"""Direct CuTe DSL NVFP4 LM-head helpers.

It is a dense-grid CuTe sidecar for benchmarking the final head
outside the persistent replay scheduler.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, const_expr
from cutlass.cute.runtime import from_dlpack


@cute.jit
def _fp4_e2m1_value(code):
    mag = code & Int32(7)
    out = mag.to(Float32) * Float32(0.5)
    out = out + cute.arch.fmax((mag - Int32(4)).to(Float32), Float32(0.0)) * Float32(0.5)
    out = out + cute.arch.fmax((mag - Int32(6)).to(Float32), Float32(0.0))
    sign = Float32(1.0) - (code >> Int32(3)).to(Float32) * Float32(2.0)
    return out * sign


@dataclass(frozen=True)
class DirectCuteNvfp4FinalConfig:
    hidden_size: int
    vocab_size: int
    group_size: int = 32
    blocks: int = 280
    threads: int = 256
    eps: float = 1e-5


class DirectCuteNvfp4FinalKernel:
    def __init__(self, cfg: DirectCuteNvfp4FinalConfig):
        self.cfg = cfg

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        norm_weight: cute.Tensor,
        weight_packed: cute.Tensor,
        weight_scales: cute.Tensor,
        block_values: cute.Tensor,
        block_indices: cute.Tensor,
        top_values: cute.Tensor,
        top_indices: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.final_kernel(
            x,
            residual,
            norm_weight,
            weight_packed,
            weight_scales,
            block_values,
            block_indices,
        ).launch(
            grid=[self.cfg.blocks, 1, 1],
            block=[self.cfg.threads, 1, 1],
            stream=stream,
        )
        self.reduce_kernel(
            block_values,
            block_indices,
            top_values,
            top_indices,
        ).launch(
            grid=[1, 1, 1],
            block=[self.cfg.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def final_kernel(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        norm_weight: cute.Tensor,
        weight_packed: cute.Tensor,
        weight_scales: cute.Tensor,
        block_values: cute.Tensor,
        block_indices: cute.Tensor,
    ):
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = tidx // Int32(32)
        lane_idx = tidx % Int32(32)
        num_warps = Int32(cfg.threads // 32)

        smem = utils.SmemAllocator()
        s_hidden = smem.allocate_tensor(Float32, cute.make_layout(cfg.hidden_size), byte_alignment=16)
        s_vals = smem.allocate_tensor(Float32, cute.make_layout(cfg.threads // 32), byte_alignment=4)
        s_idxs = smem.allocate_tensor(cutlass.Int32, cute.make_layout(cfg.threads // 32), byte_alignment=4)

        sum_sq = Float32(0.0)
        k = lane_idx
        while k < Int32(cfg.hidden_size):
            hv = x[k].to(Float32) + residual[k].to(Float32)
            sum_sq = sum_sq + hv * hv
            k = k + Int32(32)
        total_sq = cute.arch.warp_reduction(sum_sq, operator.add)
        rstd = cute.math.rsqrt(
            total_sq * Float32(1.0 / cfg.hidden_size) + Float32(cfg.eps),
            fastmath=True,
        )

        nk = tidx
        while nk < Int32(cfg.hidden_size):
            hv = x[nk].to(Float32) + residual[nk].to(Float32)
            s_hidden[nk] = hv * rstd * norm_weight[nk].to(Float32)
            nk = nk + Int32(cfg.threads)
        cute.arch.sync_threads()

        rows_per_block = (Int32(cfg.vocab_size) + Int32(cfg.blocks) - Int32(1)) // Int32(cfg.blocks)
        row_start = bidx * rows_per_block
        row_end = row_start + rows_per_block
        if row_end > Int32(cfg.vocab_size):
            row_end = Int32(cfg.vocab_size)

        best_val = Float32(-3.4028234663852886e38)
        best_idx = Int32(-1)
        v = row_start + warp_idx
        while v < row_end:
            acc = Float32(0.0)
            full_k = Int32((cfg.hidden_size // 8) * 8)
            k2 = lane_idx * Int32(8)
            while k2 < full_k:
                byte_idx = k2 >> Int32(1)
                b0 = weight_packed[(v, byte_idx)].to(Int32) & Int32(255)
                b1 = weight_packed[(v, byte_idx + Int32(1))].to(Int32) & Int32(255)
                b2 = weight_packed[(v, byte_idx + Int32(2))].to(Int32) & Int32(255)
                b3 = weight_packed[(v, byte_idx + Int32(3))].to(Int32) & Int32(255)
                if const_expr(cfg.group_size == 32):
                    scale_idx = k2 >> Int32(5)
                else:
                    scale_idx = k2 // Int32(cfg.group_size)
                scale = weight_scales[(v, scale_idx)].to(Float32)
                acc = acc + s_hidden[k2] * _fp4_e2m1_value(b0 & Int32(15)) * scale
                acc = acc + s_hidden[k2 + Int32(1)] * _fp4_e2m1_value(b0 >> Int32(4)) * scale
                acc = acc + s_hidden[k2 + Int32(2)] * _fp4_e2m1_value(b1 & Int32(15)) * scale
                acc = acc + s_hidden[k2 + Int32(3)] * _fp4_e2m1_value(b1 >> Int32(4)) * scale
                acc = acc + s_hidden[k2 + Int32(4)] * _fp4_e2m1_value(b2 & Int32(15)) * scale
                acc = acc + s_hidden[k2 + Int32(5)] * _fp4_e2m1_value(b2 >> Int32(4)) * scale
                acc = acc + s_hidden[k2 + Int32(6)] * _fp4_e2m1_value(b3 & Int32(15)) * scale
                acc = acc + s_hidden[k2 + Int32(7)] * _fp4_e2m1_value(b3 >> Int32(4)) * scale
                k2 = k2 + Int32(256)
            k2 = full_k + lane_idx
            while k2 < Int32(cfg.hidden_size):
                byte = weight_packed[(v, k2 >> Int32(1))].to(Int32) & Int32(255)
                code = byte & Int32(15)
                if (k2 & Int32(1)) != Int32(0):
                    code = byte >> Int32(4)
                if const_expr(cfg.group_size == 32):
                    scale_idx = k2 >> Int32(5)
                else:
                    scale_idx = k2 // Int32(cfg.group_size)
                scale = weight_scales[(v, scale_idx)].to(Float32)
                acc = acc + s_hidden[k2] * _fp4_e2m1_value(code) * scale
                k2 = k2 + Int32(32)
            total = cute.arch.warp_reduction(acc, operator.add)
            if lane_idx == Int32(0) and total > best_val:
                best_val = total
                best_idx = v
            v = v + num_warps

        if lane_idx == Int32(0):
            s_vals[warp_idx] = best_val
            s_idxs[warp_idx] = best_idx
        cute.arch.sync_threads()

        if tidx == Int32(0):
            block_val = Float32(-3.4028234663852886e38)
            block_idx = Int32(-1)
            wi = Int32(0)
            while wi < num_warps:
                other_val = s_vals[wi]
                if other_val > block_val:
                    block_val = other_val
                    block_idx = s_idxs[wi]
                wi = wi + Int32(1)
            block_values[bidx] = block_val
            block_indices[bidx] = block_idx

    @cute.kernel
    def reduce_kernel(
        self,
        block_values: cute.Tensor,
        block_indices: cute.Tensor,
        top_values: cute.Tensor,
        top_indices: cute.Tensor,
    ):
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        best_val = Float32(-3.4028234663852886e38)
        best_idx = Int32(-1)
        i = tidx
        while i < Int32(cfg.blocks):
            val = block_values[i]
            if val > best_val:
                best_val = val
                best_idx = block_indices[i]
            i = i + Int32(cfg.threads)

        smem = utils.SmemAllocator()
        s_vals = smem.allocate_tensor(Float32, cute.make_layout(cfg.threads), byte_alignment=4)
        s_idxs = smem.allocate_tensor(cutlass.Int32, cute.make_layout(cfg.threads), byte_alignment=4)
        s_vals[tidx] = best_val
        s_idxs[tidx] = best_idx
        cute.arch.sync_threads()

        stride = Int32(cfg.threads // 2)
        while stride > Int32(0):
            if tidx < stride:
                other_val = s_vals[tidx + stride]
                if other_val > s_vals[tidx]:
                    s_vals[tidx] = other_val
                    s_idxs[tidx] = s_idxs[tidx + stride]
            cute.arch.sync_threads()
            stride = stride >> Int32(1)

        if tidx == Int32(0):
            top_values[Int32(0)] = s_vals[Int32(0)]
            top_indices[Int32(0)] = s_idxs[Int32(0)]


_compile_cache = {}


def _as_cute(tensor: torch.Tensor, assumed_align: int = 16):
    return from_dlpack(tensor, assumed_align=assumed_align)


def get_direct_cute_nvfp4_final(
    *,
    hidden_size: int,
    vocab_size: int,
    group_size: int = 32,
    blocks: int = 280,
    threads: int = 256,
    eps: float = 1e-5,
):
    cfg = DirectCuteNvfp4FinalConfig(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        group_size=group_size,
        blocks=blocks,
        threads=threads,
        eps=eps,
    )

    def run(
        x: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        weight_packed: torch.Tensor,
        weight_scales: torch.Tensor,
        block_values: torch.Tensor,
        block_indices: torch.Tensor,
        top_values: torch.Tensor,
        top_indices: torch.Tensor,
    ):
        key = (
            hidden_size,
            vocab_size,
            group_size,
            blocks,
            threads,
            eps,
            x.dtype,
            weight_packed.dtype,
            weight_scales.dtype,
        )
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args = (
            _as_cute(x.reshape(-1)),
            _as_cute(residual.reshape(-1)),
            _as_cute(norm_weight.reshape(-1)),
            _as_cute(weight_packed),
            _as_cute(weight_scales),
            _as_cute(block_values.reshape(-1)),
            _as_cute(block_indices.reshape(-1), assumed_align=4),
            _as_cute(top_values.reshape(-1), assumed_align=4),
            _as_cute(top_indices.reshape(-1), assumed_align=4),
            stream,
        )
        if key not in _compile_cache:
            _compile_cache[key] = cute.compile(DirectCuteNvfp4FinalKernel(cfg), *args)
        _compile_cache[key](*args)

    return run
