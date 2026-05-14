# Copyright (c) 2025, Machete Authors
"""Llama-1B SM120 decode kernels.

The Llama-specific SM120 ops use staged weight matvecs: loader warps stream
weight blocks into shared memory while compute warps consume the previous
block, matching Hazy's low-latency matvec shape.
"""

import operator
import math
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, const_expr

from machete.kernels.decode_matvec.sm120 import (
    ResidualAddSm120Op as Llama1BResidualAddSm120Op,
)
from machete.megakernel.interpreter import (
    mbarrier_arrive,
    mbarrier_arrive_expect_tx,
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_wait,
    named_barrier_sync,
)
from machete.megakernel.ops import DEFAULT_PAGE_SIZE, Op, PipelineABI, PipelineSpec

from .sm100 import (
    LLAMA1B_CONSUMER_WARPS,
    LLAMA1B_HEAD_DIM,
    LLAMA1B_HIDDEN,
    LLAMA1B_INTERMEDIATE,
    LLAMA1B_KV_DIM,
    LLAMA1B_MATVEC_BLOCK,
    LLAMA1B_Q_DIM,
    LLAMA1B_ROTARY_D2,
    LLAMA1B_VOCAB,
)

LLAMA1B_SM120_MATVEC_BLOCK = 12
LLAMA1B_SM120_QKV_HEAD_BLOCK = 12
LLAMA1B_SM120_FINAL_MATVEC_BLOCK = 24
LLAMA1B_SM120_CONSUMER_WARPS = 8
LLAMA1B_SM120_REDUCTION_DIM_PER_WARP = LLAMA1B_HIDDEN // LLAMA1B_SM120_CONSUMER_WARPS
LLAMA1B_SM120_THREADS_PER_BLOCK = (LLAMA1B_SM120_CONSUMER_WARPS + 3) * 32
LLAMA1B_SM120_MIN_STAGED_PAGE_SIZE = 32 * 1024
LLAMA1B_SM120_LARGE_STAGED_PAGE_SIZE = 96 * 1024


def _validate_staged_page_size(page_size: int) -> None:
    if page_size < LLAMA1B_SM120_MIN_STAGED_PAGE_SIZE:
        raise ValueError(
            "Llama-1B SM120 staged decode requires page_size >= 32768 bytes. "
            f"Got page_size={page_size}; use 32768 or larger."
        )


def _set_exact_staged_page_size(op, page_size: int, required: int) -> None:
    _validate_staged_page_size(page_size)
    if required > page_size:
        raise ValueError(
            f"{op.op_cls.__name__} requires page_size >= {required} bytes, "
            f"but requested page_size={page_size}. The staged SM120 kernel uses "
            "the requested page size exactly; choose a compatible page_size."
        )
    op.static_dims["page_size"] = page_size


@dataclass
class Llama1BLayerSchedule:
    ops: list
    attention_config: object
    keep_alive: list


@cute.jit
def _silu(x):
    neg = Float32(0.0) - x
    exp_neg = cute.math.exp(neg, fastmath=True)
    return x / (Float32(1.0) + exp_neg)


class _Llama1BStagedWeightMatvecSm120Base(Op):
    """Hazy-style decode matvec with loader-owned staged weight pages."""

    allow_single_staged_buffer = False
    pipeline = PipelineSpec.streaming()
    pipeline_abi = PipelineABI.op_owned()
    reads = {
        "a": (None, ("B", "S", "K")),
        "weight": (None, ("O", "K")),
    }
    tile = ("B", "S", "O")
    dynamic_dims = ("B",)
    tma_loads = {"weight"}

    @classmethod
    def kernel_config(cls, ops):
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS

        page_size = max(op.static_dims.get("page_size", DEFAULT_PAGE_SIZE) for op in ops)
        return MegakernelConfig(
            threads_per_block=(LLAMA1B_SM120_CONSUMER_WARPS + NUM_DMA_WARPS) * 32,
            page_size=page_size,
            mma_reg_count=96,
        )

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name == "weight":
            return (
                tile_sizes["O"],
                static_dims.get("reduction_tile_K", LLAMA1B_SM120_REDUCTION_DIM_PER_WARP),
            )
        return None

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        if tensor_name == "weight":
            k, o = tma_tile_shape
            return f"cute.make_layout(({k}, {o}), stride=(1, {k}))"
        return None

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        page_size=DEFAULT_PAGE_SIZE,
        reduction_tile_K=None,
        **tensors,
    ):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 1)
        tile_sizes.setdefault("O", LLAMA1B_MATVEC_BLOCK)
        if tile_sizes["S"] != 1:
            raise ValueError(f"{cls.__name__} is a single-token decode matvec; got S={tile_sizes['S']}")
        if tile_sizes["O"] % 4 != 0:
            raise ValueError(f"{cls.__name__} requires O tile size divisible by 4; got O={tile_sizes['O']}")
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        weight = tensors["weight"]
        if reduction_tile_K is None:
            reduction_tile_K = LLAMA1B_SM120_REDUCTION_DIM_PER_WARP
        reduction_tile_K = min(reduction_tile_K, weight.shape[1])
        reduction_chunks = (weight.shape[1] + reduction_tile_K - 1) // reduction_tile_K
        staged_weight_chunk_bytes = tile_sizes["O"] * reduction_tile_K * weight.element_size()
        staged_weight_bytes = reduction_chunks * staged_weight_chunk_bytes
        staged_num_buffers = 2
        mbarrier_bytes = 32
        mbar_offset = staged_num_buffers * staged_weight_bytes
        rms_offset = mbar_offset + mbarrier_bytes
        partial_offset = rms_offset + reduction_chunks * 4
        scratch_bytes = reduction_chunks * 4 + reduction_chunks * tile_sizes["O"] * 4
        required = rms_offset + scratch_bytes
        if (
            required > page_size
            and cls.allow_single_staged_buffer
        ):
            compact_mbarrier_bytes = 16
            compact_required = staged_weight_bytes + compact_mbarrier_bytes + scratch_bytes
            if compact_required <= page_size:
                staged_num_buffers = 1
                mbarrier_bytes = compact_mbarrier_bytes
                mbar_offset = staged_weight_bytes
                rms_offset = mbar_offset + mbarrier_bytes
                partial_offset = rms_offset + reduction_chunks * 4
                required = rms_offset + scratch_bytes
        op.static_dims["reduction_tile_K"] = reduction_tile_K
        op.static_dims["reduction_chunks"] = reduction_chunks
        op.static_dims["staged_num_buffers"] = staged_num_buffers
        op.static_dims["staged_weight_chunk_bytes"] = staged_weight_chunk_bytes
        op.static_dims["staged_weight_bytes"] = staged_weight_bytes
        op.static_dims["mbar_offset"] = mbar_offset
        op.static_dims["rms_offset"] = rms_offset
        op.static_dims["partial_offset"] = partial_offset
        _set_exact_staged_page_size(op, page_size, required)
        return [op]

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_O, tile_3,
             weight_tma, weight_tma_gmem, work_mbar):
        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        with cute.arch.elect_one():
            mbarrier_init(bf_0, Int32(1))
            mbarrier_init(bf_1, Int32(1))
            if const_expr(self.staged_num_buffers != 1):
                mbarrier_init(kr_0, Int32(1))
                mbarrier_init(kr_1, Int32(1))
        mbarrier_init_fence()
        with cute.arch.elect_one():
            if const_expr(self.staged_num_buffers != 1):
                mbarrier_arrive(bf_1)

        block_idx = tile_O
        iter_idx = Int32(0)
        while block_idx < tile_3:
            buf_idx = Int32(0)
            if const_expr(self.staged_num_buffers != 1):
                buf_idx = iter_idx % Int32(2)
            buf_base = page_ptr + buf_idx * Int32(self.staged_weight_bytes)
            if iter_idx > Int32(0):
                bf_phase = (iter_idx - Int32(1)) % Int32(2)
                if const_expr(self.staged_num_buffers != 1):
                    bf_phase = ((iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(bf_0, bf_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(bf_1, bf_phase)

            gW = cute.local_tile(
                weight_tma_gmem,
                (self.reduction_tile_K, self.tile_size_O),
                (None, None),
            )
            nbytes = Int32(self.staged_weight_bytes)
            mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
            kr_ptr = cute.make_ptr(cutlass.Int64, bf_1, cute.AddressSpace.smem)
            if const_expr(self.staged_num_buffers != 1):
                kr_ptr = cute.make_ptr(cutlass.Int64, kr_0, cute.AddressSpace.smem)
            if iter_idx == Int32(0):
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(work_mbar, nbytes)
            if iter_idx > Int32(0):
                if const_expr(self.staged_num_buffers == 1):
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(bf_1, nbytes)
                else:
                    if buf_idx == Int32(0):
                        with cute.arch.elect_one():
                            mbarrier_arrive_expect_tx(kr_0, nbytes)
                    if buf_idx == Int32(1):
                        kr_ptr = cute.make_ptr(cutlass.Int64, kr_1, cute.AddressSpace.smem)
                        with cute.arch.elect_one():
                            mbarrier_arrive_expect_tx(kr_1, nbytes)

            for k_chunk in range(self.reduction_chunks):
                chunk_base = buf_base + Int32(k_chunk) * Int32(self.staged_weight_chunk_bytes)
                sW = cute.make_tensor(
                    cute.make_ptr(self.weight_dtype, chunk_base, cute.AddressSpace.smem, assumed_align=128),
                    cute.make_layout((self.reduction_tile_K, self.tile_size_O), stride=(1, self.reduction_tile_K)),
                )
                tWsW, tWgW = cute.nvgpu.cpasync.tma_partition(
                    weight_tma,
                    Int32(0),
                    cute.make_layout(1),
                    cute.group_modes(sW, 0, 2),
                    cute.group_modes(gW, 0, 2),
                )
                if iter_idx == Int32(0):
                    cute.copy(
                        weight_tma,
                        tWgW[(None, Int32(k_chunk), block_idx)],
                        tWsW,
                        tma_bar_ptr=mbar_ptr,
                    )
                if iter_idx > Int32(0):
                    cute.copy(
                        weight_tma,
                        tWgW[(None, Int32(k_chunk), block_idx)],
                        tWsW,
                        tma_bar_ptr=kr_ptr,
                    )

            block_idx = block_idx + Int32(1)
            iter_idx = iter_idx + Int32(1)

    @cute.jit
    def _dot_staged_partial(self, tile_B, row_idx, local_o, k_chunk, a, staged_weight):
        lane_idx = cute.arch.lane_idx()
        k_start = k_chunk * Int32(self.reduction_tile_K)
        a_base = tile_B * Int32(self.a_stride_B) + row_idx * Int32(self.a_stride_S) + k_start
        a_row = cute.make_tensor(a.iterator + a_base, cute.make_layout(self.reduction_tile_K))
        acc = Float32(0.0)
        k = lane_idx * Int32(2)
        while k < Int32(self.reduction_tile_K):
            global_k = k_start + k
            if global_k < Int32(self.K):
                acc = acc + a_row[k].to(Float32) * staged_weight[(k, local_o)].to(Float32)
            k1 = k + Int32(1)
            global_k1 = k_start + k1
            if k1 < Int32(self.reduction_tile_K):
                if global_k1 < Int32(self.K):
                    acc = acc + a_row[k1].to(Float32) * staged_weight[(k1, local_o)].to(Float32)
            k = k + Int32(64)
        return cute.arch.warp_reduction(acc, operator.add)

    @cute.jit
    def _dot_staged_norm_residual_partial4(
        self,
        tile_B,
        row_idx,
        local_o,
        k_chunk,
        rstd,
        x,
        residual_in,
        norm_weight,
        staged_weight,
    ):
        lane_idx = cute.arch.lane_idx()
        k_start = k_chunk * Int32(self.reduction_tile_K)
        x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S) + k_start
        r_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S) + k_start
        x_row = cute.make_tensor(x.iterator + x_base, cute.make_layout(self.reduction_tile_K))
        r_row = cute.make_tensor(residual_in.iterator + r_base, cute.make_layout(self.reduction_tile_K))
        norm_row = cute.make_tensor(norm_weight.iterator + k_start, cute.make_layout(self.reduction_tile_K))
        acc0 = Float32(0.0)
        acc1 = Float32(0.0)
        acc2 = Float32(0.0)
        acc3 = Float32(0.0)
        local_o1 = local_o + Int32(1)
        local_o2 = local_o + Int32(2)
        local_o3 = local_o + Int32(3)
        k = lane_idx * Int32(2)
        while k < Int32(self.reduction_tile_K):
            global_k = k_start + k
            if global_k < Int32(self.K):
                nv = (x_row[k].to(Float32) + r_row[k].to(Float32)) * rstd * norm_row[k].to(Float32)
                acc0 = acc0 + nv * staged_weight[(k, local_o)].to(Float32)
                acc1 = acc1 + nv * staged_weight[(k, local_o1)].to(Float32)
                acc2 = acc2 + nv * staged_weight[(k, local_o2)].to(Float32)
                acc3 = acc3 + nv * staged_weight[(k, local_o3)].to(Float32)
            k1 = k + Int32(1)
            global_k1 = k_start + k1
            if k1 < Int32(self.reduction_tile_K):
                if global_k1 < Int32(self.K):
                    nv1 = (x_row[k1].to(Float32) + r_row[k1].to(Float32)) * rstd * norm_row[k1].to(Float32)
                    acc0 = acc0 + nv1 * staged_weight[(k1, local_o)].to(Float32)
                    acc1 = acc1 + nv1 * staged_weight[(k1, local_o1)].to(Float32)
                    acc2 = acc2 + nv1 * staged_weight[(k1, local_o2)].to(Float32)
                    acc3 = acc3 + nv1 * staged_weight[(k1, local_o3)].to(Float32)
            k = k + Int32(64)
        return (
            cute.arch.warp_reduction(acc0, operator.add),
            cute.arch.warp_reduction(acc1, operator.add),
            cute.arch.warp_reduction(acc2, operator.add),
            cute.arch.warp_reduction(acc3, operator.add),
        )

    @cute.jit
    def _dot_staged_partial4(self, tile_B, row_idx, local_o, k_chunk, a, staged_weight):
        lane_idx = cute.arch.lane_idx()
        k_start = k_chunk * Int32(self.reduction_tile_K)
        a_base = tile_B * Int32(self.a_stride_B) + row_idx * Int32(self.a_stride_S) + k_start
        a_row = cute.make_tensor(a.iterator + a_base, cute.make_layout(self.reduction_tile_K))
        acc0 = Float32(0.0)
        acc1 = Float32(0.0)
        acc2 = Float32(0.0)
        acc3 = Float32(0.0)
        local_o1 = local_o + Int32(1)
        local_o2 = local_o + Int32(2)
        local_o3 = local_o + Int32(3)
        k = lane_idx * Int32(2)
        while k < Int32(self.reduction_tile_K):
            global_k = k_start + k
            if global_k < Int32(self.K):
                av = a_row[k].to(Float32)
                acc0 = acc0 + av * staged_weight[(k, local_o)].to(Float32)
                acc1 = acc1 + av * staged_weight[(k, local_o1)].to(Float32)
                acc2 = acc2 + av * staged_weight[(k, local_o2)].to(Float32)
                acc3 = acc3 + av * staged_weight[(k, local_o3)].to(Float32)
            k1 = k + Int32(1)
            global_k1 = k_start + k1
            if k1 < Int32(self.reduction_tile_K):
                if global_k1 < Int32(self.K):
                    av1 = a_row[k1].to(Float32)
                    acc0 = acc0 + av1 * staged_weight[(k1, local_o)].to(Float32)
                    acc1 = acc1 + av1 * staged_weight[(k1, local_o1)].to(Float32)
                    acc2 = acc2 + av1 * staged_weight[(k1, local_o2)].to(Float32)
                    acc3 = acc3 + av1 * staged_weight[(k1, local_o3)].to(Float32)
            k = k + Int32(64)
        return (
            cute.arch.warp_reduction(acc0, operator.add),
            cute.arch.warp_reduction(acc1, operator.add),
            cute.arch.warp_reduction(acc2, operator.add),
            cute.arch.warp_reduction(acc3, operator.add),
        )

    @cute.jit
    def _sum_staged_partials(self, page_ptr, local_o):
        partials = cute.make_tensor(
            cute.make_ptr(cutlass.Float32, page_ptr + Int32(self.partial_offset), cute.AddressSpace.smem),
            cute.make_layout((self.reduction_chunks, self.tile_size_O), stride=(self.tile_size_O, 1)),
        )
        total = Float32(0.0)
        k_chunk = Int32(0)
        while k_chunk < Int32(self.reduction_chunks):
            total = total + partials[(k_chunk, local_o)]
            k_chunk = k_chunk + Int32(1)
        return total


class Llama1BDownMatvecSm120Op(_Llama1BStagedWeightMatvecSm120Base):
    """Staged down/projection matvec."""

    writes = {"y": (None, ("B", "S", "O"))}

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, tile_3, a, y):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        row_start = tile_S * Int32(self.tile_size_S)
        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        block_idx = tile_O
        iter_idx = Int32(0)
        while block_idx < tile_3:
            buf_idx = iter_idx % Int32(2)
            if iter_idx > Int32(0):
                kr_phase = ((iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(kr_0, kr_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(kr_1, kr_phase)
            partials = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, page_ptr + Int32(self.partial_offset), cute.AddressSpace.smem),
                cute.make_layout((self.reduction_chunks, self.tile_size_O), stride=(self.tile_size_O, 1)),
            )
            row_idx = row_start
            if row_idx < Int32(self.S):
                o_start = block_idx * Int32(self.tile_size_O)
                if warp_idx < Int32(self.reduction_chunks):
                    staged_weight = cute.make_tensor(
                        cute.make_ptr(
                            self.weight_dtype,
                            page_ptr
                            + buf_idx * Int32(self.staged_weight_bytes)
                            + warp_idx * Int32(self.staged_weight_chunk_bytes),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        cute.make_layout((self.reduction_tile_K, self.tile_size_O), stride=(1, self.reduction_tile_K)),
                    )
                    for local_o in range(0, self.tile_size_O, 4):
                        out_idx = o_start + Int32(local_o)
                        if out_idx < Int32(self.O):
                            partial0, partial1, partial2, partial3 = self._dot_staged_partial4(
                                tile_B,
                                row_idx,
                                Int32(local_o),
                                warp_idx,
                                a,
                                staged_weight,
                            )
                            if lane_idx == Int32(0):
                                partials[(warp_idx, Int32(local_o))] = partial0
                                if out_idx + Int32(1) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 1))] = partial1
                                if out_idx + Int32(2) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 2))] = partial2
                                if out_idx + Int32(3) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 3))] = partial3
                named_barrier_sync(Int32(2), Int32(self.threads_per_row))
                if warp_idx == Int32(0):
                    local_o = lane_idx
                    while local_o < Int32(self.tile_size_O):
                        out_idx = o_start + local_o
                        if out_idx < Int32(self.O):
                            total = self._sum_staged_partials(page_ptr, local_o)
                            y_base = tile_B * Int32(self.y_stride_B) + row_idx * Int32(self.y_stride_S)
                            y_row = cute.make_tensor(y.iterator + y_base, cute.make_layout(self.O))
                            y_row[out_idx] = total.to(self.y_dtype)
                        local_o = local_o + Int32(32)
            named_barrier_sync(Int32(2), Int32(self.threads_per_row))
            if tidx == Int32(0):
                if buf_idx == Int32(0):
                    mbarrier_arrive(bf_0)
                if buf_idx == Int32(1):
                    mbarrier_arrive(bf_1)
            block_idx = block_idx + Int32(1)
            iter_idx = iter_idx + Int32(1)


class Llama1BMatvecResidualSm120Op(_Llama1BStagedWeightMatvecSm120Base):
    """Staged projection matvec with residual add."""

    reads = {
        "a": (None, ("B", "S", "K")),
        "weight": (None, ("O", "K")),
        "residual_in": (None, ("B", "S", "O")),
    }
    writes = {"residual_out": (None, ("B", "S", "O"))}

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, tile_3,
                a, residual_in, residual_out):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        row_start = tile_S * Int32(self.tile_size_S)
        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        block_idx = tile_O
        iter_idx = Int32(0)
        while block_idx < tile_3:
            buf_idx = iter_idx % Int32(2)
            if iter_idx > Int32(0):
                kr_phase = ((iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(kr_0, kr_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(kr_1, kr_phase)
            partials = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, page_ptr + Int32(self.partial_offset), cute.AddressSpace.smem),
                cute.make_layout((self.reduction_chunks, self.tile_size_O), stride=(self.tile_size_O, 1)),
            )
            row_idx = row_start
            if row_idx < Int32(self.S):
                o_start = block_idx * Int32(self.tile_size_O)
                if warp_idx < Int32(self.reduction_chunks):
                    staged_weight = cute.make_tensor(
                        cute.make_ptr(
                            self.weight_dtype,
                            page_ptr
                            + buf_idx * Int32(self.staged_weight_bytes)
                            + warp_idx * Int32(self.staged_weight_chunk_bytes),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        cute.make_layout((self.reduction_tile_K, self.tile_size_O), stride=(1, self.reduction_tile_K)),
                    )
                    for local_o in range(0, self.tile_size_O, 4):
                        out_idx = o_start + Int32(local_o)
                        if out_idx < Int32(self.O):
                            partial0, partial1, partial2, partial3 = self._dot_staged_partial4(
                                tile_B,
                                row_idx,
                                Int32(local_o),
                                warp_idx,
                                a,
                                staged_weight,
                            )
                            if lane_idx == Int32(0):
                                partials[(warp_idx, Int32(local_o))] = partial0
                                if out_idx + Int32(1) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 1))] = partial1
                                if out_idx + Int32(2) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 2))] = partial2
                                if out_idx + Int32(3) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 3))] = partial3
                named_barrier_sync(Int32(2), Int32(self.threads_per_row))
                if warp_idx == Int32(0):
                    local_o = lane_idx
                    while local_o < Int32(self.tile_size_O):
                        out_idx = o_start + local_o
                        if out_idx < Int32(self.O):
                            total = self._sum_staged_partials(page_ptr, local_o)
                            res_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S)
                            out_base = tile_B * Int32(self.residual_out_stride_B) + row_idx * Int32(self.residual_out_stride_S)
                            res_row = cute.make_tensor(residual_in.iterator + res_base, cute.make_layout(self.O))
                            out_row = cute.make_tensor(residual_out.iterator + out_base, cute.make_layout(self.O))
                            out_row[out_idx] = (total + res_row[out_idx].to(Float32)).to(self.residual_out_dtype)
                        local_o = local_o + Int32(32)
            named_barrier_sync(Int32(2), Int32(self.threads_per_row))
            if tidx == Int32(0):
                if buf_idx == Int32(0):
                    mbarrier_arrive(bf_0)
                if buf_idx == Int32(1):
                    mbarrier_arrive(bf_1)
            block_idx = block_idx + Int32(1)
            iter_idx = iter_idx + Int32(1)


class Llama1BDownAdd4Sm120Op(Op):
    """Fuse the four row-parallel down-projection partials."""

    reads = {
        "p0": (None, ("B", "S", "D")),
        "p1": (None, ("B", "S", "D")),
        "p2": (None, ("B", "S", "D")),
        "p3": (None, ("B", "S", "D")),
    }
    writes = {"y": (None, ("B", "S", "D"))}
    tile = ("B", "S", "D")
    dynamic_dims = ("B",)

    @classmethod
    def schedule(cls, tile_sizes=None, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 1)
        tile_sizes.setdefault("D", 256)
        if tile_sizes["S"] != 1:
            raise ValueError(f"{cls.__name__} is a single-token decode add; got S={tile_sizes['S']}")
        return [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_D, p0, p1, p2, p3, y):
        tidx = cute.arch.thread_idx()[0]
        row_idx = tile_S * Int32(self.tile_size_S)
        d_start = tile_D * Int32(self.tile_size_D)
        d_stop = d_start + Int32(self.tile_size_D)
        if row_idx < Int32(self.S):
            base = tile_B * Int32(self.S * self.D) + row_idx * Int32(self.D)
            p0_row = cute.make_tensor(p0.iterator + base, cute.make_layout(self.D))
            p1_row = cute.make_tensor(p1.iterator + base, cute.make_layout(self.D))
            p2_row = cute.make_tensor(p2.iterator + base, cute.make_layout(self.D))
            p3_row = cute.make_tensor(p3.iterator + base, cute.make_layout(self.D))
            y_row = cute.make_tensor(y.iterator + base, cute.make_layout(self.D))
            d = d_start + tidx
            while d < d_stop:
                if d < Int32(self.D):
                    total = (
                        p0_row[d].to(Float32)
                        + p1_row[d].to(Float32)
                        + p2_row[d].to(Float32)
                        + p3_row[d].to(Float32)
                    )
                    y_row[d] = total.to(self.y_dtype)
                d = d + Int32(self.threads_per_row)


class _Llama1BStagedRmsMatvecSm120Base(_Llama1BStagedWeightMatvecSm120Base):
    """Staged RMS-normalized matvec using Hazy's warp-split shape."""

    reads = {
        "x": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "weight": (None, ("O", "K")),
    }

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        page_size=DEFAULT_PAGE_SIZE,
        eps=1e-5,
        reduction_tile_K=None,
        **tensors,
    ):
        ops = super().schedule(
            tile_sizes=tile_sizes,
            page_size=page_size,
            reduction_tile_K=reduction_tile_K,
            **tensors,
        )
        ops[0].static_dims["eps"] = eps
        return ops

    @cute.jit
    def _row_rstd(self, page_ptr, tile_B, row_idx, x):
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        rms_partials = cute.make_tensor(
            cute.make_ptr(cutlass.Float32, page_ptr + Int32(self.rms_offset), cute.AddressSpace.smem),
            cute.make_layout(self.reduction_chunks),
        )
        if warp_idx < Int32(self.reduction_chunks):
            k_start = warp_idx * Int32(self.reduction_tile_K)
            x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S) + k_start
            x_row = cute.make_tensor(x.iterator + x_base, cute.make_layout(self.reduction_tile_K))
            sum_sq = Float32(0.0)
            k = lane_idx * Int32(2)
            while k < Int32(self.reduction_tile_K):
                global_k = k_start + k
                if global_k < Int32(self.K):
                    xv = x_row[k].to(Float32)
                    sum_sq = sum_sq + xv * xv
                k1 = k + Int32(1)
                global_k1 = k_start + k1
                if k1 < Int32(self.reduction_tile_K):
                    if global_k1 < Int32(self.K):
                        xv1 = x_row[k1].to(Float32)
                        sum_sq = sum_sq + xv1 * xv1
                k = k + Int32(64)
            total = cute.arch.warp_reduction(sum_sq, operator.add)
            if lane_idx == Int32(0):
                rms_partials[warp_idx] = total
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))
        total_sq = Float32(0.0)
        k_chunk = Int32(0)
        while k_chunk < Int32(self.reduction_chunks):
            total_sq = total_sq + rms_partials[k_chunk]
            k_chunk = k_chunk + Int32(1)
        return cute.math.rsqrt(total_sq * Float32(1.0 / self.K) + Float32(self.eps), fastmath=True)

    @cute.jit
    def _dot_staged_norm_partial(
        self,
        tile_B,
        row_idx,
        local_o,
        k_chunk,
        rstd,
        x,
        norm_weight,
        staged_weight,
    ):
        lane_idx = cute.arch.lane_idx()
        k_start = k_chunk * Int32(self.reduction_tile_K)
        x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S) + k_start
        x_row = cute.make_tensor(x.iterator + x_base, cute.make_layout(self.reduction_tile_K))
        norm_row = cute.make_tensor(norm_weight.iterator + k_start, cute.make_layout(self.reduction_tile_K))
        acc = Float32(0.0)
        k = lane_idx * Int32(2)
        while k < Int32(self.reduction_tile_K):
            global_k = k_start + k
            if global_k < Int32(self.K):
                xv = x_row[k].to(Float32)
                nv = xv * rstd * norm_row[k].to(Float32)
                acc = acc + nv * staged_weight[(k, local_o)].to(Float32)
            k1 = k + Int32(1)
            global_k1 = k_start + k1
            if k1 < Int32(self.reduction_tile_K):
                if global_k1 < Int32(self.K):
                    xv1 = x_row[k1].to(Float32)
                    nv1 = xv1 * rstd * norm_row[k1].to(Float32)
                    acc = acc + nv1 * staged_weight[(k1, local_o)].to(Float32)
            k = k + Int32(64)
        return cute.arch.warp_reduction(acc, operator.add)

    @cute.jit
    def _dot_staged_norm_partial4(
        self,
        tile_B,
        row_idx,
        local_o,
        k_chunk,
        rstd,
        x,
        norm_weight,
        staged_weight,
    ):
        lane_idx = cute.arch.lane_idx()
        k_start = k_chunk * Int32(self.reduction_tile_K)
        x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S) + k_start
        x_row = cute.make_tensor(x.iterator + x_base, cute.make_layout(self.reduction_tile_K))
        norm_row = cute.make_tensor(norm_weight.iterator + k_start, cute.make_layout(self.reduction_tile_K))
        acc0 = Float32(0.0)
        acc1 = Float32(0.0)
        acc2 = Float32(0.0)
        acc3 = Float32(0.0)
        local_o1 = local_o + Int32(1)
        local_o2 = local_o + Int32(2)
        local_o3 = local_o + Int32(3)
        k = lane_idx * Int32(2)
        while k < Int32(self.reduction_tile_K):
            global_k = k_start + k
            if global_k < Int32(self.K):
                nv = x_row[k].to(Float32) * rstd * norm_row[k].to(Float32)
                acc0 = acc0 + nv * staged_weight[(k, local_o)].to(Float32)
                acc1 = acc1 + nv * staged_weight[(k, local_o1)].to(Float32)
                acc2 = acc2 + nv * staged_weight[(k, local_o2)].to(Float32)
                acc3 = acc3 + nv * staged_weight[(k, local_o3)].to(Float32)
            k1 = k + Int32(1)
            global_k1 = k_start + k1
            if k1 < Int32(self.reduction_tile_K):
                if global_k1 < Int32(self.K):
                    nv1 = x_row[k1].to(Float32) * rstd * norm_row[k1].to(Float32)
                    acc0 = acc0 + nv1 * staged_weight[(k1, local_o)].to(Float32)
                    acc1 = acc1 + nv1 * staged_weight[(k1, local_o1)].to(Float32)
                    acc2 = acc2 + nv1 * staged_weight[(k1, local_o2)].to(Float32)
                    acc3 = acc3 + nv1 * staged_weight[(k1, local_o3)].to(Float32)
            k = k + Int32(64)
        return (
            cute.arch.warp_reduction(acc0, operator.add),
            cute.arch.warp_reduction(acc1, operator.add),
            cute.arch.warp_reduction(acc2, operator.add),
            cute.arch.warp_reduction(acc3, operator.add),
        )


class _Llama1BStagedRmsResidualMatvecSm120Base(_Llama1BStagedRmsMatvecSm120Base):
    """Staged RMS(input + residual) matvec for attention projections."""

    reads = {
        "x": (None, ("B", "S", "K")),
        "residual_in": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "weight": (None, ("O", "K")),
        "cos": (None, ("S", "HD")),
        "sin": (None, ("S", "HD")),
    }

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        page_size=DEFAULT_PAGE_SIZE,
        eps=1e-5,
        cache_pos=0,
        **tensors,
    ):
        ops = super().schedule(
            tile_sizes=tile_sizes,
            page_size=page_size,
            eps=eps,
            **tensors,
        )
        op = ops[0]
        op.static_dims["cache_pos"] = cache_pos
        op.static_dims["head_dim"] = tensors["cos"].shape[1]
        return ops

    @cute.jit
    def _row_rstd_residual(self, page_ptr, tile_B, row_idx, x, residual_in):
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        rms_partials = cute.make_tensor(
            cute.make_ptr(cutlass.Float32, page_ptr + Int32(self.rms_offset), cute.AddressSpace.smem),
            cute.make_layout(self.reduction_chunks),
        )
        if warp_idx < Int32(self.reduction_chunks):
            k_start = warp_idx * Int32(self.reduction_tile_K)
            x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S) + k_start
            r_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S) + k_start
            x_row = cute.make_tensor(x.iterator + x_base, cute.make_layout(self.reduction_tile_K))
            r_row = cute.make_tensor(residual_in.iterator + r_base, cute.make_layout(self.reduction_tile_K))
            sum_sq = Float32(0.0)
            k = lane_idx * Int32(2)
            while k < Int32(self.reduction_tile_K):
                global_k = k_start + k
                if global_k < Int32(self.K):
                    val = x_row[k].to(Float32) + r_row[k].to(Float32)
                    sum_sq = sum_sq + val * val
                k1 = k + Int32(1)
                global_k1 = k_start + k1
                if k1 < Int32(self.reduction_tile_K):
                    if global_k1 < Int32(self.K):
                        val1 = x_row[k1].to(Float32) + r_row[k1].to(Float32)
                        sum_sq = sum_sq + val1 * val1
                k = k + Int32(64)
            total = cute.arch.warp_reduction(sum_sq, operator.add)
            if lane_idx == Int32(0):
                rms_partials[warp_idx] = total
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))
        total_sq = Float32(0.0)
        k_chunk = Int32(0)
        while k_chunk < Int32(self.reduction_chunks):
            total_sq = total_sq + rms_partials[k_chunk]
            k_chunk = k_chunk + Int32(1)
        return cute.math.rsqrt(total_sq * Float32(1.0 / self.K) + Float32(self.eps), fastmath=True)

    @cute.jit
    def _store_residual(self, tile_B, row_idx, x, residual_in, residual_out):
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        if warp_idx < Int32(self.reduction_chunks):
            k_start = warp_idx * Int32(self.reduction_tile_K)
            x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S) + k_start
            r_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S) + k_start
            o_base = tile_B * Int32(self.residual_out_stride_B) + row_idx * Int32(self.residual_out_stride_S) + k_start
            x_row = cute.make_tensor(x.iterator + x_base, cute.make_layout(self.reduction_tile_K))
            r_row = cute.make_tensor(residual_in.iterator + r_base, cute.make_layout(self.reduction_tile_K))
            o_row = cute.make_tensor(residual_out.iterator + o_base, cute.make_layout(self.reduction_tile_K))
            k = lane_idx * Int32(2)
            while k < Int32(self.reduction_tile_K):
                global_k = k_start + k
                if global_k < Int32(self.K):
                    o_row[k] = (x_row[k].to(Float32) + r_row[k].to(Float32)).to(self.residual_out_dtype)
                k1 = k + Int32(1)
                global_k1 = k_start + k1
                if k1 < Int32(self.reduction_tile_K):
                    if global_k1 < Int32(self.K):
                        o_row[k1] = (x_row[k1].to(Float32) + r_row[k1].to(Float32)).to(self.residual_out_dtype)
                k = k + Int32(64)

    @cute.jit
    def _dot_staged_norm_residual_partial(
        self,
        tile_B,
        row_idx,
        local_o,
        k_chunk,
        rstd,
        x,
        residual_in,
        norm_weight,
        staged_weight,
    ):
        lane_idx = cute.arch.lane_idx()
        k_start = k_chunk * Int32(self.reduction_tile_K)
        x_base = tile_B * Int32(self.x_stride_B) + row_idx * Int32(self.x_stride_S) + k_start
        r_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S) + k_start
        x_row = cute.make_tensor(x.iterator + x_base, cute.make_layout(self.reduction_tile_K))
        r_row = cute.make_tensor(residual_in.iterator + r_base, cute.make_layout(self.reduction_tile_K))
        norm_row = cute.make_tensor(norm_weight.iterator + k_start, cute.make_layout(self.reduction_tile_K))
        acc = Float32(0.0)
        k = lane_idx * Int32(2)
        while k < Int32(self.reduction_tile_K):
            global_k = k_start + k
            if global_k < Int32(self.K):
                xv = x_row[k].to(Float32) + r_row[k].to(Float32)
                nv = xv * rstd * norm_row[k].to(Float32)
                acc = acc + nv * staged_weight[(k, local_o)].to(Float32)
            k1 = k + Int32(1)
            global_k1 = k_start + k1
            if k1 < Int32(self.reduction_tile_K):
                if global_k1 < Int32(self.K):
                    xv1 = x_row[k1].to(Float32) + r_row[k1].to(Float32)
                    nv1 = xv1 * rstd * norm_row[k1].to(Float32)
                    acc = acc + nv1 * staged_weight[(k1, local_o)].to(Float32)
            k = k + Int32(64)
        return cute.arch.warp_reduction(acc, operator.add)

    @cute.jit
    def _rope_pair_value(self, page_ptr, local_o):
        pair_o = local_o + Int32(1)
        if (local_o & Int32(1)) != Int32(0):
            pair_o = local_o - Int32(1)
        return self._sum_staged_partials(page_ptr, pair_o)

    @cute.jit
    def _apply_hazy_rope(self, page_ptr, row_idx, out_idx, local_o, value, cos, sin):
        dim = out_idx % Int32(self.head_dim)
        pair = self._rope_pair_value(page_ptr, local_o)
        cos_row = cute.make_tensor(cos.iterator + row_idx * Int32(self.cos_stride_S), cute.make_layout(self.head_dim))
        sin_row = cute.make_tensor(sin.iterator + row_idx * Int32(self.sin_stride_S), cute.make_layout(self.head_dim))
        c = cos_row[dim].to(Float32)
        s = sin_row[dim].to(Float32)
        out = value * c + pair * s
        if (local_o & Int32(1)) == Int32(0):
            out = value * c - pair * s
        return out


class Llama1BRmsQSm120Op(_Llama1BStagedRmsResidualMatvecSm120Base):
    """Staged RMS(input+residual) + Q matvec + Hazy-style RoPE."""

    allow_single_staged_buffer = True

    writes = {
        "residual_out": (None, ("B", "S", "K")),
        "q": (None, ("B", "S", "O")),
    }

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, tile_3,
                x, residual_in, norm_weight, weight, cos, sin, residual_out, q):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        row_idx = tile_S * Int32(self.tile_size_S)
        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        rstd = self._row_rstd_residual(page_ptr, tile_B, row_idx, x, residual_in)
        self._store_residual(tile_B, row_idx, x, residual_in, residual_out)
        block_idx = tile_O
        iter_idx = Int32(0)
        while block_idx < tile_3:
            buf_idx = Int32(0)
            if const_expr(self.staged_num_buffers != 1):
                buf_idx = iter_idx % Int32(2)
            if iter_idx > Int32(0):
                kr_phase = (iter_idx - Int32(1)) % Int32(2)
                if const_expr(self.staged_num_buffers == 1):
                    mbarrier_wait(bf_1, kr_phase)
                else:
                    kr_phase = ((iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                    if buf_idx == Int32(0):
                        mbarrier_wait(kr_0, kr_phase)
                    if buf_idx == Int32(1):
                        mbarrier_wait(kr_1, kr_phase)
            partials = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, page_ptr + Int32(self.partial_offset), cute.AddressSpace.smem),
                cute.make_layout((self.reduction_chunks, self.tile_size_O), stride=(self.tile_size_O, 1)),
            )
            if row_idx < Int32(self.S):
                o_start = block_idx * Int32(self.tile_size_O)
                if warp_idx < Int32(self.reduction_chunks):
                    staged_weight = cute.make_tensor(
                        cute.make_ptr(
                            self.weight_dtype,
                            page_ptr + buf_idx * Int32(self.staged_weight_bytes) + warp_idx * Int32(self.staged_weight_chunk_bytes),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        cute.make_layout((self.reduction_tile_K, self.tile_size_O), stride=(1, self.reduction_tile_K)),
                    )
                    for local_o in range(0, self.tile_size_O, 4):
                        out_idx = o_start + Int32(local_o)
                        if out_idx < Int32(self.O):
                            partial0, partial1, partial2, partial3 = self._dot_staged_norm_residual_partial4(
                                tile_B, row_idx, Int32(local_o), warp_idx, rstd, x, residual_in, norm_weight, staged_weight
                            )
                            if lane_idx == Int32(0):
                                partials[(warp_idx, Int32(local_o))] = partial0
                                if out_idx + Int32(1) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 1))] = partial1
                                if out_idx + Int32(2) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 2))] = partial2
                                if out_idx + Int32(3) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 3))] = partial3
                named_barrier_sync(Int32(2), Int32(self.threads_per_row))
                if warp_idx == Int32(0):
                    local_o = lane_idx
                    while local_o < Int32(self.tile_size_O):
                        out_idx = o_start + local_o
                        if out_idx < Int32(self.O):
                            total = self._sum_staged_partials(page_ptr, local_o)
                            total = self._apply_hazy_rope(page_ptr, row_idx, out_idx, local_o, total, cos, sin)
                            q_base = tile_B * Int32(self.q_stride_B) + row_idx * Int32(self.q_stride_S)
                            q_row = cute.make_tensor(q.iterator + q_base, cute.make_layout(self.O))
                            q_row[out_idx] = total.to(self.q_dtype)
                        local_o = local_o + Int32(32)
            named_barrier_sync(Int32(2), Int32(self.threads_per_row))
            if tidx == Int32(0):
                if const_expr(self.staged_num_buffers == 1):
                    mbarrier_arrive(bf_0)
                else:
                    if buf_idx == Int32(0):
                        mbarrier_arrive(bf_0)
                    if buf_idx == Int32(1):
                        mbarrier_arrive(bf_1)
            block_idx = block_idx + Int32(1)
            iter_idx = iter_idx + Int32(1)


class Llama1BRmsKCacheSm120Op(_Llama1BStagedRmsResidualMatvecSm120Base):
    """Staged RMS(input+residual) + K matvec + Hazy-style RoPE + cache append."""

    writes = {"dst_cache": (None, ("B", "T", "H", "HD"))}

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, tile_3,
                x, residual_in, norm_weight, weight, cos, sin, dst_cache):
        self._compute_cache_projection(
            page_ptr, tile_B, tile_S, tile_O, tile_3,
            x, residual_in, norm_weight, cos, sin, dst_cache, Int32(1),
        )

    @cute.jit
    def _compute_cache_projection(self, page_ptr, tile_B, tile_S, tile_O, tile_3,
                                  x, residual_in, norm_weight, cos, sin, dst_cache, apply_rope):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        row_idx = tile_S * Int32(self.tile_size_S)
        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        rstd = self._row_rstd_residual(page_ptr, tile_B, row_idx, x, residual_in)
        block_idx = tile_O
        iter_idx = Int32(0)
        while block_idx < tile_3:
            buf_idx = iter_idx % Int32(2)
            if iter_idx > Int32(0):
                kr_phase = ((iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(kr_0, kr_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(kr_1, kr_phase)
            partials = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, page_ptr + Int32(self.partial_offset), cute.AddressSpace.smem),
                cute.make_layout((self.reduction_chunks, self.tile_size_O), stride=(self.tile_size_O, 1)),
            )
            if row_idx < Int32(self.S):
                o_start = block_idx * Int32(self.tile_size_O)
                if warp_idx < Int32(self.reduction_chunks):
                    staged_weight = cute.make_tensor(
                        cute.make_ptr(
                            self.weight_dtype,
                            page_ptr + buf_idx * Int32(self.staged_weight_bytes) + warp_idx * Int32(self.staged_weight_chunk_bytes),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        cute.make_layout((self.reduction_tile_K, self.tile_size_O), stride=(1, self.reduction_tile_K)),
                    )
                    for local_o in range(0, self.tile_size_O, 4):
                        out_idx = o_start + Int32(local_o)
                        if out_idx < Int32(self.O):
                            partial0, partial1, partial2, partial3 = self._dot_staged_norm_residual_partial4(
                                tile_B, row_idx, Int32(local_o), warp_idx, rstd, x, residual_in, norm_weight, staged_weight
                            )
                            if lane_idx == Int32(0):
                                partials[(warp_idx, Int32(local_o))] = partial0
                                if out_idx + Int32(1) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 1))] = partial1
                                if out_idx + Int32(2) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 2))] = partial2
                                if out_idx + Int32(3) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 3))] = partial3
                named_barrier_sync(Int32(2), Int32(self.threads_per_row))
                if warp_idx == Int32(0):
                    local_o = lane_idx
                    while local_o < Int32(self.tile_size_O):
                        out_idx = o_start + local_o
                        if out_idx < Int32(self.O):
                            total = self._sum_staged_partials(page_ptr, local_o)
                            if apply_rope != Int32(0):
                                total = self._apply_hazy_rope(page_ptr, row_idx, out_idx, local_o, total, cos, sin)
                            head = out_idx // Int32(self.head_dim)
                            dim = out_idx % Int32(self.head_dim)
                            dst_base = (
                                tile_B * Int32(self.dst_cache_stride_B)
                                + (row_idx + Int32(self.cache_pos)) * Int32(self.dst_cache_stride_T)
                                + head * Int32(self.dst_cache_stride_H)
                            )
                            dst_row = cute.make_tensor(dst_cache.iterator + dst_base, cute.make_layout(self.HD))
                            dst_row[dim] = total.to(self.dst_cache_dtype)
                        local_o = local_o + Int32(32)
            named_barrier_sync(Int32(2), Int32(self.threads_per_row))
            if tidx == Int32(0):
                if buf_idx == Int32(0):
                    mbarrier_arrive(bf_0)
                if buf_idx == Int32(1):
                    mbarrier_arrive(bf_1)
            block_idx = block_idx + Int32(1)
            iter_idx = iter_idx + Int32(1)


class Llama1BRmsVCacheSm120Op(Llama1BRmsKCacheSm120Op):
    """Staged RMS(input+residual) + V matvec + cache append."""

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, tile_3,
                x, residual_in, norm_weight, weight, cos, sin, dst_cache):
        self._compute_cache_projection(
            page_ptr, tile_B, tile_S, tile_O, tile_3,
            x, residual_in, norm_weight, cos, sin, dst_cache, Int32(0),
        )


class Llama1BRmsKVCacheSm120Op(_Llama1BStagedRmsResidualMatvecSm120Base):
    """Hazy-style staged RMS + fused K/V cache append.

    The op computes the shared RMS(input + residual) once, then alternates
    staged K and V weight tiles. K applies RoPE before appending to K cache; V
    is appended directly.
    """

    reads = {
        "x": (None, ("B", "S", "K")),
        "residual_in": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "k_weight": (None, ("O", "K")),
        "v_weight": (None, ("O", "K")),
        "cos": (None, ("S", "HD")),
        "sin": (None, ("S", "HD")),
    }
    writes = {
        "k_cache": (None, ("B", "T", "H", "HD")),
        "v_cache": (None, ("B", "T", "H", "HD")),
    }
    tma_loads = {"k_weight", "v_weight"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name in ("k_weight", "v_weight"):
            return (
                tile_sizes["O"],
                static_dims.get("reduction_tile_K", LLAMA1B_SM120_REDUCTION_DIM_PER_WARP),
            )
        return None

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        if tensor_name in ("k_weight", "v_weight"):
            o, k = tma_tile_shape
            return f"cute.make_layout(({o}, {k}), stride=(1, {o}))"
        return None

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        page_size=DEFAULT_PAGE_SIZE,
        eps=1e-5,
        cache_pos=0,
        **tensors,
    ):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 1)
        tile_sizes.setdefault("O", LLAMA1B_MATVEC_BLOCK)
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        weight = tensors["k_weight"]
        reduction_tile_K = min(LLAMA1B_SM120_REDUCTION_DIM_PER_WARP, weight.shape[1])
        reduction_chunks = (weight.shape[1] + reduction_tile_K - 1) // reduction_tile_K
        staged_weight_chunk_bytes = tile_sizes["O"] * reduction_tile_K * weight.element_size()
        staged_weight_bytes = reduction_chunks * staged_weight_chunk_bytes
        mbar_offset = 2 * staged_weight_bytes
        rms_offset = mbar_offset + 32
        partial_offset = rms_offset + reduction_chunks * 4
        scratch_bytes = reduction_chunks * 4 + reduction_chunks * tile_sizes["O"] * 4
        op.static_dims["reduction_tile_K"] = reduction_tile_K
        op.static_dims["reduction_chunks"] = reduction_chunks
        op.static_dims["staged_weight_chunk_bytes"] = staged_weight_chunk_bytes
        op.static_dims["staged_weight_bytes"] = staged_weight_bytes
        op.static_dims["mbar_offset"] = mbar_offset
        op.static_dims["rms_offset"] = rms_offset
        op.static_dims["partial_offset"] = partial_offset
        _set_exact_staged_page_size(op, page_size, rms_offset + scratch_bytes)
        op.static_dims["eps"] = eps
        op.static_dims["cache_pos"] = cache_pos
        op.static_dims["head_dim"] = tensors["cos"].shape[1]
        return [op]

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_O, tile_3,
             k_weight_tma, k_weight_tma_gmem,
             v_weight_tma, v_weight_tma_gmem, work_mbar):
        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        with cute.arch.elect_one():
            mbarrier_init(bf_0, Int32(1))
            mbarrier_init(bf_1, Int32(1))
            mbarrier_init(kr_0, Int32(1))
            mbarrier_init(kr_1, Int32(1))
        mbarrier_init_fence()
        with cute.arch.elect_one():
            mbarrier_arrive(bf_1)

        iter_end = (tile_3 - tile_O) * Int32(2)
        iter_idx = Int32(0)
        while iter_idx < iter_end:
            block_idx = tile_O + (iter_idx // Int32(2))
            is_v = (iter_idx & Int32(1)) != Int32(0)
            buf_idx = iter_idx % Int32(2)
            buf_base = page_ptr + buf_idx * Int32(self.staged_weight_bytes)
            if iter_idx > Int32(0):
                bf_phase = ((iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(bf_0, bf_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(bf_1, bf_phase)

            gK = cute.local_tile(k_weight_tma_gmem, (self.reduction_tile_K, self.tile_size_O), (None, None))
            gV = cute.local_tile(v_weight_tma_gmem, (self.reduction_tile_K, self.tile_size_O), (None, None))
            nbytes = Int32(self.staged_weight_bytes)
            mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
            kr_ptr = cute.make_ptr(cutlass.Int64, kr_0, cute.AddressSpace.smem)
            if iter_idx == Int32(0):
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(work_mbar, nbytes)
            if iter_idx > Int32(0):
                if buf_idx == Int32(0):
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(kr_0, nbytes)
                if buf_idx == Int32(1):
                    kr_ptr = cute.make_ptr(cutlass.Int64, kr_1, cute.AddressSpace.smem)
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(kr_1, nbytes)

            for k_chunk in range(self.reduction_chunks):
                chunk_base = buf_base + Int32(k_chunk) * Int32(self.staged_weight_chunk_bytes)
                sW = cute.make_tensor(
                    cute.make_ptr(self.k_weight_dtype, chunk_base, cute.AddressSpace.smem, assumed_align=128),
                    cute.make_layout((self.reduction_tile_K, self.tile_size_O), stride=(1, self.reduction_tile_K)),
                )
                tKsW, tKgW = cute.nvgpu.cpasync.tma_partition(
                    k_weight_tma, Int32(0), cute.make_layout(1),
                    cute.group_modes(sW, 0, 2), cute.group_modes(gK, 0, 2),
                )
                tVsW, tVgW = cute.nvgpu.cpasync.tma_partition(
                    v_weight_tma, Int32(0), cute.make_layout(1),
                    cute.group_modes(sW, 0, 2), cute.group_modes(gV, 0, 2),
                )
                if iter_idx == Int32(0):
                    if is_v:
                        cute.copy(v_weight_tma, tVgW[(None, Int32(k_chunk), block_idx)], tVsW, tma_bar_ptr=mbar_ptr)
                    else:
                        cute.copy(k_weight_tma, tKgW[(None, Int32(k_chunk), block_idx)], tKsW, tma_bar_ptr=mbar_ptr)
                if iter_idx > Int32(0):
                    if is_v:
                        cute.copy(v_weight_tma, tVgW[(None, Int32(k_chunk), block_idx)], tVsW, tma_bar_ptr=kr_ptr)
                    else:
                        cute.copy(k_weight_tma, tKgW[(None, Int32(k_chunk), block_idx)], tKsW, tma_bar_ptr=kr_ptr)

            iter_idx = iter_idx + Int32(1)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, tile_3,
                x, residual_in, norm_weight, cos, sin, k_cache, v_cache):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        row_idx = tile_S * Int32(self.tile_size_S)
        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        rstd = self._row_rstd_residual(page_ptr, tile_B, row_idx, x, residual_in)
        iter_end = (tile_3 - tile_O) * Int32(2)
        iter_idx = Int32(0)
        while iter_idx < iter_end:
            block_idx = tile_O + (iter_idx // Int32(2))
            is_v = (iter_idx & Int32(1)) != Int32(0)
            buf_idx = iter_idx % Int32(2)
            if iter_idx > Int32(0):
                kr_phase = ((iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(kr_0, kr_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(kr_1, kr_phase)
            partials = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, page_ptr + Int32(self.partial_offset), cute.AddressSpace.smem),
                cute.make_layout((self.reduction_chunks, self.tile_size_O), stride=(self.tile_size_O, 1)),
            )
            if row_idx < Int32(self.S):
                o_start = block_idx * Int32(self.tile_size_O)
                if warp_idx < Int32(self.reduction_chunks):
                    staged_weight = cute.make_tensor(
                        cute.make_ptr(
                            self.k_weight_dtype,
                            page_ptr + buf_idx * Int32(self.staged_weight_bytes) + warp_idx * Int32(self.staged_weight_chunk_bytes),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        cute.make_layout((self.reduction_tile_K, self.tile_size_O), stride=(1, self.reduction_tile_K)),
                    )
                    for local_o in range(0, self.tile_size_O, 4):
                        out_idx = o_start + Int32(local_o)
                        if out_idx < Int32(self.O):
                            partial0, partial1, partial2, partial3 = self._dot_staged_norm_residual_partial4(
                                tile_B, row_idx, Int32(local_o), warp_idx, rstd, x, residual_in, norm_weight, staged_weight
                            )
                            if lane_idx == Int32(0):
                                partials[(warp_idx, Int32(local_o))] = partial0
                                if out_idx + Int32(1) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 1))] = partial1
                                if out_idx + Int32(2) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 2))] = partial2
                                if out_idx + Int32(3) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 3))] = partial3
                named_barrier_sync(Int32(2), Int32(self.threads_per_row))
                if warp_idx == Int32(0):
                    local_o = lane_idx
                    while local_o < Int32(self.tile_size_O):
                        out_idx = o_start + local_o
                        if out_idx < Int32(self.O):
                            total = self._sum_staged_partials(page_ptr, local_o)
                            if not is_v:
                                total = self._apply_hazy_rope(page_ptr, row_idx, out_idx, local_o, total, cos, sin)
                            head = out_idx // Int32(self.head_dim)
                            dim = out_idx % Int32(self.head_dim)
                            if is_v:
                                dst_base = (
                                    tile_B * Int32(self.v_cache_stride_B)
                                    + (row_idx + Int32(self.cache_pos)) * Int32(self.v_cache_stride_T)
                                    + head * Int32(self.v_cache_stride_H)
                                )
                                dst_row = cute.make_tensor(v_cache.iterator + dst_base, cute.make_layout(self.HD))
                                dst_row[dim] = total.to(self.v_cache_dtype)
                            else:
                                dst_base = (
                                    tile_B * Int32(self.k_cache_stride_B)
                                    + (row_idx + Int32(self.cache_pos)) * Int32(self.k_cache_stride_T)
                                    + head * Int32(self.k_cache_stride_H)
                                )
                                dst_row = cute.make_tensor(k_cache.iterator + dst_base, cute.make_layout(self.HD))
                                dst_row[dim] = total.to(self.k_cache_dtype)
                        local_o = local_o + Int32(32)
            named_barrier_sync(Int32(2), Int32(self.threads_per_row))
            if tidx == Int32(0):
                if buf_idx == Int32(0):
                    mbarrier_arrive(bf_0)
                if buf_idx == Int32(1):
                    mbarrier_arrive(bf_1)
            iter_idx = iter_idx + Int32(1)


class Llama1BRmsQKVCacheSm120Op(_Llama1BStagedRmsResidualMatvecSm120Base):
    """Hazy-style staged RMS + fused Q/K/V projection, RoPE, and KV append."""

    writes = {
        "residual_out": (None, ("B", "S", "K")),
        "q": (None, ("B", "S", "Q")),
        "k_cache": (None, ("B", "T", "H", "HD")),
        "v_cache": (None, ("B", "T", "H", "HD")),
    }

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        page_size=DEFAULT_PAGE_SIZE,
        eps=1e-5,
        cache_pos=0,
        kv_group_size=4,
        **tensors,
    ):
        ops = super().schedule(
            tile_sizes=tile_sizes,
            page_size=page_size,
            eps=eps,
            cache_pos=cache_pos,
            **tensors,
        )
        op = ops[0]
        op.static_dims["cache_pos"] = cache_pos
        op.static_dims["head_dim"] = tensors["cos"].shape[1]
        op.static_dims["q_dim"] = tensors["q"].shape[2]
        op.static_dims["kv_group_size"] = kv_group_size
        op.static_dims["q_group_cols"] = kv_group_size * tensors["cos"].shape[1]
        op.static_dims["qkv_group_cols"] = (kv_group_size + 2) * tensors["cos"].shape[1]
        return ops

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, tile_3,
                x, residual_in, norm_weight, weight, cos, sin, residual_out, q, k_cache, v_cache):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        row_idx = tile_S * Int32(self.tile_size_S)
        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        rstd = self._row_rstd_residual(page_ptr, tile_B, row_idx, x, residual_in)
        self._store_residual(tile_B, row_idx, x, residual_in, residual_out)
        block_idx = tile_O
        iter_idx = Int32(0)
        while block_idx < tile_3:
            buf_idx = iter_idx % Int32(2)
            if iter_idx > Int32(0):
                kr_phase = ((iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(kr_0, kr_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(kr_1, kr_phase)
            partials = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, page_ptr + Int32(self.partial_offset), cute.AddressSpace.smem),
                cute.make_layout((self.reduction_chunks, self.tile_size_O), stride=(self.tile_size_O, 1)),
            )
            if row_idx < Int32(self.S):
                o_start = block_idx * Int32(self.tile_size_O)
                if warp_idx < Int32(self.reduction_chunks):
                    staged_weight = cute.make_tensor(
                        cute.make_ptr(
                            self.weight_dtype,
                            page_ptr + buf_idx * Int32(self.staged_weight_bytes) + warp_idx * Int32(self.staged_weight_chunk_bytes),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        cute.make_layout((self.reduction_tile_K, self.tile_size_O), stride=(1, self.reduction_tile_K)),
                    )
                    for local_o in range(0, self.tile_size_O, 4):
                        out_idx = o_start + Int32(local_o)
                        if out_idx < Int32(self.O):
                            partial0, partial1, partial2, partial3 = self._dot_staged_norm_residual_partial4(
                                tile_B, row_idx, Int32(local_o), warp_idx, rstd, x, residual_in, norm_weight, staged_weight
                            )
                            if lane_idx == Int32(0):
                                partials[(warp_idx, Int32(local_o))] = partial0
                                if out_idx + Int32(1) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 1))] = partial1
                                if out_idx + Int32(2) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 2))] = partial2
                                if out_idx + Int32(3) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 3))] = partial3
                named_barrier_sync(Int32(2), Int32(self.threads_per_row))
                if warp_idx == Int32(0):
                    local_o = lane_idx
                    while local_o < Int32(self.tile_size_O):
                        out_idx = o_start + local_o
                        if out_idx < Int32(self.O):
                            total = self._sum_staged_partials(page_ptr, local_o)
                            group = out_idx // Int32(self.qkv_group_cols)
                            group_o = out_idx - group * Int32(self.qkv_group_cols)
                            if group_o < Int32(self.q_group_cols):
                                q_idx = group * Int32(self.q_group_cols) + group_o
                                total = self._apply_hazy_rope(page_ptr, row_idx, q_idx, local_o, total, cos, sin)
                                q_base = tile_B * Int32(self.q_stride_B) + row_idx * Int32(self.q_stride_S)
                                q_row = cute.make_tensor(q.iterator + q_base, cute.make_layout(self.q_dim))
                                q_row[q_idx] = total.to(self.q_dtype)
                            elif group_o < Int32(self.q_group_cols + self.head_dim):
                                dim = group_o - Int32(self.q_group_cols)
                                total = self._apply_hazy_rope(page_ptr, row_idx, dim, local_o, total, cos, sin)
                                dst_base = (
                                    tile_B * Int32(self.k_cache_stride_B)
                                    + (row_idx + Int32(self.cache_pos)) * Int32(self.k_cache_stride_T)
                                    + group * Int32(self.k_cache_stride_H)
                                )
                                dst_row = cute.make_tensor(k_cache.iterator + dst_base, cute.make_layout(self.HD))
                                dst_row[dim] = total.to(self.k_cache_dtype)
                            else:
                                dim = group_o - Int32(self.q_group_cols + self.head_dim)
                                dst_base = (
                                    tile_B * Int32(self.v_cache_stride_B)
                                    + (row_idx + Int32(self.cache_pos)) * Int32(self.v_cache_stride_T)
                                    + group * Int32(self.v_cache_stride_H)
                                )
                                dst_row = cute.make_tensor(v_cache.iterator + dst_base, cute.make_layout(self.HD))
                                dst_row[dim] = total.to(self.v_cache_dtype)
                        local_o = local_o + Int32(32)
            named_barrier_sync(Int32(2), Int32(self.threads_per_row))
            if tidx == Int32(0):
                if buf_idx == Int32(0):
                    mbarrier_arrive(bf_0)
                if buf_idx == Int32(1):
                    mbarrier_arrive(bf_1)
            block_idx = block_idx + Int32(1)
            iter_idx = iter_idx + Int32(1)


class Llama1BRmsUpMatvecSm120Op(_Llama1BStagedRmsMatvecSm120Base):
    """Staged RMS + up projection matvec."""

    writes = {"up": (None, ("B", "S", "O"))}

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, tile_3, x, norm_weight, up):
        self._compute_rms_up(page_ptr, tile_B, tile_S, tile_O, tile_3, x, norm_weight, up)

    @cute.jit
    def _compute_rms_up(self, page_ptr, tile_B, tile_S, tile_O, tile_3, x, norm_weight, up):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        row_idx = tile_S * Int32(self.tile_size_S)
        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        rstd = self._row_rstd(page_ptr, tile_B, row_idx, x)
        block_idx = tile_O
        iter_idx = Int32(0)
        while block_idx < tile_3:
            buf_idx = iter_idx % Int32(2)
            if iter_idx > Int32(0):
                kr_phase = ((iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(kr_0, kr_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(kr_1, kr_phase)
            partials = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, page_ptr + Int32(self.partial_offset), cute.AddressSpace.smem),
                cute.make_layout((self.reduction_chunks, self.tile_size_O), stride=(self.tile_size_O, 1)),
            )
            if row_idx < Int32(self.S):
                o_start = block_idx * Int32(self.tile_size_O)
                if warp_idx < Int32(self.reduction_chunks):
                    staged_weight = cute.make_tensor(
                        cute.make_ptr(
                            self.weight_dtype,
                            page_ptr
                            + buf_idx * Int32(self.staged_weight_bytes)
                            + warp_idx * Int32(self.staged_weight_chunk_bytes),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        cute.make_layout((self.reduction_tile_K, self.tile_size_O), stride=(1, self.reduction_tile_K)),
                    )
                    for local_o in range(0, self.tile_size_O, 4):
                        out_idx = o_start + Int32(local_o)
                        if out_idx < Int32(self.O):
                            partial0, partial1, partial2, partial3 = self._dot_staged_norm_partial4(
                                tile_B, row_idx, Int32(local_o), warp_idx, rstd, x, norm_weight, staged_weight
                            )
                            if lane_idx == Int32(0):
                                partials[(warp_idx, Int32(local_o))] = partial0
                                if out_idx + Int32(1) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 1))] = partial1
                                if out_idx + Int32(2) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 2))] = partial2
                                if out_idx + Int32(3) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 3))] = partial3
                named_barrier_sync(Int32(2), Int32(self.threads_per_row))
                if warp_idx == Int32(0):
                    local_o = lane_idx
                    while local_o < Int32(self.tile_size_O):
                        out_idx = o_start + local_o
                        if out_idx < Int32(self.O):
                            total = self._sum_staged_partials(page_ptr, local_o)
                            up_base = tile_B * Int32(self.up_stride_B) + row_idx * Int32(self.up_stride_S)
                            up_row = cute.make_tensor(up.iterator + up_base, cute.make_layout(self.O))
                            up_row[out_idx] = total.to(self.up_dtype)
                        local_o = local_o + Int32(32)
            named_barrier_sync(Int32(2), Int32(self.threads_per_row))
            if tidx == Int32(0):
                if buf_idx == Int32(0):
                    mbarrier_arrive(bf_0)
                if buf_idx == Int32(1):
                    mbarrier_arrive(bf_1)
            block_idx = block_idx + Int32(1)
            iter_idx = iter_idx + Int32(1)


class Llama1BRmsGateSiluMatvecSm120Op(_Llama1BStagedRmsMatvecSm120Base):
    """Staged RMS + gate projection, combined with staged up output."""

    reads = {
        "x": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "weight": (None, ("O", "K")),
        "up": (None, ("B", "S", "O")),
    }
    writes = {"y": (None, ("B", "S", "O"))}

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, tile_3, x, norm_weight, up, y):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        row_idx = tile_S * Int32(self.tile_size_S)
        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        rstd = self._row_rstd(page_ptr, tile_B, row_idx, x)
        block_idx = tile_O
        iter_idx = Int32(0)
        while block_idx < tile_3:
            buf_idx = iter_idx % Int32(2)
            if iter_idx > Int32(0):
                kr_phase = ((iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(kr_0, kr_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(kr_1, kr_phase)
            partials = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, page_ptr + Int32(self.partial_offset), cute.AddressSpace.smem),
                cute.make_layout((self.reduction_chunks, self.tile_size_O), stride=(self.tile_size_O, 1)),
            )
            if row_idx < Int32(self.S):
                o_start = block_idx * Int32(self.tile_size_O)
                if warp_idx < Int32(self.reduction_chunks):
                    staged_weight = cute.make_tensor(
                        cute.make_ptr(
                            self.weight_dtype,
                            page_ptr
                            + buf_idx * Int32(self.staged_weight_bytes)
                            + warp_idx * Int32(self.staged_weight_chunk_bytes),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        cute.make_layout((self.reduction_tile_K, self.tile_size_O), stride=(1, self.reduction_tile_K)),
                    )
                    for local_o in range(0, self.tile_size_O, 4):
                        out_idx = o_start + Int32(local_o)
                        if out_idx < Int32(self.O):
                            partial0, partial1, partial2, partial3 = self._dot_staged_norm_partial4(
                                tile_B, row_idx, Int32(local_o), warp_idx, rstd, x, norm_weight, staged_weight
                            )
                            if lane_idx == Int32(0):
                                partials[(warp_idx, Int32(local_o))] = partial0
                                if out_idx + Int32(1) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 1))] = partial1
                                if out_idx + Int32(2) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 2))] = partial2
                                if out_idx + Int32(3) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 3))] = partial3
                named_barrier_sync(Int32(2), Int32(self.threads_per_row))
                if warp_idx == Int32(0):
                    local_o = lane_idx
                    while local_o < Int32(self.tile_size_O):
                        out_idx = o_start + local_o
                        if out_idx < Int32(self.O):
                            gate = self._sum_staged_partials(page_ptr, local_o)
                            up_base = tile_B * Int32(self.up_stride_B) + row_idx * Int32(self.up_stride_S)
                            y_base = tile_B * Int32(self.y_stride_B) + row_idx * Int32(self.y_stride_S)
                            up_row = cute.make_tensor(up.iterator + up_base, cute.make_layout(self.O))
                            y_row = cute.make_tensor(y.iterator + y_base, cute.make_layout(self.O))
                            y_row[out_idx] = (_silu(gate) * up_row[out_idx].to(Float32)).to(self.y_dtype)
                        local_o = local_o + Int32(32)
            named_barrier_sync(Int32(2), Int32(self.threads_per_row))
            if tidx == Int32(0):
                if buf_idx == Int32(0):
                    mbarrier_arrive(bf_0)
                if buf_idx == Int32(1):
                    mbarrier_arrive(bf_1)
            block_idx = block_idx + Int32(1)
            iter_idx = iter_idx + Int32(1)


class Llama1BRmsUpGateSiluSm120Op(_Llama1BStagedRmsMatvecSm120Base):
    """Hazy-style staged RMS + up/gate projections with local SiLU fusion."""

    reads = {
        "x": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "up_weight": (None, ("O", "K")),
        "gate_weight": (None, ("O", "K")),
    }
    writes = {"y": (None, ("B", "S", "O"))}
    tma_loads = {"up_weight", "gate_weight"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name in ("up_weight", "gate_weight"):
            return (
                tile_sizes["O"],
                static_dims.get("reduction_tile_K", LLAMA1B_SM120_REDUCTION_DIM_PER_WARP),
            )
        return None

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        if tensor_name in ("up_weight", "gate_weight"):
            o, k = tma_tile_shape
            return f"cute.make_layout(({o}, {k}), stride=(1, {o}))"
        return None

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, eps=1e-5, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 1)
        tile_sizes.setdefault("O", LLAMA1B_MATVEC_BLOCK)
        if tile_sizes["S"] != 1:
            raise ValueError(f"{cls.__name__} is a single-token decode matvec; got S={tile_sizes['S']}")
        if tile_sizes["O"] % 4 != 0:
            raise ValueError(f"{cls.__name__} requires O tile size divisible by 4; got O={tile_sizes['O']}")
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        weight = tensors["up_weight"]
        reduction_tile_K = min(LLAMA1B_SM120_REDUCTION_DIM_PER_WARP, weight.shape[1])
        reduction_chunks = (weight.shape[1] + reduction_tile_K - 1) // reduction_tile_K
        staged_weight_chunk_bytes = tile_sizes["O"] * reduction_tile_K * weight.element_size()
        staged_weight_bytes = reduction_chunks * staged_weight_chunk_bytes
        mbar_offset = 2 * staged_weight_bytes
        rms_offset = mbar_offset + 32
        partial_offset = rms_offset + reduction_chunks * 4
        up_cache_offset = partial_offset + reduction_chunks * tile_sizes["O"] * 4
        scratch_bytes = reduction_chunks * 4 + reduction_chunks * tile_sizes["O"] * 4 + tile_sizes["O"] * 4
        op.static_dims["reduction_tile_K"] = reduction_tile_K
        op.static_dims["reduction_chunks"] = reduction_chunks
        op.static_dims["staged_weight_chunk_bytes"] = staged_weight_chunk_bytes
        op.static_dims["staged_weight_bytes"] = staged_weight_bytes
        op.static_dims["mbar_offset"] = mbar_offset
        op.static_dims["rms_offset"] = rms_offset
        op.static_dims["partial_offset"] = partial_offset
        op.static_dims["up_cache_offset"] = up_cache_offset
        _set_exact_staged_page_size(op, page_size, rms_offset + scratch_bytes)
        op.static_dims["eps"] = eps
        return [op]

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_O, tile_3,
             up_weight_tma, up_weight_tma_gmem,
             gate_weight_tma, gate_weight_tma_gmem, work_mbar):
        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        with cute.arch.elect_one():
            mbarrier_init(bf_0, Int32(1))
            mbarrier_init(bf_1, Int32(1))
            mbarrier_init(kr_0, Int32(1))
            mbarrier_init(kr_1, Int32(1))
        mbarrier_init_fence()
        with cute.arch.elect_one():
            mbarrier_arrive(bf_1)

        iter_end = (tile_3 - tile_O) * Int32(2)
        iter_idx = Int32(0)
        while iter_idx < iter_end:
            block_idx = tile_O + (iter_idx // Int32(2))
            is_gate = (iter_idx & Int32(1)) != Int32(0)
            buf_idx = iter_idx % Int32(2)
            buf_base = page_ptr + buf_idx * Int32(self.staged_weight_bytes)
            if iter_idx > Int32(0):
                bf_phase = ((iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(bf_0, bf_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(bf_1, bf_phase)

            gUp = cute.local_tile(
                up_weight_tma_gmem,
                (self.reduction_tile_K, self.tile_size_O),
                (None, None),
            )
            gGate = cute.local_tile(
                gate_weight_tma_gmem,
                (self.reduction_tile_K, self.tile_size_O),
                (None, None),
            )
            nbytes = Int32(self.staged_weight_bytes)
            mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
            kr_ptr = cute.make_ptr(cutlass.Int64, kr_0, cute.AddressSpace.smem)
            if iter_idx == Int32(0):
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(work_mbar, nbytes)
            if iter_idx > Int32(0):
                if buf_idx == Int32(0):
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(kr_0, nbytes)
                if buf_idx == Int32(1):
                    kr_ptr = cute.make_ptr(cutlass.Int64, kr_1, cute.AddressSpace.smem)
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(kr_1, nbytes)

            for k_chunk in range(self.reduction_chunks):
                chunk_base = buf_base + Int32(k_chunk) * Int32(self.staged_weight_chunk_bytes)
                sW = cute.make_tensor(
                    cute.make_ptr(self.up_weight_dtype, chunk_base, cute.AddressSpace.smem, assumed_align=128),
                    cute.make_layout((self.reduction_tile_K, self.tile_size_O), stride=(1, self.reduction_tile_K)),
                )
                tUpsW, tUpgW = cute.nvgpu.cpasync.tma_partition(
                    up_weight_tma,
                    Int32(0),
                    cute.make_layout(1),
                    cute.group_modes(sW, 0, 2),
                    cute.group_modes(gUp, 0, 2),
                )
                tGatesW, tGategW = cute.nvgpu.cpasync.tma_partition(
                    gate_weight_tma,
                    Int32(0),
                    cute.make_layout(1),
                    cute.group_modes(sW, 0, 2),
                    cute.group_modes(gGate, 0, 2),
                )
                if iter_idx == Int32(0):
                    if is_gate:
                        cute.copy(gate_weight_tma, tGategW[(None, Int32(k_chunk), block_idx)], tGatesW, tma_bar_ptr=mbar_ptr)
                    else:
                        cute.copy(up_weight_tma, tUpgW[(None, Int32(k_chunk), block_idx)], tUpsW, tma_bar_ptr=mbar_ptr)
                if iter_idx > Int32(0):
                    if is_gate:
                        cute.copy(gate_weight_tma, tGategW[(None, Int32(k_chunk), block_idx)], tGatesW, tma_bar_ptr=kr_ptr)
                    else:
                        cute.copy(up_weight_tma, tUpgW[(None, Int32(k_chunk), block_idx)], tUpsW, tma_bar_ptr=kr_ptr)

            iter_idx = iter_idx + Int32(1)

    @cute.jit
    def _sum_partials_at(self, page_ptr, base_offset, local_o):
        partials = cute.make_tensor(
            cute.make_ptr(cutlass.Float32, page_ptr + base_offset, cute.AddressSpace.smem),
            cute.make_layout((self.reduction_chunks, self.tile_size_O), stride=(self.tile_size_O, 1)),
        )
        total = Float32(0.0)
        k_chunk = Int32(0)
        while k_chunk < Int32(self.reduction_chunks):
            total = total + partials[(k_chunk, local_o)]
            k_chunk = k_chunk + Int32(1)
        return total

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, tile_3, x, norm_weight, y):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        row_idx = tile_S * Int32(self.tile_size_S)
        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        rstd = self._row_rstd(page_ptr, tile_B, row_idx, x)
        iter_end = (tile_3 - tile_O) * Int32(2)
        iter_idx = Int32(0)
        while iter_idx < iter_end:
            block_idx = tile_O + (iter_idx // Int32(2))
            is_gate = (iter_idx & Int32(1)) != Int32(0)
            buf_idx = iter_idx % Int32(2)
            if iter_idx > Int32(0):
                kr_phase = ((iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(kr_0, kr_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(kr_1, kr_phase)

            partials = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, page_ptr + Int32(self.partial_offset), cute.AddressSpace.smem),
                cute.make_layout((self.reduction_chunks, self.tile_size_O), stride=(self.tile_size_O, 1)),
            )
            up_cache = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, page_ptr + Int32(self.up_cache_offset), cute.AddressSpace.smem),
                cute.make_layout(self.tile_size_O),
            )
            if row_idx < Int32(self.S):
                o_start = block_idx * Int32(self.tile_size_O)
                if warp_idx < Int32(self.reduction_chunks):
                    staged_weight = cute.make_tensor(
                        cute.make_ptr(
                            self.up_weight_dtype,
                            page_ptr
                            + buf_idx * Int32(self.staged_weight_bytes)
                            + warp_idx * Int32(self.staged_weight_chunk_bytes),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        cute.make_layout((self.reduction_tile_K, self.tile_size_O), stride=(1, self.reduction_tile_K)),
                    )
                    for local_o in range(0, self.tile_size_O, 4):
                        out_idx = o_start + Int32(local_o)
                        if out_idx < Int32(self.O):
                            partial0, partial1, partial2, partial3 = self._dot_staged_norm_partial4(
                                tile_B, row_idx, Int32(local_o), warp_idx, rstd, x, norm_weight, staged_weight
                            )
                            if lane_idx == Int32(0):
                                partials[(warp_idx, Int32(local_o))] = partial0
                                if out_idx + Int32(1) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 1))] = partial1
                                if out_idx + Int32(2) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 2))] = partial2
                                if out_idx + Int32(3) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 3))] = partial3
                named_barrier_sync(Int32(2), Int32(self.threads_per_row))
                if warp_idx == Int32(0):
                    local_o = lane_idx
                    while local_o < Int32(self.tile_size_O):
                        out_idx = o_start + local_o
                        if out_idx < Int32(self.O):
                            total = self._sum_partials_at(page_ptr, Int32(self.partial_offset), local_o)
                            if is_gate:
                                y_base = tile_B * Int32(self.y_stride_B) + row_idx * Int32(self.y_stride_S)
                                y_row = cute.make_tensor(y.iterator + y_base, cute.make_layout(self.O))
                                y_row[out_idx] = (_silu(total) * up_cache[local_o]).to(self.y_dtype)
                            else:
                                up_cache[local_o] = total
                        local_o = local_o + Int32(32)
            named_barrier_sync(Int32(2), Int32(self.threads_per_row))
            if tidx == Int32(0):
                if buf_idx == Int32(0):
                    mbarrier_arrive(bf_0)
                if buf_idx == Int32(1):
                    mbarrier_arrive(bf_1)
            iter_idx = iter_idx + Int32(1)


class Llama1BFinalRmsLmHeadSm120Op(_Llama1BStagedRmsMatvecSm120Base):
    """Staged final RMS + LM-head projection."""

    writes = {"logits": (None, ("B", "S", "O"))}

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, tile_3, x, norm_weight, logits):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        row_idx = tile_S * Int32(self.tile_size_S)
        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        rstd = self._row_rstd(page_ptr, tile_B, row_idx, x)
        block_idx = tile_O
        iter_idx = Int32(0)
        while block_idx < tile_3:
            buf_idx = iter_idx % Int32(2)
            if iter_idx > Int32(0):
                kr_phase = ((iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(kr_0, kr_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(kr_1, kr_phase)
            partials = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, page_ptr + Int32(self.partial_offset), cute.AddressSpace.smem),
                cute.make_layout((self.reduction_chunks, self.tile_size_O), stride=(self.tile_size_O, 1)),
            )
            if row_idx < Int32(self.S):
                o_start = block_idx * Int32(self.tile_size_O)
                if warp_idx < Int32(self.reduction_chunks):
                    staged_weight = cute.make_tensor(
                        cute.make_ptr(
                            self.weight_dtype,
                            page_ptr
                            + buf_idx * Int32(self.staged_weight_bytes)
                            + warp_idx * Int32(self.staged_weight_chunk_bytes),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        cute.make_layout((self.reduction_tile_K, self.tile_size_O), stride=(1, self.reduction_tile_K)),
                    )
                    for local_o in range(0, self.tile_size_O, 4):
                        out_idx = o_start + Int32(local_o)
                        if out_idx < Int32(self.O):
                            partial0, partial1, partial2, partial3 = self._dot_staged_norm_partial4(
                                tile_B, row_idx, Int32(local_o), warp_idx, rstd, x, norm_weight, staged_weight
                            )
                            if lane_idx == Int32(0):
                                partials[(warp_idx, Int32(local_o))] = partial0
                                if out_idx + Int32(1) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 1))] = partial1
                                if out_idx + Int32(2) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 2))] = partial2
                                if out_idx + Int32(3) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 3))] = partial3
                named_barrier_sync(Int32(2), Int32(self.threads_per_row))
                if warp_idx == Int32(0):
                    local_o = lane_idx
                    while local_o < Int32(self.tile_size_O):
                        out_idx = o_start + local_o
                        if out_idx < Int32(self.O):
                            total = self._sum_staged_partials(page_ptr, local_o)
                            logits_base = tile_B * Int32(self.logits_stride_B) + row_idx * Int32(self.logits_stride_S)
                            logits_row = cute.make_tensor(logits.iterator + logits_base, cute.make_layout(self.O))
                            logits_row[out_idx] = total.to(self.logits_dtype)
                        local_o = local_o + Int32(32)
            named_barrier_sync(Int32(2), Int32(self.threads_per_row))
            if tidx == Int32(0):
                if buf_idx == Int32(0):
                    mbarrier_arrive(bf_0)
                if buf_idx == Int32(1):
                    mbarrier_arrive(bf_1)
            block_idx = block_idx + Int32(1)
            iter_idx = iter_idx + Int32(1)


class Llama1BFinalRmsLmHeadSingleStageSm120Op(Llama1BFinalRmsLmHeadSm120Op):
    """Single-buffer final RMS + LM-head projection for larger vocab tiles."""

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        page_size=LLAMA1B_SM120_LARGE_STAGED_PAGE_SIZE,
        eps=1e-5,
        reduction_tile_K=None,
        **tensors,
    ):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 1)
        tile_sizes.setdefault("O", LLAMA1B_MATVEC_BLOCK)
        if tile_sizes["S"] != 1:
            raise ValueError(f"{cls.__name__} is a single-token decode matvec; got S={tile_sizes['S']}")
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        weight = tensors["weight"]
        if reduction_tile_K is None:
            reduction_tile_K = LLAMA1B_SM120_REDUCTION_DIM_PER_WARP
        reduction_tile_K = min(reduction_tile_K, weight.shape[1])
        reduction_chunks = (weight.shape[1] + reduction_tile_K - 1) // reduction_tile_K
        staged_weight_chunk_bytes = tile_sizes["O"] * reduction_tile_K * weight.element_size()
        staged_weight_bytes = reduction_chunks * staged_weight_chunk_bytes
        mbar_offset = staged_weight_bytes
        rms_offset = mbar_offset + 16
        partial_offset = rms_offset + reduction_chunks * 4
        scratch_bytes = reduction_chunks * 4 + reduction_chunks * tile_sizes["O"] * 4
        op.static_dims["reduction_tile_K"] = reduction_tile_K
        op.static_dims["reduction_chunks"] = reduction_chunks
        op.static_dims["staged_weight_chunk_bytes"] = staged_weight_chunk_bytes
        op.static_dims["staged_weight_bytes"] = staged_weight_bytes
        op.static_dims["mbar_offset"] = mbar_offset
        op.static_dims["rms_offset"] = rms_offset
        op.static_dims["partial_offset"] = partial_offset
        _set_exact_staged_page_size(op, page_size, partial_offset + scratch_bytes)
        op.static_dims["eps"] = eps
        return [op]

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_O, tile_3, weight_tma, weight_tma_gmem, work_mbar):
        bf = page_ptr + Int32(self.mbar_offset)
        kr = page_ptr + Int32(self.mbar_offset + 8)
        with cute.arch.elect_one():
            mbarrier_init(bf, Int32(1))
            mbarrier_init(kr, Int32(1))
        mbarrier_init_fence()
        with cute.arch.elect_one():
            mbarrier_arrive(bf)

        block_idx = tile_O
        iter_idx = Int32(0)
        while block_idx < tile_3:
            if iter_idx > Int32(0):
                mbarrier_wait(bf, (iter_idx - Int32(1)) % Int32(2))
            gW = cute.local_tile(weight_tma_gmem, (self.reduction_tile_K, self.tile_size_O), (None, None))
            nbytes = Int32(self.staged_weight_bytes)
            mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
            if iter_idx == Int32(0):
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(work_mbar, nbytes)
            else:
                mbar_ptr = cute.make_ptr(cutlass.Int64, kr, cute.AddressSpace.smem)
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(kr, nbytes)

            for k_chunk in range(self.reduction_chunks):
                chunk_base = page_ptr + Int32(k_chunk) * Int32(self.staged_weight_chunk_bytes)
                sW = cute.make_tensor(
                    cute.make_ptr(self.weight_dtype, chunk_base, cute.AddressSpace.smem, assumed_align=128),
                    cute.make_layout((self.reduction_tile_K, self.tile_size_O), stride=(1, self.reduction_tile_K)),
                )
                tWsW, tWgW = cute.nvgpu.cpasync.tma_partition(
                    weight_tma,
                    Int32(0),
                    cute.make_layout(1),
                    cute.group_modes(sW, 0, 2),
                    cute.group_modes(gW, 0, 2),
                )
                cute.copy(weight_tma, tWgW[(None, Int32(k_chunk), block_idx)], tWsW, tma_bar_ptr=mbar_ptr)
            block_idx = block_idx + Int32(1)
            iter_idx = iter_idx + Int32(1)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_O, tile_3, x, norm_weight, logits):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        row_idx = tile_S * Int32(self.tile_size_S)
        bf = page_ptr + Int32(self.mbar_offset)
        kr = page_ptr + Int32(self.mbar_offset + 8)

        rstd = self._row_rstd(page_ptr, tile_B, row_idx, x)
        block_idx = tile_O
        iter_idx = Int32(0)
        while block_idx < tile_3:
            if iter_idx > Int32(0):
                mbarrier_wait(kr, (iter_idx - Int32(1)) % Int32(2))
            partials = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, page_ptr + Int32(self.partial_offset), cute.AddressSpace.smem),
                cute.make_layout((self.reduction_chunks, self.tile_size_O), stride=(self.tile_size_O, 1)),
            )
            if row_idx < Int32(self.S):
                o_start = block_idx * Int32(self.tile_size_O)
                if warp_idx < Int32(self.reduction_chunks):
                    staged_weight = cute.make_tensor(
                        cute.make_ptr(
                            self.weight_dtype,
                            page_ptr + warp_idx * Int32(self.staged_weight_chunk_bytes),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        cute.make_layout((self.reduction_tile_K, self.tile_size_O), stride=(1, self.reduction_tile_K)),
                    )
                    for local_o in range(0, self.tile_size_O, 4):
                        out_idx = o_start + Int32(local_o)
                        if out_idx < Int32(self.O):
                            partial0, partial1, partial2, partial3 = self._dot_staged_norm_partial4(
                                tile_B, row_idx, Int32(local_o), warp_idx, rstd, x, norm_weight, staged_weight
                            )
                            if lane_idx == Int32(0):
                                partials[(warp_idx, Int32(local_o))] = partial0
                                if out_idx + Int32(1) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 1))] = partial1
                                if out_idx + Int32(2) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 2))] = partial2
                                if out_idx + Int32(3) < Int32(self.O):
                                    partials[(warp_idx, Int32(local_o + 3))] = partial3
                named_barrier_sync(Int32(2), Int32(self.threads_per_row))
                if warp_idx == Int32(0):
                    local_o = lane_idx
                    while local_o < Int32(self.tile_size_O):
                        out_idx = o_start + local_o
                        if out_idx < Int32(self.O):
                            total = self._sum_staged_partials(page_ptr, local_o)
                            logits_base = tile_B * Int32(self.logits_stride_B) + row_idx * Int32(self.logits_stride_S)
                            logits_row = cute.make_tensor(logits.iterator + logits_base, cute.make_layout(self.O))
                            logits_row[out_idx] = total.to(self.logits_dtype)
                        local_o = local_o + Int32(32)
            named_barrier_sync(Int32(2), Int32(self.threads_per_row))
            if tidx == Int32(0):
                mbarrier_arrive(bf)
            block_idx = block_idx + Int32(1)
            iter_idx = iter_idx + Int32(1)


class Llama1BDecodeAttentionSm120Op(Op):
    """Single-token GQA decode attention for Llama-1B.

    Hazy's Llama attention instruction is scheduled per KV head and computes
    the four GQA query heads together. This scalar-CUTE version keeps the
    standard online softmax math, but schedules one tile per KV head with four
    active query-head warps.
    """

    reads = {
        "q": (None, ("B", "M", "H", "D")),
        "k": (None, ("B", "N", "H_kv", "D")),
        "v": (None, ("B", "N", "H_kv", "D")),
    }
    writes = {"o": (None, ("B", "M", "H", "D"))}
    tile = ("B", "H_kv")
    dynamic_dims = ("B", "N")

    @classmethod
    def kernel_config(cls, ops):
        from machete.megakernel import MegakernelConfig

        return MegakernelConfig(
            threads_per_block=LLAMA1B_CONSUMER_WARPS * 32,
            page_size=DEFAULT_PAGE_SIZE,
            mma_reg_count=96,
        )

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, kv_group_size=4, **tensors):
        q = tensors["q"]
        k = tensors["k"]
        if q.shape[1] != 1:
            raise ValueError(f"{cls.__name__} only supports decode M=1, got M={q.shape[1]}")
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("H_kv", 1)
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["kv_group_size"] = kv_group_size
        op.static_dims["scale"] = 1.0 / math.sqrt(q.shape[-1])
        op.static_dims["H"] = q.shape[2]
        op.static_dims["N"] = k.shape[1]
        op.static_dims["q_b_stride"] = q.stride(0)
        op.static_dims["q_m_stride"] = q.stride(1)
        op.static_dims["q_h_stride"] = q.stride(2)
        op.static_dims["k_b_stride"] = k.stride(0)
        op.static_dims["k_n_stride"] = k.stride(1)
        op.static_dims["k_h_stride"] = k.stride(2)
        op.static_dims["v_b_stride"] = tensors["v"].stride(0)
        op.static_dims["v_n_stride"] = tensors["v"].stride(1)
        op.static_dims["v_h_stride"] = tensors["v"].stride(2)
        op.static_dims["o_b_stride"] = tensors["o"].stride(0)
        op.static_dims["o_m_stride"] = tensors["o"].stride(1)
        op.static_dims["o_h_stride"] = tensors["o"].stride(2)
        return [op]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_H_kv, q, k, v, o):
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        q_h = tile_H_kv * Int32(self.kv_group_size) + warp_idx
        kv_h = tile_H_kv

        if warp_idx < Int32(self.kv_group_size) and q_h < Int32(self.H):
            q_base = (
                tile_B * Int32(self.q_b_stride)
                + q_h * Int32(self.q_h_stride)
            )
            k_head = (
                k.iterator
                + tile_B * Int32(self.k_b_stride)
                + kv_h * Int32(self.k_h_stride)
            )
            v_head = (
                v.iterator
                + tile_B * Int32(self.v_b_stride)
                + kv_h * Int32(self.v_h_stride)
            )
            q_row = cute.make_tensor(q.iterator + q_base, cute.make_layout(self.D))

            d0 = lane_idx
            d1 = lane_idx + Int32(32)
            out0 = Float32(0.0)
            out1 = Float32(0.0)
            m = Float32(-3.4028234663852886e38)
            denom = Float32(0.0)

            n = Int32(0)
            while n < Int32(self.N):
                k_row = cute.make_tensor(
                    k_head + n * Int32(self.k_n_stride),
                    cute.make_layout(self.D),
                )
                v_row = cute.make_tensor(
                    v_head + n * Int32(self.v_n_stride),
                    cute.make_layout(self.D),
                )

                partial = Float32(0.0)
                d = lane_idx
                while d < Int32(self.D):
                    partial = partial + q_row[d].to(Float32) * k_row[d].to(Float32)
                    d = d + Int32(32)
                score = cute.arch.warp_reduction(partial, operator.add) * Float32(self.scale)

                new_m = cute.arch.fmax(m, score)
                old_scale = cute.math.exp(m - new_m, fastmath=True)
                cur_scale = cute.math.exp(score - new_m, fastmath=True)
                if d0 < Int32(self.D):
                    out0 = out0 * old_scale + v_row[d0].to(Float32) * cur_scale
                if d1 < Int32(self.D):
                    out1 = out1 * old_scale + v_row[d1].to(Float32) * cur_scale
                denom = denom * old_scale + cur_scale
                m = new_m
                n = n + Int32(1)

            inv = cute.arch.rcp_approx(denom)
            o_base = (
                tile_B * Int32(self.o_b_stride)
                + q_h * Int32(self.o_h_stride)
            )
            o_row = cute.make_tensor(o.iterator + o_base, cute.make_layout(self.D))
            if d0 < Int32(self.D):
                o_row[d0] = (out0 * inv).to(self.o_dtype)
            if d1 < Int32(self.D):
                o_row[d1] = (out1 * inv).to(self.o_dtype)


class Llama1BDecodeAttentionPartialSm120Op(Op):
    """Grouped GQA decode attention over one KV split.

    One tile owns one KV head and one split of the KV cache. Four active warps
    compute the four GQA query heads and write normalized fp32 partial O plus a
    natural-log LSE for the reduction op.
    """

    reads = {
        "q": (None, ("B", "M", "H", "D")),
        "k": (None, ("B", "N", "H_kv", "D")),
        "v": (None, ("B", "N", "H_kv", "D")),
    }
    writes = {
        "o_partial": (cutlass.Float32, ("B", "H", "SPLIT", "D")),
        "lse_partial": (cutlass.Float32, ("B", "H", "SPLIT")),
    }
    tile = ("B", "H_kv", "SPLIT")
    dynamic_dims = ("B", "N", "SPLIT")

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, kv_group_size=4, num_splits=1, **tensors):
        import torch

        q = tensors["q"]
        k = tensors["k"]
        if q.shape[1] != 1:
            raise ValueError(f"{cls.__name__} only supports decode M=1, got M={q.shape[1]}")
        num_splits = max(1, min(int(num_splits), int(k.shape[1])))
        bsz, _m, q_heads, head_dim = q.shape
        o_partial = torch.empty(bsz, q_heads, num_splits, head_dim, device=q.device, dtype=torch.float32)
        lse_partial = torch.empty(bsz, q_heads, num_splits, device=q.device, dtype=torch.float32)

        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("H_kv", 1)
        tile_sizes.setdefault("SPLIT", 1)
        tensors["o_partial"] = o_partial
        tensors["lse_partial"] = lse_partial
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["kv_group_size"] = kv_group_size
        op.static_dims["num_splits"] = num_splits
        op.static_dims["tokens_per_split"] = (k.shape[1] + num_splits - 1) // num_splits
        op.static_dims["scale"] = 1.0 / math.sqrt(q.shape[-1])
        op.static_dims["H"] = q.shape[2]
        op.static_dims["N"] = k.shape[1]
        op.static_dims["q_b_stride"] = q.stride(0)
        op.static_dims["q_h_stride"] = q.stride(2)
        op.static_dims["k_b_stride"] = k.stride(0)
        op.static_dims["k_n_stride"] = k.stride(1)
        op.static_dims["k_h_stride"] = k.stride(2)
        op.static_dims["v_b_stride"] = tensors["v"].stride(0)
        op.static_dims["v_n_stride"] = tensors["v"].stride(1)
        op.static_dims["v_h_stride"] = tensors["v"].stride(2)
        return [op], o_partial, lse_partial

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_H_kv, tile_SPLIT, q, k, v, o_partial, lse_partial):
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        q_h = tile_H_kv * Int32(self.kv_group_size) + warp_idx
        kv_h = tile_H_kv

        if warp_idx < Int32(self.kv_group_size) and q_h < Int32(self.H):
            q_base = tile_B * Int32(self.q_b_stride) + q_h * Int32(self.q_h_stride)
            k_head = k.iterator + tile_B * Int32(self.k_b_stride) + kv_h * Int32(self.k_h_stride)
            v_head = v.iterator + tile_B * Int32(self.v_b_stride) + kv_h * Int32(self.v_h_stride)
            q_row = cute.make_tensor(q.iterator + q_base, cute.make_layout(self.D))

            d0 = lane_idx
            d1 = lane_idx + Int32(32)
            out0 = Float32(0.0)
            out1 = Float32(0.0)
            m = Float32(-3.4028234663852886e38)
            denom = Float32(0.0)
            n = tile_SPLIT * Int32(self.tokens_per_split)
            n_end = n + Int32(self.tokens_per_split)
            if n_end > Int32(self.N):
                n_end = Int32(self.N)

            while n < n_end:
                k_row = cute.make_tensor(k_head + n * Int32(self.k_n_stride), cute.make_layout(self.D))
                v_row = cute.make_tensor(v_head + n * Int32(self.v_n_stride), cute.make_layout(self.D))

                partial = Float32(0.0)
                d = lane_idx
                while d < Int32(self.D):
                    partial = partial + q_row[d].to(Float32) * k_row[d].to(Float32)
                    d = d + Int32(32)
                score = cute.arch.warp_reduction(partial, operator.add) * Float32(self.scale)

                new_m = cute.arch.fmax(m, score)
                old_scale = cute.math.exp(m - new_m, fastmath=True)
                cur_scale = cute.math.exp(score - new_m, fastmath=True)
                if d0 < Int32(self.D):
                    out0 = out0 * old_scale + v_row[d0].to(Float32) * cur_scale
                if d1 < Int32(self.D):
                    out1 = out1 * old_scale + v_row[d1].to(Float32) * cur_scale
                denom = denom * old_scale + cur_scale
                m = new_m
                n = n + Int32(1)

            inv = cute.arch.rcp_approx(denom)
            o_base = (
                o_partial.iterator
                + tile_B * Int32(self.H * self.num_splits * self.D)
                + q_h * Int32(self.num_splits * self.D)
                + tile_SPLIT * Int32(self.D)
            )
            o_row = cute.make_tensor(o_base, cute.make_layout(self.D))
            if denom > Float32(0.0):
                if d0 < Int32(self.D):
                    o_row[d0] = out0 * inv
                if d1 < Int32(self.D):
                    o_row[d1] = out1 * inv
                if lane_idx == Int32(0):
                    lse_offset = (
                        tile_B * Int32(self.H * self.num_splits)
                        + q_h * Int32(self.num_splits)
                        + tile_SPLIT
                    )
                    lse_row = cute.make_tensor(lse_partial.iterator + lse_offset, cute.make_layout(1))
                    lse_row[Int32(0)] = m + cute.math.log(denom)
            else:
                if d0 < Int32(self.D):
                    o_row[d0] = Float32(0.0)
                if d1 < Int32(self.D):
                    o_row[d1] = Float32(0.0)
                if lane_idx == Int32(0):
                    lse_offset = (
                        tile_B * Int32(self.H * self.num_splits)
                        + q_h * Int32(self.num_splits)
                        + tile_SPLIT
                    )
                    lse_row = cute.make_tensor(lse_partial.iterator + lse_offset, cute.make_layout(1))
                    lse_row[Int32(0)] = Float32(-3.4028234663852886e38)


class Llama1BDecodeAttentionReductionSm120Op(Op):
    """Reduce Hazy-style attention partials into the final BSHD attention output."""

    reads = {
        "o_partial": (cutlass.Float32, ("B", "H", "SPLIT", "D")),
        "lse_partial": (cutlass.Float32, ("B", "H", "SPLIT")),
    }
    writes = {"o": (None, ("B", "M", "H", "D"))}
    tile = ("B", "H")
    dynamic_dims = ("B", "SPLIT")

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, kv_group_size=4, **tensors):
        o_partial = tensors["o_partial"]
        o = tensors["o"]
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("H", kv_group_size)
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["page_size"] = page_size
        op.static_dims["kv_group_size"] = kv_group_size
        op.static_dims["H"] = o_partial.shape[1]
        op.static_dims["SPLIT"] = o_partial.shape[2]
        op.static_dims["D"] = o_partial.shape[3]
        op.static_dims["o_b_stride"] = o.stride(0)
        op.static_dims["o_h_stride"] = o.stride(2)
        return [op]

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_H, o_partial, lse_partial, o):
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        q_h = tile_H * Int32(self.tile_size_H) + warp_idx

        if warp_idx < Int32(self.kv_group_size) and q_h < Int32(self.H):
            d0 = lane_idx
            d1 = lane_idx + Int32(32)
            lse_max = Float32(-3.4028234663852886e38)
            split = Int32(0)
            while split < Int32(self.SPLIT):
                lse_offset = (
                    tile_B * Int32(self.H * self.SPLIT)
                    + q_h * Int32(self.SPLIT)
                    + split
                )
                lse_row = cute.make_tensor(lse_partial.iterator + lse_offset, cute.make_layout(1))
                lse_val = lse_row[Int32(0)]
                lse_max = cute.arch.fmax(lse_max, lse_val)
                split = split + Int32(1)

            acc0 = Float32(0.0)
            acc1 = Float32(0.0)
            scale_sum = Float32(0.0)
            split = Int32(0)
            while split < Int32(self.SPLIT):
                lse_offset = (
                    tile_B * Int32(self.H * self.SPLIT)
                    + q_h * Int32(self.SPLIT)
                    + split
                )
                lse_row = cute.make_tensor(lse_partial.iterator + lse_offset, cute.make_layout(1))
                lse_val = lse_row[Int32(0)]
                scale = cute.math.exp(lse_val - lse_max, fastmath=True)
                o_base = (
                    o_partial.iterator
                    + tile_B * Int32(self.H * self.SPLIT * self.D)
                    + q_h * Int32(self.SPLIT * self.D)
                    + split * Int32(self.D)
                )
                o_row = cute.make_tensor(o_base, cute.make_layout(self.D))
                if d0 < Int32(self.D):
                    acc0 = acc0 + scale * o_row[d0]
                if d1 < Int32(self.D):
                    acc1 = acc1 + scale * o_row[d1]
                scale_sum = scale_sum + scale
                split = split + Int32(1)

            inv = cute.arch.rcp_approx(scale_sum)
            out_base = tile_B * Int32(self.o_b_stride) + q_h * Int32(self.o_h_stride)
            out_row = cute.make_tensor(o.iterator + out_base, cute.make_layout(self.D))
            if d0 < Int32(self.D):
                out_row[d0] = (acc0 * inv).to(self.o_dtype)
            if d1 < Int32(self.D):
                out_row[d1] = (acc1 * inv).to(self.o_dtype)


def schedule_llama1b_decode_attention_sm120(
    *,
    q,
    k,
    v,
    o,
    kv_group_size=4,
    page_size=DEFAULT_PAGE_SIZE,
    num_splits=0,
):
    if num_splits and num_splits > 1:
        ops, o_partial, lse_partial = Llama1BDecodeAttentionPartialSm120Op.schedule(
            q=q,
            k=k,
            v=v,
            kv_group_size=kv_group_size,
            num_splits=num_splits,
            page_size=page_size,
        )
        ops += Llama1BDecodeAttentionReductionSm120Op.schedule(
            o_partial=o_partial,
            lse_partial=lse_partial,
            o=o,
            kv_group_size=kv_group_size,
            page_size=page_size,
        )
        return ops, [o_partial, lse_partial]
    return Llama1BDecodeAttentionSm120Op.schedule(
        q=q,
        k=k,
        v=v,
        o=o,
        kv_group_size=kv_group_size,
        page_size=page_size,
    ), []


def _annotate_llama1b_interleaved_qkv_head_barriers(*, layer_idx, qkv_op, attn_ops, head_dim, kv_group_size):
    qkv_alias = f"llama1b_layer_{layer_idx}_interleaved_qkv_head"
    qkv_op.dim_aliases["O"] = qkv_alias
    qkv_op.static_dims["barrier_tile_size_O"] = LLAMA1B_SM120_QKV_HEAD_BLOCK
    for op in attn_ops:
        if "H_kv" not in op.dim_names:
            continue
        op.static_dims.update(
            {
                "barrier_wait_q_alias_H_kv": qkv_alias,
                "barrier_wait_q_tile_size_H_kv": (kv_group_size + 2) * head_dim,
                "barrier_wait_k_alias_H_kv": qkv_alias,
                "barrier_wait_k_tile_size_H_kv": (kv_group_size + 2) * head_dim,
                "barrier_wait_v_alias_H_kv": qkv_alias,
                "barrier_wait_v_tile_size_H_kv": (kv_group_size + 2) * head_dim,
            }
        )


def _interleave_llama1b_qkv_weight(*, w_q, w_k, w_v, num_kv_heads, head_dim, kv_group_size):
    chunks = []
    q_group_cols = kv_group_size * head_dim
    for kv_head in range(num_kv_heads):
        q_start = kv_head * q_group_cols
        q_stop = q_start + q_group_cols
        k_start = kv_head * head_dim
        k_stop = k_start + head_dim
        chunks.extend(
            (
                w_q[q_start:q_stop],
                w_k[k_start:k_stop],
                w_v[k_start:k_stop],
            )
        )
    return torch.cat(chunks, dim=0)


def schedule_decode_layer_sm120(
    *,
    layer_idx,
    batch,
    seq_len,
    cache_pos,
    weights,
    k_cache,
    v_cache,
    x_in,
    residual_in,
    x_out,
    residual_out,
    q_buf,
    attn_out_buf,
    mlp_h_buf,
    page_size=DEFAULT_PAGE_SIZE,
    eps=1e-5,
    fa_num_splits=0,
    hidden_size=LLAMA1B_HIDDEN,
    intermediate_size=LLAMA1B_INTERMEDIATE,
    num_q_heads=LLAMA1B_Q_DIM // LLAMA1B_HEAD_DIM,
    num_kv_heads=LLAMA1B_KV_DIM // LLAMA1B_HEAD_DIM,
    head_dim=LLAMA1B_HEAD_DIM,
    kv_group_size=(LLAMA1B_Q_DIM // LLAMA1B_HEAD_DIM) // (LLAMA1B_KV_DIM // LLAMA1B_HEAD_DIM),
    matvec_block=LLAMA1B_SM120_MATVEC_BLOCK,
    split_upgate=True,
):
    """Schedule one Llama-1B decode layer with staged SM120 O/down matvecs."""
    _validate_staged_page_size(page_size)

    pfx = f"layer.{layer_idx}"
    cos = weights["cos"][cache_pos : cache_pos + seq_len]
    sin = weights["sin"][cache_pos : cache_pos + seq_len]
    q_4d = q_buf.view(batch, seq_len, num_q_heads, head_dim)
    k_window = k_cache[:, : cache_pos + seq_len]
    v_window = v_cache[:, : cache_pos + seq_len]
    o_4d = attn_out_buf.view(batch, seq_len, num_q_heads, head_dim)

    ops = []
    keep_alive = []
    qkv_head_barrier_op = None
    if fa_num_splits and fa_num_splits > 1:
        qkv_block = LLAMA1B_SM120_QKV_HEAD_BLOCK
        qkv_weight = _interleave_llama1b_qkv_weight(
            w_q=weights[f"{pfx}.W_q"],
            w_k=weights[f"{pfx}.W_k"],
            w_v=weights[f"{pfx}.W_v"],
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_group_size=kv_group_size,
        )
        keep_alive.append(qkv_weight)
        qkv_ops = Llama1BRmsQKVCacheSm120Op.schedule(
            x=x_in,
            residual_in=residual_in,
            norm_weight=weights[f"{pfx}.attn_norm"],
            weight=qkv_weight,
            cos=cos,
            sin=sin,
            residual_out=residual_out,
            q=q_buf,
            k_cache=k_window,
            v_cache=v_window,
            cache_pos=cache_pos,
            tile_sizes={"S": seq_len, "O": qkv_block},
            page_size=page_size,
            eps=eps,
            kv_group_size=kv_group_size,
        )
        ops += qkv_ops
        qkv_head_barrier_op = qkv_ops[0]
    else:
        ops += Llama1BRmsQSm120Op.schedule(
            x=x_in,
            residual_in=residual_in,
            norm_weight=weights[f"{pfx}.attn_norm"],
            weight=weights[f"{pfx}.W_q"],
            cos=cos,
            sin=sin,
            residual_out=residual_out,
            q=q_buf,
            tile_sizes={"S": seq_len, "O": matvec_block},
            page_size=page_size,
            eps=eps,
        )
        ops += Llama1BRmsKCacheSm120Op.schedule(
            x=x_in,
            residual_in=residual_in,
            norm_weight=weights[f"{pfx}.attn_norm"],
            weight=weights[f"{pfx}.W_k"],
            cos=cos,
            sin=sin,
            dst_cache=k_window,
            cache_pos=cache_pos,
            tile_sizes={"S": seq_len, "O": matvec_block},
            page_size=page_size,
            eps=eps,
        )
        ops += Llama1BRmsVCacheSm120Op.schedule(
            x=x_in,
            residual_in=residual_in,
            norm_weight=weights[f"{pfx}.attn_norm"],
            weight=weights[f"{pfx}.W_v"],
            cos=cos,
            sin=sin,
            dst_cache=v_window,
            cache_pos=cache_pos,
            tile_sizes={"S": seq_len, "O": matvec_block},
            page_size=page_size,
            eps=eps,
        )

    attn_ops, attn_keep = schedule_llama1b_decode_attention_sm120(
        q=q_4d,
        k=k_window,
        v=v_window,
        o=o_4d,
        kv_group_size=kv_group_size,
        page_size=page_size,
        num_splits=fa_num_splits,
    )
    if qkv_head_barrier_op is not None:
        _annotate_llama1b_interleaved_qkv_head_barriers(
            layer_idx=layer_idx,
            qkv_op=qkv_head_barrier_op,
            attn_ops=attn_ops,
            head_dim=head_dim,
            kv_group_size=kv_group_size,
        )
    ops += attn_ops

    ops += Llama1BMatvecResidualSm120Op.schedule(
        a=attn_out_buf,
        weight=weights[f"{pfx}.W_o"],
        residual_in=residual_out,
        residual_out=residual_out,
        tile_sizes={"S": seq_len, "O": matvec_block},
        page_size=page_size,
    )
    down_keep = []
    upgate_keep = []
    num_down_blocks = intermediate_size // hidden_size
    down_partial_buf = None
    if num_down_blocks == 4:
        down_partial_buf = x_out.new_empty((num_down_blocks - 1, batch, seq_len, hidden_size))
        down_keep.append(down_partial_buf)

    for reduction_block in range(num_down_blocks):
        start = reduction_block * hidden_size
        stop = start + hidden_size
        a_block = mlp_h_buf[:, :, start:stop]
        w_block = weights[f"{pfx}.W_down"][:, start:stop]
        if split_upgate:
            up_block = weights[f"{pfx}.W_up"][start:stop]
            gate_block = weights[f"{pfx}.W_gate"][start:stop]
            upgate_keep += [up_block, gate_block, a_block]
            ops += Llama1BRmsUpGateSiluSm120Op.schedule(
                x=residual_out,
                norm_weight=weights[f"{pfx}.mlp_norm"],
                up_weight=up_block,
                gate_weight=gate_block,
                y=a_block,
                tile_sizes={"S": seq_len, "O": matvec_block},
                page_size=page_size,
                eps=eps,
            )
        down_keep += [a_block, w_block]
        if down_partial_buf is not None:
            y_block = x_out if reduction_block == 0 else down_partial_buf[reduction_block - 1]
            down_keep.append(y_block)
            ops += Llama1BDownMatvecSm120Op.schedule(
                a=a_block,
                weight=w_block,
                y=y_block,
                tile_sizes={"S": seq_len, "O": matvec_block},
                page_size=page_size,
            )
        else:
            if reduction_block == 0:
                ops += Llama1BDownMatvecSm120Op.schedule(
                    a=a_block,
                    weight=w_block,
                    y=x_out,
                    tile_sizes={"S": seq_len, "O": matvec_block},
                    page_size=page_size,
                )
            else:
                ops += Llama1BMatvecResidualSm120Op.schedule(
                    a=a_block,
                    weight=w_block,
                    residual_in=x_out,
                    residual_out=x_out,
                    tile_sizes={"S": seq_len, "O": matvec_block},
                    page_size=page_size,
                )

    if not split_upgate:
        upgate_ops = Llama1BRmsUpGateSiluSm120Op.schedule(
            x=residual_out,
            norm_weight=weights[f"{pfx}.mlp_norm"],
            up_weight=weights[f"{pfx}.W_up"],
            gate_weight=weights[f"{pfx}.W_gate"],
            y=mlp_h_buf,
            tile_sizes={"S": seq_len, "O": matvec_block},
            page_size=page_size,
            eps=eps,
        )
        insert_at = len(ops) - (intermediate_size // hidden_size)
        ops[insert_at:insert_at] = upgate_ops

    if down_partial_buf is not None:
        ops += Llama1BDownAdd4Sm120Op.schedule(
            p0=x_out,
            p1=down_partial_buf[0],
            p2=down_partial_buf[1],
            p3=down_partial_buf[2],
            y=x_out,
            tile_sizes={"S": seq_len, "D": 256},
        )

    return Llama1BLayerSchedule(
        ops=ops,
        attention_config=None,
        keep_alive=[cos, sin, q_4d, k_window, v_window, o_4d, *keep_alive, *attn_keep, *upgate_keep, *down_keep],
    )


def schedule_final_sm120(
    *,
    seq_len,
    x,
    residual,
    residual_out,
    norm_weight,
    lm_head,
    logits,
    page_size=DEFAULT_PAGE_SIZE,
    eps=1e-5,
    matvec_block=LLAMA1B_MATVEC_BLOCK,
    reduction_tile_K=None,
):
    _validate_staged_page_size(page_size)

    ops = Llama1BResidualAddSm120Op.schedule(
        x=x,
        residual_in=residual,
        residual_out=residual_out,
        tile_sizes={"S": seq_len, "K": 256},
        page_size=page_size,
    )
    final_op = Llama1BFinalRmsLmHeadSingleStageSm120Op if matvec_block > 12 else Llama1BFinalRmsLmHeadSm120Op
    ops += final_op.schedule(
        x=residual_out,
        norm_weight=norm_weight,
        weight=lm_head,
        logits=logits,
        tile_sizes={"S": seq_len, "O": matvec_block},
        page_size=page_size,
        eps=eps,
        reduction_tile_K=reduction_tile_K,
    )
    return ops


def schedule_decode_model_sm120(
    *,
    batch,
    cache_pos,
    weights,
    k_cache,
    v_cache,
    x_buffers,
    residual_buffers,
    q_buf,
    attn_out_buf,
    mlp_h_buf,
    final_norm=None,
    lm_head=None,
    logits=None,
    num_layers=16,
    page_size=DEFAULT_PAGE_SIZE,
    eps=1e-5,
    matvec_block=LLAMA1B_SM120_MATVEC_BLOCK,
    final_matvec_block=LLAMA1B_SM120_FINAL_MATVEC_BLOCK,
    fa_num_splits=-1,
    split_upgate=True,
):
    """Schedule a full single-token Llama-1B SM120 decode forward pass."""
    _validate_staged_page_size(page_size)

    if len(x_buffers) < 2 or len(residual_buffers) < 2:
        raise ValueError("x_buffers and residual_buffers must each contain two ping-pong buffers")
    ops = []
    keep_alive = []
    seq_len = 1
    if fa_num_splits < 0:
        cache_len = cache_pos + seq_len
        if cache_len < 384:
            fa_num_splits = 8
        else:
            fa_num_splits = min(64, max(16, cache_len // 8))
    cur = 0
    for layer_idx in range(num_layers):
        nxt = 1 - cur
        layer = schedule_decode_layer_sm120(
            layer_idx=layer_idx,
            batch=batch,
            seq_len=seq_len,
            cache_pos=cache_pos,
            weights=weights,
            k_cache=k_cache[layer_idx],
            v_cache=v_cache[layer_idx],
            x_in=x_buffers[cur],
            residual_in=residual_buffers[cur],
            x_out=x_buffers[nxt],
            residual_out=residual_buffers[nxt],
            q_buf=q_buf,
            attn_out_buf=attn_out_buf,
            mlp_h_buf=mlp_h_buf,
            page_size=page_size,
            eps=eps,
            matvec_block=matvec_block,
            fa_num_splits=fa_num_splits,
            split_upgate=split_upgate,
        )
        ops += layer.ops
        keep_alive += layer.keep_alive
        cur = nxt

    if final_norm is not None and lm_head is not None and logits is not None:
        ops += schedule_final_sm120(
            seq_len=seq_len,
            x=x_buffers[cur],
            residual=residual_buffers[cur],
            residual_out=residual_buffers[cur],
            norm_weight=final_norm,
            lm_head=lm_head,
            logits=logits,
            page_size=page_size,
            eps=eps,
            matvec_block=final_matvec_block,
        )

    return Llama1BLayerSchedule(ops=ops, attention_config=None, keep_alive=keep_alive)


__all__ = [
    "LLAMA1B_HIDDEN",
    "LLAMA1B_HEAD_DIM",
    "LLAMA1B_ROTARY_D2",
    "LLAMA1B_Q_DIM",
    "LLAMA1B_KV_DIM",
    "LLAMA1B_INTERMEDIATE",
    "LLAMA1B_VOCAB",
    "LLAMA1B_MATVEC_BLOCK",
    "LLAMA1B_SM120_MATVEC_BLOCK",
    "LLAMA1B_SM120_FINAL_MATVEC_BLOCK",
    "LLAMA1B_SM120_CONSUMER_WARPS",
    "LLAMA1B_SM120_REDUCTION_DIM_PER_WARP",
    "LLAMA1B_SM120_THREADS_PER_BLOCK",
    "LLAMA1B_CONSUMER_WARPS",
    "Llama1BDownAdd4Sm120Op",
    "Llama1BDownMatvecSm120Op",
    "Llama1BDecodeAttentionSm120Op",
    "Llama1BDecodeAttentionPartialSm120Op",
    "Llama1BDecodeAttentionReductionSm120Op",
    "Llama1BFinalRmsLmHeadSm120Op",
    "Llama1BFinalRmsLmHeadSingleStageSm120Op",
    "Llama1BMatvecResidualSm120Op",
    "Llama1BResidualAddSm120Op",
    "Llama1BRmsGateSiluMatvecSm120Op",
    "Llama1BRmsKCacheSm120Op",
    "Llama1BRmsKVCacheSm120Op",
    "Llama1BRmsQKVCacheSm120Op",
    "Llama1BRmsQSm120Op",
    "Llama1BRmsUpGateSiluSm120Op",
    "Llama1BRmsUpMatvecSm120Op",
    "Llama1BRmsVCacheSm120Op",
    "Llama1BLayerSchedule",
    "schedule_decode_layer_sm120",
    "schedule_decode_model_sm120",
    "schedule_final_sm120",
    "schedule_llama1b_decode_attention_sm120",
]
