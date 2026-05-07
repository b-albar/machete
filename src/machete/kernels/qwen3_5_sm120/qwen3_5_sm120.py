# Copyright (c) 2025, Machete Authors
"""Qwen 3.5 decode-oriented SM120 megakernel ops.

This module is intentionally Qwen-shaped instead of reusing the Llama-1B
decode matvec schedule directly.  Qwen 3.5 applies attention RMS before the
Q/K/V projections, then per-head Q/K RMSNorm and partial RoPE after projection.
The K/V cache is native BSHD, with K normalized/rotated before cache storage and
V copied from the packed KV scratch buffer.
"""

import operator
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from machete.kernels.gemm import GemmOp
from machete.kernels.gemm.gemm import _gemm_epilogue_store_no_mbar_inval_helper
from machete.kernels.qknorm_rope import QKNormRopeOp as _BaseQKNormRopeOp
from machete.kernels.qknorm_rope.qknorm_rope import CopyBulkS2GOp, group_bulk_copy_modes
from machete.megakernel.interpreter import (
    mbarrier_arrive,
    mbarrier_arrive_expect_tx,
    mbarrier_inval,
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_wait,
    named_barrier_sync,
)
from machete.megakernel.ops import (
    DEFAULT_PAGE_SIZE,
    PipelineABI,
    PipelineSpec,
    Op,
    config_dim_i32,
)


QWEN3_5_NUM_LAYERS = 36
QWEN3_5_HIDDEN = 1024
QWEN3_5_INTERMEDIATE = 3584
QWEN3_5_NUM_Q_HEADS = 8
QWEN3_5_NUM_KV_HEADS = 2
QWEN3_5_HEAD_DIM = 256
QWEN3_5_ROTARY_DIM = 64
QWEN3_5_ROTARY_D2 = QWEN3_5_ROTARY_DIM // 2
QWEN3_5_Q_DIM = QWEN3_5_NUM_Q_HEADS * QWEN3_5_HEAD_DIM
QWEN3_5_KV_DIM = QWEN3_5_NUM_KV_HEADS * QWEN3_5_HEAD_DIM
QWEN3_5_KV_GROUP_SIZE = QWEN3_5_NUM_Q_HEADS // QWEN3_5_NUM_KV_HEADS
QWEN3_5_EPS = 1e-6
QWEN3_5_VOCAB = 151936
QWEN3_5_DECODE_S = 16
QWEN3_5_GATE_UP_TILE = {"S": 16, "N": 128, "K": 32}
QWEN3_5_REDUCTION_DIM_PER_WARP = 512


class Qwen3_5StagedDecodeGemmSm120Op(GemmOp):
    """Decode GEMM with loader-owned staged K streaming.

    The load warp streams every K block and the MMA warps consume the
    double-buffered stream through op-local mbarriers.
    """

    pipeline = PipelineSpec.streaming(
        range_axis=2,
        range_block_size=16,
    )
    pipeline_abi = PipelineABI.op_owned()

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_N,
             a_tma, a_tma_gmem, a_scale_tma, a_scale_tma_gmem,
             b_tma, b_tma_gmem,
             work_mbar):
        swz = cute.make_swizzle(self.swz_B_ab, 4, 3)

        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        with cute.arch.elect_one():
            mbarrier_init(bf_0, Int32(self.num_mma_warps))
            mbarrier_init(bf_1, Int32(self.num_mma_warps))
            mbarrier_init(kr_0, Int32(1))
            mbarrier_init(kr_1, Int32(1))
        mbarrier_init_fence()

        k_block = Int32(0)
        while k_block < Int32(self.num_k_blocks):
            buf_idx = k_block % Int32(2)
            buf_base = page_ptr + buf_idx * Int32(self.buf_stride)

            if k_block >= Int32(2):
                bf_phase = ((k_block - Int32(2)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(bf_0, bf_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(bf_1, bf_phase)

            sA_ptr = cute.recast_ptr(
                cute.make_ptr(self.a_dtype, buf_base, cute.AddressSpace.smem),
                swz,
                dtype=self.a_dtype,
            )
            sA = cute.make_tensor(
                sA_ptr,
                cute.make_layout(
                    (self.tile_K, self.tile_size_S, 1),
                    stride=(1, self.tile_K, self.tile_K * self.tile_size_S),
                ),
            )
            gA = cute.local_tile(
                a_tma_gmem,
                (self.tile_K, self.tile_size_S, 1),
                (None, None, None),
            )
            tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                a_tma,
                Int32(0),
                cute.make_layout(1),
                cute.group_modes(sA, 0, 3),
                cute.group_modes(gA, 0, 3),
            )

            sScale_ptr = cute.recast_ptr(
                cute.make_ptr(
                    self.a_scale_dtype,
                    buf_base + Int32(self.a_scale_offset),
                    cute.AddressSpace.smem,
                ),
                swz,
                dtype=self.a_scale_dtype,
            )
            sScale = cute.make_tensor(
                sScale_ptr,
                cute.make_layout(
                    (self.tile_K, self.tile_size_S, 1),
                    stride=(1, self.tile_K, self.tile_K * self.tile_size_S),
                ),
            )
            gScale = cute.local_tile(
                a_scale_tma_gmem,
                (self.tile_K, self.tile_size_S, 1),
                (None, None, None),
            )
            tScaleS, tScaleG = cute.nvgpu.cpasync.tma_partition(
                a_scale_tma,
                Int32(0),
                cute.make_layout(1),
                cute.group_modes(sScale, 0, 3),
                cute.group_modes(gScale, 0, 3),
            )

            sB_ptr = cute.recast_ptr(
                cute.make_ptr(
                    self.b_dtype,
                    buf_base + Int32(self.b_offset),
                    cute.AddressSpace.smem,
                ),
                swz,
                dtype=self.b_dtype,
            )
            sB = cute.make_tensor(
                sB_ptr,
                cute.make_layout((self.tile_K, self.tile_size_N), stride=(1, self.tile_K)),
            )
            gB = cute.local_tile(
                b_tma_gmem,
                (self.tile_K, self.tile_size_N),
                (None, None),
            )
            tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                b_tma,
                Int32(0),
                cute.make_layout(1),
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB, 0, 2),
            )

            if k_block < Int32(2):
                mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
                if k_block == Int32(0):
                    nbytes = Int32(self.tma_k_blocks * self.ab_tma_bytes)
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(work_mbar, nbytes)
                cute.copy(a_tma, tAgA[(None, k_block, tile_S, tile_B)], tAsA, tma_bar_ptr=mbar_ptr)
                if self.has_a_scale:
                    cute.copy(
                        a_scale_tma,
                        tScaleG[(None, k_block, tile_S, tile_B)],
                        tScaleS,
                        tma_bar_ptr=mbar_ptr,
                    )
                cute.copy(b_tma, tBgB[(None, k_block, tile_N)], tBsB, tma_bar_ptr=mbar_ptr)

            if k_block >= Int32(2):
                kr_ptr = cute.make_ptr(cutlass.Int64, kr_0, cute.AddressSpace.smem)
                if buf_idx == Int32(0):
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(kr_0, Int32(self.ab_tma_bytes))
                if buf_idx == Int32(1):
                    kr_ptr = cute.make_ptr(cutlass.Int64, kr_1, cute.AddressSpace.smem)
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(kr_1, Int32(self.ab_tma_bytes))
                cute.copy(a_tma, tAgA[(None, k_block, tile_S, tile_B)], tAsA, tma_bar_ptr=kr_ptr)
                if self.has_a_scale:
                    cute.copy(
                        a_scale_tma,
                        tScaleG[(None, k_block, tile_S, tile_B)],
                        tScaleS,
                        tma_bar_ptr=kr_ptr,
                    )
                cute.copy(b_tma, tBgB[(None, k_block, tile_N)], tBsB, tma_bar_ptr=kr_ptr)

            k_block = k_block + Int32(1)


class Qwen3_5LmHeadGemmSm120Op(Qwen3_5StagedDecodeGemmSm120Op):
    """Separate GEMM handler family for the Qwen final lm_head projection."""

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("S", QWEN3_5_DECODE_S)
        tile_sizes.setdefault("N", 128)
        tile_sizes.setdefault("K", 32)
        vocab = tensors["b"].shape[0]
        if vocab % tile_sizes["N"] != 0:
            raise ValueError(
                f"{cls.__name__} requires the vocab dimension ({vocab}) to be divisible "
                f"by tile N ({tile_sizes['N']})"
            )
        return super().schedule(tile_sizes=tile_sizes, page_size=page_size, **tensors)


class Qwen3_5RMSAddStagedDecodeGemmSm120Op(Qwen3_5StagedDecodeGemmSm120Op):
    """Fused residual add + RMSNorm feeding a staged decode GEMM.

    The loader warp materializes normalized A tiles in shared memory while also
    writing ``residual_out``. It then TMA-streams B tiles exactly like the staged
    decode GEMM path, so MMA compute overlaps with subsequent weight loads.
    """

    reads = {
        "a": (None, ("B", "S", "K")),
        "residual_in": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "b": (None, ("N", "K")),
    }
    writes = {
        "residual_out": (None, ("B", "S", "K")),
        "c": (None, ("B", "S", "N")),
    }
    tma_loads = {"b"}
    tma_stores = {"c"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        tile_K = static_dims.get("tile_K", 32)
        if tensor_name == "b":
            return (tile_sizes["N"], tile_K)
        if tensor_name == "c":
            return (1, tile_sizes["S"], tile_sizes["N"])
        return None

    def __init__(self, **config):
        super().__init__(**config)
        self.rms_scratch_offset = self.mbar_offset + 32
        self.rms_scratch_bytes = self.tile_size_S * 4
        assert self.rms_scratch_offset + self.rms_scratch_bytes <= self.page_size, (
            f"{type(self).__name__}: smem {self.rms_scratch_offset + self.rms_scratch_bytes}B "
            f"exceeds page_size ({self.page_size}B)."
        )

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, eps=QWEN3_5_EPS, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", QWEN3_5_DECODE_S)
        tile_sizes.setdefault("N", 128)
        tile_K = tile_sizes.pop("K", 32)
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        op.static_dims["tile_K"] = tile_K
        op.static_dims["page_size"] = page_size
        op.static_dims["eps"] = eps
        op.static_dims["has_a_scale"] = 0
        return [op]

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_N,
             a, residual_in, norm_weight, residual_out,
             b_tma, b_tma_gmem, work_mbar):
        swz = cute.make_swizzle(self.swz_B_ab, 4, 3)

        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        with cute.arch.elect_one():
            mbarrier_init(bf_0, Int32(self.num_mma_warps))
            mbarrier_init(bf_1, Int32(self.num_mma_warps))
            mbarrier_init(kr_0, Int32(1))
            mbarrier_init(kr_1, Int32(1))
        mbarrier_init_fence()

        lane_idx = cute.arch.lane_idx()
        row_start = tile_S * Int32(self.tile_size_S)
        rstd_scratch = cute.make_tensor(
            cute.make_ptr(
                cutlass.Float32,
                page_ptr + Int32(self.rms_scratch_offset),
                cute.AddressSpace.smem,
            ),
            cute.make_layout(self.tile_size_S),
        )

        for local_row in range(self.tile_size_S):
            row_idx = row_start + Int32(local_row)
            partial = Float32(0.0)
            base = tile_B * Int32(self.a_stride_B) + row_idx * Int32(self.a_stride_S)
            res_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S)
            out_base = tile_B * Int32(self.residual_out_stride_B) + row_idx * Int32(self.residual_out_stride_S)
            a_row = cute.make_tensor(a.iterator + base, cute.make_layout(self.K))
            res_row = cute.make_tensor(residual_in.iterator + res_base, cute.make_layout(self.K))
            out_row = cute.make_tensor(residual_out.iterator + out_base, cute.make_layout(self.K))
            k = lane_idx
            while k < Int32(self.K):
                val = a_row[k].to(Float32) + res_row[k].to(Float32)
                partial = partial + val * val
                out_row[k] = val.to(self.residual_out_dtype)
                k = k + Int32(32)
            total = cute.arch.warp_reduction(partial, operator.add)
            if lane_idx == Int32(0):
                rstd_scratch[local_row] = cute.math.rsqrt(
                    total * Float32(1.0 / self.K) + Float32(self.eps),
                    fastmath=True,
                )

        k_block = Int32(0)
        while k_block < Int32(self.num_k_blocks):
            buf_idx = k_block % Int32(2)
            buf_base = page_ptr + buf_idx * Int32(self.buf_stride)

            if k_block >= Int32(2):
                bf_phase = ((k_block - Int32(2)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(bf_0, bf_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(bf_1, bf_phase)

            sA = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.a_dtype, buf_base, cute.AddressSpace.smem),
                    swz,
                    dtype=self.a_dtype,
                ),
                cute.make_layout(
                    (self.tile_K, self.tile_size_S, 1),
                    stride=(1, self.tile_K, self.tile_K * self.tile_size_S),
                ),
            )
            for local_row in range(self.tile_size_S):
                row_idx = row_start + Int32(local_row)
                base = tile_B * Int32(self.a_stride_B) + row_idx * Int32(self.a_stride_S)
                res_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S)
                a_row = cute.make_tensor(a.iterator + base, cute.make_layout(self.K))
                res_row = cute.make_tensor(residual_in.iterator + res_base, cute.make_layout(self.K))
                w_row = cute.make_tensor(norm_weight.iterator, cute.make_layout(self.K))
                local_k = lane_idx
                while local_k < Int32(self.tile_K):
                    k = k_block * Int32(self.tile_K) + local_k
                    if k < Int32(self.K):
                        val = (
                            (a_row[k].to(Float32) + res_row[k].to(Float32))
                            * rstd_scratch[local_row]
                            * w_row[k].to(Float32)
                        )
                        sA[(local_k, local_row, Int32(0))] = val.to(self.a_dtype)
                    local_k = local_k + Int32(32)

            sB_ptr = cute.recast_ptr(
                cute.make_ptr(
                    self.b_dtype,
                    buf_base + Int32(self.b_offset),
                    cute.AddressSpace.smem,
                ),
                swz,
                dtype=self.b_dtype,
            )
            sB = cute.make_tensor(
                sB_ptr,
                cute.make_layout((self.tile_K, self.tile_size_N), stride=(1, self.tile_K)),
            )
            gB = cute.local_tile(
                b_tma_gmem,
                (self.tile_K, self.tile_size_N),
                (None, None),
            )
            tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                b_tma,
                Int32(0),
                cute.make_layout(1),
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB, 0, 2),
            )

            if k_block < Int32(2):
                mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
                if k_block == Int32(0):
                    nbytes = Int32(self.tma_k_blocks * self.b_tile_bytes)
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(work_mbar, nbytes)
                cute.copy(b_tma, tBgB[(None, k_block, tile_N)], tBsB, tma_bar_ptr=mbar_ptr)

            if k_block >= Int32(2):
                kr_ptr = cute.make_ptr(cutlass.Int64, kr_0, cute.AddressSpace.smem)
                if buf_idx == Int32(0):
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(kr_0, Int32(self.b_tile_bytes))
                if buf_idx == Int32(1):
                    kr_ptr = cute.make_ptr(cutlass.Int64, kr_1, cute.AddressSpace.smem)
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(kr_1, Int32(self.b_tile_bytes))
                cute.copy(b_tma, tBgB[(None, k_block, tile_N)], tBsB, tma_bar_ptr=kr_ptr)

            k_block = k_block + Int32(1)


class Qwen3_5RangedLmHeadSm120Op(Op):
    """Decode LM-head matvec with staged TMA weight tiles.

    Each instruction owns one 16-vocab block. The load warp stages the
    corresponding ``weight[N, K]`` tile into shared memory via TMA; compute
    warps then reuse that staged tile for all decode rows.
    """

    pipeline = PipelineSpec.streaming(
        range_axis=2,
        range_end_axis=3,
        range_block_size=1,
        coalesce_ranges=True,
    )
    pipeline_abi = PipelineABI.op_owned()
    reads = {
        "h": (None, ("B", "S", "K")),
        "weight": (None, ("N", "K")),
    }
    writes = {"logits": (None, ("B", "S", "N"))}
    tile = ("B", "S", "N")
    dynamic_dims = ("B",)
    tma_loads = {"weight"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name == "weight":
            return (tile_sizes["N"], static_dims["K"])
        return None

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        if tensor_name == "weight":
            k, n = tma_tile_shape
            return f"cute.make_layout(({k}, {n}), stride=(1, {k}))"
        return None

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", QWEN3_5_DECODE_S)
        tile_sizes.setdefault("N", 16)
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        weight = tensors["weight"]
        staged_bytes = tile_sizes["N"] * weight.shape[1] * weight.element_size()
        required = 2 * staged_bytes + 32
        page_size = max(page_size, required)
        op.static_dims["staged_weight_bytes"] = staged_bytes
        op.static_dims["mbar_offset"] = 2 * staged_bytes
        op.static_dims["page_size"] = page_size
        return [op]

    @cute.jit
    def load(self, page_ptr, tile_N, tile_3, weight_tma, weight_tma_gmem, work_mbar):
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

        block_idx = tile_N
        iter_idx = Int32(0)
        while block_idx < tile_3:
            buf_idx = iter_idx % Int32(2)
            buf_base = page_ptr + buf_idx * Int32(self.staged_weight_bytes)

            if iter_idx > Int32(0):
                bf_phase = ((iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(bf_0, bf_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(bf_1, bf_phase)

            sW = cute.make_tensor(
                cute.make_ptr(self.weight_dtype, buf_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.K, self.tile_size_N), stride=(1, self.K)),
            )
            gW = cute.local_tile(
                weight_tma_gmem,
                (self.K, self.tile_size_N),
                (None, None),
            )
            tWsW, tWgW = cute.nvgpu.cpasync.tma_partition(
                weight_tma,
                Int32(0),
                cute.make_layout(1),
                cute.group_modes(sW, 0, 2),
                cute.group_modes(gW, 0, 2),
            )
            nbytes = Int32(self.staged_weight_bytes)
            if iter_idx == Int32(0):
                mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(work_mbar, nbytes)
                cute.copy(weight_tma, tWgW[(None, Int32(0), block_idx)], tWsW, tma_bar_ptr=mbar_ptr)
            if iter_idx > Int32(0):
                kr_ptr = cute.make_ptr(cutlass.Int64, kr_0, cute.AddressSpace.smem)
                if buf_idx == Int32(0):
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(kr_0, nbytes)
                if buf_idx == Int32(1):
                    kr_ptr = cute.make_ptr(cutlass.Int64, kr_1, cute.AddressSpace.smem)
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(kr_1, nbytes)
                cute.copy(weight_tma, tWgW[(None, Int32(0), block_idx)], tWsW, tma_bar_ptr=kr_ptr)

            block_idx = block_idx + Int32(1)
            iter_idx = iter_idx + Int32(1)

    @cute.jit
    def _dot_vocab(self, tile_B, row_idx, local_v, h, staged_weight):
        lane_idx = cute.arch.lane_idx()
        h_base = tile_B * Int32(self.h_stride_B) + row_idx * Int32(self.h_stride_S)
        h_row = cute.make_tensor(h.iterator + h_base, cute.make_layout(self.K))
        acc = Float32(0.0)
        k = lane_idx
        while k < Int32(self.K):
            h_val = h_row[k].to(Float32)
            w_val = staged_weight[(k, local_v)].to(Float32)
            acc = acc + h_val * w_val
            k = k + Int32(32)
        return cute.arch.warp_reduction(acc, operator.add)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_N, tile_3, h, logits):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        block_idx = tile_N
        iter_idx = Int32(0)
        while block_idx < tile_3:
            buf_idx = iter_idx % Int32(2)
            if iter_idx > Int32(0):
                kr_phase = ((iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(kr_0, kr_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(kr_1, kr_phase)

            staged_weight = cute.make_tensor(
                cute.make_ptr(
                    self.weight_dtype,
                    page_ptr + buf_idx * Int32(self.staged_weight_bytes),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                cute.make_layout((self.K, self.tile_size_N), stride=(1, self.K)),
            )

            for local_row in range(warp_idx, self.tile_size_S, num_warps):
                row_idx = row_start + Int32(local_row)
                if row_idx < Int32(self.S):
                    vocab_start = block_idx * Int32(self.tile_size_N)
                    for local_v in range(self.tile_size_N):
                        vocab_idx = vocab_start + Int32(local_v)
                        if vocab_idx < Int32(self.N):
                            total = self._dot_vocab(
                                tile_B,
                                row_idx,
                                Int32(local_v),
                                h,
                                staged_weight,
                            )
                            if lane_idx == Int32(0):
                                logits_base = (
                                    tile_B * Int32(self.logits_stride_B)
                                    + row_idx * Int32(self.logits_stride_S)
                                )
                                logits_row = cute.make_tensor(
                                    logits.iterator + logits_base,
                                    cute.make_layout(self.N),
                                )
                                logits_row[vocab_idx] = total.to(self.logits_dtype)

            named_barrier_sync(Int32(2), Int32(self.threads_per_row))
            if tidx == Int32(0):
                if buf_idx == Int32(0):
                    mbarrier_arrive(bf_0)
                if buf_idx == Int32(1):
                    mbarrier_arrive(bf_1)
            block_idx = block_idx + Int32(1)
            iter_idx = iter_idx + Int32(1)


class Qwen3_5Top1LmHeadSm120Op(Qwen3_5RangedLmHeadSm120Op):
    """Streaming decode LM head that returns top-1 per row without logits."""

    pipeline = PipelineSpec.streaming(
        range_axis=2,
        range_end_axis=3,
        range_block_size=0,
        coalesce_ranges=True,
    )
    reads = {
        "h": (None, ("B", "S", "K")),
        "weight": (None, ("N", "K")),
    }
    writes = {
        "top_values": (None, ("B", "S")),
        "top_indices": (cutlass.Int32, ("B", "S")),
    }

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", 1)
        tile_sizes.setdefault("N", 16)
        ops = super().schedule(tile_sizes=tile_sizes, page_size=page_size, **tensors)
        op = ops[0]
        top_scratch_offset = op.static_dims["mbar_offset"] + 32
        # The reduction scratch below is addressed as two fixed 32-entry arrays:
        # float values followed by int32 indices.
        top_scratch_bytes = 32 * 8
        op.static_dims["top_scratch_offset"] = top_scratch_offset
        op.static_dims["page_size"] = max(
            op.static_dims["page_size"],
            top_scratch_offset + top_scratch_bytes,
        )
        return ops

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_N, tile_3, h, top_values, top_indices):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        row_idx = row_start
        best_val = Float32(-3.4028234663852886e38)
        best_idx = Int32(0)
        block_idx = tile_N
        iter_idx = Int32(0)
        while block_idx < tile_3:
            buf_idx = iter_idx % Int32(2)
            if iter_idx > Int32(0):
                kr_phase = ((iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(kr_0, kr_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(kr_1, kr_phase)

            staged_weight = cute.make_tensor(
                cute.make_ptr(
                    self.weight_dtype,
                    page_ptr + buf_idx * Int32(self.staged_weight_bytes),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                cute.make_layout((self.K, self.tile_size_N), stride=(1, self.K)),
            )

            if row_idx < Int32(self.S):
                vocab_start = block_idx * Int32(self.tile_size_N)
                local_v = warp_idx
                while local_v < Int32(self.tile_size_N):
                    vocab_idx = vocab_start + local_v
                    if vocab_idx < Int32(self.N):
                        total = self._dot_vocab(
                            tile_B,
                            row_idx,
                            local_v,
                            h,
                            staged_weight,
                        )
                        if total > best_val:
                            best_val = total
                            best_idx = vocab_idx
                    local_v = local_v + num_warps

            named_barrier_sync(Int32(2), Int32(self.threads_per_row))
            if tidx == Int32(0):
                if buf_idx == Int32(0):
                    mbarrier_arrive(bf_0)
                if buf_idx == Int32(1):
                    mbarrier_arrive(bf_1)
            block_idx = block_idx + Int32(1)
            iter_idx = iter_idx + Int32(1)

        scratch_val = cute.make_tensor(
            cute.make_ptr(
                cutlass.Float32,
                page_ptr + Int32(self.top_scratch_offset),
                cute.AddressSpace.smem,
            ),
            cute.make_layout(32),
        )
        scratch_idx = cute.make_tensor(
            cute.make_ptr(
                cutlass.Int32,
                page_ptr + Int32(self.top_scratch_offset + 32 * 4),
                cute.AddressSpace.smem,
            ),
            cute.make_layout(32),
        )
        if lane_idx == Int32(0):
            scratch_val[warp_idx] = best_val
            scratch_idx[warp_idx] = best_idx
        named_barrier_sync(Int32(2), Int32(self.threads_per_row))

        if tidx == Int32(0) and row_idx < Int32(self.S):
            final_best_val = Float32(-3.4028234663852886e38)
            final_best_idx = Int32(0)
            w = Int32(0)
            while w < num_warps:
                val = scratch_val[w]
                idx = scratch_idx[w]
                if val > final_best_val:
                    final_best_val = val
                    final_best_idx = idx
                w = w + Int32(1)
            top_values_row = cute.make_tensor(
                top_values.iterator + tile_B * Int32(self.top_values_stride_B) + row_idx,
                cute.make_layout(1),
            )
            top_indices_row = cute.make_tensor(
                top_indices.iterator + tile_B * Int32(self.top_indices_stride_B) + row_idx,
                cute.make_layout(1),
            )
            top_values_row[Int32(0)] = final_best_val.to(self.top_values_dtype)
            top_indices_row[Int32(0)] = final_best_idx

        if tidx == Int32(0):
            mbarrier_inval(bf_0)
            mbarrier_inval(bf_1)
            mbarrier_inval(kr_0)
            mbarrier_inval(kr_1)


class Qwen3_5RMSAddRangedDecodeMatvecSm120Op(Op):
    """Ranged residual-add RMSNorm + projection with normalized activation reuse."""

    pipeline = PipelineSpec.streaming(
        range_axis=2,
        range_end_axis=3,
        range_block_size=1,
        coalesce_ranges=True,
    )
    pipeline_abi = PipelineABI.op_owned()
    reads = {
        "a": (None, ("B", "S", "K")),
        "residual_in": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "b": (None, ("N", "K")),
    }
    writes = {
        "residual_out": (None, ("B", "S", "K")),
        "c": (None, ("B", "S", "N")),
    }
    tile = ("B", "S", "N")
    dynamic_dims = ("B", "S")
    tma_loads = {"b"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name == "b":
            return (tile_sizes["N"], static_dims["K"])
        return None

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        if tensor_name == "b":
            k, n = tma_tile_shape
            return f"cute.make_layout(({k}, {n}), stride=(1, {k}))"
        return None

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, eps=QWEN3_5_EPS, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", QWEN3_5_DECODE_S)
        tile_sizes.setdefault("N", 16)
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        a = tensors["a"]
        b = tensors["b"]
        elem_bytes = a.element_size()
        norm_bytes = tile_sizes["S"] * a.shape[-1] * elem_bytes
        staged_weight_bytes = tile_sizes["N"] * b.shape[1] * b.element_size()
        mbar_offset = norm_bytes + 2 * staged_weight_bytes
        required = mbar_offset + 32
        page_size = max(page_size, required)
        op.static_dims["norm_bytes"] = norm_bytes
        op.static_dims["staged_weight_bytes"] = staged_weight_bytes
        op.static_dims["mbar_offset"] = mbar_offset
        op.static_dims["page_size"] = page_size
        op.static_dims["eps"] = eps
        return [op]

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_N, tile_3,
             a, residual_in, norm_weight, residual_out,
             b_tma, b_tma_gmem, op_config_ptr, work_mbar):
        runtime_S = config_dim_i32(op_config_ptr, "S", type(self))
        lane_idx = cute.arch.lane_idx()
        row_start = tile_S * Int32(self.tile_size_S)
        norm_ptr = cute.make_ptr(self.a_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128)

        for local_row in range(self.tile_size_S):
            row_idx = row_start + Int32(local_row)
            if row_idx < runtime_S:
                partial = Float32(0.0)
                base = tile_B * Int32(self.a_stride_B) + row_idx * Int32(self.a_stride_S)
                res_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S)
                out_base = tile_B * Int32(self.residual_out_stride_B) + row_idx * Int32(self.residual_out_stride_S)
                a_row = cute.make_tensor(a.iterator + base, cute.make_layout(self.K))
                res_row = cute.make_tensor(residual_in.iterator + res_base, cute.make_layout(self.K))
                out_row = cute.make_tensor(residual_out.iterator + out_base, cute.make_layout(self.K))
                k = lane_idx
                while k < Int32(self.K):
                    val = a_row[k].to(Float32) + res_row[k].to(Float32)
                    partial = partial + val * val
                    out_row[k] = val.to(self.residual_out_dtype)
                    k = k + Int32(32)
                total = cute.arch.warp_reduction(partial, operator.add)
                rstd = cute.math.rsqrt(
                    total * Float32(1.0 / self.K) + Float32(self.eps),
                    fastmath=True,
                )
                w_row = cute.make_tensor(norm_weight.iterator, cute.make_layout(self.K))
                norm_row = cute.make_tensor(
                    norm_ptr + Int32(local_row * self.K),
                    cute.make_layout(self.K),
                )
                k2 = lane_idx
                while k2 < Int32(self.K):
                    val = (a_row[k2].to(Float32) + res_row[k2].to(Float32)) * rstd * w_row[k2].to(Float32)
                    norm_row[k2] = val.to(self.a_dtype)
                    k2 = k2 + Int32(32)

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

        block_idx = tile_N
        iter_idx = Int32(0)
        while block_idx < tile_3:
            buf_idx = iter_idx % Int32(2)
            buf_base = page_ptr + Int32(self.norm_bytes) + buf_idx * Int32(self.staged_weight_bytes)
            if iter_idx > Int32(0):
                bf_phase = ((iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(bf_0, bf_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(bf_1, bf_phase)
            sB = cute.make_tensor(
                cute.make_ptr(self.b_dtype, buf_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.K, self.tile_size_N), stride=(1, self.K)),
            )
            gB = cute.local_tile(b_tma_gmem, (self.K, self.tile_size_N), (None, None))
            tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                b_tma,
                Int32(0),
                cute.make_layout(1),
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB, 0, 2),
            )
            nbytes = Int32(self.staged_weight_bytes)
            if iter_idx == Int32(0):
                mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(work_mbar, nbytes)
                cute.copy(b_tma, tBgB[(None, Int32(0), block_idx)], tBsB, tma_bar_ptr=mbar_ptr)
            if iter_idx > Int32(0):
                kr_ptr = cute.make_ptr(cutlass.Int64, kr_0, cute.AddressSpace.smem)
                if buf_idx == Int32(0):
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(kr_0, nbytes)
                if buf_idx == Int32(1):
                    kr_ptr = cute.make_ptr(cutlass.Int64, kr_1, cute.AddressSpace.smem)
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(kr_1, nbytes)
                cute.copy(b_tma, tBgB[(None, Int32(0), block_idx)], tBsB, tma_bar_ptr=kr_ptr)
            block_idx = block_idx + Int32(1)
            iter_idx = iter_idx + Int32(1)

    @cute.jit
    def _dot(self, local_row, local_n, norm_act, staged_weight):
        lane_idx = cute.arch.lane_idx()
        h_row = cute.make_tensor(norm_act + Int32(local_row * self.K), cute.make_layout(self.K))
        acc = Float32(0.0)
        k = lane_idx
        while k < Int32(self.K):
            acc = acc + h_row[k].to(Float32) * staged_weight[(k, local_n)].to(Float32)
            k = k + Int32(32)
        return cute.arch.warp_reduction(acc, operator.add)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_N, tile_3, c, op_config_ptr):
        runtime_S = config_dim_i32(op_config_ptr, "S", type(self))
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        norm_act = cute.make_ptr(self.a_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128)
        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        block_idx = tile_N
        iter_idx = Int32(0)
        while block_idx < tile_3:
            buf_idx = iter_idx % Int32(2)
            if iter_idx > Int32(0):
                kr_phase = ((iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                if buf_idx == Int32(0):
                    mbarrier_wait(kr_0, kr_phase)
                if buf_idx == Int32(1):
                    mbarrier_wait(kr_1, kr_phase)
            staged_weight = cute.make_tensor(
                cute.make_ptr(
                    self.b_dtype,
                    page_ptr + Int32(self.norm_bytes) + buf_idx * Int32(self.staged_weight_bytes),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                cute.make_layout((self.K, self.tile_size_N), stride=(1, self.K)),
            )
            for local_row in range(warp_idx, self.tile_size_S, num_warps):
                row_idx = row_start + Int32(local_row)
                if row_idx < runtime_S:
                    n_start = block_idx * Int32(self.tile_size_N)
                    for local_n in range(self.tile_size_N):
                        n_idx = n_start + Int32(local_n)
                        if n_idx < Int32(self.N):
                            total = self._dot(Int32(local_row), Int32(local_n), norm_act, staged_weight)
                            if lane_idx == Int32(0):
                                c_base = tile_B * Int32(self.c_stride_B) + row_idx * Int32(self.c_stride_S)
                                c_row = cute.make_tensor(c.iterator + c_base, cute.make_layout(self.N))
                                c_row[n_idx] = total.to(self.c_dtype)
            named_barrier_sync(Int32(2), Int32(self.threads_per_row))
            if tidx == Int32(0):
                if buf_idx == Int32(0):
                    mbarrier_arrive(bf_0)
                if buf_idx == Int32(1):
                    mbarrier_arrive(bf_1)
            block_idx = block_idx + Int32(1)
            iter_idx = iter_idx + Int32(1)


class Qwen3_5RMSAddRangedDecodeGemmSm120Op(Op):
    """Tensor-core ranged residual-add RMSNorm + projection.

    This is the fast version of the ranged decode op: the load warp computes
    normalized activation once into shared memory, then streams a continuous
    ``(N block, K block)`` sequence. MMA warps reset accumulators per output
    block and reuse the resident normalized activation for every block in the
    coalesced range.
    """

    pipeline = PipelineSpec.streaming(
        range_axis=2,
        range_end_axis=3,
        range_block_size=1,
        coalesce_ranges=True,
    )
    pipeline_abi = PipelineABI.op_owned()
    reads = {
        "a": (None, ("B", "S", "K")),
        "residual_in": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "b": (None, ("N", "K")),
    }
    writes = {
        "residual_out": (None, ("B", "S", "K")),
        "c": (None, ("B", "S", "N")),
    }
    tile = ("B", "S", "N")
    dynamic_dims = ("B", "S")
    tma_loads = {"b"}
    tma_compute_stores = {"c"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name == "b":
            return (tile_sizes["N"], static_dims["tile_K"])
        if tensor_name == "c":
            return (1, tile_sizes["S"], tile_sizes["N"])
        return None

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        if tensor_name == "b":
            tile_k, tile_n = tma_tile_shape
            swz_b = static_dims.get("swz_B_ab", 2)
            return (
                f"cute.make_composed_layout("
                f"cute.make_swizzle({swz_b}, 4, 3), 0, "
                f"cute.make_layout(({tile_k}, {tile_n}), stride=(1, {tile_k})))"
            )
        if tensor_name == "c":
            tile_n, tile_s, one = tma_tile_shape
            swz_c = static_dims.get("swz_B_c", 3)
            return (
                f"cute.make_composed_layout("
                f"cute.make_swizzle({swz_c}, 4, 3), 0, "
                f"cute.make_layout(({tile_n}, {tile_s}, {one}), "
                f"stride=(1, {tile_n}, {tile_n * tile_s})))"
            )
        return None

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, eps=QWEN3_5_EPS, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", QWEN3_5_DECODE_S)
        tile_sizes.setdefault("N", 128)
        tile_k = tile_sizes.pop("K", 32)
        if tile_k != 32:
            raise ValueError(f"{cls.__name__} currently supports tile K=32 only")
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        a = tensors["a"]
        b = tensors["b"]
        elem_bytes = a.element_size()
        a_tile_bytes = tile_sizes["S"] * tile_k * elem_bytes
        b_tile_bytes = tile_sizes["N"] * tile_k * b.element_size()
        c_tile_bytes = tile_sizes["S"] * tile_sizes["N"] * elem_bytes
        num_k_blocks = (a.shape[-1] + tile_k - 1) // tile_k
        norm_bytes = num_k_blocks * a_tile_bytes
        buf_stride = b_tile_bytes
        c_offset = norm_bytes + 2 * buf_stride
        mbar_offset = c_offset + c_tile_bytes
        required = mbar_offset + 40
        page_size = max(page_size, required)
        if tile_k % 64 == 0 and tile_k >= 64:
            swz_b_ab = 3
        elif tile_k % 32 == 0:
            swz_b_ab = 2
        else:
            swz_b_ab = 1
        if tile_sizes["N"] % 64 == 0 and tile_sizes["N"] >= 64:
            swz_b_c = 3
        elif tile_sizes["N"] % 32 == 0:
            swz_b_c = 2
        else:
            swz_b_c = 1
        op.static_dims["tile_K"] = tile_k
        op.static_dims["norm_bytes"] = norm_bytes
        op.static_dims["a_tile_bytes"] = a_tile_bytes
        op.static_dims["b_tile_bytes"] = b_tile_bytes
        op.static_dims["c_tile_bytes"] = c_tile_bytes
        op.static_dims["buf_stride"] = buf_stride
        op.static_dims["b_offset"] = 0
        op.static_dims["c_offset"] = c_offset
        op.static_dims["mbar_offset"] = mbar_offset
        op.static_dims["page_size"] = page_size
        op.static_dims["eps"] = eps
        op.static_dims["swz_B_ab"] = swz_b_ab
        op.static_dims["swz_B_c"] = swz_b_c
        return [op]

    def __init__(self, **config):
        super().__init__(**config)
        self.elem_bytes = 2
        self.num_k_blocks = (self.K + self.tile_K - 1) // self.tile_K
        self.num_mma_warps = self.threads_per_row // 32
        self.num_mma_warps = min(self.num_mma_warps, max(1, self.tile_size_N // 32))
        self.num_mma_threads = self.num_mma_warps * 32
        self.activation = 0
        assert self.tile_K >= 16 and self.tile_K % 16 == 0, (
            f"{type(self).__name__}: tile_K={self.tile_K} must be >= 16 and a multiple of 16."
        )
        assert self.mbar_offset + 40 <= self.page_size, (
            f"{type(self).__name__}: smem {self.mbar_offset + 40}B exceeds page_size "
            f"({self.page_size}B)."
        )

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_N, tile_3,
             a, residual_in, norm_weight, residual_out,
             b_tma, b_tma_gmem, op_config_ptr, work_mbar):
        runtime_S = config_dim_i32(op_config_ptr, "S", type(self))
        lane_idx = cute.arch.lane_idx()
        row_start = tile_S * Int32(self.tile_size_S)
        swz_norm = cute.make_swizzle(self.swz_B_ab, 4, 3)

        for local_row in range(self.tile_size_S):
            row_idx = row_start + Int32(local_row)
            if row_idx < runtime_S:
                partial = Float32(0.0)
                base = tile_B * Int32(self.a_stride_B) + row_idx * Int32(self.a_stride_S)
                res_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S)
                out_base = tile_B * Int32(self.residual_out_stride_B) + row_idx * Int32(self.residual_out_stride_S)
                a_row = cute.make_tensor(a.iterator + base, cute.make_layout(self.K))
                res_row = cute.make_tensor(residual_in.iterator + res_base, cute.make_layout(self.K))
                out_row = cute.make_tensor(residual_out.iterator + out_base, cute.make_layout(self.K))
                k = lane_idx
                while k < Int32(self.K):
                    val = a_row[k].to(Float32) + res_row[k].to(Float32)
                    partial = partial + val * val
                    out_row[k] = val.to(self.residual_out_dtype)
                    k = k + Int32(32)
                total = cute.arch.warp_reduction(partial, operator.add)
                rstd = cute.math.rsqrt(
                    total * Float32(1.0 / self.K) + Float32(self.eps),
                    fastmath=True,
                )
                w_row = cute.make_tensor(norm_weight.iterator, cute.make_layout(self.K))
                k2 = lane_idx
                while k2 < Int32(self.K):
                    val = (a_row[k2].to(Float32) + res_row[k2].to(Float32)) * rstd * w_row[k2].to(Float32)
                    norm_block = k2 // Int32(self.tile_K)
                    norm_local_k = k2 - norm_block * Int32(self.tile_K)
                    sA_norm = cute.make_tensor(
                        cute.recast_ptr(
                            cute.make_ptr(
                                self.a_dtype,
                                page_ptr + norm_block * Int32(self.a_tile_bytes),
                                cute.AddressSpace.smem,
                                assumed_align=128,
                            ),
                            swz_norm,
                            dtype=self.a_dtype,
                        ),
                        cute.make_layout((self.tile_K, self.tile_size_S), stride=(1, self.tile_K)),
                    )
                    sA_norm[(norm_local_k, Int32(local_row))] = val.to(self.a_dtype)
                    k2 = k2 + Int32(32)

        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)
        with cute.arch.elect_one():
            mbarrier_init(bf_0, Int32(self.num_mma_warps))
            mbarrier_init(bf_1, Int32(self.num_mma_warps))
            mbarrier_init(kr_0, Int32(1))
            mbarrier_init(kr_1, Int32(1))
        mbarrier_init_fence()

        total_stream_blocks = (tile_3 - tile_N) * Int32(self.num_k_blocks)
        first_stream_blocks = Int32(2)
        if total_stream_blocks < Int32(2):
            first_stream_blocks = total_stream_blocks
        stream_idx = Int32(0)
        block_idx = tile_N
        while block_idx < tile_3:
            k_block = Int32(0)
            while k_block < Int32(self.num_k_blocks):
                buf_idx = stream_idx % Int32(2)
                buf_base = page_ptr + Int32(self.norm_bytes) + buf_idx * Int32(self.buf_stride)
                if stream_idx >= Int32(2):
                    bf_phase = ((stream_idx - Int32(2)) // Int32(2)) % Int32(2)
                    if buf_idx == Int32(0):
                        mbarrier_wait(bf_0, bf_phase)
                    if buf_idx == Int32(1):
                        mbarrier_wait(bf_1, bf_phase)

                sB = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(
                            self.b_dtype,
                            buf_base + Int32(self.b_offset),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        cute.make_swizzle(self.swz_B_ab, 4, 3),
                        dtype=self.b_dtype,
                    ),
                    cute.make_layout((self.tile_K, self.tile_size_N), stride=(1, self.tile_K)),
                )
                gB = cute.local_tile(b_tma_gmem, (self.tile_K, self.tile_size_N), (None, None))
                tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                    b_tma,
                    Int32(0),
                    cute.make_layout(1),
                    cute.group_modes(sB, 0, 2),
                    cute.group_modes(gB, 0, 2),
                )
                if stream_idx < Int32(2):
                    mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
                    if stream_idx == Int32(0):
                        with cute.arch.elect_one():
                            mbarrier_arrive_expect_tx(
                                work_mbar,
                                first_stream_blocks * Int32(self.b_tile_bytes),
                            )
                    cute.copy(b_tma, tBgB[(None, k_block, block_idx)], tBsB, tma_bar_ptr=mbar_ptr)
                if stream_idx >= Int32(2):
                    kr_ptr = cute.make_ptr(cutlass.Int64, kr_0, cute.AddressSpace.smem)
                    if buf_idx == Int32(0):
                        with cute.arch.elect_one():
                            mbarrier_arrive_expect_tx(kr_0, Int32(self.b_tile_bytes))
                    if buf_idx == Int32(1):
                        kr_ptr = cute.make_ptr(cutlass.Int64, kr_1, cute.AddressSpace.smem)
                        with cute.arch.elect_one():
                            mbarrier_arrive_expect_tx(kr_1, Int32(self.b_tile_bytes))
                    cute.copy(b_tma, tBgB[(None, k_block, block_idx)], tBsB, tma_bar_ptr=kr_ptr)

                k_block = k_block + Int32(1)
                stream_idx = stream_idx + Int32(1)
            block_idx = block_idx + Int32(1)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_N, tile_3, c_tma, c_tma_gmem, op_config_ptr):
        runtime_S = config_dim_i32(op_config_ptr, "S", type(self))
        tidx = cute.arch.thread_idx()[0]
        if tidx < Int32(self.num_mma_threads):
            mma_op = cute.nvgpu.warp.MmaF16BF16Op(
                self.a_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((1, self.num_mma_warps, 1)),
                permutation_mnk=(16, self.num_mma_warps * 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)
            swz = cute.make_swizzle(self.swz_B_ab, 4, 3)

            smem_copy_atom_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.a_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
            smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
            smem_copy_atom_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.b_dtype,
            )
            smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
            smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

            sB_0 = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(
                        self.b_dtype,
                        page_ptr + Int32(self.norm_bytes + self.b_offset),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    ),
                    swz,
                    dtype=self.b_dtype,
                ),
                cute.make_layout((self.tile_size_N, self.tile_K), stride=(self.tile_K, 1)),
            )
            sB_1 = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(
                        self.b_dtype,
                        page_ptr + Int32(self.norm_bytes + self.buf_stride + self.b_offset),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    ),
                    swz,
                    dtype=self.b_dtype,
                ),
                cute.make_layout((self.tile_size_N, self.tile_K), stride=(self.tile_K, 1)),
            )

            sA_template = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(
                        self.a_dtype,
                        page_ptr,
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    ),
                    swz,
                    dtype=self.a_dtype,
                ),
                cute.make_layout((self.tile_size_S, self.tile_K), stride=(self.tile_K, 1)),
            )
            tCsA = thr_mma.partition_A(sA_template)
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tCsB = thr_mma.partition_B(sB_0)
            tCrB = tiled_mma.make_fragment_B(tCsB)
            tCrA_ld = smem_thr_copy_A.retile(tCrA)
            tCrB_ld = smem_thr_copy_B.retile(tCrB)
            tBsB_ld_0 = smem_thr_copy_B.partition_S(sB_0)
            tBsB_ld_1 = smem_thr_copy_B.partition_S(sB_1)

            bf_0 = page_ptr + Int32(self.mbar_offset)
            bf_1 = page_ptr + Int32(self.mbar_offset + 8)
            kr_0 = page_ptr + Int32(self.mbar_offset + 16)
            kr_1 = page_ptr + Int32(self.mbar_offset + 24)
            kr_phase_0 = Int32(0)
            kr_phase_1 = Int32(0)
            row_start = tile_S * Int32(self.tile_size_S)
            stream_idx = Int32(0)
            block_idx = tile_N
            while block_idx < tile_3:
                acc = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.tile_size_S, self.tile_size_N)),
                    Float32,
                )
                acc.fill(0.0)
                k_idx = Int32(0)
                while k_idx < Int32(self.num_k_blocks):
                    sA_k = cute.make_tensor(
                        cute.recast_ptr(
                            cute.make_ptr(
                                self.a_dtype,
                                page_ptr + k_idx * Int32(self.a_tile_bytes),
                                cute.AddressSpace.smem,
                                assumed_align=128,
                            ),
                            swz,
                            dtype=self.a_dtype,
                        ),
                        cute.make_layout((self.tile_size_S, self.tile_K), stride=(self.tile_K, 1)),
                    )
                    tAsA_ld = smem_thr_copy_A.partition_S(sA_k)
                    if stream_idx >= Int32(2):
                        if stream_idx % Int32(2) == Int32(0):
                            mbarrier_wait(kr_0, kr_phase_0)
                            kr_phase_0 = kr_phase_0 ^ Int32(1)
                        if stream_idx % Int32(2) == Int32(1):
                            mbarrier_wait(kr_1, kr_phase_1)
                            kr_phase_1 = kr_phase_1 ^ Int32(1)

                    if stream_idx % Int32(2) == Int32(0):
                        for k_block in cutlass.range_constexpr(self.tile_K // 16):
                            cute.copy(
                                smem_tiled_copy_A,
                                tAsA_ld[None, None, k_block],
                                tCrA_ld[None, None, k_block],
                            )
                            cute.copy(
                                smem_tiled_copy_B,
                                tBsB_ld_0[None, None, k_block],
                                tCrB_ld[None, None, k_block],
                            )
                            cute.gemm(
                                tiled_mma,
                                acc,
                                tCrA[None, None, k_block],
                                tCrB[None, None, k_block],
                                acc,
                            )
                    if stream_idx % Int32(2) == Int32(1):
                        for k_block in cutlass.range_constexpr(self.tile_K // 16):
                            cute.copy(
                                smem_tiled_copy_A,
                                tAsA_ld[None, None, k_block],
                                tCrA_ld[None, None, k_block],
                            )
                            cute.copy(
                                smem_tiled_copy_B,
                                tBsB_ld_1[None, None, k_block],
                                tCrB_ld[None, None, k_block],
                            )
                            cute.gemm(
                                tiled_mma,
                                acc,
                                tCrA[None, None, k_block],
                                tCrB[None, None, k_block],
                                acc,
                            )

                    if tidx % Int32(32) == Int32(0):
                        if stream_idx % Int32(2) == Int32(0):
                            mbarrier_arrive(bf_0)
                        if stream_idx % Int32(2) == Int32(1):
                            mbarrier_arrive(bf_1)
                    stream_idx = stream_idx + Int32(1)
                    k_idx = k_idx + Int32(1)

                _gemm_epilogue_store_no_mbar_inval_helper(
                    page_ptr + Int32(self.c_offset),
                    tidx,
                    tiled_mma,
                    acc,
                    self.num_mma_threads,
                    self.swz_B_c,
                    self.c_dtype,
                    self.tile_size_S,
                    self.tile_size_N,
                    self.activation,
                )
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
                swz_c = cute.make_swizzle(self.swz_B_c, 4, 3)
                sC = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(
                            self.c_dtype,
                            page_ptr + Int32(self.c_offset),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        swz_c,
                        dtype=self.c_dtype,
                    ),
                    cute.make_layout((self.tile_size_N, self.tile_size_S, 1),
                                     stride=(1, self.tile_size_N,
                                             self.tile_size_N * self.tile_size_S)),
                )
                gC = cute.local_tile(
                    c_tma_gmem,
                    (self.tile_size_N, self.tile_size_S, 1),
                    (None, None, None),
                )
                tCsC, tCgC = cute.nvgpu.cpasync.tma_partition(
                    c_tma,
                    Int32(0),
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 3),
                    cute.group_modes(gC, 0, 3),
                )
                with cute.arch.elect_one():
                    cute.copy(c_tma, tCsC, tCgC[(None, block_idx, tile_S, tile_B)])
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
                block_idx = block_idx + Int32(1)

            if tidx == Int32(0):
                mbarrier_inval(bf_0)
                mbarrier_inval(bf_1)
                mbarrier_inval(kr_0)
                mbarrier_inval(kr_1)


class Qwen3_5DecodeMatvecGemmSm120Op(Qwen3_5StagedDecodeGemmSm120Op):
    """Qwen decode projection GEMM using staged loader/consumer streaming."""

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        tile_sizes = dict(tile_sizes or {})
        b = tensors["b"]
        n_dim, k_dim = b.shape
        tile_sizes.setdefault("S", QWEN3_5_DECODE_S)
        if n_dim == QWEN3_5_Q_DIM + 2 * QWEN3_5_KV_DIM and k_dim == QWEN3_5_HIDDEN:
            tile_sizes.setdefault("N", 64)
            tile_sizes.setdefault("K", 32)
        elif n_dim == 2 * QWEN3_5_INTERMEDIATE and k_dim == QWEN3_5_HIDDEN:
            tile_sizes.setdefault("N", 128)
            tile_sizes.setdefault("K", 32)
        elif n_dim == QWEN3_5_HIDDEN and k_dim in (QWEN3_5_Q_DIM, QWEN3_5_INTERMEDIATE):
            tile_sizes.setdefault("N", 32)
            tile_sizes.setdefault("K", 64)
        else:
            tile_sizes.setdefault("N", 64)
            tile_sizes.setdefault("K", 32)
        return super().schedule(tile_sizes=tile_sizes, page_size=page_size, **tensors)


class Qwen3_5PackedQkvChunkProjectSm120Op(Qwen3_5DecodeMatvecGemmSm120Op):
    """Packed QKV projection chunk with Q/K norm partials for the finalizer."""

    writes = {
        "c": (None, ("B", "S", "N")),
        "qk_sumsq": (cutlass.Float32, ("B", "S", "H", "C")),
    }
    tma_stores = {"c"}

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        qk_sumsq = tensors.get("qk_sumsq")
        if qk_sumsq is None:
            raise ValueError(f"{cls.__name__} requires qk_sumsq")
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("N", 64)
        tile_sizes.setdefault("K", 64)
        ops = super().schedule(tile_sizes=tile_sizes, page_size=page_size, **tensors)
        for op in ops:
            op.static_dims["num_qk_heads"] = QWEN3_5_NUM_Q_HEADS + QWEN3_5_NUM_KV_HEADS
            op.static_dims["head_dim"] = QWEN3_5_HEAD_DIM
            op.static_dims["head_chunks"] = QWEN3_5_HEAD_DIM // op.tile_sizes["N"]
        return ops

    @cute.jit
    def store(self, page_ptr, tile_B, tile_S, tile_N, qk_sumsq, c_tma, c_tma_gmem, op_config_ptr):
        swz_c = cute.make_swizzle(self.swz_B_c, 4, 3)
        sC = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.c_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                swz_c,
                dtype=self.c_dtype,
            ),
            cute.make_layout(
                (self.tile_size_N, self.tile_size_S, 1),
                stride=(1, self.tile_size_N, self.tile_size_N * self.tile_size_S),
            ),
        )

        gC = cute.local_tile(
            c_tma_gmem,
            (self.tile_size_N, self.tile_size_S, 1),
            (None, None, None),
        )
        tCsC, tCgC = cute.nvgpu.cpasync.tma_partition(
            c_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sC, 0, 3),
            cute.group_modes(gC, 0, 3),
        )
        with cute.arch.elect_one():
            cute.copy(c_tma, tCsC, tCgC[(None, tile_N, tile_S, tile_B)])

        head = tile_N // Int32(self.head_chunks)
        chunk = tile_N - head * Int32(self.head_chunks)
        if head < Int32(self.num_qk_heads):
            runtime_S = config_dim_i32(op_config_ptr, "S", type(self))
            row_start = tile_S * Int32(self.tile_size_S)
            lane_idx = cute.arch.lane_idx()
            for local_row in range(self.tile_size_S):
                row_idx = row_start + Int32(local_row)
                if row_idx < runtime_S:
                    partial = Float32(0.0)
                    local_n = lane_idx
                    while local_n < Int32(self.tile_size_N):
                        v = sC[(local_n, Int32(local_row), Int32(0))].to(Float32)
                        partial = partial + v * v
                        local_n = local_n + Int32(32)
                    total = cute.arch.warp_reduction(partial, operator.add)
                    if lane_idx == Int32(0):
                        qk_sumsq[
                            (
                                tile_B,
                                row_idx,
                                head,
                                chunk,
                            )
                        ] = total


class Qwen3_5ComputeTmaRMSAddPackedQkvChunkProjectSm120Op(Qwen3_5RMSAddStagedDecodeGemmSm120Op):
    """Compute-driven fused residual-add RMSNorm plus packed QKV projection."""

    pipeline = PipelineSpec.streaming(
        range_axis=2,
        range_end_axis=3,
        range_block_size=1,
        coalesce_ranges=True,
    )
    pipeline_abi = PipelineABI.op_owned()
    reads = {
        "a": (None, ("B", "S", "K")),
        "residual_in": (None, ("B", "S", "K")),
        "norm_weight": (None, ("K",)),
        "b": (None, ("N", "K")),
    }
    writes = {
        "residual_out": (None, ("B", "S", "K")),
        "c": (None, ("B", "S", "N")),
        "qk_sumsq": (cutlass.Float32, ("B", "S", "H", "C")),
    }
    tma_loads = set()
    tma_compute_loads = {"b"}
    tma_compute_stores = {"c"}
    tma_stores = set()

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, eps=QWEN3_5_EPS, **tensors):
        qk_sumsq = tensors.get("qk_sumsq")
        if qk_sumsq is None:
            raise ValueError(f"{cls.__name__} requires qk_sumsq")
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("N", 64)
        tile_sizes.setdefault("K", 32)
        ops = super().schedule(tile_sizes=tile_sizes, page_size=page_size, eps=eps, **tensors)
        for op in ops:
            op.static_dims["num_qk_heads"] = QWEN3_5_NUM_Q_HEADS + QWEN3_5_NUM_KV_HEADS
            op.static_dims["head_dim"] = QWEN3_5_HEAD_DIM
            op.static_dims["head_chunks"] = QWEN3_5_HEAD_DIM // op.tile_sizes["N"]
        return ops

    @cute.jit
    def load(self, page_ptr):
        pass

    @cute.jit
    def store(self, page_ptr):
        pass

    @cute.jit
    def _issue_b_tma(self, page_ptr, stream_idx, k_block, tile_N, b_tma, b_tma_gmem, kr_0, kr_1):
        swz = cute.make_swizzle(self.swz_B_ab, 4, 3)
        buf_idx = stream_idx % Int32(2)
        buf_base = page_ptr + buf_idx * Int32(self.buf_stride)
        sB = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.b_dtype,
                    buf_base + Int32(self.b_offset),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swz,
                dtype=self.b_dtype,
            ),
            cute.make_layout((self.tile_K, self.tile_size_N), stride=(1, self.tile_K)),
        )
        gB = cute.local_tile(b_tma_gmem, (self.tile_K, self.tile_size_N), (None, None))
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            b_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )
        if cute.arch.warp_idx() == Int32(0):
            if buf_idx == Int32(0):
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(kr_0, Int32(self.b_tile_bytes))
                kr_ptr = cute.make_ptr(cutlass.Int64, kr_0, cute.AddressSpace.smem)
                cute.copy(b_tma, tBgB[(None, k_block, tile_N)], tBsB, tma_bar_ptr=kr_ptr)
            if buf_idx == Int32(1):
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(kr_1, Int32(self.b_tile_bytes))
                kr_ptr = cute.make_ptr(cutlass.Int64, kr_1, cute.AddressSpace.smem)
                cute.copy(b_tma, tBgB[(None, k_block, tile_N)], tBsB, tma_bar_ptr=kr_ptr)

    @cute.jit
    def _fill_a_block(self, page_ptr, tidx, tile_B, tile_S, k_idx,
                      a, residual_in, norm_weight, rstd_scratch, op_config_ptr):
        runtime_S = config_dim_i32(op_config_ptr, "S", type(self))
        swz = cute.make_swizzle(self.swz_B_ab, 4, 3)
        buf_idx = k_idx % Int32(2)
        buf_base = page_ptr + buf_idx * Int32(self.buf_stride)
        row_start = tile_S * Int32(self.tile_size_S)
        sA = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.a_dtype, buf_base, cute.AddressSpace.smem, assumed_align=128),
                swz,
                dtype=self.a_dtype,
            ),
            cute.make_layout((self.tile_K, self.tile_size_S, 1),
                             stride=(1, self.tile_K, self.tile_K * self.tile_size_S)),
        )
        linear = tidx
        while linear < Int32(self.tile_K * self.tile_size_S):
            local_k = linear % Int32(self.tile_K)
            local_row = linear // Int32(self.tile_K)
            row_idx = row_start + local_row
            k = k_idx * Int32(self.tile_K) + local_k
            if row_idx < runtime_S and k < Int32(self.K):
                base = tile_B * Int32(self.a_stride_B) + row_idx * Int32(self.a_stride_S)
                res_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S)
                a_row = cute.make_tensor(a.iterator + base, cute.make_layout(self.K))
                res_row = cute.make_tensor(residual_in.iterator + res_base, cute.make_layout(self.K))
                w_row = cute.make_tensor(norm_weight.iterator, cute.make_layout(self.K))
                val = (
                    (a_row[k].to(Float32) + res_row[k].to(Float32))
                    * rstd_scratch[local_row]
                    * w_row[k].to(Float32)
                )
                sA[(local_k, local_row, Int32(0))] = val.to(self.a_dtype)
            linear = linear + Int32(self.num_mma_threads)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_N, tile_3,
                a, residual_in, norm_weight, residual_out, qk_sumsq,
                b_tma, b_tma_gmem, c_tma, c_tma_gmem, op_config_ptr):
        runtime_S = config_dim_i32(op_config_ptr, "S", type(self))
        tidx = cute.arch.thread_idx()[0]
        if tidx < Int32(self.num_mma_threads):
            lane_idx = cute.arch.lane_idx()
            warp_idx = cute.arch.warp_idx()
            swz = cute.make_swizzle(self.swz_B_ab, 4, 3)

            kr_0 = page_ptr + Int32(self.mbar_offset + 16)
            kr_1 = page_ptr + Int32(self.mbar_offset + 24)
            with cute.arch.elect_one():
                mbarrier_init(kr_0, Int32(1))
                mbarrier_init(kr_1, Int32(1))
            mbarrier_init_fence()

            rstd_scratch = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float32,
                    page_ptr + Int32(self.rms_scratch_offset),
                    cute.AddressSpace.smem,
                ),
                cute.make_layout(self.tile_size_S),
            )
            row_start = tile_S * Int32(self.tile_size_S)
            for local_row in range(warp_idx, self.tile_size_S, self.num_mma_warps):
                row_idx = row_start + Int32(local_row)
                if row_idx < runtime_S:
                    partial = Float32(0.0)
                    base = tile_B * Int32(self.a_stride_B) + row_idx * Int32(self.a_stride_S)
                    res_base = tile_B * Int32(self.residual_in_stride_B) + row_idx * Int32(self.residual_in_stride_S)
                    out_base = tile_B * Int32(self.residual_out_stride_B) + row_idx * Int32(self.residual_out_stride_S)
                    a_row = cute.make_tensor(a.iterator + base, cute.make_layout(self.K))
                    res_row = cute.make_tensor(residual_in.iterator + res_base, cute.make_layout(self.K))
                    out_row = cute.make_tensor(residual_out.iterator + out_base, cute.make_layout(self.K))
                    k = lane_idx
                    while k < Int32(self.K):
                        val = a_row[k].to(Float32) + res_row[k].to(Float32)
                        partial = partial + val * val
                        out_row[k] = val.to(self.residual_out_dtype)
                        k = k + Int32(32)
                    total = cute.arch.warp_reduction(partial, operator.add)
                    if lane_idx == Int32(0):
                        rstd_scratch[Int32(local_row)] = cute.math.rsqrt(
                            total * Float32(1.0 / self.K) + Float32(self.eps),
                            fastmath=True,
                        )
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            mma_op = cute.nvgpu.warp.MmaF16BF16Op(self.a_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((self.num_mma_warps, 1, 1)),
                permutation_mnk=(self.num_mma_warps * 16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)
            smem_copy_atom_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.a_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
            smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
            smem_copy_atom_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.b_dtype,
            )
            smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
            smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

            sA_0 = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.a_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                    swz,
                    dtype=self.a_dtype,
                ),
                cute.make_layout((self.tile_size_S, self.tile_K), stride=(self.tile_K, 1)),
            )
            sB_0 = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.b_dtype, page_ptr + Int32(self.b_offset), cute.AddressSpace.smem, assumed_align=128),
                    swz,
                    dtype=self.b_dtype,
                ),
                cute.make_layout((self.tile_size_N, self.tile_K), stride=(self.tile_K, 1)),
            )
            sA_1 = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.a_dtype, page_ptr + Int32(self.buf_stride), cute.AddressSpace.smem, assumed_align=128),
                    swz,
                    dtype=self.a_dtype,
                ),
                cute.make_layout((self.tile_size_S, self.tile_K), stride=(self.tile_K, 1)),
            )
            sB_1 = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.b_dtype, page_ptr + Int32(self.buf_stride + self.b_offset), cute.AddressSpace.smem, assumed_align=128),
                    swz,
                    dtype=self.b_dtype,
                ),
                cute.make_layout((self.tile_size_N, self.tile_K), stride=(self.tile_K, 1)),
            )
            tCsA = thr_mma.partition_A(sA_0)
            tCsB = thr_mma.partition_B(sB_0)
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tCrB = tiled_mma.make_fragment_B(tCsB)
            tCrA_ld = smem_thr_copy_A.retile(tCrA)
            tCrB_ld = smem_thr_copy_B.retile(tCrB)
            tAsA_ld_0 = smem_thr_copy_A.partition_S(sA_0)
            tBsB_ld_0 = smem_thr_copy_B.partition_S(sB_0)
            tAsA_ld_1 = smem_thr_copy_A.partition_S(sA_1)
            tBsB_ld_1 = smem_thr_copy_B.partition_S(sB_1)

            block_idx = tile_N
            while block_idx < tile_3:
                first_blocks = Int32(2)
                if Int32(self.num_k_blocks) < first_blocks:
                    first_blocks = Int32(self.num_k_blocks)
                preload = Int32(0)
                while preload < first_blocks:
                    self._issue_b_tma(
                        page_ptr,
                        preload,
                        preload,
                        block_idx,
                        b_tma,
                        b_tma_gmem,
                        kr_0,
                        kr_1,
                    )
                    preload = preload + Int32(1)

                acc = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.tile_size_S, self.tile_size_N)),
                    Float32,
                )
                acc.fill(0.0)
                kr_phase_0 = Int32(0)
                kr_phase_1 = Int32(0)
                k_idx = Int32(0)
                while k_idx < Int32(self.num_k_blocks):
                    self._fill_a_block(
                        page_ptr, tidx, tile_B, tile_S, k_idx,
                        a, residual_in, norm_weight, rstd_scratch, op_config_ptr,
                    )
                    named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
                    if k_idx % Int32(2) == Int32(0):
                        mbarrier_wait(kr_0, kr_phase_0)
                        for k_block in cutlass.range_constexpr(self.tile_K // 16):
                            cute.copy(smem_tiled_copy_A, tAsA_ld_0[None, None, k_block], tCrA_ld[None, None, k_block])
                            cute.copy(smem_tiled_copy_B, tBsB_ld_0[None, None, k_block], tCrB_ld[None, None, k_block])
                            cute.gemm(tiled_mma, acc, tCrA[None, None, k_block], tCrB[None, None, k_block], acc)
                        kr_phase_0 = kr_phase_0 ^ Int32(1)
                    if k_idx % Int32(2) == Int32(1):
                        mbarrier_wait(kr_1, kr_phase_1)
                        for k_block in cutlass.range_constexpr(self.tile_K // 16):
                            cute.copy(smem_tiled_copy_A, tAsA_ld_1[None, None, k_block], tCrA_ld[None, None, k_block])
                            cute.copy(smem_tiled_copy_B, tBsB_ld_1[None, None, k_block], tCrB_ld[None, None, k_block])
                            cute.gemm(tiled_mma, acc, tCrA[None, None, k_block], tCrB[None, None, k_block], acc)
                        kr_phase_1 = kr_phase_1 ^ Int32(1)

                    next_k = k_idx + Int32(2)
                    if next_k < Int32(self.num_k_blocks):
                        self._issue_b_tma(
                            page_ptr,
                            next_k,
                            next_k,
                            block_idx,
                            b_tma,
                            b_tma_gmem,
                            kr_0,
                            kr_1,
                        )
                    k_idx = k_idx + Int32(1)

                _gemm_epilogue_store_no_mbar_inval_helper(
                    page_ptr,
                    tidx,
                    tiled_mma,
                    acc,
                    self.num_mma_threads,
                    self.swz_B_c,
                    self.c_dtype,
                    self.tile_size_S,
                    self.tile_size_N,
                    self.activation,
                )
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                swz_c = cute.make_swizzle(self.swz_B_c, 4, 3)
                sC = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(self.c_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                        swz_c,
                        dtype=self.c_dtype,
                    ),
                    cute.make_layout((self.tile_size_N, self.tile_size_S, 1),
                                     stride=(1, self.tile_size_N, self.tile_size_N * self.tile_size_S)),
                )
                chunk_base = block_idx * Int32(self.tile_size_N // 64)
                for local_chunk in cutlass.range_constexpr(self.tile_size_N // 64):
                    qk_chunk = chunk_base + Int32(local_chunk)
                    head = qk_chunk // Int32(self.head_chunks)
                    chunk = qk_chunk - head * Int32(self.head_chunks)
                    if head < Int32(self.num_qk_heads):
                        if tidx < Int32(32):
                            for local_row in range(self.tile_size_S):
                                row_idx = row_start + Int32(local_row)
                                if row_idx < runtime_S:
                                    partial = Float32(0.0)
                                    local_n = lane_idx + Int32(local_chunk * 64)
                                    while local_n < Int32((local_chunk + 1) * 64):
                                        v = sC[(local_n, Int32(local_row), Int32(0))].to(Float32)
                                        partial = partial + v * v
                                        local_n = local_n + Int32(32)
                                    total = cute.arch.warp_reduction(partial, operator.add)
                                    if lane_idx == Int32(0):
                                        qk_sumsq[(tile_B, row_idx, head, chunk)] = total
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
                gC = cute.local_tile(c_tma_gmem, (self.tile_size_N, self.tile_size_S, 1), (None, None, None))
                tCsC, tCgC = cute.nvgpu.cpasync.tma_partition(
                    c_tma,
                    Int32(0),
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 3),
                    cute.group_modes(gC, 0, 3),
                )
                with cute.arch.elect_one():
                    cute.copy(c_tma, tCsC, tCgC[(None, block_idx, tile_S, tile_B)])
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
                block_idx = block_idx + Int32(1)

            if tidx == Int32(0):
                mbarrier_inval(kr_0)
                mbarrier_inval(kr_1)


class Qwen3_5PackedQkvProjectSm120Op(Op):
    """Packed QKV projection with per-head Q/K norm, RoPE, and cache writes."""

    pipeline = PipelineSpec.streaming(
        range_axis=2,
        range_block_size=1,
    )
    pipeline_abi = PipelineABI.op_owned()
    reads = {
        "a": (None, ("B", "S", "K")),
        "b": (None, ("P", "K")),
        "q_norm_weight": (None, ("D",)),
        "k_norm_weight": (None, ("D",)),
        "cos": (None, ("S", "D2")),
        "sin": (None, ("S", "D2")),
    }
    writes = {
        "qkv": (None, ("B", "S", "H", "D")),
        "dst_k": (None, ("B", "N", "HK", "D")),
        "dst_v": (None, ("B", "N", "HK", "D")),
    }
    tile = ("B", "S", "H")
    dynamic_dims = ("B", "S", "N")
    tma_loads = {"b"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name == "b":
            return (static_dims["tile_N"], static_dims["tile_K"])
        return None

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        if tensor_name == "b":
            tile_k, tile_n = tma_tile_shape
            swz_b = static_dims.get("swz_B_ab", 2)
            return (
                f"cute.make_composed_layout("
                f"cute.make_swizzle({swz_b}, 4, 3), 0, "
                f"cute.make_layout(({tile_k}, {tile_n}), stride=(1, {tile_k})))"
            )
        return None

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, eps=QWEN3_5_EPS, cache_pos=0, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", QWEN3_5_DECODE_S)
        tile_sizes.setdefault("H", 1)
        tile_k = tile_sizes.pop("K", 32)
        tile_n = tile_sizes.pop("N", 64)
        if tile_k != 32 or tile_n != 64:
            raise ValueError(f"{cls.__name__} currently requires tile K=32 and tile N=64")
        op = cls._schedule_single(tile_sizes=tile_sizes, **tensors)
        a = tensors["a"]
        elem_bytes = a.element_size()
        a_tile_bytes = tile_sizes["S"] * tile_k * elem_bytes
        b_tile_bytes = tile_n * tile_k * tensors["b"].element_size()
        head_bytes = tile_sizes["S"] * QWEN3_5_HEAD_DIM * elem_bytes
        sumsq_bytes = tile_sizes["S"] * 4
        c_tile_bytes = tile_sizes["S"] * tile_n * elem_bytes
        num_k_blocks = (a.shape[-1] + tile_k - 1) // tile_k
        norm_bytes = num_k_blocks * a_tile_bytes
        buf_stride = b_tile_bytes
        head_offset = norm_bytes + 2 * buf_stride
        sumsq_offset = head_offset + head_bytes
        c_offset = sumsq_offset + sumsq_bytes
        mbar_offset = c_offset + c_tile_bytes
        page_size = max(page_size, mbar_offset + 32)
        swz_b_ab = 2
        swz_b_c = 3
        op.static_dims["tile_K"] = tile_k
        op.static_dims["tile_N"] = tile_n
        op.static_dims["num_head_blocks"] = QWEN3_5_HEAD_DIM // tile_n
        op.static_dims["norm_bytes"] = norm_bytes
        op.static_dims["a_tile_bytes"] = a_tile_bytes
        op.static_dims["b_tile_bytes"] = b_tile_bytes
        op.static_dims["head_bytes"] = head_bytes
        op.static_dims["sumsq_bytes"] = sumsq_bytes
        op.static_dims["c_tile_bytes"] = c_tile_bytes
        op.static_dims["buf_stride"] = buf_stride
        op.static_dims["head_offset"] = head_offset
        op.static_dims["sumsq_offset"] = sumsq_offset
        op.static_dims["c_offset"] = c_offset
        op.static_dims["mbar_offset"] = mbar_offset
        op.static_dims["page_size"] = page_size
        op.static_dims["eps"] = eps
        op.static_dims["cache_pos"] = cache_pos
        op.static_dims["num_q_heads"] = QWEN3_5_NUM_Q_HEADS
        op.static_dims["num_kv_heads"] = QWEN3_5_NUM_KV_HEADS
        op.static_dims["swz_B_ab"] = swz_b_ab
        op.static_dims["swz_B_c"] = swz_b_c
        return [op]

    def __init__(self, **config):
        super().__init__(**config)
        self.elem_bytes = 2
        self.num_k_blocks = (self.K + self.tile_K - 1) // self.tile_K
        self.num_mma_warps = self.threads_per_row // 32
        self.num_mma_warps = min(self.num_mma_warps, max(1, self.tile_N // 32))
        self.num_mma_threads = self.num_mma_warps * 32
        self.activation = 0
        assert self.mbar_offset + 32 <= self.page_size, (
            f"{type(self).__name__}: smem {self.mbar_offset + 32}B exceeds page_size "
            f"({self.page_size}B)."
        )

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_H,
             a, b_tma, b_tma_gmem, op_config_ptr, work_mbar):
        runtime_S = config_dim_i32(op_config_ptr, "S", type(self))
        lane_idx = cute.arch.lane_idx()
        row_start = tile_S * Int32(self.tile_size_S)
        swz_norm = cute.make_swizzle(self.swz_B_ab, 4, 3)

        for local_row in range(self.tile_size_S):
            row_idx = row_start + Int32(local_row)
            if row_idx < runtime_S:
                base = tile_B * Int32(self.a_stride_B) + row_idx * Int32(self.a_stride_S)
                a_row = cute.make_tensor(a.iterator + base, cute.make_layout(self.K))
                k = lane_idx
                while k < Int32(self.K):
                    norm_block = k // Int32(self.tile_K)
                    norm_local_k = k - norm_block * Int32(self.tile_K)
                    sA_norm = cute.make_tensor(
                        cute.recast_ptr(
                            cute.make_ptr(
                                self.a_dtype,
                                page_ptr + norm_block * Int32(self.a_tile_bytes),
                                cute.AddressSpace.smem,
                                assumed_align=128,
                            ),
                            swz_norm,
                            dtype=self.a_dtype,
                        ),
                        cute.make_layout((self.tile_K, self.tile_size_S), stride=(1, self.tile_K)),
                    )
                    sA_norm[(norm_local_k, Int32(local_row))] = a_row[k]
                    k = k + Int32(32)

        bf_0 = page_ptr + Int32(self.mbar_offset)
        bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        kr_1 = page_ptr + Int32(self.mbar_offset + 24)
        with cute.arch.elect_one():
            mbarrier_init(bf_0, Int32(self.num_mma_warps))
            mbarrier_init(bf_1, Int32(self.num_mma_warps))
            mbarrier_init(kr_0, Int32(1))
            mbarrier_init(kr_1, Int32(1))
        mbarrier_init_fence()

        total_stream_blocks = Int32(self.num_head_blocks * self.num_k_blocks)
        first_stream_blocks = Int32(2)
        if total_stream_blocks < Int32(2):
            first_stream_blocks = total_stream_blocks
        stream_idx = Int32(0)
        head_block = Int32(0)
        while head_block < Int32(self.num_head_blocks):
            out_block = tile_H * Int32(self.num_head_blocks) + head_block
            k_block = Int32(0)
            while k_block < Int32(self.num_k_blocks):
                buf_idx = stream_idx % Int32(2)
                buf_base = page_ptr + Int32(self.norm_bytes) + buf_idx * Int32(self.buf_stride)
                if stream_idx >= Int32(2):
                    bf_phase = ((stream_idx - Int32(2)) // Int32(2)) % Int32(2)
                    if buf_idx == Int32(0):
                        mbarrier_wait(bf_0, bf_phase)
                    if buf_idx == Int32(1):
                        mbarrier_wait(bf_1, bf_phase)
                sB = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(
                            self.b_dtype,
                            buf_base,
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        cute.make_swizzle(self.swz_B_ab, 4, 3),
                        dtype=self.b_dtype,
                    ),
                    cute.make_layout((self.tile_K, self.tile_N), stride=(1, self.tile_K)),
                )
                gB = cute.local_tile(b_tma_gmem, (self.tile_K, self.tile_N), (None, None))
                tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                    b_tma,
                    Int32(0),
                    cute.make_layout(1),
                    cute.group_modes(sB, 0, 2),
                    cute.group_modes(gB, 0, 2),
                )
                if stream_idx < Int32(2):
                    mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
                    if stream_idx == Int32(0):
                        with cute.arch.elect_one():
                            mbarrier_arrive_expect_tx(
                                work_mbar,
                                first_stream_blocks * Int32(self.b_tile_bytes),
                            )
                    cute.copy(b_tma, tBgB[(None, k_block, out_block)], tBsB, tma_bar_ptr=mbar_ptr)
                if stream_idx >= Int32(2):
                    kr_ptr = cute.make_ptr(cutlass.Int64, kr_0, cute.AddressSpace.smem)
                    if buf_idx == Int32(0):
                        with cute.arch.elect_one():
                            mbarrier_arrive_expect_tx(kr_0, Int32(self.b_tile_bytes))
                    if buf_idx == Int32(1):
                        kr_ptr = cute.make_ptr(cutlass.Int64, kr_1, cute.AddressSpace.smem)
                        with cute.arch.elect_one():
                            mbarrier_arrive_expect_tx(kr_1, Int32(self.b_tile_bytes))
                    cute.copy(b_tma, tBgB[(None, k_block, out_block)], tBsB, tma_bar_ptr=kr_ptr)
                k_block = k_block + Int32(1)
                stream_idx = stream_idx + Int32(1)
            head_block = head_block + Int32(1)

    @cute.jit
    def _store_head(self, page_ptr, tile_B, tile_S, tile_H, qkv, dst_k, dst_v,
                    q_norm_weight, k_norm_weight, cos, sin, op_config_ptr):
        runtime_S = config_dim_i32(op_config_ptr, "S", type(self))
        tidx = cute.arch.thread_idx()[0]
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        num_warps = self.threads_per_row // 32
        row_start = tile_S * Int32(self.tile_size_S)
        head = tile_H
        head_smem = cute.make_tensor(
            cute.make_ptr(
                self.qkv_dtype,
                page_ptr + Int32(self.head_offset),
                cute.AddressSpace.smem,
                assumed_align=128,
            ),
            cute.make_layout((self.tile_size_S, QWEN3_5_HEAD_DIM), stride=(QWEN3_5_HEAD_DIM, 1)),
        )
        sumsq_smem = cute.make_tensor(
            cute.make_ptr(
                cutlass.Float32,
                page_ptr + Int32(self.sumsq_offset),
                cute.AddressSpace.smem,
                assumed_align=128,
            ),
            cute.make_layout(self.tile_size_S),
        )

        for local_row in range(warp_idx, self.tile_size_S, num_warps):
            row_idx = row_start + Int32(local_row)
            if row_idx < runtime_S:
                qkv_base = (
                    tile_B * Int32(self.qkv_stride_B)
                    + row_idx * Int32(self.qkv_stride_S)
                    + head * Int32(self.qkv_stride_H)
                )
                qkv_row = cute.make_tensor(qkv.iterator + qkv_base, cute.make_layout(self.D))
                if head < Int32(self.num_q_heads):
                    rstd = cute.math.rsqrt(
                        sumsq_smem[local_row] * Float32(1.0 / self.D) + Float32(self.eps),
                        fastmath=True,
                    )
                    w_row = cute.make_tensor(q_norm_weight.iterator, cute.make_layout(self.D))
                    pair = lane_idx
                    while pair < Int32(self.D2):
                        lo = pair
                        hi = pair + Int32(self.D2)
                        cos_row = cute.make_tensor(
                            cos.iterator + row_idx * Int32(self.cos_stride_S),
                            cute.make_layout(self.D2),
                        )
                        sin_row = cute.make_tensor(
                            sin.iterator + row_idx * Int32(self.sin_stride_S),
                            cute.make_layout(self.D2),
                        )
                        v0 = head_smem[(Int32(local_row), lo)].to(Float32) * rstd * w_row[lo].to(Float32)
                        v1 = head_smem[(Int32(local_row), hi)].to(Float32) * rstd * w_row[hi].to(Float32)
                        c = cos_row[pair].to(Float32)
                        sn = sin_row[pair].to(Float32)
                        out0 = (v0 * c - v1 * sn).to(self.qkv_dtype)
                        out1 = (v1 * c + v0 * sn).to(self.qkv_dtype)
                        qkv_row[lo] = out0
                        qkv_row[hi] = out1
                        pair = pair + Int32(32)
                    d_norm = lane_idx + Int32(2 * self.D2)
                    while d_norm < Int32(self.D):
                        out = (
                            head_smem[(Int32(local_row), d_norm)].to(Float32)
                            * rstd
                            * w_row[d_norm].to(Float32)
                        ).to(self.qkv_dtype)
                        qkv_row[d_norm] = out
                        d_norm = d_norm + Int32(32)
                elif head < Int32(self.num_q_heads + self.num_kv_heads):
                    kv_h = head - Int32(self.num_q_heads)
                    dst_base = (
                        tile_B * Int32(self.dst_k_stride_B)
                        + (row_idx + Int32(self.cache_pos)) * Int32(self.dst_k_stride_N)
                        + kv_h * Int32(self.dst_k_stride_HK)
                    )
                    dst_row = cute.make_tensor(dst_k.iterator + dst_base, cute.make_layout(self.D))
                    rstd = cute.math.rsqrt(
                        sumsq_smem[local_row] * Float32(1.0 / self.D) + Float32(self.eps),
                        fastmath=True,
                    )
                    w_row = cute.make_tensor(k_norm_weight.iterator, cute.make_layout(self.D))
                    pair = lane_idx
                    while pair < Int32(self.D2):
                        lo = pair
                        hi = pair + Int32(self.D2)
                        cos_row = cute.make_tensor(
                            cos.iterator + row_idx * Int32(self.cos_stride_S),
                            cute.make_layout(self.D2),
                        )
                        sin_row = cute.make_tensor(
                            sin.iterator + row_idx * Int32(self.sin_stride_S),
                            cute.make_layout(self.D2),
                        )
                        v0 = head_smem[(Int32(local_row), lo)].to(Float32) * rstd * w_row[lo].to(Float32)
                        v1 = head_smem[(Int32(local_row), hi)].to(Float32) * rstd * w_row[hi].to(Float32)
                        c = cos_row[pair].to(Float32)
                        sn = sin_row[pair].to(Float32)
                        out0 = (v0 * c - v1 * sn).to(self.qkv_dtype)
                        out1 = (v1 * c + v0 * sn).to(self.qkv_dtype)
                        qkv_row[lo] = out0
                        qkv_row[hi] = out1
                        dst_row[lo] = out0
                        dst_row[hi] = out1
                        pair = pair + Int32(32)
                    d_norm = lane_idx + Int32(2 * self.D2)
                    while d_norm < Int32(self.D):
                        out = (
                            head_smem[(Int32(local_row), d_norm)].to(Float32)
                            * rstd
                            * w_row[d_norm].to(Float32)
                        ).to(self.qkv_dtype)
                        qkv_row[d_norm] = out
                        dst_row[d_norm] = out
                        d_norm = d_norm + Int32(32)
                else:
                    kv_h = head - Int32(self.num_q_heads + self.num_kv_heads)
                    dst_base = (
                        tile_B * Int32(self.dst_v_stride_B)
                        + (row_idx + Int32(self.cache_pos)) * Int32(self.dst_v_stride_N)
                        + kv_h * Int32(self.dst_v_stride_HK)
                    )
                    dst_row = cute.make_tensor(dst_v.iterator + dst_base, cute.make_layout(self.D))
                    d = lane_idx
                    while d < Int32(self.D):
                        out = head_smem[(Int32(local_row), d)]
                        qkv_row[d] = out
                        dst_row[d] = out
                        d = d + Int32(32)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_H,
                qkv, dst_k, dst_v, q_norm_weight, k_norm_weight, cos, sin, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        if tidx < Int32(self.num_mma_threads):
            runtime_S = config_dim_i32(op_config_ptr, "S", type(self))
            mma_op = cute.nvgpu.warp.MmaF16BF16Op(
                self.a_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((1, self.num_mma_warps, 1)),
                permutation_mnk=(16, self.num_mma_warps * 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)
            swz = cute.make_swizzle(self.swz_B_ab, 4, 3)

            smem_copy_atom_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.a_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
            smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
            smem_copy_atom_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.b_dtype,
            )
            smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
            smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

            sB_0 = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(
                        self.b_dtype,
                        page_ptr + Int32(self.norm_bytes),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    ),
                    swz,
                    dtype=self.b_dtype,
                ),
                cute.make_layout((self.tile_N, self.tile_K), stride=(self.tile_K, 1)),
            )
            sB_1 = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(
                        self.b_dtype,
                        page_ptr + Int32(self.norm_bytes + self.buf_stride),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    ),
                    swz,
                    dtype=self.b_dtype,
                ),
                cute.make_layout((self.tile_N, self.tile_K), stride=(self.tile_K, 1)),
            )
            sA_template = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.a_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                    swz,
                    dtype=self.a_dtype,
                ),
                cute.make_layout((self.tile_size_S, self.tile_K), stride=(self.tile_K, 1)),
            )
            tCsA = thr_mma.partition_A(sA_template)
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tCsB = thr_mma.partition_B(sB_0)
            tCrB = tiled_mma.make_fragment_B(tCsB)
            tCrA_ld = smem_thr_copy_A.retile(tCrA)
            tCrB_ld = smem_thr_copy_B.retile(tCrB)
            tBsB_ld_0 = smem_thr_copy_B.partition_S(sB_0)
            tBsB_ld_1 = smem_thr_copy_B.partition_S(sB_1)

            bf_0 = page_ptr + Int32(self.mbar_offset)
            bf_1 = page_ptr + Int32(self.mbar_offset + 8)
            kr_0 = page_ptr + Int32(self.mbar_offset + 16)
            kr_1 = page_ptr + Int32(self.mbar_offset + 24)
            kr_phase_0 = Int32(0)
            kr_phase_1 = Int32(0)
            sumsq_smem = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float32,
                    page_ptr + Int32(self.sumsq_offset),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                cute.make_layout(self.tile_size_S),
            )
            if tidx < Int32(self.tile_size_S):
                sumsq_smem[tidx] = Float32(0.0)
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
            stream_idx = Int32(0)
            head_block = Int32(0)
            while head_block < Int32(self.num_head_blocks):
                acc = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.tile_size_S, self.tile_N)),
                    Float32,
                )
                acc.fill(0.0)
                k_idx = Int32(0)
                while k_idx < Int32(self.num_k_blocks):
                    sA_k = cute.make_tensor(
                        cute.recast_ptr(
                            cute.make_ptr(
                                self.a_dtype,
                                page_ptr + k_idx * Int32(self.a_tile_bytes),
                                cute.AddressSpace.smem,
                                assumed_align=128,
                            ),
                            swz,
                            dtype=self.a_dtype,
                        ),
                        cute.make_layout((self.tile_size_S, self.tile_K), stride=(self.tile_K, 1)),
                    )
                    tAsA_ld = smem_thr_copy_A.partition_S(sA_k)
                    if stream_idx >= Int32(2):
                        if stream_idx % Int32(2) == Int32(0):
                            mbarrier_wait(kr_0, kr_phase_0)
                            kr_phase_0 = kr_phase_0 ^ Int32(1)
                        if stream_idx % Int32(2) == Int32(1):
                            mbarrier_wait(kr_1, kr_phase_1)
                            kr_phase_1 = kr_phase_1 ^ Int32(1)
                    if stream_idx % Int32(2) == Int32(0):
                        for k_block in cutlass.range_constexpr(self.tile_K // 16):
                            cute.copy(smem_tiled_copy_A, tAsA_ld[None, None, k_block], tCrA_ld[None, None, k_block])
                            cute.copy(smem_tiled_copy_B, tBsB_ld_0[None, None, k_block], tCrB_ld[None, None, k_block])
                            cute.gemm(tiled_mma, acc, tCrA[None, None, k_block], tCrB[None, None, k_block], acc)
                    if stream_idx % Int32(2) == Int32(1):
                        for k_block in cutlass.range_constexpr(self.tile_K // 16):
                            cute.copy(smem_tiled_copy_A, tAsA_ld[None, None, k_block], tCrA_ld[None, None, k_block])
                            cute.copy(smem_tiled_copy_B, tBsB_ld_1[None, None, k_block], tCrB_ld[None, None, k_block])
                            cute.gemm(tiled_mma, acc, tCrA[None, None, k_block], tCrB[None, None, k_block], acc)
                    if tidx % Int32(32) == Int32(0):
                        if stream_idx % Int32(2) == Int32(0):
                            mbarrier_arrive(bf_0)
                        if stream_idx % Int32(2) == Int32(1):
                            mbarrier_arrive(bf_1)
                    stream_idx = stream_idx + Int32(1)
                    k_idx = k_idx + Int32(1)

                _gemm_epilogue_store_no_mbar_inval_helper(
                    page_ptr + Int32(self.c_offset),
                    tidx,
                    tiled_mma,
                    acc,
                    self.num_mma_threads,
                    self.swz_B_c,
                    self.qkv_dtype,
                    self.tile_size_S,
                    self.tile_N,
                    self.activation,
                )
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
                swz_c = cute.make_swizzle(self.swz_B_c, 4, 3)
                sC = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(
                            self.qkv_dtype,
                            page_ptr + Int32(self.c_offset),
                            cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        swz_c,
                        dtype=self.qkv_dtype,
                    ),
                    cute.make_layout((self.tile_N, self.tile_size_S, 1),
                                     stride=(1, self.tile_N, self.tile_N * self.tile_size_S)),
                )
                head_smem = cute.make_tensor(
                    cute.make_ptr(
                        self.qkv_dtype,
                        page_ptr + Int32(self.head_offset),
                        cute.AddressSpace.smem,
                        assumed_align=128,
                    ),
                    cute.make_layout((self.tile_size_S, QWEN3_5_HEAD_DIM), stride=(QWEN3_5_HEAD_DIM, 1)),
                )
                copy_idx = tidx
                while copy_idx < Int32(self.tile_size_S * self.tile_N):
                    local_s = copy_idx // Int32(self.tile_N)
                    local_n = copy_idx - local_s * Int32(self.tile_N)
                    d = head_block * Int32(self.tile_N) + local_n
                    if tile_H < Int32(self.num_q_heads + self.num_kv_heads):
                        head_smem[(local_s, d)] = sC[(local_n, local_s, Int32(0))]
                    else:
                        row_idx = tile_S * Int32(self.tile_size_S) + local_s
                        if row_idx < runtime_S:
                            kv_h = tile_H - Int32(self.num_q_heads + self.num_kv_heads)
                            out = sC[(local_n, local_s, Int32(0))]
                            qkv_base = (
                                tile_B * Int32(self.qkv_stride_B)
                                + row_idx * Int32(self.qkv_stride_S)
                                + tile_H * Int32(self.qkv_stride_H)
                            )
                            qkv_row = cute.make_tensor(qkv.iterator + qkv_base, cute.make_layout(self.D))
                            dst_base = (
                                tile_B * Int32(self.dst_v_stride_B)
                                + (row_idx + Int32(self.cache_pos)) * Int32(self.dst_v_stride_N)
                                + kv_h * Int32(self.dst_v_stride_HK)
                            )
                            dst_row = cute.make_tensor(dst_v.iterator + dst_base, cute.make_layout(self.D))
                            qkv_row[d] = out
                            dst_row[d] = out
                    copy_idx = copy_idx + Int32(self.num_mma_threads)
                if tile_H < Int32(self.num_q_heads + self.num_kv_heads):
                    lane_idx = cute.arch.lane_idx()
                    warp_idx = cute.arch.warp_idx()
                    for local_row in range(warp_idx, self.tile_size_S, self.num_mma_warps):
                        partial = Float32(0.0)
                        d = lane_idx
                        while d < Int32(self.tile_N):
                            v = sC[(d, Int32(local_row), Int32(0))].to(Float32)
                            partial = partial + v * v
                            d = d + Int32(32)
                        total = cute.arch.warp_reduction(partial, operator.add)
                        if lane_idx == Int32(0):
                            sumsq_smem[local_row] = sumsq_smem[local_row] + total
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
                head_block = head_block + Int32(1)

            if tile_H < Int32(self.num_q_heads + self.num_kv_heads):
                self._store_head(page_ptr, tile_B, tile_S, tile_H, qkv, dst_k, dst_v,
                                 q_norm_weight, k_norm_weight, cos, sin, op_config_ptr)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
            if tidx == Int32(0):
                mbarrier_inval(bf_0)
                mbarrier_inval(bf_1)
                mbarrier_inval(kr_0)
                mbarrier_inval(kr_1)


class Qwen3_5PackedQkvFinalizeSm120Op(Op):
    """Finalize packed QKV chunks: Q/K RMSNorm, RoPE, and K cache write."""

    reads = {
        "qkv": (None, ("B", "S", "H", "D")),
        "qk_sumsq": (cutlass.Float32, ("B", "S", "H", "C")),
        "q_norm_weight": (None, ("D",)),
        "k_norm_weight": (None, ("D",)),
        "cos": (None, ("S", "D2")),
        "sin": (None, ("S", "D2")),
    }
    writes = {
        "qkv": (None, ("B", "S", "H", "D")),
        "dst_k": (None, ("B", "N", "HK", "D")),
    }
    tile = ("B", "S", "H")
    dynamic_dims = ("B", "S", "N")

    @classmethod
    def schedule(cls, tile_sizes=None, cache_pos=0, page_size=DEFAULT_PAGE_SIZE, eps=QWEN3_5_EPS, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", tensors["qkv"].shape[1])
        tile_sizes.setdefault("H", 1)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["cache_pos"] = cache_pos
        ops[0].static_dims["page_size"] = page_size
        ops[0].static_dims["eps"] = eps
        ops[0].static_dims["num_q_heads"] = QWEN3_5_NUM_Q_HEADS
        ops[0].static_dims["num_kv_heads"] = QWEN3_5_NUM_KV_HEADS
        return ops

    @cute.jit
    def compute(
        self,
        page_ptr,
        tile_B,
        tile_S,
        tile_H,
        qkv,
        qk_sumsq,
        q_norm_weight,
        k_norm_weight,
        cos,
        sin,
        dst_k,
        op_config_ptr,
    ):
        runtime_S = config_dim_i32(op_config_ptr, "S", type(self))
        row_start = tile_S * Int32(self.tile_size_S)
        head = tile_H * Int32(self.tile_size_H)
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()
        num_warps = self.threads_per_row // 32

        for local_row in range(warp_idx, self.tile_size_S, num_warps):
            row_idx = row_start + Int32(local_row)
            if row_idx < runtime_S:
                qkv_base = (
                    tile_B * Int32(self.qkv_stride_B)
                    + row_idx * Int32(self.qkv_stride_S)
                    + head * Int32(self.qkv_stride_H)
                )
                qkv_row = cute.make_tensor(qkv.iterator + qkv_base, cute.make_layout(self.D))
                if head < Int32(self.num_q_heads):
                    total = Float32(0.0)
                    for chunk in cutlass.range_constexpr(QWEN3_5_HEAD_DIM // 64):
                        total = total + qk_sumsq[(tile_B, row_idx, head, Int32(chunk))]
                    rstd = cute.math.rsqrt(
                        total * Float32(1.0 / self.D) + Float32(self.eps),
                        fastmath=True,
                    )
                    w_row = cute.make_tensor(q_norm_weight.iterator, cute.make_layout(self.D))
                    pair = lane_idx
                    while pair < Int32(self.D2):
                        lo = pair
                        hi = pair + Int32(self.D2)
                        cos_row = cute.make_tensor(
                            cos.iterator + row_idx * Int32(self.cos_stride_S),
                            cute.make_layout(self.D2),
                        )
                        sin_row = cute.make_tensor(
                            sin.iterator + row_idx * Int32(self.sin_stride_S),
                            cute.make_layout(self.D2),
                        )
                        v0 = qkv_row[lo].to(Float32) * rstd * w_row[lo].to(Float32)
                        v1 = qkv_row[hi].to(Float32) * rstd * w_row[hi].to(Float32)
                        c = cos_row[pair].to(Float32)
                        sn = sin_row[pair].to(Float32)
                        qkv_row[lo] = (v0 * c - v1 * sn).to(self.qkv_dtype)
                        qkv_row[hi] = (v1 * c + v0 * sn).to(self.qkv_dtype)
                        pair = pair + Int32(32)
                    d_norm = lane_idx + Int32(2 * self.D2)
                    while d_norm < Int32(self.D):
                        qkv_row[d_norm] = (
                            qkv_row[d_norm].to(Float32)
                            * rstd
                            * w_row[d_norm].to(Float32)
                        ).to(self.qkv_dtype)
                        d_norm = d_norm + Int32(32)
                elif head < Int32(self.num_q_heads + self.num_kv_heads):
                    kv_h = head - Int32(self.num_q_heads)
                    dst_base = (
                        tile_B * Int32(self.dst_k_stride_B)
                        + (row_idx + Int32(self.cache_pos)) * Int32(self.dst_k_stride_N)
                        + kv_h * Int32(self.dst_k_stride_HK)
                    )
                    dst_row = cute.make_tensor(dst_k.iterator + dst_base, cute.make_layout(self.D))
                    total = Float32(0.0)
                    for chunk in cutlass.range_constexpr(QWEN3_5_HEAD_DIM // 64):
                        total = total + qk_sumsq[(tile_B, row_idx, head, Int32(chunk))]
                    rstd = cute.math.rsqrt(
                        total * Float32(1.0 / self.D) + Float32(self.eps),
                        fastmath=True,
                    )
                    w_row = cute.make_tensor(k_norm_weight.iterator, cute.make_layout(self.D))
                    pair = lane_idx
                    while pair < Int32(self.D2):
                        lo = pair
                        hi = pair + Int32(self.D2)
                        cos_row = cute.make_tensor(
                            cos.iterator + row_idx * Int32(self.cos_stride_S),
                            cute.make_layout(self.D2),
                        )
                        sin_row = cute.make_tensor(
                            sin.iterator + row_idx * Int32(self.sin_stride_S),
                            cute.make_layout(self.D2),
                        )
                        v0 = qkv_row[lo].to(Float32) * rstd * w_row[lo].to(Float32)
                        v1 = qkv_row[hi].to(Float32) * rstd * w_row[hi].to(Float32)
                        c = cos_row[pair].to(Float32)
                        sn = sin_row[pair].to(Float32)
                        out0 = (v0 * c - v1 * sn).to(self.qkv_dtype)
                        out1 = (v1 * c + v0 * sn).to(self.qkv_dtype)
                        qkv_row[lo] = out0
                        qkv_row[hi] = out1
                        dst_row[lo] = out0
                        dst_row[hi] = out1
                        pair = pair + Int32(32)
                    d_norm = lane_idx + Int32(2 * self.D2)
                    while d_norm < Int32(self.D):
                        out = (
                            qkv_row[d_norm].to(Float32)
                            * rstd
                            * w_row[d_norm].to(Float32)
                        ).to(self.qkv_dtype)
                        qkv_row[d_norm] = out
                        dst_row[d_norm] = out
                        d_norm = d_norm + Int32(32)


class Qwen3_5QKNormRopeKCacheStoreSm120Op(_BaseQKNormRopeOp):
    """QKNorm+partial RoPE for K, then store directly to native BSHD cache."""

    reads = {
        "q": (None, ("M", "H", "D")),
        "norm_weight": (None, ("D",)),
        "cos": (None, ("S", "D2")),
        "sin": (None, ("S", "D2")),
    }
    writes = {
        "dst_k": (None, ("B", "N", "H", "D")),
    }
    tile = ("M", "H")
    dynamic_dims = ("M",)

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        page_size=DEFAULT_PAGE_SIZE,
        eps=QWEN3_5_EPS,
        cache_pos=0,
        **tensors,
    ):
        tile_sizes = dict(tile_sizes or {})
        auto = cls._auto_tiles(page_size, **tensors)
        for key, value in auto.items():
            tile_sizes.setdefault(key, value)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        ops[0].static_dims["eps"] = eps
        ops[0].static_dims["cache_pos"] = cache_pos
        return ops

    @cute.jit
    def store(self, page_ptr, tile_M, tile_H, q, norm_weight, cos, sin, dst_k, op_config_ptr):
        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))
        s2g = cute.make_copy_atom(
            CopyBulkS2GOp(),
            self.q_dtype,
            num_bits_per_copy=self.q_nbits_per_row,
        )
        pos_start = tile_M * self.tile_size_M
        head_start = tile_H * self.tile_size_H
        for local_pos in range(self.tile_size_M):
            pos = pos_start + local_pos
            if pos < runtime_M:
                seq = pos % Int32(self.S)
                batch_idx = pos // Int32(self.S)
                s_tile = cute.make_tensor(
                    cute.make_ptr(
                        self.q_dtype,
                        page_ptr + Int32(local_pos * self.q_row_elems * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.q_row_elems,)),
                )
                dst_k_tile = cute.make_tensor(
                    dst_k.iterator
                    + batch_idx * Int32(self.dst_k_stride_B)
                    + (seq + Int32(self.cache_pos)) * Int32(self.dst_k_stride_N)
                    + head_start * Int32(self.dst_k_stride_H),
                    cute.make_layout((self.q_row_elems,)),
                )
                ssrc_k, kdst = group_bulk_copy_modes(s_tile, dst_k_tile)
                cute.copy(s2g, ssrc_k, kdst)


class Qwen3_5VCacheStoreSm120Op(Op):
    """Copy BSHD V scratch blocks into their BSHD KV-cache windows."""

    reads = {"src_v": (None, ("B", "S", "H", "D"))}
    writes = {"dst_v": (None, ("B", "N", "H", "D"))}
    tile = ("B", "S", "H")
    dynamic_dims = ("B", "S", "N")

    @classmethod
    def schedule(cls, tile_sizes=None, cache_pos=0, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("S", tensors["src_v"].shape[1])
        tile_sizes.setdefault("H", tensors["src_v"].shape[2])
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["cache_pos"] = cache_pos
        return ops

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_H, src_v, dst_v, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        runtime_S = config_dim_i32(op_config_ptr, "S", type(self))
        seq_start = tile_S * Int32(self.tile_size_S)
        head_start = tile_H * Int32(self.tile_size_H)
        for local_s in range(self.tile_size_S):
            seq = seq_start + Int32(local_s)
            if seq < runtime_S:
                for local_h in range(self.tile_size_H):
                    h = head_start + Int32(local_h)
                    if h < Int32(self.H):
                        src_v_base = (
                            tile_B * Int32(self.src_v_stride_B)
                            + seq * Int32(self.src_v_stride_S)
                            + h * Int32(self.src_v_stride_H)
                        )
                        dst_v_base = (
                            tile_B * Int32(self.dst_v_stride_B)
                            + (seq + Int32(self.cache_pos)) * Int32(self.dst_v_stride_N)
                            + h * Int32(self.dst_v_stride_H)
                        )
                        src_v_row = cute.make_tensor(src_v.iterator + src_v_base, cute.make_layout(self.D))
                        dst_v_row = cute.make_tensor(dst_v.iterator + dst_v_base, cute.make_layout(self.D))
                        elem = tidx
                        while elem < Int32(self.D):
                            dst_v_row[elem] = src_v_row[elem]
                            elem = elem + Int32(self.threads_per_row)


@dataclass
class Qwen3_5DecodeLayerSchedule:
    """Scheduled ops and view keep-alives for one Qwen 3.5 SM120 decode layer."""

    ops: list
    attention_config: object
    keep_alive: list


def schedule_decode_layer_qwen3_5_sm120(
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
    h_buf,
    q_buf,
    kv_buf,
    attn_out_buf,
    proj_buf,
    h2_buf,
    gate_up_buf,
    mlp_h_buf,
    qkv_buf=None,
    qk_sumsq_buf=None,
    page_size=DEFAULT_PAGE_SIZE,
    eps=QWEN3_5_EPS,
    fa_num_splits=0,
    rms_tile_s=1,
    gate_up_tile=None,
    packed_qkv=False,
    chunked_attention_barriers=False,
):
    """Schedule one Qwen 3.5 decode layer for an SM120 megakernel.

    The scheduled layer matches the full-model decode benchmark layout:
    fused-add RMS, Q plus packed-KV projection, Q/K norm+RoPE, native BSHD
    cache update, decode attention, O projection, pre-MLP RMS, gate/up GLU,
    and down projection.
    """
    from machete.kernels.attention import FlashAttentionOp
    from machete.kernels.attention.flash_decoding import flash_decoding_schedule
    from machete.kernels.glu import GLUOp
    from machete.kernels.qknorm_rope import QKNormRopeOp
    from machete.kernels.rms_norm import RMSNormOp

    pfx = f"layer.{layer_idx}"
    cos = weights["cos"][cache_pos : cache_pos + seq_len]
    sin = weights["sin"][cache_pos : cache_pos + seq_len]
    gate_up_tile = dict(gate_up_tile or QWEN3_5_GATE_UP_TILE)
    packed_qkv = packed_qkv or qkv_buf is not None
    ops = []

    if packed_qkv:
        if qkv_buf is None:
            raise ValueError("qkv_buf must be provided when packed_qkv is enabled")
        if qk_sumsq_buf is None:
            raise ValueError("qk_sumsq_buf must be provided when packed_qkv is enabled")
        k_window = k_cache[:, : cache_pos + seq_len, :, :]
        v_window = v_cache[:, : cache_pos + seq_len, :, :]
        qkv_flat = qkv_buf.view(
            batch,
            seq_len,
            (QWEN3_5_NUM_Q_HEADS + 2 * QWEN3_5_NUM_KV_HEADS) * QWEN3_5_HEAD_DIM,
        )
        qkv_ops = Qwen3_5ComputeTmaRMSAddPackedQkvChunkProjectSm120Op.schedule(
            a=x_in,
            residual_in=residual_in,
            norm_weight=weights[f"{pfx}.attn_norm"],
            residual_out=residual_out,
            b=weights[f"{pfx}.W_qkv"],
            c=qkv_flat,
            qk_sumsq=qk_sumsq_buf,
            page_size=page_size,
            eps=eps,
        )
        for op in qkv_ops:
            op.dim_aliases["N"] = f"qkv_chunk_{layer_idx}"
            op.static_dims["barrier_group_count_N"] = QWEN3_5_NUM_Q_HEADS + 2 * QWEN3_5_NUM_KV_HEADS
        ops += qkv_ops
    elif not packed_qkv:
        ops += RMSNormOp.schedule(
            x=x_in,
            weight=weights[f"{pfx}.attn_norm"],
            y=h_buf,
            residual_in=residual_in,
            residual_out=residual_out,
            tile_sizes={"S": rms_tile_s},
            page_size=page_size,
        )
        q_ops = Qwen3_5DecodeMatvecGemmSm120Op.schedule(
            a=h_buf,
            b=weights[f"{pfx}.W_q"],
            c=q_buf,
            page_size=page_size,
        )
        for op in q_ops:
            op.dim_aliases["N"] = f"q_head_{layer_idx}"
        ops += q_ops

        kv_flat = kv_buf.view(batch, seq_len, 2 * QWEN3_5_KV_DIM)
        kv_ops = Qwen3_5DecodeMatvecGemmSm120Op.schedule(
            a=h_buf,
            b=weights[f"{pfx}.W_kv"],
            c=kv_flat,
            page_size=page_size,
        )
        for op in kv_ops:
            op.dim_aliases["N"] = f"kv_chunk_{layer_idx}"
            op.static_dims["barrier_group_count_N"] = 2 * QWEN3_5_NUM_KV_HEADS
        ops += kv_ops

    if packed_qkv:
        q_block = qkv_buf[:, :, :QWEN3_5_NUM_Q_HEADS, :]
        q_4d = q_block.as_strided(
            (batch * seq_len, QWEN3_5_NUM_Q_HEADS, QWEN3_5_HEAD_DIM),
            (q_block.stride(1), q_block.stride(2), q_block.stride(3)),
        )
        k_block = qkv_buf[:, :, QWEN3_5_NUM_Q_HEADS : QWEN3_5_NUM_Q_HEADS + QWEN3_5_NUM_KV_HEADS, :]
        qk_block = qkv_buf[:, :, : QWEN3_5_NUM_Q_HEADS + QWEN3_5_NUM_KV_HEADS, :]
        q_fd = qkv_buf[:, :, :QWEN3_5_NUM_Q_HEADS, :]
        v_block = qkv_buf[:, :, QWEN3_5_NUM_Q_HEADS + QWEN3_5_NUM_KV_HEADS :, :]
    else:
        q_4d = q_buf.view(batch * seq_len, QWEN3_5_NUM_Q_HEADS, QWEN3_5_HEAD_DIM)
        k_block = kv_buf[:, :, :QWEN3_5_NUM_KV_HEADS, :]
        q_fd = q_buf.view(batch, seq_len, QWEN3_5_NUM_Q_HEADS, QWEN3_5_HEAD_DIM)
        v_block = kv_buf[:, :, QWEN3_5_NUM_KV_HEADS :, :]

    k_4d = k_block.as_strided(
        (batch * seq_len, QWEN3_5_NUM_KV_HEADS, QWEN3_5_HEAD_DIM),
        (k_block.stride(1), k_block.stride(2), k_block.stride(3)),
    )
    k_window = k_cache[:, : cache_pos + seq_len, :, :]
    v_window = v_cache[:, : cache_pos + seq_len, :, :]
    o_fd = attn_out_buf.view(batch, seq_len, QWEN3_5_NUM_Q_HEADS, QWEN3_5_HEAD_DIM)

    if packed_qkv:
        finalize_ops = Qwen3_5PackedQkvFinalizeSm120Op.schedule(
            qkv=qk_block,
            qk_sumsq=qk_sumsq_buf,
            q_norm_weight=weights[f"{pfx}.w_q_norm"],
            k_norm_weight=weights[f"{pfx}.w_k_norm"],
            cos=cos,
            sin=sin,
            dst_k=k_window,
            cache_pos=cache_pos,
            tile_sizes={"S": seq_len, "H": 1},
            page_size=page_size,
            eps=eps,
        )
        for op in finalize_ops:
            op.dim_aliases["H"] = f"q_head_{layer_idx}"
            op.static_dims["barrier_wait_alias_H"] = f"qkv_chunk_{layer_idx}"
            op.static_dims["barrier_wait_tile_size_H"] = QWEN3_5_HEAD_DIM
            op.static_dims["barrier_wait_acquire"] = 1
            op.static_dims["barrier_signal_alias_H"] = f"q_head_{layer_idx}"
            op.static_dims["barrier_signal_tile_size_H"] = QWEN3_5_HEAD_DIM
        ops += finalize_ops

        v_cache_ops = Qwen3_5VCacheStoreSm120Op.schedule(
            src_v=v_block,
            dst_v=v_window,
            cache_pos=cache_pos,
            tile_sizes={"S": seq_len, "H": 1},
        )
        for op in v_cache_ops:
            op.dim_aliases["H"] = f"qkv_v_{layer_idx}"
            op.static_dims["barrier_wait_alias_H"] = f"qkv_chunk_{layer_idx}"
            op.static_dims["barrier_wait_tile_size_H"] = QWEN3_5_HEAD_DIM
            op.static_dims["barrier_wait_index_offset_H"] = QWEN3_5_NUM_Q_HEADS + QWEN3_5_NUM_KV_HEADS
            op.static_dims["barrier_wait_acquire"] = 1
        ops += v_cache_ops

    else:
        q_norm_ops = QKNormRopeOp.schedule(
            q=q_4d,
            norm_weight=weights[f"{pfx}.w_q_norm"],
            cos=cos,
            sin=sin,
            tile_sizes={"M": 16, "H": 1},
            page_size=page_size,
            eps=eps,
        )
        for op in q_norm_ops:
            op.dim_aliases["H"] = f"q_head_{layer_idx}"
            op.static_dims["barrier_tile_size_H"] = QWEN3_5_HEAD_DIM
        ops += q_norm_ops

        k_cache_ops = Qwen3_5QKNormRopeKCacheStoreSm120Op.schedule(
            q=k_4d,
            norm_weight=weights[f"{pfx}.w_k_norm"],
            cos=cos,
            sin=sin,
            dst_k=k_window,
            cache_pos=cache_pos,
            tile_sizes={"M": 16, "H": 1},
            page_size=page_size,
            eps=eps,
        )
        for op in k_cache_ops:
            op.dim_aliases["H"] = f"kv_chunk_{layer_idx}"
            op.static_dims["barrier_tile_size_H"] = QWEN3_5_HEAD_DIM
            if chunked_attention_barriers:
                op.static_dims["barrier_signal_alias_H"] = f"q_head_{layer_idx}"
                op.static_dims["barrier_signal_tile_size_H"] = QWEN3_5_HEAD_DIM * QWEN3_5_KV_GROUP_SIZE
        ops += k_cache_ops

        v_cache_ops = Qwen3_5VCacheStoreSm120Op.schedule(
            src_v=v_block,
            dst_v=v_window,
            cache_pos=cache_pos,
            tile_sizes={"S": seq_len, "H": 1},
        )
        for op in v_cache_ops:
            op.dim_aliases["H"] = f"kv_chunk_{layer_idx}"
            op.static_dims["barrier_wait_alias_H"] = f"kv_chunk_{layer_idx}"
            op.static_dims["barrier_wait_tile_size_H"] = QWEN3_5_HEAD_DIM
            op.static_dims["barrier_wait_index_offset_H"] = QWEN3_5_NUM_KV_HEADS
            if chunked_attention_barriers:
                op.static_dims["barrier_signal_alias_H"] = f"q_head_{layer_idx}"
                op.static_dims["barrier_signal_tile_size_H"] = QWEN3_5_HEAD_DIM * QWEN3_5_KV_GROUP_SIZE
        ops += v_cache_ops

    if k_window.shape[1] <= 256 or page_size <= 16 * 1024:
        fa_ops = FlashAttentionOp.schedule(
            q=q_fd,
            k=k_window,
            v=v_window,
            o=o_fd,
            kv_group_size=QWEN3_5_KV_GROUP_SIZE,
            page_size=page_size,
        )
        fa_config = FlashAttentionOp.kernel_config(fa_ops)
    else:
        fa_ops, fa_config = flash_decoding_schedule(
            q=q_fd,
            k=k_window,
            v=v_window,
            o=o_fd,
            kv_group_size=QWEN3_5_KV_GROUP_SIZE,
            page_size=page_size,
            num_splits=fa_num_splits,
        )
    for op in fa_ops:
        op.dim_aliases["M"] = f"fa_M_{layer_idx}"
        op.dim_aliases["H"] = f"q_head_{layer_idx}"
        op.static_dims["barrier_tile_size_H"] = QWEN3_5_HEAD_DIM
    ops += fa_ops

    ops += Qwen3_5DecodeMatvecGemmSm120Op.schedule(
        a=attn_out_buf,
        b=weights[f"{pfx}.W_o"],
        c=proj_buf,
        page_size=page_size,
    )
    ops += RMSNormOp.schedule(
        x=proj_buf,
        weight=weights[f"{pfx}.mlp_norm"],
        y=h2_buf,
        residual_in=residual_out,
        residual_out=residual_out,
        tile_sizes={"S": rms_tile_s},
        page_size=page_size,
    )
    ops += Qwen3_5DecodeMatvecGemmSm120Op.schedule(
        a=h2_buf,
        b=weights[f"{pfx}.W_gate_up"],
        c=gate_up_buf,
        page_size=page_size,
        tile_sizes=gate_up_tile,
    )
    ops += GLUOp.schedule(
        x=gate_up_buf,
        y=mlp_h_buf,
        activation="silu",
        page_size=page_size,
    )
    ops += Qwen3_5DecodeMatvecGemmSm120Op.schedule(
        a=mlp_h_buf,
        b=weights[f"{pfx}.W_down"],
        c=x_out,
        page_size=page_size,
    )

    keep_alive = [cos, sin, q_4d, k_block, k_4d, q_fd, v_block, k_window, v_window, o_fd]
    if packed_qkv:
        keep_alive.append(qk_block)
        keep_alive.append(qk_sumsq_buf)
    if not packed_qkv:
        keep_alive.append(kv_buf.view(batch, seq_len, 2 * QWEN3_5_KV_DIM))
    return Qwen3_5DecodeLayerSchedule(ops=ops, attention_config=fa_config, keep_alive=keep_alive)


def schedule_final_qwen3_5_sm120(
    *,
    x,
    residual_in,
    residual_out,
    h_final,
    final_norm,
    lm_head=None,
    logits=None,
    top_values=None,
    top_indices=None,
    seq_len,
    page_size=DEFAULT_PAGE_SIZE,
    eps=QWEN3_5_EPS,
    rms_tile_s=1,
):
    """Schedule Qwen final fused-add RMSNorm and optional in-kernel LM head."""
    from machete.kernels.rms_norm import RMSNormOp

    if lm_head is not None:
        ops = RMSNormOp.schedule(
            x=x,
            weight=final_norm,
            y=h_final,
            residual_in=residual_in,
            residual_out=residual_out,
            tile_sizes={"S": rms_tile_s},
            page_size=page_size,
        )
        if logits is not None:
            ops += Qwen3_5LmHeadGemmSm120Op.schedule(
                a=h_final,
                b=lm_head,
                c=logits,
                page_size=page_size,
            )
        elif top_values is not None and top_indices is not None:
            ops += Qwen3_5Top1LmHeadSm120Op.schedule(
                h=h_final,
                weight=lm_head,
                top_values=top_values,
                top_indices=top_indices,
                page_size=page_size,
            )
        else:
            raise ValueError(
                "logits or both top_values/top_indices must be provided when lm_head is scheduled"
            )
        return ops
    return RMSNormOp.schedule(
        x=x,
        weight=final_norm,
        y=h_final,
        residual_in=residual_in,
        residual_out=residual_out,
        tile_sizes={"S": rms_tile_s},
        page_size=page_size,
    )


__all__ = [
    "QWEN3_5_NUM_LAYERS",
    "QWEN3_5_HIDDEN",
    "QWEN3_5_INTERMEDIATE",
    "QWEN3_5_NUM_Q_HEADS",
    "QWEN3_5_NUM_KV_HEADS",
    "QWEN3_5_HEAD_DIM",
    "QWEN3_5_ROTARY_DIM",
    "QWEN3_5_ROTARY_D2",
    "QWEN3_5_Q_DIM",
    "QWEN3_5_KV_DIM",
    "QWEN3_5_KV_GROUP_SIZE",
    "QWEN3_5_EPS",
    "QWEN3_5_VOCAB",
    "QWEN3_5_DECODE_S",
    "QWEN3_5_GATE_UP_TILE",
    "Qwen3_5DecodeLayerSchedule",
    "Qwen3_5ComputeTmaRMSAddPackedQkvChunkProjectSm120Op",
    "Qwen3_5DecodeMatvecGemmSm120Op",
    "Qwen3_5LmHeadGemmSm120Op",
    "Qwen3_5PackedQkvChunkProjectSm120Op",
    "Qwen3_5PackedQkvFinalizeSm120Op",
    "Qwen3_5PackedQkvProjectSm120Op",
    "Qwen3_5QKNormRopeKCacheStoreSm120Op",
    "Qwen3_5RMSAddRangedDecodeGemmSm120Op",
    "Qwen3_5Top1LmHeadSm120Op",
    "Qwen3_5VCacheStoreSm120Op",
    "schedule_decode_layer_qwen3_5_sm120",
    "schedule_final_qwen3_5_sm120",
]
