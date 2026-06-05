# Copyright (c) 2025, Machete Authors
"""Qwen 3.5 full-attention forward op schedule.

This module owns the Qwen-specific forward graph shape, tiling policy, and
scratch tensor layout. Benchmarks and trace scripts should build the megakernel
around this schedule instead of assembling the forward ops inline.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm

from machete.megakernel.interpreter import (
    global_memory_fence_gpu,
    mbarrier_arrive,
    mbarrier_arrive_expect_tx,
    mbarrier_init,
    mbarrier_init_fence_async_proxy,
    mbarrier_inval,
    mbarrier_wait,
    named_barrier_sync,
)
from machete.kernels.attention import FlashAttentionSm120Op as _BaseFlashAttentionSm120Op
from machete.kernels.attention.flash_decoding import (
    FlashDecodingCombineBSHDOp,
    FlashDecodingSplitBSHDOp,
    FlashPrefillCombineBSHDOp,
    FlashPrefillSplitBSHDOp,
)
from machete.kernels.decode_matvec import ResidualAddSm120Op as _BaseResidualAddSm120Op
from machete.kernels.gemm import GemmOp as _BaseGemmOp
from machete.kernels.gemm.gemm import _gemm_epilogue_store_no_mbar_inval_helper
from machete.kernels.glu import DirectGLUOp, GLUOp as _BaseGLUOp
from machete.kernels.qknorm_rope import (
    PackedQKNormRopeOp as _BasePackedQKNormRopeOp,
    QKNormRopeOp as _BaseQKNormRopeOp,
)
from machete.kernels.qknorm_rope.qknorm_rope import (
    CopyBulkG2SOp,
    CopyBulkS2GOp,
    group_bulk_copy_modes,
)
from machete.kernels.qwen_3_5.sm120 import (
    QWEN3_5_EPS,
    Qwen3_5PackedQkvChunkProjectSm120Op,
)
from machete.kernels.rms_norm.rms_norm import (
    RMSNormOp,
    SCRATCH_BYTES,
    _auto_chunked_tile_S,
    _expand_weight,
    _pick_rmsnorm_tma_tile_d,
    _rowwise_chunked_bytes,
)
from machete.megakernel.scheduling import OverlapTileScheduler
from machete.megakernel.ops import Op, config_dim_i32


DEFAULT_PAGE_SIZE = 32768
HIDDEN = 1024
INTERMEDIATE = 3584
NUM_Q_HEADS = 8
NUM_KV_HEADS = 2
HEAD_DIM = 256
Q_DIM = NUM_Q_HEADS * HEAD_DIM
KV_DIM = NUM_KV_HEADS * HEAD_DIM
KV_GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS


@dsl_user_op
def _atomic_add_f32(val: Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    nvvm.atomicrmw(
        res=T.f32(),
        op=nvvm.AtomicOpKind.FADD,
        ptr=gmem_ptr.llvm_ptr,
        a=Float32(val).ir_value(),
    )


@dataclass
class Qwen3_5ForwardSchedule:
    ops: list
    output: torch.Tensor
    residual: torch.Tensor
    keep_alive: list


class Qwen3_5ForwardRMSNormOp(RMSNormOp):
    """RMSNorm schedule variant for Qwen forward overlap.

    The generic RMSNorm schedule caps large hidden-size tiles aggressively.
    For Qwen forward, the producer tile must be compatible with the following
    projection GEMM row tile so the GEMM can start after only its row slice is
    normalized. This schedule uses the largest row tile that fits the shared
    memory page and opts into generic piecewise row dependencies.
    """

    @classmethod
    def _fill_dummies(cls, tensors):
        x, y = tensors["x"], tensors["y"]
        has_residual = "residual_in" in tensors
        has_gate = "gate" in tensors
        if not has_residual:
            tensors["residual_in"] = torch.empty_like(x)
        if "residual_out" not in tensors:
            tensors["residual_out"] = torch.empty_like(y)
        if not has_gate:
            tensors["gate"] = torch.empty_like(x)
        return has_residual, has_gate

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        residual=False,
        gemma=False,
        per_row_weight=False,
        page_size=DEFAULT_PAGE_SIZE,
        **tensors,
    ):
        tensors = dict(tensors)
        _expand_weight(tensors)
        has_residual, has_gate = cls._fill_dummies(tensors)
        tile_sizes = dict(tile_sizes or {})
        D = tensors["x"].shape[-1]
        elem_bytes = tensors["x"].element_size()
        fit_tile_s = _auto_chunked_tile_S(D, elem_bytes, page_size, SCRATCH_BYTES)
        if "S" not in tile_sizes:
            tile_sizes["S"] = fit_tile_s
        else:
            tile_sizes["S"] = min(tile_sizes["S"], fit_tile_s)
            while (
                tile_sizes["S"] > 1
                and _rowwise_chunked_bytes(D, tile_sizes["S"], elem_bytes) + SCRATCH_BYTES > page_size
            ):
                tile_sizes["S"] -= 1
        tile_sizes.setdefault("B", 1)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        if residual:
            ops[0].static_dims["residual"] = 1
        if gemma:
            ops[0].static_dims["gemma"] = 1
        if has_residual:
            ops[0].static_dims["has_residual"] = 1
        if has_gate:
            ops[0].static_dims["has_gate"] = 1
        if per_row_weight:
            ops[0].static_dims["per_row_weight"] = 1
        ops[0].static_dims["page_size"] = page_size
        ops[0].static_dims["tma_tile_D"] = _pick_rmsnorm_tma_tile_d(
            D,
            tile_sizes["S"],
            elem_bytes,
            page_size,
        )
        ops[0].static_dims["barrier_allow_piecewise_overlap"] = 1
        return ops


class Qwen3_5RmsProjOp(_BaseGemmOp):
    """Fused RMSNorm + projection GEMM, written for the Qwen forward layer.

    Computes ``C = rstd * (x @ B_fused^T)`` in ONE op, where
    ``B_fused[n,k] = w[n,k] * rmsnorm_weight[k]`` (baked at schedule time) and
    ``rstd = rsqrt(mean(x^2) + eps)`` per row.

    Unlike the generic ``RMSNormGemmOp`` (which re-reads A from global in the
    epilogue to form ``sum(x^2)`` — a net loss because the projection already
    re-reads A once per N-tile), this op accumulates ``sum(x^2)`` per row *from
    the A register fragments that are already loaded for the MMA* during the
    K-loop. That adds only cheap ALU and ZERO extra memory traffic, so the fused
    op removes the separate RMSNorm pass and its ``x0`` global round-trip for
    free. The persistent megakernel keeps ``x`` resident / prefetched, an
    advantage torch.compile cannot match (it must materialize ``x0`` to global
    between its rmsnorm and matmul kernels).

    A-fragment row mapping (m16n8k16): each warp owns 16 rows; lane ``l`` owns
    rows ``g=l//4`` (lo) and ``g+8`` (hi). The 4 lanes sharing ``g`` cover
    disjoint K columns, so a butterfly reduction over the group-of-4 sums the
    full row. This matches the C-fragment scaling (``ci%4<2`` -> lo).
    """

    dynamic_dims = ("B",)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_N, a, a_scale, b, c):
        tidx = cute.arch.thread_idx()[0]

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(self.a_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.num_mma_warps, 1, 1)),
            permutation_mnk=(self.num_mma_warps * 16, 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        swz = cute.make_swizzle(self.swz_B_ab, 4, 3)
        smem_copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.a_dtype)
        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
        smem_copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.b_dtype)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        def _mk_A(off):
            return cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.a_dtype, page_ptr + Int32(off),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.a_dtype),
                cute.make_layout((self.tile_size_S, self.tile_K), stride=(self.tile_K, 1)))

        def _mk_B(off):
            return cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.b_dtype, page_ptr + Int32(off),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.b_dtype),
                cute.make_layout((self.tile_size_N, self.tile_K), stride=(self.tile_K, 1)))

        sA_0 = _mk_A(0)
        sB_0 = _mk_B(self.b_offset)
        sA_1 = _mk_A(self.buf_stride)
        sB_1 = _mk_B(self.buf_stride + self.b_offset)

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

        _bf_0 = page_ptr + Int32(self.mbar_offset)
        _bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        _kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        _kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        acc = cute.make_fragment(
            tiled_mma.partition_shape_C((self.tile_size_S, self.tile_size_N)), Float32)
        acc.fill(0.0)
        sq_lo = Float32(0.0)
        sq_hi = Float32(0.0)

        # Accumulate sum(x^2) for this thread's two rows from the A fragment of
        # one 16-wide k-block. lo rows = fragment elems where (i//2)%2==0.
        def _accum_sq(frag, sq_lo, sq_hi):
            for i in cutlass.range_constexpr(cute.size(frag)):
                v = frag[i].to(Float32)
                if (i // 2) % 2 == 0:
                    sq_lo = sq_lo + v * v
                else:
                    sq_hi = sq_hi + v * v
            return sq_lo, sq_hi

        for kb in cutlass.range_constexpr(self.tile_K // 16):
            cute.copy(smem_tiled_copy_A, tAsA_ld_0[None, None, kb], tCrA_ld[None, None, kb])
            cute.copy(smem_tiled_copy_B, tBsB_ld_0[None, None, kb], tCrB_ld[None, None, kb])
            sq_lo, sq_hi = _accum_sq(tCrA[None, None, kb], sq_lo, sq_hi)
            cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
        if tidx % Int32(32) == Int32(0):
            mbarrier_arrive(_bf_0)

        if self.num_k_blocks >= 2:
            for kb in cutlass.range_constexpr(self.tile_K // 16):
                cute.copy(smem_tiled_copy_A, tAsA_ld_1[None, None, kb], tCrA_ld[None, None, kb])
                cute.copy(smem_tiled_copy_B, tBsB_ld_1[None, None, kb], tCrB_ld[None, None, kb])
                sq_lo, sq_hi = _accum_sq(tCrA[None, None, kb], sq_lo, sq_hi)
                cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
            if tidx % Int32(32) == Int32(0):
                mbarrier_arrive(_bf_1)

        _kr_phase_0 = Int32(0)
        _kr_phase_1 = Int32(0)
        k_idx = Int32(2)
        while k_idx < Int32(self.num_k_blocks):
            if k_idx % Int32(2) == Int32(0):
                mbarrier_wait(_kr_0, _kr_phase_0)
                _kr_phase_0 = _kr_phase_0 ^ Int32(1)
                for kb in cutlass.range_constexpr(self.tile_K // 16):
                    cute.copy(smem_tiled_copy_A, tAsA_ld_0[None, None, kb], tCrA_ld[None, None, kb])
                    cute.copy(smem_tiled_copy_B, tBsB_ld_0[None, None, kb], tCrB_ld[None, None, kb])
                    sq_lo, sq_hi = _accum_sq(tCrA[None, None, kb], sq_lo, sq_hi)
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
                if tidx % Int32(32) == Int32(0):
                    mbarrier_arrive(_bf_0)
            else:
                mbarrier_wait(_kr_1, _kr_phase_1)
                _kr_phase_1 = _kr_phase_1 ^ Int32(1)
                for kb in cutlass.range_constexpr(self.tile_K // 16):
                    cute.copy(smem_tiled_copy_A, tAsA_ld_1[None, None, kb], tCrA_ld[None, None, kb])
                    cute.copy(smem_tiled_copy_B, tBsB_ld_1[None, None, kb], tCrB_ld[None, None, kb])
                    sq_lo, sq_hi = _accum_sq(tCrA[None, None, kb], sq_lo, sq_hi)
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
                if tidx % Int32(32) == Int32(0):
                    mbarrier_arrive(_bf_1)
            k_idx = k_idx + Int32(1)

        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
        if tidx == Int32(0):
            mbarrier_inval(_bf_0)
            mbarrier_inval(_bf_1)
            mbarrier_inval(_kr_0)
            mbarrier_inval(_kr_1)

        # Reduce sum(x^2) across the 4 lanes that share each row, then rstd.
        sq_lo = sq_lo + cute.arch.shuffle_sync_bfly(sq_lo, offset=1)
        sq_lo = sq_lo + cute.arch.shuffle_sync_bfly(sq_lo, offset=2)
        sq_hi = sq_hi + cute.arch.shuffle_sync_bfly(sq_hi, offset=1)
        sq_hi = sq_hi + cute.arch.shuffle_sync_bfly(sq_hi, offset=2)
        _inv_K = Float32(1.0 / self.K)
        rstd_lo = cute.math.rsqrt(sq_lo * _inv_K + Float32(QWEN3_5_EPS), fastmath=True)
        rstd_hi = cute.math.rsqrt(sq_hi * _inv_K + Float32(QWEN3_5_EPS), fastmath=True)

        for ci in cutlass.range_constexpr(cute.size(acc)):
            if ci % 4 < 2:
                acc[ci] = acc[ci] * rstd_lo
            else:
                acc[ci] = acc[ci] * rstd_hi

        swz_c = cute.make_swizzle(self.swz_B_c, 4, 3)
        sC = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.c_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                swz_c, dtype=self.c_dtype),
            cute.make_layout((self.tile_size_S, self.tile_size_N),
                             stride=(self.tile_size_N, 1)))
        acc_out = cute.make_fragment_like(acc, self.c_dtype)
        for ci in cutlass.range_constexpr(cute.size(acc)):
            acc_out[ci] = acc[ci].to(self.c_dtype)
        r2s_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
        r2s_copy = cute.make_tiled_copy_C(r2s_atom, tiled_mma)
        r2s_thr = r2s_copy.get_slice(tidx)
        cute.copy(r2s_copy, r2s_thr.retile(acc_out), r2s_thr.partition_D(sC))

    @classmethod
    def schedule(cls, rmsnorm_weight=None, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        b = tensors["b"]
        if rmsnorm_weight is not None:
            w = rmsnorm_weight.float()
            tensors["b"] = (b.float() * w.unsqueeze(0)).to(b.dtype).contiguous()
        tile_sizes = dict(tile_sizes or {})
        a = tensors.get("a")
        if a is not None:
            _complete_qwen_forward_gemm_tile_sizes(tile_sizes, a, b, page_size)
        ops = super().schedule(tile_sizes=tile_sizes, page_size=page_size, **tensors)
        for op in ops:
            op.static_dims["barrier_allow_piecewise_overlap"] = 1
        return ops


class Qwen3_5DownResidualOp(_BaseGemmOp):
    """Fused down-projection GEMM + residual add: ``y = mlp @ w_down^T + residual``.

    Folds the final residual add into the down GEMM epilogue, removing the
    standalone residual-add op (a serial tail in the layer) and the ``mlp_out``
    global round-trip. ``residual`` is a regular (non-TMA) read, so the framework
    hands it to ``compute`` as a global CuTe tensor; the epilogue adds it through
    ``partition_C`` (identical layout to the MMA accumulator), needing no manual
    C-fragment row/col mapping.
    """

    reads = {**_BaseGemmOp.reads, "residual": (None, ("B", "S", "N"))}
    dynamic_dims = ("B",)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_N, a, a_scale, b, residual, c):
        # NOTE: param order MUST match the framework's canonical tensor order
        # (reads then writes => a, a_scale, b, residual, c). The wrapper passes
        # tensors positionally, so listing c before residual silently swaps them.
        tidx = cute.arch.thread_idx()[0]

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(self.a_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.num_mma_warps, 1, 1)),
            permutation_mnk=(self.num_mma_warps * 16, 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        swz = cute.make_swizzle(self.swz_B_ab, 4, 3)
        smem_copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.a_dtype)
        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
        smem_copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.b_dtype)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        def _mk_A(off):
            return cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.a_dtype, page_ptr + Int32(off),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.a_dtype),
                cute.make_layout((self.tile_size_S, self.tile_K), stride=(self.tile_K, 1)))

        def _mk_B(off):
            return cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.b_dtype, page_ptr + Int32(off),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.b_dtype),
                cute.make_layout((self.tile_size_N, self.tile_K), stride=(self.tile_K, 1)))

        sA_0 = _mk_A(0)
        sB_0 = _mk_B(self.b_offset)
        sA_1 = _mk_A(self.buf_stride)
        sB_1 = _mk_B(self.buf_stride + self.b_offset)

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

        _bf_0 = page_ptr + Int32(self.mbar_offset)
        _bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        _kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        _kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        acc = cute.make_fragment(
            tiled_mma.partition_shape_C((self.tile_size_S, self.tile_size_N)), Float32)
        acc.fill(0.0)

        for kb in cutlass.range_constexpr(self.tile_K // 16):
            cute.copy(smem_tiled_copy_A, tAsA_ld_0[None, None, kb], tCrA_ld[None, None, kb])
            cute.copy(smem_tiled_copy_B, tBsB_ld_0[None, None, kb], tCrB_ld[None, None, kb])
            cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
        if tidx % Int32(32) == Int32(0):
            mbarrier_arrive(_bf_0)

        if self.num_k_blocks >= 2:
            for kb in cutlass.range_constexpr(self.tile_K // 16):
                cute.copy(smem_tiled_copy_A, tAsA_ld_1[None, None, kb], tCrA_ld[None, None, kb])
                cute.copy(smem_tiled_copy_B, tBsB_ld_1[None, None, kb], tCrB_ld[None, None, kb])
                cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
            if tidx % Int32(32) == Int32(0):
                mbarrier_arrive(_bf_1)

        _kr_phase_0 = Int32(0)
        _kr_phase_1 = Int32(0)
        k_idx = Int32(2)
        while k_idx < Int32(self.num_k_blocks):
            if k_idx % Int32(2) == Int32(0):
                mbarrier_wait(_kr_0, _kr_phase_0)
                _kr_phase_0 = _kr_phase_0 ^ Int32(1)
                for kb in cutlass.range_constexpr(self.tile_K // 16):
                    cute.copy(smem_tiled_copy_A, tAsA_ld_0[None, None, kb], tCrA_ld[None, None, kb])
                    cute.copy(smem_tiled_copy_B, tBsB_ld_0[None, None, kb], tCrB_ld[None, None, kb])
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
                if tidx % Int32(32) == Int32(0):
                    mbarrier_arrive(_bf_0)
            else:
                mbarrier_wait(_kr_1, _kr_phase_1)
                _kr_phase_1 = _kr_phase_1 ^ Int32(1)
                for kb in cutlass.range_constexpr(self.tile_K // 16):
                    cute.copy(smem_tiled_copy_A, tAsA_ld_1[None, None, kb], tCrA_ld[None, None, kb])
                    cute.copy(smem_tiled_copy_B, tBsB_ld_1[None, None, kb], tCrB_ld[None, None, kb])
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
                if tidx % Int32(32) == Int32(0):
                    mbarrier_arrive(_bf_1)
            k_idx = k_idx + Int32(1)

        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
        if tidx == Int32(0):
            mbarrier_inval(_bf_0)
            mbarrier_inval(_bf_1)
            mbarrier_inval(_kr_0)
            mbarrier_inval(_kr_1)

        # Fuse the residual add: y = acc + residual. Build a global view of this
        # output tile [tile_size_S, tile_size_N] (contiguous [B,S,N], row stride
        # = N) and let the tiled MMA partition it exactly like the accumulator,
        # so acc[ci] and tCres[ci] are the same (row,col) — no manual mapping.
        _res_base = (
            tile_B * Int32(self.S * self.N)
            + tile_S * Int32(self.tile_size_S) * Int32(self.N)
            + tile_N * Int32(self.tile_size_N)
        )
        g_res = cute.make_tensor(
            residual.iterator + _res_base,
            cute.make_layout((self.tile_size_S, self.tile_size_N), stride=(self.N, 1)))
        tCres = thr_mma.partition_C(g_res)
        for ci in cutlass.range_constexpr(cute.size(acc)):
            acc[ci] = acc[ci] + tCres[ci].to(Float32)

        swz_c = cute.make_swizzle(self.swz_B_c, 4, 3)
        sC = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.c_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                swz_c, dtype=self.c_dtype),
            cute.make_layout((self.tile_size_S, self.tile_size_N),
                             stride=(self.tile_size_N, 1)))
        acc_out = cute.make_fragment_like(acc, self.c_dtype)
        for ci in cutlass.range_constexpr(cute.size(acc)):
            acc_out[ci] = acc[ci].to(self.c_dtype)
        r2s_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
        r2s_copy = cute.make_tiled_copy_C(r2s_atom, tiled_mma)
        r2s_thr = r2s_copy.get_slice(tidx)
        cute.copy(r2s_copy, r2s_thr.retile(acc_out), r2s_thr.partition_D(sC))

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        if "a_scale" not in tensors:
            a = tensors.get("a")
            if a is not None:
                tensors["a_scale"] = torch.empty_like(a)
        tile_sizes = dict(tile_sizes or {})
        a = tensors.get("a")
        b = tensors.get("b")
        if a is not None and b is not None:
            _complete_qwen_forward_gemm_tile_sizes(tile_sizes, a, b, page_size)
        ops = super().schedule(tile_sizes=tile_sizes, page_size=page_size, **tensors)
        for op in ops:
            op.static_dims["barrier_allow_piecewise_overlap"] = 1
        return ops


class Qwen3_5ForwardProjectionOp(_BaseGemmOp):
    """Qwen forward projection GEMM.

    This keeps Qwen forward scheduling policy out of the generic GEMM op.  The
    current implementation still uses the proven GEMM phase bodies, but all
    tile/dependency choices are owned here so the op can diverge without
    changing other models.
    """

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        page_size=DEFAULT_PAGE_SIZE,
        **tensors,
    ):
        if "a_scale" not in tensors:
            a = tensors.get("a")
            if a is not None:
                # Generic GEMM uses a_scale=a as a no-op placeholder when
                # has_a_scale=0. In the Qwen forward schedule that creates a
                # second, false producer dependency on every projection input.
                # Keep a same-shaped independent placeholder instead; the
                # kernel never loads it while has_a_scale=0.
                tensors["a_scale"] = torch.empty_like(a)
        tile_sizes = dict(tile_sizes or {})
        a = tensors.get("a")
        b = tensors.get("b")
        if a is not None and b is not None:
            _complete_qwen_forward_gemm_tile_sizes(tile_sizes, a, b, page_size)
        ops = super().schedule(tile_sizes=tile_sizes, page_size=page_size, **tensors)
        for op in ops:
            op.static_dims["barrier_allow_piecewise_overlap"] = 1
        return ops


class Qwen3_5ForwardCpAsyncProjectionOp(_BaseGemmOp):
    """Hybrid Qwen projection GEMM.

    The load warp TMA-prefetches the first two K-blocks so dependency/scheduler
    overlap stays identical to the regular GEMM prologue.  MMA warps then load
    later K-blocks with cp.async inside compute, leaving the framework load warp
    available for other ops instead of driving the long inner-K tail.
    """

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_N,
             a_tma, a_tma_gmem, a_scale_tma, a_scale_tma_gmem,
             b_tma, b_tma_gmem,
             work_mbar):
        swz = cute.make_swizzle(self.swz_B_ab, 4, 3)
        _mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, Int32(self.tma_k_blocks * self.ab_tma_bytes))

        _k_block = Int32(0)
        while _k_block < Int32(self.tma_k_blocks):
            _buf_base = (_k_block % Int32(2)) * Int32(self.buf_stride) + page_ptr

            sA_ptr = cute.recast_ptr(
                cute.make_ptr(self.a_dtype, _buf_base, cute.AddressSpace.smem),
                swz, dtype=self.a_dtype)
            sA = cute.make_tensor(
                sA_ptr,
                cute.make_layout((self.tile_K, self.tile_size_S, 1),
                                 stride=(1, self.tile_K, self.tile_K * self.tile_size_S)))
            gA = cute.local_tile(a_tma_gmem, (self.tile_K, self.tile_size_S, 1), (None, None, None))
            tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                a_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sA, 0, 3), cute.group_modes(gA, 0, 3))

            sB_ptr = cute.recast_ptr(
                cute.make_ptr(self.b_dtype, _buf_base + Int32(self.b_offset), cute.AddressSpace.smem),
                swz, dtype=self.b_dtype)
            sB = cute.make_tensor(
                sB_ptr,
                cute.make_layout((self.tile_K, self.tile_size_N), stride=(1, self.tile_K)))
            gB = cute.local_tile(b_tma_gmem, (self.tile_K, self.tile_size_N), (None, None))
            tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                b_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))

            cute.copy(a_tma, tAgA[(None, _k_block, tile_S, tile_B)], tAsA, tma_bar_ptr=_mbar_ptr)
            cute.copy(b_tma, tBgB[(None, _k_block, tile_N)], tBsB, tma_bar_ptr=_mbar_ptr)
            _k_block = _k_block + Int32(1)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_N, a, a_scale, b):
        tidx = cute.arch.thread_idx()[0]

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(self.a_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.num_mma_warps_m, self.num_mma_warps_n, 1)),
            permutation_mnk=(self.num_mma_warps_m * 16, self.num_mma_warps_n * 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        swz = cute.make_swizzle(self.swz_B_ab, 4, 3)
        smem_copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.a_dtype)
        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)
        smem_copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.b_dtype)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        sA_0 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.a_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.a_dtype),
            cute.make_layout((self.tile_size_S, self.tile_K), stride=(self.tile_K, 1)))
        sB_0 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.b_dtype, page_ptr + Int32(self.b_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.b_dtype),
            cute.make_layout((self.tile_size_N, self.tile_K), stride=(self.tile_K, 1)))
        sA_1 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.a_dtype, page_ptr + Int32(self.buf_stride),
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.a_dtype),
            cute.make_layout((self.tile_size_S, self.tile_K), stride=(self.tile_K, 1)))
        sB_1 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.b_dtype, page_ptr + Int32(self.buf_stride + self.b_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.b_dtype),
            cute.make_layout((self.tile_size_N, self.tile_K), stride=(self.tile_K, 1)))

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

        async_copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), self.a_dtype, num_bits_per_copy=128)
        async_copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), self.b_dtype, num_bits_per_copy=128)
        copy_thread_layout = cute.make_layout(
            (self.cpasync_copy_dim0, self.cpasync_copy_dim1),
            stride=(self.cpasync_copy_dim1, 1))
        copy_value_layout = cute.make_layout((1, self.cpasync_copy_elems))
        g2s_copy_A = cute.make_tiled_copy_tv(async_copy_atom_A, copy_thread_layout, copy_value_layout)
        g2s_copy_B = cute.make_tiled_copy_tv(async_copy_atom_B, copy_thread_layout, copy_value_layout)
        thr_copy_A = g2s_copy_A.get_slice(tidx)
        thr_copy_B = g2s_copy_B.get_slice(tidx)
        tAsA_cp_0 = thr_copy_A.partition_D(sA_0)
        tBsB_cp_0 = thr_copy_B.partition_D(sB_0)
        tAsA_cp_1 = thr_copy_A.partition_D(sA_1)
        tBsB_cp_1 = thr_copy_B.partition_D(sB_1)

        a_base = (
            tile_B * Int32(self.a_stride_B)
            + tile_S * Int32(self.tile_size_S) * Int32(self.a_stride_S)
        )
        b_base = tile_N * Int32(self.tile_size_N) * Int32(self.b_stride_N)
        gA_full = cute.make_tensor(
            (a.iterator + a_base).align(16),
            cute.make_layout((self.tile_size_S, self.K), stride=(self.a_stride_S, self.a_stride_K)))
        gB_full = cute.make_tensor(
            (b.iterator + b_base).align(16),
            cute.make_layout((self.tile_size_N, self.K), stride=(self.b_stride_N, self.b_stride_K)))

        acc = cute.make_fragment(
            tiled_mma.partition_shape_C((self.tile_size_S, self.tile_size_N)), Float32)
        acc.fill(0.0)

        k_idx = Int32(0)
        while k_idx < Int32(self.num_k_blocks):
            if k_idx >= Int32(2):
                if k_idx + Int32(1) < Int32(self.num_k_blocks):
                    cute.arch.cp_async_wait_group(1)
                if k_idx + Int32(1) >= Int32(self.num_k_blocks):
                    cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            if k_idx % Int32(2) == Int32(0):
                for k_block in cutlass.range_constexpr(self.tile_K // 16):
                    cute.copy(smem_tiled_copy_A, tAsA_ld_0[None, None, k_block], tCrA_ld[None, None, k_block])
                    cute.copy(smem_tiled_copy_B, tBsB_ld_0[None, None, k_block], tCrB_ld[None, None, k_block])
                    cute.gemm(tiled_mma, acc, tCrA[None, None, k_block], tCrB[None, None, k_block], acc)
            if k_idx % Int32(2) == Int32(1):
                for k_block in cutlass.range_constexpr(self.tile_K // 16):
                    cute.copy(smem_tiled_copy_A, tAsA_ld_1[None, None, k_block], tCrA_ld[None, None, k_block])
                    cute.copy(smem_tiled_copy_B, tBsB_ld_1[None, None, k_block], tCrB_ld[None, None, k_block])
                    cute.gemm(tiled_mma, acc, tCrA[None, None, k_block], tCrB[None, None, k_block], acc)
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            prefetch_idx = k_idx + Int32(2)
            if prefetch_idx < Int32(self.num_k_blocks):
                gA_block = cute.local_tile(gA_full, (self.tile_size_S, self.tile_K), (Int32(0), prefetch_idx))
                gB_block = cute.local_tile(gB_full, (self.tile_size_N, self.tile_K), (Int32(0), prefetch_idx))
                tAgA = thr_copy_A.partition_S(gA_block)
                tBgB = thr_copy_B.partition_S(gB_block)
                if k_idx % Int32(2) == Int32(0):
                    for ci in cutlass.range_constexpr(cute.size(tAsA_cp_0.shape[2])):
                        cute.copy(g2s_copy_A, tAgA[None, None, ci], tAsA_cp_0[None, None, ci])
                    for ci in cutlass.range_constexpr(cute.size(tBsB_cp_0.shape[2])):
                        cute.copy(g2s_copy_B, tBgB[None, None, ci], tBsB_cp_0[None, None, ci])
                if k_idx % Int32(2) == Int32(1):
                    for ci in cutlass.range_constexpr(cute.size(tAsA_cp_1.shape[2])):
                        cute.copy(g2s_copy_A, tAgA[None, None, ci], tAsA_cp_1[None, None, ci])
                    for ci in cutlass.range_constexpr(cute.size(tBsB_cp_1.shape[2])):
                        cute.copy(g2s_copy_B, tBgB[None, None, ci], tBsB_cp_1[None, None, ci])
                cute.arch.cp_async_commit_group()
            k_idx = k_idx + Int32(1)

        _gemm_epilogue_store_no_mbar_inval_helper(
            page_ptr, tidx, tiled_mma, acc,
            self.num_mma_threads, self.swz_B_c, self.c_dtype,
            self.tile_size_S, self.tile_size_N, self.activation,
        )

    def __init__(self, **config):
        super().__init__(**config)
        if self.has_a_scale:
            raise ValueError("Qwen3_5ForwardCpAsyncProjectionOp only supports has_a_scale=0")
        self.cpasync_copy_elems = 128 // (self.elem_bytes * 8)
        self.cpasync_copy_dim1 = self.tile_K // self.cpasync_copy_elems
        self.cpasync_copy_dim0 = self.num_mma_threads // self.cpasync_copy_dim1
        assert self.cpasync_copy_dim0 > 0, (
            f"Qwen3_5ForwardCpAsyncProjectionOp: not enough MMA threads "
            f"({self.num_mma_threads}) for tile_K={self.tile_K} cp.async"
        )

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        if "a_scale" not in tensors:
            a = tensors.get("a")
            if a is not None:
                tensors["a_scale"] = torch.empty_like(a)
        tile_sizes = dict(tile_sizes or {})
        a = tensors.get("a")
        b = tensors.get("b")
        if a is not None and b is not None:
            _complete_qwen_forward_gemm_tile_sizes(tile_sizes, a, b, page_size)
        ops = super().schedule(tile_sizes=tile_sizes, page_size=page_size, **tensors)
        for op in ops:
            op.static_dims["barrier_allow_piecewise_overlap"] = 1
        return ops


class Qwen3_5ForwardPackedQKProjectionOp(Qwen3_5PackedQkvChunkProjectSm120Op):
    """Qwen forward packed Q/K projection with Q/K norm partial statistics."""

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("S", 128)
        tile_sizes.setdefault("N", 64)
        tile_sizes.setdefault("K", 32)
        ops = super().schedule(tile_sizes=tile_sizes, page_size=page_size, **tensors)
        for op in ops:
            op.static_dims["barrier_allow_piecewise_overlap"] = 1
        return ops


class Qwen3_5ForwardPackedQKNormRopeOp(_BasePackedQKNormRopeOp):
    """Qwen forward packed Q/K norm+RoPE boundary op.

    Unlike the generic packed op, this local variant honors the captured
    ``qk`` strides. That lets the forward schedule pass Q/K as a last-dim slice
    of a larger QKV projection buffer without changing generic qknorm behavior.
    """

    @cute.jit
    def load(self, page_ptr, tile_M, tile_H, qk, q_norm_weight, k_norm_weight, cos, sin,
             op_config_ptr, work_mbar):
        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))
        pos_start = tile_M * self.tile_size_M
        head_start = tile_H * self.tile_size_H

        qk_per_pos_bytes = Int32(self.qk_row_elems * self.elem_bytes)
        cs_per_pos_bytes = Int32(2 * self.D2 * self.elem_bytes)
        actual_rows = Int32(self.tile_size_M)
        remaining = runtime_M - pos_start
        if remaining < Int32(self.tile_size_M):
            actual_rows = remaining
        total_bytes = actual_rows * (qk_per_pos_bytes + cs_per_pos_bytes)

        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        mbarrier_arrive_expect_tx(work_mbar, total_bytes)

        g2s_qk = cute.make_copy_atom(
            CopyBulkG2SOp(),
            self.qk_dtype,
            num_bits_per_copy=self.qk_nbits_per_row,
        )
        g2s_cs = cute.make_copy_atom(
            CopyBulkG2SOp(),
            self.qk_dtype,
            num_bits_per_copy=self.cs_nbits_per_row,
        )

        cos_smem_start = page_ptr + Int32(self.qk_tile_bytes)
        sin_smem_start = page_ptr + Int32(self.qk_tile_bytes + self.cs_tile_bytes)

        for local_pos in range(self.tile_size_M):
            pos = pos_start + local_pos
            if pos < runtime_M:
                s = pos % self.S
                g_qk = cute.make_tensor(
                    qk.iterator + pos * Int32(self.qk_stride_M) + head_start * Int32(self.qk_stride_H),
                    cute.make_layout((self.qk_row_elems,)),
                )
                s_qk = cute.make_tensor(
                    cute.make_ptr(
                        self.qk_dtype,
                        page_ptr + Int32(local_pos * self.qk_row_elems * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.qk_row_elems,)),
                )
                gsrc, sdst = group_bulk_copy_modes(g_qk, s_qk)
                cute.copy(g2s_qk, gsrc, sdst, mbar_ptr=mbar_ptr)

                g_cos = cute.make_tensor(cos.iterator + s * self.D2, cute.make_layout((self.D2,)))
                s_cos = cute.make_tensor(
                    cute.make_ptr(
                        self.qk_dtype,
                        cos_smem_start + Int32(local_pos * self.D2 * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.D2,)),
                )
                gc_src, sc_dst = group_bulk_copy_modes(g_cos, s_cos)
                cute.copy(g2s_cs, gc_src, sc_dst, mbar_ptr=mbar_ptr)

                g_sin = cute.make_tensor(sin.iterator + s * self.D2, cute.make_layout((self.D2,)))
                s_sin = cute.make_tensor(
                    cute.make_ptr(
                        self.qk_dtype,
                        sin_smem_start + Int32(local_pos * self.D2 * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.D2,)),
                )
                gs_src, ss_dst = group_bulk_copy_modes(g_sin, s_sin)
                cute.copy(g2s_cs, gs_src, ss_dst, mbar_ptr=mbar_ptr)

    @cute.jit
    def store(self, page_ptr, tile_M, tile_H, qk, q_norm_weight, k_norm_weight, cos, sin,
              op_config_ptr):
        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))
        s2g = cute.make_copy_atom(
            CopyBulkS2GOp(),
            self.qk_dtype,
            num_bits_per_copy=self.qk_nbits_per_row,
        )
        pos_start = tile_M * self.tile_size_M
        head_start = tile_H * self.tile_size_H

        for local_pos in range(self.tile_size_M):
            pos = pos_start + local_pos
            if pos < runtime_M:
                s_tile = cute.make_tensor(
                    cute.make_ptr(
                        self.qk_dtype,
                        page_ptr + Int32(local_pos * self.qk_row_elems * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.qk_row_elems,)),
                )
                g_tile = cute.make_tensor(
                    qk.iterator + pos * Int32(self.qk_stride_M) + head_start * Int32(self.qk_stride_H),
                    cute.make_layout((self.qk_row_elems,)),
                )
                ssrc, gdst = group_bulk_copy_modes(s_tile, g_tile)
                cute.copy(s2g, ssrc, gdst)


class Qwen3_5ForwardPackedQKNormRope4DOp(Qwen3_5ForwardPackedQKNormRopeOp):
    """Qwen-local packed Q/K norm+RoPE over a BMHD packed-QK view.

    This keeps the same shared-memory compute body as the 3D packed op, but
    exposes B/M/H tile axes to the dependency scheduler. That lets attention
    consume normalized row blocks instead of waiting for the whole flattened
    QK tensor.
    """

    reads = {
        "qk": (None, ("B", "M", "H", "D")),
        "q_norm_weight": (None, ("D",)),
        "k_norm_weight": (None, ("D",)),
        "cos": (None, ("S", "D2")),
        "sin": (None, ("S", "D2")),
    }
    writes = {"qk": (None, ("B", "M", "H", "D"))}
    tile = ("B", "M", "H")

    @cute.jit
    def load(self, page_ptr, tile_B, tile_M, tile_H, qk, q_norm_weight, k_norm_weight, cos, sin,
             op_config_ptr, work_mbar):
        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))
        pos_start = tile_M * self.tile_size_M
        head_start = tile_H * self.tile_size_H

        qk_per_pos_bytes = Int32(self.qk_row_elems * self.elem_bytes)
        cs_per_pos_bytes = Int32(2 * self.D2 * self.elem_bytes)
        actual_rows = Int32(self.tile_size_M)
        remaining = runtime_M - pos_start
        if remaining < Int32(self.tile_size_M):
            actual_rows = remaining
        total_bytes = actual_rows * (qk_per_pos_bytes + cs_per_pos_bytes)

        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        mbarrier_arrive_expect_tx(work_mbar, total_bytes)

        g2s_qk = cute.make_copy_atom(
            CopyBulkG2SOp(),
            self.qk_dtype,
            num_bits_per_copy=self.qk_nbits_per_row,
        )
        g2s_cs = cute.make_copy_atom(
            CopyBulkG2SOp(),
            self.qk_dtype,
            num_bits_per_copy=self.cs_nbits_per_row,
        )

        cos_smem_start = page_ptr + Int32(self.qk_tile_bytes)
        sin_smem_start = page_ptr + Int32(self.qk_tile_bytes + self.cs_tile_bytes)

        for local_pos in range(self.tile_size_M):
            pos = pos_start + local_pos
            if pos < runtime_M:
                s = pos % self.S
                g_qk = cute.make_tensor(
                    qk.iterator
                    + tile_B * Int32(self.qk_stride_B)
                    + pos * Int32(self.qk_stride_M)
                    + head_start * Int32(self.qk_stride_H),
                    cute.make_layout((self.qk_row_elems,)),
                )
                s_qk = cute.make_tensor(
                    cute.make_ptr(
                        self.qk_dtype,
                        page_ptr + Int32(local_pos * self.qk_row_elems * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.qk_row_elems,)),
                )
                gsrc, sdst = group_bulk_copy_modes(g_qk, s_qk)
                cute.copy(g2s_qk, gsrc, sdst, mbar_ptr=mbar_ptr)

                g_cos = cute.make_tensor(cos.iterator + s * self.D2, cute.make_layout((self.D2,)))
                s_cos = cute.make_tensor(
                    cute.make_ptr(
                        self.qk_dtype,
                        cos_smem_start + Int32(local_pos * self.D2 * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.D2,)),
                )
                gc_src, sc_dst = group_bulk_copy_modes(g_cos, s_cos)
                cute.copy(g2s_cs, gc_src, sc_dst, mbar_ptr=mbar_ptr)

                g_sin = cute.make_tensor(sin.iterator + s * self.D2, cute.make_layout((self.D2,)))
                s_sin = cute.make_tensor(
                    cute.make_ptr(
                        self.qk_dtype,
                        sin_smem_start + Int32(local_pos * self.D2 * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.D2,)),
                )
                gs_src, ss_dst = group_bulk_copy_modes(g_sin, s_sin)
                cute.copy(g2s_cs, gs_src, ss_dst, mbar_ptr=mbar_ptr)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_M, tile_H, op_config_ptr):
        _BasePackedQKNormRopeOp.compute(self, page_ptr, tile_M, tile_H, op_config_ptr)

    @cute.jit
    def store(self, page_ptr, tile_B, tile_M, tile_H, qk, q_norm_weight, k_norm_weight, cos, sin,
              op_config_ptr):
        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))
        s2g = cute.make_copy_atom(
            CopyBulkS2GOp(),
            self.qk_dtype,
            num_bits_per_copy=self.qk_nbits_per_row,
        )
        pos_start = tile_M * self.tile_size_M
        head_start = tile_H * self.tile_size_H

        for local_pos in range(self.tile_size_M):
            pos = pos_start + local_pos
            if pos < runtime_M:
                s_tile = cute.make_tensor(
                    cute.make_ptr(
                        self.qk_dtype,
                        page_ptr + Int32(local_pos * self.qk_row_elems * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.qk_row_elems,)),
                )
                g_tile = cute.make_tensor(
                    qk.iterator
                    + tile_B * Int32(self.qk_stride_B)
                    + pos * Int32(self.qk_stride_M)
                    + head_start * Int32(self.qk_stride_H),
                    cute.make_layout((self.qk_row_elems,)),
                )
                ssrc, gdst = group_bulk_copy_modes(s_tile, g_tile)
                cute.copy(s2g, ssrc, gdst)

    @classmethod
    def schedule(
        cls,
        tile_sizes=None,
        page_size=DEFAULT_PAGE_SIZE,
        eps=1e-6,
        num_q_heads=None,
        num_k_heads=None,
        **tensors,
    ):
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("M", 8)
        tile_sizes.setdefault("H", 5)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        ops[0].static_dims["eps"] = eps
        if num_q_heads is not None:
            ops[0].static_dims["num_q_heads"] = int(num_q_heads)
        if num_k_heads is not None:
            ops[0].static_dims["num_k_heads"] = int(num_k_heads)
        ops[0].static_dims["barrier_wait_qk_alias_M"] = "S"
        return ops


class Qwen3_5ForwardQKNormRopeOp(_BaseQKNormRopeOp):
    """Qwen forward single Q/K norm+RoPE boundary op."""


class Qwen3_5ForwardAttentionOp(_BaseFlashAttentionSm120Op):
    """Qwen forward full attention op."""

    # Q is the only DMA-warp TMA load. K/V are loaded by the attention MMA
    # warps through cp.async inside compute, so their producer dependencies can
    # be delayed until compute instead of blocking Q prefetch in the controller.
    compute_wait_inputs = {"k", "v"}


class Qwen3_5ForwardTmaAttentionOp(Qwen3_5ForwardAttentionOp):
    """Qwen forward attention with compute-issued TMA K/V loads.

    This variant keeps Q as the framework load-phase TMA, then preloads Q into
    registers and reuses the page for compact K/V buffers driven by TMA from
    compute warp 0. It targets the Qwen forward q-preload shape used at S=512.
    """

    tma_compute_loads = {"k", "v"}

    @classmethod
    def schedule(cls, tile_sizes=None, **kwargs):
        ops = super().schedule(tile_sizes=tile_sizes, **kwargs)
        for op in ops:
            # Keep the first version conservative: one compact 16-row K buffer
            # and one compact 16-row V buffer fit with op-local mbarriers in a
            # 32KB page after Q has been moved to registers.
            op.static_dims["tma_n_block"] = 16
        return ops

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name in ("k", "v"):
            return (1, static_dims["tma_n_block"], 1, static_dims["D"])
        return None

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        if tensor_name in ("k", "v"):
            # tma_tile_shape is stride-sorted to (D, H_kv=1, N, B=1).
            tile_d, one_h, tile_n, one_b = tma_tile_shape
            return (
                f"cute.make_layout(({tile_d}, {one_h}, {tile_n}, {one_b}), "
                f"stride=(1, {tile_d * tile_n}, {tile_d}, {tile_d * tile_n}))"
            )
        return super().get_tma_smem_layout_src(tensor_name, tma_tile_shape, tile_sizes, static_dims)

    def __init__(self, **config):
        super().__init__(**config)
        if self.q_in_smem:
            raise ValueError(f"{type(self).__name__} currently supports q-preload mode only")
        self.n_block = int(getattr(self, "tma_n_block", 16))
        self.smem_stride = self.D
        self.kv_tile_bytes = self.n_block * self.smem_stride * self.elem_bytes
        self.tma_k_base = 0
        self.tma_v_base = self.kv_tile_bytes
        self.tma_mbar_offset = 2 * self.kv_tile_bytes
        assert self.tma_mbar_offset + 16 <= self.page_size, (
            f"{type(self).__name__}: compact K/V buffers plus mbarriers exceed page_size"
        )
        self._bind_phase("compute", "compute_mma_tma")

    @cute.jit
    def compute_mma_tma(self, page_ptr, tile_B, tile_M, tile_H, tile_D,
                        q, k, v, o, lse,
                        k_tma, k_tma_gmem, v_tma, v_tma_gmem,
                        op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))
        runtime_N = config_dim_i32(op_config_ptr, "N", type(self))
        runtime_num_kv_blocks = (runtime_N + Int32(self.n_block - 1)) // Int32(self.n_block)

        if warp_idx < Int32(self.num_mma_warps):
            mma_op = cute.nvgpu.warp.MmaF16BF16Op(self.q_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((self.num_mma_warps, 1, 1)),
                permutation_mnk=(self.num_mma_warps * 16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)

            swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)
            sQ = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                    swz,
                    dtype=self.q_dtype,
                ),
                cute.make_layout((self.tile_size_M, self.D), stride=(self.D, 1)),
            )
            _tCsQ = thr_mma.partition_A(sQ)
            tCrQ = tiled_mma.make_fragment_A(_tCsQ)
            smem_copy_atom_Q = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.q_dtype
            )
            smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
            smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
            tQrQ_view = smem_thr_copy_Q.retile(tCrQ)
            tQsQ = smem_thr_copy_Q.partition_S(sQ)
            self._compute_load_q(page_ptr, tile_B, tile_H, tile_M, tile_D, q, runtime_M)

            for _qkb in cutlass.range_constexpr(self.D // 16):
                cute.copy(smem_tiled_copy_Q, tQsQ[None, None, _qkb], tQrQ_view[None, None, _qkb])
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            k_base = page_ptr + Int32(self.tma_k_base)
            v_base = page_ptr + Int32(self.tma_v_base)
            k_ready = page_ptr + Int32(self.tma_mbar_offset)
            v_ready = page_ptr + Int32(self.tma_mbar_offset + 8)
            k_ready_ptr = cute.make_ptr(cutlass.Int64, k_ready, cute.AddressSpace.smem)
            v_ready_ptr = cute.make_ptr(cutlass.Int64, v_ready, cute.AddressSpace.smem)
            kv_bytes = Int32(self.n_block * self.D * self.elem_bytes)

            if warp_idx == Int32(0):
                with cute.arch.elect_one():
                    mbarrier_init(k_ready, Int32(1))
                    mbarrier_init(v_ready, Int32(1))
                mbarrier_init_fence_async_proxy()
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            sK_tma = cute.make_tensor(
                cute.make_ptr(self.q_dtype, k_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.D, 1, self.n_block, 1),
                                 stride=(1, self.D * self.n_block, self.D, self.D * self.n_block)),
            )
            gK_tma = cute.local_tile(k_tma_gmem, (self.D, 1, self.n_block, 1), (None, None, None, None))
            tKsK_tma, tKgK_tma = cute.nvgpu.cpasync.tma_partition(
                k_tma,
                Int32(0),
                cute.make_layout(1),
                cute.group_modes(sK_tma, 0, 4),
                cute.group_modes(gK_tma, 0, 4),
            )
            sV_tma = cute.make_tensor(
                cute.make_ptr(self.q_dtype, v_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.D, 1, self.n_block, 1),
                                 stride=(1, self.D * self.n_block, self.D, self.D * self.n_block)),
            )
            gV_tma = cute.local_tile(v_tma_gmem, (self.D, 1, self.n_block, 1), (None, None, None, None))
            tVsV_tma, tVgV_tma = cute.nvgpu.cpasync.tma_partition(
                v_tma,
                Int32(0),
                cute.make_layout(1),
                cute.group_modes(sV_tma, 0, 4),
                cute.group_modes(gV_tma, 0, 4),
            )

            smem_copy_atom_K = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.q_dtype
            )
            smem_copy_atom_Vt = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.q_dtype
            )
            smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
            smem_tiled_copy_Vt = cute.make_tiled_copy_B(smem_copy_atom_Vt, tiled_mma)
            smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
            smem_thr_copy_Vt = smem_tiled_copy_Vt.get_slice(tidx)

            sK = cute.make_tensor(
                cute.make_ptr(self.q_dtype, k_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.n_block, self.D), stride=(self.smem_stride, 1)),
            )
            _tCsK = thr_mma.partition_B(sK)
            tCrK = tiled_mma.make_fragment_B(_tCsK)
            tKrK_view = smem_thr_copy_K.retile(tCrK)
            tKsK = smem_thr_copy_K.partition_S(sK)

            sVt = cute.make_tensor(
                cute.make_ptr(self.q_dtype, v_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.D, self.n_block), stride=(1, self.smem_stride)),
            )
            _tBsVt = thr_mma.partition_B(sVt)
            tBrVt = tiled_mma.make_fragment_B(_tBsVt)
            tVrVt_view = smem_thr_copy_Vt.retile(tBrVt)
            tVsVt = smem_thr_copy_Vt.partition_S(sVt)

            smem_copy_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.q_dtype)
            smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
            smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)

            acc_S = cute.make_fragment(tiled_mma.partition_shape_C((self.tile_size_M, self.n_block)), Float32)
            rP = cute.make_fragment_like(acc_S, self.q_dtype)
            rP_ld = cute.logical_divide(rP.layout, (None, None, 2))
            rP_mma_view = cute.make_layout(
                ((rP_ld.shape[0], rP_ld.shape[2][0]), rP_ld.shape[1], rP_ld.shape[2][1]),
                stride=((rP_ld.stride[0], rP_ld.stride[2][0]), rP_ld.stride[1], rP_ld.stride[2][1]),
            )
            tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

            acc_O = cute.make_fragment(tiled_mma.partition_shape_C((self.tile_size_M, self.D)), Float32)
            acc_O.fill(0.0)
            acc_O_shape = tiled_mma.partition_shape_C((self.tile_size_M, self.D))
            num_rows = acc_O_shape[0][1] * acc_O_shape[1]
            row_max = cute.make_fragment(cute.make_layout(num_rows), Float32)
            row_sum = cute.make_fragment(cute.make_layout(num_rows), Float32)
            for r in cutlass.range_constexpr(num_rows):
                row_max[r] = Float32(-1e20)
                row_sum[r] = Float32(0.0)

            mcS = cute.make_identity_tensor((self.tile_size_M, self.n_block))
            tScS = thr_mma.partition_C(mcS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)

            num_kv_blocks_eff = runtime_num_kv_blocks
            first_packed_row = tile_M * Int32(self.tile_size_M)
            first_row = self._seq_row_index(first_packed_row)
            causal_first_limit = first_row + (runtime_N - runtime_M)
            if cutlass.const_expr(self.causal):
                last_packed_row = tile_M * Int32(self.tile_size_M) + Int32(self.tile_size_M - 1)
                last_row = self._seq_row_index(last_packed_row)
                max_col = last_row + (runtime_N - runtime_M)
                num_kv_blocks_eff = (max_col + Int32(self.n_block)) // Int32(self.n_block)
                if num_kv_blocks_eff > runtime_num_kv_blocks:
                    num_kv_blocks_eff = runtime_num_kv_blocks

            kv_h = self._kv_head_index(tile_H)
            k_phase = Int32(0)
            v_phase = Int32(0)
            kv_idx = Int32(0)
            if warp_idx == Int32(0):
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(k_ready, kv_bytes)
                cute.copy(
                    k_tma,
                    tKgK_tma[(None, Int32(0), kv_h, kv_idx, tile_B)],
                    tKsK_tma,
                    tma_bar_ptr=k_ready_ptr,
                )

            while kv_idx < num_kv_blocks_eff:
                kv_start = kv_idx * Int32(self.n_block)

                mbarrier_wait(k_ready, k_phase)
                k_phase = k_phase ^ Int32(1)

                if warp_idx == Int32(0):
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(v_ready, kv_bytes)
                    cute.copy(
                        v_tma,
                        tVgV_tma[(None, Int32(0), kv_h, kv_idx, tile_B)],
                        tVsV_tma,
                        tma_bar_ptr=v_ready_ptr,
                    )

                acc_S.fill(0.0)
                cute.copy(smem_tiled_copy_K, tKsK[None, None, 0], tKrK_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.D // 16 - 1):
                    kb_next = kb + 1
                    cute.copy(smem_tiled_copy_K, tKsK[None, None, kb_next], tKrK_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_S, tCrQ[None, None, kb], tCrK[None, None, kb], acc_S)
                cute.gemm(
                    tiled_mma,
                    acc_S,
                    tCrQ[None, None, self.D // 16 - 1],
                    tCrK[None, None, self.D // 16 - 1],
                    acc_S,
                )

                if kv_idx + Int32(1) < num_kv_blocks_eff:
                    if warp_idx == Int32(0):
                        with cute.arch.elect_one():
                            mbarrier_arrive_expect_tx(k_ready, kv_bytes)
                        cute.copy(
                            k_tma,
                            tKgK_tma[(None, Int32(0), kv_h, kv_idx + Int32(1), tile_B)],
                            tKsK_tma,
                            tma_bar_ptr=k_ready_ptr,
                        )

                mbarrier_wait(v_ready, v_phase)
                v_phase = v_phase ^ Int32(1)

                acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
                acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

                if kv_start + Int32(self.n_block) > runtime_N:
                    for r in cutlass.range_constexpr(num_rows):
                        for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                            col_idx = tScS_mn[0, c][1]
                            global_col = kv_start + Int32(col_idx)
                            if global_col >= runtime_N:
                                acc_S_mn[r, c] = Float32(-1e20)

                if cutlass.const_expr(self.causal):
                    last_blk_col = kv_start + Int32(self.n_block - 1)
                    if last_blk_col > causal_first_limit:
                        for r in cutlass.range_constexpr(num_rows):
                            row_idx = tScS_mn[r, 0][0]
                            packed_row = first_packed_row + Int32(row_idx)
                            global_row = self._seq_row_index(packed_row)
                            causal_row_limit = global_row + (runtime_N - runtime_M)
                            for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                                col_idx = tScS_mn[0, c][1]
                                global_col = kv_start + Int32(col_idx)
                                if global_col > causal_row_limit:
                                    acc_S_mn[r, c] = Float32(-1e20)

                corrections = cute.make_fragment(cute.make_layout(num_rows), Float32)
                for r in cutlass.range_constexpr(num_rows):
                    acc_S_row = acc_S_mn[r, None].load()
                    row_max_cur = acc_S_row.reduce(cute.ReductionOp.MAX, Float32(-1e20), 0)
                    row_max_cur = self._threadquad_reduce_max(row_max_cur)
                    m_old = row_max[r]
                    m_new = cute.arch.fmax(m_old, row_max_cur)
                    acc_scale_ = (m_old - m_new) * Float32(self.scale_log2e)
                    correction = cute.math.exp2(cute.arch.fmax(acc_scale_, Float32(-126.0)), fastmath=True)
                    row_sum[r] = row_sum[r] * correction
                    corrections[r] = correction
                    acc_S_row_exp = cute.math.exp2(
                        acc_S_row * Float32(self.scale_log2e) - m_new * Float32(self.scale_log2e),
                        fastmath=True,
                    )
                    acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                    row_sum[r] = row_sum[r] + acc_S_row_sum
                    row_max[r] = m_new
                    acc_S_mn[r, None] = acc_S_row_exp

                for r in cutlass.range_constexpr(num_rows):
                    acc_O_mn[r, None] = acc_O_mn[r, None].load() * corrections[r]

                rP.store(acc_S.load().to(self.q_dtype))
                cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, 0], tVrVt_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.n_block // 16 - 1):
                    kb_next = kb + 1
                    cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, kb_next], tVrVt_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_O, tOrS[None, None, kb], tBrVt[None, None, kb], acc_O)
                cute.gemm(
                    tiled_mma,
                    acc_O,
                    tOrS[None, None, self.n_block // 16 - 1],
                    tBrVt[None, None, self.n_block // 16 - 1],
                    acc_O,
                )
                kv_idx = kv_idx + Int32(1)

            acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
            for r in cutlass.range_constexpr(num_rows):
                row_sum[r] = self._threadquad_reduce_sum(row_sum[r])

            if cutlass.const_expr(self.write_lse):
                lane_in_quad = tidx % Int32(4)
                for r in cutlass.range_constexpr(num_rows):
                    if lane_in_quad == Int32(0):
                        row_idx = tScS_mn[r, 0][0]
                        packed_row = tile_M * Int32(self.tile_size_M) + Int32(row_idx)
                        global_row = self._seq_row_index(packed_row)
                        if global_row < runtime_M:
                            q_head = self._q_head_index(tile_H, packed_row)
                            lse_head_ptr = lse.iterator + tile_B * runtime_M * Int32(self.H) + q_head
                            g_lse = cute.make_tensor(lse_head_ptr, cute.make_layout(runtime_M, stride=Int32(self.H)))
                            lse_val = row_max[r] * Float32(self.scale_val) + cute.math.log(row_sum[r])
                            g_lse[global_row] = lse_val

            for r in cutlass.range_constexpr(num_rows):
                inv_sum = cute.arch.rcp_approx(row_sum[r])
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * inv_sum

            tCrO_q = cute.make_fragment_like(acc_O, self.q_dtype)
            for i in cutlass.range_constexpr(cute.size(acc_O)):
                tCrO_q[i] = acc_O[i].to(self.q_dtype)
            o_swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)
            sO = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem),
                    o_swz,
                    dtype=self.q_dtype,
                ),
                cute.make_layout((self.tile_size_M, self.D), stride=(self.D, 1)),
            )
            tOrO = smem_thr_copy_O.retile(tCrO_q)
            tOsO = smem_thr_copy_O.partition_D(sO)
            cute.copy(smem_tiled_copy_O, tOrO, tOsO)
            self._compute_store_o(page_ptr, tile_B, tile_H, tile_M, tile_D, o, runtime_M)


class Qwen3_5ForwardTmaReduceStoreAttentionOp(Qwen3_5ForwardAttentionOp):
    """Qwen attention primitive with TMA ADD-reduce output store.

    This is the store-side primitive needed by a no-inner-KV split attention
    design.  The mathematically exact split path cannot ADD-reduce normalized
    per-split outputs directly: softmax first needs a global per-row max/LSE.
    The intended exact pipeline is:

    1. compute per-KV-block row max stats,
    2. reduce those small stats to a global row max/LSE,
    3. recompute each KV block and TMA ADD-reduce unnormalized O and L,
    4. normalize once.

    This op provides step 3's large O reduce-store mechanism while preserving
    the proven Qwen attention compute body.  Any schedule that uses it must
    zero/initialize the destination accumulator before issuing reduce stores.
    """

    tma_stores = set()
    tma_reduce_stores = {"o"}

    @cute.jit
    def store(self, page_ptr, tile_B, tile_M, tile_H, tile_D, o_tma, o_tma_gmem):
        """Warp-collective TMA ADD-reduce of O from shared to global."""
        o_swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)
        sO = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem),
                o_swz,
                dtype=self.q_dtype,
            ),
            cute.make_layout(self._o_tma_smem_shape),
        )
        gO = cute.local_tile(
            o_tma_gmem,
            self._o_tma_smem_shape,
            (None, None, None, None),
        )
        tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
            o_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sO, 0, 4),
            cute.group_modes(gO, 0, 4),
        )
        # TMA reduce is warp-collective. Do not wrap this in elect_one().
        cute.copy(o_tma, tOsO, tOgO[(None, tile_D, tile_H, tile_M, tile_B)])


class Qwen3_5ForwardNoInnerKVTmaReduceStoreAttentionOp(Qwen3_5ForwardTmaReduceStoreAttentionOp):
    """Single-KV-window reduce-store attention building block.

    This variant is deliberately constrained to K/V windows that fit in one
    in-page KV block, so the attention compute body has no long inner KV loop.
    A full exact split-KV schedule should invoke this on K/V windows after a
    global row-stat pass and reduce-store unnormalized contributions into a
    zeroed fp accumulator.
    """

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        q = tensors.get("q")
        k = tensors.get("k")
        if q is not None and k is not None:
            D = int(q.shape[-1])
            elem_bytes = int(q.element_size())
            # Mirror the Q-preload attention KV capacity. With D=256 and a
            # 32KB page this is 32 rows after dropping smem padding.
            max_n_block = page_size // (2 * D * elem_bytes)
            max_n_block = min(32, max_n_block)
            if int(k.shape[1]) > max_n_block:
                raise ValueError(
                    f"{cls.__name__} requires a sliced K/V window no larger "
                    f"than one KV block ({max_n_block} rows for D={D}, "
                    f"page_size={page_size}); got N={int(k.shape[1])}."
                )
        ops = super().schedule(tile_sizes=tile_sizes, page_size=page_size, **tensors)
        for op in ops:
            op.static_dims["no_inner_kv"] = 1
        return ops


class Qwen3_5ForwardAttentionNormalizeOp(Op):
    """Normalize split-KV attention accumulators.

    Inputs are the exact split/recompute form:

    - ``o_accum``: fp32 sum of ``exp(score - global_m) * V``.
    - ``l_accum``: fp32 sum of ``exp(score - global_m)``.
    - ``m_global``: fp32 raw per-row max before attention scaling.

    The op writes final ``o`` and optionally materializes forward LSE for bwd:
    ``lse = m_global * scale + log(l_accum)``.
    """

    reads = {
        "o_accum": (cutlass.Float32, ("B", "M", "H", "D")),
        "l_accum": (cutlass.Float32, ("B", "M", "H")),
        "m_global": (cutlass.Float32, ("B", "M", "H")),
        "accum_done": (cutlass.Int32, ("B", "M", "H", "SPLIT")),
    }
    writes = {
        "o": (None, ("B", "M", "H", "D")),
        "lse": (cutlass.Float32, ("B", "M", "H")),
    }
    tile = ("B", "M", "H", "D")
    dynamic_dims = ("B", "M")
    inline_phases = ("compute",)
    tma_loads = set()
    tma_stores = set()
    uses_smem_page = False

    def __init__(self, **config):
        super().__init__(**config)
        self.scale_val = 1.0 / (self.D ** 0.5)
        self.write_lse = getattr(self, "write_lse", 0)

    @classmethod
    def schedule(cls, tile_sizes=None, write_lse=False, **tensors):
        o = tensors.get("o")
        if o is None:
            raise ValueError(f"{cls.__name__} requires output tensor o")
        if "lse" not in tensors or tensors["lse"] is None:
            B, M, H, _D = o.shape
            tensors["lse"] = torch.empty(B, M, H, dtype=torch.float32, device=o.device)
        if "accum_done" not in tensors or tensors["accum_done"] is None:
            B, M, H, _D = o.shape
            tensors["accum_done"] = torch.empty(B, M, H, 1, dtype=torch.int32, device=o.device)
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("M", min(64, int(o.shape[1])))
        tile_sizes.setdefault("H", 1)
        tile_sizes.setdefault("D", int(o.shape[3]))
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["write_lse"] = int(bool(write_lse))
        ops[0].static_dims["barrier_signal_o_alias_M"] = "S"
        return ops

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_M, tile_H, tile_D,
                o_accum, l_accum, m_global, accum_done, o, lse, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))

        row_base = tile_M * Int32(self.tile_size_M)
        d_base = tile_D * Int32(self.tile_size_D)

        local_m = warp_idx
        while local_m < Int32(self.tile_size_M):
            row = row_base + local_m
            if row < runtime_M:
                denom = l_accum[tile_B, row, tile_H]
                inv_denom = cute.arch.rcp_approx(denom)
                if cutlass.const_expr(self.write_lse):
                    if lane_idx == Int32(0) and tile_D == Int32(0):
                        lse[tile_B, row, tile_H] = (
                            m_global[tile_B, row, tile_H] * Float32(self.scale_val)
                            + cute.math.log(denom)
                        )
                local_d = lane_idx
                while local_d < Int32(self.tile_size_D):
                    d = d_base + local_d
                    if d < Int32(self.D):
                        val = o_accum[tile_B, row, tile_H, d] * inv_denom
                        o[tile_B, row, tile_H, d] = val.to(self.o_dtype)
                    local_d = local_d + Int32(32)
            local_m = local_m + Int32(num_warps)


class Qwen3_5ForwardSplitKVMaxStatsOp(Op):
    """Correctness-first split-KV max stats for experimental attention."""

    reads = {
        "q": (None, ("B", "M", "H", "D")),
        "k": (None, ("B", "N", "H_kv", "D")),
        "_split": (cutlass.Int32, ("SPLIT",)),
    }
    writes = {"m_part": (cutlass.Float32, ("B", "M", "H", "SPLIT"))}
    tile = ("B", "M", "H", "SPLIT")
    dynamic_dims = ("B", "M", "N")
    inline_phases = ("compute",)
    tma_loads = set()
    tma_stores = set()
    uses_smem_page = False

    def __init__(self, **config):
        super().__init__(**config)
        self.scale_val = 1.0 / (self.D ** 0.5)
        self.kv_group_size = getattr(self, "kv_group_size", KV_GROUP_SIZE)
        self.causal = getattr(self, "causal", 1)

    @classmethod
    def schedule(cls, tile_sizes=None, causal=True, kv_group_size=KV_GROUP_SIZE, num_splits=16, **tensors):
        q = tensors["q"]
        B, M, H, _D = q.shape
        if "_split" not in tensors:
            tensors["_split"] = torch.empty(num_splits, dtype=torch.int32, device=q.device)
        if "m_part" not in tensors:
            tensors["m_part"] = torch.empty(B, M, H, num_splits, dtype=torch.float32, device=q.device)
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("M", min(64, M))
        tile_sizes.setdefault("H", 1)
        tile_sizes.setdefault("SPLIT", 1)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["SPLIT"] = int(num_splits)
        ops[0].static_dims["kv_group_size"] = int(kv_group_size)
        ops[0].static_dims["causal"] = int(bool(causal))
        return ops

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_M, tile_H, tile_SPLIT, q, k, _split, m_part, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))
        runtime_N = config_dim_i32(op_config_ptr, "N", type(self))
        row_base = tile_M * Int32(self.tile_size_M)
        kv_h = tile_H // Int32(self.kv_group_size)
        split_start = (runtime_N * tile_SPLIT) // Int32(self.SPLIT)
        split_end = (runtime_N * (tile_SPLIT + Int32(1))) // Int32(self.SPLIT)

        local_m = warp_idx
        while local_m < Int32(self.tile_size_M):
            row = row_base + local_m
            if row < runtime_M:
                row_max = Float32(-1.0e20)
                n = split_start
                while n < split_end:
                    valid = True
                    if cutlass.const_expr(self.causal):
                        if n > row + (runtime_N - runtime_M):
                            valid = False
                    if valid:
                        acc = Float32(0.0)
                        d = lane_idx
                        while d < Int32(self.D):
                            acc = acc + q[tile_B, row, tile_H, d].to(Float32) * k[tile_B, n, kv_h, d].to(Float32)
                            d = d + Int32(32)
                        acc = cute.arch.warp_reduction(acc, operator.add)
                        if lane_idx == Int32(0):
                            row_max = cute.arch.fmax(row_max, acc)
                    n = n + Int32(1)
                if lane_idx == Int32(0):
                    m_part[tile_B, row, tile_H, tile_SPLIT] = row_max
            local_m = local_m + Int32(num_warps)


class Qwen3_5ForwardSplitKVMaxReduceOp(Op):
    """Reduce per-split raw max stats to global raw max."""

    reads = {"m_part": (cutlass.Float32, ("B", "M", "H", "SPLIT"))}
    writes = {"m_global": (cutlass.Float32, ("B", "M", "H"))}
    tile = ("B", "M", "H")
    dynamic_dims = ("B", "M")
    inline_phases = ("compute",)
    tma_loads = set()
    tma_stores = set()
    uses_smem_page = False

    @classmethod
    def schedule(cls, tile_sizes=None, **tensors):
        m_part = tensors["m_part"]
        B, M, H, S = m_part.shape
        if "m_global" not in tensors:
            tensors["m_global"] = torch.empty(B, M, H, dtype=torch.float32, device=m_part.device)
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("M", min(64, M))
        tile_sizes.setdefault("H", 1)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["SPLIT"] = int(S)
        return ops

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_M, tile_H, m_part, m_global, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))
        row_base = tile_M * Int32(self.tile_size_M)
        row = row_base + tidx
        while row < row_base + Int32(self.tile_size_M):
            if row < runtime_M:
                mx = Float32(-1.0e20)
                s = Int32(0)
                while s < Int32(self.SPLIT):
                    mx = cute.arch.fmax(mx, m_part[tile_B, row, tile_H, s])
                    s = s + Int32(1)
                m_global[tile_B, row, tile_H] = mx
            row = row + Int32(self.threads_per_row)


class Qwen3_5ForwardSplitKVAccumAtomicOp(Op):
    """Correctness-first split-KV accumulator using compute-side atomics."""

    reads = {
        "q": (None, ("B", "M", "H", "D")),
        "k": (None, ("B", "N", "H_kv", "D")),
        "v": (None, ("B", "N", "H_kv", "D")),
        "m_global": (cutlass.Float32, ("B", "M", "H")),
        "_split": (cutlass.Int32, ("SPLIT",)),
    }
    writes = {
        "o_accum": (cutlass.Float32, ("B", "M", "H", "D")),
        "l_accum": (cutlass.Float32, ("B", "M", "H")),
        "accum_done": (cutlass.Int32, ("B", "M", "H", "SPLIT")),
    }
    tile = ("B", "M", "H", "SPLIT")
    dynamic_dims = ("B", "M", "N")
    inline_phases = ("compute",)
    tma_loads = set()
    tma_stores = set()
    compute_reduce_stores = {"o_accum", "l_accum"}
    uses_smem_page = False

    def __init__(self, **config):
        super().__init__(**config)
        self.scale_val = 1.0 / (self.D ** 0.5)
        self.kv_group_size = getattr(self, "kv_group_size", KV_GROUP_SIZE)
        self.causal = getattr(self, "causal", 1)

    @classmethod
    def schedule(cls, tile_sizes=None, causal=True, kv_group_size=KV_GROUP_SIZE, num_splits=16, **tensors):
        q = tensors["q"]
        B, M, H, D = q.shape
        if "_split" not in tensors:
            tensors["_split"] = torch.empty(num_splits, dtype=torch.int32, device=q.device)
        if "o_accum" not in tensors:
            tensors["o_accum"] = torch.zeros(B, M, H, D, dtype=torch.float32, device=q.device)
        if "l_accum" not in tensors:
            tensors["l_accum"] = torch.zeros(B, M, H, dtype=torch.float32, device=q.device)
        if "accum_done" not in tensors:
            tensors["accum_done"] = torch.empty(B, M, H, num_splits, dtype=torch.int32, device=q.device)
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("M", min(64, M))
        tile_sizes.setdefault("H", 1)
        tile_sizes.setdefault("SPLIT", 1)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["SPLIT"] = int(num_splits)
        ops[0].static_dims["kv_group_size"] = int(kv_group_size)
        ops[0].static_dims["causal"] = int(bool(causal))
        return ops

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_M, tile_H, tile_SPLIT,
                q, k, v, m_global, _split, o_accum, l_accum, accum_done, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))
        runtime_N = config_dim_i32(op_config_ptr, "N", type(self))
        row_base = tile_M * Int32(self.tile_size_M)
        kv_h = tile_H // Int32(self.kv_group_size)
        split_start = (runtime_N * tile_SPLIT) // Int32(self.SPLIT)
        split_end = (runtime_N * (tile_SPLIT + Int32(1))) // Int32(self.SPLIT)

        local_m = warp_idx
        while local_m < Int32(self.tile_size_M):
            row = row_base + local_m
            if row < runtime_M:
                mval = m_global[tile_B, row, tile_H]
                local_l = Float32(0.0)
                n_l = split_start
                while n_l < split_end:
                    valid_l = True
                    if cutlass.const_expr(self.causal):
                        if n_l > row + (runtime_N - runtime_M):
                            valid_l = False
                    if valid_l:
                        score_l = Float32(0.0)
                        rd_l = lane_idx
                        while rd_l < Int32(self.D):
                            score_l = score_l + q[tile_B, row, tile_H, rd_l].to(Float32) * k[tile_B, n_l, kv_h, rd_l].to(Float32)
                            rd_l = rd_l + Int32(32)
                        score_l = cute.arch.warp_reduction(score_l, operator.add)
                        if lane_idx == Int32(0):
                            local_l = local_l + cute.math.exp((score_l - mval) * Float32(self.scale_val), fastmath=True)
                    n_l = n_l + Int32(1)
                if lane_idx == Int32(0):
                    _atomic_add_f32(local_l, l_accum.iterator + ((tile_B * runtime_M + row) * Int32(self.H) + tile_H))

                d = lane_idx
                while d < Int32(self.D):
                    local_o = Float32(0.0)
                    n = split_start
                    while n < split_end:
                        valid = True
                        if cutlass.const_expr(self.causal):
                            if n > row + (runtime_N - runtime_M):
                                valid = False
                        if valid:
                            score = Float32(0.0)
                            rd = lane_idx
                            while rd < Int32(self.D):
                                score = score + q[tile_B, row, tile_H, rd].to(Float32) * k[tile_B, n, kv_h, rd].to(Float32)
                                rd = rd + Int32(32)
                            score = cute.arch.warp_reduction(score, operator.add)
                            p = cute.math.exp((score - mval) * Float32(self.scale_val), fastmath=True)
                            if d < Int32(self.D):
                                local_o = local_o + p * v[tile_B, n, kv_h, d].to(Float32)
                        n = n + Int32(1)
                    if d < Int32(self.D):
                        _atomic_add_f32(local_o, o_accum.iterator + (((tile_B * runtime_M + row) * Int32(self.H) + tile_H) * Int32(self.D) + d))
                    d = d + Int32(32)
                if lane_idx == Int32(0):
                    accum_done[tile_B, row, tile_H, tile_SPLIT] = Int32(1)
            local_m = local_m + Int32(num_warps)
        global_memory_fence_gpu()


class Qwen3_5ForwardSplitKVAccumTmaReduceOp(Qwen3_5ForwardSplitKVAccumAtomicOp):
    """Split-KV accumulator with TMA ADD-reduce for the large O accumulator.

    ``l_accum`` remains a compute-side scalar atomic because it is one float per
    row/head/split.  The expensive ``M x D`` contribution is staged in the page
    as fp32 and emitted by store phase through TMA reduce.
    """

    inline_phases = ("compute", "store")
    tma_reduce_stores = {"o_accum"}
    compute_reduce_stores = {"l_accum"}
    uses_smem_page = True

    def __init__(self, **config):
        super().__init__(**config)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)
        self.o_accum_elem_bytes = 4
        self.o_accum_tile_bytes = self.tile_size_M * self.D * self.o_accum_elem_bytes
        assert self.o_accum_tile_bytes <= self.page_size, (
            f"{type(self).__name__}: fp32 O accumulator tile ({self.o_accum_tile_bytes}B) "
            f"exceeds page_size ({self.page_size}B); reduce tile M or increase page size."
        )
        if self.D >= 64:
            self.o_accum_swizzle_B = 3
        elif self.D >= 32:
            self.o_accum_swizzle_B = 2
        else:
            self.o_accum_swizzle_B = 1
        self._o_accum_tma_smem_shape = (self.D, 1, self.tile_size_M, 1)

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name == "o_accum":
            return (1, tile_sizes["M"], 1, static_dims["D"])
        return None

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        if tensor_name != "o_accum":
            return None
        D = static_dims["D"]
        if D >= 64:
            B = 3
        elif D >= 32:
            B = 2
        else:
            B = 1
        dims = tma_tile_shape
        strides = [1]
        for i in range(len(dims) - 1):
            strides.append(strides[-1] * dims[i])
        shape_str = ", ".join(str(d) for d in dims)
        stride_str = ", ".join(str(s) for s in strides)
        return (
            f"cute.make_composed_layout("
            f"cute.make_swizzle({B}, 4, 3), 0, "
            f"cute.make_layout(({shape_str}), "
            f"stride=({stride_str})))"
        )

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **kwargs):
        q = kwargs["q"]
        _B, M, _H, D = q.shape
        tile_sizes = dict(tile_sizes or {})
        max_tile_m = max(1, int(page_size) // (int(D) * 4))
        tile_sizes.setdefault("M", min(32, M, max_tile_m))
        tile_sizes["M"] = min(int(tile_sizes["M"]), max_tile_m)
        ops = super().schedule(tile_sizes=tile_sizes, **kwargs)
        for op in ops:
            op.static_dims["page_size"] = int(page_size)
        return ops

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_M, tile_H, tile_SPLIT,
                q, k, v, m_global, _split, o_accum, l_accum, accum_done, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))
        runtime_N = config_dim_i32(op_config_ptr, "N", type(self))
        row_base = tile_M * Int32(self.tile_size_M)
        kv_h = tile_H // Int32(self.kv_group_size)
        split_start = (runtime_N * tile_SPLIT) // Int32(self.SPLIT)
        split_end = (runtime_N * (tile_SPLIT + Int32(1))) // Int32(self.SPLIT)

        o_swz = cute.make_swizzle(self.o_accum_swizzle_B, 4, 3)
        sO = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(Float32, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                o_swz,
                dtype=Float32,
            ),
            cute.make_layout(self._o_accum_tma_smem_shape),
        )

        # Clear the whole staged O tile before per-row accumulation.  Invalid
        # M rows must also be zero because the TMA box is fixed-size.
        clear_idx = tidx
        while clear_idx < Int32(self.tile_size_M * self.D):
            clear_m = clear_idx // Int32(self.D)
            clear_d = clear_idx - clear_m * Int32(self.D)
            sO[clear_d, Int32(0), clear_m, Int32(0)] = Float32(0.0)
            clear_idx = clear_idx + Int32(self.threads_per_row)
        named_barrier_sync(Int32(1), Int32(self.threads_per_row))

        local_m = warp_idx
        while local_m < Int32(self.tile_size_M):
            row = row_base + local_m
            if row < runtime_M:
                mval = m_global[tile_B, row, tile_H]
                local_l = Float32(0.0)
                n_l = split_start
                while n_l < split_end:
                    valid_l = True
                    if cutlass.const_expr(self.causal):
                        if n_l > row + (runtime_N - runtime_M):
                            valid_l = False
                    if valid_l:
                        score_l = Float32(0.0)
                        rd_l = lane_idx
                        while rd_l < Int32(self.D):
                            score_l = score_l + q[tile_B, row, tile_H, rd_l].to(Float32) * k[tile_B, n_l, kv_h, rd_l].to(Float32)
                            rd_l = rd_l + Int32(32)
                        score_l = cute.arch.warp_reduction(score_l, operator.add)
                        if lane_idx == Int32(0):
                            local_l = local_l + cute.math.exp((score_l - mval) * Float32(self.scale_val), fastmath=True)
                    n_l = n_l + Int32(1)
                if lane_idx == Int32(0):
                    _atomic_add_f32(local_l, l_accum.iterator + ((tile_B * runtime_M + row) * Int32(self.H) + tile_H))

                d = lane_idx
                while d < Int32(self.D):
                    local_o = Float32(0.0)
                    n = split_start
                    while n < split_end:
                        valid = True
                        if cutlass.const_expr(self.causal):
                            if n > row + (runtime_N - runtime_M):
                                valid = False
                        if valid:
                            score = Float32(0.0)
                            rd = lane_idx
                            while rd < Int32(self.D):
                                score = score + q[tile_B, row, tile_H, rd].to(Float32) * k[tile_B, n, kv_h, rd].to(Float32)
                                rd = rd + Int32(32)
                            score = cute.arch.warp_reduction(score, operator.add)
                            p = cute.math.exp((score - mval) * Float32(self.scale_val), fastmath=True)
                            local_o = local_o + p * v[tile_B, n, kv_h, d].to(Float32)
                        n = n + Int32(1)
                    sO[d, Int32(0), local_m, Int32(0)] = local_o
                    d = d + Int32(32)
                if lane_idx == Int32(0):
                    accum_done[tile_B, row, tile_H, tile_SPLIT] = Int32(1)
            local_m = local_m + Int32(num_warps)
        named_barrier_sync(Int32(1), Int32(self.threads_per_row))
        global_memory_fence_gpu()

    @cute.jit
    def store(self, page_ptr, tile_B, tile_M, tile_H, tile_SPLIT, o_accum_tma, o_accum_tma_gmem):
        o_swz = cute.make_swizzle(self.o_accum_swizzle_B, 4, 3)
        sO = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(Float32, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                o_swz,
                dtype=Float32,
            ),
            cute.make_layout(self._o_accum_tma_smem_shape),
        )
        gO = cute.local_tile(
            o_accum_tma_gmem,
            self._o_accum_tma_smem_shape,
            (None, None, None, None),
        )
        tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
            o_accum_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sO, 0, 4),
            cute.group_modes(gO, 0, 4),
        )
        cute.copy(o_accum_tma, tOsO, tOgO[(None, Int32(0), tile_H, tile_M, tile_B)])


def schedule_qwen3_5_forward_split_kv_attention_experimental(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor | None = None,
    num_splits: int = 16,
    tile_m: int = 64,
    page_size: int = DEFAULT_PAGE_SIZE,
    causal: bool = True,
    kv_group_size: int = KV_GROUP_SIZE,
    write_lse: bool = False,
    use_tma_reduce: bool = True,
):
    """Schedule complete exact experimental split-KV attention.

    The default accumulator uses TMA ADD-reduce for the large O tile and keeps a
    scalar compute atomic for L.  ``use_tma_reduce=False`` keeps the older pure
    compute-atomic baseline for comparison.
    """

    B, M, H, D = q.shape
    split = torch.empty(num_splits, dtype=torch.int32, device=q.device)
    m_part = torch.empty(B, M, H, num_splits, dtype=torch.float32, device=q.device)
    m_global = torch.empty(B, M, H, dtype=torch.float32, device=q.device)
    o_accum = torch.zeros(B, M, H, D, dtype=torch.float32, device=q.device)
    l_accum = torch.zeros(B, M, H, dtype=torch.float32, device=q.device)
    accum_done = torch.empty(B, M, H, num_splits, dtype=torch.int32, device=q.device)
    if lse is None:
        lse = torch.empty(B, M, H, dtype=torch.float32, device=q.device)

    ts = {"B": 1, "M": int(tile_m), "H": 1, "SPLIT": 1}
    ops = []
    ops += Qwen3_5ForwardSplitKVMaxStatsOp.schedule(
        q=q, k=k, _split=split, m_part=m_part, tile_sizes=ts,
        causal=causal, kv_group_size=kv_group_size, num_splits=num_splits,
    )
    ops += Qwen3_5ForwardSplitKVMaxReduceOp.schedule(
        m_part=m_part, m_global=m_global,
        tile_sizes={"B": 1, "M": int(tile_m), "H": 1},
    )
    accum_cls = Qwen3_5ForwardSplitKVAccumTmaReduceOp if use_tma_reduce else Qwen3_5ForwardSplitKVAccumAtomicOp
    accum_tile_m = int(tile_m)
    if use_tma_reduce:
        accum_tile_m = min(accum_tile_m, max(1, int(page_size) // (int(D) * 4)))
    ops += accum_cls.schedule(
        q=q, k=k, v=v, m_global=m_global, _split=split,
        o_accum=o_accum, l_accum=l_accum, accum_done=accum_done,
        tile_sizes={**ts, "M": accum_tile_m},
        page_size=page_size,
        causal=causal, kv_group_size=kv_group_size, num_splits=num_splits,
    )
    ops += Qwen3_5ForwardAttentionNormalizeOp.schedule(
        o_accum=o_accum, l_accum=l_accum, m_global=m_global, accum_done=accum_done, o=o, lse=lse,
        write_lse=write_lse,
        tile_sizes={"B": 1, "M": min(64, int(tile_m)), "H": 1, "D": D},
    )
    return ops, [split, m_part, m_global, o_accum, l_accum, accum_done, lse]


def schedule_qwen3_5_forward_split_mma_attention(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor | None = None,
    chunk_m: int = 64,
    num_splits: int = 0,
    page_size: int = 96 * 1024,
    causal: bool = True,
    kv_group_size: int = KV_GROUP_SIZE,
):
    """Schedule tensor-core split-KV attention over M chunks.

    This reuses the existing FlashDecoding split/combine MMA kernels, but
    applies them to prefill-sized tensors by slicing Q/O along M.  Split kernels
    get ``q_row_offset``/``total_M`` so causal masking uses global query rows.
    """

    B, M, H, D = q.shape
    if lse is None:
        lse = torch.empty(B, H, M, dtype=torch.float32, device=q.device)

    ops = []
    keep_alive = [lse]
    for m0 in range(0, M, int(chunk_m)):
        m1 = min(M, m0 + int(chunk_m))
        q_chunk = q[:, m0:m1, :, :]
        o_chunk = o[:, m0:m1, :, :]
        lse_chunk = lse[:, :, m0:m1]
        split_ops, o_partial, lse_partial = FlashDecodingSplitBSHDOp.schedule(
            q=q_chunk,
            k=k,
            v=v,
            num_splits=num_splits,
            page_size=page_size,
            causal=causal,
            kv_group_size=kv_group_size,
        )
        for op in split_ops:
            op.static_dims["q_row_offset"] = int(m0)
            op.static_dims["total_M"] = int(M)
        combine_ops = FlashDecodingCombineBSHDOp.schedule(
            o_partial=o_partial,
            lse_partial=lse_partial,
            o=o_chunk,
            lse=lse_chunk,
        )
        ops += split_ops + combine_ops
        keep_alive.extend([q_chunk, o_chunk, lse_chunk, o_partial, lse_partial])
    return ops, keep_alive


def schedule_qwen3_5_forward_fused_split_mma_attention(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor | None = None,
    tile_m: int = 64,
    combine_tile_m: int = 64,
    num_splits: int = 2,
    page_size: int = 96 * 1024,
    causal: bool = True,
    kv_group_size: int = KV_GROUP_SIZE,
):
    """Schedule one torch-like split-KV MMA prefill op plus one combine op."""

    B, M, H, _D = q.shape
    if lse is None:
        lse = torch.empty(B, H, M, dtype=torch.float32, device=q.device)
    split_ops, o_partial, lse_partial = FlashPrefillSplitBSHDOp.schedule(
        q=q,
        k=k,
        v=v,
        num_splits=num_splits,
        page_size=page_size,
        causal=causal,
        kv_group_size=kv_group_size,
        tile_sizes={"B": 1, "M": int(tile_m), "H": 1, "SPLIT": 1},
    )
    combine_ops = FlashPrefillCombineBSHDOp.schedule(
        o_partial=o_partial,
        lse_partial=lse_partial,
        o=o,
        lse=lse,
        tile_sizes={"B": 1, "M": int(combine_tile_m), "H": 1},
    )
    return split_ops + combine_ops, [lse, o_partial, lse_partial]


class Qwen3_5ForwardGLUOp(_BaseGLUOp):
    """Qwen forward MLP activation op."""

    @classmethod
    def schedule(cls, tile_sizes=None, activation="silu", page_size=DEFAULT_PAGE_SIZE, **tensors):
        tile_sizes = dict(tile_sizes or {})
        if activation in ("silu", "swish") and tensors["y"].shape[-1] == INTERMEDIATE:
            # Current Qwen forward uses the direct-global GLU path. The generic
            # default S=16/D=512 creates too many small tiles between gate_up and
            # down. S=64/D=256 was the best full-layer variant in the local sweep.
            tile_sizes.setdefault("S", 64)
            tile_sizes.setdefault("D", 256)
            return DirectGLUOp.schedule(
                tile_sizes=tile_sizes,
                activation=activation,
                page_size=page_size,
                **tensors,
            )
        return super().schedule(
            tile_sizes=tile_sizes,
            activation=activation,
            page_size=page_size,
            **tensors,
        )


class Qwen3_5ForwardResidualAddOp(_BaseResidualAddSm120Op):
    """Qwen forward final residual add op."""


class Qwen3_5ForwardOverlapScheduler(OverlapTileScheduler):
    """Overlap scheduler for Qwen forward producer/consumer prefixes.

    The default overlap scheduler prioritizes source depth, so a ready GEMM can
    still wait until most RMSNorm tiles have been issued. Pure ready-consumer
    priority goes too far and starves the producer. This scheduler keeps a
    bounded mix in each fetch wave when both producer and consumer work is ready.
    """

    def __init__(self, *, consumer_fraction: float = 0.35, **kwargs):
        super().__init__(**kwargs)
        self.consumer_fraction = max(0.0, min(1.0, float(consumer_fraction)))

    def _schedule_stride_waves(
        self,
        candidates,
        formulas,
        depths,
        resource_scores,
        has_waits,
        fetch_stride,
    ):
        scheduled = []
        barrier_counts = {}
        barrier_ready_wave = {}
        current_wave = 0

        while candidates:
            ready = []
            fallback_ready = []
            base_ready_entries = []
            pending_by_op = {}
            base_ready_by_op = {}
            slack_ready_by_op = {}
            for op_idx, _tile_idx, _instr in candidates:
                pending_by_op[op_idx] = pending_by_op.get(op_idx, 0) + 1
            for pos, (op_idx, tile_idx, instr) in enumerate(candidates):
                wait_formulas = formulas.get(op_idx, ([], []))[0]
                if self._waits_ready(instr, wait_formulas, barrier_counts):
                    base_ready_by_op[op_idx] = base_ready_by_op.get(op_idx, 0) + 1
                    entry = (
                        *self._priority(
                            op_idx=op_idx,
                            tile_idx=tile_idx,
                            depths=depths,
                            resource_scores=resource_scores,
                            has_waits=has_waits,
                        ),
                        pos,
                    )
                    base_ready_entries.append((entry, op_idx, instr, wait_formulas))
                    fallback_ready.append(entry)
                    if self._waits_ready_with_slack(
                        instr,
                        wait_formulas,
                        barrier_counts,
                        barrier_ready_wave,
                        current_wave,
                        op_idx,
                    ):
                        slack_ready_by_op[op_idx] = slack_ready_by_op.get(op_idx, 0) + 1

            for entry, op_idx, instr, wait_formulas in base_ready_entries:
                if (
                    self._slack_applies(op_idx)
                    and (
                        base_ready_by_op.get(op_idx, 0) < pending_by_op.get(op_idx, 0)
                        or slack_ready_by_op.get(op_idx, 0) < pending_by_op.get(op_idx, 0)
                    )
                ):
                    continue
                if self._waits_ready_with_slack(
                    instr,
                    wait_formulas,
                    barrier_counts,
                    barrier_ready_wave,
                    current_wave,
                    op_idx,
                ):
                    ready.append(entry)

            if not ready:
                if fallback_ready:
                    ready = fallback_ready
                else:
                    raise RuntimeError(
                        "Qwen3_5ForwardOverlapScheduler could not find a ready tile."
                    )

            producers = [entry for entry in ready if not has_waits[candidates[entry[-1]][0]]]
            consumers = [entry for entry in ready if has_waits[candidates[entry[-1]][0]]]
            producers.sort(reverse=True)
            consumers.sort(reverse=True)
            if producers and consumers:
                consumer_quota = max(1, int(fetch_stride * self.consumer_fraction))
                producer_quota = max(1, fetch_stride - consumer_quota)
                selected = producers[:producer_quota] + consumers[:consumer_quota]
                if len(selected) < fetch_stride:
                    used = {entry[-1] for entry in selected}
                    rest = [entry for entry in ready if entry[-1] not in used]
                    rest.sort(reverse=True)
                    selected.extend(rest[: fetch_stride - len(selected)])
            else:
                ready.sort(reverse=True)
                selected = ready[:fetch_stride]

            selected_positions = [entry[-1] for entry in selected]
            selected_records = [(pos, candidates[pos]) for pos in selected_positions]

            for pos in sorted(selected_positions, reverse=True):
                candidates.pop(pos)

            for _pos, (_op_idx, _tile_idx, instr) in selected_records:
                scheduled.append(instr)
            for _pos, (op_idx, _tile_idx, instr) in selected_records:
                signal_formulas = formulas.get(op_idx, ([], []))[1]
                self._signal(
                    instr,
                    signal_formulas,
                    barrier_counts,
                    barrier_ready_wave,
                    current_wave,
                )
            current_wave += 1

        return scheduled


def _rms_tile_sizes_for_overlap(
    batch: int,
    seq_len: int,
    scheduler,
    rms_tile_s: int | None,
) -> dict[str, int] | None:
    if rms_tile_s is not None:
        return {"S": int(rms_tile_s)}
    if scheduler is None:
        return None
    # Keep enough RMS producer tiles to feed the following row-tiled GEMMs
    # without creating excessive barrier/controller overhead. For the common
    # B=1,S=512 Qwen forward shape this selects S=4; S=3 has more overlap but
    # loses to the extra 171-tile barrier fan-out, while S>=6 delays consumers.
    target_tiles = 70 * 2
    tile_s = max(1, (int(batch) * int(seq_len) + target_tiles - 1) // target_tiles)
    return {"S": tile_s}


def _attention_tile_sizes_for_overlap(
    batch: int,
    seq_len: int,
    page_size: int,
    attention_tile_m: int | None = None,
) -> dict[str, int] | None:
    if attention_tile_m is not None:
        return {"M": int(attention_tile_m)}
    # M=64 amortizes the per-tile KV loop and softmax setup better than M=32
    # on the current packed-QKV forward path, while still leaving enough tiles
    # to overlap with the surrounding projection work. M=16 over-splits and
    # loses to per-tile overhead.
    if int(batch) == 1 and int(seq_len) <= 1024 and int(page_size) <= 32 * 1024:
        return {"M": 64}
    return None


def qwen3_5_forward_gemm_tile_sizes(
    batch: int,
    seq_len: int,
    input_k: int,
    output_n: int,
    scheduler,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    elem_bytes: int = 2,
) -> dict[str, int] | None:
    if scheduler is None:
        return None
    # Adaptive tiling for the 32KB / 3-page megakernel. The decisive lever for
    # GEMM throughput here is a DEEP K tile (large tile_k => fewer K-loop
    # iterations => less per-iteration LdMatrix/MMA overhead) combined with a C
    # tile small enough (<= half a page) that its store overlaps the next tile's
    # A/B loads across the 3-page ring, and enough tiles to fill all 70 SMs with
    # a couple of waves. The old heuristic maximized the C tile, which pinned
    # tile_k at 16 and lost ~10% vs torch; preferring deep K beats torch.
    min_tiles = 70 * 2
    best = None
    for tile_s in (256, 128, 64, 32, 16):
        s_tiles = (int(seq_len) + tile_s - 1) // tile_s
        for tile_n in (256, 128, 64, 32):
            if output_n % tile_n != 0:
                continue
            for tile_k in (64, 48, 32, 16):
                ab_total = 2 * (tile_s + tile_n) * tile_k * elem_bytes + 32
                c_total = tile_s * tile_n * elem_bytes
                if max(ab_total, c_total) > page_size:
                    continue
                tiles = int(batch) * s_tiles * ((int(output_n) + tile_n - 1) // tile_n)
                # Prefer (verified by sweep, e.g. qk -> S128/N64/K32 beats torch):
                #  1. a store-overlappable C tile (c fits in <= half a page so its
                #     TMA store overlaps the next tile's A/B loads on the 3-page ring),
                #  2. the LARGEST such C footprint (more MMA work per A/B load) up to
                #     that half-page cap,
                #  3. the deepest K (fewer K-loop iterations / less overhead),
                #  4. square-ish tiles with more rows (more MMA warps).
                c_overlap = 0 if c_total * 2 <= page_size else 1
                enough = 0 if tiles >= min_tiles else (min_tiles - tiles)
                score = (
                    c_overlap,
                    enough,
                    -min(tile_s * tile_n, page_size // (2 * elem_bytes)),
                    -tile_k,
                    abs(tile_s - tile_n),
                    -tile_s,
                )
                if best is None or score < best[0]:
                    best = (score, {"S": tile_s, "N": tile_n, "K": tile_k})
    return None if best is None else best[1]


def _complete_qwen_forward_gemm_tile_sizes(
    tile_sizes: dict[str, int],
    a: torch.Tensor,
    b: torch.Tensor,
    page_size: int,
) -> None:
    """Fill missing Qwen forward GEMM tiling, especially the inner K tile.

    Callers sometimes pin spatial tiling from an autotune/scheduler experiment
    and leave K unspecified. The generic GEMM default is K=32, which doubles the
    deepest down-projection K-loop. Keep Qwen forward on the deepest fitting K.
    """
    selected = None
    if "S" not in tile_sizes and "N" not in tile_sizes:
        selected = qwen3_5_forward_gemm_tile_sizes(
            int(a.shape[0]),
            int(a.shape[1]),
            int(a.shape[-1]),
            int(b.shape[0]),
            scheduler=True,
            page_size=page_size,
            elem_bytes=int(a.element_size()),
        )
        if selected is not None:
            tile_sizes.update(selected)

    if "K" in tile_sizes:
        return
    if "S" not in tile_sizes or "N" not in tile_sizes:
        return

    tile_s = int(tile_sizes["S"])
    tile_n = int(tile_sizes["N"])
    elem_bytes = int(a.element_size())
    for tile_k in (64, 48, 32, 16):
        ab_total = 2 * (tile_s + tile_n) * tile_k * elem_bytes + 32
        c_total = tile_s * tile_n * elem_bytes
        if max(ab_total, c_total) <= page_size:
            tile_sizes["K"] = tile_k
            return


def _flatten_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    return x.reshape(x.shape[0] * x.shape[1], heads, HEAD_DIM)


def _unflatten_heads(x: torch.Tensor, batch: int, seq_len: int, heads: int) -> torch.Tensor:
    return x.reshape(batch, seq_len, heads, HEAD_DIM)


def schedule_qwen3_5_forward_ops(
    batch: int,
    seq_len: int,
    x: torch.Tensor,
    residual: torch.Tensor,
    attn_norm: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    w_o: torch.Tensor,
    mlp_norm: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    scheduler=None,
    gemm_tile_sizes: dict[str, dict[str, int]] | None = None,
    rms_tile_s: int | None = None,
    adaptive_gemm_tiling: bool = False,
    use_packed_qk: bool = True,
    use_qwen_projection: bool = False,
    use_packed_qkv_projection: bool = True,
    use_fused_rms_proj: bool = False,
    use_fused_down_residual: bool = True,
    use_qknorm_4d: bool = True,
    use_cpasync_projection: bool = False,
    use_split_attention: bool = False,
    split_attention_splits: int = 0,
    use_tma_attention: bool = False,
    attention_tile_m: int | None = None,
    op_limit: int | None = None,
) -> Qwen3_5ForwardSchedule:
    dtype = x.dtype
    device = x.device
    cos_flat = cos.repeat(batch, 1).contiguous()
    sin_flat = sin.repeat(batch, 1).contiguous()

    if use_packed_qkv_projection and not use_packed_qk:
        raise ValueError("use_packed_qkv_projection requires use_packed_qk=True")

    x0 = torch.empty_like(x)
    qkv = None
    w_qkv = None
    if use_packed_qk:
        if use_packed_qkv_projection:
            qkv_dim = Q_DIM + 2 * KV_DIM
            qkv = torch.empty(batch, seq_len, qkv_dim, dtype=dtype, device=device)
            qk = torch.as_strided(
                qkv,
                (batch, seq_len, Q_DIM + KV_DIM),
                (seq_len * qkv_dim, qkv_dim, 1),
            )
            q = torch.as_strided(
                qkv,
                (batch, seq_len, Q_DIM),
                (seq_len * qkv_dim, qkv_dim, 1),
            )
            k = torch.as_strided(
                qkv,
                (batch, seq_len, KV_DIM),
                (seq_len * qkv_dim, qkv_dim, 1),
                storage_offset=Q_DIM,
            )
            v = torch.as_strided(
                qkv,
                (batch, seq_len, KV_DIM),
                (seq_len * qkv_dim, qkv_dim, 1),
                storage_offset=Q_DIM + KV_DIM,
            )
            w_qkv = torch.cat((w_q, w_k, w_v), dim=0).contiguous()
            w_qk = None
        else:
            qk = torch.empty(batch, seq_len, Q_DIM + KV_DIM, dtype=dtype, device=device)
            q = qk[..., :Q_DIM]
            k = qk[..., Q_DIM:]
            v = torch.empty(batch, seq_len, KV_DIM, dtype=dtype, device=device)
            w_qk = torch.cat((w_q, w_k), dim=0).contiguous()
        qk_sumsq = torch.empty(
            batch,
            seq_len,
            NUM_Q_HEADS + NUM_KV_HEADS,
            HEAD_DIM // 64,
            dtype=torch.float32,
            device=device,
        )
    else:
        qk = None
        w_qk = None
        qk_sumsq = None
        q = torch.empty(batch, seq_len, Q_DIM, dtype=dtype, device=device)
        k = torch.empty(batch, seq_len, KV_DIM, dtype=dtype, device=device)
        v = torch.empty(batch, seq_len, KV_DIM, dtype=dtype, device=device)
    attn = torch.empty(batch, seq_len, Q_DIM, dtype=dtype, device=device)
    lse = torch.empty(batch, seq_len, NUM_Q_HEADS, dtype=torch.float32, device=device)
    attn_proj = torch.empty_like(x)
    residual1 = torch.empty_like(x)
    x1 = torch.empty_like(x)
    gate_up = torch.empty(batch, seq_len, 2 * INTERMEDIATE, dtype=dtype, device=device)
    mlp = torch.empty(batch, seq_len, INTERMEDIATE, dtype=dtype, device=device)
    mlp_out = torch.empty_like(x)
    y = torch.empty_like(x)

    def _packed_heads_view(base: torch.Tensor, heads: int, offset: int = 0) -> torch.Tensor:
        return torch.as_strided(
            base,
            (batch, seq_len, heads, HEAD_DIM),
            (base.stride(0), base.stride(1), HEAD_DIM, 1),
            storage_offset=base.storage_offset() + offset,
        )

    qh = _flatten_heads(q, NUM_Q_HEADS) if not use_packed_qk else _packed_heads_view(qk, NUM_Q_HEADS, 0)
    kh = _flatten_heads(k, NUM_KV_HEADS) if not use_packed_qk else _packed_heads_view(qk, NUM_KV_HEADS, Q_DIM)
    qkh = (
        torch.as_strided(
            qk,
            (batch * seq_len, NUM_Q_HEADS + NUM_KV_HEADS, HEAD_DIM),
            (qk.stride(1), HEAD_DIM, 1),
            storage_offset=qk.storage_offset(),
        )
        if qk is not None
        else None
    )
    qkh4 = (
        torch.as_strided(
            qk,
            (batch, seq_len, NUM_Q_HEADS + NUM_KV_HEADS, HEAD_DIM),
            (qk.stride(0), qk.stride(1), HEAD_DIM, 1),
            storage_offset=qk.storage_offset(),
        )
        if qk is not None
        else None
    )
    kh4 = _packed_heads_view(qk, NUM_KV_HEADS, Q_DIM) if use_packed_qk else _unflatten_heads(k, batch, seq_len, NUM_KV_HEADS)
    vh = _packed_heads_view(qkv, NUM_KV_HEADS, Q_DIM + KV_DIM) if use_packed_qkv_projection else _unflatten_heads(v, batch, seq_len, NUM_KV_HEADS)
    attn_h = _unflatten_heads(attn, batch, seq_len, NUM_Q_HEADS)

    user_gemm_tile_sizes = gemm_tile_sizes or {}
    resolved_gemm_tile_sizes: dict[str, dict[str, int]] = {}
    if adaptive_gemm_tiling:
        resolved_gemm_tile_sizes.update(
            {
                "qk": qwen3_5_forward_gemm_tile_sizes(batch, seq_len, HIDDEN, Q_DIM + KV_DIM, scheduler, page_size=page_size),
                "qkv": qwen3_5_forward_gemm_tile_sizes(batch, seq_len, HIDDEN, Q_DIM + 2 * KV_DIM, scheduler, page_size=page_size),
                "q": qwen3_5_forward_gemm_tile_sizes(batch, seq_len, HIDDEN, Q_DIM, scheduler, page_size=page_size),
                "k": qwen3_5_forward_gemm_tile_sizes(batch, seq_len, HIDDEN, KV_DIM, scheduler, page_size=page_size),
                "v": qwen3_5_forward_gemm_tile_sizes(batch, seq_len, HIDDEN, KV_DIM, scheduler, page_size=page_size),
                "o": qwen3_5_forward_gemm_tile_sizes(batch, seq_len, Q_DIM, HIDDEN, scheduler, page_size=page_size),
                "gate_up": qwen3_5_forward_gemm_tile_sizes(batch, seq_len, HIDDEN, 2 * INTERMEDIATE, scheduler, page_size=page_size),
                "down": qwen3_5_forward_gemm_tile_sizes(batch, seq_len, INTERMEDIATE, HIDDEN, scheduler, page_size=page_size),
            }
        )
        resolved_gemm_tile_sizes = {
            name: sizes for name, sizes in resolved_gemm_tile_sizes.items() if sizes is not None
        }
    for name, sizes in user_gemm_tile_sizes.items():
        base = dict(resolved_gemm_tile_sizes.get(name) or {})
        base.update(sizes)
        resolved_gemm_tile_sizes[name] = base
    rms_tile_sizes = _rms_tile_sizes_for_overlap(batch, seq_len, scheduler, rms_tile_s)
    def _gemm(name: str, *, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        if name == "qk" and use_qwen_projection:
            scheduled = Qwen3_5ForwardPackedQKProjectionOp.schedule(
                a=a,
                b=b,
                c=c,
                qk_sumsq=qk_sumsq,
                page_size=page_size,
                tile_sizes=resolved_gemm_tile_sizes.get(name),
            )
        else:
            projection_cls = (
                Qwen3_5ForwardCpAsyncProjectionOp
                if use_cpasync_projection
                else Qwen3_5ForwardProjectionOp
            )
            scheduled = projection_cls.schedule(
                a=a,
                b=b,
                c=c,
                page_size=page_size,
                tile_sizes=resolved_gemm_tile_sizes.get(name),
            )
        for op in scheduled:
            op.static_dims["barrier_allow_piecewise_overlap"] = 1
        return scheduled

    extra_keep: list = []

    def _finish(output: torch.Tensor, residual_out: torch.Tensor = residual) -> Qwen3_5ForwardSchedule:
        keep_alive = [
            *extra_keep,
            cos_flat,
            sin_flat,
            x0,
            q,
            k,
            v,
            qk,
            qk_sumsq,
            qkv,
            w_qk,
            w_qkv,
            qh,
            kh,
            qkh,
            qkh4,
            kh4,
            vh,
            attn,
            attn_h,
            lse,
            attn_proj,
            residual1,
            x1,
            gate_up,
            mlp,
            mlp_out,
            y,
        ]
        return Qwen3_5ForwardSchedule(ops=ops, output=output, residual=residual_out, keep_alive=keep_alive)

    def _fused_rms_proj(name, b_raw, c):
        # Bake attn_norm into the projection weight once, then fuse RMSNorm into
        # the projection: the op reads raw x, computes per-row rstd from the A
        # fragment in the K-loop, and scales the output. No x0 global round-trip.
        b_baked = (b_raw.float() * attn_norm.float()).to(dtype).contiguous()
        extra_keep.append(b_baked)
        scheduled = Qwen3_5RmsProjOp.schedule(
            a=x, b=b_baked, c=c, rmsnorm_weight=None,
            page_size=page_size, tile_sizes=resolved_gemm_tile_sizes.get(name),
        )
        for op in scheduled:
            op.static_dims["barrier_allow_piecewise_overlap"] = 1
        return scheduled

    ops = []
    if not use_fused_rms_proj:
        ops += Qwen3_5ForwardRMSNormOp.schedule(
            x=x,
            weight=attn_norm,
            y=x0,
            page_size=page_size,
            eps=QWEN3_5_EPS,
            tile_sizes=rms_tile_sizes,
        )
        if op_limit is not None and len(ops) >= op_limit:
            return _finish(x0)
    proj_a = x if use_fused_rms_proj else x0
    if use_packed_qk:
        if use_packed_qkv_projection:
            ops += (
                _fused_rms_proj("qkv", w_qkv, qkv)
                if use_fused_rms_proj
                else _gemm("qkv", a=proj_a, b=w_qkv, c=qkv)
            )
        else:
            ops += _fused_rms_proj("qk", w_qk, qk) if use_fused_rms_proj else _gemm("qk", a=proj_a, b=w_qk, c=qk)
        qk_output = qk
    else:
        ops += _fused_rms_proj("q", w_q, q) if use_fused_rms_proj else _gemm("q", a=proj_a, b=w_q, c=q)
        ops += _fused_rms_proj("k", w_k, k) if use_fused_rms_proj else _gemm("k", a=proj_a, b=w_k, c=k)
        qk_output = k
    if op_limit is not None and len(ops) >= op_limit:
        return _finish(qk_output)
    if not use_packed_qkv_projection:
        ops += _fused_rms_proj("v", w_v, v) if use_fused_rms_proj else _gemm("v", a=proj_a, b=w_v, c=v)
        if op_limit is not None and len(ops) >= op_limit:
            return _finish(v)
    if use_packed_qk:
        if use_qknorm_4d:
            ops += Qwen3_5ForwardPackedQKNormRope4DOp.schedule(
                qk=qkh4,
                q_norm_weight=q_norm,
                k_norm_weight=k_norm,
                cos=cos_flat,
                sin=sin_flat,
                page_size=page_size,
                eps=QWEN3_5_EPS,
                num_q_heads=NUM_Q_HEADS,
                num_k_heads=NUM_KV_HEADS,
            )
        else:
            ops += Qwen3_5ForwardPackedQKNormRopeOp.schedule(
                qk=qkh,
                q_norm_weight=q_norm,
                k_norm_weight=k_norm,
                cos=cos_flat,
                sin=sin_flat,
                page_size=page_size,
                eps=QWEN3_5_EPS,
                num_q_heads=NUM_Q_HEADS,
                num_k_heads=NUM_KV_HEADS,
            )
    else:
        ops += Qwen3_5ForwardQKNormRopeOp.schedule(q=qh, norm_weight=q_norm, cos=cos_flat, sin=sin_flat, page_size=page_size, eps=QWEN3_5_EPS)
        ops += Qwen3_5ForwardQKNormRopeOp.schedule(q=kh, norm_weight=k_norm, cos=cos_flat, sin=sin_flat, page_size=page_size, eps=QWEN3_5_EPS)
    if op_limit is not None and len(ops) >= op_limit:
        return _finish(qk_output)
    _attn_q = qh if use_packed_qk else _unflatten_heads(q, batch, seq_len, NUM_Q_HEADS)
    if use_tma_attention:
        ops += Qwen3_5ForwardTmaAttentionOp.schedule(
            q=_attn_q,
            k=kh4,
            v=vh,
            o=attn_h,
            lse=lse,
            causal=True,
            kv_group_size=KV_GROUP_SIZE,
            page_size=page_size,
            tile_sizes=_attention_tile_sizes_for_overlap(batch, seq_len, page_size, attention_tile_m),
            write_lse=False,
        )
    elif use_split_attention:
        _split_ops, _o_partial, _lse_partial = FlashDecodingSplitBSHDOp.schedule(
            q=_attn_q,
            k=kh4,
            v=vh,
            num_splits=split_attention_splits,
            page_size=page_size,
            causal=True,
            kv_group_size=KV_GROUP_SIZE,
        )
        _combine_ops = FlashDecodingCombineBSHDOp.schedule(
            o_partial=_o_partial,
            lse_partial=_lse_partial,
            o=attn_h,
            lse=lse,
        )
        extra_keep.extend([_o_partial, _lse_partial])
        ops += _split_ops + _combine_ops
    else:
        ops += Qwen3_5ForwardAttentionOp.schedule(
            q=_attn_q,
            k=kh4,
            v=vh,
            o=attn_h,
            lse=lse,
            causal=True,
            kv_group_size=KV_GROUP_SIZE,
            page_size=page_size,
            tile_sizes=_attention_tile_sizes_for_overlap(batch, seq_len, page_size, attention_tile_m),
            write_lse=False,
        )
    if op_limit is not None and len(ops) >= op_limit:
        return _finish(attn)
    ops += _gemm("o", a=attn, b=w_o, c=attn_proj)
    if op_limit is not None and len(ops) >= op_limit:
        return _finish(attn_proj)
    ops += Qwen3_5ForwardRMSNormOp.schedule(
        x=attn_proj,
        residual_in=residual,
        residual_out=residual1,
        weight=mlp_norm,
        y=x1,
        page_size=page_size,
        eps=QWEN3_5_EPS,
        tile_sizes=rms_tile_sizes,
    )
    if op_limit is not None and len(ops) >= op_limit:
        return _finish(x1, residual1)
    ops += _gemm("gate_up", a=x1, b=w_gate_up, c=gate_up)
    if op_limit is not None and len(ops) >= op_limit:
        return _finish(gate_up, residual1)
    ops += Qwen3_5ForwardGLUOp.schedule(x=gate_up, y=mlp, activation="silu", page_size=page_size)
    if op_limit is not None and len(ops) >= op_limit:
        return _finish(mlp, residual1)
    if use_fused_down_residual:
        # Fuse y = (mlp @ w_down^T) + residual1 into the down GEMM epilogue,
        # removing the standalone residual-add op (a serial tail) + the mlp_out
        # round-trip.
        down_ops = Qwen3_5DownResidualOp.schedule(
            a=mlp, b=w_down, c=y, residual=residual1, page_size=page_size,
            tile_sizes=resolved_gemm_tile_sizes.get("down"),
        )
        for op in down_ops:
            op.static_dims["barrier_allow_piecewise_overlap"] = 1
        ops += down_ops
        return _finish(y, residual1)

    ops += _gemm("down", a=mlp, b=w_down, c=mlp_out)
    if op_limit is not None and len(ops) >= op_limit:
        return _finish(mlp_out, residual1)
    ops += Qwen3_5ForwardResidualAddOp.schedule(
        x=mlp_out,
        residual_in=residual1,
        residual_out=y,
        tile_sizes={"S": 16, "K": 256},
        page_size=page_size,
    )

    return _finish(y, residual1)


__all__ = [
    "DEFAULT_PAGE_SIZE",
    "HIDDEN",
    "INTERMEDIATE",
    "NUM_Q_HEADS",
    "NUM_KV_HEADS",
    "HEAD_DIM",
    "Q_DIM",
    "KV_DIM",
    "KV_GROUP_SIZE",
    "Qwen3_5ForwardAttentionOp",
    "Qwen3_5ForwardAttentionNormalizeOp",
    "Qwen3_5ForwardNoInnerKVTmaReduceStoreAttentionOp",
    "Qwen3_5ForwardSplitKVAccumAtomicOp",
    "Qwen3_5ForwardSplitKVAccumTmaReduceOp",
    "Qwen3_5ForwardSplitKVMaxReduceOp",
    "Qwen3_5ForwardSplitKVMaxStatsOp",
    "schedule_qwen3_5_forward_fused_split_mma_attention",
    "schedule_qwen3_5_forward_split_mma_attention",
    "Qwen3_5ForwardTmaReduceStoreAttentionOp",
    "Qwen3_5ForwardTmaAttentionOp",
    "Qwen3_5ForwardGLUOp",
    "Qwen3_5ForwardOverlapScheduler",
    "Qwen3_5ForwardPackedQKProjectionOp",
    "Qwen3_5ForwardPackedQKNormRopeOp",
    "Qwen3_5ForwardPackedQKNormRope4DOp",
    "Qwen3_5ForwardProjectionOp",
    "Qwen3_5ForwardQKNormRopeOp",
    "Qwen3_5ForwardResidualAddOp",
    "Qwen3_5ForwardSchedule",
    "Qwen3_5ForwardRMSNormOp",
    "qwen3_5_forward_gemm_tile_sizes",
    "schedule_qwen3_5_forward_ops",
]
