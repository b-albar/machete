# Copyright (c) 2025, Machete Authors
"""
Fused RMSNorm + GEMM Op for the Megakernel.

Computes C[B,S,N] = rmsnorm(A[B,S,K], weight) @ B_w[N,K]^T in a single op.

The RMSNorm weight is pre-baked into B_w at schedule time:
    B_fused[n,k] = B_w[n,k] * rmsnorm_weight[k]       (standard)
    B_fused[n,k] = B_w[n,k] * (1 + rmsnorm_weight[k]) (gemma)

This allows the GEMM to compute C = A @ B_fused^T, then scale by rstd:
    rstd = rsqrt(mean(A^2) + eps) per row
    C_out = C * rstd

Architecture:
    - Inherits all TMA load/store, mbarrier, LdMatrix from GemmOp
    - Overrides compute: standard K-loop MMA, then in epilogue reads A
      from global to compute per-row sum_sq → rstd, scales acc in
      registers before writing to swizzled smem.
    - MMA C fragment row mapping: for m16n8k16, every 4 consecutive
      elements: [0,1] = row groupID, [2,3] = row groupID+8.
      groupID = lane_idx // 4 (0..7), covering all 16 rows per warp.
"""

import operator

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32

from machete.megakernel.ops import DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import (
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_inval,
    mbarrier_arrive,
    mbarrier_arrive_expect_tx,
    mbarrier_wait,
    named_barrier_sync,
)
from machete.kernels.gemm.gemm import GemmOp
from machete.kernels.rms_norm.rms_norm import RMSNORM_EPS


class RMSNormGemmOp(GemmOp):
    """Fused RMSNorm + GEMM.

    Computes C = rstd * (A @ B_fused^T) where B_fused = diag(w) @ B_w,
    and rstd = rsqrt(mean(A^2) + eps) per row.

    Inherits load/store/TMA from GemmOp. Only compute is overridden.
    """

    def __init__(self, **config):
        super().__init__(**config)

    # =========================================================================
    # Compute: GEMM K-loop + rstd epilogue scaling
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_N,
                a, a_scale, b, c):
        """Fused RMSNorm + GEMM compute.

        Phase 1: Standard GEMM K-loop (same as GemmOp).
        Phase 2: Read A from global, compute per-row rstd.
        Phase 3: Scale acc in registers by rstd, write to swizzled smem.
        """
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()

        # --- Build tiled MMA (same as GemmOp) ---
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            self.a_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.num_mma_warps, 1, 1)),
            permutation_mnk=(self.num_mma_warps * 16, 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        # --- LdMatrix tiled copies ---
        swz = cute.make_swizzle(self.swz_B_ab, 4, 3)

        smem_copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=False, num_matrices=4), self.a_dtype)
        smem_tiled_copy_A = cute.make_tiled_copy_A(
            smem_copy_atom_A, tiled_mma)
        smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)

        smem_copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=False, num_matrices=4), self.b_dtype)
        smem_tiled_copy_B = cute.make_tiled_copy_B(
            smem_copy_atom_B, tiled_mma)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        # --- Buffer 0 and 1 smem tensors ---
        sA_0 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.a_dtype, page_ptr,
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.a_dtype),
            cute.make_layout((self.tile_size_S, self.tile_K),
                             stride=(self.tile_K, 1)),
        )
        sB_0 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.b_dtype,
                              page_ptr + Int32(self.b_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.b_dtype),
            cute.make_layout((self.tile_size_N, self.tile_K),
                             stride=(self.tile_K, 1)),
        )
        sA_1 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.a_dtype,
                              page_ptr + Int32(self.buf_stride),
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.a_dtype),
            cute.make_layout((self.tile_size_S, self.tile_K),
                             stride=(self.tile_K, 1)),
        )
        sB_1 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.b_dtype,
                              page_ptr + Int32(self.buf_stride + self.b_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.b_dtype),
            cute.make_layout((self.tile_size_N, self.tile_K),
                             stride=(self.tile_K, 1)),
        )

        # MMA partitions and register fragments
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

        # Op-managed mbarriers
        _bf_0 = page_ptr + Int32(self.mbar_offset)
        _bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        _kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        _kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        # --- Init fp32 accumulator ---
        acc = cute.make_fragment(
            tiled_mma.partition_shape_C((self.tile_size_S, self.tile_size_N)),
            Float32,
        )
        acc.fill(0.0)

        # =================================================================
        # K-block 0 from buf 0
        # =================================================================
        for k_block in cutlass.range_constexpr(self.tile_K // 16):
            cute.copy(smem_tiled_copy_A,
                      tAsA_ld_0[None, None, k_block],
                      tCrA_ld[None, None, k_block])
            cute.copy(smem_tiled_copy_B,
                      tBsB_ld_0[None, None, k_block],
                      tCrB_ld[None, None, k_block])
            cute.gemm(tiled_mma, acc,
                      tCrA[None, None, k_block],
                      tCrB[None, None, k_block], acc)

        if tidx % Int32(32) == Int32(0):
            mbarrier_arrive(_bf_0)

        # =================================================================
        # K-block 1 from buf 1
        # =================================================================
        if self.num_k_blocks >= 2:
            for k_block in cutlass.range_constexpr(self.tile_K // 16):
                cute.copy(smem_tiled_copy_A,
                          tAsA_ld_1[None, None, k_block],
                          tCrA_ld[None, None, k_block])
                cute.copy(smem_tiled_copy_B,
                          tBsB_ld_1[None, None, k_block],
                          tCrB_ld[None, None, k_block])
                cute.gemm(tiled_mma, acc,
                          tCrA[None, None, k_block],
                          tCrB[None, None, k_block], acc)

            if tidx % Int32(32) == Int32(0):
                mbarrier_arrive(_bf_1)

        # =================================================================
        # K-blocks 2+
        # =================================================================
        _kr_phase_0 = Int32(0)
        _kr_phase_1 = Int32(0)
        k_idx = Int32(2)
        while k_idx < Int32(self.num_k_blocks):
            if k_idx % Int32(2) == Int32(0):
                mbarrier_wait(_kr_0, _kr_phase_0)
                _kr_phase_0 = _kr_phase_0 ^ Int32(1)
            if k_idx % Int32(2) == Int32(1):
                mbarrier_wait(_kr_1, _kr_phase_1)
                _kr_phase_1 = _kr_phase_1 ^ Int32(1)

            if k_idx % Int32(2) == Int32(0):
                for k_block in cutlass.range_constexpr(self.tile_K // 16):
                    cute.copy(smem_tiled_copy_A,
                              tAsA_ld_0[None, None, k_block],
                              tCrA_ld[None, None, k_block])
                    cute.copy(smem_tiled_copy_B,
                              tBsB_ld_0[None, None, k_block],
                              tCrB_ld[None, None, k_block])
                    cute.gemm(tiled_mma, acc,
                              tCrA[None, None, k_block],
                              tCrB[None, None, k_block], acc)
            if k_idx % Int32(2) == Int32(1):
                for k_block in cutlass.range_constexpr(self.tile_K // 16):
                    cute.copy(smem_tiled_copy_A,
                              tAsA_ld_1[None, None, k_block],
                              tCrA_ld[None, None, k_block])
                    cute.copy(smem_tiled_copy_B,
                              tBsB_ld_1[None, None, k_block],
                              tCrB_ld[None, None, k_block])
                    cute.gemm(tiled_mma, acc,
                              tCrA[None, None, k_block],
                              tCrB[None, None, k_block], acc)

            if tidx % Int32(32) == Int32(0):
                if k_idx % Int32(2) == Int32(0):
                    mbarrier_arrive(_bf_0)
                if k_idx % Int32(2) == Int32(1):
                    mbarrier_arrive(_bf_1)

            k_idx = k_idx + Int32(1)

        # =================================================================
        # Epilogue: compute rstd, scale acc, write to smem
        # =================================================================
        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

        if tidx == Int32(0):
            mbarrier_inval(_bf_0)
            mbarrier_inval(_bf_1)
            mbarrier_inval(_kr_0)
            mbarrier_inval(_kr_1)

        # --- Compute rstd per row from global A (register-only, no smem) ---
        # Each warp handles 16 rows, 32 lanes split K columns.
        # warp_reduction returns result to ALL lanes, so each lane captures
        # its 2 needed rstd values (for groupID and groupID+8) via
        # conditional register assignments — no smem sync needed.
        _ROWS_PER_WARP = 16
        _my_row_base = warp_idx * Int32(_ROWS_PER_WARP)
        _row_start = tile_S * Int32(self.tile_size_S)
        _inv_K = Float32(1.0 / self.K)
        _thr_layout = cute.make_layout(32)

        # MMA C fragment row mapping for m16n8k16:
        #   groupID = lane_idx // 4  (0..7)
        #   ci % 4 < 2:  row = groupID      (rows 0-7 within warp's 16)
        #   ci % 4 >= 2: row = groupID + 8  (rows 8-15 within warp's 16)
        _group_id = lane_idx // Int32(4)
        _rstd_lo = Float32(1.0)
        _rstd_hi = Float32(1.0)

        for _r in cutlass.range_constexpr(_ROWS_PER_WARP):
            _global_row = _row_start + _my_row_base + Int32(_r)
            _base_off = tile_B * Int32(self.S * self.K) + _global_row * Int32(self.K)

            _a_row = cute.make_tensor(
                a.iterator + _base_off,
                cute.make_layout(self.K),
            )
            _a_part = cute.local_partition(_a_row, _thr_layout, lane_idx)

            _partial = Float32(0.0)
            for _i in range(cute.size(_a_part)):
                _v = _a_part[_i].to(Float32)
                _partial = _partial + _v * _v

            _row_sum = cute.arch.warp_reduction(_partial, operator.add)
            _rstd = cute.math.rsqrt(
                _row_sum * _inv_K + RMSNORM_EPS, fastmath=True)

            # Each lane captures rstd for its 2 MMA rows (groupID and groupID+8)
            if _group_id == Int32(_r):
                _rstd_lo = _rstd
            if _group_id + Int32(8) == Int32(_r):
                _rstd_hi = _rstd

        # Scale acc in registers
        for ci in cutlass.range_constexpr(cute.size(acc)):
            if ci % 4 < 2:
                acc[ci] = acc[ci] * _rstd_lo
            else:
                acc[ci] = acc[ci] * _rstd_hi

        # --- Write to swizzled smem (same as GemmOp epilogue) ---
        swz_c = cute.make_swizzle(self.swz_B_c, 4, 3)
        sC = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.c_dtype, page_ptr,
                              cute.AddressSpace.smem, assumed_align=128),
                swz_c, dtype=self.c_dtype),
            cute.make_layout((self.tile_size_S, self.tile_size_N),
                             stride=(self.tile_size_N, 1)),
        )

        acc_out = cute.make_fragment_like(acc, self.c_dtype)
        for ci in cutlass.range_constexpr(cute.size(acc)):
            acc_out[ci] = acc[ci].to(self.c_dtype)

        r2s_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.c_dtype)
        r2s_copy = cute.make_tiled_copy_C(r2s_atom, tiled_mma)
        r2s_thr = r2s_copy.get_slice(tidx)
        tCrC = r2s_thr.retile(acc_out)
        tCsC = r2s_thr.partition_D(sC)
        cute.copy(r2s_copy, tCrC, tCsC)

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule(cls, rmsnorm_weight=None, gemma=False,
                 tile_sizes=None, page_size=DEFAULT_PAGE_SIZE,
                 activation=None, **tensors):
        """Schedule fused RMSNorm + GEMM.

        Pre-bakes rmsnorm_weight into b:
            b_fused[n,k] = b[n,k] * rmsnorm_weight[k]

        Args:
            rmsnorm_weight: (K,) RMSNorm weight tensor.
            gemma: If True, use (1 + weight) instead of weight.
            tile_sizes: Optional dict with S, N, K keys.
            page_size: Shared memory page size.
            activation: Optional activation ('relu', 'silu').
            **tensors: a, b, c (required).
        """
        import torch

        b = tensors['b']
        if rmsnorm_weight is not None:
            w = rmsnorm_weight.float()
            if gemma:
                w = 1.0 + w
            tensors['b'] = (b.float() * w.unsqueeze(0)).to(b.dtype).contiguous()

        if "a_scale" not in tensors:
            tensors["a_scale"] = tensors["a"]

        ts = dict(tile_sizes or {})
        ts.setdefault("B", 1)
        if "S" not in ts or "N" not in ts:
            a_t = tensors.get('a')
            elem_bytes = a_t.element_size() if a_t is not None else 2
            auto_S, auto_N, auto_K = cls._auto_tiles(page_size, elem_bytes)
            ts.setdefault("S", auto_S)
            ts.setdefault("N", auto_N)
            ts.setdefault("K", auto_K)
        tile_K = ts.pop("K", 32)
        scheduled = cls._schedule_single(tile_sizes=ts, **tensors)
        scheduled.static_dims["tile_K"] = tile_K
        scheduled.static_dims["page_size"] = page_size
        scheduled.static_dims["has_a_scale"] = 0
        if activation is not None:
            from machete.kernels.activation.activation import ACT_MAP
            scheduled.static_dims["activation"] = ACT_MAP[activation]
        return [scheduled]


__all__ = ["RMSNormGemmOp"]
