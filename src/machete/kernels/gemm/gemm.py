# Copyright (c) 2025, Machete Authors
"""
GEMM Op for the Megakernel.

Computes C[B,S,N] = A[B,S,K] @ B_w[N,K]^T  (standard matmul with B pre-transposed).

Tensor shapes follow LLM conventions:
    A: (B, S, K) — batch, sequence, input dim
    B: (N, K)    — weight matrix (N = output dim, K contiguous)
    C: (B, S, N) — output

Note: B_w is expected in (N, K) layout (K contiguous). If you have B in standard
PyTorch (K, N) layout, pass b.t().contiguous() when scheduling.

Architecture (SM_90+ / Hopper+):
    - Tensor core warp MMA: MmaF16BF16Op(16, 8, 16)
    - Supports fp16 and bf16 inputs, fp32 accumulation across all K blocks
    - K is handled via an internal double-buffered loop in compute
    - TMA for all G->S loads: first 2 K-blocks by DMA warp, rest by
      elected MMA thread in compute via compute-local mbarriers
    - Regular TMA store for C
    - LdMatrix for warp-cooperative smem->register reads

Pipelined phases:
    load:    TMA G->S of first 2 K-blocks of A and B into 2 smem buffers.
    compute: Process K-blocks 0-1 (TMA-loaded by DMA), then TMA double-buffered
             loop for K-blocks 2+ (elected MMA thread). LdMatrix + MMA per K-block.
             Epilogue: R->S.
    store:   TMA store S->G of C[tile_S, tile_N]

Page layout (2-stage double buffer):
    [buf0: A_tile + B_tile] [buf1: A_tile + B_tile] [mbar0: 8B] [mbar1: 8B]
    Epilogue: [C: tile_S x tile_N]  (reuses page from offset 0)

No output pre-zeroing needed — fp32 accumulator handles full K reduction.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import (
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_inval,
    mbarrier_arrive,
    mbarrier_arrive_expect_tx,
    mbarrier_wait,
    named_barrier_sync,
)


@cute.jit
def _gemm_epilogue_store_helper(page_ptr, tidx, tiled_mma, acc,
                                _bf_0, _bf_1, _kr_0, _kr_1,
                                num_mma_threads,
                                swz_B_c,
                                c_dtype,
                                tile_size_S,
                                tile_size_N,
                                activation):
    """Finalize GEMM accumulators into swizzled shared memory."""
    named_barrier_sync(Int32(2), Int32(num_mma_threads))

    if tidx == Int32(0):
        mbarrier_inval(_bf_0)
        mbarrier_inval(_bf_1)
        mbarrier_inval(_kr_0)
        mbarrier_inval(_kr_1)

    swz_c = cute.make_swizzle(swz_B_c, 4, 3)
    sC = cute.make_tensor(
        cute.recast_ptr(
            cute.make_ptr(c_dtype, page_ptr,
                          cute.AddressSpace.smem, assumed_align=128),
            swz_c, dtype=c_dtype),
        cute.make_layout((tile_size_S, tile_size_N),
                         stride=(tile_size_N, 1)),
    )

    acc_out = cute.make_fragment_like(acc, c_dtype)
    for ci in cutlass.range_constexpr(cute.size(acc)):
        val = acc[ci]
        if activation == 1:
            val = val if val >= Float32(0.0) else Float32(0.0)
        elif activation == 2:
            neg_val = Float32(0.0) - val
            exp_neg = cute.math.exp(neg_val, fastmath=True)
            val = val / (Float32(1.0) + exp_neg)
        acc_out[ci] = val.to(c_dtype)

    r2s_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), c_dtype)
    r2s_copy = cute.make_tiled_copy_C(r2s_atom, tiled_mma)
    r2s_thr = r2s_copy.get_slice(tidx)
    tCrC = r2s_thr.retile(acc_out)
    tCsC = r2s_thr.partition_D(sC)
    cute.copy(r2s_copy, tCrC, tCsC)


@cute.jit
def _gemm_compute_unscaled_core(page_ptr, tidx,
                                a_dtype, b_dtype, c_dtype,
                                num_mma_warps, num_mma_threads,
                                tile_size_S, tile_size_N, tile_K,
                                buf_stride, b_offset, mbar_offset,
                                swz_B_ab, swz_B_c,
                                activation, num_k_blocks_i32):
    """Shared unscaled GEMM compute core for forward decode-style paths."""
    mma_op = cute.nvgpu.warp.MmaF16BF16Op(
        a_dtype, Float32, (16, 8, 16))
    tiled_mma = cute.make_tiled_mma(
        mma_op,
        cute.make_layout((num_mma_warps, 1, 1)),
        permutation_mnk=(num_mma_warps * 16, 16, 16),
    )
    thr_mma = tiled_mma.get_slice(tidx)

    swz = cute.make_swizzle(swz_B_ab, 4, 3)

    smem_copy_atom_A = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            transpose=False, num_matrices=4), a_dtype)
    smem_tiled_copy_A = cute.make_tiled_copy_A(
        smem_copy_atom_A, tiled_mma)
    smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)

    smem_copy_atom_B = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            transpose=False, num_matrices=4), b_dtype)
    smem_tiled_copy_B = cute.make_tiled_copy_B(
        smem_copy_atom_B, tiled_mma)
    smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

    sA_0 = cute.make_tensor(
        cute.recast_ptr(
            cute.make_ptr(a_dtype, page_ptr,
                          cute.AddressSpace.smem, assumed_align=128),
            swz, dtype=a_dtype),
        cute.make_layout((tile_size_S, tile_K),
                         stride=(tile_K, 1)),
    )
    sB_0 = cute.make_tensor(
        cute.recast_ptr(
            cute.make_ptr(b_dtype,
                          page_ptr + Int32(b_offset),
                          cute.AddressSpace.smem, assumed_align=128),
            swz, dtype=b_dtype),
        cute.make_layout((tile_size_N, tile_K),
                         stride=(tile_K, 1)),
    )
    sA_1 = cute.make_tensor(
        cute.recast_ptr(
            cute.make_ptr(a_dtype,
                          page_ptr + Int32(buf_stride),
                          cute.AddressSpace.smem, assumed_align=128),
            swz, dtype=a_dtype),
        cute.make_layout((tile_size_S, tile_K),
                         stride=(tile_K, 1)),
    )
    sB_1 = cute.make_tensor(
        cute.recast_ptr(
            cute.make_ptr(b_dtype,
                          page_ptr + Int32(buf_stride + b_offset),
                          cute.AddressSpace.smem, assumed_align=128),
            swz, dtype=b_dtype),
        cute.make_layout((tile_size_N, tile_K),
                         stride=(tile_K, 1)),
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

    _bf_0 = page_ptr + Int32(mbar_offset)
    _bf_1 = page_ptr + Int32(mbar_offset + 8)
    _kr_0 = page_ptr + Int32(mbar_offset + 16)
    _kr_1 = page_ptr + Int32(mbar_offset + 24)

    acc = cute.make_fragment(
        tiled_mma.partition_shape_C((tile_size_S, tile_size_N)),
        Float32,
    )
    acc.fill(0.0)

    _kr_phase_0 = Int32(0)
    _kr_phase_1 = Int32(0)
    k_idx = Int32(0)
    while k_idx < num_k_blocks_i32:
        if k_idx >= Int32(2):
            if k_idx % Int32(2) == Int32(0):
                mbarrier_wait(_kr_0, _kr_phase_0)
                _kr_phase_0 = _kr_phase_0 ^ Int32(1)
            if k_idx % Int32(2) == Int32(1):
                mbarrier_wait(_kr_1, _kr_phase_1)
                _kr_phase_1 = _kr_phase_1 ^ Int32(1)

        if k_idx % Int32(2) == Int32(0):
            for k_block in cutlass.range_constexpr(tile_K // 16):
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
            for k_block in cutlass.range_constexpr(tile_K // 16):
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

    _gemm_epilogue_store_helper(
        page_ptr, tidx, tiled_mma, acc,
        _bf_0, _bf_1, _kr_0, _kr_1,
        num_mma_threads, swz_B_c, c_dtype,
        tile_size_S, tile_size_N, activation,
    )


class GemmOp(Op):
    """GEMM operation for the megakernel framework.

    Computes C[B,S,N] = A[B,S,K] @ B_w[N,K]^T using tensor core MMA.
    B_w must be in (N, K) layout with K contiguous.

    Tile dimensions: (B, S, N) with tile_B=1 always.
    B and S are shared with downstream ops (RoPE, GDN) for tile-level barriers.

    K is handled via double-buffered TMA pipelining: the DMA warp TMA-loads
    the first 2 K-blocks, then an elected MMA thread issues TMA loads for
    K-blocks 2+ using compute-local mbarriers.
    """

    reads = {
        "a": (None, ("B", "S", "K")),
        "a_scale": (None, ("B", "S", "K")),  # element-wise A scaling (for activation backward)
        "b": (None, ("N", "K")),
    }
    writes = {"c": (None, ("B", "S", "N"))}
    tile = ("B", "S", "N")
    dynamic_dims = ("B", "S")

    tma_loads = {"a", "a_scale", "b"}
    tma_stores = {"c"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        """Custom TMA tile shapes for inner K-block sub-tiling.

        A and C are 3D (B, S, K/N) with tile_B=1.
        B (weight) stays 2D (N, K).
        """
        tile_K = static_dims.get("tile_K", 32)
        if tensor_name in ("a", "a_scale"):
            return (1, tile_sizes["S"], tile_K)
        elif tensor_name == "b":
            return (tile_sizes["N"], tile_K)
        else:  # "c"
            return (1, tile_sizes["S"], tile_sizes["N"])

    def __init__(self, **config):
        super().__init__(**config)

        if self.a_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            raise ValueError(f"GemmOp requires fp16 or bf16 input, got {self.a_dtype}")

        self.tile_K = getattr(self, 'tile_K', 32)
        self.page_size = getattr(self, 'page_size', DEFAULT_PAGE_SIZE)
        self.activation = getattr(self, 'activation', 0)  # 0=none, 1=relu, 2=silu
        self.has_a_scale = getattr(self, 'has_a_scale', 0)  # 1=scale A by a_scale

        self.a_tile_bytes = self.tile_size_S * self.tile_K * self.elem_bytes
        self.b_tile_bytes = self.tile_size_N * self.tile_K * self.elem_bytes
        self.c_tile_bytes = self.tile_size_S * self.tile_size_N * self.elem_bytes

        # Swizzle for C epilogue (R→S) to avoid smem bank conflicts.
        if self.tile_size_N % 64 == 0 and self.tile_size_N >= 64:
            self.swz_B_c = 3   # SW128
        elif self.tile_size_N % 32 == 0:
            self.swz_B_c = 2   # SW64
        else:
            self.swz_B_c = 1   # SW32
        # Per-buffer layout: [A | (a_scale) | B]
        # a_scale only allocated when has_a_scale=1 (compile-time constant)
        if self.has_a_scale:
            self.a_scale_tile_bytes = self.a_tile_bytes
            self.a_scale_offset = self.a_tile_bytes
            self.b_offset = self.a_tile_bytes + self.a_scale_tile_bytes
        else:
            self.a_scale_tile_bytes = 0
            self.a_scale_offset = 0
            self.b_offset = self.a_tile_bytes
        self.buf_stride = self.a_tile_bytes + self.a_scale_tile_bytes + self.b_tile_bytes

        assert self.tile_K >= 16 and self.tile_K % 16 == 0, (
            f"GemmOp: tile_K={self.tile_K} must be >= 16 and a multiple of 16."
        )

        self.num_k_blocks = (self.K + self.tile_K - 1) // self.tile_K
        self.num_mma_warps = self.threads_per_row // 32
        self.num_mma_threads = self.num_mma_warps * 32

        # Number of K-blocks loaded by TMA in load(iter=0): always 2 (or 1 if only 1 K-block)
        self.tma_k_blocks = min(2, self.num_k_blocks)

        # inner_iters: load(iter=0) loads first 2 K-blocks, store warp
        # calls load(iter=1..inner_iters-1) for remaining K-blocks.
        self.inner_iters = max(1, self.num_k_blocks - 1)

        # Swizzle parameters for bank-conflict-free LdMatrix reads.
        if self.tile_K % 64 == 0 and self.tile_K >= 64:
            self.swz_B_ab = 3   # SW128
        elif self.tile_K % 32 == 0:
            self.swz_B_ab = 2   # SW64
        else:
            self.swz_B_ab = 1   # SW32

        # 4 op-managed mbarriers after double-buffer data (32 bytes):
        #   buf_free[0,1]:     compute → store warp (buffer read, safe to overwrite)
        #   kblock_ready[0,1]: TMA hw → compute (new K-block data arrived)
        self.mbar_offset = 2 * self.buf_stride
        self.ab_tma_bytes = self.buf_stride  # A + (a_scale) + B per buffer
        self.mbar_bytes = 32  # 4 x 8 bytes

        # Always 2-stage double buffer + mbarrier space
        ab_bytes = 2 * self.buf_stride
        total_smem = max(ab_bytes + self.mbar_bytes, self.c_tile_bytes)
        assert total_smem <= self.page_size, (
            f"GemmOp: smem {total_smem}B exceeds page_size ({self.page_size}B). "
            f"tile_S={self.tile_size_S}, tile_N={self.tile_size_N}, "
            f"tile_K={self.tile_K}"
        )

        # The outlined unscaled compute path is useful for decode-style kernels
        # where S tiles are very small and compile scaling matters more than the
        # extra device-call boundary. For larger standalone GEMMs it is a pure
        # runtime cost, so keep the original direct compute path there.
        if not self.has_a_scale and self.tile_size_S <= 16:
            self.compute = self.compute_unscaled

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape,
                                tile_sizes, static_dims):
        """Return swizzled smem layout code for TMA descriptor creation.

        A and C are 3D (B=1 trivial outer dim), B (weight) is 2D.
        Swizzle operates on the inner 2D part; B=1 adds a trivial mode.
        """
        tile_K = static_dims.get("tile_K", 32)

        def _layout_str(swz_B, dims):
            """Build composed layout string for 2D or 3D dims."""
            if len(dims) == 2:
                d0, d1 = dims
                return (
                    f"cute.make_composed_layout("
                    f"cute.make_swizzle({swz_B}, 4, 3), 0, "
                    f"cute.make_layout(({d0}, {d1}), stride=(1, {d0})))")
            else:  # 3D: (inner0, inner1, 1) — B=1 trivial outer
                d0, d1, d2 = dims
                return (
                    f"cute.make_composed_layout("
                    f"cute.make_swizzle({swz_B}, 4, 3), 0, "
                    f"cute.make_layout(({d0}, {d1}, {d2}), "
                    f"stride=(1, {d0}, {d0 * d1})))")

        if tensor_name == "c":
            tile_N = tile_sizes.get("N", 64)
            if tile_N % 64 == 0 and tile_N >= 64:
                B_c = 3   # SW128
            elif tile_N % 32 == 0:
                B_c = 2   # SW64
            else:
                B_c = 1   # SW32
            return _layout_str(B_c, tma_tile_shape)

        if tensor_name not in ("a", "a_scale", "b"):
            return None

        if tile_K % 64 == 0 and tile_K >= 64:
            B = 3   # SW128
        elif tile_K % 32 == 0:
            B = 2   # SW64
        else:
            B = 1   # SW32
        return _layout_str(B, tma_tile_shape)

    # =========================================================================
    # Forward Load: TMA G->S (called by load warp and store warp)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_N,
             a_tma, a_tma_gmem, a_scale_tma, a_scale_tma_gmem,
             b_tma, b_tma_gmem,
             work_mbar, inner_iter_idx):
        """TMA load of A and B K-blocks.

        A/a_scale are 3D (B, S, K) — TMA coords include tile_B.
        B (weight) is 2D (N, K) — TMA coords use tile_N only.

        iter 0 (load warp):  Init mbarriers, TMA K0+K1 → buf0+buf1,
                              signal work_mbar.
        iter 1+ (store warp): Wait buf_free[buf], TMA one K-block → buf,
                              signal kblock_ready[buf].
        """
        swz = cute.make_swizzle(self.swz_B_ab, 4, 3)

        # Mbarrier pointers (4 x 8B after double-buffer data)
        _bf_0 = page_ptr + Int32(self.mbar_offset)
        _bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        _kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        _kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        if inner_iter_idx == Int32(0):
            # === ITER 0 (load warp): init mbarriers + TMA first 2 K-blocks ===
            with cute.arch.elect_one():
                mbarrier_init(_bf_0, Int32(self.num_mma_warps))
                mbarrier_init(_bf_1, Int32(self.num_mma_warps))
                mbarrier_init(_kr_0, Int32(1))
                mbarrier_init(_kr_1, Int32(1))
            mbarrier_init_fence()

            mbar_ptr = cute.make_ptr(
                cutlass.Int64, work_mbar, cute.AddressSpace.smem)
            nbytes = Int32(self.tma_k_blocks * self.ab_tma_bytes)
            with cute.arch.elect_one():
                mbarrier_arrive_expect_tx(work_mbar, nbytes)

            for _k in cutlass.range_constexpr(self.tma_k_blocks):
                _buf_base = page_ptr + Int32(_k * self.buf_stride)

                # A: 3D TMA (K, S, B) in CuTe convention
                sA_ptr = cute.recast_ptr(
                    cute.make_ptr(self.a_dtype, _buf_base,
                                  cute.AddressSpace.smem),
                    swz, dtype=self.a_dtype)
                sA = cute.make_tensor(
                    sA_ptr,
                    cute.make_layout((self.tile_K, self.tile_size_S, 1),
                                     stride=(1, self.tile_K,
                                             self.tile_K * self.tile_size_S)))
                gA = cute.local_tile(
                    a_tma_gmem, (self.tile_K, self.tile_size_S, 1),
                    (None, None, None))
                tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                    a_tma, Int32(0), cute.make_layout(1),
                    cute.group_modes(sA, 0, 3),
                    cute.group_modes(gA, 0, 3))

                # a_scale: 3D TMA (same layout as A)
                sScale_ptr = cute.recast_ptr(
                    cute.make_ptr(self.a_scale_dtype,
                                  _buf_base + Int32(self.a_scale_offset),
                                  cute.AddressSpace.smem),
                    swz, dtype=self.a_scale_dtype)
                sScale = cute.make_tensor(
                    sScale_ptr,
                    cute.make_layout((self.tile_K, self.tile_size_S, 1),
                                     stride=(1, self.tile_K,
                                             self.tile_K * self.tile_size_S)))
                gScale = cute.local_tile(
                    a_scale_tma_gmem, (self.tile_K, self.tile_size_S, 1),
                    (None, None, None))
                tScaleS, tScaleG = cute.nvgpu.cpasync.tma_partition(
                    a_scale_tma, Int32(0), cute.make_layout(1),
                    cute.group_modes(sScale, 0, 3),
                    cute.group_modes(gScale, 0, 3))

                # B (weight): 2D TMA (K, N) — unchanged
                sB_ptr = cute.recast_ptr(
                    cute.make_ptr(self.b_dtype,
                                  _buf_base + Int32(self.b_offset),
                                  cute.AddressSpace.smem),
                    swz, dtype=self.b_dtype)
                sB = cute.make_tensor(
                    sB_ptr,
                    cute.make_layout((self.tile_K, self.tile_size_N),
                                     stride=(1, self.tile_K)))
                gB = cute.local_tile(
                    b_tma_gmem, (self.tile_K, self.tile_size_N), (None, None))
                tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                    b_tma, Int32(0), cute.make_layout(1),
                    cute.group_modes(sB, 0, 2),
                    cute.group_modes(gB, 0, 2))

                # A: (None, k_block, S_tile, B_tile)
                cute.copy(a_tma, tAgA[(None, Int32(_k), tile_S, tile_B)],
                          tAsA, tma_bar_ptr=mbar_ptr)
                if self.has_a_scale:
                    cute.copy(a_scale_tma,
                              tScaleG[(None, Int32(_k), tile_S, tile_B)],
                              tScaleS, tma_bar_ptr=mbar_ptr)
                # B: (None, k_block, N_tile)
                cute.copy(b_tma, tBgB[(None, Int32(_k), tile_N)], tBsB,
                          tma_bar_ptr=mbar_ptr)

        if inner_iter_idx > Int32(0):
            # === ITER 1+ (store warp): TMA one K-block into freed buffer ===
            _k_block = inner_iter_idx + Int32(1)  # K-block index
            _buf_idx = _k_block % Int32(2)

            # Wait for compute to finish reading this buffer
            _bf_phase = ((inner_iter_idx - Int32(1)) // Int32(2)) % Int32(2)
            if _buf_idx == Int32(0):
                mbarrier_wait(_bf_0, _bf_phase)
            if _buf_idx == Int32(1):
                mbarrier_wait(_bf_1, _bf_phase)

            # Signal kblock_ready with expected TMA bytes
            _buf_base = _buf_idx * Int32(self.buf_stride) + page_ptr
            _kr_ptr = cute.make_ptr(
                cutlass.Int64, _kr_0, cute.AddressSpace.smem)
            if _buf_idx == Int32(0):
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(
                        _kr_0, Int32(self.ab_tma_bytes))
            if _buf_idx == Int32(1):
                _kr_ptr = cute.make_ptr(
                    cutlass.Int64, _kr_1, cute.AddressSpace.smem)
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(
                        _kr_1, Int32(self.ab_tma_bytes))

            # A: 3D TMA
            sA_ptr = cute.recast_ptr(
                cute.make_ptr(self.a_dtype, _buf_base,
                              cute.AddressSpace.smem),
                swz, dtype=self.a_dtype)
            sA = cute.make_tensor(
                sA_ptr,
                cute.make_layout((self.tile_K, self.tile_size_S, 1),
                                 stride=(1, self.tile_K,
                                         self.tile_K * self.tile_size_S)))
            gA = cute.local_tile(
                a_tma_gmem, (self.tile_K, self.tile_size_S, 1),
                (None, None, None))
            tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                a_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sA, 0, 3),
                cute.group_modes(gA, 0, 3))

            # a_scale: 3D TMA
            sScale_ptr = cute.recast_ptr(
                cute.make_ptr(self.a_scale_dtype,
                              _buf_base + Int32(self.a_scale_offset),
                              cute.AddressSpace.smem),
                swz, dtype=self.a_scale_dtype)
            sScale = cute.make_tensor(
                sScale_ptr,
                cute.make_layout((self.tile_K, self.tile_size_S, 1),
                                 stride=(1, self.tile_K,
                                         self.tile_K * self.tile_size_S)))
            gScale = cute.local_tile(
                a_scale_tma_gmem, (self.tile_K, self.tile_size_S, 1),
                (None, None, None))
            tScaleS, tScaleG = cute.nvgpu.cpasync.tma_partition(
                a_scale_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sScale, 0, 3),
                cute.group_modes(gScale, 0, 3))

            # B (weight): 2D TMA
            sB_ptr = cute.recast_ptr(
                cute.make_ptr(self.b_dtype,
                              _buf_base + Int32(self.b_offset),
                              cute.AddressSpace.smem),
                swz, dtype=self.b_dtype)
            sB = cute.make_tensor(
                sB_ptr,
                cute.make_layout((self.tile_K, self.tile_size_N),
                                 stride=(1, self.tile_K)))
            gB = cute.local_tile(
                b_tma_gmem, (self.tile_K, self.tile_size_N), (None, None))
            tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                b_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB, 0, 2))

            cute.copy(a_tma, tAgA[(None, _k_block, tile_S, tile_B)],
                      tAsA, tma_bar_ptr=_kr_ptr)
            if self.has_a_scale:
                cute.copy(a_scale_tma,
                          tScaleG[(None, _k_block, tile_S, tile_B)],
                          tScaleS, tma_bar_ptr=_kr_ptr)
            cute.copy(b_tma, tBgB[(None, _k_block, tile_N)], tBsB,
                      tma_bar_ptr=_kr_ptr)

    # =========================================================================
    # Forward Compute: Pure MMA (no TMA — store warp handles K-block loads)
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr):
        """GEMM compute with double-buffered K-block processing.

        K-blocks 0-1: TMA-loaded by load warp (in buf 0 and buf 1).
        K-blocks 2+:  TMA-loaded by store warp via load(iter 1+).

        Compute signals buf_free[k%2] after reading each K-block so the
        store warp can overwrite the buffer. For K-blocks 2+, compute
        waits on kblock_ready[k%2] before reading.

        After all K-blocks: epilogue converts fp32 acc to output dtype in smem.
        """
        tidx = cute.arch.thread_idx()[0]

        # --- Build tiled MMA ---
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            self.a_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.num_mma_warps, 1, 1)),
            permutation_mnk=(self.num_mma_warps * 16, 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        # --- LdMatrix tiled copies (warp-cooperative smem reads) ---
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

        # --- Buffer 0 and 1 smem tensors (swizzled for LdMatrix) ---
        sA_0 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.a_dtype, page_ptr,
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.a_dtype),
            cute.make_layout((self.tile_size_S, self.tile_K),
                             stride=(self.tile_K, 1)),
        )
        # a_scale smem tensors (always created; a_scale_offset=0 when
        # has_a_scale=0 so it harmlessly overlaps with A)
        sScale_0 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.a_scale_dtype,
                              page_ptr + Int32(self.a_scale_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.a_scale_dtype),
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
        sScale_1 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.a_scale_dtype,
                              page_ptr + Int32(self.buf_stride + self.a_scale_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.a_scale_dtype),
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

        # MMA partitions and register fragments (shapes shared across buffers)
        tCsA = thr_mma.partition_A(sA_0)
        tCsB = thr_mma.partition_B(sB_0)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrA_ld = smem_thr_copy_A.retile(tCrA)
        tCrB_ld = smem_thr_copy_B.retile(tCrB)

        # a_scale: same copy atom/partitioning as A (always created)
        smem_tiled_copy_Scale = smem_tiled_copy_A
        smem_thr_copy_Scale = smem_tiled_copy_Scale.get_slice(tidx)
        tCsScale = thr_mma.partition_A(sScale_0)
        tCrScale = tiled_mma.make_fragment_A(tCsScale)
        tCrScale_ld = smem_thr_copy_Scale.retile(tCrScale)

        # Per-buffer LdMatrix smem partitions
        tAsA_ld_0 = smem_thr_copy_A.partition_S(sA_0)
        tBsB_ld_0 = smem_thr_copy_B.partition_S(sB_0)
        tAsA_ld_1 = smem_thr_copy_A.partition_S(sA_1)
        tBsB_ld_1 = smem_thr_copy_B.partition_S(sB_1)
        tScaleS_ld_0 = smem_thr_copy_Scale.partition_S(sScale_0)
        tScaleS_ld_1 = smem_thr_copy_Scale.partition_S(sScale_1)

        # Op-managed mbarriers (initialized by load(iter=0))
        _bf_0 = page_ptr + Int32(self.mbar_offset)       # buf_free[0]
        _bf_1 = page_ptr + Int32(self.mbar_offset + 8)   # buf_free[1]
        _kr_0 = page_ptr + Int32(self.mbar_offset + 16)  # kblock_ready[0]
        _kr_1 = page_ptr + Int32(self.mbar_offset + 24)  # kblock_ready[1]

        # --- Init fp32 accumulator ---
        acc = cute.make_fragment(
            tiled_mma.partition_shape_C((self.tile_size_S, self.tile_size_N)),
            Float32,
        )
        acc.fill(0.0)

        # =====================================================================
        # Process all K-blocks through one dynamic loop.
        # K-blocks 0-1 were loaded by the load warp; K-blocks 2+ wait on the
        # store warp's compute-local TMA pipeline.
        # =====================================================================
        _kr_phase_0 = Int32(0)
        _kr_phase_1 = Int32(0)
        k_idx = Int32(0)
        if self.has_a_scale:
            while k_idx < Int32(self.num_k_blocks):
                if k_idx >= Int32(2):
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
                        cute.copy(smem_tiled_copy_Scale,
                                  tScaleS_ld_0[None, None, k_block],
                                  tCrScale_ld[None, None, k_block])
                        for si in cutlass.range_constexpr(cute.size(tCrA[None, None, k_block])):
                            tCrA[None, None, k_block][si] = (
                                tCrA[None, None, k_block][si] * tCrScale[None, None, k_block][si])
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
                        cute.copy(smem_tiled_copy_Scale,
                                  tScaleS_ld_1[None, None, k_block],
                                  tCrScale_ld[None, None, k_block])
                        for si in cutlass.range_constexpr(cute.size(tCrA[None, None, k_block])):
                            tCrA[None, None, k_block][si] = (
                                tCrA[None, None, k_block][si] * tCrScale[None, None, k_block][si])
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
        else:
            while k_idx < Int32(self.num_k_blocks):
                if k_idx >= Int32(2):
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

        _gemm_epilogue_store_helper(
            page_ptr, tidx, tiled_mma, acc,
            _bf_0, _bf_1, _kr_0, _kr_1,
            self.num_mma_threads, self.swz_B_c, self.c_dtype,
            self.tile_size_S, self.tile_size_N, self.activation,
        )

    @cute.jit
    def compute_unscaled(self, page_ptr, tile_B, tile_S, tile_N):
        """GEMM compute path specialized for the common unscaled forward case."""
        _gemm_compute_unscaled_core(
            page_ptr,
            cute.arch.thread_idx()[0],
            self.a_dtype, self.b_dtype, self.c_dtype,
            self.num_mma_warps, self.num_mma_threads,
            self.tile_size_S, self.tile_size_N, self.tile_K,
            self.buf_stride, self.b_offset, self.mbar_offset,
            self.swz_B_ab, self.swz_B_c,
            self.activation, Int32(self.num_k_blocks),
        )

    # =========================================================================
    # Forward Store (S->G): Regular TMA store of C
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_B, tile_S, tile_N, c_tma, c_tma_gmem):
        """TMA store of C from swizzled shared to global memory.

        C is 3D (B, S, N) — TMA coord includes tile_B.
        """
        swz_c = cute.make_swizzle(self.swz_B_c, 4, 3)
        sC = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.c_dtype, page_ptr,
                              cute.AddressSpace.smem, assumed_align=128),
                swz_c, dtype=self.c_dtype),
            cute.make_layout((self.tile_size_N, self.tile_size_S, 1),
                             stride=(1, self.tile_size_N,
                                     self.tile_size_N * self.tile_size_S)),
        )

        gC = cute.local_tile(
            c_tma_gmem, (self.tile_size_N, self.tile_size_S, 1),
            (None, None, None),
        )
        tCsC, tCgC = cute.nvgpu.cpasync.tma_partition(
            c_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sC, 0, 3),
            cute.group_modes(gC, 0, 3),
        )

        with cute.arch.elect_one():
            cute.copy(c_tma, tCsC, tCgC[(None, tile_N, tile_S, tile_B)])

    # =========================================================================
    # Communicate (TMA S->G to peer GPU)
    # =========================================================================

    @cute.jit
    def communicate(self, page_ptr, tile_B, tile_S, tile_N,
                    c_p0_tma, c_p0_tma_gmem):
        """Send C tile to peer GPU 0 via TMA S2G.

        The TMA atom determines the semantics: regular copy for broadcast
        (column-parallel), atomic add for reduce (row-parallel all-reduce).
        """
        swz_c = cute.make_swizzle(self.swz_B_c, 4, 3)
        sC = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.c_dtype, page_ptr,
                              cute.AddressSpace.smem, assumed_align=128),
                swz_c, dtype=self.c_dtype),
            cute.make_layout((self.tile_size_N, self.tile_size_S, 1),
                             stride=(1, self.tile_size_N,
                                     self.tile_size_N * self.tile_size_S)),
        )

        gC = cute.local_tile(
            c_p0_tma_gmem, (self.tile_size_N, self.tile_size_S, 1),
            (None, None, None),
        )
        tCsC, tCgC = cute.nvgpu.cpasync.tma_partition(
            c_p0_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sC, 0, 3),
            cute.group_modes(gC, 0, 3),
        )

        with cute.arch.elect_one():
            cute.copy(c_p0_tma, tCsC, tCgC[(None, tile_N, tile_S, tile_B)])

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def _auto_tiles(cls, page_size, elem_bytes=2, has_a_scale=False):
        """Compute best (tile_S, tile_N, tile_K) that fit in page_size.

        Constraints:
          - 2 * (a_factor*tile_S + tile_N) * tile_K * elem + mbar <= page_size  (double buf + mbarriers)
          - tile_S * tile_N * elem <= page_size  (C epilogue)
          - tile_K must be a multiple of 16

        Prefers 128×64 (best TMA/compute balance) over 128×128.
        After selecting spatial tiles, maximizes tile_K within budget.
        """
        a_factor = 2 if has_a_scale else 1
        mbar_bytes = 32  # 4 op-managed mbarriers × 8 bytes
        # Prefer 128×64: best TMA/compute balance (NCU-verified).
        for tile_S, tile_N in [(128, 64), (128, 128), (64, 64), (64, 32), (32, 32)]:
            tile_K = 32
            ab = 2 * (a_factor * tile_S + tile_N) * tile_K * elem_bytes + mbar_bytes
            c = tile_S * tile_N * elem_bytes
            if ab <= page_size and c <= page_size:
                # Try larger tile_K when page headroom allows.
                # Must be power-of-2 for valid TMA box sizes.
                for try_K in [64]:
                    ab_k = 2 * (a_factor * tile_S + tile_N) * try_K * elem_bytes + mbar_bytes
                    if ab_k <= page_size:
                        tile_K = try_K
                        break
                return tile_S, tile_N, tile_K
        return 32, 32, 16

    @classmethod
    def _shape_aware_auto_tiles(
        cls,
        page_size,
        *,
        input_k,
        output_n,
        elem_bytes=2,
        has_a_scale=False,
    ):
        """Compute tiles with a small shape-aware preference list.

        The default spatial preference is good for balanced GEMMs, but Qwen
        projection shapes are skewed:
        - expanders like Q projection benefit from a fatter N tile
        - reducers like O/down projection benefit from narrower N and larger K
        """
        a_factor = 2 if has_a_scale else 1
        mbar_bytes = 32

        preferred = []
        if output_n >= 2 * input_k:
            preferred.extend([(64, 64, 64), (128, 64, 32)])
        elif input_k >= 2 * output_n:
            # Moderately skinny reducers like Qwen O-proj (K=2048 -> N=1024)
            # behave differently standalone vs fused. For the fused one-page
            # path at larger page sizes, Qwen O-proj benefits from the narrower
            # N tile again, while very skinny reducers continue to prefer it.
            if page_size <= 32 * 1024:
                preferred.extend([(64, 32, 64), (32, 64, 64), (64, 64, 64)])
            elif input_k <= 2048:
                if page_size >= 96 * 1024:
                    preferred.extend([(64, 32, 64), (32, 64, 64), (64, 64, 64)])
                else:
                    preferred.extend([(32, 64, 64), (64, 32, 64), (64, 64, 64)])
            else:
                preferred.extend([(64, 32, 64), (64, 64, 64)])

        preferred.extend([(128, 64, 32), (128, 128, 32), (64, 64, 64), (64, 64, 32), (64, 32, 64), (32, 32, 16)])

        seen = set()
        for tile_S, tile_N, tile_K in preferred:
            key = (tile_S, tile_N, tile_K)
            if key in seen:
                continue
            seen.add(key)
            buf_stride = (a_factor * tile_S + tile_N) * tile_K * elem_bytes
            ab_total = 2 * buf_stride + mbar_bytes
            c_total = tile_S * tile_N * elem_bytes
            if max(ab_total, c_total) <= page_size:
                return tile_S, tile_N, tile_K

        return cls._auto_tiles(page_size, elem_bytes=elem_bytes, has_a_scale=has_a_scale)

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE,
                         activation=None, has_a_scale=0, **tensors):
        """Schedule forward GEMM.

        Tensors must be 3D: A (B, S, K), C (B, S, N). B_w stays 2D (N, K).

        Accepts tile_sizes with S, N, K keys. K is the inner K-block size
        (not a framework tile dimension). Auto-computed from page_size when
        not specified.

        When has_a_scale=1, the a_scale tensor is loaded alongside A and
        multiplied element-wise before each MMA (used for fused activation
        backward). When has_a_scale=0 (default), a_scale defaults to a
        (same pointer, TMA still loads it but compute skips the multiply).
        """
        # Default a_scale = a (no-op scaling, uniform interface)
        if "a_scale" not in tensors:
            tensors["a_scale"] = tensors["a"]

        ts = dict(tile_sizes or {})
        ts.setdefault("B", 1)
        if "S" not in ts or "N" not in ts:
            a = tensors.get('a')
            b = tensors.get('b')
            elem_bytes = a.element_size() if a is not None else 2
            if a is not None and b is not None:
                auto_S, auto_N, auto_K = cls._shape_aware_auto_tiles(
                    page_size,
                    input_k=a.shape[-1],
                    output_n=b.shape[0],
                    elem_bytes=elem_bytes,
                    has_a_scale=bool(has_a_scale),
                )
            else:
                auto_S, auto_N, auto_K = cls._auto_tiles(
                    page_size, elem_bytes, has_a_scale=bool(has_a_scale))
            ts.setdefault("S", auto_S)
            ts.setdefault("N", auto_N)
            ts.setdefault("K", auto_K)
        tile_K = ts.pop("K", 32)

        # Validate tiles fit in page_size; reduce tile_K if necessary.
        a_factor = 2 if has_a_scale else 1
        a = tensors.get('a')
        elem_bytes = a.element_size() if a is not None else 2
        tile_S = ts["S"]
        tile_N = ts["N"]
        mbar_bytes = 32
        while tile_K > 16:
            buf_stride = (a_factor * tile_S + tile_N) * tile_K * elem_bytes
            ab_total = 2 * buf_stride + mbar_bytes
            c_total = tile_S * tile_N * elem_bytes
            if max(ab_total, c_total) <= page_size:
                break
            tile_K //= 2

        scheduled = cls._schedule_single(tile_sizes=ts, **tensors)
        scheduled.static_dims["tile_K"] = tile_K
        scheduled.static_dims["page_size"] = page_size
        scheduled.static_dims["has_a_scale"] = has_a_scale
        if activation is not None:
            from machete.kernels.activation.activation import ACT_MAP
            scheduled.static_dims["activation"] = ACT_MAP[activation]
        return [scheduled]

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for scheduled GemmOps."""
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS
        page_size = max(op.static_dims.get('page_size', DEFAULT_PAGE_SIZE)
                        for op in ops)
        tile_m = max(op.tile_sizes.get('S', 64) for op in ops)
        num_mma_warps = tile_m // 16
        threads_per_block = (num_mma_warps + NUM_DMA_WARPS) * 32
        return MegakernelConfig(threads_per_block=threads_per_block,
                                page_size=page_size)

    @classmethod
    def schedule_backward(cls, tile_sizes=None, activation=None,
                          c=None, pre_act=None, page_size=DEFAULT_PAGE_SIZE,
                          **tensors):
        """Schedule GEMM backward as forward-equivalent ops.

        dA[B,S,K] = dout[B,S,N] @ B[N,K]  (contracts over N)
        dB[N,K] = dout^T[N,B*S] @ A[B*S,K]  (contracts over S, flattened)

        Each gradient is a standard forward GEMM with transposed inputs.
        No output zeroing needed (regular TMA store, not atomic add).

        All tensors must be 3D (B,S,K).

        When activation is set, the activation gradient is fused into the
        GEMM via A-operand scaling: dout is element-wise multiplied by
        act'(pre_act) before the matmul. This is done by TMA-loading
        act_grad alongside A (the dout operand) in the kernel.

        Args:
            activation: 'relu' or 'silu' (must match forward activation).
            c: Forward output (required for ReLU backward: act'(x) = x > 0).
            pre_act: Pre-activation values A@B^T (required for SiLU backward).
        """
        import torch

        dout, a, b = tensors['dout'], tensors['a'], tensors['b']
        da, db = tensors.get('da'), tensors.get('db')

        # Compute activation gradient tensor if needed
        act_grad = None
        if activation is not None:
            from machete.kernels.activation.activation import (
                ACT_MAP, ACT_RELU, ACT_SILU)
            act_id = ACT_MAP[activation]
            if act_id == ACT_RELU:
                assert c is not None, "ReLU backward requires c (forward output)"
                act_grad = (c > 0).to(dout.dtype).contiguous()
            elif act_id == ACT_SILU:
                assert pre_act is not None, (
                    "SiLU backward requires pre_act (pre-activation values)")
                sig = torch.sigmoid(pre_act.float())
                act_grad = (sig * (1.0 + pre_act.float() * (1.0 - sig))
                            ).to(dout.dtype).contiguous()

        use_scale = 1 if act_grad is not None else 0
        ops = []

        if da is not None:
            b_t = b.t().contiguous()
            fwd_kwargs = dict(
                a=dout, b=b_t, c=da, has_a_scale=use_scale,
            )
            if tile_sizes is not None:
                ts = tile_sizes
                fwd_kwargs["tile_sizes"] = {
                    "S": ts.get("S", 64),
                    "N": ts.get("K", 32),
                    "K": ts.get("N", 32),
                }
            # else: let schedule auto-compute via _auto_tiles
            if act_grad is not None:
                fwd_kwargs["a_scale"] = act_grad
            ops.extend(cls.schedule(**fwd_kwargs, page_size=page_size))

        if db is not None:
            # dB[N,K] = dout^T[N,B*S] @ A[B*S,K]  — flatten B*S, transpose
            BS = dout.shape[0] * dout.shape[1]
            dout_flat = dout.reshape(BS, dout.shape[2])
            a_flat = a.reshape(BS, a.shape[2])
            # a operand: (1, N, BS) — 3D for schedule
            dout_t = dout_flat.t().contiguous().unsqueeze(0)
            # b weight: (K, BS) — stays 2D
            a_t = a_flat.t().contiguous()
            fwd_kwargs = dict(
                a=dout_t, b=a_t, c=db, has_a_scale=use_scale,
            )
            if tile_sizes is not None:
                ts = tile_sizes
                fwd_kwargs["tile_sizes"] = {
                    "S": ts.get("N", 32),
                    "N": ts.get("K", 32),
                    "K": ts.get("S", 64),
                }
            # else: let schedule auto-compute via _auto_tiles
            if act_grad is not None:
                act_flat = act_grad.reshape(BS, act_grad.shape[2])
                fwd_kwargs["a_scale"] = act_flat.t().contiguous().unsqueeze(0)
            ops.extend(cls.schedule(**fwd_kwargs, page_size=page_size))

        return ops

    # =========================================================================
    # Tensor Parallelism Factory
    # =========================================================================

    @classmethod
    def schedule_tp(cls, tp_mode='column', **kwargs):
        """Schedule forward GEMM with tensor parallelism.

        Args:
            tp_mode: 'column' for column-parallel (broadcast output shard),
                     'row' for row-parallel (all-reduce via atomic add).
            **kwargs: Forwarded to schedule().

        Returns:
            List of ScheduledOps using the appropriate TP subclass.

        Notes:
            Column-parallel: each GPU has W_shard[N/P, K], computes
                C_shard = A @ W_shard^T, broadcasts to peers.
            Row-parallel: each GPU has W[N, K/P], computes
                C_partial = A_shard @ W^T, atomic-adds to all outputs.
                All output buffers must be zeroed before kernel launch.
        """
        if tp_mode == 'column':
            return GemmColumnParallelOp.schedule(**kwargs)
        elif tp_mode == 'row':
            return GemmRowParallelOp.schedule(**kwargs)
        else:
            raise ValueError(f"Unknown tp_mode: {tp_mode!r}. Use 'column' or 'row'.")


class GemmColumnParallelOp(GemmOp):
    """GemmOp with TMA S2G broadcast to peers (column-parallel TP).

    Each GPU computes C_shard[M, N/P] = A @ W_shard[N/P, K]^T and
    broadcasts the result to peer GPU buffers via TMA S2G copy.
    """

    peer_stores = {"c"}


class GemmRowParallelOp(GemmOp):
    """GemmOp with TMA S2G atomic add for row-parallel TP (all-reduce).

    Each GPU computes C_partial[M, N] = A_shard[M, K/P] @ W[N, K/P]^T
    and atomic-adds the result to both the local and all peer output buffers.
    After kernel completion, each buffer contains C_full = sum(C_partial_i).

    IMPORTANT: All output buffers must be zeroed before kernel launch.

    TMA reduce S2G (CopyReduceBulkTensorTileS2GOp) is warp-collective:
    all threads in the warp must execute the instruction, producing exactly
    one atomic add per warp. Therefore store() and communicate() must NOT
    use elect_one() — doing so causes non-elected threads to skip the
    warp-collective instruction, resulting in a hang.
    """

    tma_stores = set()          # Override parent: no regular S2G for c
    tma_reduce_stores = {"c"}   # Local store via atomic add
    peer_reduce_stores = {"c"}  # Peer stores via atomic add

    @cute.jit
    def store(self, page_ptr, tile_B, tile_S, tile_N, c_tma, c_tma_gmem):
        """TMA reduce store of C (atomic add) — warp-collective, no elect_one."""
        swz_c = cute.make_swizzle(self.swz_B_c, 4, 3)
        sC = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.c_dtype, page_ptr,
                              cute.AddressSpace.smem, assumed_align=128),
                swz_c, dtype=self.c_dtype),
            cute.make_layout((self.tile_size_N, self.tile_size_S, 1),
                             stride=(1, self.tile_size_N,
                                     self.tile_size_N * self.tile_size_S)),
        )

        gC = cute.local_tile(
            c_tma_gmem, (self.tile_size_N, self.tile_size_S, 1),
            (None, None, None),
        )
        tCsC, tCgC = cute.nvgpu.cpasync.tma_partition(
            c_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sC, 0, 3),
            cute.group_modes(gC, 0, 3),
        )

        # All warp threads issue TMA (warp-collective). 1 warp = 1 atomic add.
        cute.copy(c_tma, tCsC, tCgC[(None, tile_N, tile_S, tile_B)])

    @cute.jit
    def communicate(self, page_ptr, tile_B, tile_S, tile_N,
                    c_p0_tma, c_p0_tma_gmem):
        """TMA reduce store to peer (atomic add) — warp-collective, no elect_one."""
        swz_c = cute.make_swizzle(self.swz_B_c, 4, 3)
        sC = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.c_dtype, page_ptr,
                              cute.AddressSpace.smem, assumed_align=128),
                swz_c, dtype=self.c_dtype),
            cute.make_layout((self.tile_size_N, self.tile_size_S, 1),
                             stride=(1, self.tile_size_N,
                                     self.tile_size_N * self.tile_size_S)),
        )

        gC = cute.local_tile(
            c_p0_tma_gmem, (self.tile_size_N, self.tile_size_S, 1),
            (None, None, None),
        )
        tCsC, tCgC = cute.nvgpu.cpasync.tma_partition(
            c_p0_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sC, 0, 3),
            cute.group_modes(gC, 0, 3),
        )

        # All warp threads issue TMA (warp-collective). 1 warp = 1 atomic add.
        cute.copy(c_p0_tma, tCsC, tCgC[(None, tile_N, tile_S, tile_B)])
