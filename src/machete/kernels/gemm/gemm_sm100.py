# Copyright (c) 2025, Machete Authors
"""
SM100 (Blackwell) GEMM Op using tcgen05.mma (UMMA) instructions.

Computes C[B, S, N] = A[B, S, K] @ B_w[N, K]^T with fp32 accumulation.

Architecture:
    - tcgen05.mma (UMMA): CTA-level MMA writing to Tensor Memory (TMEM)
    - Warp 0: Issues UMMA instructions during K-loop
    - All MMA warps: Cooperate on TMEM → regs → smem epilogue
    - DMA warps: TMA load A/B (double-buffered K-blocks), TMA store C

Smem page layout (double-buffered K-blocks):
    Phase 0 (load/compute): [A_buf0 | A_buf1 | B_buf0 | B_buf1 | mbarriers]
    Phase 1 (epilogue/store): [C: tile_S × tile_N × 2B]

Usage:
    from machete.kernels.gemm import GemmSm100Op
    from machete.megakernel import Megakernel

    ops = GemmSm100Op.schedule(a=a, b=b_w, c=c, page_size=65536)
    config = GemmSm100Op.kernel_config(ops)
    kernel = Megakernel(ops, config=config)
    kernel.run()
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32
from cutlass.cute.nvgpu import tcgen05
from cutlass.utils import LayoutEnum

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import (
    mbarrier_init, mbarrier_arrive_expect_tx, mbarrier_arrive,
    mbarrier_wait, mbarrier_inval, named_barrier_sync,
)


# SM100 UMMA swizzle parameters for bf16/fp16, K-major layout.
# get_smem_layout_atom_ab(K, bf16, (M,K)) returns:
#   K>=64 → K_SW128 → swizzle(3, 4, 3), contiguous_elems = 64
#   K>=32 → K_SW64  → swizzle(2, 4, 3), contiguous_elems = 32
#   K>=16 → K_SW32  → swizzle(1, 4, 3), contiguous_elems = 16
def _sm100_swizzle_params(tile_K, elem_bits=16):
    """Return (swz_B, num_contiguous_elems) for K-major SM100 smem layout."""
    if tile_K >= 64:
        return 3, 1024 // elem_bits   # SW128, 64 elems for bf16
    elif tile_K >= 32:
        return 2, 512 // elem_bits    # SW64, 32 elems for bf16
    else:
        return 1, 256 // elem_bits    # SW32, 16 elems for bf16


class GemmSm100Op(Op):
    """SM100 GEMM using tcgen05.mma (UMMA) with TMEM accumulators.

    Tensors:
        a: (B, S, K) -- input activation (fp16 or bf16)
        b: (N, K)    -- weight matrix (contiguous K)
        c: (B, S, N) -- output

    Tiling:
        tile_B=1 (per batch), tile_S and tile_N from schedule.
        K handled via double-buffered TMA pipelining (inner_iters).
    """

    reads = {
        "a": (None, ("B", "S", "K")),
        "b": (None, ("N", "K")),
    }
    writes = {"c": (None, ("B", "S", "N"))}
    tile = ("B", "S", "N")
    dynamic_dims = ("B", "S")

    tma_loads = {"a", "b"}
    tma_stores = {"c"}

    # =========================================================================
    # TMA Layout Configuration
    # =========================================================================

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        """TMA tile shapes for A (3D), B (2D), C (3D)."""
        tile_K = static_dims.get("tile_K", 64)
        if tensor_name == "a":
            return (1, tile_sizes["S"], tile_K)
        elif tensor_name == "b":
            return (tile_sizes["N"], tile_K)
        else:  # "c"
            return (1, tile_sizes["S"], tile_sizes["N"])

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        """Swizzled smem layout for SM100 UMMA TMA descriptors.

        A and B use K-major swizzle for tcgen05.mma compatibility.
        C uses row-major swizzle for the epilogue R→S store.
        """
        tile_K = static_dims.get("tile_K", 64)

        if tensor_name == "a":
            # A: the registry has already permuted (B, S, K) into TMA order
            # (K, S, B), matching Quack/CUTLASS's K-major SM100 layout.
            swz_B, n_contig = _sm100_swizzle_params(tile_K)
            dim0, dim1, dim2 = tma_tile_shape  # (tile_K, tile_S, 1)
            return (
                f"cute.make_composed_layout("
                f"cute.make_swizzle({swz_B}, 4, 3), 0, "
                f"cute.make_layout(({dim0}, {dim1}, {dim2}), "
                f"stride=(1, {dim0}, {dim0 * dim1})))"
            )
        elif tensor_name == "b":
            # B: the registry has already permuted (N, K) into TMA order
            # (K, N), with K contiguous.
            swz_B, n_contig = _sm100_swizzle_params(tile_K)
            dim0, dim1 = tma_tile_shape  # (tile_K, tile_N)
            return (
                f"cute.make_composed_layout("
                f"cute.make_swizzle({swz_B}, 4, 3), 0, "
                f"cute.make_layout(({dim0}, {dim1}), "
                f"stride=(1, {dim0})))"
            )
        else:  # "c"
            # C: the registry has already permuted (B, S, N) into TMA order
            # (N, S, B), matching the row-major epilogue tile.
            dim0, dim1, dim2 = tma_tile_shape  # (tile_N, tile_S, 1)
            tile_N = dim0
            if tile_N >= 64:
                swz_B = 3
            elif tile_N >= 32:
                swz_B = 2
            else:
                swz_B = 1
            return (
                f"cute.make_composed_layout("
                f"cute.make_swizzle({swz_B}, 4, 3), 0, "
                f"cute.make_layout(({dim0}, {dim1}, {dim2}), "
                f"stride=(1, {dim0}, {dim0 * dim1})))"
            )

    # =========================================================================
    # Init
    # =========================================================================

    def __init__(self, **config):
        super().__init__(**config)

        if self.a_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
            self.elem_bits = 16
        else:
            raise ValueError(f"GemmSm100Op requires fp16 or bf16, got {self.a_dtype}")

        self.tile_K = getattr(self, 'tile_K', 64)
        self.page_size = getattr(self, 'page_size', DEFAULT_PAGE_SIZE)
        self.activation = getattr(self, 'activation', 0)  # 0=none, 1=relu, 2=silu

        # Validate tile_K for UMMA: must be multiple of mma_inst_k * 4 = 64 for bf16
        self.mma_inst_k = 256 // self.elem_bits  # 16 for bf16
        self.mma_k_tiles = 4  # 4 k-tiles per MMA tiler step
        self.mma_tiler_k = self.mma_inst_k * self.mma_k_tiles  # 64 for bf16
        assert self.tile_K % self.mma_tiler_k == 0, (
            f"GemmSm100Op: tile_K={self.tile_K} must be multiple of mma_tiler_k={self.mma_tiler_k}"
        )

        # Validate tile_S constraints for UMMA
        assert self.tile_size_S in (64, 128, 256), (
            f"GemmSm100Op: tile_size_S={self.tile_size_S} must be 64, 128, or 256 for tcgen05.mma"
        )
        assert self.tile_size_N >= 32 and self.tile_size_N % 32 == 0 and self.tile_size_N <= 256, (
            f"GemmSm100Op: tile_size_N={self.tile_size_N} must be 32-256, step 32"
        )

        self.a_tile_bytes = self.tile_size_S * self.tile_K * self.elem_bytes
        self.b_tile_bytes = self.tile_size_N * self.tile_K * self.elem_bytes
        self.c_tile_bytes = self.tile_size_S * self.tile_size_N * self.elem_bytes

        # Quack-style grouped staging: [A0 | A1 | B0 | B1 | mbarriers(32B)].
        # This matches blackwell_helpers.make_smem_layout_a/b staged layouts.
        self.a_offset = 0
        self.b_offset = 2 * self.a_tile_bytes
        self.mbar_offset = self.b_offset + 2 * self.b_tile_bytes
        self.ab_tma_bytes = self.a_tile_bytes + self.b_tile_bytes
        self.mbar_bytes = 32  # 4 × 8 bytes

        # K-loop: first 2 K-blocks loaded by DMA, rest by inner_iters
        self.num_k_blocks = (self.K + self.tile_K - 1) // self.tile_K
        self.tma_k_blocks = min(2, self.num_k_blocks)
        self.inner_iters = max(1, self.num_k_blocks - 1)

        self.num_mma_warps = self.threads_per_row // 32
        self.num_mma_threads = self.num_mma_warps * 32

        # Swizzle parameters
        self.swz_B_ab, self.n_contig_ab = _sm100_swizzle_params(self.tile_K, self.elem_bits)

        # C epilogue swizzle
        if self.tile_size_N >= 64 and self.tile_size_N % 64 == 0:
            self.swz_B_c = 3
        elif self.tile_size_N >= 32 and self.tile_size_N % 32 == 0:
            self.swz_B_c = 2
        else:
            self.swz_B_c = 1

        # Number of inner k-block iterations within each MMA tiler step
        self.num_k_inner = self.tile_K // self.mma_tiler_k

    # =========================================================================
    # Forward Load (DMA warp: TMA A + B)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_S, tile_N,
             a_tma, a_tma_gmem,
             b_tma, b_tma_gmem,
             work_mbar, inner_iter_idx):
        """TMA load A and B into double-buffered smem pages.

        iter=0 (DMA warp): Init op-managed mbarriers, TMA load first 2 K-blocks.
        iter>0 (Store warp): Wait buf_free, TMA load K-block iter+1.
        """
        # Op-managed mbarrier pointers
        _mbar_base = page_ptr + Int32(self.mbar_offset)
        _kr_0 = _mbar_base                      # kblock_ready[0]
        _kr_1 = _mbar_base + Int32(8)            # kblock_ready[1]
        _bf_0 = _mbar_base + Int32(16)           # buf_free[0]
        _bf_1 = _mbar_base + Int32(24)           # buf_free[1]

        swz = cute.make_swizzle(self.swz_B_ab, 4, 3)

        if inner_iter_idx == Int32(0):
            # --- DMA warp: init + first 2 K-blocks ---
            mbarrier_init(_kr_0, Int32(self.ab_tma_bytes))
            mbarrier_init(_kr_1, Int32(self.ab_tma_bytes))
            mbarrier_init(_bf_0, Int32(self.num_mma_warps))
            mbarrier_init(_bf_1, Int32(self.num_mma_warps))

            for _k in cutlass.range_constexpr(self.tma_k_blocks):
                _a_buf_base = page_ptr + Int32(_k * self.a_tile_bytes)
                _b_buf_base = page_ptr + Int32(self.b_offset + _k * self.b_tile_bytes)
                _mbar = _kr_0 if _k == 0 else _kr_1
                mbar_ptr = cute.make_ptr(cutlass.Int64, _mbar, cute.AddressSpace.smem)

                # TMA load A[tile_B, tile_S, K_block]
                sA = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(self.a_dtype, _a_buf_base,
                                      cute.AddressSpace.smem, assumed_align=128),
                        swz, dtype=self.a_dtype),
                    cute.make_layout((self.tile_K, self.tile_size_S, 1),
                                     stride=(1, self.tile_K, self.tile_K * self.tile_size_S)),
                )
                gA = cute.local_tile(
                    a_tma_gmem, (self.tile_K, self.tile_size_S, 1),
                    (None, None, None),
                )
                tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                    a_tma, Int32(0), cute.make_layout(1),
                    cute.group_modes(sA, 0, 3),
                    cute.group_modes(gA, 0, 3),
                )
                cute.copy(a_tma, tAgA[(None, Int32(_k), tile_S, tile_B)],
                          tAsA, tma_bar_ptr=mbar_ptr)

                # TMA load B[tile_N, K_block]
                sB = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(self.b_dtype, _b_buf_base,
                                      cute.AddressSpace.smem, assumed_align=128),
                        swz, dtype=self.b_dtype),
                    cute.make_layout((self.tile_K, self.tile_size_N),
                                     stride=(1, self.tile_K)),
                )
                gB = cute.local_tile(
                    b_tma_gmem, (self.tile_K, self.tile_size_N),
                    (None, None),
                )
                tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                    b_tma, Int32(0), cute.make_layout(1),
                    cute.group_modes(sB, 0, 2),
                    cute.group_modes(gB, 0, 2),
                )
                cute.copy(b_tma, tBgB[(None, Int32(_k), tile_N)],
                          tBsB, tma_bar_ptr=mbar_ptr)

            # Signal framework that load is done
            mbar_ptr_w = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
            cute.arch.mbarrier_arrive(mbar_ptr_w)

        else:
            # --- Store warp: load K-block (inner_iter_idx + 1) ---
            k_idx = inner_iter_idx + Int32(1)
            if k_idx < Int32(self.num_k_blocks):
                buf_idx = k_idx % Int32(2)
                _a_buf_base = page_ptr + buf_idx * Int32(self.a_tile_bytes)
                _b_buf_base = page_ptr + Int32(self.b_offset) + buf_idx * Int32(self.b_tile_bytes)
                _bf = _bf_0 if buf_idx == Int32(0) else _bf_1
                _kr = _kr_0 if buf_idx == Int32(0) else _kr_1

                # Wait for compute to finish reading this buffer
                mbarrier_wait(_bf, Int32(0))
                # Re-init kblock_ready for new TMA load
                mbarrier_init(_kr, Int32(self.ab_tma_bytes))

                mbar_ptr = cute.make_ptr(cutlass.Int64, _kr, cute.AddressSpace.smem)

                # TMA load A
                sA = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(self.a_dtype, _a_buf_base,
                                      cute.AddressSpace.smem, assumed_align=128),
                        swz, dtype=self.a_dtype),
                    cute.make_layout((self.tile_K, self.tile_size_S, 1),
                                     stride=(1, self.tile_K, self.tile_K * self.tile_size_S)),
                )
                gA = cute.local_tile(
                    a_tma_gmem, (self.tile_K, self.tile_size_S, 1),
                    (None, None, None),
                )
                tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                    a_tma, Int32(0), cute.make_layout(1),
                    cute.group_modes(sA, 0, 3),
                    cute.group_modes(gA, 0, 3),
                )
                cute.copy(a_tma, tAgA[(None, k_idx, tile_S, tile_B)],
                          tAsA, tma_bar_ptr=mbar_ptr)

                # TMA load B
                sB = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(self.b_dtype, _b_buf_base,
                                      cute.AddressSpace.smem, assumed_align=128),
                        swz, dtype=self.b_dtype),
                    cute.make_layout((self.tile_K, self.tile_size_N),
                                     stride=(1, self.tile_K)),
                )
                gB = cute.local_tile(
                    b_tma_gmem, (self.tile_K, self.tile_size_N),
                    (None, None),
                )
                tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                    b_tma, Int32(0), cute.make_layout(1),
                    cute.group_modes(sB, 0, 2),
                    cute.group_modes(gB, 0, 2),
                )
                cute.copy(b_tma, tBgB[(None, k_idx, tile_N)],
                          tBsB, tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Forward Compute (UMMA K-loop + TMEM epilogue)
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_N):
        """UMMA compute with double-buffered K-block processing.

        1. Allocate TMEM for accumulator
        2. K-loop: warp 0 issues tcgen05.mma, waits on kblock_ready mbarriers
        3. Epilogue: all warps cooperate on TMEM → regs → smem
        4. Deallocate TMEM
        """
        tidx = cute.arch.thread_idx()[0]
        warp_idx = tidx // Int32(32)

        import cutlass.utils.blackwell_helpers as bh

        # --- Build UMMA tiled MMA ---
        tiled_mma = bh.make_trivial_tiled_mma(
            ab_dtype=self.a_dtype,
            a_leading_mode=tcgen05.OperandMajorMode.K,
            b_leading_mode=tcgen05.OperandMajorMode.K,
            acc_dtype=Float32,
            cta_group=tcgen05.CtaGroup.ONE,
            mma_tiler_mn=(self.tile_size_S, self.tile_size_N),
        )

        # MMA tiler: (tile_S, tile_N, mma_tiler_k)
        mma_tiler = (self.tile_size_S, self.tile_size_N, self.mma_tiler_k)

        # --- TMEM allocation ---
        # Warp 0 allocates TMEM, all warps sync and retrieve pointer
        tmem_holding_buf = cute.arch.alloc_smem(Int64, 1, alignment=8)
        if warp_idx == Int32(0):
            # Compute accumulator shape and TMEM columns needed
            acc_shape = tiled_mma.partition_shape_C((self.tile_size_S, self.tile_size_N))
            acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
            num_tmem_cols = cutlass.utils.get_num_tmem_alloc_cols(acc_fake)
            cute.arch.alloc_tmem(num_tmem_cols, tmem_holding_buf)
        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
            Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_holding_buf)

        # --- Build smem tensors for A and B (staged) ---
        swz = cute.make_swizzle(self.swz_B_ab, 4, 3)

        # Smem layout for UMMA: use blackwell_helpers
        a_smem_layout = bh.make_smem_layout_a(
            tiled_mma, mma_tiler, self.a_dtype, 2)  # 2 stages
        b_smem_layout = bh.make_smem_layout_b(
            tiled_mma, mma_tiler, self.b_dtype, 2)  # 2 stages

        # Create staged smem tensors using framework page
        sA_staged = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.a_dtype, page_ptr,
                              cute.AddressSpace.smem, assumed_align=128),
                a_smem_layout.inner,
                dtype=self.a_dtype,
            ),
            a_smem_layout.outer,
        )
        sB_staged = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.b_dtype, page_ptr + Int32(self.b_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                b_smem_layout.inner,
                dtype=self.b_dtype,
            ),
            b_smem_layout.outer,
        )

        # MMA partitions: fragment_A/B give smem descriptor tensors
        tCrA = tiled_mma.make_fragment_A(sA_staged)
        tCrB = tiled_mma.make_fragment_B(sB_staged)

        # Build accumulator in TMEM
        acc_shape = tiled_mma.partition_shape_C((self.tile_size_S, self.tile_size_N))
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
        tCtAcc_base = cute.make_tensor(acc_tmem_ptr, acc_fake.layout)
        tCtAcc = tCtAcc_base[None, None, None, 0]

        # --- Op-managed mbarrier pointers ---
        _mbar_base = page_ptr + Int32(self.mbar_offset)
        _kr_0 = _mbar_base
        _kr_1 = _mbar_base + Int32(8)
        _bf_0 = _mbar_base + Int32(16)
        _bf_1 = _mbar_base + Int32(24)

        # --- K-loop (warp 0 only issues UMMA) ---
        if warp_idx == Int32(0):
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

            # Process K-block 0 from buf 0
            mbarrier_wait(_kr_0, Int32(0))
            for k_inner in cutlass.range_constexpr(self.num_k_inner):
                k_coord_0 = (None, None, k_inner, 0)
                cute.gemm(tiled_mma, tCtAcc, tCrA[k_coord_0], tCrB[k_coord_0], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # Signal buf 0 free
            if self.num_k_blocks > 1:
                mbarrier_arrive(_bf_0)

            # Process remaining K-blocks
            if self.num_k_blocks >= 2:
                # K-block 1 from buf 1
                mbarrier_wait(_kr_1, Int32(0))
                for k_inner in cutlass.range_constexpr(self.num_k_inner):
                    k_coord_1 = (None, None, k_inner, 1)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[k_coord_1], tCrB[k_coord_1], tCtAcc)
                mbarrier_arrive(_bf_1)

            if self.num_k_blocks > 2:
                # K-blocks 2+ with phase-tracked mbarrier waits
                _kr_phase_0 = Int32(1)  # Phase 1 (already consumed phase 0)
                _kr_phase_1 = Int32(1)
                k_idx = Int32(2)
                while k_idx < Int32(self.num_k_blocks):
                    buf_idx = k_idx % Int32(2)
                    if buf_idx == Int32(0):
                        mbarrier_wait(_kr_0, _kr_phase_0)
                        _kr_phase_0 = _kr_phase_0 ^ Int32(1)
                        for k_inner in cutlass.range_constexpr(self.num_k_inner):
                            k_coord = (None, None, k_inner, 0)
                            cute.gemm(tiled_mma, tCtAcc, tCrA[k_coord], tCrB[k_coord], tCtAcc)
                        mbarrier_arrive(_bf_0)
                    else:
                        mbarrier_wait(_kr_1, _kr_phase_1)
                        _kr_phase_1 = _kr_phase_1 ^ Int32(1)
                        for k_inner in cutlass.range_constexpr(self.num_k_inner):
                            k_coord = (None, None, k_inner, 1)
                            cute.gemm(tiled_mma, tCtAcc, tCrA[k_coord], tCrB[k_coord], tCtAcc)
                        mbarrier_arrive(_bf_1)
                    k_idx = k_idx + Int32(1)

        # Fence: ensure UMMA writes to TMEM are visible before epilogue reads
        cute.arch.fence_view_async_tmem_store()

        # Sync all MMA warps before epilogue
        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

        # Invalidate op-managed mbarriers before page reuse
        if tidx == Int32(0):
            mbarrier_inval(_kr_0)
            mbarrier_inval(_kr_1)
            mbarrier_inval(_bf_0)
            mbarrier_inval(_bf_1)

        # --- Epilogue: TMEM → regs → smem (all MMA warps) ---

        # Compute epilogue tile shape
        cta_tile = (self.tile_size_S, self.tile_size_N)
        epi_tile = bh.compute_epilogue_tile_shape(
            (self.tile_size_S, self.tile_size_N, self.mma_tiler_k),
            False,  # use_2cta_instrs
            bh.LayoutEnum.ROW_MAJOR,
            self.c_dtype,
        )

        # TMEM → register copy
        copy_atom_t2r = bh.get_tmem_load_op(
            (self.tile_size_S, self.tile_size_N, self.mma_tiler_k),
            bh.LayoutEnum.ROW_MAJOR,
            self.c_dtype,
            Float32,
            epi_tile,
            False,  # use_2cta_instrs
        )

        # Partition accumulator by epilogue tiles
        tCtAcc_epi = cute.flat_divide(tCtAcc, epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tCtAcc_epi[(None, None, None, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc_epi)
        cAcc = cute.make_identity_tensor(cta_tile)
        cAcc_epi = cute.flat_divide(cAcc, epi_tile)
        tTR_cAcc = thr_copy_t2r.partition_D(cAcc_epi)
        tTR_rAcc = cute.make_fragment(
            tTR_cAcc[None, None, None, 0, 0].shape, Float32)

        # Register → smem copy
        copy_atom_r2s = bh.get_smem_store_op(
            bh.LayoutEnum.ROW_MAJOR,
            self.c_dtype,
            Float32,
            tiled_copy_t2r,
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)

        # Output smem tensor (C) — reuse page from start
        swz_c = cute.make_swizzle(self.swz_B_c, 4, 3)

        # Epilogue smem layout
        epi_smem_layout = bh.make_smem_layout_epi(
            self.c_dtype,
            LayoutEnum.ROW_MAJOR,
            epi_tile,
            1,  # 1 stage for epilogue
        )
        sC = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.c_dtype, page_ptr,
                              cute.AddressSpace.smem, assumed_align=128),
                epi_smem_layout.inner,
                dtype=self.c_dtype,
            ),
            epi_smem_layout.outer,
        )

        tRS_sC = thr_copy_r2s.partition_D(sC)

        # Fragment for activated output
        tTR_rD = cute.make_fragment(tTR_rAcc.shape, Float32)

        # Process each epilogue sub-tile
        num_epi_m = cute.size(tTR_tAcc, mode=[3])
        num_epi_n = cute.size(tTR_tAcc, mode=[4])
        for epi_m in cutlass.range_constexpr(num_epi_m):
            for epi_n in cutlass.range_constexpr(num_epi_n):
                # Load from TMEM → registers
                cute.copy(tiled_copy_t2r,
                          tTR_tAcc[None, None, None, epi_m, epi_n, 0],
                          tTR_rAcc)

                # Apply activation in registers
                tRS_rAcc = tiled_copy_r2s.retile(tTR_rAcc)
                tRS_rD_retiled = tiled_copy_r2s.retile(tTR_rD)
                for ci in cutlass.range_constexpr(cute.size(tRS_rAcc)):
                    val = tRS_rAcc[ci]
                    if self.activation == 1:  # ReLU
                        val = val if val >= Float32(0.0) else Float32(0.0)
                    elif self.activation == 2:  # SiLU
                        neg_val = Float32(0.0) - val
                        exp_neg = cute.math.exp(neg_val, fastmath=True)
                        val = val / (Float32(1.0) + exp_neg)
                    tRS_rD_retiled[ci] = val.to(self.c_dtype)

                # Store to smem
                cute.copy(tiled_copy_r2s, tRS_rD_retiled,
                          tRS_sC[None, None, None, epi_m, epi_n, 0])

        # --- TMEM dealloc ---
        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
        if warp_idx == Int32(0):
            cute.arch.relinquish_tmem_alloc_permit()
            acc_shape_for_dealloc = tiled_mma.partition_shape_C((self.tile_size_S, self.tile_size_N))
            acc_fake_for_dealloc = tiled_mma.make_fragment_C(cute.append(acc_shape_for_dealloc, 1))
            num_cols = cutlass.utils.get_num_tmem_alloc_cols(acc_fake_for_dealloc)
            cute.arch.dealloc_tmem(acc_tmem_ptr, num_cols)

    # =========================================================================
    # Forward Store (TMA S→G)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_B, tile_S, tile_N, c_tma, c_tma_gmem):
        """TMA store C from smem to global."""
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
    # Scheduling
    # =========================================================================

    @classmethod
    def _auto_tiles(cls, page_size, elem_bytes=2):
        """Compute best (tile_S, tile_N, tile_K) for SM100 UMMA.

        Constraints:
          - tile_S ∈ {64, 128, 256} (UMMA M constraint)
          - tile_N ∈ {32, 64, 96, ..., 256} (UMMA N constraint, step 32)
          - tile_K must be multiple of 64 (mma_tiler_k for bf16)
          - 2 * (tile_S + tile_N) * tile_K * elem ≤ page_size (double-buf A+B)
          - tile_S * tile_N * elem ≤ page_size (C epilogue)
        """
        mbar_bytes = 32
        candidates = [
            (128, 128, 64),
            (128, 64, 64),
            (64, 128, 64),
            (64, 64, 64),
            (128, 32, 64),
            (64, 32, 64),
        ]
        for tile_S, tile_N, tile_K in candidates:
            ab = 2 * (tile_S + tile_N) * tile_K * elem_bytes + mbar_bytes
            c = tile_S * tile_N * elem_bytes
            if ab <= page_size and c <= page_size:
                return tile_S, tile_N, tile_K
        # Fallback with smaller K
        for tile_S, tile_N, tile_K in [(64, 32, 64), (64, 32, 32)]:
            ab = 2 * (tile_S + tile_N) * tile_K * elem_bytes + mbar_bytes
            c = tile_S * tile_N * elem_bytes
            if ab <= page_size and c <= page_size:
                return tile_S, tile_N, tile_K
        raise ValueError(f"GemmSm100Op: page_size={page_size} too small for any valid tile config")

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE,
                 activation=None, **tensors):
        """Schedule SM100 UMMA GEMM forward.

        Tensors must be 3D: A (B, S, K), C (B, S, N). B_w stays 2D (N, K).
        """
        ts = dict(tile_sizes or {})
        ts.setdefault("B", 1)
        if "S" not in ts or "N" not in ts:
            a = tensors.get('a')
            elem_bytes = a.element_size() if a is not None else 2
            auto_S, auto_N, auto_K = cls._auto_tiles(page_size, elem_bytes)
            ts.setdefault("S", auto_S)
            ts.setdefault("N", auto_N)
            ts.setdefault("K", auto_K)
        tile_K = ts.pop("K", 64)
        scheduled = cls._schedule_single(tile_sizes=ts, **tensors)
        scheduled.static_dims["tile_K"] = tile_K
        scheduled.static_dims["page_size"] = page_size
        if activation is not None:
            from machete.kernels.activation.activation import ACT_MAP
            scheduled.static_dims["activation"] = ACT_MAP[activation]
        return [scheduled]

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for GemmSm100Op."""
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS
        page_size = max(op.static_dims.get('page_size', DEFAULT_PAGE_SIZE)
                        for op in ops)
        # 4 MMA warps for epilogue parallelism (UMMA is CTA-level, needs fewer)
        num_mma_warps = 4
        threads_per_block = (num_mma_warps + NUM_DMA_WARPS) * 32
        return MegakernelConfig(threads_per_block=threads_per_block,
                                page_size=page_size)
