# Copyright (c) 2025, Machete Authors
"""
GEMM Op for the Megakernel.

Computes C[M,N] = A[M,K] @ B[N,K]^T  (i.e., standard matmul with B pre-transposed).

Note: B is expected in (N, K) layout (K contiguous). If you have B in standard
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
    store:   TMA store S->G of C[tile_M, tile_N]

Page layout (2-stage double buffer):
    [buf0: A_tile + B_tile] [buf1: A_tile + B_tile] [mbar0: 8B] [mbar1: 8B]
    Epilogue: [C: tile_M x tile_N]  (reuses page from offset 0)

No output pre-zeroing needed — fp32 accumulator handles full K reduction.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import (
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_arrive,
    mbarrier_arrive_expect_tx,
    mbarrier_wait,
    named_barrier_sync,
)


class GemmOp(Op):
    """GEMM operation for the megakernel framework.

    Computes C[M,N] = A[M,K] @ B[N,K]^T using tensor core MMA.
    B must be in (N, K) layout with K contiguous.

    K is handled via double-buffered TMA pipelining: the DMA warp TMA-loads
    the first 2 K-blocks, then an elected MMA thread issues TMA loads for
    K-blocks 2+ using compute-local mbarriers.
    """

    reads = {
        "a": (None, ("M", "K")),
        "b": (None, ("N", "K")),
    }
    writes = {"c": (None, ("M", "N"))}
    tile = ("M", "N")

    tma_loads = {"a", "b"}
    tma_stores = {"c"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        """Custom TMA tile shapes for inner K-block sub-tiling.

        A and B use tile_K (inner block size) instead of full K extent.
        C uses the standard tile_M x tile_N.
        """
        tile_K = static_dims.get("tile_K", 32)
        if tensor_name == "a":
            return (tile_sizes["M"], tile_K)
        elif tensor_name == "b":
            return (tile_sizes["N"], tile_K)
        else:  # "c"
            return (tile_sizes["M"], tile_sizes["N"])

    def __init__(self, **config):
        super().__init__(**config)

        if self.a_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            raise ValueError(f"GemmOp requires fp16 or bf16 input, got {self.a_dtype}")

        self.tile_K = getattr(self, 'tile_K', 32)
        self.page_size = getattr(self, 'page_size', DEFAULT_PAGE_SIZE)

        self.a_tile_bytes = self.tile_size_M * self.tile_K * self.elem_bytes
        self.b_tile_bytes = self.tile_size_N * self.tile_K * self.elem_bytes
        self.c_tile_bytes = self.tile_size_M * self.tile_size_N * self.elem_bytes
        self.buf_stride = self.a_tile_bytes + self.b_tile_bytes

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
            self.atom_K = 64
        elif self.tile_K % 32 == 0:
            self.swz_B_ab = 2   # SW64
            self.atom_K = 32
        else:
            self.swz_B_ab = 1   # SW32
            self.atom_K = 16

        # 4 op-managed mbarriers after double-buffer data (32 bytes):
        #   buf_free[0,1]:     compute → store warp (buffer read, safe to overwrite)
        #   kblock_ready[0,1]: TMA hw → compute (new K-block data arrived)
        self.mbar_offset = 2 * self.buf_stride
        self.ab_tma_bytes = self.a_tile_bytes + self.b_tile_bytes
        self.mbar_bytes = 32  # 4 x 8 bytes

        # Always 2-stage double buffer + mbarrier space
        ab_bytes = 2 * self.buf_stride
        total_smem = max(ab_bytes + self.mbar_bytes, self.c_tile_bytes)
        assert total_smem <= self.page_size, (
            f"GemmOp: smem {total_smem}B exceeds page_size ({self.page_size}B). "
            f"tile_M={self.tile_size_M}, tile_N={self.tile_size_N}, "
            f"tile_K={self.tile_K}"
        )

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape,
                                tile_sizes, static_dims):
        """Return swizzled smem layout code for TMA descriptor creation."""
        tile_K = static_dims.get("tile_K", 32)
        if tensor_name not in ("a", "b"):
            return None  # No swizzle for C

        if tile_K % 64 == 0 and tile_K >= 64:
            B = 3   # SW128
        elif tile_K % 32 == 0:
            B = 2   # SW64
        else:
            B = 1   # SW32

        dim0, dim1 = tma_tile_shape  # (tile_K, tile_M/N) in TMA convention
        return (
            f"cute.make_composed_layout("
            f"cute.make_swizzle({B}, 4, 3), 0, "
            f"cute.make_layout(({dim0}, {dim1}), stride=(1, {dim0})))"
        )

    # =========================================================================
    # Forward Load: TMA G->S (called by load warp and store warp)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_M, tile_N,
             a_tma, a_tma_gmem, b_tma, b_tma_gmem,
             work_mbar, inner_iter_idx):
        """TMA load of A and B K-blocks.

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
                mbarrier_init(_bf_0, Int32(1))
                mbarrier_init(_bf_1, Int32(1))
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
                sA_ptr = cute.recast_ptr(
                    cute.make_ptr(self.a_dtype, _buf_base,
                                  cute.AddressSpace.smem),
                    swz, dtype=self.a_dtype)
                sA = cute.make_tensor(
                    sA_ptr,
                    cute.make_layout((self.tile_K, self.tile_size_M),
                                     stride=(1, self.tile_K)))
                gA = cute.local_tile(
                    a_tma_gmem, (self.tile_K, self.tile_size_M), (None, None))
                tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                    a_tma, Int32(0), cute.make_layout(1),
                    cute.group_modes(sA, 0, 2),
                    cute.group_modes(gA, 0, 2))

                sB_ptr = cute.recast_ptr(
                    cute.make_ptr(self.b_dtype,
                                  _buf_base + Int32(self.a_tile_bytes),
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

                cute.copy(a_tma, tAgA[(None, Int32(_k), tile_M)], tAsA,
                          tma_bar_ptr=mbar_ptr)
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

            # TMA A tile
            sA_ptr = cute.recast_ptr(
                cute.make_ptr(self.a_dtype, _buf_base,
                              cute.AddressSpace.smem),
                swz, dtype=self.a_dtype)
            sA = cute.make_tensor(
                sA_ptr,
                cute.make_layout((self.tile_K, self.tile_size_M),
                                 stride=(1, self.tile_K)))
            gA = cute.local_tile(
                a_tma_gmem, (self.tile_K, self.tile_size_M), (None, None))
            tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                a_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA, 0, 2))

            # TMA B tile
            sB_ptr = cute.recast_ptr(
                cute.make_ptr(self.b_dtype,
                              _buf_base + Int32(self.a_tile_bytes),
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

            cute.copy(a_tma, tAgA[(None, _k_block, tile_M)], tAsA,
                      tma_bar_ptr=_kr_ptr)
            cute.copy(b_tma, tBgB[(None, _k_block, tile_N)], tBsB,
                      tma_bar_ptr=_kr_ptr)

    # =========================================================================
    # Forward Compute: Pure MMA (no TMA — store warp handles K-block loads)
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_M, tile_N):
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
            cute.make_layout((self.tile_size_M, self.tile_K),
                             stride=(self.tile_K, 1)),
        )
        sB_0 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.b_dtype,
                              page_ptr + Int32(self.a_tile_bytes),
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
            cute.make_layout((self.tile_size_M, self.tile_K),
                             stride=(self.tile_K, 1)),
        )
        sB_1 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.b_dtype,
                              page_ptr + Int32(self.buf_stride + self.a_tile_bytes),
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

        # Per-buffer LdMatrix smem partitions
        tAsA_ld_0 = smem_thr_copy_A.partition_S(sA_0)
        tBsB_ld_0 = smem_thr_copy_B.partition_S(sB_0)
        tAsA_ld_1 = smem_thr_copy_A.partition_S(sA_1)
        tBsB_ld_1 = smem_thr_copy_B.partition_S(sB_1)

        # Op-managed mbarriers (initialized by load(iter=0))
        _bf_0 = page_ptr + Int32(self.mbar_offset)       # buf_free[0]
        _bf_1 = page_ptr + Int32(self.mbar_offset + 8)   # buf_free[1]
        _kr_0 = page_ptr + Int32(self.mbar_offset + 16)  # kblock_ready[0]
        _kr_1 = page_ptr + Int32(self.mbar_offset + 24)  # kblock_ready[1]

        # --- Init fp32 accumulator ---
        acc = cute.make_fragment(
            tiled_mma.partition_shape_C((self.tile_size_M, self.tile_size_N)),
            Float32,
        )
        acc.fill(0.0)

        # =====================================================================
        # Process K-block 0 from buf 0 (TMA loaded by load warp)
        # =====================================================================
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

        # Sync all MMA warps, then one thread signals buf 0 is free
        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
        if tidx == Int32(0):
            mbarrier_arrive(_bf_0)

        # =====================================================================
        # Process K-block 1 from buf 1 (TMA loaded by load warp, if exists)
        # =====================================================================
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

            # Sync all MMA warps, then one thread signals buf 1 is free
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
            if tidx == Int32(0):
                mbarrier_arrive(_bf_1)

        # =====================================================================
        # K-blocks 2+ — wait for kblock_ready, process, signal buf_free
        # =====================================================================
        _kr_phase_0 = Int32(0)
        _kr_phase_1 = Int32(0)
        k_idx = Int32(2)
        while k_idx < Int32(self.num_k_blocks):
            # Wait for store warp's TMA to deliver this K-block
            if k_idx % Int32(2) == Int32(0):
                mbarrier_wait(_kr_0, _kr_phase_0)
                _kr_phase_0 = _kr_phase_0 ^ Int32(1)
            if k_idx % Int32(2) == Int32(1):
                mbarrier_wait(_kr_1, _kr_phase_1)
                _kr_phase_1 = _kr_phase_1 ^ Int32(1)

            # Process current K-block (dynamic buffer selection)
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

            # Sync all MMA warps, then one thread signals buffer is free
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
            if tidx == Int32(0):
                if k_idx % Int32(2) == Int32(0):
                    mbarrier_arrive(_bf_0)
                if k_idx % Int32(2) == Int32(1):
                    mbarrier_arrive(_bf_1)

            k_idx = k_idx + Int32(1)

        # --- Epilogue: R->S (write final C to smem for TMA store) ---
        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
        sC = cute.make_tensor(
            cute.make_ptr(self.c_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.tile_size_M, self.tile_size_N),
                             stride=(self.tile_size_N, 1)),
        )
        tCsC = thr_mma.partition_C(sC)
        for ci in cutlass.range_constexpr(cute.size(acc)):
            tCsC[ci] = acc[ci].to(self.c_dtype)

    # =========================================================================
    # Forward Store (S->G): Regular TMA store of C
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_M, tile_N, c_tma, c_tma_gmem):
        """TMA store of C from shared to global memory."""
        sC = cute.make_tensor(
            cute.make_ptr(self.c_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.tile_size_N, self.tile_size_M)),
        )

        gC = cute.local_tile(
            c_tma_gmem, (self.tile_size_N, self.tile_size_M), (None, None),
        )
        tCsC, tCgC = cute.nvgpu.cpasync.tma_partition(
            c_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(gC, 0, 2),
        )

        with cute.arch.elect_one():
            cute.copy(c_tma, tCsC, tCgC[(None, tile_N, tile_M)])

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def _auto_tiles(cls, page_size, elem_bytes=2):
        """Compute largest (tile_M, tile_N, tile_K) that fit in page_size.

        Constraints:
          - 2 * (tile_M + tile_N) * tile_K * elem <= page_size  (AB double buffer)
          - tile_M * tile_N * elem <= page_size  (C epilogue)
          - tile_K must be a multiple of 16
        """
        tile_K = 32
        for tile_M, tile_N in [(128, 64), (64, 64), (64, 32), (32, 32)]:
            ab = 2 * (tile_M + tile_N) * tile_K * elem_bytes
            c = tile_M * tile_N * elem_bytes
            if ab <= page_size and c <= page_size:
                return tile_M, tile_N, tile_K
        return 32, 32, 16

    @classmethod
    def schedule_forward(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        """Schedule forward GEMM.

        Accepts tile_sizes with M, N, K keys. K is the inner K-block size
        (not a framework tile dimension). Auto-computed from page_size when
        not specified.

        Always uses 2-stage double buffering: DMA loads first 2 K-blocks
        via TMA, compute handles remaining K-blocks via TMA (elected thread).
        """
        ts = dict(tile_sizes or {})
        if "M" not in ts or "N" not in ts:
            a = tensors.get('a')
            elem_bytes = a.element_size() if a is not None else 2
            auto_M, auto_N, auto_K = cls._auto_tiles(
                page_size, elem_bytes)
            ts.setdefault("M", auto_M)
            ts.setdefault("N", auto_N)
            ts.setdefault("K", auto_K)
        tile_K = ts.pop("K", 32)
        scheduled = cls._schedule_single(tile_sizes=ts, **tensors)
        scheduled.static_dims["tile_K"] = tile_K
        scheduled.static_dims["page_size"] = page_size
        return [scheduled]

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for scheduled GemmOps."""
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS
        page_size = ops[0].static_dims.get('page_size', DEFAULT_PAGE_SIZE)
        tile_m = ops[0].tile_sizes.get('M', 64)
        num_mma_warps = tile_m // 16
        threads_per_block = (num_mma_warps + NUM_DMA_WARPS) * 32
        return MegakernelConfig(threads_per_block=threads_per_block,
                                page_size=page_size)

    @classmethod
    def schedule_backward(cls, tile_sizes=None, **tensors):
        """Schedule GEMM backward as forward-equivalent ops.

        dA[M,K] = dout[M,N] @ B[N,K]  (contracts over N)
        dB[N,K] = dout^T[N,M] @ A[M,K]  (contracts over M)

        Each gradient is a standard forward GEMM with transposed inputs.
        No output zeroing needed (regular TMA store, not atomic add).
        """
        dout, a, b = tensors['dout'], tensors['a'], tensors['b']
        da, db = tensors.get('da'), tensors.get('db')
        ts = tile_sizes or {}
        tile_m = ts.get("M", 64)
        tile_n = ts.get("N", 32)
        tile_k = ts.get("K", 32)
        ops = []

        if da is not None:
            b_t = b.t().contiguous()
            ops.extend(cls.schedule_forward(
                a=dout, b=b_t, c=da,
                tile_sizes={"M": tile_m, "N": tile_k, "K": tile_n},
            ))

        if db is not None:
            dout_t = dout.t().contiguous()
            a_t = a.t().contiguous()
            ops.extend(cls.schedule_forward(
                a=dout_t, b=a_t, c=db,
                tile_sizes={"M": tile_n, "N": tile_k, "K": tile_m},
            ))

        return ops
