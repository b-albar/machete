# Copyright (c) 2025, Machete Authors
"""
MoE Grouped GEMM Op for the Megakernel.

Computes c[i] = sorted_x[i] @ w[expert_ids[i]]^T for each sorted token
position i. Each M-tile reads its expert_id from expert_ids and cpasync-loads
the corresponding expert weight from w.

Architecture (SM_90+ / Hopper+):
    - DMA warp:  TMA G->S of sorted_x K-blocks (double-buffered)
    - MMA warps: cpasync-load expert weight tiles from w + tensor core MMA
    - Store warp: TMA S->G of output c

Tensor layout (pre-sorted by moe_align_sort):
    sorted_x:    [total_padded, K]     — input tokens sorted by expert
    w:           [num_experts, N, K]   — expert weights (K contiguous)
    expert_ids:  [total_padded]        — expert ID per token (int32)
    c:           [total_padded, N]     — output in sorted order

Pipelined phases:
    load:    TMA G->S of sorted_x K-blocks into double buffer
    compute: Read expert_id from global, cpasync w[expert_id] tile,
             LdMatrix + MMA per K-block, epilogue R->S
    store:   TMA store S->G of c[tile_M, tile_N]

Page layout (2-stage double buffer):
    [buf0: x_tile (tile_M × tile_K)] [buf1: x_tile] [w_buf (tile_N × tile_K)]
    [mbar: 4 × 8B]
    Epilogue: [C: tile_M × tile_N] (reuses page from offset 0)
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


class MoeGemmOp(Op):
    """Grouped GEMM for Mixture-of-Experts.

    Each M-tile uses a different expert weight matrix determined by expert_ids.
    sorted_x is TMA-loaded by the DMA warp; w is cpasync-loaded by MMA warps.
    """

    reads = {
        "sorted_x":   (None, ("M", "K")),
        "w":          (None, ("E", "N", "K")),
        "expert_ids": (cutlass.Int32, ("M",)),
    }
    writes = {"c": (None, ("M", "N"))}
    tile = ("M", "N")

    tma_loads = {"sorted_x"}
    tma_stores = {"c"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        """Custom TMA tile shapes for inner K-block sub-tiling."""
        tile_K = static_dims.get("tile_K", 32)
        if tensor_name == "sorted_x":
            return (tile_sizes["M"], tile_K)
        else:  # "c"
            return (tile_sizes["M"], tile_sizes["N"])

    def __init__(self, **config):
        super().__init__(**config)

        if self.sorted_x_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            raise ValueError(f"MoeGemmOp requires fp16 or bf16, got {self.sorted_x_dtype}")

        self.tile_K = getattr(self, 'tile_K', 32)
        self.page_size = getattr(self, 'page_size', DEFAULT_PAGE_SIZE)

        self.x_tile_bytes = self.tile_size_M * self.tile_K * self.elem_bytes
        self.w_tile_bytes = self.tile_size_N * self.tile_K * self.elem_bytes
        self.c_tile_bytes = self.tile_size_M * self.tile_size_N * self.elem_bytes

        assert self.tile_K >= 16 and self.tile_K % 16 == 0, (
            f"MoeGemmOp: tile_K={self.tile_K} must be >= 16 and a multiple of 16."
        )

        self.num_k_blocks = (self.K + self.tile_K - 1) // self.tile_K
        self.num_mma_warps = self.threads_per_row // 32
        self.num_mma_threads = self.num_mma_warps * 32

        # DMA TMA-loads first 2 K-blocks of sorted_x
        self.tma_k_blocks = min(2, self.num_k_blocks)
        self.inner_iters = max(1, self.num_k_blocks - 1)

        # Swizzle for bank-conflict-free LdMatrix reads
        if self.tile_K % 64 == 0 and self.tile_K >= 64:
            self.swz_B = 3   # SW128
        elif self.tile_K % 32 == 0:
            self.swz_B = 2   # SW64
        else:
            self.swz_B = 1   # SW32

        # Smem layout:
        #   [buf0: x_tile] [buf1: x_tile] [w_buf: w_tile] [4 mbarriers]
        self.x_buf_stride = self.x_tile_bytes  # per-buffer x size
        self.w_offset = 2 * self.x_tile_bytes  # w_buf starts after double-buffered x
        self.mbar_offset = self.w_offset + self.w_tile_bytes
        self.mbar_bytes = 32  # 4 x 8 bytes
        self.x_tma_bytes = self.x_tile_bytes  # TMA bytes per x K-block

        # cpasync setup for weight loading (same pattern as FlashAttentionSm120Op)
        self.async_copy_elems = 128 // (self.elem_bytes * 8)  # 8 elements for fp16
        self.copy_dim1 = self.tile_K // self.async_copy_elems
        self.copy_dim0 = self.num_mma_threads // self.copy_dim1
        assert self.copy_dim0 > 0, (
            f"MoeGemmOp: not enough threads ({self.num_mma_threads}) for "
            f"tile_K={self.tile_K} cpasync (need at least {self.copy_dim1})"
        )

        # Weight stride: w is [E, N, K], stride_E = N * K
        self.w_stride_E = self.N * self.K

        # Verify smem fits
        data_bytes = 2 * self.x_tile_bytes + self.w_tile_bytes + self.mbar_bytes
        total_smem = max(data_bytes, self.c_tile_bytes)
        assert total_smem <= self.page_size, (
            f"MoeGemmOp: smem {total_smem}B exceeds page_size ({self.page_size}B). "
            f"tile_M={self.tile_size_M}, tile_N={self.tile_size_N}, tile_K={self.tile_K}"
        )

        # Override compute to the MMA method (FlashAttentionSm120Op pattern)
        self.compute = self.compute_mma

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape,
                                tile_sizes, static_dims):
        """Return swizzled smem layout code for TMA descriptor creation."""
        tile_K = static_dims.get("tile_K", 32)
        if tensor_name != "sorted_x":
            return None  # No swizzle for C

        if tile_K % 64 == 0 and tile_K >= 64:
            B = 3
        elif tile_K % 32 == 0:
            B = 2
        else:
            B = 1

        dim0, dim1 = tma_tile_shape
        return (
            f"cute.make_composed_layout("
            f"cute.make_swizzle({B}, 4, 3), 0, "
            f"cute.make_layout(({dim0}, {dim1}), stride=(1, {dim0})))"
        )

    # =========================================================================
    # Forward Load: TMA G->S of sorted_x (called by load/store warps)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_M, tile_N,
             sorted_x_tma, sorted_x_tma_gmem,
             work_mbar, inner_iter_idx):
        """TMA load of sorted_x K-blocks.

        iter 0 (load warp):  Init mbarriers, TMA K0+K1 → buf0+buf1.
        iter 1+ (store warp): Wait buf_free, TMA one K-block → buf.
        """
        swz = cute.make_swizzle(self.swz_B, 4, 3)

        # Mbarrier pointers (4 x 8B after data)
        _bf_0 = page_ptr + Int32(self.mbar_offset)
        _bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        _kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        _kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        if inner_iter_idx == Int32(0):
            # === ITER 0: init mbarriers + TMA first 2 K-blocks ===
            with cute.arch.elect_one():
                mbarrier_init(_bf_0, Int32(1))
                mbarrier_init(_bf_1, Int32(1))
                mbarrier_init(_kr_0, Int32(1))
                mbarrier_init(_kr_1, Int32(1))
            mbarrier_init_fence()

            mbar_ptr = cute.make_ptr(
                cutlass.Int64, work_mbar, cute.AddressSpace.smem)
            nbytes = Int32(self.tma_k_blocks * self.x_tma_bytes)
            with cute.arch.elect_one():
                mbarrier_arrive_expect_tx(work_mbar, nbytes)

            for _k in cutlass.range_constexpr(self.tma_k_blocks):
                _buf_base = page_ptr + Int32(_k * self.x_buf_stride)
                sX_ptr = cute.recast_ptr(
                    cute.make_ptr(self.sorted_x_dtype, _buf_base,
                                  cute.AddressSpace.smem),
                    swz, dtype=self.sorted_x_dtype)
                sX = cute.make_tensor(
                    sX_ptr,
                    cute.make_layout((self.tile_K, self.tile_size_M),
                                     stride=(1, self.tile_K)))
                gX = cute.local_tile(
                    sorted_x_tma_gmem, (self.tile_K, self.tile_size_M),
                    (None, None))
                tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
                    sorted_x_tma, Int32(0), cute.make_layout(1),
                    cute.group_modes(sX, 0, 2),
                    cute.group_modes(gX, 0, 2))

                cute.copy(sorted_x_tma, tXgX[(None, Int32(_k), tile_M)],
                          tXsX, tma_bar_ptr=mbar_ptr)

        if inner_iter_idx > Int32(0):
            # === ITER 1+: TMA one K-block into freed buffer ===
            _k_block = inner_iter_idx + Int32(1)
            _buf_idx = _k_block % Int32(2)

            _bf_phase = ((inner_iter_idx - Int32(1)) // Int32(2)) % Int32(2)
            if _buf_idx == Int32(0):
                mbarrier_wait(_bf_0, _bf_phase)
            if _buf_idx == Int32(1):
                mbarrier_wait(_bf_1, _bf_phase)

            _buf_base = _buf_idx * Int32(self.x_buf_stride) + page_ptr
            _kr_ptr = cute.make_ptr(
                cutlass.Int64, _kr_0, cute.AddressSpace.smem)
            if _buf_idx == Int32(0):
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(_kr_0, Int32(self.x_tma_bytes))
            if _buf_idx == Int32(1):
                _kr_ptr = cute.make_ptr(
                    cutlass.Int64, _kr_1, cute.AddressSpace.smem)
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(_kr_1, Int32(self.x_tma_bytes))

            sX_ptr = cute.recast_ptr(
                cute.make_ptr(self.sorted_x_dtype, _buf_base,
                              cute.AddressSpace.smem),
                swz, dtype=self.sorted_x_dtype)
            sX = cute.make_tensor(
                sX_ptr,
                cute.make_layout((self.tile_K, self.tile_size_M),
                                 stride=(1, self.tile_K)))
            gX = cute.local_tile(
                sorted_x_tma_gmem, (self.tile_K, self.tile_size_M),
                (None, None))
            tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
                sorted_x_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sX, 0, 2),
                cute.group_modes(gX, 0, 2))

            cute.copy(sorted_x_tma, tXgX[(None, _k_block, tile_M)],
                      tXsX, tma_bar_ptr=_kr_ptr)

    # =========================================================================
    # Forward Compute: cpasync W + MMA
    # =========================================================================

    @cute.jit
    def compute_mma(self, page_ptr, tile_M, tile_N,
                    sorted_x, w, expert_ids, c):
        """Grouped GEMM compute with cpasync weight loading.

        For each K-block:
        1. Wait for sorted_x K-block (TMA-loaded by DMA/store warp)
        2. Read expert_id from expert_ids via global pointer arithmetic
        3. cpasync w[expert_id, tile_N, k_block] → smem w_buf
        4. LdMatrix + MMA: acc += x_tile × w_tile
        After all K-blocks: epilogue converts fp32 acc → output dtype in smem.

        Note: sorted_x and c are unused (TMA load/store handles them) but must
        be in signature for expects_tensors to pass w and expert_ids.
        """
        tidx = cute.arch.thread_idx()[0]

        # --- Build tiled MMA ---
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            self.sorted_x_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.num_mma_warps, 1, 1)),
            permutation_mnk=(self.num_mma_warps * 16, 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        # --- LdMatrix tiled copies ---
        swz = cute.make_swizzle(self.swz_B, 4, 3)

        smem_copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=False, num_matrices=4), self.sorted_x_dtype)
        smem_tiled_copy_A = cute.make_tiled_copy_A(
            smem_copy_atom_A, tiled_mma)
        smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)

        smem_copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=False, num_matrices=4), self.w_dtype)
        smem_tiled_copy_B = cute.make_tiled_copy_B(
            smem_copy_atom_B, tiled_mma)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        # --- sorted_x smem tensors (buf 0 and 1, swizzled) ---
        sX_0 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.sorted_x_dtype, page_ptr,
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.sorted_x_dtype),
            cute.make_layout((self.tile_size_M, self.tile_K),
                             stride=(self.tile_K, 1)),
        )
        sX_1 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.sorted_x_dtype,
                              page_ptr + Int32(self.x_buf_stride),
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.sorted_x_dtype),
            cute.make_layout((self.tile_size_M, self.tile_K),
                             stride=(self.tile_K, 1)),
        )

        # --- w smem tensor (single buffer, swizzled) ---
        sW = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.w_dtype,
                              page_ptr + Int32(self.w_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.w_dtype),
            cute.make_layout((self.tile_size_N, self.tile_K),
                             stride=(self.tile_K, 1)),
        )

        # MMA partitions and register fragments
        tCsA = thr_mma.partition_A(sX_0)
        tCsB = thr_mma.partition_B(sW)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrA_ld = smem_thr_copy_A.retile(tCrA)
        tCrB_ld = smem_thr_copy_B.retile(tCrB)

        # Per-buffer LdMatrix smem partitions for x
        tAsA_ld_0 = smem_thr_copy_A.partition_S(sX_0)
        tAsA_ld_1 = smem_thr_copy_A.partition_S(sX_1)
        # W partition (single buffer, reused)
        tBsW_ld = smem_thr_copy_B.partition_S(sW)

        # --- cpasync setup for weight loading ---
        async_copy_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), self.w_dtype,
            num_bits_per_copy=128)
        copy_thread_layout = cute.make_layout(
            (self.copy_dim0, self.copy_dim1),
            stride=(self.copy_dim1, 1))
        copy_value_layout = cute.make_layout((1, self.async_copy_elems))
        gmem_tiled_copy = cute.make_tiled_copy_tv(
            async_copy_atom, copy_thread_layout, copy_value_layout)
        thr_copy = gmem_tiled_copy.get_slice(tidx)

        # cpasync smem destination for W (swizzled)
        sW_cp = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.w_dtype,
                              page_ptr + Int32(self.w_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.w_dtype),
            cute.make_layout((self.tile_size_N, self.tile_K),
                             stride=(self.tile_K, 1)),
        )
        tWsW_cp = thr_copy.partition_D(sW_cp)

        # --- Read expert_id for this M-tile from global memory ---
        # expert_ids is a CuTe tensor (M,). All tokens in an M-tile share
        # the same expert (guaranteed by moe_align_sort padding).
        eid_ptr = (expert_ids.iterator
                   + tile_M * Int32(self.tile_size_M)).align(4)
        g_eid = cute.make_tensor(eid_ptr, cute.make_layout(1))
        expert_id = g_eid[Int32(0)].to(Int32)

        # Compute base pointer into w for this expert:
        # w[expert_id, tile_N_start:, k_start:] with layout (N, K)
        w_expert_ptr = (w.iterator
                        + expert_id * Int32(self.w_stride_E)).align(16)
        gW_expert = cute.make_tensor(
            w_expert_ptr,
            cute.make_layout((self.N, self.K), stride=(self.K, 1)))

        # Op-managed mbarriers
        _bf_0 = page_ptr + Int32(self.mbar_offset)
        _bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        _kr_0 = page_ptr + Int32(self.mbar_offset + 16)
        _kr_1 = page_ptr + Int32(self.mbar_offset + 24)

        # --- Init fp32 accumulator ---
        acc = cute.make_fragment(
            tiled_mma.partition_shape_C((self.tile_size_M, self.tile_size_N)),
            Float32,
        )
        acc.fill(0.0)

        # =================================================================
        # K-block 0 from buf 0 (TMA loaded by load warp)
        # =================================================================
        # cpasync W K-block 0
        gW_block0 = cute.local_tile(
            gW_expert, (self.tile_size_N, self.tile_K),
            (tile_N, Int32(0)))
        tWgW0 = thr_copy.partition_S(gW_block0)
        for ci in cutlass.range_constexpr(cute.size(tWsW_cp.shape[2])):
            cute.copy(gmem_tiled_copy, tWgW0[None, None, ci],
                      tWsW_cp[None, None, ci])
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

        for k_block in cutlass.range_constexpr(self.tile_K // 16):
            cute.copy(smem_tiled_copy_A,
                      tAsA_ld_0[None, None, k_block],
                      tCrA_ld[None, None, k_block])
            cute.copy(smem_tiled_copy_B,
                      tBsW_ld[None, None, k_block],
                      tCrB_ld[None, None, k_block])
            cute.gemm(tiled_mma, acc,
                      tCrA[None, None, k_block],
                      tCrB[None, None, k_block], acc)

        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
        if tidx == Int32(0):
            mbarrier_arrive(_bf_0)

        # =================================================================
        # K-block 1 from buf 1 (if exists)
        # =================================================================
        if self.num_k_blocks >= 2:
            # cpasync W K-block 1
            gW_block1 = cute.local_tile(
                gW_expert, (self.tile_size_N, self.tile_K),
                (tile_N, Int32(1)))
            tWgW1 = thr_copy.partition_S(gW_block1)
            for ci in cutlass.range_constexpr(cute.size(tWsW_cp.shape[2])):
                cute.copy(gmem_tiled_copy, tWgW1[None, None, ci],
                          tWsW_cp[None, None, ci])
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            for k_block in cutlass.range_constexpr(self.tile_K // 16):
                cute.copy(smem_tiled_copy_A,
                          tAsA_ld_1[None, None, k_block],
                          tCrA_ld[None, None, k_block])
                cute.copy(smem_tiled_copy_B,
                          tBsW_ld[None, None, k_block],
                          tCrB_ld[None, None, k_block])
                cute.gemm(tiled_mma, acc,
                          tCrA[None, None, k_block],
                          tCrB[None, None, k_block], acc)

            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
            if tidx == Int32(0):
                mbarrier_arrive(_bf_1)

        # =================================================================
        # K-blocks 2+ — wait kblock_ready, cpasync W, process, signal
        # =================================================================
        _kr_phase_0 = Int32(0)
        _kr_phase_1 = Int32(0)
        k_idx = Int32(2)
        while k_idx < Int32(self.num_k_blocks):
            # Wait for store warp's TMA to deliver this x K-block
            if k_idx % Int32(2) == Int32(0):
                mbarrier_wait(_kr_0, _kr_phase_0)
                _kr_phase_0 = _kr_phase_0 ^ Int32(1)
            if k_idx % Int32(2) == Int32(1):
                mbarrier_wait(_kr_1, _kr_phase_1)
                _kr_phase_1 = _kr_phase_1 ^ Int32(1)

            # cpasync W for this K-block
            gW_block = cute.local_tile(
                gW_expert, (self.tile_size_N, self.tile_K),
                (tile_N, k_idx))
            tWgW = thr_copy.partition_S(gW_block)
            for ci in cutlass.range_constexpr(cute.size(tWsW_cp.shape[2])):
                cute.copy(gmem_tiled_copy, tWgW[None, None, ci],
                          tWsW_cp[None, None, ci])
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            # Process current K-block
            if k_idx % Int32(2) == Int32(0):
                for k_block in cutlass.range_constexpr(self.tile_K // 16):
                    cute.copy(smem_tiled_copy_A,
                              tAsA_ld_0[None, None, k_block],
                              tCrA_ld[None, None, k_block])
                    cute.copy(smem_tiled_copy_B,
                              tBsW_ld[None, None, k_block],
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
                              tBsW_ld[None, None, k_block],
                              tCrB_ld[None, None, k_block])
                    cute.gemm(tiled_mma, acc,
                              tCrA[None, None, k_block],
                              tCrB[None, None, k_block], acc)

            # Signal buffer free
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
            if tidx == Int32(0):
                if k_idx % Int32(2) == Int32(0):
                    mbarrier_arrive(_bf_0)
                if k_idx % Int32(2) == Int32(1):
                    mbarrier_arrive(_bf_1)

            k_idx = k_idx + Int32(1)

        # --- Epilogue: R->S (write C to smem for TMA store) ---
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
    # Forward Store: TMA S->G of C
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
        """Compute tile sizes that fit in page_size.

        Constraints:
          - 2 * tile_M * tile_K * elem + tile_N * tile_K * elem <= page_size
          - tile_M * tile_N * elem <= page_size  (C epilogue)
        """
        tile_K = 32
        for tile_M, tile_N in [(128, 128), (128, 64), (64, 64), (64, 32), (32, 32)]:
            data = (2 * tile_M + tile_N) * tile_K * elem_bytes + 32  # +mbar
            c = tile_M * tile_N * elem_bytes
            if max(data, c) <= page_size:
                return tile_M, tile_N, tile_K
        return 32, 32, 16

    @classmethod
    def schedule_forward(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE,
                         **tensors):
        """Schedule forward MoE grouped GEMM.

        Args:
            tile_sizes: Dict with M, N, K keys. Auto-computed if not given.
            page_size: Smem page size in bytes.
            **tensors: sorted_x, w, expert_ids, c tensors.
        """
        ts = dict(tile_sizes or {})
        if "M" not in ts or "N" not in ts:
            x = tensors.get('sorted_x')
            elem_bytes = x.element_size() if x is not None else 2
            auto_M, auto_N, auto_K = cls._auto_tiles(page_size, elem_bytes)
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
        """Return recommended MegakernelConfig for scheduled MoeGemmOps."""
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS
        page_size = ops[0].static_dims.get('page_size', DEFAULT_PAGE_SIZE)
        tile_m = ops[0].tile_sizes.get('M', 64)
        num_mma_warps = tile_m // 16
        threads_per_block = (num_mma_warps + NUM_DMA_WARPS) * 32
        return MegakernelConfig(threads_per_block=threads_per_block,
                                page_size=page_size)


__all__ = ["MoeGemmOp"]
