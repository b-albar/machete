# Copyright (c) 2025, Machete Authors
"""
MoE Grouped GEMM Backward (dx) Op for the Megakernel.

Computes dx[i] = dc[i] @ w[expert_ids[i]] for each sorted token position i.
This is the input gradient for the MoE forward pass c = sorted_x @ w^T.

Architecture (SM_90+ / Hopper+):
    - DMA warp:  TMA G->S of dc N-blocks (double-buffered)
    - MMA warps: cpasync-load expert weight tiles from w + tensor core MMA
    - Store warp: TMA S->G of output dx

Key difference from forward MoeGemmOp:
    - Forward: C[M,N] = X[M,K] @ W[N,K]^T,  tile(M,N), K-loop
    - Backward: dx[M,K] = dc[M,N] @ W[N,K],  tile(M,K), N-loop
    - Weight W is loaded the same way (cpasync), but used with transposed
      LdMatrix since the inner product dimension is now N (not K).

Tensor layout:
    dc:          [total_padded, N]     — gradient of loss w.r.t. c
    w:           [num_experts, N, K]   — expert weights (same as forward)
    expert_ids:  [total_padded]        — expert ID per token (int32)
    dx:          [total_padded, K]     — gradient of loss w.r.t. sorted_x

Pipelined phases:
    load:    TMA G->S of dc N-blocks into double buffer
    compute: Read expert_id from global, cpasync w[expert_id] tile,
             LdMatrix (transpose for B) + MMA per N-block, epilogue R->S
    store:   TMA store S->G of dx[tile_M, tile_K]

Page layout (2-stage double buffer):
    [buf0: dc_tile (tile_M × tile_N)] [buf1: dc_tile] [w_buf (tile_N × tile_K)]
    [mbar: 4 × 8B]
    Epilogue: [dx: tile_M × tile_K] (reuses page from offset 0)
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


class MoeGemmBwdOp(Op):
    """Backward dx for Grouped GEMM (Mixture-of-Experts).

    Computes dx[i] = dc[i] @ w[expert_ids[i]] — the input gradient.
    Each M-tile uses a different expert weight matrix determined by expert_ids.
    dc is TMA-loaded by the DMA warp; w is cpasync-loaded by MMA warps.
    """

    reads = {
        "dc":         (None, ("M", "N")),
        "w":          (None, ("E", "N", "K")),
        "expert_ids": (cutlass.Int32, ("M",)),
    }
    writes = {"dx": (None, ("M", "K"))}
    tile = ("M", "K")

    tma_loads = {"dc"}
    tma_stores = {"dx"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        """Custom TMA tile shapes for inner N-block sub-tiling."""
        tile_N = static_dims.get("tile_N", 32)
        if tensor_name == "dc":
            return (tile_sizes["M"], tile_N)
        else:  # "dx"
            return (tile_sizes["M"], tile_sizes["K"])

    def __init__(self, **config):
        super().__init__(**config)

        if self.dc_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            raise ValueError(
                f"MoeGemmBwdOp requires fp16 or bf16, got {self.dc_dtype}"
            )

        self.tile_N = getattr(self, 'tile_N', 32)
        self.page_size = getattr(self, 'page_size', DEFAULT_PAGE_SIZE)

        # Tile bytes
        self.dc_tile_bytes = self.tile_size_M * self.tile_N * self.elem_bytes
        self.w_tile_bytes = self.tile_N * self.tile_size_K * self.elem_bytes
        self.dx_tile_bytes = self.tile_size_M * self.tile_size_K * self.elem_bytes

        assert self.tile_N >= 16 and self.tile_N % 16 == 0, (
            f"MoeGemmBwdOp: tile_N={self.tile_N} must be >= 16 and multiple of 16."
        )

        self.num_n_blocks = (self.N + self.tile_N - 1) // self.tile_N
        self.num_mma_warps = self.threads_per_row // 32
        self.num_mma_threads = self.num_mma_warps * 32

        # DMA TMA-loads first 2 N-blocks of dc
        self.tma_n_blocks = min(2, self.num_n_blocks)
        self.inner_iters = max(1, self.num_n_blocks - 1)

        # Swizzle for dc (TMA-loaded, tile_N contiguous in smem)
        if self.tile_N % 64 == 0 and self.tile_N >= 64:
            self.swz_dc = 3
        elif self.tile_N % 32 == 0:
            self.swz_dc = 2
        else:
            self.swz_dc = 1

        # Swizzle for w (cpasync-loaded, tile_K contiguous in smem)
        if self.tile_size_K % 64 == 0 and self.tile_size_K >= 64:
            self.swz_w = 3
        elif self.tile_size_K % 32 == 0:
            self.swz_w = 2
        else:
            self.swz_w = 1

        # Smem layout:
        #   [buf0: dc_tile] [buf1: dc_tile] [w_buf: w_tile] [4 mbarriers]
        self.dc_buf_stride = self.dc_tile_bytes
        self.w_offset = 2 * self.dc_tile_bytes
        self.mbar_offset = self.w_offset + self.w_tile_bytes
        self.mbar_bytes = 32
        self.dc_tma_bytes = self.dc_tile_bytes

        # cpasync setup for weight loading
        self.async_copy_elems = 128 // (self.elem_bytes * 8)
        # Copy along K (contiguous) for w[tile_N, tile_K]
        self.w_copy_dim1 = self.tile_size_K // self.async_copy_elems
        self.w_copy_dim0 = self.num_mma_threads // self.w_copy_dim1
        assert self.w_copy_dim0 > 0, (
            f"MoeGemmBwdOp: not enough threads ({self.num_mma_threads}) for "
            f"tile_K={self.tile_size_K} cpasync (need >= {self.w_copy_dim1})"
        )

        # Weight stride: w is [E, N, K], stride_E = N * K
        self.w_stride_E = self.N * self.K

        # Verify smem fits
        data_bytes = 2 * self.dc_tile_bytes + self.w_tile_bytes + self.mbar_bytes
        total_smem = max(data_bytes, self.dx_tile_bytes)
        assert total_smem <= self.page_size, (
            f"MoeGemmBwdOp: smem {total_smem}B exceeds page_size ({self.page_size}B). "
            f"tile_M={self.tile_size_M}, tile_K={self.tile_size_K}, tile_N={self.tile_N}"
        )

        self.compute = self.compute_mma

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape,
                                tile_sizes, static_dims):
        """Return swizzled smem layout code for TMA descriptor creation."""
        if tensor_name == "dc":
            tile_N = static_dims.get("tile_N", 32)
            if tile_N % 64 == 0 and tile_N >= 64:
                B = 3
            elif tile_N % 32 == 0:
                B = 2
            else:
                B = 1
            dim0, dim1 = tma_tile_shape
            return (
                f"cute.make_composed_layout("
                f"cute.make_swizzle({B}, 4, 3), 0, "
                f"cute.make_layout(({dim0}, {dim1}), stride=(1, {dim0})))"
            )
        return None  # No swizzle for dx

    # =========================================================================
    # Load: TMA G->S of dc (called by load/store warps)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_M, tile_K,
             dc_tma, dc_tma_gmem,
             work_mbar, inner_iter_idx):
        """TMA load of dc N-blocks.

        iter 0 (load warp):   Init mbarriers, TMA N0+N1 → buf0+buf1.
        iter 1+ (store warp): Wait buf_free, TMA one N-block → buf.
        """
        swz = cute.make_swizzle(self.swz_dc, 4, 3)

        _bf_0 = page_ptr + Int32(self.mbar_offset)
        _bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        _nr_0 = page_ptr + Int32(self.mbar_offset + 16)
        _nr_1 = page_ptr + Int32(self.mbar_offset + 24)

        if inner_iter_idx == Int32(0):
            with cute.arch.elect_one():
                mbarrier_init(_bf_0, Int32(1))
                mbarrier_init(_bf_1, Int32(1))
                mbarrier_init(_nr_0, Int32(1))
                mbarrier_init(_nr_1, Int32(1))
            mbarrier_init_fence()

            mbar_ptr = cute.make_ptr(
                cutlass.Int64, work_mbar, cute.AddressSpace.smem)
            nbytes = Int32(self.tma_n_blocks * self.dc_tma_bytes)
            with cute.arch.elect_one():
                mbarrier_arrive_expect_tx(work_mbar, nbytes)

            for _n in cutlass.range_constexpr(self.tma_n_blocks):
                _buf_base = page_ptr + Int32(_n * self.dc_buf_stride)
                sDC_ptr = cute.recast_ptr(
                    cute.make_ptr(self.dc_dtype, _buf_base,
                                  cute.AddressSpace.smem),
                    swz, dtype=self.dc_dtype)
                sDC = cute.make_tensor(
                    sDC_ptr,
                    cute.make_layout((self.tile_N, self.tile_size_M),
                                     stride=(1, self.tile_N)))
                gDC = cute.local_tile(
                    dc_tma_gmem, (self.tile_N, self.tile_size_M),
                    (None, None))
                tDCsDC, tDCgDC = cute.nvgpu.cpasync.tma_partition(
                    dc_tma, Int32(0), cute.make_layout(1),
                    cute.group_modes(sDC, 0, 2),
                    cute.group_modes(gDC, 0, 2))

                cute.copy(dc_tma, tDCgDC[(None, Int32(_n), tile_M)],
                          tDCsDC, tma_bar_ptr=mbar_ptr)

        if inner_iter_idx > Int32(0):
            _n_block = inner_iter_idx + Int32(1)
            _buf_idx = _n_block % Int32(2)

            _bf_phase = ((inner_iter_idx - Int32(1)) // Int32(2)) % Int32(2)
            if _buf_idx == Int32(0):
                mbarrier_wait(_bf_0, _bf_phase)
            if _buf_idx == Int32(1):
                mbarrier_wait(_bf_1, _bf_phase)

            _buf_base = _buf_idx * Int32(self.dc_buf_stride) + page_ptr
            _nr_ptr = cute.make_ptr(
                cutlass.Int64, _nr_0, cute.AddressSpace.smem)
            if _buf_idx == Int32(0):
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(_nr_0, Int32(self.dc_tma_bytes))
            if _buf_idx == Int32(1):
                _nr_ptr = cute.make_ptr(
                    cutlass.Int64, _nr_1, cute.AddressSpace.smem)
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(_nr_1, Int32(self.dc_tma_bytes))

            sDC_ptr = cute.recast_ptr(
                cute.make_ptr(self.dc_dtype, _buf_base,
                              cute.AddressSpace.smem),
                swz, dtype=self.dc_dtype)
            sDC = cute.make_tensor(
                sDC_ptr,
                cute.make_layout((self.tile_N, self.tile_size_M),
                                 stride=(1, self.tile_N)))
            gDC = cute.local_tile(
                dc_tma_gmem, (self.tile_N, self.tile_size_M),
                (None, None))
            tDCsDC, tDCgDC = cute.nvgpu.cpasync.tma_partition(
                dc_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sDC, 0, 2),
                cute.group_modes(gDC, 0, 2))

            cute.copy(dc_tma, tDCgDC[(None, _n_block, tile_M)],
                      tDCsDC, tma_bar_ptr=_nr_ptr)

    # =========================================================================
    # Compute: cpasync W + transposed MMA for dx
    # =========================================================================

    @cute.jit
    def compute_mma(self, page_ptr, tile_M, tile_K,
                    dc, w, expert_ids, dx):
        """Backward dx compute with cpasync weight loading.

        For each N-block:
        1. Wait for dc N-block (TMA-loaded by DMA/store warp)
        2. Read expert_id from expert_ids via global pointer arithmetic
        3. cpasync w[expert_id, N_block, K_tile] → smem w_buf
        4. LdMatrix (A=dc non-transpose, B=w transpose) + MMA
        After all N-blocks: epilogue converts fp32 acc → output dtype in smem.
        """
        tidx = cute.arch.thread_idx()[0]

        # --- Build tiled MMA ---
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            self.dc_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.num_mma_warps, 1, 1)),
            permutation_mnk=(self.num_mma_warps * 16, 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        # --- Swizzles ---
        swz_dc = cute.make_swizzle(self.swz_dc, 4, 3)
        swz_w = cute.make_swizzle(self.swz_w, 4, 3)

        # --- LdMatrix tiled copies ---
        # A = dc: non-transposed, (tile_M, tile_N) with tile_N contiguous
        smem_copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=False, num_matrices=4), self.dc_dtype)
        smem_tiled_copy_A = cute.make_tiled_copy_A(
            smem_copy_atom_A, tiled_mma)
        smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)

        # B = w: transposed, viewed as (tile_K, tile_N) from (tile_N, tile_K) data
        smem_copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=True, num_matrices=4), self.w_dtype)
        smem_tiled_copy_B = cute.make_tiled_copy_B(
            smem_copy_atom_B, tiled_mma)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        # --- dc smem tensors (buf 0 and 1, swizzled) ---
        sDC_0 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.dc_dtype, page_ptr,
                              cute.AddressSpace.smem, assumed_align=128),
                swz_dc, dtype=self.dc_dtype),
            cute.make_layout((self.tile_size_M, self.tile_N),
                             stride=(self.tile_N, 1)),
        )
        sDC_1 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.dc_dtype,
                              page_ptr + Int32(self.dc_buf_stride),
                              cute.AddressSpace.smem, assumed_align=128),
                swz_dc, dtype=self.dc_dtype),
            cute.make_layout((self.tile_size_M, self.tile_N),
                             stride=(self.tile_N, 1)),
        )

        # --- w smem tensor: transposed view (tile_K, tile_N) for B operand ---
        # Physical data is (tile_N, tile_K) with tile_K contiguous.
        # Transposed view: (tile_K, tile_N) with stride=(1, tile_K).
        sW = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.w_dtype,
                              page_ptr + Int32(self.w_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                swz_w, dtype=self.w_dtype),
            cute.make_layout((self.tile_size_K, self.tile_N),
                             stride=(1, self.tile_size_K)),
        )

        # MMA partitions and register fragments
        tCsA = thr_mma.partition_A(sDC_0)
        tCsB = thr_mma.partition_B(sW)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrA_ld = smem_thr_copy_A.retile(tCrA)
        tCrB_ld = smem_thr_copy_B.retile(tCrB)

        # Per-buffer LdMatrix smem partitions for dc
        tAsA_ld_0 = smem_thr_copy_A.partition_S(sDC_0)
        tAsA_ld_1 = smem_thr_copy_A.partition_S(sDC_1)
        # W partition (single buffer)
        tBsW_ld = smem_thr_copy_B.partition_S(sW)

        # --- cpasync setup for weight loading ---
        async_copy_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), self.w_dtype,
            num_bits_per_copy=128)
        copy_thread_layout = cute.make_layout(
            (self.w_copy_dim0, self.w_copy_dim1),
            stride=(self.w_copy_dim1, 1))
        copy_value_layout = cute.make_layout((1, self.async_copy_elems))
        gmem_tiled_copy = cute.make_tiled_copy_tv(
            async_copy_atom, copy_thread_layout, copy_value_layout)
        thr_copy = gmem_tiled_copy.get_slice(tidx)

        # cpasync smem destination for W: physical layout (tile_N, tile_K)
        sW_cp = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.w_dtype,
                              page_ptr + Int32(self.w_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                swz_w, dtype=self.w_dtype),
            cute.make_layout((self.tile_N, self.tile_size_K),
                             stride=(self.tile_size_K, 1)),
        )
        tWsW_cp = thr_copy.partition_D(sW_cp)

        # --- Read expert_id for this M-tile from global memory ---
        eid_ptr = (expert_ids.iterator
                   + tile_M * Int32(self.tile_size_M)).align(4)
        g_eid = cute.make_tensor(eid_ptr, cute.make_layout(1))
        expert_id = g_eid[Int32(0)].to(Int32)

        # Compute base pointer into w for this expert: w[eid] is [N, K]
        w_expert_ptr = (w.iterator
                        + expert_id * Int32(self.w_stride_E)).align(16)
        gW_expert = cute.make_tensor(
            w_expert_ptr,
            cute.make_layout((self.N, self.K), stride=(self.K, 1)))

        # Op-managed mbarriers
        _bf_0 = page_ptr + Int32(self.mbar_offset)
        _bf_1 = page_ptr + Int32(self.mbar_offset + 8)
        _nr_0 = page_ptr + Int32(self.mbar_offset + 16)
        _nr_1 = page_ptr + Int32(self.mbar_offset + 24)

        # --- Init fp32 accumulator ---
        acc = cute.make_fragment(
            tiled_mma.partition_shape_C(
                (self.tile_size_M, self.tile_size_K)),
            Float32,
        )
        acc.fill(0.0)

        # =================================================================
        # N-block 0 from buf 0 (TMA loaded by load warp)
        # =================================================================
        # cpasync W N-block 0: load w[eid, 0:tile_N, K_tile]
        gW_block0 = cute.local_tile(
            gW_expert, (self.tile_N, self.tile_size_K),
            (Int32(0), tile_K))
        tWgW0 = thr_copy.partition_S(gW_block0)
        for ci in cutlass.range_constexpr(cute.size(tWsW_cp.shape[2])):
            cute.copy(gmem_tiled_copy, tWgW0[None, None, ci],
                      tWsW_cp[None, None, ci])
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

        for n_sub in cutlass.range_constexpr(self.tile_N // 16):
            cute.copy(smem_tiled_copy_A,
                      tAsA_ld_0[None, None, n_sub],
                      tCrA_ld[None, None, n_sub])
            cute.copy(smem_tiled_copy_B,
                      tBsW_ld[None, None, n_sub],
                      tCrB_ld[None, None, n_sub])
            cute.gemm(tiled_mma, acc,
                      tCrA[None, None, n_sub],
                      tCrB[None, None, n_sub], acc)

        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
        if tidx == Int32(0):
            mbarrier_arrive(_bf_0)

        # =================================================================
        # N-block 1 from buf 1 (if exists)
        # =================================================================
        if self.num_n_blocks >= 2:
            gW_block1 = cute.local_tile(
                gW_expert, (self.tile_N, self.tile_size_K),
                (Int32(1), tile_K))
            tWgW1 = thr_copy.partition_S(gW_block1)
            for ci in cutlass.range_constexpr(cute.size(tWsW_cp.shape[2])):
                cute.copy(gmem_tiled_copy, tWgW1[None, None, ci],
                          tWsW_cp[None, None, ci])
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            for n_sub in cutlass.range_constexpr(self.tile_N // 16):
                cute.copy(smem_tiled_copy_A,
                          tAsA_ld_1[None, None, n_sub],
                          tCrA_ld[None, None, n_sub])
                cute.copy(smem_tiled_copy_B,
                          tBsW_ld[None, None, n_sub],
                          tCrB_ld[None, None, n_sub])
                cute.gemm(tiled_mma, acc,
                          tCrA[None, None, n_sub],
                          tCrB[None, None, n_sub], acc)

            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
            if tidx == Int32(0):
                mbarrier_arrive(_bf_1)

        # =================================================================
        # N-blocks 2+ — wait nblock_ready, cpasync W, process, signal
        # =================================================================
        _nr_phase_0 = Int32(0)
        _nr_phase_1 = Int32(0)
        n_idx = Int32(2)
        while n_idx < Int32(self.num_n_blocks):
            if n_idx % Int32(2) == Int32(0):
                mbarrier_wait(_nr_0, _nr_phase_0)
                _nr_phase_0 = _nr_phase_0 ^ Int32(1)
            if n_idx % Int32(2) == Int32(1):
                mbarrier_wait(_nr_1, _nr_phase_1)
                _nr_phase_1 = _nr_phase_1 ^ Int32(1)

            # cpasync W for this N-block
            gW_block = cute.local_tile(
                gW_expert, (self.tile_N, self.tile_size_K),
                (n_idx, tile_K))
            tWgW = thr_copy.partition_S(gW_block)
            for ci in cutlass.range_constexpr(cute.size(tWsW_cp.shape[2])):
                cute.copy(gmem_tiled_copy, tWgW[None, None, ci],
                          tWsW_cp[None, None, ci])
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            if n_idx % Int32(2) == Int32(0):
                for n_sub in cutlass.range_constexpr(self.tile_N // 16):
                    cute.copy(smem_tiled_copy_A,
                              tAsA_ld_0[None, None, n_sub],
                              tCrA_ld[None, None, n_sub])
                    cute.copy(smem_tiled_copy_B,
                              tBsW_ld[None, None, n_sub],
                              tCrB_ld[None, None, n_sub])
                    cute.gemm(tiled_mma, acc,
                              tCrA[None, None, n_sub],
                              tCrB[None, None, n_sub], acc)
            if n_idx % Int32(2) == Int32(1):
                for n_sub in cutlass.range_constexpr(self.tile_N // 16):
                    cute.copy(smem_tiled_copy_A,
                              tAsA_ld_1[None, None, n_sub],
                              tCrA_ld[None, None, n_sub])
                    cute.copy(smem_tiled_copy_B,
                              tBsW_ld[None, None, n_sub],
                              tCrB_ld[None, None, n_sub])
                    cute.gemm(tiled_mma, acc,
                              tCrA[None, None, n_sub],
                              tCrB[None, None, n_sub], acc)

            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
            if tidx == Int32(0):
                if n_idx % Int32(2) == Int32(0):
                    mbarrier_arrive(_bf_0)
                if n_idx % Int32(2) == Int32(1):
                    mbarrier_arrive(_bf_1)

            n_idx = n_idx + Int32(1)

        # --- Epilogue: R->S (write dx to smem for TMA store) ---
        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
        sDX = cute.make_tensor(
            cute.make_ptr(self.dx_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.tile_size_M, self.tile_size_K),
                             stride=(self.tile_size_K, 1)),
        )
        tCsDX = thr_mma.partition_C(sDX)
        for ci in cutlass.range_constexpr(cute.size(acc)):
            tCsDX[ci] = acc[ci].to(self.dx_dtype)

    # =========================================================================
    # Store: TMA S->G of dx
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_M, tile_K, dx_tma, dx_tma_gmem):
        """TMA store of dx from shared to global memory."""
        sDX = cute.make_tensor(
            cute.make_ptr(self.dx_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.tile_size_K, self.tile_size_M)),
        )

        gDX = cute.local_tile(
            dx_tma_gmem, (self.tile_size_K, self.tile_size_M), (None, None),
        )
        tDXsDX, tDXgDX = cute.nvgpu.cpasync.tma_partition(
            dx_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sDX, 0, 2),
            cute.group_modes(gDX, 0, 2),
        )

        with cute.arch.elect_one():
            cute.copy(dx_tma, tDXsDX, tDXgDX[(None, tile_K, tile_M)])

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def _auto_tiles(cls, page_size, elem_bytes=2):
        """Compute tile sizes that fit in page_size.

        Constraints:
          - 2 * tile_M * tile_N * elem + tile_N * tile_K * elem <= page_size
          - tile_M * tile_K * elem <= page_size  (dx epilogue)
        """
        tile_N = 32
        for tile_M, tile_K in [(128, 64), (64, 64), (64, 32), (32, 32)]:
            data = (2 * tile_M + tile_K) * tile_N * elem_bytes + 32
            c = tile_M * tile_K * elem_bytes
            if max(data, c) <= page_size:
                return tile_M, tile_K, tile_N
        return 32, 32, 16

    @classmethod
    def schedule_backward(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE,
                          **tensors):
        """Schedule backward MoE grouped GEMM (dx computation).

        Args:
            tile_sizes: Dict with M, K keys (output tile dims).
                        Auto-computed if not given.
            page_size: Smem page size in bytes.
            **tensors: dc, w, expert_ids, dx tensors.
        """
        ts = dict(tile_sizes or {})
        if "M" not in ts or "K" not in ts:
            x = tensors.get('dc')
            elem_bytes = x.element_size() if x is not None else 2
            auto_M, auto_K, auto_N = cls._auto_tiles(page_size, elem_bytes)
            ts.setdefault("M", auto_M)
            ts.setdefault("K", auto_K)
            ts.setdefault("N", auto_N)
        tile_N = ts.pop("N", 32)
        scheduled = cls._schedule_single(tile_sizes=ts, **tensors)
        scheduled.static_dims["tile_N"] = tile_N
        scheduled.static_dims["page_size"] = page_size
        return [scheduled]

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for scheduled MoeGemmBwdOps."""
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS
        page_size = ops[0].static_dims.get('page_size', DEFAULT_PAGE_SIZE)
        tile_m = ops[0].tile_sizes.get('M', 64)
        num_mma_warps = tile_m // 16
        threads_per_block = (num_mma_warps + NUM_DMA_WARPS) * 32
        return MegakernelConfig(threads_per_block=threads_per_block,
                                page_size=page_size)


__all__ = ["MoeGemmBwdOp"]
