# Copyright (c) 2025, Machete Authors
"""Gated Delta Net Solve Op — cumsum + K@K^T + forward substitution.

Phases 1-3 of the prep stage, extracted for lower register pressure:
    Phase 1: Prefix sum for g_cumsum (sequential, thread 0)
    Phase 2: K@K^T with MMA + gating/masking on accumulators
    Phase 3: Forward substitution (I+A)^{-1} in fp32 smem (row-by-row)
    Phase 3b: Convert fp32 a_solved → fp16, write to global

Architecture:
    DMA warp:  TMA loads k sub-blocks into double buffer.
    MMA warps: All warps cooperatively compute phases 1-3.

Usage:
    from machete.kernels.gated_delta_net.solve_op import GDNSolveOp
    from machete.megakernel import Megakernel

    ops = GDNSolveOp.schedule(
        k=k, g=g, beta=beta,
        g_cumsum=g_cumsum, a_solved=a_solved,
    )
    config = GDNSolveOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)
    kernel.run()
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu import warp

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import named_barrier_sync


_BT = 64   # Chunk size (fixed by algorithm)
_BK = 64   # Default K-block


class GDNSolveOp(Op):
    """Gated Delta Net Solve — cumsum + K@K^T + forward substitution.

    Tensors (native [B, S, NH, K] layout):
        k:        (B, S, NH, K)     -- keys (fp16 or bf16)
        g:        (B, S, NH)        -- log-space gates (fp32, <= 0)
        beta:     (B, S, NH)        -- beta values (fp32, in [0,1])
        g_cumsum: (B, S, NH)        -- output: cumulative gates (fp32)
        a_solved: (B, S, NH, BT)    -- output: solved matrix rows (fp16/bf16)

    Tiling:
        tile_B=1, tile_NH=1, tile_S=BT=64.
        K is looped over in blocks of BK inside MMA (Phase 2).
    """

    reads = {
        "k":    (None, ("B", "S", "NH", "K")),
        "g":    (cutlass.Float32, ("B", "S", "NH")),
        "beta": (cutlass.Float32, ("B", "S", "NH")),
    }
    writes = {
        "g_cumsum": (cutlass.Float32, ("B", "S", "NH")),
        "a_solved": (None, ("B", "S", "NH", "BT_DIM")),
    }
    tile = ("B", "NH", "S")
    tma_loads = {"k"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name != "k":
            return None
        BK = static_dims.get("BK", _BK)
        return (_BT, tile_sizes.get("NH", 1), BK)

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape,
                                 tile_sizes, static_dims):
        if tensor_name != "k":
            return None
        BK = static_dims.get("BK", _BK)
        # SW128 swizzle for bank-conflict-free LdMatrix reads in Phase 2.
        return (
            f"cute.make_composed_layout("
            f"cute.make_swizzle(3, 4, 3), 0, "
            f"cute.make_layout(({BK}, 1, {_BT}), "
            f"stride=(1, {BK}, {BK})))"
        )

    def __init__(self, **config):
        super().__init__(**config)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)

        assert self.k_dtype in (cutlass.Float16, cutlass.BFloat16), (
            f"GDNSolveOp requires fp16 or bf16, got {self.k_dtype}"
        )
        self.elem_bytes = 2

        self.BT = _BT
        self.BK = getattr(self, "BK", _BK)
        self.NK = self.K // self.BK

        assert self.K % self.BK == 0
        assert self.S % self.BT == 0

        # MMA setup
        self.num_mma_warps = self.BT // 16  # 4 warps for BT=64
        self.num_mma_threads = self.num_mma_warps * 32

        # DMA warp loads k via TMA in NK sub-blocks, double-buffered
        self.inner_iters = max(1, self.NK - 1)
        self.inner_depth = 1

        # Per-buffer: [BT, BK] with SW128 swizzle
        self._k_buf_bytes = self.BT * self.BK * self.elem_bytes
        self._buf_stride = self._k_buf_bytes
        self._tma_k_blocks = min(2, self.NK)
        self._k_tma_bytes = self._tma_k_blocks * self._k_buf_bytes

        # Scalars after K double-buffer region
        k_bufs_end = 2 * self._buf_stride
        self._gbuf_offset = k_bufs_end
        self._beta_offset = self._gbuf_offset + self.BT * 4
        self._gc_offset = self._beta_offset + self.BT * 4

        # Phase 3: a_smem[BT, BT] fp32 at offset 0 (reuses K buffer area)
        self._a_base = 0

        # Override compute method
        self.compute = self.compute_mma

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def _auto_block_sizes(cls, page_size, K, elem_bytes=2):
        """Compute largest BK that fits in page_size.

        Binding constraints (phases reuse memory):
            Phase 1-2: 2 × [BT, BK] TMA bufs + scalars <= page_size
            Phase 3:   [BT, BT] fp32 = 16KB <= page_size
        """
        BT = _BT
        scalars = 3 * BT * 4  # g_buf + beta_buf + gc_buf
        a_smem = BT * BT * 4  # 16KB fixed

        if a_smem > page_size:
            raise ValueError(
                f"page_size={page_size} too small for a_smem ({a_smem}B). "
                f"Minimum page_size for GDNSolveOp is {a_smem}B."
            )

        for BK in [128, 64]:
            if K % BK != 0:
                continue
            if BK > BT:
                continue
            k_bufs = 2 * BT * BK * elem_bytes
            phase12 = k_bufs + scalars
            if phase12 <= page_size:
                return BK
        raise ValueError(
            f"page_size={page_size} too small for GDNSolveOp (K={K})."
        )

    @classmethod
    def schedule(cls, page_size=DEFAULT_PAGE_SIZE, tile_sizes=None, **tensors):
        """Schedule GDN solve Op."""
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("NH", 1)
        tile_sizes.setdefault("S", _BT)

        k = tensors.get("k")
        K = k.shape[-1] if k is not None else 64
        elem_bytes = k.element_size() if k is not None else 2
        BK = cls._auto_block_sizes(page_size, K, elem_bytes)

        S = k.shape[1] if k is not None else 64
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        ops[0].static_dims["BK"] = BK
        ops[0].static_dims["K"] = K
        ops[0].static_dims["S"] = S
        return ops

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig."""
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS

        num_mma_warps = _BT // 16
        threads_per_block = (num_mma_warps + NUM_DMA_WARPS) * 32
        page_size = max(
            op.static_dims.get("page_size", DEFAULT_PAGE_SIZE) for op in ops
        )
        return MegakernelConfig(
            threads_per_block=threads_per_block,
            page_size=page_size,
        )

    # =========================================================================
    # Load (DMA warp: TMA k into page)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_NH, tile_S,
             k_tma, k_tma_gmem, work_mbar, inner_iter_idx):
        """TMA k load: NK sub-blocks [BT, BK] into double buffer."""
        from machete.megakernel.interpreter import mbarrier_arrive_expect_tx

        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        swz = cute.make_swizzle(3, 4, 3)  # SW128

        merged_t = tile_B * Int32(self.S // self.BT) + tile_S

        if inner_iter_idx == Int32(0):
            nbytes = Int32(self._k_tma_bytes)
            with cute.arch.elect_one():
                mbarrier_arrive_expect_tx(work_mbar, nbytes)

            for _k in cutlass.range_constexpr(self._tma_k_blocks):
                _buf_base = page_ptr + Int32(_k * self._buf_stride)
                sK = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(self.k_dtype, _buf_base,
                                      cute.AddressSpace.smem),
                        swz, dtype=self.k_dtype),
                    cute.make_layout((self.BK, 1, self.BT),
                                     stride=(1, self.BK, self.BK)),
                )
                gK = cute.local_tile(
                    k_tma_gmem,
                    (self.BK, 1, self.BT),
                    (None, None, None),
                )
                tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
                    k_tma, Int32(0), cute.make_layout(1),
                    cute.group_modes(sK, 0, 3), cute.group_modes(gK, 0, 3),
                )
                cute.copy(k_tma, tKgK[(None, Int32(_k), tile_NH, merged_t)],
                          tKsK, tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Compute — Phases 1-3 + a_solved global write
    # =========================================================================

    @cute.jit
    def compute_mma(
        self, page_ptr, tile_B, tile_NH, tile_S,
        k, g, beta, g_cumsum, a_solved,
    ):
        """Phases 1-3: cumsum, K@K^T, forward solve, write a_solved."""
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        if warp_idx < Int32(self.num_mma_warps):
            self._phase1_cumsum(page_ptr, tidx, tile_B, tile_NH, tile_S,
                                g, beta, g_cumsum)
            self._phase2_kkt(page_ptr, tidx)
            self._phase3_solve(page_ptr, tidx)
            self._phase3b_write_a_solved(page_ptr, tidx, tile_B, tile_NH,
                                         tile_S, a_solved)

    @cute.jit
    def _phase1_cumsum(self, page_ptr, tidx, tile_B, tile_NH, tile_S,
                       g, beta, g_cumsum):
        """Phase 1: Load g + beta, prefix sum -> g_cumsum, write global."""
        g_buf = cute.make_tensor(
            cute.make_ptr(Float32, page_ptr + Int32(self._gbuf_offset),
                          cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout(self.BT),
        )
        beta_buf = cute.make_tensor(
            cute.make_ptr(Float32, page_ptr + Int32(self._beta_offset),
                          cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout(self.BT),
        )
        gc_buf = cute.make_tensor(
            cute.make_ptr(Float32, page_ptr + Int32(self._gc_offset),
                          cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout(self.BT),
        )

        chunk_idx = tile_S

        g_head_base = tile_B * Int32(self.S * self.NH) + tile_NH
        gG = cute.make_tensor(g.iterator + g_head_base,
            cute.make_layout((self.S,), stride=(self.NH,)))
        gG_tile = cute.local_tile(gG, (self.BT,), (chunk_idx,))

        gBeta = cute.make_tensor(beta.iterator + g_head_base,
            cute.make_layout((self.S,), stride=(self.NH,)))
        gBeta_tile = cute.local_tile(gBeta, (self.BT,), (chunk_idx,))

        gGC = cute.make_tensor(g_cumsum.iterator + g_head_base,
            cute.make_layout((self.S,), stride=(self.NH,)))
        gGC_tile = cute.local_tile(gGC, (self.BT,), (chunk_idx,))

        # Load g and beta
        if tidx < Int32(self.BT):
            g_buf[tidx] = gG_tile[tidx]
            beta_buf[tidx] = gBeta_tile[tidx]
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

        # Inclusive prefix sum (thread 0)
        if tidx == Int32(0):
            gc_buf[Int32(0)] = g_buf[Int32(0)]
            for si in cutlass.range_constexpr(self.BT - 1):
                gc_buf[Int32(si + 1)] = gc_buf[Int32(si)] + g_buf[Int32(si + 1)]
        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

        # Write g_cumsum to global
        if tidx < Int32(self.BT):
            gGC_tile[tidx] = gc_buf[tidx]

    @cute.jit
    def _phase2_kkt(self, page_ptr, tidx):
        """Phase 2: K@K^T with MMA + gating + strictly lower triangular mask."""
        mma_op = warp.MmaF16BF16Op(self.k_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.num_mma_warps, 1, 1)),
            permutation_mnk=(self.num_mma_warps * 16, 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        swz = cute.make_swizzle(3, 4, 3)

        smem_copy_atom_A = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.k_dtype)
        smem_tiled_copy_A = cute.make_tiled_copy_A(smem_copy_atom_A, tiled_mma)
        smem_thr_copy_A = smem_tiled_copy_A.get_slice(tidx)

        smem_copy_atom_B = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.k_dtype)
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        sK_0 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.k_dtype, page_ptr,
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.k_dtype),
            cute.make_layout((self.BT, self.BK), stride=(self.BK, 1)),
        )
        sK_1 = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.k_dtype,
                              page_ptr + Int32(self._buf_stride),
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.k_dtype),
            cute.make_layout((self.BT, self.BK), stride=(self.BK, 1)),
        )

        tCsK_A = thr_mma.partition_A(sK_0)
        tCsK_B = thr_mma.partition_B(sK_0)
        tCrK_A = tiled_mma.make_fragment_A(tCsK_A)
        tCrK_B = tiled_mma.make_fragment_B(tCsK_B)
        tCrK_A_ld = smem_thr_copy_A.retile(tCrK_A)
        tCrK_B_ld = smem_thr_copy_B.retile(tCrK_B)

        tAsK_ld_0 = smem_thr_copy_A.partition_S(sK_0)
        tBsK_ld_0 = smem_thr_copy_B.partition_S(sK_0)
        tAsK_ld_1 = smem_thr_copy_A.partition_S(sK_1)
        tBsK_ld_1 = smem_thr_copy_B.partition_S(sK_1)

        mc_AA = cute.make_identity_tensor((self.BT, self.BT))
        tCcAA = thr_mma.partition_C(mc_AA)

        gc_buf = cute.make_tensor(
            cute.make_ptr(Float32, page_ptr + Int32(self._gc_offset),
                          cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout(self.BT),
        )
        beta_buf = cute.make_tensor(
            cute.make_ptr(Float32, page_ptr + Int32(self._beta_offset),
                          cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout(self.BT),
        )

        acc_A = cute.make_fragment(
            tiled_mma.partition_shape_C((self.BT, self.BT)), Float32)
        acc_A.fill(0.0)

        # K-block 0 from buf 0
        for kb in cutlass.range_constexpr(self.BK // 16):
            cute.copy(smem_tiled_copy_A, tAsK_ld_0[None, None, kb],
                      tCrK_A_ld[None, None, kb])
            cute.copy(smem_tiled_copy_B, tBsK_ld_0[None, None, kb],
                      tCrK_B_ld[None, None, kb])
            cute.gemm(tiled_mma, acc_A, tCrK_A[None, None, kb],
                      tCrK_B[None, None, kb], acc_A)

        # K-block 1 from buf 1 (if NK >= 2)
        if self.NK >= 2:
            for kb in cutlass.range_constexpr(self.BK // 16):
                cute.copy(smem_tiled_copy_A, tAsK_ld_1[None, None, kb],
                          tCrK_A_ld[None, None, kb])
                cute.copy(smem_tiled_copy_B, tBsK_ld_1[None, None, kb],
                          tCrK_B_ld[None, None, kb])
                cute.gemm(tiled_mma, acc_A, tCrK_A[None, None, kb],
                          tCrK_B[None, None, kb], acc_A)

        # Gating + beta + strictly lower triangular mask
        for ci in cutlass.range_constexpr(cute.size(acc_A)):
            row = tCcAA[ci][0]
            col = tCcAA[ci][1]
            if Int32(row) > Int32(col):
                g_row = gc_buf[Int32(row)]
                g_col = gc_buf[Int32(col)]
                b_row = beta_buf[Int32(row)]
                gate = cute.math.exp(g_row - g_col, fastmath=True)
                acc_A[ci] = acc_A[ci] * b_row * gate
            else:
                acc_A[ci] = Float32(0.0)

        # Write to a_smem fp32
        a_smem = cute.make_tensor(
            cute.make_ptr(Float32, page_ptr + Int32(self._a_base),
                          cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((self.BT, self.BT), stride=(self.BT, 1)),
        )

        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

        for ci in cutlass.range_constexpr(cute.size(acc_A)):
            row = tCcAA[ci][0]
            col = tCcAA[ci][1]
            a_smem[Int32(row), Int32(col)] = acc_A[ci]

        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

    @cute.jit
    def _phase3_solve(self, page_ptr, tidx):
        """Phase 3: Forward substitution (I+A)^{-1} in-place in fp32 smem.

        CRITICAL: Barriers outside `if tidx < BT` guard for all MMA threads.
        """
        a_smem = cute.make_tensor(
            cute.make_ptr(Float32, page_ptr + Int32(self._a_base),
                          cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((self.BT, self.BT), stride=(self.BT, 1)),
        )

        # Row 0: identity
        if tidx < Int32(self.BT):
            col = tidx
            if col == Int32(0):
                a_smem[Int32(0), col] = Float32(1.0)
            else:
                a_smem[Int32(0), col] = Float32(0.0)

        # Rows 1..BT-1
        for ri in cutlass.range_constexpr(self.BT - 1):
            row = ri + 1
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            if tidx < Int32(self.BT):
                col = tidx
                acc_val = Float32(0.0)
                for mi in cutlass.range_constexpr(row):
                    acc_val = acc_val + a_smem[Int32(row), Int32(mi)] * a_smem[Int32(mi), col]

                i_val = Float32(0.0)
                if col == Int32(row):
                    i_val = Float32(1.0)
                a_smem[Int32(row), col] = i_val - acc_val

        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

    @cute.jit
    def _phase3b_write_a_solved(self, page_ptr, tidx, tile_B, tile_NH,
                                tile_S, a_solved):
        """Phase 3b: Convert a_smem fp32 → fp16 and write to global a_solved.

        a_solved is [B, S, NH, BT_DIM=64] with stride (S*NH*BT, NH*BT, BT, 1).
        Each tile writes [BT, BT] = one chunk's solved matrix.
        """
        a_smem = cute.make_tensor(
            cute.make_ptr(Float32, page_ptr + Int32(self._a_base),
                          cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((self.BT, self.BT), stride=(self.BT, 1)),
        )

        chunk_idx = tile_S

        # Global a_solved tensor: [B, S, NH, BT_DIM]
        # Stride: (S*NH*BT, NH*BT, BT, 1) — row-major
        a_base = (tile_B * Int32(self.S * self.NH * self.BT)
                  + chunk_idx * Int32(self.BT * self.NH * self.BT)
                  + tile_NH * Int32(self.BT))
        gA = cute.make_tensor(
            (a_solved.iterator + a_base).align(16),
            cute.make_layout((self.BT, self.BT),
                             stride=(self.NH * self.BT, 1)),
        )

        # Thread-parallel fp32→fp16 conversion + global write
        _total = self.BT * self.BT
        _nthreads = self.num_mma_threads
        _elems_per = _total // _nthreads
        for ei in cutlass.range_constexpr(_elems_per):
            flat_idx = tidx + Int32(ei * _nthreads)
            row = flat_idx // Int32(self.BT)
            col = flat_idx % Int32(self.BT)
            val = a_smem[row, col]
            gA[row, col] = val.to(self.k_dtype)


__all__ = ["GDNSolveOp"]
