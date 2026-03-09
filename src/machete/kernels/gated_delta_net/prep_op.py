# Copyright (c) 2025, Machete Authors
"""Gated Delta Net Prep — Megakernel Op (cooperative cpasync + tensor core MMA).

Fuses the 4 prep sub-stages into a single compute() call:
    Phase 1: Prefix sum for g_cumsum (sequential, thread 0)
    Phase 2: K@K^T with MMA + gating/masking on accumulators
    Phase 3: Forward substitution (I+A)^{-1} in fp32 smem (row-by-row)
    Phase 4: A_solved @ k_weighted → w, A_solved @ v_weighted → u (MMA)

Architecture (same as GDNOutputOp / FlashAttentionSm120Op):
    DMA warp:  Idle (no TMA loads or stores)
    MMA warps: All warps cooperatively load data via cpasync AND compute MMA.

Usage:
    from machete.kernels.gated_delta_net.prep_op import GDNPrepOp
    from machete.megakernel import Megakernel

    ops = GDNPrepOp.schedule_forward(
        k=k, v=v, g=g, beta=beta,
        g_cumsum=g_cumsum, w=w, u=u,
    )
    config = GDNPrepOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)
    kernel.run()
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu import warp

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import named_barrier_sync


# Block sizes (matching fla convention)
_BT = 64   # Chunk size
_BK = 64   # K-block
_BV = 64   # V-strip


class GDNPrepOp(Op):
    """Gated Delta Net Prep — cooperative cpasync + MMA megakernel Op.

    Tensors (native [B, T, H, K/V] layout, no transposes):
        k:       (B, T, H, K)  -- keys  (fp16 or bf16)
        v:       (B, T, H, V)  -- values
        g:       (B, T, H)     -- log-space gates (fp32, <= 0)
        beta:    (B, T, H)     -- beta values (fp32, in [0,1])
        g_cumsum:(B, T, H)     -- output: cumulative gates (fp32)
        w:       (B, T, H, K)  -- output: transformed keys
        u:       (B, T, H, V)  -- output: transformed values

    Tiling:
        tile_B=1, tile_H=1 (per batch-head), tile_T=BT=64 (per chunk).
        K and V are looped over in blocks of BK/BV inside compute.
    """

    reads = {
        "k":    (None, ("B", "T", "H", "K")),
        "v":    (None, ("B", "T", "H", "V")),
        "g":    (cutlass.Float32, ("B", "T", "H")),
        "beta": (cutlass.Float32, ("B", "T", "H")),
    }
    writes = {
        "g_cumsum": (cutlass.Float32, ("B", "T", "H")),
        "w":        (None, ("B", "T", "H", "K")),
        "u":        (None, ("B", "T", "H", "V")),
    }
    tile = ("B", "H", "T")

    def __init__(self, **config):
        super().__init__(**config)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)

        assert self.k_dtype in (cutlass.Float16, cutlass.BFloat16), (
            f"GDNPrepOp requires fp16 or bf16, got {self.k_dtype}"
        )
        self.elem_bytes = 2

        # Block sizes (auto-sized from page_size via schedule_forward)
        self.BT = _BT
        self.BK = getattr(self, "BK", _BK)
        self.BV = getattr(self, "BV", _BV)
        self.NK = self.K // self.BK
        self.NV = self.V // self.BV

        assert self.K % self.BK == 0
        assert self.V % self.BV == 0
        assert self.T % self.BT == 0

        # MMA setup
        self.num_mma_warps = self.BT // 16  # 4 warps for BT=64
        self.num_mma_threads = self.num_mma_warps * 32

        # Swizzle for BK=64
        if self.BK >= 64:
            self.swizzle_B = 3
        elif self.BK >= 32:
            self.swizzle_B = 2
        else:
            self.swizzle_B = 1

        # cpasync thread layout for BK-wide loads
        self.async_copy_elems = 128 // (self.elem_bytes * 8)  # 8 for fp16/bf16
        self.k_copy_dim1 = self.BK // self.async_copy_elems
        self.k_copy_dim0 = self.num_mma_threads // self.k_copy_dim1

        # cpasync thread layout for BV-wide loads
        self.v_copy_dim1 = self.BV // self.async_copy_elems
        self.v_copy_dim0 = self.num_mma_threads // self.v_copy_dim1

        # Padded strides for bank-conflict-free shared memory access.
        # stride=BK (128 bytes/row for fp16) aliases all rows to same 32 banks.
        # Padding shifts each row to different banks for conflict-free LdMatrix.
        # s_src/s_srcv use cpasync (128-bit aligned) → pad by 8 (mult of 8).
        # s_a16 uses autovec_copy (no alignment req) → pad by 2 for perfect banking.
        self.BK_PAD = self.BK + 8  # stride 72 → banks cycle with period 8 rows
        self.BV_PAD = self.BV + 8
        self.BT_PAD = self.BT + 2  # stride 66 → each row hits unique bank

        # DMA does nothing — all work in compute
        self.inner_iters = 1
        self.inner_depth = 1

        # Smem layout constants
        # Phase 1-2: g_buf[BT] + beta_buf[BT] + gc_buf[BT] = 768B
        self._gbuf_offset = 0
        self._beta_offset = self.BT * 4
        self._gc_offset = self._beta_offset + self.BT * 4
        # s_k: [BT, BK] swizzled after scalar buffers, 128B aligned
        self._sk_base = ((self._gc_offset + self.BT * 4 + 127) // 128) * 128
        self._sk_base2 = self._sk_base + self.BT * self.BK * self.elem_bytes
        # Phase 3: a_smem[BT, BT] fp32 at sk_base (16KB, reuses s_k region)
        self._a_base = self._sk_base
        # Phase 4: s_a16[BT, BT_PAD] padded fp16 at sk_base (reuses a_smem)
        # s_src[BT, BK_PAD] padded fp16 after a_smem (need fresh region)
        self._src_base = self._a_base + self.BT * self.BT * 4
        self._src_base = ((self._src_base + 127) // 128) * 128
        self._src_base2 = self._src_base + self.BT * max(self.BK_PAD, self.BV_PAD) * self.elem_bytes

        # Override compute method
        self.compute = self.compute_mma

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def _auto_block_sizes(cls, page_size, K, V, elem_bytes=2):
        """Compute largest (BK, BV) that fit in page_size.

        Binding constraint is Phase 4 smem:
            src_base + max(BT*BK, BT*BV) * elem_bytes <= page_size
        where src_base = aligned(sk_base + BT*BT*4).
        """
        BT = _BT
        scalars = 3 * BT * 4
        sk_base = ((scalars + 127) // 128) * 128
        src_base = ((sk_base + BT * BT * 4 + 127) // 128) * 128

        for BK in [128, 64]:
            if K % BK != 0:
                continue
            # Phase 2 (K@K^T) produces [BT, BT] — BK > BT needs
            # non-square B partition which has LdMatrix alignment issues.
            if BK > BT:
                continue
            for BV in [128, 64]:
                if V % BV != 0:
                    continue
                # K-loop double-buffered (2×BK_PAD), V-loop single-buffer (1×BV_PAD)
                # Padding per row for bank-conflict-free smem access
                BK_PAD = BK + 8  # cpasync requires 128-bit aligned rows
                BV_PAD = BV + 8
                phase4 = src_base + max(2 * BT * BK_PAD, BT * BV_PAD) * elem_bytes
                if phase4 <= page_size:
                    return BK, BV
        return 64, 64

    @classmethod
    def schedule_forward(cls, page_size=DEFAULT_PAGE_SIZE, tile_sizes=None, **tensors):
        """Schedule GDN prep Op."""
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("H", 1)
        tile_sizes.setdefault("T", _BT)

        k = tensors.get("k")
        K = k.shape[-1] if k is not None else 64
        v = tensors.get("v")
        V = v.shape[-1] if v is not None else 64
        elem_bytes = k.element_size() if k is not None else 2
        BK, BV = cls._auto_block_sizes(page_size, K, V, elem_bytes)

        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        ops[0].static_dims["BK"] = BK
        ops[0].static_dims["BV"] = BV
        return ops

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for the given scheduled ops."""
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS

        num_mma_warps = _BT // 16  # 4 warps
        threads_per_block = (num_mma_warps + NUM_DMA_WARPS) * 32
        page_size = ops[0].static_dims.get("page_size", DEFAULT_PAGE_SIZE)
        return MegakernelConfig(
            threads_per_block=threads_per_block,
            page_size=page_size,
        )

    # =========================================================================
    # Forward Compute — Cooperative cpasync + Tensor Core MMA
    # =========================================================================

    @cute.jit
    def compute_mma(
        self, page_ptr, tile_B, tile_H, tile_T,
        k, v, g, beta, g_cumsum, w, u,
    ):
        """Cooperative GDN prep: 4 fused sub-stages."""
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        if warp_idx < Int32(self.num_mma_warps):
            self._phase1_cumsum(page_ptr, tidx, tile_B, tile_H, tile_T, g, beta, g_cumsum)
            self._phase2_kkt(page_ptr, tidx, tile_B, tile_H, tile_T, k)
            self._phase3_solve(page_ptr, tidx)
            self._phase4_wu(page_ptr, tidx, tile_B, tile_H, tile_T, k, v, w, u)

    @cute.jit
    def _phase1_cumsum(self, page_ptr, tidx, tile_B, tile_H, tile_T, g, beta, g_cumsum):
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

        chunk_idx = tile_T

        # Global tiles — g, beta, g_cumsum are [B, T, H] with stride (T*H, H, 1)
        g_head_base = tile_B * Int32(self.T * self.H) + tile_H
        gG = cute.make_tensor(g.iterator + g_head_base,
            cute.make_layout((self.T,), stride=(self.H,)))
        gG_tile = cute.local_tile(gG, (self.BT,), (chunk_idx,))

        gBeta = cute.make_tensor(beta.iterator + g_head_base,
            cute.make_layout((self.T,), stride=(self.H,)))
        gBeta_tile = cute.local_tile(gBeta, (self.BT,), (chunk_idx,))

        gGC = cute.make_tensor(g_cumsum.iterator + g_head_base,
            cute.make_layout((self.T,), stride=(self.H,)))
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
    def _phase2_kkt(self, page_ptr, tidx, tile_B, tile_H, tile_T, k):
        """Phase 2: K@K^T with MMA + gating + strictly lower triangular mask.

        Double-buffered: cpasync prefetches k[ki+1] while MMA runs on k[ki].
        Two s_k buffers fit within the 16KB A-matrix region (A written after loop).
        """
        swz = cute.make_swizzle(self.swizzle_B, 4, 3)

        # MMA setup
        mma_op = warp.MmaF16BF16Op(self.k_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.num_mma_warps, 1, 1)),
            permutation_mnk=(self.num_mma_warps * 16, 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        # LdMatrix B
        smem_copy_atom_B = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.k_dtype,
        )
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        # cpasync setup for K loads
        k_async_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), self.k_dtype, num_bits_per_copy=128)
        k_tiled_copy = cute.make_tiled_copy_tv(
            k_async_atom,
            cute.make_layout((self.k_copy_dim0, self.k_copy_dim1),
                             stride=(self.k_copy_dim1, 1)),
            cute.make_layout((1, self.async_copy_elems)),
        )
        k_thr_copy = k_tiled_copy.get_slice(tidx)

        # Two s_k buffers (swizzled) for double-buffering
        sk_offsets = [self._sk_base, self._sk_base2]
        s_k_bufs = []
        tK_s_bufs = []
        tCsK_A_bufs = []
        tKsKt_bufs = []
        for bi in cutlass.range_constexpr(2):
            s_k_bi = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.k_dtype, page_ptr + Int32(sk_offsets[bi]),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.k_dtype,
                ),
                cute.make_layout((self.BT, self.BK), stride=(self.BK, 1)),
            )
            s_k_bufs.append(s_k_bi)
            tK_s_bufs.append(k_thr_copy.partition_D(s_k_bi))
            tCsK_A_bufs.append(thr_mma.partition_A(s_k_bi))
            s_kt_bi = cute.make_tensor(s_k_bi.iterator,
                cute.make_layout((self.BK, self.BT), stride=(1, self.BK)))
            tKsKt_bufs.append(smem_thr_copy_B.partition_S(s_kt_bi))

        # Register fragments (same shape for both buffers)
        s_kt_0 = cute.make_tensor(s_k_bufs[0].iterator,
            cute.make_layout((self.BK, self.BT), stride=(1, self.BK)))
        _tBsKt = thr_mma.partition_B(s_kt_0)
        tCrK_A = tiled_mma.make_fragment_A(tCsK_A_bufs[0])
        tCrKt_B = tiled_mma.make_fragment_B(_tBsKt)
        tKrKt_view = smem_thr_copy_B.retile(tCrKt_B)

        # Identity for coordinate extraction
        mc_AA = cute.make_identity_tensor((self.BT, self.BT))
        tCcAA = thr_mma.partition_C(mc_AA)

        # Global K — [B, T, H, K] with stride (T*H*K, H*K, K, 1)
        chunk_idx = tile_T
        kw_base = (tile_B * Int32(self.T * self.H) + tile_H) * Int32(self.K)
        gK_head = cute.make_tensor(
            (k.iterator + kw_base).align(16),
            cute.make_layout((self.T, self.K), stride=(self.H * self.K, 1)))

        # Scalar buffers for gating
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

        # Accumulator
        acc_A = cute.make_fragment(
            tiled_mma.partition_shape_C((self.BT, self.BT)), Float32)
        acc_A.fill(0.0)

        # Prefetch k[0] -> buf[0]
        gK_tile0 = cute.local_tile(gK_head, (self.BT, self.BK),
                                   (chunk_idx, Int32(0)))
        tK_g0 = k_thr_copy.partition_S(gK_tile0)
        for ci in cutlass.range_constexpr(cute.size(tK_s_bufs[0].shape[2])):
            cute.copy(k_tiled_copy, tK_g0[None, None, ci],
                      tK_s_bufs[0][None, None, ci])
        cute.arch.cp_async_commit_group()

        for ki in cutlass.range_constexpr(self.NK):
            buf = ki % 2
            cute.arch.cp_async_wait_group(0)
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            # Prefetch next tile to other buffer
            if ki + 1 < self.NK:
                next_buf = (ki + 1) % 2
                gK_next = cute.local_tile(gK_head, (self.BT, self.BK),
                                          (chunk_idx, Int32(ki + 1)))
                tK_g_next = k_thr_copy.partition_S(gK_next)
                for ci in cutlass.range_constexpr(cute.size(tK_s_bufs[next_buf].shape[2])):
                    cute.copy(k_tiled_copy, tK_g_next[None, None, ci],
                              tK_s_bufs[next_buf][None, None, ci])
                cute.arch.cp_async_commit_group()

            # MMA: acc_A += K @ K^T
            for kb in cutlass.range_constexpr(self.BK // 16):
                cute.copy(smem_tiled_copy_B, tKsKt_bufs[buf][None, None, kb],
                          tKrKt_view[None, None, kb])
                cute.autovec_copy(tCsK_A_bufs[buf][None, None, kb],
                                  tCrK_A[None, None, kb])
                cute.gemm(tiled_mma, acc_A, tCrK_A[None, None, kb],
                          tCrKt_B[None, None, kb], acc_A)

            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

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

        # Write acc_A fp32 to a_smem for Phase 3
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

        a_smem[BT, BT] initially contains strictly lower triangular A.
        After this phase, a_smem contains (I+A)^{-1}.

        Each of BT threads handles one column. Row-by-row processing ensures
        data dependencies are respected via barriers between rows.

        CRITICAL: Barriers must be outside the `if tidx < BT` guard so ALL
        num_mma_threads participate. Otherwise deadlock.
        """
        a_smem = cute.make_tensor(
            cute.make_ptr(Float32, page_ptr + Int32(self._a_base),
                          cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((self.BT, self.BT), stride=(self.BT, 1)),
        )

        # Row 0: (I+A)^{-1}[0,:] = I[0,:] since A[0,:] = 0 (strictly lower tri)
        if tidx < Int32(self.BT):
            col = tidx
            if col == Int32(0):
                a_smem[Int32(0), col] = Float32(1.0)
            else:
                a_smem[Int32(0), col] = Float32(0.0)

        # Process rows 1..BT-1. Barrier OUTSIDE the tidx guard so all MMA threads hit it.
        for ri in cutlass.range_constexpr(self.BT - 1):
            row = ri + 1
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            if tidx < Int32(self.BT):
                col = tidx
                # result[row, col] = I[row, col] - sum_{m<row} A[row,m] * result[m, col]
                acc_val = Float32(0.0)
                for mi in cutlass.range_constexpr(row):
                    acc_val = acc_val + a_smem[Int32(row), Int32(mi)] * a_smem[Int32(mi), col]

                i_val = Float32(0.0)
                if col == Int32(row):
                    i_val = Float32(1.0)
                a_smem[Int32(row), col] = i_val - acc_val

        # Final barrier — all threads must see completed result
        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

    @cute.jit
    def _phase4_wu(self, page_ptr, tidx, tile_B, tile_H, tile_T, k, v, w, u):
        """Phase 4: A_solved @ k_weighted -> w, A_solved @ v_weighted -> u.

        Double-buffered cpasync: prefetch next tile while computing current tile.
        Two s_src buffers at _src_base and _src_base2.
        """
        chunk_idx = tile_T

        # Read A_solved from fp32 smem
        a_smem = cute.make_tensor(
            cute.make_ptr(Float32, page_ptr + Int32(self._a_base),
                          cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout((self.BT, self.BT), stride=(self.BT, 1)),
        )

        # Scalar buffers (still live from Phase 1)
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

        # MMA setup
        mma_op = warp.MmaF16BF16Op(self.k_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.num_mma_warps, 1, 1)),
            permutation_mnk=(self.num_mma_warps * 16, 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        # LdMatrix B
        smem_copy_atom_B = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.k_dtype,
        )
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        # s_a16: [BT, BT] fp16 at sk_base — padded stride for bank-conflict-free access
        s_a16 = cute.make_tensor(
            cute.make_ptr(self.k_dtype, page_ptr + Int32(self._sk_base),
                          cute.AddressSpace.smem, assumed_align=128),
            cute.make_layout((self.BT, self.BT), stride=(self.BT_PAD, 1)),
        )

        # Convert A_solved fp32 -> fp16 in s_a16, thread-parallel
        _total_a = self.BT * self.BT
        _nthreads = self.num_mma_threads
        _elems_per_thread = _total_a // _nthreads
        for ei in cutlass.range_constexpr(_elems_per_thread):
            flat_idx = tidx + Int32(ei * _nthreads)
            row_a = flat_idx // Int32(self.BT)
            col_a = flat_idx % Int32(self.BT)
            val_f32 = a_smem[row_a, col_a]
            s_a16[row_a, col_a] = val_f32.to(self.k_dtype)
        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

        # MMA partitions: A_solved (s_a16) as A operand
        tCsA16 = thr_mma.partition_A(s_a16)
        tCrA16 = tiled_mma.make_fragment_A(tCsA16)

        # cpasync for BK-wide loads
        k_async_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), self.k_dtype, num_bits_per_copy=128)
        k_tiled_copy = cute.make_tiled_copy_tv(
            k_async_atom,
            cute.make_layout((self.k_copy_dim0, self.k_copy_dim1),
                             stride=(self.k_copy_dim1, 1)),
            cute.make_layout((1, self.async_copy_elems)),
        )
        k_thr_copy = k_tiled_copy.get_slice(tidx)

        # Two s_src buffers for double-buffered K loads (padded stride)
        src_offsets = [self._src_base, self._src_base2]
        s_src_bufs = []
        tSrc_s_bufs = []
        tSsSrc_bufs = []
        for bi in cutlass.range_constexpr(2):
            s_src_bi = cute.make_tensor(
                cute.make_ptr(self.k_dtype, page_ptr + Int32(src_offsets[bi]),
                              cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.BT, self.BK), stride=(self.BK_PAD, 1)),
            )
            s_src_bufs.append(s_src_bi)
            tSrc_s_bufs.append(k_thr_copy.partition_D(s_src_bi))
            s_src_B_bi = cute.make_tensor(s_src_bi.iterator,
                cute.make_layout((self.BK, self.BT), stride=(1, self.BK_PAD)))
            tSsSrc_bufs.append(smem_thr_copy_B.partition_S(s_src_B_bi))

        # Register fragments for B (same shape for both buffers)
        s_src_B_0 = cute.make_tensor(s_src_bufs[0].iterator,
            cute.make_layout((self.BK, self.BT), stride=(1, self.BK_PAD)))
        _tBsSrc = thr_mma.partition_B(s_src_B_0)
        tCrSrc_B = tiled_mma.make_fragment_B(_tBsSrc)
        tSrSrc_view = smem_thr_copy_B.retile(tCrSrc_B)

        # Identity for coordinates
        mc_out = cute.make_identity_tensor((self.BT, self.BK))
        tCcOut = thr_mma.partition_C(mc_out)

        # Global tensors — [B, T, H, K/V] with H-strided per-head access
        kw_base = (tile_B * Int32(self.T * self.H) + tile_H) * Int32(self.K)
        gK_head = cute.make_tensor(
            (k.iterator + kw_base).align(16),
            cute.make_layout((self.T, self.K), stride=(self.H * self.K, 1)))

        gW_head = cute.make_tensor(
            (w.iterator + kw_base).align(16),
            cute.make_layout((self.T, self.K), stride=(self.H * self.K, 1)))

        # ---- Phase 4a: w = A_solved @ k_weighted (double-buffered) ----
        # Prefetch k[0] -> buf[0]
        gK_tile0 = cute.local_tile(gK_head, (self.BT, self.BK),
                                   (chunk_idx, Int32(0)))
        tK_g0 = k_thr_copy.partition_S(gK_tile0)
        for ci in cutlass.range_constexpr(cute.size(tSrc_s_bufs[0].shape[2])):
            cute.copy(k_tiled_copy, tK_g0[None, None, ci],
                      tSrc_s_bufs[0][None, None, ci])
        cute.arch.cp_async_commit_group()

        for ki in cutlass.range_constexpr(self.NK):
            # Extract buffer refs before dynamic code (avoid SSA wrapping of Python ints)
            _cur_src = s_src_bufs[ki % 2]
            _cur_tSsSrc = tSsSrc_bufs[ki % 2]

            cute.arch.cp_async_wait_group(0)
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            # Prefetch next K-tile to other buffer
            if ki + 1 < self.NK:
                _next_tSrc = tSrc_s_bufs[(ki + 1) % 2]
                gK_next = cute.local_tile(gK_head, (self.BT, self.BK),
                                          (chunk_idx, Int32(ki + 1)))
                tK_g_next = k_thr_copy.partition_S(gK_next)
                for ci in cutlass.range_constexpr(cute.size(_next_tSrc.shape[2])):
                    cute.copy(k_tiled_copy, tK_g_next[None, None, ci],
                              _next_tSrc[None, None, ci])
                cute.arch.cp_async_commit_group()

            # Apply weighting in-place on current buffer
            for ei in cutlass.range_constexpr(_elems_per_thread):
                flat_idx = tidx + Int32(ei * _nthreads)
                row_s = flat_idx // Int32(self.BK)
                col_s = flat_idx % Int32(self.BK)
                if row_s < Int32(self.BT):
                    old_val = _cur_src[row_s, col_s].to(Float32)
                    weight = beta_buf[row_s] * cute.math.exp(gc_buf[row_s], fastmath=True)
                    _cur_src[row_s, col_s] = (old_val * weight).to(self.k_dtype)
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            # MMA: acc_w = A_solved @ k_weighted  [BT,BT] x [BT,BK] -> [BT,BK]
            acc_w = cute.make_fragment(
                tiled_mma.partition_shape_C((self.BT, self.BK)), Float32)
            acc_w.fill(0.0)

            for kb in cutlass.range_constexpr(self.BT // 16):
                cute.copy(smem_tiled_copy_B, _cur_tSsSrc[None, None, kb],
                          tSrSrc_view[None, None, kb])
                cute.autovec_copy(tCsA16[None, None, kb], tCrA16[None, None, kb])
                cute.gemm(tiled_mma, acc_w, tCrA16[None, None, kb],
                          tCrSrc_B[None, None, kb], acc_w)

            # Write w[BT, BK] to global
            gW_tile = cute.local_tile(gW_head, (self.BT, self.BK),
                                      (chunk_idx, Int32(ki)))
            for ci in cutlass.range_constexpr(cute.size(acc_w)):
                row_w = tCcOut[ci][0]
                col_w = tCcOut[ci][1]
                gW_tile[Int32(row_w), Int32(col_w)] = acc_w[ci].to(self.k_dtype)

            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

        # ---- Phase 4b: u = A_solved @ v_weighted (double-buffered) ----
        v_async_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), self.k_dtype, num_bits_per_copy=128)
        v_tiled_copy = cute.make_tiled_copy_tv(
            v_async_atom,
            cute.make_layout((self.v_copy_dim0, self.v_copy_dim1),
                             stride=(self.v_copy_dim1, 1)),
            cute.make_layout((1, self.async_copy_elems)),
        )
        v_thr_copy = v_tiled_copy.get_slice(tidx)

        # Two s_src_v buffers for double-buffered V loads (padded stride, reuse src regions)
        s_srcv_bufs = []
        tSrcV_s_bufs = []
        tSsSrcV_bufs = []
        for bi in cutlass.range_constexpr(2):
            s_srcv_bi = cute.make_tensor(
                cute.make_ptr(self.k_dtype, page_ptr + Int32(src_offsets[bi]),
                              cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.BT, self.BV), stride=(self.BV_PAD, 1)),
            )
            s_srcv_bufs.append(s_srcv_bi)
            tSrcV_s_bufs.append(v_thr_copy.partition_D(s_srcv_bi))
            s_srcv_B_bi = cute.make_tensor(s_srcv_bi.iterator,
                cute.make_layout((self.BV, self.BT), stride=(1, self.BV_PAD)))
            tSsSrcV_bufs.append(smem_thr_copy_B.partition_S(s_srcv_B_bi))

        # Register fragments for V B operand
        s_srcv_B_0 = cute.make_tensor(s_srcv_bufs[0].iterator,
            cute.make_layout((self.BV, self.BT), stride=(1, self.BV_PAD)))
        _tBsSrcV = thr_mma.partition_B(s_srcv_B_0)
        tCrSrcV_B = tiled_mma.make_fragment_B(_tBsSrcV)
        tSrSrcV_view = smem_thr_copy_B.retile(tCrSrcV_B)

        # Identity for V-dim output coordinates
        mc_outv = cute.make_identity_tensor((self.BT, self.BV))
        tCcOutV = thr_mma.partition_C(mc_outv)

        _elems_per_thread_v = (self.BT * self.BV) // _nthreads

        uv_base = (tile_B * Int32(self.T * self.H) + tile_H) * Int32(self.V)
        gV_head = cute.make_tensor(
            (v.iterator + uv_base).align(16),
            cute.make_layout((self.T, self.V), stride=(self.H * self.V, 1)))

        gU_head = cute.make_tensor(
            (u.iterator + uv_base).align(16),
            cute.make_layout((self.T, self.V), stride=(self.H * self.V, 1)))

        # Prefetch v[0] -> buf[0]
        gV_tile0 = cute.local_tile(gV_head, (self.BT, self.BV),
                                   (chunk_idx, Int32(0)))
        tV_g0 = v_thr_copy.partition_S(gV_tile0)
        for ci in cutlass.range_constexpr(cute.size(tSrcV_s_bufs[0].shape[2])):
            cute.copy(v_tiled_copy, tV_g0[None, None, ci],
                      tSrcV_s_bufs[0][None, None, ci])
        cute.arch.cp_async_commit_group()

        for vi in cutlass.range_constexpr(self.NV):
            # Extract buffer refs before dynamic code (avoid SSA wrapping)
            _cur_srcv = s_srcv_bufs[vi % 2]
            _cur_tSsSrcV = tSsSrcV_bufs[vi % 2]

            cute.arch.cp_async_wait_group(0)
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            # Prefetch next V-tile to other buffer
            if vi + 1 < self.NV:
                _next_tSrcV = tSrcV_s_bufs[(vi + 1) % 2]
                gV_next = cute.local_tile(gV_head, (self.BT, self.BV),
                                          (chunk_idx, Int32(vi + 1)))
                tV_g_next = v_thr_copy.partition_S(gV_next)
                for ci in cutlass.range_constexpr(cute.size(_next_tSrcV.shape[2])):
                    cute.copy(v_tiled_copy, tV_g_next[None, None, ci],
                              _next_tSrcV[None, None, ci])
                cute.arch.cp_async_commit_group()

            # Apply weighting: v_weighted = v * beta
            for ei in cutlass.range_constexpr(_elems_per_thread_v):
                flat_idx = tidx + Int32(ei * _nthreads)
                row_s = flat_idx // Int32(self.BV)
                col_s = flat_idx % Int32(self.BV)
                if row_s < Int32(self.BT):
                    old_val = _cur_srcv[row_s, col_s].to(Float32)
                    weight = beta_buf[row_s]
                    _cur_srcv[row_s, col_s] = (old_val * weight).to(self.k_dtype)
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            # MMA: acc_u = A_solved @ v_weighted  [BT,BT] x [BT,BV] -> [BT,BV]
            acc_u = cute.make_fragment(
                tiled_mma.partition_shape_C((self.BT, self.BV)), Float32)
            acc_u.fill(0.0)

            for kb in cutlass.range_constexpr(self.BT // 16):
                cute.copy(smem_tiled_copy_B, _cur_tSsSrcV[None, None, kb],
                          tSrSrcV_view[None, None, kb])
                cute.autovec_copy(tCsA16[None, None, kb], tCrA16[None, None, kb])
                cute.gemm(tiled_mma, acc_u, tCrA16[None, None, kb],
                          tCrSrcV_B[None, None, kb], acc_u)

            # Write u[BT, BV] to global
            gU_tile = cute.local_tile(gU_head, (self.BT, self.BV),
                                      (chunk_idx, Int32(vi)))
            for ci in cutlass.range_constexpr(cute.size(acc_u)):
                row_u = tCcOutV[ci][0]
                col_u = tCcOutV[ci][1]
                gU_tile[Int32(row_u), Int32(col_u)] = acc_u[ci].to(self.k_dtype)

            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))


__all__ = ["GDNPrepOp"]
