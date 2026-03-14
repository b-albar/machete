# Copyright (c) 2025, Machete Authors
"""Gated Delta Net WU Op — A_solved @ k_weighted -> w, A_solved @ v_weighted -> u.

Phase 4 of the prep stage, extracted for lower register pressure:
    - TMA loads a_solved [BT, BT] fp16 (plain layout, 8KB) + k [BT, BK] compact
    - cpasync double-buffered v loads
    - MMA: A_solved @ (beta * exp(gc) * k) -> w  (k weighted in-place in compact smem)
    - MMA: A_solved @ (beta * v) -> u              (v weighted in-place in padded smem)

Architecture:
    DMA warp:  TMA loads a_solved + K sub-blocks into page.
    MMA warps: cpasync v loads + MMA compute.

Usage:
    from machete.kernels.gated_delta_net.wu_op import GDNWUOp
    from machete.megakernel import Megakernel

    ops = GDNWUOp.schedule_forward(
        a_solved=a_solved, k=k, v=v,
        g_cumsum=g_cumsum, beta=beta,
        w=w, u=u,
    )
    config = GDNWUOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)
    kernel.run()
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu import warp

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import (
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_arrive,
    mbarrier_arrive_expect_tx,
    mbarrier_wait,
    named_barrier_sync,
)


_BT = 64   # Chunk size (fixed by algorithm)
_BK = 64   # Default K-block
_BV = 64   # Default V-block
_MBAR_BYTES = 32  # 4 mbarriers × 8 bytes (buf_free×2 + kblock_ready×2)


class GDNWUOp(Op):
    """Gated Delta Net WU — A_solved @ weighted k/v via MMA.

    Tensors (native [B, S, NH, K/V] layout):
        a_solved: (B, S, NH, BT_DIM=64) -- solved matrix (fp16/bf16, from SolveOp)
        k:        (B, S, NH, K)          -- keys (fp16/bf16)
        v:        (B, S, NH, V)          -- values (fp16/bf16)
        g_cumsum: (B, S, NH)             -- cumulative gates (fp32)
        beta:     (B, S, NH)             -- beta values (fp32)
        w:        (B, S, NH, K)          -- output: transformed keys
        u:        (B, S, NH, V)          -- output: transformed values

    Tiling:
        tile_B=1, tile_NH=1, tile_S=BT=64.
        K looped via TMA double-buffer (compact layout for in-place weighting),
        V looped via cpasync double-buffer (padded layout).
    """

    reads = {
        "a_solved": (None, ("B", "S", "NH", "BT_DIM")),
        "k":        (None, ("B", "S", "NH", "K")),
        "v":        (None, ("B", "S", "NH", "V")),
        "g_cumsum": (cutlass.Float32, ("B", "S", "NH")),
        "beta":     (cutlass.Float32, ("B", "S", "NH")),
    }
    writes = {
        "w": (None, ("B", "S", "NH", "K")),
        "u": (None, ("B", "S", "NH", "V")),
    }
    tile = ("B", "NH", "S")
    tma_loads = {"a_solved", "k"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name == "a_solved":
            return (_BT, tile_sizes.get("NH", 1), _BT)
        if tensor_name == "k":
            BK = static_dims.get("BK", _BK)
            return (_BT, tile_sizes.get("NH", 1), BK)
        return None

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape,
                                 tile_sizes, static_dims):
        if tensor_name == "a_solved":
            return (
                f"cute.make_layout(({_BT}, 1, {_BT}), "
                f"stride=(1, {_BT}, {_BT}))"
            )
        if tensor_name == "k":
            BK = static_dims.get("BK", _BK)
            # Compact layout (no swizzle) — allows in-place weighting writes
            return (
                f"cute.make_layout(({BK}, 1, {_BT}), "
                f"stride=(1, {BK}, {BK}))"
            )
        return None

    def __init__(self, **config):
        super().__init__(**config)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)

        assert self.k_dtype in (cutlass.Float16, cutlass.BFloat16), (
            f"GDNWUOp requires fp16 or bf16, got {self.k_dtype}"
        )
        self.elem_bytes = 2

        self.BT = _BT
        self.BK = getattr(self, "BK", _BK)
        self.BV = getattr(self, "BV", _BV)
        self.NK = self.K // self.BK
        self.NV = self.V // self.BV

        assert self.K % self.BK == 0
        assert self.V % self.BV == 0
        assert self.S % self.BT == 0

        # MMA setup
        self.num_mma_warps = self.BT // 16  # 4 warps for BT=64
        self.num_mma_threads = self.num_mma_warps * 32

        # cpasync thread layouts (for v only now; k via TMA)
        self.async_copy_elems = 128 // (self.elem_bytes * 8)  # 8 for fp16
        self.v_copy_dim1 = self.BV // self.async_copy_elems
        self.v_copy_dim0 = self.num_mma_threads // self.v_copy_dim1

        # Padded stride for V cpasync (bank-conflict-free)
        self.BV_PAD = self.BV + 8

        # Smem layout:
        #   a_solved [BT, BT] fp16 from TMA at offset 0 (8KB)
        #   gc_buf [BT] fp32 + beta_buf [BT] fp32
        #   K-phase: k_buf×1-2 [BT,BK] TMA compact + 4 mbarriers (32B)
        #   V-phase: v_buf×1-2 [BT,BV_PAD] cpasync padded (reuses K area)
        self._a_bytes = self.BT * self.BT * self.elem_bytes

        scalars_start = ((self._a_bytes + 15) // 16) * 16
        self._gc_offset = scalars_start
        self._beta_offset = self._gc_offset + self.BT * 4

        scalars_end = self._beta_offset + self.BT * 4
        self._src_base = ((scalars_end + 127) // 128) * 128

        # K TMA buffers (compact, no swizzle — writable for weighting)
        k_buf_bytes = self.BT * self.BK * self.elem_bytes
        k_double_total = self._src_base + 2 * k_buf_bytes + _MBAR_BYTES
        self._single_buf_k = k_double_total > self.page_size

        self._k_buf0_offset = self._src_base
        self._k_buf_stride = k_buf_bytes
        if self._single_buf_k:
            self._k_buf1_offset = self._k_buf0_offset
            self._tma_k_blocks = min(1, self.NK)
            k_region_end = self._k_buf0_offset + k_buf_bytes
        else:
            self._k_buf1_offset = self._src_base + k_buf_bytes
            self._tma_k_blocks = min(2, self.NK)
            k_region_end = self._k_buf1_offset + k_buf_bytes
        self._tma_k_bytes = self._tma_k_blocks * k_buf_bytes
        self._mbar_offset = ((k_region_end + 7) // 8) * 8

        # DMA warp loads a_solved + K via TMA
        self.inner_iters = max(1, self.NK - self._tma_k_blocks + 1)
        self.inner_depth = 1

        # V cpasync buffers (padded, reuse same smem area)
        v_buf_bytes = self.BT * self.BV_PAD * self.elem_bytes
        self._v_buf0_offset = self._src_base
        v_double_end = self._src_base + 2 * v_buf_bytes
        self._double_buf_v = v_double_end <= self.page_size
        if self._double_buf_v:
            self._v_buf1_offset = self._src_base + v_buf_bytes
        else:
            self._v_buf1_offset = self._src_base  # single-buf fallback

        # Override compute method
        self.compute = self.compute_mma

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def _auto_block_sizes(cls, page_size, K, V, elem_bytes=2):
        """Compute largest (BK, BV) that fit in page_size.

        Smem layout (two phases sharing src area):
            a_solved [BT, BT] fp16           8KB
            gc_buf + beta_buf                512B
            K-phase: 1-2 × [BT, BK] + mbar  (1 or 2)*BT*BK*2 + 32
            V-phase: 1-2 × [BT, BV_PAD]     (1 or 2)*BT*(BV+8)*2
        """
        BT = _BT
        a_bytes = BT * BT * elem_bytes
        scalars_start = ((a_bytes + 15) // 16) * 16
        scalars_end = scalars_start + 2 * BT * 4  # gc + beta
        src_base = ((scalars_end + 127) // 128) * 128

        for BK in [128, 64, 32]:
            if K % BK != 0 or BK > BT:
                continue
            for BV in [128, 64, 32]:
                if V % BV != 0 or BV > BT:
                    continue
                BV_PAD = BV + 8
                k_double = src_base + 2 * BT * BK * elem_bytes + _MBAR_BYTES
                k_single = src_base + BT * BK * elem_bytes + _MBAR_BYTES
                v_double = src_base + 2 * BT * BV_PAD * elem_bytes
                v_single = src_base + BT * BV_PAD * elem_bytes
                # Try double-buf K + double-buf V
                if max(k_double, v_double) <= page_size:
                    return BK, BV
                # Try double-buf K + single-buf V
                if max(k_double, v_single) <= page_size:
                    return BK, BV
                # Try single-buf K + double-buf V
                if max(k_single, v_double) <= page_size:
                    return BK, BV
                # Try single-buf K + single-buf V
                if max(k_single, v_single) <= page_size:
                    return BK, BV
        raise ValueError(
            f"page_size={page_size} too small for GDNWUOp (K={K}, V={V})."
        )

    @classmethod
    def schedule_forward(cls, page_size=DEFAULT_PAGE_SIZE, tile_sizes=None, **tensors):
        """Schedule GDN WU Op."""
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("NH", 1)
        tile_sizes.setdefault("S", _BT)

        k = tensors.get("k")
        K = k.shape[-1] if k is not None else 64
        v = tensors.get("v")
        V = v.shape[-1] if v is not None else 64
        elem_bytes = k.element_size() if k is not None else 2
        BK, BV = cls._auto_block_sizes(page_size, K, V, elem_bytes)

        S = k.shape[1] if k is not None else 64
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        ops[0].static_dims["BK"] = BK
        ops[0].static_dims["BV"] = BV
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
    # Load (DMA warp: TMA a_solved + K into page)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_NH, tile_S,
             a_solved_tma, a_solved_tma_gmem,
             k_tma, k_tma_gmem,
             work_mbar, inner_iter_idx):
        """TMA a_solved + K load.

        iter 0 (load warp):  TMA a_solved + K[0..min(2,NK)-1], init mbarriers.
        iter 1+ (store warp): Wait buf_free[buf], TMA K[k_block],
                              signal kblock_ready[buf].
        """
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)

        _bf_0 = page_ptr + Int32(self._mbar_offset)
        _bf_1 = page_ptr + Int32(self._mbar_offset + 8)
        _kr_0 = page_ptr + Int32(self._mbar_offset + 16)
        _kr_1 = page_ptr + Int32(self._mbar_offset + 24)

        merged_t = tile_B * Int32(self.S // self.BT) + tile_S

        if inner_iter_idx == Int32(0):
            # Init op-managed mbarriers
            with cute.arch.elect_one():
                mbarrier_init(_bf_0, Int32(1))
                mbarrier_init(_bf_1, Int32(1))
                mbarrier_init(_kr_0, Int32(1))
                mbarrier_init(_kr_1, Int32(1))
            mbarrier_init_fence()

            # Signal expected TX bytes: a_solved + K[0..min(2,NK)-1]
            nbytes = Int32(self._a_bytes + self._tma_k_bytes)
            with cute.arch.elect_one():
                mbarrier_arrive_expect_tx(work_mbar, nbytes)

            # TMA a_solved [BT, BT]
            sA = cute.make_tensor(
                cute.make_ptr(self.k_dtype, page_ptr,
                              cute.AddressSpace.smem),
                cute.make_layout((self.BT, 1, self.BT),
                                 stride=(1, self.BT, self.BT)),
            )
            gA = cute.local_tile(
                a_solved_tma_gmem,
                (self.BT, 1, self.BT),
                (None, None, None),
            )
            tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                a_solved_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sA, 0, 3), cute.group_modes(gA, 0, 3),
            )
            cute.copy(a_solved_tma, tAgA[(None, Int32(0), tile_NH, merged_t)],
                      tAsA, tma_bar_ptr=mbar_ptr)

            # TMA K[0..min(2,NK)-1] into compact double buffer
            for _k in cutlass.range_constexpr(self._tma_k_blocks):
                _buf_base = page_ptr + Int32(self._k_buf0_offset + _k * self._k_buf_stride)
                sK = cute.make_tensor(
                    cute.make_ptr(self.k_dtype, _buf_base,
                                  cute.AddressSpace.smem),
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

        if inner_iter_idx > Int32(0):
            # Load K[k_block] into freed buffer
            _k_block = inner_iter_idx + Int32(self._tma_k_blocks - 1)
            # Initialize before control flow (CuTe DSL: no constexpr-if)
            _buf_base = page_ptr + Int32(self._k_buf0_offset)
            _kr_mbar = _kr_0

            if self._single_buf_k:
                # Single-buf: always buf 0, phase alternates every iteration
                _bf_phase = (inner_iter_idx - Int32(1)) % Int32(2)
                mbarrier_wait(_bf_0, _bf_phase)
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(_kr_0, Int32(self._k_buf_stride))
            else:
                # Double-buf: alternate between buf 0 and buf 1
                _buf_idx = _k_block % Int32(2)
                _bf_phase = ((inner_iter_idx - Int32(1)) // Int32(2)) % Int32(2)
                if _buf_idx == Int32(0):
                    mbarrier_wait(_bf_0, _bf_phase)
                if _buf_idx == Int32(1):
                    mbarrier_wait(_bf_1, _bf_phase)

                if _buf_idx == Int32(1):
                    _buf_base = page_ptr + Int32(self._k_buf1_offset)
                if _buf_idx == Int32(1):
                    _kr_mbar = _kr_1
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(_kr_mbar, Int32(self._k_buf_stride))

            _kr_ptr = cute.make_ptr(cutlass.Int64, _kr_mbar,
                                    cute.AddressSpace.smem)
            sK = cute.make_tensor(
                cute.make_ptr(self.k_dtype, _buf_base,
                              cute.AddressSpace.smem),
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
            cute.copy(k_tma, tKgK[(None, _k_block, tile_NH, merged_t)],
                      tKsK, tma_bar_ptr=_kr_ptr)

    # =========================================================================
    # Compute — w = A_solved @ k_weighted, u = A_solved @ v_weighted
    # =========================================================================

    @cute.jit
    def compute_mma(
        self, page_ptr, tile_B, tile_NH, tile_S,
        a_solved, k, v, g_cumsum, beta, w, u,
    ):
        """Phase 4: A_solved @ weighted k -> w, A_solved @ weighted v -> u."""
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        if warp_idx < Int32(self.num_mma_warps):
            self._load_scalars(page_ptr, tidx, tile_B, tile_NH, tile_S,
                               g_cumsum, beta)
            self._compute_w(page_ptr, tidx, tile_B, tile_NH, tile_S, w)
            self._compute_u(page_ptr, tidx, tile_B, tile_NH, tile_S, v, u)

    @cute.jit
    def _load_scalars(self, page_ptr, tidx, tile_B, tile_NH, tile_S,
                      g_cumsum, beta):
        """Load g_cumsum and beta from global into smem scalar buffers."""
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

        chunk_idx = tile_S
        head_base = tile_B * Int32(self.S * self.NH) + tile_NH
        gGC = cute.make_tensor(head_base + g_cumsum.iterator,
            cute.make_layout((self.S,), stride=(self.NH,)))
        gGC_tile = cute.local_tile(gGC, (self.BT,), (chunk_idx,))

        gBeta = cute.make_tensor(head_base + beta.iterator,
            cute.make_layout((self.S,), stride=(self.NH,)))
        gBeta_tile = cute.local_tile(gBeta, (self.BT,), (chunk_idx,))

        if tidx < Int32(self.BT):
            gc_buf[tidx] = gGC_tile[tidx]
            beta_buf[tidx] = gBeta_tile[tidx]
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

    @cute.jit
    def _weight_mma_k_block(self, page_ptr, tidx, buf_idx, ki_val,
                             chunk_idx, tile_B, tile_NH, w):
        """Weight K block in-place and MMA with A_solved, write to w.

        Reconstructs MMA objects internally to avoid closure captures.
        buf_idx selects which double-buffer slot (0 or 1) via pointer math.
        """
        _nthreads = self.num_mma_threads

        # MMA setup
        mma_op = warp.MmaF16BF16Op(self.k_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.num_mma_warps, 1, 1)),
            permutation_mnk=(self.num_mma_warps * 16, 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        # LdMatrix B (compact layout, no swizzle)
        smem_copy_atom_B = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.k_dtype,
        )
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        # A_solved [BT, BT] at offset 0
        s_a16 = cute.make_tensor(
            cute.make_ptr(self.k_dtype, page_ptr,
                          cute.AddressSpace.smem, assumed_align=128),
            cute.make_layout((self.BT, self.BT), stride=(self.BT, 1)),
        )
        tCsA16 = thr_mma.partition_A(s_a16)
        tCrA16 = tiled_mma.make_fragment_A(tCsA16)

        # K buffer selected by buf_idx (dynamic offset)
        k_base = page_ptr + Int32(self._k_buf0_offset) + buf_idx * Int32(self._k_buf_stride)
        s_k = cute.make_tensor(
            cute.make_ptr(self.k_dtype, k_base,
                          cute.AddressSpace.smem, assumed_align=128),
            cute.make_layout((self.BT, self.BK), stride=(self.BK, 1)),
        )
        # B-operand view: transposed [BK, BT]
        s_k_B = cute.make_tensor(s_k.iterator,
            cute.make_layout((self.BK, self.BT), stride=(1, self.BK)))
        tSsSrc = smem_thr_copy_B.partition_S(s_k_B)
        _tBsSrc = thr_mma.partition_B(s_k_B)
        tCrSrc_B = tiled_mma.make_fragment_B(_tBsSrc)
        tSrSrc_view = smem_thr_copy_B.retile(tCrSrc_B)

        # Scalar buffers
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

        # Weight K in-place: k * beta * exp(gc)
        _elems_per_thread = (self.BT * self.BK) // _nthreads
        for ei in cutlass.range_constexpr(_elems_per_thread):
            flat_idx = tidx + Int32(ei * _nthreads)
            row_s = flat_idx // Int32(self.BK)
            col_s = flat_idx % Int32(self.BK)
            if row_s < Int32(self.BT):
                old_val = s_k[row_s, col_s].to(Float32)
                weight = beta_buf[row_s] * cute.math.exp(
                    gc_buf[row_s], fastmath=True)
                s_k[row_s, col_s] = (old_val * weight).to(self.k_dtype)
        named_barrier_sync(Int32(2), Int32(_nthreads))

        # MMA: A_solved @ k_weighted [BT,BT] x [BT,BK] -> [BT,BK]
        acc_w = cute.make_fragment(
            tiled_mma.partition_shape_C((self.BT, self.BK)), Float32)
        acc_w.fill(0.0)

        for kb in cutlass.range_constexpr(self.BT // 16):
            cute.copy(smem_tiled_copy_B, tSsSrc[None, None, kb],
                      tSrSrc_view[None, None, kb])
            cute.autovec_copy(tCsA16[None, None, kb],
                              tCrA16[None, None, kb])
            cute.gemm(tiled_mma, acc_w, tCrA16[None, None, kb],
                      tCrSrc_B[None, None, kb], acc_w)

        # Write w[BT, BK] to global
        mc_out = cute.make_identity_tensor((self.BT, self.BK))
        tCcOut = thr_mma.partition_C(mc_out)
        kw_base = (tile_B * Int32(self.S * self.NH) + tile_NH) * Int32(self.K)
        gW_head = cute.make_tensor(
            (w.iterator + kw_base).align(16),
            cute.make_layout((self.S, self.K), stride=(self.NH * self.K, 1)))
        gW_tile = cute.local_tile(gW_head, (self.BT, self.BK),
                                  (chunk_idx, ki_val))
        for ci in cutlass.range_constexpr(cute.size(acc_w)):
            row_w = tCcOut[ci][0]
            col_w = tCcOut[ci][1]
            gW_tile[Int32(row_w), Int32(col_w)] = acc_w[ci].to(self.k_dtype)

    @cute.jit
    def _compute_w(self, page_ptr, tidx, tile_B, tile_NH, tile_S, w):
        """w = A_solved @ (beta * exp(gc) * k), K-blocked with TMA buf.

        K loaded by DMA via TMA (compact layout). Weighted in-place before MMA.
        Supports single-buf (small pages) and double-buf (large pages) K modes.
        """
        chunk_idx = tile_S
        _nthreads = self.num_mma_threads

        # Op-managed mbarriers
        _bf_0 = page_ptr + Int32(self._mbar_offset)
        _bf_1 = page_ptr + Int32(self._mbar_offset + 8)
        _kr_0 = page_ptr + Int32(self._mbar_offset + 16)
        _kr_1 = page_ptr + Int32(self._mbar_offset + 24)

        if self._single_buf_k:
            # === Single-buf K: always buf 0 ===
            # K-block 0 (TMA pre-loaded)
            self._weight_mma_k_block(page_ptr, tidx, Int32(0), Int32(0),
                                      chunk_idx, tile_B, tile_NH, w)
            named_barrier_sync(Int32(2), Int32(_nthreads))
            if tidx == Int32(0):
                mbarrier_arrive(_bf_0)

            # K-blocks 1+ — wait kblock_ready[0], process buf 0, signal bf[0]
            _kr_phase = Int32(0)
            k_idx = Int32(1)
            while k_idx < Int32(self.NK):
                mbarrier_wait(_kr_0, _kr_phase)
                _kr_phase = _kr_phase ^ Int32(1)

                self._weight_mma_k_block(page_ptr, tidx, Int32(0), k_idx,
                                          chunk_idx, tile_B, tile_NH, w)

                named_barrier_sync(Int32(2), Int32(_nthreads))
                if tidx == Int32(0):
                    mbarrier_arrive(_bf_0)

                k_idx = k_idx + Int32(1)
        else:
            # === Double-buf K: alternate buf 0 and buf 1 ===
            # K-block 0 from buf 0 (TMA pre-loaded)
            self._weight_mma_k_block(page_ptr, tidx, Int32(0), Int32(0),
                                      chunk_idx, tile_B, tile_NH, w)
            named_barrier_sync(Int32(2), Int32(_nthreads))
            if tidx == Int32(0):
                mbarrier_arrive(_bf_0)

            # K-block 1 from buf 1 (if NK >= 2, TMA pre-loaded)
            if self.NK >= 2:
                self._weight_mma_k_block(page_ptr, tidx, Int32(1), Int32(1),
                                          chunk_idx, tile_B, tile_NH, w)
                named_barrier_sync(Int32(2), Int32(_nthreads))
                if tidx == Int32(0):
                    mbarrier_arrive(_bf_1)

            # K-blocks 2+ — wait kblock_ready, process, signal buf_free
            _kr_phase_0 = Int32(0)
            _kr_phase_1 = Int32(0)
            k_idx = Int32(2)
            while k_idx < Int32(self.NK):
                _cur_buf = k_idx % Int32(2)

                if _cur_buf == Int32(0):
                    mbarrier_wait(_kr_0, _kr_phase_0)
                    _kr_phase_0 = _kr_phase_0 ^ Int32(1)
                if _cur_buf == Int32(1):
                    mbarrier_wait(_kr_1, _kr_phase_1)
                    _kr_phase_1 = _kr_phase_1 ^ Int32(1)

                self._weight_mma_k_block(page_ptr, tidx, _cur_buf, k_idx,
                                          chunk_idx, tile_B, tile_NH, w)

                named_barrier_sync(Int32(2), Int32(_nthreads))
                if tidx == Int32(0):
                    if _cur_buf == Int32(0):
                        mbarrier_arrive(_bf_0)
                    if _cur_buf == Int32(1):
                        mbarrier_arrive(_bf_1)

                k_idx = k_idx + Int32(1)

    @cute.jit
    def _compute_u(self, page_ptr, tidx, tile_B, tile_NH, tile_S, v, u):
        """u = A_solved @ (beta * v), V-blocked with cpasync."""
        chunk_idx = tile_S
        _nthreads = self.num_mma_threads

        # MMA setup
        mma_op = warp.MmaF16BF16Op(self.k_dtype, Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout((self.num_mma_warps, 1, 1)),
            permutation_mnk=(self.num_mma_warps * 16, 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx)

        # LdMatrix B for v buffers
        smem_copy_atom_B = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.k_dtype,
        )
        smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
        smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

        # A_solved packed: [BT, BT] fp16 at offset 0
        s_a16 = cute.make_tensor(
            cute.make_ptr(self.k_dtype, page_ptr,
                          cute.AddressSpace.smem, assumed_align=128),
            cute.make_layout((self.BT, self.BT), stride=(self.BT, 1)),
        )
        tCsA16 = thr_mma.partition_A(s_a16)
        tCrA16 = tiled_mma.make_fragment_A(tCsA16)

        # Scalar buffer (only beta needed for v weighting)
        beta_buf = cute.make_tensor(
            cute.make_ptr(Float32, page_ptr + Int32(self._beta_offset),
                          cute.AddressSpace.smem, assumed_align=16),
            cute.make_layout(self.BT),
        )

        # cpasync copy for BV-wide loads
        v_async_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), self.k_dtype, num_bits_per_copy=128)
        v_tiled_copy = cute.make_tiled_copy_tv(
            v_async_atom,
            cute.make_layout((self.v_copy_dim0, self.v_copy_dim1),
                             stride=(self.v_copy_dim1, 1)),
            cute.make_layout((1, self.async_copy_elems)),
        )
        v_thr_copy = v_tiled_copy.get_slice(tidx)

        # Two src buffers for double-buffered V loads (reuse K smem area)
        v_offsets = [self._v_buf0_offset, self._v_buf1_offset]
        s_srcv_bufs = []
        tSrcV_s_bufs = []
        tSsSrcV_bufs = []
        for bi in cutlass.range_constexpr(2):
            s_srcv_bi = cute.make_tensor(
                cute.make_ptr(self.k_dtype, page_ptr + Int32(v_offsets[bi]),
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

        # Global tensors
        uv_base = (tile_B * Int32(self.S * self.NH) + tile_NH) * Int32(self.V)
        gV_head = cute.make_tensor(
            (v.iterator + uv_base).align(16),
            cute.make_layout((self.S, self.V), stride=(self.NH * self.V, 1)))
        gU_head = cute.make_tensor(
            (u.iterator + uv_base).align(16),
            cute.make_layout((self.S, self.V), stride=(self.NH * self.V, 1)))

        # Prefetch v[0] -> buf[0]
        gV_tile0 = cute.local_tile(gV_head, (self.BT, self.BV),
                                   (chunk_idx, Int32(0)))
        tV_g0 = v_thr_copy.partition_S(gV_tile0)
        for ci in cutlass.range_constexpr(cute.size(tSrcV_s_bufs[0].shape[2])):
            cute.copy(v_tiled_copy, tV_g0[None, None, ci],
                      tSrcV_s_bufs[0][None, None, ci])
        cute.arch.cp_async_commit_group()

        for vi in cutlass.range_constexpr(self.NV):
            _cur_srcv = s_srcv_bufs[vi % 2]
            _cur_tSsSrcV = tSsSrcV_bufs[vi % 2]

            cute.arch.cp_async_wait_group(0)
            named_barrier_sync(Int32(2), Int32(_nthreads))

            # Prefetch next V-tile
            if vi + 1 < self.NV:
                _next_tSrcV = tSrcV_s_bufs[(vi + 1) % 2]
                gV_next = cute.local_tile(gV_head, (self.BT, self.BV),
                                          (chunk_idx, Int32(vi + 1)))
                tV_g_next = v_thr_copy.partition_S(gV_next)
                for ci in cutlass.range_constexpr(cute.size(_next_tSrcV.shape[2])):
                    cute.copy(v_tiled_copy, tV_g_next[None, None, ci],
                              _next_tSrcV[None, None, ci])
                cute.arch.cp_async_commit_group()

            # Weight: v * beta
            for ei in cutlass.range_constexpr(_elems_per_thread_v):
                flat_idx = tidx + Int32(ei * _nthreads)
                row_s = flat_idx // Int32(self.BV)
                col_s = flat_idx % Int32(self.BV)
                if row_s < Int32(self.BT):
                    old_val = _cur_srcv[row_s, col_s].to(Float32)
                    weight = beta_buf[row_s]
                    _cur_srcv[row_s, col_s] = (old_val * weight).to(self.k_dtype)
            named_barrier_sync(Int32(2), Int32(_nthreads))

            # MMA: A_solved @ v_weighted [BT,BT] x [BT,BV] -> [BT,BV]
            acc_u = cute.make_fragment(
                tiled_mma.partition_shape_C((self.BT, self.BV)), Float32)
            acc_u.fill(0.0)

            for kb in cutlass.range_constexpr(self.BT // 16):
                cute.copy(smem_tiled_copy_B, _cur_tSsSrcV[None, None, kb],
                          tSrSrcV_view[None, None, kb])
                cute.autovec_copy(tCsA16[None, None, kb],
                                  tCrA16[None, None, kb])
                cute.gemm(tiled_mma, acc_u, tCrA16[None, None, kb],
                          tCrSrcV_B[None, None, kb], acc_u)

            # Write u[BT, BV] to global
            gU_tile = cute.local_tile(gU_head, (self.BT, self.BV),
                                      (chunk_idx, Int32(vi)))
            for ci in cutlass.range_constexpr(cute.size(acc_u)):
                row_u = tCcOutV[ci][0]
                col_u = tCcOutV[ci][1]
                gU_tile[Int32(row_u), Int32(col_u)] = acc_u[ci].to(self.k_dtype)

            named_barrier_sync(Int32(2), Int32(_nthreads))


__all__ = ["GDNWUOp"]
