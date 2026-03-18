# Copyright (c) 2025, Machete Authors
"""Gated Delta Net VNew Op — parallel v_new computation with TMA W loading.

Embarrassingly parallel over all chunks:
    v_new[chunk] = u[chunk] - w[chunk] @ h_states[chunk]

This is the second half of the original StateOp, split out so it can
run in parallel over all chunks instead of sequentially.

Architecture:
    DMA warp:  TMA loads W sub-blocks into double buffer.
    MMA warps: cpasync loads h_states + u, MMA compute.
    Each tile processes one (B, NH, chunk, V-strip) independently.

Tiling: (B, NH, S, V) — fully parallel over all chunks.
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


_BT = 64
_BK = 64
_BV = 64
_MBAR_BYTES = 32  # 4 mbarriers × 8 bytes (buf_free×2 + kblock_ready×2)


class GDNVNewOp(Op):
    """Gated Delta Net VNew — parallel v_new = u - w @ h_states.

    Tensors (native [B, S, NH, K/V] layout):
        w:        (B, S, NH, K)       -- transformed keys
        u:        (B, S, NH, V)       -- transformed values
        h_states: (B, NT, NH, K, V)   -- inter-chunk hidden states
        v_new:    (B, S, NH, V)       -- output: new values

    Tiling:
        tile_B=1, tile_NH=1, tile_S=BT (one chunk), tile_V=BV.
        Fully parallel — no inter-chunk dependencies.
    """

    reads = {
        "w":        (None, ("B", "S", "NH", "K")),
        "u":        (None, ("B", "S", "NH", "V")),
        "h_states": (None, ("B", "NT", "NH", "K", "V")),
    }
    writes = {
        "v_new":    (None, ("B", "S", "NH", "V")),
    }
    tile = ("B", "NH", "S", "V")
    tma_loads = {"w"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name != "w":
            return None
        BK = static_dims.get("BK", _BK)
        return (_BT, tile_sizes.get("NH", 1), BK)

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape,
                                 tile_sizes, static_dims):
        if tensor_name != "w":
            return None
        BK = static_dims.get("BK", _BK)
        swz_B = 3 if BK >= 64 else (2 if BK >= 32 else 1)
        return (
            f"cute.make_composed_layout("
            f"cute.make_swizzle({swz_B}, 4, 3), 0, "
            f"cute.make_layout(({BK}, 1, {_BT}), "
            f"stride=(1, {BK}, {BK})))"
        )

    def __init__(self, **config):
        super().__init__(**config)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)

        assert self.w_dtype in (cutlass.Float16, cutlass.BFloat16)
        self.elem_bytes = 2

        self.BT = _BT
        self.BK = getattr(self, "BK", _BK)
        self.BV = getattr(self, "BV", _BV)
        self.NK = self.K // self.BK

        assert self.K % self.BK == 0
        assert self.V % self.BV == 0
        assert self.S % self.BT == 0

        self.num_mma_warps = self.BT // 16
        self.num_mma_threads = self.num_mma_warps * 32

        # Swizzles
        self.swizzle_B = 3 if self.BK >= 64 else (2 if self.BK >= 32 else 1)
        self.swizzle_B_v = 3 if self.BV >= 64 else (2 if self.BV >= 32 else 1)

        # cpasync thread layouts (for h_states and u only)
        self.async_copy_elems = 128 // (self.elem_bytes * 8)
        self.uv_copy_dim1 = self.BV // self.async_copy_elems
        self.uv_copy_dim0 = self.num_mma_threads // self.uv_copy_dim1

        # TMA double-buffer for W
        self.inner_iters = max(1, self.NK - 1)
        self.inner_depth = 1
        self._tma_k_blocks = min(2, self.NK)
        self._w_buf_bytes = self.BT * self.BK * self.elem_bytes
        self._buf_stride = self._w_buf_bytes
        self._tma_w_bytes = self._tma_k_blocks * self._w_buf_bytes

        # Smem layout:
        #   w_buf×2 [BT,BK] TMA swizzled  (double-buf for W TMA)
        #   s_h     [BK,BV] swizzled       (h_states cpasync)
        #   s_uv    [BT,BV]                (u cpasync)
        #   4 mbarriers (32B)              (buf_free×2 + kblock_ready×2)
        buf_bytes = self.BT * self.BK * self.elem_bytes
        self._s_buf0_offset = 0
        self._s_buf1_offset = buf_bytes
        state_start = 2 * buf_bytes
        self._s_state_offset = ((state_start + 127) // 128) * 128
        uv_start = self._s_state_offset + self.BK * self.BV * self.elem_bytes
        self._s_uv_offset = ((uv_start + 127) // 128) * 128
        mbar_start = self._s_uv_offset + self.BT * self.BV * self.elem_bytes
        self._mbar_offset = ((mbar_start + 7) // 8) * 8

        self.compute = self.compute_mma

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def _auto_block_sizes(cls, page_size, K, V, elem_bytes=2):
        """Compute largest (BK, BV) fitting in page_size."""
        BT = _BT
        for BK in [128, 64, 32]:
            if K % BK != 0 or BK > BT:
                continue
            for BV in [128, 64, 32]:
                if V % BV != 0 or BV > BT:
                    continue
                buf = 2 * BT * BK * elem_bytes
                state_off = ((buf + 127) // 128) * 128
                uv_off = ((state_off + BK * BV * elem_bytes + 127) // 128) * 128
                mbar_off = ((uv_off + BT * BV * elem_bytes + 7) // 8) * 8
                total = mbar_off + _MBAR_BYTES
                if total <= page_size:
                    return BK, BV
        raise ValueError(
            f"page_size={page_size} too small for GDNVNewOp (K={K}, V={V})."
        )

    @classmethod
    def schedule(cls, page_size=DEFAULT_PAGE_SIZE,
                         tile_sizes=None, **tensors):
        """Schedule GDN VNew Op."""
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("NH", 1)
        tile_sizes.setdefault("S", _BT)
        tile_sizes.setdefault("V", _BV)

        w = tensors.get("w")
        u = tensors.get("u")
        K = w.shape[-1] if w is not None else 64
        V = u.shape[-1] if u is not None else K
        elem_bytes = w.element_size() if w is not None else 2
        S = w.shape[1] if w is not None else 64

        BK, BV = cls._auto_block_sizes(page_size, K, V, elem_bytes)

        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        ops[0].static_dims["BK"] = BK
        ops[0].static_dims["BV"] = BV
        ops[0].static_dims["K"] = K
        ops[0].static_dims["S"] = S
        return ops

    @classmethod
    def kernel_config(cls, ops):
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
    # Load — TMA W into double buffer
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_NH, tile_S, tile_V,
             w_tma, w_tma_gmem, work_mbar, inner_iter_idx):
        """TMA W load: NK sub-blocks [BT, BK] into double buffer.

        iter 0 (load warp):  Init mbarriers, TMA W[0..min(2,NK)-1],
                              signal work_mbar.
        iter 1+ (store warp): Wait buf_free[buf], TMA W[k_block],
                              signal kblock_ready[buf].
        """
        swz = cute.make_swizzle(self.swizzle_B, 4, 3)

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

            # Signal expected TMA bytes on framework work_mbar
            nbytes = Int32(self._tma_w_bytes)
            with cute.arch.elect_one():
                mbarrier_arrive_expect_tx(work_mbar, nbytes)

            # TMA W[0..min(2,NK)-1] into double buffer
            for _k in cutlass.range_constexpr(self._tma_k_blocks):
                _buf_base = page_ptr + Int32(_k * self._buf_stride)
                sW = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(self.w_dtype, _buf_base,
                                      cute.AddressSpace.smem),
                        swz, dtype=self.w_dtype),
                    cute.make_layout((self.BK, 1, self.BT),
                                     stride=(1, self.BK, self.BK)),
                )
                gW = cute.local_tile(
                    w_tma_gmem,
                    (self.BK, 1, self.BT),
                    (None, None, None),
                )
                tWsW, tWgW = cute.nvgpu.cpasync.tma_partition(
                    w_tma, Int32(0), cute.make_layout(1),
                    cute.group_modes(sW, 0, 3), cute.group_modes(gW, 0, 3),
                )
                cute.copy(w_tma, tWgW[(None, Int32(_k), tile_NH, merged_t)],
                          tWsW, tma_bar_ptr=mbar_ptr)

        if inner_iter_idx > Int32(0):
            # Store warp: load W[k_block] into freed buffer
            _k_block = inner_iter_idx + Int32(1)
            _buf_idx = _k_block % Int32(2)

            # Wait for compute to free this buffer
            _bf_phase = ((inner_iter_idx - Int32(1)) // Int32(2)) % Int32(2)
            if _buf_idx == Int32(0):
                mbarrier_wait(_bf_0, _bf_phase)
            if _buf_idx == Int32(1):
                mbarrier_wait(_bf_1, _bf_phase)

            # Set up TMA bar for kblock_ready
            _buf_base = _buf_idx * Int32(self._buf_stride) + page_ptr
            _kr_mbar = _kr_0
            if _buf_idx == Int32(1):
                _kr_mbar = _kr_1
            _kr_ptr = cute.make_ptr(cutlass.Int64, _kr_mbar,
                                    cute.AddressSpace.smem)
            with cute.arch.elect_one():
                mbarrier_arrive_expect_tx(_kr_mbar, Int32(self._w_buf_bytes))

            sW = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.w_dtype, _buf_base,
                                  cute.AddressSpace.smem),
                    swz, dtype=self.w_dtype),
                cute.make_layout((self.BK, 1, self.BT),
                                 stride=(1, self.BK, self.BK)),
            )
            gW = cute.local_tile(
                w_tma_gmem,
                (self.BK, 1, self.BT),
                (None, None, None),
            )
            tWsW, tWgW = cute.nvgpu.cpasync.tma_partition(
                w_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sW, 0, 3), cute.group_modes(gW, 0, 3),
            )
            cute.copy(w_tma, tWgW[(None, _k_block, tile_NH, merged_t)],
                      tWsW, tma_bar_ptr=_kr_ptr)

    # =========================================================================
    # Compute — v_new = u - w @ h_states (TMA W double-buffer + cpasync h)
    # =========================================================================

    @cute.jit
    def compute_mma(
        self, page_ptr, tile_B, tile_NH, tile_S, tile_V,
        w, u, h_states, v_new,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        if warp_idx < Int32(self.num_mma_warps):
            # === Swizzles ===
            swz = cute.make_swizzle(self.swizzle_B, 4, 3)
            swz_v = cute.make_swizzle(self.swizzle_B_v, 4, 3)

            # === Op-managed mbarriers ===
            _bf_0 = page_ptr + Int32(self._mbar_offset)
            _bf_1 = page_ptr + Int32(self._mbar_offset + 8)
            _kr_0 = page_ptr + Int32(self._mbar_offset + 16)
            _kr_1 = page_ptr + Int32(self._mbar_offset + 24)

            # === MMA setup ===
            mma_op = warp.MmaF16BF16Op(self.w_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((self.num_mma_warps, 1, 1)),
                permutation_mnk=(self.num_mma_warps * 16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)

            # === Smem W buffers (TMA swizzled) ===
            s_wk_bufs = []
            for buf_idx in cutlass.range_constexpr(2):
                offset = self._s_buf0_offset if buf_idx == 0 else self._s_buf1_offset
                s = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(self.w_dtype, page_ptr + Int32(offset),
                                      cute.AddressSpace.smem, assumed_align=128),
                        swz, dtype=self.w_dtype,
                    ),
                    cute.make_layout((self.BT, self.BK), stride=(self.BK, 1)),
                )
                s_wk_bufs.append(s)

            # === h_states smem (cpasync, swizzled) ===
            s_h = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.w_dtype, page_ptr + Int32(self._s_state_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz_v, dtype=self.w_dtype,
                ),
                cute.make_layout((self.BK, self.BV), stride=(self.BV, 1)),
            )

            # === u smem (cpasync) ===
            s_uv = cute.make_tensor(
                cute.make_ptr(self.w_dtype, page_ptr + Int32(self._s_uv_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.BT, self.BV), stride=(self.BV, 1)),
            )

            # === Copy atoms for LdMatrix ===
            smem_copy_atom_B = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                self.w_dtype,
            )
            smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
            smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

            # === MMA partitions ===
            # h as B operand [BV, BK] (transposed view)
            s_h_B = cute.make_tensor(s_h.iterator,
                cute.make_layout((self.BV, self.BK), stride=(1, self.BV)))
            _tBsH = thr_mma.partition_B(s_h_B)
            tCrB = tiled_mma.make_fragment_B(_tBsH)
            tBrB_view = smem_thr_copy_B.retile(tCrB)
            tHsH = smem_thr_copy_B.partition_S(s_h_B)

            # W as A operand from double-buf
            tCsA_bufs = []
            for buf_idx in cutlass.range_constexpr(2):
                tCsA_bufs.append(thr_mma.partition_A(s_wk_bufs[buf_idx]))
            tCrA = tiled_mma.make_fragment_A(tCsA_bufs[0])

            # === cpasync setup (h_states + u only) ===
            uv_async_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), self.w_dtype, num_bits_per_copy=128)
            uv_tiled_copy = cute.make_tiled_copy_tv(
                uv_async_atom,
                cute.make_layout((self.uv_copy_dim0, self.uv_copy_dim1),
                                 stride=(self.uv_copy_dim1, 1)),
                cute.make_layout((1, self.async_copy_elems)),
            )
            uv_thr_copy = uv_tiled_copy.get_slice(tidx)
            tU_s = uv_thr_copy.partition_D(s_uv)
            h_thr_copy = uv_thr_copy

            # === Identity tensors ===
            mc_tv = cute.make_identity_tensor((self.BT, self.BV))
            tCcTV = thr_mma.partition_C(mc_tv)

            # === Global tensors ===
            chunk_idx = tile_S
            uv_base = (tile_B * Int32(self.S * self.NH) + tile_NH) * Int32(self.V)

            gU_head = cute.make_tensor(
                (u.iterator + uv_base).align(16),
                cute.make_layout((self.S, self.V), stride=(self.NH * self.V, 1)))
            gVN_head = cute.make_tensor(
                (v_new.iterator + uv_base).align(16),
                cute.make_layout((self.S, self.V), stride=(self.NH * self.V, 1)))

            NT_val = self.S // self.BT
            hs_head_base = (tile_B * Int32(NT_val * self.NH) + tile_NH) * Int32(self.K * self.V)
            hs_chunk_base = hs_head_base + chunk_idx * Int32(self.NH * self.K * self.V)

            # === Load u into smem (cpasync) ===
            gU_tile = cute.local_tile(gU_head, (self.BT, self.BV),
                                      (chunk_idx, tile_V))
            tU_g = uv_thr_copy.partition_S(gU_tile)
            for ci in cutlass.range_constexpr(cute.size(tU_s.shape[2])):
                cute.copy(uv_tiled_copy, tU_g[None, None, ci],
                          tU_s[None, None, ci])

            # === Load h_states[chunk, ki=0] (cpasync) ===
            gHS_0 = cute.make_tensor(
                (h_states.iterator + hs_chunk_base + tile_V * Int32(self.BV)).align(16),
                cute.make_layout((self.BK, self.BV), stride=(self.V, 1)),
            )
            tHS_g0 = h_thr_copy.partition_S(gHS_0)
            tHS_s0 = h_thr_copy.partition_D(s_h)
            for ci in cutlass.range_constexpr(cute.size(tHS_s0.shape[2])):
                cute.copy(uv_tiled_copy, tHS_g0[None, None, ci],
                          tHS_s0[None, None, ci])
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            tCsU = thr_mma.partition_C(s_uv)

            # === Accumulator ===
            acc_vp = cute.make_fragment(
                tiled_mma.partition_shape_C((self.BT, self.BV)), Float32)
            acc_vp.fill(0.0)

            # === K-block 0 from buf 0 (TMA pre-loaded) ===
            for kb in cutlass.range_constexpr(self.BK // 16):
                cute.copy(smem_tiled_copy_B, tHsH[None, None, kb],
                          tBrB_view[None, None, kb])
                cute.autovec_copy(tCsA_bufs[0][None, None, kb],
                                  tCrA[None, None, kb])
                cute.gemm(tiled_mma, acc_vp,
                          tCrA[None, None, kb], tCrB[None, None, kb],
                          acc_vp)

            # Signal buf 0 free
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
            if tidx == Int32(0):
                mbarrier_arrive(_bf_0)

            # === K-block 1 from buf 1 (if NK >= 2, TMA pre-loaded) ===
            if self.NK >= 2:
                # Load h_states[chunk, ki=1] via cpasync
                gHS_1 = cute.make_tensor(
                    (h_states.iterator + hs_chunk_base
                     + Int32(1 * self.BK * self.V)
                     + tile_V * Int32(self.BV)).align(16),
                    cute.make_layout((self.BK, self.BV), stride=(self.V, 1)),
                )
                tHS_g1 = h_thr_copy.partition_S(gHS_1)
                tHS_s1 = h_thr_copy.partition_D(s_h)
                for ci in cutlass.range_constexpr(cute.size(tHS_s1.shape[2])):
                    cute.copy(uv_tiled_copy, tHS_g1[None, None, ci],
                              tHS_s1[None, None, ci])
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                for kb in cutlass.range_constexpr(self.BK // 16):
                    cute.copy(smem_tiled_copy_B, tHsH[None, None, kb],
                              tBrB_view[None, None, kb])
                    cute.autovec_copy(tCsA_bufs[1][None, None, kb],
                                      tCrA[None, None, kb])
                    cute.gemm(tiled_mma, acc_vp,
                              tCrA[None, None, kb], tCrB[None, None, kb],
                              acc_vp)

                # Signal buf 1 free
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
                if tidx == Int32(0):
                    mbarrier_arrive(_bf_1)

            # === K-blocks 2+ — wait kblock_ready, process, signal buf_free ===
            _kr_phase_0 = Int32(0)
            _kr_phase_1 = Int32(0)
            k_idx = Int32(2)
            while k_idx < Int32(self.NK):
                _cur_buf = k_idx % Int32(2)

                # Wait for store warp's TMA to deliver this K-block
                if _cur_buf == Int32(0):
                    mbarrier_wait(_kr_0, _kr_phase_0)
                    _kr_phase_0 = _kr_phase_0 ^ Int32(1)
                if _cur_buf == Int32(1):
                    mbarrier_wait(_kr_1, _kr_phase_1)
                    _kr_phase_1 = _kr_phase_1 ^ Int32(1)

                # Load h_states[chunk, k_idx] via cpasync
                gHS_ki = cute.make_tensor(
                    (h_states.iterator + hs_chunk_base
                     + k_idx * Int32(self.BK * self.V)
                     + tile_V * Int32(self.BV)).align(16),
                    cute.make_layout((self.BK, self.BV), stride=(self.V, 1)),
                )
                tHS_gki = h_thr_copy.partition_S(gHS_ki)
                tHS_ski = h_thr_copy.partition_D(s_h)
                for ci in cutlass.range_constexpr(cute.size(tHS_ski.shape[2])):
                    cute.copy(uv_tiled_copy, tHS_gki[None, None, ci],
                              tHS_ski[None, None, ci])
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # GEMM from alternating W buffer
                if _cur_buf == Int32(0):
                    for kb in cutlass.range_constexpr(self.BK // 16):
                        cute.copy(smem_tiled_copy_B, tHsH[None, None, kb],
                                  tBrB_view[None, None, kb])
                        cute.autovec_copy(tCsA_bufs[0][None, None, kb],
                                          tCrA[None, None, kb])
                        cute.gemm(tiled_mma, acc_vp,
                                  tCrA[None, None, kb], tCrB[None, None, kb],
                                  acc_vp)
                if _cur_buf == Int32(1):
                    for kb in cutlass.range_constexpr(self.BK // 16):
                        cute.copy(smem_tiled_copy_B, tHsH[None, None, kb],
                                  tBrB_view[None, None, kb])
                        cute.autovec_copy(tCsA_bufs[1][None, None, kb],
                                          tCrA[None, None, kb])
                        cute.gemm(tiled_mma, acc_vp,
                                  tCrA[None, None, kb], tCrB[None, None, kb],
                                  acc_vp)

                # Signal buffer free
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
                if tidx == Int32(0):
                    if _cur_buf == Int32(0):
                        mbarrier_arrive(_bf_0)
                    if _cur_buf == Int32(1):
                        mbarrier_arrive(_bf_1)

                k_idx = k_idx + Int32(1)

            # === v_new = u - w@h → global ===
            gVN_tile = cute.local_tile(gVN_head, (self.BT, self.BV),
                                       (chunk_idx, tile_V))
            for ci in cutlass.range_constexpr(cute.size(acc_vp)):
                vn_val = tCsU[ci].to(Float32) - acc_vp[ci]
                row = tCcTV[ci][0]
                col = tCcTV[ci][1]
                gVN_tile[Int32(row), Int32(col)] = vn_val.to(self.w_dtype)


__all__ = ["GDNVNewOp"]
