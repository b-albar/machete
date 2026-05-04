# Copyright (c) 2025, Machete Authors
"""Gated Delta Net State Recurrence Op — chunked state recurrence (h_states only).

Maintains h[BK,BV] in registers across all chunks (forward order).
Per chunk:
    1. Store h → h_states (BEFORE processing)
    2. v_partial = w @ h
    3. v_gated = (u - v_partial) * exp(g_last - g_cumsum[t])
    4. h *= exp(g_last) (decay)
    5. h += k^T @ v_gated (state update)

Does NOT write v_new — that's computed by VNewOp in parallel afterward.

Architecture:
    DMA warp:  Idle (no TMA).
    MMA warps: cpasync loads + MMA compute + global stores.
    All chunks sequential inside compute, h persistent in registers.

Tiling: (B, NH, V) — one V-strip per tile, all S chunks sequential.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu import warp

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import named_barrier_sync


from machete.kernels.gated_delta_net.chunk_size import auto_bt, _DEFAULT_BT

_BK = 64
_BV = 64


class GDNStateRecurrenceOp(Op):
    """Gated Delta Net State Recurrence — h_states only.

    Tensors (native [B, S, NH, K/V] layout):
        k:        (B, S, NH, K)       -- keys
        w:        (B, S, NH, K)       -- transformed keys
        u:        (B, S, NH, V)       -- transformed values
        g_cumsum: (B, S, NH)          -- cumulative gates (fp32)
        h_states: (B, NT, NH, K, V)   -- output: inter-chunk hidden states

    Tiling:
        tile_B=1, tile_NH=1, tile_V=BV.
        All chunks loop inside compute, h[NK][BK,BV] persistent in registers.
    """

    reads = {
        "k":        (None, ("B", "S", "NH", "K")),
        "w":        (None, ("B", "S", "NH", "K")),
        "u":        (None, ("B", "S", "NH", "V")),
        "g_cumsum": (cutlass.Float32, ("B", "S", "NH")),
    }
    writes = {
        "h_states": (None, ("B", "NT", "NH", "K", "V")),
    }
    tile = ("B", "NH", "V")
    dynamic_dims = ("B",)
    tma_loads = set()

    def __init__(self, **config):
        super().__init__(**config)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)

        assert self.k_dtype in (cutlass.Float16, cutlass.BFloat16)
        self.elem_bytes = 2

        self.BT = getattr(self, "BT", _DEFAULT_BT)
        self.BK = getattr(self, "BK", _BK)
        self.BV = getattr(self, "BV", _BV)
        self.NK = self.K // self.BK
        self.NT_val = self.S // self.BT

        assert self.K % self.BK == 0
        assert self.V % self.BV == 0
        assert self.S % self.BT == 0

        self.num_mma_warps = self.BT // 16
        self.num_mma_threads = self.num_mma_warps * 32

        # Swizzles
        self.swizzle_B = 3 if self.BK >= 64 else (2 if self.BK >= 32 else 1)
        self.swizzle_B_v = 3 if self.BV >= 64 else (2 if self.BV >= 32 else 1)

        # cpasync thread layouts
        self.async_copy_elems = 128 // (self.elem_bytes * 8)
        self.wk_copy_dim1 = self.BK // self.async_copy_elems
        self.wk_copy_dim0 = self.num_mma_threads // self.wk_copy_dim1
        self.uv_copy_dim1 = self.BV // self.async_copy_elems
        self.uv_copy_dim0 = self.num_mma_threads // self.uv_copy_dim1

        # Configurable pipeline stages (can be 1 or 2)
        self.num_stages = getattr(self, "num_stages", 2)

        # Smem layout (no v_new output → no s_uv needed for output):
        #   wk_buf×2 [BT,BK] swizzled   (double-buf for w/k cpasync)
        #   s_state  [BK,BV] swizzled    (h register→smem staging for GEMM)
        #   s_uv     [BT,BV]             (u load for v_gated computation)
        #   g_buf    [BT] fp32
        buf_bytes = self.BT * self.BK * self.elem_bytes
        self._s_buf0_offset = 0
        self._s_buf1_offset = buf_bytes
        state_start = 2 * buf_bytes
        self._s_state_offset = ((state_start + 127) // 128) * 128
        uv_start = self._s_state_offset + self.BK * self.BV * self.elem_bytes
        self._s_uv_offset = ((uv_start + 127) // 128) * 128
        gbuf_start = self._s_uv_offset + self.BT * self.BV * self.elem_bytes
        self._gbuf_offset = ((gbuf_start + 15) // 16) * 16

        self.compute = self.compute_mma

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def _auto_block_sizes(cls, page_size, K, V, BT, elem_bytes=2):
        """Compute largest (BK, BV) fitting in page_size."""
        for BK in [128, 64, 32]:
            if K % BK != 0 or BK > BT:
                continue
            for BV in [128, 64, 32]:
                if V % BV != 0 or BV > BT:
                    continue
                buf = 2 * BT * BK * elem_bytes
                state_off = ((buf + 127) // 128) * 128
                uv_off = ((state_off + BK * BV * elem_bytes + 127) // 128) * 128
                gbuf_off = ((uv_off + BT * BV * elem_bytes + 15) // 16) * 16
                total = gbuf_off + BT * 4
                if total <= page_size:
                    return BK, BV
        raise ValueError(
            f"page_size={page_size} too small for GDNStateRecurrenceOp (K={K}, V={V})."
        )

    @classmethod
    def schedule(cls, page_size=DEFAULT_PAGE_SIZE,
                         num_stages=2, tile_sizes=None, **tensors):
        """Schedule GDN state recurrence Op."""
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("NH", 1)
        tile_sizes.setdefault("V", _BV)

        BT = auto_bt(page_size)

        k = tensors.get("k")
        u = tensors.get("u")
        K = k.shape[-1] if k is not None else 64
        V = u.shape[-1] if u is not None else K
        elem_bytes = k.element_size() if k is not None else 2
        S = k.shape[1] if k is not None else 64

        BK, BV = cls._auto_block_sizes(page_size, K, V, BT, elem_bytes)

        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        ops[0].static_dims["BT"] = BT
        ops[0].static_dims["BK"] = BK
        ops[0].static_dims["BV"] = BV
        ops[0].static_dims["K"] = K
        ops[0].static_dims["S"] = S
        ops[0].static_dims["num_stages"] = num_stages
        return ops

    @classmethod
    def kernel_config(cls, ops):
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS

        BT = max(op.static_dims.get("BT", _DEFAULT_BT) for op in ops)
        num_mma_warps = BT // 16
        threads_per_block = (num_mma_warps + NUM_DMA_WARPS) * 32
        page_size = max(
            op.static_dims.get("page_size", DEFAULT_PAGE_SIZE) for op in ops
        )
        return MegakernelConfig(
            threads_per_block=threads_per_block,
            page_size=page_size,
        )

    # =========================================================================
    # Load — no TMA, idle DMA warp
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_NH, tile_V):
        """No TMA loads — framework auto-arrives on work_mbar."""
        pass

    # =========================================================================
    # Compute — all chunks sequential, h persistent in registers
    # =========================================================================

    @cute.jit
    def compute_mma(
        self, page_ptr, tile_B, tile_NH, tile_V,
        k, w, u, g_cumsum, h_states,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        if warp_idx < Int32(self.num_mma_warps):
            # === Swizzles ===
            swz = cute.make_swizzle(self.swizzle_B, 4, 3)
            swz_v = cute.make_swizzle(self.swizzle_B_v, 4, 3)

            # === MMA setup ===
            mma_op = warp.MmaF16BF16Op(self.k_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((self.num_mma_warps, 1, 1)),
                permutation_mnk=(self.num_mma_warps * 16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)

            # === Smem regions ===
            s_wk_bufs = []
            for buf_idx in cutlass.range_constexpr(2):
                offset = self._s_buf0_offset if buf_idx == 0 else self._s_buf1_offset
                s = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(self.k_dtype, page_ptr + Int32(offset),
                                      cute.AddressSpace.smem, assumed_align=128),
                        swz, dtype=self.k_dtype,
                    ),
                    cute.make_layout((self.BT, self.BK), stride=(self.BK, 1)),
                )
                s_wk_bufs.append(s)

            s_h = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.k_dtype, page_ptr + Int32(self._s_state_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz_v, dtype=self.k_dtype,
                ),
                cute.make_layout((self.BK, self.BV), stride=(self.BV, 1)),
            )

            # s_uv: NOT swizzled (read via partition_C indexing)
            s_uv = cute.make_tensor(
                cute.make_ptr(self.k_dtype, page_ptr + Int32(self._s_uv_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.BT, self.BV), stride=(self.BV, 1)),
            )

            # s_v: swizzled (same region, for GEMM B operand reads)
            s_v = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.k_dtype, page_ptr + Int32(self._s_uv_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz_v, dtype=self.k_dtype,
                ),
                cute.make_layout((self.BT, self.BV), stride=(self.BV, 1)),
            )

            g_buf = cute.make_tensor(
                cute.make_ptr(Float32, page_ptr + Int32(self._gbuf_offset),
                              cute.AddressSpace.smem, assumed_align=16),
                cute.make_layout(self.BT),
            )

            # === Copy atoms ===
            smem_copy_atom_B = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                self.k_dtype,
            )
            smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
            smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

            smem_copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.k_dtype)
            smem_tiled_copy_C = cute.make_tiled_copy_C(smem_copy_atom_C, tiled_mma)
            smem_thr_copy_C = smem_tiled_copy_C.get_slice(tidx)

            # === MMA partitions ===
            # h as B operand [BV, BK]
            s_h_B = cute.make_tensor(s_h.iterator,
                cute.make_layout((self.BV, self.BK), stride=(1, self.BV)))
            _tBsH = thr_mma.partition_B(s_h_B)
            tCrB = tiled_mma.make_fragment_B(_tBsH)
            tBrB_view = smem_thr_copy_B.retile(tCrB)
            tHsH = smem_thr_copy_B.partition_S(s_h_B)

            # w/k as A operand from double-buf
            tCsA_bufs = []
            tWK_s_bufs = []
            tCsKt_bufs = []  # K^T for state update GEMM
            for buf_idx in cutlass.range_constexpr(2):
                tCsA_bufs.append(thr_mma.partition_A(s_wk_bufs[buf_idx]))
                # K^T view [BK, BT] for state update
                s_kt_b = cute.make_tensor(s_wk_bufs[buf_idx].iterator,
                    cute.make_layout((self.BK, self.BT), stride=(1, self.BK)))
                tCsKt_bufs.append(thr_mma.partition_A(s_kt_b))
            tCrA = tiled_mma.make_fragment_A(tCsA_bufs[0])

            # v_gated as B operand [BV, BT]
            s_v_B = cute.make_tensor(s_v.iterator,
                cute.make_layout((self.BV, self.BT), stride=(1, self.BV)))
            tVsVt = smem_thr_copy_B.partition_S(s_v_B)

            # === cpasync setup ===
            wk_async_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), self.k_dtype, num_bits_per_copy=128)
            wk_tiled_copy = cute.make_tiled_copy_tv(
                wk_async_atom,
                cute.make_layout((self.wk_copy_dim0, self.wk_copy_dim1),
                                 stride=(self.wk_copy_dim1, 1)),
                cute.make_layout((1, self.async_copy_elems)),
            )
            wk_thr_copy = wk_tiled_copy.get_slice(tidx)

            for buf_idx in cutlass.range_constexpr(2):
                tWK_s_bufs.append(wk_thr_copy.partition_D(s_wk_bufs[buf_idx]))

            uv_async_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), self.k_dtype, num_bits_per_copy=128)
            uv_tiled_copy = cute.make_tiled_copy_tv(
                uv_async_atom,
                cute.make_layout((self.uv_copy_dim0, self.uv_copy_dim1),
                                 stride=(self.uv_copy_dim1, 1)),
                cute.make_layout((1, self.async_copy_elems)),
            )
            uv_thr_copy = uv_tiled_copy.get_slice(tidx)
            tU_s = uv_thr_copy.partition_D(s_uv)

            # === h accumulators (persistent across chunks) ===
            acc_h_shape = tiled_mma.partition_shape_C((self.BK, self.BV))
            h_accs = []
            for _ki in cutlass.range_constexpr(self.NK):
                _acc = cute.make_fragment(acc_h_shape, Float32)
                _acc.fill(0.0)
                h_accs.append(_acc)

            # Reusable dtype fragment for conversions
            frag_dtype = cute.make_fragment(acc_h_shape, self.k_dtype)

            # === Identity tensors (hoisted out of loop) ===
            mc_tv = cute.make_identity_tensor((self.BT, self.BV))
            tCcTV = thr_mma.partition_C(mc_tv)
            mc_kv = cute.make_identity_tensor((self.BK, self.BV))
            tCcKV = thr_mma.partition_C(mc_kv)

            # === Global tensors ===
            kw_base = (tile_B * Int32(self.S * self.NH) + tile_NH) * Int32(self.K)
            uv_base = (tile_B * Int32(self.S * self.NH) + tile_NH) * Int32(self.V)
            g_head_base = tile_B * Int32(self.S * self.NH) + tile_NH

            gK_head = cute.make_tensor(
                (k.iterator + kw_base).align(16),
                cute.make_layout((self.S, self.K), stride=(self.NH * self.K, 1)))
            gW_head = cute.make_tensor(
                (w.iterator + kw_base).align(16),
                cute.make_layout((self.S, self.K), stride=(self.NH * self.K, 1)))
            gU_head = cute.make_tensor(
                (u.iterator + uv_base).align(16),
                cute.make_layout((self.S, self.V), stride=(self.NH * self.V, 1)))
            gG_head = cute.make_tensor(g_cumsum.iterator + g_head_base,
                cute.make_layout((self.S,), stride=(self.NH,)))

            # h_states: [B, NT, NH, K, V]
            hs_head_base = (tile_B * Int32(self.NT_val * self.NH) + tile_NH) * Int32(self.K * self.V)

            # === Prologue: prefetch u[0] ===
            _gU_pre = cute.local_tile(gU_head, (self.BT, self.BV),
                                      (Int32(0), tile_V))
            _tU_g_pre = uv_thr_copy.partition_S(_gU_pre)
            for ci in cutlass.range_constexpr(cute.size(tU_s.shape[2])):
                cute.copy(uv_tiled_copy, _tU_g_pre[None, None, ci],
                          tU_s[None, None, ci])
            cute.arch.cp_async_commit_group()

            # ===============================================================
            # Main chunk loop
            # ===============================================================
            chunk_idx = Int32(0)
            while chunk_idx < Int32(self.NT_val):

                # Load g_cumsum
                gG_tile = cute.local_tile(gG_head, (self.BT,), (chunk_idx,))
                if tidx < Int32(self.BT):
                    g_buf[tidx] = gG_tile[tidx]

                # Wait for u cpasync
                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                tCsU = thr_mma.partition_C(s_uv)

                # -----------------------------------------------------------
                # Fused: h_states store + w@h GEMM
                # Direct register→global for h_states (no smem roundtrip)
                # h→smem only for GEMM B operand
                # -----------------------------------------------------------
                hs_chunk_base = hs_head_base + chunk_idx * Int32(self.NH * self.K * self.V)

                acc_vp = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), Float32)
                acc_vp.fill(0.0)

                # Prefetch w[0]
                _gW0 = cute.local_tile(gW_head, (self.BT, self.BK),
                                        (chunk_idx, Int32(0)))
                _tWg0 = wk_thr_copy.partition_S(_gW0)
                for ci in cutlass.range_constexpr(cute.size(tWK_s_bufs[0].shape[2])):
                    cute.copy(wk_tiled_copy, _tWg0[None, None, ci],
                              tWK_s_bufs[0][None, None, ci])
                cute.arch.cp_async_commit_group()

                for ki in cutlass.range_constexpr(self.NK):
                    cur = ki % 2

                    # (a) Direct h_states store: register → global
                    gHS_block = cute.make_tensor(
                        (h_states.iterator + hs_chunk_base
                         + Int32(ki * self.BK * self.V)
                         + tile_V * Int32(self.BV)).align(16),
                        cute.make_layout((self.BK, self.BV),
                                         stride=(self.V, 1)),
                    )
                    for ci in cutlass.range_constexpr(cute.size(h_accs[ki])):
                        row_k = tCcKV[ci][0]
                        col_v = tCcKV[ci][1]
                        gHS_block[Int32(row_k), Int32(col_v)] = h_accs[ki][ci].to(self.k_dtype)

                    # (b) h → smem for GEMM B operand
                    for ci in cutlass.range_constexpr(cute.size(h_accs[ki])):
                        frag_dtype[ci] = h_accs[ki][ci].to(self.k_dtype)
                    tOrH = smem_thr_copy_C.retile(frag_dtype)
                    tOsH = smem_thr_copy_C.partition_D(s_h)
                    cute.copy(smem_tiled_copy_C, tOrH, tOsH)

                    # (c) Wait w cpasync + sync (also ensures s_h writes visible)
                    cute.arch.cp_async_wait_group(0)
                    named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                    # (d) GEMM: acc_vp += w[ki] @ h[ki]
                    for kb in cutlass.range_constexpr(self.BK // 16):
                        cute.copy(smem_tiled_copy_B, tHsH[None, None, kb],
                                  tBrB_view[None, None, kb])
                        cute.autovec_copy(tCsA_bufs[cur][None, None, kb],
                                          tCrA[None, None, kb])
                        cute.gemm(tiled_mma, acc_vp,
                                  tCrA[None, None, kb], tCrB[None, None, kb],
                                  acc_vp)

                    # (e) Prefetch w[ki+1]
                    _nxt_safe = min(ki + 1, self.NK - 1)
                    _nxt_cur_safe = _nxt_safe % 2
                    if ki < self.NK - 1:
                        _gW_nxt = cute.local_tile(gW_head, (self.BT, self.BK),
                                                   (chunk_idx, Int32(_nxt_safe)))
                        _tWg_nxt = wk_thr_copy.partition_S(_gW_nxt)
                        for ci in cutlass.range_constexpr(
                                cute.size(tWK_s_bufs[_nxt_cur_safe].shape[2])):
                            cute.copy(wk_tiled_copy, _tWg_nxt[None, None, ci],
                                      tWK_s_bufs[_nxt_cur_safe][None, None, ci])
                        cute.arch.cp_async_commit_group()

                    # (f) Inter-ki sync
                    if ki < self.NK - 1:
                        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # -----------------------------------------------------------
                # v_gated = (u - w@h) * exp(g_last - g[t])
                # No v_new write — just compute v_gated for state update
                # -----------------------------------------------------------
                g_last = g_buf[Int32(self.BT - 1)]

                vgated_regs = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), self.k_dtype)

                for ci in cutlass.range_constexpr(cute.size(acc_vp)):
                    vn_val = tCsU[ci].to(Float32) - acc_vp[ci]
                    row = tCcTV[ci][0]
                    gate = cute.math.exp(g_last - g_buf[Int32(row)], fastmath=True)
                    vgated_regs[ci] = (vn_val * gate).to(self.k_dtype)

                # Write v_gated → s_v for GEMM
                tOrVG = smem_thr_copy_C.retile(vgated_regs)
                tOsVG = smem_thr_copy_C.partition_D(s_v)
                cute.copy(smem_tiled_copy_C, tOrVG, tOsVG)

                # -----------------------------------------------------------
                # Decay + state update h += k^T @ v_gated
                # -----------------------------------------------------------
                decay = cute.math.exp(g_last, fastmath=True)

                # Prefetch k[0] while writing v_gated
                _gK0 = cute.local_tile(gK_head, (self.BT, self.BK),
                                        (chunk_idx, Int32(0)))
                _tKg0 = wk_thr_copy.partition_S(_gK0)
                for ci in cutlass.range_constexpr(cute.size(tWK_s_bufs[0].shape[2])):
                    cute.copy(wk_tiled_copy, _tKg0[None, None, ci],
                              tWK_s_bufs[0][None, None, ci])
                cute.arch.cp_async_commit_group()

                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                for ki in cutlass.range_constexpr(self.NK):
                    cur = ki % 2

                    cute.arch.cp_async_wait_group(0)
                    named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                    # Prefetch k[ki+1]
                    _nxt_safe = min(ki + 1, self.NK - 1)
                    _nxt_cur_safe = _nxt_safe % 2
                    if ki < self.NK - 1:
                        _gK_nxt = cute.local_tile(gK_head, (self.BT, self.BK),
                                                   (chunk_idx, Int32(_nxt_safe)))
                        _tKg_nxt = wk_thr_copy.partition_S(_gK_nxt)
                        for ci in cutlass.range_constexpr(
                                cute.size(tWK_s_bufs[_nxt_cur_safe].shape[2])):
                            cute.copy(wk_tiled_copy, _tKg_nxt[None, None, ci],
                                      tWK_s_bufs[_nxt_cur_safe][None, None, ci])
                        cute.arch.cp_async_commit_group()

                    # Decay h_accs[ki]
                    for ci in cutlass.range_constexpr(cute.size(h_accs[ki])):
                        h_accs[ki][ci] = h_accs[ki][ci] * decay

                    # MMA: h[ki] += k^T @ v_gated
                    for kb in cutlass.range_constexpr(self.BT // 16):
                        cute.copy(smem_tiled_copy_B, tVsVt[None, None, kb],
                                  tBrB_view[None, None, kb])
                        cute.autovec_copy(tCsKt_bufs[cur][None, None, kb],
                                          tCrA[None, None, kb])
                        cute.gemm(tiled_mma, h_accs[ki],
                                  tCrA[None, None, kb], tCrB[None, None, kb],
                                  h_accs[ki])

                    if ki < self.NK - 1:
                        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # Pipeline: prefetch u[chunk+1]
                next_chunk = chunk_idx + Int32(1)
                if next_chunk < Int32(self.NT_val):
                    gU_nxt = cute.local_tile(gU_head, (self.BT, self.BV),
                                              (next_chunk, tile_V))
                    tU_g_nxt = uv_thr_copy.partition_S(gU_nxt)
                    for ci in cutlass.range_constexpr(cute.size(tU_s.shape[2])):
                        cute.copy(uv_tiled_copy, tU_g_nxt[None, None, ci],
                                  tU_s[None, None, ci])
                    cute.arch.cp_async_commit_group()

                chunk_idx = chunk_idx + Int32(1)


__all__ = ["GDNStateRecurrenceOp"]
