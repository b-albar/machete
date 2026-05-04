# Copyright (c) 2025, Machete Authors
"""Gated Delta Net Output Op — chunked output computation.

Per chunk:
    Phase A (K-loop): acc_o += Q @ h_states[chunk],  acc_A += Q @ K^T
    Phase B: Gating on acc_o, gating+mask on acc_A
    Phase C: scores → smem, v_new → smem
    Phase D: acc_intra = scores @ v_new
    Phase E: o = (acc_o + acc_intra) * scale → global

Architecture:
    MMA warps: cpasync loads for q, k, v_new, h_states + MMA compute.
    All chunks sequential inside compute.

Tiling: (B, NH, V) — one V-strip per tile, all S chunks sequential.
"""

import struct

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu import warp

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import named_barrier_sync


from machete.kernels.gated_delta_net.chunk_size import auto_bt, _DEFAULT_BT

_BK = 64
_BV = 64


class GDNOutputOp(Op):
    """Gated Delta Net Output — chunked output computation.

    Tensors (native [B, S, NH, K/V] layout):
        q:        (B, S, NH, K)       -- queries
        k:        (B, S, NH, K)       -- keys
        v_new:    (B, S, NH, V)       -- new values (from StateOp)
        h_states: (B, NT, NH, K, V)   -- inter-chunk states (from StateOp)
        g_cumsum: (B, S, NH)          -- cumulative gates (fp32)
        o:        (B, S, NH, V)       -- output

    Tiling:
        tile_B=1, tile_NH=1, tile_V=BV.
        All chunks loop inside compute.
    """

    reads = {
        "q":        (None, ("B", "S", "NH", "K")),
        "k":        (None, ("B", "S", "NH", "K")),
        "v_new":    (None, ("B", "S", "NH", "V")),
        "h_states": (None, ("B", "NT", "NH", "K", "V")),
        "g_cumsum": (cutlass.Float32, ("B", "S", "NH")),
    }
    writes = {
        "o": (None, ("B", "S", "NH", "V")),
    }
    tile = ("B", "NH", "V")
    dynamic_dims = ("B",)
    tma_loads = set()

    def __init__(self, **config):
        super().__init__(**config)
        self.scale_val = struct.unpack("f", struct.pack("I", self.scale_bits))[0]

        assert self.q_dtype in (cutlass.Float16, cutlass.BFloat16)
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

        # Smem layout (no primary Q buffer — Q loaded per K-block):
        #   buf_a:  [BT, BK]     (Q_bk cpasync; reused for scores [BT,BT] after K-loop)
        #   buf_b:  [BT, BK]     (K_bk cpasync; reused for scores after K-loop)
        #   s_h:    [BK, BV]     (h_states cpasync per K-block)
        #   s_v:    [BT, BV]     (v_new cpasync)
        #   g_buf:  [BT] fp32
        buf_bytes = self.BT * self.BK * self.elem_bytes
        self._s_buf0_offset = 0
        self._s_buf1_offset = buf_bytes
        state_start = 2 * buf_bytes
        self._s_state_offset = ((state_start + 127) // 128) * 128
        v_start = self._s_state_offset + self.BK * self.BV * self.elem_bytes
        self._s_v_offset = ((v_start + 127) // 128) * 128
        gbuf_start = self._s_v_offset + self.BT * self.BV * self.elem_bytes
        self._gbuf_offset = ((gbuf_start + 15) // 16) * 16

        self.compute = self.compute_mma

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def _auto_block_sizes(cls, page_size, K, V, BT, elem_bytes=2):
        """Compute largest (BK, BV) fitting in page_size.

        Smem: 2*BT*BK*eb + BK*BV*eb + BT*BV*eb + BT*4 + align
        Constraint: 2*BK >= BT (scores [BT,BT] must fit in buf_a+buf_b).
        """
        for BK in [128, 64, 32]:
            if K % BK != 0 or BK > BT:
                continue
            if 2 * BK < BT:
                continue  # scores [BT,BT] won't fit in buf_a+buf_b
            for BV in [128, 64, 32]:
                if V % BV != 0 or BV > BT:
                    continue
                buf_bytes = BT * BK * elem_bytes
                state_start = 2 * buf_bytes
                sh_off = ((state_start + 127) // 128) * 128
                sv_off = ((sh_off + BK * BV * elem_bytes + 127) // 128) * 128
                gbuf_off = ((sv_off + BT * BV * elem_bytes + 15) // 16) * 16
                total = gbuf_off + BT * 4
                if total <= page_size:
                    return BK, BV
        raise ValueError(
            f"page_size={page_size} too small for GDNOutputOp (K={K}, V={V})."
        )

    @classmethod
    def schedule(cls, scale=None, page_size=DEFAULT_PAGE_SIZE,
                         tile_sizes=None, **tensors):
        """Schedule GDN output Op."""
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("NH", 1)
        tile_sizes.setdefault("V", _BV)

        BT = auto_bt(page_size)

        q = tensors.get("q")
        v_new = tensors.get("v_new")
        K = q.shape[-1] if q is not None else 64
        V = v_new.shape[-1] if v_new is not None else K
        elem_bytes = q.element_size() if q is not None else 2
        S = q.shape[1] if q is not None else 64

        if scale is None:
            scale = K ** -0.5

        BK, BV = cls._auto_block_sizes(page_size, K, V, BT, elem_bytes)
        scale_bits = struct.unpack("I", struct.pack("f", scale))[0]

        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        ops[0].static_dims["BT"] = BT
        ops[0].static_dims["scale_bits"] = scale_bits
        ops[0].static_dims["BK"] = BK
        ops[0].static_dims["BV"] = BV
        ops[0].static_dims["K"] = K
        ops[0].static_dims["S"] = S
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
    # Compute — all chunks sequential
    # =========================================================================

    @cute.jit
    def compute_mma(
        self, page_ptr, tile_B, tile_NH, tile_V,
        q, k, v_new, h_states, g_cumsum, o,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        if warp_idx < Int32(self.num_mma_warps):
            # === Swizzles ===
            swz = cute.make_swizzle(self.swizzle_B, 4, 3)
            swz_v = cute.make_swizzle(self.swizzle_B_v, 4, 3)

            # === MMA setup ===
            mma_op = warp.MmaF16BF16Op(self.q_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((self.num_mma_warps, 1, 1)),
                permutation_mnk=(self.num_mma_warps * 16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)

            # === Smem regions ===
            # buf_a: Q_bk [BT, BK] swizzled
            s_q_swz = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr + Int32(self._s_buf0_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.q_dtype,
                ),
                cute.make_layout((self.BT, self.BK), stride=(self.BK, 1)),
            )
            # buf_a: Q_bk plain (for cpasync dest)
            s_q_plain = cute.make_tensor(
                cute.make_ptr(self.q_dtype, page_ptr + Int32(self._s_buf0_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.BT, self.BK), stride=(self.BK, 1)),
            )

            # buf_b: K_bk [BT, BK] swizzled
            s_k_swz = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr + Int32(self._s_buf1_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.q_dtype,
                ),
                cute.make_layout((self.BT, self.BK), stride=(self.BK, 1)),
            )
            # buf_b: K_bk plain (for cpasync dest)
            s_k_plain = cute.make_tensor(
                cute.make_ptr(self.q_dtype, page_ptr + Int32(self._s_buf1_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.BT, self.BK), stride=(self.BK, 1)),
            )

            s_h = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr + Int32(self._s_state_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz_v, dtype=self.q_dtype,
                ),
                cute.make_layout((self.BK, self.BV), stride=(self.BV, 1)),
            )

            s_v_plain = cute.make_tensor(
                cute.make_ptr(self.q_dtype, page_ptr + Int32(self._s_v_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.BT, self.BV), stride=(self.BV, 1)),
            )

            s_v_swz = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr + Int32(self._s_v_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz_v, dtype=self.q_dtype,
                ),
                cute.make_layout((self.BT, self.BV), stride=(self.BV, 1)),
            )

            g_buf = cute.make_tensor(
                cute.make_ptr(Float32, page_ptr + Int32(self._gbuf_offset),
                              cute.AddressSpace.smem, assumed_align=16),
                cute.make_layout(self.BT),
            )

            # === Copy atoms ===
            # transpose=True for column-major B operands (h, v_new)
            smem_copy_atom_Bt = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                self.q_dtype,
            )
            smem_tiled_copy_Bt = cute.make_tiled_copy_B(smem_copy_atom_Bt, tiled_mma)
            smem_thr_copy_Bt = smem_tiled_copy_Bt.get_slice(tidx)
            # transpose=False for row-major B operands (K)
            smem_copy_atom_Bk = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.q_dtype,
            )
            smem_tiled_copy_Bk = cute.make_tiled_copy_B(smem_copy_atom_Bk, tiled_mma)
            smem_thr_copy_Bk = smem_tiled_copy_Bk.get_slice(tidx)

            smem_copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.q_dtype)
            smem_tiled_copy_C = cute.make_tiled_copy_C(smem_copy_atom_C, tiled_mma)
            smem_thr_copy_C = smem_tiled_copy_C.get_slice(tidx)

            # === MMA partitions ===
            # h as B operand [BV, BK] column-major (transpose=True copy)
            s_h_B = cute.make_tensor(s_h.iterator,
                cute.make_layout((self.BV, self.BK), stride=(1, self.BV)))
            _tBsH = thr_mma.partition_B(s_h_B)
            tCrB = tiled_mma.make_fragment_B(_tBsH)
            tBrB_view = smem_thr_copy_Bt.retile(tCrB)
            tHsH = smem_thr_copy_Bt.partition_S(s_h_B)

            # K as B operand [BT, BK] row-major (transpose=False copy)
            s_kt_B = cute.make_tensor(s_k_swz.iterator,
                cute.make_layout((self.BT, self.BK), stride=(self.BK, 1)))
            _tBsKt = thr_mma.partition_B(s_kt_B)
            tCrKt = tiled_mma.make_fragment_B(_tBsKt)
            tKrKt_view = smem_thr_copy_Bk.retile(tCrKt)
            tKsKt = smem_thr_copy_Bk.partition_S(s_kt_B)

            # Q as A operand [BT, BK] (from buf_a)
            tCsQ = thr_mma.partition_A(s_q_swz)
            tCrQ = tiled_mma.make_fragment_A(tCsQ)

            # v_new as B operand [BV, BT]
            s_v_B = cute.make_tensor(s_v_swz.iterator,
                cute.make_layout((self.BV, self.BT), stride=(1, self.BV)))
            tVsVt = smem_thr_copy_Bt.partition_S(s_v_B)

            # Scores as A operand [BT, BT] (reuses buf_a offset)
            s_a = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr + Int32(self._s_buf0_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.q_dtype,
                ),
                cute.make_layout((self.BT, self.BT), stride=(self.BT, 1)),
            )
            tCsA_scores = thr_mma.partition_A(s_a)

            # === cpasync setup ===
            wk_async_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), self.q_dtype, num_bits_per_copy=128)
            wk_tiled_copy = cute.make_tiled_copy_tv(
                wk_async_atom,
                cute.make_layout((self.wk_copy_dim0, self.wk_copy_dim1),
                                 stride=(self.wk_copy_dim1, 1)),
                cute.make_layout((1, self.async_copy_elems)),
            )
            wk_thr_copy = wk_tiled_copy.get_slice(tidx)

            tQ_s = wk_thr_copy.partition_D(s_q_plain)
            tK_s = wk_thr_copy.partition_D(s_k_plain)

            uv_async_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), self.q_dtype, num_bits_per_copy=128)
            uv_tiled_copy = cute.make_tiled_copy_tv(
                uv_async_atom,
                cute.make_layout((self.uv_copy_dim0, self.uv_copy_dim1),
                                 stride=(self.uv_copy_dim1, 1)),
                cute.make_layout((1, self.async_copy_elems)),
            )
            uv_thr_copy = uv_tiled_copy.get_slice(tidx)
            tV_s = uv_thr_copy.partition_D(s_v_plain)

            # h cpasync (same dims as uv since BK*BV reads)
            h_thr_copy = uv_thr_copy  # reuse same copy atom

            # Fragment for acc_A → smem scores conversion (BT×BT shape)
            conv_scores = cute.make_fragment(
                tiled_mma.partition_shape_C((self.BT, self.BT)), self.q_dtype)

            # === Identity tensors ===
            mc_tv = cute.make_identity_tensor((self.BT, self.BV))
            tCcTV = thr_mma.partition_C(mc_tv)
            mc_AA = cute.make_identity_tensor((self.BT, self.BT))
            tCcAA = thr_mma.partition_C(mc_AA)

            # === Global tensors ===
            kw_base = (tile_B * Int32(self.S * self.NH) + tile_NH) * Int32(self.K)
            uv_base = (tile_B * Int32(self.S * self.NH) + tile_NH) * Int32(self.V)
            g_head_base = tile_B * Int32(self.S * self.NH) + tile_NH

            gQ_head = cute.make_tensor(
                (q.iterator + kw_base).align(16),
                cute.make_layout((self.S, self.K), stride=(self.NH * self.K, 1)))
            gK_head = cute.make_tensor(
                (k.iterator + kw_base).align(16),
                cute.make_layout((self.S, self.K), stride=(self.NH * self.K, 1)))
            gVN_head = cute.make_tensor(
                (v_new.iterator + uv_base).align(16),
                cute.make_layout((self.S, self.V), stride=(self.NH * self.V, 1)))
            gG_head = cute.make_tensor(g_cumsum.iterator + g_head_base,
                cute.make_layout((self.S,), stride=(self.NH,)))
            gO_head = cute.make_tensor(
                (o.iterator + uv_base).align(16),
                cute.make_layout((self.S, self.V), stride=(self.NH * self.V, 1)))

            # h_states: [B, NT, NH, K, V]
            hs_head_base = (tile_B * Int32(self.NT_val * self.NH) + tile_NH) * Int32(self.K * self.V)

            # === Prologue: issue v_new[0] cpasync ===
            _gV_pre = cute.local_tile(gVN_head, (self.BT, self.BV),
                                      (Int32(0), tile_V))
            _tV_g_pre = uv_thr_copy.partition_S(_gV_pre)
            for ci in cutlass.range_constexpr(cute.size(tV_s.shape[2])):
                cute.copy(uv_tiled_copy, _tV_g_pre[None, None, ci],
                          tV_s[None, None, ci])
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

                # Wait for v_new cpasync
                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # ---------------------------------------------------
                # Phase A: K-loop — Q@h + Q@K^T
                # ---------------------------------------------------
                acc_o = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), Float32)
                acc_o.fill(0.0)
                acc_A = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BT)), Float32)
                acc_A.fill(0.0)

                _hs_base = hs_head_base + chunk_idx * Int32(self.NH * self.K * self.V)

                # Issue cpasync for Q[0], K[0], h[0]
                _gQ0 = cute.local_tile(gQ_head, (self.BT, self.BK),
                                        (chunk_idx, Int32(0)))
                _tQg0 = wk_thr_copy.partition_S(_gQ0)
                for ci in cutlass.range_constexpr(cute.size(tQ_s.shape[2])):
                    cute.copy(wk_tiled_copy, _tQg0[None, None, ci],
                              tQ_s[None, None, ci])

                _gK0 = cute.local_tile(gK_head, (self.BT, self.BK),
                                        (chunk_idx, Int32(0)))
                _tKg0 = wk_thr_copy.partition_S(_gK0)
                for ci in cutlass.range_constexpr(cute.size(tK_s.shape[2])):
                    cute.copy(wk_tiled_copy, _tKg0[None, None, ci],
                              tK_s[None, None, ci])

                gHS_0 = cute.make_tensor(
                    (h_states.iterator + _hs_base + tile_V * Int32(self.BV)).align(16),
                    cute.make_layout((self.BK, self.BV), stride=(self.V, 1)),
                )
                tHS_g0 = h_thr_copy.partition_S(gHS_0)
                tHS_s0 = h_thr_copy.partition_D(s_h)
                for ci in cutlass.range_constexpr(cute.size(tHS_s0.shape[2])):
                    cute.copy(uv_tiled_copy, tHS_g0[None, None, ci],
                              tHS_s0[None, None, ci])
                cute.arch.cp_async_commit_group()

                for ki in cutlass.range_constexpr(self.NK):
                    if ki > 0:
                        # Issue K[ki] + h[ki] cpasync (Q[ki] already in
                        # flight from previous iteration's prefetch)
                        _gK_ki = cute.local_tile(gK_head, (self.BT, self.BK),
                                                  (chunk_idx, Int32(ki)))
                        _tKg_ki = wk_thr_copy.partition_S(_gK_ki)
                        for ci in cutlass.range_constexpr(cute.size(tK_s.shape[2])):
                            cute.copy(wk_tiled_copy, _tKg_ki[None, None, ci],
                                      tK_s[None, None, ci])

                        gHS_ki = cute.make_tensor(
                            (h_states.iterator + _hs_base
                             + Int32(ki * self.BK * self.V)
                             + tile_V * Int32(self.BV)).align(16),
                            cute.make_layout((self.BK, self.BV),
                                             stride=(self.V, 1)),
                        )
                        tHS_gki = h_thr_copy.partition_S(gHS_ki)
                        tHS_ski = h_thr_copy.partition_D(s_h)
                        for ci in cutlass.range_constexpr(cute.size(tHS_ski.shape[2])):
                            cute.copy(uv_tiled_copy, tHS_gki[None, None, ci],
                                      tHS_ski[None, None, ci])
                        cute.arch.cp_async_commit_group()

                    cute.arch.cp_async_wait_group(0)
                    named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                    # Load Q_bk from buf_a smem → registers
                    for qkb in cutlass.range_constexpr(self.BK // 16):
                        cute.autovec_copy(tCsQ[None, None, qkb],
                                          tCrQ[None, None, qkb])

                    # Pipeline: issue Q[ki+1] cpasync (buf_a free, Q in regs)
                    if ki + 1 < self.NK:
                        _gQ_nxt = cute.local_tile(gQ_head, (self.BT, self.BK),
                                                   (chunk_idx, Int32(ki + 1)))
                        _tQg_nxt = wk_thr_copy.partition_S(_gQ_nxt)
                        for ci in cutlass.range_constexpr(cute.size(tQ_s.shape[2])):
                            cute.copy(wk_tiled_copy, _tQg_nxt[None, None, ci],
                                      tQ_s[None, None, ci])
                        cute.arch.cp_async_commit_group()

                    # GEMM: acc_o += Q_bk @ h (with fragment prefetch)
                    cute.copy(smem_tiled_copy_Bt, tHsH[None, None, 0],
                              tBrB_view[None, None, 0])
                    for kb in cutlass.range_constexpr(self.BK // 16):
                        kb_next = (kb + 1) % (self.BK // 16)
                        cute.copy(smem_tiled_copy_Bt, tHsH[None, None, kb_next],
                                  tBrB_view[None, None, kb_next])
                        cute.gemm(tiled_mma, acc_o,
                                  tCrQ[None, None, kb],
                                  tCrB[None, None, kb], acc_o)

                    # GEMM: acc_A += Q_bk @ K^T_bk (with fragment prefetch)
                    cute.copy(smem_tiled_copy_Bk, tKsKt[None, None, 0],
                              tKrKt_view[None, None, 0])
                    for kb in cutlass.range_constexpr(self.BK // 16):
                        kb_next = (kb + 1) % (self.BK // 16)
                        cute.copy(smem_tiled_copy_Bk, tKsKt[None, None, kb_next],
                                  tKrKt_view[None, None, kb_next])
                        cute.gemm(tiled_mma, acc_A,
                                  tCrQ[None, None, kb],
                                  tCrKt[None, None, kb], acc_A)

                # ---------------------------------------------------
                # Phase B: Gating on acc_o, gating+mask on acc_A
                # ---------------------------------------------------
                for ci in cutlass.range_constexpr(cute.size(acc_o)):
                    row = tCcTV[ci][0]
                    acc_o[ci] = acc_o[ci] * cute.math.exp(
                        g_buf[Int32(row)], fastmath=True)

                for ci in cutlass.range_constexpr(cute.size(acc_A)):
                    row = tCcAA[ci][0]
                    col = tCcAA[ci][1]
                    if Int32(col) > Int32(row):
                        acc_A[ci] = Float32(0.0)
                    else:
                        acc_A[ci] = acc_A[ci] * cute.math.exp(
                            g_buf[Int32(row)] - g_buf[Int32(col)],
                            fastmath=True)

                # ---------------------------------------------------
                # Phase C: scores → smem for GEMM
                # ---------------------------------------------------
                for ci in cutlass.range_constexpr(cute.size(acc_A)):
                    conv_scores[ci] = acc_A[ci].to(self.q_dtype)
                tOrA = smem_thr_copy_C.retile(conv_scores)
                tOsA = smem_thr_copy_C.partition_D(s_a)
                cute.copy(smem_tiled_copy_C, tOrA, tOsA)

                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # ---------------------------------------------------
                # Phase D: acc_intra = scores @ v_new
                # ---------------------------------------------------
                _tBsVt = thr_mma.partition_B(s_v_B)
                tCrVt = tiled_mma.make_fragment_B(_tBsVt)
                tVrVt_view = smem_thr_copy_Bt.retile(tCrVt)
                tCrA_scores = tiled_mma.make_fragment_A(tCsA_scores)
                acc_intra = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), Float32)
                acc_intra.fill(0.0)

                # scores @ v_new GEMM with fragment prefetch
                cute.copy(smem_tiled_copy_Bt, tVsVt[None, None, 0],
                          tVrVt_view[None, None, 0])
                cute.autovec_copy(tCsA_scores[None, None, 0],
                                  tCrA_scores[None, None, 0])
                for kb in cutlass.range_constexpr(self.BT // 16):
                    kb_next = (kb + 1) % (self.BT // 16)
                    cute.copy(smem_tiled_copy_Bt, tVsVt[None, None, kb_next],
                              tVrVt_view[None, None, kb_next])
                    cute.autovec_copy(tCsA_scores[None, None, kb_next],
                                      tCrA_scores[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_intra,
                              tCrA_scores[None, None, kb],
                              tCrVt[None, None, kb], acc_intra)

                # Pipeline: v_new[chunk+1] cpasync
                next_chunk = chunk_idx + Int32(1)
                if next_chunk < Int32(self.NT_val):
                    gV_nxt = cute.local_tile(gVN_head, (self.BT, self.BV),
                                              (next_chunk, tile_V))
                    tV_g_nxt = uv_thr_copy.partition_S(gV_nxt)
                    for ci in cutlass.range_constexpr(cute.size(tV_s.shape[2])):
                        cute.copy(uv_tiled_copy, tV_g_nxt[None, None, ci],
                                  tV_s[None, None, ci])
                    cute.arch.cp_async_commit_group()

                # ---------------------------------------------------
                # Phase E: Write O to global
                # ---------------------------------------------------
                gO_tile = cute.local_tile(gO_head, (self.BT, self.BV),
                                          (chunk_idx, tile_V))
                for ci in cutlass.range_constexpr(cute.size(acc_o)):
                    row = tCcTV[ci][0]
                    col = tCcTV[ci][1]
                    o_val = (acc_o[ci] + acc_intra[ci]) * Float32(self.scale_val)
                    gO_tile[Int32(row), Int32(col)] = o_val.to(self.q_dtype)

                chunk_idx = chunk_idx + Int32(1)


__all__ = ["GDNOutputOp"]
