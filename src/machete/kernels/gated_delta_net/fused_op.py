# Copyright (c) 2025, Machete Authors
"""Gated Delta Net Fused State+Output — single Op, no intermediate global stores.

Fuses StateOp and OutputOp: maintains h[BK,BV] in registers across all chunks,
computes output directly from register h, computes v_new on the fly.
Eliminates all h_states and v_new global memory traffic.

Per chunk, 5 GEMMs:
    Phase A (K-loop, 3 fused GEMMs sharing h from smem):
        GEMM1-state:  acc_vp[BT,BV]  += w[BT,BK]  @ h[BK,BV]
        GEMM1-output: acc_o[BT,BV]   += Q[BT,BK]  @ h[BK,BV]
        GEMM2-output: acc_A[BT,BT]   += Q[BT,BK]  @ K^T[BK,BT]
    Phase B: Gating on acc_o, gating+mask on acc_A
    Phase C: v_new = u - acc_vp (non-gated → s_v for GEMM3, gated → regs for state GEMM2)
    Phase D: GEMM3: acc_intra += scores @ v_new
    Phase E: Write O to global (element-wise)
    Phase F (K-loop): h[ki] = decay*h[ki] + k^T[ki] @ v_gated (state update)

Architecture:
    DMA warp:  Idle (no TMA).
    MMA warps: cpasync Q/K/w/u/g loads + all MMA compute + element-wise O stores.

Tiling: (B, H, V) — all chunks sequential inside compute.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu import warp

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import named_barrier_sync


_BT = 64
_BK = 64
_BV = 64


class GDNFusedOp(Op):
    """Gated Delta Net Fused State+Output Op.

    Tensors:
        q:       (B, T, H, K)
        k:       (B, T, H, K)
        w:       (B, T, H, K)
        u:       (B, T, H, V)
        g_cumsum:(B, T, H)       fp32
        o:       (B, T, H, V)    output

    Tiling: tile_B=1, tile_H=1, tile_V=BV. All chunks loop inside compute.
    """

    reads = {
        "q":       (None, ("B", "T", "H", "K")),
        "k":       (None, ("B", "T", "H", "K")),
        "w":       (None, ("B", "T", "H", "K")),
        "u":       (None, ("B", "T", "H", "V")),
        "g_cumsum": (cutlass.Float32, ("B", "T", "H")),
    }
    writes = {
        "o": (None, ("B", "T", "H", "V")),
    }
    tile = ("B", "H", "V")

    def __init__(self, **config):
        super().__init__(**config)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)

        assert self.q_dtype in (cutlass.Float16, cutlass.BFloat16)
        self.elem_bytes = 2

        import struct
        self.scale_val = struct.unpack("f", struct.pack("I", self.scale_bits))[0]

        self.BT = _BT
        self.BK = _BK
        self.BV = _BV
        self.NK = self.K // self.BK
        self.NT_val = self.T // self.BT

        assert self.K % self.BK == 0
        assert self.V % self.BV == 0
        assert self.T % self.BT == 0
        assert self.NK <= 2, "Q loading assumes NK <= 2 (K <= 128)"

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

        self.inner_iters = 1
        self.inner_depth = 1

        # Smem layout (32KB page):
        # s_wk: [BT, BK] swizzled × 2 (double-buffered) = 16KB
        wk_bytes = self.BT * self.BK * self.elem_bytes  # 8KB
        self._swk0_offset = 0
        self._swk1_offset = wk_bytes
        # s_h: [BK, BV] swizzled = 8KB
        self._sh_offset = 2 * wk_bytes
        self._sh_offset = ((self._sh_offset + 127) // 128) * 128
        # s_uv: [BT, BV] = 8KB (u load, then v_new staging)
        self._suv_offset = self._sh_offset + self.BK * self.BV * self.elem_bytes
        self._suv_offset = ((self._suv_offset + 127) // 128) * 128
        # g_buf: [BT] fp32 = 256B
        self._gbuf_offset = self._suv_offset + self.BT * self.BV * self.elem_bytes
        self._gbuf_offset = ((self._gbuf_offset + 15) // 16) * 16
        self._total_smem = self._gbuf_offset + self.BT * 4

        self.compute = self.compute_mma

    @classmethod
    def schedule_forward(cls, scale=None, page_size=None, tile_sizes=None, **tensors):
        import struct

        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("H", 1)
        tile_sizes.setdefault("V", _BV)

        q = tensors.get("q")
        if q is not None:
            K = q.shape[-1]
            if scale is None:
                scale = K ** -0.5

        scale_bits = struct.unpack("I", struct.pack("f", scale))[0]

        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]

        if page_size is None:
            eb = 2
            wk_bytes = _BT * _BK * eb
            sh_off = 2 * wk_bytes
            sh_off = ((sh_off + 127) // 128) * 128
            suv_off = sh_off + _BK * _BV * eb
            suv_off = ((suv_off + 127) // 128) * 128
            gbuf_off = suv_off + _BT * _BV * eb
            gbuf_off = ((gbuf_off + 15) // 16) * 16
            page_size = gbuf_off + _BT * 4

        ops[0].static_dims["page_size"] = page_size
        ops[0].static_dims["scale_bits"] = scale_bits
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
    # Compute — All chunks sequential, state + output fused
    # =========================================================================

    @cute.jit
    def compute_mma(
        self, page_ptr, tile_B, tile_H, tile_V,
        q, k, w, u, g_cumsum, o,
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
            s_wk_bufs = []
            for buf_idx in cutlass.range_constexpr(2):
                offset = self._swk0_offset if buf_idx == 0 else self._swk1_offset
                s = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(self.q_dtype, page_ptr + Int32(offset),
                                      cute.AddressSpace.smem, assumed_align=128),
                        swz, dtype=self.q_dtype,
                    ),
                    cute.make_layout((self.BT, self.BK), stride=(self.BK, 1)),
                )
                s_wk_bufs.append(s)

            s_h = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr + Int32(self._sh_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz_v, dtype=self.q_dtype,
                ),
                cute.make_layout((self.BK, self.BV), stride=(self.BV, 1)),
            )

            # s_u: NOT swizzled (read via partition_C indexing)
            s_u = cute.make_tensor(
                cute.make_ptr(self.q_dtype, page_ptr + Int32(self._suv_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.BT, self.BV), stride=(self.BV, 1)),
            )

            # s_v: swizzled (same region as s_u, for GEMM B operand)
            s_v = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr + Int32(self._suv_offset),
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
            smem_copy_atom_B = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                self.q_dtype,
            )
            smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
            smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

            smem_copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.q_dtype)
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

            # K^T as B operand [BK, BT]
            s_kt_B_template = cute.make_tensor(s_wk_bufs[0].iterator,
                cute.make_layout((self.BK, self.BT), stride=(1, self.BK)))
            _tBsKt = thr_mma.partition_B(s_kt_B_template)
            tCrKt = tiled_mma.make_fragment_B(_tBsKt)
            tKrKt_view = smem_thr_copy_B.retile(tCrKt)

            # Per-buffer partitions
            tKsKt_bufs = []
            tCsA_bufs = []
            tCsKt_bufs = []
            tWK_s_bufs = []
            for buf_idx in cutlass.range_constexpr(2):
                s_kt_b = cute.make_tensor(s_wk_bufs[buf_idx].iterator,
                    cute.make_layout((self.BK, self.BT), stride=(1, self.BK)))
                tKsKt_bufs.append(smem_thr_copy_B.partition_S(s_kt_b))
                tCsA_bufs.append(thr_mma.partition_A(s_wk_bufs[buf_idx]))
                tCsKt_bufs.append(thr_mma.partition_A(
                    cute.make_tensor(s_wk_bufs[buf_idx].iterator,
                        cute.make_layout((self.BK, self.BT), stride=(1, self.BK)))
                ))

            tCrA = tiled_mma.make_fragment_A(tCsA_bufs[0])

            # v_new as B operand [BV, BT]
            s_v_B = cute.make_tensor(s_v.iterator,
                cute.make_layout((self.BV, self.BT), stride=(1, self.BV)))
            _tBsVt = thr_mma.partition_B(s_v_B)
            tCrVt = tiled_mma.make_fragment_B(_tBsVt)
            tVrVt_view = smem_thr_copy_B.retile(tCrVt)
            tVsVt = smem_thr_copy_B.partition_S(s_v_B)

            # Scores as A operand [BT, BT] (reuses s_wk0 region)
            s_a = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr,
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.q_dtype,
                ),
                cute.make_layout((self.BT, self.BT), stride=(self.BT, 1)),
            )
            tCsA_scores = thr_mma.partition_A(s_a)
            tCrA_scores = tiled_mma.make_fragment_A(tCsA_scores)

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

            for buf_idx in cutlass.range_constexpr(2):
                tWK_s_bufs.append(wk_thr_copy.partition_D(s_wk_bufs[buf_idx]))

            uv_async_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), self.q_dtype, num_bits_per_copy=128)
            uv_tiled_copy = cute.make_tiled_copy_tv(
                uv_async_atom,
                cute.make_layout((self.uv_copy_dim0, self.uv_copy_dim1),
                                 stride=(self.uv_copy_dim1, 1)),
                cute.make_layout((1, self.async_copy_elems)),
            )
            uv_thr_copy = uv_tiled_copy.get_slice(tidx)
            tU_s = uv_thr_copy.partition_D(s_u)

            # === h accumulators (persistent across chunks) ===
            acc_h_shape = tiled_mma.partition_shape_C((self.BK, self.BV))
            h_accs = []
            for _ki in cutlass.range_constexpr(self.NK):
                _acc = cute.make_fragment(acc_h_shape, Float32)
                _acc.fill(0.0)
                h_accs.append(_acc)

            # === Identity tensors ===
            mc_tv = cute.make_identity_tensor((self.BT, self.BV))
            tCcTV = thr_mma.partition_C(mc_tv)
            mc_AA = cute.make_identity_tensor((self.BT, self.BT))
            tCcAA = thr_mma.partition_C(mc_AA)

            # === Global tensors ===
            kw_base = (tile_B * Int32(self.T * self.H) + tile_H) * Int32(self.K)
            uv_base = (tile_B * Int32(self.T * self.H) + tile_H) * Int32(self.V)
            g_head_base = tile_B * Int32(self.T * self.H) + tile_H

            gQ_head = cute.make_tensor(
                (q.iterator + kw_base).align(16),
                cute.make_layout((self.T, self.K), stride=(self.H * self.K, 1)))
            gK_head = cute.make_tensor(
                (k.iterator + kw_base).align(16),
                cute.make_layout((self.T, self.K), stride=(self.H * self.K, 1)))
            gW_head = cute.make_tensor(
                (w.iterator + kw_base).align(16),
                cute.make_layout((self.T, self.K), stride=(self.H * self.K, 1)))
            gU_head = cute.make_tensor(
                (u.iterator + uv_base).align(16),
                cute.make_layout((self.T, self.V), stride=(self.H * self.V, 1)))
            gG_head = cute.make_tensor(g_cumsum.iterator + g_head_base,
                cute.make_layout((self.T,), stride=(self.H,)))
            gO_head = cute.make_tensor(
                (o.iterator + uv_base).align(16),
                cute.make_layout((self.T, self.V), stride=(self.H * self.V, 1)))

            # === Prologue: issue u[0] cpasync ===
            _gU_pre = cute.local_tile(gU_head, (self.BT, self.BV), (Int32(0), tile_V))
            _tU_g_pre = uv_thr_copy.partition_S(_gU_pre)
            for ci in cutlass.range_constexpr(cute.size(tU_s.shape[2])):
                cute.copy(uv_tiled_copy, _tU_g_pre[None, None, ci], tU_s[None, None, ci])
            cute.arch.cp_async_commit_group()

            # ===================================================================
            # Main chunk loop
            # ===================================================================
            chunk_idx = Int32(0)
            while chunk_idx < Int32(self.NT_val):

                # --- Load Q[BT,K] via cpasync into s_wk buffers ---
                for qi in cutlass.range_constexpr(self.NK):
                    gQ_tile = cute.local_tile(gQ_head, (self.BT, self.BK),
                                              (chunk_idx, Int32(qi)))
                    tQ_g = wk_thr_copy.partition_S(gQ_tile)
                    for ci in cutlass.range_constexpr(cute.size(tWK_s_bufs[qi].shape[2])):
                        cute.copy(wk_tiled_copy, tQ_g[None, None, ci],
                                  tWK_s_bufs[qi][None, None, ci])
                cute.arch.cp_async_commit_group()

                # Load g_cumsum while cpasync is in flight
                gG_tile = cute.local_tile(gG_head, (self.BT,), (chunk_idx,))
                if tidx < Int32(self.BT):
                    g_buf[tidx] = gG_tile[tidx]

                # Wait for u (from prologue/prev chunk) AND Q
                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # Q smem → registers (full K, per BK-block)
                tCrQ_full_parts = []
                for qi in cutlass.range_constexpr(self.NK):
                    tCsQ_part = thr_mma.partition_A(s_wk_bufs[qi])
                    tCrQ_part = tiled_mma.make_fragment_A(tCsQ_part)
                    for qkb in cutlass.range_constexpr(self.BK // 16):
                        cute.autovec_copy(tCsQ_part[None, None, qkb], tCrQ_part[None, None, qkb])
                    tCrQ_full_parts.append(tCrQ_part)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # u stays in smem (read via partition_C)
                tCsU = thr_mma.partition_C(s_u)

                # -------------------------------------------------------
                # Phase A: K-loop — 3 fused GEMMs
                # -------------------------------------------------------
                acc_vp = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), Float32)
                acc_vp.fill(0.0)
                acc_o = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), Float32)
                acc_o.fill(0.0)
                acc_A = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BT)), Float32)
                acc_A.fill(0.0)

                for ki in cutlass.range_constexpr(self.NK):
                    cur = ki % 2

                    # h[ki] → s_h (fp32→fp16)
                    h_tmp = cute.make_fragment_like(h_accs[ki], self.q_dtype)
                    for ci in cutlass.range_constexpr(cute.size(h_accs[ki])):
                        h_tmp[ci] = h_accs[ki][ci].to(self.q_dtype)
                    tOrH = smem_thr_copy_C.retile(h_tmp)
                    tOsH = smem_thr_copy_C.partition_D(s_h)
                    cute.copy(smem_tiled_copy_C, tOrH, tOsH)

                    # Load w[ki] → s_wk[cur]
                    gW_tile = cute.local_tile(gW_head, (self.BT, self.BK),
                                              (chunk_idx, Int32(ki)))
                    tWK_g = wk_thr_copy.partition_S(gW_tile)
                    for ci in cutlass.range_constexpr(cute.size(tWK_s_bufs[cur].shape[2])):
                        cute.copy(wk_tiled_copy, tWK_g[None, None, ci],
                                  tWK_s_bufs[cur][None, None, ci])
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                    # GEMM1-state + GEMM1-output (shared h B operand)
                    for kb in cutlass.range_constexpr(self.BK // 16):
                        cute.copy(smem_tiled_copy_B, tHsH[None, None, kb],
                                  tBrB_view[None, None, kb])
                        cute.autovec_copy(tCsA_bufs[cur][None, None, kb],
                                          tCrA[None, None, kb])
                        # acc_vp += w @ h
                        cute.gemm(tiled_mma, acc_vp,
                                  tCrA[None, None, kb], tCrB[None, None, kb], acc_vp)
                        # acc_o += Q @ h
                        cute.gemm(tiled_mma, acc_o,
                                  tCrQ_full_parts[ki][None, None, kb],
                                  tCrB[None, None, kb], acc_o)

                    named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                    # Load K[ki] → s_wk[cur] for GEMM2-output
                    gK_tile = cute.local_tile(gK_head, (self.BT, self.BK),
                                              (chunk_idx, Int32(ki)))
                    tK_g = wk_thr_copy.partition_S(gK_tile)
                    for ci in cutlass.range_constexpr(cute.size(tWK_s_bufs[cur].shape[2])):
                        cute.copy(wk_tiled_copy, tK_g[None, None, ci],
                                  tWK_s_bufs[cur][None, None, ci])
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                    # GEMM2-output: acc_A += Q @ K^T
                    for kb in cutlass.range_constexpr(self.BK // 16):
                        cute.copy(smem_tiled_copy_B, tKsKt_bufs[cur][None, None, kb],
                                  tKrKt_view[None, None, kb])
                        cute.gemm(tiled_mma, acc_A,
                                  tCrQ_full_parts[ki][None, None, kb],
                                  tCrKt[None, None, kb], acc_A)

                    if ki < self.NK - 1:
                        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # -------------------------------------------------------
                # Phase B: Gating on acc_o, gating+mask on acc_A
                # -------------------------------------------------------
                for ci in cutlass.range_constexpr(cute.size(acc_o)):
                    row = tCcTV[ci][0]
                    acc_o[ci] = acc_o[ci] * cute.math.exp(g_buf[Int32(row)], fastmath=True)

                for ci in cutlass.range_constexpr(cute.size(acc_A)):
                    row = tCcAA[ci][0]
                    col = tCcAA[ci][1]
                    if Int32(col) > Int32(row):
                        acc_A[ci] = Float32(0.0)
                    else:
                        acc_A[ci] = acc_A[ci] * cute.math.exp(
                            g_buf[Int32(row)] - g_buf[Int32(col)], fastmath=True)

                # -------------------------------------------------------
                # Phase C: v_new, scores → smem for GEMM3
                # -------------------------------------------------------
                g_last = g_buf[Int32(self.BT - 1)]

                # Compute v_new (non-gated) and v_gated in one pass
                vn_regs = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), self.q_dtype)
                vgated_regs = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), self.q_dtype)
                for ci in cutlass.range_constexpr(cute.size(acc_vp)):
                    vn_val = tCsU[ci].to(Float32) - acc_vp[ci]
                    vn_regs[ci] = vn_val.to(self.q_dtype)
                    row = tCcTV[ci][0]
                    gate = cute.math.exp(g_last - g_buf[Int32(row)], fastmath=True)
                    vgated_regs[ci] = (vn_val * gate).to(self.q_dtype)

                # Write non-gated v_new → s_v for GEMM3
                tOrVN = smem_thr_copy_C.retile(vn_regs)
                tOsVN = smem_thr_copy_C.partition_D(s_v)
                cute.copy(smem_tiled_copy_C, tOrVN, tOsVN)

                # Write scores → s_a (at page_ptr, [BT,BT])
                a_tmp = cute.make_fragment_like(acc_A, self.q_dtype)
                for ci in cutlass.range_constexpr(cute.size(acc_A)):
                    a_tmp[ci] = acc_A[ci].to(self.q_dtype)
                tOrA = smem_thr_copy_C.retile(a_tmp)
                tOsA = smem_thr_copy_C.partition_D(s_a)
                cute.copy(smem_tiled_copy_C, tOrA, tOsA)

                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # -------------------------------------------------------
                # Phase D: GEMM3 — acc_intra = scores @ v_new
                # -------------------------------------------------------
                acc_intra = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), Float32)
                acc_intra.fill(0.0)

                for kb in cutlass.range_constexpr(self.BT // 16):
                    cute.copy(smem_tiled_copy_B, tVsVt[None, None, kb],
                              tVrVt_view[None, None, kb])
                    cute.autovec_copy(tCsA_scores[None, None, kb],
                                      tCrA_scores[None, None, kb])
                    cute.gemm(tiled_mma, acc_intra,
                              tCrA_scores[None, None, kb],
                              tCrVt[None, None, kb], acc_intra)

                # -------------------------------------------------------
                # Phase E: Write O to global (element-wise)
                # -------------------------------------------------------
                gO_tile = cute.local_tile(gO_head, (self.BT, self.BV),
                                          (chunk_idx, tile_V))
                for ci in cutlass.range_constexpr(cute.size(acc_o)):
                    row = tCcTV[ci][0]
                    col = tCcTV[ci][1]
                    o_val = (acc_o[ci] + acc_intra[ci]) * Float32(self.scale_val)
                    gO_tile[Int32(row), Int32(col)] = o_val.to(self.q_dtype)

                # -------------------------------------------------------
                # Phase F: State GEMM2 — h[ki] = decay*h[ki] + k^T @ v_gated
                # -------------------------------------------------------
                # Write gated v_new → s_v (overwrite non-gated)
                tOrVG = smem_thr_copy_C.retile(vgated_regs)
                tOsVG = smem_thr_copy_C.partition_D(s_v)
                cute.copy(smem_tiled_copy_C, tOrVG, tOsVG)

                # Load k[0] → s_wk0
                gK_tile0 = cute.local_tile(gK_head, (self.BT, self.BK),
                                            (chunk_idx, Int32(0)))
                tK_g0 = wk_thr_copy.partition_S(gK_tile0)
                for ci in cutlass.range_constexpr(cute.size(tWK_s_bufs[0].shape[2])):
                    cute.copy(wk_tiled_copy, tK_g0[None, None, ci],
                              tWK_s_bufs[0][None, None, ci])
                cute.arch.cp_async_commit_group()

                decay = cute.math.exp(g_last, fastmath=True)

                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                for ki in cutlass.range_constexpr(self.NK):
                    cur = ki % 2

                    if ki > 0:
                        cute.arch.cp_async_wait_group(0)
                        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                    if ki < self.NK - 1:
                        nxt = (ki + 1) % 2
                        gK_tile_nxt = cute.local_tile(gK_head, (self.BT, self.BK),
                                                       (chunk_idx, Int32(ki + 1)))
                        tK_g_nxt = wk_thr_copy.partition_S(gK_tile_nxt)
                        for ci in cutlass.range_constexpr(cute.size(tWK_s_bufs[nxt].shape[2])):
                            cute.copy(wk_tiled_copy, tK_g_nxt[None, None, ci],
                                      tWK_s_bufs[nxt][None, None, ci])
                        cute.arch.cp_async_commit_group()

                    # Decay h_accs[ki]
                    for ci in cutlass.range_constexpr(cute.size(h_accs[ki])):
                        h_accs[ki][ci] = h_accs[ki][ci] * decay

                    # MMA: h[ki] += k^T[ki] @ v_gated
                    for kb in cutlass.range_constexpr(self.BT // 16):
                        cute.copy(smem_tiled_copy_B, tVsVt[None, None, kb],
                                  tBrB_view[None, None, kb])
                        cute.autovec_copy(tCsKt_bufs[cur][None, None, kb],
                                          tCrA[None, None, kb])
                        cute.gemm(tiled_mma, h_accs[ki],
                                  tCrA[None, None, kb], tCrB[None, None, kb], h_accs[ki])

                # Pipeline: issue u[chunk+1] cpasync
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


__all__ = ["GDNFusedOp"]
