# Copyright (c) 2025, Machete Authors
"""Gated Delta Net State Recurrence — Megakernel Op (cooperative cpasync + tensor core MMA).

Processes all NT chunks sequentially inside compute_mma, accumulating the
hidden state h[BK,BV] in register accumulators across chunks.

Two GEMMs per chunk:
    GEMM1: v_partial[BT,BV] += w[BT,BK] @ h[BK,BV]  (per K-block)
    GEMM2: h[BK,BV]        += k^T[BK,BT] @ v_gated[BT,BV]  (per K-block)

Architecture (same as GDNPrepOp / GDNOutputOp):
    DMA warp:  Idle (no TMA loads or stores)
    MMA warps: All warps cooperatively load data via cpasync AND compute MMA.

Optimizations (A+B+D+E):
    A. Merged barriers: h→s_h + w→s_wk issued together (one barrier instead
       of two); last K-block post-MMA barrier skipped (no smem conflict after).
    B. Double-buffered s_wk: cpasync for next K-block overlaps with current
       MMA, hiding global memory latency behind tensor-core compute.
    D. Early cpasync prefetch: w[0] issued during h_stores, k[0] issued before
       v_new compute — hides long_scoreboard stalls.
    E. BV=64 V-tiling: maximizes tensor core utilization per tile.

Usage:
    from machete.kernels.gated_delta_net.state_op import GDNStateOp
    from machete.megakernel import Megakernel

    ops = GDNStateOp.schedule_forward(
        k=k, w=w, u=u, g_cumsum=g_cumsum,
        h=h, v_new=v_new,
    )
    config = GDNStateOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)
    kernel.run()
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu import warp
from cutlass.cute.nvgpu.cpasync import CopyBulkS2GOp, group_bulk_copy_modes

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import named_barrier_sync


# Block sizes (matching standalone state_cuda.py)
_BT = 64   # Chunk size
_BK = 64   # K-block
_BV = 64   # V-strip (doubled for higher tensor core utilization)


class GDNStateOp(Op):
    """Gated Delta Net State Recurrence — cooperative cpasync + MMA megakernel Op.

    Tensors (native [B, T, H, K/V] layout, no transposes):
        k:       (B, T, H, K)  -- keys  (fp16 or bf16)
        w:       (B, T, H, K)  -- transformed keys from prep
        u:       (B, T, H, V)  -- transformed values from prep
        g_cumsum:(B, T, H)     -- cumulative gates (fp32)
        h:       (B, NT, H, K, V) -- output: inter-chunk states
        v_new:   (B, T, H, V)  -- output: corrected values

    Tiling:
        tile_B=1, tile_H=1 (per batch-head), tile_V=BV (V-strip).
        All T chunks and K-blocks are looped over inside compute.
    """

    reads = {
        "k":       (None, ("B", "T", "H", "K")),
        "w":       (None, ("B", "T", "H", "K")),
        "u":       (None, ("B", "T", "H", "V")),
        "g_cumsum": (cutlass.Float32, ("B", "T", "H")),
    }
    writes = {
        "h":     (None, ("B", "NT", "H", "K", "V")),
        "v_new": (None, ("B", "T", "H", "V")),
    }
    tile = ("B", "H", "V")

    def __init__(self, **config):
        super().__init__(**config)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)

        assert self.k_dtype in (cutlass.Float16, cutlass.BFloat16), (
            f"GDNStateOp requires fp16 or bf16, got {self.k_dtype}"
        )
        self.elem_bytes = 2

        # Block sizes
        self.BT = _BT
        self.BK = _BK
        self.BV = _BV
        self.NK = self.K // self.BK
        self.NV = self.V // self.BV
        self.NT_val = self.T // self.BT

        assert self.K % self.BK == 0
        assert self.V % self.BV == 0
        assert self.T % self.BT == 0

        # MMA setup
        self.num_mma_warps = self.BT // 16  # 4 warps for BT=64
        self.num_mma_threads = self.num_mma_warps * 32

        # Swizzle for BK-sized buffers (s_wk)
        if self.BK >= 64:
            self.swizzle_B = 3
        elif self.BK >= 32:
            self.swizzle_B = 2
        else:
            self.swizzle_B = 1

        # Swizzle for BV-sized buffers (s_h, s_v)
        if self.BV >= 64:
            self.swizzle_B_v = 3
        elif self.BV >= 32:
            self.swizzle_B_v = 2
        else:
            self.swizzle_B_v = 1

        # cpasync thread layout for BK-wide loads (w, k)
        self.async_copy_elems = 128 // (self.elem_bytes * 8)  # 8 for fp16/bf16
        self.wk_copy_dim1 = self.BK // self.async_copy_elems
        self.wk_copy_dim0 = self.num_mma_threads // self.wk_copy_dim1

        # cpasync thread layout for BV-wide loads (u)
        self.uv_copy_dim1 = self.BV // self.async_copy_elems
        self.uv_copy_dim0 = self.num_mma_threads // self.uv_copy_dim1

        # DMA does nothing — all work in compute
        self.inner_iters = 1
        self.inner_depth = 1

        # Smem layout: manual byte offsets from page_ptr
        # Double-buffered s_wk for pipelined cpasync + MMA overlap
        wk_bytes = self.BT * self.BK * self.elem_bytes  # 8KB per buffer

        # s_wk0: [BT, BK] swizzled at offset 0
        self._swk0_offset = 0
        # s_wk1: [BT, BK] swizzled at offset 8KB (128B aligned: 8192 % 128 == 0)
        self._swk1_offset = wk_bytes
        # s_h: [BK, BV] swizzled after s_wk1, 128B aligned
        self._sh_offset = self._swk1_offset + wk_bytes
        self._sh_offset = ((self._sh_offset + 127) // 128) * 128
        # s_uv: [BT, BV] after s_h, 128B aligned
        self._suv_offset = self._sh_offset + self.BK * self.BV * self.elem_bytes
        self._suv_offset = ((self._suv_offset + 127) // 128) * 128
        # g_buf: [BT] fp32 after s_uv, 16B aligned
        self._gbuf_offset = self._suv_offset + self.BT * self.BV * self.elem_bytes
        self._gbuf_offset = ((self._gbuf_offset + 15) // 16) * 16
        # Total smem needed
        self._total_smem = self._gbuf_offset + self.BT * 4

        # Override compute method
        self.compute = self.compute_mma

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule_forward(cls, page_size=None, tile_sizes=None, **tensors):
        """Schedule GDN state recurrence Op."""
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("H", 1)
        tile_sizes.setdefault("V", _BV)

        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        # Compute required page_size from smem layout (double-buffered s_wk)
        if page_size is None:
            eb = 2  # fp16/bf16
            K = tensors["k"].shape[-1]
            V = tensors["u"].shape[-1]
            T = tensors["k"].shape[1]
            wk_bytes = _BT * _BK * eb
            sh_off = 2 * wk_bytes  # after two s_wk buffers
            sh_off = ((sh_off + 127) // 128) * 128
            suv_off = sh_off + _BK * _BV * eb
            suv_off = ((suv_off + 127) // 128) * 128
            gbuf_off = suv_off + _BT * _BV * eb
            gbuf_off = ((gbuf_off + 15) // 16) * 16
            page_size = gbuf_off + _BT * 4
        ops[0].static_dims["page_size"] = page_size
        return ops

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for the given scheduled ops."""
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS

        num_mma_warps = _BT // 16  # 4 warps
        threads_per_block = (num_mma_warps + NUM_DMA_WARPS) * 32
        # Take max page_size across all ops for fused megakernel compatibility
        page_size = max(
            op.static_dims.get("page_size", DEFAULT_PAGE_SIZE) for op in ops
        )
        return MegakernelConfig(
            threads_per_block=threads_per_block,
            page_size=page_size,
        )

    # =========================================================================
    # Forward Compute — Cooperative cpasync + Tensor Core MMA
    # =========================================================================

    @cute.jit
    def compute_mma(
        self, page_ptr, tile_B, tile_H, tile_V,
        k, w, u, g_cumsum, h, v_new,
    ):
        """Cooperative GDN state recurrence with merged barriers + double-buffered s_wk."""
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        if warp_idx < Int32(self.num_mma_warps):
            # === Swizzles ===
            swz = cute.make_swizzle(self.swizzle_B, 4, 3)      # BK-wide (s_wk)
            swz_v = cute.make_swizzle(self.swizzle_B_v, 4, 3)  # BV-wide (s_h, s_v)

            # === MMA setup ===
            mma_op = warp.MmaF16BF16Op(self.k_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((self.num_mma_warps, 1, 1)),
                permutation_mnk=(self.num_mma_warps * 16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)

            # === Smem regions ===
            # Double-buffered s_wk: [BT, BK] swizzled × 2
            s_wk_bufs = []
            for buf_idx in cutlass.range_constexpr(2):
                offset = self._swk0_offset if buf_idx == 0 else self._swk1_offset
                s = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(self.k_dtype, page_ptr + Int32(offset),
                                      cute.AddressSpace.smem, assumed_align=128),
                        swz, dtype=self.k_dtype,
                    ),
                    cute.make_layout((self.BT, self.BK), stride=(self.BK, 1)),
                )
                s_wk_bufs.append(s)

            # s_h: [BK, BV] swizzled — for h accumulator → fp16 → MMA B operand
            s_h = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.k_dtype, page_ptr + Int32(self._sh_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz_v, dtype=self.k_dtype,
                ),
                cute.make_layout((self.BK, self.BV), stride=(self.BV, 1)),
            )
            # s_u: [BT, BV] NOT swizzled — read via partition_C indexing
            s_u = cute.make_tensor(
                cute.make_ptr(self.k_dtype, page_ptr + Int32(self._suv_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.BT, self.BV), stride=(self.BV, 1)),
            )
            # s_v: [BT, BV] swizzled — same physical region as s_u, for GEMM2 B operand
            s_v = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.k_dtype, page_ptr + Int32(self._suv_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz_v, dtype=self.k_dtype,
                ),
                cute.make_layout((self.BT, self.BV), stride=(self.BV, 1)),
            )
            # g_buf: [BT] fp32
            g_buf = cute.make_tensor(
                cute.make_ptr(Float32, page_ptr + Int32(self._gbuf_offset),
                              cute.AddressSpace.smem, assumed_align=16),
                cute.make_layout(self.BT),
            )

            # === LdMatrix B for swizzled B operands ===
            smem_copy_atom_B = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                self.k_dtype,
            )
            smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
            smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

            # CopyUniversal for C accumulators → swizzled smem
            smem_copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.k_dtype)
            smem_tiled_copy_C = cute.make_tiled_copy_C(smem_copy_atom_C, tiled_mma)
            smem_thr_copy_C = smem_tiled_copy_C.get_slice(tidx)

            # === MMA partitions ===
            # Shared B fragment: both GEMM1 (h) and GEMM2 (v_gated) use [BV, 64]
            # B operands with same shape, so we share one register fragment.
            s_h_B = cute.make_tensor(s_h.iterator,
                cute.make_layout((self.BV, self.BK), stride=(1, self.BV)))
            _tBsH = thr_mma.partition_B(s_h_B)
            tCrB = tiled_mma.make_fragment_B(_tBsH)
            tBrB_view = smem_thr_copy_B.retile(tCrB)
            tHsH = smem_thr_copy_B.partition_S(s_h_B)

            # v_gated smem partition (same fragment tCrB reused)
            s_v_B = cute.make_tensor(s_v.iterator,
                cute.make_layout((self.BV, self.BT), stride=(1, self.BV)))
            tVsVt = smem_thr_copy_B.partition_S(s_v_B)

            # Per-buffer MMA A partitions and cpasync destinations
            tCsA_bufs = []
            tWK_s_bufs = []
            # Transposed views for GEMM2 (k^T)
            tCsKt_bufs = []
            for buf_idx in cutlass.range_constexpr(2):
                tCsA_bufs.append(thr_mma.partition_A(s_wk_bufs[buf_idx]))
                tCsKt_bufs.append(thr_mma.partition_A(
                    cute.make_tensor(s_wk_bufs[buf_idx].iterator,
                        cute.make_layout((self.BK, self.BT), stride=(1, self.BK)))
                ))

            # MMA A register fragment (same shape for both buffers)
            tCrA = tiled_mma.make_fragment_A(tCsA_bufs[0])

            # === cpasync setup for BK-wide loads (w, k) ===
            wk_async_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), self.k_dtype,
                num_bits_per_copy=128)
            wk_tiled_copy = cute.make_tiled_copy_tv(
                wk_async_atom,
                cute.make_layout((self.wk_copy_dim0, self.wk_copy_dim1),
                                 stride=(self.wk_copy_dim1, 1)),
                cute.make_layout((1, self.async_copy_elems)),
            )
            wk_thr_copy = wk_tiled_copy.get_slice(tidx)

            # Per-buffer cpasync destinations
            for buf_idx in cutlass.range_constexpr(2):
                tWK_s_bufs.append(wk_thr_copy.partition_D(s_wk_bufs[buf_idx]))

            # cpasync setup for BV-wide loads (u)
            uv_async_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), self.k_dtype,
                num_bits_per_copy=128)
            uv_tiled_copy = cute.make_tiled_copy_tv(
                uv_async_atom,
                cute.make_layout((self.uv_copy_dim0, self.uv_copy_dim1),
                                 stride=(self.uv_copy_dim1, 1)),
                cute.make_layout((1, self.async_copy_elems)),
            )
            uv_thr_copy = uv_tiled_copy.get_slice(tidx)
            tU_s = uv_thr_copy.partition_D(s_u)

            # === Persistent h accumulators (in registers, across chunks) ===
            acc_h_shape = tiled_mma.partition_shape_C((self.BK, self.BV))
            h_accs = []
            for _ki in cutlass.range_constexpr(self.NK):
                _acc = cute.make_fragment(acc_h_shape, Float32)
                _acc.fill(0.0)
                h_accs.append(_acc)

            # === Identity tensors for coordinate extraction ===
            mc_h = cute.make_identity_tensor((self.BK, self.BV))
            tCcH = thr_mma.partition_C(mc_h)
            mc_tv = cute.make_identity_tensor((self.BT, self.BV))
            tCcTV = thr_mma.partition_C(mc_tv)

            # === Per-head global tensors (strided [B, T, H, K/V] layout) ===
            # Compute base offsets (scalars, not full tensor objects)
            kw_base = (tile_B * Int32(self.T * self.H) + tile_H) * Int32(self.K)
            uv_base = (tile_B * Int32(self.T * self.H) + tile_H) * Int32(self.V)
            g_head_base = tile_B * Int32(self.T * self.H) + tile_H
            h_head_base = (tile_B * Int32(self.NT_val * self.H) + tile_H) * Int32(self.K * self.V)

            # K-dim tensors (k, w) — needed throughout
            gK_head = cute.make_tensor(
                (k.iterator + kw_base).align(16),
                cute.make_layout((self.T, self.K), stride=(self.H * self.K, 1)))
            gW_head = cute.make_tensor(
                (w.iterator + kw_base).align(16),
                cute.make_layout((self.T, self.K), stride=(self.H * self.K, 1)))

            # V-dim tensor (u) — needed for cpasync + v_new compute
            gU_head = cute.make_tensor(
                (u.iterator + uv_base).align(16),
                cute.make_layout((self.T, self.V), stride=(self.H * self.V, 1)))

            gG_head = cute.make_tensor(g_cumsum.iterator + g_head_base,
                cute.make_layout((self.T,), stride=(self.H,)))

            # === Prologue: issue u[0] cpasync (pipelined, waited in first iteration) ===
            _gU_pre = cute.local_tile(
                gU_head, (self.BT, self.BV), (Int32(0), tile_V))
            _tU_g_pre = uv_thr_copy.partition_S(_gU_pre)
            for ci in cutlass.range_constexpr(cute.size(tU_s.shape[2])):
                cute.copy(uv_tiled_copy, _tU_g_pre[None, None, ci],
                          tU_s[None, None, ci])
            cute.arch.cp_async_commit_group()

            # === Main chunk loop ===
            chunk_idx = Int32(0)
            while chunk_idx < Int32(self.NT_val):

                # --- 1. Store h_states[chunk] to global ---
                # h_states: [B, NT, H, K, V] — stride between chunks = H * K * V
                h_chunk_ptr = h.iterator + h_head_base + chunk_idx * Int32(self.H * self.K * self.V)
                for ki in cutlass.range_constexpr(self.NK):
                    gH_chunk = cute.make_tensor(
                        h_chunk_ptr + Int32(ki * self.BK * self.V),
                        cute.make_layout((self.BK, self.V), stride=(self.V, 1)))
                    gH_tile = cute.local_tile(
                        gH_chunk, (self.BK, self.BV), (Int32(0), tile_V))
                    for ci in cutlass.range_constexpr(cute.size(h_accs[ki])):
                        row = tCcH[ci][0]
                        col = tCcH[ci][1]
                        gH_tile[row, col] = h_accs[ki][ci].to(self.k_dtype)

                # --- D. Prefetch w[0]→s_wk0 (pipelined during g load + u→regs) ---
                gW_tile = cute.local_tile(
                    gW_head, (self.BT, self.BK), (chunk_idx, Int32(0)))
                tWK_g = wk_thr_copy.partition_S(gW_tile)
                for ci in cutlass.range_constexpr(cute.size(tWK_s_bufs[0].shape[2])):
                    cute.copy(wk_tiled_copy, tWK_g[None, None, ci],
                              tWK_s_bufs[0][None, None, ci])
                cute.arch.cp_async_commit_group()

                # --- 2. Wait for u AND w[0] cpasync + load g ---
                gG_tile = cute.local_tile(
                    gG_head, (self.BT,), (chunk_idx,))
                if tidx < Int32(self.BT):
                    g_buf[tidx] = gG_tile[tidx]

                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # --- 3. u stays in smem (read directly in step 5) ---
                tCsU = thr_mma.partition_C(s_u)

                # --- 4. GEMM1: v_partial = Σ_ki w[ki] @ h[ki] ---
                # Merged barriers: h→s_h + w→s_wk issued together, one barrier.
                # Double-buffered s_wk: prefetch w[ki+1] during MMA of ki.
                acc_vp = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), Float32)
                acc_vp.fill(0.0)

                # h[0]→s_h — w[0] already waited at step 2
                h_tmp = cute.make_fragment_like(h_accs[0], self.k_dtype)
                for ci in cutlass.range_constexpr(cute.size(h_accs[0])):
                    h_tmp[ci] = h_accs[0][ci].to(self.k_dtype)
                tOrH = smem_thr_copy_C.retile(h_tmp)
                tOsH = smem_thr_copy_C.partition_D(s_h)
                cute.copy(smem_tiled_copy_C, tOrH, tOsH)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # GEMM1 K-block loop
                for ki in cutlass.range_constexpr(self.NK):
                    cur = ki % 2

                    if ki > 0:
                        # h[ki] → s_h (CopyUniversal, immediate)
                        h_tmp = cute.make_fragment_like(h_accs[ki], self.k_dtype)
                        for ci in cutlass.range_constexpr(cute.size(h_accs[ki])):
                            h_tmp[ci] = h_accs[ki][ci].to(self.k_dtype)
                        tOrH = smem_thr_copy_C.retile(h_tmp)
                        tOsH = smem_thr_copy_C.partition_D(s_h)
                        cute.copy(smem_tiled_copy_C, tOrH, tOsH)
                        # Wait for prefetched w[ki] in s_wk[cur]
                        cute.arch.cp_async_wait_group(0)
                        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                    # Prefetch w[ki+1] → s_wk[nxt] (overlaps with MMA below)
                    if ki < self.NK - 1:
                        nxt = (ki + 1) % 2
                        gW_tile_nxt = cute.local_tile(
                            gW_head, (self.BT, self.BK), (chunk_idx, Int32(ki + 1)))
                        tWK_g_nxt = wk_thr_copy.partition_S(gW_tile_nxt)
                        for ci in cutlass.range_constexpr(cute.size(tWK_s_bufs[nxt].shape[2])):
                            cute.copy(wk_tiled_copy, tWK_g_nxt[None, None, ci],
                                      tWK_s_bufs[nxt][None, None, ci])
                        cute.arch.cp_async_commit_group()

                    # MMA: acc_vp += w[ki] @ h[ki]
                    for kb in cutlass.range_constexpr(self.BK // 16):
                        cute.copy(smem_tiled_copy_B, tHsH[None, None, kb],
                                  tBrB_view[None, None, kb])
                        cute.autovec_copy(tCsA_bufs[cur][None, None, kb],
                                          tCrA[None, None, kb])
                        cute.gemm(tiled_mma, acc_vp,
                                  tCrA[None, None, kb],
                                  tCrB[None, None, kb], acc_vp)

                    # Barrier after MMA to protect s_h before next ki overwrites it
                    # (skip at last K-block — no more overwrites coming)
                    if ki < self.NK - 1:
                        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # --- D. Prefetch k[0]→s_wk0 (pipelined during v_new + decay) ---
                gK_tile = cute.local_tile(
                    gK_head, (self.BT, self.BK), (chunk_idx, Int32(0)))
                tK_g = wk_thr_copy.partition_S(gK_tile)
                for ci in cutlass.range_constexpr(cute.size(tWK_s_bufs[0].shape[2])):
                    cute.copy(wk_tiled_copy, tK_g[None, None, ci],
                              tWK_s_bufs[0][None, None, ci])
                cute.arch.cp_async_commit_group()

                # --- 5. v_new = u - v_partial, gate, store ---
                g_last = g_buf[Int32(self.BT - 1)]

                gVN_head = cute.make_tensor(
                    (v_new.iterator + uv_base).align(16),
                    cute.make_layout((self.T, self.V), stride=(self.H * self.V, 1)))
                gVN_tile = cute.local_tile(
                    gVN_head, (self.BT, self.BV), (chunk_idx, tile_V))

                vgated_regs = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), self.k_dtype)

                for ci in cutlass.range_constexpr(cute.size(acc_vp)):
                    vn_val = tCsU[ci].to(Float32) - acc_vp[ci]
                    vn_fp16 = vn_val.to(self.k_dtype)
                    row = tCcTV[ci][0]
                    col = tCcTV[ci][1]
                    gVN_tile[row, col] = vn_fp16
                    gate = cute.math.exp(g_last - g_buf[Int32(row)],
                                         fastmath=True)
                    vgated_regs[ci] = (vn_val * gate).to(self.k_dtype)

                # Write gated v → swizzled s_v (CopyUniversal, for GEMM2)
                tOrV = smem_thr_copy_C.retile(vgated_regs)
                tOsV = smem_thr_copy_C.partition_D(s_v)
                cute.copy(smem_tiled_copy_C, tOrV, tOsV)

                # --- 6. Decay h ---
                decay = cute.math.exp(g_last, fastmath=True)
                for ki in cutlass.range_constexpr(self.NK):
                    for ci in cutlass.range_constexpr(cute.size(h_accs[ki])):
                        h_accs[ki][ci] = h_accs[ki][ci] * decay

                # --- 7. GEMM2: h += k^T @ v_gated ---
                # k[0] was prefetched before step 5; wait for completion
                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # GEMM2 K-block loop
                for ki in cutlass.range_constexpr(self.NK):
                    cur = ki % 2

                    if ki > 0:
                        # Wait for prefetched k[ki] in s_wk[cur]
                        cute.arch.cp_async_wait_group(0)
                        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                    # Prefetch k[ki+1] → s_wk[nxt] (overlaps with MMA below)
                    if ki < self.NK - 1:
                        nxt = (ki + 1) % 2
                        gK_tile_nxt = cute.local_tile(
                            gK_head, (self.BT, self.BK), (chunk_idx, Int32(ki + 1)))
                        tK_g_nxt = wk_thr_copy.partition_S(gK_tile_nxt)
                        for ci in cutlass.range_constexpr(cute.size(tWK_s_bufs[nxt].shape[2])):
                            cute.copy(wk_tiled_copy, tK_g_nxt[None, None, ci],
                                      tWK_s_bufs[nxt][None, None, ci])
                        cute.arch.cp_async_commit_group()

                    # MMA: h[ki] += k^T[ki] @ v_gated
                    for kb in cutlass.range_constexpr(self.BT // 16):
                        cute.copy(smem_tiled_copy_B, tVsVt[None, None, kb],
                                  tBrB_view[None, None, kb])
                        cute.autovec_copy(tCsKt_bufs[cur][None, None, kb],
                                          tCrA[None, None, kb])
                        cute.gemm(tiled_mma, h_accs[ki],
                                  tCrA[None, None, kb],
                                  tCrB[None, None, kb], h_accs[ki])

                # Pipeline: issue u[chunk+1] cpasync (overlaps with next h_stores + g load)
                # s_u/s_v region is free (GEMM2 done reading s_v)
                next_chunk = chunk_idx + Int32(1)
                if next_chunk < Int32(self.NT_val):
                    gU_nxt = cute.local_tile(
                        gU_head, (self.BT, self.BV), (next_chunk, tile_V))
                    tU_g_nxt = uv_thr_copy.partition_S(gU_nxt)
                    for ci in cutlass.range_constexpr(cute.size(tU_s.shape[2])):
                        cute.copy(uv_tiled_copy, tU_g_nxt[None, None, ci],
                                  tU_s[None, None, ci])
                    cute.arch.cp_async_commit_group()

                chunk_idx = chunk_idx + Int32(1)


__all__ = ["GDNStateOp"]
