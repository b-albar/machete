# Copyright (c) 2025, Machete Authors
"""Gated Delta Net Output — Megakernel Op (cooperative cpasync + tensor core MMA).

Computes the final output for each chunk by combining:
    - Inter-chunk contribution: Q @ h_states[chunk] (cross-chunk state)
    - Intra-chunk contribution: causal_mask(Q @ K^T) @ v_new (local attention)

Both terms are gated by exp(g_cumsum) and scaled by 1/sqrt(K).

Architecture:
    DMA warp:  TMA Q load (single shot) + TMA O store.
    MMA warps: Copy Q smem→regs, then cpasync K/h/v_new + MMA compute.
               Output written to smem for TMA store.

Three GEMMs per tile:
    GEMM1 (inter-chunk): acc_o[BT,BV]  += Q[BT,BK] @ h[BK,BV]     (per K-block)
    GEMM2 (scores):      acc_A[BT,BT]  += Q[BT,BK] @ K^T[BK,BT]   (per K-block)
    GEMM3 (intra-chunk): acc_intra[BT,BV] += A_fp16[BT,BT] @ v_new[BT,BV]

Usage:
    from machete.kernels.gated_delta_net.output_op import GDNOutputOp
    from machete.megakernel import Megakernel

    ops = GDNOutputOp.schedule_forward(
        q=q, k=k, v_new=vn, h=h, g_cumsum=gc, o=o,
        scale=scale,
    )
    config = GDNOutputOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)
    kernel.run()
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu import warp

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import named_barrier_sync


# Block sizes (matching standalone output_cuda.py)
_BT = 64  # Chunk size
_BK = 64  # K-block
_BV = 64  # V-strip


class GDNOutputOp(Op):
    """Gated Delta Net Output — cooperative cpasync + MMA megakernel Op.

    Tensors (native [B, T, H, K/V] layout, no transposes):
        q:       (B, T, H, K)  -- queries  (fp16 or bf16)
        k:       (B, T, H, K)  -- keys
        v_new:   (B, T, H, V)  -- transformed values from state recurrence
        h:       (B, NT, H, K, V) -- inter-chunk states
        g_cumsum:(B, T, H)     -- cumulative gates (fp32)
        o:       (B, T, H, V)  -- output

    Tiling:
        tile_B=1, tile_H=1 (per batch-head), tile_T=BT=64 (per chunk),
        tile_V=BV=64 (V-strip).
    """

    reads = {
        "q":       (None, ("B", "T", "H", "K")),
        "k":       (None, ("B", "T", "H", "K")),
        "v_new":   (None, ("B", "T", "H", "V")),
        "h":       (None, ("B", "NT", "H", "K", "V")),
        "g_cumsum": (cutlass.Float32, ("B", "T", "H")),
    }
    writes = {
        "o": (None, ("B", "T", "H", "V")),
    }
    tile = ("B", "H", "T", "V")

    tma_loads = {"q"}
    tma_stores = {"o"}

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        """Swizzled smem layout for O TMA store descriptor."""
        if tensor_name != "o":
            return None

        BV = 64  # V-block size
        if BV >= 64:
            B = 3
        elif BV >= 32:
            B = 2
        else:
            B = 1

        # tma_tile_shape is (BV, 1, BT, 1) in reversed dim order (V, H, T, B)
        dim0, dim1, dim2, dim3 = tma_tile_shape
        return (
            f"cute.make_composed_layout("
            f"cute.make_swizzle({B}, 4, 3), 0, "
            f"cute.make_layout(({dim0}, {dim1}, {dim2}, {dim3}), "
            f"stride=(1, {dim0}, {dim0 * dim1}, {dim0 * dim1 * dim2})))"
        )

    def __init__(self, **config):
        super().__init__(**config)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)

        assert self.q_dtype in (cutlass.Float16, cutlass.BFloat16), (
            f"GDNOutputOp requires fp16 or bf16, got {self.q_dtype}"
        )
        self.elem_bytes = 2

        # Decode scale from int bits
        import struct
        self.scale_val = struct.unpack("f", struct.pack("I", self.scale_bits))[0]

        # Block sizes
        self.BT = _BT
        self.BK = _BK
        self.BV = _BV
        self.NK = self.K // self.BK
        self.NT_val = self.T // self.BT

        self.q_tile_bytes = self.BT * self.K * self.elem_bytes

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

        # cpasync thread layout for BK-wide loads (K)
        self.async_copy_elems = 128 // (self.elem_bytes * 8)  # 8 for fp16/bf16
        self.qk_copy_dim1 = self.BK // self.async_copy_elems
        self.qk_copy_dim0 = self.num_mma_threads // self.qk_copy_dim1

        # cpasync thread layout for BV-wide loads (h, v_new)
        self.hv_copy_dim1 = self.BV // self.async_copy_elems
        self.hv_copy_dim0 = self.num_mma_threads // self.hv_copy_dim1

        # DMA loads Q once, everything else in compute
        self.inner_iters = 1
        self.inner_depth = 1

        # Override compute method
        self.compute = self.compute_mma

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule_forward(cls, scale=None, page_size=DEFAULT_PAGE_SIZE, tile_sizes=None, **tensors):
        """Schedule GDN output Op."""
        import struct

        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("H", 1)
        tile_sizes.setdefault("T", _BT)
        tile_sizes.setdefault("V", _BV)

        q = tensors.get("q")
        if q is not None:
            K = q.shape[-1]
            if scale is None:
                scale = K ** -0.5

        scale_bits = struct.unpack("I", struct.pack("f", scale))[0]

        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        ops[0].static_dims["scale_bits"] = scale_bits
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
    # Forward Load (DMA warp: TMA Q only)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_H, tile_T, tile_V, q_tma, q_tma_gmem, work_mbar):
        """TMA Q load into page (single shot, full K columns)."""
        from machete.megakernel.interpreter import mbarrier_arrive_expect_tx

        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)

        # Smem: Q at page_ptr, TMA-ordered (K, H, T, B) → (K, 1, BT, 1)
        sQ = cute.make_tensor(
            cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.K, 1, self.BT, 1)),
        )
        gQ = cute.local_tile(
            q_tma_gmem,
            (self.K, 1, self.BT, 1),
            (None, None, None, None),
        )
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            q_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 4),
            cute.group_modes(gQ, 0, 4),
        )
        nbytes = Int32(self.q_tile_bytes)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        # Copy index: (None, K_rep=0, H_rep=tile_H, T_rep=tile_T, B_rep=tile_B)
        cute.copy(q_tma, tQgQ[(None, Int32(0), tile_H, tile_T, tile_B)], tQsQ,
                  tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Forward Compute — Cooperative cpasync + Tensor Core MMA
    # =========================================================================

    @cute.jit
    def compute_mma(
        self, page_ptr, tile_B, tile_H, tile_T, tile_V,
        q, k, v_new, h, g_cumsum, o,
    ):
        """Cooperative GDN output: TMA Q load + MMA warps do cpasync + 3 GEMMs.

        Phase 0: Read Q from smem (TMA-loaded) to registers, barrier to free smem.
        Phase 1: cpasync g_cumsum → smem, wait.
        Phase 2: K-loop: cpasync K,h → smem, GEMM1 (Q@h), GEMM2 (Q@K^T) — Q from regs.
        Phase 3: Apply gating + causal mask on accumulators.
        Phase 4: Write scores→smem, cpasync v_new→smem, GEMM3 (scores@v_new).
        Phase 5: Scale, write O→smem for TMA store.
        """
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        if warp_idx < Int32(self.num_mma_warps):
            # === Swizzle ===
            swz = cute.make_swizzle(self.swizzle_B, 4, 3)

            # === MMA setup ===
            mma_op = warp.MmaF16BF16Op(self.q_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((self.num_mma_warps, 1, 1)),
                permutation_mnk=(self.num_mma_warps * 16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)

            # =============================================================
            # Phase 0: Read Q from TMA-loaded smem to registers (full K)
            # =============================================================
            sQ_full = cute.make_tensor(
                cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.BT, self.K), stride=(self.K, 1)),
            )
            tCsQ_full = thr_mma.partition_A(sQ_full)
            tCrQ_full = tiled_mma.make_fragment_A(tCsQ_full)
            for qkb in cutlass.range_constexpr(self.K // 16):
                cute.autovec_copy(tCsQ_full[None, None, qkb], tCrQ_full[None, None, qkb])
            # Barrier: all MMA warps done reading Q, smem is free for reuse
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            # === Smem regions (page freed after Q→regs, shifted layout) ===
            # s_k: [BT, BK] swizzled (8KB) — at page_ptr
            s_k = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.q_dtype,
                ),
                cute.make_layout((self.BT, self.BK), stride=(self.BK, 1)),
            )
            # s_h: [BK, BV] swizzled (8KB) — after s_k
            _sk_offset = self.BT * self.BK * self.elem_bytes
            _sk_offset = ((_sk_offset + 127) // 128) * 128
            s_h = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr + Int32(_sk_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.q_dtype,
                ),
                cute.make_layout((self.BK, self.BV), stride=(self.BV, 1)),
            )
            # g_buf: [BT] fp32 (256B) — after s_h
            _sh_offset = _sk_offset + self.BK * self.BV * self.elem_bytes
            _sh_offset = ((_sh_offset + 127) // 128) * 128
            g_buf = cute.make_tensor(
                cute.make_ptr(Float32, page_ptr + Int32(_sh_offset),
                              cute.AddressSpace.smem, assumed_align=16),
                cute.make_layout(self.BT),
            )

            # After K-loop, reuse s_k region for scores + v_new
            # s_a: [BT, BT] swizzled (8KB) — at page_ptr (reuse s_k)
            s_a = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.q_dtype,
                ),
                cute.make_layout((self.BT, self.BT), stride=(self.BT, 1)),
            )
            # s_v: [BT, BV] swizzled (8KB) — at _sk_offset (reuse s_h)
            s_v = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr + Int32(_sk_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.q_dtype,
                ),
                cute.make_layout((self.BT, self.BV), stride=(self.BV, 1)),
            )
            # s_o: [BT, BV] swizzled — at _sh_offset for TMA store
            s_o = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr + Int32(_sh_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.q_dtype,
                ),
                cute.make_layout((self.BT, self.BV), stride=(self.BV, 1)),
            )

            # === LdMatrix copy atoms for B operands ===
            smem_copy_atom_B = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                self.q_dtype,
            )
            smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
            smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

            # CopyUniversal for C → smem (scores → s_a, output → s_o)
            smem_copy_atom_C = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.q_dtype)
            smem_tiled_copy_C = cute.make_tiled_copy_C(smem_copy_atom_C, tiled_mma)
            smem_thr_copy_C = smem_tiled_copy_C.get_slice(tidx)

            # === MMA partitions ===
            # h as B operand (transposed: [BV, BK] stride (1, BV))
            s_h_B = cute.make_tensor(s_h.iterator,
                cute.make_layout((self.BV, self.BK), stride=(1, self.BV)))
            _tBsH = thr_mma.partition_B(s_h_B)
            tCrH = tiled_mma.make_fragment_B(_tBsH)
            tHrH_view = smem_thr_copy_B.retile(tCrH)
            tHsH = smem_thr_copy_B.partition_S(s_h_B)

            # K^T as B operand (transposed: [BK, BT] stride (1, BK))
            s_kt_B = cute.make_tensor(s_k.iterator,
                cute.make_layout((self.BK, self.BT), stride=(1, self.BK)))
            _tBsKt = thr_mma.partition_B(s_kt_B)
            tCrKt = tiled_mma.make_fragment_B(_tBsKt)
            tKrKt_view = smem_thr_copy_B.retile(tCrKt)
            tKsKt = smem_thr_copy_B.partition_S(s_kt_B)

            # Scores as A operand for GEMM3
            tCsA = thr_mma.partition_A(s_a)
            tCrA = tiled_mma.make_fragment_A(tCsA)

            # v_new as B operand (transposed: [BV, BT] stride (1, BV))
            s_v_B = cute.make_tensor(s_v.iterator,
                cute.make_layout((self.BV, self.BT), stride=(1, self.BV)))
            _tBsVt = thr_mma.partition_B(s_v_B)
            tCrVt = tiled_mma.make_fragment_B(_tBsVt)
            tVrVt_view = smem_thr_copy_B.retile(tCrVt)
            tVsVt = smem_thr_copy_B.partition_S(s_v_B)

            # === cpasync setup for BK-wide loads (K) ===
            qk_async_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), self.q_dtype, num_bits_per_copy=128)
            qk_tiled_copy = cute.make_tiled_copy_tv(
                qk_async_atom,
                cute.make_layout((self.qk_copy_dim0, self.qk_copy_dim1),
                                 stride=(self.qk_copy_dim1, 1)),
                cute.make_layout((1, self.async_copy_elems)),
            )
            qk_thr_copy = qk_tiled_copy.get_slice(tidx)

            # cpasync setup for BV-wide loads (h, v_new)
            hv_async_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), self.q_dtype, num_bits_per_copy=128)
            hv_tiled_copy = cute.make_tiled_copy_tv(
                hv_async_atom,
                cute.make_layout((self.hv_copy_dim0, self.hv_copy_dim1),
                                 stride=(self.hv_copy_dim1, 1)),
                cute.make_layout((1, self.async_copy_elems)),
            )
            hv_thr_copy = hv_tiled_copy.get_slice(tidx)

            # cpasync smem destinations
            tK_s = qk_thr_copy.partition_D(s_k)
            tH_s = hv_thr_copy.partition_D(s_h)
            tV_s = hv_thr_copy.partition_D(s_v)

            # === Global memory sources — [B, T, H, K/V] with H-strided access ===
            # Q is already in registers (TMA-loaded in Phase 0)
            qk_base = (tile_B * Int32(self.T * self.H) + tile_H) * Int32(self.K)
            gK_head = cute.make_tensor(
                (k.iterator + qk_base).align(16),
                cute.make_layout((self.T, self.K), stride=(self.H * self.K, 1)))

            vn_base = (tile_B * Int32(self.T * self.H) + tile_H) * Int32(self.V)
            gVN_head = cute.make_tensor(
                (v_new.iterator + vn_base).align(16),
                cute.make_layout((self.T, self.V), stride=(self.H * self.V, 1)))

            # h: [B, NT, H, K, V] — per-head base, stride H*K*V between chunks
            h_head_offset = (
                tile_B * Int32(self.NT_val * self.H) + tile_H
            ) * Int32(self.K * self.V)
            h_head_ptr = (h.iterator + h_head_offset).align(16)

            # g_cumsum: [B, T, H] with stride (T*H, H, 1)
            g_head_base = tile_B * Int32(self.T * self.H) + tile_H
            gG_head = cute.make_tensor(
                g_cumsum.iterator + g_head_base,
                cute.make_layout((self.T,), stride=(self.H,)))

            # Row tile = chunk index within this head
            chunk_idx = tile_T

            # === Identity tensors for coordinate extraction ===
            mc_ov = cute.make_identity_tensor((self.BT, self.BV))
            tCcOV = thr_mma.partition_C(mc_ov)
            mc_AA = cute.make_identity_tensor((self.BT, self.BT))
            tCcAA = thr_mma.partition_C(mc_AA)

            # === Accumulators ===
            acc_o = cute.make_fragment(
                tiled_mma.partition_shape_C((self.BT, self.BV)), Float32)
            acc_o.fill(0.0)
            acc_A = cute.make_fragment(
                tiled_mma.partition_shape_C((self.BT, self.BT)), Float32)
            acc_A.fill(0.0)

            # =============================================================
            # Phase 1: Load g_cumsum
            # =============================================================
            gG_tile = cute.local_tile(gG_head, (self.BT,),
                                      (chunk_idx,))
            if tidx < Int32(self.BT):
                g_buf[tidx] = gG_tile[tidx]
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            # =============================================================
            # Phase 2: K-loop (GEMM1 + GEMM2)
            # =============================================================
            for ki in cutlass.range_constexpr(self.NK):
                # cpasync K[BT, BK]
                gK_tile = cute.local_tile(gK_head, (self.BT, self.BK),
                                          (chunk_idx, Int32(ki)))
                tK_g = qk_thr_copy.partition_S(gK_tile)
                for ci in cutlass.range_constexpr(cute.size(tK_s.shape[2])):
                    cute.copy(qk_tiled_copy, tK_g[None, None, ci], tK_s[None, None, ci])

                # cpasync h[BK, BV] — h is [B, NT, H, K, V], per-head ptr with H*K*V stride
                h_chunk_ptr = h_head_ptr + chunk_idx * Int32(self.H * self.K * self.V)
                gH_chunk = cute.make_tensor(
                    (h_chunk_ptr + Int32(ki * self.BK * self.V)).align(16),
                    cute.make_layout((self.BK, self.V), stride=(self.V, 1)))
                gH_tile = cute.local_tile(gH_chunk, (self.BK, self.BV), (Int32(0), tile_V))
                tH_g = hv_thr_copy.partition_S(gH_tile)
                for ci in cutlass.range_constexpr(cute.size(tH_s.shape[2])):
                    cute.copy(hv_tiled_copy, tH_g[None, None, ci], tH_s[None, None, ci])

                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # Fused GEMM1+GEMM2: Q from registers (Phase 0), K and h from smem
                _BK16 = self.BK // 16
                for kb in cutlass.range_constexpr(self.BK // 16):
                    q_kb = Int32(ki * _BK16 + kb)
                    # GEMM1: acc_o += Q @ h
                    cute.copy(smem_tiled_copy_B, tHsH[None, None, kb],
                              tHrH_view[None, None, kb])
                    cute.gemm(tiled_mma, acc_o,
                              tCrQ_full[None, None, q_kb],
                              tCrH[None, None, kb], acc_o)
                    # GEMM2: acc_A += Q @ K^T
                    cute.copy(smem_tiled_copy_B, tKsKt[None, None, kb],
                              tKrKt_view[None, None, kb])
                    cute.gemm(tiled_mma, acc_A,
                              tCrQ_full[None, None, q_kb],
                              tCrKt[None, None, kb], acc_A)

                # Barrier to protect smem before next ki overwrites it
                # (skip at last K-block — no more writes coming)
                if ki < self.NK - 1:
                    named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            # =============================================================
            # Phase 3: Gating + causal mask
            # =============================================================
            # acc_o[t, v] *= exp(g[t])
            for ci in cutlass.range_constexpr(cute.size(acc_o)):
                row = tCcOV[ci][0]
                acc_o[ci] = acc_o[ci] * cute.math.exp(g_buf[Int32(row)], fastmath=True)

            # acc_A[i, j] *= exp(g[i] - g[j]), causal mask where col > row
            for ci in cutlass.range_constexpr(cute.size(acc_A)):
                row = tCcAA[ci][0]
                col = tCcAA[ci][1]
                g_row = g_buf[Int32(row)]
                g_col = g_buf[Int32(col)]
                gate = cute.math.exp(g_row - g_col, fastmath=True)
                if Int32(col) > Int32(row):
                    acc_A[ci] = Float32(0.0)
                else:
                    acc_A[ci] = acc_A[ci] * gate

            # =============================================================
            # Phase 4: Write scores to smem + load v_new for GEMM3
            # =============================================================
            # Need barrier before reusing s_q/s_k region for s_a/s_v
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            # Issue v_new cpasync FIRST (async, writes to s_v at s_k offset).
            # Overlaps with the synchronous scores→s_a write below since
            # s_v (at _sk_offset) and s_a (at page_ptr) are disjoint regions.
            gVN_tile = cute.local_tile(gVN_head, (self.BT, self.BV),
                                        (chunk_idx, tile_V))
            tVN_g = hv_thr_copy.partition_S(gVN_tile)
            for ci in cutlass.range_constexpr(cute.size(tV_s.shape[2])):
                cute.copy(hv_tiled_copy, tVN_g[None, None, ci], tV_s[None, None, ci])
            cute.arch.cp_async_commit_group()

            # acc_A (fp32) → fp16 → s_a via tiled_copy_C (synchronous, to s_q region)
            a_tmp = cute.make_fragment_like(acc_A, self.q_dtype)
            for ci in cutlass.range_constexpr(cute.size(acc_A)):
                a_tmp[ci] = acc_A[ci].to(self.q_dtype)
            tOrA = smem_thr_copy_C.retile(a_tmp)
            tOsA = smem_thr_copy_C.partition_D(s_a)
            cute.copy(smem_tiled_copy_C, tOrA, tOsA)

            # Wait for v_new (scores→s_a is already done since CopyUniversal is sync)
            cute.arch.cp_async_wait_group(0)
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            # =============================================================
            # Phase 5: GEMM3 — intra-chunk: scores @ v_new
            # =============================================================
            acc_intra = cute.make_fragment(
                tiled_mma.partition_shape_C((self.BT, self.BV)), Float32)
            acc_intra.fill(0.0)

            for kb in cutlass.range_constexpr(self.BT // 16):
                cute.copy(smem_tiled_copy_B, tVsVt[None, None, kb], tVrVt_view[None, None, kb])
                cute.autovec_copy(tCsA[None, None, kb], tCrA[None, None, kb])
                cute.gemm(tiled_mma, acc_intra, tCrA[None, None, kb], tCrVt[None, None, kb], acc_intra)

            # =============================================================
            # Phase 6: Scale and write O to smem for TMA store
            # =============================================================
            # o = scale * (acc_o + acc_intra)
            # Convert to output dtype and write to swizzled smem (s_o)
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            acc_final = cute.make_fragment_like(acc_o, self.q_dtype)
            for ci in cutlass.range_constexpr(cute.size(acc_o)):
                o_val = (acc_o[ci] + acc_intra[ci]) * Float32(self.scale_val)
                acc_final[ci] = o_val.to(self.q_dtype)

            tOrO = smem_thr_copy_C.retile(acc_final)
            tOsO = smem_thr_copy_C.partition_D(s_o)
            cute.copy(smem_tiled_copy_C, tOrO, tOsO)

    # =========================================================================
    # Forward Store (TMA S->G for O)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_B, tile_H, tile_T, tile_V, o_tma, o_tma_gmem):
        """TMA store of O from shared to global memory (swizzled).

        o is [B, T, H, V] → reversed for TMA: (V, H, T, B).
        """
        # s_o is at _sh_offset (after s_k at 0 and s_h at _sk_offset)
        _sk_offset = self.BT * self.BK * self.elem_bytes
        _sk_offset = ((_sk_offset + 127) // 128) * 128
        _sh_offset = _sk_offset + self.BK * self.BV * self.elem_bytes
        _sh_offset = ((_sh_offset + 127) // 128) * 128

        _o_swz = cute.make_swizzle(self.swizzle_B, 4, 3)
        sO = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.q_dtype, page_ptr + Int32(_sh_offset),
                              cute.AddressSpace.smem),
                _o_swz, dtype=self.q_dtype,
            ),
            cute.make_layout((self.BV, 1, self.BT, 1)),
        )
        gO = cute.local_tile(
            o_tma_gmem,
            (self.BV, 1, self.BT, 1),
            (None, None, None, None),
        )
        tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
            o_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sO, 0, 4),
            cute.group_modes(gO, 0, 4),
        )
        with cute.arch.elect_one():
            cute.copy(o_tma, tOsO, tOgO[(None, tile_V, tile_H, tile_T, tile_B)])


__all__ = ["GDNOutputOp"]
