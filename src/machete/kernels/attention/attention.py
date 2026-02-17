# Copyright (c) 2025, Machete Authors
"""
Flash Attention Op for the Megakernel (fp16/bf16 tensor core MMA).

Computes scaled dot-product attention with online softmax:
    O[BH, M, D] = softmax(Q[BH, M, D] @ K[BH, N, D]^T / sqrt(D)) @ V[BH, N, D]

Pipelined load/compute/store:
    load:    3D TMA G->S for Q tile
    compute: Read Q from smem, iterate KV blocks with CopyG2SOp (per-thread
             cp.async) into swizzled smem, LdMatrix for smem→reg, online
             softmax with P-in-registers, MMA GEMMs.
    store:   3D TMA S->G for O tile

K/V are loaded inside compute via CopyG2SOp with swizzled shared memory
layouts for bank-conflict-free LdMatrix reads.  Single-buffer design: K and V
share the same smem region (sequential use), enabling large n_block sizes
(64 for D=128, 112 for D=64) that maximize MMA utilization.

Supports optional causal masking (lower-left aligned):
    Row i in Q can attend to K/V positions 0..(i + N - M).
    Handles prefill (M=N), decode (M=1), and general (M<N).

Usage:
    from machete.kernels.attention import FlashAttentionOp
    from machete.megakernel import Megakernel, MegakernelConfig

    q = q.view(BH, M, D).contiguous()  # fp16 or bf16
    k = k.view(BH, N, D).contiguous()
    v = v.view(BH, N, D).contiguous()
    o = torch.zeros_like(q)
    ops = FlashAttentionOp.schedule(q=q, k=k, v=v, o=o)
    kernel = Megakernel(ops, config=MegakernelConfig())
    kernel.run()
"""

from math import gcd

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu.cpasync import CopyG2SOp, LoadCacheMode
from cutlass.cute.nvgpu import warp

from machete.megakernel.ops import Op
from machete.megakernel.interpreter import (
    mbarrier_arrive_expect_tx,
    named_barrier_sync,
)
from machete.megakernel.paged_memory import PAGE_SIZE


class FlashAttentionOp(Op):
    """Flash Attention operation for the megakernel framework.

    Tensors:
        q: (BH, M, D) — query  (fp16 or bf16)
        k: (BH, N, D) — key
        v: (BH, N, D) — value
        o: (BH, M, D) — output

    Tiling:
        tile_BH=1 (per head), tile_M from schedule, tile_D=D (full).

    Smem page layout:
        Load/store phases: Q/O at [0, q_tile_bytes) via 3D TMA
        Compute phase: Q pre-loaded to registers, page reused for KV:
          [0, n_block × D × 2) — single swizzled KV region (K then V)
    """

    reads = {
        "q": (None, ("BH", "M", "D")),
        "k": (None, ("BH", "N", "D")),
        "v": (None, ("BH", "N", "D")),
    }
    writes = {"o": (None, ("BH", "M", "D"))}
    tile = ("BH", "M", "D")

    tma_loads = {"q"}
    tma_stores = {"o"}

    def __init__(self, **config):
        super().__init__(**config)
        self.causal = getattr(self, 'causal', 0)

        assert self.q_dtype in (cutlass.Float16, cutlass.BFloat16), (
            f"FlashAttentionOp requires fp16 or bf16, got {self.q_dtype}")
        self.elem_bytes = 2

        self.scale_val = 1.0 / (self.D ** 0.5)
        self.kv_row_bytes = self.D * self.elem_bytes
        self.q_tile_bytes = self.tile_size_M * self.D * self.elem_bytes

        self._init_mma()

    def _init_mma(self):
        """Init tensor core MMA path with dynamic n_block and swizzle.

        Smem layout (compute phase, overlapping Q/O page):
          Single swizzled KV region at offset 0: n_block × D × elem_bytes.
          K and V share this region (loaded sequentially, not concurrently).
          No mbarrier needed — uses cp_async_commit/wait + named_barrier.
        """
        # Multi-warp MMA: tile_M = num_warps * 16
        assert self.tile_size_M % 16 == 0 and self.tile_size_M >= 16, (
            f"FlashAttentionOp: tile_size_M={self.tile_size_M} must be "
            f"a positive multiple of 16.")
        self.num_mma_warps = self.tile_size_M // 16
        max_warps = self.threads_per_row // 32
        assert self.num_mma_warps <= max_warps, (
            f"FlashAttentionOp: tile_size_M={self.tile_size_M} requires "
            f"{self.num_mma_warps} warps but only {max_warps} available.")
        self.num_mma_threads = self.num_mma_warps * 32

        assert self.D >= 16 and self.D % 16 == 0, (
            f"FlashAttentionOp: D={self.D} must be >= 16 and ×16.")

        assert self.q_tile_bytes <= PAGE_SIZE, (
            f"FlashAttentionOp: Q tile ({self.q_tile_bytes}B) > "
            f"PAGE_SIZE ({PAGE_SIZE}B). Reduce tile_size_M.")

        # --- Dynamic n_block computation ---
        # CopyG2SOp: 128-bit (16 bytes) per thread per cp.async
        self.async_copy_elems = 128 // (self.elem_bytes * 8)  # 8 for fp16
        # Swizzle atom covers full D columns — no column tiling needed.
        # Swizzle params: B=min(3, log2(D)-3), M=3, S=log2(D)-3
        # ensures yyy captures row bits, zzz targets col bank bits.
        log2_D = self.D.bit_length() - 1
        assert (1 << log2_D) == self.D, (
            f"D={self.D} must be a power of 2")
        self.swizzle_S = log2_D - 3
        self.swizzle_B = min(3, self.swizzle_S)
        self.swizzle_M = 3
        self.tQKV_dim_1 = self.D // self.async_copy_elems
        self.copy_atom_rows = self.num_mma_threads // self.tQKV_dim_1

        # n_block must be:
        # 1. ×lcm(copy_atom_rows, 16) for CopyG2SOp + MMA tiling
        # 2. ×32 for rP_mma_view (logical_divide by 2 needs n_block/16 even)
        n_block_base = (self.copy_atom_rows * 16) // gcd(
            self.copy_atom_rows, 16)
        n_block_align = (n_block_base * 32) // gcd(n_block_base, 32)
        max_n_block = PAGE_SIZE // (self.D * self.elem_bytes)
        self.n_block = (max_n_block // n_block_align) * n_block_align
        self.n_block = max(self.n_block, 32)
        assert self.n_block % 32 == 0, (
            f"n_block={self.n_block} must be ×32 for rP_mma_view. "
            f"Use schedule_forward() to auto-select valid tile_M.")
        self.num_kv_blocks = (self.N + self.n_block - 1) // self.n_block

        self.kv_smem_bytes = self.n_block * self.D * self.elem_bytes
        assert self.kv_smem_bytes <= PAGE_SIZE, (
            f"FlashAttentionOp: KV smem ({self.kv_smem_bytes}B) > "
            f"PAGE_SIZE ({PAGE_SIZE}B).")

        # exp2-based softmax
        self.scale_log2e = self.scale_val * 1.4426950408889634074

        # Override compute method
        self.compute = self.compute_mma

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule_forward(cls, tile_sizes=None, causal=False, **tensors):
        """Schedule flash attention forward, optionally with causal masking.

        Picks the largest tile_M (×16) ≤ M that gives a valid n_block:
        n_block must be ×32 (for rP_mma_view, which needs n_block/16 even)
        and ×lcm(copy_atom_rows, 16) (for CopyG2SOp tiling).
        """
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("BH", 1)
        q = tensors.get('q')
        if q is not None and "M" not in tile_sizes:
            assert q.element_size() == 2, (
                f"FlashAttentionOp requires fp16/bf16, "
                f"got element_size={q.element_size()}")
            D = q.shape[-1]
            M = q.shape[1]
            elem = q.element_size()
            max_tile_M_page = PAGE_SIZE // (D * elem)
            max_n_block = max_tile_M_page
            async_copy_elems = 128 // (elem * 8)
            tQKV_dim_1 = D // async_copy_elems

            # Find largest tile_M giving valid n_block (×32)
            max_nw = min(7, max_tile_M_page // 16, M // 16)
            tile_M = 16
            for nw in range(max_nw, 0, -1):
                num_threads = nw * 32
                copy_rows = num_threads // tQKV_dim_1
                base_align = (copy_rows * 16) // gcd(copy_rows, 16)
                full_align = (base_align * 32) // gcd(base_align, 32)
                nb = (max_n_block // full_align) * full_align
                if nb >= 32:
                    tile_M = nw * 16
                    break
            tile_sizes["M"] = max(16, tile_M)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        if causal:
            ops[0].static_dims['causal'] = 1
        return ops

    # =========================================================================
    # Forward Load (3D TMA G->S for Q)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_BH, tile_M, tile_D, q_tma, q_tma_gmem,
             work_mbar):
        """TMA load of Q tile from global to shared memory.

        3D TMA: Q(BH, M, D) permuted to (D, M, BH) for CuTe mode 0 = D.
        Smem layout: (D, tile_M, 1) col-major = (tile_M, D) row-major.
        """
        sQ = cute.make_tensor(
            cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.D, self.tile_size_M, 1)),
        )
        gQ = cute.local_tile(
            q_tma_gmem, (self.D, self.tile_size_M, 1), (None, None, None),
        )
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            q_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(gQ, 0, 3),
        )

        nbytes = Int32(self.q_tile_bytes)
        mbar_ptr = cute.make_ptr(
            cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(q_tma, tQgQ[(None, tile_D, tile_M, tile_BH)], tQsQ,
                  tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # MMA Helpers
    # =========================================================================

    def _make_acc_tensor_mn_view(self, acc):
        """Reshape MMA accumulator to (M, N) view for per-row softmax."""
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        s = acc_layout_col_major.shape
        st = acc_layout_col_major.stride
        acc_layout_mn = cute.make_layout(
            ((s[0][1], s[1]), (s[0][0], s[2])),
            stride=((st[0][1], st[1]), (st[0][0], st[2])),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    def _threadquad_reduce(self, val, op):
        """Reduce a scalar across 4 threads in an MMA thread quad."""
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1,
                                        mask_and_clamp=31),
        )
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1,
                                        mask_and_clamp=31),
        )
        return val

    def _threadquad_reduce_max(self, val):
        return self._threadquad_reduce(
            val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val):
        return self._threadquad_reduce(val, lambda x, y: x + y)

    # =========================================================================
    # Forward Compute — Tensor Core MMA (fp16/bf16)
    # =========================================================================

    @cute.jit
    def compute_mma(self, page_ptr, tile_BH, tile_M, tile_D, q, k, v, o):
        """Flash attention forward with multi-warp tensor core MMA.

        Uses CopyG2SOp + swizzled smem for bank-conflict-free KV loads.
        Single-buffer: K and V share the same smem region (loaded
        sequentially). Pipeline: load K → S GEMM → load V (async) →
        softmax (overlaps with V load) → O GEMM.
        """
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        if warp_idx < Int32(self.num_mma_warps):
            # === MMA setup (multi-warp) ===
            mma_op = warp.MmaF16BF16Op(
                self.q_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((self.num_mma_warps, 1, 1)),
                permutation_mnk=(self.num_mma_warps * 16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)

            # === Preload Q to registers (before KV overwrites smem) ===
            sQ = cute.make_tensor(
                cute.make_ptr(self.q_dtype, page_ptr,
                              cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.tile_size_M, self.D),
                                 stride=(self.D, 1)),
            )
            tCsQ = thr_mma.partition_A(sQ)
            tCrQ = tiled_mma.make_fragment_A(tCsQ)
            for kb in cutlass.range_constexpr(self.D // 16):
                cute.autovec_copy(
                    tCsQ[None, None, kb], tCrQ[None, None, kb])
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            # === KV smem (swizzled for bank-conflict-free LdMatrix) ===
            kv_smem_ptr = cute.make_ptr(
                self.q_dtype, page_ptr,
                cute.AddressSpace.smem, assumed_align=128)
            swz = cute.make_swizzle(
                self.swizzle_B, self.swizzle_M, self.swizzle_S)
            # Swizzled (n_block, D) for K: atom (8, D) row-major
            sKV_atom = cute.make_composed_layout(
                swz, 0,
                cute.make_layout(
                    (8, self.D), stride=(self.D, 1)))
            sKV = cute.make_tensor(
                kv_smem_ptr,
                cute.tile_to_shape(
                    sKV_atom,
                    (self.n_block, self.D), (0, 1)))
            # Transposed swizzled (D, n_block) for V^T reads
            sKVt_atom = cute.make_composed_layout(
                swz, 0,
                cute.make_layout(
                    (self.D, 8), stride=(1, self.D)))
            sKVt = cute.make_tensor(
                kv_smem_ptr,
                cute.tile_to_shape(
                    sKVt_atom,
                    (self.D, self.n_block), (0, 1)))

            # === CopyG2SOp tiled copy (all MMA threads) ===
            g2s_atom = cute.make_copy_atom(
                CopyG2SOp(cache_mode=LoadCacheMode.GLOBAL),
                self.q_dtype, num_bits_per_copy=128)
            thr_layout = cute.make_layout(
                (self.copy_atom_rows, self.tQKV_dim_1),
                stride=(self.tQKV_dim_1, 1))
            val_layout = cute.make_layout(
                (1, self.async_copy_elems))
            gmem_tiled_copy = cute.make_tiled_copy_tv(
                g2s_atom, thr_layout, val_layout)
            gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)

            # Smem destination partition (same every block)
            tKVsKV = gmem_thr_copy.partition_D(sKV)

            # Identity tensor for N-boundary predicates
            mcKV = cute.make_identity_tensor(
                (self.n_block, self.D))
            tKVcKV = gmem_thr_copy.partition_S(mcKV)

            # === LdMatrix for smem→reg reads (K and Vt) ===
            smem_copy_atom_K = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(
                    transpose=False, num_matrices=4),
                self.q_dtype)
            smem_copy_atom_Vt = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(
                    transpose=True, num_matrices=4),
                self.q_dtype)
            smem_tiled_copy_K = cute.make_tiled_copy_B(
                smem_copy_atom_K, tiled_mma)
            smem_tiled_copy_Vt = cute.make_tiled_copy_B(
                smem_copy_atom_Vt, tiled_mma)
            smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
            smem_thr_copy_Vt = smem_tiled_copy_Vt.get_slice(tidx)

            # === MMA register fragments ===
            tCsK = thr_mma.partition_B(sKV)
            tCrK = tiled_mma.make_fragment_B(tCsK)
            tBsVt = thr_mma.partition_B(sKVt)
            tBrVt = tiled_mma.make_fragment_B(tBsVt)

            # LdMatrix partitions
            tKsK = smem_thr_copy_K.partition_S(sKV)
            tKrK_copy_view = smem_thr_copy_K.retile(tCrK)
            tVsVt = smem_thr_copy_Vt.partition_S(sKVt)
            tVrVt_copy_view = smem_thr_copy_Vt.retile(tBrVt)

            # === Accumulators ===
            acc_O = cute.make_fragment(
                tiled_mma.partition_shape_C(
                    (self.tile_size_M, self.D)),
                Float32)
            acc_O.fill(0.0)
            acc_S = cute.make_fragment(
                tiled_mma.partition_shape_C(
                    (self.tile_size_M, self.n_block)),
                Float32)

            # Softmax state
            acc_O_shape = tiled_mma.partition_shape_C(
                (self.tile_size_M, self.D))
            num_rows = acc_O_shape[0][1] * acc_O_shape[1]
            row_max = cute.make_fragment(
                cute.make_layout(num_rows), Float32)
            row_sum = cute.make_fragment(
                cute.make_layout(num_rows), Float32)
            for r in cutlass.range_constexpr(num_rows):
                row_max[r] = Float32(-1e30)
                row_sum[r] = Float32(0.0)

            # Identity tensor for N-masking in softmax
            mcS = cute.make_identity_tensor(
                (self.tile_size_M, self.n_block))
            tScS = thr_mma.partition_C(mcS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)

            # === Global memory setup ===
            head_offset = tile_BH * Int32(self.N * self.D)

            # Number of outer copy tiles (for boundary loop)
            num_copy_outer = cute.size(tKVsKV.shape[1])

            # === KV block loop (single-buffer pipeline) ===
            # Pipeline per block:
            #   1. Load K → commit → wait → barrier
            #   2. S GEMM: Q(regs) @ K(smem)^T
            #   3. Load V → commit (don't wait, V loads in background)
            #   4. Softmax (pure register work, overlaps V load)
            #   5. Wait V → barrier
            #   6. O GEMM: P(regs) @ V(smem)^T
            for kv_blk in range(self.num_kv_blocks):
                kv_start = kv_blk * Int32(self.n_block)

                # --- 1. Load K[i] → swizzled smem ---
                gK = cute.make_tensor(
                    cute.make_ptr(
                        self.q_dtype,
                        (k.iterator + head_offset
                         + kv_start * Int32(self.D)).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16),
                    cute.make_layout(
                        (self.n_block, self.D),
                        stride=(self.D, 1)))
                tKVgK = gmem_thr_copy.partition_S(gK)

                for nt in cutlass.range_constexpr(num_copy_outer):
                    row_coord = tKVcKV[0, nt, 0][0]
                    if kv_start + row_coord < Int32(self.N):
                        cute.copy(gmem_tiled_copy,
                                  tKVgK[None, nt, None],
                                  tKVsKV[None, nt, None])
                    else:
                        tKVsKV[None, nt, None].fill(0.0)

                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(
                    Int32(2), Int32(self.num_mma_threads))

                # --- 2. S GEMM: acc_S = Q(regs) @ K(smem)^T ---
                acc_S.fill(0.0)
                for kb in cutlass.range_constexpr(self.D // 16):
                    cute.copy(smem_tiled_copy_K,
                              tKsK[None, None, kb],
                              tKrK_copy_view[None, None, kb])
                    cute.gemm(tiled_mma, acc_S,
                              tCrQ[None, None, kb],
                              tCrK[None, None, kb], acc_S)

                # Barrier: all warps must finish reading K before
                # any warp starts V cp.async (overwrites same smem).
                named_barrier_sync(
                    Int32(2), Int32(self.num_mma_threads))

                # --- 3. Load V[i] → same smem (async) ---
                gV = cute.make_tensor(
                    cute.make_ptr(
                        self.q_dtype,
                        (v.iterator + head_offset
                         + kv_start * Int32(self.D)).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16),
                    cute.make_layout(
                        (self.n_block, self.D),
                        stride=(self.D, 1)))
                tKVgV = gmem_thr_copy.partition_S(gV)

                for nt in cutlass.range_constexpr(num_copy_outer):
                    row_coord = tKVcKV[0, nt, 0][0]
                    if kv_start + row_coord < Int32(self.N):
                        cute.copy(gmem_tiled_copy,
                                  tKVgV[None, nt, None],
                                  tKVsKV[None, nt, None])
                    else:
                        tKVsKV[None, nt, None].fill(0.0)

                cute.arch.cp_async_commit_group()
                # Don't wait yet — softmax overlaps with V load

                # --- 4. Mask + online softmax (registers only) ---
                acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
                acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

                for r in cutlass.range_constexpr(num_rows):
                    row_idx = tScS_mn[r, 0][0]
                    for c in cutlass.range_constexpr(
                            cute.size(tScS_mn.shape[1])):
                        col_idx = tScS_mn[0, c][1]
                        global_col = kv_start + Int32(col_idx)
                        # N-boundary mask
                        if global_col >= Int32(self.N):
                            acc_S_mn[r, c] = Float32(-1e30)
                        # Causal mask: row i attends to
                        # cols 0..(i + N - M)
                        if self.causal:
                            global_row = (
                                tile_M * Int32(self.tile_size_M)
                                + Int32(row_idx))
                            if global_col > global_row + Int32(
                                    self.N - self.M):
                                acc_S_mn[r, c] = Float32(
                                    -1e30)

                for r in cutlass.range_constexpr(num_rows):
                    acc_S_row = acc_S_mn[r, None].load()

                    row_max_cur = acc_S_row.reduce(
                        cute.ReductionOp.MAX, Float32(-1e30), 0)
                    row_max_cur = self._threadquad_reduce_max(
                        row_max_cur)

                    m_old = row_max[r]
                    m_new = cute.arch.fmax(m_old, row_max_cur)
                    correction = cute.math.exp2(
                        (m_old - m_new) * Float32(
                            self.scale_log2e),
                        fastmath=True,
                    )

                    row_sum[r] = row_sum[r] * correction
                    acc_O_mn[r, None] = (
                        acc_O_mn[r, None].load() * correction)

                    acc_S_row_exp = cute.math.exp2(
                        acc_S_row * Float32(self.scale_log2e)
                        - m_new * Float32(self.scale_log2e),
                        fastmath=True,
                    )

                    acc_S_row_sum = acc_S_row_exp.reduce(
                        cute.ReductionOp.ADD, Float32(0.0), 0)
                    row_sum[r] = row_sum[r] + acc_S_row_sum

                    row_max[r] = m_new
                    acc_S_mn[r, None] = acc_S_row_exp

                # --- 5. P in registers (rP_mma_view trick) ---
                rP = cute.make_fragment_like(acc_S, self.q_dtype)
                rP.store(acc_S.load().to(self.q_dtype))
                rP_ld = cute.logical_divide(
                    rP.layout, (None, None, 2))
                rP_mma_view = cute.make_layout(
                    ((rP_ld.shape[0], rP_ld.shape[2][0]),
                     rP_ld.shape[1],
                     rP_ld.shape[2][1]),
                    stride=(
                        (rP_ld.stride[0], rP_ld.stride[2][0]),
                        rP_ld.stride[1],
                        rP_ld.stride[2][1]),
                )
                tOrS = cute.make_tensor(
                    rP.iterator, rP_mma_view)

                # --- 6. Wait for V ---
                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(
                    Int32(2), Int32(self.num_mma_threads))

                # --- 7. O GEMM: acc_O += P(regs) @ V(smem)^T ---
                for kb in cutlass.range_constexpr(
                        self.n_block // 16):
                    cute.copy(smem_tiled_copy_Vt,
                              tVsVt[None, None, kb],
                              tVrVt_copy_view[None, None, kb])
                    cute.gemm(tiled_mma, acc_O,
                              tOrS[None, None, kb],
                              tBrVt[None, None, kb], acc_O)

            # === Normalize O by row_sum ===
            acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
            for r in cutlass.range_constexpr(num_rows):
                row_sum[r] = self._threadquad_reduce_sum(
                    row_sum[r])
                inv_sum = cute.arch.rcp_approx(row_sum[r])
                acc_O_mn[r, None] = (
                    acc_O_mn[r, None].load() * inv_sum)

            # === Write O to smem (for TMA store) ===
            named_barrier_sync(
                Int32(2), Int32(self.num_mma_threads))
            sO = cute.make_tensor(
                cute.make_ptr(self.q_dtype, page_ptr,
                              cute.AddressSpace.smem),
                cute.make_layout((self.tile_size_M, self.D),
                                 stride=(self.D, 1)),
            )
            tCsO = thr_mma.partition_C(sO)
            for i in cutlass.range_constexpr(cute.size(acc_O)):
                tCsO[i] = acc_O[i].to(self.q_dtype)

    # =========================================================================
    # Forward Store (3D TMA S->G for O)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_BH, tile_M, tile_D, o_tma, o_tma_gmem):
        """TMA store of O from shared to global memory."""
        sO = cute.make_tensor(
            cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.D, self.tile_size_M, 1)),
        )
        gO = cute.local_tile(
            o_tma_gmem, (self.D, self.tile_size_M, 1), (None, None, None),
        )
        tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
            o_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sO, 0, 3),
            cute.group_modes(gO, 0, 3),
        )
        with cute.arch.elect_one():
            cute.copy(o_tma, tOsO, tOgO[(None, tile_D, tile_M, tile_BH)])


__all__ = ["FlashAttentionOp"]
