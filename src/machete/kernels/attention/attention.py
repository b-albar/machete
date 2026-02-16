# Copyright (c) 2025, Machete Authors
"""
Flash Attention Op for the Megakernel.

Computes scaled dot-product attention with online softmax:
    O[BH, M, D] = softmax(Q[BH, M, D] @ K[BH, N, D]^T / sqrt(D)) @ V[BH, N, D]

Pipelined load/compute/store:
    load:    3D TMA G->S for Q tile
    compute: Read Q from smem, iterate KV positions with async bulk copy
             (CopyBulkG2SOp + mbarrier), online softmax, write O to smem
    store:   3D TMA S->G for O tile

K/V are loaded inside compute via CopyBulkG2SOp (cp.async.bulk — same TMA
hardware) with a compute-local mbarrier for synchronization.

Supports optional causal masking (lower-left aligned):
    Row i in Q can attend to K/V positions 0..(i + N - M).
    Handles prefill (M=N), decode (M=1), and general (M<N).

Usage:
    from machete.kernels.attention import FlashAttentionOp
    from machete.megakernel import Megakernel, MegakernelConfig

    q = q.view(BH, M, D).contiguous()
    k = k.view(BH, N, D).contiguous()
    v = v.view(BH, N, D).contiguous()
    o = torch.zeros_like(q)
    ops = FlashAttentionOp.schedule(q=q, k=k, v=v, o=o, tile_sizes={"M": 4})
    kernel = Megakernel(ops, config=MegakernelConfig())
    kernel.run()
"""

import operator

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkG2SOp,
    group_bulk_copy_modes,
)
from cutlass.cute.nvgpu import warp

from machete.megakernel.ops import Op
from machete.megakernel.interpreter import (
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_arrive_expect_tx,
    mbarrier_wait,
    named_barrier_sync,
)
from machete.megakernel.paged_memory import PAGE_SIZE


def _align_up(v, a):
    return (v + a - 1) // a * a


class FlashAttentionOp(Op):
    """Flash Attention operation for the megakernel framework.

    Tensors:
        q: (BH, M, D) — query
        k: (BH, N, D) — key
        v: (BH, N, D) — value
        o: (BH, M, D) — output

    Tiling:
        tile_BH=1 (per head), tile_M from schedule, tile_D=D (full).

    Smem page layout (double-buffered K/V for 2-stage pipelining):
        [Q/O region:  tile_M × D × elem_bytes]
        [mbarrier:    8 bytes, 8-byte aligned]
        [K0 row:      D × elem_bytes, 16-aligned]
        [V0 row:      D × elem_bytes, 16-aligned]
        [K1 row:      D × elem_bytes, 16-aligned]
        [V1 row:      D × elem_bytes, 16-aligned]
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

        if self.q_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
            self.use_mma = True
        elif self.q_dtype == cutlass.Float32:
            self.elem_bytes = 4
            self.use_mma = False
        else:
            self.elem_bytes = 4
            self.use_mma = False

        # Scale factor
        self.scale_val = 1.0 / (self.D ** 0.5)

        # CopyBulkG2SOp config for K/V rows
        self.kv_row_bytes = self.D * self.elem_bytes
        self.kv_row_nbits = self.kv_row_bytes * 8

        # Q/O tile bytes (shared between both paths)
        self.q_tile_bytes = self.tile_size_M * self.D * self.elem_bytes
        self.mbar_offset = _align_up(self.q_tile_bytes, 8)

        if self.use_mma:
            self._init_mma()
        else:
            self._init_scalar()

    def _init_mma(self):
        """Init for tensor core MMA path (fp16/bf16).

        Multi-warp MMA with overlapping smem layout:
          Load/store phases: Q/O at [0, q_tile_bytes)
          Compute phase: Q pre-loaded to registers, smem reused for KV+P:
            [0,8) = mbar, [16..] = K_block, V_block, P_region
        """
        self.n_block = 16  # KV rows per block (matches MMA K-atom)
        self.num_kv_blocks = (self.N + self.n_block - 1) // self.n_block

        # Multi-warp MMA: tile_M = num_warps * 16 (MMA atom M-size)
        assert self.tile_size_M % 16 == 0 and self.tile_size_M >= 16, (
            f"FlashAttentionOp MMA: tile_size_M={self.tile_size_M} must be "
            f"a positive multiple of 16."
        )
        self.num_mma_warps = self.tile_size_M // 16
        max_warps = self.threads_per_row // 32
        assert self.num_mma_warps <= max_warps, (
            f"FlashAttentionOp MMA: tile_size_M={self.tile_size_M} requires "
            f"{self.num_mma_warps} warps but only {max_warps} available "
            f"(threads_per_row={self.threads_per_row})."
        )
        self.num_mma_threads = self.num_mma_warps * 32

        assert self.D >= 16 and self.D % 16 == 0, (
            f"FlashAttentionOp MMA: D={self.D} must be >= 16 and a multiple of 16."
        )

        # Q/O must fit in page (load/store phases)
        assert self.q_tile_bytes <= PAGE_SIZE, (
            f"FlashAttentionOp MMA: Q tile ({self.q_tile_bytes}B) exceeds "
            f"PAGE_SIZE ({PAGE_SIZE}B). Reduce tile_size_M."
        )

        # Compute phase KV+P layout (overlapping Q/O region)
        kv_block_bytes = self.n_block * self.kv_row_bytes
        p_bytes = self.tile_size_M * self.n_block * self.elem_bytes
        self.kv_mbar_offset = 0  # compute-local mbar at page start
        self.k_block_offset = 16  # 16-byte aligned after mbar
        self.v_block_offset = _align_up(16 + kv_block_bytes, 16)
        self.p_offset = _align_up(self.v_block_offset + kv_block_bytes, 16)
        kv_total = self.p_offset + p_bytes
        assert kv_total <= PAGE_SIZE, (
            f"FlashAttentionOp MMA: KV+P smem ({kv_total}B) exceeds "
            f"PAGE_SIZE ({PAGE_SIZE}B)."
        )

        # exp2-based softmax: scale_log2e = scale * log2(e)
        self.scale_log2e = self.scale_val * 1.4426950408889634074

        # Override compute to MMA version
        self.compute = self.compute_mma

    def _init_scalar(self):
        """Init for scalar warp-parallel path (fp32)."""
        # Double-buffered single-row K/V for 2-stage pipelining
        # cp.async.bulk requires 16-byte aligned shared memory addresses
        k0 = _align_up(self.mbar_offset + 8, 16)
        v0 = _align_up(k0 + self.kv_row_bytes, 16)
        k1 = _align_up(v0 + self.kv_row_bytes, 16)
        v1 = _align_up(k1 + self.kv_row_bytes, 16)
        self.k_smem_offsets = [k0, k1]
        self.v_smem_offsets = [v0, v1]
        total_smem = v1 + self.kv_row_bytes

        assert self.D >= 32, f"FlashAttentionOp requires D >= 32, got D={self.D}"
        assert total_smem <= PAGE_SIZE, (
            f"FlashAttentionOp: smem ({total_smem}B) exceeds PAGE_SIZE ({PAGE_SIZE}B). "
            f"Reduce tile_size_M={self.tile_size_M}."
        )

        self.num_warps = self.threads_per_row // 32
        self.elems_per_lane = self.D // 32
        self.rows_per_warp = (self.tile_size_M + self.num_warps - 1) // self.num_warps

        # Strides between buf[0] and buf[1] for arithmetic buffer selection
        self.k_buf_base = self.k_smem_offsets[0]
        self.k_buf_stride = self.k_smem_offsets[1] - self.k_smem_offsets[0]
        self.v_buf_base = self.v_smem_offsets[0]
        self.v_buf_stride = self.v_smem_offsets[1] - self.v_smem_offsets[0]

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule_forward(cls, tile_sizes=None, causal=False, **tensors):
        """Schedule flash attention forward, optionally with causal masking."""
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("BH", 1)
        q = tensors.get('q')
        if q is not None and q.element_size() == 2:
            # MMA path: maximize tile_M for multi-warp MMA.
            # Constraint: Q/O tile ≤ PAGE_SIZE (overlapping smem layout).
            D = q.shape[-1]
            elem = q.element_size()
            max_tile_M_page = PAGE_SIZE // (D * elem)
            # Cap at default compute warps: (threads_per_block - 32) / 32
            max_warps = 7  # default: (256 - 32) / 32
            max_tile_M = min(max_tile_M_page, max_warps * 16)
            max_tile_M = (max_tile_M // 16) * 16
            tile_sizes.setdefault("M", max(16, max_tile_M))
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
            cutlass.Int64, work_mbar, cute.AddressSpace.smem
        )
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(q_tma, tQgQ[(None, tile_D, tile_M, tile_BH)], tQsQ,
                  tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # MMA Helpers (for tensor core path)
    # =========================================================================

    def _make_acc_tensor_mn_view(self, acc):
        """Reshape MMA accumulator to (M, N) view for per-row softmax.

        MMA C-partition layout is ((atom_v_m, atom_v_n), rest_M, rest_N).
        This reshapes to ((atom_v_m, rest_M), (atom_v_n, rest_N)) = (M, N).
        Adapted from FA2 flash_attention_v2.py:1072-1104.
        """
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
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val):
        return self._threadquad_reduce(val, lambda x, y: x + y)

    # =========================================================================
    # Forward Compute — Tensor Core MMA (fp16/bf16)
    # =========================================================================

    @cute.jit
    def compute_mma(self, page_ptr, tile_BH, tile_M, tile_D, q, k, v, o):
        """Flash attention forward with multi-warp tensor core MMA.

        Uses MmaF16BF16Op(16,8,16) with num_mma_warps warps tiling M.
        Two GEMMs per KV block:
          S[tile_M, n_block] = Q[tile_M, D] @ K[n_block, D]^T
          O[tile_M, D] += softmax(S) @ V[n_block, D]

        Overlapping smem: Q pre-loaded to registers, then page reused
        for KV blocks + P region. Online softmax with exp2.
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

            # === Q smem view (at page start, before overlap) ===
            sQ = cute.make_tensor(
                cute.make_ptr(self.q_dtype, page_ptr,
                              cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.tile_size_M, self.D),
                                 stride=(self.D, 1)),
            )

            # === Preload Q to registers (before KV overwrites smem) ===
            tCsQ = thr_mma.partition_A(sQ)
            tCrQ = tiled_mma.make_fragment_A(tCsQ)
            for kb in cutlass.range_constexpr(self.D // 16):
                cute.autovec_copy(
                    tCsQ[None, None, kb], tCrQ[None, None, kb])

            # Sync: all MMA threads done reading Q from smem
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            # === Smem now reused for KV + P (overlapping Q region) ===
            # K block: (n_block, D) row-major
            sK = cute.make_tensor(
                cute.make_ptr(
                    self.q_dtype,
                    page_ptr + Int32(self.k_block_offset),
                    cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.n_block, self.D),
                                 stride=(self.D, 1)),
            )
            # V block: (n_block, D) row-major — transposed for O GEMM
            sV = cute.make_tensor(
                cute.make_ptr(
                    self.q_dtype,
                    page_ptr + Int32(self.v_block_offset),
                    cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.n_block, self.D),
                                 stride=(self.D, 1)),
            )
            sVt = cute.composition(
                sV,
                cute.make_layout((self.D, self.n_block),
                                 stride=(self.n_block, 1)),
            )
            # P region: (tile_M, n_block) — separate from K/V
            sP = cute.make_tensor(
                cute.make_ptr(
                    self.q_dtype,
                    page_ptr + Int32(self.p_offset),
                    cute.AddressSpace.smem),
                cute.make_layout((self.tile_size_M, self.n_block),
                                 stride=(self.n_block, 1)),
            )

            # === MMA partitions for K, V, P ===
            tCsK = thr_mma.partition_B(sK)
            tCrK = tiled_mma.make_fragment_B(tCsK)
            tAsP = thr_mma.partition_A(sP)
            tArP = tiled_mma.make_fragment_A(tAsP)
            tBsVt = thr_mma.partition_B(sVt)
            tBrVt = tiled_mma.make_fragment_B(tBsVt)

            # === Accumulators ===
            acc_O = cute.make_fragment(
                tiled_mma.partition_shape_C(
                    (self.tile_size_M, self.D)),
                Float32,
            )
            acc_O.fill(0.0)
            acc_S = cute.make_fragment(
                tiled_mma.partition_shape_C(
                    (self.tile_size_M, self.n_block)),
                Float32,
            )

            # Softmax state: 2 rows per thread (atom_v_m * rest_M)
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

            # Identity tensor for coordinate-based masking
            mcS = cute.make_identity_tensor(
                (self.tile_size_M, self.n_block))
            tScS = thr_mma.partition_C(mcS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)

            # === Init compute-local mbarrier (overlapping Q smem) ===
            # Use tidx==0 (not elect_one) to ensure exactly ONE thread
            # across all warps issues mbar ops.
            kv_mbar_addr = page_ptr + Int32(self.kv_mbar_offset)
            mbar_ptr = cute.make_ptr(
                cutlass.Int64, kv_mbar_addr, cute.AddressSpace.smem)
            if tidx == Int32(0):
                mbarrier_init(kv_mbar_addr, Int32(1))
            mbarrier_init_fence()

            # CopyBulkG2SOp for K/V row copies
            g2s = cute.make_copy_atom(
                CopyBulkG2SOp(), self.q_dtype,
                num_bits_per_copy=self.kv_row_nbits,
            )
            head_offset = tile_BH * Int32(self.N * self.D)

            # === KV block loop ===
            for kv_blk in range(self.num_kv_blocks):
                phase = kv_blk % Int32(2)
                kv_start = kv_blk * Int32(self.n_block)

                # -- Load K + V block via CopyBulkG2SOp --
                remaining = Int32(self.N) - kv_start
                valid_rows = remaining
                if remaining > Int32(self.n_block):
                    valid_rows = Int32(self.n_block)
                tx_bytes = valid_rows * Int32(2 * self.kv_row_bytes)

                if tidx == Int32(0):
                    mbarrier_arrive_expect_tx(kv_mbar_addr, tx_bytes)
                    for row in cutlass.range_constexpr(self.n_block):
                        global_kv = kv_start + Int32(row)
                        if global_kv < Int32(self.N):
                            row_off = (
                                head_offset + global_kv * Int32(self.D))
                            # K row
                            g_k = cute.make_tensor(
                                k.iterator + row_off,
                                cute.make_layout((self.D,)),
                            )
                            s_k = cute.make_tensor(
                                cute.make_ptr(
                                    self.q_dtype,
                                    page_ptr + Int32(
                                        self.k_block_offset
                                        + row * self.kv_row_bytes),
                                    cute.AddressSpace.smem),
                                cute.make_layout((self.D,)),
                            )
                            gk, sk = group_bulk_copy_modes(g_k, s_k)
                            cute.copy(g2s, gk, sk, mbar_ptr=mbar_ptr)
                            # V row
                            g_v = cute.make_tensor(
                                v.iterator + row_off,
                                cute.make_layout((self.D,)),
                            )
                            s_v = cute.make_tensor(
                                cute.make_ptr(
                                    self.q_dtype,
                                    page_ptr + Int32(
                                        self.v_block_offset
                                        + row * self.kv_row_bytes),
                                    cute.AddressSpace.smem),
                                cute.make_layout((self.D,)),
                            )
                            gv, sv = group_bulk_copy_modes(g_v, s_v)
                            cute.copy(g2s, gv, sv, mbar_ptr=mbar_ptr)

                # Wait for K+V data
                mbarrier_wait(kv_mbar_addr, phase)

                # -- S GEMM: acc_S = Q(regs) @ K(smem)^T --
                acc_S.fill(0.0)
                for kb in cutlass.range_constexpr(self.D // 16):
                    cute.autovec_copy(
                        tCsK[None, None, kb], tCrK[None, None, kb])
                    cute.gemm(tiled_mma, acc_S,
                              tCrQ[None, None, kb],
                              tCrK[None, None, kb], acc_S)

                # -- Mask invalid S scores (N-boundary) --
                acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
                acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

                for r in cutlass.range_constexpr(num_rows):
                    for c in cutlass.range_constexpr(
                            cute.size(tScS_mn.shape[1])):
                        col_idx = tScS_mn[0, c][1]
                        if kv_start + Int32(col_idx) >= Int32(self.N):
                            acc_S_mn[r, c] = Float32(-1e30)

                # -- Online softmax --
                for r in cutlass.range_constexpr(num_rows):
                    acc_S_row = acc_S_mn[r, None].load()

                    # Row max (local reduce + quad shuffle)
                    row_max_cur = acc_S_row.reduce(
                        cute.ReductionOp.MAX, Float32(-1e30), 0)
                    row_max_cur = self._threadquad_reduce_max(
                        row_max_cur)

                    # Correction factor for running max
                    m_old = row_max[r]
                    m_new = cute.arch.fmax(m_old, row_max_cur)
                    correction = cute.math.exp2(
                        (m_old - m_new) * Float32(self.scale_log2e),
                        fastmath=True,
                    )

                    # Rescale running state
                    row_sum[r] = row_sum[r] * correction
                    acc_O_mn[r, None] = (
                        acc_O_mn[r, None].load() * correction)

                    # exp2(score * scale_log2e - max * scale_log2e)
                    acc_S_row_exp = cute.math.exp2(
                        acc_S_row * Float32(self.scale_log2e)
                        - m_new * Float32(self.scale_log2e),
                        fastmath=True,
                    )

                    # Update running sum
                    acc_S_row_sum = acc_S_row_exp.reduce(
                        cute.ReductionOp.ADD, Float32(0.0), 0)
                    row_sum[r] = row_sum[r] + acc_S_row_sum

                    # Update max and store P values
                    row_max[r] = m_new
                    acc_S_mn[r, None] = acc_S_row_exp

                # -- Write P (fp16) to P smem region --
                tCsP = thr_mma.partition_C(sP)
                for i in cutlass.range_constexpr(cute.size(acc_S)):
                    tCsP[i] = acc_S[i].to(self.q_dtype)

                # Sync so all MMA threads see P in smem
                named_barrier_sync(
                    Int32(2), Int32(self.num_mma_threads))

                # -- O GEMM: acc_O += P @ V --
                for kb in cutlass.range_constexpr(self.n_block // 16):
                    cute.autovec_copy(
                        tAsP[None, None, kb], tArP[None, None, kb])
                    cute.autovec_copy(
                        tBsVt[None, None, kb], tBrVt[None, None, kb])
                    cute.gemm(tiled_mma, acc_O,
                              tArP[None, None, kb],
                              tBrVt[None, None, kb], acc_O)

            # === Normalize O by row_sum (with quad reduction) ===
            acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
            for r in cutlass.range_constexpr(num_rows):
                row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
                inv_sum = cute.arch.rcp_approx(row_sum[r])
                acc_O_mn[r, None] = (
                    acc_O_mn[r, None].load() * inv_sum)

            # === Write O to smem (at page start, for TMA store) ===
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
    # Forward Compute — Scalar (fp32)
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_BH, tile_M, tile_D, q, k, v, o):
        """Flash attention forward with online softmax (scalar path).

        1. Read Q rows from smem to registers (loaded by TMA in load phase)
        2. Init compute-local mbarrier for async K/V copies
        3. For each KV position:
           a. Async bulk copy K+V rows from global to smem
           b. Wait on mbarrier
           c. Dot product, causal mask, online softmax, O accumulation
        4. Write O to smem for TMA store
        """
        q_smem = cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem)

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.num_warps
        thr_layout = cute.make_layout(32)
        elems = self.elems_per_lane

        # Head offset for K/V global addressing
        head_offset = tile_BH * Int32(self.N * self.D)

        # Mbarrier address in smem
        kv_mbar_addr = page_ptr + Int32(self.mbar_offset)
        mbar_ptr = cute.make_ptr(
            cutlass.Int64, kv_mbar_addr, cute.AddressSpace.smem
        )

        # CopyBulkG2SOp atom for K/V row copies
        g2s = cute.make_copy_atom(
            CopyBulkG2SOp(), self.q_dtype,
            num_bits_per_copy=self.kv_row_nbits,
        )

        # ----- 1. Read Q rows from smem to fp32 registers -----
        q_f32 = []
        for r in cutlass.range_constexpr(self.rows_per_warp):
            local_row = warp_idx + Int32(r * num_warps)
            q_row = cute.make_tensor(
                q_smem + local_row * self.D,
                cute.make_layout(self.D),
            )
            q_part = cute.local_partition(q_row, thr_layout, lane_idx)
            q_reg = cute.make_fragment_like(q_part)
            cute.autovec_copy(q_part, q_reg)
            q_f32.append([q_reg[i].to(Float32) for i in range(elems)])

        # ----- 2. Init compute-local mbarrier -----
        if warp_idx == Int32(0):
            with cute.arch.elect_one():
                mbarrier_init(kv_mbar_addr, Int32(1))
        mbarrier_init_fence()

        # ----- 3. Init per-row accumulators as register fragments -----
        # Register fragments (backed by alloca) survive dynamic loop mutations,
        # unlike Python lists which the DSL can't track across dynamic range().
        rpw = self.rows_per_warp
        m_frag = cute.make_fragment(cute.make_layout(rpw), Float32)
        l_frag = cute.make_fragment(cute.make_layout(rpw), Float32)
        o_frag = cute.make_fragment(
            cute.make_layout(rpw * elems), Float32,
        )
        for r in cutlass.range_constexpr(rpw):
            m_frag[r] = Float32(-1e30)
            l_frag[r] = Float32(0.0)
            for i in cutlass.range_constexpr(elems):
                o_frag[r * elems + i] = Float32(0.0)

        # ----- 4. Prefetch first KV pair into buf[0] -----
        if warp_idx == Int32(0):
            with cute.arch.elect_one():
                mbarrier_arrive_expect_tx(
                    kv_mbar_addr, Int32(2 * self.kv_row_bytes),
                )
                # Copy K[0] → buf[0]
                g_k = cute.make_tensor(
                    k.iterator + head_offset,
                    cute.make_layout((self.D,)),
                )
                s_k = cute.make_tensor(
                    cute.make_ptr(
                        self.q_dtype,
                        page_ptr + Int32(self.k_buf_base),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.D,)),
                )
                gk_src, sk_dst = group_bulk_copy_modes(g_k, s_k)
                cute.copy(g2s, gk_src, sk_dst, mbar_ptr=mbar_ptr)
                # Copy V[0] → buf[0]
                g_v = cute.make_tensor(
                    v.iterator + head_offset,
                    cute.make_layout((self.D,)),
                )
                s_v = cute.make_tensor(
                    cute.make_ptr(
                        self.q_dtype,
                        page_ptr + Int32(self.v_buf_base),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.D,)),
                )
                gv_src, sv_dst = group_bulk_copy_modes(g_v, s_v)
                cute.copy(g2s, gv_src, sv_dst, mbar_ptr=mbar_ptr)

        # ----- 5. KV loop (2-stage pipelined, dynamic range) -----
        # Using range() (not range_constexpr) so N doesn't unroll the loop.
        # Buffer selection is arithmetic: base + (kv_idx % 2) * stride.
        for kv_idx in range(self.N):
            phase = kv_idx % Int32(2)
            k_off = Int32(self.k_buf_base) + phase * Int32(self.k_buf_stride)
            v_off = Int32(self.v_buf_base) + phase * Int32(self.v_buf_stride)

            # 5a. Wait for current buffer
            mbarrier_wait(kv_mbar_addr, phase)

            # 5b. Prefetch next KV pair into alternate buffer (overlaps compute)
            if kv_idx < Int32(self.N - 1):
                next_phase = (kv_idx + Int32(1)) % Int32(2)
                next_k_off = (
                    Int32(self.k_buf_base)
                    + next_phase * Int32(self.k_buf_stride)
                )
                next_v_off = (
                    Int32(self.v_buf_base)
                    + next_phase * Int32(self.v_buf_stride)
                )
                next_global_off = head_offset + (kv_idx + Int32(1)) * Int32(self.D)
                if warp_idx == Int32(0):
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(
                            kv_mbar_addr, Int32(2 * self.kv_row_bytes),
                        )
                        # Copy K[kv_idx+1] → next buf
                        g_k = cute.make_tensor(
                            k.iterator + next_global_off,
                            cute.make_layout((self.D,)),
                        )
                        s_k = cute.make_tensor(
                            cute.make_ptr(
                                self.q_dtype,
                                page_ptr + next_k_off,
                                cute.AddressSpace.smem,
                            ),
                            cute.make_layout((self.D,)),
                        )
                        gk_src, sk_dst = group_bulk_copy_modes(g_k, s_k)
                        cute.copy(g2s, gk_src, sk_dst, mbar_ptr=mbar_ptr)
                        # Copy V[kv_idx+1] → next buf
                        g_v = cute.make_tensor(
                            v.iterator + next_global_off,
                            cute.make_layout((self.D,)),
                        )
                        s_v = cute.make_tensor(
                            cute.make_ptr(
                                self.q_dtype,
                                page_ptr + next_v_off,
                                cute.AddressSpace.smem,
                            ),
                            cute.make_layout((self.D,)),
                        )
                        gv_src, sv_dst = group_bulk_copy_modes(g_v, s_v)
                        cute.copy(g2s, gv_src, sv_dst, mbar_ptr=mbar_ptr)

            # 5c. Read K and V from current buffer (arithmetic offset)
            k_smem_row = cute.make_tensor(
                cute.make_ptr(
                    self.q_dtype,
                    page_ptr + k_off,
                    cute.AddressSpace.smem,
                ),
                cute.make_layout(self.D),
            )
            k_part = cute.local_partition(k_smem_row, thr_layout, lane_idx)
            k_reg = cute.make_fragment_like(k_part)
            cute.autovec_copy(k_part, k_reg)

            v_smem_row = cute.make_tensor(
                cute.make_ptr(
                    self.q_dtype,
                    page_ptr + v_off,
                    cute.AddressSpace.smem,
                ),
                cute.make_layout(self.D),
            )
            v_part = cute.local_partition(v_smem_row, thr_layout, lane_idx)
            v_reg = cute.make_fragment_like(v_part)
            cute.autovec_copy(v_part, v_reg)

            # 5d. Process each Q row assigned to this warp
            for r in cutlass.range_constexpr(self.rows_per_warp):
                local_row = warp_idx + Int32(r * num_warps)
                if local_row < Int32(self.tile_size_M):
                    # Dot product q·k
                    partial = Float32(0.0)
                    for i in cutlass.range_constexpr(elems):
                        partial = partial + q_f32[r][i] * k_reg[i].to(Float32)
                    s = cute.arch.warp_reduction(partial, operator.add)
                    s = s * Float32(self.scale_val)

                    # Causal mask
                    if self.causal:
                        q_pos = tile_M * self.tile_size_M + local_row
                        max_kv = q_pos + Int32(self.N - self.M)
                        if kv_idx > max_kv:
                            s = Float32(-1e30)

                    # Online softmax (register fragment accumulators)
                    m_old = m_frag[r]
                    m_new = m_old
                    if s > m_old:
                        m_new = s
                    correction = cute.math.exp(
                        m_old - m_new, fastmath=True
                    )
                    p = cute.math.exp(s - m_new, fastmath=True)
                    l_frag[r] = l_frag[r] * correction + p

                    # O accumulation
                    for i in cutlass.range_constexpr(elems):
                        o_frag[r * elems + i] = (
                            o_frag[r * elems + i] * correction
                            + p * v_reg[i].to(Float32)
                        )

                    m_frag[r] = m_new

        # ----- 6. Write O to smem Q region -----
        for r in cutlass.range_constexpr(self.rows_per_warp):
            local_row = warp_idx + Int32(r * num_warps)
            row_idx = tile_M * self.tile_size_M + local_row
            if local_row < Int32(self.tile_size_M):
                if row_idx < Int32(self.M):
                    inv_l = Float32(1.0) / l_frag[r]
                    o_row = cute.make_tensor(
                        q_smem + local_row * self.D,
                        cute.make_layout(self.D),
                    )
                    o_part = cute.local_partition(
                        o_row, thr_layout, lane_idx
                    )
                    o_reg = cute.make_fragment_like(o_part)
                    for i in cutlass.range_constexpr(elems):
                        o_reg[i] = (
                            o_frag[r * elems + i] * inv_l
                        ).to(self.q_dtype)
                    cute.autovec_copy(o_reg, o_part)

    # =========================================================================
    # Forward Store (3D TMA S->G for O)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_BH, tile_M, tile_D, o_tma, o_tma_gmem):
        """TMA store of O from shared to global memory.

        Smem (D, tile_M, 1) col-major = (tile_M, D) row-major.
        TMA handles boundary conditions for partial last M tile.
        """
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
