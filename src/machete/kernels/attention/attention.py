# Copyright (c) 2025, Machete Authors
"""
Flash Attention Op for the Megakernel (fp16/bf16 tensor core MMA).

Computes scaled dot-product attention with online softmax:
    O[BH, M, D] = softmax(Q[BH, M, D] @ K[BH, N, D]^T / sqrt(D)) @ V[BH, N, D]

Pipelined load/compute/store via framework inner_iters:
    load iter 0:  3D TMA G->S for Q tile (plain smem layout)
    load iter 1+: 3D TMA G->S for K+V blocks (swizzled smem, double-buffered)
    compute:      Read Q from smem to registers, then KV block loop with
                  LdMatrix reads, online softmax, MMA GEMMs.
    store:        3D TMA S->G for O tile

All TMA loads are issued by the DMA warp via the framework's inner_iters
mechanism.  Double-buffer design: two copies of (K + V) in the page, with
DMA prefetching the next KV block while compute processes the current one.

Supports optional causal masking (lower-left aligned):
    Row i in Q can attend to K/V positions 0..(i + N - M).

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

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu import warp

from machete.megakernel.ops import Op
from machete.megakernel.interpreter import (
    mbarrier_arrive,
    mbarrier_arrive_expect_tx,
    mbarrier_wait,
    named_barrier_sync,
)
from machete.megakernel.paged_memory import PAGE_SIZE


class FlashAttentionOp(Op):
    """Flash Attention operation for the megakernel framework.

    Tensors:
        q: (BH, M, D) -- query  (fp16 or bf16)
        k: (BH, N, D) -- key
        v: (BH, N, D) -- value
        o: (BH, M, D) -- output

    Tiling:
        tile_BH=1 (per head), tile_M from schedule, tile_D=D (full).

    Smem page layout (double-buffered KV):
        buf0: [K0: n_block x D] [V0: n_block x D]  (= kv_buf_stride bytes)
        buf1: [K1: n_block x D] [V1: n_block x D]  (= kv_buf_stride bytes)
        Q occupies buf0 during iter 0, then buf0 is reused for KV blocks.
    """

    reads = {
        "q": (None, ("BH", "M", "D")),
        "k": (None, ("BH", "N", "D")),
        "v": (None, ("BH", "N", "D")),
    }
    writes = {"o": (None, ("BH", "M", "D"))}
    tile = ("BH", "M", "D")

    tma_loads = {"q", "k", "v"}
    tma_stores = {"o"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        """Custom TMA tile shapes for K/V (n_block sub-tiling)."""
        if tensor_name in ("k", "v"):
            n_block = static_dims["n_block"]
            D = static_dims["D"]
            return (1, n_block, D)
        return None  # Q/O use defaults

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape,
                                tile_sizes, static_dims):
        """Swizzled smem layout for K/V TMA descriptors."""
        if tensor_name not in ("k", "v"):
            return None

        D = static_dims["D"]
        # Swizzle B: largest SW mode whose atom fits in one D-row.
        # SW128(B=3)→D≥64, SW64(B=2)→D≥32, SW32(B=1)→D≥16.
        # M=4, S=3 fixed (same as GemmOp): guarantees M≠S for all D.
        if D >= 64:
            B = 3
        elif D >= 32:
            B = 2
        else:
            B = 1

        dim0, dim1, dim2 = tma_tile_shape  # (D, n_block, 1) after reversal
        return (
            f"cute.make_composed_layout("
            f"cute.make_swizzle({B}, 4, 3), 0, "
            f"cute.make_layout(({dim0}, {dim1}, {dim2}), "
            f"stride=(1, {dim0}, {dim0 * dim1})))"
        )

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
        """Init tensor core MMA path with dynamic n_block and swizzle."""
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
            f"FlashAttentionOp: D={self.D} must be >= 16 and x16.")

        assert self.q_tile_bytes <= PAGE_SIZE, (
            f"FlashAttentionOp: Q tile ({self.q_tile_bytes}B) > "
            f"PAGE_SIZE ({PAGE_SIZE}B). Reduce tile_size_M.")

        # --- Swizzle parameters (must match get_tma_smem_layout_src) ---
        # SW128(B=3)→D≥64, SW64(B=2)→D≥32, SW32(B=1)→D≥16.
        # M=4, S=3 fixed (GemmOp convention): guarantees M≠S for all D.
        if self.D >= 64:
            self.swizzle_B = 3
        elif self.D >= 32:
            self.swizzle_B = 2
        else:
            self.swizzle_B = 1
        self.swizzle_M = 4
        self.swizzle_S = 3

        # --- Dynamic n_block computation (double-buffer constraint) ---
        # 2 buffers x (K + V) = 4 x n_block x D x elem_bytes <= PAGE_SIZE
        max_n_block = PAGE_SIZE // (4 * self.D * self.elem_bytes)
        self.n_block = (max_n_block // 16) * 16
        self.n_block = min(self.n_block, self.N)
        # Round down to x16 after clamping to N (TMA zero-fills partial tiles)
        self.n_block = max(16, (self.n_block // 16) * 16)
        self.num_kv_blocks = (self.N + self.n_block - 1) // self.n_block

        # Smem layout: buf0=[K0, V0], buf1=[K1, V1]
        self.kv_tile_bytes = self.n_block * self.D * self.elem_bytes
        self.kv_buf_stride = 2 * self.kv_tile_bytes  # K + V per buffer
        total_kv_smem = 2 * self.kv_buf_stride
        assert total_kv_smem <= PAGE_SIZE, (
            f"FlashAttentionOp: KV double-buffer ({total_kv_smem}B) > "
            f"PAGE_SIZE ({PAGE_SIZE}B).")

        # Q must fit in one buffer slot (Q loaded into buf0, then reused for KV)
        assert self.q_tile_bytes <= self.kv_buf_stride, (
            f"FlashAttentionOp: Q tile ({self.q_tile_bytes}B) > "
            f"kv_buf_stride ({self.kv_buf_stride}B). Reduce tile_size_M.")

        # Framework inner iterations: DMA warp calls load() for Q + each KV block
        self.inner_iters = 1 + self.num_kv_blocks
        self.inner_depth = 2  # Double buffering

        # exp2-based softmax
        self.scale_log2e = self.scale_val * 1.4426950408889634074

        # Override compute method
        self.compute = self.compute_mma

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule_forward(cls, tile_sizes=None, causal=False, **tensors):
        """Schedule flash attention forward, optionally with causal masking."""
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("BH", 1)
        q = tensors.get('q')
        n_block = 16  # default
        if q is not None:
            assert q.element_size() == 2, (
                f"FlashAttentionOp requires fp16/bf16, "
                f"got element_size={q.element_size()}")
            D = q.shape[-1]
            M = q.shape[1]
            N = tensors['k'].shape[1]
            elem = q.element_size()
            # n_block from double-buffer constraint
            max_n_block = PAGE_SIZE // (4 * D * elem)
            n_block = (max_n_block // 16) * 16
            n_block = min(n_block, N)
            n_block = max(16, (n_block // 16) * 16)
            if "M" not in tile_sizes:
                # Q must fit in one buffer = kv_buf_stride = 2*n_block*D*elem
                max_tile_M_buf = 2 * n_block
                max_nw = min(7, max_tile_M_buf // 16, M // 16)
                tile_M = max(16, max_nw * 16)
                tile_sizes["M"] = tile_M
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        if q is not None:
            ops[0].static_dims['n_block'] = n_block
        if causal:
            ops[0].static_dims['causal'] = 1
        return ops

    # =========================================================================
    # Forward Load (DMA warp: Q on iter 0, K+V on iter 1+)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_BH, tile_M, tile_D,
             q_tma, q_tma_gmem, k_tma, k_tma_gmem, v_tma, v_tma_gmem,
             work_mbar, inner_iter_idx):
        """TMA load dispatched by inner_iter_idx.

        iter 0:  TMA Q tile (plain smem, no swizzle)
        iter 1+: TMA K+V block (swizzled smem for bank-conflict-free reads)
        """
        _buf_base = page_ptr + (
            inner_iter_idx % Int32(self.inner_depth)
        ) * Int32(self.kv_buf_stride)
        mbar_ptr = cute.make_ptr(
            cutlass.Int64, work_mbar, cute.AddressSpace.smem)

        # --- Q load (iter 0) ---
        if inner_iter_idx == Int32(0):
            sQ = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _buf_base,
                              cute.AddressSpace.smem),
                cute.make_layout((self.D, self.tile_size_M, 1)),
            )
            gQ = cute.local_tile(
                q_tma_gmem, (self.D, self.tile_size_M, 1),
                (None, None, None),
            )
            tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
                q_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(gQ, 0, 3),
            )
            nbytes = Int32(self.q_tile_bytes)
            with cute.arch.elect_one():
                mbarrier_arrive_expect_tx(work_mbar, nbytes)
            cute.copy(q_tma, tQgQ[(None, tile_D, tile_M, tile_BH)],
                      tQsQ, tma_bar_ptr=mbar_ptr)

        # --- K+V load (iter 1+) ---
        if inner_iter_idx > Int32(0):
            kv_block_idx = inner_iter_idx - Int32(1)
            swz = cute.make_swizzle(
                self.swizzle_B, self.swizzle_M, self.swizzle_S)

            # K at buf_base
            sK = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, _buf_base,
                                  cute.AddressSpace.smem),
                    swz, dtype=self.q_dtype),
                cute.make_layout(
                    (self.D, self.n_block, 1),
                    stride=(1, self.D, self.D * self.n_block)),
            )
            gK = cute.local_tile(
                k_tma_gmem, (self.D, self.n_block, 1),
                (None, None, None),
            )
            tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
                k_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(gK, 0, 3),
            )

            # V at buf_base + kv_tile_bytes
            sV = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype,
                                  _buf_base + Int32(self.kv_tile_bytes),
                                  cute.AddressSpace.smem),
                    swz, dtype=self.q_dtype),
                cute.make_layout(
                    (self.D, self.n_block, 1),
                    stride=(1, self.D, self.D * self.n_block)),
            )
            gV = cute.local_tile(
                v_tma_gmem, (self.D, self.n_block, 1),
                (None, None, None),
            )
            tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
                v_tma, Int32(0), cute.make_layout(1),
                cute.group_modes(sV, 0, 3),
                cute.group_modes(gV, 0, 3),
            )

            nbytes = Int32(2 * self.kv_tile_bytes)
            with cute.arch.elect_one():
                mbarrier_arrive_expect_tx(work_mbar, nbytes)
            cute.copy(k_tma,
                      tKgK[(None, Int32(0), kv_block_idx, tile_BH)],
                      tKsK, tma_bar_ptr=mbar_ptr)
            cute.copy(v_tma,
                      tVgV[(None, Int32(0), kv_block_idx, tile_BH)],
                      tVsV, tma_bar_ptr=mbar_ptr)

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
    # Forward Compute -- Tensor Core MMA (fp16/bf16)
    # =========================================================================

    @cute.jit
    def compute_mma(self, page_ptr, tile_BH, tile_M, tile_D,
                    work_mbar, smem_consumed_mbar, work_mbar_phase):
        """Flash attention forward with multi-warp tensor core MMA.

        Uses framework inner_iters for DMA-driven KV loads.
        k=0 (Q): read from buf 0 to registers, signal smem_consumed.
        k=1+ (KV blocks): S GEMM + softmax + O GEMM from alternating buffers.
        """
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_id = tidx % Int32(32)

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

            # === Q register fragment (filled during k=0) ===
            sQ = cute.make_tensor(
                cute.make_ptr(self.q_dtype, page_ptr,
                              cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.tile_size_M, self.D),
                                 stride=(self.D, 1)),
            )
            tCsQ = thr_mma.partition_A(sQ)
            tCrQ = tiled_mma.make_fragment_A(tCsQ)

            # === Swizzle + LdMatrix setup for KV ===
            swz = cute.make_swizzle(
                self.swizzle_B, self.swizzle_M, self.swizzle_S)

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

            # Per-buffer K and V^T smem tensors + LdMatrix partitions
            _buf_tKsK = []
            _buf_tVsVt = []
            _buf_tCrK = []
            _buf_tBrVt = []
            _buf_tKrK_view = []
            _buf_tVrVt_view = []
            for _d in cutlass.range_constexpr(2):
                buf_base = page_ptr + Int32(_d * self.kv_buf_stride)

                # K: (n_block, D) row-major with swizzle
                sK_ptr = cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, buf_base,
                                  cute.AddressSpace.smem,
                                  assumed_align=128),
                    swz, dtype=self.q_dtype)
                sK = cute.make_tensor(
                    sK_ptr,
                    cute.make_layout((self.n_block, self.D),
                                     stride=(self.D, 1)))

                # V^T: (D, n_block) transposed view with same swizzle
                sVt_ptr = cute.recast_ptr(
                    cute.make_ptr(self.q_dtype,
                                  buf_base + Int32(self.kv_tile_bytes),
                                  cute.AddressSpace.smem,
                                  assumed_align=128),
                    swz, dtype=self.q_dtype)
                sVt = cute.make_tensor(
                    sVt_ptr,
                    cute.make_layout((self.D, self.n_block),
                                     stride=(1, self.D)))

                # MMA fragments
                tCsK = thr_mma.partition_B(sK)
                tCrK = tiled_mma.make_fragment_B(tCsK)
                tBsVt = thr_mma.partition_B(sVt)
                tBrVt = tiled_mma.make_fragment_B(tBsVt)

                # LdMatrix partitions
                tKsK = smem_thr_copy_K.partition_S(sK)
                tKrK_view = smem_thr_copy_K.retile(tCrK)
                tVsVt = smem_thr_copy_Vt.partition_S(sVt)
                tVrVt_view = smem_thr_copy_Vt.retile(tBrVt)

                _buf_tKsK.append(tKsK)
                _buf_tVsVt.append(tVsVt)
                _buf_tCrK.append(tCrK)
                _buf_tBrVt.append(tBrVt)
                _buf_tKrK_view.append(tKrK_view)
                _buf_tVrVt_view.append(tVrVt_view)

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

            # === Phase tracking (same as GemmOp pattern) ===
            _kb_phases = [work_mbar_phase]
            for _d in cutlass.range_constexpr(self.inner_depth - 1):
                _kb_phases.append(Int32(0))

            # === Main loop: k=0 is Q, k=1+ are KV blocks ===
            for k in cutlass.range_constexpr(1 + self.num_kv_blocks):
                if k > 0:
                    # Signal previous buffer consumed
                    prev_buf = (k - 1) % 2
                    named_barrier_sync(
                        Int32(2), Int32(self.num_mma_threads))
                    if lane_id == Int32(0):
                        mbarrier_arrive(
                            smem_consumed_mbar + Int32(prev_buf * 8))

                    # Wait for next buffer from DMA
                    cur_buf = k % 2
                    mbarrier_wait(
                        work_mbar + Int32(cur_buf * 8),
                        _kb_phases[cur_buf])
                    _kb_phases[cur_buf] = (
                        Int32(1) - _kb_phases[cur_buf])

                if k == 0:
                    # === Read Q from buf 0 to registers ===
                    for kb in cutlass.range_constexpr(self.D // 16):
                        cute.autovec_copy(
                            tCsQ[None, None, kb],
                            tCrQ[None, None, kb])
                else:
                    # === KV block processing ===
                    cur_buf = k % 2
                    kv_start = Int32((k - 1) * self.n_block)

                    # --- S GEMM: acc_S = Q(regs) @ K(smem)^T ---
                    tKsK_cur = _buf_tKsK[cur_buf]
                    tCrK_cur = _buf_tCrK[cur_buf]
                    tKrK_view_cur = _buf_tKrK_view[cur_buf]

                    acc_S.fill(0.0)
                    for kb in cutlass.range_constexpr(self.D // 16):
                        cute.copy(smem_tiled_copy_K,
                                  tKsK_cur[None, None, kb],
                                  tKrK_view_cur[None, None, kb])
                        cute.gemm(tiled_mma, acc_S,
                                  tCrQ[None, None, kb],
                                  tCrK_cur[None, None, kb], acc_S)

                    # --- Mask + online softmax ---
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
                            # Causal mask
                            if self.causal:
                                global_row = (
                                    tile_M * Int32(self.tile_size_M)
                                    + Int32(row_idx))
                                if global_col > global_row + Int32(
                                        self.N - self.M):
                                    acc_S_mn[r, c] = Float32(-1e30)

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

                    # --- P in registers (rP_mma_view trick) ---
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

                    # --- O GEMM: acc_O += P(regs) @ V(smem)^T ---
                    tVsVt_cur = _buf_tVsVt[cur_buf]
                    tBrVt_cur = _buf_tBrVt[cur_buf]
                    tVrVt_view_cur = _buf_tVrVt_view[cur_buf]

                    for kb in cutlass.range_constexpr(
                            self.n_block // 16):
                        cute.copy(smem_tiled_copy_Vt,
                                  tVsVt_cur[None, None, kb],
                                  tVrVt_view_cur[None, None, kb])
                        cute.gemm(tiled_mma, acc_O,
                                  tOrS[None, None, kb],
                                  tBrVt_cur[None, None, kb], acc_O)

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
