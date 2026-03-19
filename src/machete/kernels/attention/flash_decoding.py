# Copyright (c) 2025, Machete Authors
"""
FlashDecoding: Split-KV Attention for decode workloads.

For small BH (batch×heads) and small M (query length), the standard FA kernel
uses only 1 CTA per head, leaving most SMs idle. FlashDecoding splits the KV
sequence across multiple CTAs (splits), each producing partial O and LSE in
fp32. A combine kernel reduces the partials into the final output.

Architecture:
    FlashDecodingSplitOp: tile=(BH, SPLIT) — each CTA handles a KV range.
    Combine is fused into the store warp epilogue (no separate op needed).

Usage:
    from machete.kernels.attention.flash_decoding import flash_decoding_schedule
    from machete.megakernel import Megakernel

    ops, config = flash_decoding_schedule(q=q, k=k, v=v, o=o)
    kernel = Megakernel(ops, config=config)
    kernel.run()
"""

import math

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32
from cutlass.cute.nvgpu import warp

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import named_barrier_sync, atomic_add_acq_rel_gpu_i32


class FlashDecodingSplitOp(Op):
    """Split-KV attention: each tile processes a subset of KV blocks.

    Tensors:
        q: (BH, M, D) — query (fp16/bf16)
        k: (BH_kv, N, D) — key
        v: (BH_kv, N, D) — value
        o_partial: (BH, SPLIT, M, D) — partial output (fp32)
        lse_partial: (BH, SPLIT, M) — partial log-sum-exp (fp32)

    Tiling:
        tile_BH=1, tile_SPLIT=1 → BH*num_splits tiles total.

    Each tile computes attention over KV blocks [kv_start, kv_end) and writes
    fp32 partial O and LSE to global memory (no TMA store needed).
    """

    reads = {
        "q": (None, ("BH", "M", "D")),
        "k": (None, ("BH_kv", "N", "D")),
        "v": (None, ("BH_kv", "N", "D")),
    }
    writes = {
        "o_partial": (cutlass.Float32, ("BH", "SPLIT", "M", "D")),
        "lse_partial": (cutlass.Float32, ("BH", "SPLIT", "M")),
        "o": (None, ("BH", "M", "D")),
        "lse": (cutlass.Float32, ("BH", "M")),
        "split_counter": (cutlass.Int32, ("BH",)),
    }
    tile = ("BH", "SPLIT")

    # Q via TMA (DMA warp loads once), K/V via cpasync in compute
    tma_loads = {"q"}
    tma_stores = set()  # Partials written to global from compute

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        """Swizzled smem layout for Q TMA load descriptor."""
        if tensor_name != "q":
            return None
        D = static_dims["D"]
        if D >= 64:
            B = 3
        elif D >= 32:
            B = 2
        else:
            B = 1
        dim0, dim1, dim2 = tma_tile_shape
        return (
            f"cute.make_composed_layout("
            f"cute.make_swizzle({B}, 4, 3), 0, "
            f"cute.make_layout(({dim0}, {dim1}, {dim2}), "
            f"stride=(1, {dim0}, {dim0 * dim1})))"
        )

    def __init__(self, **config):
        super().__init__(**config)
        self.causal = getattr(self, "causal", 0)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)
        self.kv_group_size = getattr(self, "kv_group_size", 1)

        assert self.q_dtype in (cutlass.Float16, cutlass.BFloat16), (
            f"FlashDecodingSplitOp requires fp16 or bf16, got {self.q_dtype}"
        )
        self.elem_bytes = 2

        self.scale_val = 1.0 / (self.D ** 0.5)
        self.q_tile_bytes = self.M * self.D * self.elem_bytes

        # Raw pointer for atomic counter (set via static_dims)
        self.split_counter_ptr = getattr(self, "split_counter_ptr", 0)

        # Stride overrides for strided K/V
        self.k_bh_stride = getattr(self, "k_bh_stride", self.N * self.D)
        self.k_n_stride = getattr(self, "k_n_stride", self.D)
        self.v_bh_stride = getattr(self, "v_bh_stride", self.N * self.D)
        self.v_n_stride = getattr(self, "v_n_stride", self.D)

        # num_splits stored as SPLIT dim extent
        self.num_splits = self.SPLIT

        self._init_mma()

    def _init_mma(self):
        """Init MMA path — tile_M = M (full query, since M is small for decode)."""
        self.tile_size_M = self.M  # Full M per tile (decode: M is small)
        assert self.tile_size_M % 16 == 0 and self.tile_size_M >= 16, (
            f"FlashDecodingSplitOp: M={self.M} must be a positive multiple of 16."
        )
        self.num_mma_warps = self.tile_size_M // 16
        max_warps = self.threads_per_row // 32
        assert self.num_mma_warps <= max_warps, (
            f"FlashDecodingSplitOp: M={self.M} requires "
            f"{self.num_mma_warps} warps but only {max_warps} available."
        )
        self.num_mma_threads = self.num_mma_warps * 32

        assert self.D >= 16 and self.D % 16 == 0

        assert self.q_tile_bytes <= self.page_size, (
            f"FlashDecodingSplitOp: Q tile ({self.q_tile_bytes}B) > page_size ({self.page_size}B)."
        )

        # KV smem padding: same technique as Sm120Op — distributes LdMatrix
        # accesses across banks with zero register overhead. Free for decode
        # because n_block rounds to the same power-of-2 (plenty of kv_budget).
        self.kv_pad = 8

        # n_block: Q persists + KV double-buffer (with padding)
        kv_budget = self.page_size - self.q_tile_bytes
        assert kv_budget > 0
        max_n_block = kv_budget // (2 * (self.D + self.kv_pad) * self.elem_bytes)
        self.n_block = 1 << int(math.log2(max(16, max_n_block)))
        if self.N < self.n_block:
            self.n_block = max(16, (self.N // 16) * 16)
        self.num_kv_blocks = (self.N + self.n_block - 1) // self.n_block

        self.kv_tile_bytes = self.n_block * (self.D + self.kv_pad) * self.elem_bytes

        # Compute KV blocks per split
        self.blocks_per_split = (self.num_kv_blocks + self.num_splits - 1) // self.num_splits

        self.inner_iters = 1
        self.inner_depth = 1

        # Swizzle for Q (TMA-constrained: S=3)
        if self.D >= 64:
            self.swizzle_B = 3
        elif self.D >= 32:
            self.swizzle_B = 2
        else:
            self.swizzle_B = 1
        self.swizzle_M = 4
        self.swizzle_S = 3

        # cpasync thread layout
        self.async_copy_elems = 128 // (self.elem_bytes * 8)
        self.copy_dim1 = self.D // self.async_copy_elems
        self.copy_dim0 = self.num_mma_threads // self.copy_dim1
        assert self.n_block % self.copy_dim0 == 0

        # exp2-based softmax
        self.scale_log2e = self.scale_val * 1.4426950408889634074
        self.rescale_threshold = 8.0

        # Override compute
        self.compute = self.compute_mma

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule(cls, tile_sizes=None, causal=False,
                         page_size=DEFAULT_PAGE_SIZE, kv_group_size=1,
                         num_splits=0, **tensors):
        """Schedule split-KV forward pass with fused combine.

        Allocates intermediate buffers (o_partial, lse_partial, split_counter)
        and schedules split ops. The combine is fused into the store warp
        epilogue — no separate CombineOp needed.

        Requires 'o' and 'lse' in tensors for the fused combine output.
        """
        import torch

        q = tensors["q"]
        k = tensors["k"]
        BH, M, D = q.shape
        N = k.shape[1]

        assert q.element_size() == 2
        assert "o" in tensors, "FlashDecodingSplitOp.schedule requires 'o' tensor"

        # Auto num_splits
        if num_splits <= 0:
            num_SMs = torch.cuda.get_device_properties(q.device).multi_processor_count
            elem = q.element_size()
            # Estimate n_block
            q_tile_bytes = M * D * elem
            kv_budget = page_size - q_tile_bytes
            max_n_block = kv_budget // (2 * (D + 8) * elem)  # +8 = kv_pad
            n_block = 1 << int(math.log2(max(16, max_n_block)))
            if N < n_block:
                n_block = max(16, (N // 16) * 16)
            num_n_blocks = (N + n_block - 1) // n_block

            total_mblocks = BH  # 1 M-tile per head for decode
            # Cap splits so each has >= MIN_BLOCKS_PER_SPLIT KV blocks.
            # This enables cpasync K/V pipelining within each split and
            # reduces combine overhead (fewer fp32 partials to reduce).
            min_blocks_per_split = 4
            max_splits = max(1, num_n_blocks // min_blocks_per_split)
            num_splits = min(num_SMs // max(total_mblocks, 1), max_splits)
            num_splits = max(num_splits, 1)

        # Allocate intermediate buffers
        o_partial = torch.empty(BH, num_splits, M, D, dtype=torch.float32, device=q.device)
        lse_partial = torch.empty(BH, num_splits, M, dtype=torch.float32, device=q.device)

        # Allocate split counter for fused combine (atomic, one per head)
        split_counter = torch.zeros(BH, dtype=torch.int32, device=q.device)

        # Allocate LSE output if not provided
        if "lse" not in tensors:
            tensors["lse"] = torch.empty(BH, M, dtype=torch.float32, device=q.device)

        tile_sizes = dict(tile_sizes or {})
        tile_sizes["BH"] = 1
        tile_sizes["SPLIT"] = 1

        tensors["o_partial"] = o_partial
        tensors["lse_partial"] = lse_partial
        tensors["split_counter"] = split_counter

        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        ops[0].static_dims["num_splits"] = num_splits
        if causal:
            ops[0].static_dims["causal"] = 1
        if kv_group_size > 1:
            ops[0].static_dims["kv_group_size"] = kv_group_size

        # Raw pointer for atomic counter
        ops[0].static_dims["split_counter_ptr"] = split_counter.data_ptr()

        # Strided K/V support
        if not k.is_contiguous():
            ops[0].static_dims["k_bh_stride"] = k.stride(0)
            ops[0].static_dims["k_n_stride"] = k.stride(1)
        v = tensors.get("v")
        if v is not None and not v.is_contiguous():
            ops[0].static_dims["v_bh_stride"] = v.stride(0)
            ops[0].static_dims["v_n_stride"] = v.stride(1)

        return ops

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for split ops."""
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS

        M = ops[0].static_dims["M"]
        num_mma_warps = M // 16
        threads_per_block = (num_mma_warps + NUM_DMA_WARPS) * 32
        page_size = ops[0].static_dims.get("page_size", DEFAULT_PAGE_SIZE)
        return MegakernelConfig(threads_per_block=threads_per_block, page_size=page_size)

    # =========================================================================
    # Load (TMA Q only — same as FlashAttentionSm120Op)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_BH, tile_SPLIT, q_tma, q_tma_gmem, work_mbar):
        """TMA Q load into page (single shot, swizzled for LdMatrix reads).

        Note: tile_SPLIT is unused here — Q is the same for all splits of a head.
        The TMA coord uses tile_BH to select the head.
        """
        from machete.megakernel.interpreter import mbarrier_arrive_expect_tx

        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)

        _q_swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)
        sQ = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem),
                _q_swz,
                dtype=self.q_dtype,
            ),
            cute.make_layout((self.D, self.tile_size_M, 1)),
        )
        gQ = cute.local_tile(
            q_tma_gmem,
            (self.D, self.tile_size_M, 1),
            (None, None, None),
        )
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            q_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(gQ, 0, 3),
        )
        nbytes = Int32(self.q_tile_bytes)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        # Q tile indices: D and M are full-extent (not tile dims), so tile index = 0
        cute.copy(q_tma, tQgQ[(None, Int32(0), Int32(0), tile_BH)], tQsQ, tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # MMA Helpers (same as Sm120Op)
    # =========================================================================

    def _make_acc_tensor_mn_view(self, acc):
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
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31))
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31))
        return val

    def _threadquad_reduce_max(self, val):
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val):
        return self._threadquad_reduce(val, lambda x, y: x + y)

    # =========================================================================
    # Compute — split-KV variant of FlashAttentionSm120Op.compute_mma
    # =========================================================================

    @cute.jit
    def compute_mma(self, page_ptr, tile_BH, tile_SPLIT,
                    q, k, v, o_partial, lse_partial, o, lse, split_counter):
        """Split-KV flash attention: process KV blocks [kv_start, kv_end).

        Same MMA pipeline as FlashAttentionSm120Op but:
        - Only processes a subset of KV blocks (determined by tile_SPLIT)
        - Writes fp32 partial O and LSE to global memory (no smem O write)
        - No TMA store (store warp is a no-op)
        """
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        if warp_idx < Int32(self.num_mma_warps):
            # === Determine KV range for this split ===
            kv_start_block = tile_SPLIT * Int32(self.blocks_per_split)
            kv_end_block = (tile_SPLIT + Int32(1)) * Int32(self.blocks_per_split)
            if kv_end_block > Int32(self.num_kv_blocks):
                kv_end_block = Int32(self.num_kv_blocks)

            # === MMA setup ===
            mma_op = warp.MmaF16BF16Op(self.q_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((self.num_mma_warps, 1, 1)),
                permutation_mnk=(self.num_mma_warps * 16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)

            # === Swizzle + LdMatrix setup ===
            swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)
            sQ = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.q_dtype,
                ),
                cute.make_layout((self.tile_size_M, self.D), stride=(self.D, 1)),
            )
            _tCsQ = thr_mma.partition_A(sQ)
            tCrQ = tiled_mma.make_fragment_A(_tCsQ)

            smem_copy_atom_Q = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.q_dtype
            )
            smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
            smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
            tQrQ_view = smem_thr_copy_Q.retile(tCrQ)
            tQsQ = smem_thr_copy_Q.partition_S(sQ)

            smem_copy_atom_K = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.q_dtype
            )
            smem_copy_atom_Vt = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.q_dtype
            )
            smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
            smem_tiled_copy_Vt = cute.make_tiled_copy_B(smem_copy_atom_Vt, tiled_mma)
            smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
            smem_thr_copy_Vt = smem_tiled_copy_Vt.get_slice(tidx)

            # === KV buffer base (after persistent Q, padded rows) ===
            _kv_base = page_ptr + Int32(self.q_tile_bytes)
            _kv_ptr = cute.make_ptr(self.q_dtype, _kv_base, cute.AddressSpace.smem, assumed_align=128)
            _kv_stride = self.D + self.kv_pad  # padded row stride (elements)

            # K smem (buf0, padded stride)
            _sK = cute.make_tensor(
                _kv_ptr,
                cute.make_layout((self.n_block, self.D), stride=(_kv_stride, 1)),
            )
            _tCsK = thr_mma.partition_B(_sK)
            tCrK = tiled_mma.make_fragment_B(_tCsK)
            tKrK_view = smem_thr_copy_K.retile(tCrK)
            tKsK = smem_thr_copy_K.partition_S(_sK)

            # V smem (buf1, padded stride)
            _buf1_base = _kv_base + Int32(self.kv_tile_bytes)
            _buf1_ptr = cute.make_ptr(self.q_dtype, _buf1_base, cute.AddressSpace.smem, assumed_align=128)
            _sVt = cute.make_tensor(
                _buf1_ptr,
                cute.make_layout((self.D, self.n_block), stride=(1, _kv_stride)),
            )
            _tBsVt = thr_mma.partition_B(_sVt)
            tBrVt = tiled_mma.make_fragment_B(_tBsVt)
            tVrVt_view = smem_thr_copy_Vt.retile(tBrVt)
            tVsVt = smem_thr_copy_Vt.partition_S(_sVt)

            # === cpasync tiled copy setup ===
            async_copy_atom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), self.q_dtype, num_bits_per_copy=128)
            copy_thread_layout = cute.make_layout((self.copy_dim0, self.copy_dim1), stride=(self.copy_dim1, 1))
            copy_value_layout = cute.make_layout((1, self.async_copy_elems))
            gmem_tiled_copy = cute.make_tiled_copy_tv(async_copy_atom, copy_thread_layout, copy_value_layout)
            thr_copy = gmem_tiled_copy.get_slice(tidx)

            sK_cp = cute.make_tensor(
                _kv_ptr,
                cute.make_layout((self.n_block, self.D), stride=(_kv_stride, 1)),
            )
            sV_cp = cute.make_tensor(
                _buf1_ptr,
                cute.make_layout((self.n_block, self.D), stride=(_kv_stride, 1)),
            )
            tKsK_cp = thr_copy.partition_D(sK_cp)
            tVsV_cp = thr_copy.partition_D(sV_cp)

            # Global K/V sources
            kv_bh = tile_BH // Int32(self.kv_group_size)
            k_head_ptr = (k.iterator + kv_bh * Int32(self.k_bh_stride)).align(16)
            v_head_ptr = (v.iterator + kv_bh * Int32(self.v_bh_stride)).align(16)
            gK_head = cute.make_tensor(k_head_ptr, cute.make_layout((self.N, self.D), stride=(self.k_n_stride, 1)))
            gV_head = cute.make_tensor(v_head_ptr, cute.make_layout((self.N, self.D), stride=(self.v_n_stride, 1)))

            # P register fragment
            acc_S = cute.make_fragment(tiled_mma.partition_shape_C((self.tile_size_M, self.n_block)), Float32)
            rP = cute.make_fragment_like(acc_S, self.q_dtype)
            rP_ld = cute.logical_divide(rP.layout, (None, None, 2))
            rP_mma_view = cute.make_layout(
                ((rP_ld.shape[0], rP_ld.shape[2][0]), rP_ld.shape[1], rP_ld.shape[2][1]),
                stride=((rP_ld.stride[0], rP_ld.stride[2][0]), rP_ld.stride[1], rP_ld.stride[2][1]),
            )
            tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

            # Accumulators
            acc_O = cute.make_fragment(tiled_mma.partition_shape_C((self.tile_size_M, self.D)), Float32)
            acc_O.fill(0.0)

            acc_O_shape = tiled_mma.partition_shape_C((self.tile_size_M, self.D))
            num_rows = acc_O_shape[0][1] * acc_O_shape[1]
            row_max = cute.make_fragment(cute.make_layout(num_rows), Float32)
            row_sum = cute.make_fragment(cute.make_layout(num_rows), Float32)
            for r in cutlass.range_constexpr(num_rows):
                row_max[r] = Float32(-1e30)
                row_sum[r] = Float32(0.0)

            # Identity tensor for masking
            mcS = cute.make_identity_tensor((self.tile_size_M, self.n_block))
            tScS = thr_mma.partition_C(mcS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)

            # =============================================================
            # Preload Q into registers (once, before KV loop)
            # =============================================================
            for _qkb in cutlass.range_constexpr(self.D // 16):
                cute.copy(smem_tiled_copy_Q, tQsQ[None, None, _qkb], tQrQ_view[None, None, _qkb])

            # =============================================================
            # KV loop over [kv_start_block, kv_end_block)
            # =============================================================

            # Prologue: cpasync K[kv_start_block]
            gK_block0 = cute.local_tile(gK_head, (self.n_block, self.D), (kv_start_block, Int32(0)))
            tKgK0 = thr_copy.partition_S(gK_block0)
            for ci in cutlass.range_constexpr(cute.size(tKsK_cp.shape[2])):
                cute.copy(gmem_tiled_copy, tKgK0[None, None, ci], tKsK_cp[None, None, ci])
            cute.arch.cp_async_commit_group()

            kv_idx = kv_start_block
            while kv_idx < kv_end_block:
                kv_start = kv_idx * Int32(self.n_block)

                # Wait for K[i]
                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # Start V[i] cpasync
                gV_block = cute.local_tile(gV_head, (self.n_block, self.D), (kv_idx, Int32(0)))
                tVgV = thr_copy.partition_S(gV_block)
                for ci in cutlass.range_constexpr(cute.size(tVsV_cp.shape[2])):
                    cute.copy(gmem_tiled_copy, tVgV[None, None, ci], tVsV_cp[None, None, ci])
                cute.arch.cp_async_commit_group()

                # S GEMM (Q already in registers)
                acc_S.fill(0.0)
                cute.copy(smem_tiled_copy_K, tKsK[None, None, 0], tKrK_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.D // 16):
                    kb_next = (kb + 1) % (self.D // 16)
                    cute.copy(smem_tiled_copy_K, tKsK[None, None, kb_next], tKrK_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_S, tCrQ[None, None, kb], tCrK[None, None, kb], acc_S)

                # Wait for V[i]
                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # Start K[i+1] cpasync (if not last)
                if kv_idx + Int32(1) < kv_end_block:
                    gK_next = cute.local_tile(gK_head, (self.n_block, self.D), (kv_idx + Int32(1), Int32(0)))
                    tKgK_next = thr_copy.partition_S(gK_next)
                    for ci in cutlass.range_constexpr(cute.size(tKsK_cp.shape[2])):
                        cute.copy(gmem_tiled_copy, tKgK_next[None, None, ci], tKsK_cp[None, None, ci])
                    cute.arch.cp_async_commit_group()

                # Masking
                acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
                acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

                # N-boundary mask
                if kv_start + Int32(self.n_block) > Int32(self.N):
                    for r in cutlass.range_constexpr(num_rows):
                        for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                            col_idx = tScS_mn[0, c][1]
                            global_col = kv_start + Int32(col_idx)
                            if global_col >= Int32(self.N):
                                acc_S_mn[r, c] = Float32(-1e30)

                # Causal mask
                if self.causal:
                    last_blk_col = kv_start + Int32(self.n_block - 1)
                    # tile_M=0 always for decode (M is full extent, single tile)
                    first_row = Int32(0)
                    if last_blk_col > first_row + Int32(self.N - self.M):
                        for r in cutlass.range_constexpr(num_rows):
                            row_idx = tScS_mn[r, 0][0]
                            global_row = Int32(row_idx)
                            for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                                col_idx = tScS_mn[0, c][1]
                                global_col = kv_start + Int32(col_idx)
                                if global_col > global_row + Int32(self.N - self.M):
                                    acc_S_mn[r, c] = Float32(-1e30)

                # Online softmax
                _any_correction = Int32(0)
                corrections = cute.make_fragment(cute.make_layout(num_rows), Float32)
                for r in cutlass.range_constexpr(num_rows):
                    acc_S_row = acc_S_mn[r, None].load()
                    row_max_cur = acc_S_row.reduce(cute.ReductionOp.MAX, Float32(-1e30), 0)
                    row_max_cur = self._threadquad_reduce_max(row_max_cur)

                    m_old = row_max[r]
                    m_new = cute.arch.fmax(m_old, row_max_cur)

                    acc_scale_ = (m_old - m_new) * Float32(self.scale_log2e)
                    correction = cute.math.exp2(cute.arch.fmax(acc_scale_, Float32(-126.0)), fastmath=True)
                    if acc_scale_ >= Float32(-self.rescale_threshold):
                        m_new = m_old
                        correction = Float32(1.0)
                    row_sum[r] = row_sum[r] * correction
                    corrections[r] = correction
                    if m_new > m_old:
                        _any_correction = Int32(1)

                    acc_S_row_exp = cute.math.exp2(
                        acc_S_row * Float32(self.scale_log2e) - m_new * Float32(self.scale_log2e), fastmath=True
                    )
                    acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                    row_sum[r] = row_sum[r] + acc_S_row_sum
                    row_max[r] = m_new
                    acc_S_mn[r, None] = acc_S_row_exp

                # Deferred O rescale
                _skip_rescale = cute.arch.vote_all_sync(_any_correction == Int32(0))
                if not _skip_rescale:
                    for r in cutlass.range_constexpr(num_rows):
                        acc_O_mn[r, None] = acc_O_mn[r, None].load() * corrections[r]

                # P conversion + O GEMM
                rP.store(acc_S.load().to(self.q_dtype))
                cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, 0], tVrVt_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.n_block // 16):
                    kb_next = (kb + 1) % (self.n_block // 16)
                    cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, kb_next], tVrVt_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_O, tOrS[None, None, kb], tBrVt[None, None, kb], acc_O)

                kv_idx = kv_idx + Int32(1)

            # =============================================================
            # Write partial O (fp32) and LSE (fp32) to global memory
            # =============================================================
            acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

            # Normalize O by row_sum (keep in fp32)
            for r in cutlass.range_constexpr(num_rows):
                row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
                inv_sum = cute.arch.rcp_approx(row_sum[r])
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * inv_sum

            # Write partial O to global: o_partial[BH, SPLIT, M, D]
            # Each thread writes its owned elements
            lane_in_quad = tidx % Int32(4)
            o_partial_base = (
                o_partial.iterator
                + tile_BH * Int32(self.num_splits * self.M * self.D)
                + tile_SPLIT * Int32(self.M * self.D)
            )
            g_o_partial = cute.make_tensor(
                o_partial_base,
                cute.make_layout((self.tile_size_M, self.D), stride=(self.D, 1)),
            )
            tCgO = thr_mma.partition_C(g_o_partial)
            for i in cutlass.range_constexpr(cute.size(acc_O)):
                tCgO[i] = acc_O[i]

            # Write LSE to global: lse_partial[BH, SPLIT, M]
            # LSE = row_max * scale_val + log(row_sum)
            lse_base = (
                lse_partial.iterator
                + tile_BH * Int32(self.num_splits * self.M)
                + tile_SPLIT * Int32(self.M)
            )
            g_lse = cute.make_tensor(lse_base, cute.make_layout(self.M))
            for r in cutlass.range_constexpr(num_rows):
                if lane_in_quad == Int32(0):
                    row_idx = tScS_mn[r, 0][0]
                    if Int32(row_idx) < Int32(self.M):
                        lse_val = row_max[r] * Float32(self.scale_val) + cute.math.log(row_sum[r])
                        g_lse[Int32(row_idx)] = lse_val


    # =========================================================================
    # Store — fused combine epilogue
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_BH, tile_SPLIT,
              q, k, v, o_partial, lse_partial, o, lse, split_counter):
        """Fused combine epilogue in the store warp.

        After compute writes partials to global, the store warp atomically
        increments a per-head counter. The LAST split's store warp detects
        this and performs the combine reduction across all splits.

        All 32 threads of the store warp participate.
        """
        lane_id = cute.arch.thread_idx()[0] % Int32(32)

        # Atomic increment — thread 0 does the atomic, broadcasts result
        old_count = Int32(0)
        if lane_id == Int32(0):
            # Use raw pointer (Int64) from static_dims for atomic
            counter_base = Int64(self.split_counter_ptr)
            old_count = atomic_add_acq_rel_gpu_i32(counter_base, tile_BH)
        old_count = cute.arch.shuffle_sync(old_count, offset=0, mask=-1, mask_and_clamp=31)

        if (old_count + Int32(1)) % Int32(self.num_splits) == Int32(0):
            # Last split: combine all partials into final O and LSE
            # All 32 threads participate in the reduction

            # Create CuTe tensors for global memory access
            # o_partial: (BH, SPLIT, M, D) fp32
            g_op = cute.make_tensor(
                o_partial.iterator + tile_BH * Int32(self.num_splits * self.M * self.D),
                cute.make_layout((self.num_splits, self.M, self.D),
                                 stride=(self.M * self.D, self.D, 1)),
            )
            # lse_partial: (BH, SPLIT, M) fp32
            g_lp = cute.make_tensor(
                lse_partial.iterator + tile_BH * Int32(self.num_splits * self.M),
                cute.make_layout((self.num_splits, self.M),
                                 stride=(self.M, 1)),
            )
            # o: (BH, M, D) output
            g_o = cute.make_tensor(
                o.iterator + tile_BH * Int32(self.M * self.D),
                cute.make_layout((self.M, self.D), stride=(self.D, 1)),
            )
            # lse: (BH, M) output
            g_lse = cute.make_tensor(
                lse.iterator + tile_BH * Int32(self.M),
                cute.make_layout(self.M),
            )

            # Each thread handles elements in a strided pattern over M*D
            total_elems = Int32(self.M * self.D)
            elem_idx = lane_id
            while elem_idx < total_elems:
                row = elem_idx // Int32(self.D)
                col = elem_idx % Int32(self.D)

                # Find max LSE across splits for this row
                lse_max = Float32(-1e30)
                si = Int32(0)
                while si < Int32(self.num_splits):
                    lse_val = g_lp[si, row]
                    lse_max = cute.arch.fmax(lse_max, lse_val)
                    si = si + Int32(1)

                # Accumulate scaled O and total scale
                acc = Float32(0.0)
                scale_sum = Float32(0.0)
                si = Int32(0)
                while si < Int32(self.num_splits):
                    lse_val = g_lp[si, row]
                    scale = cute.math.exp(lse_val - lse_max)
                    o_val = g_op[si, row, col]
                    acc = acc + scale * o_val
                    scale_sum = scale_sum + scale
                    si = si + Int32(1)

                # Normalize and write output
                inv_scale_sum = cute.arch.rcp_approx(scale_sum)
                result = acc * inv_scale_sum
                g_o[row, col] = result.to(self.q_dtype)

                # Write LSE (only once per row, from the thread handling col=0)
                if col == Int32(0):
                    g_lse[row] = lse_max + cute.math.log(scale_sum)

                elem_idx = elem_idx + Int32(32)


# =============================================================================
# Public API
# =============================================================================


def flash_decoding_schedule(q, k, v, o, num_splits=0,
                            page_size=DEFAULT_PAGE_SIZE,
                            causal=False, kv_group_size=1):
    """Schedule FlashDecoding split-KV with fused combine.

    The combine is fused into the store warp epilogue of the split op,
    eliminating the need for a separate CombineOp.

    Returns:
        (ops, config): List of ScheduledOps and MegakernelConfig.
    """
    import torch

    BH, M, D = q.shape
    lse = torch.empty(BH, M, dtype=torch.float32, device=q.device)

    # Schedule split ops with fused combine
    split_ops = FlashDecodingSplitOp.schedule(
        q=q, k=k, v=v, o=o, lse=lse,
        num_splits=num_splits,
        page_size=page_size, causal=causal, kv_group_size=kv_group_size,
    )

    config = FlashDecodingSplitOp.kernel_config(split_ops)
    return split_ops, config


__all__ = ["FlashDecodingSplitOp", "flash_decoding_schedule"]
