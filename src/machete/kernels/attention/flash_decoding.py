# Copyright (c) 2025, Machete Authors
"""
FlashDecoding: Split-KV Attention for decode workloads.

For small B×H and small M (query length), the standard FA kernel
uses only 1 CTA per head, leaving most SMs idle. FlashDecoding splits the KV
sequence across multiple CTAs (splits), each producing partial O and LSE in
fp32. A separate CombineOp reduces the partials into the final output.

Architecture:
    FlashDecodingSplitOp (B × H × num_splits tiles):
        DMA warp: TMA Q load (single shot).
        MMA warps: cooperative cp.async K/V loads + MMA compute.
        Writes fp32 partials (o_partial, lse_partial) to global.

    FlashDecodingCombineOp (B × H tiles):
        All MMA warps reduce partials into final O (bf16) and LSE (fp32).
        Framework guarantees all SplitOp tiles complete before CombineOp starts.

Usage:
    from machete.kernels.attention.flash_decoding import flash_decoding_schedule
    from machete.megakernel import Megakernel

    ops, config = flash_decoding_schedule(q=q, k=k, v=v, o=o)
    kernel = Megakernel(ops, config=config)
    kernel.run()

Public layout contract:
    - 4D tensors use native `(B, S, H, D)` / KV-cache `(B, N, H_kv, D)`.
"""

import math

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu import warp

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import (
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_arrive_expect_tx,
    mbarrier_wait,
    named_barrier_sync,
)

# Op-managed mbarriers: kblock_ready_K + kblock_ready_V = 2 × 8B
_MBAR_BYTES = 16


class _FlashDecodingSplitTmaOp(Op):
    """Split-KV attention with compute-driven TMA K/V loads.

    Tensors:
        q: (B, H, M, D) — query (fp16/bf16)
        k: (B, H_kv, N, D) — key
        v: (B, H_kv, N, D) — value
        o_partial: (B, H, SPLIT, M, D) — partial output (fp32)
        lse_partial: (B, H, SPLIT, M) — partial log-sum-exp (fp32)

    Tiling:
        tile_B=1, tile_H=1, tile_SPLIT=1 → B*H*num_splits tiles total.

    Each tile computes attention over KV blocks [kv_start, kv_end) and writes
    fp32 partial O and LSE to global memory (no TMA store needed).
    """

    reads = {
        "q": (None, ("B", "H", "M", "D")),
        "k": (None, ("B", "H_kv", "N", "D")),
        "v": (None, ("B", "H_kv", "N", "D")),
    }
    writes = {
        "o_partial": (cutlass.Float32, ("B", "H", "SPLIT", "M", "D")),
        "lse_partial": (cutlass.Float32, ("B", "H", "SPLIT", "M")),
    }
    tile = ("B", "H", "SPLIT")
    dynamic_dims = ("B", "M", "N", "SPLIT")

    # Q/K/V via TMA: Q loaded by DMA warp, K/V by compute warp 0
    tma_loads = {"q", "k", "v"}
    tma_stores = set()  # Partials written to global from compute

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        """Custom TMA tile shapes for K/V (n_block sub-tiling).

        Returns shape in PyTorch dim order (B, H_kv, N, D) with sub-tiling on N.
        """
        if tensor_name in ("k", "v"):
            n_block = static_dims["n_block"]
            D = static_dims["D"]
            return (1, 1, n_block, D)
        return None  # Q uses defaults

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        """Plain smem layout for TMA — no swizzle for multi-row TMA boxes.

        Without swizzle, TMA can use full-width boxes (256B for D=128 bf16)
        × all n_block rows in a single cp.async.bulk.tensor.2d, reducing
        128 TMA ops per K/V load to just 1.
        """
        # Return None → framework uses default plain layout
        return None

    def __init__(self, **config):
        super().__init__(**config)
        self.causal = getattr(self, "causal", 0)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)
        self.kv_group_size = getattr(self, "kv_group_size", 1)

        # 4D: store H for pointer arithmetic
        self.H = getattr(self, 'H', self.tile_size_B * self.tile_size_H)

        assert self.q_dtype in (cutlass.Float16, cutlass.BFloat16), (
            f"FlashDecodingSplitOp requires fp16 or bf16, got {self.q_dtype}"
        )
        self.elem_bytes = 2

        self.scale_val = 1.0 / (self.D ** 0.5)
        self.q_tile_bytes = self.M * self.D * self.elem_bytes

        # num_splits stored as SPLIT dim extent
        self.num_splits = self.SPLIT

        self._init_mma()

    def _init_mma(self):
        """Init compute-driven TMA MMA path."""
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

        # Op-managed mbarriers: kblock_ready_K, kblock_ready_V (2 × 8B = 16B)
        self.mbar_bytes = _MBAR_BYTES

        # n_block: page = Q + 2×KV + mbarriers
        kv_budget = self.page_size - self.q_tile_bytes - self.mbar_bytes
        assert kv_budget > 0, (
            f"FlashDecodingSplitOp: page_size ({self.page_size}B) must be > "
            f"Q tile + mbarriers ({self.q_tile_bytes + self.mbar_bytes}B)."
        )
        max_n_block = kv_budget // (2 * self.D * self.elem_bytes)
        self.n_block = 1 << int(math.log2(max(16, max_n_block)))
        if self.N < self.n_block:
            self.n_block = max(16, (self.N // 16) * 16)
        self.num_kv_blocks = (self.N + self.n_block - 1) // self.n_block

        self.kv_tile_bytes = self.n_block * self.D * self.elem_bytes
        self.mbar_offset = self.q_tile_bytes + 2 * self.kv_tile_bytes

        total_smem = self.q_tile_bytes + 2 * self.kv_tile_bytes + self.mbar_bytes
        assert total_smem <= self.page_size, (
            f"FlashDecodingSplitOp: Q + KV + mbar ({total_smem}B) > page_size ({self.page_size}B)."
        )

        # Compute KV blocks per split
        self.blocks_per_split = (self.num_kv_blocks + self.num_splits - 1) // self.num_splits

        self.inner_iters = 1
        self.inner_depth = 1

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
        """Schedule split-KV forward pass (partials only, no combine)."""
        import torch

        q = tensors["q"]
        k = tensors["k"]

        assert q.ndim == 4, f"Expected 4D q tensor, got shape={tuple(q.shape)}"
        assert k.ndim == 4, f"Expected 4D k tensor, got shape={tuple(k.shape)}"
        assert tensors["v"].ndim == 4, f"Expected 4D v tensor, got shape={tuple(tensors['v'].shape)}"

        B, H, M, D = q.shape
        N = k.shape[2]

        assert q.element_size() == 2

        # Compute-TMA path needs op-managed mbarrier space. The public split op
        # is the cp.async BSHD path and does not.
        effective_page_size = page_size + _MBAR_BYTES if cls is _FlashDecodingSplitTmaOp else page_size

        elem = q.element_size()
        q_tile_bytes = M * D * elem
        kv_budget = effective_page_size - q_tile_bytes - _MBAR_BYTES
        max_n_block = kv_budget // (2 * D * elem)
        n_block = 1 << int(math.log2(max(16, max_n_block)))
        if N < n_block:
            n_block = max(16, (N // 16) * 16)
        num_n_blocks = (N + n_block - 1) // n_block

        # Short decode contexts are combiner dominated. Long contexts benefit
        # from smaller KV chunks, but oversplitting can leave numerically empty
        # or too-small split work and has shown nonfinite outputs in decode.
        if D >= 256 and N <= 4096:
            min_blocks_per_split = 16
        elif N < 1024:
            min_blocks_per_split = 4
        else:
            min_blocks_per_split = 8
        max_splits = max(1, num_n_blocks // min_blocks_per_split)
        max_requested_splits = num_n_blocks if N < 1024 else max_splits

        # Auto num_splits
        if num_splits <= 0:
            num_SMs = torch.cuda.get_device_properties(q.device).multi_processor_count
            total_mblocks = B * H  # 1 M-tile per head for decode
            num_splits = min(num_SMs // max(total_mblocks, 1), max_splits)
        else:
            num_splits = min(num_splits, max_requested_splits)
        num_splits = max(num_splits, 1)

        # Allocate intermediate buffers
        o_partial = torch.empty(B, H, num_splits, M, D, dtype=torch.float32, device=q.device)
        lse_partial = torch.empty(B, H, num_splits, M, dtype=torch.float32, device=q.device)

        tile_sizes = dict(tile_sizes or {})
        tile_sizes["B"] = 1
        tile_sizes["H"] = 1
        tile_sizes["SPLIT"] = 1

        tensors["o_partial"] = o_partial
        tensors["lse_partial"] = lse_partial

        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]

        # Compute n_block for static_dims (must match _init_mma)
        elem = q.element_size()
        _q_tile = M * D * elem
        _kv_budget = effective_page_size - _q_tile - _MBAR_BYTES
        if _kv_budget > 0:
            _max_n = _kv_budget // (2 * D * elem)
            _n_block = 1 << int(math.log2(max(16, _max_n)))
            if N < _n_block:
                _n_block = max(16, (N // 16) * 16)
            ops[0].static_dims["n_block"] = _n_block

        ops[0].static_dims["page_size"] = effective_page_size
        ops[0].static_dims["num_splits"] = num_splits
        ops[0].static_dims["M"] = M
        ops[0].static_dims["N"] = N
        ops[0].static_dims["H"] = H
        ops[0].static_dims["SPLIT"] = num_splits
        if causal:
            ops[0].static_dims["causal"] = 1
        if kv_group_size > 1:
            ops[0].static_dims["kv_group_size"] = kv_group_size

        return ops, o_partial, lse_partial

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for flash decoding ops."""
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS
        import torch

        num_mma_warps = max(
            1,
            max(op.static_dims.get("M", 16) // 16 for op in ops),
        )
        threads_per_block = (num_mma_warps + NUM_DMA_WARPS) * 32
        page_size = ops[0].static_dims.get("page_size", DEFAULT_PAGE_SIZE)

        total_tiles = sum(op.total_tiles for op in ops)
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        num_sms = min(num_sms, total_tiles)

        return MegakernelConfig(
            threads_per_block=threads_per_block,
            page_size=page_size,
            noinline=True,
            num_sms=num_sms,
        )

    # =========================================================================
    # Load (TMA Q + init mbarriers)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_H, tile_SPLIT, q_tma, q_tma_gmem, work_mbar):
        """TMA Q load + init op-managed mbarriers for compute-driven K/V TMA.

        Mbarriers (initialized here, used by compute warp 0):
          kblock_ready_K (offset +0): signals K data arrived in smem
          kblock_ready_V (offset +8): signals V data arrived in smem
        """
        # Init mbarriers for compute-driven K/V TMA
        _kr_K = page_ptr + Int32(self.mbar_offset)
        _kr_V = page_ptr + Int32(self.mbar_offset + 8)
        with cute.arch.elect_one():
            mbarrier_init(_kr_K, Int32(1))
            mbarrier_init(_kr_V, Int32(1))
        mbarrier_init_fence()

        # TMA Q → page start (plain layout, no swizzle for multi-row TMA)
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        sQ = cute.make_tensor(
            cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.D, self.tile_size_M, 1, 1)),
        )
        gQ = cute.local_tile(
            q_tma_gmem,
            (self.D, self.tile_size_M, 1, 1),
            (None, None, None, None),
        )
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            q_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sQ, 0, 4),
            cute.group_modes(gQ, 0, 4),
        )
        nbytes = Int32(self.q_tile_bytes)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(q_tma, tQgQ[(None, Int32(0), Int32(0), tile_H, tile_B)], tQsQ, tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # MMA Helpers
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
    # Compute — compute-driven TMA split-KV flash attention
    # =========================================================================

    @cute.jit
    def compute_mma(self, page_ptr, tile_B, tile_H, tile_SPLIT,
                    q, k, v, o_partial, lse_partial,
                    k_tma, k_tma_gmem, v_tma, v_tma_gmem):
        """Compute-driven TMA split-KV flash attention.

        Same pipeline as FlashAttentionSm120Op but processes only KV blocks
        [kv_start, kv_end) and writes fp32 partials to global memory.
        """
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        # GQA: map Q head index to KV head index
        kv_h = tile_H // Int32(self.kv_group_size)

        # === TMA partition setup (must be at function top level — ===
        # === TMA atoms cannot cross MLIR SCF region boundaries)  ===

        _k_base = page_ptr + Int32(self.q_tile_bytes)
        _v_base = page_ptr + Int32(self.q_tile_bytes + self.kv_tile_bytes)

        sK_tma = cute.make_tensor(
            cute.make_ptr(self.q_dtype, _k_base, cute.AddressSpace.smem),
            cute.make_layout(
                (self.D, self.n_block, 1, 1),
                stride=(1, self.D, self.D * self.n_block, self.D * self.n_block)),
        )
        gK_tma = cute.local_tile(
            k_tma_gmem, (self.D, self.n_block, 1, 1), (None, None, None, None),
        )
        tKsK_tma, tKgK_tma = cute.nvgpu.cpasync.tma_partition(
            k_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sK_tma, 0, 4),
            cute.group_modes(gK_tma, 0, 4),
        )

        sV_tma = cute.make_tensor(
            cute.make_ptr(self.q_dtype, _v_base, cute.AddressSpace.smem),
            cute.make_layout(
                (self.D, self.n_block, 1, 1),
                stride=(1, self.D, self.D * self.n_block, self.D * self.n_block)),
        )
        gV_tma = cute.local_tile(
            v_tma_gmem, (self.D, self.n_block, 1, 1), (None, None, None, None),
        )
        tVsV_tma, tVgV_tma = cute.nvgpu.cpasync.tma_partition(
            v_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sV_tma, 0, 4),
            cute.group_modes(gV_tma, 0, 4),
        )

        # Op-managed mbarrier pointers
        _kr_K = page_ptr + Int32(self.mbar_offset)
        _kr_V = page_ptr + Int32(self.mbar_offset + 8)
        _kr_K_ptr = cute.make_ptr(cutlass.Int64, _kr_K, cute.AddressSpace.smem)
        _kr_V_ptr = cute.make_ptr(cutlass.Int64, _kr_V, cute.AddressSpace.smem)
        nbytes_kv = Int32(self.kv_tile_bytes)

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

            # === Q smem tensor (persistent, plain layout) ===
            sQ = cute.make_tensor(
                cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
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

            # === K smem tensor + LdMatrix fragments (plain layout) ===
            _sK = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _k_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.n_block, self.D), stride=(self.D, 1)),
            )
            _tCsK = thr_mma.partition_B(_sK)
            tCrK = tiled_mma.make_fragment_B(_tCsK)
            tKrK_view = smem_thr_copy_K.retile(tCrK)
            tKsK = smem_thr_copy_K.partition_S(_sK)

            # === V smem tensor + LdMatrix fragments (plain layout, transposed) ===
            _sVt = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _v_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.D, self.n_block), stride=(1, self.D)),
            )
            _tBsVt = thr_mma.partition_B(_sVt)
            tBrVt = tiled_mma.make_fragment_B(_tBsVt)
            tVrVt_view = smem_thr_copy_Vt.retile(tBrVt)
            tVsVt = smem_thr_copy_Vt.partition_S(_sVt)

            # P register fragment + MMA view
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
            # Preload Q into registers
            # =============================================================
            for _qkb in cutlass.range_constexpr(self.D // 16):
                cute.copy(smem_tiled_copy_Q, tQsQ[None, None, _qkb], tQrQ_view[None, None, _qkb])

            # =============================================================
            # Pipelined KV loop with compute-driven TMA
            # =============================================================

            # Prologue: issue TMA K[kv_start_block] from warp 0
            kv_idx = kv_start_block
            kv_phase = Int32(0)
            if warp_idx == Int32(0):
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(_kr_K, nbytes_kv)
                cute.copy(k_tma,
                          tKgK_tma[(None, Int32(0), kv_idx, kv_h, tile_B)],
                          tKsK_tma, tma_bar_ptr=_kr_K_ptr)

            while kv_idx < kv_end_block:
                kv_start = kv_idx * Int32(self.n_block)

                # --- Wait for K[i] (TMA from warp 0) ---
                mbarrier_wait(_kr_K, kv_phase)

                # --- Issue TMA V[i] from warp 0 (overlap with S GEMM) ---
                if warp_idx == Int32(0):
                    with cute.arch.elect_one():
                        mbarrier_arrive_expect_tx(_kr_V, nbytes_kv)
                    cute.copy(v_tma,
                              tVgV_tma[(None, Int32(0), kv_idx, kv_h, tile_B)],
                              tVsV_tma, tma_bar_ptr=_kr_V_ptr)

                # --- S GEMM: Q_regs × K_smem ---
                acc_S.fill(0.0)
                cute.copy(smem_tiled_copy_K, tKsK[None, None, 0], tKrK_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.D // 16):
                    kb_next = (kb + 1) % (self.D // 16)
                    cute.copy(smem_tiled_copy_K, tKsK[None, None, kb_next], tKrK_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_S, tCrQ[None, None, kb], tCrK[None, None, kb], acc_S)

                # --- Sync: all warps done reading K, buffer safe for reuse ---
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # --- Issue TMA K[i+1] from warp 0 (overlap with V processing) ---
                if kv_idx + Int32(1) < kv_end_block:
                    if warp_idx == Int32(0):
                        with cute.arch.elect_one():
                            mbarrier_arrive_expect_tx(_kr_K, nbytes_kv)
                        cute.copy(k_tma,
                                  tKgK_tma[(None, Int32(0), kv_idx + Int32(1), kv_h, tile_B)],
                                  tKsK_tma, tma_bar_ptr=_kr_K_ptr)

                # --- Wait for V[i] (TMA from warp 0) ---
                mbarrier_wait(_kr_V, kv_phase)

                # --- Masking ---
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
                    first_row = Int32(0)  # tile_M=0 always for decode
                    if last_blk_col > first_row + Int32(self.N - self.M):
                        for r in cutlass.range_constexpr(num_rows):
                            row_idx = tScS_mn[r, 0][0]
                            global_row = Int32(row_idx)
                            for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                                col_idx = tScS_mn[0, c][1]
                                global_col = kv_start + Int32(col_idx)
                                if global_col > global_row + Int32(self.N - self.M):
                                    acc_S_mn[r, c] = Float32(-1e30)

                # --- Online softmax ---
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

                # --- P conversion + O GEMM ---
                rP.store(acc_S.load().to(self.q_dtype))
                cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, 0], tVrVt_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.n_block // 16):
                    kb_next = (kb + 1) % (self.n_block // 16)
                    cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, kb_next], tVrVt_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_O, tOrS[None, None, kb], tBrVt[None, None, kb], acc_O)

                # --- Sync: all warps done reading V, buffer safe for reuse ---
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                kv_idx = kv_idx + Int32(1)
                kv_phase = Int32(1) - kv_phase

            # =============================================================
            # Write partial O (fp32) and LSE (fp32) to global memory
            # =============================================================
            acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

            # Normalize O by row_sum (keep in fp32)
            for r in cutlass.range_constexpr(num_rows):
                row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
                inv_sum = cute.arch.rcp_approx(row_sum[r])
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * inv_sum

            # Write partial O to global: o_partial[B, H, SPLIT, M, D]
            lane_in_quad = tidx % Int32(4)
            o_partial_base = (
                o_partial.iterator
                + tile_B * Int32(self.H * self.num_splits * self.M * self.D)
                + tile_H * Int32(self.num_splits * self.M * self.D)
                + tile_SPLIT * Int32(self.M * self.D)
            )
            g_o_partial = cute.make_tensor(
                o_partial_base,
                cute.make_layout((self.tile_size_M, self.D), stride=(self.D, 1)),
            )
            tCgO = thr_mma.partition_C(g_o_partial)
            for i in cutlass.range_constexpr(cute.size(acc_O)):
                tCgO[i] = acc_O[i]

            # Write LSE to global: lse_partial[B, H, SPLIT, M]
            lse_base = (
                lse_partial.iterator
                + tile_B * Int32(self.H * self.num_splits * self.M)
                + tile_H * Int32(self.num_splits * self.M)
                + tile_SPLIT * Int32(self.M)
            )
            g_lse = cute.make_tensor(lse_base, cute.make_layout(self.M))
            for r in cutlass.range_constexpr(num_rows):
                if lane_in_quad == Int32(0):
                    row_idx = tScS_mn[r, 0][0]
                    if Int32(row_idx) < Int32(self.M):
                        lse_val = row_max[r] * Float32(self.scale_val) + cute.math.log(row_sum[r])
                        g_lse[Int32(row_idx)] = lse_val


class _FlashDecodingSplitCpAsyncBase(_FlashDecodingSplitTmaOp):
    """Split-KV decode attention with cp.async K/V loading in compute.

    This variant keeps Q as a framework-managed TMA load, but removes K/V TMA
    use from the compute phase. K/V are loaded cooperatively from global memory
    via cp.async, mirroring the main SM120 attention kernels.
    """

    tma_loads = {"q"}

    def _init_mma(self):
        """Init cp.async-driven MMA path."""
        self.tile_size_M = self.M
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

        # Plain row-major K/V smem. This keeps the output math path identical to
        # the existing split op while changing only the transport path.
        self.smem_stride = self.D

        self.async_copy_elems = 128 // (self.elem_bytes * 8)
        self.copy_dim1 = self.D // self.async_copy_elems
        self.copy_dim0 = self.num_mma_threads // self.copy_dim1

        kv_budget = self.page_size - self.q_tile_bytes
        assert kv_budget > 0, (
            f"FlashDecodingSplitOp: page_size ({self.page_size}B) must be > "
            f"Q tile ({self.q_tile_bytes}B)."
        )
        max_n_block = kv_budget // (2 * self.smem_stride * self.elem_bytes)
        self.n_block = 1 << int(math.log2(max(16, max_n_block)))
        if self.N < self.n_block:
            self.n_block = max(16, (self.N // 16) * 16)

        self.num_kv_blocks = (self.N + self.n_block - 1) // self.n_block
        self.kv_tile_bytes = self.n_block * self.smem_stride * self.elem_bytes
        total_smem = self.q_tile_bytes + 2 * self.kv_tile_bytes
        assert total_smem <= self.page_size, (
            f"FlashDecodingSplitOp: Q + KV ({total_smem}B) > page_size ({self.page_size}B)."
        )

        self.blocks_per_split = (self.num_kv_blocks + self.num_splits - 1) // self.num_splits
        self.inner_iters = 1
        self.inner_depth = 1
        self.scale_log2e = self.scale_val * 1.4426950408889634074
        self.rescale_threshold = 8.0
        self.compute = self.compute_mma

    @cute.jit
    def load(self, page_ptr, tile_B, tile_H, tile_SPLIT, q_tma, q_tma_gmem, work_mbar):
        """TMA Q load only. cp.async K/V are issued in compute."""
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        sQ = cute.make_tensor(
            cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.D, self.tile_size_M, 1, 1)),
        )
        gQ = cute.local_tile(
            q_tma_gmem,
            (self.D, self.tile_size_M, 1, 1),
            (None, None, None, None),
        )
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            q_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sQ, 0, 4),
            cute.group_modes(gQ, 0, 4),
        )
        nbytes = Int32(self.q_tile_bytes)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(
            q_tma,
            tQgQ[(None, Int32(0), Int32(0), tile_H, tile_B)],
            tQsQ,
            tma_bar_ptr=mbar_ptr,
        )

    @cute.jit
    def compute_mma(self, page_ptr, tile_B, tile_H, tile_SPLIT,
                    q, k, v, o_partial, lse_partial):
        """Split-KV decode attention with cp.async K/V loading."""
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        kv_h = tile_H // Int32(self.kv_group_size)
        _k_base = page_ptr + Int32(self.q_tile_bytes)
        _v_base = page_ptr + Int32(self.q_tile_bytes + self.kv_tile_bytes)

        if warp_idx < Int32(self.num_mma_warps):
            kv_start_block = tile_SPLIT * Int32(self.blocks_per_split)
            kv_end_block = (tile_SPLIT + Int32(1)) * Int32(self.blocks_per_split)
            if kv_end_block > Int32(self.num_kv_blocks):
                kv_end_block = Int32(self.num_kv_blocks)

            mma_op = warp.MmaF16BF16Op(self.q_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((self.num_mma_warps, 1, 1)),
                permutation_mnk=(self.num_mma_warps * 16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)

            sQ = cute.make_tensor(
                cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
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

            _sK = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _k_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.n_block, self.D), stride=(self.smem_stride, 1)),
            )
            _tCsK = thr_mma.partition_B(_sK)
            tCrK = tiled_mma.make_fragment_B(_tCsK)
            tKrK_view = smem_thr_copy_K.retile(tCrK)
            tKsK = smem_thr_copy_K.partition_S(_sK)

            _sVt = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _v_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.D, self.n_block), stride=(1, self.smem_stride)),
            )
            _tBsVt = thr_mma.partition_B(_sVt)
            tBrVt = tiled_mma.make_fragment_B(_tBsVt)
            tVrVt_view = smem_thr_copy_Vt.retile(tBrVt)
            tVsVt = smem_thr_copy_Vt.partition_S(_sVt)

            async_copy_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), self.q_dtype, num_bits_per_copy=128
            )
            copy_thread_layout = cute.make_layout(
                (self.copy_dim0, self.copy_dim1), stride=(self.copy_dim1, 1)
            )
            copy_value_layout = cute.make_layout((1, self.async_copy_elems))
            gmem_tiled_copy = cute.make_tiled_copy_tv(
                async_copy_atom, copy_thread_layout, copy_value_layout
            )
            thr_copy = gmem_tiled_copy.get_slice(tidx)

            sK_cp = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _k_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.n_block, self.D), stride=(self.smem_stride, 1)),
            )
            sV_cp = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _v_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.n_block, self.D), stride=(self.smem_stride, 1)),
            )
            tKsK_cp = thr_copy.partition_D(sK_cp)
            tVsV_cp = thr_copy.partition_D(sV_cp)

            k_head_ptr = (
                k.iterator
                + tile_B * Int32(self.H_kv * self.N * self.D)
                + kv_h * Int32(self.N * self.D)
            ).align(16)
            v_head_ptr = (
                v.iterator
                + tile_B * Int32(self.H_kv * self.N * self.D)
                + kv_h * Int32(self.N * self.D)
            ).align(16)
            gK_head = cute.make_tensor(
                k_head_ptr,
                cute.make_layout((self.N, self.D), stride=(self.D, 1)),
            )
            gV_head = cute.make_tensor(
                v_head_ptr,
                cute.make_layout((self.N, self.D), stride=(self.D, 1)),
            )

            acc_S = cute.make_fragment(
                tiled_mma.partition_shape_C((self.tile_size_M, self.n_block)), Float32
            )
            rP = cute.make_fragment_like(acc_S, self.q_dtype)
            rP_ld = cute.logical_divide(rP.layout, (None, None, 2))
            rP_mma_view = cute.make_layout(
                ((rP_ld.shape[0], rP_ld.shape[2][0]), rP_ld.shape[1], rP_ld.shape[2][1]),
                stride=((rP_ld.stride[0], rP_ld.stride[2][0]), rP_ld.stride[1], rP_ld.stride[2][1]),
            )
            tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

            acc_O = cute.make_fragment(
                tiled_mma.partition_shape_C((self.tile_size_M, self.D)), Float32
            )
            acc_O.fill(0.0)

            acc_O_shape = tiled_mma.partition_shape_C((self.tile_size_M, self.D))
            num_rows = acc_O_shape[0][1] * acc_O_shape[1]
            row_max = cute.make_fragment(cute.make_layout(num_rows), Float32)
            row_sum = cute.make_fragment(cute.make_layout(num_rows), Float32)
            for r in cutlass.range_constexpr(num_rows):
                row_max[r] = Float32(-1e30)
                row_sum[r] = Float32(0.0)

            mcS = cute.make_identity_tensor((self.tile_size_M, self.n_block))
            tScS = thr_mma.partition_C(mcS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)

            for _qkb in cutlass.range_constexpr(self.D // 16):
                cute.copy(smem_tiled_copy_Q, tQsQ[None, None, _qkb], tQrQ_view[None, None, _qkb])

            full_kv_end_block = kv_end_block
            if Int32(self.N % self.n_block) != Int32(0) and kv_end_block == Int32(self.num_kv_blocks):
                full_kv_end_block = kv_end_block - Int32(1)

            kv_idx = kv_start_block
            if kv_idx < full_kv_end_block:
                gK_block0 = cute.local_tile(gK_head, (self.n_block, self.D), (kv_start_block, Int32(0)))
                tKgK0 = thr_copy.partition_S(gK_block0)
                for ci in cutlass.range_constexpr(cute.size(tKsK_cp.shape[2])):
                    cute.copy(gmem_tiled_copy, tKgK0[None, None, ci], tKsK_cp[None, None, ci])
                cute.arch.cp_async_commit_group()

            while kv_idx < full_kv_end_block:
                kv_start = kv_idx * Int32(self.n_block)

                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                gV_block = cute.local_tile(gV_head, (self.n_block, self.D), (kv_idx, Int32(0)))
                tVgV = thr_copy.partition_S(gV_block)
                for ci in cutlass.range_constexpr(cute.size(tVsV_cp.shape[2])):
                    cute.copy(gmem_tiled_copy, tVgV[None, None, ci], tVsV_cp[None, None, ci])
                cute.arch.cp_async_commit_group()

                acc_S.fill(0.0)
                cute.copy(smem_tiled_copy_K, tKsK[None, None, 0], tKrK_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.D // 16):
                    kb_next = (kb + 1) % (self.D // 16)
                    cute.copy(smem_tiled_copy_K, tKsK[None, None, kb_next], tKrK_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_S, tCrQ[None, None, kb], tCrK[None, None, kb], acc_S)

                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                if kv_idx + Int32(1) < full_kv_end_block:
                    gK_next = cute.local_tile(
                        gK_head, (self.n_block, self.D), (kv_idx + Int32(1), Int32(0))
                    )
                    tKgK_next = thr_copy.partition_S(gK_next)
                    for ci in cutlass.range_constexpr(cute.size(tKsK_cp.shape[2])):
                        cute.copy(gmem_tiled_copy, tKgK_next[None, None, ci], tKsK_cp[None, None, ci])
                cute.arch.cp_async_commit_group()

                cute.arch.cp_async_wait_group(1)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
                acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

                if kv_start + Int32(self.n_block) > Int32(self.N):
                    for r in cutlass.range_constexpr(num_rows):
                        for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                            col_idx = tScS_mn[0, c][1]
                            global_col = kv_start + Int32(col_idx)
                            if global_col >= Int32(self.N):
                                acc_S_mn[r, c] = Float32(-1e30)

                if self.causal:
                    last_blk_col = kv_start + Int32(self.n_block - 1)
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

                _any_correction = Int32(0)
                corrections = cute.make_fragment(cute.make_layout(num_rows), Float32)
                for r in cutlass.range_constexpr(num_rows):
                    acc_S_row = acc_S_mn[r, None].load()
                    row_max_cur = acc_S_row.reduce(cute.ReductionOp.MAX, Float32(-1e30), 0)
                    row_max_cur = self._threadquad_reduce_max(row_max_cur)

                    m_old = row_max[r]
                    m_new = cute.arch.fmax(m_old, row_max_cur)

                    acc_scale_ = (m_old - m_new) * Float32(self.scale_log2e)
                    correction = cute.math.exp2(
                        cute.arch.fmax(acc_scale_, Float32(-126.0)), fastmath=True
                    )
                    if acc_scale_ >= Float32(-self.rescale_threshold):
                        m_new = m_old
                        correction = Float32(1.0)
                    row_sum[r] = row_sum[r] * correction
                    corrections[r] = correction
                    if m_new > m_old:
                        _any_correction = Int32(1)

                    acc_S_row_exp = cute.math.exp2(
                        acc_S_row * Float32(self.scale_log2e)
                        - m_new * Float32(self.scale_log2e),
                        fastmath=True,
                    )
                    acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                    row_sum[r] = row_sum[r] + acc_S_row_sum
                    row_max[r] = m_new
                    acc_S_mn[r, None] = acc_S_row_exp

                _skip_rescale = cute.arch.vote_all_sync(_any_correction == Int32(0))
                if not _skip_rescale:
                    for r in cutlass.range_constexpr(num_rows):
                        acc_O_mn[r, None] = acc_O_mn[r, None].load() * corrections[r]

                rP.store(acc_S.load().to(self.q_dtype))
                cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, 0], tVrVt_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.n_block // 16):
                    kb_next = (kb + 1) % (self.n_block // 16)
                    cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, kb_next], tVrVt_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_O, tOrS[None, None, kb], tBrVt[None, None, kb], acc_O)

                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                kv_idx = kv_idx + Int32(1)

            if full_kv_end_block < kv_end_block:
                kv_idx = full_kv_end_block
                kv_start = kv_idx * Int32(self.n_block)

                elem_idx = Int32(tidx)
                total_kv_elems = Int32(self.n_block * self.D)
                while elem_idx < total_kv_elems:
                    row = elem_idx // Int32(self.D)
                    col = elem_idx % Int32(self.D)
                    global_row = kv_start + row
                    if global_row < Int32(self.N):
                        sK_cp[row, col] = gK_head[global_row, col]
                        sV_cp[row, col] = gV_head[global_row, col]
                    else:
                        zero = Float32(0.0).to(self.q_dtype)
                        sK_cp[row, col] = zero
                        sV_cp[row, col] = zero
                    elem_idx = elem_idx + Int32(self.num_mma_threads)

                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                acc_S.fill(0.0)
                cute.copy(smem_tiled_copy_K, tKsK[None, None, 0], tKrK_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.D // 16):
                    kb_next = (kb + 1) % (self.D // 16)
                    cute.copy(smem_tiled_copy_K, tKsK[None, None, kb_next], tKrK_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_S, tCrQ[None, None, kb], tCrK[None, None, kb], acc_S)

                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
                acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

                for r in cutlass.range_constexpr(num_rows):
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        col_idx = tScS_mn[0, c][1]
                        global_col = kv_start + Int32(col_idx)
                        if global_col >= Int32(self.N):
                            acc_S_mn[r, c] = Float32(-1e30)

                if self.causal:
                    last_blk_col = kv_start + Int32(self.n_block - 1)
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

                _any_correction = Int32(0)
                corrections = cute.make_fragment(cute.make_layout(num_rows), Float32)
                for r in cutlass.range_constexpr(num_rows):
                    acc_S_row = acc_S_mn[r, None].load()
                    row_max_cur = acc_S_row.reduce(cute.ReductionOp.MAX, Float32(-1e30), 0)
                    row_max_cur = self._threadquad_reduce_max(row_max_cur)

                    m_old = row_max[r]
                    m_new = cute.arch.fmax(m_old, row_max_cur)

                    acc_scale_ = (m_old - m_new) * Float32(self.scale_log2e)
                    correction = cute.math.exp2(
                        cute.arch.fmax(acc_scale_, Float32(-126.0)), fastmath=True
                    )
                    if acc_scale_ >= Float32(-self.rescale_threshold):
                        m_new = m_old
                        correction = Float32(1.0)
                    row_sum[r] = row_sum[r] * correction
                    corrections[r] = correction
                    if m_new > m_old:
                        _any_correction = Int32(1)

                    acc_S_row_exp = cute.math.exp2(
                        acc_S_row * Float32(self.scale_log2e)
                        - m_new * Float32(self.scale_log2e),
                        fastmath=True,
                    )
                    acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                    row_sum[r] = row_sum[r] + acc_S_row_sum
                    row_max[r] = m_new
                    acc_S_mn[r, None] = acc_S_row_exp

                _skip_rescale = cute.arch.vote_all_sync(_any_correction == Int32(0))
                if not _skip_rescale:
                    for r in cutlass.range_constexpr(num_rows):
                        acc_O_mn[r, None] = acc_O_mn[r, None].load() * corrections[r]

                rP.store(acc_S.load().to(self.q_dtype))
                cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, 0], tVrVt_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.n_block // 16):
                    kb_next = (kb + 1) % (self.n_block // 16)
                    cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, kb_next], tVrVt_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_O, tOrS[None, None, kb], tBrVt[None, None, kb], acc_O)

            acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
            for r in cutlass.range_constexpr(num_rows):
                row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
                inv_sum = cute.arch.rcp_approx(row_sum[r])
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * inv_sum

            lane_in_quad = tidx % Int32(4)
            o_partial_base = (
                o_partial.iterator
                + tile_B * Int32(self.H * self.num_splits * self.M * self.D)
                + tile_H * Int32(self.num_splits * self.M * self.D)
                + tile_SPLIT * Int32(self.M * self.D)
            )
            g_o_partial = cute.make_tensor(
                o_partial_base,
                cute.make_layout((self.tile_size_M, self.D), stride=(self.D, 1)),
            )
            tCgO = thr_mma.partition_C(g_o_partial)
            for i in cutlass.range_constexpr(cute.size(acc_O)):
                tCgO[i] = acc_O[i]

            lse_base = (
                lse_partial.iterator
                + tile_B * Int32(self.H * self.num_splits * self.M)
                + tile_H * Int32(self.num_splits * self.M)
                + tile_SPLIT * Int32(self.M)
            )
            g_lse = cute.make_tensor(lse_base, cute.make_layout(self.M))
            for r in cutlass.range_constexpr(num_rows):
                if lane_in_quad == Int32(0):
                    row_idx = tScS_mn[r, 0][0]
                    if Int32(row_idx) < Int32(self.M):
                        lse_val = row_max[r] * Float32(self.scale_val) + cute.math.log(row_sum[r])
                        g_lse[Int32(row_idx)] = lse_val


class FlashDecodingSplitBSHDOp(_FlashDecodingSplitCpAsyncBase):
    """Native BSHD split-KV decode attention.

    Tensors:
        q: (B, M, H, D)
        k/v: (B, N, H_kv, D), including regular KV-cache tensors
    """

    reads = {
        "q": (None, ("B", "M", "H", "D")),
        "k": (None, ("B", "N", "H_kv", "D")),
        "v": (None, ("B", "N", "H_kv", "D")),
    }
    writes = {
        "o_partial": (cutlass.Float32, ("B", "H", "SPLIT", "M", "D")),
        "lse_partial": (cutlass.Float32, ("B", "H", "SPLIT", "M")),
    }
    tile = ("B", "H", "SPLIT")
    dynamic_dims = ("B", "M", "N", "SPLIT")
    tma_loads = {"q"}

    @classmethod
    def schedule(cls, tile_sizes=None, causal=False,
                         page_size=DEFAULT_PAGE_SIZE, kv_group_size=1,
                         num_splits=0, **tensors):
        """Schedule native BSHD split-KV forward pass."""
        import torch

        q = tensors["q"]
        k = tensors["k"]
        assert q.ndim == 4, f"Expected BSHD q, got shape={tuple(q.shape)}"
        assert k.ndim == 4, f"Expected BSHD KV cache k, got shape={tuple(k.shape)}"
        assert tensors["v"].ndim == 4, f"Expected BSHD KV cache v, got shape={tuple(tensors['v'].shape)}"

        B, M, H, D = q.shape
        N = k.shape[1]
        assert k.shape[0] == B and k.shape[3] == D
        assert tensors["v"].shape == k.shape
        assert q.element_size() == 2

        effective_page_size = page_size

        elem = q.element_size()
        q_tile_bytes = M * D * elem
        kv_budget = effective_page_size - q_tile_bytes
        max_n_block = kv_budget // (2 * D * elem)
        n_block = 1 << int(math.log2(max(16, max_n_block)))
        if N < n_block:
            n_block = max(16, (N // 16) * 16)
        num_n_blocks = (N + n_block - 1) // n_block

        if D >= 256 and N <= 4096:
            min_blocks_per_split = 16
        elif N < 1024:
            min_blocks_per_split = 4
        else:
            min_blocks_per_split = 8
        max_splits = max(1, num_n_blocks // min_blocks_per_split)
        max_requested_splits = num_n_blocks if N < 1024 else max_splits

        if num_splits <= 0:
            num_SMs = torch.cuda.get_device_properties(q.device).multi_processor_count
            total_mblocks = B * H
            num_splits = min(num_SMs // max(total_mblocks, 1), max_splits)
        else:
            num_splits = min(num_splits, max_requested_splits)
        num_splits = max(num_splits, 1)

        o_partial = torch.empty(B, H, num_splits, M, D, dtype=torch.float32, device=q.device)
        lse_partial = torch.empty(B, H, num_splits, M, dtype=torch.float32, device=q.device)

        tile_sizes = dict(tile_sizes or {})
        tile_sizes["B"] = 1
        tile_sizes["H"] = 1
        tile_sizes["SPLIT"] = 1

        tensors["o_partial"] = o_partial
        tensors["lse_partial"] = lse_partial

        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]

        elem = q.element_size()
        _q_tile = M * D * elem
        _kv_budget = effective_page_size - _q_tile
        if _kv_budget > 0:
            _max_n = _kv_budget // (2 * D * elem)
            _n_block = 1 << int(math.log2(max(16, _max_n)))
            if N < _n_block:
                _n_block = max(16, (N // 16) * 16)
            ops[0].static_dims["n_block"] = _n_block

        ops[0].static_dims["page_size"] = effective_page_size
        ops[0].static_dims["num_splits"] = num_splits
        ops[0].static_dims["M"] = M
        ops[0].static_dims["N"] = N
        ops[0].static_dims["H"] = H
        ops[0].static_dims["SPLIT"] = num_splits
        ops[0].static_dims["k_b_stride"] = k.stride(0)
        ops[0].static_dims["k_n_stride"] = k.stride(1)
        ops[0].static_dims["k_h_stride"] = k.stride(2)
        ops[0].static_dims["v_b_stride"] = tensors["v"].stride(0)
        ops[0].static_dims["v_n_stride"] = tensors["v"].stride(1)
        ops[0].static_dims["v_h_stride"] = tensors["v"].stride(2)
        if causal:
            ops[0].static_dims["causal"] = 1
        if kv_group_size > 1:
            ops[0].static_dims["kv_group_size"] = kv_group_size

        return ops, o_partial, lse_partial

    @cute.jit
    def load(self, page_ptr, tile_B, tile_H, tile_SPLIT, q_tma, q_tma_gmem, work_mbar):
        """TMA Q load for native BSHD tensors.

        BSHD contiguous tensors are TMA stride-sorted as (D, H, M, B), so the
        head coordinate is the second TMA coordinate.
        """
        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        sQ = cute.make_tensor(
            cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.D, 1, self.tile_size_M, 1)),
        )
        gQ = cute.local_tile(
            q_tma_gmem,
            (self.D, 1, self.tile_size_M, 1),
            (None, None, None, None),
        )
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            q_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sQ, 0, 4),
            cute.group_modes(gQ, 0, 4),
        )
        nbytes = Int32(self.q_tile_bytes)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(
            q_tma,
            tQgQ[(None, Int32(0), tile_H, Int32(0), tile_B)],
            tQsQ,
            tma_bar_ptr=mbar_ptr,
        )

    @cute.jit
    def compute_mma(self, page_ptr, tile_B, tile_H, tile_SPLIT,
                    q, k, v, o_partial, lse_partial):
        """Split-KV decode attention with cp.async K/V loading from BSHD cache."""
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        kv_h = tile_H // Int32(self.kv_group_size)
        _k_base = page_ptr + Int32(self.q_tile_bytes)
        _v_base = page_ptr + Int32(self.q_tile_bytes + self.kv_tile_bytes)

        if warp_idx < Int32(self.num_mma_warps):
            kv_start_block = tile_SPLIT * Int32(self.blocks_per_split)
            kv_end_block = (tile_SPLIT + Int32(1)) * Int32(self.blocks_per_split)
            if kv_end_block > Int32(self.num_kv_blocks):
                kv_end_block = Int32(self.num_kv_blocks)

            mma_op = warp.MmaF16BF16Op(self.q_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((self.num_mma_warps, 1, 1)),
                permutation_mnk=(self.num_mma_warps * 16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)

            sQ = cute.make_tensor(
                cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
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

            _sK = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _k_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.n_block, self.D), stride=(self.smem_stride, 1)),
            )
            _tCsK = thr_mma.partition_B(_sK)
            tCrK = tiled_mma.make_fragment_B(_tCsK)
            tKrK_view = smem_thr_copy_K.retile(tCrK)
            tKsK = smem_thr_copy_K.partition_S(_sK)

            _sVt = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _v_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.D, self.n_block), stride=(1, self.smem_stride)),
            )
            _tBsVt = thr_mma.partition_B(_sVt)
            tBrVt = tiled_mma.make_fragment_B(_tBsVt)
            tVrVt_view = smem_thr_copy_Vt.retile(tBrVt)
            tVsVt = smem_thr_copy_Vt.partition_S(_sVt)

            async_copy_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), self.q_dtype, num_bits_per_copy=128
            )
            copy_thread_layout = cute.make_layout(
                (self.copy_dim0, self.copy_dim1), stride=(self.copy_dim1, 1)
            )
            copy_value_layout = cute.make_layout((1, self.async_copy_elems))
            gmem_tiled_copy = cute.make_tiled_copy_tv(
                async_copy_atom, copy_thread_layout, copy_value_layout
            )
            thr_copy = gmem_tiled_copy.get_slice(tidx)

            sK_cp = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _k_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.n_block, self.D), stride=(self.smem_stride, 1)),
            )
            sV_cp = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _v_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.n_block, self.D), stride=(self.smem_stride, 1)),
            )
            tKsK_cp = thr_copy.partition_D(sK_cp)
            tVsV_cp = thr_copy.partition_D(sV_cp)

            k_head_ptr = (
                k.iterator
                + tile_B * Int32(self.k_b_stride)
                + kv_h * Int32(self.k_h_stride)
            ).align(16)
            v_head_ptr = (
                v.iterator
                + tile_B * Int32(self.v_b_stride)
                + kv_h * Int32(self.v_h_stride)
            ).align(16)
            gK_head = cute.make_tensor(
                k_head_ptr,
                cute.make_layout((self.N, self.D), stride=(self.k_n_stride, 1)),
            )
            gV_head = cute.make_tensor(
                v_head_ptr,
                cute.make_layout((self.N, self.D), stride=(self.v_n_stride, 1)),
            )

            acc_S = cute.make_fragment(
                tiled_mma.partition_shape_C((self.tile_size_M, self.n_block)), Float32
            )
            rP = cute.make_fragment_like(acc_S, self.q_dtype)
            rP_ld = cute.logical_divide(rP.layout, (None, None, 2))
            rP_mma_view = cute.make_layout(
                ((rP_ld.shape[0], rP_ld.shape[2][0]), rP_ld.shape[1], rP_ld.shape[2][1]),
                stride=((rP_ld.stride[0], rP_ld.stride[2][0]), rP_ld.stride[1], rP_ld.stride[2][1]),
            )
            tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

            acc_O = cute.make_fragment(
                tiled_mma.partition_shape_C((self.tile_size_M, self.D)), Float32
            )
            acc_O.fill(0.0)

            acc_O_shape = tiled_mma.partition_shape_C((self.tile_size_M, self.D))
            num_rows = acc_O_shape[0][1] * acc_O_shape[1]
            row_max = cute.make_fragment(cute.make_layout(num_rows), Float32)
            row_sum = cute.make_fragment(cute.make_layout(num_rows), Float32)
            for r in cutlass.range_constexpr(num_rows):
                row_max[r] = Float32(-1e30)
                row_sum[r] = Float32(0.0)

            mcS = cute.make_identity_tensor((self.tile_size_M, self.n_block))
            tScS = thr_mma.partition_C(mcS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)

            for _qkb in cutlass.range_constexpr(self.D // 16):
                cute.copy(smem_tiled_copy_Q, tQsQ[None, None, _qkb], tQrQ_view[None, None, _qkb])

            full_kv_end_block = kv_end_block
            if Int32(self.N % self.n_block) != Int32(0) and kv_end_block == Int32(self.num_kv_blocks):
                full_kv_end_block = kv_end_block - Int32(1)

            kv_idx = kv_start_block
            if kv_idx < full_kv_end_block:
                gK_block0 = cute.local_tile(gK_head, (self.n_block, self.D), (kv_start_block, Int32(0)))
                tKgK0 = thr_copy.partition_S(gK_block0)
                for ci in cutlass.range_constexpr(cute.size(tKsK_cp.shape[2])):
                    cute.copy(gmem_tiled_copy, tKgK0[None, None, ci], tKsK_cp[None, None, ci])
                cute.arch.cp_async_commit_group()

            while kv_idx < full_kv_end_block:
                kv_start = kv_idx * Int32(self.n_block)

                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                gV_block = cute.local_tile(gV_head, (self.n_block, self.D), (kv_idx, Int32(0)))
                tVgV = thr_copy.partition_S(gV_block)
                for ci in cutlass.range_constexpr(cute.size(tVsV_cp.shape[2])):
                    cute.copy(gmem_tiled_copy, tVgV[None, None, ci], tVsV_cp[None, None, ci])
                cute.arch.cp_async_commit_group()

                acc_S.fill(0.0)
                cute.copy(smem_tiled_copy_K, tKsK[None, None, 0], tKrK_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.D // 16):
                    kb_next = (kb + 1) % (self.D // 16)
                    cute.copy(smem_tiled_copy_K, tKsK[None, None, kb_next], tKrK_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_S, tCrQ[None, None, kb], tCrK[None, None, kb], acc_S)

                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                if kv_idx + Int32(1) < full_kv_end_block:
                    gK_next = cute.local_tile(
                        gK_head, (self.n_block, self.D), (kv_idx + Int32(1), Int32(0))
                    )
                    tKgK_next = thr_copy.partition_S(gK_next)
                    for ci in cutlass.range_constexpr(cute.size(tKsK_cp.shape[2])):
                        cute.copy(gmem_tiled_copy, tKgK_next[None, None, ci], tKsK_cp[None, None, ci])
                cute.arch.cp_async_commit_group()

                cute.arch.cp_async_wait_group(1)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
                acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

                if kv_start + Int32(self.n_block) > Int32(self.N):
                    for r in cutlass.range_constexpr(num_rows):
                        for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                            col_idx = tScS_mn[0, c][1]
                            global_col = kv_start + Int32(col_idx)
                            if global_col >= Int32(self.N):
                                acc_S_mn[r, c] = Float32(-1e30)

                if self.causal:
                    last_blk_col = kv_start + Int32(self.n_block - 1)
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

                _any_correction = Int32(0)
                corrections = cute.make_fragment(cute.make_layout(num_rows), Float32)
                for r in cutlass.range_constexpr(num_rows):
                    acc_S_row = acc_S_mn[r, None].load()
                    row_max_cur = acc_S_row.reduce(cute.ReductionOp.MAX, Float32(-1e30), 0)
                    row_max_cur = self._threadquad_reduce_max(row_max_cur)

                    m_old = row_max[r]
                    m_new = cute.arch.fmax(m_old, row_max_cur)

                    acc_scale_ = (m_old - m_new) * Float32(self.scale_log2e)
                    correction = cute.math.exp2(
                        cute.arch.fmax(acc_scale_, Float32(-126.0)), fastmath=True
                    )
                    if acc_scale_ >= Float32(-self.rescale_threshold):
                        m_new = m_old
                        correction = Float32(1.0)
                    row_sum[r] = row_sum[r] * correction
                    corrections[r] = correction
                    if m_new > m_old:
                        _any_correction = Int32(1)

                    acc_S_row_exp = cute.math.exp2(
                        acc_S_row * Float32(self.scale_log2e)
                        - m_new * Float32(self.scale_log2e),
                        fastmath=True,
                    )
                    acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                    row_sum[r] = row_sum[r] + acc_S_row_sum
                    row_max[r] = m_new
                    acc_S_mn[r, None] = acc_S_row_exp

                _skip_rescale = cute.arch.vote_all_sync(_any_correction == Int32(0))
                if not _skip_rescale:
                    for r in cutlass.range_constexpr(num_rows):
                        acc_O_mn[r, None] = acc_O_mn[r, None].load() * corrections[r]

                rP.store(acc_S.load().to(self.q_dtype))
                cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, 0], tVrVt_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.n_block // 16):
                    kb_next = (kb + 1) % (self.n_block // 16)
                    cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, kb_next], tVrVt_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_O, tOrS[None, None, kb], tBrVt[None, None, kb], acc_O)

                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                kv_idx = kv_idx + Int32(1)

            if full_kv_end_block < kv_end_block:
                kv_idx = full_kv_end_block
                kv_start = kv_idx * Int32(self.n_block)

                elem_idx = Int32(tidx)
                total_kv_elems = Int32(self.n_block * self.D)
                while elem_idx < total_kv_elems:
                    row = elem_idx // Int32(self.D)
                    col = elem_idx % Int32(self.D)
                    global_row = kv_start + row
                    if global_row < Int32(self.N):
                        sK_cp[row, col] = gK_head[global_row, col]
                        sV_cp[row, col] = gV_head[global_row, col]
                    else:
                        zero = Float32(0.0).to(self.q_dtype)
                        sK_cp[row, col] = zero
                        sV_cp[row, col] = zero
                    elem_idx = elem_idx + Int32(self.num_mma_threads)

                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                acc_S.fill(0.0)
                cute.copy(smem_tiled_copy_K, tKsK[None, None, 0], tKrK_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.D // 16):
                    kb_next = (kb + 1) % (self.D // 16)
                    cute.copy(smem_tiled_copy_K, tKsK[None, None, kb_next], tKrK_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_S, tCrQ[None, None, kb], tCrK[None, None, kb], acc_S)

                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
                acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

                for r in cutlass.range_constexpr(num_rows):
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        col_idx = tScS_mn[0, c][1]
                        global_col = kv_start + Int32(col_idx)
                        if global_col >= Int32(self.N):
                            acc_S_mn[r, c] = Float32(-1e30)

                if self.causal:
                    last_blk_col = kv_start + Int32(self.n_block - 1)
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

                _any_correction = Int32(0)
                corrections = cute.make_fragment(cute.make_layout(num_rows), Float32)
                for r in cutlass.range_constexpr(num_rows):
                    acc_S_row = acc_S_mn[r, None].load()
                    row_max_cur = acc_S_row.reduce(cute.ReductionOp.MAX, Float32(-1e30), 0)
                    row_max_cur = self._threadquad_reduce_max(row_max_cur)

                    m_old = row_max[r]
                    m_new = cute.arch.fmax(m_old, row_max_cur)

                    acc_scale_ = (m_old - m_new) * Float32(self.scale_log2e)
                    correction = cute.math.exp2(
                        cute.arch.fmax(acc_scale_, Float32(-126.0)), fastmath=True
                    )
                    if acc_scale_ >= Float32(-self.rescale_threshold):
                        m_new = m_old
                        correction = Float32(1.0)
                    row_sum[r] = row_sum[r] * correction
                    corrections[r] = correction
                    if m_new > m_old:
                        _any_correction = Int32(1)

                    acc_S_row_exp = cute.math.exp2(
                        acc_S_row * Float32(self.scale_log2e)
                        - m_new * Float32(self.scale_log2e),
                        fastmath=True,
                    )
                    acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                    row_sum[r] = row_sum[r] + acc_S_row_sum
                    row_max[r] = m_new
                    acc_S_mn[r, None] = acc_S_row_exp

                _skip_rescale = cute.arch.vote_all_sync(_any_correction == Int32(0))
                if not _skip_rescale:
                    for r in cutlass.range_constexpr(num_rows):
                        acc_O_mn[r, None] = acc_O_mn[r, None].load() * corrections[r]

                rP.store(acc_S.load().to(self.q_dtype))
                cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, 0], tVrVt_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.n_block // 16):
                    kb_next = (kb + 1) % (self.n_block // 16)
                    cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, kb_next], tVrVt_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_O, tOrS[None, None, kb], tBrVt[None, None, kb], acc_O)

            acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
            for r in cutlass.range_constexpr(num_rows):
                row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
                inv_sum = cute.arch.rcp_approx(row_sum[r])
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * inv_sum

            lane_in_quad = tidx % Int32(4)
            o_partial_base = (
                o_partial.iterator
                + tile_B * Int32(self.H * self.num_splits * self.M * self.D)
                + tile_H * Int32(self.num_splits * self.M * self.D)
                + tile_SPLIT * Int32(self.M * self.D)
            )
            g_o_partial = cute.make_tensor(
                o_partial_base,
                cute.make_layout((self.tile_size_M, self.D), stride=(self.D, 1)),
            )
            tCgO = thr_mma.partition_C(g_o_partial)
            for i in cutlass.range_constexpr(cute.size(acc_O)):
                tCgO[i] = acc_O[i]

            lse_base = (
                lse_partial.iterator
                + tile_B * Int32(self.H * self.num_splits * self.M)
                + tile_H * Int32(self.num_splits * self.M)
                + tile_SPLIT * Int32(self.M)
            )
            g_lse = cute.make_tensor(lse_base, cute.make_layout(self.M))
            for r in cutlass.range_constexpr(num_rows):
                if lane_in_quad == Int32(0):
                    row_idx = tScS_mn[r, 0][0]
                    if Int32(row_idx) < Int32(self.M):
                        lse_val = row_max[r] * Float32(self.scale_val) + cute.math.log(row_sum[r])
                        g_lse[Int32(row_idx)] = lse_val

# Public split op name is native BSHD-only. The internal bases only provide
# shared MMA setup/helpers.
FlashDecodingSplitOp = FlashDecodingSplitBSHDOp


# =============================================================================
# CombineOp — parallel reduction of split partials
# =============================================================================


class FlashDecodingCombineOp(Op):
    """Combine partial O and LSE from split-KV attention into final output.

    All MMA warps participate in the reduction — 8x more parallelism than
    the previous 32-thread store warp approach.

    Framework guarantees all SplitOp tiles complete before CombineOp starts
    (automatic dependency detection via shared o_partial tensor pointer).
    """

    reads = {
        "o_partial": (cutlass.Float32, ("B", "H", "SPLIT", "M", "D")),
        "lse_partial": (cutlass.Float32, ("B", "H", "SPLIT", "M")),
    }
    writes = {
        "o": (None, ("B", "H", "M", "D")),
        "lse": (cutlass.Float32, ("B", "H", "M")),
    }
    tile = ("B", "H")
    dynamic_dims = ("B", "M", "SPLIT")
    tma_loads = set()
    tma_stores = set()

    def __init__(self, **config):
        super().__init__(**config)
        self.num_splits = self.SPLIT
        self.num_mma_threads = self.threads_per_row
        self.inner_iters = 1
        self.inner_depth = 1

    @classmethod
    def schedule(cls, tile_sizes=None, **tensors):
        """Schedule combine op — one tile per head."""
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("H", 1)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        o_partial = tensors.get("o_partial")
        if o_partial is not None:
            ops[0].static_dims["SPLIT"] = o_partial.shape[2]
            ops[0].static_dims["M"] = o_partial.shape[3]
        return ops

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for combine ops."""
        return FlashDecodingSplitOp.kernel_config(ops)

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_H, o_partial, lse_partial, o, lse):
        """Reduce num_splits partial O/LSE into final output.

        All MMA warps participate. Each thread handles a strided subset
        of M×D output elements, reading all splits per element.
        """
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        num_mma_warps = Int32(self.num_mma_threads // 32)

        if warp_idx < num_mma_warps:
            # Global memory tensors for this head
            # o_partial: (B, H, SPLIT, M, D) fp32
            _op_head = (
                o_partial.iterator
                + tile_B * Int32(self.H * self.num_splits * self.M * self.D)
                + tile_H * Int32(self.num_splits * self.M * self.D)
            )
            g_op = cute.make_tensor(
                _op_head,
                cute.make_layout((self.num_splits, self.M, self.D),
                                 stride=(self.M * self.D, self.D, 1)),
            )
            # lse_partial: (B, H, SPLIT, M) fp32
            _lp_head = (
                lse_partial.iterator
                + tile_B * Int32(self.H * self.num_splits * self.M)
                + tile_H * Int32(self.num_splits * self.M)
            )
            g_lp = cute.make_tensor(
                _lp_head,
                cute.make_layout((self.num_splits, self.M),
                                 stride=(self.M, 1)),
            )
            # o: (B, H, M, D) output
            _o_head = (
                o.iterator
                + tile_B * Int32(self.H * self.M * self.D)
                + tile_H * Int32(self.M * self.D)
            )
            g_o = cute.make_tensor(
                _o_head,
                cute.make_layout((self.M, self.D), stride=(self.D, 1)),
            )
            # lse: (B, H, M) output
            _lse_head = (
                lse.iterator
                + tile_B * Int32(self.H * self.M)
                + tile_H * Int32(self.M)
            )
            g_lse = cute.make_tensor(
                _lse_head,
                cute.make_layout(self.M),
            )

            # Each thread handles elements in a strided pattern over M*D
            total_elems = Int32(self.M * self.D)
            elem_idx = tidx
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
                g_o[row, col] = result.to(self.o_dtype)

                # Write LSE (only once per row, from the thread handling col=0)
                if col == Int32(0):
                    g_lse[row] = lse_max + cute.math.log(scale_sum)

                elem_idx = elem_idx + Int32(self.num_mma_threads)


class FlashDecodingCombineBSHDOp(FlashDecodingCombineOp):
    """Combine split partials into native BSHD output."""

    writes = {
        "o": (None, ("B", "M", "H", "D")),
        "lse": (cutlass.Float32, ("B", "H", "M")),
    }

    @classmethod
    def schedule(cls, tile_sizes=None, **tensors):
        ops = super().schedule(tile_sizes=tile_sizes, **tensors)
        o = tensors.get("o")
        if o is not None:
            ops[0].static_dims["o_b_stride"] = o.stride(0)
            ops[0].static_dims["o_m_stride"] = o.stride(1)
            ops[0].static_dims["o_h_stride"] = o.stride(2)
        return ops

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_H, o_partial, lse_partial, o, lse):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        num_mma_warps = Int32(self.num_mma_threads // 32)

        if warp_idx < num_mma_warps:
            _op_head = (
                o_partial.iterator
                + tile_B * Int32(self.H * self.num_splits * self.M * self.D)
                + tile_H * Int32(self.num_splits * self.M * self.D)
            )
            g_op = cute.make_tensor(
                _op_head,
                cute.make_layout((self.num_splits, self.M, self.D),
                                 stride=(self.M * self.D, self.D, 1)),
            )
            _lp_head = (
                lse_partial.iterator
                + tile_B * Int32(self.H * self.num_splits * self.M)
                + tile_H * Int32(self.num_splits * self.M)
            )
            g_lp = cute.make_tensor(
                _lp_head,
                cute.make_layout((self.num_splits, self.M),
                                 stride=(self.M, 1)),
            )
            _o_head = (
                o.iterator
                + tile_B * Int32(self.o_b_stride)
                + tile_H * Int32(self.o_h_stride)
            )
            g_o = cute.make_tensor(
                _o_head,
                cute.make_layout((self.M, self.D), stride=(self.o_m_stride, 1)),
            )
            _lse_head = (
                lse.iterator
                + tile_B * Int32(self.H * self.M)
                + tile_H * Int32(self.M)
            )
            g_lse = cute.make_tensor(_lse_head, cute.make_layout(self.M))

            total_elems = Int32(self.M * self.D)
            elem_idx = tidx
            while elem_idx < total_elems:
                row = elem_idx // Int32(self.D)
                col = elem_idx % Int32(self.D)

                lse_max = Float32(-1e30)
                si = Int32(0)
                while si < Int32(self.num_splits):
                    lse_val = g_lp[si, row]
                    lse_max = cute.arch.fmax(lse_max, lse_val)
                    si = si + Int32(1)

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

                inv_scale_sum = cute.arch.rcp_approx(scale_sum)
                result = acc * inv_scale_sum
                g_o[row, col] = result.to(self.o_dtype)

                if col == Int32(0):
                    g_lse[row] = lse_max + cute.math.log(scale_sum)

                elem_idx = elem_idx + Int32(self.num_mma_threads)


# =============================================================================
# Public API
# =============================================================================


def flash_decoding_schedule(q, k, v, o, num_splits=0,
                            page_size=DEFAULT_PAGE_SIZE,
                            causal=False, kv_group_size=1):
    """Schedule FlashDecoding split-KV with separate CombineOp.

    Layout contract:
        - q/o are native BSHD `(B, S, H, D)`.
        - k/v are native KV-cache BSHD `(B, N, H_kv, D)`.

    Returns:
        (ops, config): List of ScheduledOps and MegakernelConfig.
    """
    import torch

    assert q.ndim == 4, f"Expected BSHD query tensor, got shape={tuple(q.shape)}"
    assert k.ndim == 4, f"Expected BSHD key cache tensor, got shape={tuple(k.shape)}"
    assert v.ndim == 4, f"Expected BSHD value cache tensor, got shape={tuple(v.shape)}"
    assert o.ndim == 4, f"Expected BSHD output tensor, got shape={tuple(o.shape)}"
    B, M, H, D = q.shape
    lse = torch.empty(B, H, M, dtype=torch.float32, device=q.device)

    # Schedule split ops (writes fp32 partials)
    split_ops, o_partial, lse_partial = FlashDecodingSplitBSHDOp.schedule(
        q=q, k=k, v=v,
        num_splits=num_splits,
        page_size=page_size, causal=causal, kv_group_size=kv_group_size,
    )

    # Schedule combine op (reduces partials → final output)
    combine_ops = FlashDecodingCombineBSHDOp.schedule(
        o_partial=o_partial, lse_partial=lse_partial, o=o, lse=lse,
    )

    ops = split_ops + combine_ops
    config = FlashDecodingSplitBSHDOp.kernel_config(ops)
    return ops, config


__all__ = [
    "FlashDecodingSplitOp",
    "FlashDecodingSplitBSHDOp",
    "FlashDecodingCombineOp",
    "FlashDecodingCombineBSHDOp",
    "flash_decoding_schedule",
]
