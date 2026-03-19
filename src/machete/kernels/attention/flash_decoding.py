# Copyright (c) 2025, Machete Authors
"""
FlashDecoding: Split-KV Attention for decode workloads.

For small BH (batch×heads) and small M (query length), the standard FA kernel
uses only 1 CTA per head, leaving most SMs idle. FlashDecoding splits the KV
sequence across multiple CTAs (splits), each producing partial O and LSE in
fp32. A combine kernel reduces the partials into the final output.

Architecture (Compute-driven TMA):
    DMA warp: TMA Q load (single shot) + init mbarriers.
    MMA warps: Warp 0 issues TMA K/V loads. All warps do MMA compute.
    Store warp: Fused combine epilogue (atomic counter, reduce partials).

    Overlap:
        V[i] loads during S GEMM (Q × K[i]^T).
        K[i+1] loads during softmax + O GEMM (P × V[i]).

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
from machete.megakernel.interpreter import (
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_arrive_expect_tx,
    mbarrier_wait,
    named_barrier_sync,
    atomic_add_acq_rel_gpu_i32,
)

# Op-managed mbarriers: kblock_ready_K + kblock_ready_V = 2 × 8B
_MBAR_BYTES = 16


class FlashDecodingSplitOp(Op):
    """Split-KV attention with compute-driven TMA K/V loads.

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

    # Q/K/V via TMA: Q loaded by DMA warp, K/V by compute warp 0
    tma_loads = {"q", "k", "v"}
    tma_stores = set()  # Partials written to global from compute

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        """Custom TMA tile shapes for K/V (n_block sub-tiling).

        Returns shape in CuTe mode order (D, n_block, BH).
        The framework permutes PyTorch (BH, N, D) → CuTe (D, N, BH),
        so tile_shape[0] maps to D (contiguous), tile_shape[1] to N.
        """
        if tensor_name in ("k", "v"):
            n_block = static_dims["n_block"]
            D = static_dims["D"]
            return (1, n_block, D)
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

        assert self.q_dtype in (cutlass.Float16, cutlass.BFloat16), (
            f"FlashDecodingSplitOp requires fp16 or bf16, got {self.q_dtype}"
        )
        self.elem_bytes = 2

        self.scale_val = 1.0 / (self.D ** 0.5)
        self.q_tile_bytes = self.M * self.D * self.elem_bytes

        # Raw pointer for atomic counter (set via static_dims)
        self.split_counter_ptr = getattr(self, "split_counter_ptr", 0)

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
        """Schedule split-KV forward pass with fused combine."""
        import torch

        q = tensors["q"]
        k = tensors["k"]
        BH, M, D = q.shape
        N = k.shape[1]

        assert q.element_size() == 2
        assert "o" in tensors, "FlashDecodingSplitOp.schedule requires 'o' tensor"

        # Bump page_size to accommodate op-managed mbarriers
        effective_page_size = page_size + _MBAR_BYTES

        # Auto num_splits
        if num_splits <= 0:
            num_SMs = torch.cuda.get_device_properties(q.device).multi_processor_count
            elem = q.element_size()
            q_tile_bytes = M * D * elem
            kv_budget = effective_page_size - q_tile_bytes - _MBAR_BYTES
            max_n_block = kv_budget // (2 * D * elem)
            n_block = 1 << int(math.log2(max(16, max_n_block)))
            if N < n_block:
                n_block = max(16, (N // 16) * 16)
            num_n_blocks = (N + n_block - 1) // n_block

            total_mblocks = BH  # 1 M-tile per head for decode
            min_blocks_per_split = 2
            max_splits = max(1, num_n_blocks // min_blocks_per_split)
            num_splits = min(num_SMs // max(total_mblocks, 1), max_splits)
            num_splits = max(num_splits, 1)

        # Allocate intermediate buffers
        o_partial = torch.empty(BH, num_splits, M, D, dtype=torch.float32, device=q.device)
        lse_partial = torch.empty(BH, num_splits, M, dtype=torch.float32, device=q.device)
        split_counter = torch.zeros(BH, dtype=torch.int32, device=q.device)

        if "lse" not in tensors:
            tensors["lse"] = torch.empty(BH, M, dtype=torch.float32, device=q.device)

        tile_sizes = dict(tile_sizes or {})
        tile_sizes["BH"] = 1
        tile_sizes["SPLIT"] = 1

        tensors["o_partial"] = o_partial
        tensors["lse_partial"] = lse_partial
        tensors["split_counter"] = split_counter

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
        if causal:
            ops[0].static_dims["causal"] = 1
        if kv_group_size > 1:
            ops[0].static_dims["kv_group_size"] = kv_group_size

        # Raw pointer for atomic counter
        ops[0].static_dims["split_counter_ptr"] = split_counter.data_ptr()

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

        # Limit grid to actual tiles for standalone decode attention
        total_tiles = sum(op.total_tiles for op in ops)
        import torch
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        num_sms = min(num_sms, total_tiles)

        return MegakernelConfig(
            threads_per_block=threads_per_block,
            page_size=page_size,
            noinline=True,
            num_pages=2,
            num_sms=num_sms,
        )

    # =========================================================================
    # Load (TMA Q + init mbarriers)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_BH, tile_SPLIT, q_tma, q_tma_gmem, work_mbar):
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
            cute.make_layout((self.D, self.tile_size_M, 1)),
        )
        gQ = cute.local_tile(
            q_tma_gmem,
            (self.D, self.tile_size_M, 1),
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
        cute.copy(q_tma, tQgQ[(None, Int32(0), Int32(0), tile_BH)], tQsQ, tma_bar_ptr=mbar_ptr)

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
    def compute_mma(self, page_ptr, tile_BH, tile_SPLIT,
                    q, k, v, o_partial, lse_partial, o, lse, split_counter,
                    k_tma, k_tma_gmem, v_tma, v_tma_gmem):
        """Compute-driven TMA split-KV flash attention.

        Same pipeline as FlashAttentionSm120Op but processes only KV blocks
        [kv_start, kv_end) and writes fp32 partials to global memory.
        """
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        # GQA: map Q head index to KV head index
        kv_bh = tile_BH // Int32(self.kv_group_size)

        # === TMA partition setup (must be at function top level — ===
        # === TMA atoms cannot cross MLIR SCF region boundaries)  ===

        _k_base = page_ptr + Int32(self.q_tile_bytes)
        _v_base = page_ptr + Int32(self.q_tile_bytes + self.kv_tile_bytes)

        sK_tma = cute.make_tensor(
            cute.make_ptr(self.q_dtype, _k_base, cute.AddressSpace.smem),
            cute.make_layout(
                (self.D, self.n_block, 1),
                stride=(1, self.D, self.D * self.n_block)),
        )
        gK_tma = cute.local_tile(
            k_tma_gmem, (self.D, self.n_block, 1), (None, None, None),
        )
        tKsK_tma, tKgK_tma = cute.nvgpu.cpasync.tma_partition(
            k_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sK_tma, 0, 3),
            cute.group_modes(gK_tma, 0, 3),
        )

        sV_tma = cute.make_tensor(
            cute.make_ptr(self.q_dtype, _v_base, cute.AddressSpace.smem),
            cute.make_layout(
                (self.D, self.n_block, 1),
                stride=(1, self.D, self.D * self.n_block)),
        )
        gV_tma = cute.local_tile(
            v_tma_gmem, (self.D, self.n_block, 1), (None, None, None),
        )
        tVsV_tma, tVgV_tma = cute.nvgpu.cpasync.tma_partition(
            v_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sV_tma, 0, 3),
            cute.group_modes(gV_tma, 0, 3),
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
                          tKgK_tma[(None, Int32(0), kv_idx, kv_bh)],
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
                              tVgV_tma[(None, Int32(0), kv_idx, kv_bh)],
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
                                  tKgK_tma[(None, Int32(0), kv_idx + Int32(1), kv_bh)],
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

            # Write partial O to global: o_partial[BH, SPLIT, M, D]
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
