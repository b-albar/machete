# Copyright (c) 2025, Machete Authors
"""
Cooperative Flash Attention Op for the Megakernel (fp16/bf16 tensor core MMA).

Computes scaled dot-product attention with online softmax:
    O[BH, M, D] = softmax(Q[BH, M, D] @ K[BH, N, D]^T / sqrt(D)) @ V[BH, N, D]

Architecture (cute_fa2-like cooperative):
    DMA warp:  TMA Q load (single shot) + TMA O store
    MMA warps: All warps cooperatively load K/V via cpasync AND compute MMA.

Pipelined KV loop (inside compute, all MMA warps):
    prologue:  cpasync K[0] → buf0
    loop:      wait K → cpasync V → S GEMM (overlap V load)
               wait V → cpasync K[next] → softmax → O GEMM (overlap K load)
    epilogue:  write O to smem for TMA store

Smem page layout (16KB, sequential reuse):
    Phase 1: Q tile fills page (tile_M × D × 2 bytes)
    Phase 2: After Q→registers, page reused for KV double-buffer:
        buf0: [n_block × D] for K blocks
        buf1: [n_block × D] for V blocks

Key improvement: tile_M doubles vs FlashAttentionSm100Op (64 vs 32 for D=128)
because Q gets the full 16KB page (not sharing a buffer with V).

Supports optional causal masking (lower-left aligned):
    Row i in Q can attend to K/V positions 0..(i + N - M).

Usage:
    from machete.kernels.attention import FlashAttentionSm120Op
    from machete.megakernel import Megakernel, MegakernelConfig

    q = q.view(BH, M, D).contiguous()  # fp16 or bf16
    k = k.view(BH, N, D).contiguous()
    v = v.view(BH, N, D).contiguous()
    o = torch.zeros_like(q)
    ops = FlashAttentionSm120Op.schedule(q=q, k=k, v=v, o=o)
    tile_m = ops[0].tile_sizes["M"]
    tpb = (tile_m // 16 + 1) * 32
    kernel = Megakernel(ops, config=MegakernelConfig(threads_per_block=tpb))
    kernel.run()
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu import warp

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import (
    named_barrier_sync,
)


class FlashAttentionSm120Op(Op):
    """Cooperative Flash Attention — MMA warps do both cpasync loads and MMA.

    Tensors:
        q: (BH, M, D) -- query  (fp16 or bf16)
        k: (BH, N, D) -- key
        v: (BH, N, D) -- value
        o: (BH, M, D) -- output

    Tiling:
        tile_BH=1 (per head), tile_M from schedule, tile_D=D (full).

    Smem page layout:
        Phase 1: Q fills page (tile_M × D × 2 bytes)
        Phase 2: buf0=[n_block × D] for K, buf1=[n_block × D] for V
    """

    reads = {
        "q": (None, ("BH", "M", "D")),
        "k": (None, ("BH", "N", "D")),
        "v": (None, ("BH", "N", "D")),
    }
    writes = {"o": (None, ("BH", "M", "D"))}
    tile = ("BH", "M", "D")

    # Only Q via TMA (DMA warp), K/V loaded by cpasync in compute
    tma_loads = {"q"}
    tma_stores = {"o"}

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        """Swizzled smem layout for O TMA store descriptor."""
        if tensor_name != "o":
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

        assert self.q_dtype in (cutlass.Float16, cutlass.BFloat16), (
            f"FlashAttentionSm120Op requires fp16 or bf16, got {self.q_dtype}"
        )
        self.elem_bytes = 2

        self.scale_val = 1.0 / (self.D**0.5)
        self.kv_row_bytes = self.D * self.elem_bytes
        self.q_tile_bytes = self.tile_size_M * self.D * self.elem_bytes

        self._init_mma()

    def _init_mma(self):
        """Init cooperative MMA path with cpasync K/V loading."""
        assert self.tile_size_M % 16 == 0 and self.tile_size_M >= 16, (
            f"FlashAttentionSm120Op: tile_size_M={self.tile_size_M} must be a positive multiple of 16."
        )
        self.num_mma_warps = self.tile_size_M // 16
        max_warps = self.threads_per_row // 32
        assert self.num_mma_warps <= max_warps, (
            f"FlashAttentionSm120Op: tile_size_M={self.tile_size_M} requires "
            f"{self.num_mma_warps} warps but only {max_warps} available."
        )
        self.num_mma_threads = self.num_mma_warps * 32

        assert self.D >= 16 and self.D % 16 == 0, f"FlashAttentionSm120Op: D={self.D} must be >= 16 and x16."

        assert self.q_tile_bytes <= self.page_size, (
            f"FlashAttentionSm120Op: Q tile ({self.q_tile_bytes}B) > page_size ({self.page_size}B). Reduce tile_size_M."
        )

        # --- n_block: K/V double-buffer must fit in page after Q→regs ---
        # 2 × n_block × D × elem_bytes <= page_size
        max_n_block = self.page_size // (2 * self.D * self.elem_bytes)
        self.n_block = (max_n_block // 16) * 16
        self.n_block = min(self.n_block, self.N)
        self.n_block = max(16, (self.n_block // 16) * 16)
        self.num_kv_blocks = (self.N + self.n_block - 1) // self.n_block

        self.kv_tile_bytes = self.n_block * self.D * self.elem_bytes
        total_kv_smem = 2 * self.kv_tile_bytes
        assert total_kv_smem <= self.page_size, (
            f"FlashAttentionSm120Op: KV double-buffer ({total_kv_smem}B) > page_size ({self.page_size}B)."
        )

        # DMA loads Q once, then compute handles everything
        self.inner_iters = 1
        self.inner_depth = 1

        # --- Swizzle parameters (same as FlashAttentionSm100Op) ---
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

        # cpasync thread layout for K/V loading
        # 128-bit copies = 8 fp16 elements per thread per copy
        self.async_copy_elems = 128 // (self.elem_bytes * 8)  # 8 for fp16
        self.copy_dim1 = self.D // self.async_copy_elems
        self.copy_dim0 = self.num_mma_threads // self.copy_dim1

        # exp2-based softmax
        self.scale_log2e = self.scale_val * 1.4426950408889634074
        # Rescale threshold (log2-space): skip O rescale when correction factor
        # >= 2^(-threshold), i.e. row max changed insignificantly.
        # 8.0 matches flash_fwd_sm100 for fp16/bf16 (2^-8 = 1/256 worst-case).
        self.rescale_threshold = 8.0

        # Override compute method
        self.compute = self.compute_mma

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule_forward(cls, tile_sizes=None, causal=False, page_size=DEFAULT_PAGE_SIZE, **tensors):
        """Schedule cooperative flash attention forward."""
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("BH", 1)
        q = tensors.get("q")
        if q is not None:
            assert q.element_size() == 2, (
                f"FlashAttentionSm120Op requires fp16/bf16, got element_size={q.element_size()}"
            )
            D = q.shape[-1]
            M = q.shape[1]
            elem = q.element_size()
            if "M" not in tile_sizes:
                # Q gets the full page: tile_M × D × elem ≤ page_size
                max_tile_M_page = page_size // (D * elem)
                max_nw = min(7, max_tile_M_page // 16, M // 16)
                tile_M = max(16, max_nw * 16)
                tile_sizes["M"] = tile_M
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        if causal:
            ops[0].static_dims["causal"] = 1
        return ops

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for the given scheduled ops."""
        from machete.megakernel import MegakernelConfig

        tile_m = ops[0].tile_sizes["M"]
        num_mma_warps = tile_m // 16
        threads_per_block = (num_mma_warps + 1) * 32
        page_size = ops[0].static_dims.get("page_size", DEFAULT_PAGE_SIZE)
        return MegakernelConfig(
            threads_per_block=threads_per_block,
            page_size=page_size,
        )

    # =========================================================================
    # Forward Load (DMA warp: TMA Q only)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_BH, tile_M, tile_D, q_tma, q_tma_gmem, work_mbar, inner_iter_idx):
        """TMA Q load into page (single shot)."""
        from machete.megakernel.interpreter import mbarrier_arrive_expect_tx

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
            q_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(gQ, 0, 3),
        )
        nbytes = Int32(self.q_tile_bytes)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(q_tma, tQgQ[(None, tile_D, tile_M, tile_BH)], tQsQ, tma_bar_ptr=mbar_ptr)

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
            cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31),
        )
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31),
        )
        return val

    def _threadquad_reduce_max(self, val):
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val):
        return self._threadquad_reduce(val, lambda x, y: x + y)

    # =========================================================================
    # Forward Compute -- Cooperative cpasync + Tensor Core MMA
    # =========================================================================

    @cute.jit
    def compute_mma(
        self, page_ptr, tile_BH, tile_M, tile_D, q, k, v, o, work_mbar, smem_consumed_mbar, work_mbar_phase
    ):
        """Cooperative flash attention: MMA warps do both cpasync loads and MMA.

        Phase 1: Read Q from smem (TMA-loaded) to registers.
        Phase 2: Signal smem consumed (page free for KV).
        Phase 3: Pipelined KV loop with cpasync K/V + MMA.
        Phase 4: Write O to smem for TMA store.

        Note: q and o are unused here (loaded/stored by TMA) but must be in
        the signature because the framework passes all tensors when
        expects_tensors=True (all-or-none).
        """
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_id = tidx % Int32(32)

        if warp_idx < Int32(self.num_mma_warps):
            # === MMA setup (multi-warp) ===
            mma_op = warp.MmaF16BF16Op(self.q_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((self.num_mma_warps, 1, 1)),
                permutation_mnk=(self.num_mma_warps * 16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)

            # === Q register fragment (from smem) ===
            sQ = cute.make_tensor(
                cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.tile_size_M, self.D), stride=(self.D, 1)),
            )
            tCsQ = thr_mma.partition_A(sQ)
            tCrQ = tiled_mma.make_fragment_A(tCsQ)

            # =============================================================
            # Phase 1: Read Q from smem to registers
            # =============================================================
            for kb in cutlass.range_constexpr(self.D // 16):
                cute.autovec_copy(tCsQ[None, None, kb], tCrQ[None, None, kb])

            # =============================================================
            # Phase 2: Signal smem consumed (page now free for KV)
            # =============================================================
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
            if lane_id == Int32(0):
                from machete.megakernel.interpreter import mbarrier_arrive

                mbarrier_arrive(smem_consumed_mbar)

            # === Swizzle + LdMatrix setup for KV ===
            swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)

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

            # === CopyUniversal for O write to swizzled smem ===
            smem_copy_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.q_dtype)
            smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
            smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)

            # === K smem tensor + LdMatrix fragments (buf0, swizzled) ===
            _sK = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                    swz,
                    dtype=self.q_dtype,
                ),
                cute.make_layout((self.n_block, self.D), stride=(self.D, 1)),
            )
            _tCsK = thr_mma.partition_B(_sK)
            tCrK = tiled_mma.make_fragment_B(_tCsK)
            tKrK_view = smem_thr_copy_K.retile(tCrK)
            tKsK = smem_thr_copy_K.partition_S(_sK)

            # === V smem tensor + LdMatrix fragments (buf1, swizzled transposed) ===
            _buf1_base = page_ptr + Int32(self.kv_tile_bytes)
            _sVt = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, _buf1_base, cute.AddressSpace.smem, assumed_align=128),
                    swz,
                    dtype=self.q_dtype,
                ),
                cute.make_layout((self.D, self.n_block), stride=(1, self.D)),
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

            # cpasync smem destinations (buf0=K, buf1=V, swizzled)
            sK_cp = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                    swz,
                    dtype=self.q_dtype,
                ),
                cute.make_layout((self.n_block, self.D), stride=(self.D, 1)),
            )
            sV_cp = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, _buf1_base, cute.AddressSpace.smem, assumed_align=128),
                    swz,
                    dtype=self.q_dtype,
                ),
                cute.make_layout((self.n_block, self.D), stride=(self.D, 1)),
            )
            tKsK_cp = thr_copy.partition_D(sK_cp)
            tVsV_cp = thr_copy.partition_D(sV_cp)

            # cpasync global sources: K and V for current head
            # k, v are CuTe tensors (BH, N, D) from framework.
            # .align(16) annotates 16-byte alignment for 128-bit cpasync.
            k_head_ptr = (k.iterator + tile_BH * Int32(self.N * self.D)).align(16)
            v_head_ptr = (v.iterator + tile_BH * Int32(self.N * self.D)).align(16)
            gK_head = cute.make_tensor(k_head_ptr, cute.make_layout((self.N, self.D), stride=(self.D, 1)))
            gV_head = cute.make_tensor(v_head_ptr, cute.make_layout((self.N, self.D), stride=(self.D, 1)))

            # P register fragment + MMA view (pre-allocated)
            acc_S = cute.make_fragment(tiled_mma.partition_shape_C((self.tile_size_M, self.n_block)), Float32)
            rP = cute.make_fragment_like(acc_S, self.q_dtype)
            rP_ld = cute.logical_divide(rP.layout, (None, None, 2))
            rP_mma_view = cute.make_layout(
                ((rP_ld.shape[0], rP_ld.shape[2][0]), rP_ld.shape[1], rP_ld.shape[2][1]),
                stride=((rP_ld.stride[0], rP_ld.stride[2][0]), rP_ld.stride[1], rP_ld.stride[2][1]),
            )
            tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

            # === Accumulators ===
            acc_O = cute.make_fragment(tiled_mma.partition_shape_C((self.tile_size_M, self.D)), Float32)
            acc_O.fill(0.0)

            # Softmax state
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
            # Phase 3: Pipelined KV loop with cpasync
            # =============================================================

            # Prologue: cpasync K[0] → buf0
            gK_block0 = cute.local_tile(gK_head, (self.n_block, self.D), (Int32(0), Int32(0)))
            tKgK0 = thr_copy.partition_S(gK_block0)
            for ci in cutlass.range_constexpr(cute.size(tKsK_cp.shape[2])):
                cute.copy(gmem_tiled_copy, tKgK0[None, None, ci], tKsK_cp[None, None, ci])
            cute.arch.cp_async_commit_group()

            kv_idx = Int32(0)
            while kv_idx < Int32(self.num_kv_blocks):
                kv_start = kv_idx * Int32(self.n_block)

                # --- Wait for K[i] in buf0 ---
                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # --- Start V[i] cpasync → buf1 ---
                gV_block = cute.local_tile(gV_head, (self.n_block, self.D), (kv_idx, Int32(0)))
                tVgV = thr_copy.partition_S(gV_block)
                for ci in cutlass.range_constexpr(cute.size(tVsV_cp.shape[2])):
                    cute.copy(gmem_tiled_copy, tVgV[None, None, ci], tVsV_cp[None, None, ci])
                cute.arch.cp_async_commit_group()

                # --- S GEMM with register pipeline (K in buf0) ---
                acc_S.fill(0.0)
                cute.copy(smem_tiled_copy_K, tKsK[None, None, 0], tKrK_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.D // 16):
                    kb_next = (kb + 1) % (self.D // 16)
                    cute.copy(smem_tiled_copy_K, tKsK[None, None, kb_next], tKrK_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_S, tCrQ[None, None, kb], tCrK[None, None, kb], acc_S)

                # --- Wait for V[i] in buf1 ---
                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # --- Start K[i+1] cpasync → buf0 (if not last) ---
                if kv_idx + Int32(1) < Int32(self.num_kv_blocks):
                    gK_next = cute.local_tile(gK_head, (self.n_block, self.D), (kv_idx + Int32(1), Int32(0)))
                    tKgK_next = thr_copy.partition_S(gK_next)
                    for ci in cutlass.range_constexpr(cute.size(tKsK_cp.shape[2])):
                        cute.copy(gmem_tiled_copy, tKgK_next[None, None, ci], tKsK_cp[None, None, ci])
                    cute.arch.cp_async_commit_group()

                # --- Masking (boundary blocks only) ---
                acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
                acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

                # N-boundary mask (only last KV block)
                if kv_start + Int32(self.n_block) > Int32(self.N):
                    for r in cutlass.range_constexpr(num_rows):
                        for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                            col_idx = tScS_mn[0, c][1]
                            global_col = kv_start + Int32(col_idx)
                            if global_col >= Int32(self.N):
                                acc_S_mn[r, c] = Float32(-1e30)

                # Causal mask (only blocks near diagonal)
                if self.causal:
                    last_blk_col = kv_start + Int32(self.n_block - 1)
                    first_row = tile_M * Int32(self.tile_size_M)
                    if last_blk_col > first_row + Int32(self.N - self.M):
                        for r in cutlass.range_constexpr(num_rows):
                            row_idx = tScS_mn[r, 0][0]
                            global_row = tile_M * Int32(self.tile_size_M) + Int32(row_idx)
                            for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                                col_idx = tScS_mn[0, c][1]
                                global_col = kv_start + Int32(col_idx)
                                if global_col > global_row + Int32(self.N - self.M):
                                    acc_S_mn[r, c] = Float32(-1e30)

                # --- Online softmax ---
                _any_correction = Int32(0)
                corrections = cute.make_fragment(
                    cute.make_layout(num_rows), Float32)
                for r in cutlass.range_constexpr(num_rows):
                    acc_S_row = acc_S_mn[r, None].load()
                    row_max_cur = acc_S_row.reduce(cute.ReductionOp.MAX, Float32(-1e30), 0)
                    row_max_cur = self._threadquad_reduce_max(row_max_cur)

                    m_old = row_max[r]
                    m_new = cute.arch.fmax(m_old, row_max_cur)

                    if kv_idx > Int32(0):
                        acc_scale_ = (m_old - m_new) * Float32(self.scale_log2e)
                        correction = cute.math.exp2(acc_scale_, fastmath=True)
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

                # Deferred O rescale: skip if max unchanged across all rows/threads
                if kv_idx > Int32(0):
                    _skip_rescale = cute.arch.vote_all_sync(
                        _any_correction == Int32(0))
                    if not _skip_rescale:
                        for r in cutlass.range_constexpr(num_rows):
                            acc_O_mn[r, None] = acc_O_mn[r, None].load() * corrections[r]

                # --- P conversion + O GEMM with register pipeline ---
                rP.store(acc_S.load().to(self.q_dtype))

                # V in buf1: tVsVt already set up
                cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, 0], tVrVt_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.n_block // 16):
                    kb_next = (kb + 1) % (self.n_block // 16)
                    cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, kb_next], tVrVt_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_O, tOrS[None, None, kb], tBrVt[None, None, kb], acc_O)

                kv_idx = kv_idx + Int32(1)

            # =============================================================
            # Phase 4: Normalize O and write to smem for TMA store
            # =============================================================
            acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
            for r in cutlass.range_constexpr(num_rows):
                row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
                inv_sum = cute.arch.rcp_approx(row_sum[r])
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * inv_sum

            # Write O to smem (at page_ptr, swizzled layout)
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
            _o_swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)
            sO = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem), _o_swz, dtype=self.q_dtype
                ),
                cute.make_layout((self.tile_size_M, self.D), stride=(self.D, 1)),
            )
            # Convert acc_O (f32) → q_dtype in registers
            tCrO_q = cute.make_fragment_like(acc_O, self.q_dtype)
            for i in cutlass.range_constexpr(cute.size(acc_O)):
                tCrO_q[i] = acc_O[i].to(self.q_dtype)
            # Retile register fragment for copy atom, partition smem
            tOrO = smem_thr_copy_O.retile(tCrO_q)
            tOsO = smem_thr_copy_O.partition_D(sO)
            cute.copy(smem_tiled_copy_O, tOrO, tOsO)

    # =========================================================================
    # Forward Store (3D TMA S->G for O)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_BH, tile_M, tile_D, o_tma, o_tma_gmem):
        """TMA store of O from shared to global memory (swizzled)."""
        _o_swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)
        sO = cute.make_tensor(
            cute.recast_ptr(cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem), _o_swz, dtype=self.q_dtype),
            cute.make_layout((self.D, self.tile_size_M, 1)),
        )
        gO = cute.local_tile(
            o_tma_gmem,
            (self.D, self.tile_size_M, 1),
            (None, None, None),
        )
        tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
            o_tma,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sO, 0, 3),
            cute.group_modes(gO, 0, 3),
        )
        with cute.arch.elect_one():
            cute.copy(o_tma, tOsO, tOgO[(None, tile_D, tile_M, tile_BH)])


__all__ = ["FlashAttentionSm120Op"]
