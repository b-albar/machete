# Copyright (c) 2025, Machete Authors
"""
Flash Attention Op for the Megakernel (fp16/bf16 tensor core MMA).

Computes scaled dot-product attention with online softmax:
    O[BH, M, D] = softmax(Q[BH, M, D] @ K[BH, N, D]^T / sqrt(D)) @ V[BH, N, D]

Pipelined load/compute/store via framework inner_iters:
    load iter 0:        3D TMA G->S for Q tile (plain smem, buf 0)
    load odd iters 1+:  3D TMA G->S for K block (swizzled smem, buf 1)
    load even iters 2+: 3D TMA G->S for V block (swizzled smem, buf 0)
    compute:            Read Q from buf 0 to registers, then KV block loop:
                        wait K (buf 1) -> S GEMM -> wait V (buf 0) ->
                        softmax -> O GEMM.
    store:              3D TMA S->G for O tile

K and V are loaded as separate inner_iters into alternating buffers.
This doubles n_block vs bundled K+V (2x fewer KV iterations) while
keeping the same double-buffer overlap between DMA and compute.

Supports optional causal masking (lower-left aligned):
    Row i in Q can attend to K/V positions 0..(i + N - M).

Usage:
    from machete.kernels.attention import FlashAttentionSm100Op
    from machete.megakernel import Megakernel, MegakernelConfig

    q = q.view(BH, M, D).contiguous()  # fp16 or bf16
    k = k.view(BH, N, D).contiguous()
    v = v.view(BH, N, D).contiguous()
    o = torch.zeros_like(q)
    ops = FlashAttentionSm100Op.schedule(q=q, k=k, v=v, o=o)
    kernel = Megakernel(ops, config=MegakernelConfig())
    kernel.run()
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu import warp

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import (
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_arrive,
    mbarrier_arrive_expect_tx,
    mbarrier_wait,
    named_barrier_sync,
)


class FlashAttentionSm100Op(Op):
    """Flash Attention operation for the megakernel framework.

    Tensors:
        q: (BH, M, D) -- query  (fp16 or bf16)
        k: (BH, N, D) -- key
        v: (BH, N, D) -- value
        o: (BH, M, D) -- output

    Tiling:
        tile_BH=1 (per head), tile_M from schedule, tile_D=D (full).

    Smem page layout (K/V separated, double-buffered):
        buf0: [n_block x D]  -- Q (iter 0), then V blocks (even iters)
        buf1: [n_block x D]  -- K blocks (odd iters)
        K always in buf 1, V always in buf 0.
    """

    reads = {
        "q": (None, ("BH", "M", "D")),
        "k": (None, ("BH_kv", "N", "D")),
        "v": (None, ("BH_kv", "N", "D")),
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
        if tensor_name not in ("k", "v", "o"):
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
        self.page_size = getattr(self, 'page_size', DEFAULT_PAGE_SIZE)
        self.kv_group_size = getattr(self, 'kv_group_size', 1)

        assert self.q_dtype in (cutlass.Float16, cutlass.BFloat16), (
            f"FlashAttentionSm100Op requires fp16 or bf16, got {self.q_dtype}")
        self.elem_bytes = 2

        self.scale_val = 1.0 / (self.D ** 0.5)
        self.kv_row_bytes = self.D * self.elem_bytes
        self.q_tile_bytes = self.tile_size_M * self.D * self.elem_bytes

        self._init_mma()

    def _init_mma(self):
        """Init tensor core MMA path with dynamic n_block and swizzle."""
        assert self.tile_size_M % 16 == 0 and self.tile_size_M >= 16, (
            f"FlashAttentionSm100Op: tile_size_M={self.tile_size_M} must be "
            f"a positive multiple of 16.")
        self.num_mma_warps = self.tile_size_M // 16
        max_warps = self.threads_per_row // 32
        assert self.num_mma_warps <= max_warps, (
            f"FlashAttentionSm100Op: tile_size_M={self.tile_size_M} requires "
            f"{self.num_mma_warps} warps but only {max_warps} available.")
        self.num_mma_threads = self.num_mma_warps * 32

        assert self.D >= 16 and self.D % 16 == 0, (
            f"FlashAttentionSm100Op: D={self.D} must be >= 16 and x16.")

        assert self.q_tile_bytes <= self.page_size, (
            f"FlashAttentionSm100Op: Q tile ({self.q_tile_bytes}B) > "
            f"page_size ({self.page_size}B). Reduce tile_size_M.")

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

        # Op-managed mbarrier overhead (always 32B: 4 × 8B)
        self.mbar_bytes = 32

        # --- Dynamic n_block computation (K/V separated double-buffer) ---
        # 2 buffers x 1 matrix (K or V) + mbarriers <= page_size
        max_n_block = (self.page_size - self.mbar_bytes) // (2 * self.D * self.elem_bytes)
        self.n_block = (max_n_block // 16) * 16
        self.n_block = min(self.n_block, self.N)
        # Round down to x16 after clamping to N (TMA zero-fills partial tiles)
        self.n_block = max(16, (self.n_block // 16) * 16)
        self.num_kv_blocks = (self.N + self.n_block - 1) // self.n_block

        # Smem layout: buf0=[single matrix], buf1=[single matrix], mbarriers
        # K always in buf 1, V always in buf 0 (Q loaded into buf 0 first)
        self.kv_tile_bytes = self.n_block * self.D * self.elem_bytes

        # 4 op-managed mbarriers after double-buffer data (32 bytes):
        #   smem_consumed[0,1]: compute → store warp (buffer read done)
        #   kblock_ready[0,1]:  TMA hw → compute (new K/V data arrived)
        self.mbar_offset = 2 * self.kv_tile_bytes
        self.mbar_bytes = 32  # 4 × 8B

        total_smem = 2 * self.kv_tile_bytes + self.mbar_bytes
        assert total_smem <= self.page_size, (
            f"FlashAttentionSm100Op: KV double-buffer + mbarriers ({total_smem}B) > "
            f"page_size ({self.page_size}B).")

        # Q must fit in buf 0 (loaded first, then buf 0 reused for V)
        assert self.q_tile_bytes <= self.kv_tile_bytes, (
            f"FlashAttentionSm100Op: Q tile ({self.q_tile_bytes}B) > "
            f"kv_tile_bytes ({self.kv_tile_bytes}B). Reduce tile_size_M.")

        # Framework inner iterations: Q + (K + V) per KV block
        self.inner_iters = 1 + 2 * self.num_kv_blocks
        self.inner_depth = 2  # Double buffering

        # exp2-based softmax
        self.scale_log2e = self.scale_val * 1.4426950408889634074

        # Override compute method
        self.compute = self.compute_mma

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule_forward(cls, tile_sizes=None, causal=False,
                         kv_group_size=1,
                         page_size=DEFAULT_PAGE_SIZE, **tensors):
        """Schedule flash attention forward, optionally with causal masking."""
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("BH", 1)
        q = tensors.get('q')
        n_block = 16  # default
        if q is not None:
            assert q.element_size() == 2, (
                f"FlashAttentionSm100Op requires fp16/bf16, "
                f"got element_size={q.element_size()}")
            D = q.shape[-1]
            M = q.shape[1]
            N = tensors['k'].shape[1]
            elem = q.element_size()
            # n_block from K/V separated double-buffer + mbarrier constraint
            mbar_bytes = 32  # 4 op-managed mbarriers × 8B
            max_n_block = (page_size - mbar_bytes) // (2 * D * elem)
            n_block = (max_n_block // 16) * 16
            n_block = min(n_block, N)
            n_block = max(16, (n_block // 16) * 16)
            if "M" not in tile_sizes:
                # Q must fit in buf 0 = kv_tile_bytes = n_block*D*elem
                max_tile_M_buf = n_block
                max_nw = min(7, max_tile_M_buf // 16, M // 16)
                tile_M = max(16, max_nw * 16)
                tile_sizes["M"] = tile_M
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims['page_size'] = page_size
        ops[0].static_dims['inner_depth'] = 2  # Double-buffered K/V
        ops[0].static_dims['kv_group_size'] = kv_group_size
        if q is not None:
            ops[0].static_dims['n_block'] = n_block
        if causal:
            ops[0].static_dims['causal'] = 1
        return ops

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for the given scheduled ops."""
        from machete.megakernel import MegakernelConfig
        tile_m = ops[0].tile_sizes["M"]
        from machete.megakernel.megakernel import NUM_DMA_WARPS
        num_mma_warps = tile_m // 16
        threads_per_block = (num_mma_warps + NUM_DMA_WARPS) * 32
        page_size = ops[0].static_dims.get('page_size', DEFAULT_PAGE_SIZE)
        return MegakernelConfig(
            threads_per_block=threads_per_block,
            page_size=page_size,
        )

    # =========================================================================
    # Forward Load (DMA warp: Q iter 0, K odd iters, V even iters)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_BH, tile_M, tile_D,
             q_tma, q_tma_gmem, k_tma, k_tma_gmem, v_tma, v_tma_gmem,
             work_mbar, inner_iter_idx):
        """TMA load dispatched by inner_iter_idx.

        iter 0:        Init op-managed mbarriers, TMA Q into buf 0
                       (uses framework work_mbar for Q).
        odd iters 1+:  Wait smem_consumed[1], TMA K → buf 1 (kblock_ready[1])
        even iters 2+: Wait smem_consumed[0], TMA V → buf 0 (kblock_ready[0])
        """
        # GQA: map Q head index to KV head index
        kv_bh = tile_BH // Int32(self.kv_group_size)

        _buf_base = page_ptr + (
            inner_iter_idx % Int32(self.inner_depth)
        ) * Int32(self.kv_tile_bytes)

        # Op-managed mbarrier addresses
        _sc_0 = page_ptr + Int32(self.mbar_offset)       # smem_consumed[0]
        _sc_1 = page_ptr + Int32(self.mbar_offset + 8)   # smem_consumed[1]
        _kr_0 = page_ptr + Int32(self.mbar_offset + 16)  # kblock_ready[0]
        _kr_1 = page_ptr + Int32(self.mbar_offset + 24)  # kblock_ready[1]

        # --- Q load (iter 0) -> buf 0 ---
        if inner_iter_idx == Int32(0):
            # Init op-managed mbarriers
            with cute.arch.elect_one():
                mbarrier_init(_sc_0, Int32(1))
                mbarrier_init(_sc_1, Int32(1))
                mbarrier_init(_kr_0, Int32(1))
                mbarrier_init(_kr_1, Int32(1))
            mbarrier_init_fence()
            # Pre-arrive smem_consumed[1]: buf 1 starts empty
            with cute.arch.elect_one():
                mbarrier_arrive(_sc_1)

            mbar_ptr = cute.make_ptr(
                cutlass.Int64, work_mbar, cute.AddressSpace.smem)
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

        # --- K load (odd iters: 1, 3, 5, ...) -> buf 1 ---
        if inner_iter_idx % Int32(2) == Int32(1):
            kv_block_idx = (inner_iter_idx - Int32(1)) // Int32(2)

            # Wait for compute to free buf 1
            _sc_phase = kv_block_idx % Int32(2)
            mbarrier_wait(_sc_1, _sc_phase)

            swz = cute.make_swizzle(
                self.swizzle_B, self.swizzle_M, self.swizzle_S)
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
            _kr_1_ptr = cute.make_ptr(
                cutlass.Int64, _kr_1, cute.AddressSpace.smem)
            nbytes = Int32(self.kv_tile_bytes)
            with cute.arch.elect_one():
                mbarrier_arrive_expect_tx(_kr_1, nbytes)
            cute.copy(k_tma,
                      tKgK[(None, Int32(0), kv_block_idx, kv_bh)],
                      tKsK, tma_bar_ptr=_kr_1_ptr)

        # --- V load (even iters >= 2: 2, 4, 6, ...) -> buf 0 ---
        if inner_iter_idx > Int32(0):
            if inner_iter_idx % Int32(2) == Int32(0):
                kv_block_idx = (inner_iter_idx - Int32(2)) // Int32(2)

                # Wait for compute to free buf 0
                _sc_phase = kv_block_idx % Int32(2)
                mbarrier_wait(_sc_0, _sc_phase)

                swz = cute.make_swizzle(
                    self.swizzle_B, self.swizzle_M, self.swizzle_S)
                sV = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(self.q_dtype, _buf_base,
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
                _kr_0_ptr = cute.make_ptr(
                    cutlass.Int64, _kr_0, cute.AddressSpace.smem)
                nbytes = Int32(self.kv_tile_bytes)
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(_kr_0, nbytes)
                cute.copy(v_tma,
                          tVgV[(None, Int32(0), kv_block_idx, kv_bh)],
                          tVsV, tma_bar_ptr=_kr_0_ptr)

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
    def compute_mma(self, page_ptr, tile_BH, tile_M, tile_D):
        """Flash attention forward with multi-warp tensor core MMA.

        K/V separated: K always in buf 1, V always in buf 0.
        Op-managed mbarriers derived from page_ptr (GemmOp pattern).
        Phase 1: read Q from buf 0 to registers.
        Phase 2: dynamic while loop over KV blocks with:
          - Signal smem_consumed[0], wait kblock_ready[1] -> S GEMM
          - Signal smem_consumed[1], wait kblock_ready[0] -> softmax + O GEMM
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

            # === Q register fragment (from buf 0) ===
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

            # === CopyUniversal for O write to swizzled smem ===
            smem_copy_atom_O = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.q_dtype)
            smem_tiled_copy_O = cute.make_tiled_copy_C(
                smem_copy_atom_O, tiled_mma)
            smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)

            # === K smem tensor + fragments (always buf 1) ===
            _buf1_base = page_ptr + Int32(self.kv_tile_bytes)
            _sK = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, _buf1_base,
                                  cute.AddressSpace.smem,
                                  assumed_align=128),
                    swz, dtype=self.q_dtype),
                cute.make_layout((self.n_block, self.D),
                                 stride=(self.D, 1)))
            _tCsK = thr_mma.partition_B(_sK)
            tCrK = tiled_mma.make_fragment_B(_tCsK)
            tKrK_view = smem_thr_copy_K.retile(tCrK)
            tKsK = smem_thr_copy_K.partition_S(_sK)

            # === V smem tensor + fragments (always buf 0) ===
            _sVt = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr,
                                  cute.AddressSpace.smem,
                                  assumed_align=128),
                    swz, dtype=self.q_dtype),
                cute.make_layout((self.D, self.n_block),
                                 stride=(1, self.D)))
            _tBsVt = thr_mma.partition_B(_sVt)
            tBrVt = tiled_mma.make_fragment_B(_tBsVt)
            tVrVt_view = smem_thr_copy_Vt.retile(tBrVt)
            tVsVt = smem_thr_copy_Vt.partition_S(_sVt)

            # P register fragment + MMA view (pre-allocated)
            acc_S = cute.make_fragment(
                tiled_mma.partition_shape_C(
                    (self.tile_size_M, self.n_block)),
                Float32)
            rP = cute.make_fragment_like(acc_S, self.q_dtype)
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
            tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

            # === Accumulators ===
            acc_O = cute.make_fragment(
                tiled_mma.partition_shape_C(
                    (self.tile_size_M, self.D)),
                Float32)
            acc_O.fill(0.0)

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

            # Identity tensor for masking
            mcS = cute.make_identity_tensor(
                (self.tile_size_M, self.n_block))
            tScS = thr_mma.partition_C(mcS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)

            # =============================================================
            # Phase 1: Read Q from buf 0 to registers
            # =============================================================
            for kb in cutlass.range_constexpr(self.D // 16):
                cute.autovec_copy(
                    tCsQ[None, None, kb],
                    tCrQ[None, None, kb])

            # =============================================================
            # Phase 2: KV block loop (K/V separated)
            #   Op-managed mbarriers (GemmOp pattern):
            #   smem_consumed[0,1] signaled by compute after reading
            #   kblock_ready[0,1] signaled by TMA on arrival
            # =============================================================
            _sc_0 = page_ptr + Int32(self.mbar_offset)
            _sc_1 = page_ptr + Int32(self.mbar_offset + 8)
            _kr_0 = page_ptr + Int32(self.mbar_offset + 16)
            _kr_1 = page_ptr + Int32(self.mbar_offset + 24)

            kv_idx = Int32(0)
            while kv_idx < Int32(self.num_kv_blocks):
                # --- Signal buf 0 consumed (Q on first, prev V after) ---
                named_barrier_sync(
                    Int32(2), Int32(self.num_mma_threads))
                if tidx == Int32(0):
                    mbarrier_arrive(_sc_0)

                # --- Wait for K on buf 1 ---
                k_phase = kv_idx % Int32(2)
                mbarrier_wait(_kr_1, k_phase)

                kv_start = kv_idx * Int32(self.n_block)

                # --- S GEMM with register pipeline (K in buf 1) ---
                acc_S.fill(0.0)
                cute.copy(smem_tiled_copy_K,
                          tKsK[None, None, 0],
                          tKrK_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.D // 16):
                    kb_next = (kb + 1) % (self.D // 16)
                    cute.copy(smem_tiled_copy_K,
                              tKsK[None, None, kb_next],
                              tKrK_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_S,
                              tCrQ[None, None, kb],
                              tCrK[None, None, kb], acc_S)

                # --- Signal buf 1 consumed (K done) ---
                named_barrier_sync(
                    Int32(2), Int32(self.num_mma_threads))
                if tidx == Int32(0):
                    mbarrier_arrive(_sc_1)

                # --- Wait for V on buf 0 ---
                v_phase = kv_idx % Int32(2)
                mbarrier_wait(_kr_0, v_phase)

                # --- Masking (boundary blocks only) ---
                acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
                acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

                # N-boundary mask (only last KV block)
                if kv_start + Int32(self.n_block) > Int32(self.N):
                    for r in cutlass.range_constexpr(num_rows):
                        for c in cutlass.range_constexpr(
                                cute.size(tScS_mn.shape[1])):
                            col_idx = tScS_mn[0, c][1]
                            global_col = kv_start + Int32(col_idx)
                            if global_col >= Int32(self.N):
                                acc_S_mn[r, c] = Float32(-1e30)

                # Causal mask (only blocks near diagonal)
                if self.causal:
                    last_blk_col = kv_start + Int32(
                        self.n_block - 1)
                    first_row = tile_M * Int32(self.tile_size_M)
                    if last_blk_col > first_row + Int32(
                            self.N - self.M):
                        for r in cutlass.range_constexpr(num_rows):
                            row_idx = tScS_mn[r, 0][0]
                            global_row = (
                                tile_M * Int32(self.tile_size_M)
                                + Int32(row_idx))
                            for c in cutlass.range_constexpr(
                                    cute.size(tScS_mn.shape[1])):
                                col_idx = tScS_mn[0, c][1]
                                global_col = (
                                    kv_start + Int32(col_idx))
                                if global_col > global_row + Int32(
                                        self.N - self.M):
                                    acc_S_mn[r, c] = Float32(-1e30)

                # --- Online softmax ---
                for r in cutlass.range_constexpr(num_rows):
                    acc_S_row = acc_S_mn[r, None].load()
                    row_max_cur = acc_S_row.reduce(
                        cute.ReductionOp.MAX, Float32(-1e30), 0)
                    row_max_cur = self._threadquad_reduce_max(
                        row_max_cur)

                    m_old = row_max[r]
                    m_new = cute.arch.fmax(m_old, row_max_cur)

                    if kv_idx > Int32(0):
                        correction = cute.math.exp2(
                            (m_old - m_new) * Float32(
                                self.scale_log2e),
                            fastmath=True)
                        row_sum[r] = row_sum[r] * correction
                        acc_O_mn[r, None] = (
                            acc_O_mn[r, None].load() * correction)

                    acc_S_row_exp = cute.math.exp2(
                        acc_S_row * Float32(self.scale_log2e)
                        - m_new * Float32(self.scale_log2e),
                        fastmath=True)
                    acc_S_row_sum = acc_S_row_exp.reduce(
                        cute.ReductionOp.ADD, Float32(0.0), 0)
                    row_sum[r] = row_sum[r] + acc_S_row_sum
                    row_max[r] = m_new
                    acc_S_mn[r, None] = acc_S_row_exp

                # --- P conversion + O GEMM with register pipeline
                rP.store(acc_S.load().to(self.q_dtype))

                # V in buf 0: tVsVt already set up
                cute.copy(smem_tiled_copy_Vt,
                          tVsVt[None, None, 0],
                          tVrVt_view[None, None, 0])
                for kb in cutlass.range_constexpr(
                        self.n_block // 16):
                    kb_next = (kb + 1) % (self.n_block // 16)
                    cute.copy(smem_tiled_copy_Vt,
                              tVsVt[None, None, kb_next],
                              tVrVt_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_O,
                              tOrS[None, None, kb],
                              tBrVt[None, None, kb], acc_O)

                kv_idx = kv_idx + Int32(1)

            # === Normalize O by row_sum ===
            acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
            for r in cutlass.range_constexpr(num_rows):
                row_sum[r] = self._threadquad_reduce_sum(
                    row_sum[r])
                inv_sum = cute.arch.rcp_approx(row_sum[r])
                acc_O_mn[r, None] = (
                    acc_O_mn[r, None].load() * inv_sum)

            # === Write O to smem (for TMA store, swizzled) ===
            named_barrier_sync(
                Int32(2), Int32(self.num_mma_threads))
            _o_swz = cute.make_swizzle(
                self.swizzle_B, self.swizzle_M, self.swizzle_S)
            sO = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr,
                                  cute.AddressSpace.smem),
                    _o_swz, dtype=self.q_dtype),
                cute.make_layout((self.tile_size_M, self.D),
                                 stride=(self.D, 1)),
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
        _o_swz = cute.make_swizzle(
            self.swizzle_B, self.swizzle_M, self.swizzle_S)
        sO = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem),
                _o_swz, dtype=self.q_dtype),
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


__all__ = ["FlashAttentionSm100Op"]
