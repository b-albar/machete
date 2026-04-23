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

Smem page layout (48KB default, persistent Q + KV double-buffer):
    [Q: tile_M × D × 2 bytes]  (persistent, swizzled for LdMatrix reads)
    [buf0: n_block × D × 2 bytes]  for K blocks
    [buf1: n_block × D × 2 bytes]  for V blocks

Q stays in smem for the entire KV loop. Each S GEMM k-block reloads Q
via LdMatrix, trading smem bandwidth for ~28 fewer registers/thread.

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
    tpb = (tile_m // 16 + 2) * 32  # +2 for load warp + store warp
    kernel = Megakernel(ops, config=MegakernelConfig(threads_per_block=tpb))
    kernel.run()
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu import warp
import torch

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE, config_dim_i32
from machete.megakernel.interpreter import (
    named_barrier_sync,
)


def _max_attention_page_size(device=None):
    if device is None:
        device = torch.cuda.current_device()
    max_smem = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    return ((max_smem - 512) // 128) * 128


def _effective_page_size(q, page_size):
    if page_size is None:
        return _max_attention_page_size(q.device)
    return page_size


def _bshd_to_bhsd(x):
    if x is None:
        return None
    assert x.ndim == 4, f"Expected 4D BSHD tensor, got shape={tuple(x.shape)}"
    return x.permute(0, 2, 1, 3)


class FlashAttentionSm120Op(Op):
    """Cooperative Flash Attention — MMA warps do both cpasync loads and MMA.

    Tensors:
        q: (BH, M, D) -- query  (fp16 or bf16)
        k: (BH, N, D) -- key
        v: (BH, N, D) -- value
        o: (BH, M, D) -- output

    Tiling:
        tile_BH=1 (per head), tile_M from schedule, tile_D=D (full).

    Smem page layout (persistent Q + KV):
        [Q: tile_M × D × 2B] [buf0: n_block × D] [buf1: n_block × D]
    """

    reads = {
        "q": (None, ("B", "H", "M", "D")),
        "k": (None, ("B", "H_kv", "N", "D")),
        "v": (None, ("B", "H_kv", "N", "D")),
    }
    writes = {"o": (None, ("B", "H", "M", "D")), "lse": (cutlass.Float32, ("B", "H", "M"))}
    tile = ("B", "H", "M", "D")
    dynamic_dims = ("B", "M", "N")

    # Only Q via TMA (DMA warp), K/V loaded by cpasync in compute
    tma_loads = {"q"}
    tma_stores = {"o"}

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        """Swizzled smem layout for Q TMA load and O TMA store descriptors."""
        if tensor_name not in ("o", "q"):
            return None

        D = static_dims["D"]
        if D >= 64:
            B = 3
        elif D >= 32:
            B = 2
        else:
            B = 1

        # Generic for any ndim (3D or 4D): build row-major strides
        dims = tma_tile_shape
        strides = [1]
        for i in range(len(dims) - 1):
            strides.append(strides[-1] * dims[i])
        shape_str = ", ".join(str(d) for d in dims)
        stride_str = ", ".join(str(s) for s in strides)
        return (
            f"cute.make_composed_layout("
            f"cute.make_swizzle({B}, 4, 3), 0, "
            f"cute.make_layout(({shape_str}), "
            f"stride=({stride_str})))"
        )

    def __init__(self, **config):
        super().__init__(**config)
        self.causal = getattr(self, "causal", 0)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)
        self.kv_group_size = getattr(self, "kv_group_size", 1)

        assert self.q_dtype in (cutlass.Float16, cutlass.BFloat16), (
            f"FlashAttentionSm120Op requires fp16 or bf16, got {self.q_dtype}"
        )
        self.elem_bytes = 2

        self.scale_val = 1.0 / (self.D**0.5)
        self.q_tile_bytes = self.tile_size_M * self.D * self.elem_bytes

        # Stride overrides for strided K/V (e.g., view+permute of GEMM output).
        # Defaults assume contiguous (B, H_kv, N, D) layout.
        self.k_b_stride = getattr(self, "k_b_stride", None)
        self.k_h_stride = getattr(self, "k_h_stride", None)
        self.k_n_stride = getattr(self, "k_n_stride", self.D)
        self.v_b_stride = getattr(self, "v_b_stride", None)
        self.v_h_stride = getattr(self, "v_h_stride", None)
        self.v_n_stride = getattr(self, "v_n_stride", self.D)

        # TMA coordinate order flag: when Q/O are strided (from view+permute),
        # the framework sorts TMA dims by stride, giving (D, H, M, B) instead
        # of the contiguous (D, M, H, B). Load/store must match.
        self.q_tma_permuted = bool(getattr(self, "q_tma_permuted", 0))
        self.o_tma_permuted = bool(getattr(self, "o_tma_permuted", 0))

        self._init_mma()

        # Pre-compute TMA smem shapes outside JIT (CuTe DSL can't define
        # Python tuples inside dynamic if — they don't escape SCF regions).
        if self.q_tma_permuted:
            self._q_tma_smem_shape = (self.D, 1, self.tile_size_M, 1)
        else:
            self._q_tma_smem_shape = (self.D, self.tile_size_M, 1, 1)
        if self.o_tma_permuted:
            self._o_tma_smem_shape = (self.D, 1, self.tile_size_M, 1)
        else:
            self._o_tma_smem_shape = (self.D, self.tile_size_M, 1, 1)

    def _init_mma(self):
        """Init cooperative MMA path with cpasync K/V loading."""
        assert self.tile_size_M % 16 == 0 and self.tile_size_M >= 16, (
            f"FlashAttentionSm120Op: tile_size_M={self.tile_size_M} must be a positive multiple of 16."
        )

        # num_mma_warps from schedule (supports m_reps > 1 where tile_M > warps*16)
        self.num_mma_warps = getattr(self, "num_mma_warps", self.tile_size_M // 16)
        max_warps = self.threads_per_row // 32
        assert self.num_mma_warps <= max_warps, (
            f"FlashAttentionSm120Op: tile_size_M={self.tile_size_M} requires "
            f"{self.num_mma_warps} warps but only {max_warps} available."
        )
        self.num_mma_threads = self.num_mma_warps * 32

        # M-repetitions per warp: tile_M / (warps * 16)
        self.mma_tile_M = self.num_mma_warps * 16
        self.m_reps = self.tile_size_M // self.mma_tile_M
        assert self.tile_size_M == self.mma_tile_M * self.m_reps, (
            f"tile_size_M={self.tile_size_M} must be exact multiple of "
            f"mma_tile_M={self.mma_tile_M} (num_mma_warps={self.num_mma_warps})"
        )

        # Q-in-smem mode: when m_reps > 1, Q stays in smem, KV uses remaining space.
        # Q-preload mode: when m_reps == 1, Q preloaded to regs, full page for KV.
        self.q_in_smem = (self.m_reps > 1)

        assert self.D >= 16 and self.D % 16 == 0, f"FlashAttentionSm120Op: D={self.D} must be >= 16 and x16."

        assert self.q_tile_bytes <= self.page_size, (
            f"FlashAttentionSm120Op: Q tile ({self.q_tile_bytes}B) > page_size ({self.page_size}B). Reduce tile_size_M."
        )

        import math

        # --- Q/O swizzle (TMA requires it, loaded once → minimal reg impact) ---
        if self.D >= 64:
            self.swizzle_B = 3
        elif self.D >= 32:
            self.swizzle_B = 2
        else:
            self.swizzle_B = 1
        self.swizzle_M = 4
        self.swizzle_S = 3

        # --- KV row padding (replaces swizzle to avoid address precomputation regs) ---
        # Pad each row by 8 elements (16 bytes) so consecutive rows hit different banks.
        # Bank offset per row = (D+pad)*elem_bytes/4 % 32 = 4 → 8 rows span 8 bank groups.
        self.smem_pad = 8  # bf16/fp16 elements
        self.smem_stride = self.D + self.smem_pad

        # cpasync thread layout for K/V loading — compute copy_dim0 BEFORE
        # n_block so we can round n_block to a multiple of copy_dim0.
        # 128-bit copies = 8 fp16 elements per thread per copy
        self.async_copy_elems = 128 // (self.elem_bytes * 8)  # 8 for fp16
        self.copy_dim1 = self.D // self.async_copy_elems
        self.copy_dim0 = self.num_mma_threads // self.copy_dim1

        # --- n_block: max KV rows per double-buffer slot (padded stride) ---
        if self.q_in_smem:
            # Q stays in smem: KV uses remaining page space
            kv_budget = self.page_size - self.q_tile_bytes
            max_n_block = kv_budget // (2 * self.smem_stride * self.elem_bytes)
        else:
            # Q preloaded to registers: full page available for KV
            max_n_block = self.page_size // (2 * self.smem_stride * self.elem_bytes)

        # Dynamic-N path uses a fixed KV block size so one compiled kernel can
        # be reused across sequence lengths safely. Keep it static, but use a
        # less conservative tile when page budget allows it: 32 rows halves the
        # KV-loop trip count versus 16 without making the shape runtime-
        # dependent. Round down to the legal cpasync/MMA granularity.
        n_block_granularity = max(16, self.copy_dim0)
        max_fixed_n_block = min(32, max_n_block)
        self.n_block = (max_fixed_n_block // n_block_granularity) * n_block_granularity
        if self.n_block < 16:
            self.n_block = 16
        self.kv_tile_bytes = self.n_block * self.smem_stride * self.elem_bytes
        total_kv_smem = 2 * self.kv_tile_bytes
        if self.q_in_smem:
            assert self.q_tile_bytes + total_kv_smem <= self.page_size, (
                f"FlashAttentionSm120Op: Q({self.q_tile_bytes}B) + KV({total_kv_smem}B) "
                f"> page_size ({self.page_size}B)."
            )
        else:
            assert total_kv_smem <= self.page_size, (
                f"FlashAttentionSm120Op: KV double-buffer ({total_kv_smem}B) > page_size ({self.page_size}B)."
            )

        assert self.n_block % self.copy_dim0 == 0, (
            f"n_block={self.n_block} must be divisible by copy_dim0={self.copy_dim0} "
            f"(num_mma_warps={self.num_mma_warps})."
        )

        # DMA loads Q once, then compute handles everything
        self.inner_iters = 1
        self.inner_depth = 1

        # exp2-based softmax
        self.scale_log2e = self.scale_val * 1.4426950408889634074
        # Override compute method (select Q-in-smem vs Q-preload path)
        self.compute = self.compute_mma_q_smem if self.q_in_smem else self.compute_mma

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule(cls, tile_sizes=None, causal=False, page_size=DEFAULT_PAGE_SIZE,
                         kv_group_size=1, **tensors):
        """Schedule cooperative flash attention forward.

        Accepts 4D tensors in BSHD layout or 3D tensors `(BH, M, D)` for
        backward compatibility. 4D tensors are converted to zero-copy BHSD
        views internally before scheduling the underlying kernel.

        Two modes depending on page_size:
        - Q-in-smem: Q persists in smem alongside KV. tile_M can be 2x warps
          (m_reps=2) for better KV reuse. Used when page fits Q(128)+KV.
        - Q-preload: Q preloaded to registers, smem reclaimed for KV.
          tile_M = warps * 16. Used for smaller page_size.
        """
        # 3D backward compat: reshape (BH, M, D) → (1, BH, M, D)
        for name in ("q", "k", "v", "o"):
            t = tensors.get(name)
            if t is not None and t.ndim == 3:
                tensors[name] = t.unsqueeze(0)
        if "lse" in tensors and tensors["lse"] is not None and tensors["lse"].ndim == 2:
            tensors["lse"] = tensors["lse"].unsqueeze(0)
        elif tensors.get("q") is not None and tensors["q"].ndim == 4:
            q = tensors["q"]
            page_size = _effective_page_size(q, page_size)
            for name in ("q", "k", "v", "o"):
                if tensors.get(name) is not None:
                    tensors[name] = _bshd_to_bhsd(tensors[name])

        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        tile_sizes.setdefault("H", 1)
        q = tensors.get("q")
        nw = None
        if q is not None:
            assert q.element_size() == 2, (
                f"FlashAttentionSm120Op requires fp16/bf16, got element_size={q.element_size()}"
            )
            B, H, M, D = q.shape
            elem = q.element_size()

            if "M" not in tile_sizes:
                min_kv_bytes = 2 * 16 * D * elem  # 2 buffers × 16 rows
                smem_pad_elems = 8
                smem_stride = D + smem_pad_elems

                # Target 4 MMA warps (matches SDPA, optimal barrier overhead)
                max_nw = min(4, M // 16)
                nw = 1
                while nw * 2 <= max_nw:
                    nw *= 2

                # Pick mode that maximizes n_block (larger KV blocks = fewer
                # syncs per tile and better memory bandwidth per cpasync).
                # Q-preload reclaims Q smem for KV → nearly always wins.
                preload_n = page_size // (2 * smem_stride * elem)
                tile_M_2x = nw * 32
                q_bytes_2x = tile_M_2x * D * elem
                smem_n = 0
                if (page_size >= q_bytes_2x + min_kv_bytes
                        and M >= tile_M_2x):
                    smem_n = (page_size - q_bytes_2x) // (2 * smem_stride * elem)

                if smem_n > preload_n:
                    tile_M = tile_M_2x  # Q-in-smem: rare, only if page >> Q
                else:
                    tile_M = max(16, nw * 16)  # Q-preload: default

                # For short-prefill Qwen-style shapes under the fixed fused
                # 224-TPB regime, larger pages should spend the smem budget on
                # a fatter M tile rather than leaving the attention block at
                # the smaller 2-warp shape. At 96KB, 64x4 is the best variant
                # we measured for M=128,D=256 and it matches the fused global
                # TPB target exactly.
                if M <= 128 and page_size <= 32 * 1024:
                    tile_M = 16
                    nw = 1
                elif M <= 128 and page_size >= 96 * 1024:
                    tile_M = 64 if M >= 64 else 32
                    nw = 4 if tile_M == 64 else 2
                elif M <= 128 and page_size >= 48 * 1024:
                    tile_M = 32
                    nw = 2

                tile_sizes["M"] = tile_M
        # Auto-allocate lse output if not provided
        if "lse" not in tensors and q is not None:
            B, H, M, _D = q.shape
            tensors["lse"] = torch.empty(B, H, M, dtype=torch.float32, device=q.device)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        if nw is not None:
            ops[0].static_dims["num_mma_warps"] = nw
        if causal:
            ops[0].static_dims["causal"] = 1
        if kv_group_size > 1:
            ops[0].static_dims["kv_group_size"] = kv_group_size

        # Detect strided tensors (e.g., view+permute of GEMM outputs)
        # and store per-dimension strides for correct pointer arithmetic.
        k = tensors.get("k")
        if k is not None and not k.is_contiguous():
            ops[0].static_dims["k_b_stride"] = k.stride(0)
            ops[0].static_dims["k_h_stride"] = k.stride(1)
            ops[0].static_dims["k_n_stride"] = k.stride(2)
        v = tensors.get("v")
        if v is not None and not v.is_contiguous():
            ops[0].static_dims["v_b_stride"] = v.stride(0)
            ops[0].static_dims["v_h_stride"] = v.stride(1)
            ops[0].static_dims["v_n_stride"] = v.stride(2)

        # For strided Q/O (from view+permute), the stride-sorted TMA dim
        # order differs from simple reversal. The op needs to know this to
        # use the correct TMA coordinates in load/store.
        q = tensors.get("q")
        if q is not None and not q.is_contiguous():
            ops[0].static_dims["q_tma_permuted"] = 1
        o = tensors.get("o")
        if o is not None and not o.is_contiguous():
            ops[0].static_dims["o_tma_permuted"] = 1

        return ops

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for the given scheduled ops.

        Uses num_pages=1 because attention does KV pipelining internally via
        cpasync — the framework's ring-buffer page pipeline provides no benefit
        (DMA warp only loads the small Q tile). This gives the op the full smem
        budget instead of splitting across 2 pages.
        """
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS

        tile_m = ops[0].tile_sizes["M"]
        num_mma_warps = ops[0].static_dims.get("num_mma_warps", tile_m // 16)
        threads_per_block = (num_mma_warps + NUM_DMA_WARPS) * 32
        page_size = ops[0].static_dims.get("page_size", DEFAULT_PAGE_SIZE)
        return MegakernelConfig(
            threads_per_block=threads_per_block,
            page_size=page_size,
            num_pages=1,
        )

    # =========================================================================
    # Forward Load (DMA warp: TMA Q only)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_H, tile_M, tile_D, q_tma, q_tma_gmem, work_mbar):
        """TMA Q load into page (single shot, swizzled for LdMatrix reads)."""
        from machete.megakernel.interpreter import mbarrier_arrive_expect_tx

        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)

        _q_swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)

        # TMA dim order depends on whether Q is strided (from view+permute).
        # Contiguous: reversed dims (D, M, H, B). Strided: stride-sorted (D, H, M, B).
        # Shape pre-computed in __init__ (can't define Python tuples inside JIT if).
        sQ = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem),
                _q_swz,
                dtype=self.q_dtype,
            ),
            cute.make_layout(self._q_tma_smem_shape),
        )
        gQ = cute.local_tile(
            q_tma_gmem,
            self._q_tma_smem_shape,
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
        if self.q_tma_permuted:
            cute.copy(q_tma, tQgQ[(None, tile_D, tile_H, tile_M, tile_B)], tQsQ, tma_bar_ptr=mbar_ptr)
        else:
            cute.copy(q_tma, tQgQ[(None, tile_D, tile_M, tile_H, tile_B)], tQsQ, tma_bar_ptr=mbar_ptr)

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
    def compute_mma(self, page_ptr, tile_B, tile_H, tile_M, tile_D, q, k, v, o, lse,
                    op_config_ptr):
        """Cooperative flash attention: MMA warps do both cpasync loads and MMA.

        Smem layout phases:
          Phase 0: [Q: tile_M×D×2B]  (TMA-loaded, preloaded to registers)
          Phase 1: [K buf: n_block×D×2B] [V buf: n_block×D×2B]  (Q reclaimed)
          Phase 2: [O: tile_M×D×2B]  (overwrites KV, for TMA store)

        Q preload to registers frees the full page for KV double-buffering,
        doubling n_block vs the Q-persistent layout.

        """
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))
        runtime_N = config_dim_i32(op_config_ptr, "N", type(self))
        runtime_k_b_stride = Int32(self.k_b_stride) if self.k_b_stride is not None else Int32(self.H_kv) * runtime_N * Int32(self.D)
        runtime_k_h_stride = Int32(self.k_h_stride) if self.k_h_stride is not None else runtime_N * Int32(self.D)
        runtime_v_b_stride = Int32(self.v_b_stride) if self.v_b_stride is not None else Int32(self.H_kv) * runtime_N * Int32(self.D)
        runtime_v_h_stride = Int32(self.v_h_stride) if self.v_h_stride is not None else runtime_N * Int32(self.D)
        runtime_num_kv_blocks = (runtime_N + Int32(self.n_block - 1)) // Int32(self.n_block)

        if warp_idx < Int32(self.num_mma_warps):
            # === MMA setup (multi-warp) ===
            mma_op = warp.MmaF16BF16Op(self.q_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((self.num_mma_warps, 1, 1)),
                permutation_mnk=(self.num_mma_warps * 16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)

            # === Swizzle + LdMatrix setup ===
            swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)

            # === Q smem tensor (persistent, swizzled for LdMatrix) ===
            sQ = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                    swz,
                    dtype=self.q_dtype,
                ),
                cute.make_layout((self.tile_size_M, self.D), stride=(self.D, 1)),
            )
            _tCsQ = thr_mma.partition_A(sQ)
            tCrQ = tiled_mma.make_fragment_A(_tCsQ)

            # LdMatrix copy atom for Q (A-side of MMA)
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

            # === CopyUniversal for O write to swizzled smem ===
            smem_copy_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.q_dtype)
            smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
            smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)

            # =============================================================
            # Phase 0: Preload Q from TMA smem → registers, then reclaim
            # =============================================================

            # Preload ALL Q k-blocks into registers. After this, Q smem
            # at page_ptr is free for KV double-buffering (2x n_block).
            for _qkb in cutlass.range_constexpr(self.D // 16):
                cute.copy(smem_tiled_copy_Q, tQsQ[None, None, _qkb], tQrQ_view[None, None, _qkb])

            # Barrier: all warps must finish Q reads before K overwrites smem
            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            # =============================================================
            # Phase 1: Setup KV at page_ptr (reclaimed Q space)
            # =============================================================

            # KV buffer base: reclaim Q smem → full page for KV
            _kv_base = page_ptr

            # K smem tensor + LdMatrix fragments (buf0, row-padded)
            _sK = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _kv_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.n_block, self.D), stride=(self.smem_stride, 1)),
            )
            _tCsK = thr_mma.partition_B(_sK)
            tCrK = tiled_mma.make_fragment_B(_tCsK)
            tKrK_view = smem_thr_copy_K.retile(tCrK)
            tKsK = smem_thr_copy_K.partition_S(_sK)

            # V smem tensor + LdMatrix fragments (buf1, row-padded, transposed)
            _buf1_base = _kv_base + Int32(self.kv_tile_bytes)
            _sVt = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _buf1_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.D, self.n_block), stride=(1, self.smem_stride)),
            )
            _tBsVt = thr_mma.partition_B(_sVt)
            tBrVt = tiled_mma.make_fragment_B(_tBsVt)
            tVrVt_view = smem_thr_copy_Vt.retile(tBrVt)
            tVsVt = smem_thr_copy_Vt.partition_S(_sVt)

            # cpasync tiled copy setup
            async_copy_atom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), self.q_dtype, num_bits_per_copy=128)
            copy_thread_layout = cute.make_layout((self.copy_dim0, self.copy_dim1), stride=(self.copy_dim1, 1))
            copy_value_layout = cute.make_layout((1, self.async_copy_elems))
            gmem_tiled_copy = cute.make_tiled_copy_tv(async_copy_atom, copy_thread_layout, copy_value_layout)
            thr_copy = gmem_tiled_copy.get_slice(tidx)

            # cpasync smem destinations (buf0=K at page_ptr, buf1=V, row-padded)
            sK_cp = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _kv_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.n_block, self.D), stride=(self.smem_stride, 1)),
            )
            sV_cp = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _buf1_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.n_block, self.D), stride=(self.smem_stride, 1)),
            )
            tKsK_cp = thr_copy.partition_D(sK_cp)
            tVsV_cp = thr_copy.partition_D(sV_cp)

            # cpasync global sources: K and V for current head
            kv_h = tile_H // Int32(self.kv_group_size)
            k_head_ptr = (k.iterator + tile_B * runtime_k_b_stride + kv_h * runtime_k_h_stride).align(16)
            v_head_ptr = (v.iterator + tile_B * runtime_v_b_stride + kv_h * runtime_v_h_stride).align(16)
            gK_head = cute.make_tensor(k_head_ptr, cute.make_layout((runtime_N, self.D), stride=(self.k_n_stride, 1)))
            gV_head = cute.make_tensor(v_head_ptr, cute.make_layout((runtime_N, self.D), stride=(self.v_n_stride, 1)))

            # P register fragment + MMA view (pre-allocated)
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
            # Phase 2: Pipelined KV loop
            # =============================================================

            # Causal block skipping: compute effective number of KV blocks
            # for this M-tile. Row i can attend to columns 0..(i + N - M),
            # so we only need KV blocks up to the last attending column.
            num_kv_blocks_eff = runtime_num_kv_blocks
            if self.causal:
                _last_row = tile_M * Int32(self.tile_size_M) + Int32(self.tile_size_M - 1)
                _max_col = _last_row + (runtime_N - runtime_M)
                num_kv_blocks_eff = (_max_col + Int32(self.n_block)) // Int32(self.n_block)
                if num_kv_blocks_eff > runtime_num_kv_blocks:
                    num_kv_blocks_eff = runtime_num_kv_blocks

            # Prologue: cpasync K[0] → buf0
            gK_block0 = cute.local_tile(gK_head, (self.n_block, self.D), (Int32(0), Int32(0)))
            tKgK0 = thr_copy.partition_S(gK_block0)
            for ci in cutlass.range_constexpr(cute.size(tKsK_cp.shape[2])):
                cute.copy(gmem_tiled_copy, tKgK0[None, None, ci], tKsK_cp[None, None, ci])
            cute.arch.cp_async_commit_group()

            kv_idx = Int32(0)
            while kv_idx < num_kv_blocks_eff:
                kv_start = kv_idx * Int32(self.n_block)

                # --- Wait for K[i] in buf0 ---
                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # --- Start V[i] cpasync → buf1  (group 1) ---
                gV_block = cute.local_tile(gV_head, (self.n_block, self.D), (kv_idx, Int32(0)))
                tVgV = thr_copy.partition_S(gV_block)
                for ci in cutlass.range_constexpr(cute.size(tVsV_cp.shape[2])):
                    cute.copy(gmem_tiled_copy, tVgV[None, None, ci], tVsV_cp[None, None, ci])
                cute.arch.cp_async_commit_group()

                # --- S GEMM with K-only LdMatrix pipeline (Q in registers) ---
                # K[i] in buf0 is consumed here via LdMatrix into registers.
                # After this loop, buf0 is free for K[i+1].
                acc_S.fill(0.0)
                cute.copy(smem_tiled_copy_K, tKsK[None, None, 0], tKrK_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.D // 16):
                    kb_next = (kb + 1) % (self.D // 16)
                    cute.copy(smem_tiled_copy_K, tKsK[None, None, kb_next], tKrK_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_S, tCrQ[None, None, kb], tCrK[None, None, kb], acc_S)

                # --- Start K[i+1] cpasync → buf0 (overlaps with V wait + softmax + O GEMM) ---
                # After S GEMM, buf0 (K) is fully consumed. Start loading next K
                # before waiting for V so K[i+1] load overlaps with the rest of
                # this iteration. Commit as a separate group so wait_group(1)
                # drains V while K[i+1] stays in flight.
                # On the last iteration we still commit an empty group so the
                # wait_group(1) count is consistent.
                if kv_idx + Int32(1) < num_kv_blocks_eff:
                    gK_next = cute.local_tile(gK_head, (self.n_block, self.D), (kv_idx + Int32(1), Int32(0)))
                    tKgK_next = thr_copy.partition_S(gK_next)
                    for ci in cutlass.range_constexpr(cute.size(tKsK_cp.shape[2])):
                        cute.copy(gmem_tiled_copy, tKgK_next[None, None, ci], tKsK_cp[None, None, ci])
                cute.arch.cp_async_commit_group()

                # --- Wait for V[i] in buf1 (leave K[i+1] in flight) ---
                # 2 groups outstanding: V (group 1) + K[i+1] (group 2).
                # wait_group(1) drains until ≤1 outstanding → V completes, K[i+1] stays.
                # On last iteration: empty K group + V → wait_group(1) drains V.
                cute.arch.cp_async_wait_group(1)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # --- Masking (boundary blocks only) ---
                acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
                acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

                # N-boundary mask (only last KV block)
                if kv_start + Int32(self.n_block) > runtime_N:
                    for r in cutlass.range_constexpr(num_rows):
                        for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                            col_idx = tScS_mn[0, c][1]
                            global_col = kv_start + Int32(col_idx)
                            if global_col >= runtime_N:
                                acc_S_mn[r, c] = Float32(-1e30)

                # Causal mask (only blocks near diagonal)
                if self.causal:
                    last_blk_col = kv_start + Int32(self.n_block - 1)
                    first_row = tile_M * Int32(self.tile_size_M)
                    if last_blk_col > first_row + (runtime_N - runtime_M):
                        for r in cutlass.range_constexpr(num_rows):
                            row_idx = tScS_mn[r, 0][0]
                            global_row = tile_M * Int32(self.tile_size_M) + Int32(row_idx)
                            for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                                col_idx = tScS_mn[0, c][1]
                                global_col = kv_start + Int32(col_idx)
                                if global_col > global_row + (runtime_N - runtime_M):
                                    acc_S_mn[r, c] = Float32(-1e30)

                # --- Online softmax ---
                corrections = cute.make_fragment(cute.make_layout(num_rows), Float32)
                for r in cutlass.range_constexpr(num_rows):
                    acc_S_row = acc_S_mn[r, None].load()
                    row_max_cur = acc_S_row.reduce(cute.ReductionOp.MAX, Float32(-1e30), 0)
                    row_max_cur = self._threadquad_reduce_max(row_max_cur)

                    m_old = row_max[r]
                    m_new = cute.arch.fmax(m_old, row_max_cur)

                    acc_scale_ = (m_old - m_new) * Float32(self.scale_log2e)
                    # Clamp to PTX ex2.approx valid range [-126, 128] to avoid
                    # NaN on first iteration (m_old=-1e30 → acc_scale_≈-1e30).
                    correction = cute.math.exp2(cute.arch.fmax(acc_scale_, Float32(-126.0)), fastmath=True)
                    row_sum[r] = row_sum[r] * correction
                    corrections[r] = correction

                    acc_S_row_exp = cute.math.exp2(
                        acc_S_row * Float32(self.scale_log2e) - m_new * Float32(self.scale_log2e), fastmath=True
                    )
                    acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                    row_sum[r] = row_sum[r] + acc_S_row_sum
                    row_max[r] = m_new
                    acc_S_mn[r, None] = acc_S_row_exp

                # O rescale (unconditional — CuTe DSL dynamic-if executes both
                # branches anyway, so vote_all_sync + conditional was pure overhead)
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
            # Phase 2: Normalize O and write to smem for TMA store
            # =============================================================
            acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
            for r in cutlass.range_constexpr(num_rows):
                row_sum[r] = self._threadquad_reduce_sum(row_sum[r])

            # Write LSE = row_max / log2e + log(row_sum) = row_max * scale_val / (scale_val * log2e) + log(row_sum)
            # Since row_max is in raw score space (pre-scale), LSE = row_max + log(row_sum)
            # Actually: softmax used S * scale_log2e, so row_max is in log2-scaled space.
            # P = exp2(S * scale_log2e - row_max * scale_log2e)
            # = exp(S * scale - row_max * scale)  [since exp2(x * log2e) = exp(x)]
            # Wait — row_max stores raw S values (not scaled). Let's trace:
            #   acc_S_row values are raw Q@K^T scores
            #   row_max_cur = max(acc_S_row)  → raw score space
            #   m_new = fmax(m_old, row_max_cur)  → raw score space
            #   exp2(acc_S_row * scale_log2e - m_new * scale_log2e) = exp((acc_S_row - m_new) * scale)
            # So P_ij = exp((S_ij - row_max_i) * scale)
            # True softmax: P_ij = exp(S_ij * scale) / sum_j exp(S_ij * scale)
            #             = exp((S_ij - row_max_i) * scale) / sum_j exp((S_ij - row_max_i) * scale)
            # row_sum = sum_j exp((S_ij - row_max_i) * scale)
            # LSE_i = log(sum_j exp(S_ij * scale)) = row_max_i * scale + log(row_sum_i)
            # Write LSE to global via CuTe tensor. Each thread in the MMA partition owns
            # specific rows identified by the identity tensor tScS_mn[r, 0][0].
            # Only one thread per quad writes (threads 0,1,2,3 in a quad share the same row).
            lse_head_ptr = lse.iterator + (tile_B * Int32(self.H) + tile_H) * runtime_M
            g_lse = cute.make_tensor(lse_head_ptr, cute.make_layout(runtime_M))
            lane_in_quad = tidx % Int32(4)
            for r in cutlass.range_constexpr(num_rows):
                if lane_in_quad == Int32(0):
                    row_idx = tScS_mn[r, 0][0]
                    global_row = tile_M * Int32(self.tile_size_M) + Int32(row_idx)
                    if global_row < runtime_M:
                        lse_val = row_max[r] * Float32(self.scale_val) + cute.math.log(row_sum[r])
                        g_lse[global_row] = lse_val

            for r in cutlass.range_constexpr(num_rows):
                inv_sum = cute.arch.rcp_approx(row_sum[r])
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * inv_sum

            # Write O to smem (at page_ptr, swizzled layout)
            # No barrier needed: each warp writes to disjoint M rows.
            # DMA store warp waits on framework's compute_done barrier.
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
    # Forward Compute -- Q-in-smem mode (m_reps > 1)
    # =========================================================================

    @cute.jit
    def compute_mma_q_smem(self, page_ptr, tile_B, tile_H, tile_M, tile_D, q, k, v, o, lse,
                           op_config_ptr):
        """Cooperative flash attention with Q persistent in smem (m_reps > 1).

        Smem layout (persistent throughout compute):
          [Q: tile_M×D×2B]  (TMA-loaded, read per k-block via LdMatrix)
          [K buf: n_block×D×2B]  (cpasync double-buffer slot 0)
          [V buf: n_block×D×2B]  (cpasync double-buffer slot 1)

        Q stays in smem for the entire KV loop. Each S GEMM k-block reloads Q
        via LdMatrix, trading smem bandwidth for ~28 fewer registers/thread.
        With m_reps=2 (tile_M=128, 4 warps), this halves total tile count
        vs Q-preload mode, reducing total KV bandwidth by 2x.
        """
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        runtime_M = config_dim_i32(op_config_ptr, "M", type(self))
        runtime_N = config_dim_i32(op_config_ptr, "N", type(self))
        runtime_k_b_stride = Int32(self.k_b_stride) if self.k_b_stride is not None else Int32(self.H_kv) * runtime_N * Int32(self.D)
        runtime_k_h_stride = Int32(self.k_h_stride) if self.k_h_stride is not None else runtime_N * Int32(self.D)
        runtime_v_b_stride = Int32(self.v_b_stride) if self.v_b_stride is not None else Int32(self.H_kv) * runtime_N * Int32(self.D)
        runtime_v_h_stride = Int32(self.v_h_stride) if self.v_h_stride is not None else runtime_N * Int32(self.D)
        runtime_num_kv_blocks = (runtime_N + Int32(self.n_block - 1)) // Int32(self.n_block)

        if warp_idx < Int32(self.num_mma_warps):
            # === MMA setup (multi-warp, m_reps > 1) ===
            mma_op = warp.MmaF16BF16Op(self.q_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((self.num_mma_warps, 1, 1)),
                permutation_mnk=(self.tile_size_M, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)

            # === Swizzle + LdMatrix setup ===
            swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)

            # === Q smem tensor (persistent, swizzled for LdMatrix) ===
            sQ = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem, assumed_align=128),
                    swz,
                    dtype=self.q_dtype,
                ),
                cute.make_layout((self.tile_size_M, self.D), stride=(self.D, 1)),
            )
            _tCsQ = thr_mma.partition_A(sQ)
            tCrQ = tiled_mma.make_fragment_A(_tCsQ)

            # LdMatrix copy atoms
            smem_copy_atom_Q = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.q_dtype
            )
            smem_copy_atom_K = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.q_dtype
            )
            smem_copy_atom_Vt = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.q_dtype
            )
            smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
            smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
            smem_tiled_copy_Vt = cute.make_tiled_copy_B(smem_copy_atom_Vt, tiled_mma)
            smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
            smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
            smem_thr_copy_Vt = smem_tiled_copy_Vt.get_slice(tidx)

            tQrQ_view = smem_thr_copy_Q.retile(tCrQ)
            tQsQ = smem_thr_copy_Q.partition_S(sQ)

            # CopyUniversal for O write to swizzled smem
            smem_copy_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.q_dtype)
            smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
            smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)

            # =============================================================
            # KV smem setup (after Q, persistent)
            # =============================================================

            _kv_base = page_ptr + Int32(self.q_tile_bytes)

            # K smem tensor + LdMatrix fragments (buf0, row-padded)
            _sK = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _kv_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.n_block, self.D), stride=(self.smem_stride, 1)),
            )
            _tCsK = thr_mma.partition_B(_sK)
            tCrK = tiled_mma.make_fragment_B(_tCsK)
            tKrK_view = smem_thr_copy_K.retile(tCrK)
            tKsK = smem_thr_copy_K.partition_S(_sK)

            # V smem tensor + LdMatrix fragments (buf1, row-padded, transposed)
            _buf1_base = _kv_base + Int32(self.kv_tile_bytes)
            _sVt = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _buf1_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.D, self.n_block), stride=(1, self.smem_stride)),
            )
            _tBsVt = thr_mma.partition_B(_sVt)
            tBrVt = tiled_mma.make_fragment_B(_tBsVt)
            tVrVt_view = smem_thr_copy_Vt.retile(tBrVt)
            tVsVt = smem_thr_copy_Vt.partition_S(_sVt)

            # cpasync tiled copy setup
            async_copy_atom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), self.q_dtype, num_bits_per_copy=128)
            copy_thread_layout = cute.make_layout((self.copy_dim0, self.copy_dim1), stride=(self.copy_dim1, 1))
            copy_value_layout = cute.make_layout((1, self.async_copy_elems))
            gmem_tiled_copy = cute.make_tiled_copy_tv(async_copy_atom, copy_thread_layout, copy_value_layout)
            thr_copy = gmem_tiled_copy.get_slice(tidx)

            # cpasync smem destinations (row-padded)
            sK_cp = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _kv_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.n_block, self.D), stride=(self.smem_stride, 1)),
            )
            sV_cp = cute.make_tensor(
                cute.make_ptr(self.q_dtype, _buf1_base, cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.n_block, self.D), stride=(self.smem_stride, 1)),
            )
            tKsK_cp = thr_copy.partition_D(sK_cp)
            tVsV_cp = thr_copy.partition_D(sV_cp)

            # cpasync global sources
            kv_h = tile_H // Int32(self.kv_group_size)
            k_head_ptr = (k.iterator + tile_B * runtime_k_b_stride + kv_h * runtime_k_h_stride).align(16)
            v_head_ptr = (v.iterator + tile_B * runtime_v_b_stride + kv_h * runtime_v_h_stride).align(16)
            gK_head = cute.make_tensor(k_head_ptr, cute.make_layout((runtime_N, self.D), stride=(self.k_n_stride, 1)))
            gV_head = cute.make_tensor(v_head_ptr, cute.make_layout((runtime_N, self.D), stride=(self.v_n_stride, 1)))

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
            # Pipelined KV loop (Q read from smem each k-block)
            # =============================================================

            num_kv_blocks_eff = runtime_num_kv_blocks
            if self.causal:
                _last_row = tile_M * Int32(self.tile_size_M) + Int32(self.tile_size_M - 1)
                _max_col = _last_row + (runtime_N - runtime_M)
                num_kv_blocks_eff = (_max_col + Int32(self.n_block)) // Int32(self.n_block)
                if num_kv_blocks_eff > runtime_num_kv_blocks:
                    num_kv_blocks_eff = runtime_num_kv_blocks

            # Prologue: cpasync K[0] → buf0
            gK_block0 = cute.local_tile(gK_head, (self.n_block, self.D), (Int32(0), Int32(0)))
            tKgK0 = thr_copy.partition_S(gK_block0)
            for ci in cutlass.range_constexpr(cute.size(tKsK_cp.shape[2])):
                cute.copy(gmem_tiled_copy, tKgK0[None, None, ci], tKsK_cp[None, None, ci])
            cute.arch.cp_async_commit_group()

            kv_idx = Int32(0)
            while kv_idx < num_kv_blocks_eff:
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

                # --- S GEMM: Q from smem + K from smem, pipelined LdMatrix ---
                acc_S.fill(0.0)
                cute.copy(smem_tiled_copy_Q, tQsQ[None, None, 0], tQrQ_view[None, None, 0])
                cute.copy(smem_tiled_copy_K, tKsK[None, None, 0], tKrK_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.D // 16):
                    kb_next = (kb + 1) % (self.D // 16)
                    cute.copy(smem_tiled_copy_Q, tQsQ[None, None, kb_next], tQrQ_view[None, None, kb_next])
                    cute.copy(smem_tiled_copy_K, tKsK[None, None, kb_next], tKrK_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_S, tCrQ[None, None, kb], tCrK[None, None, kb], acc_S)

                # --- Start K[i+1] cpasync → buf0 (overlaps with V wait + softmax + O GEMM) ---
                if kv_idx + Int32(1) < num_kv_blocks_eff:
                    gK_next = cute.local_tile(gK_head, (self.n_block, self.D), (kv_idx + Int32(1), Int32(0)))
                    tKgK_next = thr_copy.partition_S(gK_next)
                    for ci in cutlass.range_constexpr(cute.size(tKsK_cp.shape[2])):
                        cute.copy(gmem_tiled_copy, tKgK_next[None, None, ci], tKsK_cp[None, None, ci])
                cute.arch.cp_async_commit_group()

                # --- Wait for V[i] in buf1 (leave K[i+1] in flight) ---
                cute.arch.cp_async_wait_group(1)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # --- Masking ---
                acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
                acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

                if kv_start + Int32(self.n_block) > runtime_N:
                    for r in cutlass.range_constexpr(num_rows):
                        for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                            col_idx = tScS_mn[0, c][1]
                            global_col = kv_start + Int32(col_idx)
                            if global_col >= runtime_N:
                                acc_S_mn[r, c] = Float32(-1e30)

                if self.causal:
                    last_blk_col = kv_start + Int32(self.n_block - 1)
                    first_row = tile_M * Int32(self.tile_size_M)
                    if last_blk_col > first_row + (runtime_N - runtime_M):
                        for r in cutlass.range_constexpr(num_rows):
                            row_idx = tScS_mn[r, 0][0]
                            global_row = tile_M * Int32(self.tile_size_M) + Int32(row_idx)
                            for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                                col_idx = tScS_mn[0, c][1]
                                global_col = kv_start + Int32(col_idx)
                                if global_col > global_row + (runtime_N - runtime_M):
                                    acc_S_mn[r, c] = Float32(-1e30)

                # --- Online softmax ---
                corrections = cute.make_fragment(cute.make_layout(num_rows), Float32)
                for r in cutlass.range_constexpr(num_rows):
                    acc_S_row = acc_S_mn[r, None].load()
                    row_max_cur = acc_S_row.reduce(cute.ReductionOp.MAX, Float32(-1e30), 0)
                    row_max_cur = self._threadquad_reduce_max(row_max_cur)

                    m_old = row_max[r]
                    m_new = cute.arch.fmax(m_old, row_max_cur)

                    acc_scale_ = (m_old - m_new) * Float32(self.scale_log2e)
                    correction = cute.math.exp2(cute.arch.fmax(acc_scale_, Float32(-126.0)), fastmath=True)
                    row_sum[r] = row_sum[r] * correction
                    corrections[r] = correction

                    acc_S_row_exp = cute.math.exp2(
                        acc_S_row * Float32(self.scale_log2e) - m_new * Float32(self.scale_log2e), fastmath=True
                    )
                    acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, Float32(0.0), 0)
                    row_sum[r] = row_sum[r] + acc_S_row_sum
                    row_max[r] = m_new
                    acc_S_mn[r, None] = acc_S_row_exp

                # O rescale
                for r in cutlass.range_constexpr(num_rows):
                    acc_O_mn[r, None] = acc_O_mn[r, None].load() * corrections[r]

                # --- P conversion + O GEMM ---
                rP.store(acc_S.load().to(self.q_dtype))

                cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, 0], tVrVt_view[None, None, 0])
                for kb in cutlass.range_constexpr(self.n_block // 16):
                    kb_next = (kb + 1) % (self.n_block // 16)
                    cute.copy(smem_tiled_copy_Vt, tVsVt[None, None, kb_next], tVrVt_view[None, None, kb_next])
                    cute.gemm(tiled_mma, acc_O, tOrS[None, None, kb], tBrVt[None, None, kb], acc_O)

                kv_idx = kv_idx + Int32(1)

            # =============================================================
            # Normalize O and write to smem for TMA store
            # =============================================================
            acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
            for r in cutlass.range_constexpr(num_rows):
                row_sum[r] = self._threadquad_reduce_sum(row_sum[r])

            lse_head_ptr = lse.iterator + (tile_B * Int32(self.H) + tile_H) * runtime_M
            g_lse = cute.make_tensor(lse_head_ptr, cute.make_layout(runtime_M))
            lane_in_quad = tidx % Int32(4)
            for r in cutlass.range_constexpr(num_rows):
                if lane_in_quad == Int32(0):
                    row_idx = tScS_mn[r, 0][0]
                    global_row = tile_M * Int32(self.tile_size_M) + Int32(row_idx)
                    if global_row < runtime_M:
                        lse_val = row_max[r] * Float32(self.scale_val) + cute.math.log(row_sum[r])
                        g_lse[global_row] = lse_val

            for r in cutlass.range_constexpr(num_rows):
                inv_sum = cute.arch.rcp_approx(row_sum[r])
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * inv_sum

            # Write O to smem (overwrites Q, which is no longer needed)
            _o_swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)
            sO = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem), _o_swz, dtype=self.q_dtype
                ),
                cute.make_layout((self.tile_size_M, self.D), stride=(self.D, 1)),
            )
            tCrO_q = cute.make_fragment_like(acc_O, self.q_dtype)
            for i in cutlass.range_constexpr(cute.size(acc_O)):
                tCrO_q[i] = acc_O[i].to(self.q_dtype)
            tOrO = smem_thr_copy_O.retile(tCrO_q)
            tOsO = smem_thr_copy_O.partition_D(sO)
            cute.copy(smem_tiled_copy_O, tOrO, tOsO)

    # =========================================================================
    # Forward Store (3D TMA S->G for O)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_B, tile_H, tile_M, tile_D, o_tma, o_tma_gmem):
        """TMA store of O from shared to global memory (swizzled)."""
        _o_swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)

        sO = cute.make_tensor(
            cute.recast_ptr(cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem), _o_swz, dtype=self.q_dtype),
            cute.make_layout(self._o_tma_smem_shape),
        )
        gO = cute.local_tile(
            o_tma_gmem,
            self._o_tma_smem_shape,
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
            if self.o_tma_permuted:
                cute.copy(o_tma, tOsO, tOgO[(None, tile_D, tile_H, tile_M, tile_B)])
            else:
                cute.copy(o_tma, tOsO, tOgO[(None, tile_D, tile_M, tile_H, tile_B)])


__all__ = ["FlashAttentionSm120Op"]
