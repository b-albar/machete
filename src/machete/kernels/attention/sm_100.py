# Copyright (c) 2025, Machete Authors
"""
Flash Attention 4 Op for the Megakernel (SM100 Blackwell UMMA).

Computes scaled dot-product attention with online softmax using tcgen05.mma:
    O[BH, M, D] = softmax(Q[BH, M, D] @ K[BH, N, D]^T / sqrt(D)) @ V[BH, N, D]

Architecture (FA4-style warp specialization):
    Warp 0-3:  Softmax warps — read S from TMEM, compute online softmax, write P to TMEM
    Warp 4-7:  Correction warps — rescale O in TMEM when row_max changes, final epilogue
    Warp 8:    MMA warp — issues tcgen05.mma (UMMA) for QK and PV GEMMs
    Warp 9:    KV loader warp — cpasync loads K/V tiles into smem
    Warp 10-12: Framework DMA warps (controller + loader + store)

TMEM layout (256 columns, q_stage=1):
    [S: 0..127]   QK attention scores (128 cols = n_block_size)
    [O: 128..255]  PV accumulator (128 cols = head_dim)
    P overlaps S at offset 64 (half-precision packs into half the columns)

Smem page layout (page_size = 98304 = 96KB):
    [sQ: tile_M × D × 2B]     Q tile (loaded by framework TMA)
    [sKV: n_block × D × 2B]   K/V single buffer (K and V alternate)
    [sScale: tile_M × 4B]     row_max scale for correction warps
    [mbarriers: 9 × 8B]       op-managed mbarriers

Usage:
    from machete.kernels.attention import FlashAttentionSm100Op
    from machete.megakernel import Megakernel, MegakernelConfig

    ops = FlashAttentionSm100Op.schedule(q=q, k=k, v=v, o=o, page_size=98304)
    config = FlashAttentionSm100Op.kernel_config(ops)
    kernel = Megakernel(ops, config=config)
    kernel.run()
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu import tcgen05

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import (
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_arrive,
    mbarrier_arrive_expect_tx,
    mbarrier_wait,
    named_barrier_sync,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

M_BLOCK = 128        # Q tile rows (fixed for UMMA 128×128)
N_BLOCK = 128        # KV tile rows (fixed for UMMA 128×128)
NUM_SOFTMAX_WARPS = 4   # Warps 0-3: softmax
NUM_CORR_WARPS = 4      # Warps 4-7: correction/epilogue
NUM_MMA_WARPS_COMPUTE = 1   # Warp 8: UMMA issuer
NUM_LOAD_WARPS = 1      # Warp 9: internal KV TMA loader
NUM_MMA_TOTAL = NUM_SOFTMAX_WARPS + NUM_CORR_WARPS + NUM_MMA_WARPS_COMPUTE + NUM_LOAD_WARPS  # 10

# TMEM column offsets (q_stage=1)
TMEM_S_OFFSET = 0       # S accumulator: columns 0..127
TMEM_O_OFFSET = 128     # O accumulator: columns 128..255
TMEM_P_OFFSET = 64      # P (half-prec) overlaps S at col 64
TMEM_ALLOC_COLS = 256    # Total TMEM columns needed

# Mbarrier layout within page (after Q + single KV buffer + sScale)
# 0: K_ready      — KV loader → MMA warp (K data ready in sKV)
# 1: K_consumed   — MMA warp → KV loader (K consumed, safe to overwrite)
# 2: V_ready      — KV loader → MMA warp (V data ready in sKV)
# 3: V_consumed   — MMA warp → KV loader (V consumed, safe to overwrite)
# 4: S_full       — MMA warp → softmax warps (S ready in TMEM)
# 5: P_full_O_rescaled — softmax+correction → MMA warp (P ready, O rescaled)
# 6: softmax_corr_full — softmax → correction (scale ready in smem)
# 7: softmax_corr_empty — correction → softmax (done reading scale)
# 8: O_full       — MMA warp → correction (final O ready in TMEM)
NUM_MBARRIERS = 9
MBAR_BYTES = NUM_MBARRIERS * 8

# Default page size for FA4 on B200 (228KB smem, need 2 pages)
FA4_PAGE_SIZE = 98304  # 96KB


class FlashAttentionSm100Op(Op):
    """Flash Attention 4 operation using SM100 UMMA (tcgen05.mma).

    Tensors:
        q: (BH, M, D) — query  (fp16 or bf16)
        k: (BH_kv, N, D) — key
        v: (BH_kv, N, D) — value
        o: (BH, M, D) — output
        lse: (BH, M) — log-sum-exp (optional, fp32)

    Tiling:
        tile_BH=1, tile_M=128 (fixed for UMMA), tile_D=D (full head dim).

    Internal KV loop in compute() with cpasync KV loading by loader warp.
    """

    reads = {
        "q": (None, ("B", "H", "M", "D")),
        "k": (None, ("B", "H_kv", "N", "D")),
        "v": (None, ("B", "H_kv", "N", "D")),
    }
    writes = {
        "o": (None, ("B", "H", "M", "D")),
        "lse": (cutlass.Float32, ("B", "H", "M")),
    }
    tile = ("B", "H", "M", "D")
    dynamic_dims = ("B", "M", "N")

    tma_loads = {"q"}
    tma_stores = {"o"}

    # ------------------------------------------------------------------
    # TMA configuration
    # ------------------------------------------------------------------

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape,
                                tile_sizes, static_dims):
        """Swizzled smem layout for O TMA store descriptor."""
        if tensor_name != "o":
            return None
        D = static_dims.get("D", tile_sizes.get("D", 128))
        if D >= 64:
            B = 3
        elif D >= 32:
            B = 2
        else:
            B = 1
        dims = tma_tile_shape  # (D, tile_M, 1, ...) after reversal
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

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(self, **config):
        super().__init__(**config)
        self.causal = getattr(self, 'causal', 0)
        self.page_size = getattr(self, 'page_size', FA4_PAGE_SIZE)
        self.kv_group_size = getattr(self, 'kv_group_size', 1)

        # 4D: store H, H_kv and per-dimension KV strides
        self.H = getattr(self, 'H', self.tile_size_B * self.tile_size_H)
        self.H_kv = getattr(self, 'H_kv', self.H // self.kv_group_size)
        self.k_b_stride = getattr(self, 'k_b_stride', self.H_kv * self.N * self.D)
        self.k_h_stride = getattr(self, 'k_h_stride', self.N * self.D)
        self.k_n_stride = getattr(self, 'k_n_stride', self.D)
        self.v_b_stride = getattr(self, 'v_b_stride', self.H_kv * self.N * self.D)
        self.v_h_stride = getattr(self, 'v_h_stride', self.N * self.D)
        self.v_n_stride = getattr(self, 'v_n_stride', self.D)

        # TMA coordinate order flag for strided Q/O (see sm_120.py for details)
        self.q_tma_permuted = bool(getattr(self, 'q_tma_permuted', 0))
        self.o_tma_permuted = bool(getattr(self, 'o_tma_permuted', 0))

        assert self.q_dtype in (cutlass.Float16, cutlass.BFloat16), (
            f"FlashAttentionSm100Op requires fp16 or bf16, got {self.q_dtype}")
        self.elem_bytes = 2

        # Fixed tile sizes for UMMA
        self.m_block = M_BLOCK
        self.n_block = N_BLOCK

        # Pre-compute TMA smem shapes outside JIT (CuTe DSL can't define
        # Python tuples inside dynamic if — they don't escape SCF regions).
        if self.q_tma_permuted:
            self._q_tma_smem_shape = (self.D, 1, self.m_block, 1)
        else:
            self._q_tma_smem_shape = (self.D, self.m_block, 1, 1)
        if self.o_tma_permuted:
            self._o_tma_smem_shape = (self.D, 1, self.m_block, 1)
        else:
            self._o_tma_smem_shape = (self.D, self.m_block, 1, 1)
        assert self.tile_size_M == self.m_block, (
            f"FlashAttentionSm100Op: tile_size_M must be {self.m_block}")
        assert self.D >= 16 and self.D % 16 == 0

        # Softmax scale
        self.scale_val = 1.0 / (self.D ** 0.5)
        self.scale_log2e = self.scale_val * 1.4426950408889634074

        # KV block count
        self.num_kv_blocks = (self.N + self.n_block - 1) // self.n_block

        # Smem layout sizes
        self.q_tile_bytes = self.m_block * self.D * self.elem_bytes
        self.kv_tile_bytes = self.n_block * self.D * self.elem_bytes
        self.scale_bytes = self.m_block * 4  # fp32 per row

        # Smem offsets within page (single KV buffer, K/V loaded alternately)
        self.sQ_offset = 0
        self.sKV_offset = self.q_tile_bytes
        self.sScale_offset = self.sKV_offset + self.kv_tile_bytes
        self.mbar_offset = self.sScale_offset + self.scale_bytes

        total_smem = self.mbar_offset + MBAR_BYTES
        assert total_smem <= self.page_size, (
            f"FlashAttentionSm100Op: smem layout ({total_smem}B) > "
            f"page_size ({self.page_size}B)")

        # cpasync setup for KV loader warp (32 threads)
        self.async_copy_elems = 128 // self.q_dtype.width  # 8 for bf16
        self.copy_dim1 = max(1, self.D // self.async_copy_elems)
        self.copy_dim0 = max(1, 32 // self.copy_dim1)

        # Swizzle for O store (same as existing sm_100)
        if self.D >= 64:
            self.swizzle_B = 3
        elif self.D >= 32:
            self.swizzle_B = 2
        else:
            self.swizzle_B = 1

        # Framework inner iterations: Q loaded once by framework TMA
        self.inner_iters = 1
        self.inner_depth = 1

        # Number of MMA warps visible to framework
        self.num_mma_warps = NUM_MMA_TOTAL
        self.num_mma_threads = self.num_mma_warps * 32

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    @classmethod
    def schedule(cls, tile_sizes=None, causal=False,
                 kv_group_size=1, page_size=FA4_PAGE_SIZE, **tensors):
        """Schedule FA4 forward pass."""
        import torch
        ts = dict(tile_sizes or {})
        ts.setdefault("B", 1)
        ts.setdefault("H", 1)
        ts["M"] = M_BLOCK  # Fixed for UMMA

        q = tensors.get('q')
        k = tensors.get('k')
        if q is not None:
            # 3D backward compat: (BH, M, D) → (1, BH, M, D)
            if q.ndim == 3:
                BH, M, D = q.shape
                tensors['q'] = q.unsqueeze(0)       # (1, BH, M, D)
                tensors['o'] = tensors['o'].unsqueeze(0)
                tensors['k'] = k.unsqueeze(0)        # (1, BH_kv, N, D)
                tensors['v'] = tensors['v'].unsqueeze(0)
                if 'lse' in tensors and tensors['lse'] is not None:
                    tensors['lse'] = tensors['lse'].unsqueeze(0)
                q = tensors['q']
                k = tensors['k']

            assert q.element_size() == 2, (
                f"FlashAttentionSm100Op requires fp16/bf16")
            B, H, M_dim, D = q.shape
            N = k.shape[2]
            H_kv = k.shape[1]
        else:
            D = ts.get('D', 128)
            N = 128
            B = 1
            H = 1
            H_kv = 1

        # Auto-allocate lse if not provided
        if tensors.get('lse') is None and q is not None:
            tensors['lse'] = torch.empty(B, H, M_dim, dtype=torch.float32, device=q.device)

        ops = [cls._schedule_single(tile_sizes=ts, **tensors)]
        ops[0].static_dims['page_size'] = page_size
        ops[0].static_dims['kv_group_size'] = kv_group_size
        ops[0].static_dims['D'] = D
        ops[0].static_dims['H'] = H
        ops[0].static_dims['H_kv'] = H_kv
        if causal:
            ops[0].static_dims['causal'] = 1

        # Store per-dimension KV strides for strided tensors
        if k is not None and k.stride(-1) == 1:
            ops[0].static_dims['k_b_stride'] = k.stride(0)
            ops[0].static_dims['k_h_stride'] = k.stride(1)
            ops[0].static_dims['k_n_stride'] = k.stride(2)
            v = tensors['v']
            ops[0].static_dims['v_b_stride'] = v.stride(0)
            ops[0].static_dims['v_h_stride'] = v.stride(1)
            ops[0].static_dims['v_n_stride'] = v.stride(2)

        # TMA coordinate order flags for strided Q/O
        q = tensors.get('q')
        if q is not None and not q.is_contiguous():
            ops[0].static_dims['q_tma_permuted'] = 1
        o = tensors.get('o')
        if o is not None and not o.is_contiguous():
            ops[0].static_dims['o_tma_permuted'] = 1

        return ops

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for FA4."""
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS
        page_size = ops[0].static_dims.get('page_size', FA4_PAGE_SIZE)
        threads_per_block = (NUM_MMA_TOTAL + NUM_DMA_WARPS) * 32
        return MegakernelConfig(
            threads_per_block=threads_per_block,
            page_size=page_size,
        )

    # ------------------------------------------------------------------
    # Forward Load (framework DMA: TMA Q into sQ)
    # ------------------------------------------------------------------

    @cute.jit
    def load(self, page_ptr, tile_B, tile_H, tile_M, tile_D,
             q_tma, q_tma_gmem,
             work_mbar, inner_iter_idx):
        """TMA load Q into sQ region of page. Init op-managed mbarriers."""
        # Init op-managed mbarriers
        _mbar_base = page_ptr + Int32(self.mbar_offset)
        with cute.arch.elect_one():
            for mi in cutlass.range_constexpr(NUM_MBARRIERS):
                mbarrier_init(_mbar_base + Int32(mi * 8), Int32(1))
        mbarrier_init_fence()

        # TMA load Q into sQ
        sQ_base = page_ptr + Int32(self.sQ_offset)
        mbar_ptr = cute.make_ptr(
            cutlass.Int64, work_mbar, cute.AddressSpace.smem)

        sQ = cute.make_tensor(
            cute.make_ptr(self.q_dtype, sQ_base,
                          cute.AddressSpace.smem),
            cute.make_layout(self._q_tma_smem_shape),
        )
        gQ = cute.local_tile(
            q_tma_gmem, self._q_tma_smem_shape,
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
        if self.q_tma_permuted:
            cute.copy(q_tma, tQgQ[(None, tile_D, tile_H, tile_M, tile_B)],
                      tQsQ, tma_bar_ptr=mbar_ptr)
        else:
            cute.copy(q_tma, tQgQ[(None, tile_D, tile_M, tile_H, tile_B)],
                      tQsQ, tma_bar_ptr=mbar_ptr)

    # ------------------------------------------------------------------
    # Forward Compute (FA4 warp-specialized UMMA loop)
    # ------------------------------------------------------------------

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_H, tile_M, tile_D,
                k_gmem, v_gmem):
        """FA4 forward compute with warp specialization.

        Warps 0-3: softmax (read S from TMEM, write P to TMEM)
        Warps 4-7: correction (rescale O in TMEM, final epilogue to smem)
        Warp 8:    MMA (issue UMMA for QK and PV GEMMs)
        Warp 9:    KV loader (TMA load K/V into double-buffered smem)
        """
        import cutlass.utils.blackwell_helpers as bh

        tidx = cute.arch.thread_idx()[0]
        warp_idx = tidx // Int32(32)

        # GQA: map Q head to KV head
        kv_h = tile_H // Int32(self.kv_group_size)

        # Mbarrier base
        _mbar_base = page_ptr + Int32(self.mbar_offset)
        _K_ready = _mbar_base                       # K_ready
        _K_consumed = _mbar_base + Int32(8)          # K_consumed
        _V_ready = _mbar_base + Int32(2 * 8)         # V_ready
        _V_consumed = _mbar_base + Int32(3 * 8)      # V_consumed
        _S_full = _mbar_base + Int32(4 * 8)          # S_full
        _P_full = _mbar_base + Int32(5 * 8)          # P_full_O_rescaled
        _sc_full = _mbar_base + Int32(6 * 8)         # softmax_corr_full
        _sc_empty = _mbar_base + Int32(7 * 8)        # softmax_corr_empty
        _O_full = _mbar_base + Int32(8 * 8)          # O_full

        # ===============================================================
        # Build UMMA tiled_mma objects
        # ===============================================================

        # QK GEMM: Q(smem, K-major) × K(smem, K-major) → S(TMEM)
        tiled_mma_qk = bh.make_trivial_tiled_mma(
            ab_dtype=self.q_dtype,
            a_leading_mode=tcgen05.OperandMajorMode.K,
            b_leading_mode=tcgen05.OperandMajorMode.K,
            acc_dtype=Float32,
            cta_group=tcgen05.CtaGroup.ONE,
            mma_tiler_mn=(self.m_block, self.n_block),
        )

        # PV GEMM: P(TMEM, K-major) × V(smem, MN-major) → O(TMEM)
        tiled_mma_pv = bh.make_trivial_tiled_mma(
            ab_dtype=self.q_dtype,
            a_leading_mode=tcgen05.OperandMajorMode.K,
            b_leading_mode=tcgen05.OperandMajorMode.MN,
            acc_dtype=Float32,
            cta_group=tcgen05.CtaGroup.ONE,
            mma_tiler_mn=(self.m_block, self.D),
            a_source=tcgen05.OperandSource.TMEM,
        )

        # MMA tilers
        mma_tiler_qk = (self.m_block, self.n_block, self.D)
        mma_tiler_pv = (self.m_block, self.D, self.n_block)

        # ===============================================================
        # TMEM allocation (warp 0)
        # ===============================================================
        tmem_holding_buf = cute.arch.alloc_smem(8, alignment=8)
        if warp_idx == Int32(0):
            cute.arch.alloc_tmem(Int32(TMEM_ALLOC_COLS), tmem_holding_buf)
        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
            Float32, alignment=16,
            ptr_to_buffer_holding_addr=tmem_holding_buf)

        # ===============================================================
        # Smem layout for Q, KV, Scale, O
        # ===============================================================

        # Q smem (plain layout, loaded by framework TMA)
        sQ_base = page_ptr + Int32(self.sQ_offset)
        sQ_layout_qk = bh.make_smem_layout_a(
            tiled_mma_qk, mma_tiler_qk, self.q_dtype, 1)

        sQ = cute.make_tensor(
            cute.make_ptr(self.q_dtype, sQ_base,
                          cute.AddressSpace.smem, assumed_align=128),
            sQ_layout_qk,
        )

        # KV smem (single buffer, K and V loaded alternately via cpasync)
        sKV_base = page_ptr + Int32(self.sKV_offset)

        # K view: for QK GEMM (B operand, K-major, 1 stage)
        sK_layout = bh.make_smem_layout_b(
            tiled_mma_qk, mma_tiler_qk, self.q_dtype, 1)
        sK = cute.make_tensor(
            cute.make_ptr(self.q_dtype, sKV_base,
                          cute.AddressSpace.smem, assumed_align=128),
            sK_layout,
        )

        # V view: for PV GEMM (B operand, MN-major, 1 stage)
        sV_layout = bh.make_smem_layout_b(
            tiled_mma_pv, mma_tiler_pv, self.q_dtype, 1)
        sV = cute.make_tensor(
            cute.make_ptr(self.q_dtype, sKV_base,
                          cute.AddressSpace.smem, assumed_align=128),
            sV_layout,
        )

        # sScale for correction warps (fp32 per M row)
        sScale_base = page_ptr + Int32(self.sScale_offset)

        # ===============================================================
        # MMA fragments
        # ===============================================================

        # QK fragments
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)

        # Build S accumulator in TMEM
        acc_S_shape = tiled_mma_qk.partition_shape_C(
            (self.m_block, self.n_block))
        tStS = cute.make_tensor(acc_tmem_ptr, cute.make_layout(acc_S_shape))

        # PV fragments: P from TMEM, V from smem
        tOrV = tiled_mma_pv.make_fragment_B(sV)

        # Build O accumulator in TMEM (offset by TMEM_O_OFFSET)
        acc_O_shape = tiled_mma_pv.partition_shape_C(
            (self.m_block, self.D))
        tmem_o_ptr = cute.make_ptr(
            Float32, acc_tmem_ptr.toint() + Int32(TMEM_O_OFFSET * 4),
            mem_space=cute.AddressSpace.tmem, assumed_align=16)
        tOtO = cute.make_tensor(tmem_o_ptr, cute.make_layout(acc_O_shape))

        # P fragment (TMEM, for PV GEMM operand A)
        tmem_p_ptr = cute.make_ptr(
            Float32, acc_tmem_ptr.toint() + Int32(TMEM_P_OFFSET * 4),
            mem_space=cute.AddressSpace.tmem, assumed_align=16)
        # P fragment shape matches tiled_mma_pv A partition
        tOrP_shape = tiled_mma_pv.partition_shape_A(
            (self.m_block, self.n_block))
        tOrP = cute.make_tensor(tmem_p_ptr, cute.make_layout(tOrP_shape))

        # ===============================================================
        # TMEM copy atoms for softmax/correction
        # ===============================================================

        # TMEM → regs (for reading S, reading O)
        cta_tile_qk = (self.m_block, self.n_block)
        epi_tile_qk = bh.compute_epilogue_tile_shape(
            mma_tiler_qk, False, bh.LayoutEnum.ROW_MAJOR, self.q_dtype)
        copy_atom_t2r_S = bh.get_tmem_load_op(
            mma_tiler_qk, bh.LayoutEnum.ROW_MAJOR, self.q_dtype,
            Float32, epi_tile_qk, False)

        cta_tile_pv = (self.m_block, self.D)
        epi_tile_pv = bh.compute_epilogue_tile_shape(
            mma_tiler_pv, False, bh.LayoutEnum.ROW_MAJOR, self.q_dtype)
        copy_atom_t2r_O = bh.get_tmem_load_op(
            mma_tiler_pv, bh.LayoutEnum.ROW_MAJOR, self.q_dtype,
            Float32, epi_tile_pv, False)

        # ===============================================================
        # Warp dispatch
        # ===============================================================

        # --- KV LOADER WARP (warp 9) ---
        if warp_idx == Int32(NUM_SOFTMAX_WARPS + NUM_CORR_WARPS + NUM_MMA_WARPS_COMPUTE):
            self._kv_loader(
                page_ptr, tile_B, kv_h, k_gmem, v_gmem,
                _K_ready, _K_consumed, _V_ready, _V_consumed,
            )

        # --- MMA WARP (warp 8) ---
        if warp_idx == Int32(NUM_SOFTMAX_WARPS + NUM_CORR_WARPS):
            self._mma_warp(
                tiled_mma_qk, tiled_mma_pv,
                tSrQ, tSrK, tOrV, tStS, tOtO, tOrP,
                sQ, sK, sV,
                _K_ready, _K_consumed, _V_ready, _V_consumed,
                _S_full, _P_full, _O_full,
            )

        # --- SOFTMAX WARPS (warps 0-3) ---
        if warp_idx < Int32(NUM_SOFTMAX_WARPS):
            self._softmax_warps(
                tidx, warp_idx,
                tStS, acc_tmem_ptr,
                tiled_mma_qk, copy_atom_t2r_S,
                cta_tile_qk, epi_tile_qk,
                sScale_base, tile_M,
                _S_full, _P_full, _sc_full, _sc_empty,
            )

        # --- CORRECTION WARPS (warps 4-7) ---
        if warp_idx >= Int32(NUM_SOFTMAX_WARPS):
            if warp_idx < Int32(NUM_SOFTMAX_WARPS + NUM_CORR_WARPS):
                self._correction_warps(
                    tidx, warp_idx, page_ptr,
                    tOtO, acc_tmem_ptr,
                    tiled_mma_pv, copy_atom_t2r_O,
                    cta_tile_pv, epi_tile_pv,
                    sScale_base,
                    _P_full, _sc_full, _sc_empty, _O_full,
                )

        # ===============================================================
        # Post-loop: sync all MMA warps, TMEM dealloc
        # ===============================================================
        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))
        if warp_idx == Int32(0):
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(acc_tmem_ptr, Int32(TMEM_ALLOC_COLS))

    # ------------------------------------------------------------------
    # KV Loader Warp
    # ------------------------------------------------------------------

    @cute.jit
    def _kv_loader(self, page_ptr, tile_B, kv_h,
                   k_gmem, v_gmem,
                   _K_ready, _K_consumed, _V_ready, _V_consumed):
        """Internal cpasync loader for K and V tiles (single-buffered).

        K and V alternate in the same sKV buffer. Protocol per iteration:
            Load K → signal K_ready → wait K_consumed →
            Load V → signal V_ready → wait V_consumed
        """
        lane_idx = cute.arch.thread_idx()[0] % Int32(32)

        # cpasync setup: 32 threads, 128-bit copies (8 bf16 elements each)
        async_copy_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), self.q_dtype,
            num_bits_per_copy=128)
        copy_thread_layout = cute.make_layout(
            (self.copy_dim0, self.copy_dim1),
            stride=(self.copy_dim1, 1))
        copy_value_layout = cute.make_layout((1, self.async_copy_elems))
        gmem_tiled_copy = cute.make_tiled_copy_tv(
            async_copy_atom, copy_thread_layout, copy_value_layout)
        thr_copy = gmem_tiled_copy.get_slice(lane_idx)

        # Swizzled smem destination for cpasync (SW128 for bf16/fp16)
        swz = cute.make_swizzle(3, 4, 3)
        sKV_base = page_ptr + Int32(self.sKV_offset)
        sKV_cp = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.q_dtype, sKV_base,
                              cute.AddressSpace.smem, assumed_align=128),
                swz, dtype=self.q_dtype),
            cute.make_layout((self.n_block, self.D), stride=(self.D, 1)),
        )
        tKVsKV_cp = thr_copy.partition_D(sKV_cp)

        # Global K and V source tensors for this KV head
        gK_head = cute.make_tensor(
            (k_gmem.iterator + tile_B * Int32(self.k_b_stride) + kv_h * Int32(self.k_h_stride)).align(16),
            cute.make_layout((self.N, self.D), stride=(self.k_n_stride, 1)))
        gV_head = cute.make_tensor(
            (v_gmem.iterator + tile_B * Int32(self.v_b_stride) + kv_h * Int32(self.v_h_stride)).align(16),
            cute.make_layout((self.N, self.D), stride=(self.v_n_stride, 1)))

        _kc_phase = Int32(0)
        _vc_phase = Int32(0)

        kv_idx = Int32(0)
        while kv_idx < Int32(self.num_kv_blocks):
            # --- Load K[kv_idx] into sKV ---
            gK_block = cute.local_tile(
                gK_head, (self.n_block, self.D), (kv_idx, Int32(0)))
            tKgK = thr_copy.partition_S(gK_block)
            for ci in cutlass.range_constexpr(cute.size(tKVsKV_cp.shape[2])):
                cute.copy(gmem_tiled_copy, tKgK[None, None, ci],
                          tKVsKV_cp[None, None, ci])
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared)

            # Signal K ready, wait for K consumed
            mbarrier_arrive(_K_ready)
            mbarrier_wait(_K_consumed, _kc_phase)
            _kc_phase = _kc_phase ^ Int32(1)

            # --- Load V[kv_idx] into sKV (overwrites K) ---
            gV_block = cute.local_tile(
                gV_head, (self.n_block, self.D), (kv_idx, Int32(0)))
            tVgV = thr_copy.partition_S(gV_block)
            for ci in cutlass.range_constexpr(cute.size(tKVsKV_cp.shape[2])):
                cute.copy(gmem_tiled_copy, tVgV[None, None, ci],
                          tKVsKV_cp[None, None, ci])
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared)

            # Signal V ready, wait for V consumed
            mbarrier_arrive(_V_ready)
            mbarrier_wait(_V_consumed, _vc_phase)
            _vc_phase = _vc_phase ^ Int32(1)

            kv_idx = kv_idx + Int32(1)

    # ------------------------------------------------------------------
    # MMA Warp
    # ------------------------------------------------------------------

    @cute.jit
    def _mma_warp(self, tiled_mma_qk, tiled_mma_pv,
                  tSrQ, tSrK, tOrV, tStS, tOtO, tOrP,
                  sQ, sK, sV,
                  _K_ready, _K_consumed, _V_ready, _V_consumed,
                  _S_full, _P_full, _O_full):
        """Issue UMMA instructions for QK and PV GEMMs.

        Protocol per KV block:
            Wait K_ready → QK GEMM → signal K_consumed + S_full →
            Wait P_full (softmax done, O rescaled) →
            Wait V_ready → PV GEMM → signal V_consumed
        """
        _kr_phase = Int32(0)
        _vr_phase = Int32(0)
        _p_phase = Int32(0)

        kv_idx = Int32(0)
        while kv_idx < Int32(self.num_kv_blocks):
            # --- QK GEMM: S = Q × K^T → TMEM ---
            mbarrier_wait(_K_ready, _kr_phase)
            _kr_phase = _kr_phase ^ Int32(1)

            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
            cute.gemm(tiled_mma_qk, tStS,
                      tSrQ[(None, None, None, 0)],
                      tSrK[(None, None, None, 0)],
                      tStS)

            # K consumed, S ready for softmax
            mbarrier_arrive(_K_consumed)
            with cute.arch.elect_one():
                tcgen05.commit(_S_full)

            # Wait for softmax+correction to finish (P ready, O rescaled)
            mbarrier_wait(_P_full, _p_phase)
            _p_phase = _p_phase ^ Int32(1)

            # --- PV GEMM: O += P × V → TMEM ---
            mbarrier_wait(_V_ready, _vr_phase)
            _vr_phase = _vr_phase ^ Int32(1)

            if kv_idx == Int32(0):
                tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, False)
            else:
                tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
            cute.gemm(tiled_mma_pv, tOtO,
                      tOrP,
                      tOrV[(None, None, None, 0)],
                      tOtO)

            # V consumed
            mbarrier_arrive(_V_consumed)

            kv_idx = kv_idx + Int32(1)

        # Signal O ready for final epilogue
        with cute.arch.elect_one():
            tcgen05.commit(_O_full)

    # ------------------------------------------------------------------
    # Softmax Warps
    # ------------------------------------------------------------------

    @cute.jit
    def _softmax_warps(self, tidx, warp_idx,
                       tStS, acc_tmem_ptr,
                       tiled_mma_qk, copy_atom_t2r_S,
                       cta_tile, epi_tile,
                       sScale_base, tile_M,
                       _S_full, _P_full, _sc_full, _sc_empty):
        """4 softmax warps: read S from TMEM, online softmax, write P to TMEM."""
        # Softmax thread index within the 4-warp group (threads 0..127)
        local_tidx = tidx

        # Softmax state (one row per thread since 128 threads / 128 rows)
        row_max = Float32(-1e30)
        row_sum = Float32(0.0)

        # --- TMEM load setup for reading S ---
        # S is at TMEM offset 0, shape (m_block, n_block)
        tStSi = cute.composition(tStS, cute.make_layout(
            (self.m_block, self.n_block)))

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            Float32)
        thr_tmem_load = tcgen05.make_tmem_copy(
            tmem_load_atom, tStSi).get_slice(local_tidx)
        tStS_t2r = thr_tmem_load.partition_S(tStSi)
        tScS = cute.make_identity_tensor((self.m_block, self.n_block))
        tSrS_shape = thr_tmem_load.partition_D(tScS).shape
        tSrS = cute.make_fragment(tSrS_shape, Float32)

        # --- TMEM store setup for writing P ---
        # P overlaps S at TMEM_P_OFFSET columns, stored in half precision
        # P has shape (m_block, n_block/2) in fp32 cols because half packs 2:1
        p_cols = self.n_block // 32 * self.q_dtype.width
        tStP = cute.make_tensor(
            cute.make_ptr(Float32,
                          acc_tmem_ptr.toint() + Int32(TMEM_P_OFFSET * 4),
                          mem_space=cute.AddressSpace.tmem, assumed_align=16),
            cute.make_layout((self.m_block, p_cols)))

        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)),
            Float32)
        thr_tmem_store = tcgen05.make_tmem_copy(
            tmem_store_atom, tStP).get_slice(local_tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)

        # Register fragment for P in fp32 (will be recast to half for TMEM store)
        tScP = cute.make_identity_tensor((self.m_block, p_cols))
        tSrP_f32 = cute.make_fragment(
            thr_tmem_store.partition_S(tScP).shape, Float32)

        # sScale pointer for passing acc_scale to correction warps
        sScale_ptr = cute.make_ptr(Float32, sScale_base, cute.AddressSpace.smem)

        # Main KV loop
        _s_phase = Int32(0)
        kv_idx = Int32(0)
        while kv_idx < Int32(self.num_kv_blocks):
            # Wait for S ready from MMA warp
            mbarrier_wait(_S_full, _s_phase)

            # Read S from TMEM into registers
            cute.copy(thr_tmem_load, tStS_t2r, tSrS)

            # Online softmax: find row max
            new_max = Float32(-1e30)
            for si in cutlass.range_constexpr(cute.size(tSrS)):
                new_max = cute.arch.fmax(new_max, tSrS[si])

            # Warp-level reduce max (across 4 threads in MMA quad)
            new_max = cute.arch.fmax(
                new_max,
                cute.arch.shuffle_sync_bfly(new_max, offset=2, mask=-1,
                                            mask_and_clamp=31))
            new_max = cute.arch.fmax(
                new_max,
                cute.arch.shuffle_sync_bfly(new_max, offset=1, mask=-1,
                                            mask_and_clamp=31))

            m_old = row_max
            m_new = cute.arch.fmax(m_old, new_max)

            # Compute rescale factor for O correction
            acc_scale = Float32(1.0)
            if kv_idx > Int32(0):
                acc_scale = cute.math.exp2(
                    (m_old - m_new) * Float32(self.scale_log2e),
                    fastmath=True)
                row_sum = row_sum * acc_scale

            # Write acc_scale to sScale for correction warps
            sScale_ptr[local_tidx] = acc_scale

            # Signal correction warps (scale ready)
            mbarrier_arrive(_sc_full)

            # Scale-subtract and exp2: S[i] → exp2((S[i] - m_new) * scale_log2e)
            local_sum = Float32(0.0)
            for si in cutlass.range_constexpr(cute.size(tSrS)):
                val = cute.math.exp2(
                    tSrS[si] * Float32(self.scale_log2e)
                    - m_new * Float32(self.scale_log2e),
                    fastmath=True)
                tSrS[si] = val
                local_sum = local_sum + val

            # Reduce sum across quad
            local_sum = local_sum + cute.arch.shuffle_sync_bfly(
                local_sum, offset=2, mask=-1, mask_and_clamp=31)
            local_sum = local_sum + cute.arch.shuffle_sync_bfly(
                local_sum, offset=1, mask=-1, mask_and_clamp=31)

            row_sum = row_sum + local_sum
            row_max = m_new

            # Convert S_exp (fp32) → P (half), reinterpret as fp32, write to TMEM
            # P at TMEM_P_OFFSET overlaps S; half-precision packs 2 values per fp32 col
            tSrP_half = cute.make_tensor(
                cute.recast_ptr(tSrP_f32.iterator, dtype=self.q_dtype),
                tSrS.layout)
            for si in cutlass.range_constexpr(cute.size(tSrS)):
                tSrP_half[si] = tSrS[si].to(self.q_dtype)

            # Write P to TMEM
            cute.copy(thr_tmem_store, tSrP_f32, tStP_r2t)
            cute.arch.fence_view_async_tmem_store()

            # Wait for correction to finish reading scale
            mbarrier_wait(_sc_empty, _s_phase)

            # Signal P ready + O rescaled
            mbarrier_arrive(_P_full)

            _s_phase = _s_phase ^ Int32(1)
            kv_idx = kv_idx + Int32(1)

        # Store final row_sum for correction epilogue
        sScale_ptr[local_tidx] = row_sum

    # ------------------------------------------------------------------
    # Correction Warps
    # ------------------------------------------------------------------

    @cute.jit
    def _correction_warps(self, tidx, warp_idx, page_ptr,
                          tOtO, acc_tmem_ptr,
                          tiled_mma_pv, copy_atom_t2r_O,
                          cta_tile, epi_tile,
                          sScale_base,
                          _P_full, _sc_full, _sc_empty, _O_full):
        """4 correction warps: rescale O in TMEM, final epilogue to smem."""
        import cutlass.utils.blackwell_helpers as bh

        local_tidx = tidx - Int32(NUM_SOFTMAX_WARPS * 32)
        sScale_ptr = cute.make_ptr(Float32, sScale_base, cute.AddressSpace.smem)

        # --- TMEM load/store for O rescaling (FA4 correction_rescale pattern) ---
        corr_tile_size = 16
        tmem_load_corr_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            Float32)
        tmem_store_corr_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            Float32)

        # Compose O TMEM tensor to (m_block, corr_tile_size) view
        tOtO_corr = cute.composition(
            tOtO, cute.make_layout((self.m_block, corr_tile_size)))
        tOcO_corr = cute.composition(
            cute.make_identity_tensor((self.m_block, self.D)),
            cute.make_layout((self.m_block, corr_tile_size)))

        thr_tmem_load_corr = tcgen05.make_tmem_copy(
            tmem_load_corr_atom, tOtO_corr).get_slice(local_tidx)
        thr_tmem_store_corr = tcgen05.make_tmem_copy(
            tmem_store_corr_atom, tOtO_corr).get_slice(local_tidx)

        tOtO_t2r = thr_tmem_load_corr.partition_S(tOtO_corr)
        tOrO_shape = thr_tmem_load_corr.partition_D(tOcO_corr).shape
        tOtO_r2t = thr_tmem_store_corr.partition_D(tOtO_corr)

        corr_frg_count = self.D // corr_tile_size

        # --- First KV block: no rescaling needed ---
        # Softmax signals sc_full for first block, but scale=1.0, so just consume
        mbarrier_wait(_sc_full, Int32(0))
        mbarrier_arrive(_sc_empty)
        mbarrier_arrive(_P_full)

        # --- KV loop: rescale O ---
        _sc_phase = Int32(1)
        kv_idx = Int32(1)
        while kv_idx < Int32(self.num_kv_blocks):
            # Wait for softmax to write acc_scale
            mbarrier_wait(_sc_full, _sc_phase)

            # Read acc_scale from sScale
            scale = sScale_ptr[local_tidx]

            # Rescale O in TMEM: O *= scale
            should_rescale = cute.arch.vote_ballot_sync(scale < Float32(1.0)) != Int32(0)
            if should_rescale:
                for fi in cutlass.range_constexpr(corr_frg_count):
                    tOrO_frg = cute.make_fragment(tOrO_shape, Float32)
                    tOtO_t2r_i = cute.make_tensor(
                        tOtO_t2r.iterator + Int32(fi * corr_tile_size),
                        tOtO_t2r.layout)
                    cute.copy(thr_tmem_load_corr, tOtO_t2r_i, tOrO_frg)
                    for ji in cutlass.range_constexpr(cute.size(tOrO_frg)):
                        tOrO_frg[ji] = tOrO_frg[ji] * scale
                    tOtO_r2t_i = cute.make_tensor(
                        tOtO_r2t.iterator + Int32(fi * corr_tile_size),
                        tOtO_r2t.layout)
                    cute.copy(thr_tmem_store_corr, tOrO_frg, tOtO_r2t_i)
                cute.arch.fence_view_async_tmem_store()

            # Signal softmax can proceed (done reading scale)
            mbarrier_arrive(_sc_empty)

            # Signal MMA that P ready + O rescaled
            mbarrier_arrive(_P_full)

            _sc_phase = _sc_phase ^ Int32(1)
            kv_idx = kv_idx + Int32(1)

        # --- Final epilogue: O from TMEM → smem ---
        # Wait for final O ready from MMA warp
        mbarrier_wait(_O_full, Int32(0))

        # Read row_sum from sScale (softmax stored it there after loop)
        inv_sum = cute.arch.rcp_approx(sScale_ptr[local_tidx])

        # TMEM → regs → scale by 1/row_sum → convert to half → write to smem
        tOtO_epi = cute.flat_divide(tOtO[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r_O, tOtO_epi[(None, None, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(local_tidx)
        tTR_tOtO = thr_copy_t2r.partition_S(tOtO_epi)
        tTR_cOtO = thr_copy_t2r.partition_D(
            cute.flat_divide(cute.make_identity_tensor(cta_tile), epi_tile))
        tTR_rO = cute.make_fragment(
            tTR_cOtO[None, None, None, 0, 0].shape, Float32)

        # R→S copy for O output
        copy_atom_r2s_O = bh.get_smem_store_op(
            bh.LayoutEnum.ROW_MAJOR, self.q_dtype, Float32, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s_O, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(local_tidx)

        # Output smem (reuse page start for O, swizzled for TMA store)
        epi_smem_layout = bh.make_smem_layout_epi(
            tiled_copy_t2r, cta_tile, self.q_dtype, 1)
        sO = cute.make_tensor(
            cute.make_ptr(self.q_dtype, page_ptr,
                          cute.AddressSpace.smem, assumed_align=128),
            epi_smem_layout,
        )
        tRS_sO = thr_copy_r2s.partition_D(sO)

        # Process epilogue sub-tiles
        tTR_rD = cute.make_fragment(tTR_rO.shape, Float32)
        num_epi_m = cute.size(tTR_tOtO, mode=[3])
        num_epi_n = cute.size(tTR_tOtO, mode=[4])
        for epi_m in cutlass.range_constexpr(num_epi_m):
            for epi_n in cutlass.range_constexpr(num_epi_n):
                # TMEM → regs
                cute.copy(tiled_copy_t2r,
                          tTR_tOtO[None, None, None, epi_m, epi_n, 0],
                          tTR_rO)
                # Scale by 1/row_sum and convert to half
                tRS_rO = tiled_copy_r2s.retile(tTR_rO)
                tRS_rD_retiled = tiled_copy_r2s.retile(tTR_rD)
                for ci in cutlass.range_constexpr(cute.size(tRS_rO)):
                    tRS_rD_retiled[ci] = (tRS_rO[ci] * inv_sum).to(self.q_dtype)
                # Store to smem
                cute.copy(tiled_copy_r2s, tRS_rD_retiled,
                          tRS_sO[None, None, None, epi_m, epi_n, 0])

        cute.arch.fence_proxy(cute.arch.ProxyKind.async_shared)

    # ------------------------------------------------------------------
    # Forward Store (framework DMA: TMA O from smem)
    # ------------------------------------------------------------------

    @cute.jit
    def store(self, page_ptr, tile_B, tile_H, tile_M, tile_D, o_tma, o_tma_gmem):
        """TMA store O from smem to global memory (swizzled)."""
        _o_swz = cute.make_swizzle(self.swizzle_B, 4, 3)

        sO = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem),
                _o_swz, dtype=self.q_dtype),
            cute.make_layout(self._o_tma_smem_shape),
        )
        gO = cute.local_tile(
            o_tma_gmem, self._o_tma_smem_shape, (None, None, None, None),
        )
        tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
            o_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sO, 0, 4),
            cute.group_modes(gO, 0, 4),
        )
        with cute.arch.elect_one():
            if self.o_tma_permuted:
                cute.copy(o_tma, tOsO, tOgO[(None, tile_D, tile_H, tile_M, tile_B)])
            else:
                cute.copy(o_tma, tOsO, tOgO[(None, tile_D, tile_M, tile_H, tile_B)])


__all__ = ["FlashAttentionSm100Op"]
