# Copyright (c) 2025, Machete Authors
"""
Cooperative Flash Attention Backward Op for SM120 (Blackwell).

Computes dQ, dK, dV given dO (gradient of loss w.r.t. output O).

Architecture: "dKV-centric" — tile on N (outer), iterate M (inner).
    DMA warps: TMA load K[n], V[n] → smem. TMA store dK[n], dV[n].
    MMA warps: cpasync Q[m], dO[m] per M-block. 5 GEMMs per block:
        S  = Q @ K^T           (m_block, tile_N)   K-dim = D
        dP = dO @ V^T          (m_block, tile_N)   K-dim = D
        dV += P^T @ dO         (tile_N, D)         K-dim = m_block
        dK += dS^T @ Q         (tile_N, D)         K-dim = m_block
        dQ += dS @ K * scale   (m_block, D)        K-dim = tile_N

Math:
    P  = exp(S * scale - LSE)           (reconstructed via LSE from fwd)
    dS = P * (dP - dPsum)               (dPsum precomputed in Python)
    dV = sum_m P^T @ dO                 (accumulated across M-blocks)
    dK = sum_m dS^T @ Q * scale         (accumulated across M-blocks)
    dQ += dS @ K * scale                (atomicAdd across N-tiles)

Smem page layout (all regions coexist, page_size >= 48KB):
    [K: tile_N×D] [V: tile_N×D] [Q_buf: m_block×D] [dO_buf: m_block×D]
    [P_buf: m_block×tile_N] [dS_buf: m_block×tile_N]
    K/V stay in smem throughout compute. Q_buf/dO_buf refreshed each M-block.
    Epilogue: dK/dV overwrite K/V areas for TMA store.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu import warp
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import (
    named_barrier_sync,
)


@dsl_user_op
def _atomic_add_f32(val: Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """Atomic f32 add to global memory via nvvm."""
    nvvm.atomicrmw(
        res=T.f32(), op=nvvm.AtomicOpKind.FADD,
        ptr=gmem_ptr.llvm_ptr, a=Float32(val).ir_value()
    )


class FlashAttentionSm120BwdOp(Op):
    """Backward pass for cooperative Flash Attention (SM120, Blackwell).

    Tensors:
        k:  (BH, N, D) -- key (fp16/bf16)
        v:  (BH, N, D) -- value
        q:  (BH, M, D) -- query (cpasync in compute)
        dout: (BH, M, D) -- gradient of loss w.r.t. O
        lse: (BH, M)    -- logsumexp from forward (f32)
        dpsum: (BH, M)  -- rowsum(dO * O), precomputed (f32)
        dq_accum: (BH, M, D) -- atomicAdd target (f32)
        dk: (BH, N, D) -- output gradient w.r.t. K
        dv: (BH, N, D) -- output gradient w.r.t. V

    Tiling: tile_BH=1, tile_N from schedule, tile_D=D (full).
    """

    reads = {
        "k": (None, ("BH", "N", "D")),
        "v": (None, ("BH", "N", "D")),
        "q": (None, ("BH", "M", "D")),
        "dout": (None, ("BH", "M", "D")),
        "lse": (cutlass.Float32, ("BH", "M")),
        "dpsum": (cutlass.Float32, ("BH", "M")),
        "dq_accum": (cutlass.Float32, ("BH", "M", "D")),
    }
    writes = {
        "dk": (None, ("BH", "N", "D")),
        "dv": (None, ("BH", "N", "D")),
    }
    tile = ("BH", "N", "D")

    tma_loads = {"k", "v"}
    tma_stores = {"dk", "dv"}

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        """Swizzled smem layout for K/V TMA load and dK/dV TMA store."""
        if tensor_name not in ("k", "v", "dk", "dv"):
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

        assert self.k_dtype in (cutlass.Float16, cutlass.BFloat16), (
            f"FlashAttentionSm120BwdOp requires fp16 or bf16, got {self.k_dtype}"
        )
        self.elem_bytes = 2
        self.dtype = self.k_dtype

        self.scale_val = 1.0 / (self.D ** 0.5)
        self.scale_log2e = self.scale_val * 1.4426950408889634074

        self._init_mma()

    def _init_mma(self):
        """Init tile sizes, smem offsets, and cpasync layout."""
        # tile_N and m_block are equal for simplicity (all GEMMs use same MMA config)
        self.m_block = self.tile_size_N  # tile_N = m_block

        assert self.tile_size_N % 16 == 0 and self.tile_size_N >= 16
        assert self.D >= 16 and self.D % 16 == 0

        self.num_mma_warps = self.tile_size_N // 16
        self.num_mma_threads = self.num_mma_warps * 32

        self.num_m_blocks = (self.M + self.m_block - 1) // self.m_block

        # Smem layout: K/V stay in smem throughout, Q/dO/P/dS after them.
        self.kv_tile_bytes = self.tile_size_N * self.D * self.elem_bytes
        self.q_buf_bytes = self.m_block * self.D * self.elem_bytes
        self.do_buf_bytes = self.m_block * self.D * self.elem_bytes
        self.p_buf_bytes = self.m_block * self.tile_size_N * self.elem_bytes
        self.ds_buf_bytes = self.m_block * self.tile_size_N * self.elem_bytes

        # Offsets within page
        self.q_buf_offset = 2 * self.kv_tile_bytes
        self.do_buf_offset = self.q_buf_offset + self.q_buf_bytes
        self.p_buf_offset = self.do_buf_offset + self.do_buf_bytes
        self.ds_buf_offset = self.p_buf_offset + self.p_buf_bytes

        total_smem = self.ds_buf_offset + self.ds_buf_bytes
        assert total_smem <= self.page_size, (
            f"Total smem ({total_smem}B) exceeds page_size ({self.page_size}B). "
            f"tile_N={self.tile_size_N}, D={self.D}"
        )

        # DMA loads K+V once, compute handles everything
        self.inner_iters = 1
        self.inner_depth = 1

        # Swizzle for smem (same as forward)
        if self.D >= 64:
            self.swizzle_B = 3
        elif self.D >= 32:
            self.swizzle_B = 2
        else:
            self.swizzle_B = 1
        self.swizzle_M = 4
        self.swizzle_S = 3

        # cpasync thread layout for Q/dO loading (128-bit copies)
        self.async_copy_elems = 128 // (self.elem_bytes * 8)  # 8 for fp16
        self.copy_dim1 = self.D // self.async_copy_elems
        self.copy_dim0 = self.num_mma_threads // self.copy_dim1
        assert self.m_block % self.copy_dim0 == 0, (
            f"m_block={self.m_block} must be divisible by copy_dim0={self.copy_dim0}"
        )

        # Override compute method
        self.compute = self.compute_mma

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule_backward(cls, tile_sizes=None, causal=False, page_size=49152, **tensors):
        """Schedule backward pass."""
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("BH", 1)
        k = tensors.get("k")
        if k is not None:
            assert k.element_size() == 2
            D = k.shape[-1]
            if "N" not in tile_sizes:
                # Total smem = 2*tile_N*D*2 + 2*tile_N*D*2 + 2*tile_N*tile_N*2
                # = 4*tile_N*D*2 + 2*tile_N^2*2 = tile_N*(8*D + 4*tile_N) bytes
                # For simplicity, start with tile_N from K+V constraint, check total
                max_tile_N = page_size // (2 * D * k.element_size())
                max_nw = min(8, max_tile_N // 16, k.shape[1] // 16)
                # Power-of-2 warps for cpasync divisibility
                nw = 1
                while nw * 2 <= max_nw:
                    nw *= 2
                tile_N = max(16, nw * 16)
                # Verify total smem fits
                eb = k.element_size()
                total = 2 * tile_N * D * eb + 2 * tile_N * D * eb + 2 * tile_N * tile_N * eb
                while total > page_size and nw > 1:
                    nw = nw // 2
                    tile_N = nw * 16
                    total = 2 * tile_N * D * eb + 2 * tile_N * D * eb + 2 * tile_N * tile_N * eb
                tile_sizes["N"] = tile_N
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        if causal:
            ops[0].static_dims["causal"] = 1
        return ops

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig."""
        from machete.megakernel import MegakernelConfig
        from machete.megakernel.megakernel import NUM_DMA_WARPS

        tile_n = ops[0].tile_sizes["N"]
        num_mma_warps = tile_n // 16
        threads_per_block = (num_mma_warps + NUM_DMA_WARPS) * 32
        page_size = ops[0].static_dims.get("page_size", 49152)
        return MegakernelConfig(
            threads_per_block=threads_per_block,
            page_size=page_size,
        )

    # =========================================================================
    # Forward Load (DMA warp: TMA K + V)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_BH, tile_N, tile_D,
             k_tma, k_tma_gmem, v_tma, v_tma_gmem, work_mbar):
        """TMA load K and V into page (K at offset 0, V at offset kv_tile_bytes)."""
        from machete.megakernel.interpreter import mbarrier_arrive_expect_tx

        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)

        # K at page start
        sK = cute.make_tensor(
            cute.make_ptr(self.dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.D, self.tile_size_N, 1)),
        )
        gK = cute.local_tile(
            k_tma_gmem, (self.D, self.tile_size_N, 1), (None, None, None),
        )
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            k_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sK, 0, 3), cute.group_modes(gK, 0, 3),
        )

        # V at page_ptr + kv_tile_bytes
        v_base = page_ptr + Int32(self.kv_tile_bytes)
        sV = cute.make_tensor(
            cute.make_ptr(self.dtype, v_base, cute.AddressSpace.smem),
            cute.make_layout((self.D, self.tile_size_N, 1)),
        )
        gV = cute.local_tile(
            v_tma_gmem, (self.D, self.tile_size_N, 1), (None, None, None),
        )
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            v_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sV, 0, 3), cute.group_modes(gV, 0, 3),
        )

        nbytes = Int32(2 * self.kv_tile_bytes)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(k_tma, tKgK[(None, tile_D, tile_N, tile_BH)], tKsK, tma_bar_ptr=mbar_ptr)
        cute.copy(v_tma, tVgV[(None, tile_D, tile_N, tile_BH)], tVsV, tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # MMA Helpers
    # =========================================================================

    def _make_acc_tensor_mn_view(self, acc):
        """Reshape MMA accumulator to (M, N) view for per-row operations."""
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        s = acc_layout_col_major.shape
        st = acc_layout_col_major.stride
        acc_layout_mn = cute.make_layout(
            ((s[0][1], s[1]), (s[0][0], s[2])),
            stride=((st[0][1], st[1]), (st[0][0], st[2])),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    # =========================================================================
    # Backward Compute -- M-block loop with 5 GEMMs per block
    # =========================================================================

    @cute.jit
    def compute_mma(
        self, page_ptr, tile_BH, tile_N, tile_D,
        k, v, q, dout, lse, dpsum, dq_accum, dk, dv
    ):
        """Backward compute: iterate M-blocks, accumulate dK/dV, atomicAdd dQ.

        K/V stay in smem throughout (TMA-loaded at page offsets 0 and kv_tile_bytes).
        Q_buf/dO_buf at offsets after K+V, refreshed each M-block via cpasync.
        All GEMM operands read from smem via LdMatrix per GEMM.
        """
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        if warp_idx < Int32(self.num_mma_warps):
            # === MMA setup ===
            mma_op = warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((self.num_mma_warps, 1, 1)),
                permutation_mnk=(self.num_mma_warps * 16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)

            swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)

            # === Smem pointers ===
            k_smem_ptr = page_ptr
            v_smem_ptr = page_ptr + Int32(self.kv_tile_bytes)
            q_buf_ptr = page_ptr + Int32(self.q_buf_offset)
            do_buf_ptr = page_ptr + Int32(self.do_buf_offset)
            p_buf_ptr = page_ptr + Int32(self.p_buf_offset)
            ds_buf_ptr = page_ptr + Int32(self.ds_buf_offset)

            # ---------------------------------------------------------------
            # LdMatrix setup for each GEMM's operands (all from smem)
            # ---------------------------------------------------------------

            # -- S GEMM: C(m,n) = Q(m,D) @ K^T -> A=Q, B=K, K-dim=D --
            # A = Q_buf (m_block, D), non-transposed
            _sQ_A = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(self.dtype, q_buf_ptr, cute.AddressSpace.smem, assumed_align=128), swz, dtype=self.dtype),
                cute.make_layout((self.m_block, self.D), stride=(self.D, 1)),
            )
            smem_copy_Q_A = cute.make_tiled_copy_A(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype), tiled_mma)
            thr_copy_Q_A = smem_copy_Q_A.get_slice(tidx)
            tCrQ_A = tiled_mma.make_fragment_A(thr_mma.partition_A(_sQ_A))
            tQrQ_A_v = thr_copy_Q_A.retile(tCrQ_A)
            tQsQ_A = thr_copy_Q_A.partition_S(_sQ_A)

            # B = K (tile_N, D), non-transposed
            _sK_B = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(self.dtype, k_smem_ptr, cute.AddressSpace.smem, assumed_align=128), swz, dtype=self.dtype),
                cute.make_layout((self.tile_size_N, self.D), stride=(self.D, 1)),
            )
            smem_copy_K_B = cute.make_tiled_copy_B(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype), tiled_mma)
            thr_copy_K_B = smem_copy_K_B.get_slice(tidx)
            tCrK_B = tiled_mma.make_fragment_B(thr_mma.partition_B(_sK_B))
            tKrK_B_v = thr_copy_K_B.retile(tCrK_B)
            tKsK_B = thr_copy_K_B.partition_S(_sK_B)

            # -- dP GEMM: same shape as S. A=dO, B=V, K-dim=D --
            # A = dO_buf (m_block, D), non-transposed
            _sdO_A = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(self.dtype, do_buf_ptr, cute.AddressSpace.smem, assumed_align=128), swz, dtype=self.dtype),
                cute.make_layout((self.m_block, self.D), stride=(self.D, 1)),
            )
            smem_copy_dO_A = cute.make_tiled_copy_A(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype), tiled_mma)
            thr_copy_dO_A = smem_copy_dO_A.get_slice(tidx)
            tCrdO_A = tiled_mma.make_fragment_A(thr_mma.partition_A(_sdO_A))
            tdOrO_A_v = thr_copy_dO_A.retile(tCrdO_A)
            tdOsdO_A = thr_copy_dO_A.partition_S(_sdO_A)

            # B = V (tile_N, D), non-transposed (same layout as K)
            _sV_B = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(self.dtype, v_smem_ptr, cute.AddressSpace.smem, assumed_align=128), swz, dtype=self.dtype),
                cute.make_layout((self.tile_size_N, self.D), stride=(self.D, 1)),
            )
            smem_copy_V_B = cute.make_tiled_copy_B(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype), tiled_mma)
            thr_copy_V_B = smem_copy_V_B.get_slice(tidx)
            tCrV_B = tiled_mma.make_fragment_B(thr_mma.partition_B(_sV_B))
            tVrV_B_v = thr_copy_V_B.retile(tCrV_B)
            tVsV_B = thr_copy_V_B.partition_S(_sV_B)

            # -- dV GEMM: C(tile_N, D) = P^T(tile_N, m_block) @ dO(m_block, D)
            #    A = P^T from P_buf, transposed LdMatrix. K-dim = m_block.
            #    B = dO^T(D, m_block) from dO_buf, transposed LdMatrix.
            _sPt_A = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(self.dtype, p_buf_ptr, cute.AddressSpace.smem, assumed_align=128), swz, dtype=self.dtype),
                cute.make_layout((self.tile_size_N, self.m_block), stride=(1, self.tile_size_N)),
            )
            smem_copy_Pt_A = cute.make_tiled_copy_A(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype), tiled_mma)
            thr_copy_Pt_A = smem_copy_Pt_A.get_slice(tidx)
            tCrPt_A = tiled_mma.make_fragment_A(thr_mma.partition_A(_sPt_A))
            tPrPt_A_v = thr_copy_Pt_A.retile(tCrPt_A)
            tPsPt_A = thr_copy_Pt_A.partition_S(_sPt_A)

            _sdOt_B = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(self.dtype, do_buf_ptr, cute.AddressSpace.smem, assumed_align=128), swz, dtype=self.dtype),
                cute.make_layout((self.D, self.m_block), stride=(1, self.D)),
            )
            smem_copy_dOt_B = cute.make_tiled_copy_B(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype), tiled_mma)
            thr_copy_dOt_B = smem_copy_dOt_B.get_slice(tidx)
            tCrdOt_B = tiled_mma.make_fragment_B(thr_mma.partition_B(_sdOt_B))
            tdOrdOt_B_v = thr_copy_dOt_B.retile(tCrdOt_B)
            tdOsdOt_B = thr_copy_dOt_B.partition_S(_sdOt_B)

            # -- dK GEMM: C(tile_N, D) = dS^T(tile_N, m_block) @ Q(m_block, D)
            #    A = dS^T from dS_buf, transposed LdMatrix. K-dim = m_block.
            #    B = Q^T(D, m_block) from Q_buf, transposed LdMatrix.
            _sdSt_A = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(self.dtype, ds_buf_ptr, cute.AddressSpace.smem, assumed_align=128), swz, dtype=self.dtype),
                cute.make_layout((self.tile_size_N, self.m_block), stride=(1, self.tile_size_N)),
            )
            smem_copy_dSt_A = cute.make_tiled_copy_A(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype), tiled_mma)
            thr_copy_dSt_A = smem_copy_dSt_A.get_slice(tidx)
            tCrdSt_A = tiled_mma.make_fragment_A(thr_mma.partition_A(_sdSt_A))
            tdSrdSt_A_v = thr_copy_dSt_A.retile(tCrdSt_A)
            tdSsdSt_A = thr_copy_dSt_A.partition_S(_sdSt_A)

            _sQt_B = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(self.dtype, q_buf_ptr, cute.AddressSpace.smem, assumed_align=128), swz, dtype=self.dtype),
                cute.make_layout((self.D, self.m_block), stride=(1, self.D)),
            )
            smem_copy_Qt_B = cute.make_tiled_copy_B(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype), tiled_mma)
            thr_copy_Qt_B = smem_copy_Qt_B.get_slice(tidx)
            tCrQt_B = tiled_mma.make_fragment_B(thr_mma.partition_B(_sQt_B))
            tQrQt_B_v = thr_copy_Qt_B.retile(tCrQt_B)
            tQsQt_B = thr_copy_Qt_B.partition_S(_sQt_B)

            # -- dQ GEMM: C(m_block, D) = dS(m_block, tile_N) @ K(tile_N, D)
            #    A = dS from dS_buf, non-transposed. K-dim = tile_N.
            #    B = K^T(D, tile_N) from K smem, transposed LdMatrix.
            _sdS_dQ_A = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(self.dtype, ds_buf_ptr, cute.AddressSpace.smem, assumed_align=128), swz, dtype=self.dtype),
                cute.make_layout((self.m_block, self.tile_size_N), stride=(self.tile_size_N, 1)),
            )
            smem_copy_dS_dQ_A = cute.make_tiled_copy_A(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype), tiled_mma)
            thr_copy_dS_dQ_A = smem_copy_dS_dQ_A.get_slice(tidx)
            tCrdS_dQ_A = tiled_mma.make_fragment_A(thr_mma.partition_A(_sdS_dQ_A))
            tdSrdS_dQ_A_v = thr_copy_dS_dQ_A.retile(tCrdS_dQ_A)
            tdSsdS_dQ_A = thr_copy_dS_dQ_A.partition_S(_sdS_dQ_A)

            _sKt_dQ_B = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(self.dtype, k_smem_ptr, cute.AddressSpace.smem, assumed_align=128), swz, dtype=self.dtype),
                cute.make_layout((self.D, self.tile_size_N), stride=(1, self.D)),
            )
            smem_copy_Kt_dQ_B = cute.make_tiled_copy_B(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype), tiled_mma)
            thr_copy_Kt_dQ_B = smem_copy_Kt_dQ_B.get_slice(tidx)
            tCrKt_dQ_B = tiled_mma.make_fragment_B(thr_mma.partition_B(_sKt_dQ_B))
            tKrKt_dQ_B_v = thr_copy_Kt_dQ_B.retile(tCrKt_dQ_B)
            tKsKt_dQ_B = thr_copy_Kt_dQ_B.partition_S(_sKt_dQ_B)

            # -- CopyUniversal for R→S (P and dS accumulators to smem) --
            smem_copy_C = cute.make_tiled_copy_C(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype), tiled_mma
            )
            thr_copy_C = smem_copy_C.get_slice(tidx)
            _sP_dst = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(self.dtype, p_buf_ptr, cute.AddressSpace.smem, assumed_align=128), swz, dtype=self.dtype),
                cute.make_layout((self.m_block, self.tile_size_N), stride=(self.tile_size_N, 1)),
            )
            tPsP_dst = thr_copy_C.partition_D(_sP_dst)
            _sdS_dst = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(self.dtype, ds_buf_ptr, cute.AddressSpace.smem, assumed_align=128), swz, dtype=self.dtype),
                cute.make_layout((self.m_block, self.tile_size_N), stride=(self.tile_size_N, 1)),
            )
            tdSsdS_dst = thr_copy_C.partition_D(_sdS_dst)

            # === cpasync setup for Q and dO ===
            async_copy_atom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), self.dtype, num_bits_per_copy=128)
            copy_thread_layout = cute.make_layout((self.copy_dim0, self.copy_dim1), stride=(self.copy_dim1, 1))
            copy_value_layout = cute.make_layout((1, self.async_copy_elems))
            gmem_tiled_copy = cute.make_tiled_copy_tv(async_copy_atom, copy_thread_layout, copy_value_layout)
            thr_copy_cp = gmem_tiled_copy.get_slice(tidx)

            # cpasync smem destinations (swizzled)
            sQ_cp = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(self.dtype, q_buf_ptr, cute.AddressSpace.smem, assumed_align=128), swz, dtype=self.dtype),
                cute.make_layout((self.m_block, self.D), stride=(self.D, 1)),
            )
            sdO_cp = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(self.dtype, do_buf_ptr, cute.AddressSpace.smem, assumed_align=128), swz, dtype=self.dtype),
                cute.make_layout((self.m_block, self.D), stride=(self.D, 1)),
            )
            tQsQ_cp = thr_copy_cp.partition_D(sQ_cp)
            tdOsdO_cp = thr_copy_cp.partition_D(sdO_cp)

            # Global Q and dO pointers for current head
            q_head_ptr = (q.iterator + tile_BH * Int32(self.M * self.D)).align(16)
            do_head_ptr = (dout.iterator + tile_BH * Int32(self.M * self.D)).align(16)
            gQ_head = cute.make_tensor(q_head_ptr, cute.make_layout((self.M, self.D), stride=(self.D, 1)))
            gdO_head = cute.make_tensor(do_head_ptr, cute.make_layout((self.M, self.D), stride=(self.D, 1)))

            # === Accumulators ===
            acc_dK = cute.make_fragment(tiled_mma.partition_shape_C((self.tile_size_N, self.D)), Float32)
            acc_dV = cute.make_fragment(tiled_mma.partition_shape_C((self.tile_size_N, self.D)), Float32)
            acc_dK.fill(0.0)
            acc_dV.fill(0.0)
            acc_S = cute.make_fragment(tiled_mma.partition_shape_C((self.m_block, self.tile_size_N)), Float32)
            acc_dP = cute.make_fragment(tiled_mma.partition_shape_C((self.m_block, self.tile_size_N)), Float32)
            acc_dQ = cute.make_fragment(tiled_mma.partition_shape_C((self.m_block, self.D)), Float32)

            # Identity tensor for S masking
            mcS = cute.make_identity_tensor((self.m_block, self.tile_size_N))
            tScS = thr_mma.partition_C(mcS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)
            acc_S_shape = tiled_mma.partition_shape_C((self.m_block, self.tile_size_N))
            num_rows_S = acc_S_shape[0][1] * acc_S_shape[1]

            # Identity for dQ atomicAdd
            mc_dQ = cute.make_identity_tensor((self.m_block, self.D))
            tSc_dQ = thr_mma.partition_C(mc_dQ)
            tSc_dQ_mn = self._make_acc_tensor_mn_view(tSc_dQ)
            dq_shape = tiled_mma.partition_shape_C((self.m_block, self.D))
            num_rows_dQ = dq_shape[0][1] * dq_shape[1]

            # P and dS register fragments for R→S copy
            rP = cute.make_fragment_like(acc_S, self.dtype)
            rdS = cute.make_fragment_like(acc_S, self.dtype)

            # Global LSE, dPsum, dQ_accum
            lse_head_ptr = lse.iterator + tile_BH * Int32(self.M)
            g_lse = cute.make_tensor(lse_head_ptr, cute.make_layout(self.M))
            dpsum_head_ptr = dpsum.iterator + tile_BH * Int32(self.M)
            g_dpsum = cute.make_tensor(dpsum_head_ptr, cute.make_layout(self.M))
            dqa_head_ptr = dq_accum.iterator + tile_BH * Int32(self.M * self.D)
            g_dq_accum = cute.make_tensor(
                dqa_head_ptr, cute.make_layout((self.M, self.D), stride=(self.D, 1))
            )

            # =============================================================
            # M-block loop
            # =============================================================
            m_idx = Int32(0)
            while m_idx < Int32(self.num_m_blocks):
                m_start = m_idx * Int32(self.m_block)

                # --- cpasync Q[m] and dO[m] → smem ---
                gQ_block = cute.local_tile(gQ_head, (self.m_block, self.D), (m_idx, Int32(0)))
                tQgQ = thr_copy_cp.partition_S(gQ_block)
                for ci in cutlass.range_constexpr(cute.size(tQsQ_cp.shape[2])):
                    cute.copy(gmem_tiled_copy, tQgQ[None, None, ci], tQsQ_cp[None, None, ci])

                gdO_block = cute.local_tile(gdO_head, (self.m_block, self.D), (m_idx, Int32(0)))
                tdOgdO = thr_copy_cp.partition_S(gdO_block)
                for ci in cutlass.range_constexpr(cute.size(tdOsdO_cp.shape[2])):
                    cute.copy(gmem_tiled_copy, tdOgdO[None, None, ci], tdOsdO_cp[None, None, ci])

                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # --- GEMM 1: S = Q @ K^T (m_block × tile_N), K-dim=D ---
                for kb in cutlass.range_constexpr(self.D // 16):
                    cute.copy(smem_copy_Q_A, tQsQ_A[None, None, kb], tQrQ_A_v[None, None, kb])
                for kb in cutlass.range_constexpr(self.D // 16):
                    cute.copy(smem_copy_K_B, tKsK_B[None, None, kb], tKrK_B_v[None, None, kb])
                acc_S.fill(0.0)
                for kb in cutlass.range_constexpr(self.D // 16):
                    cute.gemm(tiled_mma, acc_S, tCrQ_A[None, None, kb], tCrK_B[None, None, kb], acc_S)

                # --- Softmax reconstruction: P = exp2(S * scale_log2e - LSE * log2e) ---
                acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
                kv_start = tile_N * Int32(self.tile_size_N)

                # N-boundary mask
                if kv_start + Int32(self.tile_size_N) > Int32(self.N):
                    for r in cutlass.range_constexpr(num_rows_S):
                        for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                            col_idx = tScS_mn[0, c][1]
                            global_col = kv_start + Int32(col_idx)
                            if global_col >= Int32(self.N):
                                acc_S_mn[r, c] = Float32(-1e30)

                # Causal mask
                if self.causal:
                    last_blk_col = kv_start + Int32(self.tile_size_N - 1)
                    if last_blk_col > m_start + Int32(self.N - self.M):
                        for r in cutlass.range_constexpr(num_rows_S):
                            row_idx = tScS_mn[r, 0][0]
                            global_row = m_start + Int32(row_idx)
                            for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                                col_idx = tScS_mn[0, c][1]
                                global_col = kv_start + Int32(col_idx)
                                if global_col > global_row + Int32(self.N - self.M):
                                    acc_S_mn[r, c] = Float32(-1e30)

                # P = exp2(S * scale_log2e - LSE * log2e)
                for r in cutlass.range_constexpr(num_rows_S):
                    row_idx = tScS_mn[r, 0][0]
                    global_row = m_start + Int32(row_idx)
                    lse_val = Float32(0.0)
                    if global_row < Int32(self.M):
                        lse_val = g_lse[global_row]
                    lse_log2e = lse_val * Float32(1.4426950408889634074)
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        acc_S_mn[r, c] = cute.math.exp2(
                            acc_S_mn[r, c] * Float32(self.scale_log2e) - lse_log2e,
                            fastmath=True,
                        )

                # --- GEMM 2: dP = dO @ V^T (m_block × tile_N), K-dim=D ---
                for kb in cutlass.range_constexpr(self.D // 16):
                    cute.copy(smem_copy_dO_A, tdOsdO_A[None, None, kb], tdOrO_A_v[None, None, kb])
                for kb in cutlass.range_constexpr(self.D // 16):
                    cute.copy(smem_copy_V_B, tVsV_B[None, None, kb], tVrV_B_v[None, None, kb])
                acc_dP.fill(0.0)
                for kb in cutlass.range_constexpr(self.D // 16):
                    cute.gemm(tiled_mma, acc_dP, tCrdO_A[None, None, kb], tCrV_B[None, None, kb], acc_dP)

                # --- dS = P * (dP - dPsum) ---
                acc_dP_mn = self._make_acc_tensor_mn_view(acc_dP)
                for r in cutlass.range_constexpr(num_rows_S):
                    row_idx = tScS_mn[r, 0][0]
                    global_row = m_start + Int32(row_idx)
                    dpsum_val = Float32(0.0)
                    if global_row < Int32(self.M):
                        dpsum_val = g_dpsum[global_row]
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val)

                # --- Store P and dS to smem for dKV GEMMs ---
                rP.store(acc_S.load().to(self.dtype))
                tPrP = thr_copy_C.retile(rP)
                cute.copy(smem_copy_C, tPrP, tPsP_dst)

                rdS.store(acc_dP.load().to(self.dtype))
                tdSrdS = thr_copy_C.retile(rdS)
                cute.copy(smem_copy_C, tdSrdS, tdSsdS_dst)

                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # --- GEMM 3: dV += P^T @ dO (tile_N × D), K-dim=m_block ---
                for kb in cutlass.range_constexpr(self.m_block // 16):
                    cute.copy(smem_copy_Pt_A, tPsPt_A[None, None, kb], tPrPt_A_v[None, None, kb])
                for kb in cutlass.range_constexpr(self.m_block // 16):
                    cute.copy(smem_copy_dOt_B, tdOsdOt_B[None, None, kb], tdOrdOt_B_v[None, None, kb])
                for kb in cutlass.range_constexpr(self.m_block // 16):
                    cute.gemm(tiled_mma, acc_dV, tCrPt_A[None, None, kb], tCrdOt_B[None, None, kb], acc_dV)

                # --- GEMM 4: dK += dS^T @ Q (tile_N × D), K-dim=m_block ---
                for kb in cutlass.range_constexpr(self.m_block // 16):
                    cute.copy(smem_copy_dSt_A, tdSsdSt_A[None, None, kb], tdSrdSt_A_v[None, None, kb])
                for kb in cutlass.range_constexpr(self.m_block // 16):
                    cute.copy(smem_copy_Qt_B, tQsQt_B[None, None, kb], tQrQt_B_v[None, None, kb])
                for kb in cutlass.range_constexpr(self.m_block // 16):
                    cute.gemm(tiled_mma, acc_dK, tCrdSt_A[None, None, kb], tCrQt_B[None, None, kb], acc_dK)

                # --- GEMM 5: dQ = dS @ K (m_block × D), K-dim=tile_N ---
                for kb in cutlass.range_constexpr(self.tile_size_N // 16):
                    cute.copy(smem_copy_dS_dQ_A, tdSsdS_dQ_A[None, None, kb], tdSrdS_dQ_A_v[None, None, kb])
                for kb in cutlass.range_constexpr(self.tile_size_N // 16):
                    cute.copy(smem_copy_Kt_dQ_B, tKsKt_dQ_B[None, None, kb], tKrKt_dQ_B_v[None, None, kb])
                acc_dQ.fill(0.0)
                for kb in cutlass.range_constexpr(self.tile_size_N // 16):
                    cute.gemm(tiled_mma, acc_dQ, tCrdS_dQ_A[None, None, kb], tCrKt_dQ_B[None, None, kb], acc_dQ)

                # Scale dQ and atomicAdd to global dq_accum
                acc_dQ_mn = self._make_acc_tensor_mn_view(acc_dQ)
                for r in cutlass.range_constexpr(num_rows_dQ):
                    row_idx = tSc_dQ_mn[r, 0][0]
                    global_row = m_start + Int32(row_idx)
                    if global_row < Int32(self.M):
                        for c in cutlass.range_constexpr(cute.size(tSc_dQ_mn.shape[1])):
                            col_idx = tSc_dQ_mn[0, c][1]
                            scaled_val = acc_dQ_mn[r, c] * Float32(self.scale_val)
                            elem_offset = global_row * Int32(self.D) + Int32(col_idx)
                            _atomic_add_f32(scaled_val, g_dq_accum.iterator + elem_offset)

                m_idx = m_idx + Int32(1)

            # =============================================================
            # Epilogue: Scale dK, write dK/dV to K/V smem areas for TMA store
            # =============================================================

            # Scale dK
            acc_dK_mn = self._make_acc_tensor_mn_view(acc_dK)
            mc_dKV = cute.make_identity_tensor((self.tile_size_N, self.D))
            tSc_dKV = thr_mma.partition_C(mc_dKV)
            tSc_dKV_mn = self._make_acc_tensor_mn_view(tSc_dKV)
            dkv_shape = tiled_mma.partition_shape_C((self.tile_size_N, self.D))
            num_rows_dKV = dkv_shape[0][1] * dkv_shape[1]
            for r in cutlass.range_constexpr(num_rows_dKV):
                for c in cutlass.range_constexpr(cute.size(tSc_dKV_mn.shape[1])):
                    acc_dK_mn[r, c] = acc_dK_mn[r, c] * Float32(self.scale_val)

            named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

            # CopyUniversal R→S for dK/dV (reuse K/V smem areas)
            _o_swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)
            smem_copy_out = cute.make_tiled_copy_C(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype), tiled_mma
            )
            thr_copy_out = smem_copy_out.get_slice(tidx)

            # dK → page offset 0
            sdK_out = cute.make_tensor(
                cute.recast_ptr(cute.make_ptr(self.dtype, page_ptr, cute.AddressSpace.smem), _o_swz, dtype=self.dtype),
                cute.make_layout((self.tile_size_N, self.D), stride=(self.D, 1)),
            )
            tCrdK_q = cute.make_fragment_like(acc_dK, self.dtype)
            for i in cutlass.range_constexpr(cute.size(acc_dK)):
                tCrdK_q[i] = acc_dK[i].to(self.dtype)
            tOrdK = thr_copy_out.retile(tCrdK_q)
            tOsdK = thr_copy_out.partition_D(sdK_out)
            cute.copy(smem_copy_out, tOrdK, tOsdK)

            # dV → page offset kv_tile_bytes
            sdV_out = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.dtype, page_ptr + Int32(self.kv_tile_bytes), cute.AddressSpace.smem),
                    _o_swz, dtype=self.dtype,
                ),
                cute.make_layout((self.tile_size_N, self.D), stride=(self.D, 1)),
            )
            tCrdV_q = cute.make_fragment_like(acc_dV, self.dtype)
            for i in cutlass.range_constexpr(cute.size(acc_dV)):
                tCrdV_q[i] = acc_dV[i].to(self.dtype)
            tOrdV = thr_copy_out.retile(tCrdV_q)
            tOsdV = thr_copy_out.partition_D(sdV_out)
            cute.copy(smem_copy_out, tOrdV, tOsdV)

    # =========================================================================
    # Backward Store (TMA S->G for dK and dV)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_BH, tile_N, tile_D,
              dk_tma, dk_tma_gmem, dv_tma, dv_tma_gmem):
        """TMA store dK and dV from shared to global memory."""
        _o_swz = cute.make_swizzle(self.swizzle_B, self.swizzle_M, self.swizzle_S)

        # dK at page offset 0
        sdK = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.dtype, page_ptr, cute.AddressSpace.smem), _o_swz, dtype=self.dtype
            ),
            cute.make_layout((self.D, self.tile_size_N, 1)),
        )
        gdK = cute.local_tile(
            dk_tma_gmem, (self.D, self.tile_size_N, 1), (None, None, None),
        )
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            dk_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sdK, 0, 3), cute.group_modes(gdK, 0, 3),
        )
        with cute.arch.elect_one():
            cute.copy(dk_tma, tKsK, tKgK[(None, tile_D, tile_N, tile_BH)])

        # dV at page offset kv_tile_bytes
        sdV = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(self.dtype, page_ptr + Int32(self.kv_tile_bytes), cute.AddressSpace.smem),
                _o_swz, dtype=self.dtype,
            ),
            cute.make_layout((self.D, self.tile_size_N, 1)),
        )
        gdV = cute.local_tile(
            dv_tma_gmem, (self.D, self.tile_size_N, 1), (None, None, None),
        )
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            dv_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sdV, 0, 3), cute.group_modes(gdV, 0, 3),
        )
        with cute.arch.elect_one():
            cute.copy(dv_tma, tVsV, tVgV[(None, tile_D, tile_N, tile_BH)])


__all__ = ["FlashAttentionSm120BwdOp"]
