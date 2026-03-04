# Copyright (c) 2025, Machete Authors
"""Gated Delta Net Fused Ops — forward and backward.

Forward (GDNFusedOp):
    Fuses state recurrence + output. Maintains h[BK,BV] in registers across
    all chunks (forward order), computes output directly from register h.

    Per chunk, 5 GEMMs:
        Phase A: w@h, Q@h, Q@K^T (K-loop, shared h from smem)
        Phase B: Gating on acc_o, gating+mask on acc_A
        Phase C: v_new = u - w@h, v_gated
        Phase D: scores @ v_new (intra-chunk output)
        Phase E: Write O to global
        Phase F: h = decay*h + K^T @ v_gated (state update)

Backward (GDNFusedBwdOp):
    Fuses dv_local (Stage 1) + backward state recurrence (Stage 2).
    Maintains b_dh[BK,BV] in registers across all chunks (reverse order).

    Per chunk, 5 GEMMs:
        Phase A: k @ q^T (intra-chunk backward attention), A_bwd @ do → dv_local
        Phase B: k @ b_dh + dv_local → b_dv (gated)
        Phase C: b_dh decay, q_gated^T @ do, w^T @ b_dv (state gradient update)

Architecture: DMA warp does TMA load (q for fwd, do for bwd).
              MMA warps do cpasync loads + all compute + global stores.
Tiling: (B, H, V) — all T chunks sequential inside compute.
"""

import struct

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu import warp

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import named_barrier_sync


# =============================================================================
# Constants
# =============================================================================

_BT = 64
_BK = 64
_BV = 64
_PAGE_SIZE = 48 * 1024  # 48KB


# =============================================================================
# Shared helpers
# =============================================================================

def _auto_block_sizes(page_size, K, V, elem_bytes=2):
    """Compute largest (BK, BV) that fit in page_size.

    All regions are live simultaneously:
        BT*K*eb + 2*BT*BK*eb + BK*BV*eb + BT*BV*eb + BT*4 + align <= page_size
    Also: s_a (scores [BT,BT]) reuses s_wk0, so BT*BK*eb >= BT*BT*eb.
    """
    BT = _BT
    q_bytes = BT * K * elem_bytes
    for BK in [128, 64]:
        if K % BK != 0:
            continue
        for BV in [128, 64]:
            if V % BV != 0:
                continue
            wk_bytes = BT * BK * elem_bytes
            if wk_bytes < BT * BT * elem_bytes:
                continue
            sh_off = ((q_bytes + 2 * wk_bytes + 127) // 128) * 128
            suv_off = ((sh_off + BK * BV * elem_bytes + 127) // 128) * 128
            gbuf_off = ((suv_off + BT * BV * elem_bytes + 15) // 16) * 16
            total = gbuf_off + BT * 4
            if total <= page_size:
                return BK, BV
    return 64, 64


def _kernel_config(ops):
    """Shared kernel config for GDNFused forward and backward."""
    from machete.megakernel import MegakernelConfig
    from machete.megakernel.megakernel import NUM_DMA_WARPS

    num_mma_warps = _BT // 16
    threads_per_block = (num_mma_warps + NUM_DMA_WARPS) * 32
    page_size = max(
        op.static_dims.get("page_size", DEFAULT_PAGE_SIZE) for op in ops
    )
    return MegakernelConfig(
        threads_per_block=threads_per_block,
        page_size=page_size,
    )


def _fused_init(self):
    """Common __init__ logic for GDNFusedOp and GDNFusedBwdOp."""
    self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)
    self.elem_bytes = 2
    self.scale_val = struct.unpack("f", struct.pack("I", self.scale_bits))[0]

    self.BT = _BT
    self.BK = getattr(self, "BK", _BK)
    self.BV = getattr(self, "BV", _BV)
    self.NK = self.K // self.BK
    self.NT_val = self.T // self.BT

    assert self.K % self.BK == 0
    assert self.V % self.BV == 0
    assert self.T % self.BT == 0
    assert self.NK <= 2, "K-block loading assumes NK <= 2 (K <= 128)"

    self.num_mma_warps = self.BT // 16
    self.num_mma_threads = self.num_mma_warps * 32

    # Swizzles
    self.swizzle_B = 3 if self.BK >= 64 else (2 if self.BK >= 32 else 1)
    self.swizzle_B_v = 3 if self.BV >= 64 else (2 if self.BV >= 32 else 1)

    # cpasync thread layouts (128-bit copies)
    self.async_copy_elems = 128 // (self.elem_bytes * 8)
    self.wk_copy_dim1 = self.BK // self.async_copy_elems
    self.wk_copy_dim0 = self.num_mma_threads // self.wk_copy_dim1
    self.uv_copy_dim1 = self.BV // self.async_copy_elems
    self.uv_copy_dim0 = self.num_mma_threads // self.uv_copy_dim1

    self.inner_iters = 1
    self.inner_depth = 1

    # Smem layout (48KB page):
    #   [0]           s_primary: [BT, K or BV]  (q for fwd, do for bwd)
    #   [primary_end] s_buf:     [BT, BK] × 2   double-buffered (w/k for fwd, q/k/w for bwd)
    #   [aligned]     s_state:   [BK, BV]        swizzled (h for fwd, b_dh for bwd)
    #   [aligned]     s_aux:     [BT, BV]        (u/v_new for fwd, dv_local/b_dv for bwd)
    #   [aligned]     g_buf:     [BT] fp32
    primary_bytes = self.BT * self.K * self.elem_bytes  # fwd: BT*K, bwd overrides
    self._primary_bytes = getattr(self, '_primary_bytes', primary_bytes)
    buf_bytes = self.BT * self.BK * self.elem_bytes
    self._s_primary_offset = 0
    self._s_buf0_offset = self._primary_bytes
    self._s_buf1_offset = self._primary_bytes + buf_bytes
    self._s_state_offset = self._primary_bytes + 2 * buf_bytes
    self._s_state_offset = ((self._s_state_offset + 127) // 128) * 128
    self._s_aux_offset = self._s_state_offset + self.BK * self.BV * self.elem_bytes
    self._s_aux_offset = ((self._s_aux_offset + 127) // 128) * 128
    self._gbuf_offset = self._s_aux_offset + self.BT * self.BV * self.elem_bytes
    self._gbuf_offset = ((self._gbuf_offset + 15) // 16) * 16
    self._total_smem = self._gbuf_offset + self.BT * 4


def _schedule(cls, scale, page_size, tile_sizes, tensors, k_tensor_name, v_tensor_name):
    """Shared scheduling logic for forward and backward."""
    tile_sizes = dict(tile_sizes or {})
    tile_sizes.setdefault("B", 1)
    tile_sizes.setdefault("H", 1)
    tile_sizes.setdefault("V", _BV)

    k_tens = tensors.get(k_tensor_name)
    v_tens = tensors.get(v_tensor_name)
    if k_tens is not None:
        K = k_tens.shape[-1]
        V = v_tens.shape[-1] if v_tens is not None else K
        elem_bytes = k_tens.element_size()
        if scale is None:
            scale = K ** -0.5
    else:
        K, V, elem_bytes = 64, 64, 2
        if scale is None:
            scale = K ** -0.5

    BK, BV = _auto_block_sizes(page_size, K, V, elem_bytes)
    scale_bits = struct.unpack("I", struct.pack("f", scale))[0]

    ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
    ops[0].static_dims["page_size"] = page_size
    ops[0].static_dims["scale_bits"] = scale_bits
    ops[0].static_dims["BK"] = BK
    ops[0].static_dims["BV"] = BV
    return ops


# =============================================================================
# Forward Op
# =============================================================================

class GDNFusedOp(Op):
    """Gated Delta Net forward fused state+output Op.

    Tiling: tile_B=1, tile_H=1, tile_V=BV. All chunks loop inside compute.
    """

    reads = {
        "q":        (None, ("B", "T", "H", "K")),
        "k":        (None, ("B", "T", "H", "K")),
        "w":        (None, ("B", "T", "H", "K")),
        "u":        (None, ("B", "T", "H", "V")),
        "g_cumsum": (cutlass.Float32, ("B", "T", "H")),
    }
    writes = {
        "o": (None, ("B", "T", "H", "V")),
    }
    tile = ("B", "H", "V")
    tma_loads = {"q"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name != "q":
            return None
        K = static_dims["K"]
        return (tile_sizes.get("B", 1), _BT, tile_sizes.get("H", 1), K)

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        return None

    def __init__(self, **config):
        super().__init__(**config)
        assert self.q_dtype in (cutlass.Float16, cutlass.BFloat16)
        self.q_tile_bytes = _BT * self.K * 2
        _fused_init(self)
        self.compute = self.compute_mma

    @classmethod
    def schedule_forward(cls, scale=None, page_size=_PAGE_SIZE, tile_sizes=None, **tensors):
        return _schedule(cls, scale, page_size, tile_sizes, tensors, "q", "u")

    kernel_config = staticmethod(_kernel_config)

    # =========================================================================
    # Load (DMA warp: TMA Q for chunk 0)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_H, tile_V,
             q_tma, q_tma_gmem, work_mbar):
        """TMA Q load into page (single shot, plain layout [BT, K])."""
        from machete.megakernel.interpreter import mbarrier_arrive_expect_tx

        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)

        # Q TMA tile shape (reversed from (B, T, H, K)): (K, H, BT, B)
        sQ = cute.make_tensor(
            cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.K, 1, self.BT, 1)),
        )
        gQ = cute.local_tile(
            q_tma_gmem,
            (self.K, 1, self.BT, 1),
            (None, None, None, None),
        )
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            q_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sQ, 0, 4), cute.group_modes(gQ, 0, 4),
        )

        nbytes = Int32(self.q_tile_bytes)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        # Copy index: (None, K_coord=0, H_coord, T_coord=0, B_coord)
        cute.copy(q_tma, tQgQ[(None, Int32(0), tile_H, Int32(0), tile_B)],
                  tQsQ, tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Compute — All chunks sequential, state + output fused
    # =========================================================================

    @cute.jit
    def compute_mma(
        self, page_ptr, tile_B, tile_H, tile_V,
        q, k, w, u, g_cumsum, o,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        if warp_idx < Int32(self.num_mma_warps):
            # === Swizzles ===
            swz = cute.make_swizzle(self.swizzle_B, 4, 3)
            swz_v = cute.make_swizzle(self.swizzle_B_v, 4, 3)

            # === MMA setup ===
            mma_op = warp.MmaF16BF16Op(self.q_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((self.num_mma_warps, 1, 1)),
                permutation_mnk=(self.num_mma_warps * 16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)

            # === Smem regions ===
            # s_q_bufs: [BT, BK] × NK — SWIZZLED (for K cpasync dest + LdMatrix reads)
            s_q_bufs = []
            for buf_idx in cutlass.range_constexpr(self.NK):
                q_off = self._s_primary_offset + buf_idx * self.BT * self.BK * self.elem_bytes
                s = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(self.q_dtype, page_ptr + Int32(q_off),
                                      cute.AddressSpace.smem, assumed_align=128),
                        swz, dtype=self.q_dtype,
                    ),
                    cute.make_layout((self.BT, self.BK), stride=(self.BK, 1)),
                )
                s_q_bufs.append(s)

            # s_q_plain_bufs: [BT, BK] × NK — PLAIN (for Q reading, stride K)
            # TMA loads Q[BT,K] contiguously at page_ptr. cpasync Q (chunks 1+)
            # also writes here with stride K so both use the same row-major layout.
            s_q_plain_bufs = []
            for buf_idx in cutlass.range_constexpr(self.NK):
                q_col_off = self._s_primary_offset + buf_idx * self.BK * self.elem_bytes
                s = cute.make_tensor(
                    cute.make_ptr(self.q_dtype, page_ptr + Int32(q_col_off),
                                  cute.AddressSpace.smem, assumed_align=128),
                    cute.make_layout((self.BT, self.BK), stride=(self.K, 1)),
                )
                s_q_plain_bufs.append(s)

            s_wk_bufs = []
            for buf_idx in cutlass.range_constexpr(2):
                offset = self._s_buf0_offset if buf_idx == 0 else self._s_buf1_offset
                s = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(self.q_dtype, page_ptr + Int32(offset),
                                      cute.AddressSpace.smem, assumed_align=128),
                        swz, dtype=self.q_dtype,
                    ),
                    cute.make_layout((self.BT, self.BK), stride=(self.BK, 1)),
                )
                s_wk_bufs.append(s)

            s_h = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr + Int32(self._s_state_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz_v, dtype=self.q_dtype,
                ),
                cute.make_layout((self.BK, self.BV), stride=(self.BV, 1)),
            )

            # s_u: NOT swizzled (read via partition_C indexing)
            s_u = cute.make_tensor(
                cute.make_ptr(self.q_dtype, page_ptr + Int32(self._s_aux_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.BT, self.BV), stride=(self.BV, 1)),
            )

            # s_v: swizzled (same region as s_u, for GEMM B operand)
            s_v = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr + Int32(self._s_aux_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz_v, dtype=self.q_dtype,
                ),
                cute.make_layout((self.BT, self.BV), stride=(self.BV, 1)),
            )

            g_buf = cute.make_tensor(
                cute.make_ptr(Float32, page_ptr + Int32(self._gbuf_offset),
                              cute.AddressSpace.smem, assumed_align=16),
                cute.make_layout(self.BT),
            )

            # === Copy atoms ===
            smem_copy_atom_B = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                self.q_dtype,
            )
            smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
            smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

            smem_copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.q_dtype)
            smem_tiled_copy_C = cute.make_tiled_copy_C(smem_copy_atom_C, tiled_mma)
            smem_thr_copy_C = smem_tiled_copy_C.get_slice(tidx)

            # === MMA partitions ===
            # h as B operand [BV, BK]
            s_h_B = cute.make_tensor(s_h.iterator,
                cute.make_layout((self.BV, self.BK), stride=(1, self.BV)))
            _tBsH = thr_mma.partition_B(s_h_B)
            tCrB = tiled_mma.make_fragment_B(_tBsH)
            tBrB_view = smem_thr_copy_B.retile(tCrB)
            tHsH = smem_thr_copy_B.partition_S(s_h_B)

            # K^T as B operand [BK, BT]
            s_kt_B_template = cute.make_tensor(s_wk_bufs[0].iterator,
                cute.make_layout((self.BK, self.BT), stride=(1, self.BK)))
            _tBsKt = thr_mma.partition_B(s_kt_B_template)
            tCrKt = tiled_mma.make_fragment_B(_tBsKt)
            tKrKt_view = smem_thr_copy_B.retile(tCrKt)

            # Per-buffer partitions
            tKsKt_bufs = []
            tCsA_bufs = []
            tCsKt_bufs = []
            tWK_s_bufs = []
            for buf_idx in cutlass.range_constexpr(2):
                s_kt_b = cute.make_tensor(s_wk_bufs[buf_idx].iterator,
                    cute.make_layout((self.BK, self.BT), stride=(1, self.BK)))
                tKsKt_bufs.append(smem_thr_copy_B.partition_S(s_kt_b))
                tCsA_bufs.append(thr_mma.partition_A(s_wk_bufs[buf_idx]))
                tCsKt_bufs.append(thr_mma.partition_A(
                    cute.make_tensor(s_wk_bufs[buf_idx].iterator,
                        cute.make_layout((self.BK, self.BT), stride=(1, self.BK)))
                ))

            tCrA = tiled_mma.make_fragment_A(tCsA_bufs[0])

            # v_new as B operand [BV, BT]
            s_v_B = cute.make_tensor(s_v.iterator,
                cute.make_layout((self.BV, self.BT), stride=(1, self.BV)))
            _tBsVt = thr_mma.partition_B(s_v_B)
            tCrVt = tiled_mma.make_fragment_B(_tBsVt)
            tVrVt_view = smem_thr_copy_B.retile(tCrVt)
            tVsVt = smem_thr_copy_B.partition_S(s_v_B)

            # Scores as A operand [BT, BT] (reuses s_buf0 region)
            s_a = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.q_dtype, page_ptr + Int32(self._s_buf0_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.q_dtype,
                ),
                cute.make_layout((self.BT, self.BT), stride=(self.BT, 1)),
            )
            tCsA_scores = thr_mma.partition_A(s_a)
            tCrA_scores = tiled_mma.make_fragment_A(tCsA_scores)

            # === cpasync setup ===
            wk_async_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), self.q_dtype, num_bits_per_copy=128)
            wk_tiled_copy = cute.make_tiled_copy_tv(
                wk_async_atom,
                cute.make_layout((self.wk_copy_dim0, self.wk_copy_dim1),
                                 stride=(self.wk_copy_dim1, 1)),
                cute.make_layout((1, self.async_copy_elems)),
            )
            wk_thr_copy = wk_tiled_copy.get_slice(tidx)

            tQ_s_plain_bufs = []  # cpasync Q dest (plain, stride K)
            tK_s_bufs = []  # cpasync K dest (swizzled, stride BK)
            tKsKt_q_bufs = []  # B partition (LdMatrix) of K^T from s_q_bufs
            tCsKt_q_bufs = []  # A partition of K^T from s_q_bufs
            for buf_idx in cutlass.range_constexpr(self.NK):
                tQ_s_plain_bufs.append(wk_thr_copy.partition_D(s_q_plain_bufs[buf_idx]))
                tK_s_bufs.append(wk_thr_copy.partition_D(s_q_bufs[buf_idx]))
                s_kt_q = cute.make_tensor(s_q_bufs[buf_idx].iterator,
                    cute.make_layout((self.BK, self.BT), stride=(1, self.BK)))
                tKsKt_q_bufs.append(smem_thr_copy_B.partition_S(s_kt_q))
                tCsKt_q_bufs.append(thr_mma.partition_A(s_kt_q))

            for buf_idx in cutlass.range_constexpr(2):
                tWK_s_bufs.append(wk_thr_copy.partition_D(s_wk_bufs[buf_idx]))

            uv_async_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), self.q_dtype, num_bits_per_copy=128)
            uv_tiled_copy = cute.make_tiled_copy_tv(
                uv_async_atom,
                cute.make_layout((self.uv_copy_dim0, self.uv_copy_dim1),
                                 stride=(self.uv_copy_dim1, 1)),
                cute.make_layout((1, self.async_copy_elems)),
            )
            uv_thr_copy = uv_tiled_copy.get_slice(tidx)
            tU_s = uv_thr_copy.partition_D(s_u)

            # === h accumulators (persistent across chunks) ===
            acc_h_shape = tiled_mma.partition_shape_C((self.BK, self.BV))
            h_accs = []
            for _ki in cutlass.range_constexpr(self.NK):
                _acc = cute.make_fragment(acc_h_shape, Float32)
                _acc.fill(0.0)
                h_accs.append(_acc)

            # === Identity tensors ===
            mc_tv = cute.make_identity_tensor((self.BT, self.BV))
            tCcTV = thr_mma.partition_C(mc_tv)
            mc_AA = cute.make_identity_tensor((self.BT, self.BT))
            tCcAA = thr_mma.partition_C(mc_AA)

            # === Global tensors ===
            kw_base = (tile_B * Int32(self.T * self.H) + tile_H) * Int32(self.K)
            uv_base = (tile_B * Int32(self.T * self.H) + tile_H) * Int32(self.V)
            g_head_base = tile_B * Int32(self.T * self.H) + tile_H

            gQ_head = cute.make_tensor(
                (q.iterator + kw_base).align(16),
                cute.make_layout((self.T, self.K), stride=(self.H * self.K, 1)))
            gK_head = cute.make_tensor(
                (k.iterator + kw_base).align(16),
                cute.make_layout((self.T, self.K), stride=(self.H * self.K, 1)))
            gW_head = cute.make_tensor(
                (w.iterator + kw_base).align(16),
                cute.make_layout((self.T, self.K), stride=(self.H * self.K, 1)))
            gU_head = cute.make_tensor(
                (u.iterator + uv_base).align(16),
                cute.make_layout((self.T, self.V), stride=(self.H * self.V, 1)))
            gG_head = cute.make_tensor(g_cumsum.iterator + g_head_base,
                cute.make_layout((self.T,), stride=(self.H,)))
            gO_head = cute.make_tensor(
                (o.iterator + uv_base).align(16),
                cute.make_layout((self.T, self.V), stride=(self.H * self.V, 1)))

            # === Prologue: issue u[0] cpasync (Q[0] loaded by TMA) ===
            _gU_pre = cute.local_tile(gU_head, (self.BT, self.BV), (Int32(0), tile_V))
            _tU_g_pre = uv_thr_copy.partition_S(_gU_pre)
            for ci in cutlass.range_constexpr(cute.size(tU_s.shape[2])):
                cute.copy(uv_tiled_copy, _tU_g_pre[None, None, ci], tU_s[None, None, ci])
            cute.arch.cp_async_commit_group()

            # ===================================================================
            # Main chunk loop
            # ===================================================================
            chunk_idx = Int32(0)
            while chunk_idx < Int32(self.NT_val):

                # Load g_cumsum while cpasync is in flight
                gG_tile = cute.local_tile(gG_head, (self.BT,), (chunk_idx,))
                if tidx < Int32(self.BT):
                    g_buf[tidx] = gG_tile[tidx]

                # Wait for u cpasync (chunk 0: Q already in smem from TMA,
                # chunk 1+: Q from prev chunk's pipeline cpasync)
                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # Q smem → registers (from plain Q views, stride K per row)
                tCrQ_full_parts = []
                for qi in cutlass.range_constexpr(self.NK):
                    tCsQ_part = thr_mma.partition_A(s_q_plain_bufs[qi])
                    tCrQ_part = tiled_mma.make_fragment_A(tCsQ_part)
                    for qkb in cutlass.range_constexpr(self.BK // 16):
                        cute.autovec_copy(tCsQ_part[None, None, qkb], tCrQ_part[None, None, qkb])
                    tCrQ_full_parts.append(tCrQ_part)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # u stays in smem (read via partition_C)
                tCsU = thr_mma.partition_C(s_u)

                # -------------------------------------------------------
                # Phase A: K-loop — 3 fused GEMMs
                # -------------------------------------------------------
                acc_vp = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), Float32)
                acc_vp.fill(0.0)
                acc_o = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), Float32)
                acc_o.fill(0.0)
                acc_A = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BT)), Float32)
                acc_A.fill(0.0)

                for ki in cutlass.range_constexpr(self.NK):
                    cur = ki % 2

                    # h[ki] → s_h (fp32→fp16)
                    h_tmp = cute.make_fragment_like(h_accs[ki], self.q_dtype)
                    for ci in cutlass.range_constexpr(cute.size(h_accs[ki])):
                        h_tmp[ci] = h_accs[ki][ci].to(self.q_dtype)
                    tOrH = smem_thr_copy_C.retile(h_tmp)
                    tOsH = smem_thr_copy_C.partition_D(s_h)
                    cute.copy(smem_tiled_copy_C, tOrH, tOsH)

                    # Load w[ki] → s_wk[cur] and K[ki] → s_q_bufs[ki]
                    gW_tile = cute.local_tile(gW_head, (self.BT, self.BK),
                                              (chunk_idx, Int32(ki)))
                    tWK_g = wk_thr_copy.partition_S(gW_tile)
                    for ci in cutlass.range_constexpr(cute.size(tWK_s_bufs[cur].shape[2])):
                        cute.copy(wk_tiled_copy, tWK_g[None, None, ci],
                                  tWK_s_bufs[cur][None, None, ci])
                    gK_tile = cute.local_tile(gK_head, (self.BT, self.BK),
                                              (chunk_idx, Int32(ki)))
                    tK_g = wk_thr_copy.partition_S(gK_tile)
                    for ci in cutlass.range_constexpr(cute.size(tK_s_bufs[ki].shape[2])):
                        cute.copy(wk_tiled_copy, tK_g[None, None, ci],
                                  tK_s_bufs[ki][None, None, ci])
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                    # GEMM1-state + GEMM1-output (shared h B operand)
                    for kb in cutlass.range_constexpr(self.BK // 16):
                        cute.copy(smem_tiled_copy_B, tHsH[None, None, kb],
                                  tBrB_view[None, None, kb])
                        cute.autovec_copy(tCsA_bufs[cur][None, None, kb],
                                          tCrA[None, None, kb])
                        # acc_vp += w @ h
                        cute.gemm(tiled_mma, acc_vp,
                                  tCrA[None, None, kb], tCrB[None, None, kb], acc_vp)
                        # acc_o += Q @ h
                        cute.gemm(tiled_mma, acc_o,
                                  tCrQ_full_parts[ki][None, None, kb],
                                  tCrB[None, None, kb], acc_o)

                    # GEMM2-output: acc_A += Q @ K^T (K from s_q_bufs[ki])
                    for kb in cutlass.range_constexpr(self.BK // 16):
                        cute.copy(smem_tiled_copy_B, tKsKt_q_bufs[ki][None, None, kb],
                                  tKrKt_view[None, None, kb])
                        cute.gemm(tiled_mma, acc_A,
                                  tCrQ_full_parts[ki][None, None, kb],
                                  tCrKt[None, None, kb], acc_A)

                    if ki < self.NK - 1:
                        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # -------------------------------------------------------
                # Phase B: Gating on acc_o, gating+mask on acc_A
                # -------------------------------------------------------
                for ci in cutlass.range_constexpr(cute.size(acc_o)):
                    row = tCcTV[ci][0]
                    acc_o[ci] = acc_o[ci] * cute.math.exp(g_buf[Int32(row)], fastmath=True)

                for ci in cutlass.range_constexpr(cute.size(acc_A)):
                    row = tCcAA[ci][0]
                    col = tCcAA[ci][1]
                    if Int32(col) > Int32(row):
                        acc_A[ci] = Float32(0.0)
                    else:
                        acc_A[ci] = acc_A[ci] * cute.math.exp(
                            g_buf[Int32(row)] - g_buf[Int32(col)], fastmath=True)

                # -------------------------------------------------------
                # Phase C: v_new, scores → smem for GEMM3
                # -------------------------------------------------------
                g_last = g_buf[Int32(self.BT - 1)]

                # Compute v_new (non-gated) and v_gated in one pass
                vn_regs = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), self.q_dtype)
                vgated_regs = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), self.q_dtype)
                for ci in cutlass.range_constexpr(cute.size(acc_vp)):
                    vn_val = tCsU[ci].to(Float32) - acc_vp[ci]
                    vn_regs[ci] = vn_val.to(self.q_dtype)
                    row = tCcTV[ci][0]
                    gate = cute.math.exp(g_last - g_buf[Int32(row)], fastmath=True)
                    vgated_regs[ci] = (vn_val * gate).to(self.q_dtype)

                # Write non-gated v_new → s_v for GEMM3
                tOrVN = smem_thr_copy_C.retile(vn_regs)
                tOsVN = smem_thr_copy_C.partition_D(s_v)
                cute.copy(smem_tiled_copy_C, tOrVN, tOsVN)

                # Write scores → s_a (at page_ptr, [BT,BT])
                a_tmp = cute.make_fragment_like(acc_A, self.q_dtype)
                for ci in cutlass.range_constexpr(cute.size(acc_A)):
                    a_tmp[ci] = acc_A[ci].to(self.q_dtype)
                tOrA = smem_thr_copy_C.retile(a_tmp)
                tOsA = smem_thr_copy_C.partition_D(s_a)
                cute.copy(smem_tiled_copy_C, tOrA, tOsA)

                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # -------------------------------------------------------
                # Phase D: GEMM3 — acc_intra = scores @ v_new
                # -------------------------------------------------------
                acc_intra = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), Float32)
                acc_intra.fill(0.0)

                for kb in cutlass.range_constexpr(self.BT // 16):
                    cute.copy(smem_tiled_copy_B, tVsVt[None, None, kb],
                              tVrVt_view[None, None, kb])
                    cute.autovec_copy(tCsA_scores[None, None, kb],
                                      tCrA_scores[None, None, kb])
                    cute.gemm(tiled_mma, acc_intra,
                              tCrA_scores[None, None, kb],
                              tCrVt[None, None, kb], acc_intra)

                # -------------------------------------------------------
                # Phase E: Write O to global (element-wise)
                # -------------------------------------------------------
                gO_tile = cute.local_tile(gO_head, (self.BT, self.BV),
                                          (chunk_idx, tile_V))
                for ci in cutlass.range_constexpr(cute.size(acc_o)):
                    row = tCcTV[ci][0]
                    col = tCcTV[ci][1]
                    o_val = (acc_o[ci] + acc_intra[ci]) * Float32(self.scale_val)
                    gO_tile[Int32(row), Int32(col)] = o_val.to(self.q_dtype)

                # -------------------------------------------------------
                # Phase F: State GEMM2 — h[ki] = decay*h[ki] + k^T @ v_gated
                #   K[ki] already in s_q_bufs[ki] from Phase A (single K-loop)
                # -------------------------------------------------------
                # Write gated v_new → s_v (overwrite non-gated)
                tOrVG = smem_thr_copy_C.retile(vgated_regs)
                tOsVG = smem_thr_copy_C.partition_D(s_v)
                cute.copy(smem_tiled_copy_C, tOrVG, tOsVG)

                decay = cute.math.exp(g_last, fastmath=True)

                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                for ki in cutlass.range_constexpr(self.NK):
                    # Decay h_accs[ki]
                    for ci in cutlass.range_constexpr(cute.size(h_accs[ki])):
                        h_accs[ki][ci] = h_accs[ki][ci] * decay

                    # MMA: h[ki] += k^T[ki] @ v_gated (K from s_q_bufs[ki])
                    for kb in cutlass.range_constexpr(self.BT // 16):
                        cute.copy(smem_tiled_copy_B, tVsVt[None, None, kb],
                                  tBrB_view[None, None, kb])
                        cute.autovec_copy(tCsKt_q_bufs[ki][None, None, kb],
                                          tCrA[None, None, kb])
                        cute.gemm(tiled_mma, h_accs[ki],
                                  tCrA[None, None, kb], tCrB[None, None, kb], h_accs[ki])

                # Pipeline: Q[chunk+1] + u[chunk+1] cpasync
                next_chunk = chunk_idx + Int32(1)
                if next_chunk < Int32(self.NT_val):
                    for qi in cutlass.range_constexpr(self.NK):
                        gQ_nxt = cute.local_tile(
                            gQ_head, (self.BT, self.BK),
                            (next_chunk, Int32(qi)))
                        tQ_g_nxt = wk_thr_copy.partition_S(gQ_nxt)
                        for ci in cutlass.range_constexpr(
                                cute.size(tQ_s_plain_bufs[qi].shape[2])):
                            cute.copy(wk_tiled_copy,
                                      tQ_g_nxt[None, None, ci],
                                      tQ_s_plain_bufs[qi][None, None, ci])
                    gU_nxt = cute.local_tile(gU_head, (self.BT, self.BV),
                                              (next_chunk, tile_V))
                    tU_g_nxt = uv_thr_copy.partition_S(gU_nxt)
                    for ci in cutlass.range_constexpr(cute.size(tU_s.shape[2])):
                        cute.copy(uv_tiled_copy, tU_g_nxt[None, None, ci],
                                  tU_s[None, None, ci])
                    cute.arch.cp_async_commit_group()

                chunk_idx = chunk_idx + Int32(1)


# =============================================================================
# Backward Op
# =============================================================================

class GDNFusedBwdOp(Op):
    """Gated Delta Net backward fused dv_local + state recurrence Op.

    Fuses Stage 1 (dv_local) and Stage 2 (backward state recurrence).
    Iterates chunks in reverse order, maintaining b_dh[BK,BV] in registers.

    Per chunk (reverse order), 5 GEMMs:
        Phase A: GEMM1 k@q^T → A_bwd, gate+upper-tri mask, GEMM2 A_bwd@do → dv_local
        Phase B: GEMM3 k@b_dh → acc_bdv, gate, add dv_local → b_dv, write dv
        Phase C: decay b_dh, GEMM4 q_gated^T@do, GEMM5 w^T@b_dv → update b_dh

    Tiling: tile_B=1, tile_H=1, tile_V=BV. All chunks loop inside compute.
    """

    reads = {
        "q":        (None, ("B", "T", "H", "K")),
        "k":        (None, ("B", "T", "H", "K")),
        "w":        (None, ("B", "T", "H", "K")),
        "g_cumsum": (cutlass.Float32, ("B", "T", "H")),
        "do":       (None, ("B", "T", "H", "V")),
    }
    writes = {
        "dv": (None, ("B", "T", "H", "V")),
    }
    tile = ("B", "H", "V")
    tma_loads = {"do"}

    @classmethod
    def get_tma_tile_shape(cls, tensor_name, tile_sizes, static_dims):
        if tensor_name != "do":
            return None
        BV = static_dims.get("BV", _BV)
        return (tile_sizes.get("B", 1), _BT, tile_sizes.get("H", 1), BV)

    @classmethod
    def get_tma_smem_layout_src(cls, tensor_name, tma_tile_shape, tile_sizes, static_dims):
        return None

    def __init__(self, **config):
        super().__init__(**config)
        assert self.do_dtype in (cutlass.Float16, cutlass.BFloat16)
        BV = getattr(self, "BV", _BV)
        # Primary region holds do[BT, BV] (smaller than forward's q[BT, K])
        self._primary_bytes = _BT * BV * 2
        self.do_tile_bytes = self._primary_bytes
        _fused_init(self)
        self.compute = self.compute_mma

    @classmethod
    def schedule_forward(cls, scale=None, page_size=_PAGE_SIZE, tile_sizes=None, **tensors):
        return _schedule(cls, scale, page_size, tile_sizes, tensors, "q", "do")

    kernel_config = staticmethod(_kernel_config)

    # =========================================================================
    # Load (DMA warp: TMA do for last chunk)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_B, tile_H, tile_V,
             do_tma, do_tma_gmem, work_mbar):
        """TMA do load into page (single shot, plain layout [BT, BV])."""
        from machete.megakernel.interpreter import mbarrier_arrive_expect_tx

        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)

        # do TMA tile shape (reversed from (B, T, H, V)): (BV, H, BT, B)
        sDO = cute.make_tensor(
            cute.make_ptr(self.do_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.BV, 1, self.BT, 1)),
        )
        gDO = cute.local_tile(
            do_tma_gmem,
            (self.BV, 1, self.BT, 1),
            (None, None, None, None),
        )
        tDOsDO, tDOgDO = cute.nvgpu.cpasync.tma_partition(
            do_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sDO, 0, 4), cute.group_modes(gDO, 0, 4),
        )

        nbytes = Int32(self.do_tile_bytes)
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        # Copy index: (None, V_coord=tile_V, H_coord, T_coord=last_chunk, B_coord)
        last_chunk = Int32(self.NT_val - 1)
        cute.copy(do_tma, tDOgDO[(None, tile_V, tile_H, last_chunk * Int32(self.BT), tile_B)],
                  tDOsDO, tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Compute — All chunks reverse sequential, dv_local + bwd_state fused
    # =========================================================================

    @cute.jit
    def compute_mma(
        self, page_ptr, tile_B, tile_H, tile_V,
        q, k, w, g_cumsum, do, dv,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()

        if warp_idx < Int32(self.num_mma_warps):
            # === Swizzles ===
            swz = cute.make_swizzle(self.swizzle_B, 4, 3)
            swz_v = cute.make_swizzle(self.swizzle_B_v, 4, 3)

            # === MMA setup ===
            mma_op = warp.MmaF16BF16Op(self.do_dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(
                mma_op,
                cute.make_layout((self.num_mma_warps, 1, 1)),
                permutation_mnk=(self.num_mma_warps * 16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)

            # === Smem regions ===
            # s_do: [BT, BV] at primary offset — do from TMA (plain, for reading)
            s_do = cute.make_tensor(
                cute.make_ptr(self.do_dtype, page_ptr + Int32(self._s_primary_offset),
                              cute.AddressSpace.smem, assumed_align=128),
                cute.make_layout((self.BT, self.BV), stride=(self.BV, 1)),
            )
            # s_do swizzled (same region, for LdMatrix B operand)
            s_do_swz = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.do_dtype, page_ptr + Int32(self._s_primary_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz_v, dtype=self.do_dtype,
                ),
                cute.make_layout((self.BT, self.BV), stride=(self.BV, 1)),
            )

            # s_buf: [BT, BK] × 2 double-buffered — for q/k/w cpasync
            s_buf = []
            for buf_idx in cutlass.range_constexpr(2):
                offset = self._s_buf0_offset if buf_idx == 0 else self._s_buf1_offset
                s = cute.make_tensor(
                    cute.recast_ptr(
                        cute.make_ptr(self.do_dtype, page_ptr + Int32(offset),
                                      cute.AddressSpace.smem, assumed_align=128),
                        swz, dtype=self.do_dtype,
                    ),
                    cute.make_layout((self.BT, self.BK), stride=(self.BK, 1)),
                )
                s_buf.append(s)

            # s_dh: [BK, BV] swizzled — b_dh staging (register → smem for GEMM)
            s_dh = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.do_dtype, page_ptr + Int32(self._s_state_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz_v, dtype=self.do_dtype,
                ),
                cute.make_layout((self.BK, self.BV), stride=(self.BV, 1)),
            )

            # s_bdv: [BT, BV] swizzled — for b_dv staging (GEMM5 B operand)
            s_bdv = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.do_dtype, page_ptr + Int32(self._s_aux_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz_v, dtype=self.do_dtype,
                ),
                cute.make_layout((self.BT, self.BV), stride=(self.BV, 1)),
            )

            # s_a: [BT, BT] swizzled — A_bwd scores (reuses s_buf0 region)
            s_a = cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(self.do_dtype, page_ptr + Int32(self._s_buf0_offset),
                                  cute.AddressSpace.smem, assumed_align=128),
                    swz, dtype=self.do_dtype,
                ),
                cute.make_layout((self.BT, self.BT), stride=(self.BT, 1)),
            )

            g_buf = cute.make_tensor(
                cute.make_ptr(Float32, page_ptr + Int32(self._gbuf_offset),
                              cute.AddressSpace.smem, assumed_align=16),
                cute.make_layout(self.BT),
            )

            # === Copy atoms ===
            smem_copy_atom_B = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                self.do_dtype,
            )
            smem_tiled_copy_B = cute.make_tiled_copy_B(smem_copy_atom_B, tiled_mma)
            smem_thr_copy_B = smem_tiled_copy_B.get_slice(tidx)

            smem_copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.do_dtype)
            smem_tiled_copy_C = cute.make_tiled_copy_C(smem_copy_atom_C, tiled_mma)
            smem_thr_copy_C = smem_tiled_copy_C.get_slice(tidx)

            # === MMA partitions ===
            # b_dh as B operand [BV, BK] (transposed for k @ b_dh)
            s_dh_B = cute.make_tensor(s_dh.iterator,
                cute.make_layout((self.BV, self.BK), stride=(1, self.BV)))
            _tBsDH = thr_mma.partition_B(s_dh_B)
            tCrB = tiled_mma.make_fragment_B(_tBsDH)
            tBrB_view = smem_thr_copy_B.retile(tCrB)
            tDHsDH = smem_thr_copy_B.partition_S(s_dh_B)

            # q^T / k^T as B operand [BK, BT] (from s_buf)
            s_buf_B_template = cute.make_tensor(s_buf[0].iterator,
                cute.make_layout((self.BK, self.BT), stride=(1, self.BK)))
            _tBsBuf = thr_mma.partition_B(s_buf_B_template)
            tCrBufB = tiled_mma.make_fragment_B(_tBsBuf)
            tBufBrB_view = smem_thr_copy_B.retile(tCrBufB)

            # do^T as B operand [BV, BT] (for q_gated^T @ do)
            s_do_B = cute.make_tensor(s_do_swz.iterator,
                cute.make_layout((self.BV, self.BT), stride=(1, self.BV)))
            _tBsDO = thr_mma.partition_B(s_do_B)
            tCrDOB = tiled_mma.make_fragment_B(_tBsDO)
            tDOrDOB_view = smem_thr_copy_B.retile(tCrDOB)
            tDOsDO_B = smem_thr_copy_B.partition_S(s_do_B)

            # b_dv^T as B operand [BV, BT] (for w^T @ b_dv)
            s_bdv_B = cute.make_tensor(s_bdv.iterator,
                cute.make_layout((self.BV, self.BT), stride=(1, self.BV)))
            _tBsBDV = thr_mma.partition_B(s_bdv_B)
            tCrBDVB = tiled_mma.make_fragment_B(_tBsBDV)
            tBDVrB_view = smem_thr_copy_B.retile(tCrBDVB)
            tBDVsBDV_B = smem_thr_copy_B.partition_S(s_bdv_B)

            # Per-buffer partitions for s_buf
            tCsA_bufs = []      # A partition of s_buf[i] as [BT, BK]
            tCsAt_bufs = []     # A partition of s_buf[i]^T as [BK, BT]
            tBufsBuf_B = []     # B partition (LdMatrix) of s_buf^T as [BK, BT]
            tBuf_s_bufs = []    # cpasync dest
            for buf_idx in cutlass.range_constexpr(2):
                tCsA_bufs.append(thr_mma.partition_A(s_buf[buf_idx]))
                s_bt = cute.make_tensor(s_buf[buf_idx].iterator,
                    cute.make_layout((self.BK, self.BT), stride=(1, self.BK)))
                tBufsBuf_B.append(smem_thr_copy_B.partition_S(s_bt))
                tCsAt_bufs.append(thr_mma.partition_A(s_bt))

            tCrA = tiled_mma.make_fragment_A(tCsA_bufs[0])

            # Scores A_bwd as A operand [BT, BT]
            tCsA_scores = thr_mma.partition_A(s_a)
            tCrA_scores = tiled_mma.make_fragment_A(tCsA_scores)

            # === cpasync setup ===
            wk_async_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), self.do_dtype, num_bits_per_copy=128)
            wk_tiled_copy = cute.make_tiled_copy_tv(
                wk_async_atom,
                cute.make_layout((self.wk_copy_dim0, self.wk_copy_dim1),
                                 stride=(self.wk_copy_dim1, 1)),
                cute.make_layout((1, self.async_copy_elems)),
            )
            wk_thr_copy = wk_tiled_copy.get_slice(tidx)

            for buf_idx in cutlass.range_constexpr(2):
                tBuf_s_bufs.append(wk_thr_copy.partition_D(s_buf[buf_idx]))

            uv_async_atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), self.do_dtype, num_bits_per_copy=128)
            uv_tiled_copy = cute.make_tiled_copy_tv(
                uv_async_atom,
                cute.make_layout((self.uv_copy_dim0, self.uv_copy_dim1),
                                 stride=(self.uv_copy_dim1, 1)),
                cute.make_layout((1, self.async_copy_elems)),
            )
            uv_thr_copy = uv_tiled_copy.get_slice(tidx)
            tDO_s = uv_thr_copy.partition_D(s_do)

            # === b_dh accumulators (persistent across chunks, reverse order) ===
            acc_dh_shape = tiled_mma.partition_shape_C((self.BK, self.BV))
            b_dh_accs = []
            for _ki in cutlass.range_constexpr(self.NK):
                _acc = cute.make_fragment(acc_dh_shape, Float32)
                _acc.fill(0.0)
                b_dh_accs.append(_acc)

            # === Identity tensors ===
            mc_tv = cute.make_identity_tensor((self.BT, self.BV))
            tCcTV = thr_mma.partition_C(mc_tv)
            mc_AA = cute.make_identity_tensor((self.BT, self.BT))
            tCcAA = thr_mma.partition_C(mc_AA)
            # === Global tensors ===
            kw_base = (tile_B * Int32(self.T * self.H) + tile_H) * Int32(self.K)
            uv_base = (tile_B * Int32(self.T * self.H) + tile_H) * Int32(self.V)
            g_head_base = tile_B * Int32(self.T * self.H) + tile_H

            gQ_head = cute.make_tensor(
                (q.iterator + kw_base).align(16),
                cute.make_layout((self.T, self.K), stride=(self.H * self.K, 1)))
            gK_head = cute.make_tensor(
                (k.iterator + kw_base).align(16),
                cute.make_layout((self.T, self.K), stride=(self.H * self.K, 1)))
            gW_head = cute.make_tensor(
                (w.iterator + kw_base).align(16),
                cute.make_layout((self.T, self.K), stride=(self.H * self.K, 1)))
            gDO_head = cute.make_tensor(
                (do.iterator + uv_base).align(16),
                cute.make_layout((self.T, self.V), stride=(self.H * self.V, 1)))
            gG_head = cute.make_tensor(g_cumsum.iterator + g_head_base,
                cute.make_layout((self.T,), stride=(self.H,)))
            gDV_head = cute.make_tensor(
                (dv.iterator + uv_base).align(16),
                cute.make_layout((self.T, self.V), stride=(self.H * self.V, 1)))

            # ===================================================================
            # Main chunk loop (REVERSE order)
            # ===================================================================
            chunk_idx = Int32(self.NT_val - 1)
            while chunk_idx >= Int32(0):

                # Load g_cumsum for this chunk
                gG_tile = cute.local_tile(gG_head, (self.BT,), (chunk_idx,))
                if tidx < Int32(self.BT):
                    g_buf[tidx] = gG_tile[tidx]

                # Wait for do (chunk NT-1: TMA loaded, others: cpasync pipelined)
                cute.arch.cp_async_wait_group(0)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # do smem → registers
                tCsDO = thr_mma.partition_C(s_do)
                do_regs = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), Float32)
                for ci in cutlass.range_constexpr(cute.size(do_regs)):
                    do_regs[ci] = tCsDO[ci].to(Float32)

                g_last = g_buf[Int32(self.BT - 1)]

                # -----------------------------------------------------------
                # Phase A: dv_local — intra-chunk backward attention
                #   GEMM1: acc_A_bwd[BT,BT] += k[BT,BK] @ q^T[BK,BT]
                #   Gate + upper-tri mask
                #   GEMM2: dv_local[BT,BV] = A_bwd[BT,BT] @ do[BT,BV]
                # -----------------------------------------------------------
                acc_A = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BT)), Float32)
                acc_A.fill(0.0)

                for ki in cutlass.range_constexpr(self.NK):
                    # Load k[ki] → s_buf[0], q[ki] → s_buf[1]
                    gK_tile = cute.local_tile(gK_head, (self.BT, self.BK),
                                              (chunk_idx, Int32(ki)))
                    tK_g = wk_thr_copy.partition_S(gK_tile)
                    for ci in cutlass.range_constexpr(cute.size(tBuf_s_bufs[0].shape[2])):
                        cute.copy(wk_tiled_copy, tK_g[None, None, ci],
                                  tBuf_s_bufs[0][None, None, ci])

                    gQ_tile = cute.local_tile(gQ_head, (self.BT, self.BK),
                                              (chunk_idx, Int32(ki)))
                    tQ_g = wk_thr_copy.partition_S(gQ_tile)
                    for ci in cutlass.range_constexpr(cute.size(tBuf_s_bufs[1].shape[2])):
                        cute.copy(wk_tiled_copy, tQ_g[None, None, ci],
                                  tBuf_s_bufs[1][None, None, ci])
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                    # GEMM1: acc_A += k @ q^T  (k from s_buf[0], q^T from s_buf[1])
                    for kb in cutlass.range_constexpr(self.BK // 16):
                        cute.copy(smem_tiled_copy_B, tBufsBuf_B[1][None, None, kb],
                                  tBufBrB_view[None, None, kb])
                        cute.autovec_copy(tCsA_bufs[0][None, None, kb],
                                          tCrA[None, None, kb])
                        cute.gemm(tiled_mma, acc_A,
                                  tCrA[None, None, kb], tCrBufB[None, None, kb], acc_A)

                    if ki < self.NK - 1:
                        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # Scale + gate + upper-tri mask:
                # A_bwd[i,j] = scale * k[i] @ q[j]^T * exp(g[j] - g[i]) for i <= j
                for ci in cutlass.range_constexpr(cute.size(acc_A)):
                    row = tCcAA[ci][0]
                    col = tCcAA[ci][1]
                    if Int32(row) > Int32(col):
                        acc_A[ci] = Float32(0.0)
                    else:
                        acc_A[ci] = acc_A[ci] * Float32(self.scale_val) * cute.math.exp(
                            g_buf[Int32(col)] - g_buf[Int32(row)], fastmath=True)

                # Write A_bwd → s_a for GEMM2
                a_tmp = cute.make_fragment_like(acc_A, self.do_dtype)
                for ci in cutlass.range_constexpr(cute.size(acc_A)):
                    a_tmp[ci] = acc_A[ci].to(self.do_dtype)
                tOrA = smem_thr_copy_C.retile(a_tmp)
                tOsA = smem_thr_copy_C.partition_D(s_a)
                cute.copy(smem_tiled_copy_C, tOrA, tOsA)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # GEMM2: dv_local[BT,BV] = A_bwd[BT,BT] @ do[BT,BV]
                # do is in s_do (swizzled for B operand)
                acc_dvl = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), Float32)
                acc_dvl.fill(0.0)
                for kb in cutlass.range_constexpr(self.BT // 16):
                    cute.copy(smem_tiled_copy_B, tDOsDO_B[None, None, kb],
                              tDOrDOB_view[None, None, kb])
                    cute.autovec_copy(tCsA_scores[None, None, kb],
                                      tCrA_scores[None, None, kb])
                    cute.gemm(tiled_mma, acc_dvl,
                              tCrA_scores[None, None, kb],
                              tCrDOB[None, None, kb], acc_dvl)

                # -----------------------------------------------------------
                # Phase B: b_dv = k @ b_dh * gate + dv_local
                #   GEMM3: acc_bdv[BT,BV] += k[BT,BK] @ b_dh[BK,BV]
                # -----------------------------------------------------------
                acc_bdv = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), Float32)
                acc_bdv.fill(0.0)

                for ki in cutlass.range_constexpr(self.NK):
                    # Convert b_dh_accs[ki] → fp16, write → s_dh
                    dh_tmp = cute.make_fragment_like(b_dh_accs[ki], self.do_dtype)
                    for ci in cutlass.range_constexpr(cute.size(b_dh_accs[ki])):
                        dh_tmp[ci] = b_dh_accs[ki][ci].to(self.do_dtype)
                    tOrDH = smem_thr_copy_C.retile(dh_tmp)
                    tOsDH = smem_thr_copy_C.partition_D(s_dh)
                    cute.copy(smem_tiled_copy_C, tOrDH, tOsDH)

                    # Reload k[ki] → s_buf[0] (needed for GEMM3)
                    gK_tile = cute.local_tile(gK_head, (self.BT, self.BK),
                                              (chunk_idx, Int32(ki)))
                    tK_g = wk_thr_copy.partition_S(gK_tile)
                    for ci in cutlass.range_constexpr(cute.size(tBuf_s_bufs[0].shape[2])):
                        cute.copy(wk_tiled_copy, tK_g[None, None, ci],
                                  tBuf_s_bufs[0][None, None, ci])
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                    # GEMM3: acc_bdv += k @ b_dh
                    for kb in cutlass.range_constexpr(self.BK // 16):
                        cute.copy(smem_tiled_copy_B, tDHsDH[None, None, kb],
                                  tBrB_view[None, None, kb])
                        cute.autovec_copy(tCsA_bufs[0][None, None, kb],
                                          tCrA[None, None, kb])
                        cute.gemm(tiled_mma, acc_bdv,
                                  tCrA[None, None, kb], tCrB[None, None, kb], acc_bdv)

                    if ki < self.NK - 1:
                        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # Gate: acc_bdv *= exp(g_last - g[t]), then add dv_local
                for ci in cutlass.range_constexpr(cute.size(acc_bdv)):
                    row = tCcTV[ci][0]
                    gate = cute.math.exp(g_last - g_buf[Int32(row)], fastmath=True)
                    acc_bdv[ci] = acc_bdv[ci] * gate + acc_dvl[ci]

                # Write dv to global
                gDV_tile = cute.local_tile(gDV_head, (self.BT, self.BV),
                                            (chunk_idx, tile_V))
                for ci in cutlass.range_constexpr(cute.size(acc_bdv)):
                    row = tCcTV[ci][0]
                    col = tCcTV[ci][1]
                    gDV_tile[Int32(row), Int32(col)] = acc_bdv[ci].to(self.do_dtype)

                # Write b_dv → s_bdv (fp16, for GEMM5: w^T @ b_dv)
                bdv_tmp = cute.make_fragment_like(acc_bdv, self.do_dtype)
                for ci in cutlass.range_constexpr(cute.size(acc_bdv)):
                    bdv_tmp[ci] = acc_bdv[ci].to(self.do_dtype)
                tOrBDV = smem_thr_copy_C.retile(bdv_tmp)
                tOsBDV = smem_thr_copy_C.partition_D(s_bdv)
                cute.copy(smem_tiled_copy_C, tOrBDV, tOsBDV)

                # -----------------------------------------------------------
                # Phase C: Update b_dh
                #   Decay: b_dh *= exp(g_last)
                #   GEMM4: b_dh += q^T @ do_gated  (do_gated = do * exp(g) * scale)
                #   GEMM5: b_dh -= w^T @ b_dv
                #
                # Key insight: q_gated^T @ do = q^T @ do_gated when gating is
                # per-row scalar. Gate do instead of q to avoid modifying
                # swizzled smem.
                # -----------------------------------------------------------
                decay = cute.math.exp(g_last, fastmath=True)
                for ki in cutlass.range_constexpr(self.NK):
                    for ci in cutlass.range_constexpr(cute.size(b_dh_accs[ki])):
                        b_dh_accs[ki][ci] = b_dh_accs[ki][ci] * decay

                # Compute do_gated = do * exp(g[row]) * scale → overwrite s_do
                do_gated_fp16 = cute.make_fragment(
                    tiled_mma.partition_shape_C((self.BT, self.BV)), self.do_dtype)
                for ci in cutlass.range_constexpr(cute.size(do_regs)):
                    row = tCcTV[ci][0]
                    do_gated_fp16[ci] = (
                        do_regs[ci] * cute.math.exp(g_buf[Int32(row)], fastmath=True)
                        * Float32(self.scale_val)
                    ).to(self.do_dtype)
                tOrDOG = smem_thr_copy_C.retile(do_gated_fp16)
                tOsDOG = smem_thr_copy_C.partition_D(s_do_swz)
                cute.copy(smem_tiled_copy_C, tOrDOG, tOsDOG)
                named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # Pre-allocate GEMM5 accumulators (separate, to subtract later)
                acc_w_list = []
                for _ki in cutlass.range_constexpr(self.NK):
                    _aw = cute.make_fragment(acc_dh_shape, Float32)
                    _aw.fill(0.0)
                    acc_w_list.append(_aw)

                for ki in cutlass.range_constexpr(self.NK):
                    # Load q[ki] → s_buf[0], w[ki] → s_buf[1]
                    gQ_tile = cute.local_tile(gQ_head, (self.BT, self.BK),
                                              (chunk_idx, Int32(ki)))
                    tQ_g = wk_thr_copy.partition_S(gQ_tile)
                    for ci in cutlass.range_constexpr(cute.size(tBuf_s_bufs[0].shape[2])):
                        cute.copy(wk_tiled_copy, tQ_g[None, None, ci],
                                  tBuf_s_bufs[0][None, None, ci])

                    gW_tile = cute.local_tile(gW_head, (self.BT, self.BK),
                                              (chunk_idx, Int32(ki)))
                    tW_g = wk_thr_copy.partition_S(gW_tile)
                    for ci in cutlass.range_constexpr(cute.size(tBuf_s_bufs[1].shape[2])):
                        cute.copy(wk_tiled_copy, tW_g[None, None, ci],
                                  tBuf_s_bufs[1][None, None, ci])
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                    # GEMM4: b_dh[ki] += q^T[BK,BT] @ do_gated^T[BV,BT]
                    for kb in cutlass.range_constexpr(self.BT // 16):
                        cute.copy(smem_tiled_copy_B, tDOsDO_B[None, None, kb],
                                  tDOrDOB_view[None, None, kb])
                        cute.autovec_copy(tCsAt_bufs[0][None, None, kb],
                                          tCrA[None, None, kb])
                        cute.gemm(tiled_mma, b_dh_accs[ki],
                                  tCrA[None, None, kb], tCrDOB[None, None, kb],
                                  b_dh_accs[ki])

                    # GEMM5: acc_w[ki] += w^T[BK,BT] @ b_dv^T[BV,BT]
                    for kb in cutlass.range_constexpr(self.BT // 16):
                        cute.copy(smem_tiled_copy_B, tBDVsBDV_B[None, None, kb],
                                  tBDVrB_view[None, None, kb])
                        cute.autovec_copy(tCsAt_bufs[1][None, None, kb],
                                          tCrA[None, None, kb])
                        cute.gemm(tiled_mma, acc_w_list[ki],
                                  tCrA[None, None, kb], tCrBDVB[None, None, kb],
                                  acc_w_list[ki])

                    if ki < self.NK - 1:
                        named_barrier_sync(Int32(2), Int32(self.num_mma_threads))

                # Subtract GEMM5 result: b_dh -= w^T @ b_dv
                for ki in cutlass.range_constexpr(self.NK):
                    for ci in cutlass.range_constexpr(cute.size(b_dh_accs[ki])):
                        b_dh_accs[ki][ci] = b_dh_accs[ki][ci] - acc_w_list[ki][ci]

                # Pipeline: do[chunk-1] cpasync
                prev_chunk = chunk_idx - Int32(1)
                if prev_chunk >= Int32(0):
                    gDO_prev = cute.local_tile(gDO_head, (self.BT, self.BV),
                                               (prev_chunk, tile_V))
                    tDO_g_prev = uv_thr_copy.partition_S(gDO_prev)
                    for ci in cutlass.range_constexpr(cute.size(tDO_s.shape[2])):
                        cute.copy(uv_tiled_copy, tDO_g_prev[None, None, ci],
                                  tDO_s[None, None, ci])
                    cute.arch.cp_async_commit_group()

                chunk_idx = chunk_idx - Int32(1)


__all__ = ["GDNFusedOp", "GDNFusedBwdOp"]
