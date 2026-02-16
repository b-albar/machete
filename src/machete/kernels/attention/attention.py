# Copyright (c) 2025, Machete Authors
"""
Flash Attention Op for the Megakernel.

Computes scaled dot-product attention with online softmax:
    O[BH, M, D] = softmax(Q[BH, M, D] @ K[BH, N, D]^T / sqrt(D)) @ V[BH, N, D]

Pipelined load/compute/store:
    load:    3D TMA G->S for Q tile
    compute: Read Q from smem, iterate KV positions with async bulk copy
             (CopyBulkG2SOp + mbarrier), online softmax, write O to smem
    store:   3D TMA S->G for O tile

K/V are loaded inside compute via CopyBulkG2SOp (cp.async.bulk — same TMA
hardware) with a compute-local mbarrier for synchronization.

Supports optional causal masking (lower-left aligned):
    Row i in Q can attend to K/V positions 0..(i + N - M).
    Handles prefill (M=N), decode (M=1), and general (M<N).

Usage:
    from machete.kernels.attention import FlashAttentionOp
    from machete.megakernel import Megakernel, MegakernelConfig

    q = q.view(BH, M, D).contiguous()
    k = k.view(BH, N, D).contiguous()
    v = v.view(BH, N, D).contiguous()
    o = torch.zeros_like(q)
    ops = FlashAttentionOp.schedule(q=q, k=k, v=v, o=o, tile_sizes={"M": 4})
    kernel = Megakernel(ops, config=MegakernelConfig())
    kernel.run()
"""

import operator

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkG2SOp,
    group_bulk_copy_modes,
)

from machete.megakernel.ops import Op
from machete.megakernel.interpreter import (
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_arrive_expect_tx,
    mbarrier_wait,
)
from machete.megakernel.paged_memory import PAGE_SIZE


def _align_up(v, a):
    return (v + a - 1) // a * a


class FlashAttentionOp(Op):
    """Flash Attention operation for the megakernel framework.

    Tensors:
        q: (BH, M, D) — query
        k: (BH, N, D) — key
        v: (BH, N, D) — value
        o: (BH, M, D) — output

    Tiling:
        tile_BH=1 (per head), tile_M from schedule, tile_D=D (full).

    Smem page layout:
        [Q/O region:  tile_M × D × elem_bytes]
        [mbarrier:    8 bytes, 8-byte aligned]
        [K row:       D × elem_bytes]
        [V row:       D × elem_bytes]
    """

    reads = {
        "q": (None, ("BH", "M", "D")),
        "k": (None, ("BH", "N", "D")),
        "v": (None, ("BH", "N", "D")),
    }
    writes = {"o": (None, ("BH", "M", "D"))}
    tile = ("BH", "M", "D")

    tma_loads = {"q"}
    tma_stores = {"o"}

    def __init__(self, **config):
        super().__init__(**config)
        self.causal = getattr(self, 'causal', 0)

        if self.q_dtype == cutlass.Float32:
            self.elem_bytes = 4
        elif self.q_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            self.elem_bytes = 4

        # Smem layout offsets
        # cp.async.bulk requires 16-byte aligned shared memory addresses
        self.q_tile_bytes = self.tile_size_M * self.D * self.elem_bytes
        self.mbar_offset = _align_up(self.q_tile_bytes, 8)
        self.k_smem_offset = _align_up(self.mbar_offset + 8, 16)
        self.v_smem_offset = _align_up(self.k_smem_offset + self.D * self.elem_bytes, 16)
        total_smem = self.v_smem_offset + self.D * self.elem_bytes

        assert self.D >= 32, f"FlashAttentionOp requires D >= 32, got D={self.D}"
        assert total_smem <= PAGE_SIZE, (
            f"FlashAttentionOp: smem ({total_smem}B) exceeds PAGE_SIZE ({PAGE_SIZE}B). "
            f"Reduce tile_size_M={self.tile_size_M}."
        )

        self.num_warps = self.threads_per_row // 32
        self.elems_per_lane = self.D // 32
        self.rows_per_warp = (self.tile_size_M + self.num_warps - 1) // self.num_warps

        # Scale factor
        self.scale_val = 1.0 / (self.D ** 0.5)

        # CopyBulkG2SOp config for K/V rows
        self.kv_row_nbits = self.D * self.elem_bytes * 8
        self.kv_row_bytes = self.D * self.elem_bytes

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule_forward(cls, tile_sizes=None, causal=False, **tensors):
        """Schedule flash attention forward, optionally with causal masking."""
        tile_sizes = dict(tile_sizes or {})
        # Always tile BH with size 1 (one head per tile) — the compute
        # loop indexes K/V by tile_BH, so each tile must be a single head.
        tile_sizes.setdefault("BH", 1)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        if causal:
            ops[0].static_dims['causal'] = 1
        return ops

    # =========================================================================
    # Forward Load (3D TMA G->S for Q)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_BH, tile_M, tile_D, q_tma, q_tma_gmem,
             work_mbar):
        """TMA load of Q tile from global to shared memory.

        3D TMA: Q(BH, M, D) permuted to (D, M, BH) for CuTe mode 0 = D.
        Smem layout: (D, tile_M, 1) col-major = (tile_M, D) row-major.
        """
        sQ = cute.make_tensor(
            cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem),
            cute.make_layout((self.D, self.tile_size_M, 1)),
        )
        gQ = cute.local_tile(
            q_tma_gmem, (self.D, self.tile_size_M, 1), (None, None, None),
        )
        tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
            q_tma, Int32(0), cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(gQ, 0, 3),
        )

        nbytes = Int32(self.q_tile_bytes)
        mbar_ptr = cute.make_ptr(
            cutlass.Int64, work_mbar, cute.AddressSpace.smem
        )
        with cute.arch.elect_one():
            mbarrier_arrive_expect_tx(work_mbar, nbytes)
        cute.copy(q_tma, tQgQ[(None, tile_D, tile_M, tile_BH)], tQsQ,
                  tma_bar_ptr=mbar_ptr)

    # =========================================================================
    # Forward Compute
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_BH, tile_M, tile_D, q, k, v, o):
        """Flash attention forward with online softmax.

        1. Read Q rows from smem to registers (loaded by TMA in load phase)
        2. Init compute-local mbarrier for async K/V copies
        3. For each KV position:
           a. Async bulk copy K+V rows from global to smem
           b. Wait on mbarrier
           c. Dot product, causal mask, online softmax, O accumulation
        4. Write O to smem for TMA store
        """
        q_smem = cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem)

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.num_warps
        thr_layout = cute.make_layout(32)
        elems = self.elems_per_lane

        # Head offset for K/V global addressing
        head_offset = tile_BH * Int32(self.N * self.D)

        # Mbarrier address in smem
        kv_mbar_addr = page_ptr + Int32(self.mbar_offset)
        mbar_ptr = cute.make_ptr(
            cutlass.Int64, kv_mbar_addr, cute.AddressSpace.smem
        )

        # CopyBulkG2SOp atom for K/V row copies
        g2s = cute.make_copy_atom(
            CopyBulkG2SOp(), self.q_dtype,
            num_bits_per_copy=self.kv_row_nbits,
        )

        # ----- 1. Read Q rows from smem to fp32 registers -----
        q_f32 = []
        for r in cutlass.range_constexpr(self.rows_per_warp):
            local_row = warp_idx + Int32(r * num_warps)
            q_row = cute.make_tensor(
                q_smem + local_row * self.D,
                cute.make_layout(self.D),
            )
            q_part = cute.local_partition(q_row, thr_layout, lane_idx)
            q_reg = cute.make_fragment_like(q_part)
            cute.autovec_copy(q_part, q_reg)
            q_f32.append([q_reg[i].to(Float32) for i in range(elems)])

        # ----- 2. Init compute-local mbarrier -----
        if warp_idx == Int32(0):
            with cute.arch.elect_one():
                mbarrier_init(kv_mbar_addr, Int32(1))
        mbarrier_init_fence()

        # ----- 3. Init per-row accumulators -----
        m_vals = [Float32(-1e30) for _ in range(self.rows_per_warp)]
        l_vals = [Float32(0.0) for _ in range(self.rows_per_warp)]
        o_acc = [[Float32(0.0) for _ in range(elems)]
                 for _ in range(self.rows_per_warp)]

        # ----- 4. KV loop (unrolled via range_constexpr) -----
        for kv_idx in cutlass.range_constexpr(self.N):
            # 4a. One thread issues async K+V row copy
            kv_offset = Int32(kv_idx * self.D)
            k_global_offset = head_offset + kv_offset
            v_global_offset = head_offset + kv_offset

            if warp_idx == Int32(0):
                with cute.arch.elect_one():
                    mbarrier_arrive_expect_tx(
                        kv_mbar_addr,
                        Int32(2 * self.kv_row_bytes),
                    )
                    # Copy K row
                    g_k = cute.make_tensor(
                        k.iterator + k_global_offset,
                        cute.make_layout((self.D,)),
                    )
                    s_k = cute.make_tensor(
                        cute.make_ptr(
                            self.q_dtype,
                            page_ptr + Int32(self.k_smem_offset),
                            cute.AddressSpace.smem,
                        ),
                        cute.make_layout((self.D,)),
                    )
                    gk_src, sk_dst = group_bulk_copy_modes(g_k, s_k)
                    cute.copy(g2s, gk_src, sk_dst, mbar_ptr=mbar_ptr)

                    # Copy V row
                    g_v = cute.make_tensor(
                        v.iterator + v_global_offset,
                        cute.make_layout((self.D,)),
                    )
                    s_v = cute.make_tensor(
                        cute.make_ptr(
                            self.q_dtype,
                            page_ptr + Int32(self.v_smem_offset),
                            cute.AddressSpace.smem,
                        ),
                        cute.make_layout((self.D,)),
                    )
                    gv_src, sv_dst = group_bulk_copy_modes(g_v, s_v)
                    cute.copy(g2s, gv_src, sv_dst, mbar_ptr=mbar_ptr)

            # 4b. All threads wait for K+V data
            mbarrier_wait(kv_mbar_addr, Int32(kv_idx % 2))

            # 4c. Read K and V from smem
            k_smem_row = cute.make_tensor(
                cute.make_ptr(
                    self.q_dtype,
                    page_ptr + Int32(self.k_smem_offset),
                    cute.AddressSpace.smem,
                ),
                cute.make_layout(self.D),
            )
            k_part = cute.local_partition(k_smem_row, thr_layout, lane_idx)
            k_reg = cute.make_fragment_like(k_part)
            cute.autovec_copy(k_part, k_reg)

            v_smem_row = cute.make_tensor(
                cute.make_ptr(
                    self.q_dtype,
                    page_ptr + Int32(self.v_smem_offset),
                    cute.AddressSpace.smem,
                ),
                cute.make_layout(self.D),
            )
            v_part = cute.local_partition(v_smem_row, thr_layout, lane_idx)
            v_reg = cute.make_fragment_like(v_part)
            cute.autovec_copy(v_part, v_reg)

            # 4d. Process each Q row assigned to this warp
            for r in cutlass.range_constexpr(self.rows_per_warp):
                local_row = warp_idx + Int32(r * num_warps)
                if local_row < Int32(self.tile_size_M):
                    # Dot product q·k
                    partial = Float32(0.0)
                    for i in cutlass.range_constexpr(elems):
                        partial = partial + q_f32[r][i] * k_reg[i].to(Float32)
                    s = cute.arch.warp_reduction(partial, operator.add)
                    s = s * Float32(self.scale_val)

                    # Causal mask
                    if self.causal:
                        q_pos = tile_M * self.tile_size_M + local_row
                        max_kv = q_pos + Int32(self.N - self.M)
                        if Int32(kv_idx) > max_kv:
                            s = Float32(-1e30)

                    # Online softmax
                    m_new = m_vals[r]
                    if s > m_vals[r]:
                        m_new = s
                    correction = cute.math.exp(
                        m_vals[r] - m_new, fastmath=True
                    )
                    p = cute.math.exp(s - m_new, fastmath=True)
                    l_vals[r] = l_vals[r] * correction + p

                    # O accumulation
                    for i in cutlass.range_constexpr(elems):
                        o_acc[r][i] = (
                            o_acc[r][i] * correction
                            + p * v_reg[i].to(Float32)
                        )

                    m_vals[r] = m_new

        # ----- 5. Write O to smem Q region -----
        for r in cutlass.range_constexpr(self.rows_per_warp):
            local_row = warp_idx + Int32(r * num_warps)
            row_idx = tile_M * self.tile_size_M + local_row
            if local_row < Int32(self.tile_size_M):
                if row_idx < Int32(self.M):
                    inv_l = Float32(1.0) / l_vals[r]
                    o_row = cute.make_tensor(
                        q_smem + local_row * self.D,
                        cute.make_layout(self.D),
                    )
                    o_part = cute.local_partition(
                        o_row, thr_layout, lane_idx
                    )
                    o_reg = cute.make_fragment_like(o_part)
                    for i in cutlass.range_constexpr(elems):
                        o_reg[i] = (
                            o_acc[r][i] * inv_l
                        ).to(self.q_dtype)
                    cute.autovec_copy(o_reg, o_part)

    # =========================================================================
    # Forward Store (3D TMA S->G for O)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_BH, tile_M, tile_D, o_tma, o_tma_gmem):
        """TMA store of O from shared to global memory.

        Smem (D, tile_M, 1) col-major = (tile_M, D) row-major.
        TMA handles boundary conditions for partial last M tile.
        """
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
