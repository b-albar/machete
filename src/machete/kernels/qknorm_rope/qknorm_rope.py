# Copyright (c) 2025, Machete Authors
"""
Fused per-head RMSNorm + RoPE Op for the Megakernel.

Applies per-head RMSNorm normalization followed by rotary position embedding
in a single pass through shared memory, eliminating an extra global memory
round-trip compared to separate RMSNormOp + RopeOp.

Forward:
    For each head h in [0, H):
        rms = sqrt(mean(q[..., h, :]^2) + eps)
        q_normed = q[..., h, :] / rms * norm_weight
        q_normed[..., :D2]      = q_normed[..., :D2] * cos - q_normed[..., D2:2*D2] * sin
        q_normed[..., D2:2*D2]  = q_normed[..., D2:2*D2] * cos + q_normed[..., :D2] * sin
        q_normed[..., 2*D2:]    = unchanged (partial RoPE passthrough)

Usage:
    from machete.kernels.qknorm_rope import QKNormRopeOp

    q_flat = q.view(b * s, h, d).contiguous()
    ops = QKNormRopeOp.schedule(q=q_flat, norm_weight=weight, cos=cos, sin=sin)
"""

import operator

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkG2SOp,
    CopyBulkS2GOp,
    group_bulk_copy_modes,
)

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import mbarrier_arrive_expect_tx, named_barrier_sync


class QKNormRopeOp(Op):
    """Fused per-head RMSNorm + RoPE forward operation.

    Processes a single (M, H, D) tensor in-place: normalizes each head's
    D-element row, then applies rotary embedding to the first 2*D2 dims.
    Schedule once for Q (with q_norm weight) and once for K (with k_norm
    weight) to handle GQA naturally.

    Shared memory layout per page:
        [q_tile:   tile_size_M * tile_size_H * D elements]
        [cos_tile: tile_size_M * D2 elements]
        [sin_tile: tile_size_M * D2 elements]

    Tensor declarations:
        q:           (M, H, D)  — query/key tensor, modified in-place
        norm_weight: (D,)       — RMSNorm weight, shared across heads
        cos:         (S, D2)    — cosine table (D2 = rotary_dim // 2)
        sin:         (S, D2)    — sine table
    """

    reads = {
        "q": (None, ("M", "H", "D")),
        "norm_weight": (None, ("D",)),
        "cos": (None, ("S", "D2")),
        "sin": (None, ("S", "D2")),
    }
    writes = {"q": (None, ("M", "H", "D"))}
    tile = ("M", "H")

    def __init__(self, **config):
        super().__init__(**config)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)
        self.eps = getattr(self, "eps", 1e-6)

        if self.q_dtype == cutlass.Float32:
            self.elem_bytes = 4
        elif self.q_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            self.elem_bytes = 4

        self.q_row_elems = self.tile_size_H * self.D
        self.q_tile_bytes = self.tile_size_M * self.q_row_elems * self.elem_bytes
        self.cs_tile_bytes = self.tile_size_M * self.D2 * self.elem_bytes
        total_smem = self.q_tile_bytes + 2 * self.cs_tile_bytes

        assert self.D >= 32, f"QKNormRopeOp requires D >= 32, got D={self.D}"
        assert self.D2 >= 16, f"QKNormRopeOp requires D2 >= 16, got D2={self.D2}"
        assert self.H % self.tile_size_H == 0, (
            f"QKNormRopeOp: tile_size_H={self.tile_size_H} must divide H={self.H}"
        )
        assert total_smem <= self.page_size, (
            f"QKNormRopeOp: tile smem ({total_smem}B) exceeds page_size "
            f"({self.page_size}B). Reduce tile_size_M or tile_size_H."
        )

        self.q_nbits_per_row = self.q_row_elems * self.elem_bytes * 8
        self.cs_nbits_per_row = self.D2 * self.elem_bytes * 8

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def _auto_tiles(cls, page_size, **tensors):
        """Compute tile_sizes M and H that fit in page_size."""
        q = tensors.get("q")
        if q is None:
            return {}
        M, H, D = q.shape
        cos = tensors.get("cos")
        D2 = cos.shape[1] if cos is not None else D // 2
        elem_bytes = q.element_size()
        tiles = {}
        min_tile_H = max(8, 2048 // (D * elem_bytes))
        tile_H = min(H, min_tile_H)
        while tile_H > 1 and H % tile_H != 0:
            tile_H -= 1
        tiles["H"] = tile_H
        row_bytes = (tile_H * D + 2 * D2) * elem_bytes
        tiles["M"] = max(1, page_size // row_bytes)
        return tiles

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE,
                         eps=1e-6, **tensors):
        """Schedule fused QKNorm+RoPE forward with auto-computed tile sizes."""
        tile_sizes = dict(tile_sizes or {})
        auto = cls._auto_tiles(page_size, **tensors)
        for k, v in auto.items():
            tile_sizes.setdefault(k, v)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        ops[0].static_dims["eps"] = eps
        return ops

    @classmethod
    def kernel_config(cls, ops):
        from machete.megakernel import MegakernelConfig
        page_size = ops[0].static_dims.get("page_size", DEFAULT_PAGE_SIZE)
        return MegakernelConfig(page_size=page_size)

    # =========================================================================
    # Load (G→S)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_M, tile_H, q, norm_weight, cos, sin,
             work_mbar):
        """Load q/cos/sin tile from global to shared memory.

        norm_weight is in the signature because the framework passes all
        tensors when expects_tensors=True, but it's read from global in
        compute — not loaded to smem.
        """
        pos_start = tile_M * self.tile_size_M
        head_start = tile_H * self.tile_size_H

        q_per_pos_bytes = Int32(self.q_row_elems * self.elem_bytes)
        cs_per_pos_bytes = Int32(2 * self.D2 * self.elem_bytes)
        actual_rows = Int32(self.tile_size_M)
        remaining = Int32(self.M) - pos_start
        if remaining < Int32(self.tile_size_M):
            actual_rows = remaining
        total_bytes = actual_rows * (q_per_pos_bytes + cs_per_pos_bytes)

        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        mbarrier_arrive_expect_tx(work_mbar, total_bytes)

        g2s_q = cute.make_copy_atom(
            CopyBulkG2SOp(),
            self.q_dtype,
            num_bits_per_copy=self.q_nbits_per_row,
        )
        g2s_cs = cute.make_copy_atom(
            CopyBulkG2SOp(),
            self.q_dtype,
            num_bits_per_copy=self.cs_nbits_per_row,
        )

        cos_smem_start = page_ptr + Int32(self.q_tile_bytes)
        sin_smem_start = page_ptr + Int32(self.q_tile_bytes + self.cs_tile_bytes)

        for local_pos in range(self.tile_size_M):
            pos = pos_start + local_pos
            if pos < self.M:
                s = pos % self.S

                g_q = cute.make_tensor(
                    q.iterator + pos * (self.H * self.D) + head_start * self.D,
                    cute.make_layout((self.q_row_elems,)),
                )
                s_q = cute.make_tensor(
                    cute.make_ptr(
                        self.q_dtype,
                        page_ptr + Int32(local_pos * self.q_row_elems * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.q_row_elems,)),
                )
                gsrc, sdst = group_bulk_copy_modes(g_q, s_q)
                cute.copy(g2s_q, gsrc, sdst, mbar_ptr=mbar_ptr)

                g_cos = cute.make_tensor(
                    cos.iterator + s * self.D2,
                    cute.make_layout((self.D2,)),
                )
                s_cos = cute.make_tensor(
                    cute.make_ptr(
                        self.q_dtype,
                        cos_smem_start + Int32(local_pos * self.D2 * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.D2,)),
                )
                gc_src, sc_dst = group_bulk_copy_modes(g_cos, s_cos)
                cute.copy(g2s_cs, gc_src, sc_dst, mbar_ptr=mbar_ptr)

                g_sin = cute.make_tensor(
                    sin.iterator + s * self.D2,
                    cute.make_layout((self.D2,)),
                )
                s_sin = cute.make_tensor(
                    cute.make_ptr(
                        self.q_dtype,
                        sin_smem_start + Int32(local_pos * self.D2 * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.D2,)),
                )
                gs_src, ss_dst = group_bulk_copy_modes(g_sin, s_sin)
                cute.copy(g2s_cs, gs_src, ss_dst, mbar_ptr=mbar_ptr)

    # =========================================================================
    # Compute — Fused RMSNorm + RoPE
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, norm_weight):
        """Per-head RMSNorm then RoPE rotation."""
        q_smem = cute.make_ptr(self.q_dtype, page_ptr, cute.AddressSpace.smem)
        cos_smem = cute.make_ptr(
            self.q_dtype,
            page_ptr + Int32(self.q_tile_bytes),
            cute.AddressSpace.smem,
        )
        sin_smem = cute.make_ptr(
            self.q_dtype,
            page_ptr + Int32(self.q_tile_bytes + self.cs_tile_bytes),
            cute.AddressSpace.smem,
        )

        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        num_warps = self.threads_per_row // 32
        thr_layout_D = cute.make_layout(32)

        # Load norm_weight from global → regs (once, reused for all pos/heads)
        w_row = cute.make_tensor(
            norm_weight.iterator,
            cute.make_layout(self.D),
        )
        w_part = cute.local_partition(w_row, thr_layout_D, lane_idx)
        w_reg = cute.make_fragment_like(w_part)
        cute.autovec_copy(w_part, w_reg)

        eps_val = Float32(self.eps)
        inv_D = Float32(1.0 / self.D)

        if self.D2 >= 32:
            # =================================================================
            # Path A: D2 >= 32 — fused normalize + rotate in registers
            # =================================================================
            d2_regs = self.D2 // 32  # register elements per lane in rotary part

            for local_pos in range(self.tile_size_M):
                # Load cos/sin for this position (32-lane partition over D2)
                cos_row = cute.make_tensor(
                    cos_smem + local_pos * self.D2,
                    cute.make_layout(self.D2),
                )
                sin_row = cute.make_tensor(
                    sin_smem + local_pos * self.D2,
                    cute.make_layout(self.D2),
                )
                cos_part = cute.local_partition(cos_row, thr_layout_D, lane_idx)
                sin_part = cute.local_partition(sin_row, thr_layout_D, lane_idx)
                cos_reg = cute.make_fragment_like(cos_part)
                sin_reg = cute.make_fragment_like(sin_part)
                cute.autovec_copy(cos_part, cos_reg)
                cute.autovec_copy(sin_part, sin_reg)

                for local_h in range(warp_idx, self.tile_size_H, num_warps):
                    q_base = (local_pos * self.tile_size_H + local_h) * self.D

                    # Load full D-row from smem → regs
                    q_full_row = cute.make_tensor(
                        q_smem + q_base, cute.make_layout(self.D),
                    )
                    q_full_part = cute.local_partition(
                        q_full_row, thr_layout_D, lane_idx,
                    )
                    q_reg = cute.make_fragment_like(q_full_part)
                    cute.autovec_copy(q_full_part, q_reg)

                    # RMSNorm: sum of squares → warp reduction
                    partial_sq = Float32(0.0)
                    for i in range(cute.size(q_reg)):
                        val = q_reg[i].to(Float32)
                        partial_sq = partial_sq + val * val
                    sum_sq = cute.arch.warp_reduction(partial_sq, operator.add)
                    rstd = cute.math.rsqrt(sum_sq * inv_D + eps_val, fastmath=True)

                    # Normalize + apply weight
                    for i in range(cute.size(q_reg)):
                        q_reg[i] = (
                            q_reg[i].to(Float32) * rstd * w_reg[i].to(Float32)
                        ).to(self.q_dtype)

                    # RoPE in-register: reg layout with stride-32 partition
                    # reg[k] corresponds to position lane_idx + k*32
                    # q0 = positions [0, D2) → reg indices [0, d2_regs)
                    # q1 = positions [D2, 2*D2) → reg indices [d2_regs, 2*d2_regs)
                    for k in range(d2_regs):
                        v0 = q_reg[k].to(Float32)
                        v1 = q_reg[k + d2_regs].to(Float32)
                        c = cos_reg[k].to(Float32)
                        sn = sin_reg[k].to(Float32)
                        q_reg[k] = (v0 * c - v1 * sn).to(self.q_dtype)
                        q_reg[k + d2_regs] = (v1 * c + v0 * sn).to(self.q_dtype)

                    # Write back full D-row to smem
                    cute.autovec_copy(q_reg, q_full_part)
        else:
            # =================================================================
            # Path B: D2 < 32 — two-pass with barrier
            # =================================================================
            thr_count_d2 = self.D2
            thr_layout_d2 = cute.make_layout(thr_count_d2)
            compute_threads = Int32(self.threads_per_row)

            for local_pos in range(self.tile_size_M):
                # Pass 1: RMSNorm (all 32 lanes per warp)
                for local_h in range(warp_idx, self.tile_size_H, num_warps):
                    q_base = (local_pos * self.tile_size_H + local_h) * self.D

                    q_full_row = cute.make_tensor(
                        q_smem + q_base, cute.make_layout(self.D),
                    )
                    q_full_part = cute.local_partition(
                        q_full_row, thr_layout_D, lane_idx,
                    )
                    q_reg = cute.make_fragment_like(q_full_part)
                    cute.autovec_copy(q_full_part, q_reg)

                    partial_sq = Float32(0.0)
                    for i in range(cute.size(q_reg)):
                        val = q_reg[i].to(Float32)
                        partial_sq = partial_sq + val * val
                    sum_sq = cute.arch.warp_reduction(partial_sq, operator.add)
                    rstd = cute.math.rsqrt(sum_sq * inv_D + eps_val, fastmath=True)

                    for i in range(cute.size(q_reg)):
                        q_reg[i] = (
                            q_reg[i].to(Float32) * rstd * w_reg[i].to(Float32)
                        ).to(self.q_dtype)

                    cute.autovec_copy(q_reg, q_full_part)

                # Barrier: ensure all norm writes are visible before RoPE reads
                named_barrier_sync(Int32(2), compute_threads)

                # Pass 2: RoPE (D2 lanes only)
                if lane_idx < Int32(thr_count_d2):
                    cos_row = cute.make_tensor(
                        cos_smem + local_pos * self.D2,
                        cute.make_layout(self.D2),
                    )
                    sin_row = cute.make_tensor(
                        sin_smem + local_pos * self.D2,
                        cute.make_layout(self.D2),
                    )
                    cos_part = cute.local_partition(
                        cos_row, thr_layout_d2, lane_idx,
                    )
                    sin_part = cute.local_partition(
                        sin_row, thr_layout_d2, lane_idx,
                    )
                    cos_reg = cute.make_fragment_like(cos_part)
                    sin_reg = cute.make_fragment_like(sin_part)
                    cute.autovec_copy(cos_part, cos_reg)
                    cute.autovec_copy(sin_part, sin_reg)

                    for local_h in range(warp_idx, self.tile_size_H, num_warps):
                        q_base = (
                            (local_pos * self.tile_size_H + local_h) * self.D
                        )
                        q0_row = cute.make_tensor(
                            q_smem + q_base, cute.make_layout(self.D2),
                        )
                        q1_row = cute.make_tensor(
                            q_smem + q_base + self.D2,
                            cute.make_layout(self.D2),
                        )
                        q0_part = cute.local_partition(
                            q0_row, thr_layout_d2, lane_idx,
                        )
                        q1_part = cute.local_partition(
                            q1_row, thr_layout_d2, lane_idx,
                        )
                        q0_reg = cute.make_fragment_like(q0_part)
                        q1_reg = cute.make_fragment_like(q1_part)
                        cute.autovec_copy(q0_part, q0_reg)
                        cute.autovec_copy(q1_part, q1_reg)

                        out0_reg = cute.make_fragment_like(q0_reg)
                        out1_reg = cute.make_fragment_like(q1_reg)
                        for i in range(cute.size(q0_reg)):
                            c = cos_reg[i].to(Float32)
                            sn = sin_reg[i].to(Float32)
                            v0 = q0_reg[i].to(Float32)
                            v1 = q1_reg[i].to(Float32)
                            out0_reg[i] = (v0 * c - v1 * sn).to(self.q_dtype)
                            out1_reg[i] = (v1 * c + v0 * sn).to(
                                self.q_dtype,
                            )

                        cute.autovec_copy(out0_reg, q0_part)
                        cute.autovec_copy(out1_reg, q1_part)

                # Barrier after RoPE writes before next position iteration
                named_barrier_sync(Int32(2), compute_threads)

    # =========================================================================
    # Store (S→G)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_M, tile_H, q, norm_weight, cos, sin):
        """Store modified q from shared to global memory."""
        s2g = cute.make_copy_atom(
            CopyBulkS2GOp(),
            self.q_dtype,
            num_bits_per_copy=self.q_nbits_per_row,
        )
        pos_start = tile_M * self.tile_size_M
        head_start = tile_H * self.tile_size_H

        for local_pos in range(self.tile_size_M):
            pos = pos_start + local_pos
            if pos < self.M:
                s_tile = cute.make_tensor(
                    cute.make_ptr(
                        self.q_dtype,
                        page_ptr + Int32(
                            local_pos * self.q_row_elems * self.elem_bytes
                        ),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.q_row_elems,)),
                )
                g_tile = cute.make_tensor(
                    q.iterator + pos * (self.H * self.D) + head_start * self.D,
                    cute.make_layout((self.q_row_elems,)),
                )
                ssrc, gdst = group_bulk_copy_modes(s_tile, g_tile)
                cute.copy(s2g, ssrc, gdst)


__all__ = ["QKNormRopeOp"]
