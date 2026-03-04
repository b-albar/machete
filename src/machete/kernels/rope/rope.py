# Copyright (c) 2025, Machete Authors
"""
RoPE (Rotary Position Embedding) Ops for the Megakernel.

Applies rotary position embedding in-place to a query tensor using
pipelined load/compute/store with shared memory staging.

Forward rotation:
    out[..., :half_d] = q[..., :half_d] * cos - q[..., half_d:] * sin
    out[..., half_d:] = q[..., half_d:] * cos + q[..., :half_d] * sin

Backward (inverse rotation — transpose of the rotation matrix):
    out[..., :half_d] = q[..., :half_d] * cos + q[..., half_d:] * sin
    out[..., half_d:] = q[..., half_d:] * cos - q[..., :half_d] * sin

Usage:
    from machete.kernels.rope import RopeOp, RopeBwdOp
    from machete.megakernel import Megakernel

    q_flat = q.view(b * s, h, d).contiguous()
    fwd_ops = RopeOp.schedule(q=q_flat, cos=cos, sin=sin)
    bwd_ops = RopeBwdOp.schedule(q=dq_flat, cos=cos, sin=sin)
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.nvgpu.cpasync import (
    CopyBulkG2SOp,
    CopyBulkS2GOp,
    group_bulk_copy_modes,
)

from machete.megakernel.ops import Op, DEFAULT_PAGE_SIZE
from machete.megakernel.interpreter import mbarrier_arrive_expect_tx


class RopeOp(Op):
    """RoPE forward operation for the megakernel framework.

    Applies rotary position embedding in-place to a query tensor.
    Pipelined: uses load/compute/store with shared memory staging.

    Shared memory layout per page:
        [q_tile:   tile_size_M * tile_size_H * D elements]
        [cos_tile: tile_size_M * D2 elements]
        [sin_tile: tile_size_M * D2 elements]

    Tensor declarations:
        q:   (M, H, D)   — query tensor, bf16/fp16/fp32, modified in-place
        cos: (S, D2)     — cosine table, same dtype as q
        sin: (S, D2)     — sine table, same dtype as q

    Tiling:
        tile_M indexes M (positions), tile_H indexes H (heads).

    Requirements:
        D2 >= 32 (warp-parallel vectorized access)
        tile_size_H evenly divides H
        Tile smem footprint fits in page_size
    """

    # dtype=None means infer from tensor at schedule time (supports bf16/fp16/fp32)
    reads = {
        "q": (None, ("M", "H", "D")),
        "cos": (None, ("S", "D2")),
        "sin": (None, ("S", "D2")),
    }
    writes = {"q": (None, ("M", "H", "D"))}
    tile = ("M", "H")

    def __init__(self, **config):
        super().__init__(**config)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)

        # Element size from dtype (runs at compile time, not in cute.jit)
        if self.q_dtype == cutlass.Float32:
            self.elem_bytes = 4
        elif self.q_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            self.elem_bytes = 4  # fallback

        # Shared memory layout constants
        self.q_row_elems = self.tile_size_H * self.D
        self.q_tile_bytes = self.tile_size_M * self.q_row_elems * self.elem_bytes
        self.cs_tile_bytes = self.tile_size_M * self.D2 * self.elem_bytes
        total_smem = self.q_tile_bytes + 2 * self.cs_tile_bytes

        assert self.D2 >= 32, f"RopeOp requires D2 >= 32, got D2={self.D2} (D={self.D})"
        assert self.H % self.tile_size_H == 0, f"RopeOp: tile_size_H={self.tile_size_H} must divide H={self.H}"
        assert total_smem <= self.page_size, (
            f"RopeOp: tile smem ({total_smem}B) exceeds page_size ({self.page_size}B). "
            f"Reduce tile_size_M or tile_size_H."
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
        D2 = D // 2
        elem_bytes = q.element_size()
        tiles = {}
        # tile_H: target ≥2048 bytes per q row for efficient DMA.
        # For D=128/bf16 this gives tile_H=8, for D=64/bf16 it gives tile_H=16.
        min_tile_H = max(8, 2048 // (D * elem_bytes))
        tile_H = min(H, min_tile_H)
        while tile_H > 1 and H % tile_H != 0:
            tile_H -= 1
        tiles["H"] = tile_H
        # tile_M: q(tile_M * tile_H * D) + cos(tile_M * D2) + sin(tile_M * D2)
        row_bytes = (tile_H * D + D) * elem_bytes
        tiles["M"] = max(1, page_size // row_bytes)
        return tiles

    @classmethod
    def schedule_forward(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        """Schedule RoPE forward with auto-computed tile sizes."""
        tile_sizes = dict(tile_sizes or {})
        auto = cls._auto_tiles(page_size, **tensors)
        for k, v in auto.items():
            tile_sizes.setdefault(k, v)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        return ops

    @classmethod
    def kernel_config(cls, ops):
        """Return recommended MegakernelConfig for scheduled RopeOps."""
        from machete.megakernel import MegakernelConfig

        page_size = ops[0].static_dims.get("page_size", DEFAULT_PAGE_SIZE)
        return MegakernelConfig(page_size=page_size)

    # =========================================================================
    # Load (G→S)
    # =========================================================================

    @cute.jit
    def load(self, page_ptr, tile_M, tile_H, q, cos, sin, work_mbar, prev_tile_M):
        """Load q/cos/sin tile from global to shared memory.

        Having work_mbar in the signature tells the framework this is an
        async load. The framework wraps non-TMA loads in elect_one(), so
        only one DMA-warp thread executes this body.

        prev_tile_M: tile_M of the previous op that used this page slot,
        or -1 if different op (or first use). When M matches, cos/sin
        are still valid in smem (compute doesn't touch them) — skip
        their G2S loads to save bandwidth.
        """
        pos_start = tile_M * self.tile_size_M
        head_start = tile_H * self.tile_size_H

        # When same op reused this page slot with same M, cos/sin
        # are still valid in smem (compute only touches q region).
        _load_cs = Int32(1)
        if prev_tile_M == tile_M:
            _load_cs = Int32(0)

        # Compute total bytes (handle partial M tiles at boundary)
        q_per_pos_bytes = Int32(self.q_row_elems * self.elem_bytes)
        cs_per_pos_bytes = Int32(2 * self.D2 * self.elem_bytes)
        actual_rows = Int32(self.tile_size_M)
        remaining = Int32(self.M) - pos_start
        if remaining < Int32(self.tile_size_M):
            actual_rows = remaining
        total_bytes = actual_rows * q_per_pos_bytes
        if _load_cs == Int32(1):
            total_bytes = total_bytes + actual_rows * cs_per_pos_bytes

        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        mbarrier_arrive_expect_tx(work_mbar, total_bytes)

        # Copy atoms: q rows (tile_size_H*D) and cos/sin rows (D2)
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

                # q: tile_size_H × D contiguous elements per position
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

                # cos/sin: skip if same M (still valid in page from prev use)
                if _load_cs == Int32(1):
                    # cos: D2 elements
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

                    # sin: D2 elements
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
    # Forward Compute
    # =========================================================================

    @cute.jit
    def compute(self, page_ptr, tile_M, tile_H):
        """Apply RoPE rotation for tile_size_M positions × tile_size_H heads."""
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
        thr_layout = cute.make_layout(32)

        for local_pos in range(self.tile_size_M):
            cos_row = cute.make_tensor(
                cos_smem + local_pos * self.D2,
                cute.make_layout(self.D2),
            )
            sin_row = cute.make_tensor(
                sin_smem + local_pos * self.D2,
                cute.make_layout(self.D2),
            )
            cos_part = cute.local_partition(cos_row, thr_layout, lane_idx)
            sin_part = cute.local_partition(sin_row, thr_layout, lane_idx)
            cos_reg = cute.make_fragment_like(cos_part)
            sin_reg = cute.make_fragment_like(sin_part)
            cute.autovec_copy(cos_part, cos_reg)
            cute.autovec_copy(sin_part, sin_reg)

            for local_h in range(warp_idx, self.tile_size_H, num_warps):
                q_base = (local_pos * self.tile_size_H + local_h) * self.D

                q0_row = cute.make_tensor(
                    q_smem + q_base,
                    cute.make_layout(self.D2),
                )
                q1_row = cute.make_tensor(
                    q_smem + q_base + self.D2,
                    cute.make_layout(self.D2),
                )

                q0_part = cute.local_partition(q0_row, thr_layout, lane_idx)
                q1_part = cute.local_partition(q1_row, thr_layout, lane_idx)
                q0_reg = cute.make_fragment_like(q0_part)
                q1_reg = cute.make_fragment_like(q1_part)
                cute.autovec_copy(q0_part, q0_reg)
                cute.autovec_copy(q1_part, q1_reg)

                # Compute rotation in fp32
                out0_reg = cute.make_fragment_like(q0_reg)
                out1_reg = cute.make_fragment_like(q1_reg)
                for i in range(cute.size(q0_reg)):
                    c = cos_reg[i].to(Float32)
                    sn = sin_reg[i].to(Float32)
                    v0 = q0_reg[i].to(Float32)
                    v1 = q1_reg[i].to(Float32)
                    out0_reg[i] = (v0 * c - v1 * sn).to(self.q_dtype)
                    out1_reg[i] = (v1 * c + v0 * sn).to(self.q_dtype)

                # Store back to shared memory
                cute.autovec_copy(out0_reg, q0_part)
                cute.autovec_copy(out1_reg, q1_part)

    # =========================================================================
    # Store (S→G)
    # =========================================================================

    @cute.jit
    def store(self, page_ptr, tile_M, tile_H, q, cos, sin):
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
                        page_ptr + Int32(local_pos * self.q_row_elems * self.elem_bytes),
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


class RopeBwdOp(Op):
    """RoPE backward (inverse rotation) operation.

    Applies inverse rotary embedding: transpose of the forward rotation matrix.
    Reuses RopeOp's load and store — only the compute sign differs.
    """

    reads = {
        "q": (None, ("M", "H", "D")),
        "cos": (None, ("S", "D2")),
        "sin": (None, ("S", "D2")),
    }
    writes = {"q": (None, ("M", "H", "D"))}
    tile = ("M", "H")

    def __init__(self, **config):
        super().__init__(**config)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)

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

        assert self.D2 >= 32, f"RopeBwdOp requires D2 >= 32, got D2={self.D2} (D={self.D})"
        assert self.H % self.tile_size_H == 0, f"RopeBwdOp: tile_size_H={self.tile_size_H} must divide H={self.H}"
        assert total_smem <= self.page_size, (
            f"RopeBwdOp: tile smem ({total_smem}B) exceeds page_size ({self.page_size}B). "
            f"Reduce tile_size_M or tile_size_H."
        )

        self.q_nbits_per_row = self.q_row_elems * self.elem_bytes * 8
        self.cs_nbits_per_row = self.D2 * self.elem_bytes * 8

    # Reuse RopeOp's load and store — zero duplication
    load = RopeOp.load
    store = RopeOp.store

    @classmethod
    def schedule_forward(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        """Schedule RoPE backward with auto-computed tile sizes."""
        tile_sizes = dict(tile_sizes or {})
        auto = RopeOp._auto_tiles(page_size, **tensors)
        for k, v in auto.items():
            tile_sizes.setdefault(k, v)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        return ops

    kernel_config = RopeOp.kernel_config

    @cute.jit
    def compute(self, page_ptr, tile_M, tile_H):
        """Inverse RoPE rotation (transpose of forward rotation matrix)."""
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
        thr_layout = cute.make_layout(32)

        for local_pos in range(self.tile_size_M):
            cos_row = cute.make_tensor(
                cos_smem + local_pos * self.D2,
                cute.make_layout(self.D2),
            )
            sin_row = cute.make_tensor(
                sin_smem + local_pos * self.D2,
                cute.make_layout(self.D2),
            )
            cos_part = cute.local_partition(cos_row, thr_layout, lane_idx)
            sin_part = cute.local_partition(sin_row, thr_layout, lane_idx)
            cos_reg = cute.make_fragment_like(cos_part)
            sin_reg = cute.make_fragment_like(sin_part)
            cute.autovec_copy(cos_part, cos_reg)
            cute.autovec_copy(sin_part, sin_reg)

            for local_h in range(warp_idx, self.tile_size_H, num_warps):
                q_base = (local_pos * self.tile_size_H + local_h) * self.D

                q0_row = cute.make_tensor(
                    q_smem + q_base,
                    cute.make_layout(self.D2),
                )
                q1_row = cute.make_tensor(
                    q_smem + q_base + self.D2,
                    cute.make_layout(self.D2),
                )

                q0_part = cute.local_partition(q0_row, thr_layout, lane_idx)
                q1_part = cute.local_partition(q1_row, thr_layout, lane_idx)
                q0_reg = cute.make_fragment_like(q0_part)
                q1_reg = cute.make_fragment_like(q1_part)
                cute.autovec_copy(q0_part, q0_reg)
                cute.autovec_copy(q1_part, q1_reg)

                # Compute inverse rotation in fp32 (signs flipped vs forward)
                out0_reg = cute.make_fragment_like(q0_reg)
                out1_reg = cute.make_fragment_like(q1_reg)
                for i in range(cute.size(q0_reg)):
                    c = cos_reg[i].to(Float32)
                    sn = sin_reg[i].to(Float32)
                    v0 = q0_reg[i].to(Float32)
                    v1 = q1_reg[i].to(Float32)
                    out0_reg[i] = (v0 * c + v1 * sn).to(self.q_dtype)
                    out1_reg[i] = (v1 * c - v0 * sn).to(self.q_dtype)

                cute.autovec_copy(out0_reg, q0_part)
                cute.autovec_copy(out1_reg, q1_part)


__all__ = ["RopeOp", "RopeBwdOp"]
