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

Tensor shapes follow LLM conventions:
    q:   (B, S, NH, HD) — batch, sequence, num_heads, head_dim
    cos: (SL, D2)       — sequence length (cos table), rotary_dim // 2
    sin: (SL, D2)       — same as cos

Usage:
    from machete.kernels.rope import RopeOp, RopeBwdOp
    from machete.megakernel import Megakernel

    fwd_ops = RopeOp.schedule(q=q, cos=cos, sin=sin)
    bwd_ops = RopeBwdOp.schedule(q=dq, cos=cos, sin=sin)
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

    Supports partial RoPE (partial_rotary_factor < 1.0): only the first
    2*D2 dimensions of each head are rotated, the rest pass through
    unchanged. D2 is determined by the cos/sin table shape.

    Shared memory layout per page:
        [q_tile:   tile_size_S * tile_size_NH * HD elements]
        [cos_tile: tile_size_S * D2 elements]
        [sin_tile: tile_size_S * D2 elements]

    Tensor declarations:
        q:   (B, S, NH, HD) — query tensor, bf16/fp16/fp32, modified in-place
        cos: (SL, D2)       — cosine table, same dtype as q (D2 = rotary_dim // 2)
        sin: (SL, D2)       — sine table, same dtype as q

    Tiling:
        tile_B indexes B (batch, always 1), tile_S indexes S (positions),
        tile_NH indexes NH (heads).

    Requirements:
        D2 >= 16 (warp-parallel vectorized access)
        tile_size_NH evenly divides NH
        Tile smem footprint fits in page_size
    """

    # dtype=None means infer from tensor at schedule time (supports bf16/fp16/fp32)
    reads = {
        "q": (None, ("B", "S", "NH", "HD")),
        "cos": (None, ("SL", "D2")),
        "sin": (None, ("SL", "D2")),
    }
    writes = {"q": (None, ("B", "S", "NH", "HD"))}
    tile = ("B", "S", "NH")

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
        self.q_row_elems = self.tile_size_NH * self.HD
        self.q_tile_bytes = self.tile_size_S * self.q_row_elems * self.elem_bytes
        self.cs_tile_bytes = self.tile_size_S * self.D2 * self.elem_bytes
        total_smem = self.q_tile_bytes + 2 * self.cs_tile_bytes

        assert self.D2 >= 16, f"RopeOp requires D2 >= 16, got D2={self.D2} (HD={self.HD})"
        assert self.NH % self.tile_size_NH == 0, f"RopeOp: tile_size_NH={self.tile_size_NH} must divide NH={self.NH}"
        assert total_smem <= self.page_size, (
            f"RopeOp: tile smem ({total_smem}B) exceeds page_size ({self.page_size}B). "
            f"Reduce tile_size_S or tile_size_NH."
        )

        self.q_nbits_per_row = self.q_row_elems * self.elem_bytes * 8
        self.cs_nbits_per_row = self.D2 * self.elem_bytes * 8

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def _auto_tiles(cls, page_size, **tensors):
        """Compute tile_sizes S and NH that fit in page_size."""
        q = tensors.get("q")
        if q is None:
            return {}
        B, S, NH, HD = q.shape
        cos = tensors.get("cos")
        D2 = cos.shape[1] if cos is not None else HD // 2
        elem_bytes = q.element_size()
        tiles = {}
        # tile_NH: target ≥2048 bytes per q row for efficient DMA.
        # For HD=128/bf16 this gives tile_NH=8, for HD=64/bf16 it gives tile_NH=16.
        min_tile_NH = max(8, 2048 // (HD * elem_bytes))
        tile_NH = min(NH, min_tile_NH)
        while tile_NH > 1 and NH % tile_NH != 0:
            tile_NH -= 1
        tiles["NH"] = tile_NH
        # tile_S: q(tile_S * tile_NH * HD) + cos(tile_S * D2) + sin(tile_S * D2)
        # Round down to power-of-2 so tile_S divides common GEMM tile_S (128),
        # enabling tile-level barrier dependencies when fused with GemmOp.
        row_bytes = (tile_NH * HD + 2 * D2) * elem_bytes
        raw_S = max(1, page_size // row_bytes)
        tile_S = 1
        while tile_S * 2 <= raw_S:
            tile_S *= 2
        tiles["S"] = tile_S
        tiles["B"] = 1
        return tiles

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        """Schedule RoPE forward with auto-computed tile sizes.

        q must be 4D (B, S, NH, HD).
        """
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
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
    def load(self, page_ptr, tile_B, tile_S, tile_NH, q, cos, sin, work_mbar):
        """Load q/cos/sin tile from global to shared memory.

        Having work_mbar in the signature tells the framework this is an
        async load. The framework wraps non-TMA loads in elect_one(), so
        only one DMA-warp thread executes this body.
        """
        pos_start = tile_S * self.tile_size_S
        head_start = tile_NH * self.tile_size_NH

        # Compute total bytes (handle partial S tiles at boundary)
        q_per_pos_bytes = Int32(self.q_row_elems * self.elem_bytes)
        cs_per_pos_bytes = Int32(2 * self.D2 * self.elem_bytes)
        actual_rows = Int32(self.tile_size_S)
        remaining = Int32(self.S) - pos_start
        if remaining < Int32(self.tile_size_S):
            actual_rows = remaining
        total_bytes = actual_rows * (q_per_pos_bytes + cs_per_pos_bytes)

        mbar_ptr = cute.make_ptr(cutlass.Int64, work_mbar, cute.AddressSpace.smem)
        mbarrier_arrive_expect_tx(work_mbar, total_bytes)

        # Copy atoms: q rows (tile_size_NH*HD) and cos/sin rows (D2)
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

        for local_pos in range(self.tile_size_S):
            pos = pos_start + local_pos
            if pos < self.S:
                sl = pos % self.SL

                # q: tile_size_NH × HD contiguous elements per position
                q_offset = (tile_B * (self.S * self.NH * self.HD)
                            + pos * (self.NH * self.HD) + head_start * self.HD)
                g_q = cute.make_tensor(
                    q.iterator + q_offset,
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

                # cos: D2 elements
                g_cos = cute.make_tensor(
                    cos.iterator + sl * self.D2,
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
                    sin.iterator + sl * self.D2,
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
    def compute(self, page_ptr, tile_B, tile_S, tile_NH):
        """Apply RoPE rotation for tile_size_S positions × tile_size_NH heads."""
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
        thr_count = min(self.D2, 32)
        thr_layout = cute.make_layout(thr_count)

        for local_pos in range(self.tile_size_S):
            if lane_idx < Int32(thr_count):
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

                for local_h in range(warp_idx, self.tile_size_NH, num_warps):
                    q_base = (local_pos * self.tile_size_NH + local_h) * self.HD

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
    def store(self, page_ptr, tile_B, tile_S, tile_NH, q, cos, sin):
        """Store modified q from shared to global memory."""
        s2g = cute.make_copy_atom(
            CopyBulkS2GOp(),
            self.q_dtype,
            num_bits_per_copy=self.q_nbits_per_row,
        )
        pos_start = tile_S * self.tile_size_S
        head_start = tile_NH * self.tile_size_NH

        for local_pos in range(self.tile_size_S):
            pos = pos_start + local_pos
            if pos < self.S:
                s_tile = cute.make_tensor(
                    cute.make_ptr(
                        self.q_dtype,
                        page_ptr + Int32(local_pos * self.q_row_elems * self.elem_bytes),
                        cute.AddressSpace.smem,
                    ),
                    cute.make_layout((self.q_row_elems,)),
                )
                q_offset = (tile_B * (self.S * self.NH * self.HD)
                            + pos * (self.NH * self.HD) + head_start * self.HD)
                g_tile = cute.make_tensor(
                    q.iterator + q_offset,
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
        "q": (None, ("B", "S", "NH", "HD")),
        "cos": (None, ("SL", "D2")),
        "sin": (None, ("SL", "D2")),
    }
    writes = {"q": (None, ("B", "S", "NH", "HD"))}
    tile = ("B", "S", "NH")

    def __init__(self, **config):
        super().__init__(**config)
        self.page_size = getattr(self, "page_size", DEFAULT_PAGE_SIZE)

        if self.q_dtype == cutlass.Float32:
            self.elem_bytes = 4
        elif self.q_dtype in (cutlass.Float16, cutlass.BFloat16):
            self.elem_bytes = 2
        else:
            self.elem_bytes = 4

        self.q_row_elems = self.tile_size_NH * self.HD
        self.q_tile_bytes = self.tile_size_S * self.q_row_elems * self.elem_bytes
        self.cs_tile_bytes = self.tile_size_S * self.D2 * self.elem_bytes
        total_smem = self.q_tile_bytes + 2 * self.cs_tile_bytes

        assert self.D2 >= 16, f"RopeBwdOp requires D2 >= 16, got D2={self.D2} (HD={self.HD})"
        assert self.NH % self.tile_size_NH == 0, f"RopeBwdOp: tile_size_NH={self.tile_size_NH} must divide NH={self.NH}"
        assert total_smem <= self.page_size, (
            f"RopeBwdOp: tile smem ({total_smem}B) exceeds page_size ({self.page_size}B). "
            f"Reduce tile_size_S or tile_size_NH."
        )

        self.q_nbits_per_row = self.q_row_elems * self.elem_bytes * 8
        self.cs_nbits_per_row = self.D2 * self.elem_bytes * 8

    # Reuse RopeOp's load and store — zero duplication
    load = RopeOp.load
    store = RopeOp.store

    @classmethod
    def schedule(cls, tile_sizes=None, page_size=DEFAULT_PAGE_SIZE, **tensors):
        """Schedule RoPE backward with auto-computed tile sizes.

        q must be 4D (B, S, NH, HD).
        """
        tile_sizes = dict(tile_sizes or {})
        tile_sizes.setdefault("B", 1)
        auto = RopeOp._auto_tiles(page_size, **tensors)
        for k, v in auto.items():
            tile_sizes.setdefault(k, v)
        ops = [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]
        ops[0].static_dims["page_size"] = page_size
        return ops

    kernel_config = RopeOp.kernel_config

    @cute.jit
    def compute(self, page_ptr, tile_B, tile_S, tile_NH):
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
        thr_count = min(self.D2, 32)
        thr_layout = cute.make_layout(thr_count)

        for local_pos in range(self.tile_size_S):
            if lane_idx < Int32(thr_count):
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

                for local_h in range(warp_idx, self.tile_size_NH, num_warps):
                    q_base = (local_pos * self.tile_size_NH + local_h) * self.HD

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
