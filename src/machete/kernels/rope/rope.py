# Copyright (c) 2025, Machete Authors
"""
RoPE (Rotary Position Embedding) Op for the Megakernel.

Applies rotary position embedding in-place to a query tensor using the
standard load/compute/store decomposition. Both forward and backward
passes are implemented.

Forward rotation:
    out[..., :half_d] = q[..., :half_d] * cos - q[..., half_d:] * sin
    out[..., half_d:] = q[..., half_d:] * cos + q[..., :half_d] * sin

Backward (inverse rotation — transpose of the rotation matrix):
    out[..., :half_d] = q[..., :half_d] * cos + q[..., half_d:] * sin
    out[..., half_d:] = q[..., half_d:] * cos - q[..., :half_d] * sin

Usage:
    from machete.kernels.rope import RopeOp
    from machete.megakernel import Megakernel

    q_flat = q.view(b * s, h, d).contiguous()
    ops = [RopeOp.schedule(q=q_flat, cos=cos, sin=sin)]
    kernel = Megakernel(ops)
    kernel.run()
"""

from cutlass import Int32, Int64, Float32

from machete.megakernel.ops import Op


class RopeOp(Op):
    """RoPE operation for the megakernel framework.

    Applies rotary position embedding in-place to a query tensor.
    Zero-page op: all data accessed directly from global memory via
    typed CuTe tensor views constructed from op_config_ptr.

    Tensor declarations:
        q:   (M, H, D)   — query tensor, float32, modified in-place
        cos: (S, D2)     — cosine table, float32
        sin: (S, D2)     — sine table, float32

    Tiling:
        tile_m indexes M (one tile per position, M = batch * seqlen).
    """

    reads = {
        "q": (Float32, "M, H, D"),
        "cos": (Float32, "S, D2"),
        "sin": (Float32, "S, D2"),
    }
    writes = {"q": (Float32, "M, H, D")}
    tile = ("M",)

    # --- Forward ---

    @staticmethod
    def compute_forward(
        smem_base: Int32, config_ptr: Int32, page_ids: tuple,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Apply RoPE rotation for one position (tile_m) across all heads."""
        s = tile_m % S
        total_work = H * D2
        for work_idx in range(tidx, total_work, num_threads):
            h = work_idx // D2
            i = work_idx % D2

            cs_idx = s * D2 + i
            q0_idx = tile_m * H * D + h * D + i
            q1_idx = q0_idx + D2

            c = cos[cs_idx]
            sn = sin[cs_idx]
            q0 = q[q0_idx]
            q1 = q[q1_idx]

            # Forward rotation: [[cos, -sin], [sin, cos]]
            q[q0_idx] = q0 * c - q1 * sn
            q[q1_idx] = q1 * c + q0 * sn

    # --- Backward ---

    @staticmethod
    def compute_backward(
        smem_base: Int32, config_ptr: Int32, page_ids: tuple,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Inverse RoPE rotation (transpose of forward rotation matrix)."""
        s = tile_m % S
        total_work = H * D2
        for work_idx in range(tidx, total_work, num_threads):
            h = work_idx // D2
            i = work_idx % D2

            cs_idx = s * D2 + i
            q0_idx = tile_m * H * D + h * D + i
            q1_idx = q0_idx + D2

            c = cos[cs_idx]
            sn = sin[cs_idx]
            q0 = q[q0_idx]
            q1 = q[q1_idx]

            # Inverse rotation: [[cos, sin], [-sin, cos]]
            q[q0_idx] = q0 * c + q1 * sn
            q[q1_idx] = q1 * c - q0 * sn


__all__ = ["RopeOp"]
