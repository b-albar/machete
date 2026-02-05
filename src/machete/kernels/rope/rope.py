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
        q:   (M, H, D)   — query tensor, bf16/fp16/fp32, modified in-place
        cos: (S, D2)     — cosine table, same dtype as q
        sin: (S, D2)     — sine table, same dtype as q

    Tiling:
        tile_m indexes M (one tile per position, M = batch * seqlen).
    """

    # dtype=None means infer from tensor at schedule time (supports bf16/fp16/fp32)
    reads = {
        "q": (None, "M, H, D"),
        "cos": (None, "S, D2"),
        "sin": (None, "S, D2"),
    }
    writes = {"q": (None, "M, H, D")}
    tile = ("M",)

    # --- Forward ---

    @staticmethod
    def forward(
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

            # Load and convert to fp32 for computation
            c = cos[cs_idx].to(Float32)
            sn = sin[cs_idx].to(Float32)
            q0 = q[q0_idx].to(Float32)
            q1 = q[q1_idx].to(Float32)

            # Forward rotation: [[cos, -sin], [sin, cos]]
            # Store back in input dtype
            q[q0_idx] = (q0 * c - q1 * sn).to(q_dtype)
            q[q1_idx] = (q1 * c + q0 * sn).to(q_dtype)

    # --- Backward ---

    @staticmethod
    def backward(
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

            # Load and convert to fp32 for computation
            c = cos[cs_idx].to(Float32)
            sn = sin[cs_idx].to(Float32)
            q0 = q[q0_idx].to(Float32)
            q1 = q[q1_idx].to(Float32)

            # Inverse rotation: [[cos, sin], [-sin, cos]]
            # Store back in input dtype
            q[q0_idx] = (q0 * c + q1 * sn).to(q_dtype)
            q[q1_idx] = (q1 * c - q0 * sn).to(q_dtype)


__all__ = ["RopeOp"]
