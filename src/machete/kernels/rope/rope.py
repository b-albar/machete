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

import cutlass.cute as cute
from cutlass import Float32

from machete.megakernel.ops import Op


class RopeOp(Op):
    """RoPE operation for the megakernel framework.

    Applies rotary position embedding in-place to a query tensor.
    Zero-page op: all data accessed directly from global memory via
    typed CuTe tensor views.

    Tensor declarations:
        q:   (M, H, D)   — query tensor, bf16/fp16/fp32, modified in-place
        cos: (S, D2)     — cosine table, same dtype as q
        sin: (S, D2)     — sine table, same dtype as q

    Tiling:
        tile_M indexes M (one tile per position, M = batch * seqlen).
    """

    # dtype=None means infer from tensor at schedule time (supports bf16/fp16/fp32)
    reads = {
        "q": (None, ("M", "H", "D")),
        "cos": (None, ("S", "D2")),
        "sin": (None, ("S", "D2")),
    }
    writes = {"q": (None, ("M", "H", "D"))}
    tile = ("M",)

    # --- Forward (Compute Phase) ---

    @cute.jit
    def compute(self, page_ptr, tile_M, q, cos, sin):
        """Apply RoPE rotation for one position (tile_M) across all heads."""
        # Flatten N-D tensors for scalar indexing
        q = cute.make_tensor(q.iterator, cute.make_layout(self.M * self.H * self.D))
        cos = cute.make_tensor(cos.iterator, cute.make_layout(self.S * self.D2))
        sin = cute.make_tensor(sin.iterator, cute.make_layout(self.S * self.D2))

        tidx = cute.arch.thread_idx()[0]
        s = tile_M % self.S
        total_work = self.H * self.D2
        for work_idx in range(tidx, total_work, self.threads_per_row):
            h = work_idx // self.D2
            i = work_idx % self.D2

            cs_idx = s * self.D2 + i
            q0_idx = tile_M * self.H * self.D + h * self.D + i
            q1_idx = q0_idx + self.D2

            # Load and convert to fp32 for computation
            c = cos[cs_idx].to(Float32)
            sn = sin[cs_idx].to(Float32)
            q0 = q[q0_idx].to(Float32)
            q1 = q[q1_idx].to(Float32)

            # Forward rotation: [[cos, -sin], [sin, cos]]
            q[q0_idx] = (q0 * c - q1 * sn).to(self.q_dtype)
            q[q1_idx] = (q1 * c + q0 * sn).to(self.q_dtype)

    # --- Backward (Compute Phase) ---

    @cute.jit
    def backward_compute(self, page_ptr, tile_M, q, cos, sin):
        """Inverse RoPE rotation (transpose of forward rotation matrix)."""
        # Flatten N-D tensors for scalar indexing
        q = cute.make_tensor(q.iterator, cute.make_layout(self.M * self.H * self.D))
        cos = cute.make_tensor(cos.iterator, cute.make_layout(self.S * self.D2))
        sin = cute.make_tensor(sin.iterator, cute.make_layout(self.S * self.D2))

        tidx = cute.arch.thread_idx()[0]
        s = tile_M % self.S
        total_work = self.H * self.D2
        for work_idx in range(tidx, total_work, self.threads_per_row):
            h = work_idx // self.D2
            i = work_idx % self.D2

            cs_idx = s * self.D2 + i
            q0_idx = tile_M * self.H * self.D + h * self.D + i
            q1_idx = q0_idx + self.D2

            # Load and convert to fp32 for computation
            c = cos[cs_idx].to(Float32)
            sn = sin[cs_idx].to(Float32)
            q0 = q[q0_idx].to(Float32)
            q1 = q[q1_idx].to(Float32)

            # Inverse rotation: [[cos, sin], [-sin, cos]]
            q[q0_idx] = (q0 * c + q1 * sn).to(self.q_dtype)
            q[q1_idx] = (q1 * c - q0 * sn).to(self.q_dtype)


__all__ = ["RopeOp"]
