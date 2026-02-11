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

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from machete.megakernel.ops import Op


class RopeOp(Op):
    """RoPE operation for the megakernel framework.

    Applies rotary position embedding in-place to a query tensor.
    Zero-page op: all data accessed directly from global memory via
    typed CuTe tensor views.

    Parallelism modes based on half-dimension D2 = D/2:
    - D2 >= 32: Warp-parallel (each warp processes one head, vectorized loads/stores)
    - D2 < 32: Scalar fallback (lane-strided access)

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
        s = tile_M % self.S

        if cutlass.const_expr(self.D2 >= 32):
            # === Vectorized path: each warp processes one head ===
            warp_idx = cute.arch.warp_idx()
            lane_idx = cute.arch.lane_idx()
            num_warps = self.threads_per_row // 32
            thr_layout = cute.make_layout(32)

            # Load cos/sin for this position (shared across all heads)
            cos_row = cute.make_tensor(
                cos.iterator + s * self.D2,
                cute.make_layout(self.D2),
            )
            sin_row = cute.make_tensor(
                sin.iterator + s * self.D2,
                cute.make_layout(self.D2),
            )
            cos_part = cute.local_partition(cos_row, thr_layout, lane_idx)
            sin_part = cute.local_partition(sin_row, thr_layout, lane_idx)
            cos_reg = cute.make_fragment_like(cos_part)
            sin_reg = cute.make_fragment_like(sin_part)
            cute.autovec_copy(cos_part, cos_reg)
            cute.autovec_copy(sin_part, sin_reg)

            for h in range(warp_idx, self.H, num_warps):
                q_base = tile_M * (self.H * self.D) + h * self.D

                # q0 = q[tile_M, h, :D2], q1 = q[tile_M, h, D2:]
                q0_row = cute.make_tensor(
                    q.iterator + q_base,
                    cute.make_layout(self.D2),
                )
                q1_row = cute.make_tensor(
                    q.iterator + q_base + self.D2,
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

                # Store back (vectorized)
                cute.autovec_copy(out0_reg, q0_part)
                cute.autovec_copy(out1_reg, q1_part)

        else:
            # === Scalar fallback for small D2 (< 32) ===
            q = cute.make_tensor(q.iterator, cute.make_layout(self.M * self.H * self.D))
            cos = cute.make_tensor(cos.iterator, cute.make_layout(self.S * self.D2))
            sin = cute.make_tensor(sin.iterator, cute.make_layout(self.S * self.D2))

            tidx = cute.arch.thread_idx()[0]
            total_work = self.H * self.D2
            for work_idx in range(tidx, total_work, self.threads_per_row):
                h = work_idx // self.D2
                i = work_idx % self.D2

                cs_idx = s * self.D2 + i
                q0_idx = tile_M * self.H * self.D + h * self.D + i
                q1_idx = q0_idx + self.D2

                c = cos[cs_idx].to(Float32)
                sn = sin[cs_idx].to(Float32)
                q0 = q[q0_idx].to(Float32)
                q1 = q[q1_idx].to(Float32)

                q[q0_idx] = (q0 * c - q1 * sn).to(self.q_dtype)
                q[q1_idx] = (q1 * c + q0 * sn).to(self.q_dtype)

    # --- Backward (Compute Phase) ---

    @cute.jit
    def backward_compute(self, page_ptr, tile_M, q, cos, sin):
        """Inverse RoPE rotation (transpose of forward rotation matrix)."""
        s = tile_M % self.S

        if cutlass.const_expr(self.D2 >= 32):
            # === Vectorized path: each warp processes one head ===
            warp_idx = cute.arch.warp_idx()
            lane_idx = cute.arch.lane_idx()
            num_warps = self.threads_per_row // 32
            thr_layout = cute.make_layout(32)

            # Load cos/sin for this position (shared across all heads)
            cos_row = cute.make_tensor(
                cos.iterator + s * self.D2,
                cute.make_layout(self.D2),
            )
            sin_row = cute.make_tensor(
                sin.iterator + s * self.D2,
                cute.make_layout(self.D2),
            )
            cos_part = cute.local_partition(cos_row, thr_layout, lane_idx)
            sin_part = cute.local_partition(sin_row, thr_layout, lane_idx)
            cos_reg = cute.make_fragment_like(cos_part)
            sin_reg = cute.make_fragment_like(sin_part)
            cute.autovec_copy(cos_part, cos_reg)
            cute.autovec_copy(sin_part, sin_reg)

            for h in range(warp_idx, self.H, num_warps):
                q_base = tile_M * (self.H * self.D) + h * self.D

                q0_row = cute.make_tensor(
                    q.iterator + q_base,
                    cute.make_layout(self.D2),
                )
                q1_row = cute.make_tensor(
                    q.iterator + q_base + self.D2,
                    cute.make_layout(self.D2),
                )

                q0_part = cute.local_partition(q0_row, thr_layout, lane_idx)
                q1_part = cute.local_partition(q1_row, thr_layout, lane_idx)
                q0_reg = cute.make_fragment_like(q0_part)
                q1_reg = cute.make_fragment_like(q1_part)
                cute.autovec_copy(q0_part, q0_reg)
                cute.autovec_copy(q1_part, q1_reg)

                # Compute inverse rotation in fp32
                out0_reg = cute.make_fragment_like(q0_reg)
                out1_reg = cute.make_fragment_like(q1_reg)
                for i in range(cute.size(q0_reg)):
                    c = cos_reg[i].to(Float32)
                    sn = sin_reg[i].to(Float32)
                    v0 = q0_reg[i].to(Float32)
                    v1 = q1_reg[i].to(Float32)
                    out0_reg[i] = (v0 * c + v1 * sn).to(self.q_dtype)
                    out1_reg[i] = (v1 * c - v0 * sn).to(self.q_dtype)

                # Store back (vectorized)
                cute.autovec_copy(out0_reg, q0_part)
                cute.autovec_copy(out1_reg, q1_part)

        else:
            # === Scalar fallback for small D2 (< 32) ===
            q = cute.make_tensor(q.iterator, cute.make_layout(self.M * self.H * self.D))
            cos = cute.make_tensor(cos.iterator, cute.make_layout(self.S * self.D2))
            sin = cute.make_tensor(sin.iterator, cute.make_layout(self.S * self.D2))

            tidx = cute.arch.thread_idx()[0]
            total_work = self.H * self.D2
            for work_idx in range(tidx, total_work, self.threads_per_row):
                h = work_idx // self.D2
                i = work_idx % self.D2

                cs_idx = s * self.D2 + i
                q0_idx = tile_M * self.H * self.D + h * self.D + i
                q1_idx = q0_idx + self.D2

                c = cos[cs_idx].to(Float32)
                sn = sin[cs_idx].to(Float32)
                q0 = q[q0_idx].to(Float32)
                q1 = q[q1_idx].to(Float32)

                q[q0_idx] = (q0 * c + q1 * sn).to(self.q_dtype)
                q[q1_idx] = (q1 * c - q0 * sn).to(self.q_dtype)


__all__ = ["RopeOp"]
