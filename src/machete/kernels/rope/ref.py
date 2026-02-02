# Copyright (c) 2025, Machete Authors
"""Reference RoPE implementations for testing and benchmarking.

Provides PyTorch and Triton (Unsloth-style) reference implementations of
Rotary Position Embedding (RoPE) for correctness verification and
performance comparison against the megakernel RopeOp.

All functions expect tensors in (B, S, H, D) layout.
"""

import torch

# =============================================================================
# PyTorch Reference
# =============================================================================


def rope_pytorch(q, cos, sin):
    """Pure PyTorch RoPE forward reference.

    Args:
        q: (B, S, H, D) float32
        cos: (S, D//2) float32
        sin: (S, D//2) float32

    Returns:
        (B, S, H, D) float32 — new tensor with RoPE applied
    """
    b, s, h, d = q.shape
    half_d = d // 2
    q0 = q[..., :half_d]
    q1 = q[..., half_d:]
    cos_exp = cos[:s].view(1, s, 1, half_d)
    sin_exp = sin[:s].view(1, s, 1, half_d)
    r0 = q0 * cos_exp - q1 * sin_exp
    r1 = q1 * cos_exp + q0 * sin_exp
    return torch.cat([r0, r1], dim=-1)


def rope_pytorch_backward(q, cos, sin):
    """Pure PyTorch inverse RoPE (backward) reference.

    Applies transpose of the rotation matrix: [[cos, sin], [-sin, cos]].

    Args:
        q: (B, S, H, D) float32
        cos: (S, D//2) float32
        sin: (S, D//2) float32

    Returns:
        (B, S, H, D) float32 — new tensor with inverse RoPE applied
    """
    b, s, h, d = q.shape
    half_d = d // 2
    q0 = q[..., :half_d]
    q1 = q[..., half_d:]
    cos_exp = cos[:s].view(1, s, 1, half_d)
    sin_exp = sin[:s].view(1, s, 1, half_d)
    r0 = q0 * cos_exp + q1 * sin_exp
    r1 = q1 * cos_exp - q0 * sin_exp
    return torch.cat([r0, r1], dim=-1)


# =============================================================================
# Triton Reference (optional)
# =============================================================================


try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _rope_embedding_triton(
        q,
        q_stride,
        cos,
        cos_stride,
        sin,
        sin_stride,
        seqlen,
        head_dim: tl.constexpr,
        n_heads: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row_position = tl.program_id(0)
        group_head_position = tl.program_id(1)
        col_offsets = tl.arange(0, block_size)
        half_head_dim = head_dim // 2
        mask = col_offsets < half_head_dim

        s_idx = row_position % seqlen
        sin1 = tl.load(sin + s_idx * sin_stride + col_offsets, mask=mask, other=0)
        cos1 = tl.load(cos + s_idx * cos_stride + col_offsets, mask=mask, other=0)

        head_idx = group_head_position
        offs_q1 = row_position * q_stride + head_idx * head_dim + col_offsets
        offs_q2 = (
            row_position * q_stride
            + head_idx * head_dim
            + col_offsets
            + half_head_dim
        )

        q1 = tl.load(q + offs_q1, mask=mask, other=0).to(tl.float32)
        q2 = tl.load(q + offs_q2, mask=mask, other=0).to(tl.float32)

        tl.store(
            q + offs_q1, (q1 * cos1 - q2 * sin1).to(q.dtype.element_ty), mask=mask
        )
        tl.store(
            q + offs_q2, (q2 * cos1 + q1 * sin1).to(q.dtype.element_ty), mask=mask
        )

    def rope_triton(q, cos, sin):
        """Triton (Unsloth-style) in-place RoPE. Modifies q in-place.

        Args:
            q: (B, S, H, D) float32, CUDA. Modified in-place.
            cos: (S, D//2) float32, CUDA.
            sin: (S, D//2) float32, CUDA.

        Returns:
            q (modified in-place)
        """
        b, s, h, d = q.shape
        q_view = q.view(b * s, h * d)
        block_size = triton.next_power_of_2(d // 2)
        _rope_embedding_triton[(b * s, h)](
            q_view,
            q_view.stride(0),
            cos,
            cos.stride(0),
            sin,
            sin.stride(0),
            s,
            d,
            h,
            block_size,
        )
        return q


__all__ = [
    "rope_pytorch",
    "rope_pytorch_backward",
    "HAS_TRITON",
]

# Conditionally export Triton functions
if HAS_TRITON:
    __all__.append("rope_triton")
