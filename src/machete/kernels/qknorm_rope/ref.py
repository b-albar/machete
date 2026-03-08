# Copyright (c) 2025, Machete Authors
"""Reference implementations of fused per-head RMSNorm + RoPE.

All functions expect tensors in (B, S, H, D) layout.
"""

import torch


def qknorm_rope_pytorch(q, norm_weight, cos, sin, eps=1e-6):
    """Per-head RMSNorm then partial RoPE (PyTorch reference).

    Args:
        q: (B, S, H, D) float32
        norm_weight: (D,) float32 — RMSNorm weight, shared across heads
        cos: (S, D2) float32 — D2 = rotary_dim // 2
        sin: (S, D2) float32
        eps: RMSNorm epsilon

    Returns:
        (B, S, H, D) float32 — new tensor with RMSNorm + RoPE applied
    """
    b, s, h, d = q.shape
    d2 = cos.shape[1]

    # Per-head RMSNorm
    q_f32 = q.float()
    rms = (q_f32.pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
    q_normed = (q_f32 * rms * norm_weight.float()).to(q.dtype)

    # Partial RoPE
    q0 = q_normed[..., :d2]
    q1 = q_normed[..., d2:2 * d2]
    cos_exp = cos[:s].view(1, s, 1, d2)
    sin_exp = sin[:s].view(1, s, 1, d2)
    out = q_normed.clone()
    out[..., :d2] = q0 * cos_exp - q1 * sin_exp
    out[..., d2:2 * d2] = q1 * cos_exp + q0 * sin_exp
    return out


__all__ = ["qknorm_rope_pytorch"]
