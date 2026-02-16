# Copyright (c) 2025, Machete Authors
"""Reference Flash Attention implementations for testing and benchmarking.

Provides a PyTorch reference implementation of scaled dot-product attention
with optional causal masking for correctness verification against the
megakernel FlashAttentionOp.
"""

import torch


def flash_attention_pytorch(q, k, v, causal=False, scale=None):
    """Pure PyTorch scaled dot-product attention reference.

    Uses bmm + softmax (not memory-efficient, but numerically stable
    for reference comparison).

    Args:
        q: (BH, M, D) float32 — query
        k: (BH, N, D) float32 — key
        v: (BH, N, D) float32 — value
        causal: if True, apply lower-left aligned causal mask
        scale: attention scale factor (default: 1/sqrt(D))

    Returns:
        o: (BH, M, D) float32 — attention output
    """
    scale = scale or (q.shape[-1] ** -0.5)
    scores = torch.bmm(q.float(), k.float().transpose(-2, -1)) * scale

    if causal:
        M, N = q.shape[1], k.shape[1]
        # Lower-left aligned: row i attends to cols 0..(i + N - M)
        row_idx = torch.arange(M, device=q.device).unsqueeze(1)
        col_idx = torch.arange(N, device=q.device).unsqueeze(0)
        mask = col_idx > row_idx + (N - M)
        scores.masked_fill_(mask.unsqueeze(0), float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    o = torch.bmm(attn, v.float())
    return o.to(q.dtype)


__all__ = ["flash_attention_pytorch"]
