# Copyright (c) 2025, Machete Authors
"""Reference Flash Attention implementations for testing and benchmarking.

Provides a PyTorch reference implementation of scaled dot-product attention
with optional causal masking for correctness verification against the
megakernel FlashAttentionSm100Op / FlashAttentionSm120Op.
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


def flash_attention_backward_pytorch(q, k, v, o, dout, causal=False, scale=None):
    """Pure PyTorch backward for scaled dot-product attention.

    Computes dQ, dK, dV given dO (gradient of loss w.r.t. output O).

    Args:
        q: (BH, M, D) — query
        k: (BH, N, D) — key
        v: (BH, N, D) — value
        o: (BH, M, D) — forward output (used for dPsum)
        dout: (BH, M, D) — gradient of loss w.r.t. O
        causal: if True, apply lower-left aligned causal mask
        scale: attention scale factor (default: 1/sqrt(D))

    Returns:
        dq: (BH, M, D), dk: (BH, N, D), dv: (BH, N, D)
    """
    scale = scale or (q.shape[-1] ** -0.5)
    q_f, k_f, v_f = q.float(), k.float(), v.float()
    dout_f = dout.float()

    # Recompute P = softmax(Q @ K^T * scale)
    scores = torch.bmm(q_f, k_f.transpose(-2, -1)) * scale
    if causal:
        M, N = q.shape[1], k.shape[1]
        row_idx = torch.arange(M, device=q.device).unsqueeze(1)
        col_idx = torch.arange(N, device=q.device).unsqueeze(0)
        mask = col_idx > row_idx + (N - M)
        scores.masked_fill_(mask.unsqueeze(0), float('-inf'))
    P = torch.softmax(scores, dim=-1)  # (BH, M, N)

    # dV = P^T @ dO
    dv = torch.bmm(P.transpose(-2, -1), dout_f)

    # dP = dO @ V^T
    dP = torch.bmm(dout_f, v_f.transpose(-2, -1))

    # dS = P * (dP - dPsum) where dPsum = rowsum(dO * O)
    dPsum = (dout_f * o.float()).sum(dim=-1, keepdim=True)  # (BH, M, 1)
    dS = P * (dP - dPsum)

    # dQ = dS @ K * scale (scale already in dS via P, but dS = P*(dP-dPsum), need extra scale)
    # Actually: d(softmax(S*scale)) = P*(dP - dPsum), so dScore = P*(dP-dPsum)
    # dQ = dScore @ K, but Score = Q@K^T*scale, so dQ = dScore @ K * scale? No.
    # Chain rule: dL/dQ = dL/dScore @ dScore/dQ = dS @ K where dS already accounts for scale
    # Score = Q @ K^T * scale → dQ = dS @ K * scale, dK = dS^T @ Q * scale
    dq = torch.bmm(dS, k_f) * scale
    dk = torch.bmm(dS.transpose(-2, -1), q_f) * scale

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


__all__ = ["flash_attention_pytorch", "flash_attention_backward_pytorch"]
