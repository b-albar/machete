# Copyright (c) 2025, Machete Authors
"""Reference Flash Attention implementations for testing and benchmarking.

Provides a PyTorch reference implementation of scaled dot-product attention
with optional causal masking for correctness verification against the
megakernel FlashAttentionSm100Op / FlashAttentionSm120Op.
"""

import torch


def flash_attention_pytorch(q, k, v, causal=False, scale=None, kv_group_size=1):
    """Pure PyTorch scaled dot-product attention reference.

    Uses bmm + softmax (not memory-efficient, but numerically stable
    for reference comparison).

    Args:
        q: (BH_q, M, D) float32 — query
        k: (BH_kv, N, D) float32 — key (BH_kv = BH_q when kv_group_size=1)
        v: (BH_kv, N, D) float32 — value
        causal: if True, apply lower-left aligned causal mask
        scale: attention scale factor (default: 1/sqrt(D))
        kv_group_size: number of Q heads per KV head (1 = MHA, >1 = GQA)

    Returns:
        o: (BH_q, M, D) float32 — attention output
    """
    if kv_group_size > 1:
        k = k.repeat_interleave(kv_group_size, dim=0)
        v = v.repeat_interleave(kv_group_size, dim=0)
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


def flash_attention_backward_pytorch(q, k, v, o, dout, causal=False, scale=None,
                                     kv_group_size=1):
    """Pure PyTorch backward for scaled dot-product attention.

    Computes dQ, dK, dV given dO (gradient of loss w.r.t. output O).

    Args:
        q: (BH_q, M, D) — query
        k: (BH_kv, N, D) — key
        v: (BH_kv, N, D) — value
        o: (BH_q, M, D) — forward output (used for dPsum)
        dout: (BH_q, M, D) — gradient of loss w.r.t. O
        causal: if True, apply lower-left aligned causal mask
        scale: attention scale factor (default: 1/sqrt(D))
        kv_group_size: number of Q heads per KV head (1 = MHA, >1 = GQA)

    Returns:
        dq: (BH_q, M, D), dk: (BH_kv, N, D), dv: (BH_kv, N, D)
    """
    if kv_group_size > 1:
        k_exp = k.repeat_interleave(kv_group_size, dim=0)
        v_exp = v.repeat_interleave(kv_group_size, dim=0)
    else:
        k_exp, v_exp = k, v

    N = k.shape[1]
    scale = scale or (q.shape[-1] ** -0.5)
    q_f, k_f, v_f = q.float(), k_exp.float(), v_exp.float()
    dout_f = dout.float()

    # Recompute P = softmax(Q @ K^T * scale)
    scores = torch.bmm(q_f, k_f.transpose(-2, -1)) * scale
    if causal:
        M = q.shape[1]
        row_idx = torch.arange(M, device=q.device).unsqueeze(1)
        col_idx = torch.arange(N, device=q.device).unsqueeze(0)
        mask = col_idx > row_idx + (N - M)
        scores.masked_fill_(mask.unsqueeze(0), float('-inf'))
    P = torch.softmax(scores, dim=-1)  # (BH_q, M, N)

    # dV = P^T @ dO  — (BH_q, N, D)
    dv_exp = torch.bmm(P.transpose(-2, -1), dout_f)

    # dP = dO @ V^T
    dP = torch.bmm(dout_f, v_f.transpose(-2, -1))

    # dS = P * (dP - dPsum) where dPsum = rowsum(dO * O)
    dPsum = (dout_f * o.float()).sum(dim=-1, keepdim=True)  # (BH_q, M, 1)
    dS = P * (dP - dPsum)

    # dQ = dS @ K * scale, dK = dS^T @ Q * scale
    dq = torch.bmm(dS, k_f) * scale
    dk_exp = torch.bmm(dS.transpose(-2, -1), q_f) * scale

    # Reduce dk/dv from BH_q to BH_kv
    if kv_group_size > 1:
        D = k.shape[-1]
        dk = dk_exp.view(-1, kv_group_size, N, D).sum(1)
        dv = dv_exp.view(-1, kv_group_size, N, D).sum(1)
    else:
        dk, dv = dk_exp, dv_exp

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


__all__ = ["flash_attention_pytorch", "flash_attention_backward_pytorch"]
