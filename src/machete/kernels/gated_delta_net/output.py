# Copyright (c) 2025, Machete Authors
"""Gated Delta Net output computation (stage 5 of the chunked algorithm).

Computes the final output for each chunk by combining:
    - Inter-chunk contribution: Q @ h_states[chunk] (cross-chunk state)
    - Intra-chunk contribution: causal_mask(Q @ K^T) @ v_new (local attention)

Both terms are gated by exp(g_cumsum) and scaled by 1/sqrt(K).

Currently a PyTorch reference implementation matching fla's chunk_fwd_kernel_o.
Will be replaced with a CuTe DSL megakernel.
"""

import torch


BT = 64  # Chunk size (matching fla convention)


def run_output(
    q: torch.Tensor,        # [B, T, H, K]
    k: torch.Tensor,        # [B, T, H, K]
    v_new: torch.Tensor,    # [B, T, H, V]
    h: torch.Tensor,        # [B, NT, H, K, V]
    g_cumsum: torch.Tensor,  # [B, T, H] (fp32)
    scale: float | None = None,
) -> torch.Tensor:
    """Output computation matching fla's chunk_fwd_kernel_o.

    For each chunk i_t:
        1. K-loop: b_o += Q @ h[i_t]      (inter-chunk, [BT,K]@[K,V]->[BT,V])
                   b_A += Q @ K^T           (intra-chunk scores, [BT,K]@[K,BT]->[BT,BT])
        2. Gating: b_o *= exp(g[t]),  b_A *= exp(g[i] - g[j])
        3. Causal mask: b_A[i,j] = 0 for j > i
        4. Output: o = b_o * scale + (b_A @ v_new) * scale

    Args:
        q: Queries [B, T, H, K] (fp16/bf16)
        k: Keys [B, T, H, K] (fp16/bf16)
        v_new: New values from state recurrence [B, T, H, V] (fp16/bf16)
        h: Inter-chunk hidden states [B, NT, H, K, V] (same dtype as k)
        g_cumsum: Cumulative gating values [B, T, H] (fp32)
        scale: Attention scale (default: K^-0.5)

    Returns:
        o: Output [B, T, H, V] (same dtype as q)
    """
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    NT = (T + BT - 1) // BT
    dtype = q.dtype

    if scale is None:
        scale = K ** -0.5

    o = torch.empty(B, T, H, V, device=q.device, dtype=dtype)

    for chunk_idx in range(NT):
        t_start = chunk_idx * BT
        t_end = min((chunk_idx + 1) * BT, T)
        cl = t_end - t_start

        # Chunk slices (permute to [B, H, cl, K/V] for batched matmul)
        q_c = q[:, t_start:t_end].permute(0, 2, 1, 3).float()   # [B, H, cl, K]
        k_c = k[:, t_start:t_end].permute(0, 2, 1, 3).float()   # [B, H, cl, K]
        v_c = v_new[:, t_start:t_end].permute(0, 2, 1, 3)        # [B, H, cl, V]
        g_c = g_cumsum[:, t_start:t_end].permute(0, 2, 1)         # [B, H, cl]
        h_c = h[:, chunk_idx]                                      # [B, H, K, V]

        # 1. Inter-chunk: b_o = Q @ h  (simulate fp16×fp16→fp32)
        b_o = torch.matmul(q_c, h_c.float())  # [B, H, cl, V]

        # 2. Intra-chunk scores: b_A = Q @ K^T
        b_A = torch.matmul(q_c, k_c.transpose(-2, -1))  # [B, H, cl, cl]

        # 3. Apply gating
        g_exp = torch.exp(g_c)  # [B, H, cl]
        b_o = b_o * g_exp.unsqueeze(-1)                            # [B, H, cl, 1]
        b_A = b_A * torch.exp(g_c.unsqueeze(-1) - g_c.unsqueeze(-2))  # [B, H, cl, cl]

        # 4. Causal mask (lower triangular, matching fla: o_t[i] >= o_t[j])
        causal_mask = torch.ones(cl, cl, device=q.device, dtype=torch.bool).tril()
        b_A = b_A.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), 0.0)

        # 5. Final output: o = b_o * scale + (b_A @ v_new) * scale
        #    Match fla: b_A cast to input dtype before matmul with v
        intra = torch.matmul(b_A.to(dtype).float(), v_c.float())  # [B, H, cl, V]
        o_c = b_o * scale + intra * scale

        o[:, t_start:t_end] = o_c.permute(0, 2, 1, 3).to(dtype)

    return o
