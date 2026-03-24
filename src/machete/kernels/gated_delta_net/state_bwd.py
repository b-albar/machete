# Copyright (c) 2025, Machete Authors
"""Gated Delta Net backward state recurrence.

Reverse-order state recurrence for computing dh, dv2, and dw (Stage 2 of backward).
Processes chunks from NT-1 down to 0, maintaining persistent gradient state b_dh.

Matches fla's chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64 for dh/dv2,
and fla's chunk_bwd_kernel_dqkwg for dw (fused here to avoid a separate pass).
"""

import torch


BT = 64


def run_bwd_state_recurrence(
    q: torch.Tensor,          # [B, T, H, K]
    k: torch.Tensor,          # [B, T, H, K]
    w: torch.Tensor,          # [B, T, H, K]
    g_cumsum: torch.Tensor,   # [B, T, H] (fp32)
    h0: torch.Tensor | None,  # [B, H, K, V] (initial state, optional)
    dht: torch.Tensor | None, # [B, H, K, V] (final state gradient, optional)
    do: torch.Tensor,         # [B, T, H, V] (output gradient)
    dv_local: torch.Tensor,   # [B, T, H, V] (local dv from Stage 1)
    h: torch.Tensor,          # [B, NT, H, K, V] (forward hidden states)
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Backward state recurrence (reverse time order).

    For each chunk i_t = NT-1 down to 0:
        1. Store dh[i_t] = b_dh  (state gradient BEFORE this chunk's update)
        2. b_dv = k @ b_dh       (propagate state gradient to values)
        3. b_dv *= exp(g_last - g[t])  (gating)
        4. b_dv += dv_local      (add local attention contribution)
        5. Store dv2[i_t] = b_dv
        6. dw[i_t] = -b_dv @ h[i_t]^T  (weight gradient from v_new = u - w@h)
        7. b_dh *= exp(g_last)   (decay state gradient)
        8. b_dh += (q * exp(g))^T @ do * scale  (output contribution)
        9. b_dh -= w^T @ b_dv    (state update contribution)

    Returns:
        dh: [B, NT, H, K, V] state gradients (k.dtype)
        dh0: [B, H, K, V] initial state gradient (fp32, or None)
        dv2: [B, T, H, V] accumulated value gradient (k.dtype)
        dw: [B, T, H, K] weight gradient (k.dtype)
    """
    B, T, H, K = q.shape
    V = do.shape[-1]
    NT = (T + BT - 1) // BT
    dtype = q.dtype
    device = q.device

    # Persistent state gradient b_dh[B, H, K, V] in fp32
    b_dh = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)
    if dht is not None:
        b_dh = dht.float().clone()

    dh = torch.empty(B, NT, H, K, V, device=device, dtype=dtype)
    dv2 = torch.empty(B, T, H, V, device=device, dtype=dtype)
    dw = torch.empty(B, T, H, K, device=device, dtype=dtype)

    for chunk_idx in range(NT - 1, -1, -1):
        t_start = chunk_idx * BT
        t_end = min((chunk_idx + 1) * BT, T)
        last_idx = t_end - 1

        # 1. Store dh BEFORE processing (fla convention)
        dh[:, chunk_idx] = b_dh.to(dtype)

        # 2. Chunk slices → [B, H, cl, K/V]
        k_c = k[:, t_start:t_end].permute(0, 2, 1, 3)    # [B, H, cl, K]
        q_c = q[:, t_start:t_end].permute(0, 2, 1, 3)    # [B, H, cl, K]
        w_c = w[:, t_start:t_end].permute(0, 2, 1, 3)    # [B, H, cl, K]
        do_c = do[:, t_start:t_end].permute(0, 2, 1, 3)   # [B, H, cl, V]
        dv_c = dv_local[:, t_start:t_end].permute(0, 2, 1, 3)  # [B, H, cl, V]
        g_c = g_cumsum[:, t_start:t_end].permute(0, 2, 1)  # [B, H, cl]

        # 3. b_dv = k @ b_dh (fp16×fp16→fp32)
        b_dh_rounded = b_dh.to(dtype).float()
        b_dv = torch.matmul(k_c.float(), b_dh_rounded)  # [B, H, cl, V]

        # 4. Gating: b_dv *= exp(g_last - g[t])
        g_last = g_cumsum[:, last_idx:last_idx + 1, :]  # [B, 1, H]
        gate = torch.exp(g_last - g_cumsum[:, t_start:t_end])  # [B, cl, H]
        b_dv = b_dv * gate.permute(0, 2, 1).unsqueeze(-1)

        # 5. Add local contribution
        b_dv = b_dv + dv_c.float()

        # 6. Store dv2
        dv2[:, t_start:t_end] = b_dv.permute(0, 2, 1, 3).to(dtype)

        # 7. dw = -dv2 @ h^T  (from forward: v_new = u - w@h, so dw = -dv_new @ h^T)
        #    Match fla's dtype casting: both operands cast to input dtype before matmul
        h_c = h[:, chunk_idx]  # [B, H, K, V]
        dw_c = -torch.matmul(
            b_dv.to(dtype).float(),              # [B, H, cl, V]
            h_c.to(dtype).float().transpose(-2, -1),  # [B, H, V, K]
        )  # [B, H, cl, K]
        dw[:, t_start:t_end] = dw_c.permute(0, 2, 1, 3).to(dtype)

        # 8. Decay: b_dh *= exp(g_last)
        decay = torch.exp(g_cumsum[:, last_idx, :])  # [B, H]
        b_dh = b_dh * decay.unsqueeze(-1).unsqueeze(-1)

        # 9. Gate q: q_gated = q * exp(g)
        q_gated = q_c.float() * torch.exp(g_c).unsqueeze(-1)  # [B, H, cl, K]

        # 10. b_dh += q_gated^T @ do * scale - w^T @ b_dv
        b_dh = b_dh + torch.matmul(
            q_gated.transpose(-2, -1), do_c.float()
        ) * scale
        b_dh = b_dh - torch.matmul(
            w_c.transpose(-2, -1).float(), b_dv.to(dtype).float()
        )

    dh0 = b_dh if h0 is not None else None
    return dh, dh0, dv2, dw
