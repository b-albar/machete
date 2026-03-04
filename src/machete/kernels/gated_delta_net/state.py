# Copyright (c) 2025, Machete Authors
"""Gated Delta Net state recurrence (stage 4 of the chunked algorithm).

Implements the sequential state recurrence:
    For each chunk:
        1. Store h to h_states (BEFORE processing)
        2. v_partial = w @ h            (K-blocked matmul)
        3. v_new = u - v_partial         (store BEFORE gating)
        4. v_gated = v_new * exp(g_last - g_cumsum[t])
        5. h *= exp(g_last)              (decay state)
        6. h += k^T @ v_gated           (K-blocked matmul)

Currently a PyTorch reference implementation matching fla's Triton kernel
(chunk_gated_delta_rule_fwd_kernel_h_blockdim64). Will be replaced with a
CuTe DSL megakernel with pipelined TMA/cpasync.
"""

import torch


BT = 64  # Chunk size (matching fla convention)


def run_state_recurrence(
    k: torch.Tensor,        # [B, T, H, K]
    w: torch.Tensor,        # [B, T, H, K]
    u: torch.Tensor,        # [B, T, H, V]
    g_cumsum: torch.Tensor,  # [B, T, H] (fp32)
    initial_state: torch.Tensor | None = None,  # [B, H, K, V]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked state recurrence matching fla's chunk_gated_delta_rule_fwd_h.

    Processes chunks of BT=64 timesteps sequentially, maintaining persistent
    hidden state h[K,V] across chunks. Each V-strip (BV=64) is independent
    and processed via batched matmul.

    Algorithm per chunk (matching fla Triton kernel):
        1. Store h_states[chunk] = h  (state BEFORE this chunk's update)
        2. b_v = w @ h               (fp16 matmul with fp32 accumulation)
        3. b_v = u - b_v             (fp32 subtraction)
        4. v_new[chunk] = b_v        (stored in input dtype, BEFORE gating)
        5. b_v *= exp(g_last - g[t]) (per-timestep gating, fp32)
        6. b_v = b_v.to(dtype)       (cast to fp16, matching fla line 192)
        7. h *= exp(g_last)          (scalar decay of entire state)
        8. h += k^T @ b_v            (fp16 matmul with fp32 accumulation)

    Args:
        k: Keys [B, T, H, K] (fp16/bf16)
        w: Transformed keys from WY representation [B, T, H, K] (fp16/bf16)
        u: Transformed values from WY representation [B, T, H, V] (fp16/bf16)
        g_cumsum: Cumulative gating values [B, T, H] (fp32)
        initial_state: Optional initial hidden state [B, H, K, V] (fp32)

    Returns:
        h_states: Inter-chunk hidden states [B, NT, H, K, V] (same dtype as k)
        v_new: New values before gating [B, T, H, V] (same dtype as u)
    """
    B, T, H, K = k.shape
    V = u.shape[-1]
    NT = (T + BT - 1) // BT

    device = k.device
    dtype = k.dtype

    # Persistent state h[B, H, K, V] in fp32
    h = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)
    if initial_state is not None:
        h = initial_state.float().clone()

    # Outputs (matching fla's allocation)
    h_states = torch.empty(B, NT, H, K, V, device=device, dtype=dtype)
    v_new = torch.empty(B, T, H, V, device=device, dtype=dtype)

    for chunk_idx in range(NT):
        t_start = chunk_idx * BT
        t_end = min((chunk_idx + 1) * BT, T)
        last_idx = t_end - 1

        # 1. Store h BEFORE processing (fla convention: h_states[i] = state
        #    entering chunk i, not after processing it)
        h_states[:, chunk_idx] = h.to(dtype)

        # 2. Chunk slices
        w_c = w[:, t_start:t_end].permute(0, 2, 1, 3)  # [B, H, cl, K]
        u_c = u[:, t_start:t_end].permute(0, 2, 1, 3)  # [B, H, cl, V]
        k_c = k[:, t_start:t_end].permute(0, 2, 1, 3)  # [B, H, cl, K]
        g_c = g_cumsum[:, t_start:t_end]                 # [B, cl, H]

        # 3. b_v = w @ h  (simulate fp16 × fp16 → fp32 accumulation)
        #    Cast h to dtype first (fp16 rounding), then to float for matmul
        h_rounded = h.to(dtype).float()
        b_v = torch.matmul(w_c.float(), h_rounded)  # [B, H, cl, V]

        # 4. b_v = u - b_v  (fp32)
        b_v = u_c.float() - b_v

        # 5. Store v_new BEFORE gating (fla stores v_new at this point)
        v_new[:, t_start:t_end] = b_v.permute(0, 2, 1, 3).to(dtype)

        # 6. Apply gating: b_v *= exp(g_last - g[t]) per timestep
        g_last = g_cumsum[:, last_idx:last_idx + 1, :]  # [B, 1, H]
        gate = torch.exp(g_last - g_c)                   # [B, cl, H]
        b_v = b_v * gate.permute(0, 2, 1).unsqueeze(-1)  # [B, H, cl, 1] broadcast

        # 7. Cast b_v to input dtype (matching fla line 192: b_v = b_v.to(k.dtype))
        b_v = b_v.to(dtype)

        # 8. Decay state: h *= exp(g_last)
        decay = torch.exp(g_cumsum[:, last_idx, :])       # [B, H]
        h = h * decay.unsqueeze(-1).unsqueeze(-1)          # broadcast [B, H, 1, 1]

        # 9. State update: h += k^T @ b_v  (fp16 × fp16 → fp32)
        #    k_c: [B, H, cl, K] → transpose last two → [B, H, K, cl]
        h = h + torch.matmul(k_c.transpose(-2, -1).float(), b_v.float())

    return h_states, v_new
