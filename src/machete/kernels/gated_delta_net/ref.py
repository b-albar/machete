# Copyright (c) 2025, Machete Authors
"""Reference implementations for Gated Delta Net testing.

Wraps fla's internal stages so each machete op can be tested individually
against the Triton reference, and provides a pure-PyTorch naive implementation
for gradient checking.
"""

import torch
import torch.nn.functional as F


# =============================================================================
# Pure PyTorch naive implementation (for gradient checking)
# =============================================================================


def gated_delta_rule_naive(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    g: torch.Tensor,  # [B, T, H]  (log-space gate, always <= 0)
    beta: torch.Tensor,  # [B, T, H]
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,  # [B, H, K, V]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token recurrence (O(T*K*V) per head). For validation only."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    device, dtype = q.device, q.dtype
    h = torch.zeros(B, H, K, V, device=device, dtype=torch.float32)
    if initial_state is not None:
        h = initial_state.float()

    o = torch.zeros(B, T, H, V, device=device, dtype=dtype)

    for t in range(T):
        q_t = q[:, t]  # [B, H, K]
        k_t = k[:, t]  # [B, H, K]
        v_t = v[:, t]  # [B, H, V]
        g_t = g[:, t]  # [B, H]
        b_t = beta[:, t]  # [B, H]

        # Decay state
        h = h * torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)  # [B, H, K, V]

        # Delta rule: v_corrected = beta * (v - h^T @ k)
        # h^T @ k: [B, H, K, V]^T @ [B, H, K, 1] = [B, H, V]
        retrieved = torch.einsum("bhkv,bhk->bhv", h, k_t.float())
        delta = b_t.unsqueeze(-1) * (v_t.float() - retrieved)  # [B, H, V]

        # State update: h += k outer delta
        h = h + torch.einsum("bhk,bhv->bhkv", k_t.float(), delta)

        # Output
        o[:, t] = torch.einsum("bhk,bhkv->bhv", q_t.float() * scale, h).to(dtype)

    return o, h


# =============================================================================
# fla wrapper: individual stages for per-op testing
# =============================================================================


def fla_prep_stage(
    k: torch.Tensor,  # [B, T, H, K]
    v: torch.Tensor,  # [B, T, H, V]
    g: torch.Tensor,  # [B, T, H]
    beta: torch.Tensor,  # [B, T, H]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fla's preprocessing stages 1-3: cumsum + kkt + solve + w/u.

    Returns:
        g_cumsum: [B, T, H] (float32)
        A: [B, T, H, 64] (solved WY matrix)
        w: [B, T, H, K]
        u: [B, T, H, V]
    """
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from fla.ops.utils import solve_tril
    from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd

    g_cumsum = chunk_local_cumsum(g, chunk_size=64)
    A = chunk_scaled_dot_kkt_fwd(k=k, g=g_cumsum, beta=beta, output_dtype=torch.float32)
    A = solve_tril(A=A, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g=g_cumsum)

    return g_cumsum, A, w, u


def fla_state_recurrence(
    k: torch.Tensor,  # [B, T, H, K]
    w: torch.Tensor,  # [B, T, H, K]
    u: torch.Tensor,  # [B, T, H, V]
    g_cumsum: torch.Tensor,  # [B, T, H]
    initial_state: torch.Tensor | None = None,  # [B, H, K, V]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run fla's state recurrence (stage 4).

    Returns:
        h: [B, NT, H, K, V] (float32, inter-chunk hidden states)
        v_new: [B, T, H, V]
        final_state: [B, H, K, V] or None
    """
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g_cumsum,
        initial_state=initial_state,
        output_final_state=True,
    )
    return h, v_new, final_state


def fla_output(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, H, K]
    v_new: torch.Tensor,  # [B, T, H, V]
    h: torch.Tensor,  # [B, NT, H, K, V]
    g_cumsum: torch.Tensor,  # [B, T, H]
    scale: float | None = None,
) -> torch.Tensor:
    """Run fla's output computation (stage 5).

    Returns:
        o: [B, T, H, V]
    """
    from fla.ops.common.chunk_o import chunk_fwd_o

    K = q.shape[-1]
    if scale is None:
        scale = K ** -0.5
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g_cumsum, scale=scale)
    return o


def fla_full_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run full fla forward (all 5 stages).

    Returns:
        o: [B, T, H, V]
        final_state: [B, H, K, V] or None
    """
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    o, final_state = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale, initial_state=initial_state,
        output_final_state=(initial_state is not None),
    )
    return o, final_state


# =============================================================================
# fla backward stages (for per-op backward testing)
# =============================================================================


def fla_bwd_dv_local(
    q: torch.Tensor,
    k: torch.Tensor,
    g_cumsum: torch.Tensor,
    do: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Run fla's local dv backward (stage 3 of backward)."""
    from fla.ops.common.chunk_o import chunk_bwd_dv_local

    K = q.shape[-1]
    if scale is None:
        scale = K ** -0.5
    return chunk_bwd_dv_local(q=q, k=k, g=g_cumsum, do=do, scale=scale)


def fla_bwd_state_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    g_cumsum: torch.Tensor,
    h0: torch.Tensor | None,
    dht: torch.Tensor | None,
    do: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Run fla's backward state recurrence (stage 4 of backward)."""
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu

    K = q.shape[-1]
    if scale is None:
        scale = K ** -0.5
    return chunk_gated_delta_rule_bwd_dhu(
        q=q, k=k, w=w, g=g_cumsum, h0=h0, dht=dht, do=do, dv=dv, scale=scale,
    )
