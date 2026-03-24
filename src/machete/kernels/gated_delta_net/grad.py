# Copyright (c) 2025, Machete Authors
"""Gated Delta Net backward gradient computation stages.

Stage 1: dv_local — local causal attention backward
Stage 3: dqkwg — query/key/w/gate gradients
Stage 4: wy_bwd — backward through WY representation
Stage 5: reverse cumsum of dg

Currently wraps fla's implementations. Will be replaced with CuTe DSL kernels.
"""

import torch


BT = 64


def run_bwd_dv_local(
    q: torch.Tensor,        # [B, T, H, K]
    k: torch.Tensor,        # [B, T, H, K]
    g_cumsum: torch.Tensor,  # [B, T, H] (fp32)
    do: torch.Tensor,       # [B, T, H, V]
    scale: float,
) -> torch.Tensor:
    """Compute local dv from causal attention backward.

    For each chunk:
        A[i,j] = scale * k[i] @ q[j]^T * exp(g[j] - g[i]) for i <= j
        dv = A @ do

    This is the transpose of the forward attention pattern.

    Returns:
        dv: [B, T, H, V]
    """
    B, T, H, K = q.shape
    V = do.shape[-1]
    NT = (T + BT - 1) // BT
    dtype = q.dtype
    device = q.device

    dv = torch.empty(B, T, H, V, device=device, dtype=dtype)

    for chunk_idx in range(NT):
        t_start = chunk_idx * BT
        t_end = min((chunk_idx + 1) * BT, T)
        cl = t_end - t_start

        # [B, H, cl, K/V]
        q_c = q[:, t_start:t_end].permute(0, 2, 1, 3).float()
        k_c = k[:, t_start:t_end].permute(0, 2, 1, 3).float()
        do_c = do[:, t_start:t_end].permute(0, 2, 1, 3).float()
        g_c = g_cumsum[:, t_start:t_end].permute(0, 2, 1)  # [B, H, cl]

        # A = k @ q^T * scale: [B, H, cl, cl]
        b_A = torch.matmul(k_c, q_c.transpose(-2, -1)) * scale

        # Gating: A *= exp(g[j] - g[i])  (note: transposed from forward)
        b_A = b_A * torch.exp(g_c.unsqueeze(-2) - g_c.unsqueeze(-1))

        # Upper-triangular mask (i <= j): transpose of forward's lower-tri
        mask = torch.ones(cl, cl, device=device, dtype=torch.bool).triu()
        b_A = b_A.masked_fill(~mask, 0.0)

        # dv = A @ do
        dv_c = torch.matmul(b_A.to(dtype).float(), do_c)
        dv[:, t_start:t_end] = dv_c.permute(0, 2, 1, 3).to(dtype)

    return dv


def run_bwd_dqkwg(
    q: torch.Tensor,          # [B, T, H, K]
    k: torch.Tensor,          # [B, T, H, K]
    v_new: torch.Tensor,      # [B, T, H, V]
    w: torch.Tensor,          # [B, T, H, K]
    g_cumsum: torch.Tensor,   # [B, T, H] (fp32)
    h: torch.Tensor,          # [B, NT, H, K, V]
    dv: torch.Tensor,         # [B, T, H, V]
    do: torch.Tensor,         # [B, T, H, V]
    dh: torch.Tensor,         # [B, NT, H, K, V]
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute dq, dk, dw, dg from attention and gate backward.

    Wraps fla's chunk_bwd_dqkwg internally.

    Returns:
        dq: [B, T, H, K]
        dk: [B, T, H, K]
        dw: [B, T, H, K]
        dg: [B, T, H]
    """
    from fla.ops.common.chunk_o import chunk_bwd_dqkwg
    return chunk_bwd_dqkwg(
        q=q, k=k, v=v_new, w=w, g=g_cumsum,
        h=h, dv=dv, do=do, dh=dh, scale=scale,
    )


def run_bwd_wy(
    k: torch.Tensor,          # [B, T, H, K]
    v: torch.Tensor,          # [B, T, H, V]
    beta: torch.Tensor,       # [B, T, H]
    g_cumsum: torch.Tensor,   # [B, T, H]
    A: torch.Tensor,          # [B, T, H, BT]
    dw: torch.Tensor,         # [B, T, H, K]
    du: torch.Tensor,         # [B, T, H, V]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward through WY representation.

    Wraps fla's prepare_wy_repr_bwd internally.

    Returns:
        dk2: [B, T, H, K]
        dv: [B, T, H, V]
        db: [B, T, H]
        dg2: [B, T, H]
    """
    from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd
    return prepare_wy_repr_bwd(
        k=k, v=v, beta=beta, A=A, dw=dw, du=du, g=g_cumsum,
    )


def reverse_cumsum(dg: torch.Tensor) -> torch.Tensor:
    """Reverse cumulative sum within each chunk (stage 5 of backward).

    This undoes the forward cumsum applied to g.

    Args:
        dg: [B, T, H]

    Returns:
        dg_reversed: [B, T, H]
    """
    from fla.ops.utils import chunk_local_cumsum
    return chunk_local_cumsum(dg, chunk_size=BT, reverse=True)
