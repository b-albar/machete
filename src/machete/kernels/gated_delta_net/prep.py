# Copyright (c) 2025, Machete Authors
"""Gated Delta Net preprocessing (stages 1-3 of the chunked algorithm).

Computes the WY representation needed for the state recurrence:
    Stage 1: g_cumsum = chunk_local_cumsum(g, BT=64)
    Stage 2: A = beta * K@K^T * exp(g_diff), strictly lower triangular
    Stage 3: A_solved = (I + A)^{-1}
    Stage 4: w = A_solved @ (k*beta*exp(g)), u = A_solved @ (v*beta)

All operations are per-chunk and embarrassingly parallel (no cross-chunk deps).

Currently a PyTorch reference implementation matching fla's Triton kernels
(chunk_local_cumsum, chunk_scaled_dot_kkt_fwd, solve_tril, recompute_w_u_fwd).
Will be replaced with a CuTe DSL megakernel.
"""

import torch


BT = 64  # Chunk size (matching fla convention)


def _chunk_local_cumsum(g: torch.Tensor, chunk_size: int = BT) -> torch.Tensor:
    """Cumulative sum within each chunk along the T dimension.

    Args:
        g: [B, T, H] gate values (fp32)

    Returns:
        g_cumsum: [B, T, H] cumulative sums reset at chunk boundaries (fp32)
    """
    B, T, H = g.shape
    assert T % chunk_size == 0, f"T ({T}) must be divisible by chunk_size ({chunk_size})"
    NT = T // chunk_size
    # Reshape to [B, NT, BT, H], cumsum along BT dim, reshape back
    g_chunked = g.reshape(B, NT, chunk_size, H)
    g_cumsum = torch.cumsum(g_chunked, dim=2)
    return g_cumsum.reshape(B, T, H)


def _chunk_scaled_dot_kkt(
    k: torch.Tensor,        # [B, T, H, K]
    g_cumsum: torch.Tensor,  # [B, T, H]
    beta: torch.Tensor,     # [B, T, H]
) -> torch.Tensor:
    """Compute A[i,j] = beta[i] * k[i]@k[j]^T * exp(g[i]-g[j]) for i>j.

    Strictly lower triangular within each chunk.

    Returns:
        A: [B, T, H, BT] (fp32)
    """
    B, T, H, K = k.shape
    device = k.device
    NT = T // BT

    # Reshape to chunks: [B, NT, BT, H, K] → [B*NT, H, BT, K]
    k_c = k.reshape(B, NT, BT, H, K).permute(0, 1, 3, 2, 4).reshape(B * NT, H, BT, K).float()
    g_c = g_cumsum.reshape(B, NT, BT, H).permute(0, 1, 3, 2).reshape(B * NT, H, BT)
    beta_c = beta.reshape(B, NT, BT, H).permute(0, 1, 3, 2).reshape(B * NT, H, BT)

    # K @ K^T: [B*NT, H, BT, K] @ [B*NT, H, K, BT] = [B*NT, H, BT, BT]
    kkt = torch.matmul(k_c, k_c.transpose(-2, -1))

    # Gating: exp(g[i] - g[j])
    kkt = kkt * torch.exp(g_c.unsqueeze(-1) - g_c.unsqueeze(-2))

    # Beta row-wise scaling
    kkt = kkt * beta_c.unsqueeze(-1)

    # Strictly lower triangular
    mask = torch.ones(BT, BT, device=device, dtype=torch.bool).tril(diagonal=-1)
    kkt.masked_fill_(~mask, 0.0)

    # Reshape back: [B*NT, H, BT, BT] → [B, T, H, BT]
    A = kkt.reshape(B, NT, H, BT, BT).permute(0, 1, 3, 2, 4).reshape(B, T, H, BT)
    return A


def _solve_tril(A: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Compute (I + A)^{-1} where A is strictly lower triangular.

    Uses batched matrix inverse across all chunks simultaneously.

    Args:
        A: [B, T, H, BT] strictly lower triangular (fp32)
        dtype: output dtype

    Returns:
        Ai: [B, T, H, BT] the inverse, same shape (in `dtype`)
    """
    B, T, H, _ = A.shape
    device = A.device
    NT = T // BT

    # Reshape to [B*NT, H, BT, BT]
    A_c = A.reshape(B, NT, BT, H, BT).permute(0, 1, 3, 2, 4).reshape(B * NT, H, BT, BT)

    # (I + A) is unit lower triangular → invertible
    I_plus_A = torch.eye(BT, device=device, dtype=torch.float32) + A_c
    inv = torch.linalg.inv(I_plus_A)

    # Reshape back: [B*NT, H, BT, BT] → [B, T, H, BT]
    Ai = inv.reshape(B, NT, H, BT, BT).permute(0, 1, 3, 2, 4).reshape(B, T, H, BT)
    return Ai.to(dtype)


def _recompute_w_u(
    k: torch.Tensor,        # [B, T, H, K]
    v: torch.Tensor,        # [B, T, H, V]
    beta: torch.Tensor,     # [B, T, H]
    A: torch.Tensor,        # [B, T, H, BT] (solved, in k.dtype)
    g_cumsum: torch.Tensor,  # [B, T, H]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute w = A @ (k*beta*exp(g)), u = A @ (v*beta).

    Returns:
        w: [B, T, H, K] (k.dtype)
        u: [B, T, H, V] (v.dtype)
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    dtype = k.dtype
    NT = T // BT

    # Reshape to chunks: [B*NT, H, BT, ...]
    A_c = A.reshape(B, NT, BT, H, BT).permute(0, 1, 3, 2, 4).reshape(B * NT, H, BT, BT).float()
    k_c = k.reshape(B, NT, BT, H, K).permute(0, 1, 3, 2, 4).reshape(B * NT, H, BT, K).float()
    v_c = v.reshape(B, NT, BT, H, V).permute(0, 1, 3, 2, 4).reshape(B * NT, H, BT, V).float()
    beta_c = beta.reshape(B, NT, BT, H).permute(0, 1, 3, 2).reshape(B * NT, H, BT)
    g_c = g_cumsum.reshape(B, NT, BT, H).permute(0, 1, 3, 2).reshape(B * NT, H, BT)

    # Weighted inputs
    k_weighted = k_c * (beta_c * torch.exp(g_c)).unsqueeze(-1)
    v_weighted = v_c * beta_c.unsqueeze(-1)

    # Batched matmul: [B*NT, H, BT, BT] @ [B*NT, H, BT, K/V]
    w_c = torch.matmul(A_c, k_weighted)
    u_c = torch.matmul(A_c, v_weighted)

    # Reshape back: [B*NT, H, BT, K] → [B, T, H, K]
    w = w_c.reshape(B, NT, H, BT, K).permute(0, 1, 3, 2, 4).reshape(B, T, H, K).to(dtype)
    u = u_c.reshape(B, NT, H, BT, V).permute(0, 1, 3, 2, 4).reshape(B, T, H, V).to(dtype)

    return w, u


def run_prep(
    k: torch.Tensor,     # [B, T, H, K]
    v: torch.Tensor,     # [B, T, H, V]
    g: torch.Tensor,     # [B, T, H] (fp32, log-space gates <= 0)
    beta: torch.Tensor,  # [B, T, H] (fp32, in [0,1])
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full preprocessing: cumsum → kkt → solve → w/u.

    Matches fla's pipeline: chunk_local_cumsum → chunk_scaled_dot_kkt_fwd →
    solve_tril → recompute_w_u_fwd.

    Args:
        k: Keys [B, T, H, K] (fp16/bf16)
        v: Values [B, T, H, V] (fp16/bf16)
        g: Log-space gates [B, T, H] (fp32, always <= 0)
        beta: Beta values [B, T, H] (fp32, in [0,1])

    Returns:
        g_cumsum: [B, T, H] cumulative gates (fp32)
        A: [B, T, H, BT] solved A matrix (k.dtype)
        w: [B, T, H, K] transformed keys (k.dtype)
        u: [B, T, H, V] transformed values (v.dtype)
    """
    # Stage 1: Chunk-local cumulative sum
    g_cumsum = _chunk_local_cumsum(g)

    # Stage 2: Scaled K@K^T
    A = _chunk_scaled_dot_kkt(k, g_cumsum, beta)

    # Stage 3: Triangular solve
    A = _solve_tril(A, dtype=k.dtype)

    # Stage 4: WY representation
    w, u = _recompute_w_u(k, v, beta, A, g_cumsum)

    return g_cumsum, A, w, u
