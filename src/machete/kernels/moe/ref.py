# Copyright (c) 2025, Machete Authors
"""Reference implementations for MoE grouped GEMM testing.

Provides simple PyTorch loop-based implementations to validate the
CuTe DSL MoeGemmOp against known-correct behavior.
"""

import torch


def moe_gemm_ref(
    sorted_x: torch.Tensor,
    w: torch.Tensor,
    expert_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Reference grouped GEMM: c[i] = sorted_x[i] @ w[expert_ids[i]]^T.

    Processes token-by-token, selecting the correct expert weight matrix
    for each position based on expert_ids.

    Args:
        sorted_x: [total_padded, K] — input tokens in sorted order (fp16/bf16)
        w: [num_experts, N, K] — expert weights (fp16/bf16)
        expert_ids: [total_padded] — expert ID per sorted position (int32)
        sorted_token_ids: [total_padded] — original token indices.
            Positions where sorted_token_ids == num_original_tokens are padding.

    Returns:
        c: [total_padded, N] — output in sorted order (same dtype as sorted_x)
    """
    total_padded, K = sorted_x.shape
    num_experts, N, _ = w.shape
    dtype = sorted_x.dtype
    device = sorted_x.device

    c = torch.zeros(total_padded, N, dtype=dtype, device=device)

    # Group by expert for efficiency
    for e in range(num_experts):
        mask = expert_ids == e
        if not mask.any():
            continue
        # x_e: [tokens_for_expert, K]
        x_e = sorted_x[mask].float()
        # w_e: [N, K]
        w_e = w[e].float()
        # GEMM: [tokens_for_expert, K] @ [K, N] = [tokens_for_expert, N]
        c_e = x_e @ w_e.t()
        c[mask] = c_e.to(dtype)

    return c


def moe_full_ref(
    x: torch.Tensor,
    w: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Full MoE reference: route, gather, GEMM, scatter.

    Performs the complete MoE forward pass including routing, expert
    computation, and weighted output combination. Used as an end-to-end
    correctness reference.

    Args:
        x: [num_tokens, K] — input hidden states (fp16/bf16)
        w: [num_experts, N, K] — expert weights (fp16/bf16)
        topk_ids: [num_tokens, topk] — selected expert IDs per token
        topk_weights: [num_tokens, topk] — routing weights (float32)
        num_experts: Total number of experts

    Returns:
        output: [num_tokens, N] — weighted expert outputs (same dtype as x)
    """
    num_tokens, K = x.shape
    _, N, _ = w.shape
    topk = topk_ids.shape[1]
    dtype = x.dtype
    device = x.device

    output = torch.zeros(num_tokens, N, dtype=dtype, device=device)

    for t in range(num_tokens):
        for k_idx in range(topk):
            expert_id = topk_ids[t, k_idx].item()
            weight = topk_weights[t, k_idx].item()
            # Single token GEMM: [1, K] @ [K, N] = [1, N]
            token_out = (x[t].float() @ w[expert_id].float().t()).to(dtype)
            output[t] += weight * token_out

    return output


def moe_gemm_bwd_dx_ref(
    dc: torch.Tensor,
    w: torch.Tensor,
    expert_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Reference backward dx: dx[i] = dc[i] @ w[expert_ids[i]].

    Args:
        dc: [total_padded, N] — gradient w.r.t. output c (fp16/bf16)
        w: [num_experts, N, K] — expert weights (same as forward)
        expert_ids: [total_padded] — expert ID per sorted position (int32)
        sorted_token_ids: [total_padded] — original token indices.

    Returns:
        dx: [total_padded, K] — gradient w.r.t. sorted_x (same dtype as dc)
    """
    total_padded, N = dc.shape
    num_experts, _, K = w.shape
    dtype = dc.dtype
    device = dc.device

    dx = torch.zeros(total_padded, K, dtype=dtype, device=device)

    for e in range(num_experts):
        mask = expert_ids == e
        if not mask.any():
            continue
        dc_e = dc[mask].float()
        w_e = w[e].float()  # [N, K]
        # dx = dc @ w (no transpose): [tokens, N] @ [N, K] = [tokens, K]
        dx[mask] = (dc_e @ w_e).to(dtype)

    return dx


def moe_gemm_bwd_dw_ref(
    dc: torch.Tensor,
    sorted_x: torch.Tensor,
    expert_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Reference backward dw: dw[e] = dc_e^T @ sorted_x_e.

    Args:
        dc: [total_padded, N] — gradient w.r.t. output c (fp16/bf16)
        sorted_x: [total_padded, K] — input tokens in sorted order (fp16/bf16)
        expert_ids: [total_padded] — expert ID per sorted position (int32)
        num_experts: Total number of experts

    Returns:
        dw: [num_experts, N, K] — gradient w.r.t. weights (float32)
    """
    _, N = dc.shape
    _, K = sorted_x.shape
    device = dc.device

    dw = torch.zeros(num_experts, N, K, dtype=torch.float32, device=device)

    for e in range(num_experts):
        mask = expert_ids == e
        if not mask.any():
            continue
        dc_e = dc[mask].float()    # [tokens_e, N]
        x_e = sorted_x[mask].float()  # [tokens_e, K]
        # dw[e] = dc_e^T @ x_e: [N, tokens_e] @ [tokens_e, K] = [N, K]
        dw[e] = dc_e.t() @ x_e

    return dw
