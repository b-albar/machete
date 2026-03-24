# Copyright (c) 2025, Machete Authors
"""MoE token align and sort — host-side preprocessing for grouped GEMM.

Sorts tokens by expert assignment and pads each expert's token count to
a multiple of block_size (tile_M). This produces contiguous, aligned
token blocks that the MoeGemmOp can process tile-by-tile.

Algorithm (follows vLLM/SGLang moe_align_block_size pattern):
    1. Count tokens per expert
    2. Pad each expert's count to a multiple of block_size
    3. Compute cumulative offsets
    4. Scatter tokens to sorted positions

Usage:
    from machete.kernels.moe.align_sort import moe_align_sort

    sorted_token_ids, expert_ids, sorted_weights, num_tokens_per_expert = (
        moe_align_sort(topk_ids, topk_weights, num_experts=128, block_size=64)
    )
"""

import torch


def moe_align_sort(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort tokens by expert and pad to block_size alignment.

    Args:
        topk_ids: [num_tokens, topk] — selected expert IDs per token (int32/int64)
        topk_weights: [num_tokens, topk] — routing weights (float32)
        num_experts: Total number of experts
        block_size: Alignment granularity (= tile_size_M of MoeGemmOp)

    Returns:
        sorted_token_ids: [total_padded] — original token indices in sorted order (int32).
            Padding positions contain num_tokens (sentinel value).
        expert_ids: [total_padded] — expert ID per sorted position (int32).
            All positions within a block_size-aligned block share the same expert.
        sorted_weights: [total_padded] — routing weight per sorted position (float32).
            Padding positions contain 0.0.
        num_tokens_per_expert: [num_experts] — actual token count per expert (int32).
    """
    num_tokens, topk = topk_ids.shape
    device = topk_ids.device

    # Flatten topk assignments: (num_tokens * topk,)
    flat_ids = topk_ids.reshape(-1)             # expert_id for each (token, k) pair
    flat_weights = topk_weights.reshape(-1)     # routing weight for each pair
    # Token indices: [0,0,..,1,1,..,2,2,..] repeated topk times
    flat_token_ids = torch.arange(num_tokens, device=device).repeat_interleave(topk)

    # Count tokens per expert
    num_tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=device)
    num_tokens_per_expert.scatter_add_(
        0, flat_ids.to(torch.int64), torch.ones_like(flat_ids, dtype=torch.int32)
    )

    # Pad each expert's count to multiple of block_size
    padded_counts = ((num_tokens_per_expert + block_size - 1) // block_size) * block_size
    total_padded = int(padded_counts.sum().item())

    # Compute cumulative offsets per expert
    cumsum_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    cumsum_offsets[1:] = padded_counts.cumsum(0)

    # Sort by expert_id (stable sort preserves token order within each expert)
    sort_indices = torch.argsort(flat_ids, stable=True)
    sorted_flat_ids = flat_ids[sort_indices]
    sorted_flat_token_ids = flat_token_ids[sort_indices]
    sorted_flat_weights = flat_weights[sort_indices]

    # Allocate output arrays with sentinel/zero padding
    sorted_token_ids = torch.full(
        (total_padded,), num_tokens, dtype=torch.int32, device=device
    )
    expert_ids_out = torch.zeros(total_padded, dtype=torch.int32, device=device)
    sorted_weights_out = torch.zeros(total_padded, dtype=torch.float32, device=device)

    # Place sorted tokens into padded positions
    # For each expert, fill cumsum_offsets[e] .. cumsum_offsets[e] + actual_count
    actual_cumsum = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    actual_cumsum[1:] = num_tokens_per_expert.cumsum(0)

    for e in range(num_experts):
        actual_count = int(num_tokens_per_expert[e].item())
        if actual_count == 0:
            continue

        src_start = int(actual_cumsum[e].item())
        dst_start = int(cumsum_offsets[e].item())

        sorted_token_ids[dst_start:dst_start + actual_count] = (
            sorted_flat_token_ids[src_start:src_start + actual_count].to(torch.int32)
        )
        expert_ids_out[dst_start:dst_start + actual_count] = e
        sorted_weights_out[dst_start:dst_start + actual_count] = (
            sorted_flat_weights[src_start:src_start + actual_count]
        )

        # Fill padding positions with expert_id (so tile-based expert lookup works)
        padded_count = int(padded_counts[e].item())
        if padded_count > actual_count:
            expert_ids_out[dst_start + actual_count:dst_start + padded_count] = e

    return sorted_token_ids, expert_ids_out, sorted_weights_out, num_tokens_per_expert
