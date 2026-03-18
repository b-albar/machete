# Copyright (c) 2025, Machete Authors
"""MoE (Mixture-of-Experts) grouped GEMM kernels.

Provides MoeGemmOp for grouped GEMM in the megakernel framework,
plus host-side utilities for token sorting and alignment.

Usage:
    from machete.kernels.moe import MoeGemmOp, moe_align_sort

    # Host-side: sort tokens by expert, pad to tile_M alignment
    sorted_token_ids, expert_ids, sorted_weights, num_tokens_per_expert = (
        moe_align_sort(topk_ids, topk_weights, num_experts, block_size=64)
    )

    # Gather sorted input
    sorted_x = x[sorted_token_ids.clamp(max=x.shape[0]-1)]

    # Schedule and run grouped GEMM
    ops = MoeGemmOp.schedule(sorted_x=sorted_x, w=w,
                                      expert_ids=expert_ids, c=c)
    config = MoeGemmOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)
    kernel.run()
"""

from machete.kernels.moe.moe_gemm import MoeGemmOp
from machete.kernels.moe.moe_gemm_bwd import MoeGemmBwdOp
from machete.kernels.moe.align_sort import moe_align_sort

__all__ = ["MoeGemmOp", "MoeGemmBwdOp", "moe_align_sort"]
