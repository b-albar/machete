# Copyright (c) 2025, Machete Authors
"""Shared optimized operations that work across model types."""

from machete.patching.ops.cross_entropy import MacheteCrossEntropyLoss, patch_cross_entropy_loss
from machete.patching.ops.rmsnorm import make_rmsnorm_forward, patch_rmsnorm, unpatch_rmsnorm
from machete.patching.ops.linear import MacheteLinear, patch_linear, unpatch_linear
from machete.patching.ops.rope import MacheteRoPE, apply_rope
from machete.patching.ops.mlp import MacheteMLP, MacheteGatedMLP, patch_gated_mlp, unpatch_gated_mlp
from machete.patching.ops.attention import MacheteAttention, flash_attention
from machete.patching.ops.linear_cross_entropy import MacheteLinearCrossEntropy, fused_linear_cross_entropy

__all__ = [
    # Cross Entropy
    "MacheteCrossEntropyLoss",
    "patch_cross_entropy_loss",
    # RMSNorm
    "make_rmsnorm_forward",
    "patch_rmsnorm",
    "unpatch_rmsnorm",
    # Linear (GEMM)
    "MacheteLinear",
    "patch_linear",
    "unpatch_linear",
    # RoPE
    "MacheteRoPE",
    "apply_rope",
    # MLP
    "MacheteMLP",
    "MacheteGatedMLP",
    "patch_gated_mlp",
    "unpatch_gated_mlp",
    # Attention
    "MacheteAttention",
    "flash_attention",
    # Linear Cross Entropy (fused)
    "MacheteLinearCrossEntropy",
    "fused_linear_cross_entropy",
]
