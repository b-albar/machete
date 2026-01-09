# Copyright (c) 2025, Machete Authors
"""Shared optimized operations that work across model types."""

from machete.patching.ops.cross_entropy import MacheteCrossEntropyLoss, patch_cross_entropy_loss
from machete.patching.ops.rmsnorm import make_rmsnorm_forward, patch_rmsnorm, unpatch_rmsnorm

__all__ = [
    "MacheteCrossEntropyLoss",
    "patch_cross_entropy_loss",
    "make_rmsnorm_forward",
    "patch_rmsnorm",
    "unpatch_rmsnorm",
]
