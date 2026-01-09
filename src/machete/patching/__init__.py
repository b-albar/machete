# Copyright (c) 2025, Machete Authors
"""Patching utilities for HuggingFace model optimization.

Structure:
- llama.py: Llama model patches (attention, MLP)
- qwen.py: Qwen model patches (attention, MLP)
- ops/: Shared optimized operations (cross_entropy, rmsnorm)
"""

from machete.patching import llama, qwen
from machete.patching.ops import (
    MacheteCrossEntropyLoss,
    patch_cross_entropy_loss,
    patch_rmsnorm,
    unpatch_rmsnorm,
)

__all__ = [
    "llama",
    "qwen",
    "MacheteCrossEntropyLoss",
    "patch_cross_entropy_loss",
    "patch_rmsnorm",
    "unpatch_rmsnorm",
]
