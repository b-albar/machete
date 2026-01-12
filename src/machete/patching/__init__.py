# Copyright (c) 2025, Machete Authors
"""Patching utilities for HuggingFace model optimization.

Structure:
- llama.py: Llama model patches (attention, MLP)
- qwen.py: Qwen model patches (attention, MLP)
- ops/: Shared optimized operations (cross_entropy, rmsnorm)
"""

# We avoid top-level imports of llama and qwen to keep the package
# importable even when transformers is not installed (e.g. for kernel tests)

__all__ = [
    "MacheteCrossEntropyLoss",
    "patch_cross_entropy_loss",
    "patch_rmsnorm",
    "unpatch_rmsnorm",
]


def __getattr__(name):
    if name in ("llama", "qwen"):
        import importlib

        return importlib.import_module(f"machete.patching.{name}")

    if name in __all__:
        import machete.patching.ops as ops

        return getattr(ops, name)

    raise AttributeError(f"module {__name__} has no attribute {name}")
