# Copyright (c) 2025, Machete Authors
"""Optimized RMSNorm forward using quack."""

import types

import torch

try:
    from quack.rmsnorm import rmsnorm_fwd

    HAS_QUACK_RMSNORM = True
except ImportError:
    HAS_QUACK_RMSNORM = False
    rmsnorm_fwd = None


def make_rmsnorm_forward():
    """Create optimized forward for RMSNorm layers."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Reshape if needed - quack expects 2D input
        original_shape = hidden_states.shape
        if hidden_states.dim() > 2:
            hidden_states = hidden_states.view(-1, original_shape[-1])

        out, _, _ = rmsnorm_fwd(
            hidden_states,
            weight=self.weight,
            eps=self.variance_epsilon,
        )

        if len(original_shape) > 2:
            out = out.view(original_shape)

        return out

    return forward


def patch_rmsnorm(module):
    """Patch RMSNorm module with optimized forward.

    No-op if quack is not available.
    """
    if not HAS_QUACK_RMSNORM:
        return  # Quack not available, skip patching

    if hasattr(module, "_machete_original_forward"):
        return  # Already patched

    module._machete_original_forward = module.forward
    module.forward = types.MethodType(make_rmsnorm_forward(), module)


def unpatch_rmsnorm(module):
    """Restore original RMSNorm forward."""
    if hasattr(module, "_machete_original_forward"):
        module.forward = module._machete_original_forward
        del module._machete_original_forward
