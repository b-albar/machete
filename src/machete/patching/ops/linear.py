# Copyright (c) 2025, Machete Authors
"""Optimized Linear layer using quack GEMM with compute capability detection."""

import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from quack.linear import linear_func
    from quack.cute_dsl_utils import get_device_capacity

    HAS_QUACK_LINEAR = True
except ImportError:
    HAS_QUACK_LINEAR = False
    linear_func = None
    get_device_capacity = None


def _check_compute_capability(device: torch.device) -> bool:
    """Check if device supports quack GEMM (SM90+)."""
    if not HAS_QUACK_LINEAR:
        return False
    if device.type != "cuda":
        return False
    try:
        cap = get_device_capacity(device)
        return cap[0] >= 9  # SM90 or higher
    except Exception:
        return False


class MacheteLinear(nn.Linear):
    """Drop-in replacement for nn.Linear using quack optimized GEMM.

    Automatically falls back to standard PyTorch linear when:
    - quack is not available
    - Device is not CUDA
    - GPU compute capability < SM90
    - Dimensions are not divisible by 8
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        fuse_grad_accum: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.fuse_grad_accum = fuse_grad_accum
        self._supports_quack: Optional[bool] = None

    def _can_use_quack(self, input: Tensor) -> bool:
        """Check if we can use quack for this forward pass."""
        if self._supports_quack is None:
            self._supports_quack = _check_compute_capability(input.device)

        return (
            self._supports_quack
            and input.is_cuda
            and self.bias is None  # quack linear_func works best without bias
            and self.in_features % 8 == 0
            and self.out_features % 8 == 0
        )

    def forward(self, input: Tensor) -> Tensor:
        if self._can_use_quack(input):
            return linear_func(
                input,
                self.weight,
                bias=self.bias,
                fuse_grad_accum=self.fuse_grad_accum,
            )
        return F.linear(input, self.weight, self.bias)


def make_linear_forward(fuse_grad_accum: bool = False):
    """Create optimized forward for Linear layers."""

    def forward(self, input: Tensor) -> Tensor:
        # Check compute capability on first call
        if not hasattr(self, "_machete_supports_quack"):
            self._machete_supports_quack = _check_compute_capability(input.device)

        can_use_quack = (
            self._machete_supports_quack
            and input.is_cuda
            and self.bias is None
            and self.in_features % 8 == 0
            and self.out_features % 8 == 0
        )

        if can_use_quack:
            return linear_func(
                input,
                self.weight,
                bias=self.bias,
                fuse_grad_accum=fuse_grad_accum,
            )
        return F.linear(input, self.weight, self.bias)

    return forward


def patch_linear(module: nn.Linear, fuse_grad_accum: bool = False):
    """Patch Linear module with optimized forward.

    No-op if quack is not available.
    """
    if not HAS_QUACK_LINEAR:
        return

    if hasattr(module, "_machete_original_forward"):
        return  # Already patched

    module._machete_original_forward = module.forward
    module.forward = types.MethodType(make_linear_forward(fuse_grad_accum), module)


def unpatch_linear(module: nn.Linear):
    """Restore original Linear forward."""
    if hasattr(module, "_machete_original_forward"):
        module.forward = module._machete_original_forward
        del module._machete_original_forward
        if hasattr(module, "_machete_supports_quack"):
            del module._machete_supports_quack
