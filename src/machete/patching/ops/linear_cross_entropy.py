# Copyright (c) 2025, Machete Authors
"""Fused Linear + Cross Entropy using quack chunked implementation."""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from quack.linear_cross_entropy import (
        chunked_linear_cross_entropy,
        linear_cross_entropy_func,
        LinearCrossEntropy as QuackLinearCrossEntropy,
    )
    from quack.cute_dsl_utils import get_device_capacity

    HAS_QUACK_LINEAR_CE = True
except ImportError:
    HAS_QUACK_LINEAR_CE = False
    chunked_linear_cross_entropy = None
    linear_cross_entropy_func = None
    QuackLinearCrossEntropy = None
    get_device_capacity = None


def _check_compute_capability(device: torch.device) -> bool:
    """Check if device supports quack linear cross entropy (SM90+)."""
    if not HAS_QUACK_LINEAR_CE:
        return False
    if device.type != "cuda":
        return False
    try:
        cap = get_device_capacity(device)
        return cap[0] >= 9  # SM90 or higher
    except Exception:
        return False


class MacheteLinearCrossEntropy(nn.Module):
    """Fused Linear + Cross Entropy loss.

    Combines the final linear projection (lm_head) with cross entropy loss
    computation in a memory-efficient manner. Uses chunked computation to
    avoid materializing the full vocabulary logits tensor.

    This is especially useful for large vocabulary models where the logits
    tensor (batch * seq_len, vocab_size) can be very large.

    Automatically falls back to separate linear + cross_entropy when:
    - quack is not available
    - Device is not CUDA
    - GPU compute capability < SM90
    - Dimensions are not aligned

    Args:
        in_features: Input dimension (hidden size)
        out_features: Output dimension (vocabulary size)
        bias: Whether to use bias (default: False)
        ignore_index: Target index to ignore in loss
        reduction: Loss reduction type ("mean" or "sum")
        chunk_size: Chunk size for chunked computation. If None, uses
            non-chunked version. Must be divisible by 8.
        inplace_backward: Whether to do backward in-place (non-chunked only)
        tuned: Whether to use autotuned kernels
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        ignore_index: int = -100,
        reduction: Literal["mean", "sum"] = "mean",
        chunk_size: Optional[int] = 4096,
        inplace_backward: bool = False,
        tuned: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.chunk_size = chunk_size
        self.inplace_backward = inplace_backward
        self.tuned = tuned

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self._supports_quack: Optional[bool] = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _can_use_quack_chunked(self, input: Tensor) -> bool:
        """Check if we can use quack chunked version."""
        if self._supports_quack is None:
            self._supports_quack = _check_compute_capability(input.device)

        return (
            self._supports_quack
            and input.is_cuda
            and self.bias is None
            and input.stride(-1) == 1
            and self.in_features % 8 == 0
            and self.out_features % 8 == 0
            and input.shape[:-1].numel() % 8 == 0
            and self.chunk_size is not None
            and self.chunk_size % 8 == 0
            and self.reduction in ["mean", "sum"]
        )

    def _can_use_quack(self, input: Tensor) -> bool:
        """Check if we can use quack non-chunked version."""
        if self._supports_quack is None:
            self._supports_quack = _check_compute_capability(input.device)

        return self._supports_quack and input.is_cuda

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Compute fused linear + cross entropy loss.

        Args:
            input: Hidden states of shape (..., in_features)
            target: Target indices of shape (...)

        Returns:
            Loss scalar (with specified reduction)
        """
        # Flatten input and target
        original_shape = input.shape[:-1]
        input_flat = input.view(-1, self.in_features)
        target_flat = target.view(-1)

        # Ensure target is proper dtype
        if target_flat.dtype not in (torch.int32, torch.int64):
            target_flat = target_flat.to(torch.int64)

        # Try chunked quack version
        if self._can_use_quack_chunked(input_flat):
            return chunked_linear_cross_entropy(
                input_flat,
                self.weight,
                target_flat,
                chunk_size=self.chunk_size,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
                tuned=self.tuned,
            )

        # Try non-chunked quack version
        if self._can_use_quack(input_flat):
            return linear_cross_entropy_func(
                input_flat,
                self.weight,
                self.bias,
                target_flat,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
                inplace_backward=self.inplace_backward,
            )

        # Fallback to standard PyTorch
        logits = F.linear(input_flat, self.weight, self.bias)
        return F.cross_entropy(
            logits,
            target_flat,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


def fused_linear_cross_entropy(
    input: Tensor,
    weight: Tensor,
    target: Tensor,
    bias: Optional[Tensor] = None,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum"] = "mean",
    chunk_size: Optional[int] = 4096,
    tuned: bool = True,
) -> Tensor:
    """Functional API for fused linear + cross entropy.

    Args:
        input: Input tensor of shape (..., in_features)
        weight: Weight tensor of shape (vocab_size, in_features)
        target: Target indices of shape (...)
        bias: Optional bias tensor of shape (vocab_size,)
        ignore_index: Index to ignore in loss computation
        reduction: Type of reduction ("mean" or "sum")
        chunk_size: Chunk size for chunked computation
        tuned: Whether to use autotuned kernels

    Returns:
        Loss tensor with specified reduction
    """
    # Flatten input and target
    input_flat = input.view(-1, input.shape[-1])
    target_flat = target.view(-1)

    if target_flat.dtype not in (torch.int32, torch.int64):
        target_flat = target_flat.to(torch.int64)

    can_use_chunked = (
        HAS_QUACK_LINEAR_CE
        and _check_compute_capability(input.device)
        and input.is_cuda
        and bias is None
        and input_flat.stride(-1) == 1
        and weight.shape[1] % 8 == 0
        and weight.shape[0] % 8 == 0
        and input_flat.shape[0] % 8 == 0
        and chunk_size is not None
        and chunk_size % 8 == 0
        and reduction in ["mean", "sum"]
    )

    if can_use_chunked:
        return chunked_linear_cross_entropy(
            input_flat,
            weight,
            target_flat,
            chunk_size=chunk_size,
            ignore_index=ignore_index,
            reduction=reduction,
            tuned=tuned,
        )

    can_use_quack = (
        HAS_QUACK_LINEAR_CE
        and _check_compute_capability(input.device)
        and input.is_cuda
    )

    if can_use_quack:
        return linear_cross_entropy_func(
            input_flat,
            weight,
            bias,
            target_flat,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    # Fallback
    logits = F.linear(input_flat, weight, bias)
    return F.cross_entropy(
        logits,
        target_flat,
        ignore_index=ignore_index,
        reduction=reduction,
    )
