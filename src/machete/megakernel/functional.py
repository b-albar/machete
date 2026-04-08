# Copyright (c) 2025, Machete Authors
"""
User-facing wrappers for autograd-enabled megakernel ops.

This module exposes the smallest public surface for running fused kernels
without instantiating ``Megakernel`` directly.

Usage:
    # Functional API
    q_rotated = megakernel_apply(RopeAutogradOp(), q=q, cos=cos, sin=sin)

    # Module API
    rope = MegakernelModule(RopeAutogradOp())
    q_rotated = rope(q=q, cos=cos, sin=sin)
"""

from typing import Optional

import torch
import torch.nn as nn

from .autograd import MegakernelFunction
from .autograd_op import AutogradOp
from .megakernel import MegakernelConfig


def megakernel_apply(
    *autograd_ops: AutogradOp,
    config: Optional[MegakernelConfig] = None,
    **named_tensors: torch.Tensor,
) -> torch.Tensor:
    """Execute one or more autograd-aware megakernel ops.

    Args:
        *autograd_ops: One or more AutogradOp instances.
        config: Megakernel config (optional, auto-detects num_sms).
        **named_tensors: Input tensors by name (must match tensor_specs).

    Returns:
        Output tensor or tuple of output tensors.

    Example::

        q_rotated = megakernel_apply(
            RopeAutogradOp(),
            q=q_tensor, cos=cos_cache, sin=sin_cache,
        )
    """
    if config is None:
        config = MegakernelConfig()

    # Flatten inputs in spec order
    flat_inputs = []
    for aop in autograd_ops:
        for spec in aop.input_specs():
            if spec.name not in named_tensors:
                raise ValueError(
                    f"Missing input tensor '{spec.name}' for "
                    f"{aop.__class__.__name__}"
                )
            flat_inputs.append(named_tensors[spec.name])

    return MegakernelFunction.apply(
        list(autograd_ops), config, *flat_inputs
    )


class MegakernelModule(nn.Module):
    """``nn.Module`` wrapper over :func:`megakernel_apply`.

    Example::

        rope = MegakernelModule(RopeAutogradOp())
        q_out = rope(q=q, cos=cos, sin=sin)
    """

    def __init__(
        self,
        *autograd_ops: AutogradOp,
        config: Optional[MegakernelConfig] = None,
    ):
        super().__init__()
        self.autograd_ops = list(autograd_ops)
        self.config = config or MegakernelConfig()

    def forward(self, **tensors: torch.Tensor) -> torch.Tensor:
        """Apply the configured autograd ops to named input tensors."""
        return megakernel_apply(
            *self.autograd_ops,
            config=self.config,
            **tensors,
        )


__all__ = ["megakernel_apply", "MegakernelModule"]
