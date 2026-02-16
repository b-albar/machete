# Copyright (c) 2025, Machete Authors
"""Reference activation implementations for testing."""

import torch
import torch.nn.functional as F


def activation_pytorch(x, activation='relu'):
    """Pure PyTorch activation reference.

    Args:
        x: tensor of any shape
        activation: 'relu' or 'silu'

    Returns:
        activated tensor (same shape/dtype)
    """
    if activation == 'relu':
        return torch.relu(x)
    elif activation == 'silu':
        return F.silu(x)
    else:
        raise ValueError(f"Unknown activation: {activation}")


__all__ = ["activation_pytorch"]
