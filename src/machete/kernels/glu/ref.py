# Copyright (c) 2025, Machete Authors
"""Reference GLU implementations for testing."""

import torch
import torch.nn.functional as F


def glu_pytorch(x, activation='silu'):
    """Pure PyTorch GLU forward reference.

    Args:
        x: (..., 2D) input tensor — last dim split into gate and up halves.
        activation: 'silu', 'relu', or 'identity'.

    Returns:
        y: (..., D) output tensor.
    """
    D = x.shape[-1] // 2
    gate = x[..., :D].float()
    up = x[..., D:].float()

    if activation == 'silu':
        act_gate = F.silu(gate)
    elif activation == 'relu':
        act_gate = F.relu(gate)
    elif activation == 'identity':
        act_gate = gate
    else:
        raise ValueError(f"Unknown activation: {activation}")

    return (act_gate * up).to(x.dtype)


def glu_backward_pytorch(dy, x, activation='silu'):
    """Pure PyTorch GLU backward reference.

    Args:
        dy: (..., D) gradient of output.
        x: (..., 2D) forward input (saved for backward).
        activation: 'silu', 'relu', or 'identity'.

    Returns:
        dx: (..., 2D) gradient w.r.t. input.
    """
    D = x.shape[-1] // 2
    gate = x[..., :D].float()
    up = x[..., D:].float()
    dy_f = dy.float()

    if activation == 'silu':
        sig = torch.sigmoid(gate)
        silu_val = gate * sig
        silu_grad = sig * (1.0 + gate * (1.0 - sig))
        d_up = dy_f * silu_val
        d_gate = dy_f * up * silu_grad
    elif activation == 'relu':
        mask = (gate > 0).float()
        d_up = dy_f * F.relu(gate)
        d_gate = dy_f * up * mask
    elif activation == 'identity':
        d_up = dy_f * gate
        d_gate = dy_f * up
    else:
        raise ValueError(f"Unknown activation: {activation}")

    return torch.cat([d_gate.to(x.dtype), d_up.to(x.dtype)], dim=-1)


__all__ = ["glu_pytorch", "glu_backward_pytorch"]
