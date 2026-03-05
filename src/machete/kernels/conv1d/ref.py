# Copyright (c) 2025, Machete Authors
"""Reference causal depthwise Conv1d implementations for testing.

Follows Tri Dao's causal-conv1d semantics:
    y[b, l, d] = bias[d] + sum(w[d, K-1-j] * x[b, l-j, d] for j in range(K))
    where x[b, l-j, d] = 0 when l-j < 0.

Optionally applies SiLU activation after convolution.
"""

import torch
import torch.nn.functional as F


def causal_conv1d_ref(x, weight, bias=None, activation=None):
    """Pure PyTorch causal depthwise conv1d reference.

    Args:
        x: (B, L, D) input tensor, channels-last.
        weight: (D, K) depthwise conv kernel.
        bias: optional (D,) bias.
        activation: None or 'silu'.

    Returns:
        (B, L, D) output tensor.
    """
    B, L, D = x.shape
    K = weight.shape[1]

    # F.conv1d expects (B, D, L) layout and (D, 1, K) weight
    x_t = x.transpose(1, 2).contiguous()  # (B, D, L)
    w_conv = weight.unsqueeze(1)  # (D, 1, K)

    out = F.conv1d(x_t, w_conv, bias, padding=K - 1, groups=D)
    out = out[..., :L]  # remove trailing padding

    # Back to channels-last
    out = out.transpose(1, 2).contiguous()  # (B, L, D)

    if activation == 'silu':
        out = F.silu(out)

    return out


__all__ = ["causal_conv1d_ref"]
