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


def causal_conv1d_bwd_ref(dy, x, weight, activation=None):
    """Reference backward for causal depthwise conv1d.

    Args:
        dy: (B, L, D) gradient of output, channels-last.
        x: (B, L, D) forward input tensor, channels-last.
        weight: (D, K) depthwise conv kernel.
        activation: None or 'silu'. If set, dy is pre-activation gradient.

    Returns:
        dx: (B, L, D) gradient w.r.t. input.
        dw: (D, K) gradient w.r.t. weight.
    """
    B, L, D = x.shape
    K = weight.shape[1]

    # If activation was applied in forward, chain rule through it
    if activation == 'silu':
        y = causal_conv1d_ref(x, weight, activation=None)
        y_t = y.transpose(1, 2).contiguous()  # (B, D, L)
        sigmoid_y = torch.sigmoid(y_t)
        silu_grad = sigmoid_y * (1 + y_t * (1 - sigmoid_y))
        dy_pre = dy.transpose(1, 2).contiguous() * silu_grad
        dy_pre = dy_pre.transpose(1, 2).contiguous()
    else:
        dy_pre = dy

    # dx: cross-correlate dy with weight (flip kernel, pad future)
    # dx[b,l,d] = sum_j w[d,K-1-j] * dy[b, l+j, d]  for j=0..K-1
    # This is F.conv1d with flipped kernel and future padding
    dy_t = dy_pre.transpose(1, 2).contiguous()  # (B, D, L)
    w_flip = weight.flip(1).unsqueeze(1)  # (D, 1, K) — flipped kernel
    dx_t = F.conv1d(dy_t, w_flip, padding=K - 1, groups=D)
    dx_t = dx_t[..., K - 1:]  # take last L elements (future-causal)
    dx = dx_t.transpose(1, 2).contiguous()  # (B, L, D)

    # dw: outer product of x and dy, summed over B and L
    # dw[d,k] = sum_{b,l} dy[b,l,d] * x[b, l-K+1+k, d]  (zero-pad x for l-K+1+k < 0)
    x_t = x.transpose(1, 2).contiguous()  # (B, D, L)
    # Pad x with K-1 zeros on the left
    x_padded = F.pad(x_t, (K - 1, 0))  # (B, D, L+K-1)
    # dw[d,k] = sum_{b,l} dy_t[b,d,l] * x_padded[b,d, l+k]
    dw = torch.zeros_like(weight)
    for k in range(K):
        dw[:, k] = (dy_t * x_padded[:, :, k:k + L]).sum(dim=(0, 2))

    return dx, dw


__all__ = ["causal_conv1d_ref", "causal_conv1d_bwd_ref"]
