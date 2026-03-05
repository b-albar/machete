# Copyright (c) 2025, Machete Authors
"""Reference PyTorch implementations for cross-entropy loss."""

import torch
import torch.nn.functional as F


def cross_entropy_ref(logits, targets, ignore_index=-100):
    """Reference cross-entropy forward + backward.

    Args:
        logits: (BT, V) float tensor
        targets: (BT,) long/int tensor
        ignore_index: target value to ignore

    Returns:
        loss: (BT,) per-row CE loss (fp32)
        grad_logits: (BT, V) dL/dlogits (same dtype as logits)
    """
    logits_f32 = logits.float()

    # Per-row loss
    loss = F.cross_entropy(
        logits_f32, targets.long(), ignore_index=ignore_index, reduction="none"
    )

    # Gradient: softmax - one_hot
    softmax = torch.softmax(logits_f32, dim=-1)
    one_hot = torch.zeros_like(softmax)
    mask = targets != ignore_index
    valid_targets = targets[mask].long()
    one_hot[mask, valid_targets] = 1.0

    grad_logits = softmax - one_hot
    grad_logits[~mask] = 0.0

    return loss.float(), grad_logits.to(logits.dtype)


__all__ = ["cross_entropy_ref"]
