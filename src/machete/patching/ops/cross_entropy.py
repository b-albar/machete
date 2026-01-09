# Copyright (c) 2025, Machete Authors
"""Cross-entropy loss patching using quack."""

from typing import Optional

import torch
import torch.nn as nn

try:
    from quack.cross_entropy import cross_entropy

    HAS_QUACK_CROSS_ENTROPY = True
except ImportError:
    HAS_QUACK_CROSS_ENTROPY = False
    cross_entropy = None


class MacheteCrossEntropyLoss(nn.Module):
    """Drop-in replacement for nn.CrossEntropyLoss using quack."""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        # Note: weight and label_smoothing not yet supported by quack
        if weight is not None:
            raise NotImplementedError("weight parameter not supported in MacheteCrossEntropyLoss")
        if label_smoothing != 0.0:
            raise NotImplementedError("label_smoothing not supported in MacheteCrossEntropyLoss")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if HAS_QUACK_CROSS_ENTROPY and input.is_cuda:
            # quack cross_entropy expects (M, N) for input and (M,) for target
            # HF models typically pass (batch * seq, vocab) and (batch * seq,)
            if input.dim() > 2:
                input = input.view(-1, input.size(-1))
            if target.dim() > 1:
                target = target.view(-1)

            # Ensure target is int32 or int64
            if target.dtype not in (torch.int32, torch.int64):
                target = target.to(torch.int64)

            return cross_entropy(
                input,
                target,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
            )
        else:
            return nn.functional.cross_entropy(
                input,
                target,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
            )


def patch_cross_entropy_loss(model):
    """Replace CrossEntropyLoss in model with optimized version."""
    # This patches any CrossEntropyLoss found in the model
    for name, module in model.named_modules():
        if isinstance(module, nn.CrossEntropyLoss):
            parent_name = ".".join(name.split(".")[:-1])
            module_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(
                parent,
                module_name,
                MacheteCrossEntropyLoss(
                    ignore_index=module.ignore_index,
                    reduction=module.reduction,
                    label_smoothing=getattr(module, "label_smoothing", 0.0),
                ),
            )
