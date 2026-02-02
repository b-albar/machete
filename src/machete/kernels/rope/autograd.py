# Copyright (c) 2025, Machete Authors
"""
AutogradOp for RoPE — bridges RopeOp with PyTorch autograd.

Usage:
    from machete.kernels.rope.autograd import RopeAutogradOp
    from machete.megakernel.functional import megakernel_apply

    q_rotated = megakernel_apply(RopeAutogradOp(), q=q, cos=cos, sin=sin)
    # q_rotated is q modified in-place, with autograd support.
"""

from typing import Dict, List

import torch

from machete.megakernel.autograd_op import AutogradOp, TensorSpec
from .rope import RopeOp


class RopeAutogradOp(AutogradOp):
    """AutogradOp for Rotary Position Embedding.

    Forward: applies rotation [[cos, -sin], [sin, cos]] to q in-place.
    Backward: applies inverse rotation [[cos, sin], [-sin, cos]] to grad_q.

    Input tensors:
        q: (B, S, H, D) float32 — modified in-place
        cos: (S, D//2) float32
        sin: (S, D//2) float32

    Output:
        q_rotated: same tensor as q, after in-place rotation
    """

    op_cls = RopeOp

    def tensor_specs(self) -> List[TensorSpec]:
        return [
            TensorSpec("q", needs_grad=True),
            TensorSpec("cos"),
            TensorSpec("sin"),
            TensorSpec("q_rotated", is_output=True, mutated_from="q"),
        ]

    def prepare_tensors(self, q, cos, sin, **kw) -> Dict[str, torch.Tensor]:
        b, s, h, d = q.shape
        return {"q": q.view(b * s, h, d).contiguous(), "cos": cos, "sin": sin}


__all__ = ["RopeAutogradOp"]
