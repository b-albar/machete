# Copyright (c) 2025, Machete Authors
"""
Host-side autograd protocol for megakernel Ops.

AutogradOp is a companion to the GPU-side Op class. It describes the
tensor-level contract for PyTorch autograd: which tensors are inputs/outputs,
which need gradients, and what to save for backward.

Subclasses must set ``op_cls`` and implement ``tensor_specs()``.
Override ``prepare_tensors()`` if autograd tensors need reshaping
before the Op (e.g., 4D → 3D).

Usage:
    class RopeAutogradOp(AutogradOp):
        op_cls = RopeOp

        def tensor_specs(self):
            return [
                TensorSpec("q", needs_grad=True),
                TensorSpec("cos"),
                TensorSpec("sin"),
                TensorSpec("q_rotated", is_output=True, mutated_from="q"),
            ]

        def prepare_tensors(self, q, cos, sin, **kw):
            b, s, h, d = q.shape
            return {"q": q.view(b * s, h, d).contiguous(), "cos": cos, "sin": sin}
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Type

import torch

from .ops import Op


@dataclass
class TensorSpec:
    """Declares a tensor's role in the autograd graph.

    Attributes:
        name: Identifier (e.g., "q", "cos", "grad_q").
        needs_grad: Whether this input requires a gradient in backward.
        is_output: True if this tensor is produced by the op.
        mutated_from: If this output is an in-place mutation of an input,
            name the input. The same tensor object is returned (zero-copy).
            None means the output is a fresh tensor.
    """

    name: str
    needs_grad: bool = False
    is_output: bool = False
    mutated_from: Optional[str] = None


class AutogradOp(ABC):
    """Host-side autograd contract for a megakernel Op.

    Pairs with a GPU-side Op class. Defines tensor I/O and
    save-for-backward semantics. Does NOT modify the GPU Op.

    Config packing and scheduling are handled by Op.schedule().

    Subclasses must set ``op_cls`` and implement ``tensor_specs()``.
    Override ``prepare_tensors()`` when autograd tensors have different
    shapes than what the Op expects.
    """

    op_cls: ClassVar[Type[Op]]

    @abstractmethod
    def tensor_specs(self) -> List[TensorSpec]:
        """Declare all input and output tensors with their autograd roles.

        Order matters: inputs first, outputs second.

        Example for RoPE::

            [
                TensorSpec("q", needs_grad=True),
                TensorSpec("cos"),
                TensorSpec("sin"),
                TensorSpec("q_rotated", is_output=True, mutated_from="q"),
            ]
        """
        ...

    def save_for_backward(self, **tensors) -> Dict[str, torch.Tensor]:
        """Select which tensors to save for backward.

        Default: save all inputs that don't require grad (constants like
        cos, sin that are needed for the backward computation).
        """
        result = {}
        for spec in self.input_specs():
            if not spec.needs_grad and spec.name in tensors:
                result[spec.name] = tensors[spec.name]
        return result

    def prepare_tensors(self, **tensors) -> Dict[str, torch.Tensor]:
        """Prepare tensors for the Op (e.g., reshape 4D → 3D).

        Returns a dict of tensors matching the Op's declared shapes,
        used by Op.schedule() for config packing and tile computation.
        Default: pass through unchanged.
        """
        return tensors

    # --- Helpers ---

    def input_specs(self) -> List[TensorSpec]:
        """Return only input specs (not outputs)."""
        return [s for s in self.tensor_specs() if not s.is_output]

    def output_specs(self) -> List[TensorSpec]:
        """Return only output specs."""
        return [s for s in self.tensor_specs() if s.is_output]


__all__ = ["TensorSpec", "AutogradOp"]
