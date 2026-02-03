# Copyright (c) 2025, Machete Authors
"""Tests for the megakernel autograd framework.

Tests the framework-level autograd machinery: imports, AutogradOp protocol,
KernelCache, and MegakernelModule. Kernel-specific correctness tests
(forward/backward/gradcheck) live in tests/kernels/test_<kernel>.py.
"""

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def test_imports():
    """Verify all autograd components can be imported."""
    from machete.megakernel import (
        AutogradOp,
        TensorSpec,
        MegakernelFunction,
        KernelCache,
        megakernel_apply,
        MegakernelModule,
    )

    assert hasattr(AutogradOp, "tensor_specs")
    assert hasattr(AutogradOp, "save_for_backward")
    assert hasattr(AutogradOp, "prepare_tensors")


def test_tensor_spec_defaults():
    """TensorSpec defaults are correct."""
    from machete.megakernel import TensorSpec

    s = TensorSpec("x")
    assert s.name == "x"
    assert s.needs_grad is False
    assert s.is_output is False
    assert s.mutated_from is None


def test_autograd_op_input_output_helpers():
    """AutogradOp.input_specs() / output_specs() filter correctly."""
    from machete.megakernel import AutogradOp, TensorSpec

    class DummyOp(AutogradOp):
        op_cls = None

        def tensor_specs(self):
            return [
                TensorSpec("a", needs_grad=True),
                TensorSpec("b"),
                TensorSpec("c", is_output=True, mutated_from="a"),
            ]

    op = DummyOp()
    assert [s.name for s in op.input_specs()] == ["a", "b"]
    assert [s.name for s in op.output_specs()] == ["c"]
    assert op.output_specs()[0].mutated_from == "a"

    # Default save_for_backward: saves inputs without needs_grad
    saved = op.save_for_backward(a=torch.zeros(1), b=torch.ones(1))
    assert list(saved.keys()) == ["b"]


def test_kernel_cache_singleton():
    """KernelCache.get() returns the same instance."""
    from machete.megakernel import KernelCache

    c1 = KernelCache.get()
    c2 = KernelCache.get()
    assert c1 is c2


def test_kernel_cache_clear():
    """KernelCache.clear() empties the cache."""
    from machete.megakernel import KernelCache

    cache = KernelCache.get()
    cache.clear()
    assert len(cache) == 0


def test_megakernel_module_is_nn_module():
    """MegakernelModule is a proper nn.Module subclass."""
    from machete.megakernel import MegakernelModule

    assert issubclass(MegakernelModule, torch.nn.Module)
