# Copyright (c) 2025, Machete Authors
"""Tests for CrossEntropyOp — forward+backward correctness.

Tests run on GPU (Hopper+) and compare the megakernel CrossEntropyOp against
a pure PyTorch reference implementation.
"""

import contextlib
import io
import importlib.util

import pytest
import torch

if importlib.util.find_spec("cutlass") is None:
    pytest.skip("Requires CUTLASS", allow_module_level=True)

from machete.kernels.cross_entropy.ref import cross_entropy_ref
from tests.kernels.support import requires_hopper_cutlass


requires_gpu = requires_hopper_cutlass

BASIC_CASES = [
    (1, 128, torch.bfloat16),
    (4, 1024, torch.float32),
    (32, 1024, torch.bfloat16),
    (128, 32000, torch.bfloat16),
]


def _run_cross_entropy(logits, targets, ignore_index=-100):
    """Run CrossEntropyOp and return loss, grad_logits."""
    from machete.megakernel import Megakernel
    from machete.kernels.cross_entropy import CrossEntropyOp

    BT, V = logits.shape
    loss = torch.zeros(BT, dtype=torch.float32, device=logits.device)
    grad_logits = torch.zeros_like(logits)

    ops = CrossEntropyOp.schedule(
        logits=logits,
        targets=targets.int(),
        loss=loss,
        grad_logits=grad_logits,
        ignore_index=ignore_index,
    )
    config = CrossEntropyOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return loss, grad_logits


# =============================================================================
# Forward + Backward Tests
# =============================================================================


@requires_gpu
@pytest.mark.parametrize("BT,V,dtype", BASIC_CASES)
def test_cross_entropy_basic(BT, V, dtype):
    """Test forward loss and backward gradients against PyTorch reference."""
    torch.manual_seed(42)
    logits = torch.randn(BT, V, dtype=dtype, device="cuda")
    targets = torch.randint(0, V, (BT,), device="cuda")

    loss, grad_logits = _run_cross_entropy(logits, targets)
    ref_loss, ref_grad = cross_entropy_ref(logits, targets)

    # Forward loss
    torch.testing.assert_close(loss, ref_loss, atol=1e-4, rtol=1e-4)

    # Backward gradients
    if dtype == torch.float32:
        torch.testing.assert_close(grad_logits, ref_grad, atol=1e-5, rtol=1e-4)
    else:
        torch.testing.assert_close(grad_logits, ref_grad, atol=1e-2, rtol=1e-2)


@requires_gpu
@pytest.mark.parametrize("BT", [4, 32])
def test_cross_entropy_ignore_index(BT):
    """Test ignore_index: ignored rows should have loss=0 and grad=0."""
    V = 1024
    torch.manual_seed(42)
    logits = torch.randn(BT, V, dtype=torch.bfloat16, device="cuda")
    targets = torch.randint(0, V, (BT,), device="cuda")

    # Set half the targets to ignore_index
    ignore_mask = torch.arange(BT, device="cuda") % 2 == 0
    targets[ignore_mask] = -100

    loss, grad_logits = _run_cross_entropy(logits, targets, ignore_index=-100)
    ref_loss, ref_grad = cross_entropy_ref(logits, targets, ignore_index=-100)

    torch.testing.assert_close(loss, ref_loss, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(grad_logits, ref_grad, atol=1e-2, rtol=1e-2)

    # Verify ignored rows are exactly zero
    assert (loss[ignore_mask] == 0).all()
    assert (grad_logits[ignore_mask] == 0).all()


@requires_gpu
def test_cross_entropy_all_ignored():
    """Test when all targets are ignore_index."""
    BT, V = 8, 512
    logits = torch.randn(BT, V, dtype=torch.bfloat16, device="cuda")
    targets = torch.full((BT,), -100, device="cuda", dtype=torch.long)

    loss, grad_logits = _run_cross_entropy(logits, targets)

    assert (loss == 0).all()
    assert (grad_logits == 0).all()


@requires_gpu
def test_cross_entropy_single_token():
    """Test single token (BT=1)."""
    V = 32000
    torch.manual_seed(42)
    logits = torch.randn(1, V, dtype=torch.bfloat16, device="cuda")
    targets = torch.tensor([1234], device="cuda")

    loss, grad_logits = _run_cross_entropy(logits, targets)
    ref_loss, ref_grad = cross_entropy_ref(logits, targets)

    torch.testing.assert_close(loss, ref_loss, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(grad_logits, ref_grad, atol=1e-2, rtol=1e-2)


@requires_gpu
def test_cross_entropy_large_vocab():
    """Test with large vocabulary (128256 for Qwen 3.5)."""
    BT, V = 4, 128256
    torch.manual_seed(42)
    logits = torch.randn(BT, V, dtype=torch.bfloat16, device="cuda")
    targets = torch.randint(0, V, (BT,), device="cuda")

    loss, grad_logits = _run_cross_entropy(logits, targets)
    ref_loss, ref_grad = cross_entropy_ref(logits, targets)

    torch.testing.assert_close(loss, ref_loss, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(grad_logits, ref_grad, atol=1e-2, rtol=1e-2)
