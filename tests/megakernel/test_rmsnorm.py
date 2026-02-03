# Copyright (c) 2025, Machete Authors
"""Tests for RMSNormOp — forward and backward correctness.

Tests run on GPU (Hopper+) and compare the megakernel RMSNormOp against
a pure PyTorch reference implementation.
"""

import contextlib
import io

import pytest
import torch

from machete.kernels.rms_norm.ref import rmsnorm_pytorch, rmsnorm_backward_pytorch


def is_hopper_or_newer():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


try:
    import cutlass  # noqa: F401

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


requires_gpu = pytest.mark.skipif(
    not (is_hopper_or_newer() and CUTLASS_AVAILABLE),
    reason="Requires Hopper+ GPU with CUTLASS",
)


# =============================================================================
# Helpers
# =============================================================================


def _run_rmsnorm_forward(x_2d, weight, eps=1e-6):
    """Run RMSNormOp forward and return output tensor."""
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.rms_norm import RMSNormOp

    y = torch.zeros_like(x_2d)
    ops = [RMSNormOp.schedule(x=x_2d, weight=weight, y=y)]
    kernel = Megakernel(ops, config=MegakernelConfig())

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return y


def _run_rmsnorm_backward(dout_2d, x_2d, weight, eps=1e-6):
    """Run RMSNormOp backward and return dx tensor."""
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.rms_norm import RMSNormOp

    dx = torch.zeros_like(x_2d)
    ops = [RMSNormOp.schedule(
        backward=True, dout=dout_2d, x=x_2d, weight=weight, dx=dx,
    )]
    kernel = Megakernel(ops, config=MegakernelConfig(), backward=True)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return dx


# =============================================================================
# Forward Tests
# =============================================================================


class TestRMSNormForward:
    """Forward pass correctness tests."""

    @requires_gpu
    @pytest.mark.parametrize("M,D", [
        (1, 64),
        (4, 128),
        (32, 256),
        (128, 512),
        (256, 1024),
    ])
    def test_forward_shapes(self, M, D):
        """RMSNorm forward produces correct output for various shapes."""
        torch.manual_seed(42)
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")

        y_mk = _run_rmsnorm_forward(x, weight)
        y_ref = rmsnorm_pytorch(x, weight, eps=1e-6)

        torch.testing.assert_close(y_mk, y_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_forward_ones_weight(self):
        """With weight=1, output should be x * rstd."""
        torch.manual_seed(42)
        M, D = 16, 128
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.ones(D, dtype=torch.float32, device="cuda")

        y_mk = _run_rmsnorm_forward(x, weight)
        y_ref = rmsnorm_pytorch(x, weight, eps=1e-6)

        torch.testing.assert_close(y_mk, y_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_forward_constant_input(self):
        """Constant input: all elements equal → normalized to weight."""
        M, D = 8, 64
        x = torch.full((M, D), 2.0, dtype=torch.float32, device="cuda")
        weight = torch.ones(D, dtype=torch.float32, device="cuda")

        y_mk = _run_rmsnorm_forward(x, weight)
        # RMS of constant 2.0 = 2.0, so rstd = 1/2, y = 2 * 0.5 * 1 = 1.0
        expected = torch.ones(M, D, dtype=torch.float32, device="cuda")
        torch.testing.assert_close(y_mk, expected, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_forward_batch_independence(self):
        """Each row is normalized independently."""
        torch.manual_seed(42)
        M, D = 32, 128
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")

        y_full = _run_rmsnorm_forward(x, weight)

        # Process first row alone
        y_single = _run_rmsnorm_forward(x[:1], weight)

        torch.testing.assert_close(y_full[0], y_single[0], atol=1e-5, rtol=1e-5)

    @requires_gpu
    def test_forward_kernel_cache(self):
        """Second run with same shapes should reuse compiled kernel."""
        torch.manual_seed(42)
        M, D = 16, 128
        x1 = torch.randn(M, D, dtype=torch.float32, device="cuda")
        x2 = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")

        y1 = _run_rmsnorm_forward(x1, weight)
        y2 = _run_rmsnorm_forward(x2, weight)

        # Both should match reference
        torch.testing.assert_close(y1, rmsnorm_pytorch(x1, weight), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(y2, rmsnorm_pytorch(x2, weight), atol=1e-3, rtol=1e-3)


# =============================================================================
# Backward Tests
# =============================================================================


class TestRMSNormBackward:
    """Backward pass correctness tests."""

    @requires_gpu
    @pytest.mark.parametrize("M,D", [
        (1, 64),
        (4, 128),
        (32, 256),
        (128, 512),
    ])
    def test_backward_shapes(self, M, D):
        """RMSNorm backward produces correct gradients for various shapes."""
        torch.manual_seed(42)
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")
        dout = torch.randn(M, D, dtype=torch.float32, device="cuda")

        dx_mk = _run_rmsnorm_backward(dout, x, weight)
        dx_ref = rmsnorm_backward_pytorch(dout, x, weight, eps=1e-6)

        torch.testing.assert_close(dx_mk, dx_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_backward_zero_grad(self):
        """Zero grad_output → zero grad_input."""
        M, D = 8, 128
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")
        dout = torch.zeros(M, D, dtype=torch.float32, device="cuda")

        dx = _run_rmsnorm_backward(dout, x, weight)
        torch.testing.assert_close(dx, torch.zeros_like(dx), atol=1e-6, rtol=1e-6)


# =============================================================================
# Reference-only tests (no GPU required)
# =============================================================================


class TestRMSNormReference:
    """Tests for the PyTorch reference implementation (CPU)."""

    def test_pytorch_ref_basic(self):
        torch.manual_seed(42)
        x = torch.randn(4, 64)
        w = torch.ones(64)
        y = rmsnorm_pytorch(x, w)
        # Each row should have unit RMS
        rms = y.pow(2).mean(-1).sqrt()
        torch.testing.assert_close(rms, torch.ones(4), atol=1e-3, rtol=1e-3)

    def test_pytorch_backward_ref(self):
        """PyTorch backward matches autograd."""
        torch.manual_seed(42)
        x = torch.randn(4, 64, requires_grad=True)
        w = torch.randn(64)

        # Forward with autograd
        variance = x.float().pow(2).mean(-1, keepdim=True)
        y = (x.float() * torch.rsqrt(variance + 1e-6) * w.float()).to(x.dtype)
        dout = torch.randn_like(y)
        y.backward(dout)
        dx_autograd = x.grad.clone()

        # Our reference
        dx_ref = rmsnorm_backward_pytorch(dout, x.detach(), w)
        torch.testing.assert_close(dx_ref, dx_autograd, atol=1e-4, rtol=1e-4)
