# Copyright (c) 2025, Machete Authors
"""Tests for RMSNormOp — forward and backward correctness.

Tests run on GPU (Hopper+) and compare the megakernel RMSNormOp against
a pure PyTorch reference implementation. Covers all variants:
    - Standard RMSNorm
    - Residual (y = rmsnorm(x) + x)
    - Gemma-style (w_eff = 1 + w)
    - Fused add + RMSNorm (residual_out = x + residual_in, y = rmsnorm(residual_out))
    - Gated RMSNorm (y = rmsnorm(x) * silu(gate))
    - Per-row weight (weight is (B, S, D) instead of shared (D,))
"""

import contextlib
import io

import pytest
import torch

from machete.kernels.rms_norm.ref import (
    rmsnorm_pytorch,
    rmsnorm_backward_pytorch,
    fused_add_rmsnorm_pytorch,
    fused_add_rmsnorm_backward_pytorch,
    rmsnorm_gated_pytorch,
    rmsnorm_gated_backward_pytorch,
)


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


def _to_3d(t):
    """Wrap 2D (M, D) tensor to 3D (1, M, D) for RMSNormOp."""
    if t.ndim == 2:
        return t.unsqueeze(0)
    return t


def _run_rmsnorm_forward(x_2d, weight, eps=1e-6, residual=False, gemma=False,
                         per_row_weight=False):
    """Run RMSNormOp forward and return output tensor."""
    from machete.megakernel import Megakernel
    from machete.kernels.rms_norm import RMSNormOp

    x = _to_3d(x_2d)
    y = torch.zeros_like(x)
    ops = RMSNormOp.schedule(
        x=x, weight=_to_3d(weight) if weight.ndim == 2 else weight,
        y=y, residual=residual, gemma=gemma, per_row_weight=per_row_weight,
    )
    config = RMSNormOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return y.squeeze(0)


def _run_rmsnorm_backward(dout_2d, x_2d, weight, eps=1e-6, residual=False,
                          gemma=False, per_row_weight=False):
    """Run RMSNormBwdOp backward and return dx tensor."""
    from machete.megakernel import Megakernel
    from machete.kernels.rms_norm import RMSNormBwdOp

    x = _to_3d(x_2d)
    dout = _to_3d(dout_2d)
    dx = torch.zeros_like(x)
    ops = RMSNormBwdOp.schedule(
        dout=dout, x=x,
        weight=_to_3d(weight) if weight.ndim == 2 else weight,
        dx=dx, residual=residual, gemma=gemma, per_row_weight=per_row_weight,
    )
    config = RMSNormBwdOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return dx.squeeze(0)


def _run_fused_add_rmsnorm_forward(x_2d, residual_in, weight):
    """Run RMSNormOp forward with fused add (residual_in provided)."""
    from machete.megakernel import Megakernel
    from machete.kernels.rms_norm import RMSNormOp

    x = _to_3d(x_2d)
    y = torch.zeros_like(x)
    residual_out = torch.zeros_like(x)
    ops = RMSNormOp.schedule(
        x=x, weight=weight, y=y,
        residual_in=_to_3d(residual_in), residual_out=residual_out,
    )
    config = RMSNormOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return y.squeeze(0), residual_out.squeeze(0)


def _run_fused_add_rmsnorm_backward(dout_2d, residual_out_2d, weight):
    """Run RMSNormBwdOp backward with fused add (d_residual output)."""
    from machete.megakernel import Megakernel
    from machete.kernels.rms_norm import RMSNormBwdOp

    dout = _to_3d(dout_2d)
    x = _to_3d(residual_out_2d)
    dx = torch.zeros_like(dout)
    d_residual = torch.zeros_like(dout)
    ops = RMSNormBwdOp.schedule(
        dout=dout, x=x, weight=weight,
        dx=dx, d_residual=d_residual,
    )
    config = RMSNormBwdOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return dx.squeeze(0), d_residual.squeeze(0)


def _run_rmsnorm_gated_forward(x_2d, gate, weight):
    """Run RMSNormOp forward with gating (gate provided)."""
    from machete.megakernel import Megakernel
    from machete.kernels.rms_norm import RMSNormOp

    x = _to_3d(x_2d)
    y = torch.zeros_like(x)
    ops = RMSNormOp.schedule(
        x=x, weight=weight, y=y, gate=_to_3d(gate),
    )
    config = RMSNormOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return y.squeeze(0)


def _run_rmsnorm_gated_backward(dout_2d, x_2d, gate, weight):
    """Run RMSNormBwdOp backward with gating."""
    from machete.megakernel import Megakernel
    from machete.kernels.rms_norm import RMSNormBwdOp

    x = _to_3d(x_2d)
    dout = _to_3d(dout_2d)
    dx = torch.zeros_like(x)
    dgate = torch.zeros_like(_to_3d(gate))
    ops = RMSNormBwdOp.schedule(
        dout=dout, x=x, weight=weight,
        dx=dx, gate=_to_3d(gate), dgate=dgate,
    )
    config = RMSNormBwdOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return dx.squeeze(0), dgate.squeeze(0)


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
        (32, 4096),
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
        """Constant input: all elements equal -> normalized to weight."""
        M, D = 8, 64
        x = torch.full((M, D), 2.0, dtype=torch.float32, device="cuda")
        weight = torch.ones(D, dtype=torch.float32, device="cuda")

        y_mk = _run_rmsnorm_forward(x, weight)
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
        (32, 4096),
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
        """Zero grad_output -> zero grad_input."""
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
        rms = y.pow(2).mean(-1).sqrt()
        torch.testing.assert_close(rms, torch.ones(4), atol=1e-3, rtol=1e-3)

    def test_pytorch_backward_ref(self):
        """PyTorch backward matches autograd."""
        torch.manual_seed(42)
        x = torch.randn(4, 64, requires_grad=True)
        w = torch.randn(64)

        variance = x.float().pow(2).mean(-1, keepdim=True)
        y = (x.float() * torch.rsqrt(variance + 1e-6) * w.float()).to(x.dtype)
        dout = torch.randn_like(y)
        y.backward(dout)
        dx_autograd = x.grad.clone()

        dx_ref = rmsnorm_backward_pytorch(dout, x.detach(), w)
        torch.testing.assert_close(dx_ref, dx_autograd, atol=1e-4, rtol=1e-4)


# =============================================================================
# Residual Forward Tests
# =============================================================================


class TestRMSNormResidualForward:
    """Forward pass with residual connection: y = rmsnorm(x) + x."""

    @requires_gpu
    @pytest.mark.parametrize("M,D", [
        (1, 64),
        (4, 128),
        (32, 256),
        (128, 512),
        (32, 4096),
    ])
    def test_residual_forward_shapes(self, M, D):
        torch.manual_seed(42)
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")

        y_mk = _run_rmsnorm_forward(x, weight, residual=True)
        y_ref = rmsnorm_pytorch(x, weight, eps=1e-6, residual=True)

        torch.testing.assert_close(y_mk, y_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_residual_vs_no_residual(self):
        torch.manual_seed(42)
        M, D = 16, 128
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")

        y_no_res = _run_rmsnorm_forward(x, weight, residual=False)
        y_res = _run_rmsnorm_forward(x, weight, residual=True)

        torch.testing.assert_close(y_res, y_no_res + x, atol=1e-3, rtol=1e-3)


# =============================================================================
# Residual Backward Tests
# =============================================================================


class TestRMSNormResidualBackward:
    """Backward pass with residual: dx = dx_rmsnorm + dout."""

    @requires_gpu
    @pytest.mark.parametrize("M,D", [
        (1, 64),
        (4, 128),
        (32, 256),
        (128, 512),
        (32, 4096),
    ])
    def test_residual_backward_shapes(self, M, D):
        torch.manual_seed(42)
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")
        dout = torch.randn(M, D, dtype=torch.float32, device="cuda")

        dx_mk = _run_rmsnorm_backward(dout, x, weight, residual=True)
        dx_ref = rmsnorm_backward_pytorch(dout, x, weight, eps=1e-6, residual=True)

        torch.testing.assert_close(dx_mk, dx_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_residual_vs_no_residual_backward(self):
        torch.manual_seed(42)
        M, D = 16, 128
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")
        dout = torch.randn(M, D, dtype=torch.float32, device="cuda")

        dx_no_res = _run_rmsnorm_backward(dout, x, weight, residual=False)
        dx_res = _run_rmsnorm_backward(dout, x, weight, residual=True)

        torch.testing.assert_close(dx_res, dx_no_res + dout, atol=1e-3, rtol=1e-3)


# =============================================================================
# GemmaRMSNorm Tests
# =============================================================================


class TestGemmaRMSNormForward:
    """Gemma-style RMSNorm: y = x * (1+w) * rstd."""

    @requires_gpu
    @pytest.mark.parametrize("M,D", [
        (1, 64),
        (4, 128),
        (32, 256),
        (128, 512),
    ])
    def test_gemma_forward_shapes(self, M, D):
        torch.manual_seed(42)
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")

        y_mk = _run_rmsnorm_forward(x, weight, gemma=True)
        y_ref = rmsnorm_pytorch(x, weight, eps=1e-6, gemma=True)

        torch.testing.assert_close(y_mk, y_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_gemma_vs_standard(self):
        torch.manual_seed(42)
        M, D = 16, 128
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")

        y_std = _run_rmsnorm_forward(x, weight)
        y_gemma = _run_rmsnorm_forward(x, weight, gemma=True)

        assert not torch.allclose(y_std, y_gemma, atol=1e-3)


class TestGemmaRMSNormBackward:
    """Gemma-style RMSNorm backward."""

    @requires_gpu
    @pytest.mark.parametrize("M,D", [
        (1, 64),
        (4, 128),
        (32, 256),
        (128, 512),
    ])
    def test_gemma_backward_shapes(self, M, D):
        torch.manual_seed(42)
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")
        dout = torch.randn(M, D, dtype=torch.float32, device="cuda")

        dx_mk = _run_rmsnorm_backward(dout, x, weight, gemma=True)
        dx_ref = rmsnorm_backward_pytorch(dout, x, weight, eps=1e-6, gemma=True)

        torch.testing.assert_close(dx_mk, dx_ref, atol=1e-3, rtol=1e-3)


# =============================================================================
# Fused Add + RMSNorm Tests
# =============================================================================


class TestFusedAddRMSNormForward:
    """Fused add + RMSNorm forward: residual_out = x + res, y = rmsnorm(residual_out)."""

    @requires_gpu
    @pytest.mark.parametrize("M,D", [
        (1, 64),
        (4, 128),
        (32, 256),
        (128, 512),
    ])
    def test_fused_add_forward_shapes(self, M, D):
        torch.manual_seed(42)
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        residual_in = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")

        y_mk, res_out_mk = _run_fused_add_rmsnorm_forward(x, residual_in, weight)
        y_ref, res_out_ref = fused_add_rmsnorm_pytorch(x, residual_in, weight)

        torch.testing.assert_close(res_out_mk, res_out_ref, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(y_mk, y_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_fused_add_zero_residual(self):
        """With zero residual, should match plain rmsnorm."""
        torch.manual_seed(42)
        M, D = 16, 128
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        residual_in = torch.zeros(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")

        y_fused, _ = _run_fused_add_rmsnorm_forward(x, residual_in, weight)
        y_plain = _run_rmsnorm_forward(x, weight)

        torch.testing.assert_close(y_fused, y_plain, atol=1e-3, rtol=1e-3)


class TestFusedAddRMSNormBackward:
    """Fused add + RMSNorm backward."""

    @requires_gpu
    @pytest.mark.parametrize("M,D", [
        (1, 64),
        (4, 128),
        (32, 256),
        (128, 512),
    ])
    def test_fused_add_backward_shapes(self, M, D):
        torch.manual_seed(42)
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        residual_in = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")
        dout = torch.randn(M, D, dtype=torch.float32, device="cuda")

        _, residual_out = fused_add_rmsnorm_pytorch(x, residual_in, weight)
        residual_out = residual_out.cuda()

        dx_mk, dres_mk = _run_fused_add_rmsnorm_backward(dout, residual_out, weight)
        dx_ref, dres_ref = fused_add_rmsnorm_backward_pytorch(dout, residual_out, weight)

        torch.testing.assert_close(dx_mk, dx_ref, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(dres_mk, dres_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_fused_add_backward_dx_equals_dres(self):
        """dx and d_residual should be identical (sum gradient)."""
        torch.manual_seed(42)
        M, D = 16, 128
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        residual_in = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")
        dout = torch.randn(M, D, dtype=torch.float32, device="cuda")

        _, residual_out = fused_add_rmsnorm_pytorch(x, residual_in, weight)
        residual_out = residual_out.cuda()

        dx_mk, dres_mk = _run_fused_add_rmsnorm_backward(dout, residual_out, weight)
        torch.testing.assert_close(dx_mk, dres_mk, atol=1e-6, rtol=1e-6)


# =============================================================================
# RMSNorm Gated Tests
# =============================================================================


class TestRMSNormGatedForward:
    """Gated RMSNorm forward: y = rmsnorm(x, w) * silu(gate)."""

    @requires_gpu
    @pytest.mark.parametrize("M,D", [
        (1, 64),
        (4, 128),
        (32, 256),
        (128, 512),
    ])
    def test_gated_forward_shapes(self, M, D):
        torch.manual_seed(42)
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        gate = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")

        y_mk = _run_rmsnorm_gated_forward(x, gate, weight)
        y_ref = rmsnorm_gated_pytorch(x, gate, weight)

        torch.testing.assert_close(y_mk, y_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_gated_ones_gate(self):
        """silu(0) = 0, so gate=0 -> y should be ~0."""
        torch.manual_seed(42)
        M, D = 16, 128
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        gate = torch.zeros(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")

        y_mk = _run_rmsnorm_gated_forward(x, gate, weight)
        torch.testing.assert_close(y_mk, torch.zeros_like(y_mk), atol=1e-3, rtol=1e-3)


class TestRMSNormGatedBackward:
    """Gated RMSNorm backward."""

    @requires_gpu
    @pytest.mark.parametrize("M,D", [
        (1, 64),
        (4, 128),
        (32, 256),
        (128, 512),
    ])
    def test_gated_backward_shapes(self, M, D):
        torch.manual_seed(42)
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        gate = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")
        dout = torch.randn(M, D, dtype=torch.float32, device="cuda")

        dx_mk, dgate_mk = _run_rmsnorm_gated_backward(dout, x, gate, weight)
        dx_ref, dgate_ref = rmsnorm_gated_backward_pytorch(dout, x, gate, weight)

        torch.testing.assert_close(dx_mk, dx_ref, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(dgate_mk, dgate_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_gated_backward_zero_dout(self):
        """Zero dout -> zero dx, dgate."""
        M, D = 8, 128
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        gate = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(D, dtype=torch.float32, device="cuda")
        dout = torch.zeros(M, D, dtype=torch.float32, device="cuda")

        dx, dgate = _run_rmsnorm_gated_backward(dout, x, gate, weight)
        torch.testing.assert_close(dx, torch.zeros_like(dx), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(dgate, torch.zeros_like(dgate), atol=1e-6, rtol=1e-6)


# =============================================================================
# Per-Row Weight Tests
# =============================================================================


class TestPerRowWeightForward:
    """Per-row weight RMSNorm: each row has its own weight vector."""

    @requires_gpu
    @pytest.mark.parametrize("M,D", [
        (1, 64),
        (4, 128),
        (32, 256),
        (128, 512),
    ])
    def test_per_row_weight_forward(self, M, D):
        torch.manual_seed(42)
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(M, D, dtype=torch.float32, device="cuda")

        y_mk = _run_rmsnorm_forward(x, weight, per_row_weight=True)
        y_ref = rmsnorm_pytorch(x, weight)  # Broadcasting handles (M, D) weight

        torch.testing.assert_close(y_mk, y_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_per_row_weight_matches_shared(self):
        """Per-row weight with identical rows should match shared weight."""
        torch.manual_seed(42)
        M, D = 16, 128
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight_1d = torch.randn(D, dtype=torch.float32, device="cuda")
        weight_2d = weight_1d.unsqueeze(0).expand(M, -1).contiguous()

        y_shared = _run_rmsnorm_forward(x, weight_1d)
        y_per_row = _run_rmsnorm_forward(x, weight_2d, per_row_weight=True)

        torch.testing.assert_close(y_shared, y_per_row, atol=1e-5, rtol=1e-5)

    @requires_gpu
    def test_per_row_weight_different_rows(self):
        """Different weight per row produces different results than shared."""
        torch.manual_seed(42)
        M, D = 8, 64
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight_2d = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight_1d = weight_2d[0]  # Shared = first row's weight

        y_per_row = _run_rmsnorm_forward(x, weight_2d, per_row_weight=True)
        y_shared = _run_rmsnorm_forward(x, weight_1d)

        # Different weights per row should produce different outputs
        assert not torch.allclose(y_per_row, y_shared, atol=1e-3)


class TestPerRowWeightBackward:
    """Per-row weight RMSNorm backward."""

    @requires_gpu
    @pytest.mark.parametrize("M,D", [
        (1, 64),
        (4, 128),
        (32, 256),
        (128, 512),
    ])
    def test_per_row_weight_backward(self, M, D):
        torch.manual_seed(42)
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight = torch.randn(M, D, dtype=torch.float32, device="cuda")
        dout = torch.randn(M, D, dtype=torch.float32, device="cuda")

        dx_mk = _run_rmsnorm_backward(dout, x, weight, per_row_weight=True)
        dx_ref = rmsnorm_backward_pytorch(dout, x, weight)

        torch.testing.assert_close(dx_mk, dx_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_per_row_weight_backward_matches_shared(self):
        """Per-row backward with identical rows matches shared backward."""
        torch.manual_seed(42)
        M, D = 16, 128
        x = torch.randn(M, D, dtype=torch.float32, device="cuda")
        weight_1d = torch.randn(D, dtype=torch.float32, device="cuda")
        weight_2d = weight_1d.unsqueeze(0).expand(M, -1).contiguous()
        dout = torch.randn(M, D, dtype=torch.float32, device="cuda")

        dx_shared = _run_rmsnorm_backward(dout, x, weight_1d)
        dx_per_row = _run_rmsnorm_backward(dout, x, weight_2d, per_row_weight=True)

        torch.testing.assert_close(dx_shared, dx_per_row, atol=1e-5, rtol=1e-5)


class TestRMSNormScratchOverflow:
    """Regression test: tile_S must be clamped so scratch fits in page.

    When tile_S * D * elem_bytes == page_size, the 64-byte scratch area
    overflows the page boundary, corrupting the next page's data. This
    caused NaN in combined megakernels (e.g., RMSNorm + GEMM).
    """

    @requires_gpu
    def test_tile_s_clamped(self):
        """tile_sizes={'S': 16} must be clamped to 15 for D=1024, pg=32768."""
        from machete.kernels.rms_norm import RMSNormOp

        D = 1024
        x = torch.randn(1, 64, D, dtype=torch.bfloat16, device="cuda")
        y = torch.zeros_like(x)
        w = torch.randn(D, dtype=torch.bfloat16, device="cuda")

        # Explicit tile_S=16 would overflow: 16*1024*2 = 32768 = page_size
        ops = RMSNormOp.schedule(
            x=x, weight=w, y=y, tile_sizes={"S": 16}, page_size=32768,
        )
        assert ops[0].tile_sizes["S"] <= 15, (
            f"tile_S should be clamped to 15, got {ops[0].tile_sizes['S']}"
        )

    @requires_gpu
    def test_combined_rmsnorm_gemm_no_nan(self):
        """RMSNorm + independent GEMM must not produce NaN (scratch overflow)."""
        from machete.megakernel import Megakernel
        from machete.kernels.rms_norm import RMSNormOp
        from machete.kernels.gemm import GemmOp

        torch.manual_seed(42)
        B, S, D = 1, 1024, 1024
        page_size = 32768
        dtype = torch.bfloat16
        dev = "cuda"

        x = torch.randn(B, S, D, dtype=dtype, device=dev)
        residual = torch.randn(B, S, D, dtype=dtype, device=dev)
        w = torch.randn(D, dtype=dtype, device=dev)
        res_out = torch.empty(B, S, D, dtype=dtype, device=dev)
        h = torch.empty(B, S, D, dtype=dtype, device=dev)

        a = torch.randn(B, S, 2048, dtype=dtype, device=dev)
        W = torch.randn(D, 2048, dtype=dtype, device=dev) * 0.02
        c = torch.empty(B, S, D, dtype=dtype, device=dev)

        ops = (
            RMSNormOp.schedule(
                x=x, weight=w, y=h,
                residual_in=residual, residual_out=res_out,
                tile_sizes={"S": 16}, page_size=page_size,
            )
            + GemmOp.schedule(a=a, b=W, c=c, page_size=page_size)
        )
        config = GemmOp.kernel_config(ops)
        kernel = Megakernel(ops, config=config)
        with contextlib.redirect_stdout(io.StringIO()):
            kernel.run()
        torch.cuda.synchronize()

        assert not torch.isnan(h).any(), "RMSNorm output h has NaN"
        assert not torch.isnan(c).any(), "GEMM output c has NaN"
