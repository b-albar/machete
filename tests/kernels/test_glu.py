# Copyright (c) 2025, Machete Authors
"""Tests for GLUOp — forward and backward correctness.

Tests run on GPU (Hopper+) and compare the megakernel GLUOp against
a pure PyTorch reference implementation. Covers:
    - SwiGLU (SiLU activation, default)
    - ReGLU (ReLU activation)
    - Plain gating (identity activation)
    - Multiple shapes and dtypes
    - Backward gradient correctness
"""

import contextlib
import io
import importlib.util

import pytest
import torch

if importlib.util.find_spec("cutlass") is None:
    pytest.skip("Requires CUTLASS", allow_module_level=True)

from machete.kernels.glu.ref import glu_pytorch, glu_backward_pytorch
from tests.kernels.support import requires_hopper_cutlass


requires_gpu = requires_hopper_cutlass

CORE_SHAPES = [
    (1, 64),
    (32, 256),
    (128, 512),
    (16, 4096),
]


# =============================================================================
# Helpers
# =============================================================================


def _to_3d(t):
    """Wrap 2D (M, N) tensor to 3D (1, M, N) for megakernel ops."""
    if t.ndim == 2:
        return t.unsqueeze(0)
    return t


def _run_glu_forward(x_2d, activation='silu'):
    """Run GLUOp forward and return output tensor."""
    from machete.megakernel import Megakernel
    from machete.kernels.glu import GLUOp

    x = _to_3d(x_2d)
    D = x.shape[2] // 2
    y = torch.zeros(*x.shape[:2], D, dtype=x.dtype, device=x.device)

    ops = GLUOp.schedule(x=x, y=y, activation=activation)
    config = GLUOp.kernel_config(ops)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel = Megakernel(ops, config=config)
        kernel.run()
    torch.cuda.synchronize()

    return y.squeeze(0) if x_2d.ndim == 2 else y


def _run_glu_backward(dy_2d, x_2d, activation='silu'):
    """Run GLUBwdOp and return dx tensor."""
    from machete.megakernel import Megakernel
    from machete.kernels.glu import GLUBwdOp

    x = _to_3d(x_2d)
    dy = _to_3d(dy_2d)
    dx = torch.zeros_like(x)

    ops = GLUBwdOp.schedule(dy=dy, x=x, dx=dx, activation=activation)
    config = GLUBwdOp.kernel_config(ops)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel = Megakernel(ops, config=config)
        kernel.run()
    torch.cuda.synchronize()

    return dx.squeeze(0) if x_2d.ndim == 2 else dx


# =============================================================================
# Forward Tests
# =============================================================================


class TestGLUForward:
    """Test GLUOp forward against PyTorch reference."""

    @requires_gpu
    @pytest.mark.parametrize("M,D", CORE_SHAPES)
    @pytest.mark.parametrize("activation", ["silu", "relu", "identity"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_forward(self, M, D, activation, dtype):
        """Compare GLUOp output against PyTorch reference."""
        torch.manual_seed(42)
        x = torch.randn(M, 2 * D, dtype=dtype, device="cuda")

        y_mk = _run_glu_forward(x, activation=activation)
        y_ref = glu_pytorch(x, activation=activation)

        torch.testing.assert_close(
            y_mk.float(), y_ref.float(), atol=1e-2, rtol=1e-2,
        )

    @requires_gpu
    def test_forward_batch(self):
        """Test with explicit batch dimension."""
        torch.manual_seed(42)
        B, S, D = 2, 64, 128
        x = torch.randn(B, S, 2 * D, dtype=torch.bfloat16, device="cuda")
        y = torch.zeros(B, S, D, dtype=torch.bfloat16, device="cuda")

        from machete.megakernel import Megakernel
        from machete.kernels.glu import GLUOp

        ops = GLUOp.schedule(x=x, y=y, activation='silu')
        config = GLUOp.kernel_config(ops)
        with contextlib.redirect_stdout(io.StringIO()):
            kernel = Megakernel(ops, config=config)
            kernel.run()
        torch.cuda.synchronize()

        y_ref = glu_pytorch(x, activation='silu')
        torch.testing.assert_close(
            y.float(), y_ref.float(), atol=1e-2, rtol=1e-2,
        )

    @requires_gpu
    def test_direct_forward_rejects_non_divisible_d_tile(self):
        """Direct wide GLU must reject D tiles that do not cover rows exactly."""
        from machete.kernels.glu import DirectGLUOp

        x = torch.empty(1, 16, 2 * 3584, dtype=torch.bfloat16, device="cuda")
        y = torch.empty(1, 16, 3584, dtype=torch.bfloat16, device="cuda")

        with pytest.raises(ValueError, match="D % tile_size_D"):
            DirectGLUOp.schedule(
                x=x,
                y=y,
                activation="silu",
                tile_sizes={"S": 16, "D": 1024},
            )


# =============================================================================
# Backward Tests
# =============================================================================


class TestGLUBackward:
    """Test GLUBwdOp against PyTorch reference."""

    @requires_gpu
    @pytest.mark.parametrize("M,D", CORE_SHAPES)
    @pytest.mark.parametrize("activation", ["silu", "relu", "identity"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_backward(self, M, D, activation, dtype):
        """Compare GLUBwdOp dx against PyTorch reference."""
        torch.manual_seed(42)
        x = torch.randn(M, 2 * D, dtype=dtype, device="cuda")
        dy = torch.randn(M, D, dtype=dtype, device="cuda")

        dx_mk = _run_glu_backward(dy, x, activation=activation)
        dx_ref = glu_backward_pytorch(dy, x, activation=activation)

        torch.testing.assert_close(
            dx_mk.float(), dx_ref.float(), atol=1e-2, rtol=1e-2,
        )

    @requires_gpu
    @pytest.mark.parametrize("activation", ["silu", "relu", "identity"])
    def test_backward_vs_autograd(self, activation):
        """Verify backward matches PyTorch autograd."""
        torch.manual_seed(42)
        M, D = 32, 128

        # Autograd reference
        x_ag = torch.randn(M, 2 * D, dtype=torch.float32, device="cuda",
                           requires_grad=True)
        gate_ag = x_ag[:, :D]
        up_ag = x_ag[:, D:]

        if activation == 'silu':
            y_ag = torch.nn.functional.silu(gate_ag) * up_ag
        elif activation == 'relu':
            y_ag = torch.nn.functional.relu(gate_ag) * up_ag
        elif activation == 'identity':
            y_ag = gate_ag * up_ag

        dy = torch.randn(M, D, dtype=torch.float32, device="cuda")
        y_ag.backward(dy)
        dx_autograd = x_ag.grad

        # Megakernel backward
        x_bf = x_ag.detach().bfloat16()
        dy_bf = dy.bfloat16()
        dx_mk = _run_glu_backward(dy_bf, x_bf, activation=activation)

        torch.testing.assert_close(
            dx_mk.float(), dx_autograd.float(),
            atol=5e-2, rtol=5e-2,
        )

    @requires_gpu
    def test_backward_zero_grad(self):
        """dy=0 should produce dx=0."""
        M, D = 16, 128
        x = torch.randn(M, 2 * D, dtype=torch.bfloat16, device="cuda")
        dy = torch.zeros(M, D, dtype=torch.bfloat16, device="cuda")

        dx = _run_glu_backward(dy, x, activation='silu')
        assert torch.all(dx == 0), "dx should be zero when dy is zero"

    @requires_gpu
    def test_backward_rejects_non_power_of_two_d_tile(self):
        """Reject TMA chunk shapes that can fault at runtime."""
        from machete.kernels.glu import GLUBwdOp

        x = torch.empty(1, 16, 2 * 3584, dtype=torch.bfloat16, device="cuda")
        dy = torch.empty(1, 16, 3584, dtype=torch.bfloat16, device="cuda")
        dx = torch.empty_like(x)

        with pytest.raises(ValueError, match="power-of-two tile_size_D"):
            GLUBwdOp.schedule(
                dy=dy,
                x=x,
                dx=dx,
                activation="silu",
                tile_sizes={"S": 8, "D": 896},
            )


# =============================================================================
# Reference Tests (CPU only, no GPU required)
# =============================================================================


class TestGLUReference:
    """Test PyTorch reference implementations."""

    @pytest.mark.parametrize("activation", ["silu", "relu", "identity"])
    def test_ref_basic(self, activation):
        """Basic ref forward correctness."""
        M, D = 8, 32
        x = torch.randn(M, 2 * D)
        y = glu_pytorch(x, activation=activation)
        assert y.shape == (M, D)
        assert not torch.isnan(y).any()

    @pytest.mark.parametrize("activation", ["silu", "relu", "identity"])
    def test_ref_backward_matches_autograd(self, activation):
        """Verify ref backward matches autograd."""
        M, D = 8, 32
        x = torch.randn(M, 2 * D, requires_grad=True)
        gate, up = x[:, :D], x[:, D:]

        if activation == 'silu':
            y = torch.nn.functional.silu(gate) * up
        elif activation == 'relu':
            y = torch.nn.functional.relu(gate) * up
        elif activation == 'identity':
            y = gate * up

        dy = torch.randn(M, D)
        y.backward(dy)

        dx_ref = glu_backward_pytorch(dy, x.detach(), activation=activation)
        torch.testing.assert_close(
            dx_ref.float(), x.grad.float(), atol=1e-5, rtol=1e-5,
        )
