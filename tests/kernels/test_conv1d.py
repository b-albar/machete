# Copyright (c) 2025, Machete Authors
"""Tests for Conv1dOp — causal depthwise convolution.

Tests run on GPU (Hopper+) and compare the megakernel Conv1dOp against
PyTorch reference (F.conv1d with causal padding).
"""

import contextlib
import io
import importlib.util

import pytest
import torch

if importlib.util.find_spec("cutlass") is None:
    pytest.skip("Requires CUTLASS", allow_module_level=True)

from machete.kernels.conv1d.ref import causal_conv1d_ref, causal_conv1d_bwd_ref
from tests.kernels.support import requires_hopper_cutlass


requires_gpu = requires_hopper_cutlass

CORE_SHAPES = [
    (1, 64, 128, 4),
    (2, 128, 64, 4),
    (1, 256, 256, 4),
    (4, 32, 128, 3),
]


# =============================================================================
# Helpers
# =============================================================================


def _run_conv1d(x, w, activation=None, tile_s=None, page_size=None):
    """Run Conv1dOp and return output tensor."""
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.conv1d import Conv1dOp

    D = x.shape[2]
    K = w.shape[1] if w.ndim == 2 and w.shape[0] != w.shape[1] else w.shape[0]
    # After transpose, w is (K, D); before transpose, w is (D, K).
    # Detect: if w.shape == (D, K) with D == x.shape[2], K is w.shape[1].
    if w.shape[0] == D and w.ndim == 2:
        K = w.shape[1]
    else:
        K = w.shape[0]
    elem_bytes = 2 if x.dtype in (torch.float16, torch.bfloat16) else 4
    if page_size is None:
        page_size = 49152  # 48KB
    if tile_s is None:
        # Smem reuse: halo (K-1 rows) + x tile (tile_s rows), y overwrites
        total_rows = page_size // (D * elem_bytes)
        tile_s = max(1, total_rows - (K - 1))

    y = torch.empty_like(x)
    ops = Conv1dOp.schedule(
        x=x, w=w, y=y, activation=activation,
        tile_sizes={"S": tile_s}, page_size=page_size,
    )
    config = Conv1dOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return y


# =============================================================================
# Tests
# =============================================================================


class TestConv1dStandalone:
    """Standalone causal conv1d correctness tests."""

    @requires_gpu
    @pytest.mark.parametrize("B,L,D,K", CORE_SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_conv1d_shapes(self, B, L, D, K, dtype):
        """Conv1d produces correct output for various shapes."""
        torch.manual_seed(42)
        x = torch.randn(B, L, D, dtype=dtype, device="cuda")
        w = torch.randn(D, K, dtype=dtype, device="cuda") * 0.1

        y = _run_conv1d(x, w)
        y_ref = causal_conv1d_ref(x, w).to(dtype)

        rtol, atol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
        torch.testing.assert_close(y, y_ref, rtol=rtol, atol=atol)

    @requires_gpu
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_conv1d_silu(self, dtype):
        """Conv1d with SiLU activation matches reference."""
        torch.manual_seed(42)
        B, L, D, K = 2, 64, 128, 4
        x = torch.randn(B, L, D, dtype=dtype, device="cuda")
        w = torch.randn(D, K, dtype=dtype, device="cuda") * 0.1

        y = _run_conv1d(x, w, activation='silu')
        y_ref = causal_conv1d_ref(x, w, activation='silu').to(dtype)

        rtol, atol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
        torch.testing.assert_close(y, y_ref, rtol=rtol, atol=atol)

    @requires_gpu
    def test_conv1d_causality(self):
        """Output at position l depends only on positions <= l."""
        torch.manual_seed(42)
        B, L, D, K = 1, 32, 64, 4
        dtype = torch.float16

        x = torch.randn(B, L, D, dtype=dtype, device="cuda")
        w = torch.randn(D, K, dtype=dtype, device="cuda") * 0.1

        y_full = _run_conv1d(x, w)

        # Zero out future positions and recompute — should match
        for t in range(L):
            x_trunc = x.clone()
            x_trunc[:, t + 1:, :] = 0.0
            y_trunc = _run_conv1d(x_trunc, w)
            torch.testing.assert_close(
                y_full[:, t, :], y_trunc[:, t, :],
                rtol=1e-3, atol=1e-3,
                msg=f"Causality violated at position {t}",
            )

    @requires_gpu
    def test_conv1d_small_tile(self):
        """Conv1d works with very small tile sizes (many tiles)."""
        torch.manual_seed(42)
        B, L, D, K = 1, 64, 128, 4
        dtype = torch.float16

        x = torch.randn(B, L, D, dtype=dtype, device="cuda")
        w = torch.randn(D, K, dtype=dtype, device="cuda") * 0.1

        y = _run_conv1d(x, w, tile_s=4)
        y_ref = causal_conv1d_ref(x, w).to(dtype)

        torch.testing.assert_close(y, y_ref, rtol=1e-3, atol=1e-3)

    @requires_gpu
    def test_conv1d_single_position(self):
        """Conv1d with L=1 (decode-like)."""
        torch.manual_seed(42)
        B, D, K = 4, 128, 4
        dtype = torch.float16

        x = torch.randn(B, 1, D, dtype=dtype, device="cuda")
        w = torch.randn(D, K, dtype=dtype, device="cuda") * 0.1

        y = _run_conv1d(x, w)
        y_ref = causal_conv1d_ref(x, w).to(dtype)

        torch.testing.assert_close(y, y_ref, rtol=1e-3, atol=1e-3)


def _run_conv1d_bwd(dy, w, tile_s=None, page_size=None):
    """Run Conv1dBwdOp and return dx tensor."""
    from machete.megakernel import Megakernel
    from machete.kernels.conv1d import Conv1dBwdOp

    D = dy.shape[2]
    if w.shape[0] == D and w.ndim == 2:
        K = w.shape[1]
    else:
        K = w.shape[0]
    elem_bytes = 2 if dy.dtype in (torch.float16, torch.bfloat16) else 4
    if page_size is None:
        page_size = 49152  # 48KB
    if tile_s is None:
        # Smem reuse: dy tile (tile_s rows) + halo (K-1 rows), dx overwrites
        total_rows = page_size // (D * elem_bytes)
        tile_s = max(1, total_rows - (K - 1))

    dx = torch.empty_like(dy)
    ops = Conv1dBwdOp.schedule(
        dy=dy, w=w, dx=dx,
        tile_sizes={"S": tile_s}, page_size=page_size,
    )
    config = Conv1dBwdOp.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return dx


class TestConv1dBackward:
    """Backward (dx gradient) correctness tests."""

    @requires_gpu
    @pytest.mark.parametrize("B,L,D,K", CORE_SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_conv1d_bwd_shapes(self, B, L, D, K, dtype):
        """Conv1d backward dx matches reference for various shapes."""
        torch.manual_seed(42)
        x = torch.randn(B, L, D, dtype=dtype, device="cuda")
        w = torch.randn(D, K, dtype=dtype, device="cuda") * 0.1
        dy = torch.randn(B, L, D, dtype=dtype, device="cuda")

        dx = _run_conv1d_bwd(dy, w)
        dx_ref, _ = causal_conv1d_bwd_ref(dy, x, w)
        dx_ref = dx_ref.to(dtype)

        rtol, atol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
        torch.testing.assert_close(dx, dx_ref, rtol=rtol, atol=atol)

    @requires_gpu
    def test_conv1d_bwd_small_tile(self):
        """Conv1d backward with small tile (many tiles, tests halo at boundaries)."""
        torch.manual_seed(42)
        B, L, D, K = 1, 64, 128, 4
        dtype = torch.float16

        w = torch.randn(D, K, dtype=dtype, device="cuda") * 0.1
        dy = torch.randn(B, L, D, dtype=dtype, device="cuda")
        x = torch.randn(B, L, D, dtype=dtype, device="cuda")

        dx = _run_conv1d_bwd(dy, w, tile_s=4)
        dx_ref, _ = causal_conv1d_bwd_ref(dy, x, w)
        dx_ref = dx_ref.to(dtype)

        torch.testing.assert_close(dx, dx_ref, rtol=1e-3, atol=1e-3)

    @requires_gpu
    def test_conv1d_bwd_single_position(self):
        """Conv1d backward with L=1."""
        torch.manual_seed(42)
        B, D, K = 4, 128, 4
        dtype = torch.float16

        w = torch.randn(D, K, dtype=dtype, device="cuda") * 0.1
        dy = torch.randn(B, 1, D, dtype=dtype, device="cuda")
        x = torch.randn(B, 1, D, dtype=dtype, device="cuda")

        dx = _run_conv1d_bwd(dy, w)
        dx_ref, _ = causal_conv1d_bwd_ref(dy, x, w)
        dx_ref = dx_ref.to(dtype)

        torch.testing.assert_close(dx, dx_ref, rtol=1e-3, atol=1e-3)

    @requires_gpu
    def test_conv1d_bwd_gradient_check(self):
        """Verify dx via finite differences (autograd consistency)."""
        torch.manual_seed(42)
        B, L, D, K = 1, 32, 64, 4
        dtype = torch.float32

        x = torch.randn(B, L, D, dtype=dtype, device="cuda", requires_grad=True)
        w = torch.randn(D, K, dtype=dtype, device="cuda")

        # Forward with autograd
        y = causal_conv1d_ref(x, w)
        dy = torch.randn_like(y)
        y.backward(dy)
        dx_autograd = x.grad.clone()

        # Our reference
        dx_ref, _ = causal_conv1d_bwd_ref(dy, x.detach(), w)

        torch.testing.assert_close(dx_ref, dx_autograd, rtol=1e-4, atol=1e-4)
