# Copyright (c) 2025, Machete Authors
"""Tests for Conv1dOp — causal depthwise convolution.

Tests run on GPU (Hopper+) and compare the megakernel Conv1dOp against
PyTorch reference (F.conv1d with causal padding).
"""

import contextlib
import io

import pytest
import torch

from machete.kernels.conv1d.ref import causal_conv1d_ref


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


def _run_conv1d(x, w, activation=None, tile_s=None, page_size=None):
    """Run Conv1dOp and return output tensor."""
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.conv1d import Conv1dOp

    D = x.shape[2]
    elem_bytes = 2 if x.dtype in (torch.float16, torch.bfloat16) else 4
    if page_size is None:
        page_size = 49152  # 48KB
    if tile_s is None:
        # 2x smem: x tile (TMA loaded) + y tile (compute output)
        tile_s = max(1, page_size // (2 * D * elem_bytes))

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
    @pytest.mark.parametrize("B,L,D,K", [
        (1, 64, 128, 4),
        (2, 128, 64, 4),
        (1, 256, 256, 4),
        (4, 32, 128, 3),
    ])
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
