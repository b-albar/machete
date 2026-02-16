# Copyright (c) 2025, Machete Authors
"""Tests for ActivationOp — standalone and fused with GEMM.

Tests run on GPU (Hopper+) and compare the megakernel ActivationOp against
PyTorch reference implementations.
"""

import contextlib
import io

import pytest
import torch

from machete.kernels.activation.ref import activation_pytorch


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


def _tile_size_M(D, elem_bytes=2):
    """Compute largest tile_size_M that fits in PAGE_SIZE (16KB)."""
    return min(4, max(1, 16384 // (D * elem_bytes)))


def _run_activation(x_2d, activation='relu'):
    """Run ActivationOp and return output tensor (in-place on x_2d)."""
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.activation import ActivationOp

    D = x_2d.shape[1]
    elem_bytes = 2 if x_2d.dtype in (torch.float16, torch.bfloat16) else 4
    tile_m = _tile_size_M(D, elem_bytes)
    ops = ActivationOp.schedule(
        x=x_2d, activation=activation, tile_sizes={"M": tile_m},
    )
    kernel = Megakernel(ops, config=MegakernelConfig())

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return x_2d


def _run_gemm_activation(a, b_t, activation='relu', tile_m=64, tile_n=32, tile_k=32):
    """Run fused GEMM + Activation and return output tensor C."""
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.gemm import GemmOp
    from machete.kernels.activation import ActivationOp

    M, K = a.shape
    N = b_t.shape[0]
    c = torch.zeros(M, N, dtype=a.dtype, device=a.device)

    elem_bytes = 2 if a.dtype in (torch.float16, torch.bfloat16) else 4
    act_tile_m = _tile_size_M(N, elem_bytes)

    ops = GemmOp.schedule(
        a=a, b=b_t, c=c,
        tile_sizes={"M": tile_m, "N": tile_n, "K": tile_k},
    )
    ops += ActivationOp.schedule(
        x=c, activation=activation, tile_sizes={"M": act_tile_m},
    )

    config = MegakernelConfig(threads_per_block=160)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    return c


# =============================================================================
# Standalone Activation Tests
# =============================================================================


class TestActivationStandalone:
    """Standalone activation correctness tests."""

    @requires_gpu
    @pytest.mark.parametrize("M,D", [
        (1, 64),
        (4, 128),
        (32, 256),
        (128, 512),
    ])
    @pytest.mark.parametrize("activation", ['relu', 'silu'])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_activation_shapes(self, M, D, activation, dtype):
        """Activation produces correct output for various shapes."""
        torch.manual_seed(42)
        x = torch.randn(M, D, dtype=dtype, device="cuda")
        x_ref = x.clone()

        _run_activation(x, activation=activation)
        y_ref = activation_pytorch(x_ref, activation=activation)

        rtol, atol = (1e-2, 1e-2) if dtype == torch.bfloat16 else (1e-3, 1e-3)
        torch.testing.assert_close(x, y_ref, rtol=rtol, atol=atol)

    @requires_gpu
    def test_relu_zeros_negatives(self):
        """ReLU should zero out all negative values."""
        M, D = 16, 128
        x = torch.randn(M, D, dtype=torch.float16, device="cuda")
        x_ref = x.clone()

        _run_activation(x, activation='relu')

        assert (x[x_ref < 0] == 0).all(), "ReLU should zero negatives"
        torch.testing.assert_close(
            x[x_ref >= 0], x_ref[x_ref >= 0], rtol=1e-3, atol=1e-3,
        )


# =============================================================================
# Fused GEMM + Activation Tests
# =============================================================================


class TestGemmActivation:
    """Fused GEMM + Activation correctness tests."""

    @requires_gpu
    @pytest.mark.parametrize("activation", ['relu', 'silu'])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fused_gemm_activation(self, activation, dtype):
        """Fused GEMM + activation matches sequential reference."""
        torch.manual_seed(42)
        M, K, N = 64, 32, 32

        a = torch.randn(M, K, dtype=dtype, device="cuda")
        b = torch.randn(N, K, dtype=dtype, device="cuda")

        c = _run_gemm_activation(a, b, activation=activation)

        # Reference: matmul then activation
        c_ref = (a.float() @ b.float().t()).to(dtype)
        c_ref = activation_pytorch(c_ref, activation=activation)

        rtol, atol = (5e-2, 5e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
        torch.testing.assert_close(c, c_ref, rtol=rtol, atol=atol)

    @requires_gpu
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fused_gemm_relu_multi_k(self, dtype):
        """Fused GEMM + ReLU with multiple K tiles."""
        torch.manual_seed(42)
        M, K, N = 64, 64, 32

        a = torch.randn(M, K, dtype=dtype, device="cuda")
        b = torch.randn(N, K, dtype=dtype, device="cuda")

        c = _run_gemm_activation(a, b, activation='relu', tile_k=32)

        c_ref = (a.float() @ b.float().t()).to(dtype)
        c_ref = torch.relu(c_ref)

        rtol, atol = (5e-2, 5e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)
        torch.testing.assert_close(c, c_ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
