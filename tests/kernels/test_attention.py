# Copyright (c) 2025, Machete Authors
"""Tests for FlashAttentionOp — forward correctness.

Tests run on GPU (Hopper+) and compare the megakernel FlashAttentionOp
against a pure PyTorch reference implementation.
"""

import contextlib
import io

import pytest
import torch

from machete.kernels.attention.ref import flash_attention_pytorch


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


def _tile_size_M(D, elem_bytes=4):
    """Compute tile_size_M that fits Q/O + mbar + KV row in PAGE_SIZE (16KB)."""
    # (tile_M + 2) * D * elem_bytes + 8 <= 16384
    max_tile = (16384 - 8) // (D * elem_bytes) - 2
    return min(4, max(1, max_tile))


def _run_attention_forward(q, k, v, tile_m=None, causal=False):
    """Run FlashAttentionOp forward and return output tensor."""
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.kernels.attention import FlashAttentionOp

    BH, M, D = q.shape
    if tile_m is None:
        tile_m = _tile_size_M(D, q.element_size())
    o = torch.zeros_like(q)
    ops = FlashAttentionOp.schedule(
        q=q, k=k, v=v, o=o,
        tile_sizes={"M": tile_m},
        causal=causal,
    )
    kernel = Megakernel(ops, config=MegakernelConfig())

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return o


# =============================================================================
# Forward Tests
# =============================================================================


class TestFlashAttentionForward:
    """Forward pass correctness tests."""

    @requires_gpu
    @pytest.mark.parametrize("BH,M,N,D", [
        (1, 4, 4, 64),
        (1, 4, 4, 128),
        (1, 8, 8, 64),
        (4, 4, 4, 64),
    ])
    def test_forward_shapes(self, BH, M, N, D):
        """Flash attention forward for various shapes."""
        torch.manual_seed(42)
        q = torch.randn(BH, M, D, dtype=torch.float32, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")

        o_mk = _run_attention_forward(q, k, v)
        o_ref = flash_attention_pytorch(q, k, v)

        torch.testing.assert_close(o_mk, o_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_forward_decode(self):
        """Decode: M=1, N=8 (single query attending to 8 keys)."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 1, 8, 64
        q = torch.randn(BH, M, D, dtype=torch.float32, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")

        o_mk = _run_attention_forward(q, k, v, tile_m=1)
        o_ref = flash_attention_pytorch(q, k, v)

        torch.testing.assert_close(o_mk, o_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_forward_multi_tile(self):
        """Multiple M tiles: M=16, tile_m=4."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 16, 8, 64
        q = torch.randn(BH, M, D, dtype=torch.float32, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")

        o_mk = _run_attention_forward(q, k, v, tile_m=4)
        o_ref = flash_attention_pytorch(q, k, v)

        torch.testing.assert_close(o_mk, o_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_forward_non_divisible_m(self):
        """Non-divisible M: M=10, tile_m=4."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 10, 8, 64
        q = torch.randn(BH, M, D, dtype=torch.float32, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")

        o_mk = _run_attention_forward(q, k, v, tile_m=4)
        o_ref = flash_attention_pytorch(q, k, v)

        torch.testing.assert_close(o_mk, o_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_forward_uniform_attention(self):
        """Uniform keys → each output row equals mean of V."""
        BH, M, N, D = 1, 4, 4, 64
        q = torch.randn(BH, M, D, dtype=torch.float32, device="cuda")
        # All K rows identical → uniform attention weights
        k_row = torch.randn(1, 1, D, dtype=torch.float32, device="cuda")
        k = k_row.expand(BH, N, D).contiguous()
        v = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")

        o_mk = _run_attention_forward(q, k, v)
        # With uniform attention, output = mean(V) for each row
        expected = v.float().mean(dim=1, keepdim=True).expand_as(q)
        torch.testing.assert_close(o_mk, expected.to(q.dtype), atol=1e-3, rtol=1e-3)


# =============================================================================
# Causal Tests
# =============================================================================


class TestFlashAttentionCausal:
    """Causal masking tests."""

    @requires_gpu
    def test_causal_square(self):
        """Square causal: M=N=8, lower-triangular attention."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 8, 8, 64
        q = torch.randn(BH, M, D, dtype=torch.float32, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")

        o_mk = _run_attention_forward(q, k, v, causal=True)
        o_ref = flash_attention_pytorch(q, k, v, causal=True)

        torch.testing.assert_close(o_mk, o_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_causal_decode(self):
        """Decode causal: M=1, N=8 → attends to all positions."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 1, 8, 64
        q = torch.randn(BH, M, D, dtype=torch.float32, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")

        o_mk = _run_attention_forward(q, k, v, tile_m=1, causal=True)
        o_ref = flash_attention_pytorch(q, k, v, causal=True)

        torch.testing.assert_close(o_mk, o_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_causal_asymmetric(self):
        """Asymmetric causal: M=4, N=16."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 4, 16, 64
        q = torch.randn(BH, M, D, dtype=torch.float32, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")

        o_mk = _run_attention_forward(q, k, v, causal=True)
        o_ref = flash_attention_pytorch(q, k, v, causal=True)

        torch.testing.assert_close(o_mk, o_ref, atol=1e-3, rtol=1e-3)

    @requires_gpu
    def test_causal_multi_head(self):
        """Multi-head causal: BH=4."""
        torch.manual_seed(42)
        BH, M, N, D = 4, 8, 8, 64
        q = torch.randn(BH, M, D, dtype=torch.float32, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float32, device="cuda")

        o_mk = _run_attention_forward(q, k, v, causal=True)
        o_ref = flash_attention_pytorch(q, k, v, causal=True)

        torch.testing.assert_close(o_mk, o_ref, atol=1e-3, rtol=1e-3)


# =============================================================================
# Reference-only tests (no GPU required)
# =============================================================================


class TestFlashAttentionReference:
    """Tests for the PyTorch reference implementation (CPU)."""

    def test_ref_basic(self):
        torch.manual_seed(42)
        q = torch.randn(1, 4, 64)
        k = torch.randn(1, 4, 64)
        v = torch.randn(1, 4, 64)
        o = flash_attention_pytorch(q, k, v)
        assert o.shape == (1, 4, 64)

    def test_ref_causal(self):
        """Causal: first row should only attend to first KV position."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 4, 4, 32
        q = torch.randn(BH, M, D)
        k = torch.randn(BH, N, D)
        v = torch.randn(BH, N, D)
        o = flash_attention_pytorch(q, k, v, causal=True)
        # First row (i=0) attends only to j=0, so output = V[0]
        torch.testing.assert_close(o[0, 0], v[0, 0].float(), atol=1e-5, rtol=1e-5)

    def test_ref_identity_attention(self):
        """With identity-like scores, output should match specific V rows."""
        BH, M, N, D = 1, 2, 2, 32
        # Q and K such that q[0]·k[0] >> q[0]·k[1]
        q = torch.zeros(BH, M, D)
        k = torch.zeros(BH, N, D)
        v = torch.randn(BH, N, D)

        # Make q[0] strongly prefer k[0]
        q[0, 0, 0] = 10.0
        k[0, 0, 0] = 10.0
        k[0, 1, 0] = -10.0

        o = flash_attention_pytorch(q, k, v)
        # Row 0 should be close to V[0]
        torch.testing.assert_close(o[0, 0], v[0, 0].float(), atol=1e-3, rtol=1e-3)
