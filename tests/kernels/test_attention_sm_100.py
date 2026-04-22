# Copyright (c) 2025, Machete Authors
"""Tests for FlashAttentionSm100Op — forward correctness (fp16/bf16 MMA only).

Tests run on GPU (Hopper+) and compare the megakernel FlashAttentionSm100Op
against a pure PyTorch reference implementation.
"""

import contextlib
import io
import importlib.util

import pytest
import torch

if importlib.util.find_spec("cutlass") is None:
    pytest.skip("Requires CUTLASS", allow_module_level=True)

from machete.kernels.attention.ref import flash_attention_pytorch
from tests.kernels.support import CUTLASS_AVAILABLE


def is_hopper_sm100():
    """SM100 attention requires Hopper (SM9x/10x), not Blackwell (SM12x)."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return 9 <= major <= 10


try:
    import cutlass  # noqa: F401

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


requires_gpu = pytest.mark.skipif(
    not (is_hopper_sm100() and CUTLASS_AVAILABLE),
    reason="Requires Hopper (SM100) GPU with CUTLASS — not supported on Blackwell",
)

MMA_FORWARD_CASES = [
    (1, 16, 16, 64),
    (1, 16, 16, 128),
    (1, 16, 64, 128),
    (4, 16, 32, 128),
    (1, 32, 32, 64),
    (1, 16, 20, 64),
    (1, 16, 256, 128),
]

MULTIWARP_FORWARD_CASES = [
    (1, 64, 64, 64),
    (1, 96, 64, 64),
    (1, 128, 128, 128),
    (4, 64, 64, 128),
    (1, 64, 50, 128),
    (1, 80, 64, 128),
    (4, 16, 128, 128),
    (1, 64, 512, 128),
]

CAUSAL_CASES = [
    (1, 32, 32, 64),
    (1, 16, 64, 64),
    (4, 32, 32, 64),
    (1, 64, 64, 128),
]

GQA_CASES = [
    (4, 2, 32, 32, 64, False),
    (8, 2, 32, 32, 128, False),
    (8, 4, 64, 64, 128, False),
    (4, 2, 32, 32, 64, True),
    (8, 2, 64, 64, 128, True),
]


# =============================================================================
# Helpers
# =============================================================================


def _run_attention_forward(q, k, v, tile_m=None, causal=False, kv_group_size=1):
    """Run FlashAttentionSm100Op forward and return output tensor."""
    from machete.megakernel import Megakernel
    from machete.kernels.attention import FlashAttentionSm100Op

    tile_sizes = {}
    if tile_m is not None:
        tile_sizes["M"] = tile_m
    # For fp16/bf16, let schedule compute optimal tile_M
    o = torch.zeros_like(q)
    ops = FlashAttentionSm100Op.schedule(
        q=q, k=k, v=v, o=o,
        tile_sizes=tile_sizes,
        causal=causal,
        kv_group_size=kv_group_size,
    )
    config = FlashAttentionSm100Op.kernel_config(ops)
    kernel = Megakernel(ops, config=config)

    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()

    torch.cuda.synchronize()
    return o


def _assert_attention_forward_close(q, k, v, *, tile_m=None, causal=False,
                                    kv_group_size=1):
    o_mk = _run_attention_forward(
        q, k, v, tile_m=tile_m, causal=causal, kv_group_size=kv_group_size)
    o_ref = flash_attention_pytorch(
        q.float(),
        k.float(),
        v.float(),
        causal=causal,
        kv_group_size=kv_group_size,
    ).half()
    torch.testing.assert_close(o_mk, o_ref, atol=5e-2, rtol=5e-2)


# =============================================================================
# fp16 MMA Tests (tensor core path)
# =============================================================================


class TestFlashAttentionMMA:
    """fp16 tensor core MMA path correctness tests."""

    @requires_gpu
    @pytest.mark.parametrize("BH,M,N,D", MMA_FORWARD_CASES)
    def test_mma_forward_matrix(self, BH, M, N, D):
        """Representative MMA forward matrix."""
        torch.manual_seed(42)
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        _assert_attention_forward_close(q, k, v, tile_m=16)

    @requires_gpu
    def test_mma_uniform_attention(self):
        """Uniform keys -> each output row equals mean of V."""
        BH, M, N, D = 1, 16, 16, 64
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k_row = torch.randn(1, 1, D, dtype=torch.float16, device="cuda")
        k = k_row.expand(BH, N, D).contiguous()
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o_mk = _run_attention_forward(q, k, v, tile_m=16)
        expected = v.float().mean(dim=1, keepdim=True).expand_as(q).half()

        torch.testing.assert_close(o_mk, expected, atol=5e-2, rtol=5e-2)

# =============================================================================
# Multi-warp MMA Tests (auto tile_M)
# =============================================================================


class TestFlashAttentionMultiWarp:
    """Multi-warp MMA tests — let schedule pick optimal tile_M."""

    @requires_gpu
    @pytest.mark.parametrize("BH,M,N,D", MULTIWARP_FORWARD_CASES)
    def test_multi_warp_forward_matrix(self, BH, M, N, D):
        """Representative multi-warp forward matrix."""
        torch.manual_seed(42)
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        _assert_attention_forward_close(q, k, v)


# =============================================================================
# Causal Tests (fp16 MMA)
# =============================================================================


class TestFlashAttentionCausal:
    """Causal masking tests (fp16)."""

    @requires_gpu
    @pytest.mark.parametrize("BH,M,N,D", CAUSAL_CASES)
    def test_causal_matrix(self, BH, M, N, D):
        """Representative causal forward matrix."""
        torch.manual_seed(42)
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        _assert_attention_forward_close(q, k, v, causal=True)


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


# =============================================================================
# GQA Tests (Grouped Query Attention)
# =============================================================================


class TestFlashAttentionGQA:
    """GQA tests: multiple Q heads share K/V heads."""

    @requires_gpu
    @pytest.mark.parametrize("BH_q,BH_kv,M,N,D,causal", GQA_CASES)
    def test_gqa_forward(self, BH_q, BH_kv, M, N, D, causal):
        """GQA forward matches PyTorch reference with repeat_interleave."""
        kv_group_size = BH_q // BH_kv
        torch.manual_seed(42)
        q = torch.randn(BH_q, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH_kv, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH_kv, N, D, dtype=torch.float16, device="cuda")

        _assert_attention_forward_close(
            q, k, v, causal=causal, kv_group_size=kv_group_size)
