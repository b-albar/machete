# Copyright (c) 2025, Machete Authors
"""Tests for FlashDecoding (split-KV attention)."""

import contextlib
import io
import importlib.util

import pytest
import torch

if importlib.util.find_spec("cutlass") is None:
    pytest.skip("Requires CUTLASS", allow_module_level=True)
from tests.kernels.support import requires_hopper_cutlass


requires_gpu = requires_hopper_cutlass

FORWARD_CASES = [
    (1, 16, 128, 128),
    (1, 16, 2048, 128),
    (8, 16, 2048, 128),
    (1, 16, 1024, 64),
]

CAUSAL_CASES = [
    (1, 16, 128, 128),
    (4, 16, 512, 128),
]

GQA_CASES = [
    (8, 2, 16, 512, 128, False),
    (8, 2, 16, 2048, 256, False),
    (8, 2, 16, 512, 128, True),
]


def _run_flash_decoding(q, k, v, num_splits=0, causal=False, kv_group_size=1):
    """Run FlashDecoding and return output."""
    from machete.kernels.attention.flash_decoding import flash_decoding_schedule
    from machete.megakernel import Megakernel

    o = torch.zeros_like(q)
    ops, config = flash_decoding_schedule(
        q=q, k=k, v=v, o=o,
        num_splits=num_splits,
        causal=causal, kv_group_size=kv_group_size,
    )
    kernel = Megakernel(ops, config=config)
    with contextlib.redirect_stdout(io.StringIO()):
        kernel.run()
    torch.cuda.synchronize()
    return o


def _ref_attention(q, k, v, causal=False, kv_group_size=1):
    """Reference attention using project's lower-left aligned causal mask."""
    from machete.kernels.attention.ref import flash_attention_pytorch

    B, M, H, D = q.shape
    q_flat = q.permute(0, 2, 1, 3).reshape(B * H, M, D)
    k_flat = k.permute(0, 2, 1, 3).reshape(B * k.shape[2], k.shape[1], D)
    v_flat = v.permute(0, 2, 1, 3).reshape(B * v.shape[2], v.shape[1], D)
    out = flash_attention_pytorch(
        q_flat, k_flat, v_flat, causal=causal, kv_group_size=kv_group_size
    )
    return out.view(B, H, M, D).permute(0, 2, 1, 3).contiguous()


@requires_gpu
class TestFlashDecodingBasic:
    """Basic FlashDecoding correctness tests."""

    @pytest.mark.parametrize("H,M,N,D", FORWARD_CASES)
    def test_forward_matrix(self, H, M, N, D):
        """Representative non-causal decode matrix."""
        torch.manual_seed(42)
        q = torch.randn(1, M, H, D, dtype=torch.float16, device="cuda")
        k = torch.randn(1, N, H, D, dtype=torch.float16, device="cuda")
        v = torch.randn(1, N, H, D, dtype=torch.float16, device="cuda")

        o = _run_flash_decoding(q, k, v)
        o_ref = _ref_attention(q, k, v)

        torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=5e-2)

    def test_explicit_splits(self):
        """Test with explicitly specified num_splits."""
        torch.manual_seed(42)
        H, M, N, D = 1, 16, 512, 128
        q = torch.randn(1, M, H, D, dtype=torch.float16, device="cuda")
        k = torch.randn(1, N, H, D, dtype=torch.float16, device="cuda")
        v = torch.randn(1, N, H, D, dtype=torch.float16, device="cuda")

        for num_splits in [2, 4, 8]:
            o = _run_flash_decoding(q, k, v, num_splits=num_splits)
            o_ref = _ref_attention(q, k, v)
            torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=5e-2)

    def test_bf16(self):
        """Test with bf16 dtype."""
        torch.manual_seed(42)
        H, M, N, D = 1, 16, 512, 128
        q = torch.randn(1, M, H, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, N, H, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(1, N, H, D, dtype=torch.bfloat16, device="cuda")

        o = _run_flash_decoding(q, k, v)
        o_ref = _ref_attention(q, k, v)

        torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=5e-2)

@requires_gpu
class TestFlashDecodingCausal:
    """FlashDecoding with causal masking."""

    @pytest.mark.parametrize("H,M,N,D", CAUSAL_CASES)
    def test_causal_matrix(self, H, M, N, D):
        torch.manual_seed(42)
        q = torch.randn(1, M, H, D, dtype=torch.float16, device="cuda")
        k = torch.randn(1, N, H, D, dtype=torch.float16, device="cuda")
        v = torch.randn(1, N, H, D, dtype=torch.float16, device="cuda")

        o = _run_flash_decoding(q, k, v, causal=True)
        o_ref = _ref_attention(q, k, v, causal=True)

        torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=5e-2)


@requires_gpu
class TestFlashDecodingGQA:
    """FlashDecoding with grouped query attention."""

    @pytest.mark.parametrize("H,H_kv,M,N,D,causal", GQA_CASES)
    def test_gqa_matrix(self, H, H_kv, M, N, D, causal):
        torch.manual_seed(42)
        kv_group_size = H // H_kv
        q = torch.randn(1, M, H, D, dtype=torch.float16, device="cuda")
        k = torch.randn(1, N, H_kv, D, dtype=torch.float16, device="cuda")
        v = torch.randn(1, N, H_kv, D, dtype=torch.float16, device="cuda")

        o = _run_flash_decoding(
            q, k, v, causal=causal, kv_group_size=kv_group_size)
        o_ref = _ref_attention(
            q, k, v, causal=causal, kv_group_size=kv_group_size)

        torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=5e-2)
