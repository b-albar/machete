# Copyright (c) 2025, Machete Authors
"""Tests for FlashDecoding (split-KV attention)."""

import contextlib
import io

import pytest
import torch

try:
    import cutlass
    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


def is_hopper_or_newer():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_hopper_or_newer() and CUTLASS_AVAILABLE),
    reason="Requires Hopper+ GPU with CUTLASS",
)


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
    return flash_attention_pytorch(q, k, v, causal=causal, kv_group_size=kv_group_size)


@requires_gpu
class TestFlashDecodingBasic:
    """Basic FlashDecoding correctness tests."""

    @pytest.mark.parametrize("BH,M,N,D", [
        (1, 16, 128, 128),
        (1, 16, 256, 128),
        (1, 16, 512, 128),
        (1, 16, 2048, 128),
    ])
    def test_basic_decode(self, BH, M, N, D):
        """Single-head decode: BH=1, M=16 (typical decode config)."""
        torch.manual_seed(42)
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o = _run_flash_decoding(q, k, v)
        o_ref = _ref_attention(q, k, v)

        torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("BH,M,N,D", [
        (4, 16, 512, 128),
        (8, 16, 2048, 128),
        (1, 16, 1024, 64),
    ])
    def test_multi_head(self, BH, M, N, D):
        """Multi-head decode."""
        torch.manual_seed(42)
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o = _run_flash_decoding(q, k, v)
        o_ref = _ref_attention(q, k, v)

        torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=5e-2)

    def test_explicit_splits(self):
        """Test with explicitly specified num_splits."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 16, 512, 128
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        for num_splits in [2, 4, 8]:
            o = _run_flash_decoding(q, k, v, num_splits=num_splits)
            o_ref = _ref_attention(q, k, v)
            torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=5e-2)

    def test_bf16(self):
        """Test with bf16 dtype."""
        torch.manual_seed(42)
        BH, M, N, D = 1, 16, 512, 128
        q = torch.randn(BH, M, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.bfloat16, device="cuda")

        o = _run_flash_decoding(q, k, v)
        o_ref = _ref_attention(q, k, v)

        torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=5e-2)

    def test_long_decode_path(self):
        torch.manual_seed(42)
        BH, M, N, D = 1, 16, 2048, 128
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o = _run_flash_decoding(q, k, v)
        o_ref = _ref_attention(q, k, v)

        torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=5e-2)


@requires_gpu
class TestFlashDecodingCausal:
    """FlashDecoding with causal masking."""

    @pytest.mark.parametrize("BH,M,N,D", [
        (1, 16, 128, 128),
        (4, 16, 512, 128),
    ])
    def test_causal(self, BH, M, N, D):
        torch.manual_seed(42)
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH, N, D, dtype=torch.float16, device="cuda")

        o = _run_flash_decoding(q, k, v, causal=True)
        o_ref = _ref_attention(q, k, v, causal=True)

        torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=5e-2)


@requires_gpu
class TestFlashDecodingGQA:
    """FlashDecoding with grouped query attention."""

    @pytest.mark.parametrize("BH,BH_kv,M,N,D", [
        (8, 2, 16, 512, 128),
        (8, 2, 16, 2048, 256),
    ])
    def test_gqa(self, BH, BH_kv, M, N, D):
        torch.manual_seed(42)
        kv_group_size = BH // BH_kv
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH_kv, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH_kv, N, D, dtype=torch.float16, device="cuda")

        o = _run_flash_decoding(q, k, v, kv_group_size=kv_group_size)
        o_ref = _ref_attention(q, k, v, kv_group_size=kv_group_size)

        torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=5e-2)

    def test_gqa_causal(self):
        """GQA + causal masking."""
        torch.manual_seed(42)
        BH, BH_kv, M, N, D = 8, 2, 16, 512, 128
        kv_group_size = BH // BH_kv
        q = torch.randn(BH, M, D, dtype=torch.float16, device="cuda")
        k = torch.randn(BH_kv, N, D, dtype=torch.float16, device="cuda")
        v = torch.randn(BH_kv, N, D, dtype=torch.float16, device="cuda")

        o = _run_flash_decoding(q, k, v, causal=True, kv_group_size=kv_group_size)
        o_ref = _ref_attention(q, k, v, causal=True, kv_group_size=kv_group_size)

        torch.testing.assert_close(o, o_ref, atol=5e-2, rtol=5e-2)
