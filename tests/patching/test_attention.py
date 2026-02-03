# Copyright (c) 2025, Machete Authors
"""Tests for MacheteAttention using flash-attn-cute."""

import math

import pytest
import torch
import torch.nn.functional as F

from machete.patching.ops.attention import (
    MacheteAttention,
    flash_attention,
    HAS_FLASH_ATTN_CUTE,
)


def is_cuda_available():
    return torch.cuda.is_available()


def attention_pytorch_ref(query, key, value, causal=True):
    """PyTorch reference attention implementation.

    Args:
        query: (B, S_q, H, D)
        key: (B, S_k, H_kv, D)
        value: (B, S_k, H_kv, D)
        causal: Whether to apply causal masking

    Returns:
        Output tensor of shape (B, S_q, H, D)
    """
    b, s_q, h, d = query.shape
    _, s_k, h_kv, _ = key.shape

    # Transpose to (B, H, S, D)
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    # Expand K, V for GQA if needed
    if h_kv != h:
        n_rep = h // h_kv
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    # Use PyTorch's scaled_dot_product_attention
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    # Transpose back to (B, S, H, D)
    return out.transpose(1, 2)


# =============================================================================
# Unit Tests - MacheteAttention Module
# =============================================================================


class TestMacheteAttentionCPU:
    """Test MacheteAttention on CPU (always uses SDPA fallback)."""

    def test_forward_shape(self):
        """MacheteAttention produces correct output shape."""
        embed_dim, num_heads = 64, 4
        batch_size, seq_len = 2, 16

        attn = MacheteAttention(embed_dim, num_heads)
        x = torch.randn(batch_size, seq_len, embed_dim)

        out, _, _ = attn(x)
        assert out.shape == (batch_size, seq_len, embed_dim)

    def test_forward_matches_reference(self):
        """MacheteAttention forward matches reference on CPU."""
        torch.manual_seed(42)
        embed_dim, num_heads = 64, 4
        batch_size, seq_len = 2, 16

        attn = MacheteAttention(embed_dim, num_heads, bias=False, causal=True)
        x = torch.randn(batch_size, seq_len, embed_dim)

        out, _, _ = attn(x)

        # Manually compute reference
        q = attn.q_proj(x).view(batch_size, seq_len, num_heads, embed_dim // num_heads)
        k = attn.k_proj(x).view(batch_size, seq_len, num_heads, embed_dim // num_heads)
        v = attn.v_proj(x).view(batch_size, seq_len, num_heads, embed_dim // num_heads)

        attn_out = attention_pytorch_ref(q, k, v, causal=True)
        attn_out = attn_out.reshape(batch_size, seq_len, -1)
        out_ref = attn.o_proj(attn_out)

        assert torch.allclose(out, out_ref, atol=1e-5)

    def test_gqa(self):
        """MacheteAttention with grouped query attention."""
        embed_dim, num_heads, num_kv_heads = 64, 8, 2
        batch_size, seq_len = 2, 16

        attn = MacheteAttention(embed_dim, num_heads, num_kv_heads=num_kv_heads)
        x = torch.randn(batch_size, seq_len, embed_dim)

        out, _, _ = attn(x)
        assert out.shape == (batch_size, seq_len, embed_dim)

    def test_non_causal(self):
        """MacheteAttention without causal masking."""
        embed_dim, num_heads = 64, 4

        attn = MacheteAttention(embed_dim, num_heads, causal=False)
        x = torch.randn(2, 16, embed_dim)

        out, _, _ = attn(x)
        assert out.shape == (2, 16, embed_dim)

    def test_kv_cache(self):
        """MacheteAttention with KV cache."""
        embed_dim, num_heads = 64, 4
        head_dim = embed_dim // num_heads

        attn = MacheteAttention(embed_dim, num_heads)

        # First forward pass
        x1 = torch.randn(1, 10, embed_dim)
        out1, _, cache1 = attn(x1, use_cache=True)

        assert cache1 is not None
        past_key, past_value = cache1
        assert past_key.shape == (1, 10, num_heads, head_dim)

        # Second forward pass with cache
        x2 = torch.randn(1, 1, embed_dim)
        out2, _, cache2 = attn(x2, past_key_value=cache1, use_cache=True)

        assert out2.shape == (1, 1, embed_dim)
        past_key2, past_value2 = cache2
        assert past_key2.shape == (1, 11, num_heads, head_dim)  # 10 + 1


@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
class TestMacheteAttentionGPU:
    """Test MacheteAttention on GPU."""

    @pytest.mark.parametrize("embed_dim,num_heads", [
        (64, 4),
        (128, 8),
        (256, 16),
    ])
    def test_forward_shape(self, embed_dim, num_heads):
        """MacheteAttention produces correct output shape on GPU."""
        batch_size, seq_len = 2, 16

        # flash-attn-cute requires float16 or bfloat16
        attn = MacheteAttention(embed_dim, num_heads).cuda().half()
        x = torch.randn(batch_size, seq_len, embed_dim, device="cuda", dtype=torch.float16)

        try:
            out, _, _ = attn(x)
            assert out.shape == (batch_size, seq_len, embed_dim)
        except Exception as e:
            if "OpError" in type(e).__name__ or "arch" in str(e).lower():
                pytest.skip(f"flash-attn-cute unsupported on this GPU: {e}")
            raise

    @pytest.mark.skipif(not HAS_FLASH_ATTN_CUTE, reason="flash-attn-cute not available")
    def test_forward_matches_reference(self):
        """MacheteAttention forward matches reference on GPU."""
        torch.manual_seed(42)
        embed_dim, num_heads = 64, 4
        batch_size, seq_len = 2, 16

        # flash-attn-cute requires float16 or bfloat16
        attn = MacheteAttention(embed_dim, num_heads, bias=False).cuda().half()
        x = torch.randn(batch_size, seq_len, embed_dim, device="cuda", dtype=torch.float16)

        try:
            out, _, _ = attn(x)

            # Compute reference
            q = attn.q_proj(x).view(batch_size, seq_len, num_heads, embed_dim // num_heads)
            k = attn.k_proj(x).view(batch_size, seq_len, num_heads, embed_dim // num_heads)
            v = attn.v_proj(x).view(batch_size, seq_len, num_heads, embed_dim // num_heads)

            attn_out = attention_pytorch_ref(q.contiguous(), k.contiguous(), v.contiguous())
            attn_out = attn_out.reshape(batch_size, seq_len, -1)
            out_ref = attn.o_proj(attn_out)

            assert torch.allclose(out, out_ref, atol=1e-2, rtol=1e-2)
        except Exception as e:
            if "OpError" in type(e).__name__ or "arch" in str(e).lower():
                pytest.skip(f"flash-attn-cute unsupported on this GPU: {e}")
            raise

    def test_backward_gradient(self):
        """MacheteAttention supports autograd backward."""
        torch.manual_seed(42)
        embed_dim, num_heads = 64, 4

        # flash-attn-cute requires float16 or bfloat16
        attn = MacheteAttention(embed_dim, num_heads, bias=False).cuda().half()
        x = torch.randn(2, 16, embed_dim, device="cuda", dtype=torch.float16, requires_grad=True)

        try:
            out, _, _ = attn(x)
            loss = out.sum()
            loss.backward()

            assert x.grad is not None
            assert x.grad.shape == x.shape
        except Exception as e:
            if "OpError" in type(e).__name__ or "arch" in str(e).lower():
                pytest.skip(f"flash-attn-cute unsupported on this GPU: {e}")
            raise


# =============================================================================
# Unit Tests - Functional API
# =============================================================================


class TestFlashAttentionFunctional:
    """Test flash_attention functional API."""

    def test_flash_attention_cpu(self):
        """flash_attention works on CPU."""
        torch.manual_seed(42)
        b, s, h, d = 2, 16, 4, 64

        q = torch.randn(b, s, h, d)
        k = torch.randn(b, s, h, d)
        v = torch.randn(b, s, h, d)

        out = flash_attention(q, k, v, causal=True)
        out_ref = attention_pytorch_ref(q, k, v, causal=True)

        assert torch.allclose(out, out_ref, atol=1e-5)

    @pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
    @pytest.mark.skipif(not HAS_FLASH_ATTN_CUTE, reason="flash-attn-cute not available")
    def test_flash_attention_gpu(self):
        """flash_attention uses flash-attn-cute on GPU."""
        torch.manual_seed(42)
        b, s, h, d = 2, 16, 4, 64

        # flash-attn-cute requires float16 or bfloat16
        q = torch.randn(b, s, h, d, device="cuda", dtype=torch.float16)
        k = torch.randn(b, s, h, d, device="cuda", dtype=torch.float16)
        v = torch.randn(b, s, h, d, device="cuda", dtype=torch.float16)

        try:
            out = flash_attention(q, k, v, causal=True)
            out_ref = attention_pytorch_ref(q, k, v, causal=True)

            assert torch.allclose(out, out_ref, atol=1e-2, rtol=1e-2)
        except Exception as e:
            if "OpError" in type(e).__name__ or "arch" in str(e).lower():
                pytest.skip(f"flash-attn-cute unsupported on this GPU: {e}")
            raise

    def test_gqa_functional(self):
        """flash_attention with GQA."""
        b, s, h, h_kv, d = 2, 16, 8, 2, 64

        q = torch.randn(b, s, h, d)
        k = torch.randn(b, s, h_kv, d)
        v = torch.randn(b, s, h_kv, d)

        out = flash_attention(q, k, v, causal=True)
        assert out.shape == (b, s, h, d)


# =============================================================================
# Edge Cases
# =============================================================================


class TestAttentionEdgeCases:
    """Test edge cases for attention modules."""

    def test_single_token(self):
        """Attention with single token."""
        attn = MacheteAttention(64, 4)
        x = torch.randn(1, 1, 64)

        out, _, _ = attn(x)
        assert out.shape == (1, 1, 64)

    def test_large_batch(self):
        """Attention with large batch size."""
        attn = MacheteAttention(64, 4)
        x = torch.randn(32, 16, 64)

        out, _, _ = attn(x)
        assert out.shape == (32, 16, 64)

    def test_long_sequence(self):
        """Attention with long sequence."""
        attn = MacheteAttention(64, 4)
        x = torch.randn(1, 512, 64)

        out, _, _ = attn(x)
        assert out.shape == (1, 512, 64)

    def test_custom_head_dim(self):
        """Attention with custom head dimension."""
        attn = MacheteAttention(embed_dim=128, num_heads=4, head_dim=64)
        x = torch.randn(2, 16, 128)

        out, _, _ = attn(x)
        assert out.shape == (2, 16, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
