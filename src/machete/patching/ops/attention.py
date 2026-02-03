# Copyright (c) 2025, Machete Authors
"""Optimized Attention using flash-attn-cute."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from flash_attn.cute import flash_attn_func

    HAS_FLASH_ATTN_CUTE = True
except ImportError:
    HAS_FLASH_ATTN_CUTE = False
    flash_attn_func = None


class MacheteAttention(nn.Module):
    """Multi-head attention using flash-attn-cute.

    This is a standalone attention module that uses flash-attn-cute for
    efficient attention computation. It handles Q, K, V projections and
    output projection internally.

    Automatically falls back to standard scaled dot-product attention when:
    - flash-attn-cute is not available
    - Device is not CUDA

    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for GQA). Default: num_heads
        head_dim: Dimension per head. Default: embed_dim // num_heads
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in projections (default: False)
        causal: Whether to use causal masking (default: True)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or embed_dim // num_heads
        self.dropout = dropout
        self.causal = causal

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=bias, **factory_kwargs)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor, Tensor]]]:
        """Forward pass.

        Args:
            hidden_states: Input tensor of shape (B, S, embed_dim)
            attention_mask: Optional attention mask (not used with flash attention)
            position_ids: Optional position IDs (for RoPE, applied externally)
            past_key_value: Optional cached key/value tensors
            output_attentions: Whether to return attention weights (not supported)
            use_cache: Whether to return cached key/value tensors

        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        bsz, seq_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to (B, S, H, D) for flash attention
        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        # Handle KV cache
        if past_key_value is not None:
            # past_key_value is (past_key, past_value) each of shape (B, S_past, H, D)
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=1)
            value_states = torch.cat([past_value, value_states], dim=1)

        if use_cache:
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None

        # Make contiguous for flash attention
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        # Determine if causal
        causal = self.causal and seq_len > 1

        # Compute attention
        # flash-attn-cute requires float16 or bfloat16
        use_flash = (
            HAS_FLASH_ATTN_CUTE
            and hidden_states.is_cuda
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
        )
        if use_flash:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                causal=causal,
            )
            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]
        else:
            # Fallback to scaled dot-product attention
            attn_output = self._sdpa_attention(
                query_states, key_states, value_states, causal=causal
            )

        # Reshape and project output
        attn_output = attn_output.reshape(bsz, seq_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None, present_key_value

    def _sdpa_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        causal: bool = True,
    ) -> Tensor:
        """Scaled dot-product attention fallback.

        Args:
            query: (B, S_q, H, D)
            key: (B, S_k, H_kv, D)
            value: (B, S_k, H_kv, D)
            causal: Whether to apply causal masking

        Returns:
            Output tensor of shape (B, S_q, H, D)
        """
        # Transpose to (B, H, S, D) for SDPA
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Expand K, V for GQA if needed
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(n_rep, dim=1)
            value = value.repeat_interleave(n_rep, dim=1)

        # Use PyTorch's scaled_dot_product_attention
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )

        # Transpose back to (B, S, H, D)
        return attn_output.transpose(1, 2)


def flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
) -> Tensor:
    """Functional API for flash attention.

    Args:
        query: Query tensor of shape (B, S_q, H, D)
        key: Key tensor of shape (B, S_k, H_kv, D)
        value: Value tensor of shape (B, S_k, H_kv, D)
        causal: Whether to apply causal masking
        softmax_scale: Optional softmax scale. Default: 1/sqrt(D)

    Returns:
        Output tensor of shape (B, S_q, H, D)
    """
    # flash-attn-cute requires float16 or bfloat16
    use_flash = (
        HAS_FLASH_ATTN_CUTE
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
    )
    if use_flash:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        attn_output = flash_attn_func(
            query,
            key,
            value,
            causal=causal,
        )
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        return attn_output
    else:
        # Fallback to SDPA
        b, s_q, h, d = query.shape
        _, s_k, h_kv, _ = key.shape

        # Transpose to (B, H, S, D)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Expand K, V for GQA if needed
        if h_kv != h:
            n_rep = h // h_kv
            key = key.repeat_interleave(n_rep, dim=1)
            value = value.repeat_interleave(n_rep, dim=1)

        scale = softmax_scale or (1.0 / math.sqrt(d))

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=scale,
        )

        # Transpose back to (B, S, H, D)
        return attn_output.transpose(1, 2)
