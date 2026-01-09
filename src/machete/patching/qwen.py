# Copyright (c) 2025, Machete Authors
"""Qwen model patching using flash-attn-cute."""

import types
from typing import Optional, Tuple, Dict
import torch
import transformers
from machete.patching.llama import flash_attn_func

# Module class names for Qwen models
ATTENTION_CLASSES = ("Qwen2Attention", "Qwen2SdpaAttention", "Qwen2FlashAttention2", "Qwen3Attention")
RMSNORM_CLASSES = ("Qwen2RMSNorm", "Qwen3RMSNorm")


def make_qwen_attention_forward():
    """Create optimized forward for QwenAttention (supports Qwen2 and Qwen3)."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[object] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.shape

        # Qwen3Attention uses config for num_heads check, Qwen2 might have attributes
        if hasattr(self, "num_heads"):
            num_heads = self.num_heads
            num_kv_heads = self.num_key_value_heads
        else:
            num_heads = self.config.num_attention_heads
            num_kv_heads = self.config.num_key_value_heads

        head_dim = self.head_dim

        # 1. Projections and Reshape
        # Qwen2/3 applies norm on (bsz, seqlen, num_heads, head_dim)
        query_states = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim)

        # Apply QK Norm if available
        if hasattr(self, "q_norm") and self.q_norm is not None:
            query_states = self.q_norm(query_states)
        if hasattr(self, "k_norm") and self.k_norm is not None:
            key_states = self.k_norm(key_states)

        # 2. RoPE
        # Transpose to (bsz, num_heads, seqlen, head_dim) for apply_rotary_pos_emb
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Get apply_rotary_pos_emb. Try qwen3, then qwen2, then fallback
        # Ideally we'd import this at top level but it depends on transformers version
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        # position_embeddings passed as (cos, sin) usually
        # Qwen3 signature: position_embeddings: tuple[torch.Tensor, torch.Tensor]
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 3. KV Cache
        if past_key_values is not None:
            # Qwen3 uses cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # We need to adapt based on what update expects.
            # Assuming standard DynamicCache/Cache interface

            # Note: Qwen3 passes sin/cos to update!?
            cache_kwargs = {}
            if cache_position is not None:
                cache_kwargs["cache_position"] = cache_position
            if position_embeddings is not None:
                cos, sin = position_embeddings
                cache_kwargs["sin"] = sin
                cache_kwargs["cos"] = cos

            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 4. Flash Attention using flash_attn_func
        # flash_attn_func expects (bsz, seqlen, num_heads, head_dim)
        query_states = query_states.transpose(1, 2).contiguous()
        key_states = key_states.transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()

        # Qwen3 logic for causal: "input_shape = hidden_states.shape[:-1]; is_causal = True"
        # We can pass causal=True for generation/prefill usually

        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)

        # 5. Output Projection
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    return forward


def patch_attention(module):
    """Patch Qwen attention module with optimized forward.

    No-op if flash-attn-cute is not available.
    """
    if flash_attn_func is None:
        return  # flash-attn-cute not available, skip patching

    if hasattr(module, "_machete_original_forward"):
        return  # Already patched

    module._machete_original_forward = module.forward
    module.forward = types.MethodType(make_qwen_attention_forward(), module)


def unpatch_attention(module):
    """Restore original Qwen attention forward."""
    if hasattr(module, "_machete_original_forward"):
        module.forward = module._machete_original_forward
        del module._machete_original_forward
