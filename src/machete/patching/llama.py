# Copyright (c) 2025, Machete Authors
"""Llama model patching using flash-attn-cute and quack."""

import types
from typing import Optional, Tuple

import torch

try:
    from flash_attn.cute import flash_attn_func
except ImportError as e:
    print(f"Failed to import flash_attn_cute: {e}")
    flash_attn_func = None


# Module class names for Llama models
ATTENTION_CLASSES = ("LlamaAttention", "LlamaSdpaAttention", "LlamaFlashAttention2")
RMSNORM_CLASSES = ("LlamaRMSNorm",)


def make_attention_forward():
    """Create optimized forward for LlamaAttention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Use existing projection layers (preserves LoRA compatibility)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to (batch, seq, heads, head_dim)
        num_heads = getattr(self, "num_heads", None) or self.config.num_attention_heads
        num_kv_heads = getattr(self, "num_key_value_heads", None) or self.config.num_key_value_heads
        query_states = query_states.view(bsz, q_len, num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, num_kv_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, num_kv_heads, self.head_dim)

        # Apply rotary embeddings if position_embeddings provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Transpose for apply_rotary_pos_emb which expects (bsz, num_heads, seq_len, head_dim)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

            # Import here to avoid circular imports
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # Transpose back to (bsz, seq_len, num_heads, head_dim) for flash attention
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

        # Handle KV cache
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states.transpose(1, 2), value_states.transpose(1, 2), self.layer_idx, cache_position
            )
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

        # Prepare for flash attention - need contiguous tensors
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

        # Determine causal mask
        causal = self.is_causal and q_len > 1

        # Use flash-attn-cute
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            causal=causal,
        )
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]

        # Reshape output
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        # Output projection (preserves LoRA compatibility)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    return forward


def patch_attention(module):
    """Patch Llama attention module with optimized forward.

    No-op if flash-attn-cute is not available.
    """
    if flash_attn_func is None:
        return  # flash-attn-cute not available, skip patching

    if hasattr(module, "_machete_original_forward"):
        return  # Already patched

    module._machete_original_forward = module.forward
    module.forward = types.MethodType(make_attention_forward(), module)


def unpatch_attention(module):
    """Restore original Llama attention forward."""
    if hasattr(module, "_machete_original_forward"):
        module.forward = module._machete_original_forward
        del module._machete_original_forward
