# Copyright (c) 2025, Machete Authors
"""Rotary Position Embedding (RoPE) using machete megakernel."""

import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

try:
    from machete.kernels.rope import RopeAutogradOp
    from machete.megakernel.functional import megakernel_apply

    HAS_MEGAKERNEL_ROPE = True
except ImportError:
    HAS_MEGAKERNEL_ROPE = False
    RopeAutogradOp = None
    megakernel_apply = None


class MacheteRoPE(nn.Module):
    """Rotary Position Embedding using machete megakernel.

    This module applies rotary position embeddings to query and key tensors
    using the optimized megakernel implementation.

    The rotation is applied in-place for efficiency.

    Args:
        dim: Head dimension (must be even)
        max_seq_len: Maximum sequence length for precomputing cos/sin cache
        base: Base for the frequency computation (default: 10000.0)

    Example:
        rope = MacheteRoPE(dim=64, max_seq_len=2048)
        q_rotated = rope(q, position_ids)  # q shape: (B, S, H, D)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build cos/sin cache
        self._build_cache(max_seq_len, device=device, dtype=dtype or torch.float32)

    def _build_cache(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Build cos/sin cache for positions 0 to seq_len-1."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device or self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (S, D/2)

        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Extend cache if needed."""
        if seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len, device=device, dtype=dtype)

    def forward(
        self,
        q: Tensor,
        position_ids: Optional[Tensor] = None,
        k: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Apply rotary embeddings to q (and optionally k).

        Args:
            q: Query tensor of shape (B, S, H, D)
            position_ids: Position indices of shape (B, S). If None, uses 0..S-1.
            k: Optional key tensor of shape (B, S, H_kv, D)

        Returns:
            Tuple of (q_rotated, k_rotated) or (q_rotated, None) if k is None.
        """
        b, s, h, d = q.shape
        device = q.device
        dtype = q.dtype

        # Ensure cache is large enough
        self._update_cache(s, device, dtype)

        # Get cos/sin for these positions
        if position_ids is None:
            cos = self.cos_cached[:s]  # (S, D/2)
            sin = self.sin_cached[:s]
        else:
            # Gather cos/sin for specific positions
            cos = self.cos_cached[position_ids]  # (B, S, D/2)
            sin = self.sin_cached[position_ids]
            # For megakernel, we need (S, D/2) - use first batch if all same
            if cos.dim() == 3:
                cos = cos[0]  # Assume same positions across batch
                sin = sin[0]

        # Apply RoPE using megakernel if available and on CUDA
        if HAS_MEGAKERNEL_ROPE and q.is_cuda and q.dtype == torch.float32:
            q_rotated = self._apply_megakernel_rope(q, cos, sin)
            if k is not None:
                k_rotated = self._apply_megakernel_rope(k, cos, sin)
            else:
                k_rotated = None
        else:
            q_rotated = self._apply_rope_pytorch(q, cos, sin)
            if k is not None:
                k_rotated = self._apply_rope_pytorch(k, cos, sin)
            else:
                k_rotated = None

        return q_rotated, k_rotated

    def _apply_megakernel_rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply RoPE using megakernel (in-place)."""
        # megakernel expects (B, S, H, D) and modifies in place
        # Clone if x requires grad and is a leaf to avoid in-place modification error
        if x.requires_grad and x.is_leaf:
            x = x.clone()
        x = x.contiguous()
        cos = cos.contiguous()
        sin = sin.contiguous()

        # Use megakernel_apply with RopeAutogradOp
        x_rotated = megakernel_apply(RopeAutogradOp(), q=x, cos=cos, sin=sin)
        return x_rotated

    def _apply_rope_pytorch(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply RoPE using PyTorch (fallback)."""
        # x: (B, S, H, D)
        # cos, sin: (S, D/2)
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]

        # Broadcast cos/sin to match x shape
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D/2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        # Apply rotation
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin

        return torch.cat([out1, out2], dim=-1)


def apply_rope(
    q: Tensor,
    cos: Tensor,
    sin: Tensor,
    k: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Functional API for applying RoPE.

    Args:
        q: Query tensor of shape (B, S, H, D)
        cos: Cosine cache of shape (S, D/2)
        sin: Sine cache of shape (S, D/2)
        k: Optional key tensor of shape (B, S, H_kv, D)

    Returns:
        Tuple of (q_rotated, k_rotated) or (q_rotated, None) if k is None.
    """
    if HAS_MEGAKERNEL_ROPE and q.is_cuda and q.dtype == torch.float32:
        q = q.contiguous()
        cos = cos.contiguous()
        sin = sin.contiguous()
        q_rotated = megakernel_apply(RopeAutogradOp(), q=q, cos=cos, sin=sin)

        if k is not None:
            k = k.contiguous()
            k_rotated = megakernel_apply(RopeAutogradOp(), q=k, cos=cos, sin=sin)
        else:
            k_rotated = None
    else:
        # PyTorch fallback
        d = q.shape[-1]
        cos_b = cos.unsqueeze(0).unsqueeze(2)
        sin_b = sin.unsqueeze(0).unsqueeze(2)

        q1, q2 = q[..., : d // 2], q[..., d // 2 :]
        q_rotated = torch.cat([q1 * cos_b - q2 * sin_b, q2 * cos_b + q1 * sin_b], dim=-1)

        if k is not None:
            k1, k2 = k[..., : d // 2], k[..., d // 2 :]
            k_rotated = torch.cat([k1 * cos_b - k2 * sin_b, k2 * cos_b + k1 * sin_b], dim=-1)
        else:
            k_rotated = None

    return q_rotated, k_rotated
