# Copyright (c) 2025, Machete Authors
import torch
from .sm80 import RopeSM80

# Global kernel cache for direct-call function
# Key: (dtype, head_dim, n_heads)
_KERNEL_CACHE: dict[tuple, RopeSM80] = {}


def get_gpu_capability():
    if not torch.cuda.is_available():
        return 0, 0
    return torch.cuda.get_device_capability()


def rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE (Rotary Position Embedding) to query tensor.

    This is a direct-call function that auto-detects dtype and dimensions
    from the input tensor. Can be used like Triton kernels without
    pre-instantiation.

    Args:
        q: Query tensor of shape (B, S, H, D)
        cos: Cosine values of shape (S, D)
        sin: Sine values of shape (S, D)

    Returns:
        Rotated query tensor (same shape as input, modified in-place)

    Example:
        >>> q = torch.randn(2, 1024, 32, 128, device='cuda', dtype=torch.float16)
        >>> cos = torch.randn(1024, 128, device='cuda', dtype=torch.float16)
        >>> sin = torch.randn(1024, 128, device='cuda', dtype=torch.float16)
        >>> out = rope(q, cos, sin)
    """
    dtype = q.dtype
    _, _, n_heads, head_dim = q.shape

    cache_key = (dtype, head_dim, n_heads)
    if cache_key not in _KERNEL_CACHE:
        _KERNEL_CACHE[cache_key] = RopeSM80(dtype, head_dim, n_heads)

    kernel = _KERNEL_CACHE[cache_key]
    return kernel(q, cos, sin)


class Rope:
    """RoPE (Rotary Position Embedding) kernel.

    Supports both forward and backward passes with autograd integration
    via the SingleKernel/Megakernel infrastructure.

    Args:
        dtype: PyTorch dtype (float16, bfloat16, float32)
        head_dim: Dimension of each attention head
        n_heads: Optional number of attention heads (inferred from tensor if None)
    """

    def __init__(self, dtype: torch.dtype, head_dim: int, n_heads: int | None = None):
        self.dtype = dtype
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.major, self.minor = get_gpu_capability()
        self._kernel_cache: dict[int, RopeSM80] = {}

    def _get_kernel(self, n_heads: int) -> RopeSM80:
        """Get or create kernel for given number of heads."""
        if n_heads not in self._kernel_cache:
            self._kernel_cache[n_heads] = RopeSM80(self.dtype, self.head_dim, n_heads)
        return self._kernel_cache[n_heads]

    def __call__(self, q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to query tensor.

        Args:
            q: Query tensor of shape (B, S, H, D)
            cos: Cosine values of shape (S, D)
            sin: Sine values of shape (S, D)

        Returns:
            Rotated query tensor (same shape as input)
        """
        n_heads = q.shape[2] if self.n_heads is None else self.n_heads
        kernel = self._get_kernel(n_heads)
        # RopeSM80 inherits from SingleKernel which handles autograd via apply_autograd
        return kernel(q, cos, sin)
