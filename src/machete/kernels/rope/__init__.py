# Copyright (c) 2025, Machete Authors
import torch
from .sm80 import RopeSM80


def get_gpu_capability():
    if not torch.cuda.is_available():
        return 0, 0
    return torch.cuda.get_device_capability()


class Rope:
    def __init__(self, dtype: torch.dtype, head_dim: int, n_heads: int | None = None):
        self.dtype = dtype
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.major, self.minor = get_gpu_capability()
        self._kernel_cache: dict[int, RopeSM80] = {}

    def __call__(self, q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # Get n_heads from tensor shape
        n_heads = q.shape[2] if self.n_heads is None else self.n_heads

        # Cache kernel per n_heads
        if n_heads not in self._kernel_cache:
            self._kernel_cache[n_heads] = RopeSM80(self.dtype, self.head_dim, n_heads)

        return self._kernel_cache[n_heads](q, cos, sin)
