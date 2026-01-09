# Copyright (c) 2025, Machete Authors
import torch
from .base import Rope as RopeBase


def get_gpu_capability():
    if not torch.cuda.is_available():
        return 0, 0
    return torch.cuda.get_device_capability()


class Rope:
    def __init__(self, dtype: torch.dtype, head_dim: int):
        self.dtype = dtype
        self.head_dim = head_dim
        self.major, self.minor = get_gpu_capability()
        self._base_rope = None

    def __call__(self, q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, backward: bool = False):
        if self.major in (9, 10):
            from .sm90 import RopeSM90

            return RopeSM90.apply(q, cos, sin, backward)

        # Fallback to base implementation
        if self._base_rope is None:
            self._base_rope = RopeBase(self.dtype, self.head_dim)

        return self._base_rope(q, cos, sin, backward=backward)
