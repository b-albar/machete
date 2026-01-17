# Copyright (c) 2025, Machete Authors
import torch
from functools import lru_cache
from .sm80 import GatedLinearSM80


def get_gpu_capability():
    if not torch.cuda.is_available():
        return 0, 0
    return torch.cuda.get_device_capability()


# Global LRU cache for kernels to limit memory usage
# Key: (dtype, act_type, n_rows, n_cols)
@lru_cache(maxsize=8)
def _get_kernel(dtype: torch.dtype, act_type: str, n_rows: int, n_cols: int) -> GatedLinearSM80:
    return GatedLinearSM80(dtype, act_type, n_rows, n_cols)


def clear_kernel_cache():
    """Clear the kernel cache to free GPU memory."""
    _get_kernel.cache_clear()


class GatedLinear:
    def __init__(self, dtype: torch.dtype, act_type: str = "gelu"):
        self.dtype = dtype
        self.act_type = act_type

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ori_shape = a.shape
        n_cols = ori_shape[-1]
        a_flat = a.view(-1, n_cols)
        b_flat = b.view(-1, n_cols)
        n_rows = a_flat.shape[0]

        # Get kernel from global LRU cache
        kernel = _get_kernel(self.dtype, self.act_type, n_rows, n_cols)

        c_flat = kernel(a_flat, b_flat)
        return c_flat.view(*ori_shape)


def geglu_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Direct instantiation, relying on Megakernel global cache for performance
    return GatedLinear(a.dtype, "gelu")(a, b)


def swiglu_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return GatedLinear(a.dtype, "silu")(a, b)


def reglu_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return GatedLinear(a.dtype, "relu")(a, b)
