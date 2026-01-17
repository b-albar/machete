# Copyright (c) 2025, Machete Authors
import torch
from .sm80 import GatedLinearSM80


def get_gpu_capability():
    if not torch.cuda.is_available():
        return 0, 0
    return torch.cuda.get_device_capability()


class GatedLinear:
    def __init__(self, dtype: torch.dtype, act_type: str = "gelu"):
        self.dtype = dtype
        self.act_type = act_type
        # Cache kernels per (n_rows, n_cols) since shapes are baked into kernel
        self._kernel_cache: dict[tuple[int, int], GatedLinearSM80] = {}

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ori_shape = a.shape
        n_cols = ori_shape[-1]
        a_flat = a.view(-1, n_cols)
        b_flat = b.view(-1, n_cols)
        n_rows = a_flat.shape[0]

        # Get or create kernel for this shape
        key = (n_rows, n_cols)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = GatedLinearSM80(self.dtype, self.act_type, n_rows, n_cols)

        c_flat = self._kernel_cache[key](a_flat, b_flat)
        return c_flat.view(*ori_shape)


def geglu_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Direct instantiation, relying on Megakernel global cache for performance
    return GatedLinear(a.dtype, "gelu")(a, b)


def swiglu_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return GatedLinear(a.dtype, "silu")(a, b)


def reglu_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return GatedLinear(a.dtype, "relu")(a, b)
