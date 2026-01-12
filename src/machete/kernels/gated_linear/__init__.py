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
        self._kernel = GatedLinearSM80(dtype, act_type)

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._kernel(a, b)


def geglu_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Direct instantiation, relying on Megakernel global cache for performance
    return GatedLinear(a.dtype, "gelu")(a, b)


def swiglu_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return GatedLinear(a.dtype, "silu")(a, b)


def reglu_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return GatedLinear(a.dtype, "relu")(a, b)
