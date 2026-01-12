# Copyright (c) 2025, Machete Authors
import torch
from .sm80 import GatedLinear as GatedLinearAutograd


def get_gpu_capability():
    if not torch.cuda.is_available():
        return 0, 0
    return torch.cuda.get_device_capability()


class GatedLinear:
    @staticmethod
    def apply(a: torch.Tensor, b: torch.Tensor, act_type: str = "gelu") -> torch.Tensor:
        return GatedLinearAutograd.apply(a, b, act_type)


def geglu_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return GatedLinear.apply(a, b, "gelu")


def swiglu_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return GatedLinear.apply(a, b, "silu")


def reglu_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return GatedLinear.apply(a, b, "relu")
