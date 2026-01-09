# Copyright (c) 2025, Machete Authors
import torch
from .base import GatedLinear as GatedLinearBase


def get_gpu_capability():
    if not torch.cuda.is_available():
        return 0, 0
    return torch.cuda.get_device_capability()


class GatedLinear:
    @staticmethod
    def apply(a: torch.Tensor, b: torch.Tensor, act_type: str = "gelu") -> torch.Tensor:
        major, _ = get_gpu_capability()
        if major == 9 or major == 10:
            from .sm90 import GatedLinearSM90

            return GatedLinearSM90.apply(a, b, act_type)

        return GatedLinearBase.apply(a, b, act_type)


def geglu_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return GatedLinear.apply(a, b, "gelu")


def swiglu_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return GatedLinear.apply(a, b, "silu")


def reglu_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return GatedLinear.apply(a, b, "relu")
