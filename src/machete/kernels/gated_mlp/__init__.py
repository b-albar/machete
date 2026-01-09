# Copyright (c) 2025, Machete Authors
import torch
from .base import GatedMLP as GatedMLPBase


def get_gpu_capability():
    if not torch.cuda.is_available():
        return 0, 0
    return torch.cuda.get_device_capability()


class GatedMLP:
    @staticmethod
    def apply(x: torch.Tensor, weight: torch.Tensor, act_type: str = "silu") -> torch.Tensor:
        # Use Triton implementation for GPU if available, else fallback
        if x.is_cuda and weight.is_cuda:
            cc = get_gpu_capability()
            if cc >= (9, 0):
                # Use SM90 implementation
                from .sm90 import gated_mlp_sm90

                return gated_mlp_sm90(x, weight)

            from .triton_impl import gated_mlp_triton

            return gated_mlp_triton(x, weight, activation=act_type)
        return GatedMLPBase.apply(x, weight, act_type)


def gated_mlp_func(x: torch.Tensor, weight: torch.Tensor, act_type: str = "silu") -> torch.Tensor:
    return GatedMLP.apply(x, weight, act_type)
