# Copyright (c) 2025, Machete Authors
import torch


def get_gpu_capability():
    if not torch.cuda.is_available():
        return 0, 0
    return torch.cuda.get_device_capability()


class GatedMLP:
    @staticmethod
    def apply(x: torch.Tensor, weight: torch.Tensor, act_type: str = "silu") -> torch.Tensor:
        # Use optimized implementations for GPU if available, else fallback to Triton
        if x.is_cuda and weight.is_cuda:
            major, _ = get_gpu_capability()

            if major == 9 or major == 10:
                # For now, SM90+ uses SM90 implementation
                from .sm90 import gated_mlp_sm90

                sm90_act_map = {"silu": "swiglu", "gelu": "geglu"}
                return gated_mlp_sm90(x, weight, act_type=sm90_act_map.get(act_type, act_type))

            # Use SM80 (Ampere/Ada) implementation
            from .sm80 import gated_mlp_sm80

            return gated_mlp_sm80(x, weight, act_type=act_type)


def gated_mlp_func(x: torch.Tensor, weight: torch.Tensor, act_type: str = "silu") -> torch.Tensor:
    return GatedMLP.apply(x, weight, act_type)
