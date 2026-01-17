# Copyright (c) 2025, Machete Authors
import torch
from functools import lru_cache
from .sm80 import GatedMLPSM80


def get_gpu_capability():
    if not torch.cuda.is_available():
        return 0, 0
    return torch.cuda.get_device_capability()


# Global LRU cache for kernels to limit memory usage
# Key: (dtype, act_type, m_dim, k_dim, n_dim)
@lru_cache(maxsize=8)
def _get_kernel(dtype: torch.dtype, act_type: str, m_dim: int, k_dim: int, n_dim: int) -> GatedMLPSM80:
    return GatedMLPSM80(dtype, act_type, m_dim, k_dim, n_dim)


def clear_kernel_cache():
    """Clear the kernel cache to free GPU memory."""
    _get_kernel.cache_clear()


class GatedMLP:
    def __init__(self, dtype: torch.dtype, act_type: str = "silu"):
        self.dtype = dtype
        self.act_type = act_type

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        m_dim, k_dim = x_2d.shape
        n2_dim = weight.shape[1]
        n_dim = n2_dim // 2

        # Get kernel from global LRU cache
        kernel = _get_kernel(self.dtype, self.act_type, m_dim, k_dim, n_dim)

        out = kernel(x_2d, weight)
        return out.view(*orig_shape[:-1], n_dim)

    @staticmethod
    def apply(x: torch.Tensor, weight: torch.Tensor, act_type: str = "silu") -> torch.Tensor:
        """Legacy interface for backward compatibility."""
        major, _ = get_gpu_capability()

        if major >= 9:
            # SM90+ can use either implementation
            from .sm90 import gated_mlp_sm90
            sm90_act_map = {"silu": "swiglu", "gelu": "geglu"}
            return gated_mlp_sm90(x, weight, act_type=sm90_act_map.get(act_type, act_type))

        # SM80+ uses new L/C/S implementation
        return GatedMLP(x.dtype, act_type)(x, weight)


def gated_mlp_func(x: torch.Tensor, weight: torch.Tensor, act_type: str = "silu") -> torch.Tensor:
    """Convenience function for gated MLP forward pass."""
    return GatedMLP(x.dtype, act_type)(x, weight)


def swiglu_mlp_func(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """SwiGLU: SiLU(x @ W_gate) * (x @ W_up)"""
    return GatedMLP(x.dtype, "silu")(x, weight)


def geglu_mlp_func(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """GeGLU: GELU(x @ W_gate) * (x @ W_up)"""
    return GatedMLP(x.dtype, "gelu")(x, weight)
