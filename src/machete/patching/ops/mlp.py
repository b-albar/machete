# Copyright (c) 2025, Machete Authors
"""Optimized MLP using quack fused kernels."""

import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from quack.mlp import mlp_func, MLP as QuackMLP
    from quack.cute_dsl_utils import get_device_capacity

    HAS_QUACK_MLP = True
except ImportError:
    HAS_QUACK_MLP = False
    mlp_func = None
    QuackMLP = None
    get_device_capacity = None


def _check_compute_capability(device: torch.device) -> bool:
    """Check if device supports quack MLP (SM90+)."""
    if not HAS_QUACK_MLP:
        return False
    if device.type != "cuda":
        return False
    try:
        cap = get_device_capacity(device)
        return cap[0] >= 9  # SM90 or higher
    except Exception:
        return False


class MacheteMLP(nn.Module):
    """Optimized MLP using quack fused GEMM + activation kernels.

    This is a standard 2-layer MLP (fc1 -> activation -> fc2) that uses
    quack's fused linear_act and act_linear operations for efficiency.

    Automatically falls back to standard PyTorch when:
    - quack is not available
    - Device is not CUDA
    - GPU compute capability < SM90
    - Dimensions are not divisible by 8
    - Bias is used (quack MLP works best without bias)

    Args:
        in_features: Input dimension
        hidden_features: Hidden dimension (default: 4 * in_features)
        out_features: Output dimension (default: in_features)
        activation: Activation function name (default: "gelu")
        bias: Whether to use bias (default: False for optimal performance)
        fuse_grad_accum: Whether to fuse gradient accumulation
        tuned: Whether to use autotuned kernels
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: str = "gelu",
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        fuse_grad_accum: bool = False,
        tuned: bool = True,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features or 4 * in_features
        self.out_features = out_features or in_features
        self.activation = activation
        self.fuse_grad_accum = fuse_grad_accum
        self.tuned = tuned

        self.fc1 = nn.Linear(in_features, self.hidden_features, bias=bias, **factory_kwargs)
        self.fc2 = nn.Linear(self.hidden_features, self.out_features, bias=bias, **factory_kwargs)

        self._supports_quack: Optional[bool] = None

    def _can_use_quack(self, input: Tensor) -> bool:
        """Check if we can use quack for this forward pass."""
        if self._supports_quack is None:
            self._supports_quack = _check_compute_capability(input.device)

        return (
            self._supports_quack
            and input.is_cuda
            and self.fc1.bias is None
            and self.fc2.bias is None
            and input.stride(-1) == 1
            and self.fc1.in_features % 8 == 0
            and self.fc1.out_features % 8 == 0
            and self.fc2.out_features % 8 == 0
        )

    def forward(self, input: Tensor) -> Tensor:
        if self._can_use_quack(input):
            return mlp_func(
                input,
                self.fc1.weight,
                self.fc2.weight,
                activation=self.activation,
                fuse_grad_accum=self.fuse_grad_accum,
                tuned=self.tuned,
            )
        # Fallback to standard implementation
        x = self.fc1(input)
        if self.activation == "gelu":
            x = F.gelu(x)
        elif self.activation == "silu" or self.activation == "swish":
            x = F.silu(x)
        elif self.activation == "relu":
            x = F.relu(x)
        else:
            # Default to gelu
            x = F.gelu(x)
        return self.fc2(x)


class MacheteGatedMLP(nn.Module):
    """Optimized Gated MLP (SwiGLU-style) using quack.

    This implements the gated MLP used in models like LLaMA:
        output = down_proj(act(gate_proj(x)) * up_proj(x))

    Args:
        in_features: Input dimension
        hidden_features: Hidden dimension
        out_features: Output dimension (default: in_features)
        activation: Activation function name (default: "silu")
        bias: Whether to use bias (default: False)
        fuse_grad_accum: Whether to fuse gradient accumulation
        tuned: Whether to use autotuned kernels
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        activation: str = "silu",
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        fuse_grad_accum: bool = False,
        tuned: bool = True,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features or in_features
        self.activation = activation
        self.fuse_grad_accum = fuse_grad_accum
        self.tuned = tuned

        self.gate_proj = nn.Linear(in_features, hidden_features, bias=bias, **factory_kwargs)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=bias, **factory_kwargs)
        self.down_proj = nn.Linear(hidden_features, self.out_features, bias=bias, **factory_kwargs)

        self._supports_quack: Optional[bool] = None

    def _can_use_quack(self, input: Tensor) -> bool:
        """Check if we can use quack for this forward pass."""
        if self._supports_quack is None:
            self._supports_quack = _check_compute_capability(input.device)

        return (
            self._supports_quack
            and input.is_cuda
            and self.gate_proj.bias is None
            and self.up_proj.bias is None
            and self.down_proj.bias is None
            and input.stride(-1) == 1
            and self.in_features % 8 == 0
            and self.hidden_features % 8 == 0
            and self.out_features % 8 == 0
        )

    def forward(self, input: Tensor) -> Tensor:
        # Gated MLP doesn't directly map to quack's mlp_func,
        # so we use individual linear operations
        if self._can_use_quack(input):
            try:
                from quack.linear import linear_func

                gate = linear_func(input, self.gate_proj.weight, fuse_grad_accum=self.fuse_grad_accum)
                up = linear_func(input, self.up_proj.weight, fuse_grad_accum=self.fuse_grad_accum)

                if self.activation == "silu" or self.activation == "swish":
                    gate = F.silu(gate)
                elif self.activation == "gelu":
                    gate = F.gelu(gate)
                else:
                    gate = F.silu(gate)

                hidden = gate * up
                return linear_func(hidden, self.down_proj.weight, fuse_grad_accum=self.fuse_grad_accum)
            except Exception:
                pass  # Fall through to PyTorch fallback

        # Fallback to standard implementation
        gate = self.gate_proj(input)
        if self.activation == "silu" or self.activation == "swish":
            gate = F.silu(gate)
        elif self.activation == "gelu":
            gate = F.gelu(gate)
        else:
            gate = F.silu(gate)

        up = self.up_proj(input)
        hidden = gate * up
        return self.down_proj(hidden)


def make_gated_mlp_forward(
    gate_proj_attr: str = "gate_proj",
    up_proj_attr: str = "up_proj",
    down_proj_attr: str = "down_proj",
    activation: str = "silu",
    fuse_grad_accum: bool = False,
):
    """Create optimized forward for gated MLP modules (LLaMA-style).

    Args:
        gate_proj_attr: Attribute name for gate projection
        up_proj_attr: Attribute name for up projection
        down_proj_attr: Attribute name for down projection
        activation: Activation function name
        fuse_grad_accum: Whether to fuse gradient accumulation
    """

    def forward(self, input: Tensor) -> Tensor:
        # Check compute capability on first call
        if not hasattr(self, "_machete_supports_quack"):
            self._machete_supports_quack = _check_compute_capability(input.device)

        gate_proj = getattr(self, gate_proj_attr)
        up_proj = getattr(self, up_proj_attr)
        down_proj = getattr(self, down_proj_attr)

        can_use_quack = (
            self._machete_supports_quack
            and input.is_cuda
            and gate_proj.bias is None
            and up_proj.bias is None
            and down_proj.bias is None
            and input.stride(-1) == 1
            and gate_proj.in_features % 8 == 0
            and gate_proj.out_features % 8 == 0
            and down_proj.out_features % 8 == 0
        )

        if can_use_quack and HAS_QUACK_MLP:
            try:
                from quack.linear import linear_func

                gate = linear_func(input, gate_proj.weight, fuse_grad_accum=fuse_grad_accum)
                up = linear_func(input, up_proj.weight, fuse_grad_accum=fuse_grad_accum)

                if activation == "silu" or activation == "swish":
                    gate = F.silu(gate)
                elif activation == "gelu":
                    gate = F.gelu(gate)
                else:
                    gate = F.silu(gate)

                hidden = gate * up
                return linear_func(hidden, down_proj.weight, fuse_grad_accum=fuse_grad_accum)
            except Exception:
                pass  # Fall through to PyTorch fallback

        # Fallback
        gate = gate_proj(input)
        if activation == "silu" or activation == "swish":
            gate = F.silu(gate)
        elif activation == "gelu":
            gate = F.gelu(gate)
        else:
            gate = F.silu(gate)

        up = up_proj(input)
        return down_proj(gate * up)

    return forward


def patch_gated_mlp(
    module: nn.Module,
    gate_proj_attr: str = "gate_proj",
    up_proj_attr: str = "up_proj",
    down_proj_attr: str = "down_proj",
    activation: str = "silu",
    fuse_grad_accum: bool = False,
):
    """Patch a gated MLP module with optimized forward.

    No-op if quack is not available.
    """
    if not HAS_QUACK_MLP:
        return

    if hasattr(module, "_machete_original_forward"):
        return  # Already patched

    module._machete_original_forward = module.forward
    module.forward = types.MethodType(
        make_gated_mlp_forward(gate_proj_attr, up_proj_attr, down_proj_attr, activation, fuse_grad_accum),
        module,
    )


def unpatch_gated_mlp(module: nn.Module):
    """Restore original gated MLP forward."""
    if hasattr(module, "_machete_original_forward"):
        module.forward = module._machete_original_forward
        del module._machete_original_forward
        if hasattr(module, "_machete_supports_quack"):
            del module._machete_supports_quack
