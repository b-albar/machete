# Copyright (c) 2025, Machete Authors
"""
Tensor Specification System for Machete Kernels.

This module provides classes for declaratively specifying tensor inputs/outputs
with symbolic dimensions that resolve at runtime.
"""

from dataclasses import dataclass, field
from typing import Tuple, Any, Optional, Dict, List, Union
from enum import Enum, auto


class MemorySpace(Enum):
    """Memory space for tensor allocation."""
    GMEM = auto()  # Global memory
    SMEM = auto()  # Shared memory
    RMEM = auto()  # Register memory


@dataclass
class TensorSpec:
    """Specification for declaring a kernel tensor with symbolic dimensions.

    TensorSpec allows kernels to declare their tensor inputs/outputs with
    symbolic dimension names (like "n_tokens", "n_heads") that resolve to
    actual values at runtime from scalar arguments.

    Example:
        spec = TensorSpec(
            name="q",
            dtype=cute.Float16,
            shape_expr=("n_tokens", "n_heads", "head_dim"),
            is_input=True,
            is_output=True,  # In-place operation
        )
    """

    name: str
    dtype: Any  # CuTe dtype (e.g., cute.Float16)
    shape_expr: Tuple[str, ...]  # Symbolic dimension names
    stride_expr: Optional[Tuple[str, ...]] = None  # Optional stride expressions
    is_input: bool = True
    is_output: bool = False
    memory_space: MemorySpace = MemorySpace.GMEM
    alignment: int = 16  # Byte alignment for async copy

    def resolve_shape(self, scalars: Dict[str, int]) -> Tuple[int, ...]:
        """Resolve symbolic shape to concrete integer values.

        Args:
            scalars: Dict mapping dimension names to values

        Returns:
            Tuple of concrete dimension sizes

        Raises:
            KeyError: If a dimension name is not found in scalars
        """
        resolved = []
        for dim in self.shape_expr:
            if isinstance(dim, int):
                resolved.append(dim)
            elif dim in scalars:
                resolved.append(scalars[dim])
            else:
                raise KeyError(f"Dimension '{dim}' not found in scalars: {list(scalars.keys())}")
        return tuple(resolved)

    def resolve_stride(self, scalars: Dict[str, int]) -> Tuple[int, ...]:
        """Resolve strides - default to row-major if not specified.

        Args:
            scalars: Dict mapping dimension names to values

        Returns:
            Tuple of strides for each dimension
        """
        if self.stride_expr:
            resolved = []
            for s in self.stride_expr:
                if isinstance(s, int):
                    resolved.append(s)
                elif s in scalars:
                    resolved.append(scalars[s])
                else:
                    raise KeyError(f"Stride expression '{s}' not found in scalars")
            return tuple(resolved)

        # Default: row-major (C-contiguous) strides
        shape = self.resolve_shape(scalars)
        strides = []
        stride = 1
        for dim in reversed(shape):
            strides.append(stride)
            stride *= dim
        return tuple(reversed(strides))

    def generate_layout_code(self, scalars: Dict[str, int]) -> str:
        """Generate CuTe layout creation code.

        Args:
            scalars: Dict mapping dimension names to values

        Returns:
            String of Python code to create the layout
        """
        shape = self.resolve_shape(scalars)
        stride = self.resolve_stride(scalars)
        return f"cute.make_layout({shape}, stride={stride})"

    def generate_make_tensor_code(self, ptr_var: str, scalars: Dict[str, int]) -> str:
        """Generate cute.make_tensor code.

        Args:
            ptr_var: Name of the pointer variable
            scalars: Dict mapping dimension names to values

        Returns:
            String of Python code to create the tensor
        """
        layout_code = self.generate_layout_code(scalars)
        return f"cute.make_tensor({ptr_var}, {layout_code})"

    def generate_symbolic_layout_code(self) -> str:
        """Generate layout code with symbolic dimension names (for templates).

        Returns:
            String with symbolic references like (n_tokens, n_heads, head_dim)
        """
        shape_str = ", ".join(self.shape_expr)
        if self.stride_expr:
            stride_str = ", ".join(self.stride_expr)
            return f"cute.make_layout(({shape_str}), stride=({stride_str}))"
        else:
            return f"cute.make_layout(({shape_str}))"

    def element_size_bytes(self) -> int:
        """Get element size in bytes."""
        if hasattr(self.dtype, 'width'):
            return self.dtype.width // 8
        # Fallback for common types
        return 2  # Assume fp16


@dataclass
class KernelSignature:
    """Complete signature for a kernel including all tensors and scalars.

    This aggregates TensorSpecs and scalar parameters to fully describe
    a kernel's interface.
    """

    tensors: Dict[str, TensorSpec] = field(default_factory=dict)
    scalars: List[str] = field(default_factory=list)  # Scalar parameter names

    def get_inputs(self) -> Dict[str, TensorSpec]:
        """Get all input tensors."""
        return {k: v for k, v in self.tensors.items() if v.is_input}

    def get_outputs(self) -> Dict[str, TensorSpec]:
        """Get all output tensors."""
        return {k: v for k, v in self.tensors.items() if v.is_output}

    def get_inplace(self) -> Dict[str, TensorSpec]:
        """Get tensors that are both input and output (in-place)."""
        return {k: v for k, v in self.tensors.items() if v.is_input and v.is_output}

    def validate_scalars(self, provided: Dict[str, Any]) -> bool:
        """Check that all required scalars are provided."""
        # Collect all dimension names used
        required = set()
        for spec in self.tensors.values():
            for dim in spec.shape_expr:
                if isinstance(dim, str):
                    required.add(dim)
            if spec.stride_expr:
                for s in spec.stride_expr:
                    if isinstance(s, str):
                        required.add(s)

        # Check all are provided
        provided_keys = set(provided.keys())
        missing = required - provided_keys
        if missing:
            raise ValueError(f"Missing scalar values: {missing}")
        return True


class TensorRegistry:
    """Registry for managing tensor specs across fused kernels.

    When fusing multiple kernels, this registry tracks all tensors
    and validates that fused kernels have compatible interfaces.
    """

    def __init__(self):
        self.specs: Dict[str, TensorSpec] = {}
        self.kernel_specs: Dict[str, KernelSignature] = {}  # Per-kernel signatures

    def register_kernel(self, kernel_name: str, signature: KernelSignature):
        """Register all specs from a kernel signature.

        Args:
            kernel_name: Unique name for this kernel
            signature: The kernel's signature
        """
        self.kernel_specs[kernel_name] = signature

        # Add tensors with kernel prefix to avoid collisions
        for name, spec in signature.tensors.items():
            full_name = f"{kernel_name}.{name}"
            self.specs[full_name] = spec

    def get_tensor(self, kernel_name: str, tensor_name: str) -> TensorSpec:
        """Get a tensor spec by kernel and tensor name."""
        full_name = f"{kernel_name}.{tensor_name}"
        if full_name not in self.specs:
            raise KeyError(f"Tensor '{tensor_name}' not found for kernel '{kernel_name}'")
        return self.specs[full_name]

    def validate_fusion(self) -> bool:
        """Validate that fused kernels have compatible dimensions.

        For now, just checks that all kernels can be registered.
        Future: Add dimension compatibility checking.
        """
        return True


# Convenience function for creating specs
def tensor(
    name: str,
    dtype: Any,
    shape: Tuple[Union[str, int], ...],
    *,
    stride: Optional[Tuple[Union[str, int], ...]] = None,
    is_input: bool = True,
    is_output: bool = False,
    alignment: int = 16,
) -> TensorSpec:
    """Convenience function to create a TensorSpec.

    Example:
        q = tensor("q", cute.Float16, ("n_tokens", "n_heads", "head_dim"),
                   is_input=True, is_output=True)
    """
    return TensorSpec(
        name=name,
        dtype=dtype,
        shape_expr=tuple(str(s) if isinstance(s, int) else s for s in shape),
        stride_expr=tuple(str(s) if isinstance(s, int) else s for s in stride) if stride else None,
        is_input=is_input,
        is_output=is_output,
        alignment=alignment,
    )


