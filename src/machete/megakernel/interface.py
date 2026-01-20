# Copyright (c) 2025, Machete Authors
"""
Megakernel Operation Interfaces.

This module provides base classes and decorators for defining operations
that can be fused into megakernels.
"""

from abc import abstractmethod
from typing import Callable, Tuple, List, Optional
from dataclasses import dataclass
import cutlass.cute as cute

# Re-export dependency decorators, warp roles, and scheduler config for convenience
from machete.megakernel.scheduler import (
    # Dependency decorators
    reads,
    writes,
    warp_role,
    async_load,
    TensorDependency,
    # Warp configuration
    WarpRole,
    WarpConfig,
    # Logical blocks
    LogicalCoord,
    LogicalGridInfo,
    # Barrier config
    BarrierConfig,
    # Page config
    PageConfig,
    PageSemaphores,
)

__all__ = [
    # Dependency decorators
    "reads",
    "writes",
    "warp_role",
    "async_load",
    "TensorDependency",
    # Warp configuration
    "WarpRole",
    "WarpConfig",
    # Logical blocks
    "LogicalCoord",
    "LogicalGridInfo",
    # Barrier config
    "BarrierConfig",
    # Page config
    "PageConfig",
    "PageSemaphores",
    # Data definitions
    "TensorParam",
    "TensorDef",
    # Operation classes
    "machete_op",
    "MegakernelOp",
    "FusableOp",
    "FusableKernel",
    "WarpSpecializedKernel",
]


@dataclass
class TensorDef:
    """Declarative tensor definition for clean kernel interfaces.

    Use TensorDef to define your kernel's tensor inputs. The megakernel will
    automatically generate a JIT wrapper that:
    1. Creates CuTe tensors from Int64 pointers
    2. Inlines your compute_forward body with the tensors ready to use
    3. Handles the verbose 32-slot signature internally

    Your compute_forward method should NOT have @cute.jit decorator - the
    wrapper handles JIT compilation. Parameters are: idx (global thread index),
    then tensors/scalars matching your tensor_defs names.

    For tensors, you specify which dimensions of the PyTorch tensor to use
    for the CuTe tensor shape and stride.

    Example:
        class AddKernel(FusableKernel):
            tensor_defs = [
                TensorDef("input_t", cute.Float32, shape=(0,), stride=(0,)),
                TensorDef("bias_t", cute.Float32, shape=(0,), stride=(0,)),
                TensorDef("output_t", cute.Float32, shape=(0,), stride=(0,)),
                TensorDef("n_elements", is_scalar=True),
            ]

            # NO @cute.jit! Parameters: idx, then tensor_defs names
            def compute_forward(self, idx, input_t, bias_t, output_t, n_elements):
                if idx < n_elements:
                    output_t[idx] = input_t[idx] + bias_t[idx]

    Args:
        name: Variable name for the tensor (passed to compute_forward)
        dtype: CuTe dtype (e.g., cute.Float32, cute.Float16)
        shape: Tuple of indices into tensor.shape for CuTe tensor dimensions.
               E.g., (0,) for 1D using dim 0, (0, 1) for 2D using dims 0 and 1.
        stride: Tuple of indices into tensor.stride() for CuTe tensor strides.
                Usually matches shape indices for standard tensors.
        is_scalar: If True, pass raw Int64 value instead of creating tensor
    """
    name: str
    dtype: any = None  # CuTe dtype like cute.Float32
    shape: Tuple[int, ...] = None  # Indices into tensor.shape
    stride: Tuple[int, ...] = None  # Indices into tensor.stride()
    is_scalar: bool = False


@dataclass
class TensorParam:
    """Defines a tensor parameter for automatic CuTe tensor creation.

    Uses indices into the PyTorch tensor's shape/stride at runtime.

    Args:
        name: Name of the tensor parameter (used for generated variable naming)
        shape: Tuple of indices into tensor.shape (e.g., (0, 1) for 2D tensor)
        stride: Tuple of indices into tensor.stride() (e.g., (0, 1) for actual strides)
        dtype_attr: Optional name of kernel attribute containing the CuTe dtype.
                   If provided, uses getattr(kernel, dtype_attr) for make_tensor.
                   If None, dtype is inferred from the PyTorch tensor.

    Examples:
        # 2D tensor using dims 0 and 1, dtype inferred from PyTorch tensor
        TensorParam("a", shape=(0, 1), stride=(0, 1))

        # 3D tensor with explicit dtype from kernel attribute
        TensorParam("q", shape=(0, 1, 2), stride=(0, 1, 2), dtype_attr="cute_dtype")
    """

    name: str
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype_attr: Optional[str] = None


class MegakernelOp:
    """
    Base class for operations that can be run either as a simple kernel
    or fused into a megakernel.

    Operations implement Load/Compute/Store phases:
    - load(): Load data from global memory to shared memory
    - compute(): Perform computation using shared memory
    - store(): Store results from shared memory back to global memory

    Logical Blocks API (optional):
    - get_logical_grid_size(): Return total logical blocks for this kernel
    - get_logical_coord(): Map linear logical_idx to kernel-specific coordinates
    """

    @property
    def smem_size_fwd(self) -> int:
        """Shared memory size needed by this operation in bytes for forward pass."""
        return 0

    @property
    def smem_size_bwd(self) -> int:
        """Shared memory size needed by this operation in bytes for backward pass."""
        return 0

    # ========== Logical Blocks API ==========

    def get_logical_grid_size(self, *args) -> int:
        """Return the total number of logical blocks for this operation.

        Override this for kernels that want to use the Logical Blocks abstraction.
        The default returns 1 (single block).

        Args:
            *args: Same arguments passed to load/compute/store

        Returns:
            Total number of logical blocks (units of work)
        """
        return 1

    def get_logical_coord(self, logical_idx: int, *args) -> Tuple[int, ...]:
        """Map a linear logical block index to kernel-specific coordinates.

        Override this along with get_logical_grid_size() to use the
        Logical Blocks abstraction.

        Args:
            logical_idx: Linear index in [0, get_logical_grid_size())
            *args: Same arguments passed to load/compute/store

        Returns:
            Tuple of coordinates specific to this kernel (e.g., (batch, seq, head))
        """
        return (logical_idx,)

    def get_logical_grid_info(self, *args) -> LogicalGridInfo:
        """Return full logical grid information including coordinate names.

        Override this for better debugging and tracing support.

        Returns:
            LogicalGridInfo with grid size and coordinate dimension info
        """
        return LogicalGridInfo(
            logical_grid_size=self.get_logical_grid_size(*args),
            coord_names=("idx",),
            coord_dims=(self.get_logical_grid_size(*args),),
        )

    @cute.jit
    def load(self, logical_idx, *args):
        """Load phase: move data from global memory to shared memory.

        Args:
            logical_idx: The global logical block index (use this for offsets!)
            *args: Remaining operation arguments
        """
        pass

    @abstractmethod
    @cute.jit
    def compute(self, logical_idx, *args, **kwargs):
        """Compute phase: perform the operation logic.

        Args:
            logical_idx: The global logical block index
            *args: Operation arguments
        """
        pass

    @cute.jit
    def store(self, logical_idx, *args):
        """Store phase: move results from shared memory to global memory.

        Args:
            logical_idx: Global logical block index
            *args: Remaining arguments
        """
        pass

    @abstractmethod
    def launch(self, *args, **kwargs):
        """Launch the operation as a standalone kernel."""
        pass


class FusableOp(MegakernelOp):
    """
    A generic wrapper that turns any @cute.jit function or method into a MegakernelOp.
    """

    def __init__(
        self,
        compute_func: Callable,
        smem_size_fwd: int = 0,
        smem_size_bwd: int = 0,
        load_func: Callable = None,
        store_func: Callable = None,
        launch_func: Callable = None,
        logical_grid_func: Callable = None,
        logical_coord_func: Callable = None,
    ):
        self._compute_func = compute_func
        self._load_func = load_func
        self._store_func = store_func
        self._smem_size = smem_size_fwd
        self._smem_size_bwd = smem_size_bwd
        self._launch_func = launch_func
        self._logical_grid_func = logical_grid_func
        self._logical_coord_func = logical_coord_func

    @property
    def smem_size_fwd(self) -> int:
        return self._smem_size

    @property
    def smem_size_bwd(self) -> int:
        return self._smem_size_bwd

    def get_logical_grid_size(self, *args) -> int:
        if self._logical_grid_func:
            return self._logical_grid_func(*args)
        return 1

    def get_logical_coord(self, logical_idx: int, *args) -> Tuple[int, ...]:
        if self._logical_coord_func:
            return self._logical_coord_func(logical_idx, *args)
        return (logical_idx,)

    @cute.jit
    def load(self, logical_idx, *args):
        pass

    @cute.jit
    def compute(self, logical_idx, *args, **kwargs):
        self._compute_func(logical_idx, *args, **kwargs)

    @cute.jit
    def store(self, logical_idx, *args):
        pass

    def launch(self, *args, **kwargs):
        if self._launch_func:
            self._launch_func(*args, **kwargs)
        else:
            raise NotImplementedError("Standalone launch not configured for this FusableOp")


class FusableKernel:
    """
    Base class for kernels that support both forward and backward
    passes in a megakernel with Load/Compute/Store phases.

    Supports Logical Blocks abstraction:
    - Override get_logical_grid_size() to define total work units
    - Override get_logical_coord() to map linear index to coordinates
    - Coordinates are passed to L/C/S methods instead of raw block indices
    """

    # ========== CuTe DSL Protocol Methods ==========
    # These are required for the kernel to be passed through the DSL JIT system.
    # Defined here so subclasses don't need to implement them.

    def __extract_mlir_values__(self):
        """Extract MLIR values for DSL serialization."""
        return []

    def __new_from_mlir_values__(self, values):
        """Reconstruct from MLIR values."""
        return self

    def __c_pointers__(self):
        """Return C pointers for FFI."""
        return []

    # ========== Execution Configuration ==========

    @property
    def execution_mode(self) -> str:
        """Execution mode for this kernel: 'sequential' or 'warp_specialized'.

        Override this to specify how the kernel should be executed:
        - 'sequential': All warps participate in Load → Sync → Compute → Sync → Store
        - 'warp_specialized': Different warps handle different phases concurrently

        Returns:
            'sequential' (default) or 'warp_specialized'
        """
        return "sequential"

    # ========== Shared Memory Configuration ==========

    @property
    def smem_size_fwd(self) -> int:
        """Shared memory size needed by this kernel in bytes for forward pass."""
        return 0

    @property
    def smem_size_bwd(self) -> int:
        """Shared memory size needed by this kernel in bytes for backward pass."""
        return 0

    # ========== Logical Blocks API ==========

    def get_logical_grid_size(self, *args) -> int:
        """Return the total number of logical blocks for this kernel.

        Override this method to enable the Logical Blocks abstraction.
        The scheduler will launch with grid_size = max(kernel.get_logical_grid_size())
        across all fused kernels.

        Args:
            *args: Kernel arguments (tensors, scalars) - same as passed to L/C/S

        Returns:
            Total number of logical blocks (units of work)
        """
        return 1

    def get_logical_coord(self, logical_idx: int, *args) -> Tuple[int, ...]:
        """Map a linear logical block index to kernel-specific coordinates.

        Override this method along with get_logical_grid_size() to use the
        Logical Blocks abstraction.

        Args:
            logical_idx: Linear index in [0, get_logical_grid_size())
            *args: Kernel arguments (tensors, scalars) - same as passed to L/C/S

        Returns:
            Tuple of coordinates (e.g., (batch_idx, seq_chunk_idx, head_idx))
        """
        return (logical_idx,)

    def get_logical_coord_names(self) -> Tuple[str, ...]:
        """Return names for the logical coordinate dimensions.

        Override this for better debugging and trace visualization.

        Returns:
            Tuple of coordinate names (e.g., ("batch", "seq_chunk", "head"))
        """
        return ("idx",)

    def get_logical_grid_info(self, *args) -> LogicalGridInfo:
        """Return full logical grid information.

        Returns:
            LogicalGridInfo with grid size and coordinate metadata
        """
        grid_size = self.get_logical_grid_size(*args)
        return LogicalGridInfo(
            logical_grid_size=grid_size,
            coord_names=self.get_logical_coord_names(),
            coord_dims=(grid_size,),  # Default: 1D grid
        )

    # ========== Forward Pass L/C/S ==========

    @cute.jit
    def load_forward(self, logical_idx, *args):
        """Forward load phase (optional)."""
        pass

    @cute.jit
    def compute_forward(self, logical_idx, *args, **kwargs):
        """Forward pass compute logic."""
        pass

    @cute.jit
    def store_forward(self, logical_idx, *args):
        """Forward store phase (optional)."""
        pass

    # ========== Backward Pass L/C/S ==========

    @cute.jit
    def load_backward(self, logical_idx, *args):
        """Backward load phase (optional)."""
        pass

    @cute.jit
    def compute_backward(self, logical_idx, *args, **kwargs):
        """Backward pass compute logic."""
        pass

    @cute.jit
    def store_backward(self, logical_idx, *args):
        """Backward store phase (optional)."""
        pass


def machete_op(smem_size: int = 0):
    """Decorator to mark a function or method as a Machete Megakernel operation.

    Args:
        smem_size: Total shared memory size needed by this operation in bytes.

    Example:
        class MyKernel:
            @machete_op(smem_size=16384)
            @cute.jit
            def compute(self, input, weight, output):
                ...
    """

    def decorator(func):
        func._machete_is_op = True
        func._machete_smem_size = smem_size
        return func

    return decorator


class WarpSpecializedKernel(FusableKernel):
    """Kernel supporting warp specialization.

    Extends FusableKernel with warp role configuration. Override warp_config
    to customize the warp allocation for your kernel.

    Warp Layout: [Consumer warps...][Loader][Storer][Launcher][Controller]
    """

    @property
    def uses_warp_specialization(self) -> bool:
        """Returns True since this kernel uses warp specialization."""
        return True

    @property
    def warp_config(self) -> WarpConfig:
        """Override to configure warp roles.

        Returns:
            WarpConfig with warp counts for each role.
        """
        return WarpConfig()
