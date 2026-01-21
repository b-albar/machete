# Copyright (c) 2025, Machete Authors
"""
Megakernel Operation Interfaces.

This module provides the MacheteKernel base class for defining operations
that can be fused into megakernels.

MacheteKernel Interface:
    The MacheteKernel base class provides a clean interface where:
    - Tensors are declared via declare_tensors() with symbolic dimensions
    - L/C/S methods receive CuTe tensors directly (not pointers)
    - setup_kernel() runs per logical_idx for shared state setup and smem allocation
    - Code is extracted via inspect and inlined into a single @cute.jit kernel

    For warp-specialized kernels (producer/consumer pipelines):
    - Set uses_warp_specialization property to True
    - Configure warp roles via warp_config property
    - The full L/C/S pipeline is contained in a single @cute.jit function
    - Shared barriers between load/compute/store phases
    - Supports pipelined execution when fusing multiple warp-specialized kernels
    - Load of next op can overlap with compute of previous op if smem available
"""

from typing import Any, Dict, Optional, Tuple

# Re-export dependency decorators, warp roles, and scheduler config for convenience
from machete.megakernel.scheduler import (
    # Dependency decorators
    reads,
    writes,
    # Warp configuration
    WarpRole,
    WarpConfig,
    # Logical blocks
    LogicalCoord,
    LogicalGridInfo,
    # Barrier config
    BarrierConfig,
)

# Import tensor spec types
from machete.megakernel.tensor_spec import (
    TensorSpec,
    KernelSignature,
    MemorySpace,
    tensor,
)

__all__ = [
    # Dependency decorators
    "reads",
    "writes",
    # Warp configuration
    "WarpRole",
    "WarpConfig",
    # Logical blocks
    "LogicalCoord",
    "LogicalGridInfo",
    # Barrier config
    "BarrierConfig",
    # Tensor specification
    "TensorSpec",
    "KernelSignature",
    "MemorySpace",
    "tensor",
    # Operation classes
    "machete_op",
    "MacheteKernel",
]


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


# =============================================================================
# MacheteKernel: Unified Interface for All Kernel Types
# =============================================================================


class MacheteKernel:
    """Base class for GPU kernels with tensor-first interface.

    MacheteKernel provides a unified interface for defining GPU kernels where:
    - Tensors are declared via declare_tensors() with symbolic dimensions
    - L/C/S methods receive CuTe tensors directly (not raw pointers)
    - setup_kernel() runs per logical_idx for shared state and smem allocation
    - Code is extracted via inspect and inlined into a single @cute.jit kernel

    Execution Modes:
    1. **Sequential (default)**: All threads execute L -> C -> S sequentially
       with sync_threads() barriers between phases.

    2. **Warp-Specialized**: Different warps execute different roles concurrently:
       - Loader warps: Execute load phases (global -> shared)
       - Consumer warps: Execute compute phases (math/MMA)
       - Storer warps: Execute store phases (shared -> global)

       Enable by overriding `uses_warp_specialization` to return True and
       configuring `warp_config`.

    Inter-Op Pipelining (Warp-Specialized Only):
    When fusing multiple warp-specialized kernels (Op A -> Op B -> Op C),
    the system can overlap execution:
    - While consumer warps compute Op A, loader warps can prefetch Op B's data
    - This requires sufficient shared memory for both operations
    - Control via `supports_inter_op_pipelining()` and `get_inter_op_smem_requirement()`

    Example (Sequential):
        class RopeSM80(MacheteKernel):
            NUM_THREADS = 256

            @property
            def smem_size_fwd(self) -> int:
                return self.half_d * 2 * 2  # cos + sin cache

            def declare_tensors(self) -> Dict[str, TensorSpec]:
                return {
                    "q": TensorSpec("q", self.cute_dtype,
                                   shape_expr=("n_tokens", "n_heads", "head_dim"),
                                   is_input=True, is_output=True),
                }

            def setup_kernel(self, logical_idx, smem, q, seq_len, n_tokens):
                self.m = logical_idx
                alloc = MacheteSmemAllocator(smem)
                self.s_cos = alloc.allocate_array(self.cute_dtype, self.half_d)

            def load_forward(self, logical_idx, smem, q, seq_len, n_tokens):
                # Load cos/sin to shared memory
                ...

            def compute_forward(self, logical_idx, smem, q, seq_len, n_tokens):
                # Apply rotation using shared memory
                ...

    Example (Warp-Specialized):
        class FlashAttentionKernel(MacheteKernel):
            NUM_STAGES = 2  # Double buffering

            @property
            def uses_warp_specialization(self) -> bool:
                return True

            @property
            def warp_config(self) -> WarpConfig:
                return WarpConfig(num_consumer_warps=12)

            @property
            def smem_size_fwd(self) -> int:
                return self.NUM_STAGES * (q_tile + k_tile + v_tile) + sem_bytes

            def load_forward(self, logical_idx, smem, *args):
                # Loader warps execute this
                ...

            def compute_forward(self, logical_idx, smem, *args):
                # Consumer warps execute this
                ...

            def store_forward(self, logical_idx, smem, *args):
                # Storer warps execute this
                ...
    """

    # ========== Class-Level Configuration ==========

    # Pipeline depth for warp-specialized kernels (2 = double buffer, 3 = triple)
    NUM_STAGES: int = 1

    # ========== Shared Memory Size ==========

    @property
    def smem_size_fwd(self) -> int:
        """Shared memory size in bytes for forward pass.

        Override this property to specify the shared memory your kernel needs.
        """
        return 0

    @property
    def smem_size_bwd(self) -> int:
        """Shared memory size in bytes for backward pass.

        Override this property if backward pass needs different smem size.
        Defaults to smem_size_fwd.
        """
        return self.smem_size_fwd

    # ========== Tensor Declaration Interface ==========

    def declare_tensors(self) -> Dict[str, TensorSpec]:
        """Declare all input and output tensors for this kernel.

        Override this method to specify the tensors your kernel uses.
        Dimension names in shape_expr are symbolic and will be resolved
        from scalar arguments at runtime.

        Returns:
            Dict mapping tensor name to TensorSpec

        Example:
            return {
                "x": TensorSpec("x", cute.Float16, ("batch", "seq", "hidden"),
                               is_input=True),
                "y": TensorSpec("y", cute.Float16, ("batch", "seq", "hidden"),
                               is_output=True),
            }
        """
        return {}

    def declare_scalars(self) -> Tuple[str, ...]:
        """Declare scalar parameters for this kernel.

        Override to specify which scalar parameters your kernel expects.
        These are used to resolve symbolic dimensions in tensor specs.

        Returns:
            Tuple of scalar parameter names

        Example:
            return ("n_tokens", "seq_len", "n_heads", "head_dim", "half_d")
        """
        return ()

    # ========== Host Setup ==========

    def setup_host(self, **scalars) -> Optional[Any]:
        """Host-side setup called before kernel launch.

        Override to create a cute.struct or other data structure
        that will be passed to the kernel. This runs on CPU.

        Args:
            **scalars: All scalar values for this kernel invocation

        Returns:
            Optional data structure to pass to kernel (e.g., cute.struct)
        """
        return None

    # ========== Kernel Setup (Per Logical Index) ==========

    def setup_kernel(self, logical_idx, smem, *args, **kwargs):
        """Per-logical_idx setup that runs at the start of each block's work.

        This method is called ONCE per logical_idx before L/C/S phases.
        Use it to:
        1. Allocate shared memory regions using MacheteSmemAllocator
        2. Set up shared state that all three phases can access via self.xxx

        NOTE: This method should NOT have @cute.jit decorator.
        Its code will be extracted and inlined into the generated kernel.

        Args:
            logical_idx: The logical block index for this invocation
            smem: Raw shared memory pointer for allocation
            *args: Tensor arguments (already CuTe tensors, not pointers)
            **kwargs: Scalar arguments
        """
        pass

    # ========== Forward Pass L/C/S ==========

    def load_forward(self, logical_idx, smem, *args, **kwargs):
        """Forward pass load phase: global memory -> shared memory.

        In warp-specialized mode, only loader warps execute this.

        NOTE: This method should NOT have @cute.jit decorator.
        Its code will be extracted and inlined into the generated kernel.

        Args:
            logical_idx: The logical block index
            smem: Raw shared memory pointer
            *args: Tensor arguments (already CuTe tensors)
            **kwargs: Scalar arguments
        """
        pass

    def compute_forward(self, logical_idx, smem, *args, **kwargs):
        """Forward pass compute phase: main computation logic.

        In warp-specialized mode, only consumer warps execute this.

        NOTE: This method should NOT have @cute.jit decorator.

        Args:
            logical_idx: The logical block index
            smem: Raw shared memory pointer
            *args: Tensor arguments (already CuTe tensors)
            **kwargs: Scalar arguments
        """
        pass

    def store_forward(self, logical_idx, smem, *args, **kwargs):
        """Forward pass store phase: shared memory -> global memory.

        In warp-specialized mode, only storer warps execute this.

        NOTE: This method should NOT have @cute.jit decorator.

        Args:
            logical_idx: The logical block index
            smem: Raw shared memory pointer
            *args: Tensor arguments (already CuTe tensors)
            **kwargs: Scalar arguments
        """
        pass

    # ========== Backward Pass L/C/S ==========

    def load_backward(self, logical_idx, smem, *args, **kwargs):
        """Backward pass load phase."""
        pass

    def compute_backward(self, logical_idx, smem, *args, **kwargs):
        """Backward pass compute phase."""
        pass

    def store_backward(self, logical_idx, smem, *args, **kwargs):
        """Backward pass store phase."""
        pass

    # ========== Logical Blocks API ==========

    def get_logical_grid_size(self, *args) -> int:
        """Return total number of logical blocks for this kernel.

        Override to enable the Logical Blocks abstraction.

        Args:
            *args: Kernel arguments

        Returns:
            Total number of logical blocks (units of work)
        """
        return 1

    def get_logical_coord(self, logical_idx: int, *args) -> Tuple[int, ...]:
        """Map linear logical block index to kernel-specific coordinates.

        Args:
            logical_idx: Linear index in [0, get_logical_grid_size())
            *args: Kernel arguments

        Returns:
            Tuple of coordinates (e.g., (batch_idx, seq_idx, head_idx))
        """
        return (logical_idx,)

    def get_logical_coord_names(self) -> Tuple[str, ...]:
        """Return names for logical coordinate dimensions."""
        return ("idx",)

    # ========== Kernel Signature ==========

    def get_kernel_signature(self) -> KernelSignature:
        """Build complete kernel signature from declarations."""
        return KernelSignature(
            tensors=self.declare_tensors(),
            scalars=list(self.declare_scalars()),
        )

    # ========== Warp Specialization Configuration ==========

    @property
    def uses_warp_specialization(self) -> bool:
        """Whether this kernel uses warp specialization.

        Override to return True to enable warp-specialized execution where:
        - Loader warps execute load_forward/load_backward
        - Consumer warps execute compute_forward/compute_backward
        - Storer warps execute store_forward/store_backward

        All three warp types run concurrently with semaphore synchronization.
        """
        return False

    @property
    def warp_config(self) -> WarpConfig:
        """Return warp configuration for warp-specialized execution.

        Override to customize the number of warps for each role.
        Only used when uses_warp_specialization returns True.

        Default configuration:
        - 16 consumer warps (for compute)
        - 1 loader warp
        - 1 storer warp
        - 1 launcher warp (auxiliary tasks)
        - 1 controller warp (pipeline coordination)

        Returns:
            WarpConfig specifying warp counts for each role
        """
        return WarpConfig()
