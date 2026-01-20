# Copyright (c) 2025, Machete Authors
"""
Megakernel Operation Interfaces.

This module provides base classes and decorators for defining operations
that can be fused into megakernels.
"""

from abc import abstractmethod
from typing import Callable, Tuple
import cutlass.cute as cute

# Re-export dependency decorators, warp roles, and scheduler config for convenience
from machete.megakernel.scheduler import (
    # Dependency decorators
    reads,
    writes,
    warp_role,
    async_load,
    prefetchable,
    depends_on,
    # Warp configuration
    WarpRole,
    WarpConfig,
    # Logical blocks
    LogicalCoord,
    LogicalGridInfo,
    # Barrier config
    BarrierConfig,
)

__all__ = [
    # Dependency decorators
    "reads",
    "writes",
    "warp_role",
    "async_load",
    "prefetchable",
    "depends_on",
    # Warp configuration
    "WarpRole",
    "WarpConfig",
    # Logical blocks
    "LogicalCoord",
    "LogicalGridInfo",
    # Barrier config
    "BarrierConfig",
    # Operation classes
    "machete_op",
    "MegakernelOp",
    "FusableOp",
    "FusableKernel",
    "WarpSpecializedKernel",
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


class WarpSpecializedKernel(FusableKernel):
    """Base class for kernels that use warp specialization (No Bubbles pattern).

    This class extends FusableKernel with explicit support for warp-specialized
    execution where different warps execute different roles concurrently:
    - LOADER warps: Execute load_* methods (Global -> Shared)
    - CONSUMER warps: Execute compute_* methods (math/MMA operations)
    - STORER warps: Execute store_* methods (Shared -> Global)
    - CONTROLLER warp: Manages instruction scheduling
    - LAUNCHER warp: Handles auxiliary async tasks

    Producer-Consumer Synchronization:
    Instead of sync_threads() which blocks all warps, this class uses
    semaphore-based synchronization that allows true concurrent execution:

    - loader_ready[stage]: Loader signals "data loaded into stage"
    - consumer_done[stage]: Consumer signals "done with stage, can be reused"

    Pipeline Flow (with num_stages=2):
        Iteration 0: Loader fills stage 0
        Iteration 1: Loader fills stage 1, Consumer processes stage 0
        Iteration 2: Loader fills stage 0, Consumer processes stage 1, Storer writes stage 0 result
        ...

    Example usage:
        class MyWarpSpecializedKernel(WarpSpecializedKernel):
            TILE_M = 128
            TILE_N = 128

            @property
            def warp_config(self) -> WarpConfig:
                return WarpConfig(num_consumer_warps=12)

            @property
            def num_stages(self) -> int:
                return 2  # Double buffering

            @property
            def page_size(self) -> int:
                return self.TILE_M * self.TILE_N * 2  # fp16

            @warp_role(WarpRole.LOADER)
            @reads("input", "weight")
            @cute.jit
            def load_forward(self, logical_idx, page_ptr, stage, *args):
                # TMA/cp.async loads into page_ptr
                ...

            @warp_role(WarpRole.CONSUMER)
            @writes("output")
            @cute.jit
            def compute_forward(self, logical_idx, page_ptr, stage, *args):
                # MMA operations reading from page_ptr
                ...

            @warp_role(WarpRole.STORER)
            @cute.jit
            def store_forward(self, logical_idx, page_ptr, stage, *args):
                # Write results to global memory
                ...

    The Megakernel runtime will:
    1. Partition warps according to warp_config
    2. Allocate paged buffers with semaphores in shared memory
    3. Generate code that:
       - Dispatches methods based on warp role
       - Uses semaphore acquire/release instead of sync_threads()
       - Enables true overlap between load/compute/store phases
    """

    @property
    def warp_config(self) -> WarpConfig:
        """Configuration for warp specialization.

        Override this to customize the warp distribution. Default is H100-optimized
        with 16 consumer warps and 4 system warps (20 total = 640 threads).

        Returns:
            WarpConfig specifying warp counts per role
        """
        return WarpConfig()

    @property
    def num_stages(self) -> int:
        """Number of pipeline stages (buffer slots).

        Override this to enable multi-stage buffering:
        - 1: No buffering (sequential L->C->S)
        - 2: Double buffering (recommended)
        - 3: Triple buffering (for hiding very long latencies)

        Returns:
            Number of stages (default 2 for double buffering)
        """
        return 2

    @property
    def page_size(self) -> int:
        """Size of each data page in bytes.

        Override this to specify how much data each stage holds.
        Should be large enough for one tile of data.

        Returns:
            Page size in bytes (default 16KB)
        """
        return 16384

    @property
    def uses_warp_specialization(self) -> bool:
        """Whether this kernel uses warp specialization.

        This is always True for WarpSpecializedKernel. The Megakernel
        runtime uses this to determine scheduling strategy.
        """
        return True

    @property
    def smem_size_fwd(self) -> int:
        """Total shared memory needed for forward pass.

        Includes paged buffers + semaphores.
        Override page_size and num_stages instead of this.
        """
        # Data pages + semaphores (2 per stage, 4 bytes each)
        return (self.num_stages * self.page_size) + (self.num_stages * 2 * 4)

    @property
    def smem_size_bwd(self) -> int:
        """Total shared memory needed for backward pass."""
        return self.smem_size_fwd

    # ========== Semaphore Offsets ==========

    @property
    def _sem_base_offset(self) -> int:
        """Byte offset to semaphore array in shared memory."""
        return self.num_stages * self.page_size

    # ========== Warp-Role Specific Methods ==========

    @warp_role(WarpRole.LOADER)
    @cute.jit
    def loader_main(self, logical_idx, smem_base, num_iterations, *args):
        """Main loop for loader warps with pipelined execution.

        Implements the loader side of the producer-consumer protocol:
        1. Wait for consumer to release the stage (consumer_done)
        2. Load data into the stage
        3. Signal that data is ready (loader_ready)

        Override load_forward() to implement the actual load logic.

        Args:
            logical_idx: Base logical block index
            smem_base: Base pointer to shared memory
            num_iterations: Number of tiles/iterations to process
            *args: Kernel arguments
        """
        from cutlass import Int32, const_expr
        from machete.megakernel.paged_buffer import (
            loader_acquire_stage,
            loader_release_stage,
            get_page_ptr,
        )

        num_stages = const_expr(self.num_stages)
        page_size = const_expr(self.page_size)
        sem_offset = const_expr(self._sem_base_offset)

        for iteration in range(num_iterations):
            stage = iteration % num_stages

            # Wait for this stage to be available (consumer finished with it)
            loader_acquire_stage(smem_base, Int32(stage), Int32(iteration), num_stages, sem_offset)

            # Get pointer to this stage's data page
            page_ptr = get_page_ptr(smem_base, Int32(stage), page_size, cute.Float16)

            # Perform the actual load
            self.load_forward(logical_idx + iteration, page_ptr, stage, *args)

            # Signal that data is ready for consumption
            loader_release_stage(smem_base, Int32(stage), num_stages, sem_offset)

    @warp_role(WarpRole.CONSUMER)
    @cute.jit
    def consumer_main(self, logical_idx, smem_base, num_iterations, *args):
        """Main loop for consumer warps with pipelined execution.

        Implements the consumer side of the producer-consumer protocol:
        1. Wait for loader to fill the stage (loader_ready)
        2. Process the data
        3. Signal that stage can be reused (consumer_done)

        Override compute_forward() to implement the actual compute logic.

        Args:
            logical_idx: Base logical block index
            smem_base: Base pointer to shared memory
            num_iterations: Number of tiles/iterations to process
            *args: Kernel arguments
        """
        from cutlass import Int32, const_expr
        from machete.megakernel.paged_buffer import (
            consumer_acquire_stage,
            consumer_release_stage,
            get_page_ptr,
        )

        num_stages = const_expr(self.num_stages)
        page_size = const_expr(self.page_size)
        sem_offset = const_expr(self._sem_base_offset)

        for iteration in range(num_iterations):
            stage = iteration % num_stages

            # Wait for loader to fill this stage
            consumer_acquire_stage(smem_base, Int32(stage), Int32(iteration), num_stages, sem_offset)

            # Get pointer to this stage's data page
            page_ptr = get_page_ptr(smem_base, Int32(stage), page_size, cute.Float16)

            # Perform the actual computation
            self.compute_forward(logical_idx + iteration, page_ptr, stage, *args)

            # Signal that we're done with this stage (loader can reuse it)
            consumer_release_stage(smem_base, Int32(stage), num_stages, sem_offset)

    @warp_role(WarpRole.STORER)
    @cute.jit
    def storer_main(self, logical_idx, smem_base, num_iterations, *args):
        """Main loop for storer warps.

        For kernels with separate store phase, the storer runs one stage
        behind the consumer. Override store_forward() to implement store logic.

        Note: Many kernels write directly in compute_forward() and don't
        need a separate storer. In that case, leave store_forward() as no-op.

        Args:
            logical_idx: Base logical block index
            smem_base: Base pointer to shared memory
            num_iterations: Number of tiles/iterations to process
            *args: Kernel arguments
        """
        from cutlass import Int32, const_expr
        from machete.megakernel.paged_buffer import get_page_ptr

        num_stages = const_expr(self.num_stages)
        page_size = const_expr(self.page_size)

        for iteration in range(num_iterations):
            stage = iteration % num_stages

            # Get pointer to this stage's data page
            page_ptr = get_page_ptr(smem_base, Int32(stage), page_size, cute.Float16)

            # Perform the store
            self.store_forward(logical_idx + iteration, page_ptr, stage, *args)

    @warp_role(WarpRole.CONTROLLER)
    @cute.jit
    def controller_main(self, instruction_buffer, *args):
        """Main loop for controller warp.

        Default is no-op. Override for custom instruction scheduling
        or inter-block coordination.
        """
        pass

    @warp_role(WarpRole.LAUNCHER)
    @cute.jit
    def launcher_main(self, *args):
        """Main loop for launcher warp.

        Default is no-op. Override for auxiliary async tasks like
        K/V cache prefetching or TMA descriptor updates.
        """
        pass

    # ========== Initialization ==========

    @cute.jit
    def init_semaphores(self, smem_base):
        """Initialize semaphores at kernel start.

        Must be called by thread 0 followed by sync_threads() before
        the main loop begins.

        Args:
            smem_base: Base pointer to shared memory
        """
        from machete.megakernel.paged_buffer import paged_buffer_init_semaphores
        from cutlass import const_expr

        paged_buffer_init_semaphores(
            smem_base,
            const_expr(self.num_stages),
            const_expr(self._sem_base_offset),
        )
