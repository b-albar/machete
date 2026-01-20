# Copyright (c) 2025, Machete Authors
"""
Megakernel module for fusing multiple operations into a single GPU kernel.

This module implements the "No Bubbles" pattern from HazyResearch for maximum
overlap between load, compute, and store phases across fused operations.

Key Features:
- Logical Blocks: Abstract block coordinates for flexible kernel decomposition
- Warp Specialization: Dedicated warps for load, compute, store, and control
- Static Shared Memory: Computed from operation graph at compile time
- Dependency-Aware Scheduling: Optimal overlap based on read/write analysis
- Mixed Kernel Types: Unified support for LCS and Producer/Consumer patterns

Example:
    from machete.megakernel import Megakernel, FusableKernel, reads, writes

    class MyKernel(FusableKernel):
        def get_logical_grid_size(self, *args):
            return batch * seq_len // tile_size

        @reads("input")
        @cute.jit
        def load_forward(self, logical_idx, *args):
            ...

        @writes("output")
        @cute.jit
        def store_forward(self, logical_idx, *args):
            ...

    mk = Megakernel()
    mk.add(MyKernel(), input, output, batch=2, seq_len=512)
    mk.launch_logical(block=(256, 1, 1))
"""

# Core classes
from .core import Megakernel
from .interface import (
    # Operation base classes
    MegakernelOp,
    FusableOp,
    FusableKernel,
    WarpSpecializedKernel,
    machete_op,
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

# Utilities
from .utils import (
    nanosleep,
    atomic_add_i32,
    atomic_load_acquire_i32,
    atomic_store_release_i32,
    semaphore_init,
    semaphore_signal,
    semaphore_wait,
    semaphore_try_wait,
)

# Paged buffer system
from .paged_buffer import (
    PagedBufferConfig,
    PagedBufferLayout,
    PagedBufferAllocator,
    paged_buffer_init_semaphores,
    loader_acquire_stage,
    loader_release_stage,
    consumer_acquire_stage,
    consumer_release_stage,
    get_page_ptr,
    # Inter-op semaphores
    InterOpSemaphoreConfig,
    InterOpSemaphoreLayout,
    inter_op_init_semaphores,
    inter_op_wait_for_dependency,
    inter_op_signal_done,
    inter_op_try_acquire,
)

__all__ = [
    # Core
    "Megakernel",
    # Operation base classes
    "MegakernelOp",
    "FusableOp",
    "FusableKernel",
    "WarpSpecializedKernel",
    "machete_op",
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
    # Utilities - atomics and sleep
    "nanosleep",
    "atomic_add_i32",
    "atomic_load_acquire_i32",
    "atomic_store_release_i32",
    # Utilities - semaphores
    "semaphore_init",
    "semaphore_signal",
    "semaphore_wait",
    "semaphore_try_wait",
    # Paged buffer system
    "PagedBufferConfig",
    "PagedBufferLayout",
    "PagedBufferAllocator",
    "paged_buffer_init_semaphores",
    "loader_acquire_stage",
    "loader_release_stage",
    "consumer_acquire_stage",
    "consumer_release_stage",
    "get_page_ptr",
    # Inter-op semaphores
    "InterOpSemaphoreConfig",
    "InterOpSemaphoreLayout",
    "inter_op_init_semaphores",
    "inter_op_wait_for_dependency",
    "inter_op_signal_done",
    "inter_op_try_acquire",
]
