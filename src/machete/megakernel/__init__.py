# Copyright (c) 2025, Machete Authors
"""
Megakernel module for fusing multiple operations into a single GPU kernel.

This module implements the "No Bubbles" pattern from HazyResearch for maximum
overlap between load, compute, and store phases across fused operations.

Key Features:
- Logical Blocks: Abstract block coordinates for flexible kernel decomposition
- Warp Specialization: Dedicated warps for load, compute, store, and control
- Paged Shared Memory: Dynamic page allocation with semaphore-based sync
- Dependency-Aware Scheduling: Optimal overlap based on read/write analysis

Example:
    from machete.megakernel import Megakernel, FusableKernel, reads, writes

    class MyKernel(FusableKernel):
        def get_logical_grid_size(self, *args):
            return batch * seq_len // tile_size

        @reads("input")
        @cute.jit
        def load_forward(self, paged_pool, page_idx, *args):
            ...

        @writes("output")
        @cute.jit
        def store_forward(self, paged_pool, page_idx, *args):
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
    # Warp configuration
    WarpRole,
    WarpConfig,
    # Logical blocks
    LogicalCoord,
    LogicalGridInfo,
    # Scheduler configuration
    NoBubblesConfig,
    PageSemaphoreConfig,
    BarrierConfig,
    # Scheduling modes for mixed kernel support
    SchedulingMode,
    MixedModeScheduler,
)

# Utilities
from .utils import nanosleep

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
    # Warp configuration
    "WarpRole",
    "WarpConfig",
    # Logical blocks
    "LogicalCoord",
    "LogicalGridInfo",
    # Scheduler configuration
    "NoBubblesConfig",
    "PageSemaphoreConfig",
    "BarrierConfig",
    # Scheduling modes for mixed kernel support
    "SchedulingMode",
    "MixedModeScheduler",
    # Utilities
    "nanosleep",
]
