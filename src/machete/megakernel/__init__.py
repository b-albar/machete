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
    # Warp configuration
    WarpRole,
    WarpConfig,
    # Barrier config
    BarrierConfig,
    # Page config
    PageConfig,
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
    "async_load",
    # Warp configuration
    "WarpRole",
    "WarpConfig",
    # Barrier config
    "BarrierConfig",
    # Page config
    "PageConfig",
    # Utilities
    "nanosleep",
]
