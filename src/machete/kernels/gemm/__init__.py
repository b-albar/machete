# Copyright (c) 2025, Machete Authors
"""
GEMM Kernels for Machete Megakernel Framework.

This module provides high-performance GEMM (General Matrix Multiply) operations
that integrate with the machete megakernel framework for fusion with other ops.

Available ops:
- GemmOp: Basic GEMM with element-wise computation

Usage:
    from machete.kernels.gemm import GemmOp
    from machete.megakernel import Megakernel, MegakernelConfig

    # Schedule GEMM operation
    scheduled_op = GemmOp.schedule(a=A, b=B, c=C)

    kernel = Megakernel([scheduled_op], config=MegakernelConfig())
    kernel.run()
"""

from .sm_120 import (
    GemmOp,
    gemm_megakernel,
    is_blackwell_available,
    is_hopper_available,
)

__all__ = [
    "GemmOp",
    "gemm_megakernel",
    "is_blackwell_available",
    "is_hopper_available",
]
