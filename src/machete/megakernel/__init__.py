# Copyright (c) 2025, Machete Authors
"""
Megakernel Module for Low-Latency LLM Inference.

This module implements a "No Bubbles" megakernel architecture with
instruction stream and fine-grained tile-level barriers for maximum
pipeline overlap between operations.

Usage:
    from machete.megakernel import Megakernel
    from machete.kernels.rms_norm import RMSNormOp

    ops = [RMSNormOp.schedule(x=x, weight=w, y=y, tile_sizes={"M": 4})]
    kernel = Megakernel(ops)
    kernel.run()
"""

from .ops import (
    Op,
    ScheduledOp,
    TensorMeta,
    BarrierFormula,
    INSTRUCTION_WORDS,
    TileInstruction,
    InstructionStreamBuilder,
    validate_op_compatibility,
)

from .compile import (
    cleanup_linecache,
)

from .megakernel import (
    MegakernelConfig,
    Megakernel,
    create_megakernel,
)

from .interpreter import (
    global_barrier_wait,
    global_barrier_signal,
    load_instruction_to_smem,
    st_global_i32,
    get_smem_base_ptr,
)

from .autograd_op import AutogradOp, TensorSpec
from .autograd import MegakernelFunction
from .kernel_cache import KernelCache
from .functional import megakernel_apply, MegakernelModule

__all__ = [
    # Operation Protocol
    "Op",
    "ScheduledOp",
    "TensorMeta",
    "validate_op_compatibility",
    # Barrier Formulas
    "BarrierFormula",
    # Compilation
    "cleanup_linecache",
    # Instruction Stream
    "INSTRUCTION_WORDS",
    "TileInstruction",
    "InstructionStreamBuilder",
    # Configuration & Megakernel
    "MegakernelConfig",
    "Megakernel",
    "create_megakernel",
    # Device-side primitives
    "global_barrier_wait",
    "global_barrier_signal",
    "load_instruction_to_smem",
    "st_global_i32",
    # PTX utilities
    "get_smem_base_ptr",
    # Autograd
    "AutogradOp",
    "TensorSpec",
    "MegakernelFunction",
    "KernelCache",
    "megakernel_apply",
    "MegakernelModule",
]
