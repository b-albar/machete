# Copyright (c) 2025, Machete Authors
"""
Megakernel Module for Low-Latency LLM Inference.

This module implements a "No Bubbles" megakernel architecture with
instruction stream and fine-grained tile-level barriers for maximum
pipeline overlap between operations.

Usage:
    from machete.megakernel import Megakernel, config_dim_i32, config_flat_tensor
    from machete.kernels.rms_norm import RMSNormOp

    ops = RMSNormOp.schedule(x=x, weight=w, y=y)
    kernel = Megakernel(ops)
    kernel.run()
"""

from .ops import (
    DEFAULT_PAGE_SIZE,
    InstructionPageProtocol,
    Op,
    PageRole,
    PipelineABI,
    PipelineSpec,
    ScheduledOp,
    SemaphoreRole,
    TensorMeta,
    TileRange,
    build_op_config,
    config_dim_i32,
    config_ptr_i64,
    config_flat_tensor,
)

from .registries import (
    PeerBufferRegistry,
    PeerTMARegistry,
    validate_op_compatibility,
)

from .scheduling import (
    BarrierFormula,
    INSTRUCTION_WORDS,
    INSTR_BARRIER_META_IDX,
    INSTR_OP_IDX,
    INSTR_TILE_0,
    INSTR_TILE_1,
    INSTR_TILE_2,
    INSTR_TILE_3,
    INSTR_TILE_4,
    TileInstruction,
    InstructionStreamBuilder,
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
from .utils import dump_ptx, dump_sass, dump_cubin, extract_cubin

__all__ = [
    # Operation Protocol
    "Op",
    "PageRole",
    "SemaphoreRole",
    "InstructionPageProtocol",
    "PipelineSpec",
    "TileRange",
    "PipelineABI",
    "ScheduledOp",
    "TensorMeta",
    "PeerBufferRegistry",
    "PeerTMARegistry",
    "validate_op_compatibility",
    "build_op_config",
    "config_dim_i32",
    "config_ptr_i64",
    "config_flat_tensor",
    # Barrier Formulas
    "BarrierFormula",
    # Compilation
    "cleanup_linecache",
    # Instruction Stream
    "INSTRUCTION_WORDS",
    "INSTR_BARRIER_META_IDX",
    "INSTR_OP_IDX",
    "INSTR_TILE_0",
    "INSTR_TILE_1",
    "INSTR_TILE_2",
    "INSTR_TILE_3",
    "INSTR_TILE_4",
    "TileInstruction",
    "InstructionStreamBuilder",
    # Configuration & Megakernel
    "DEFAULT_PAGE_SIZE",
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
    # Debug utilities
    "dump_ptx",
    "dump_sass",
    "dump_cubin",
    "extract_cubin",
]
