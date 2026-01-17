# Copyright (c) 2025, Machete Authors
"""
Utility functions for megakernel code generation.

This module provides PTX intrinsics and helper functions used in
the generated megakernel code.
"""

from cutlass import Int32
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def atomic_add_i32(val: Int32, ptr, *, loc=None, ip=None) -> Int32:
    """Atomic add for signed 32-bit integer.

    Performs an atomic add operation on shared memory and returns
    the previous value.

    Args:
        val: Value to add
        ptr: Pointer to the target address

    Returns:
        The old value before the add
    """
    from cutlass._mlir import ir

    result = llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [Int32(val).ir_value(loc=loc, ip=ip), ptr.llvm_ptr],
        "atom.add.u32 $0, [$1], $2;",
        "=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(result)


@dsl_user_op
def nanosleep(ns: int | Int32, *, loc=None, ip=None) -> None:
    """Sleep for approximately ns nanoseconds.

    This uses the PTX nanosleep.u32 instruction to put the thread
    into a low-power sleep state. This is useful for spin-wait loops
    to reduce power consumption while waiting for a counter.

    Available on SM 70+ (Volta and later).

    Args:
        ns: Sleep duration in nanoseconds (typically 100-1000)

    Note:
        The actual sleep duration may vary due to hardware implementation.
        This is not a precise timer, just a hint to reduce power usage.
    """
    llvm.inline_asm(
        None,  # No return value
        [Int32(ns).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",  # Input constraint: r = 32-bit register
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
