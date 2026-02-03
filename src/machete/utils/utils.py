# Copyright (c) 2025, Machete Authors
"""
Utility functions for megakernel code generation.

This module provides PTX intrinsics and helper functions used in
the generated megakernel code, including:
- Atomic operations for shared memory
- Producer-consumer semaphore primitives
- Sleep/wait utilities
"""

from cutlass import Int32
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


# ============================================================================
# Atomic Operations
# ============================================================================


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
def atomic_load_acquire_i32(ptr, *, loc=None, ip=None) -> Int32:
    """Atomic load with acquire semantics for shared memory.

    Ensures all subsequent reads see writes that happened before
    the corresponding release store.

    Args:
        ptr: Pointer to shared memory location

    Returns:
        The loaded value
    """
    from cutlass._mlir import ir

    result = llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [ptr.llvm_ptr],
        "ld.acquire.sys.shared.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Int32(result)


@dsl_user_op
def atomic_store_release_i32(val: Int32, ptr, *, loc=None, ip=None) -> None:
    """Atomic store with release semantics for shared memory.

    Ensures all prior writes are visible before this store is seen
    by threads doing acquire loads.

    Args:
        val: Value to store
        ptr: Pointer to shared memory location
    """
    llvm.inline_asm(
        None,
        [ptr.llvm_ptr, Int32(val).ir_value(loc=loc, ip=ip)],
        "st.release.sys.shared.b32 [$1], $2;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# ============================================================================
# Producer-Consumer Semaphore Primitives
# ============================================================================


@dsl_user_op
def semaphore_init(ptr, initial_value: Int32 = Int32(0), *, loc=None, ip=None) -> None:
    """Initialize a semaphore in shared memory.

    Should be called by a single thread (e.g., thread 0) before use.

    Args:
        ptr: Pointer to semaphore location in shared memory (4 bytes)
        initial_value: Initial semaphore value (default 0)
    """
    llvm.inline_asm(
        None,
        [ptr.llvm_ptr, Int32(initial_value).ir_value(loc=loc, ip=ip)],
        "st.shared.b32 [$1], $2;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def semaphore_signal(ptr, *, loc=None, ip=None) -> None:
    """Signal (increment) a semaphore with release semantics.

    Used by producers to indicate data is ready. The release semantics
    ensure all prior writes (to the data buffer) are visible before
    the semaphore increment is seen.

    Args:
        ptr: Pointer to semaphore in shared memory
    """
    # Atomic increment with release semantics
    llvm.inline_asm(
        None,
        [ptr.llvm_ptr],
        "atom.add.release.sys.shared.u32 _, [$1], 1;",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def semaphore_wait(ptr, expected: Int32, *, loc=None, ip=None) -> None:
    """Wait until semaphore reaches expected value with acquire semantics.

    Spins until the semaphore value >= expected. Uses acquire semantics
    so that all writes done before the producer's signal are visible.

    Args:
        ptr: Pointer to semaphore in shared memory
        expected: Value to wait for (typically stage number)
    """
    # Spin-wait with acquire load and nanosleep for power efficiency
    # The loop: while (ld.acquire(ptr) < expected) nanosleep(100)
    llvm.inline_asm(
        None,
        [ptr.llvm_ptr, Int32(expected).ir_value(loc=loc, ip=ip)],
        """
        {{
            .reg .pred %p;
            .reg .u32 %val;
        $L_wait_loop:
            ld.acquire.sys.shared.b32 %val, [$1];
            setp.ge.u32 %p, %val, $2;
            @%p bra $L_wait_done;
            nanosleep.u32 100;
            bra $L_wait_loop;
        $L_wait_done:
        }}
        """,
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def semaphore_try_wait(ptr, expected: Int32, *, loc=None, ip=None) -> Int32:
    """Non-blocking check if semaphore has reached expected value.

    Returns 1 if semaphore >= expected, 0 otherwise.

    Args:
        ptr: Pointer to semaphore in shared memory
        expected: Value to check for

    Returns:
        1 if ready, 0 if not
    """
    from cutlass._mlir import ir

    result = llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [ptr.llvm_ptr, Int32(expected).ir_value(loc=loc, ip=ip)],
        """
        {{
            .reg .pred %p;
            .reg .u32 %val;
            ld.acquire.sys.shared.b32 %val, [$1];
            setp.ge.u32 %p, %val, $2;
            selp.u32 $0, 1, 0, %p;
        }}
        """,
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
