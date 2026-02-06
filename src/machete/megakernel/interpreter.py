# Copyright (c) 2025, Machete Authors
"""
Device-Side Primitives for Megakernel.

This module provides PTX inline assembly primitives used by the persistent
megakernel kernel:
- Global memory barriers (inter-block synchronization)
- Instruction loading from global memory to shared memory
- Global memory load/store utilities
"""

from cutlass import Int32, Int64
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

# Counter for unique PTX labels across inline asm invocations.
# PTX labels are function-scoped (not block-scoped), so multiple
# inlined copies of the same asm template would collide without
# unique names.
_ptx_label_counter = 0


# =============================================================================
# Global Barrier (Inter-Block Synchronization)
# =============================================================================


@dsl_user_op
def global_barrier_wait(
    barrier_ptr: Int64,
    barrier_idx: Int32,
    expected: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Wait on a global memory barrier.

    Spins until barrier[barrier_idx] >= expected.
    Used for inter-block synchronization (e.g., layer dependencies).

    Args:
        barrier_ptr: Pointer to barrier array in global memory (64-bit)
        barrier_idx: Index of barrier to wait on
        expected: Expected count
    """
    global _ptx_label_counter
    _ptx_label_counter += 1
    _loop = "LBB_bwait_loop_{}".format(_ptx_label_counter)
    _done = "LBB_bwait_done_{}".format(_ptx_label_counter)

    llvm.inline_asm(
        None,
        [barrier_ptr.ir_value(loc=loc, ip=ip), barrier_idx.ir_value(loc=loc, ip=ip), expected.ir_value(loc=loc, ip=ip)],
        "{{ "
        ".reg .pred %p; "
        ".reg .u32 %val; "
        ".reg .u64 %addr; "
        "mad.wide.u32 %addr, $1, 4, $0; "
        "{loop}: "
        "ld.acquire.sys.global.b32 %val, [%addr]; "
        "setp.ge.u32 %p, %val, $2; "
        "@%p bra {done}; "
        "nanosleep.u32 8; "
        "bra {loop}; "
        "{done}: "
        "}}".format(loop=_loop, done=_done),
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def check_barrier_ready(
    barrier_ptr: Int64,
    barrier_idx: Int32,
    expected: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Non-blocking check if a global memory barrier is ready.

    Returns 1 if barrier[barrier_idx] >= expected, 0 otherwise.
    Does NOT spin - returns immediately. Used for speculative pipelining
    to check if dependencies are satisfied without blocking.

    Args:
        barrier_ptr: Pointer to barrier array in global memory (64-bit)
        barrier_idx: Index of barrier to check
        expected: Expected count threshold

    Returns:
        Int32: 1 if barrier >= expected (ready), 0 otherwise (not ready)
    """
    from cutlass._mlir import ir

    result = llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [
            barrier_ptr.ir_value(loc=loc, ip=ip),
            barrier_idx.ir_value(loc=loc, ip=ip),
            expected.ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .pred %p;
            .reg .u32 %val;
            .reg .u64 %addr;
            mad.wide.u32 %addr, $2, 4, $1;
            ld.acquire.sys.global.b32 %val, [%addr];
            setp.ge.u32 %p, %val, $3;
            selp.u32 $0, 1, 0, %p;
        }
        """,
        "=r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def global_barrier_signal(
    barrier_ptr: Int64,
    barrier_idx: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Signal (increment) a global memory barrier.

    Atomically increments barrier[barrier_idx].

    Args:
        barrier_ptr: Pointer to barrier array in global memory (64-bit)
        barrier_idx: Index of barrier to signal
    """
    llvm.inline_asm(
        None,
        [barrier_ptr.ir_value(loc=loc, ip=ip), barrier_idx.ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .u64 %addr;
            mad.wide.u32 %addr, $1, 4, $0;
            atom.add.release.sys.global.u32 _, [%addr], 1;
        }
        """,
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# =============================================================================
# Instruction Stream Access
# =============================================================================


@dsl_user_op
def load_instruction_to_smem(
    instr_ptr: Int64,
    instr_idx: Int32,
    smem_dest: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Load a full instruction (4 words) from global memory to shared memory.

    Uses ld.global.v4.b32 to load all 4 instruction words in a single
    vector load, then st.shared.v4.b32 to write them to shared memory
    in a single store. Both require 16-byte alignment.

    Args:
        instr_ptr: Pointer to instruction array in global memory (64-bit)
        instr_idx: Instruction index (word offset = instr_idx * 4)
        smem_dest: Shared memory destination address (must be 16-byte aligned)
    """
    llvm.inline_asm(
        None,
        [
            instr_ptr.ir_value(loc=loc, ip=ip),
            instr_idx.ir_value(loc=loc, ip=ip),
            smem_dest.ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .u64 %gaddr;
            .reg .u32 %w0, %w1, %w2, %w3;
            mad.wide.u32 %gaddr, $1, 16, $0;
            ld.global.v4.b32 {%w0, %w1, %w2, %w3}, [%gaddr];
            st.shared.v4.b32 [$2], {%w0, %w1, %w2, %w3};
        }
        """,
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def ld_global_i32(
    base_ptr: Int64,
    offset: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Load a 32-bit integer from global memory.

    Reads base_ptr[offset] (i.e., base_ptr + offset * 4).

    Args:
        base_ptr: Base pointer to global memory array (64-bit)
        offset: Element index (multiplied by 4 internally)

    Returns:
        Loaded 32-bit value (Int32)
    """
    from cutlass._mlir import ir

    result = llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [base_ptr.ir_value(loc=loc, ip=ip), offset.ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .u64 %addr;
            mad.wide.u32 %addr, $2, 4, $1;
            ld.global.b32 $0, [%addr];
        }
        """,
        "=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def ld_global_i64(
    base_ptr: Int64,
    offset: Int32,
    *,
    loc=None,
    ip=None,
) -> Int64:
    """Load a 64-bit integer from global memory.

    Reads base_ptr[offset] (i.e., base_ptr + offset * 8).

    Args:
        base_ptr: Base pointer to global memory array (64-bit)
        offset: Element index (multiplied by 8 internally)

    Returns:
        Loaded 64-bit value (Int64)
    """
    from cutlass._mlir import ir

    result = llvm.inline_asm(
        ir.IntegerType.get_signless(64),
        [base_ptr.ir_value(loc=loc, ip=ip), offset.ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .u64 %addr;
            mad.wide.u32 %addr, $2, 8, $1;
            ld.global.b64 $0, [%addr];
        }
        """,
        "=l,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int64(result)


@dsl_user_op
def st_global_i32(
    base_ptr: Int64,
    offset: Int32,
    value: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store a 32-bit integer to global memory.

    Writes value to base_ptr[offset] (i.e., base_ptr + offset * 4).

    Args:
        base_ptr: Base pointer to global memory array (64-bit)
        offset: Element index (word offset, multiplied by 4 internally)
        value: 32-bit value to store
    """
    llvm.inline_asm(
        None,
        [
            base_ptr.ir_value(loc=loc, ip=ip),
            offset.ir_value(loc=loc, ip=ip),
            value.ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .u64 %addr;
            mad.wide.u32 %addr, $1, 4, $0;
            st.global.b32 [%addr], $2;
        }
        """,
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# =============================================================================
# Async Copy Synchronization
# =============================================================================


@dsl_user_op
def cp_async_wait_group(
    n: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Wait until at most N async copy groups remain in flight.

    This is used for N-page pipelining where we want to wait for the oldest
    async copy to complete while keeping newer ones in flight.

    cp.async.wait_group N waits until at most N commit groups are pending.
    - wait_group 0: wait for all (same as cp.async.wait_all)
    - wait_group 1: wait until at most 1 group pending
    - wait_group N: wait until at most N groups pending

    Args:
        n: Maximum number of groups to leave in flight (0 = wait for all)
    """
    # Note: PTX cp.async.wait_group requires an immediate constant.
    # We generate a switch on common values. For very deep pipelines,
    # the kernel loop caps tiles_in_flight anyway.
    llvm.inline_asm(
        None,
        [n.ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred %p0, %p1, %p2, %p3, %p4, %p5, %p6, %p7;
            setp.eq.u32 %p0, $0, 0;
            setp.eq.u32 %p1, $0, 1;
            setp.eq.u32 %p2, $0, 2;
            setp.eq.u32 %p3, $0, 3;
            setp.eq.u32 %p4, $0, 4;
            setp.eq.u32 %p5, $0, 5;
            setp.eq.u32 %p6, $0, 6;
            setp.ge.u32 %p7, $0, 7;
            @%p0 cp.async.wait_group 0;
            @%p1 cp.async.wait_group 1;
            @%p2 cp.async.wait_group 2;
            @%p3 cp.async.wait_group 3;
            @%p4 cp.async.wait_group 4;
            @%p5 cp.async.wait_group 5;
            @%p6 cp.async.wait_group 6;
            @%p7 cp.async.wait_group 7;
        }
        """,
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


__all__ = [
    "global_barrier_wait",
    "global_barrier_signal",
    "check_barrier_ready",
    "load_instruction_to_smem",
    "ld_global_i32",
    "ld_global_i64",
    "st_global_i32",
    "cp_async_wait_group",
]
