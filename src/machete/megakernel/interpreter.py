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


# =============================================================================
# mbarrier Primitives (SM90+ Hardware Barriers)
# =============================================================================


@dsl_user_op
def mbarrier_init(
    smem_addr: Int32,
    expected_count: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Initialize an mbarrier object in shared memory.

    Sets the expected arrival count for the barrier. Must be called by
    exactly one thread before any arrive/wait operations.

    Args:
        smem_addr: Shared memory address of the mbarrier (8-byte aligned, 8 bytes)
        expected_count: Number of arrivals expected before the barrier triggers
    """
    llvm.inline_asm(
        None,
        [smem_addr.ir_value(loc=loc, ip=ip), expected_count.ir_value(loc=loc, ip=ip)],
        """
        {
            mbarrier.init.shared.b64 [$0], $1;
        }
        """,
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def mbarrier_init_fence(
    *,
    loc=None,
    ip=None,
) -> None:
    """Fence after mbarrier initialization.

    Issues a fence.proxy.async to ensure mbarrier initialization is visible
    to all threads before any arrive/wait operations. Must be called after
    all mbarrier_init calls and before any mbarrier_arrive/mbarrier_wait.
    """
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.async.shared::cta;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def mbarrier_arrive(
    smem_addr: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Signal arrival at an mbarrier.

    Decrements the expected arrival count by 1. When all expected arrivals
    have occurred, the barrier phase flips and waiters are released.

    Args:
        smem_addr: Shared memory address of the mbarrier
    """
    llvm.inline_asm(
        None,
        [smem_addr.ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b64 %state;
            mbarrier.arrive.shared.b64 %state, [$0];
        }
        """,
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def mbarrier_wait(
    smem_addr: Int32,
    phase: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Wait for an mbarrier to complete the given phase.

    Spins using mbarrier.try_wait.parity until the barrier's phase matches
    the expected phase. The phase alternates (0/1) each time the barrier
    completes a full cycle of arrivals.

    Args:
        smem_addr: Shared memory address of the mbarrier
        phase: Expected phase (0 or 1). Flips each time the barrier completes.
    """
    global _ptx_label_counter
    _ptx_label_counter += 1
    _loop = "LBB_mbar_wait_loop_{}".format(_ptx_label_counter)
    _done = "LBB_mbar_wait_done_{}".format(_ptx_label_counter)

    llvm.inline_asm(
        None,
        [smem_addr.ir_value(loc=loc, ip=ip), phase.ir_value(loc=loc, ip=ip)],
        "{{ "
        ".reg .pred %p; "
        "{loop}: "
        "mbarrier.try_wait.parity.shared.b64 %p, [$0], $1; "
        "@%p bra {done}; "
        "nanosleep.u32 8; "
        "bra {loop}; "
        "{done}: "
        "}}".format(loop=_loop, done=_done),
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def mbarrier_try_wait(
    smem_addr: Int32,
    phase: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Non-blocking test if an mbarrier has completed the given phase.

    Returns 1 if the barrier's phase has advanced past the expected phase,
    0 otherwise. Does NOT spin — returns immediately.

    Args:
        smem_addr: Shared memory address of the mbarrier
        phase: Expected phase (0 or 1)

    Returns:
        Int32: 1 if phase completed (ready), 0 if still pending
    """
    from cutlass._mlir import ir

    result = llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [smem_addr.ir_value(loc=loc, ip=ip), phase.ir_value(loc=loc, ip=ip)],
        "{ "
        ".reg .pred %p; "
        "mbarrier.try_wait.parity.shared.b64 %p, [$1], $2; "
        "selp.u32 $0, 1, 0, %p; "
        "}",
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def nanosleep(
    duration: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Sleep for approximately the given number of nanoseconds.

    Used to reduce power consumption when the DMA warp has no work to do.

    Args:
        duration: Sleep duration in nanoseconds (approximate)
    """
    llvm.inline_asm(
        None,
        [duration.ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# =============================================================================
# Named Barrier (Thread Subset Synchronization)
# =============================================================================


@dsl_user_op
def named_barrier_sync(
    barrier_id: Int32,
    thread_count: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Synchronize a subset of threads using a named barrier.

    Uses PTX bar.sync to synchronize exactly thread_count threads on the
    given barrier ID. Unlike __syncthreads() which syncs ALL threads in a
    block, this only waits for the specified number of threads to arrive.

    Used to replace __syncthreads() in warp-specialized kernels where the
    DMA warp doesn't participate in compute synchronization.

    Args:
        barrier_id: Barrier ID (0-15). 0 is equivalent to __syncthreads when
            thread_count equals block size. Use 1+ for compute-only barriers.
        thread_count: Number of threads that must arrive before any can proceed.
    """
    llvm.inline_asm(
        None,
        [barrier_id.ir_value(loc=loc, ip=ip), thread_count.ir_value(loc=loc, ip=ip)],
        "bar.sync $0, $1;",
        "r,r",
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
    # mbarrier primitives
    "mbarrier_init",
    "mbarrier_init_fence",
    "mbarrier_arrive",
    "mbarrier_wait",
    "mbarrier_try_wait",
    # Named barrier
    "named_barrier_sync",
    # Misc
    "nanosleep",
]
