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
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

# Counter for unique PTX labels across inline asm invocations.
# PTX labels are function-scoped (not block-scoped), so multiple
# inlined copies of the same asm template would collide without
# unique names.
_ptx_label_counter = 0

# The instruction tensor is stored as [op_idx, linear_tile_idx, tile_0..tile_4]
# i.e. 7 int32 words per row.
_INSTRUCTION_ROW_STRIDE_BYTES = 7 * 4


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
        "ld.acquire.gpu.global.b32 %val, [%addr]; "
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
            ld.acquire.gpu.global.b32 %val, [%addr];
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


@dsl_user_op
def check_barrier_ready_gpu(
    barrier_ptr: Int64,
    barrier_idx: Int32,
    expected: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Non-blocking barrier check with GPU scope (cheaper than sys scope).

    Same as check_barrier_ready but uses .gpu scope instead of .sys.
    Sufficient for intra-GPU barriers (not cross-GPU NVLink).
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
            ld.acquire.gpu.global.b32 %val, [%addr];
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
def global_barrier_signal_gpu(
    barrier_ptr: Int64,
    barrier_idx: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Signal (increment) a global barrier with GPU scope (cheaper than sys scope).

    Same as global_barrier_signal but uses .gpu scope instead of .sys.
    Sufficient for intra-GPU barriers (not cross-GPU NVLink).
    """
    llvm.inline_asm(
        None,
        [barrier_ptr.ir_value(loc=loc, ip=ip), barrier_idx.ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .u64 %addr;
            mad.wide.u32 %addr, $1, 4, $0;
            atom.add.release.gpu.global.u32 _, [%addr], 1;
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
    """Load an instruction header (2 words) from global memory to shared memory.

    The instruction tensor is laid out as
    ``[num_instructions, INSTRUCTION_WORDS]`` int32 values, but only the
    first two words (``op_idx`` and ``linear_tile_idx``) are needed here.
    This helper therefore loads those two words from the start of the
    selected row, using the full row stride. The row stride is 28 bytes,
    so the second row is only 4-byte aligned; use scalar loads rather than
    ``ld.global.v2.b32`` which requires 8-byte alignment.

    Args:
        instr_ptr: Pointer to instruction array in global memory (64-bit)
        instr_idx: Instruction row index
        smem_dest: Shared memory destination address (must be 8-byte aligned)
    """
    llvm.inline_asm(
        None,
        [
            instr_ptr.ir_value(loc=loc, ip=ip),
            instr_idx.ir_value(loc=loc, ip=ip),
            smem_dest.ir_value(loc=loc, ip=ip),
            Int32(_INSTRUCTION_ROW_STRIDE_BYTES).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .u64 %gaddr;
            .reg .u32 %w0, %w1;
            mad.wide.u32 %gaddr, $1, $3, $0;
            ld.global.b32 %w0, [%gaddr];
            ld.global.b32 %w1, [%gaddr+4];
            st.shared.v2.b32 [$2], {%w0, %w1};
        }
        """,
        "l,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
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
def mbarrier_arrive_expect_tx(
    smem_addr: Int32,
    tx_bytes: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Signal arrival at an mbarrier with expected async transaction bytes.

    Combines arrival with setting the expected number of bytes that will
    be delivered by async copy operations (cp.async.bulk). The mbarrier
    only completes when both all arrivals have occurred AND all expected
    bytes have been delivered.

    Used by async G2S bulk copy loads that signal the mbarrier via
    cute.copy(..., mbar_ptr=...).

    Args:
        smem_addr: Shared memory address of the mbarrier
        tx_bytes: Number of bytes expected from async transactions
    """
    llvm.inline_asm(
        None,
        [smem_addr.ir_value(loc=loc, ip=ip), tx_bytes.ir_value(loc=loc, ip=ip)],
        "mbarrier.arrive.expect_tx.shared.b64 _, [$0], $1;",
        "r,r",
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
def mbarrier_inval(
    smem_addr: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Invalidate an mbarrier object, releasing it for regular smem reuse.

    After invalidation, the shared memory locations occupied by the mbarrier
    can be safely overwritten with regular data (e.g., by TMA loads for a
    different op reusing the same page).

    Must only be called when no thread has pending arrive/wait operations
    on this mbarrier.

    Args:
        smem_addr: Shared memory address of the mbarrier (8-byte aligned)
    """
    llvm.inline_asm(
        None,
        [smem_addr.ir_value(loc=loc, ip=ip)],
        "mbarrier.inval.shared.b64 [$0];",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


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


# =============================================================================
# Shared Memory Base Pointer
# =============================================================================


@dsl_user_op
def get_smem_base_ptr(*, loc=None, ip=None) -> Int32:
    """Get the base pointer to shared memory using PTX.

    Returns:
        32-bit unsigned integer address of shared memory base.
    """
    result = llvm.inline_asm(
        T.i32(),
        [],
        """
        {
            .reg .u64 smem_ptr64;
            cvta.shared.u64 smem_ptr64, 0;
            cvt.u32.u64 $0, smem_ptr64;
        }
        """,
        "=r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def prefetch_instruction(
    instr_ptr: Int64,
    instr_idx: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Prefetch an instruction from global memory into L2 cache.

    Issues a ``prefetch.global.L2`` for the instruction at the given index.
    This is a non-blocking hint — the hardware starts fetching the cache line
    so that a subsequent ``ld.global`` hits L2 (~30-50 cycles) instead of
    DRAM (~200-400 cycles).

    Args:
        instr_ptr: Pointer to instruction array in global memory (64-bit)
        instr_idx: Instruction index (byte offset = instr_idx * 8)
    """
    llvm.inline_asm(
        None,
        [
            instr_ptr.ir_value(loc=loc, ip=ip),
            instr_idx.ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .u64 %gaddr;
            mad.wide.u32 %gaddr, $1, 8, $0;
            prefetch.global.L2 [%gaddr];
        }
        """,
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


__all__ = [
    "global_barrier_wait",
    "global_barrier_signal",
    "global_barrier_signal_gpu",
    "check_barrier_ready",
    "check_barrier_ready_gpu",
    "load_instruction_to_smem",
    "prefetch_instruction",
    "ld_global_i32",
    "ld_global_i64",
    "st_global_i32",
    # mbarrier primitives
    "mbarrier_init",
    "mbarrier_init_fence",
    "mbarrier_inval",
    "mbarrier_arrive",
    "mbarrier_arrive_expect_tx",
    "mbarrier_wait",
    "mbarrier_try_wait",
    # Named barrier
    "named_barrier_sync",
    # Shared memory
    "get_smem_base_ptr",
    # Misc
    "nanosleep",
]
