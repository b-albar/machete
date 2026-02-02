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


__all__ = [
    "global_barrier_wait",
    "global_barrier_signal",
    "load_instruction_to_smem",
    "ld_global_i32",
    "ld_global_i64",
    "st_global_i32",
]


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
