# Copyright (c) 2025, Machete Authors
"""
Async Copy Utilities for Memory/Compute Overlap
================================================

Provides PTX-based async copy primitives for overlapping global memory loads
with computation using shared memory staging.

cp.async enables the memory subsystem to perform loads independently of the
compute pipeline. Combined with mbarrier synchronization, this allows:
1. Load row N+1 to shared memory (async)
2. Compute on row N from shared memory
3. Wait for row N+1 load to complete
4. Swap buffers and repeat

Usage:
    from machete.kernels.utils.async_copy import (
        cp_async_f32, cp_async_commit, cp_async_wait_all
    )

    # Async load 4 floats from gmem to smem
    cp_async_f32(smem_ptr, gmem_ptr)

    # Commit all pending async copies
    cp_async_commit()

    # Wait for all committed copies to complete
    cp_async_wait_all()
"""

from cutlass import Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def cp_async_f32(
    smem_base: Int32,
    smem_offset: Int32,
    gmem_ptr: Int64,
    *,
    loc=None,
    ip=None,
) -> None:
    """Async copy 4 bytes (1 float32) from global to shared memory.

    Uses cp.async.ca.shared.global with cache-all policy.
    The copy is asynchronous and must be followed by cp_async_commit()
    and cp_async_wait_all() to ensure completion.

    Args:
        smem_base: Shared memory base address (32-bit, from alloc_smem)
        smem_offset: Byte offset from base (32-bit)
        gmem_ptr: Global memory source address (64-bit)
    """
    llvm.inline_asm(
        None,
        [smem_base.ir_value(loc=loc, ip=ip),
         smem_offset.ir_value(loc=loc, ip=ip),
         gmem_ptr.ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .u32 %smem_addr;
            add.u32 %smem_addr, $0, $1;
            cp.async.ca.shared.global [%smem_addr], [$2], 4;
        }
        """,
        "r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cp_async_f32x4(
    smem_ptr: Int32,
    gmem_ptr: Int64,
    *,
    loc=None,
    ip=None,
) -> None:
    """Async copy 16 bytes (4 float32s) from global to shared memory.

    Uses cp.async.ca.shared.global with cache-all policy.
    Both pointers must be 16-byte aligned for optimal performance.

    Args:
        smem_ptr: Shared memory destination address (32-bit, 16-byte aligned)
        gmem_ptr: Global memory source address (64-bit, 16-byte aligned)
    """
    llvm.inline_asm(
        None,
        [smem_ptr.ir_value(loc=loc, ip=ip), gmem_ptr.ir_value(loc=loc, ip=ip)],
        """
        {
            cp.async.ca.shared.global [%0], [%1], 16;
        }
        """,
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cp_async_commit(*, loc=None, ip=None) -> None:
    """Commit all pending async copies to a group.

    Creates a synchronization point for all previous cp.async operations.
    Must be called before cp_async_wait_all().
    """
    llvm.inline_asm(
        None,
        [],
        """
        {
            cp.async.commit_group;
        }
        """,
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cp_async_wait_all(*, loc=None, ip=None) -> None:
    """Wait for all committed async copy groups to complete.

    Blocks until all previously committed cp.async operations have finished.
    """
    llvm.inline_asm(
        None,
        [],
        """
        {
            cp.async.wait_all;
        }
        """,
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cp_async_wait_group(n: int, *, loc=None, ip=None) -> None:
    """Wait until at most N async copy groups are pending.

    Blocks until at most N previously committed groups remain in-flight.
    cp_async_wait_group(0) is equivalent to cp_async_wait_all().

    Args:
        n: Maximum number of pending groups (compile-time constant)
    """
    llvm.inline_asm(
        None,
        [],
        f"""
        {{
            cp.async.wait_group {n};
        }}
        """,
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def smem_ptr_to_int(smem_addr, *, loc=None, ip=None) -> Int32:
    """Convert a shared memory pointer/address to a 32-bit integer.

    Useful for passing smem addresses to PTX inline assembly that
    expects raw integer addresses.

    Args:
        smem_addr: Shared memory address (from alloc_smem or pointer arithmetic)

    Returns:
        32-bit integer representation of the address
    """
    # If smem_addr is already an Int32, just return it
    if isinstance(smem_addr, Int32):
        return smem_addr
    # Otherwise try to convert
    return Int32(smem_addr.toint(loc=loc, ip=ip).ir_value())


__all__ = [
    "cp_async_f32",
    "cp_async_f32x4",
    "cp_async_commit",
    "cp_async_wait_all",
    "cp_async_wait_group",
    "smem_ptr_to_int",
]
