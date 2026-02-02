# Copyright (c) 2025, Machete Authors
"""
Paged Shared Memory Manager for Megakernel.

This module implements a circular buffer paging system for shared memory
management within the megakernel. Key features:

1. Physical Pages: Divide shared memory into fixed-size pages (16KB each)
2. Logical-to-Physical Mapping: Instructions reference logical page IDs (LID)
   which are dynamically mapped to physical page IDs (PID)
3. Page Release/Acquire: Automatic rotation as instructions complete

Shared Memory Layout:
    [Control/Scratch] -> [Page Table Config + States + Free List]
    -> [Page Data]
"""

from dataclasses import dataclass

import cutlass.cute as cute
from cutlass import Int32
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


# =============================================================================
# Constants
# =============================================================================

PAGE_SIZE: int = 16 * 1024  # 16KB per page
MAX_PAGES: int = 16  # Maximum number of pages


# =============================================================================
# Shared Memory Load/Store Primitives (Device Side)
# =============================================================================


@dsl_user_op
def st_shared_i32(
    smem_addr: Int32,
    value: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store a 32-bit integer to shared memory."""
    llvm.inline_asm(
        None,
        [smem_addr.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.shared.b32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def ld_shared_i32(
    smem_addr: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Load a 32-bit integer from shared memory."""
    from cutlass._mlir import ir

    result = llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [smem_addr.ir_value(loc=loc, ip=ip)],
        "ld.shared.b32 $0, [$1];",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


# =============================================================================
# Page Table Configuration (Host Side)
# =============================================================================


@dataclass
class PageTableConfig:
    """Configuration for the page table.

    This is the host-side mirror of what gets written into shared memory
    at page_table_offset. Layout in smem (12 bytes):
        [0]  num_pages     (int32)
        [4]  page_size     (int32)
        [8]  base_offset   (int32) — offset from smem_base to page data

    Attributes:
        num_pages: Total number of pages
        page_size: Size of each page in bytes
        base_offset: Offset from start of shared memory to page data
    """

    num_pages: int = 4
    page_size: int = PAGE_SIZE
    base_offset: int = 0


# =============================================================================
# Page Table Initialization (Device Side)
# =============================================================================


@cute.jit
def init_page_table(
    smem_base: Int32,
    page_table_offset: Int32,
    num_pages: Int32,
    page_size: Int32,
    page_data_offset: Int32,
) -> None:
    """Write PageTableConfig into shared memory. Called by thread 0.

    Args:
        smem_base: Base address of shared memory
        page_table_offset: Offset to page table region
        num_pages: Number of physical pages
        page_size: Size of each page in bytes
        page_data_offset: Offset from smem_base to page data region
    """
    config_ptr = smem_base + page_table_offset
    st_shared_i32(config_ptr, num_pages)
    st_shared_i32(config_ptr + 4, page_size)
    st_shared_i32(config_ptr + 8, page_data_offset)


@cute.jit
def init_page_states(
    smem_base: Int32,
    page_table_offset: Int32,
    num_pages: Int32,
) -> None:
    """Initialize page states and free list. Called by thread 0.

    Page states are stored after PageTableConfig (12 bytes).
    Each PageState is 16 bytes: [owner(4), reserved(12)].
    Free list follows: num_pages * 4 bytes + head(4) + tail(4).

    All pages start as free (owner = -1), free list = [0, 1, ..., N-1].

    Args:
        smem_base: Base address of shared memory
        page_table_offset: Offset to page table region
        num_pages: Number of physical pages
    """
    # PageState array starts after config (12 bytes)
    page_state_ptr = smem_base + page_table_offset + 12

    # Free list starts after page states
    free_list_ptr = page_state_ptr + num_pages * 16

    # Head and tail pointers after free list
    free_list_head_ptr = free_list_ptr + num_pages * 4
    free_list_tail_ptr = free_list_head_ptr + 4

    i = Int32(0)
    while i < num_pages:
        # Mark page as free (owner = -1)
        st_shared_i32(page_state_ptr + i * 16, Int32(-1))
        # Initialize free list entry
        st_shared_i32(free_list_ptr + i * 4, i)
        i = i + Int32(1)

    # Head = 0 (next page to acquire), tail = 0 (next slot to release into)
    # With all N pages in the list, head==tail means full (we track count implicitly
    # by only acquiring what we release)
    st_shared_i32(free_list_head_ptr, Int32(0))
    st_shared_i32(free_list_tail_ptr, Int32(0))


# =============================================================================
# Page Pointer Access (Device Side)
# =============================================================================


@cute.jit
def get_page_data_ptr(smem_base: Int32, page_table_config_ptr: Int32, page_id: Int32) -> Int32:
    """Get pointer to data region of a physical page.

    Args:
        smem_base: Base address of shared memory
        page_table_config_ptr: Pointer to PageTableConfig in shared memory
        page_id: Physical page ID

    Returns:
        Shared memory pointer to page data
    """
    # Config layout: [num_pages, page_size, base_offset]
    page_size = ld_shared_i32(page_table_config_ptr + 4)
    base_offset = ld_shared_i32(page_table_config_ptr + 8)
    return smem_base + base_offset + page_id * page_size


# =============================================================================
# Shared Memory Layout Calculator
# =============================================================================


@dataclass
class SharedMemoryLayout:
    """Calculates and stores the shared memory layout for megakernel.

    Layout (all regions 128-byte aligned):
        [Control/Scratch]  — scratch area for broadcasting page IDs
        [Page Table]       — PageTableConfig (12B) + PageState[] + free list
        [Page Data]        — num_pages * page_size

    Note: Global barriers live in global memory (_barriers_tensor),
    not in shared memory.
    """

    def __init__(
        self,
        num_pages: int = 4,
        page_size: int = PAGE_SIZE,
    ):
        self.num_pages = num_pages
        self.page_size = page_size
        self._calculate_layout()

    def _calculate_layout(self):
        """Calculate memory layout offsets."""
        offset = 0

        # Control/scratch region: used to broadcast page IDs from thread 0
        # Size: MAX_PAGES * 4 bytes (one int32 per page ID slot)
        self.control_offset = offset
        control_size = MAX_PAGES * 4
        control_size = (control_size + 127) // 128 * 128
        offset += control_size

        # Page table: config + page states + free list
        self.page_table_offset = offset
        # PageTableConfig: 12 bytes
        # PageState array: num_pages * 16 bytes (owner + 12 reserved)
        # Free list: num_pages * 4 bytes
        # Free list head + tail: 8 bytes
        page_table_size = 12 + self.num_pages * 16 + self.num_pages * 4 + 8
        page_table_size = (page_table_size + 127) // 128 * 128
        offset += page_table_size

        # Page data
        self.page_data_offset = offset
        page_data_size = self.num_pages * self.page_size
        offset += page_data_size

        self.total_size = offset

    def to_config(self) -> PageTableConfig:
        """Create PageTableConfig from layout."""
        return PageTableConfig(
            num_pages=self.num_pages,
            page_size=self.page_size,
            base_offset=self.page_data_offset,
        )

    def __repr__(self) -> str:
        return (
            f"SharedMemoryLayout(\n"
            f"  total_size={self.total_size / 1024:.1f}KB,\n"
            f"  control_offset={self.control_offset},\n"
            f"  page_table_offset={self.page_table_offset},\n"
            f"  page_data_offset={self.page_data_offset},\n"
            f"  num_pages={self.num_pages},\n"
            f"  page_size={self.page_size / 1024:.1f}KB\n"
            f")"
        )


__all__ = [
    "PAGE_SIZE",
    "MAX_PAGES",
    "st_shared_i32",
    "ld_shared_i32",
    "PageTableConfig",
    "init_page_table",
    "init_page_states",
    "get_page_data_ptr",
    "SharedMemoryLayout",
]
