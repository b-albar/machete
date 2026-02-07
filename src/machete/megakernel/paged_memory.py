# Copyright (c) 2025, Machete Authors
"""
Paged Shared Memory Manager for Megakernel.

This module provides shared memory layout calculation for pipelined execution:

1. NPageLayout: N-page ring buffer for overlapping loads with compute
2. Shared memory primitives: st_shared_i32, ld_shared_i32

Layout:
    [Scratch] -> [Page 0..N-1]

With N pages, up to N-1 loads can overlap with compute.
"""

from dataclasses import dataclass

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
# N-Page Buffer Layout (Generalized Pipelined Execution)
# =============================================================================


def _align_up(value: int, alignment: int) -> int:
    """Round up value to next multiple of alignment."""
    return (value + alignment - 1) // alignment * alignment


@dataclass
class NPageLayout:
    """N-page shared memory layout for pipelined execution.

    Provides a flexible memory layout where N pages are allocated based on
    available shared memory. With N pages, up to N-1 loads can overlap with
    compute, improving performance for memory-bound operations.

    Layout (all regions 128-byte aligned):
        [Scratch]       — Instruction decode area + ring buffer metadata
        [Page 0..N-1]   — N data pages (computed via pages_start + idx * aligned_page_size)

    The number of pages is determined by:
    1. User-specified num_pages (if provided)
    2. Otherwise, maximum that fits in max_smem

    Usage:
        # Auto-detect max pages for GPU
        layout = NPageLayout.for_device(page_size=16384)

        # Force specific number of pages
        layout = NPageLayout(num_pages=4, page_size=16384)

    Attributes:
        num_pages: Number of pages (2 to MAX_PAGES)
        page_size: Size of each page in bytes
        scratch_size: Size of scratch area (128-byte aligned)
        pages_start: Offset to first page (for arithmetic computation)
        aligned_page_size: Page size aligned to 128 bytes
        total_size: Total shared memory required
    """

    num_pages: int = 2
    page_size: int = PAGE_SIZE

    # Scratch area layout (ring buffer):
    # - Per-page tile info: num_pages * 16 bytes [op_idx, tile_m, tile_n, tile_l] per page
    # - Current instruction: 16 bytes [op_idx, tile_m, tile_n, tile_l]
    # - Flags: 4 bytes [done flag for warp-uniform loop exit]
    # - Mbarriers: 2 * num_pages * 8 bytes [work_notify + compute_done]
    _TILE_INFO_SIZE: int = 16  # Per-page: op_idx, tile_m, tile_n, tile_l
    _INSTR_SIZE: int = 16
    _FLAGS_SIZE: int = 4  # Single int32 done flag
    _MBARRIER_SIZE: int = 8  # Per mbarrier object (8 bytes, 8-byte aligned)

    def __post_init__(self):
        """Calculate layout offsets after initialization."""
        if self.num_pages < 2:
            raise ValueError(f"num_pages must be >= 2, got {self.num_pages}")
        if self.num_pages > MAX_PAGES:
            raise ValueError(f"num_pages must be <= {MAX_PAGES}, got {self.num_pages}")

        # Scratch region layout: [tile_info][instr][flags][mbarriers]
        self.ring_state_offset = 0
        self.instr_offset = self.num_pages * self._TILE_INFO_SIZE
        self.flags_offset = self.instr_offset + self._INSTR_SIZE

        # mbarrier array: 2 * num_pages mbarriers (work_notify[0..N-1], compute_done[0..N-1])
        # Each mbarrier is 8 bytes and MUST be 8-byte aligned (PTX requirement).
        self.mbarrier_offset = _align_up(
            self.flags_offset + self._FLAGS_SIZE, 8
        )

        raw_scratch_size = (
            self.mbarrier_offset
            + 2 * self.num_pages * self._MBARRIER_SIZE
        )
        self.scratch_size = _align_up(raw_scratch_size, 128)

        # Page layout - pages start right after scratch
        # Page pointer computed as: smem_base + pages_start + page_idx * aligned_page_size
        self.aligned_page_size = _align_up(self.page_size, 128)
        self.pages_start = self.scratch_size

        # Total size
        self.total_size = self.pages_start + self.num_pages * self.aligned_page_size

    def get_page_offset(self, page_idx: int) -> int:
        """Get offset for a specific page."""
        if page_idx < 0 or page_idx >= self.num_pages:
            raise ValueError(
                f"Invalid page_idx {page_idx}, must be 0 to {self.num_pages - 1}"
            )
        return self.pages_start + page_idx * self.aligned_page_size

    def work_notify_mbar_offset(self, slot_idx: int) -> int:
        """Get offset to the work_notify mbarrier for a given work slot."""
        return self.mbarrier_offset + slot_idx * self._MBARRIER_SIZE

    def compute_done_mbar_offset(self, page_idx: int) -> int:
        """Get offset to the compute_done mbarrier for a given page."""
        return self.mbarrier_offset + self.num_pages * self._MBARRIER_SIZE + page_idx * self._MBARRIER_SIZE

    @classmethod
    def for_device(
        cls,
        page_size: int = PAGE_SIZE,
        max_smem: int | None = None,
        min_pages: int = 2,
    ) -> "NPageLayout":
        """Create layout with maximum pages that fit in device shared memory.

        Args:
            page_size: Size of each page in bytes
            max_smem: Maximum shared memory (None = auto-detect from GPU)
            min_pages: Minimum number of pages required

        Returns:
            NPageLayout configured for maximum pages
        """
        import torch

        if max_smem is None:
            if not torch.cuda.is_available():
                # Default for CPU testing
                max_smem = 228 * 1024  # 228KB (Hopper default)
            else:
                props = torch.cuda.get_device_properties(torch.cuda.current_device())
                max_smem = props.shared_memory_per_block_optin

        # Find max pages that fit (search from max down)
        aligned_page_size = _align_up(page_size, 128)

        # Estimate scratch overhead (conservative)
        scratch_overhead = 256 + MAX_PAGES * 4  # Ring state + page offsets

        available = max_smem - scratch_overhead
        max_possible = min(available // aligned_page_size, MAX_PAGES)

        if max_possible < min_pages:
            raise ValueError(
                f"Cannot fit {min_pages} pages of {page_size // 1024}KB in "
                f"{max_smem // 1024}KB shared memory"
            )

        # Try from max down to find largest that actually fits
        for n in range(max_possible, min_pages - 1, -1):
            layout = cls(num_pages=n, page_size=page_size)
            if layout.total_size <= max_smem:
                return layout

        # Fallback to minimum
        return cls(num_pages=min_pages, page_size=page_size)

    def __repr__(self) -> str:
        return (
            f"NPageLayout(\n"
            f"  num_pages={self.num_pages},\n"
            f"  page_size={self.page_size // 1024}KB,\n"
            f"  total_size={self.total_size // 1024}KB,\n"
            f"  scratch_size={self.scratch_size}B,\n"
            f"  pages_start={self.pages_start},\n"
            f"  aligned_page_size={self.aligned_page_size}\n"
            f")"
        )



__all__ = [
    "PAGE_SIZE",
    "MAX_PAGES",
    "st_shared_i32",
    "ld_shared_i32",
    "NPageLayout",
]
