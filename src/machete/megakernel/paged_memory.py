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
from typing import Tuple

from cutlass import Int32, Int64
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

from .instruction_layout import INSTRUCTION_ROW_BYTES


# =============================================================================
# Constants
# =============================================================================

MAX_PAGES: int = 16  # Maximum number of pages
IQ_DEPTH: int = 4  # Instruction queue depth for out-of-order loading

# Per-slot tile-info layout in shared memory (int32 slots, plus one int64).
TILE_INFO_OP_IDX: int = 0
TILE_INFO_HANDLER_IDX: int = 1
TILE_INFO_TILE_0: int = 2
TILE_INFO_TILE_1: int = 3
TILE_INFO_TILE_2: int = 4
TILE_INFO_TILE_3: int = 5
TILE_INFO_INSTRUCTION_IDX: int = 6
TILE_INFO_OP_CONFIG: int = 8
TILE_INFO_PAGE_ID: int = 10

# Flag offsets within the flags region (each int32 = 4 bytes).
# Used by controller, loader, and store warps for inter-warp communication.
#
# Warp roles (see megakernel.py _kernel_loop_ring for full protocol):
#   Controller (warp num_mma_warps):     Fetches instructions, manages barriers
#   Loader     (warp num_mma_warps + 1): Dispatches TMA loads
#   Store      (warp num_mma_warps + 2): Dispatches TMA stores after compute
#
# Offsets 0 and 8 are reserved (unused).
FLAG_DISPATCH_LOAD: int = 4    # Controller → Loader: page slot to load next
FLAG_PRODUCE_IDX: int = 16     # Controller: produce counter (read by store warp)
FLAG_STORE_IDX: int = 20       # Store warp: store counter (read by controller)
FLAG_LOAD_DONE: int = 24       # Controller: signals all instructions consumed
FLAG_DATA_RELEASE_IDX: int = 28  # Store warp: physical page release counter
FLAG_DATA_PRODUCE_IDX: int = 32  # Controller: physical page produce counter


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


@dsl_user_op
def st_shared_release_cta_i32(
    smem_addr: Int32,
    value: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Release-store a 32-bit integer to shared memory (CTA scope).

    Ensures all prior shared memory writes are visible to other warps
    before this store becomes visible. Use for inter-warp signaling.
    """
    llvm.inline_asm(
        None,
        [smem_addr.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.release.cta.shared.b32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def ld_shared_acquire_cta_i32(
    smem_addr: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Acquire-load a 32-bit integer from shared memory (CTA scope).

    Ensures all subsequent shared memory reads see writes that were
    visible before the corresponding release-store. Use for inter-warp signaling.
    """
    from cutlass._mlir import ir

    result = llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [smem_addr.ir_value(loc=loc, ip=ip)],
        "ld.acquire.cta.shared.b32 $0, [$1];",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def ld_shared_v2_b32(
    smem_addr: Int32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Int32, Int32]:
    """Load two consecutive 32-bit integers from shared memory (v2)."""
    from cutlass._mlir import ir

    i32 = ir.IntegerType.get_signless(32)
    struct_ty = llvm.StructType.get_literal([i32, i32])
    result = llvm.inline_asm(
        struct_ty,
        [smem_addr.ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.b32 {$0, $1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(i32, result, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(i32, result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def st_shared_v2_b32(
    smem_addr: Int32,
    v0: Int32,
    v1: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store two consecutive 32-bit integers to shared memory (v2)."""
    llvm.inline_asm(
        None,
        [
            smem_addr.ir_value(loc=loc, ip=ip),
            v0.ir_value(loc=loc, ip=ip),
            v1.ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v2.b32 [$0], {$1, $2};",
        "r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def st_shared_i64(
    smem_addr: Int32,
    value: Int64,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store a 64-bit integer to shared memory."""
    llvm.inline_asm(
        None,
        [smem_addr.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "st.shared.b64 [$0], $1;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def ld_shared_i64(
    smem_addr: Int32,
    *,
    loc=None,
    ip=None,
) -> Int64:
    """Load a 64-bit integer from shared memory."""
    from cutlass._mlir import ir

    result = llvm.inline_asm(
        ir.IntegerType.get_signless(64),
        [smem_addr.ir_value(loc=loc, ip=ip)],
        "ld.shared.b64 $0, [$1];",
        "=l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int64(result)


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

    Layout:
        [Scratch: tile_info + flags + IQ + mbarriers] [Page 0..N-1]
    """

    num_pages: int = 2
    page_size: int = 16384
    num_slots: int | None = None

    # Scratch area layout (ring buffer):
    # - Per-slot tile info: num_slots * 48 bytes
    #   [op_idx, handler_idx, tile_0..tile_3, instruction_idx, op_config_ptr, page_id]
    # - Flags: 32 bytes (see FLAG_* constants for active offsets;
    #          offsets 0 and 8 are reserved/unused)
    # - Instruction queue: IQ_DEPTH * replay instruction row bytes
    # - Mbarriers: 2 * num_slots * 8 bytes [work_notify + compute_done]
    _TILE_INFO_SIZE: int = 48  # Per-slot tile metadata with aligned int64 config pointer
    _IQ_ENTRY_SIZE: int = INSTRUCTION_ROW_BYTES
    _FLAGS_SIZE: int = 36  # See FLAG_* constants for layout; includes 2 reserved slots
    _MBARRIER_SIZE: int = 8  # Per mbarrier object (8 bytes, 8-byte aligned)

    def __post_init__(self):
        """Calculate layout offsets after initialization."""
        if self.num_pages < 1:
            raise ValueError(f"num_pages must be >= 1, got {self.num_pages}")
        if self.num_pages > MAX_PAGES:
            raise ValueError(f"num_pages must be <= {MAX_PAGES}, got {self.num_pages}")
        if self.num_slots is None:
            self.num_slots = self.num_pages
        if self.num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {self.num_slots}")
        if self.num_slots > MAX_PAGES:
            raise ValueError(f"num_slots must be <= {MAX_PAGES}, got {self.num_slots}")

        # Scratch region layout: [tile_info][flags][iq][mbarriers]
        self.ring_state_offset = 0
        self.flags_offset = self.num_slots * self._TILE_INFO_SIZE
        self.iq_offset = _align_up(self.flags_offset + self._FLAGS_SIZE, 8)

        # mbarrier array layout (each 8 bytes, MUST be 8-byte aligned per PTX):
        #   work_notify[N]:  DMA→MMA signal: data loaded, ready to compute (1 arrive)
        #   compute_done[N]: MMA→DMA signal: compute finished (num_mma_warps arrives)
        # Phase alternates 0/1 per use (hardware auto-reset).
        self.mbarrier_offset = _align_up(
            self.iq_offset + IQ_DEPTH * self._IQ_ENTRY_SIZE, 8
        )

        num_mbarriers = 2 * self.num_slots  # work_notify + compute_done
        raw_scratch_size = (
            self.mbarrier_offset
            + num_mbarriers * self._MBARRIER_SIZE
        )

        # 128-byte alignment: required for TMA base address alignment (PTX spec)
        self.scratch_size = _align_up(raw_scratch_size, 128)

        # Page layout - pages start right after scratch
        # Page pointer computed as: smem_base + pages_start + page_idx * aligned_page_size
        self.aligned_page_size = _align_up(self.page_size, 128)
        self.pages_start = self.scratch_size

        # Total size
        self.total_size = self.pages_start + self.num_pages * self.aligned_page_size

    def work_notify_mbar_offset(self, slot_idx: int) -> int:
        """Get offset to the work_notify mbarrier for a given work slot."""
        return self.mbarrier_offset + slot_idx * self._MBARRIER_SIZE

    def compute_done_mbar_offset(self, slot_idx: int) -> int:
        """Get offset to the compute_done mbarrier for a given work slot."""
        return self.mbarrier_offset + self.num_slots * self._MBARRIER_SIZE + slot_idx * self._MBARRIER_SIZE

    @classmethod
    def for_device(
        cls,
        page_size: int = 16384,
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
            f"  num_slots={self.num_slots},\n"
            f"  page_size={self.page_size // 1024}KB,\n"
            f"  total_size={self.total_size // 1024}KB,\n"
            f"  scratch_size={self.scratch_size}B,\n"
            f"  pages_start={self.pages_start},\n"
            f"  aligned_page_size={self.aligned_page_size}\n"
            f")"
        )


@dataclass(frozen=True)
class PipelinePageLayout:
    """Layout for pipeline-local storage inside one megakernel page."""

    page_count: int
    page_bytes: int
    semaphore_count: int = 0
    scratch_bytes: int = 0

    def __post_init__(self):
        if self.page_count < 1:
            raise ValueError("page_count must be >= 1")
        if self.page_bytes < 0:
            raise ValueError("page_bytes must be non-negative")
        if self.semaphore_count < 0:
            raise ValueError("semaphore_count must be non-negative")
        if self.scratch_bytes < 0:
            raise ValueError("scratch_bytes must be non-negative")

    @property
    def pages_offset(self) -> int:
        return 0

    @property
    def semaphores_offset(self) -> int:
        return _align_up(self.page_count * self.page_bytes, 8)

    @property
    def scratch_offset(self) -> int:
        return _align_up(
            self.semaphores_offset + self.semaphore_count * NPageLayout._MBARRIER_SIZE,
            16,
        )

    @property
    def total_size(self) -> int:
        return _align_up(self.scratch_offset + self.scratch_bytes, 128)

    def page_offset(self, page_idx: int) -> int:
        if page_idx < 0 or page_idx >= self.page_count:
            raise IndexError(
                f"pipeline page {page_idx} out of range for {self.page_count} pages"
            )
        return self.pages_offset + page_idx * self.page_bytes

    def activation_page_offset(self) -> int:
        return self.page_offset(0)

    def weight_page_offset(
        self,
        *,
        input_stage: int,
        page_in_stage: int,
        input_stages: int,
        stage_pages: int,
    ) -> int:
        if input_stage < 0 or input_stage >= input_stages:
            raise IndexError(
                f"input_stage {input_stage} out of range for {input_stages} stages"
            )
        if page_in_stage < 0 or page_in_stage >= stage_pages:
            raise IndexError(
                f"page_in_stage {page_in_stage} out of range for {stage_pages} pages"
            )
        return self.page_offset(1 + input_stage * stage_pages + page_in_stage)

    def semaphore_offset(self, sem_idx: int) -> int:
        if sem_idx < 0 or sem_idx >= self.semaphore_count:
            raise IndexError(
                f"pipeline semaphore {sem_idx} out of range for {self.semaphore_count} semaphores"
            )
        return self.semaphores_offset + sem_idx * NPageLayout._MBARRIER_SIZE

    @staticmethod
    def activations_arrived_sem() -> int:
        return 0

    @staticmethod
    def weights_arrived_sem(input_stage: int) -> int:
        return 1 + input_stage

    @staticmethod
    def weights_finished_sem(input_stages: int, input_stage: int) -> int:
        return 1 + input_stages + input_stage

    @staticmethod
    def outputs_arrived_sem(input_stages: int, output_stage: int) -> int:
        return 1 + 2 * input_stages + output_stage

    @staticmethod
    def outputs_finished_sem(
        input_stages: int,
        output_stages: int,
        output_stage: int,
    ) -> int:
        return 1 + 2 * input_stages + output_stages + output_stage

    def output_scratch_offset(
        self,
        *,
        output_stage: int,
        output_stages: int,
    ) -> int:
        if output_stage < 0 or output_stage >= output_stages:
            raise IndexError(
                f"output_stage {output_stage} out of range for {output_stages} stages"
            )
        stage_bytes = self.scratch_bytes // max(1, output_stages)
        return self.scratch_offset + output_stage * stage_bytes


__all__ = [
    "MAX_PAGES",
    "IQ_DEPTH",
    "FLAG_DISPATCH_LOAD",
    "FLAG_PRODUCE_IDX",
    "FLAG_STORE_IDX",
    "FLAG_LOAD_DONE",
    "st_shared_i32",
    "ld_shared_i32",
    "st_shared_release_cta_i32",
    "ld_shared_acquire_cta_i32",
    "ld_shared_v2_b32",
    "st_shared_v2_b32",
    "NPageLayout",
    "PipelinePageLayout",
]
