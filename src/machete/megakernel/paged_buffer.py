# Copyright (c) 2025, Machete Authors
"""
Paged Buffer System for Megakernel Producer-Consumer Pipelines.

This module implements a circular buffer system with semaphore-based
synchronization for warp-specialized kernels. It enables the "No Bubbles"
pattern where loader, consumer, and storer warps operate concurrently
on different buffer stages.

Key concepts:
- Pages: Fixed-size regions of shared memory for data staging
- Stages: Multiple pages enabling double/triple buffering
- Semaphores: Producer-consumer synchronization per stage

Memory Layout (example with 2 stages):
    +------------------+------------------+------------------+
    | Stage 0 Data     | Stage 1 Data     | Semaphores       |
    | (page_size)      | (page_size)      | (num_sems * 4B)  |
    +------------------+------------------+------------------+

Semaphore Protocol:
    - loader_ready[stage]: Loader signals "data loaded into stage"
    - consumer_done[stage]: Consumer signals "done processing stage"

    Loader:  wait(consumer_done[s]) -> load -> signal(loader_ready[s])
    Consumer: wait(loader_ready[s]) -> compute -> signal(consumer_done[s])
"""

from dataclasses import dataclass
from typing import Tuple
import cutlass.cute as cute
from cutlass import Int32, const_expr

from machete.megakernel.utils import (
    semaphore_init,
    semaphore_signal,
    semaphore_wait,
    semaphore_try_wait,
)


@dataclass
class PagedBufferConfig:
    """Configuration for a paged buffer system.

    Attributes:
        num_stages: Number of buffer stages (2 = double buffering, 3 = triple)
        page_size: Size of each data page in bytes
        element_type: CuTe dtype for the buffer elements (e.g., cute.Float16)
    """
    num_stages: int = 2
    page_size: int = 16384  # 16KB default
    element_type: type = None  # Set at runtime

    def __post_init__(self):
        if self.element_type is None:
            self.element_type = cute.Float16

    @property
    def element_size(self) -> int:
        """Size of each element in bytes."""
        return self.element_type.width // 8

    @property
    def elements_per_page(self) -> int:
        """Number of elements that fit in one page."""
        return self.page_size // self.element_size

    @property
    def num_semaphores(self) -> int:
        """Number of semaphores needed.

        We need 2 semaphores per stage:
        - loader_ready[stage]: loader -> consumer handoff
        - consumer_done[stage]: consumer -> loader handoff (for reuse)
        """
        return self.num_stages * 2

    @property
    def semaphore_bytes(self) -> int:
        """Total bytes needed for semaphores."""
        return self.num_semaphores * 4  # 4 bytes per semaphore (int32)

    @property
    def total_smem_bytes(self) -> int:
        """Total shared memory needed for this buffer configuration."""
        return (self.num_stages * self.page_size) + self.semaphore_bytes


class PagedBufferLayout:
    """Describes the memory layout of a paged buffer in shared memory.

    This class computes offsets and provides accessors for the paged buffer
    system. It's used at compile time to generate the correct memory access
    patterns.

    Layout:
        [Stage 0 Data][Stage 1 Data]...[Stage N-1 Data][Semaphores]

    Semaphore array layout:
        [loader_ready_0, loader_ready_1, ..., consumer_done_0, consumer_done_1, ...]
    """

    def __init__(self, config: PagedBufferConfig, base_offset: int = 0):
        """Initialize the buffer layout.

        Args:
            config: Buffer configuration
            base_offset: Starting offset in shared memory (bytes)
        """
        self.config = config
        self.base_offset = base_offset

        # Compute offsets
        self._data_offset = base_offset
        self._sem_offset = base_offset + (config.num_stages * config.page_size)

    def get_page_offset(self, stage: int) -> int:
        """Get byte offset of a data page.

        Args:
            stage: Stage index (0 to num_stages-1)

        Returns:
            Byte offset from shared memory base
        """
        return self._data_offset + (stage * self.config.page_size)

    def get_loader_ready_sem_offset(self, stage: int) -> int:
        """Get byte offset of loader_ready semaphore for a stage."""
        return self._sem_offset + (stage * 4)

    def get_consumer_done_sem_offset(self, stage: int) -> int:
        """Get byte offset of consumer_done semaphore for a stage."""
        return self._sem_offset + (self.config.num_stages * 4) + (stage * 4)

    @property
    def total_bytes(self) -> int:
        """Total bytes used by this buffer layout."""
        return self.config.total_smem_bytes


class PagedBufferAllocator:
    """Allocates paged buffers from shared memory.

    This class manages the allocation of multiple paged buffer systems
    within the shared memory space, tracking offsets and ensuring proper
    alignment.
    """

    def __init__(self, smem_base=None):
        """Initialize the allocator.

        Args:
            smem_base: Base pointer to shared memory (optional, uses offset 0 if None)
        """
        self._current_offset = 0
        self._smem_base = smem_base
        self._layouts: list[PagedBufferLayout] = []

    def allocate(self, config: PagedBufferConfig, alignment: int = 128) -> PagedBufferLayout:
        """Allocate a paged buffer with the given configuration.

        Args:
            config: Buffer configuration
            alignment: Byte alignment for the buffer (default 128 for optimal smem access)

        Returns:
            PagedBufferLayout describing the allocated buffer
        """
        # Align current offset
        if self._current_offset % alignment != 0:
            self._current_offset += alignment - (self._current_offset % alignment)

        layout = PagedBufferLayout(config, self._current_offset)
        self._layouts.append(layout)
        self._current_offset += layout.total_bytes

        return layout

    @property
    def total_allocated(self) -> int:
        """Total bytes allocated so far."""
        return self._current_offset


# ============================================================================
# JIT-compatible Buffer Access Functions
# ============================================================================


@cute.jit
def paged_buffer_init_semaphores(
    smem_base,
    num_stages: int,
    sem_base_offset: int,
):
    """Initialize all semaphores for a paged buffer system.

    Should be called by thread 0 at kernel start, followed by sync_threads().

    Args:
        smem_base: Base pointer to shared memory
        num_stages: Number of buffer stages
        sem_base_offset: Byte offset to semaphore array
    """
    tidx, _, _ = cute.arch.thread_idx()

    if tidx == Int32(0):
        # Get pointer to semaphore region
        sem_ptr = cute.recast_ptr(smem_base, dtype=cute.Uint8) + sem_base_offset
        sem_i32_ptr = cute.recast_ptr(sem_ptr, dtype=cute.Int32)

        # Initialize loader_ready semaphores to 0 (no data ready yet)
        for i in range(num_stages):
            semaphore_init(sem_i32_ptr + i, Int32(0))

        # Initialize consumer_done semaphores to 1 (all stages initially available)
        # This allows the loader to start filling stage 0 immediately
        for i in range(num_stages):
            semaphore_init(sem_i32_ptr + num_stages + i, Int32(1))

    cute.arch.sync_threads()


@cute.jit
def loader_acquire_stage(
    smem_base,
    stage: Int32,
    iteration: Int32,
    num_stages: int,
    sem_base_offset: int,
):
    """Loader waits until a stage is available for loading.

    The loader must wait for the consumer to finish with this stage
    before overwriting it with new data.

    Args:
        smem_base: Base pointer to shared memory
        stage: Stage index to acquire
        iteration: Current iteration (used for semaphore counting)
        num_stages: Number of buffer stages
        sem_base_offset: Byte offset to semaphore array
    """
    # Get pointer to consumer_done semaphore for this stage
    sem_ptr = cute.recast_ptr(smem_base, dtype=cute.Uint8) + sem_base_offset
    sem_i32_ptr = cute.recast_ptr(sem_ptr, dtype=cute.Int32)
    consumer_done_ptr = sem_i32_ptr + num_stages + stage

    # For iteration i, consumer_done should be >= i+1
    # (consumer has processed iteration i-1 for this stage)
    # But for iteration 0, we initialized to 1, so we wait for >= 1
    expected = iteration + Int32(1)
    semaphore_wait(consumer_done_ptr, expected)


@cute.jit
def loader_release_stage(
    smem_base,
    stage: Int32,
    num_stages: int,
    sem_base_offset: int,
):
    """Loader signals that a stage is loaded and ready for consumption.

    Args:
        smem_base: Base pointer to shared memory
        stage: Stage index that was loaded
        num_stages: Number of buffer stages
        sem_base_offset: Byte offset to semaphore array
    """
    # Get pointer to loader_ready semaphore for this stage
    sem_ptr = cute.recast_ptr(smem_base, dtype=cute.Uint8) + sem_base_offset
    sem_i32_ptr = cute.recast_ptr(sem_ptr, dtype=cute.Int32)
    loader_ready_ptr = sem_i32_ptr + stage

    semaphore_signal(loader_ready_ptr)


@cute.jit
def consumer_acquire_stage(
    smem_base,
    stage: Int32,
    iteration: Int32,
    num_stages: int,
    sem_base_offset: int,
):
    """Consumer waits until a stage has data ready.

    Args:
        smem_base: Base pointer to shared memory
        stage: Stage index to acquire
        iteration: Current iteration (used for semaphore counting)
        num_stages: Number of buffer stages
        sem_base_offset: Byte offset to semaphore array
    """
    # Get pointer to loader_ready semaphore for this stage
    sem_ptr = cute.recast_ptr(smem_base, dtype=cute.Uint8) + sem_base_offset
    sem_i32_ptr = cute.recast_ptr(sem_ptr, dtype=cute.Int32)
    loader_ready_ptr = sem_i32_ptr + stage

    # For iteration i, loader_ready should be >= i+1
    expected = iteration + Int32(1)
    semaphore_wait(loader_ready_ptr, expected)


@cute.jit
def consumer_release_stage(
    smem_base,
    stage: Int32,
    num_stages: int,
    sem_base_offset: int,
):
    """Consumer signals that it's done with a stage.

    This allows the loader to reuse this stage for new data.

    Args:
        smem_base: Base pointer to shared memory
        stage: Stage index that was consumed
        num_stages: Number of buffer stages
        sem_base_offset: Byte offset to semaphore array
    """
    # Get pointer to consumer_done semaphore for this stage
    sem_ptr = cute.recast_ptr(smem_base, dtype=cute.Uint8) + sem_base_offset
    sem_i32_ptr = cute.recast_ptr(sem_ptr, dtype=cute.Int32)
    consumer_done_ptr = sem_i32_ptr + num_stages + stage

    semaphore_signal(consumer_done_ptr)


@cute.jit
def get_page_ptr(
    smem_base,
    stage: Int32,
    page_size: int,
    element_type,
):
    """Get a typed pointer to a specific page in the buffer.

    Args:
        smem_base: Base pointer to shared memory
        stage: Stage index
        page_size: Size of each page in bytes
        element_type: CuTe dtype for the returned pointer

    Returns:
        Pointer to the start of the page, typed as element_type
    """
    offset = stage * page_size
    byte_ptr = cute.recast_ptr(smem_base, dtype=cute.Uint8) + offset
    return cute.recast_ptr(byte_ptr, dtype=element_type)


# ============================================================================
# Inter-Operation Semaphores for Fused Kernel Pipelines
# ============================================================================
#
# When fusing multiple operations (Op0 -> Op1 -> Op2), we need inter-op
# semaphores so that:
#   - Op1's loader can start loading while Op0's consumer is still computing
#   - No global sync_threads() barrier between operations
#
# Memory Layout for inter-op semaphores (appended after per-op semaphores):
#   [op0_store_done, op1_store_done, ..., opN_store_done]
#
# Protocol:
#   - Op N storer: signal(op_N_store_done) after writing to global memory
#   - Op N+1 loader: wait(op_N_store_done) before loading (if dependent)
#
# For independent ops (no data dependency), no wait is needed.


@dataclass
class InterOpSemaphoreConfig:
    """Configuration for inter-operation semaphores in fused kernels.

    Attributes:
        num_ops: Number of operations in the fused kernel
        num_stages: Number of pipeline stages per operation
    """
    num_ops: int
    num_stages: int = 1  # Usually 1 for inter-op (per logical block)

    @property
    def num_semaphores(self) -> int:
        """Number of inter-op semaphores needed.

        One semaphore per operation to signal "store complete for this block".
        """
        return self.num_ops

    @property
    def semaphore_bytes(self) -> int:
        """Total bytes needed for inter-op semaphores."""
        return self.num_semaphores * 4


class InterOpSemaphoreLayout:
    """Describes the layout of inter-operation semaphores.

    Layout in shared memory:
        [op_0_done, op_1_done, ..., op_N-1_done]

    Each semaphore is incremented when that operation completes its store
    for the current logical block.
    """

    def __init__(self, num_ops: int, base_offset: int = 0):
        """Initialize the layout.

        Args:
            num_ops: Number of operations
            base_offset: Starting byte offset in shared memory
        """
        self.num_ops = num_ops
        self.base_offset = base_offset

    def get_op_done_sem_offset(self, op_idx: int) -> int:
        """Get byte offset of the 'done' semaphore for an operation."""
        return self.base_offset + (op_idx * 4)

    @property
    def total_bytes(self) -> int:
        """Total bytes used by inter-op semaphores."""
        return self.num_ops * 4


@cute.jit
def inter_op_init_semaphores(
    smem_base,
    num_ops: int,
    sem_base_offset: int,
):
    """Initialize inter-operation semaphores.

    Should be called by thread 0 at kernel start, followed by sync_threads().

    Args:
        smem_base: Base pointer to shared memory
        num_ops: Number of operations in fused kernel
        sem_base_offset: Byte offset to inter-op semaphore array
    """
    tidx, _, _ = cute.arch.thread_idx()

    if tidx == Int32(0):
        sem_ptr = cute.recast_ptr(smem_base, dtype=cute.Uint8) + sem_base_offset
        sem_i32_ptr = cute.recast_ptr(sem_ptr, dtype=cute.Int32)

        # Initialize all op_done semaphores to 0
        for i in range(num_ops):
            semaphore_init(sem_i32_ptr + i, Int32(0))

    cute.arch.sync_threads()


@cute.jit
def inter_op_wait_for_dependency(
    smem_base,
    dep_op_idx: Int32,
    num_ops: int,
    sem_base_offset: int,
):
    """Wait for a dependent operation to complete its store.

    Used by Op N+1's loader to wait for Op N's store to complete
    before loading data that Op N produced.

    Args:
        smem_base: Base pointer to shared memory
        dep_op_idx: Index of the operation we depend on
        num_ops: Number of operations (for bounds checking)
        sem_base_offset: Byte offset to inter-op semaphore array
    """
    sem_ptr = cute.recast_ptr(smem_base, dtype=cute.Uint8) + sem_base_offset
    sem_i32_ptr = cute.recast_ptr(sem_ptr, dtype=cute.Int32)
    op_done_ptr = sem_i32_ptr + dep_op_idx

    # Wait for the dependent op to signal completion (value >= 1)
    semaphore_wait(op_done_ptr, Int32(1))


@cute.jit
def inter_op_signal_done(
    smem_base,
    op_idx: Int32,
    num_ops: int,
    sem_base_offset: int,
):
    """Signal that an operation has completed its store phase.

    Called by Op N's storer after writing results to global memory.
    This allows Op N+1's loader to proceed if it was waiting.

    Args:
        smem_base: Base pointer to shared memory
        op_idx: Index of the operation that completed
        num_ops: Number of operations
        sem_base_offset: Byte offset to inter-op semaphore array
    """
    sem_ptr = cute.recast_ptr(smem_base, dtype=cute.Uint8) + sem_base_offset
    sem_i32_ptr = cute.recast_ptr(sem_ptr, dtype=cute.Int32)
    op_done_ptr = sem_i32_ptr + op_idx

    semaphore_signal(op_done_ptr)


@cute.jit
def inter_op_try_acquire(
    smem_base,
    dep_op_idx: Int32,
    num_ops: int,
    sem_base_offset: int,
) -> Int32:
    """Non-blocking check if a dependent operation has completed.

    Returns 1 if the dependency is satisfied, 0 otherwise.
    Useful for speculative execution or polling.

    Args:
        smem_base: Base pointer to shared memory
        dep_op_idx: Index of the operation we depend on
        num_ops: Number of operations
        sem_base_offset: Byte offset to inter-op semaphore array

    Returns:
        1 if dependency satisfied, 0 otherwise
    """
    sem_ptr = cute.recast_ptr(smem_base, dtype=cute.Uint8) + sem_base_offset
    sem_i32_ptr = cute.recast_ptr(sem_ptr, dtype=cute.Int32)
    op_done_ptr = sem_i32_ptr + dep_op_idx

    return semaphore_try_wait(op_done_ptr, Int32(1))
