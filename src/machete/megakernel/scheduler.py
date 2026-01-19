# Copyright (c) 2025, Machete Authors
"""Scheduler and data structures for persistent megakernel.

This module provides:
- WarpRole, WarpConfig: Warp specialization configuration
- PageConfig: Paged shared memory configuration
- BarrierConfig: Global scoreboard configuration
- Instruction: Runtime instruction encoding
- InstructionScheduler: Dependency-aware scheduling
- Decorators: @reads, @writes, @warp_role, @async_load
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Tuple, List, Dict, Any, Optional


# ============ Warp Roles ============


class WarpRole(IntEnum):
    """Warp specialization roles for the megakernel."""

    CONSUMER = 0  # MMA and math (warps 0 to N-1)
    LOADER = 1  # Async loads HBM→SMEM
    STORER = 2  # Stores SMEM→HBM
    LAUNCHER = 3  # Auxiliary async (TMA prefetch, etc.)
    CONTROLLER = 4  # Instruction fetch & coordination


@dataclass
class WarpConfig:
    """Warp specialization configuration.

    Layout: [Consumer warps...][Loader][Storer][Launcher][Controller]
    """

    num_consumer_warps: int = 12
    num_loader_warps: int = 1
    num_storer_warps: int = 1
    num_launcher_warps: int = 1
    num_controller_warps: int = 1

    @property
    def total_warps(self) -> int:
        return (
            self.num_consumer_warps
            + self.num_loader_warps
            + self.num_storer_warps
            + self.num_launcher_warps
            + self.num_controller_warps
        )

    @property
    def total_threads(self) -> int:
        return self.total_warps * 32

    def get_role(self, warp_id: int) -> WarpRole:
        """Map warp ID to role."""
        if warp_id < self.num_consumer_warps:
            return WarpRole.CONSUMER
        warp_id -= self.num_consumer_warps
        if warp_id < self.num_loader_warps:
            return WarpRole.LOADER
        warp_id -= self.num_loader_warps
        if warp_id < self.num_storer_warps:
            return WarpRole.STORER
        warp_id -= self.num_storer_warps
        if warp_id < self.num_launcher_warps:
            return WarpRole.LAUNCHER
        return WarpRole.CONTROLLER

    @property
    def loader_warp_start(self) -> int:
        return self.num_consumer_warps

    @property
    def storer_warp_start(self) -> int:
        return self.num_consumer_warps + self.num_loader_warps

    @property
    def launcher_warp_start(self) -> int:
        return self.storer_warp_start + self.num_storer_warps

    @property
    def controller_warp_start(self) -> int:
        return self.launcher_warp_start + self.num_launcher_warps


# ============ Logical Grid ============


@dataclass
class LogicalCoord:
    """Represents logical coordinates for a work unit."""

    values: Tuple[int, ...]


@dataclass
class LogicalGridInfo:
    """Full logical grid information."""

    logical_grid_size: int
    coord_names: Tuple[str, ...] = ("idx",)
    coord_dims: Tuple[int, ...] = None


# ============ Dependencies ============


@dataclass
class TensorDependency:
    """Tracks tensor read/write for dependency analysis."""

    name: str
    is_read: bool  # True=@reads, False=@writes


@dataclass
class BarrierConfig:
    """Global scoreboard configuration: Bar[op_id][chunk_id]."""

    num_ops: int
    num_chunks: int

    def get_index(self, op_id: int, chunk_id: int) -> int:
        """Compute flat index into barrier array."""
        return op_id * self.num_chunks + chunk_id

    @property
    def total_size(self) -> int:
        return self.num_ops * self.num_chunks


# ============ Paged Memory ============


@dataclass
class PageSemaphores:
    """Per-page semaphore state for backpressure control.

    Each page has two semaphores:
    - sem_loaded: Signaled when Loader finishes writing to this page
    - sem_consumed: Signaled when Consumer finishes reading from this page

    This prevents WAR (Write-After-Read) hazards where the Loader wraps
    around and overwrites data the Consumer is still processing.
    """

    num_pages: int

    def get_sem_loaded_offset(self, page_id: int) -> int:
        """Offset for sem_loaded[page_id] in semaphore array."""
        return page_id * 2

    def get_sem_consumed_offset(self, page_id: int) -> int:
        """Offset for sem_consumed[page_id] in semaphore array."""
        return page_id * 2 + 1

    @property
    def total_semaphores(self) -> int:
        """Total number of semaphores (2 per page)."""
        return self.num_pages * 2

    @property
    def smem_size(self) -> int:
        """Shared memory needed for semaphores (8 bytes each)."""
        return self.total_semaphores * 8


@dataclass
class PageConfig:
    """Paged shared memory configuration.

    The number of pages is computed dynamically based on available
    shared memory on the device. Only page_size is configurable.

    Page lifecycle with backpressure:
    1. Loader waits on sem_consumed[page] (page is free)
    2. Loader loads data into page
    3. Loader signals sem_loaded[page]
    4. Consumer waits on sem_loaded[page]
    5. Consumer processes data
    6. Consumer signals sem_consumed[page] (releases page for reuse)

    Typical shared memory sizes:
    - H100: ~227KB -> ~14 pages at 16KB
    - A100: ~164KB -> ~10 pages at 16KB
    - RTX 4090: ~100KB -> ~6 pages at 16KB
    """

    page_size: int = 16384  # 16KB per page (configurable)
    alignment: int = 128  # Alignment for each page
    reserved_smem: int = 4096  # Reserved for semaphores, ring buffer, etc.

    def get_num_pages(self, total_smem: int) -> int:
        """Compute number of pages based on available shared memory.

        Args:
            total_smem: Total shared memory available on device in bytes.

        Returns:
            Number of pages that fit in available memory.
        """
        available = total_smem - self.reserved_smem
        return max(1, available // self.page_size)

    def get_total_smem_for_pages(self, num_pages: int) -> int:
        """Get total shared memory used by pages."""
        return self.page_size * num_pages

    def get_semaphores(self, num_pages: int) -> PageSemaphores:
        """Create semaphore configuration for given page count."""
        return PageSemaphores(num_pages=num_pages)


# ============ Instructions ============


@dataclass
class Instruction:
    """Runtime instruction for megakernel interpreter.

    Encoding: [opcode, logical_idx, num_in, in_pages..., num_out, out_pages...,
               num_wait, wait_bars..., num_signal, signal_bars...]

    Dependencies are classified as:
    - Local: Same logical_idx, sequential ops within same block (use syncthreads)
    - Cross-block: Different logical_idx or requiring global coordination (use g_bar)

    For local dependencies, global scoreboard is skipped (redundant).
    """

    opcode: int  # Operation type (index into op registry)
    logical_idx: int  # Logical work unit index
    input_pages: List[int] = field(default_factory=list)  # Logical page IDs for inputs
    output_pages: List[int] = field(default_factory=list)  # Logical page IDs for outputs
    wait_barriers: List[int] = field(default_factory=list)  # Cross-block barrier indices to wait on
    signal_barriers: List[int] = field(default_factory=list)  # Barriers to signal
    local_deps: List[int] = field(default_factory=list)  # Local op dependencies (same block, use sync)
    arg_slots: List[int] = field(default_factory=list)  # Indices into global tensor pointer array
    needs_page_wait: bool = False  # True if must wait for page to be free (backpressure)

    def encode(self) -> List[int]:
        """Encode instruction as int array for GPU.

        Format: [opcode, logical_idx, flags, num_in, in_pages..., num_out, out_pages...,
                 num_wait, wait_bars..., num_signal, signal_bars..., num_local, local_deps...]

        Flags byte: bit 0 = needs_page_wait
        """
        flags = 1 if self.needs_page_wait else 0
        data = [self.opcode, self.logical_idx, flags]
        data.append(len(self.input_pages))
        data.extend(self.input_pages)
        data.append(len(self.output_pages))
        data.extend(self.output_pages)
        data.append(len(self.wait_barriers))
        data.extend(self.wait_barriers)
        data.append(len(self.signal_barriers))
        data.extend(self.signal_barriers)
        data.append(len(self.local_deps))
        data.extend(self.local_deps)
        data.append(len(self.arg_slots))
        data.extend(self.arg_slots)
        return data

    @staticmethod
    def decode(data: List[int], offset: int = 0) -> Tuple["Instruction", int]:
        """Decode instruction from int array. Returns (instruction, next_offset)."""
        opcode = data[offset]
        logical_idx = data[offset + 1]
        flags = data[offset + 2]
        needs_page_wait = bool(flags & 1)
        pos = offset + 3

        num_in = data[pos]
        pos += 1
        input_pages = data[pos : pos + num_in]
        pos += num_in

        num_out = data[pos]
        pos += 1
        output_pages = data[pos : pos + num_out]
        pos += num_out

        num_wait = data[pos]
        pos += 1
        wait_barriers = data[pos : pos + num_wait]
        pos += num_wait

        num_signal = data[pos]
        pos += 1
        signal_barriers = data[pos : pos + num_signal]
        pos += num_signal

        num_local = data[pos]
        pos += 1
        local_deps = data[pos : pos + num_local]
        pos += num_local

        num_slots = data[pos]
        pos += 1
        arg_slots = data[pos : pos + num_slots]
        pos += num_slots

        return Instruction(
            opcode=opcode,
            logical_idx=logical_idx,
            input_pages=list(input_pages),
            output_pages=list(output_pages),
            wait_barriers=list(wait_barriers),
            signal_barriers=list(signal_barriers),
            local_deps=list(local_deps),
            arg_slots=list(arg_slots),
            needs_page_wait=needs_page_wait,
        ), pos


# ============ Decorators ============


def reads(*tensor_names: str):
    """Mark method as reading from tensors for dependency tracking."""

    def decorator(func):
        # Store in deps list for scheduler
        existing = getattr(func, "_machete_deps", [])
        for name in tensor_names:
            existing.append(TensorDependency(name=name, is_read=True))
        func._machete_deps = existing
        # Also store as set for compatibility
        func._machete_reads = set(tensor_names)
        return func

    return decorator


def writes(*tensor_names: str):
    """Mark method as writing to tensors for dependency tracking."""

    def decorator(func):
        # Store in deps list for scheduler
        existing = getattr(func, "_machete_deps", [])
        for name in tensor_names:
            existing.append(TensorDependency(name=name, is_read=False))
        func._machete_deps = existing
        # Also store as set for compatibility
        func._machete_writes = set(tensor_names)
        return func

    return decorator


def warp_role(role: WarpRole):
    """Assign method to specific warp role."""

    def decorator(func):
        func._machete_warp_role = role
        return func

    return decorator


def async_load(func):
    """Mark load as using async copy (TMA/cp.async)."""
    func._machete_async_load = True
    return func


# ============ Scheduler ============


class InstructionScheduler:
    """Schedules ops into instruction stream with fine-grained dependencies.

    The scheduler:
    1. Tracks tensor read/write dependencies
    2. Classifies dependencies as local (same block) or cross-block
    3. Generates instructions for each (op, chunk) pair
    4. Computes barrier wait/signal indices for cross-block dependencies
    5. Allocates logical pages (round-robin) with backpressure tracking

    Dependency Classification:
    - Local: Producer and consumer process same logical_idx in same block.
             Use syncthreads() instead of global scoreboard (faster).
    - Cross-block: Different logical_idx or different blocks.
             Use global scoreboard (g_bar) for synchronization.

    Backpressure (WAR Protection):
    - When logical_idx wraps around pages (logical_idx >= num_pages),
      the Loader must wait for Consumer to release the page.
    - This prevents overwriting data the Consumer is still processing.
    """

    def __init__(self, barrier_config: BarrierConfig, page_config: PageConfig, num_pages: int = 8):
        self.barrier_config = barrier_config
        self.page_config = page_config
        self.num_pages = num_pages  # Actual number of pages (computed from device smem)
        self.ops: List[Dict] = []
        self.instructions: List[Instruction] = []
        # Track which op last wrote each tensor
        self.tensor_producers: Dict[str, int] = {}  # tensor_name -> op_id
        # Track global tensor registration
        self.tensor_to_slot: Dict[int, int] = {}  # tensor_id -> slot_index
        self.unique_tensors: List[Any] = []

    def add_op(
        self, op_id: int, op: Any, logical_grid_size: int, reads_list: List[str], writes_list: List[str]
    ) -> None:
        """Add operation to schedule."""
        self.ops.append(
            {
                "op_id": op_id,
                "op": op,
                "logical_grid_size": logical_grid_size,
                "reads": reads_list,
                "writes": writes_list,
                "args": [],  # Filled during registry build
            }
        )

    def register_tensor(self, tensor: Any) -> int:
        """Register a tensor and return its global slot index."""
        tid = id(tensor)
        if tid not in self.tensor_to_slot:
            slot = len(self.unique_tensors)
            self.tensor_to_slot[tid] = slot
            self.unique_tensors.append(tensor)
        return self.tensor_to_slot[tid]

    def _is_local_dependency(self, producer_op: int, consumer_op: int, chunk_id: int) -> bool:
        """Check if dependency is local (same block, sequential ops).

        Local dependencies occur when:
        1. Operations are sequential in the op list (producer_op < consumer_op)
        2. Same logical_idx/chunk_id is being processed
        3. Single persistent block processes both ops for this chunk

        In the persistent model with warp specialization, ops for the same
        logical_idx are processed by the same block, so syncthreads() suffices.
        """
        # If ops are sequential and processing same chunk, it's local
        return producer_op < consumer_op

    def schedule(self) -> List[Instruction]:
        """Generate instruction stream with dependencies.

        Each instruction represents one (op, chunk) pair.
        Dependencies are classified as local or cross-block.
        """
        self.instructions = []

        for op_info in self.ops:
            op_id = op_info["op_id"]
            logical_grid_size = op_info["logical_grid_size"]
            reads_list = op_info["reads"]
            writes_list = op_info["writes"]

            # Map op arguments to global slots
            arg_slots = [self.register_tensor(arg) for arg in op_info.get("raw_args", [])]

            for chunk_id in range(logical_grid_size):
                # Classify dependencies
                wait_barriers = []  # Cross-block only
                local_deps = []  # Same block, use syncthreads

                for tensor_name in reads_list:
                    if tensor_name in self.tensor_producers:
                        producer_op = self.tensor_producers[tensor_name]
                        if self._is_local_dependency(producer_op, op_id, chunk_id):
                            # Local dependency: just need syncthreads between ops
                            local_deps.append(producer_op)
                        else:
                            # Cross-block dependency: need global scoreboard
                            wait_idx = self.barrier_config.get_index(producer_op, chunk_id)
                            wait_barriers.append(wait_idx)

                # Compute signal barrier (always signal for potential cross-block consumers)
                signal_barriers = []
                signal_idx = self.barrier_config.get_index(op_id, chunk_id)
                signal_barriers.append(signal_idx)

                # Page allocation: round-robin
                page_id = chunk_id % self.num_pages
                input_pages = [page_id]
                output_pages = [(chunk_id + 1) % self.num_pages]

                # Backpressure: if chunk_id >= num_pages, the page may still be
                # in use by a previous iteration. Must wait for consumer to release.
                needs_page_wait = chunk_id >= self.num_pages

                inst = Instruction(
                    opcode=op_id,
                    logical_idx=chunk_id,
                    input_pages=input_pages,
                    output_pages=output_pages,
                    wait_barriers=wait_barriers,
                    signal_barriers=signal_barriers,
                    local_deps=local_deps,
                    arg_slots=arg_slots,
                    needs_page_wait=needs_page_wait,
                )
                self.instructions.append(inst)

            # Update producers for next ops
            for tensor_name in writes_list:
                self.tensor_producers[tensor_name] = op_id

        return self.instructions

    def get_encoded_instructions(self) -> List[int]:
        """Get flattened instruction buffer for GPU."""
        encoded = []
        for inst in self.instructions:
            encoded.extend(inst.encode())
        return encoded

    def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze scheduled instructions for debugging.

        Returns:
            Dict with statistics about local vs cross-block dependencies.
        """
        total_local = sum(len(inst.local_deps) for inst in self.instructions)
        total_cross = sum(len(inst.wait_barriers) for inst in self.instructions)
        total_backpressure = sum(1 for inst in self.instructions if inst.needs_page_wait)

        return {
            "total_instructions": len(self.instructions),
            "local_dependencies": total_local,
            "cross_block_dependencies": total_cross,
            "backpressure_waits": total_backpressure,
            "num_pages": self.num_pages,
        }
