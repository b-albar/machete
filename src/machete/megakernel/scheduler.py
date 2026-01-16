# Copyright (c) 2025, Machete Authors
"""
No Bubbles Scheduler for Megakernels.

This module implements the scheduling infrastructure for the No Bubbles pattern:
1. Paged shared memory management
2. Instruction sequencing per SM
3. Global synchronization via atomic counters
4. Page request/release coordination
5. **Dependency-aware scheduling** for optimal load/compute overlap
6. **Logical Blocks** abstraction for flexible coordinate mapping

Based on: "Look Ma, No Bubbles!" - HazyResearch
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Callable, TypeVar, Tuple
from enum import Enum, auto
from functools import wraps  # noqa: F401 - may be used by decorated functions

F = TypeVar("F", bound=Callable)


# ============================================================================
# Warp Role Definitions (HazyResearch Model)
# ============================================================================


class WarpRole(Enum):
    """Warp specialization roles for the megakernel.

    Based on HazyResearch's design:
    - CONSUMER: 12-16 warps for compute (MMA, math operations)
    - LOADER: 1 warp for async loads (Global -> Shared via TMA/cp.async)
    - STORER: 1 warp for async stores (Shared -> Global)
    - LAUNCHER: 1 warp for auxiliary async tasks (K/V cache prefetch, TMA descriptors)
    - CONTROLLER: 1 warp for instruction fetch/decode and pipeline coordination
    """

    CONSUMER = auto()  # Math/MMA operations (12-16 warps)
    LOADER = auto()  # Global -> Shared memory loads (1 warp)
    STORER = auto()  # Shared -> Global memory stores (1 warp)
    LAUNCHER = auto()  # Auxiliary async tasks (1 warp)
    CONTROLLER = auto()  # Instruction fetch/decode (1 warp)


@dataclass
class WarpConfig:
    """Configuration for warp specialization.

    Typical H100 configuration:
    - 20 warps total per block (640 threads)
    - 16 consumer warps (80% of warps)
    - 4 system warps (controller, launcher, loader, storer)
    """

    num_consumer_warps: int = 16
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

    def get_warp_role(self, warp_id: int) -> WarpRole:
        """Determine the role of a warp by its ID."""
        if warp_id < self.num_consumer_warps:
            return WarpRole.CONSUMER
        offset = self.num_consumer_warps
        if warp_id < offset + self.num_loader_warps:
            return WarpRole.LOADER
        offset += self.num_loader_warps
        if warp_id < offset + self.num_storer_warps:
            return WarpRole.STORER
        offset += self.num_storer_warps
        if warp_id < offset + self.num_launcher_warps:
            return WarpRole.LAUNCHER
        return WarpRole.CONTROLLER


# ============================================================================
# Logical Block Coordinate Mapping
# ============================================================================


@dataclass
class LogicalCoord:
    """A logical coordinate that maps to kernel-specific dimensions.

    This abstracts the physical block index into a logical coordinate system
    that each kernel can interpret differently (e.g., (batch, seq, head) for RoPE).
    """

    values: Tuple[int, ...]
    names: Tuple[str, ...] = ()

    def __getitem__(self, idx: int) -> int:
        return self.values[idx]

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        if self.names:
            parts = [f"{n}={v}" for n, v in zip(self.names, self.values)]
            return f"LogicalCoord({', '.join(parts)})"
        return f"LogicalCoord{self.values}"


@dataclass
class LogicalGridInfo:
    """Information about the logical grid for a kernel.

    Each kernel computes its own logical grid size based on the problem shape.
    The scheduler uses this to determine TotalLogicalBlocks and coordinate mapping.
    """

    logical_grid_size: int  # Total logical blocks for this kernel
    coord_names: Tuple[str, ...] = ()  # Names for coordinate dimensions
    coord_dims: Tuple[int, ...] = ()  # Size of each coordinate dimension


# ============================================================================
# Dependency Declaration Decorators for L/C/S Methods
# ============================================================================


def reads(*tensors: str) -> Callable[[F], F]:
    """Decorator to declare which tensors/memory regions an L/C/S method reads from.

    Usage:
        @reads("input", "weight")
        @cute.jit
        def load_forward(self, paged_pool, page_idx, input, weight, output):
            ...

    The tensor names should match the argument names or be symbolic identifiers
    that the scheduler uses for dependency analysis.
    """

    def decorator(func: F) -> F:
        existing = getattr(func, "_machete_reads", set())
        func._machete_reads = existing | set(tensors)
        return func

    return decorator


def writes(*tensors: str) -> Callable[[F], F]:
    """Decorator to declare which tensors/memory regions an L/C/S method writes to.

    Usage:
        @writes("output")
        @cute.jit
        def store_forward(self, paged_pool, page_idx, input, weight, output):
            ...
    """

    def decorator(func: F) -> F:
        existing = getattr(func, "_machete_writes", set())
        func._machete_writes = existing | set(tensors)
        return func

    return decorator


def independent() -> Callable[[F], F]:
    """Decorator to mark an L/C/S method as having no dependencies on previous ops.

    This is a shortcut for operations that read from unique memory regions
    and can always be scheduled as early as possible.

    Usage:
        @independent()
        @cute.jit
        def load_forward(self, paged_pool, page_idx, ...):
            # This load can overlap with any previous compute/store
            ...
    """

    def decorator(func: F) -> F:
        func._machete_independent = True
        return func

    return decorator


def warp_role(role: WarpRole) -> Callable[[F], F]:
    """Decorator to annotate which warp role should execute a method.

    Usage:
        @warp_role(WarpRole.LOADER)
        @cute.jit
        def load_forward(self, paged_pool, page_idx, ...):
            # Only loader warps execute this
            ...
    """

    def decorator(func: F) -> F:
        func._machete_warp_role = role
        return func

    return decorator


def get_method_dependencies(method: Callable) -> tuple[Set[str], Set[str], bool]:
    """Extract dependency info from a decorated L/C/S method.

    Returns:
        (reads, writes, is_independent)
    """
    reads_set = getattr(method, "_machete_reads", set())
    writes_set = getattr(method, "_machete_writes", set())
    is_independent = getattr(method, "_machete_independent", False)
    return reads_set, writes_set, is_independent


def get_method_warp_role(method: Callable) -> Optional[WarpRole]:
    """Extract warp role annotation from a method."""
    return getattr(method, "_machete_warp_role", None)


def build_op_descriptor_from_kernel(kernel, op_idx: int, mode: str = "forward") -> "OpDescriptor":
    """Build an OpDescriptor by extracting dependency info from decorated L/C/S methods.

    This function examines the kernel's load/compute/store methods for the given mode
    and extracts @reads/@writes annotations to build the dependency graph.

    Args:
        kernel: A FusableKernel or MegakernelOp instance with L/C/S methods
        op_idx: The operation index in the schedule
        mode: "forward" or "backward"

    Returns:
        OpDescriptor with reads/writes populated from method decorators

    Example:
        class MyKernel(FusableKernel):
            @reads("input", "weight")
            @cute.jit
            def load_forward(self, paged_pool, page_idx, input, weight, output):
                ...

            @writes("output")
            @cute.jit
            def store_forward(self, paged_pool, page_idx, input, weight, output):
                ...

        # Scheduler can now determine that MyKernel doesn't depend on previous ops
        # that write to different tensors
        desc = build_op_descriptor_from_kernel(kernel, 0, "forward")

    Note:
    Nota:
        Synchronization between L/C/S phases is now handled via fine-grained semaphores.
    """
    load_method = getattr(kernel, f"load_{mode}", None)
    compute_method = getattr(kernel, f"compute_{mode}", None)
    store_method = getattr(kernel, f"store_{mode}", None)

    all_reads: Set[str] = set()
    all_writes: Set[str] = set()
    is_independent = False

    for method in [load_method, compute_method, store_method]:
        if method is not None:
            reads_set, writes_set, indep = get_method_dependencies(method)
            all_reads |= reads_set
            all_writes |= writes_set
            is_independent = is_independent or indep

    # Get kernel class name for debugging
    name = getattr(kernel, "__class__", type(kernel)).__name__

    # Extract logical grid info if available
    logical_grid_size = None
    if hasattr(kernel, "get_logical_grid_size"):
        # Defer actual computation to runtime when args are available
        logical_grid_size = -1  # Sentinel: compute at runtime

    return OpDescriptor(
        name=name,
        op_idx=op_idx,
        reads=all_reads,
        writes=all_writes,
        allow_early_load=is_independent or len(all_reads) > 0,
        logical_grid_size=logical_grid_size,
    )


class MicroOpType(Enum):
    # Synchronous operations (legacy/simple mode)
    LOAD = auto()
    COMPUTE = auto()
    STORE = auto()
    SYNC_BLOCK = auto()
    SYNC_GLOBAL = auto()
    ADVANCE_PAGE = auto()

    # Async operations (for overlapped scheduling)
    LOAD_ASYNC = auto()  # Issue async load (cp.async / TMA)
    WAIT_LOAD = auto()  # Wait for async load to complete
    STORE_ASYNC = auto()  # Issue async store
    WAIT_STORE = auto()  # Wait for async store completion
    COMMIT_GROUP = auto()  # Commit async group (cp.async.commit_group)
    WAIT_GROUP = auto()  # Wait for async group (cp.async.wait_group)

    # Page semaphore operations (No Bubbles pattern)
    # These use hardware mbarrier for efficient intra-block synchronization
    PAGE_ACQUIRE = auto()  # Wait for page semaphore (sem_finished from previous user)
    PAGE_RELEASE = auto()  # Signal page semaphore (sem_arrived for next user)
    SEM_WAIT = auto()  # Generic semaphore wait
    SEM_SIGNAL = auto()  # Generic semaphore signal


@dataclass
class MicroOp:
    """A micro-operation in the schedule with explicit dependencies."""

    id: int  # Unique ID for dependency tracking
    type: MicroOpType
    op_idx: int
    desc: str = ""
    # Dependencies: set of MicroOp IDs that must complete before this one
    depends_on: Set[int] = field(default_factory=set)

    def __hash__(self):
        return self.id


@dataclass
class OpDescriptor:
    """Describes an operation's memory access patterns for dependency analysis.

    This allows the scheduler to determine which operations can run in parallel.
    Operations that read from different memory than others write to can overlap.

    Note: Synchronization between L/C/S phases is now always handled via
    fine-grained semaphores, not sync_threads().
    """

    name: str
    op_idx: int

    # Memory regions this operation reads from (as tensor IDs or symbolic names)
    reads: Set[str] = field(default_factory=set)
    # Memory regions this operation writes to
    writes: Set[str] = field(default_factory=set)

    # If True, the load for this op can overlap with previous compute/store
    # if there's no read-after-write dependency
    allow_early_load: bool = True

    # If True, requires global sync (cross-SM barrier)
    needs_global_sync: bool = False

    # Logical grid size for this operation (-1 means compute at runtime)
    logical_grid_size: Optional[int] = None

    # Logical offset in the unified grid (for operations with different grid sizes)
    logical_offset: int = 0

    # Number of logical blocks this operation processes
    logical_count: int = 0


@dataclass
class Instruction:
    """An instruction for a single SM to execute.

    Extended with logical block information for the No Bubbles pattern.
    """

    opcode: int
    op_idx: int
    args: List
    page_ids: List[int]
    depends_on: List[int] = field(default_factory=list)
    signals: List[int] = field(default_factory=list)

    # Logical block mapping
    logical_offset: int = 0  # Starting logical block index for this instruction
    logical_count: int = 1  # Number of logical blocks this instruction processes


@dataclass
class NoBubblesConfig:
    """Configuration for the No Bubbles scheduler.

    H100 defaults based on HazyResearch paper:
    - 13 pages (227KB shared memory / 16KB page size)
    - 64 sync counters for barrier coordination

    Use from_device_smem() to compute num_pages based on available shared memory.
    """

    num_pages: int = 13
    page_size_bytes: int = 16384
    num_sync_counters: int = 64
    max_instructions_per_sm: int = 256

    # Warp configuration
    warp_config: WarpConfig = field(default_factory=WarpConfig)

    # Reserved shared memory for non-page uses (semaphores, instruction buffer, etc.)
    reserved_smem_bytes: int = 4096  # 4KB default

    @classmethod
    def from_device_smem(
        cls,
        device_smem_bytes: int,
        page_size_bytes: int = 16384,
        reserved_smem_bytes: int = 4096,
        warp_config: Optional[WarpConfig] = None,
    ) -> "NoBubblesConfig":
        """Create config with num_pages computed from device shared memory.

        Args:
            device_smem_bytes: Total shared memory available on device (e.g., 227KB for H100)
            page_size_bytes: Size of each page in bytes (default 16KB)
            reserved_smem_bytes: Memory reserved for non-page uses (default 4KB)
            warp_config: Optional warp configuration

        Returns:
            NoBubblesConfig with num_pages computed from available memory

        Example:
            # H100 with 227KB shared memory
            config = NoBubblesConfig.from_device_smem(227 * 1024)
            # -> num_pages = (227KB - 4KB) / 16KB = 13 pages

            # A100 with 164KB shared memory
            config = NoBubblesConfig.from_device_smem(164 * 1024)
            # -> num_pages = (164KB - 4KB) / 16KB = 10 pages
        """
        available_smem = device_smem_bytes - reserved_smem_bytes
        num_pages = available_smem // page_size_bytes

        if num_pages < 2:
            raise ValueError(
                f"Not enough shared memory for paging: {device_smem_bytes} bytes "
                f"with {page_size_bytes} byte pages and {reserved_smem_bytes} reserved"
            )

        return cls(
            num_pages=num_pages,
            page_size_bytes=page_size_bytes,
            reserved_smem_bytes=reserved_smem_bytes,
            warp_config=warp_config or WarpConfig(),
        )

    @property
    def total_page_smem_bytes(self) -> int:
        """Total shared memory used by pages."""
        return self.num_pages * self.page_size_bytes

    @property
    def total_smem_bytes(self) -> int:
        """Total shared memory required (pages + reserved)."""
        return self.total_page_smem_bytes + self.reserved_smem_bytes


# ============================================================================
# Barrier Tensor Configuration for Logical Blocks
# ============================================================================


@dataclass
class BarrierConfig:
    """Configuration for the global barrier tensor.

    The barrier tensor is addressed as: Bar[OpIdx][LogicalID]
    - Producer: atomicAdd(&Bar[op_idx, logical_id], 1) after store
    - Consumer: while(Bar[dep_op_idx, logical_id] < target) nanosleep()

    This allows fine-grained synchronization where each logical block
    can proceed as soon as its specific data is ready.
    """

    num_ops: int  # Number of operations in the megakernel
    total_logical_blocks: int  # Max logical grid size across all ops

    @property
    def tensor_size(self) -> Tuple[int, int]:
        """Size of the barrier tensor: (num_ops, total_logical_blocks)."""
        return (self.num_ops, self.total_logical_blocks)

    @property
    def total_counters(self) -> int:
        """Total number of barrier counters needed."""
        return self.num_ops * self.total_logical_blocks

    def get_barrier_index(self, op_idx: int, logical_id: int) -> int:
        """Get the linear index into the barrier tensor."""
        return op_idx * self.total_logical_blocks + logical_id


class NoBubblesScheduler:
    """Scheduler with dependency-aware scheduling for optimal overlap.

    The key insight is that Load[i+1] can often start before Compute[i] or Store[i]
    completes, as long as there's no data dependency. This scheduler builds a
    dependency graph and schedules loads as early as possible.

    Extended with Logical Blocks support:
    - Each operation defines get_logical_grid_size() and get_logical_coord()
    - The scheduler uses TotalLogicalBlocks = max(op.logical_grid_size) for grid dimension
    - Barriers are indexed by (OpIdx, LogicalID) for fine-grained sync
    """

    def __init__(self, config: Optional[NoBubblesConfig] = None):
        self.config = config or NoBubblesConfig()
        self.instructions: List[Instruction] = []
        self.micro_ops: List[MicroOp] = []
        self._next_micro_op_id = 0
        self.barrier_config: Optional[BarrierConfig] = None

    def _new_micro_op(
        self,
        type: MicroOpType,
        op_idx: int,
        desc: str = "",
        depends_on: Optional[Set[int]] = None,
    ) -> MicroOp:
        """Create a new MicroOp with a unique ID."""
        op = MicroOp(
            id=self._next_micro_op_id,
            type=type,
            op_idx=op_idx,
            desc=desc,
            depends_on=depends_on or set(),
        )
        self._next_micro_op_id += 1
        self.micro_ops.append(op)
        return op

    def add_micro_op(self, type: MicroOpType, op_idx: int, desc: str = "") -> MicroOp:
        """Add a micro-op (legacy interface, no explicit dependencies)."""
        return self._new_micro_op(type, op_idx, desc)

    def _check_dependency(
        self,
        op_reads: Set[str],
        op_writes: Set[str],
        prev_writes: Set[str],
        prev_reads: Set[str],
    ) -> bool:
        """Check if there's a data dependency between two operations.

        Returns True if the operation depends on the previous operation:
        - Read-After-Write (RAW): op reads what prev wrote
        - Write-After-Write (WAW): op writes what prev wrote
        - Write-After-Read (WAR): op writes what prev read (anti-dependency)
        """
        # RAW: we read something that was written
        if op_reads & prev_writes:
            return True
        # WAW: we write something that was written
        if op_writes & prev_writes:
            return True
        # WAR: we write something that was read (anti-dependency)
        if op_writes & prev_reads:
            return True
        return False

    def calculate_total_logical_blocks(self, ops: List[OpDescriptor]) -> int:
        """Calculate TotalLogicalBlocks = max(op.logical_grid_size) across all ops.

        For the "No Bubbles" pattern, all operations typically share the same
        logical grid space (1:1 mapping for producer/consumer).
        """
        max_logical = 1
        for op in ops:
            if op.logical_grid_size is not None and op.logical_grid_size > 0:
                max_logical = max(max_logical, op.logical_grid_size)
        return max_logical

    def configure_barriers(self, num_ops: int, total_logical_blocks: int):
        """Configure the barrier tensor for logical block synchronization."""
        self.barrier_config = BarrierConfig(
            num_ops=num_ops,
            total_logical_blocks=total_logical_blocks,
        )

    def generate_dependency_aware_schedule(self, ops: List[OpDescriptor]) -> List[MicroOp]:
        """Generate a schedule with explicit dependencies for maximum overlap.

        This method analyzes read/write sets to determine the minimal set of
        dependencies, allowing loads to be scheduled as early as possible.

        The schedule structure for maximum overlap:
        - Load[i] depends on: Store[j] for any j where op[j].writes ∩ op[i].reads != empty
        - Compute[i] depends on: Load[i] (always)
        - Store[i] depends on: Compute[i] (always)

        This allows Load[i+1] to run in parallel with Compute[i] and Store[i]
        when there's no data dependency.
        """
        self.micro_ops.clear()
        self._next_micro_op_id = 0
        n_ops = len(ops)

        if n_ops == 0:
            return self.micro_ops

        # Configure barriers based on logical grid
        total_logical = self.calculate_total_logical_blocks(ops)
        self.configure_barriers(n_ops, total_logical)

        # Track micro-ops by (op_idx, type) for dependency resolution
        load_ops: Dict[int, MicroOp] = {}
        compute_ops: Dict[int, MicroOp] = {}
        store_ops: Dict[int, MicroOp] = {}
        sync_ops: List[MicroOp] = []

        # Phase 1: Create all micro-ops and compute dependencies
        for i, op in enumerate(ops):
            # --- LOAD[i] ---
            load_deps: Set[int] = set()

            # Load[i] depends on Store[j] if op[i] reads what op[j] wrote
            for j in range(i):
                if self._check_dependency(
                    op_reads=op.reads,
                    op_writes=set(),  # Load doesn't write
                    prev_writes=ops[j].writes,
                    prev_reads=set(),  # Only care about what prev wrote
                ):
                    # Must wait for Store[j] to complete
                    if j in store_ops:
                        load_deps.add(store_ops[j].id)

            # Also depends on any previous global sync
            if sync_ops:
                load_deps.add(sync_ops[-1].id)

            load_op = self._new_micro_op(
                MicroOpType.LOAD,
                i,
                f"Load[{i}]" + (" (early)" if not load_deps else ""),
                load_deps,
            )
            load_ops[i] = load_op

            # --- COMPUTE[i] ---
            # Always depends on its own Load
            compute_deps = {load_op.id}

            compute_op = self._new_micro_op(MicroOpType.COMPUTE, i, f"Compute[{i}]", compute_deps)
            compute_ops[i] = compute_op

            # --- STORE[i] ---
            # Always depends on its own Compute
            store_deps = {compute_op.id}

            store_op = self._new_micro_op(MicroOpType.STORE, i, f"Store[{i}]", store_deps)
            store_ops[i] = store_op

            # --- SYNC (if needed) ---
            # Note: Block sync via sync_threads() is deprecated.
            # Fine-grained semaphore synchronization is used instead.
            # Only global sync (cross-SM barrier) is still supported here.
            if op.needs_global_sync:
                sync_global = self._new_micro_op(
                    MicroOpType.SYNC_GLOBAL,
                    i,
                    f"GlobalSync[{i}]",
                    {store_op.id} | ({sync_ops[-1].id} if sync_ops else set()),
                )
                sync_ops.append(sync_global)

        return self.micro_ops

    def generate_pipeline_schedule(self, ops: List[dict], use_pipeline: bool = True):
        """DEPRECATED: Legacy pipeline schedule generator.

        This method is kept for backward compatibility but the generated schedule
        is no longer used. All scheduling modes now use fine-grained semaphore
        synchronization implemented directly in the code generator.

        Note: Synchronization between L/C/S phases is handled via atomic semaphores (sem_load_done, sem_compute_done).
        """
        self.micro_ops.clear()
        self._next_micro_op_id = 0
        n_ops = len(ops)

        if not use_pipeline:
            # Serial Schedule: L -> C -> S (sync handled via semaphores)
            for i in range(n_ops):
                self.add_micro_op(MicroOpType.LOAD, i)
                self.add_micro_op(MicroOpType.COMPUTE, i)
                self.add_micro_op(MicroOpType.STORE, i)

                if ops[i].get("needs_sync", False):
                    self.add_micro_op(MicroOpType.SYNC_GLOBAL, i)
            return

        # No Bubbles Pipeline Schedule (legacy structure, sync via semaphores)
        if n_ops > 0:
            self.add_micro_op(MicroOpType.LOAD, 0, "Prologue")

        for i in range(n_ops):
            self.add_micro_op(MicroOpType.COMPUTE, i)

            if i + 1 < n_ops:
                self.add_micro_op(MicroOpType.LOAD, i + 1, "Prefetch")

            self.add_micro_op(MicroOpType.STORE, i)

            if ops[i].get("needs_sync", False):
                self.add_micro_op(MicroOpType.SYNC_GLOBAL, i)

    def topological_sort(self) -> List[MicroOp]:
        """Return micro-ops in a valid execution order respecting dependencies.

        Uses Kahn's algorithm for topological sorting.
        """
        from collections import deque

        # Build adjacency list and in-degree count
        in_degree: Dict[int, int] = {op.id: len(op.depends_on) for op in self.micro_ops}
        dependents: Dict[int, List[int]] = {op.id: [] for op in self.micro_ops}
        id_to_op: Dict[int, MicroOp] = {op.id: op for op in self.micro_ops}

        for op in self.micro_ops:
            for dep_id in op.depends_on:
                if dep_id in dependents:
                    dependents[dep_id].append(op.id)

        # Start with ops that have no dependencies
        queue = deque([op.id for op in self.micro_ops if in_degree[op.id] == 0])
        result: List[MicroOp] = []

        while queue:
            op_id = queue.popleft()
            result.append(id_to_op[op_id])

            for dependent_id in dependents[op_id]:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        if len(result) != len(self.micro_ops):
            raise ValueError("Cycle detected in dependency graph!")

        return result

    def get_parallelizable_groups(self) -> List[List[MicroOp]]:
        """Group micro-ops that can execute in parallel.

        Returns a list of "waves" where all ops in each wave can run concurrently.
        This is useful for visualizing the schedule and estimating performance.
        """
        sorted_ops = self.topological_sort()
        if not sorted_ops:
            return []

        # Calculate the earliest "wave" each op can be in
        wave_of: Dict[int, int] = {}
        for op in sorted_ops:
            if not op.depends_on:
                wave_of[op.id] = 0
            else:
                wave_of[op.id] = max(wave_of[dep] for dep in op.depends_on) + 1

        # Group by wave
        max_wave = max(wave_of.values()) if wave_of else 0
        waves: List[List[MicroOp]] = [[] for _ in range(max_wave + 1)]
        for op in sorted_ops:
            waves[wave_of[op.id]].append(op)

        return waves

    def visualize_schedule(self) -> str:
        """Create an ASCII visualization of the parallel schedule."""
        waves = self.get_parallelizable_groups()
        lines = []
        lines.append("=" * 60)
        lines.append("Dependency-Aware Schedule Visualization")
        lines.append("=" * 60)

        for i, wave in enumerate(waves):
            ops_str = " | ".join(f"{op.type.name}[{op.op_idx}]" for op in wave)
            lines.append(f"Wave {i}: {ops_str}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# Paged Memory Allocator with Acquire/Release Semantics
# ============================================================================


@dataclass
class PageSemaphoreConfig:
    """Configuration for page semaphores in shared memory.

    Each page has two semaphores (following HazyResearch design):
    - sem_arrived: Loader signals when data is ready, Consumer waits
    - sem_finished: Consumer signals when done, Loader waits before reusing

    This creates a circular buffer where pages can be reused as soon as
    the consumer is done, even if the storer is still writing to global memory.
    """

    num_pages: int
    # Each semaphore is typically 8 bytes (mbarrier on H100)
    semaphore_size_bytes: int = 8

    @property
    def total_semaphores(self) -> int:
        """Two semaphores per page: arrived + finished."""
        return self.num_pages * 2

    @property
    def total_size_bytes(self) -> int:
        """Total shared memory needed for all semaphores."""
        return self.total_semaphores * self.semaphore_size_bytes

    def get_arrived_semaphore_idx(self, page_id: int) -> int:
        """Get index of the 'data arrived' semaphore for a page."""
        return page_id * 2

    def get_finished_semaphore_idx(self, page_id: int) -> int:
        """Get index of the 'consumer finished' semaphore for a page."""
        return page_id * 2 + 1


@dataclass
class PageAllocation:
    """Represents a page allocation for an operation."""

    op_idx: int
    page_ids: List[int]  # Which pages this op holds
    acquired_at: int  # MicroOp ID when pages were acquired
    released_at: Optional[int] = None  # MicroOp ID when pages were released


@dataclass
class PageAwareMicroOp:
    """A micro-operation with page allocation and semaphore information.

    Supports the No Bubbles paged memory pattern:
    - Loader: wait(sem_finished[page]) -> load -> signal(sem_arrived[page])
    - Consumer: wait(sem_arrived[page]) -> compute -> signal(sem_finished[page])
    - Storer: (runs after compute, uses output page)
    """

    id: int
    type: MicroOpType
    op_idx: int
    desc: str = ""

    # Data dependencies (from OpDescriptor reads/writes)
    depends_on: Set[int] = field(default_factory=set)

    # Page dependencies: must wait until these pages are released
    waits_for_pages: Set[int] = field(default_factory=set)

    # Pages acquired by this micro-op (for LOAD operations)
    acquires_pages: List[int] = field(default_factory=list)

    # Pages released by this micro-op (for STORE operations)
    releases_pages: List[int] = field(default_factory=list)

    # Semaphore operations (No Bubbles pattern)
    # List of (semaphore_idx, is_signal) tuples
    # - wait on sem_finished before loading (page is free)
    # - signal sem_arrived after loading (data ready)
    # - wait on sem_arrived before compute (data ready)
    # - signal sem_finished after compute (page can be reused)
    sem_waits: List[int] = field(default_factory=list)  # Semaphore indices to wait on
    sem_signals: List[int] = field(default_factory=list)  # Semaphore indices to signal

    # Warp role that should execute this micro-op (None = all warps)
    warp_role: Optional[WarpRole] = None

    def __hash__(self):
        return self.id


class PagedMemoryAllocator:
    """Manages paged shared memory with acquire/release semantics.

    This implements the "No Bubbles" pattern where:
    1. Each operation needs N pages to start its LOAD phase
    2. LOAD acquires pages (waits if not enough available)
    3. STORE releases pages (allows next op to acquire)

    The key insight is that Load[i+1] can start as soon as Store[j] releases
    enough pages, even if Compute[i] is still running.

    Example with 4 pages, each op needs 2 pages:
        Time 0: Load[0] acquires pages [0,1]
        Time 1: Compute[0] uses pages [0,1], Load[1] acquires pages [2,3]
        Time 2: Store[0] releases [0,1], Load[2] acquires [0,1]
        Time 3: Compute[1], Compute[2] can overlap!
    """

    def __init__(self, num_pages: int, page_size_bytes: int = 16384):
        self.num_pages = num_pages
        self.page_size_bytes = page_size_bytes

        # Track which pages are currently held by which op
        self.page_owner: Dict[int, Optional[int]] = dict.fromkeys(range(num_pages), None)

        # Track allocations for each operation
        self.allocations: Dict[int, PageAllocation] = {}

        # Track release events: op_idx -> micro_op_id that releases its pages
        self.release_events: Dict[int, int] = {}

    def pages_required(self, op_desc: OpDescriptor) -> int:
        """Calculate how many pages an operation needs.

        For now, we use a simple heuristic: 2 pages for double buffering.
        In practice, this would be computed from op.smem_size / page_size_bytes.
        """
        return 2  # Default: double buffering

    def try_acquire(self, op_idx: int, num_pages_needed: int) -> Optional[List[int]]:
        """Try to acquire pages for an operation.

        Returns list of page IDs if successful, None if not enough pages available.
        """
        available = [p for p, owner in self.page_owner.items() if owner is None]
        if len(available) >= num_pages_needed:
            allocated = available[:num_pages_needed]
            for p in allocated:
                self.page_owner[p] = op_idx
            return allocated
        return None

    def release(self, op_idx: int) -> List[int]:
        """Release all pages held by an operation."""
        released = []
        for p, owner in self.page_owner.items():
            if owner == op_idx:
                self.page_owner[p] = None
                released.append(p)
        return released

    def get_blocking_ops(self, num_pages_needed: int) -> Set[int]:
        """Get the set of operations that must release pages before we can acquire.

        Returns the minimal set of ops that, once they release, would provide
        enough pages.
        """
        available = sum(1 for owner in self.page_owner.values() if owner is None)
        if available >= num_pages_needed:
            return set()  # No blocking

        # Find which ops are holding pages
        holders: Dict[int, int] = {}  # op_idx -> num_pages_held
        for owner in self.page_owner.values():
            if owner is not None:
                holders[owner] = holders.get(owner, 0) + 1

        needed = num_pages_needed - available
        blocking = set()

        # Greedily pick ops to release (prefer those holding more pages)
        for op_idx, count in sorted(holders.items(), key=lambda x: -x[1]):
            if needed <= 0:
                break
            blocking.add(op_idx)
            needed -= count

        return blocking


class PageAwareScheduler:
    """Scheduler that considers both data dependencies AND page availability.

    This extends the dependency-aware scheduler to also track when pages
    become available, allowing loads to start as early as possible.

    Enhanced with Logical Block support and page semaphores for the No Bubbles pattern.

    Semaphore Protocol (per page):
    - sem_arrived[page]: Loader signals after load, Consumer waits before compute
    - sem_finished[page]: Consumer signals after compute, Loader waits before reusing

    This allows:
    - Load[N+1] to start as soon as Consumer[N] finishes (not waiting for Store[N])
    - Maximum overlap between Load, Compute, and Store phases
    """

    def __init__(self, config: NoBubblesConfig):
        self.config = config
        self.allocator = PagedMemoryAllocator(config.num_pages, config.page_size_bytes)
        self.micro_ops: List[PageAwareMicroOp] = []
        self._next_id = 0
        self.barrier_config: Optional[BarrierConfig] = None
        self.semaphore_config: Optional[PageSemaphoreConfig] = None

        # Initialize semaphore config
        if config.num_pages > 0:
            self.semaphore_config = PageSemaphoreConfig(num_pages=config.num_pages)

    def _new_micro_op(
        self,
        type: MicroOpType,
        op_idx: int,
        desc: str = "",
        depends_on: Optional[Set[int]] = None,
        waits_for_pages: Optional[Set[int]] = None,
        acquires_pages: Optional[List[int]] = None,
        releases_pages: Optional[List[int]] = None,
        sem_waits: Optional[List[int]] = None,
        sem_signals: Optional[List[int]] = None,
        warp_role: Optional[WarpRole] = None,
    ) -> PageAwareMicroOp:
        op = PageAwareMicroOp(
            id=self._next_id,
            type=type,
            op_idx=op_idx,
            desc=desc,
            depends_on=depends_on or set(),
            waits_for_pages=waits_for_pages or set(),
            acquires_pages=acquires_pages or [],
            releases_pages=releases_pages or [],
            sem_waits=sem_waits or [],
            sem_signals=sem_signals or [],
            warp_role=warp_role,
        )
        self._next_id += 1
        self.micro_ops.append(op)
        return op

    def configure_barriers(self, num_ops: int, total_logical_blocks: int):
        """Configure the barrier tensor for logical block synchronization."""
        self.barrier_config = BarrierConfig(
            num_ops=num_ops,
            total_logical_blocks=total_logical_blocks,
        )

    def generate_page_aware_schedule(self, ops: List[OpDescriptor], pages_per_op: int = 2) -> List[PageAwareMicroOp]:
        """Generate a schedule considering both data deps and page availability.

        This produces a schedule where:
        - Load[i] depends on Store[j] if there's a data dependency (RAW)
        - Load[i] ALSO waits for pages to be released (page dependency)
        - The schedule is optimal in terms of overlapping L/C/S phases

        Args:
            ops: List of operation descriptors with reads/writes info
            pages_per_op: Number of pages each operation needs (default 2 for double buffering)

        Returns:
            List of PageAwareMicroOp with full dependency information
        """
        self.micro_ops.clear()
        self._next_id = 0
        self.allocator = PagedMemoryAllocator(self.config.num_pages, self.config.page_size_bytes)

        n_ops = len(ops)
        if n_ops == 0:
            return self.micro_ops

        # Configure barriers for logical blocks
        total_logical = max((op.logical_grid_size or 1) for op in ops)
        self.configure_barriers(n_ops, total_logical)

        # Track micro-ops by (op_idx, type)
        load_ops: Dict[int, PageAwareMicroOp] = {}
        compute_ops: Dict[int, PageAwareMicroOp] = {}
        store_ops: Dict[int, PageAwareMicroOp] = {}

        for i, op in enumerate(ops):
            # --- Determine page dependencies ---
            # Try to acquire pages; if not enough, wait for previous stores
            pages = self.allocator.try_acquire(i, pages_per_op)
            page_deps: Set[int] = set()

            if pages is None:
                # Not enough pages - find which stores we need to wait for
                blocking_ops = self.allocator.get_blocking_ops(pages_per_op)
                for blocked_by in blocking_ops:
                    if blocked_by in store_ops:
                        page_deps.add(store_ops[blocked_by].id)

                # Force acquire after we've recorded the dependency
                # (Simulation: assume pages become available)
                # Release pages from oldest ops first
                for blocked_by in sorted(blocking_ops):
                    self.allocator.release(blocked_by)

                pages = self.allocator.try_acquire(i, pages_per_op)
                assert pages is not None, "Should have pages after release"

            # --- Determine data dependencies (RAW) ---
            data_deps: Set[int] = set()
            for j in range(i):
                if op.reads & ops[j].writes:  # RAW dependency
                    if j in store_ops:
                        data_deps.add(store_ops[j].id)

            # --- LOAD[i] ---
            all_deps = data_deps | page_deps
            load_op = self._new_micro_op(
                MicroOpType.LOAD,
                i,
                f"Load[{i}]" + (f" (acquires pages {pages})" if pages else ""),
                depends_on=all_deps,
                acquires_pages=pages or [],
            )
            load_ops[i] = load_op
            self.allocator.allocations[i] = PageAllocation(op_idx=i, page_ids=pages or [], acquired_at=load_op.id)

            # Note: Block sync via sync_threads() is deprecated.
            # Fine-grained semaphore synchronization is used instead.

            # --- COMPUTE[i] ---
            compute_op = self._new_micro_op(MicroOpType.COMPUTE, i, f"Compute[{i}]", depends_on={load_op.id})
            compute_ops[i] = compute_op

            # --- STORE[i] ---
            store_op = self._new_micro_op(
                MicroOpType.STORE,
                i,
                f"Store[{i}]" + (f" (releases pages {pages})" if pages else ""),
                depends_on={compute_op.id},
                releases_pages=pages or [],
            )
            store_ops[i] = store_op
            self.allocator.release_events[i] = store_op.id
            if i in self.allocator.allocations:
                self.allocator.allocations[i].released_at = store_op.id

            # Only global sync (cross-SM barrier) is still supported
            if op.needs_global_sync:
                self._new_micro_op(MicroOpType.SYNC_GLOBAL, i, f"GlobalSync[{i}]", depends_on={store_op.id})

        return self.micro_ops

    def generate_overlapped_schedule(self, ops: List[OpDescriptor], pages_per_op: int = 2) -> List[PageAwareMicroOp]:
        """Generate schedule optimized for async load overlap.

        This method assumes loads are async (cp.async/TMA style) and reorders
        operations to maximize overlap:

        - Issue as many LOADs as possible upfront (limited by available pages)
        - Interleave COMPUTE[i] with LOAD[i+k] where k = pages / pages_per_op
        - STORE[i] follows COMPUTE[i]

        The resulting schedule looks like a "sliding window":
            LOAD[0] LOAD[1] COMPUTE[0] LOAD[2] STORE[0] COMPUTE[1] LOAD[3] STORE[1] ...

        This is the optimal pattern from the "No Bubbles" paper.
        """
        n_ops = len(ops)
        if n_ops == 0:
            return self.micro_ops

        total_pages = self.config.num_pages
        max_concurrent = total_pages // pages_per_op  # How many ops can overlap

        # Track micro-ops
        load_ops: Dict[int, PageAwareMicroOp] = {}
        compute_ops: Dict[int, PageAwareMicroOp] = {}
        store_ops: Dict[int, PageAwareMicroOp] = {}

        # Build data dependency graph first (RAW dependencies between ops)
        data_deps_for_load: Dict[int, Set[int]] = {}  # op_idx -> set of op_idx it depends on
        for i, op in enumerate(ops):
            deps = set()
            for j in range(i):
                if op.reads & ops[j].writes:  # RAW dependency
                    deps.add(j)
            data_deps_for_load[i] = deps

        # Phase 1: Issue initial LOADs (as many as we have pages for)
        initial_loads = min(max_concurrent, n_ops)
        for i in range(initial_loads):
            pages = list(range(i * pages_per_op, (i + 1) * pages_per_op))

            # Check data dependencies - need to wait for stores of deps
            data_deps = set()
            for dep_idx in data_deps_for_load[i]:
                if dep_idx in store_ops:
                    data_deps.add(store_ops[dep_idx].id)

            load_op = self._new_micro_op(
                MicroOpType.LOAD,
                i,
                f"Load[{i}] (async, pages {pages})",
                depends_on=data_deps,
                acquires_pages=pages,
            )
            load_ops[i] = load_op
            self.allocator.allocations[i] = PageAllocation(op_idx=i, page_ids=pages, acquired_at=load_op.id)

        # Phase 2: Interleaved execution
        # For each op i, we execute COMPUTE[i], then potentially LOAD[i+k], then STORE[i]
        for i in range(n_ops):
            # COMPUTE[i] depends on LOAD[i]
            compute_op = self._new_micro_op(MicroOpType.COMPUTE, i, f"Compute[{i}]", depends_on={load_ops[i].id})
            compute_ops[i] = compute_op

            # Check if we can issue a new LOAD (for op i+max_concurrent)
            next_load_idx = i + max_concurrent
            if next_load_idx < n_ops and next_load_idx not in load_ops:
                # Reuse pages from op i (which will be released by STORE[i])
                pages = self.allocator.allocations[i].page_ids

                # This LOAD depends on STORE[i] (to free pages) and any data deps
                data_deps = set()
                for dep_idx in data_deps_for_load[next_load_idx]:
                    if dep_idx in store_ops:
                        data_deps.add(store_ops[dep_idx].id)

                # Since STORE[i] hasn't been created yet, we'll create it first
                # and add the dep after. For now, mark as "waits for pages"
                waits_for = set()
                if i in store_ops:
                    waits_for.add(store_ops[i].id)

                load_op = self._new_micro_op(
                    MicroOpType.LOAD,
                    next_load_idx,
                    f"Load[{next_load_idx}] (async, reuses pages from op {i})",
                    depends_on=waits_for | data_deps,
                    acquires_pages=pages,
                    waits_for_pages=[i],  # Wait for STORE[i] to release
                )
                load_ops[next_load_idx] = load_op
                self.allocator.allocations[next_load_idx] = PageAllocation(
                    op_idx=next_load_idx, page_ids=pages, acquired_at=load_op.id
                )

            # STORE[i] depends on COMPUTE[i]
            pages = self.allocator.allocations[i].page_ids if i in self.allocator.allocations else []
            store_op = self._new_micro_op(
                MicroOpType.STORE,
                i,
                f"Store[{i}]",
                depends_on={compute_op.id},
                releases_pages=pages,
            )
            store_ops[i] = store_op
            self.allocator.release_events[i] = store_op.id

            # Now update any pending LOADs that were waiting for this STORE
            for pending_idx in range(i + 1, min(i + max_concurrent + 1, n_ops)):
                if pending_idx in load_ops:
                    pending_load = load_ops[pending_idx]
                    if i in (pending_load.waits_for_pages or []):
                        pending_load.depends_on.add(store_op.id)

        return self.micro_ops

    def visualize_page_schedule(self) -> str:
        """Visualize the schedule with page allocation info."""
        lines = []
        lines.append("=" * 70)
        lines.append("Page-Aware Schedule (with page acquire/release)")
        lines.append(f"Total pages: {self.config.num_pages}")
        if self.barrier_config:
            lines.append(
                f"Barrier tensor: {self.barrier_config.num_ops} ops x "
                f"{self.barrier_config.total_logical_blocks} logical blocks"
            )
        lines.append("=" * 70)

        for op in self.micro_ops:
            deps_str = f"deps={list(op.depends_on)}" if op.depends_on else ""
            pages_str = ""
            if op.acquires_pages:
                pages_str = f" [ACQUIRE pages {op.acquires_pages}]"
            if op.releases_pages:
                pages_str = f" [RELEASE pages {op.releases_pages}]"

            lines.append(f"  {op.id}: {op.type.name}[{op.op_idx}] {deps_str}{pages_str}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def generate_async_pipeline_schedule(
        self, ops: List[OpDescriptor], pages_per_op: int = 2
    ) -> List[PageAwareMicroOp]:
        """Generate fully async pipeline with producer/consumer barriers.

        This implements the optimal "No Bubbles" pattern with async operations:

        1. LOAD_ASYNC[i] - Issue async load (non-blocking)
        2. COMMIT_GROUP[i] - Commit async group
        3. WAIT_LOAD[i] - Wait for load completion (before compute)
        4. COMPUTE[i] - Execute computation
        5. STORE_ASYNC[i] - Issue async store (non-blocking)
        6. (STORE completes in background, pages released when next op needs them)

        The schedule maximizes overlap by:
        - Issuing all LOADs as early as possible (up to page limit)
        - Interleaving WAIT_LOAD[i] / COMPUTE[i] with LOAD_ASYNC[i+k]
        - Issuing STORE_ASYNC immediately after COMPUTE

        Uses logical block barriers: Bar[OpIdx][LogicalID] for fine-grained sync.
        """
        n_ops = len(ops)
        if n_ops == 0:
            return self.micro_ops

        total_pages = self.config.num_pages
        max_concurrent = total_pages // pages_per_op

        # Configure barriers for logical blocks
        total_logical = max((op.logical_grid_size or 1) for op in ops)
        self.configure_barriers(n_ops, total_logical)

        # Track operations
        load_async_ops: Dict[int, PageAwareMicroOp] = {}
        wait_load_ops: Dict[int, PageAwareMicroOp] = {}
        compute_ops: Dict[int, PageAwareMicroOp] = {}
        store_async_ops: Dict[int, PageAwareMicroOp] = {}

        # Data dependencies
        data_deps = self._compute_data_dependencies(ops)

        # Phase 1: Issue initial async loads
        for i in range(min(max_concurrent, n_ops)):
            pages = list(range(i * pages_per_op, (i + 1) * pages_per_op))
            load_async_ops[i] = self._issue_async_load(i, pages, data_deps, store_async_ops)

        # Phase 2: Interleaved execution with async stores
        for i in range(n_ops):
            # WAIT_LOAD[i] - Wait for data to arrive
            wait_load_op = self._new_micro_op(
                MicroOpType.WAIT_LOAD, i, f"WaitLoad[{i}]", depends_on={load_async_ops[i].id}
            )
            wait_load_ops[i] = wait_load_op

            # Note: Block sync via sync_threads() is deprecated.
            # Fine-grained semaphore synchronization is used instead.

            # COMPUTE[i]
            compute_op = self._new_micro_op(MicroOpType.COMPUTE, i, f"Compute[{i}]", depends_on={wait_load_op.id})
            compute_ops[i] = compute_op

            # STORE_ASYNC[i] - Issue async store immediately
            pages = load_async_ops[i].acquires_pages
            store_async_op = self._new_micro_op(
                MicroOpType.STORE_ASYNC,
                i,
                f"StoreAsync[{i}]",
                depends_on={compute_op.id},
                releases_pages=pages,
            )
            store_async_ops[i] = store_async_op
            self.allocator.release_events[i] = store_async_op.id

            # Issue next async load if available
            next_idx = i + max_concurrent
            if next_idx < n_ops and next_idx not in load_async_ops:
                # Reuse pages from op i, but must wait for its store
                load_async_ops[next_idx] = self._issue_async_load(
                    next_idx, pages, data_deps, store_async_ops, wait_for_store=i
                )

        return self.micro_ops

    def _compute_data_dependencies(self, ops: List[OpDescriptor]) -> Dict[int, Set[int]]:
        """Compute RAW data dependencies between operations."""
        deps: Dict[int, Set[int]] = {}
        for i, op in enumerate(ops):
            deps[i] = set()
            for j in range(i):
                if op.reads & ops[j].writes:
                    deps[i].add(j)
        return deps

    def _issue_async_load(
        self,
        op_idx: int,
        pages: List[int],
        data_deps: Dict[int, Set[int]],
        store_ops: Dict[int, PageAwareMicroOp],
        wait_for_store: Optional[int] = None,
    ) -> PageAwareMicroOp:
        """Issue an async load operation."""
        # Compute dependencies
        deps: Set[int] = set()

        # Data dependencies: wait for stores of ops we read from
        for dep_idx in data_deps.get(op_idx, set()):
            if dep_idx in store_ops:
                deps.add(store_ops[dep_idx].id)

        # Page dependency: wait for previous op's store to release pages
        if wait_for_store is not None and wait_for_store in store_ops:
            deps.add(store_ops[wait_for_store].id)

        load_op = self._new_micro_op(
            MicroOpType.LOAD_ASYNC,
            op_idx,
            f"LoadAsync[{op_idx}] pages={pages}",
            depends_on=deps,
            acquires_pages=pages,
        )
        self.allocator.allocations[op_idx] = PageAllocation(op_idx=op_idx, page_ids=pages, acquired_at=load_op.id)

        # COMMIT_GROUP after load
        self._new_micro_op(MicroOpType.COMMIT_GROUP, op_idx, f"CommitGroup[{op_idx}]", depends_on={load_op.id})

        return load_op

    def generate_warp_specialized_schedule(
        self, ops: List[OpDescriptor], pages_per_op: int = 2
    ) -> List[PageAwareMicroOp]:
        """Generate schedule with warp specialization and page semaphores.

        This implements the full No Bubbles pattern from HazyResearch:

        Warp Roles:
        - LOADER warp: wait(sem_finished) -> async_load -> signal(sem_arrived)
        - CONSUMER warps: wait(sem_arrived) -> compute -> signal(sem_finished)
        - STORER warp: async_store (after compute signals)
        - CONTROLLER warp: orchestrates instruction flow

        Semaphore Protocol (per page):
        - sem_arrived[p]: Loader signals, Consumer waits (data is ready)
        - sem_finished[p]: Consumer signals, Loader waits (page can be reused)

        This allows maximum overlap:
        - Load[N+1] can start as soon as Compute[N] finishes (not waiting for Store[N])
        - Store[N] runs in parallel with Load[N+1] and Compute[N+1]
        """
        self.micro_ops.clear()
        self._next_id = 0
        self.allocator = PagedMemoryAllocator(self.config.num_pages, self.config.page_size_bytes)

        n_ops = len(ops)
        if n_ops == 0:
            return self.micro_ops

        total_pages = self.config.num_pages
        max_concurrent = total_pages // pages_per_op

        # Configure barriers for logical blocks
        total_logical = max((op.logical_grid_size or 1) for op in ops)
        self.configure_barriers(n_ops, total_logical)

        # Ensure semaphore config exists
        if self.semaphore_config is None:
            self.semaphore_config = PageSemaphoreConfig(num_pages=total_pages)

        # Data dependencies
        data_deps = self._compute_data_dependencies(ops)

        # Track operations by type
        load_ops: Dict[int, PageAwareMicroOp] = {}
        compute_ops: Dict[int, PageAwareMicroOp] = {}
        store_ops: Dict[int, PageAwareMicroOp] = {}

        for i, op in enumerate(ops):
            # Determine which pages this op uses (circular allocation)
            page_start = (i * pages_per_op) % total_pages
            pages = [(page_start + j) % total_pages for j in range(pages_per_op)]

            # Calculate semaphore indices for these pages
            sem_arrived = [self.semaphore_config.get_arrived_semaphore_idx(p) for p in pages]
            sem_finished = [self.semaphore_config.get_finished_semaphore_idx(p) for p in pages]

            # --- LOADER: wait(sem_finished) -> load -> signal(sem_arrived) ---
            # For first iteration of each page, sem_finished is pre-initialized to 1
            # For subsequent uses, we wait for the previous consumer to finish
            load_deps: Set[int] = set()

            # Data dependencies: wait for stores of ops we read from
            for dep_idx in data_deps.get(i, set()):
                if dep_idx in store_ops:
                    load_deps.add(store_ops[dep_idx].id)

            # Page dependency: if pages were used before, wait for that compute to signal sem_finished
            prev_user = i - max_concurrent
            if prev_user >= 0 and prev_user in compute_ops:
                load_deps.add(compute_ops[prev_user].id)

            load_op = self._new_micro_op(
                MicroOpType.LOAD_ASYNC,
                i,
                f"Load[{i}] pages={pages}",
                depends_on=load_deps,
                acquires_pages=pages,
                sem_waits=sem_finished if prev_user >= 0 else [],  # Wait for page to be free
                sem_signals=sem_arrived,  # Signal data is ready
                warp_role=WarpRole.LOADER,
            )
            load_ops[i] = load_op
            self.allocator.allocations[i] = PageAllocation(op_idx=i, page_ids=pages, acquired_at=load_op.id)

            # COMMIT_GROUP (loader warp)
            self._new_micro_op(
                MicroOpType.COMMIT_GROUP,
                i,
                f"CommitGroup[{i}]",
                depends_on={load_op.id},
                warp_role=WarpRole.LOADER,
            )

            # --- CONSUMER: wait(sem_arrived) -> compute -> signal(sem_finished) ---
            compute_op = self._new_micro_op(
                MicroOpType.COMPUTE,
                i,
                f"Compute[{i}]",
                depends_on={load_op.id},  # Structural dependency
                sem_waits=sem_arrived,  # Wait for data to be ready
                sem_signals=sem_finished,  # Signal page can be reused
                warp_role=WarpRole.CONSUMER,
            )
            compute_ops[i] = compute_op

            # --- STORER: store output to global memory ---
            store_op = self._new_micro_op(
                MicroOpType.STORE_ASYNC,
                i,
                f"Store[{i}]",
                depends_on={compute_op.id},
                releases_pages=pages,
                warp_role=WarpRole.STORER,
            )
            store_ops[i] = store_op
            self.allocator.release_events[i] = store_op.id

        return self.micro_ops

    def visualize_warp_schedule(self) -> str:
        """Visualize the warp-specialized schedule."""
        lines = []
        lines.append("=" * 80)
        lines.append("Warp-Specialized Schedule (No Bubbles)")
        num_sems = self.semaphore_config.total_semaphores if self.semaphore_config else 0
        lines.append(f"Pages: {self.config.num_pages}, Semaphores: {num_sems}")
        if self.barrier_config:
            lines.append(
                f"Barriers: {self.barrier_config.num_ops} ops x {self.barrier_config.total_logical_blocks} blocks"
            )
        lines.append("=" * 80)

        # Group by warp role
        by_role: Dict[Optional[WarpRole], List[PageAwareMicroOp]] = {}
        for op in self.micro_ops:
            role = op.warp_role
            if role not in by_role:
                by_role[role] = []
            by_role[role].append(op)

        for role in [WarpRole.LOADER, WarpRole.CONSUMER, WarpRole.STORER, None]:
            if role in by_role:
                role_name = role.name if role else "ALL"
                lines.append(f"\n--- {role_name} Warp ---")
                for op in by_role[role]:
                    sem_str = ""
                    if op.sem_waits:
                        sem_str += f" wait({op.sem_waits})"
                    if op.sem_signals:
                        sem_str += f" signal({op.sem_signals})"
                    pages_str = ""
                    if op.acquires_pages:
                        pages_str = f" pages={op.acquires_pages}"
                    lines.append(f"  {op.id}: {op.type.name}[{op.op_idx}]{pages_str}{sem_str}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)
