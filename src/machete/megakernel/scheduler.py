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


def async_load() -> Callable[[F], F]:
    """Decorator to mark a load as async (non-blocking).

    Async loads can be issued early and overlapped with previous operations.
    The scheduler will:
    1. Issue the load as early as possible (during previous compute/store)
    2. Only wait for completion when the data is actually needed

    Usage:
        @async_load()
        @reads("weights")
        @warp_role(WarpRole.LOADER)
        @cute.jit
        def load_forward(self, paged_pool, page_idx, weights, ...):
            # This load uses cp.async or TMA and can overlap
            cute.arch.cp_async_bulk(...)
            ...

    This enables the key "No Bubbles" optimization:
    - Start loading weights for Op[i+1] while Op[i] is still computing
    - No idle GPU cycles waiting for memory transfers
    """

    def decorator(func: F) -> F:
        func._machete_async_load = True
        return func

    return decorator


def prefetchable(*memory_regions: str) -> Callable[[F], F]:
    """Decorator to mark which memory regions can be prefetched.

    Prefetchable regions are loaded early (during previous op's compute)
    to hide memory latency. Typically used for weights that don't depend
    on previous operation's output.

    Usage:
        @prefetchable("weights", "bias")
        @async_load()
        @warp_role(WarpRole.LOADER)
        @cute.jit
        def load_forward(self, paged_pool, page_idx, weights, bias, activations):
            # weights and bias can be prefetched, activations cannot
            ...

    The scheduler will:
    1. Start loading weights/bias during previous op's compute
    2. Wait for activations only when they're ready (barrier sync)
    """

    def decorator(func: F) -> F:
        existing = getattr(func, "_machete_prefetchable", set())
        func._machete_prefetchable = existing | set(memory_regions)
        return func

    return decorator


def depends_on(
    op_name: str,
    granularity: str = "logical_block",
    producer_dims: Optional[Tuple[int, ...]] = None,
    consumer_dims: Optional[Tuple[int, ...]] = None,
) -> Callable[[F], F]:
    """Decorator to declare fine-grained dependency on another operation.

    This enables logical block-level synchronization instead of waiting
    for the entire previous operation to complete.

    Usage (1:1 mapping):
        @depends_on("qkv_projection")
        @reads("Q", "K", "V")
        def load_forward(self, ...):
            # Wait only for this logical block's Q, K, V to be ready
            ...

    Usage (many-to-one reduction):
        @depends_on("attention", producer_dims=(2, 8, 512), consumer_dims=(2, 8))
        def load_forward(self, ...):
            # Consumer (batch, head) waits for ALL producer (batch, head, seq) blocks
            # Granularity auto-inferred as "reduction"
            ...

    Usage (one-to-many broadcast):
        @depends_on("weights", producer_dims=(2, 8), consumer_dims=(2, 8, 512))
        def load_forward(self, ...):
            # Consumer (batch, head, seq) waits for producer (batch, head)
            # Granularity auto-inferred as "broadcast"
            ...

    Args:
        op_name: Name of the producer operation to depend on
        granularity: "kernel", "logical_block", "reduction", "broadcast", or "tile"
                     Auto-inferred from dims if producer_dims/consumer_dims provided
        producer_dims: Shape of producer's logical grid (e.g., (batch, head, seq))
        consumer_dims: Shape of consumer's logical grid (e.g., (batch, head))

    Granularity options:
    - "kernel": Wait for entire operation (coarse, like cudaGridDependencySynchronize)
    - "logical_block": Wait for specific logical block (fine-grained 1:1, default)
    - "reduction": Wait for multiple producer blocks (many-to-one)
    - "broadcast": Multiple consumers wait for one producer (one-to-many)
    - "tile": Wait for specific tile within logical block (finest)
    """

    def decorator(func: F) -> F:
        existing = getattr(func, "_machete_depends_on", [])
        # Store as tuple: (op_name, granularity, producer_dims, consumer_dims)
        func._machete_depends_on = existing + [(op_name, granularity, producer_dims, consumer_dims)]
        return func

    return decorator


def get_method_dependencies(method: Callable) -> tuple[Set[str], Set[str]]:
    """Extract dependency info from a decorated L/C/S method.

    Returns:
        (reads, writes)
    """
    reads_set = getattr(method, "_machete_reads", set())
    writes_set = getattr(method, "_machete_writes", set())
    return reads_set, writes_set


def get_method_warp_role(method: Callable) -> Optional[WarpRole]:
    """Extract warp role annotation from a method."""
    return getattr(method, "_machete_warp_role", None)


def is_async_load(method: Callable) -> bool:
    """Check if a method is marked as async load."""
    return getattr(method, "_machete_async_load", False)


def get_prefetchable_regions(method: Callable) -> Set[str]:
    """Get memory regions that can be prefetched."""
    return getattr(method, "_machete_prefetchable", set())


@dataclass
class CrossOpDependencyInfo:
    """Parsed cross-operation dependency information from @depends_on decorator."""

    op_name: str
    granularity: str
    producer_dims: Optional[Tuple[int, ...]] = None
    consumer_dims: Optional[Tuple[int, ...]] = None

    def to_dimension_mapping(self) -> Optional["DimensionMapping"]:
        """Create DimensionMapping if dims are specified."""
        if self.producer_dims is not None and self.consumer_dims is not None:
            return DimensionMapping(
                producer_dims=self.producer_dims,
                consumer_dims=self.consumer_dims,
            )
        return None


def get_cross_op_dependencies(method: Callable) -> List[CrossOpDependencyInfo]:
    """Get cross-operation dependencies declared on a method.

    Returns:
        List of CrossOpDependencyInfo objects
    """
    raw_deps = getattr(method, "_machete_depends_on", [])
    result = []
    for dep in raw_deps:
        if len(dep) == 2:
            # Legacy format: (op_name, granularity)
            result.append(CrossOpDependencyInfo(op_name=dep[0], granularity=dep[1]))
        elif len(dep) == 4:
            # New format: (op_name, granularity, producer_dims, consumer_dims)
            result.append(CrossOpDependencyInfo(
                op_name=dep[0],
                granularity=dep[1],
                producer_dims=dep[2],
                consumer_dims=dep[3],
            ))
    return result


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
    has_async_load = False
    prefetchable: Set[str] = set()
    cross_deps: List[CrossOpDependencyInfo] = []

    for method in [load_method, compute_method, store_method]:
        if method is not None:
            reads_set, writes_set = get_method_dependencies(method)
            all_reads |= reads_set
            all_writes |= writes_set

            # Extract async load info from load method
            if method == load_method:
                has_async_load = is_async_load(method)
                prefetchable |= get_prefetchable_regions(method)
                cross_deps.extend(get_cross_op_dependencies(method))

    # Get kernel class name for debugging
    name = getattr(kernel, "__class__", type(kernel)).__name__

    # Allow override from kernel attribute (for explicit naming)
    if hasattr(kernel, "op_name"):
        name = kernel.op_name

    # Extract logical grid info if available
    logical_grid_size = None
    if hasattr(kernel, "get_logical_grid_size"):
        # Defer actual computation to runtime when args are available
        logical_grid_size = -1  # Sentinel: compute at runtime

    # Check for warp specialization
    uses_warp_spec = getattr(kernel, "uses_warp_specialization", False)

    # Build descriptor with inferred scheduling mode
    desc = OpDescriptor(
        name=name,
        op_idx=op_idx,
        reads=all_reads,
        writes=all_writes,
        allow_early_load=True,  # Determined by dependency analysis at schedule time
        logical_grid_size=logical_grid_size,
        async_load=has_async_load,
        uses_warp_specialization=uses_warp_spec,
        prefetchable_regions=prefetchable,
        cross_op_deps=cross_deps,
    )

    # Auto-infer and set scheduling mode
    desc.scheduling_mode = desc.infer_scheduling_mode()

    return desc


class SchedulingMode(Enum):
    """Per-operation scheduling mode.

    Each operation in a megakernel can have its own scheduling mode,
    allowing heterogeneous kernels to be fused together optimally.

    - SEQUENTIAL: Simple L->C->S, all threads participate. Best for simple kernels.
    - ASYNC: Async loads (cp.async/TMA) with overlapped L/C/S. Best for memory-bound kernels.
    - WARP_SPECIALIZED: Dedicated loader/consumer/storer warps. Best for compute-heavy kernels.

    The scheduler automatically detects the appropriate mode from kernel attributes,
    or it can be set explicitly.
    """

    SEQUENTIAL = auto()  # Simple L->C->S, all threads participate
    ASYNC = auto()  # Async loads with overlapped L/C/S phases
    WARP_SPECIALIZED = auto()  # Dedicated warps for L/C/S roles


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
    """Describes an operation's memory access patterns and scheduling mode.

    This allows the scheduler to determine which operations can run in parallel
    and how each operation should be scheduled (sequential, async, warp-specialized).

    Note: Synchronization between L/C/S phases is handled via fine-grained
    semaphores, not sync_threads().
    """

    name: str
    op_idx: int

    # Memory regions this operation reads from (as tensor IDs or symbolic names)
    reads: Set[str] = field(default_factory=set)
    # Memory regions this operation writes to
    writes: Set[str] = field(default_factory=set)

    # Scheduling mode for this operation (auto-detected or explicitly set)
    scheduling_mode: SchedulingMode = SchedulingMode.SEQUENTIAL

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

    # Cross-kernel scheduling support
    # If True, the load uses async mechanisms (cp.async, TMA) and can be issued early
    async_load: bool = False

    # If True, uses warp specialization (dedicated loader/consumer/storer warps)
    uses_warp_specialization: bool = False

    # Memory regions that can be prefetched (loaded during previous op's compute)
    # These typically include weights that don't depend on previous activations
    prefetchable_regions: Set[str] = field(default_factory=set)

    # Cross-operation dependencies with dimension mapping support
    # Each entry contains op_name, granularity, and optional producer/consumer dims
    cross_op_deps: List["CrossOpDependencyInfo"] = field(default_factory=list)

    def infer_scheduling_mode(self) -> SchedulingMode:
        """Infer the best scheduling mode based on operation attributes."""
        if self.uses_warp_specialization:
            return SchedulingMode.WARP_SPECIALIZED
        elif self.async_load:
            return SchedulingMode.ASYNC
        return SchedulingMode.SEQUENTIAL


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


# ============================================================================
# Cross-Kernel Dependencies (Fine-Grained Inter-Operation Sync)
# ============================================================================


class DependencyGranularity(Enum):
    """Granularity of inter-operation dependencies.

    This determines at what level operations synchronize:
    - KERNEL: Wait for entire previous operation to complete (coarse, like PDL)
    - LOGICAL_BLOCK: Wait for specific logical block to complete (fine-grained, 1:1)
    - REDUCTION: Wait for multiple producer blocks to complete (many-to-one)
    - BROADCAST: One producer block feeds multiple consumer blocks (one-to-many)
    - TILE: Wait for specific tile within a logical block (finest)
    """

    KERNEL = auto()  # Coarse: wait for entire op
    LOGICAL_BLOCK = auto()  # Fine: wait for specific logical block (1:1)
    REDUCTION = auto()  # Many-to-one: consumer waits for multiple producer blocks
    BROADCAST = auto()  # One-to-many: multiple consumers wait for one producer block
    TILE = auto()  # Finest: wait for specific tile


@dataclass
class DimensionMapping:
    """Describes dimension mapping between producer and consumer operations.

    Supports three patterns:
    1. REDUCTION (many-to-one): Consumer has fewer dims, waits for multiple producers
       - Producer: (batch, head, seq) -> Consumer: (batch, head)
       - Consumer block (b,h) waits for ALL producer blocks (b,h,0..seq-1)

    2. BROADCAST (one-to-many): Consumer has more dims, one producer feeds many consumers
       - Producer: (batch, head) -> Consumer: (batch, head, seq)
       - Consumer block (b,h,s) waits for producer block (b,h)

    3. 1:1 mapping: Same dims, direct correspondence
       - Producer: (batch, head) -> Consumer: (batch, head)

    The mapping automatically detects which pattern applies based on dims.

    Example (Reduction):
        DimensionMapping(
            producer_dims=(2, 8, 512),  # (batch, head, seq)
            consumer_dims=(2, 8),        # (batch, head)
        )
        # Automatically detects: reduction_axes=(2,)
        # Consumer block 5 waits for producer blocks (0,5,0)..(0,5,511)

    Example (Broadcast):
        DimensionMapping(
            producer_dims=(2, 8),        # (batch, head)
            consumer_dims=(2, 8, 512),   # (batch, head, seq)
        )
        # Automatically detects: broadcast_axes=(2,)
        # Consumer block (0,5,100) waits for producer block (0,5)
    """

    producer_dims: Tuple[int, ...]  # Shape of producer logical grid
    consumer_dims: Tuple[int, ...]  # Shape of consumer logical grid

    # Auto-computed fields
    _reduction_axes: Tuple[int, ...] = field(default=(), repr=False, init=False)
    _broadcast_axes: Tuple[int, ...] = field(default=(), repr=False, init=False)
    _is_reduction: bool = field(default=False, repr=False, init=False)
    _is_broadcast: bool = field(default=False, repr=False, init=False)

    def __post_init__(self):
        """Auto-detect mapping type from dimensions."""
        p_len, c_len = len(self.producer_dims), len(self.consumer_dims)

        if p_len == c_len:
            # Same rank - check for 1:1 or partial match
            if self.producer_dims == self.consumer_dims:
                # Pure 1:1 mapping
                pass
            else:
                raise ValueError(
                    f"Same-rank dims must match exactly: producer={self.producer_dims}, "
                    f"consumer={self.consumer_dims}"
                )
        elif p_len > c_len:
            # Reduction: producer has more dims
            self._is_reduction = True
            # Find which axes are reduced (removed in consumer)
            # Assume consumer dims are a prefix/subset of producer dims
            self._reduction_axes = self._find_reduced_axes()
        else:
            # Broadcast: consumer has more dims
            self._is_broadcast = True
            # Find which axes are broadcast (added in consumer)
            self._broadcast_axes = self._find_broadcast_axes()

    def _find_reduced_axes(self) -> Tuple[int, ...]:
        """Find axes that are reduced (present in producer, absent in consumer)."""
        # Try to match consumer dims to producer dims
        # Common case: reduction over trailing dims
        c_len = len(self.consumer_dims)
        if self.producer_dims[:c_len] == self.consumer_dims:
            # Trailing reduction
            return tuple(range(c_len, len(self.producer_dims)))

        # General case: find which dims match
        reduced = []
        c_idx = 0
        for p_idx, p_dim in enumerate(self.producer_dims):
            if c_idx < len(self.consumer_dims) and p_dim == self.consumer_dims[c_idx]:
                c_idx += 1
            else:
                reduced.append(p_idx)

        if c_idx != len(self.consumer_dims):
            raise ValueError(
                f"Cannot match consumer dims {self.consumer_dims} to producer dims "
                f"{self.producer_dims} for reduction"
            )
        return tuple(reduced)

    def _find_broadcast_axes(self) -> Tuple[int, ...]:
        """Find axes that are broadcast (absent in producer, present in consumer)."""
        # Try to match producer dims to consumer dims
        p_len = len(self.producer_dims)
        if self.consumer_dims[:p_len] == self.producer_dims:
            # Trailing broadcast
            return tuple(range(p_len, len(self.consumer_dims)))

        # General case: find which dims are added
        broadcast = []
        p_idx = 0
        for c_idx, c_dim in enumerate(self.consumer_dims):
            if p_idx < len(self.producer_dims) and c_dim == self.producer_dims[p_idx]:
                p_idx += 1
            else:
                broadcast.append(c_idx)

        if p_idx != len(self.producer_dims):
            raise ValueError(
                f"Cannot match producer dims {self.producer_dims} to consumer dims "
                f"{self.consumer_dims} for broadcast"
            )
        return tuple(broadcast)

    @property
    def is_reduction(self) -> bool:
        """True if this is a many-to-one (reduction) mapping."""
        return self._is_reduction

    @property
    def is_broadcast(self) -> bool:
        """True if this is a one-to-many (broadcast) mapping."""
        return self._is_broadcast

    @property
    def is_one_to_one(self) -> bool:
        """True if this is a 1:1 mapping."""
        return not self._is_reduction and not self._is_broadcast

    @property
    def reduction_axes(self) -> Tuple[int, ...]:
        """Axes reduced in producer (for REDUCTION pattern)."""
        return self._reduction_axes

    @property
    def broadcast_axes(self) -> Tuple[int, ...]:
        """Axes broadcast in consumer (for BROADCAST pattern)."""
        return self._broadcast_axes

    @property
    def producer_total_blocks(self) -> int:
        """Total number of producer logical blocks."""
        result = 1
        for d in self.producer_dims:
            result *= d
        return result

    @property
    def consumer_total_blocks(self) -> int:
        """Total number of consumer logical blocks."""
        result = 1
        for d in self.consumer_dims:
            result *= d
        return result

    @property
    def blocks_per_consumer(self) -> int:
        """Number of producer blocks each consumer block depends on."""
        if self._is_reduction:
            result = 1
            for axis in self._reduction_axes:
                result *= self.producer_dims[axis]
            return result
        return 1  # Broadcast and 1:1 always wait for 1 producer

    def _linear_to_coords(self, linear: int, dims: Tuple[int, ...]) -> Tuple[int, ...]:
        """Convert linear index to coordinate tuple."""
        coords = []
        remaining = linear
        for dim in reversed(dims):
            coords.append(remaining % dim)
            remaining //= dim
        return tuple(reversed(coords))

    def _coords_to_linear(self, coords: Tuple[int, ...], dims: Tuple[int, ...]) -> int:
        """Convert coordinate tuple to linear index."""
        result = 0
        stride = 1
        for i in range(len(dims) - 1, -1, -1):
            result += coords[i] * stride
            stride *= dims[i]
        return result

    def get_producer_blocks(self, consumer_logical: int) -> List[int]:
        """Get all producer logical blocks that this consumer block depends on.

        For REDUCTION: returns multiple producer blocks
        For BROADCAST: returns single producer block
        For 1:1: returns single producer block (same index)
        """
        if self.is_one_to_one:
            return [consumer_logical]

        consumer_coords = self._linear_to_coords(consumer_logical, self.consumer_dims)

        if self._is_broadcast:
            # One-to-many: consumer (b,h,s) depends on producer (b,h)
            # Extract only the non-broadcast coordinates
            producer_coords = tuple(
                c for i, c in enumerate(consumer_coords) if i not in self._broadcast_axes
            )
            return [self._coords_to_linear(producer_coords, self.producer_dims)]

        if self._is_reduction:
            # Many-to-one: consumer (b,h) depends on producer (b,h,0..seq-1)
            # Generate all producer blocks by iterating over reduced axes
            non_reduced_coords = tuple(
                c for i, c in enumerate(consumer_coords)
            )

            # Build all combinations of reduction indices
            reduction_ranges = [range(self.producer_dims[axis]) for axis in self._reduction_axes]

            producer_blocks = []
            if not reduction_ranges:
                return [consumer_logical]

            # Generate cartesian product
            indices_list = [[]]
            for r in reduction_ranges:
                indices_list = [existing + [val] for existing in indices_list for val in r]

            for reduction_indices in indices_list:
                # Insert reduction indices into coordinates
                producer_coords = list(non_reduced_coords)
                for axis, idx in zip(sorted(self._reduction_axes), reduction_indices):
                    producer_coords.insert(axis, idx)
                producer_blocks.append(self._coords_to_linear(tuple(producer_coords), self.producer_dims))

            return producer_blocks

        return [consumer_logical]

    def get_producer_block_range(self, consumer_logical: int) -> Tuple[int, int, int]:
        """Get producer block range for optimized contiguous access.

        If reduction is over trailing axes (most common), blocks are contiguous.

        Returns:
            (start, end, stride) tuple, or (0, 0, 0) if non-contiguous
        """
        if not self._is_reduction:
            return (0, 0, 0)

        # Check if reduction is over trailing axes only (contiguous case)
        p_len = len(self.producer_dims)
        trailing_axes = tuple(range(len(self.consumer_dims), p_len))
        if self._reduction_axes == trailing_axes:
            # Contiguous case: blocks are sequential
            blocks_count = self.blocks_per_consumer
            base = consumer_logical * blocks_count
            return (base, base + blocks_count, 1)

        return (0, 0, 0)

    def infer_granularity(self) -> DependencyGranularity:
        """Infer the appropriate DependencyGranularity from this mapping."""
        if self._is_reduction:
            return DependencyGranularity.REDUCTION
        if self._is_broadcast:
            return DependencyGranularity.BROADCAST
        return DependencyGranularity.LOGICAL_BLOCK


@dataclass
class InterOpDependency:
    """Declares a dependency between two operations with flexible dimension mapping.

    This enables fine-grained cross-operation synchronization where:
    - Attention head N can start as soon as Q[N], K[N], V[N] are ready
    - No need to wait for all Q, K, V to complete

    Example 1: 1:1 dependency (Attention depends on QKV projection)
        InterOpDependency(
            producer_op="qkv_proj",
            consumer_op="attention",
            producer_outputs={"Q", "K", "V"},
            consumer_inputs={"Q", "K", "V"},
        )

    Example 2: Many-to-one reduction (Output proj depends on all seq blocks of attention)
        InterOpDependency(
            producer_op="attention",
            consumer_op="output_proj",
            dim_mapping=DimensionMapping(
                producer_dims=(2, 8, 512),  # (batch, head, seq)
                consumer_dims=(2, 8),        # (batch, head)
            ),
            # granularity auto-inferred as REDUCTION
        )

    Example 3: One-to-many broadcast (Attention heads depend on shared weights)
        InterOpDependency(
            producer_op="weight_load",
            consumer_op="attention",
            dim_mapping=DimensionMapping(
                producer_dims=(2, 8),        # (batch, head)
                consumer_dims=(2, 8, 512),   # (batch, head, seq)
            ),
            # granularity auto-inferred as BROADCAST
        )
    """

    producer_op: str  # Name/ID of producing operation
    consumer_op: str  # Name/ID of consuming operation

    # Granularity can be set explicitly or auto-inferred from dim_mapping
    granularity: Optional[DependencyGranularity] = None

    # Memory regions that create the dependency
    producer_outputs: Set[str] = field(default_factory=set)  # What producer writes
    consumer_inputs: Set[str] = field(default_factory=set)  # What consumer reads

    # Dimension mapping for non-1:1 dependencies (reduction, broadcast)
    # When set, granularity is auto-inferred if not explicitly provided
    dim_mapping: Optional[DimensionMapping] = None

    # Legacy: custom mapping function for advanced cases
    logical_mapping: Optional[Callable[[int], int]] = None

    def __post_init__(self):
        """Validate and auto-configure."""
        # Auto-infer granularity from dim_mapping if not set
        if self.dim_mapping is not None and self.granularity is None:
            self.granularity = self.dim_mapping.infer_granularity()
        elif self.granularity is None:
            self.granularity = DependencyGranularity.LOGICAL_BLOCK

    def get_producer_logical_block(self, consumer_logical: int) -> int:
        """Get the producer logical block that this consumer block depends on.

        For 1:1 and broadcast mappings. Use get_producer_logical_blocks() for reductions.
        """
        if self.dim_mapping is not None:
            blocks = self.dim_mapping.get_producer_blocks(consumer_logical)
            return blocks[0] if blocks else consumer_logical
        if self.logical_mapping is not None:
            return self.logical_mapping(consumer_logical)
        return consumer_logical  # Default: 1:1 mapping

    def get_producer_logical_blocks(self, consumer_logical: int) -> List[int]:
        """Get all producer logical blocks that this consumer block depends on.

        For REDUCTION granularity, returns multiple blocks.
        For BROADCAST/LOGICAL_BLOCK, returns a single-element list.
        """
        if self.dim_mapping is not None:
            return self.dim_mapping.get_producer_blocks(consumer_logical)
        return [self.get_producer_logical_block(consumer_logical)]

    def get_wait_count(self, consumer_logical: int) -> int:
        """Get the number of producer completions to wait for.

        For REDUCTION: number of producer blocks to wait for
        For BROADCAST/LOGICAL_BLOCK: 1
        For KERNEL: total producer blocks (returns -1 as sentinel)
        """
        if self.dim_mapping is not None:
            return self.dim_mapping.blocks_per_consumer
        elif self.granularity == DependencyGranularity.KERNEL:
            return -1  # Sentinel: use producer's total_logical_blocks
        return 1

    def is_contiguous_reduction(self) -> bool:
        """Check if reduction blocks form a contiguous range (optimization)."""
        if self.dim_mapping is None or not self.dim_mapping.is_reduction:
            return False
        # Check via get_producer_block_range
        start, end, stride = self.dim_mapping.get_producer_block_range(0)
        return stride > 0

    def get_producer_block_range(self, consumer_logical: int) -> Tuple[int, int, int]:
        """Get contiguous producer block range for optimized waiting.

        Returns (start, end, stride) if contiguous, (0, 0, 0) otherwise.
        """
        if self.dim_mapping:
            return self.dim_mapping.get_producer_block_range(consumer_logical)
        return (0, 0, 0)


@dataclass
class GlobalBarrierTensor:
    """Configuration for a persistent global barrier tensor across operations.

    This tensor lives in global memory and persists across megakernel launches,
    enabling cross-kernel synchronization without kernel launch overhead.

    Layout: Bar[op_idx, logical_id] -> int32 counter
    - Producer: atomicAdd(&Bar[op_idx, logical_id], 1) after store completes
    - Consumer: while(Bar[dep_op_idx, logical_id] < expected) nanosleep()

    This replaces coarse-grained PDL (cudaGridDependencySynchronize) with
    fine-grained logical block-level synchronization.
    """

    num_ops: int  # Total operations across all fused kernels
    total_logical_blocks: int  # Max logical grid size
    tensor: Optional[object] = None  # The actual torch tensor (set at runtime)

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the barrier tensor."""
        return (self.num_ops, self.total_logical_blocks)

    @property
    def num_counters(self) -> int:
        """Total number of barrier counters."""
        return self.num_ops * self.total_logical_blocks

    def reset(self):
        """Reset all barrier counters to 0 (call between inference batches)."""
        if self.tensor is not None:
            self.tensor.zero_()


@dataclass
class ReductionBarrierConfig:
    """Configuration for reduction barriers (many-to-one dependencies).

    When a consumer operation has fewer logical blocks than its producer
    (e.g., attention (batch,head,seq) -> output_proj (batch,head)), we need
    a two-level barrier system:

    1. Fine-grained barriers: Producer signals per logical block (existing)
    2. Reduction barriers: Accumulated counter for consumer to wait on

    Layout: ReductionBar[consumer_op_idx, consumer_logical_id] -> int32 counter
    - Producer: atomicAdd(&ReductionBar[consumer_op, consumer_logical], 1) after store
    - Consumer: while(ReductionBar[op_idx, logical_id] < blocks_per_consumer) nanosleep()

    Example:
        Producer: attention with (batch=2, head=8, seq=512) = 8192 blocks
        Consumer: output_proj with (batch=2, head=8) = 16 blocks

        Each consumer block waits for 512 producer blocks to complete.
        ReductionBar[output_proj_idx, 0] counts completions for (batch=0, head=0)
        ReductionBar[output_proj_idx, 1] counts completions for (batch=0, head=1)
        etc.
    """

    # List of reduction dependencies: (producer_op_idx, consumer_op_idx, dim_mapping)
    reductions: List[Tuple[int, int, "DimensionMapping"]] = field(default_factory=list)

    # Total consumer blocks across all reduction consumers
    total_consumer_blocks: int = 0

    # Mapping: (consumer_op_idx, consumer_logical) -> barrier_index
    # This allows multiple reduction dependencies to share one tensor
    _barrier_offsets: Dict[int, int] = field(default_factory=dict, repr=False)

    tensor: Optional[object] = None  # The actual torch tensor (set at runtime)

    def add_reduction(
        self,
        producer_op_idx: int,
        consumer_op_idx: int,
        dim_mapping: "DimensionMapping",
    ):
        """Register a reduction dependency."""
        self.reductions.append((producer_op_idx, consumer_op_idx, dim_mapping))

        # Allocate barrier space for this consumer
        if consumer_op_idx not in self._barrier_offsets:
            self._barrier_offsets[consumer_op_idx] = self.total_consumer_blocks
            self.total_consumer_blocks += dim_mapping.consumer_total_blocks

    def get_barrier_index(self, consumer_op_idx: int, consumer_logical: int) -> int:
        """Get the barrier index for a consumer logical block."""
        offset = self._barrier_offsets.get(consumer_op_idx, 0)
        return offset + consumer_logical

    def get_consumer_logical_from_producer(
        self,
        producer_op_idx: int,
        consumer_op_idx: int,
        producer_logical: int,
    ) -> int:
        """Get the consumer logical block that a producer block contributes to.

        For reduction, multiple producer blocks map to one consumer block.
        This returns which consumer block to signal when producer completes.
        """
        for prod_idx, cons_idx, dim_mapping in self.reductions:
            if prod_idx == producer_op_idx and cons_idx == consumer_op_idx:
                # Reverse mapping: find which consumer block this producer belongs to
                # For trailing-axis reduction, it's: consumer_logical = producer_logical // blocks_per_consumer
                if dim_mapping.is_reduction:
                    return producer_logical // dim_mapping.blocks_per_consumer
        return producer_logical  # Fallback to 1:1

    def get_wait_count(self, consumer_op_idx: int) -> int:
        """Get the number of producer completions to wait for."""
        for _, cons_idx, dim_mapping in self.reductions:
            if cons_idx == consumer_op_idx:
                return dim_mapping.blocks_per_consumer
        return 1

    def get_reduction_info(self, consumer_op_idx: int) -> Optional[Tuple[int, "DimensionMapping"]]:
        """Get reduction info for a consumer operation.

        Returns:
            (producer_op_idx, dim_mapping) if consumer has a reduction dependency, else None
        """
        for prod_idx, cons_idx, dim_mapping in self.reductions:
            if cons_idx == consumer_op_idx:
                return (prod_idx, dim_mapping)
        return None

    @property
    def shape(self) -> Tuple[int,]:
        """Shape of the reduction barrier tensor."""
        return (max(1, self.total_consumer_blocks),)

    def reset(self):
        """Reset all reduction barrier counters to 0."""
        if self.tensor is not None:
            self.tensor.zero_()


@dataclass
class PrefetchDescriptor:
    """Describes a prefetch operation for overlapping load with previous compute.

    This enables the key optimization: start loading weights for Op[i+1]
    while Op[i] is still computing, as long as there's no data dependency.
    """

    op_idx: int  # Operation whose data to prefetch
    prefetch_during: int  # Operation during whose compute to prefetch
    memory_regions: Set[str]  # Which memory regions to prefetch (e.g., "weights")
    pages_needed: int  # Number of shared memory pages needed

    # Barrier to wait on before prefetch can start (producer must have stored)
    wait_barrier: Optional[Tuple[int, int]] = None  # (op_idx, logical_id) or None


# ============================================================================
# Modular Code Generator Architecture
# ============================================================================


@dataclass
class CodeGenContext:
    """Context passed to code generators for generating kernel code.

    This provides all necessary information and utilities for generating
    operation-specific code, supporting modularity and reuse.
    """

    instructions: List[Dict]  # Megakernel instruction list
    mapping: Dict[int, List[int]]  # Argument mapping per operation
    traced: bool  # Whether tracing is enabled
    paged_pool_bytes: int  # Size of paged pool
    num_stages: int  # Number of pipeline stages

    # Utility functions passed from core
    make_smem_arg_getter: Callable[[int], Callable[[str], str]]
    emit_trace_start: Optional[Callable] = None
    emit_trace_end: Optional[Callable] = None

    # Trace type dicts
    tracing_phases: Optional[Dict] = None
    sync_trace_types: Optional[Dict] = None

    # Reduction barrier config
    reduction_barrier_config: Optional["ReductionBarrierConfig"] = None


class OpCodeGenerator:
    """Base class for per-operation code generators.

    Each scheduling mode has its own code generator that produces
    the L/C/S code for operations using that mode.

    Subclasses must implement:
    - generate_op_code(): Generate code for a single operation
    - generate_init_code(): Generate mode-specific initialization (optional)
    - generate_finalize_code(): Generate mode-specific cleanup (optional)
    """

    def generate_init_code(self, ctx: CodeGenContext, ops: List[Dict]) -> List[str]:
        """Generate initialization code for this scheduling mode.

        Called once before any operations of this mode are generated.
        """
        return []

    def generate_op_code(
        self,
        ctx: CodeGenContext,
        op_idx: int,
        indent: str = "        ",
    ) -> List[str]:
        """Generate L/C/S code for a single operation.

        Args:
            ctx: Code generation context
            op_idx: Index of the operation to generate
            indent: Base indentation for generated code

        Returns:
            List of code lines
        """
        raise NotImplementedError

    def generate_finalize_code(self, ctx: CodeGenContext, ops: List[Dict]) -> List[str]:
        """Generate cleanup code for this scheduling mode.

        Called once after all operations of this mode are generated.
        """
        return []


class SequentialCodeGenerator(OpCodeGenerator):
    """Code generator for sequential (all-threads) L/C/S execution.

    Generates simple Load -> Compute -> Store sequence where all threads
    participate in each phase with semaphore synchronization between phases.
    """

    def generate_init_code(self, ctx: CodeGenContext, ops: List[Dict]) -> List[str]:
        """Allocate semaphores for sequential synchronization."""
        code = []
        code.append("")
        code.append("        # Sequential mode: fine-grained synchronization semaphores")
        code.append("        seq_semaphores = smem_alloc.allocate_tensor(cute.Int32, cute.make_layout(2))")
        code.append("        if tidx == 0:")
        code.append("            seq_semaphores[0] = Int32(0)  # sem_load_done")
        code.append("            seq_semaphores[1] = Int32(0)  # sem_compute_done")
        code.append("        cute.arch.fence_view_async_shared()")
        return code

    def generate_op_code(
        self,
        ctx: CodeGenContext,
        op_idx: int,
        indent: str = "        ",
    ) -> List[str]:
        """Generate sequential L/C/S code for one operation."""
        ops = []
        args_str = ", ".join([f"arg_{idx}" for idx in ctx.mapping[op_idx]])
        get_smem_arg = ctx.make_smem_arg_getter(op_idx)
        smem_str = get_smem_arg("page_idx")

        # Reset semaphores
        ops.append(f"{indent}# Op {op_idx} (Sequential mode)")
        ops.append(f"{indent}if tidx == 0:")
        ops.append(f"{indent}    seq_semaphores[0] = Int32(0)")
        ops.append(f"{indent}    seq_semaphores[1] = Int32(0)")
        ops.append(f"{indent}cute.arch.fence_view_async_shared()")

        # LOAD
        if ctx.traced and ctx.emit_trace_start:
            ctx.emit_trace_start(ops, indent)
        ops.append(f"{indent}op_{op_idx}_load(paged_pool, page_idx, logical_idx, {smem_str}{args_str})")
        if ctx.traced and ctx.emit_trace_end:
            ctx.emit_trace_end(ops, indent, ctx.tracing_phases, MicroOpType.LOAD, op_idx)

        # Signal load done
        ops.append(f"{indent}cute.arch.fence_view_async_shared()")
        ops.append(f"{indent}if tidx == 0:")
        ops.append(f"{indent}    atomic_add_i32(1, seq_semaphores.iterator)")

        # Wait for load
        ops.append(f"{indent}if tidx != 0:")
        ops.append(f"{indent}    while atomic_add_i32(0, seq_semaphores.iterator) < Int32(1):")
        ops.append(f"{indent}        nanosleep(Int32(20))")

        # COMPUTE
        if ctx.traced and ctx.emit_trace_start:
            ctx.emit_trace_start(ops, indent)
        ops.append(f"{indent}op_{op_idx}_compute(logical_idx, {smem_str}{args_str})")
        if ctx.traced and ctx.emit_trace_end:
            ctx.emit_trace_end(ops, indent, ctx.tracing_phases, MicroOpType.COMPUTE, op_idx)

        # Signal compute done
        ops.append(f"{indent}cute.arch.fence_view_async_shared()")
        ops.append(f"{indent}if tidx == 0:")
        ops.append(f"{indent}    atomic_add_i32(1, seq_semaphores.iterator + Int32(1))")

        # Wait for compute
        ops.append(f"{indent}if tidx != 0:")
        ops.append(f"{indent}    while atomic_add_i32(0, seq_semaphores.iterator + Int32(1)) < Int32(1):")
        ops.append(f"{indent}        nanosleep(Int32(20))")

        # STORE
        if ctx.traced and ctx.emit_trace_start:
            ctx.emit_trace_start(ops, indent)
        ops.append(f"{indent}op_{op_idx}_store(paged_pool, page_idx, logical_idx, {smem_str}{args_str})")
        if ctx.traced and ctx.emit_trace_end:
            ctx.emit_trace_end(ops, indent, ctx.tracing_phases, MicroOpType.STORE, op_idx)

        ops.append("")
        return ops


class WarpSpecializedCodeGenerator(OpCodeGenerator):
    """Code generator for warp-specialized L/C/S execution.

    Generates code where dedicated warps handle loader/consumer/storer roles,
    enabling maximum overlap between phases via the No Bubbles pattern.
    """

    def __init__(self, warp_config: Optional[WarpConfig] = None):
        self.warp_config = warp_config or WarpConfig()
        self.num_consumer = self.warp_config.num_consumer_warps
        self.loader_warp = self.num_consumer
        self.storer_warp = self.num_consumer + self.warp_config.num_loader_warps

    def generate_init_code(self, ctx: CodeGenContext, ops: List[Dict]) -> List[str]:
        """Initialize warp-specialized semaphores and persistent loop variables."""
        code = []
        num_ops = len(ops)
        num_stages = max(1, ctx.num_stages)

        code.append("")
        code.append("        # Warp-Specialized mode: persistent No Bubbles pattern")
        code.append(f"        # Consumer warps: 0-{self.num_consumer - 1}, "
                    f"Loader: {self.loader_warp}, Storer: {self.storer_warp}")
        code.append("        warp_id = tidx // Int32(32)")
        code.append("        lane_id = tidx % Int32(32)")

        # Semaphores: stages * ops * 3 types (LOAD_DONE, COMPUTE_DONE, FINISHED)
        num_sems = num_stages * num_ops * 3 + 1
        code.append(f"        warp_semaphores = smem_alloc.allocate_tensor(cute.Int32, cute.make_layout({num_sems}))")
        code.append(f"        warp_init_sem_idx = Int32({num_stages * num_ops * 3})")

        # Initialize semaphores
        code.append("        if tidx == 0:")
        code.append("            warp_semaphores[warp_init_sem_idx] = Int32(0)")
        code.append(f"            for i in range({num_stages * num_ops * 3}):")
        code.append("                if i % 3 == 2:")
        code.append("                    warp_semaphores[i] = Int32(1)  # FINISHED = 1 (page free)")
        code.append("                else:")
        code.append("                    warp_semaphores[i] = Int32(0)")
        code.append("            cute.arch.fence_view_async_shared()")
        code.append("            atomic_add_i32(1, warp_semaphores.iterator + warp_init_sem_idx)")

        # Wait for initialization
        code.append("        if tidx != 0:")
        code.append("            while atomic_add_i32(0, warp_semaphores.iterator + warp_init_sem_idx) < Int32(1):")
        code.append("                nanosleep(Int32(20))")
        code.append("        cute.arch.fence_view_async_shared()")

        # Persistent loop variables
        code.append("        local_block_idx = Int32(0)")
        code.append("        persistent_block_idx = bidx")
        code.append("        n_blocks_val = n_blocks")
        code.append("        g_dim, _, _ = cute.arch.grid_dim()")

        return code

    def generate_persistent_loop_start(self, ctx: CodeGenContext) -> List[str]:
        """Generate the start of the persistent grid loop."""
        num_stages = max(1, ctx.num_stages)
        code = []
        code.append("")
        code.append("        while persistent_block_idx < n_blocks_val:")
        code.append(f"            page_idx = local_block_idx % Int32({num_stages})")
        code.append(f"            iter_target = local_block_idx // Int32({num_stages}) + Int32(1)")
        return code

    def generate_op_code(
        self,
        ctx: CodeGenContext,
        op_idx: int,
        indent: str = "            ",
    ) -> List[str]:
        """Generate warp-specialized L/C/S code for one operation.

        This generates the role dispatch for loader/consumer/storer warps.
        """
        ops = []
        inst = ctx.instructions[op_idx]
        args_str = ", ".join([f"arg_{idx}" for idx in ctx.mapping[op_idx]])
        get_smem_arg = ctx.make_smem_arg_getter(op_idx)
        smem_str = get_smem_arg("page_idx")
        num_ops = len(ctx.instructions)

        # Semaphore indices
        idx_fin = f"(page_idx * Int32({num_ops}) + Int32({op_idx})) * Int32(3) + Int32(2)"
        idx_load = f"(page_idx * Int32({num_ops}) + Int32({op_idx})) * Int32(3) + Int32(0)"
        idx_comp = f"(page_idx * Int32({num_ops}) + Int32({op_idx})) * Int32(3) + Int32(1)"

        logical_size = inst.get("logical_grid_size", -1)
        bounds_prefix = f"if persistent_block_idx < Int32({logical_size}): " if logical_size != -1 else ""

        # LOADER role
        ops.append(f"{indent}# Op {op_idx} (Warp-Specialized mode)")
        ops.append(f"{indent}if warp_id == Int32({self.loader_warp}):")
        ops.append(f"{indent}    while atomic_add_i32(0, warp_semaphores.iterator + {idx_fin}) < iter_target:")
        ops.append(f"{indent}        nanosleep(Int32(20))")

        # Reduction wait if needed
        reduction_wait = inst.get("reduction_wait")
        if reduction_wait and ctx.reduction_barrier_config:
            wait_count = reduction_wait["wait_count"]
            producer_op_idx = reduction_wait["producer_op_idx"]
            barrier_offset = ctx.reduction_barrier_config._barrier_offsets.get(op_idx, 0)
            red_bar_idx = f"Int32({barrier_offset}) + persistent_block_idx"
            ops.append(f"{indent}    # Reduction wait: {wait_count} blocks from op {producer_op_idx}")
            ops.append(
                f"{indent}    while atomic_add_i32(0, reduction_barrier.iterator + {red_bar_idx}) "
                f"< Int32({wait_count}):"
            )
            ops.append(f"{indent}        nanosleep(Int32(20))")

        ops.append(
            f"{indent}    {bounds_prefix}op_{op_idx}_load("
            f"paged_pool, page_idx, persistent_block_idx, {smem_str}{args_str})"
        )
        ops.append(f"{indent}    cute.arch.sync_warp()")
        ops.append(f"{indent}    cute.arch.fence_view_async_shared()")
        ops.append(f"{indent}    if lane_id == 0:")
        ops.append(f"{indent}        atomic_add_i32(1, warp_semaphores.iterator + {idx_load})")

        # CONSUMER role
        ops.append(f"{indent}elif warp_id < Int32({self.num_consumer}):")
        ops.append(f"{indent}    while atomic_add_i32(0, warp_semaphores.iterator + {idx_load}) < iter_target:")
        ops.append(f"{indent}        nanosleep(Int32(20))")
        ops.append(
            f"{indent}    {bounds_prefix}op_{op_idx}_compute(persistent_block_idx, {smem_str}{args_str})"
        )
        ops.append(f"{indent}    cute.arch.sync_warp()")
        ops.append(f"{indent}    cute.arch.fence_view_async_shared()")
        ops.append(f"{indent}    if lane_id == 0:")
        ops.append(f"{indent}        atomic_add_i32(1, warp_semaphores.iterator + {idx_comp})")

        # STORER role
        wait_target = f"iter_target * Int32({self.num_consumer})"
        ops.append(f"{indent}elif warp_id == Int32({self.storer_warp}):")
        ops.append(f"{indent}    while atomic_add_i32(0, warp_semaphores.iterator + {idx_comp}) < {wait_target}:")
        ops.append(f"{indent}        nanosleep(Int32(20))")
        ops.append(
            f"{indent}    {bounds_prefix}op_{op_idx}_store("
            f"paged_pool, page_idx, persistent_block_idx, {smem_str}{args_str})"
        )
        ops.append(f"{indent}    cute.arch.sync_warp()")
        ops.append(f"{indent}    cute.arch.fence_view_async_shared()")
        ops.append(f"{indent}    if lane_id == 0:")
        ops.append(f"{indent}        atomic_add_i32(1, warp_semaphores.iterator + {idx_fin})")

        # Reduction signal if needed
        reduction_signals = inst.get("reduction_signals", [])
        for red_sig in reduction_signals:
            dim_mapping = red_sig["dim_mapping"]
            consumer_op_idx = red_sig["consumer_op_idx"]
            blocks_per_consumer = dim_mapping.blocks_per_consumer
            if ctx.reduction_barrier_config:
                barrier_offset = ctx.reduction_barrier_config._barrier_offsets.get(consumer_op_idx, 0)
                red_bar_idx = f"Int32({barrier_offset}) + reduction_consumer_idx"
                ops.append(f"{indent}        # Signal reduction to consumer op {consumer_op_idx}")
                ops.append(
                    f"{indent}        reduction_consumer_idx = "
                    f"persistent_block_idx // Int32({blocks_per_consumer})"
                )
                ops.append(
                    f"{indent}        atomic_add_i32(1, reduction_barrier.iterator + {red_bar_idx})"
                )

        return ops

    def generate_persistent_loop_end(self) -> List[str]:
        """Generate the end of the persistent grid loop."""
        code = []
        code.append("")
        code.append("            # Persistence update")
        code.append("            local_block_idx += Int32(1)")
        code.append("            persistent_block_idx += g_dim")
        code.append("        # End persistent loop")
        return code


class MixedModeScheduler:
    """Unified scheduler that handles mixed scheduling modes per-operation.

    This scheduler analyzes each operation's scheduling mode and generates
    appropriate code, handling transitions between modes with proper
    synchronization barriers.

    Supported patterns:
    - Sequential -> Sequential: No barrier needed
    - Sequential -> WarpSpecialized: Barrier to establish warp roles
    - WarpSpecialized -> Sequential: Barrier to reunify warps
    - WarpSpecialized -> WarpSpecialized: No barrier (persistent loop handles it)
    """

    def __init__(self, config: Optional[NoBubblesConfig] = None):
        self.config = config or NoBubblesConfig()
        self.generators: Dict[SchedulingMode, OpCodeGenerator] = {
            SchedulingMode.SEQUENTIAL: SequentialCodeGenerator(),
            SchedulingMode.ASYNC: SequentialCodeGenerator(),  # TODO: implement async
            SchedulingMode.WARP_SPECIALIZED: WarpSpecializedCodeGenerator(self.config.warp_config),
        }

    def _group_ops_by_mode(self, op_descriptors: List[OpDescriptor]) -> List[Tuple[SchedulingMode, List[int]]]:
        """Group consecutive operations by their scheduling mode.

        Returns list of (mode, [op_indices]) tuples representing contiguous
        groups that can be generated together.
        """
        if not op_descriptors:
            return []

        groups = []
        current_mode = op_descriptors[0].scheduling_mode
        current_indices = [0]

        for i in range(1, len(op_descriptors)):
            op_mode = op_descriptors[i].scheduling_mode
            if op_mode == current_mode:
                current_indices.append(i)
            else:
                groups.append((current_mode, current_indices))
                current_mode = op_mode
                current_indices = [i]

        groups.append((current_mode, current_indices))
        return groups

    def generate_schedule(
        self,
        ctx: CodeGenContext,
        op_descriptors: List[OpDescriptor],
    ) -> List[str]:
        """Generate complete schedule code handling mixed modes.

        Args:
            ctx: Code generation context
            op_descriptors: List of operation descriptors with scheduling modes

        Returns:
            List of code lines for the complete schedule
        """
        code = []
        groups = self._group_ops_by_mode(op_descriptors)

        if not groups:
            return code

        # Generate initialization for modes that need it
        init_modes = set()
        for mode, _ in groups:
            if mode not in init_modes:
                gen = self.generators.get(mode)
                if gen:
                    code.extend(gen.generate_init_code(ctx, ctx.instructions))
                init_modes.add(mode)

        # Generate code for each group
        prev_mode = None
        warp_spec_gen = self.generators.get(SchedulingMode.WARP_SPECIALIZED)

        for group_idx, (mode, op_indices) in enumerate(groups):
            gen = self.generators.get(mode)
            if not gen:
                continue

            # Add mode transition barrier if needed
            if prev_mode is not None and prev_mode != mode:
                code.extend(self._generate_mode_transition(prev_mode, mode))

            # For warp-specialized mode, use persistent loop
            if mode == SchedulingMode.WARP_SPECIALIZED and isinstance(warp_spec_gen, WarpSpecializedCodeGenerator):
                code.extend(warp_spec_gen.generate_persistent_loop_start(ctx))
                for op_idx in op_indices:
                    code.extend(gen.generate_op_code(ctx, op_idx, indent="            "))
                # Check if next group is also warp-specialized
                next_is_warp_spec = (
                    group_idx + 1 < len(groups) and
                    groups[group_idx + 1][0] == SchedulingMode.WARP_SPECIALIZED
                )
                if not next_is_warp_spec:
                    code.extend(warp_spec_gen.generate_persistent_loop_end())
            else:
                # Sequential/async modes
                for op_idx in op_indices:
                    code.extend(gen.generate_op_code(ctx, op_idx))

            prev_mode = mode

        return code

    def _generate_mode_transition(self, from_mode: SchedulingMode, to_mode: SchedulingMode) -> List[str]:
        """Generate synchronization barrier for mode transition."""
        code = []
        code.append("")
        code.append(f"        # Mode transition: {from_mode.name} -> {to_mode.name}")

        if from_mode == SchedulingMode.WARP_SPECIALIZED:
            # All warps must synchronize before transitioning
            code.append("        # Synchronize all warps before mode change")
            code.append("        cute.arch.fence_view_async_shared()")
            # Use a simple barrier pattern
            code.append("        if tidx == 0:")
            code.append("            atomic_add_i32(1, barrier_tensor.iterator)")
            code.append("        target = (sync_step + Int32(1)) * g_dim")
            code.append("        while atomic_add_i32(0, barrier_tensor.iterator) < target:")
            code.append("            nanosleep(Int32(20))")
            code.append("        sync_step = sync_step + Int32(1)")

        elif to_mode == SchedulingMode.WARP_SPECIALIZED:
            # Entering warp-specialized mode - no special barrier needed
            # The warp_id dispatch will handle role assignment
            code.append("        # Entering warp-specialized mode")
            code.append("        cute.arch.fence_view_async_shared()")

        return code


# ============================================================================
# Sequential Scheduler (Simple L/C/S execution)
# ============================================================================


class SequentialScheduler:
    """Simple sequential scheduler for L/C/S execution.

    This scheduler generates a straightforward sequence of Load -> Compute -> Store
    for each operation, with fine-grained semaphore synchronization between phases.

    Unlike the warp-specialized schedulers, this runs ALL threads through all phases
    sequentially, making it suitable for:
    - Simple kernels without warp role specialization
    - Debugging and baseline comparison
    - Kernels where all threads participate in L/C/S equally

    Synchronization Protocol:
    - sem_load_done: Thread 0 signals after load, others wait before compute
    - sem_compute_done: Thread 0 signals after compute, others wait before store

    This ensures all threads see consistent data between phases without full
    block barriers (__syncthreads).
    """

    def __init__(self):
        self.instructions: list = []

    def generate_schedule_ops(
        self,
        instructions: list,
        mapping: dict,
        paged_pool_bytes: int,
        traced: bool,
        make_smem_arg_getter,
        emit_trace_start,
        emit_trace_end,
        tracing_phases: dict,
        sync_trace_types: dict,
    ) -> list:
        """Generate simple scheduling code with fine-grained semaphore synchronization.

        Args:
            instructions: List of instruction dicts from Megakernel
            mapping: Argument mapping per operation
            paged_pool_bytes: Size of paged pool (affects page_idx initialization)
            traced: Whether tracing is enabled
            make_smem_arg_getter: Function to create smem argument getter
            emit_trace_start: Function to emit trace start
            emit_trace_end: Function to emit trace end
            tracing_phases: Dict of tracing phase types (LOAD, COMPUTE, STORE)
            sync_trace_types: Dict of sync trace types (SEM_SIGNAL_*, SEM_WAIT_*)

        Returns:
            List of code lines for the schedule
        """
        ops = []

        # Initialize page index
        if paged_pool_bytes > 0:
            ops.append("        page_idx = Int32(0)")
            ops.append("        next_page = Int32(0)")
        else:
            ops.append("        page_idx = Int32(0)")

        # Check if we have heterogeneous logical grid sizes
        logical_sizes = [inst.get("logical_grid_size", 1) for inst in instructions]
        has_heterogeneous_grids = len(set(logical_sizes)) > 1

        if has_heterogeneous_grids:
            ops.append("")
            ops.append("        # Heterogeneous logical grids - ops check bounds via logical_idx")

        # Allocate semaphores for fine-grained synchronization
        ops.append("")
        ops.append("        # Fine-grained synchronization semaphores")
        ops.append("        semaphores = smem_alloc.allocate_tensor(cute.Int32, cute.make_layout(2))")
        ops.append("")
        ops.append("        # Initialize semaphores (single thread)")
        ops.append("        if tidx == 0:")
        ops.append("            semaphores[0] = Int32(0)  # sem_load_done")
        ops.append("            semaphores[1] = Int32(0)  # sem_compute_done")
        ops.append("        cute.arch.fence_view_async_shared()")
        ops.append("")

        # Generate L/C/S for each operation with semaphore-based sync
        for i, inst in enumerate(instructions):
            args_str = ", ".join([f"arg_{idx}" for idx in mapping[i]])
            get_smem_arg = make_smem_arg_getter(i)
            logical_size = inst.get("logical_grid_size", 1)

            # For heterogeneous grids, wrap ops in bounds check
            needs_bounds_check = has_heterogeneous_grids and logical_size < max(logical_sizes)
            indent = "            " if needs_bounds_check else "        "

            if needs_bounds_check:
                ops.append(f"        # Op {i}: {logical_size} logical blocks")
                ops.append(f"        if logical_idx < Int32({logical_size}):")

            # Reset semaphores for this operation (only for ops after the first)
            if i > 0:
                ops.append(f"{indent}# Reset semaphores for Op {i}")
                ops.append(f"{indent}if tidx == 0:")
                ops.append(f"{indent}    semaphores[0] = Int32(0)")
                ops.append(f"{indent}    semaphores[1] = Int32(0)")
                ops.append(f"{indent}cute.arch.fence_view_async_shared()")
                ops.append("")

            # LOAD phase
            if traced:
                emit_trace_start(ops, indent)

            smem_str = get_smem_arg("page_idx")
            ops.append(f"{indent}op_{i}_load(paged_pool, page_idx, logical_idx, {smem_str}{args_str})")

            if traced:
                emit_trace_end(ops, indent, tracing_phases, MicroOpType.LOAD, i)

            # Signal load complete
            ops.append(f"{indent}cute.arch.fence_view_async_shared()")
            ops.append(f"{indent}if tidx == 0:")
            if traced:
                emit_trace_start(ops, indent + "    ", "t_sig")
            ops.append(f"{indent}    atomic_add_i32(1, semaphores.iterator)")
            if traced:
                emit_trace_end(ops, indent + "    ", sync_trace_types, "SEM_SIGNAL_LOAD", i, "t_sig")
            ops.append("")

            # Wait for load to complete before compute (non-thread-0 threads wait)
            ops.append(f"{indent}# Wait for load to complete")
            ops.append(f"{indent}if tidx != 0:")
            if traced:
                emit_trace_start(ops, indent + "    ", "t_wait")
            ops.append(f"{indent}    while atomic_add_i32(0, semaphores.iterator) < Int32(1):")
            ops.append(f"{indent}        nanosleep(Int32(20))")
            if traced:
                emit_trace_end(ops, indent + "    ", sync_trace_types, "SEM_WAIT_LOAD", i, "t_wait")
            ops.append("")

            # COMPUTE phase
            if traced:
                emit_trace_start(ops, indent)

            ops.append(f"{indent}op_{i}_compute(logical_idx, {smem_str}{args_str})")

            if traced:
                emit_trace_end(ops, indent, tracing_phases, MicroOpType.COMPUTE, i)

            # Signal compute complete
            ops.append(f"{indent}cute.arch.fence_view_async_shared()")
            ops.append(f"{indent}if tidx == 0:")
            if traced:
                emit_trace_start(ops, indent + "    ", "t_sig")
            ops.append(f"{indent}    atomic_add_i32(1, semaphores.iterator + Int32(1))")
            if traced:
                emit_trace_end(ops, indent + "    ", sync_trace_types, "SEM_SIGNAL_COMPUTE", i, "t_sig")
            ops.append("")

            # Wait for compute to complete before store (non-thread-0 threads wait)
            ops.append(f"{indent}# Wait for compute to complete")
            ops.append(f"{indent}if tidx != 0:")
            if traced:
                emit_trace_start(ops, indent + "    ", "t_wait")
            ops.append(f"{indent}    while atomic_add_i32(0, semaphores.iterator + Int32(1)) < Int32(1):")
            ops.append(f"{indent}        nanosleep(Int32(20))")
            if traced:
                emit_trace_end(ops, indent + "    ", sync_trace_types, "SEM_WAIT_COMPUTE", i, "t_wait")
            ops.append("")

            # STORE phase
            if traced:
                emit_trace_start(ops, indent)

            ops.append(f"{indent}op_{i}_store(paged_pool, page_idx, logical_idx, {smem_str}{args_str})")

            if traced:
                emit_trace_end(ops, indent, tracing_phases, MicroOpType.STORE, i)

            ops.append("")

        return ops


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
        """Three semaphores per page: arrived + compute_done + finished."""
        return self.num_pages * 3

    @property
    def total_size_bytes(self) -> int:
        """Total shared memory needed for semaphores."""
        return self.total_semaphores * self.semaphore_size_bytes

    def get_arrived_semaphore_idx(self, page_id: int) -> int:
        """Get index of the 'data arrived' (load done) semaphore for a page."""
        return page_id * 3

    def get_compute_done_semaphore_idx(self, page_id: int) -> int:
        """Get index of the 'compute done' semaphore for a page."""
        return page_id * 3 + 1

    def get_finished_semaphore_idx(self, page_id: int) -> int:
        """Get index of the 'page finished' (store done) semaphore for a page."""
        return page_id * 3 + 2


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


class NoBubblesScheduler:
    """Unified scheduler for the No Bubbles pattern with dependency-aware scheduling.

    This scheduler combines:
    1. Dependency-aware scheduling for optimal L/C/S overlap
    2. Paged shared memory management with acquire/release semantics
    3. Page semaphores for fine-grained synchronization
    4. Warp specialization support (LOADER, CONSUMER, STORER roles)
    5. Logical Block support for flexible coordinate mapping
    6. Cross-kernel dependencies with dimension mapping (reduction/broadcast)

    The key insight is that Load[i+1] can often start before Compute[i] or Store[i]
    completes, as long as there's no data dependency. This scheduler builds a
    dependency graph and schedules loads as early as possible.

    Cross-Kernel Dependency Patterns:
    - 1:1 (LOGICAL_BLOCK): Consumer block i waits for producer block i
    - Many-to-one (REDUCTION): Consumer (b,h) waits for ALL producer (b,h,0..seq-1)
    - One-to-many (BROADCAST): Consumer (b,h,s) waits for producer (b,h)

    Semaphore Protocol (per page):
    - sem_arrived[page]: Loader signals after load, Consumer waits before compute
    - sem_finished[page]: Consumer signals after compute, Loader waits before reusing

    This allows:
    - Load[N+1] to start as soon as Consumer[N] finishes (not waiting for Store[N])
    - Maximum overlap between Load, Compute, and Store phases
    - Fine-grained cross-kernel sync (unlike coarse PDL/cudaGridDependencySynchronize)

    Based on: "Look Ma, No Bubbles!" - HazyResearch
    """

    def __init__(self, config: Optional[NoBubblesConfig] = None):
        self.config = config or NoBubblesConfig()
        self.allocator = PagedMemoryAllocator(self.config.num_pages, self.config.page_size_bytes)
        self.micro_ops: List[PageAwareMicroOp] = []
        self._next_id = 0
        self.barrier_config: Optional[BarrierConfig] = None
        self.semaphore_config: Optional[PageSemaphoreConfig] = None

        # Cross-kernel dependency tracking
        self.operations: List[OpDescriptor] = []
        self.inter_op_dependencies: List[InterOpDependency] = []
        self._dep_graph: Dict[int, List[InterOpDependency]] = {}

        # Initialize semaphore config
        if self.config.num_pages > 0:
            self.semaphore_config = PageSemaphoreConfig(num_pages=self.config.num_pages)

    # =========================================================================
    # Dependency Analysis Methods
    # =========================================================================

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

    # =========================================================================
    # Cross-Kernel Dependency Management
    # =========================================================================

    def add_operation(self, op: OpDescriptor):
        """Add an operation to the scheduler."""
        self.operations.append(op)

    def add_dependency(self, dep: InterOpDependency):
        """Add an inter-operation dependency.

        Example (1:1):
            scheduler.add_dependency(InterOpDependency(
                producer_op="qkv_proj",
                consumer_op="attention",
            ))

        Example (reduction - many-to-one):
            scheduler.add_dependency(InterOpDependency(
                producer_op="attention",
                consumer_op="output_proj",
                dim_mapping=DimensionMapping(
                    producer_dims=(2, 8, 512),  # (batch, head, seq)
                    consumer_dims=(2, 8),        # (batch, head)
                ),
            ))

        Example (broadcast - one-to-many):
            scheduler.add_dependency(InterOpDependency(
                producer_op="weights",
                consumer_op="attention",
                dim_mapping=DimensionMapping(
                    producer_dims=(2, 8),
                    consumer_dims=(2, 8, 512),
                ),
            ))
        """
        self.inter_op_dependencies.append(dep)

    def build_cross_kernel_dependency_graph(self) -> Dict[int, List[InterOpDependency]]:
        """Build the cross-kernel dependency graph from declared dependencies.

        Returns:
            Dict mapping consumer op_idx to list of InterOpDependency objects
        """
        # Build name -> idx mapping
        name_to_idx = {op.name: op.op_idx for op in self.operations}

        self._dep_graph = {op.op_idx: [] for op in self.operations}

        for dep in self.inter_op_dependencies:
            if dep.producer_op in name_to_idx and dep.consumer_op in name_to_idx:
                consumer_idx = name_to_idx[dep.consumer_op]
                self._dep_graph[consumer_idx].append(dep)

        return self._dep_graph

    def get_cross_kernel_deps_for_op(self, op_idx: int) -> List[InterOpDependency]:
        """Get cross-kernel dependencies for an operation."""
        if not self._dep_graph:
            self.build_cross_kernel_dependency_graph()
        return self._dep_graph.get(op_idx, [])

    def get_producer_blocks_for_consumer(
        self,
        consumer_op_idx: int,
        consumer_logical: int,
    ) -> Dict[int, List[int]]:
        """Get all producer blocks that a consumer logical block depends on.

        This handles all dependency patterns:
        - 1:1 (LOGICAL_BLOCK): Returns single producer block
        - Many-to-one (REDUCTION): Returns all producer blocks to wait for
        - One-to-many (BROADCAST): Returns the single producer block

        Args:
            consumer_op_idx: Consumer operation index
            consumer_logical: Consumer logical block index

        Returns:
            Dict mapping producer_op_idx to list of producer logical block indices
        """
        deps = self.get_cross_kernel_deps_for_op(consumer_op_idx)
        name_to_idx = {op.name: op.op_idx for op in self.operations}

        result: Dict[int, List[int]] = {}

        for dep in deps:
            if dep.producer_op in name_to_idx:
                producer_idx = name_to_idx[dep.producer_op]
                producer_blocks = dep.get_producer_logical_blocks(consumer_logical)
                result[producer_idx] = producer_blocks

        return result

    def get_wait_count_for_consumer(
        self,
        consumer_op_idx: int,
        consumer_logical: int,
    ) -> Dict[int, int]:
        """Get the number of producer completions to wait for per dependency.

        Args:
            consumer_op_idx: Consumer operation index
            consumer_logical: Consumer logical block index

        Returns:
            Dict mapping producer_op_idx to wait count
        """
        deps = self.get_cross_kernel_deps_for_op(consumer_op_idx)
        name_to_idx = {op.name: op.op_idx for op in self.operations}

        result: Dict[int, int] = {}

        for dep in deps:
            if dep.producer_op in name_to_idx:
                producer_idx = name_to_idx[dep.producer_op]
                wait_count = dep.get_wait_count(consumer_logical)
                if wait_count == -1:
                    # KERNEL granularity: wait for all producer blocks
                    producer_op = next(
                        (op for op in self.operations if op.op_idx == producer_idx), None
                    )
                    if producer_op and producer_op.logical_grid_size:
                        wait_count = producer_op.logical_grid_size
                    else:
                        wait_count = 1
                result[producer_idx] = wait_count

        return result

    def can_prefetch_op(self, op_idx: int) -> bool:
        """Check if an operation's load can be prefetched during previous op's compute.

        An operation can prefetch if:
        1. It has no KERNEL granularity dependencies
        2. For REDUCTION deps, prefetch can start but must wait for all blocks
        """
        deps = self.get_cross_kernel_deps_for_op(op_idx)
        if not deps:
            return True  # No dependencies, can always prefetch

        # Can prefetch if no coarse KERNEL-level deps
        return all(dep.granularity != DependencyGranularity.KERNEL for dep in deps)

    # =========================================================================
    # Schedule Analysis and Visualization
    # =========================================================================

    def topological_sort(self) -> List[PageAwareMicroOp]:
        """Return micro-ops in a valid execution order respecting dependencies.

        Uses Kahn's algorithm for topological sorting.
        """
        from collections import deque

        # Build adjacency list and in-degree count
        in_degree: Dict[int, int] = {op.id: len(op.depends_on) for op in self.micro_ops}
        dependents: Dict[int, List[int]] = {op.id: [] for op in self.micro_ops}
        id_to_op: Dict[int, PageAwareMicroOp] = {op.id: op for op in self.micro_ops}

        for op in self.micro_ops:
            for dep_id in op.depends_on:
                if dep_id in dependents:
                    dependents[dep_id].append(op.id)

        # Start with ops that have no dependencies
        queue = deque([op.id for op in self.micro_ops if in_degree[op.id] == 0])
        result: List[PageAwareMicroOp] = []

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

    def get_parallelizable_groups(self) -> List[List[PageAwareMicroOp]]:
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
        waves: List[List[PageAwareMicroOp]] = [[] for _ in range(max_wave + 1)]
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

    # =========================================================================
    # MicroOp Creation
    # =========================================================================

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
            sem_compute_done = [self.semaphore_config.get_compute_done_semaphore_idx(p) for p in pages]
            sem_finished = [self.semaphore_config.get_finished_semaphore_idx(p) for p in pages]

            # --- LOADER: wait(sem_finished) -> load -> signal(sem_arrived) ---
            # For first iteration of each page, sem_finished is pre-initialized to 1
            # For subsequent uses, we wait for the previous storer to finish
            load_deps: Set[int] = set()

            # Data dependencies: wait for stores of ops we read from
            for dep_idx in data_deps.get(i, set()):
                if dep_idx in store_ops:
                    load_deps.add(store_ops[dep_idx].id)

            # Page dependency: if pages were used before, wait for that storer to signal sem_finished
            prev_user = i - max_concurrent
            if prev_user >= 0 and prev_user in store_ops:
                load_deps.add(store_ops[prev_user].id)

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

            # --- CONSUMER: wait(sem_arrived) -> compute -> signal(sem_compute_done) ---
            compute_op = self._new_micro_op(
                MicroOpType.COMPUTE,
                i,
                f"Compute[{i}]",
                depends_on={load_op.id},  # Structural dependency
                sem_waits=sem_arrived,  # Wait for data to be ready
                sem_signals=sem_compute_done,  # Signal compute is done
                warp_role=WarpRole.CONSUMER,
            )
            compute_ops[i] = compute_op

            # --- STORER: wait(sem_compute_done) -> store output -> signal(sem_finished) ---
            store_op = self._new_micro_op(
                MicroOpType.STORE_ASYNC,
                i,
                f"Store[{i}]",
                depends_on={compute_op.id},
                sem_waits=sem_compute_done,  # Wait for compute to be done
                sem_signals=sem_finished,  # Signal page can be reused
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
