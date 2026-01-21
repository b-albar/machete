# Copyright (c) 2025, Machete Authors
"""
Operation Graph Scheduler for Megakernels.

This module implements a simplified scheduling infrastructure:
1. Operation graph with data dependencies
2. Static shared memory planning (no dynamic paging)
3. Schedule optimization (load movement, overlap)
4. Unified mixed kernel type support (LCS and Producer/Consumer)

Based on: "Look Ma, No Bubbles!" - HazyResearch
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Callable, TypeVar, Tuple
from enum import Enum, auto
from collections import deque

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
# Dependency Declaration Decorators
# ============================================================================


def reads(*tensors: str) -> Callable[[F], F]:
    """Decorator to declare which tensors/memory regions an L/C/S method reads from.

    Usage:
        @reads("input", "weight")
        @cute.jit
        def load_forward(self, logical_idx, input, weight, output):
            ...
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
        def store_forward(self, logical_idx, input, weight, output):
            ...
    """

    def decorator(func: F) -> F:
        existing = getattr(func, "_machete_writes", set())
        func._machete_writes = existing | set(tensors)
        return func

    return decorator


# ============================================================================
# Helper Functions for Extracting Decorator Info
# ============================================================================


def get_method_dependencies(method: Callable) -> Tuple[Set[str], Set[str]]:
    """Extract dependency info from a decorated L/C/S method.

    Returns:
        (reads, writes)
    """
    reads_set = getattr(method, "_machete_reads", set())
    writes_set = getattr(method, "_machete_writes", set())
    return reads_set, writes_set


# ============================================================================
# Configuration Classes
# ============================================================================


@dataclass
class BarrierConfig:
    """Configuration for the global barrier tensor.

    The barrier tensor is addressed as: Bar[OpIdx][LogicalID]
    - Producer: atomicAdd(&Bar[op_idx, logical_id], 1) after store
    - Consumer: while(Bar[dep_op_idx, logical_id] < target) nanosleep()
    """

    num_ops: int
    total_logical_blocks: int

    @property
    def tensor_shape(self) -> Tuple[int, int]:
        """Shape of the barrier tensor: (num_ops, total_logical_blocks)."""
        return (self.num_ops, self.total_logical_blocks)

    @property
    def total_counters(self) -> int:
        """Total number of barrier counters needed."""
        return self.num_ops * self.total_logical_blocks


# ============================================================================
# Kernel Type for Mixed Mode Support
# ============================================================================


class KernelType(Enum):
    """Per-operation kernel type.

    Each operation declares its type. The scheduler builds a unified graph
    where all kernel types coexist.
    """

    LCS = auto()  # Load/Compute/Store phases (traditional)
    PRODUCER_CONSUMER = auto()  # Producer/Consumer warp pattern


# ============================================================================
# Operation Graph Data Structures
# ============================================================================


@dataclass
class OpNode:
    """A node in the operation graph representing a single operation."""

    op_idx: int
    name: str

    # Shared memory requirements (bytes) - static, computed from kernel
    smem_bytes_fwd: int = 0
    smem_bytes_bwd: int = 0

    # Active smem bytes based on mode (set by graph builder)
    smem_bytes: int = 0

    # Kernel type (LCS or Producer/Consumer)
    kernel_type: KernelType = KernelType.LCS

    # Logical grid info
    logical_grid_size: int = 1
    coord_dims: Tuple[int, ...] = ()

    # Dependency information (tensor names)
    reads: Set[str] = field(default_factory=set)
    writes: Set[str] = field(default_factory=set)

    # Graph edges (indices of dependent operations)
    depends_on: Set[int] = field(default_factory=set)
    dependents: Set[int] = field(default_factory=set)

    # For warp specialization
    uses_warp_specialization: bool = False
    warp_config: Optional[WarpConfig] = None


@dataclass
class OpEdge:
    """An edge in the operation graph representing a dependency."""

    producer_idx: int
    consumer_idx: int

    # What data flows along this edge
    tensors: Set[str] = field(default_factory=set)

    # Dependency pattern
    is_reduction: bool = False  # Many-to-one
    is_broadcast: bool = False  # One-to-many

    # For reduction/broadcast, dimension mapping
    producer_dims: Tuple[int, ...] = ()
    consumer_dims: Tuple[int, ...] = ()

    @property
    def blocks_per_consumer(self) -> int:
        """For reduction: how many producer blocks per consumer."""
        if not self.is_reduction:
            return 1
        p_total = 1
        for d in self.producer_dims:
            p_total *= d
        c_total = 1
        for d in self.consumer_dims:
            c_total *= d
        return p_total // c_total if c_total > 0 else 1


class OperationGraph:
    """Graph of operations for scheduling and optimization.

    This is the central data structure for the scheduler.
    It builds a DAG of operations and provides analysis methods.
    """

    def __init__(self):
        self.nodes: Dict[int, OpNode] = {}
        self.edges: List[OpEdge] = []
        self._total_smem: Optional[int] = None
        self._topo_order: Optional[List[int]] = None

    def add_node(self, node: OpNode):
        """Add an operation node to the graph."""
        self.nodes[node.op_idx] = node
        self._total_smem = None
        self._topo_order = None

    def add_edge(self, edge: OpEdge):
        """Add a dependency edge to the graph."""
        self.edges.append(edge)
        if edge.producer_idx in self.nodes:
            self.nodes[edge.producer_idx].dependents.add(edge.consumer_idx)
        if edge.consumer_idx in self.nodes:
            self.nodes[edge.consumer_idx].depends_on.add(edge.producer_idx)
        self._topo_order = None

    @classmethod
    def from_instructions(
        cls, instructions: List[Dict], mode: str = "forward"
    ) -> "OperationGraph":
        """Build operation graph from a list of instruction dicts.

        Args:
            instructions: List of instruction dicts with kernel, load, compute, store
            mode: "forward" or "backward"

        Returns:
            OperationGraph with nodes and inferred edges
        """
        graph = cls()

        # First pass: create nodes
        for i, inst in enumerate(instructions):
            node = cls._node_from_instruction(inst, i, mode)
            graph.add_node(node)

        # Second pass: infer edges from reads/writes
        graph._infer_edges()

        return graph

    @staticmethod
    def _node_from_instruction(inst: Dict, op_idx: int, mode: str) -> OpNode:
        """Create an OpNode from an instruction dict."""
        kernel = inst.get("kernel")

        # Get L/C/S methods
        load_fn = inst.get("load") or getattr(kernel, f"load_{mode}", None)
        compute_fn = inst.get("compute") or getattr(kernel, f"compute_{mode}", None)
        store_fn = inst.get("store") or getattr(kernel, f"store_{mode}", None)

        # Extract reads/writes from decorators
        all_reads: Set[str] = set()
        all_writes: Set[str] = set()

        for fn in [load_fn, compute_fn, store_fn]:
            if fn:
                r, w = get_method_dependencies(fn)
                all_reads |= r
                all_writes |= w

        # Determine kernel type
        uses_warp_spec = getattr(kernel, "uses_warp_specialization", False)
        if uses_warp_spec:
            kernel_type = KernelType.PRODUCER_CONSUMER
        else:
            kernel_type = KernelType.LCS

        # Get smem sizes for forward and backward passes
        # Priority: instruction dict > kernel property > legacy smem_size
        smem_bytes_fwd = (
            inst.get("smem_size_fwd", 0)
            or getattr(kernel, "smem_size_fwd", 0)
            or inst.get("smem_size", 0)
            or getattr(kernel, "smem_size", 0)
        )
        smem_bytes_bwd = (
            inst.get("smem_size_bwd", 0)
            or getattr(kernel, "smem_size_bwd", 0)
            or inst.get("smem_size", 0)
            or getattr(kernel, "smem_size", 0)
        )

        # Select active smem based on mode
        smem_bytes = smem_bytes_fwd if mode == "forward" else smem_bytes_bwd

        # Get logical grid size if available
        logical_grid_size = 1
        if hasattr(kernel, "get_logical_grid_size"):
            args = inst.get("args", [])
            try:
                logical_grid_size = kernel.get_logical_grid_size(*args)
            except Exception:
                logical_grid_size = 1

        return OpNode(
            op_idx=op_idx,
            name=type(kernel).__name__ if kernel else f"op_{op_idx}",
            smem_bytes_fwd=smem_bytes_fwd,
            smem_bytes_bwd=smem_bytes_bwd,
            smem_bytes=smem_bytes,
            kernel_type=kernel_type,
            logical_grid_size=logical_grid_size,
            reads=all_reads,
            writes=all_writes,
            uses_warp_specialization=uses_warp_spec,
            warp_config=getattr(kernel, "warp_config", None) if uses_warp_spec else None,
        )

    def _infer_edges(self):
        """Infer dependency edges from reads/writes relationships."""
        for i, node_i in self.nodes.items():
            for j, node_j in self.nodes.items():
                if i >= j:
                    continue

                # Check if j depends on i
                # RAW: j reads what i wrote
                raw = node_j.reads & node_i.writes
                # WAW: j writes what i wrote
                waw = node_j.writes & node_i.writes
                # WAR: j writes what i read
                war = node_j.writes & node_i.reads

                if raw or waw or war:
                    edge = OpEdge(
                        producer_idx=i,
                        consumer_idx=j,
                        tensors=raw | waw | war,
                    )
                    self.add_edge(edge)

    # ========== Analysis Methods ==========

    def compute_total_smem(self) -> int:
        """Compute total shared memory needed from graph analysis.

        For sequential execution: max(op.smem_bytes)
        For overlapped execution: may need sum of overlapping ops
        """
        if self._total_smem is not None:
            return self._total_smem

        # Simple case: max of all ops (sequential execution)
        max_smem = max((n.smem_bytes for n in self.nodes.values()), default=0)
        self._total_smem = max_smem
        return self._total_smem

    def compute_logical_grid_size(self) -> int:
        """Compute total logical blocks = max across all ops."""
        return max((n.logical_grid_size for n in self.nodes.values()), default=1)

    def get_topological_order(self) -> List[int]:
        """Return operation indices in topological order."""
        if self._topo_order is not None:
            return self._topo_order

        in_degree = {i: len(n.depends_on) for i, n in self.nodes.items()}
        queue = deque([i for i, d in in_degree.items() if d == 0])
        result = []

        while queue:
            op_idx = queue.popleft()
            result.append(op_idx)
            for dep in self.nodes[op_idx].dependents:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        self._topo_order = result
        return result

    def can_move_load_early(self, op_idx: int) -> bool:
        """Check if op's load can be moved earlier (overlap with prev compute).

        A load can be moved early if it has no dependencies on previous ops.
        Dependencies are inferred from reads/writes on TensorSpecs.
        """
        node = self.nodes[op_idx]

        # If this op depends on ANY previous op, don't move load early
        if node.depends_on:
            return False

        return True


# ============================================================================
# Shared Memory Planner
# ============================================================================


@dataclass
class SmemAllocation:
    """Describes shared memory allocation for an operation."""

    op_idx: int
    offset: int  # Byte offset in smem
    size: int  # Size in bytes


class SmemPlanner:
    """Plans static shared memory allocation from operation graph.

    Unlike legacy paged approach, this computes a fixed allocation
    at compile time based on graph analysis.

    Supports three allocation modes:
    1. Sequential (num_stages=1): All ops share one buffer (max size)
    2. Multi-stage (num_stages>1): Each op gets separate buffer
    3. Early-load overlap: Ops with overlapping loads get separate buffers

    Semaphore allocation (for warp-specialized kernels):
    - Per-op intra-op semaphores: load_done, compute_done (2 per op, 4 bytes each)
    - Inter-op semaphores: op_done (1 per op, 4 bytes each)
    """

    def __init__(self, graph: OperationGraph):
        self.graph = graph
        self.allocations: Dict[int, SmemAllocation] = {}
        self._total_bytes: int = 0
        self._early_load_ops: Set[int] = set()  # Ops that need separate smem for overlap
        self._semaphore_base_offset: int = 0  # Where semaphores start
        self._use_semaphores: bool = False
        self._num_ops: int = 0

    def plan(
        self,
        num_stages: int = 1,
        enable_early_load: bool = True,
        use_semaphores: bool = False,
    ) -> int:
        """Plan smem allocation.

        Args:
            num_stages: Number of pipeline stages (1 = no buffering, 2 = double buffer)
            enable_early_load: If True, allocate separate smem for ops that can overlap
            use_semaphores: If True, allocate semaphores for inter-op overlap

        Returns:
            Total shared memory bytes needed
        """
        # First, identify ops that can have their load moved early
        if enable_early_load:
            self._identify_early_load_ops()

        self._use_semaphores = use_semaphores
        self._num_ops = len(self.graph.nodes)

        max_size = 0
        current_offset = 0
        topo_order = self.graph.get_topological_order()

        for op_idx in topo_order:
            node = self.graph.nodes[op_idx]
            size = node.smem_bytes

            # Determine if this op needs its own buffer
            needs_separate = (
                num_stages > 1 or
                op_idx in self._early_load_ops
            )

            if needs_separate:
                self.allocations[op_idx] = SmemAllocation(
                    op_idx=op_idx,
                    offset=current_offset,
                    size=size,
                )
                current_offset += size
            else:
                # Share buffer with other sequential ops
                self.allocations[op_idx] = SmemAllocation(
                    op_idx=op_idx,
                    offset=0,
                    size=size,
                )
                max_size = max(max_size, size)

        # Total is sum of separate buffers + max of shared buffers
        if current_offset > 0:
            data_bytes = current_offset + max_size
        else:
            data_bytes = max_size

        # Record where semaphores start (after data buffers)
        self._semaphore_base_offset = data_bytes

        # Add semaphore space if needed
        if use_semaphores:
            semaphore_bytes = self._calculate_semaphore_bytes()
            self._total_bytes = data_bytes + semaphore_bytes
        else:
            self._total_bytes = data_bytes

        return self._total_bytes

    def _calculate_semaphore_bytes(self) -> int:
        """Calculate bytes needed for semaphores.

        Layout:
        - Per-op intra-op semaphores: [op0_load_done, op0_compute_done, op1_load_done, ...]
        - Inter-op semaphores: [op0_done, op1_done, ...]

        Each semaphore is 4 bytes (int32).
        """
        num_ops = self._num_ops
        # 2 intra-op semaphores per op (load_done, compute_done)
        intra_op_sems = num_ops * 2
        # 1 inter-op semaphore per op (op_done for dependent ops)
        inter_op_sems = num_ops
        total_sems = intra_op_sems + inter_op_sems
        return total_sems * 4  # 4 bytes per semaphore

    @property
    def semaphore_base_offset(self) -> int:
        """Byte offset where semaphores start in shared memory."""
        return self._semaphore_base_offset

    def get_intra_op_sem_offset(self, op_idx: int, sem_type: str) -> int:
        """Get byte offset for an intra-op semaphore.

        Args:
            op_idx: Operation index
            sem_type: "load_done" or "compute_done"

        Returns:
            Byte offset from smem base
        """
        base = self._semaphore_base_offset
        if sem_type == "load_done":
            return base + (op_idx * 2 * 4)
        elif sem_type == "compute_done":
            return base + (op_idx * 2 * 4) + 4
        else:
            raise ValueError(f"Unknown semaphore type: {sem_type}")

    def get_inter_op_sem_offset(self, op_idx: int) -> int:
        """Get byte offset for an inter-op semaphore.

        Args:
            op_idx: Operation index

        Returns:
            Byte offset from smem base
        """
        # Inter-op semaphores come after intra-op semaphores
        intra_op_bytes = self._num_ops * 2 * 4
        return self._semaphore_base_offset + intra_op_bytes + (op_idx * 4)

    def _identify_early_load_ops(self):
        """Identify ops that can have their load overlapped with previous compute."""
        self._early_load_ops.clear()
        topo_order = self.graph.get_topological_order()

        for i, op_idx in enumerate(topo_order):
            if i == 0:
                continue  # First op can't be early-loaded

            # Check if this op can be loaded early
            if self.graph.can_move_load_early(op_idx):
                node = self.graph.nodes[op_idx]
                if node.smem_bytes > 0:
                    # This op needs separate smem because its load overlaps
                    # with the previous op's compute
                    self._early_load_ops.add(op_idx)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def get_allocation(self, op_idx: int) -> Optional[SmemAllocation]:
        return self.allocations.get(op_idx)

    def needs_separate_buffer(self, op_idx: int) -> bool:
        """Check if an op needs its own smem buffer (not shared)."""
        return op_idx in self._early_load_ops
