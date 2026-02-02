# Copyright (c) 2025, Machete Authors
"""
GPU-Executable Operations for Megakernel.

This module defines the operation protocol for GPU execution using CuTe DSL.
Operations are templates that get inlined at compile time, enabling full
compiler optimization with no runtime dispatch overhead.

All operations must implement @cute.jit forward methods:
- load_forward: Load data from global to shared memory
- compute_forward: Perform the computation
- store_forward: Store results to global memory

Operations may optionally implement backward methods:
- load_backward: Load data/gradients from global to shared memory
- compute_backward: Compute gradients
- store_backward: Store gradients to global memory

Usage:
    from machete.megakernel import ScheduledOp, NOPOp, Megakernel

    ops = [
        ScheduledOp(NOPOp, tiles_m=32),
    ]

    kernel = Megakernel(ops)
    kernel.run()
"""

import enum
import linecache
import struct
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Type

import torch
from cutlass import Int32, Int64


# Counter for unique filenames in linecache for auto-generated init methods.
_init_counter = 0
# Track linecache entries for cleanup.
_linecache_entries: List[str] = []


def _parse_dims(dim_str: str) -> List[str]:
    """Parse a dim string like "M, H, D" into ["M", "H", "D"]."""
    return [d.strip() for d in dim_str.split(",")]


def _build_tensor_and_dim_lists(reads, writes):
    """Build ordered unique tensor list and dim list from reads/writes.

    Returns:
        unique_tensors: List of (name, dtype, dims) in declaration order, deduped.
        unique_dims: List of (dim_name, tensor_name, axis_index), ordered by first
            appearance, deduped by dim_name.
    """
    seen_tensors = set()
    unique_tensors = []
    seen_dims = set()
    unique_dims = []

    for name, (dtype, dim_str) in reads.items():
        if name not in seen_tensors:
            seen_tensors.add(name)
            dims = _parse_dims(dim_str)
            unique_tensors.append((name, dtype, dims))
            for i, d in enumerate(dims):
                if d not in seen_dims:
                    seen_dims.add(d)
                    unique_dims.append((d, name, i))

    for name, (dtype, dim_str) in writes.items():
        if name not in seen_tensors:
            seen_tensors.add(name)
            dims = _parse_dims(dim_str)
            unique_tensors.append((name, dtype, dims))
            for i, d in enumerate(dims):
                if d not in seen_dims:
                    seen_dims.add(d)
                    unique_dims.append((d, name, i))

    return unique_tensors, unique_dims


def _gen_init_source(unique_tensors, unique_dims):
    """Generate init_forward/init_backward method source code.

    The generated function reads pointers and dims from the config tensor
    in global memory, then creates CuTe flat tensor views. Variables are
    visible to compute_forward/compute_backward after inlining by compile.py.
    """
    lines = []
    lines.append("tidx = cute.arch.thread_idx()[0]")
    lines.append("num_threads = cute.arch.block_dim()[0]")

    # Read pointers (int64, 2 int32 words each)
    for i, (name, dtype, dims) in enumerate(unique_tensors):
        lines.append(
            f"{name}_ptr_raw = ld_global_i64(op_config_ptr, Int32({i}))"
        )

    # Read dims (int32, after pointers)
    dim_offset = 2 * len(unique_tensors)
    for j, (dim_name, _, _) in enumerate(unique_dims):
        lines.append(
            f"{dim_name} = ld_global_i32(op_config_ptr, Int32({dim_offset + j}))"
        )

    # Create CuTe tensor views
    for name, dtype, dims in unique_tensors:
        dtype_name = dtype.__name__ if hasattr(dtype, '__name__') else str(dtype)
        lines.append(
            f"{name} = cute.make_tensor("
            f"cute.make_ptr({dtype_name}, {name}_ptr_raw, cute.AddressSpace.gmem), "
            f"cute.make_layout(_FLAT))"
        )

    return "\n".join(lines)


def _gen_pack_config(cls, unique_tensors, unique_dims, reads, writes):
    """Generate and attach pack_config / pack_backward_config staticmethods."""

    def _make_pack_fn(tensors_decl, dims_list):
        """Create a pack_config function for a given tensor/dim specification."""
        num_unique = len(tensors_decl)
        num_dims = len(dims_list)
        config_size = 2 * num_unique + num_dims

        # Pre-compute dim extraction info: (dim_name, tensor_name, axis_index)
        dim_info = dims_list

        def pack_config(**tensors):
            config = torch.zeros(config_size, dtype=torch.int32,
                                 device=next(iter(tensors.values())).device)
            # Pack pointers
            for i, (name, dtype, dims) in enumerate(tensors_decl):
                t = tensors[name]
                assert t.is_contiguous(), f"{name} must be contiguous"
                lo, hi = struct.unpack('<2i', struct.pack('<Q', t.data_ptr()))
                config[2 * i] = lo
                config[2 * i + 1] = hi
            # Pack dims
            offset = 2 * num_unique
            for j, (dim_name, tensor_name, axis_idx) in enumerate(dim_info):
                config[offset + j] = tensors[tensor_name].shape[axis_idx]
            return config

        return staticmethod(pack_config)

    cls.pack_config = _make_pack_fn(unique_tensors, unique_dims)

    # Backward config: use backward_reads/backward_writes if defined
    bwd_reads = getattr(cls, 'backward_reads', None) or reads
    bwd_writes = getattr(cls, 'backward_writes', None) or writes
    if bwd_reads is reads and bwd_writes is writes:
        cls.pack_backward_config = cls.pack_config
    else:
        bwd_tensors, bwd_dims = _build_tensor_and_dim_lists(bwd_reads, bwd_writes)
        cls.pack_backward_config = _make_pack_fn(bwd_tensors, bwd_dims)


def _gen_init_method(cls, source, method_name):
    """Generate an init method from source, register in linecache, attach to cls."""
    global _init_counter
    _init_counter += 1
    unique_filename = f"<{cls.__name__}.{method_name}_{_init_counter}>"

    fn_source = (
        f"def {method_name}(smem_base, config_ptr, page_ids,"
        " tile_m, tile_n, tile_l, op_config_ptr):\n"
        + textwrap.indent(source, "    ")
        + "\n"
    )

    # Register in linecache for inspect.getsource() (needed by CuTe DSL)
    linecache.cache[unique_filename] = (
        len(fn_source),
        None,
        fn_source.splitlines(True),
        unique_filename,
    )
    _linecache_entries.append(unique_filename)

    code = compile(fn_source, unique_filename, "exec")
    exec_globals = {
        "cute": __import__("cutlass").cute,
        "Int32": Int32,
        "Int64": Int64,
    }
    # Import dtype names used in tensor declarations
    import cutlass
    for name in dir(cutlass):
        obj = getattr(cutlass, name)
        if isinstance(obj, type) or (hasattr(obj, '__name__') and
                                      name[0].isupper()):
            exec_globals[name] = obj

    # Import interpreter primitives
    from machete.megakernel.interpreter import ld_global_i64, ld_global_i32
    exec_globals["ld_global_i64"] = ld_global_i64
    exec_globals["ld_global_i32"] = ld_global_i32
    exec_globals["_FLAT"] = 1 << 24

    exec(code, exec_globals)
    setattr(cls, method_name, staticmethod(exec_globals[method_name]))


def _noop_method():
    """Return a no-op staticmethod for load/store stubs."""
    @staticmethod
    def noop(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l,
             op_config_ptr):
        pass
    return noop


def _process_op_declarations(cls):
    """Process reads/writes/tile declarations on an Op subclass.

    Auto-generates: INPUTS, OUTPUTS, NUM_INPUT_PAGES, NUM_OUTPUT_PAGES,
    DIM_NAMES, compute_tiles, tiles_n, tiles_l, pack_config,
    pack_backward_config, init_forward, init_backward, load/store stubs,
    and schedule() classmethod.
    """
    reads = cls.reads
    writes = cls.writes
    tile = cls.tile

    # Normalize tile to tuple
    if isinstance(tile, str):
        tile = (tile,)

    # Build tensor and dim lists
    unique_tensors, unique_dims = _build_tensor_and_dim_lists(reads, writes)

    # Set INPUTS / OUTPUTS
    cls.INPUTS = list(reads.keys())
    cls.OUTPUTS = list(writes.keys())

    # Zero-page (global memory) ops
    cls.NUM_INPUT_PAGES = 0
    cls.NUM_OUTPUT_PAGES = 0

    # DIM_NAMES: map tile dim names to axes
    axis_names = ("m", "n", "l")
    cls.DIM_NAMES = {tile[i]: axis_names[i] for i in range(len(tile))}

    # compute_tiles / tiles_n / tiles_l
    # Find which tensor and axis each tile dim comes from
    dim_lookup = {d: (tname, idx) for d, tname, idx in unique_dims}

    tile_dims = []
    for t in tile:
        if t not in dim_lookup:
            raise ValueError(
                f"Tile dim '{t}' not found in any tensor shape declaration"
            )
        tile_dims.append(dim_lookup[t])

    def _make_tile_fn(tensor_name, axis_idx):
        def tile_fn(**tensors):
            return tensors[tensor_name].shape[axis_idx]
        return staticmethod(tile_fn)

    cls.compute_tiles = _make_tile_fn(*tile_dims[0])
    if len(tile) > 1:
        cls.tiles_n = _make_tile_fn(*tile_dims[1])
    if len(tile) > 2:
        cls.tiles_l = _make_tile_fn(*tile_dims[2])

    # Generate pack_config / pack_backward_config
    _gen_pack_config(cls, unique_tensors, unique_dims, reads, writes)

    # Generate init_forward
    fwd_init_source = _gen_init_source(unique_tensors, unique_dims)
    _gen_init_method(cls, fwd_init_source, "init_forward")

    # Generate init_backward (same layout if backward_reads/writes not set)
    bwd_reads = getattr(cls, 'backward_reads', None) or reads
    bwd_writes = getattr(cls, 'backward_writes', None) or writes
    if bwd_reads is reads and bwd_writes is writes:
        bwd_init_source = fwd_init_source
    else:
        bwd_tensors, bwd_dims = _build_tensor_and_dim_lists(bwd_reads, bwd_writes)
        bwd_init_source = _gen_init_source(bwd_tensors, bwd_dims)
    _gen_init_method(cls, bwd_init_source, "init_backward")

    # Generate load/store stubs (pass) if not already user-defined
    for method_name in ("load_forward", "store_forward",
                        "load_backward", "store_backward"):
        existing = getattr(cls, method_name, None)
        # Check if it's the base Op's default or not overridden
        if existing is None or existing is getattr(Op, method_name, None):
            setattr(cls, method_name, _noop_method())

    # Add schedule() classmethod
    @classmethod
    def schedule(cls, **tensors):
        """Create a ScheduledOp from tensor kwargs."""
        tiles_m = cls.compute_tiles(**tensors)
        tiles_n = cls.tiles_n(**tensors)
        tiles_l = cls.tiles_l(**tensors)
        config_data = cls.pack_config(**tensors)
        return ScheduledOp(
            op_cls=cls,
            tiles_m=tiles_m,
            tiles_n=tiles_n,
            tiles_l=tiles_l,
            config_data=config_data,
            dim_names=cls.DIM_NAMES,
        )

    cls.schedule = schedule


# =============================================================================
# Execution Mode
# =============================================================================


class ExecutionMode(enum.Enum):
    """Execution strategy for an op's load/compute/store phases.

    SEQUENTIAL: All threads execute load → sync → compute → sync → store.
    WARP_SPECIALIZED: Producer warps do load/store, consumer warps do compute,
                      synchronized via hardware mbarrier.
    """

    SEQUENTIAL = "sequential"
    WARP_SPECIALIZED = "warp_specialized"


# =============================================================================
# Operation Protocol
# =============================================================================


class Op(ABC):
    """Abstract base for GPU-executable operations.

    Each operation implements static methods that get inlined into the
    megakernel at compile time. The methods receive raw CuTe DSL types
    (Int32 pointers and indices) so they can execute on the GPU.

    Operations must implement forward methods:
    - load_forward: Load data from global to shared memory
    - compute_forward: Perform the computation
    - store_forward: Store results to global memory

    Operations may optionally implement backward methods:
    - load_backward: Load data/gradients from global to shared memory
    - compute_backward: Compute gradients
    - store_backward: Store gradients to global memory

    Operations may optionally implement init methods:
    - init_forward: Shared setup that runs before warp split (e.g., pipeline
      init, TMA partitions, smem tensor views). Its body is inlined into the
      same function scope as load/compute/store, so local variables are visible
      to all phases.
    - init_backward: Same as init_forward but for the backward pass.

    Method signature (same for forward and backward):
        def method(smem_base: Int32, config_ptr: Int32,
                   page_ids: tuple[Int32, ...],
                   tile_m: Int32, tile_n: Int32, tile_l: Int32,
                   op_config_ptr: Int64) -> None

    Where:
        smem_base: Base pointer to shared memory
        config_ptr: Pointer to PageTableConfig in shared memory
        page_ids: Tuple of physical page IDs (length = NUM_INPUT + NUM_OUTPUT)
                  Use get_page_data_ptr(smem_base, config_ptr, pid) to get data pointer
        tile_m/n/l: Tile indices for this work item
        op_config_ptr: Pointer to per-op config struct in global memory (64-bit).
                       Each op defines its own config layout (tensor pointers, shapes, etc.).

    Class attributes:
        NUM_INPUT_PAGES: Number of input shared memory pages needed
        NUM_OUTPUT_PAGES: Number of output shared memory pages needed
        INPUTS: Named global memory buffers this op reads from
        OUTPUTS: Named global memory buffers this op produces
    """

    NUM_INPUT_PAGES: ClassVar[int] = 1
    NUM_OUTPUT_PAGES: ClassVar[int] = 1

    # Named buffer declarations for automatic dependency inference.
    # Each string names a global memory buffer. The builder matches
    # producer OUTPUTS to consumer INPUTS to build the dependency DAG.
    INPUTS: ClassVar[List[str]] = []
    OUTPUTS: ClassVar[List[str]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'reads') and hasattr(cls, 'writes'):
            _process_op_declarations(cls)

    # --- Forward pass ---

    @staticmethod
    def init_forward(
        smem_base: Int32, config_ptr: Int32, page_ids: tuple,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Shared setup before warp split (optional).

        Runs before load/compute/store in both sequential and warp-specialized
        modes. In warp-specialized mode, its body is inlined before the warp_id
        branch, so any variables defined here are visible to both producer and
        consumer warps.

        Use this for pipeline initialization, TMA descriptor partitioning,
        shared memory tensor views, or any setup that must be shared across
        warp roles.
        """
        pass

    @staticmethod
    @abstractmethod
    def load_forward(
        smem_base: Int32, config_ptr: Int32, page_ids: tuple,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Load data from global to shared memory."""
        ...

    @staticmethod
    @abstractmethod
    def compute_forward(
        smem_base: Int32, config_ptr: Int32, page_ids: tuple,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Perform the forward computation."""
        ...

    @staticmethod
    @abstractmethod
    def store_forward(
        smem_base: Int32, config_ptr: Int32, page_ids: tuple,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Store results to global memory."""
        ...

    # --- Backward pass (optional) ---

    @staticmethod
    def init_backward(
        smem_base: Int32, config_ptr: Int32, page_ids: tuple,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Shared setup before warp split for the backward pass (optional).

        Same role as init_forward but for gradient computation.
        """
        pass

    @staticmethod
    def load_backward(
        smem_base: Int32, config_ptr: Int32, page_ids: tuple,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Load data/gradients from global to shared memory for backward pass."""
        pass

    @staticmethod
    def compute_backward(
        smem_base: Int32, config_ptr: Int32, page_ids: tuple,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Compute gradients."""
        pass

    @staticmethod
    def store_backward(
        smem_base: Int32, config_ptr: Int32, page_ids: tuple,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Store gradients to global memory."""
        pass

    # --- Host-side tiling (used by autograd and scheduling) ---

    @staticmethod
    def compute_tiles(**tensors) -> int:
        """Compute tiles_m from input tensor shapes.

        Args:
            **tensors: Named input tensors (e.g., q=q_tensor, cos=cos_tensor).

        Returns:
            tiles_m for the ScheduledOp.
        """
        raise NotImplementedError(
            "Op subclasses must implement compute_tiles()"
        )

    @staticmethod
    def tiles_n(**tensors) -> int:
        """Compute tiles_n from input tensor shapes. Default: 1."""
        return 1

    @staticmethod
    def tiles_l(**tensors) -> int:
        """Compute tiles_l from input tensor shapes. Default: 1."""
        return 1


# =============================================================================
# NOP Operation
# =============================================================================


class NOPOp(Op):
    """No-operation for synchronization barriers. Requires 0 pages."""

    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0

    @staticmethod
    def init_forward(
        smem_base: Int32, config_ptr: Int32, page_ids: tuple,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        pass

    @staticmethod
    def load_forward(
        smem_base: Int32, config_ptr: Int32, page_ids: tuple,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        pass

    @staticmethod
    def compute_forward(
        smem_base: Int32, config_ptr: Int32, page_ids: tuple,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        pass

    @staticmethod
    def store_forward(
        smem_base: Int32, config_ptr: Int32, page_ids: tuple,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        pass


# =============================================================================
# Scheduled Operation
# =============================================================================


@dataclass
class ScheduledOp:
    """An operation scheduled for execution with tile requirements.

    Pairs an operation class with the number of tiles to process and
    an execution strategy.

    Attributes:
        op_cls: Operation class (must be Op subclass)
        tiles_m: Number of tiles in M dimension
        tiles_n: Number of tiles in N dimension
        tiles_l: Number of tiles in L dimension
        execution_mode: How to execute load/compute/store phases
        num_producer_warps: Producer warps for WARP_SPECIALIZED mode
        params: Operation-specific parameters
        dim_names: Maps semantic dimension names to tile axes ("m", "n", "l").
            Example: {"batch": "m", "seqlen": "n"} means tile_m indexes batch
            and tile_n indexes seqlen. Used by the builder to compute tile
            mappings between ops with different grid shapes.
    """

    op_cls: Type[Op]
    tiles_m: int
    tiles_n: int = 1
    tiles_l: int = 1
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    num_producer_warps: int = 1
    params: Dict[str, Any] = field(default_factory=dict)
    dim_names: Dict[str, str] = None
    config_data: Any = None  # Optional torch.Tensor with per-op config in global memory

    def __post_init__(self):
        if self.dim_names is None:
            self.dim_names = getattr(self.op_cls, 'DIM_NAMES', {})

    @property
    def total_tiles(self) -> int:
        """Total number of tiles for this operation."""
        return self.tiles_m * self.tiles_n * self.tiles_l

    def tiles_for_axis(self, axis: str) -> int:
        """Get tile count for a given axis ("m", "n", or "l")."""
        return {"m": self.tiles_m, "n": self.tiles_n, "l": self.tiles_l}[axis]

    def axis_for_dim(self, dim_name: str) -> Optional[str]:
        """Get the tile axis for a semantic dimension name, or None."""
        return self.dim_names.get(dim_name)


# =============================================================================
# Compile-Time Barrier Formulas
# =============================================================================


@dataclass
class BarrierFormula:
    """Compile-time formula for computing a barrier index from tile coordinates.

    barrier_idx = base + coeff_m * tile_m + coeff_n * tile_n + coeff_l * tile_l

    Used by the megakernel handler to bake barrier wait/signal calls directly
    into each op's handler at JIT compile time. No per-instruction encoding
    needed — the formula coefficients are Python-level constants captured
    by closure.

    Attributes:
        base: Barrier base offset
        coeff_m: Multiplier for tile_m
        coeff_n: Multiplier for tile_n
        coeff_l: Multiplier for tile_l
        expected: For wait deps: how many signals to wait for (default 1)
        guard_max: Only execute when the computed linear index
            (coeff_m * tile_m + coeff_n * tile_n + coeff_l * tile_l)
            is less than this value. Defaults to NO_GUARD (always passes).
            Used for legacy mode with mismatched tile counts.
    """
    # Sentinel: guard_max value that always passes (larger than any tile count)
    NO_GUARD: ClassVar[int] = 2**30

    base: int
    coeff_m: int = 0
    coeff_n: int = 0
    coeff_l: int = 0
    expected: int = 1
    guard_max: int = NO_GUARD

    def compute_index(self, tile_m: int, tile_n: int, tile_l: int) -> int:
        """Compute barrier index for a given tile (host-side, for testing)."""
        return (self.base
                + self.coeff_m * tile_m
                + self.coeff_n * tile_n
                + self.coeff_l * tile_l)

    def is_guarded(self, tile_m: int, tile_n: int, tile_l: int) -> bool:
        """Check if the guard passes for a given tile (host-side, for testing)."""
        linear = self.coeff_m * tile_m + self.coeff_n * tile_n + self.coeff_l * tile_l
        return linear < self.guard_max

    @property
    def has_guard(self) -> bool:
        """Whether this formula has an active guard (not NO_GUARD)."""
        return self.guard_max != self.NO_GUARD


# =============================================================================
# Instruction Stream (Lightweight — barriers baked into handlers)
# =============================================================================

INSTRUCTION_WORDS = 4


@dataclass
class TileInstruction:
    """A single tile work instruction for the persistent megakernel.

    Lightweight encoding in global memory (4 x int32):
    [0]  op_idx: Which operation (indexes into op list), or -1 for end marker
    [1]  tile_m: M tile index within this op
    [2]  tile_n: N tile index within this op
    [3]  tile_l: L tile index within this op

    Barrier wait/signal logic is baked into op handlers at compile time
    via BarrierFormula, not encoded in the instruction stream.
    """

    op_idx: int
    tile_m: int
    tile_n: int
    tile_l: int

    # Sentinel for end of stream
    END_MARKER: int = -1

    def pack(self) -> List[int]:
        """Pack into list of int32."""
        return [self.op_idx, self.tile_m, self.tile_n, self.tile_l]

    @classmethod
    def end_instruction(cls) -> "TileInstruction":
        """Create end-of-stream marker."""
        return cls(op_idx=cls.END_MARKER, tile_m=0, tile_n=0, tile_l=0)


# =============================================================================
# Dependency Resolution Helpers
# =============================================================================


@dataclass
class _OpRecord:
    """Internal record for an op added to the builder."""
    op_idx: int
    op: ScheduledOp
    # Flat list of tile coordinate tuples: [(m, n, l), ...]
    tiles: List[Tuple[int, int, int]]


@dataclass
class _DepEdge:
    """A dependency edge between two ops for tile scheduling.

    Attributes:
        producer_idx: Index of the producing op
        consumer_idx: Index of the consuming op
        kind: Dependency type — determines how many producer tiles
              must be emitted before each consumer tile:
              "one_to_one": consumer tile k needs producer tile k
              "many_to_one": each consumer tile needs ALL producer tiles
              "one_to_many": consumer tile k needs a proportional subset
    """
    producer_idx: int
    consumer_idx: int
    kind: str  # "one_to_one", "many_to_one", "one_to_many"


def _compute_formula_coeffs(
    source_op: ScheduledOp,
    target_op: ScheduledOp,
    shared_dims: Set[str],
) -> Tuple[int, int, int]:
    """Compute (coeff_m, coeff_n, coeff_l) for source computing target's barrier index.

    Maps shared dimension values from source tile coordinates to target's
    linear index. The target's strides are: m=1, n=target.tiles_m,
    l=target.tiles_m * target.tiles_n.

    For each shared dim, find which axis it maps to on each side, then
    accumulate the target stride onto the source axis coefficient.
    """
    s_dims = source_op.dim_names
    t_dims = target_op.dim_names

    t_strides = {
        "m": 1,
        "n": target_op.tiles_m,
        "l": target_op.tiles_m * target_op.tiles_n,
    }

    cm, cn, cl = 0, 0, 0

    if shared_dims:
        for dim in shared_dims:
            s_axis = s_dims[dim]
            t_axis = t_dims[dim]
            stride = t_strides[t_axis]

            if s_axis == "m":
                cm += stride
            elif s_axis == "n":
                cn += stride
            elif s_axis == "l":
                cl += stride
    else:
        # No dim_names: use source's own linear index
        cm = 1
        cn = source_op.tiles_m
        cl = source_op.tiles_m * source_op.tiles_n

    return cm, cn, cl


# =============================================================================
# Instruction Stream Builder
# =============================================================================


class InstructionStreamBuilder:
    """Builds instruction stream with compile-time baked barrier formulas.

    Supports automatic dependency inference from named buffers declared on ops.
    Dependencies are resolved at build time and expressed as BarrierFormula
    objects that get baked into op handlers at JIT compile time.

    Tile mappings between ops with different grid shapes are computed via
    named dimensions (dim_names on ScheduledOp):
    - Shared dims: 1:1 tile mapping
    - Producer-only dims (many-to-one): consumer waits for all producer tiles
    - Consumer-only dims (one-to-many): all consumer tiles wait on one producer

    Example with named buffers:
        builder = InstructionStreamBuilder()
        builder.add_op(OpA, tiles_m=4, tiles_n=32,
                       dim_names={"batch": "m", "seqlen": "n"})
        builder.add_op(OpB, tiles_m=4,
                       dim_names={"batch": "m"})
        # OpB reads OpA's output → OpB batch=0 waits for all 32 seqlen tiles

    Legacy mode (no INPUTS/OUTPUTS): linear chain with 1:1 tile mapping.
    """

    def __init__(self):
        self._op_records: List[_OpRecord] = []
        self._cached_formulas: Optional[Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]]] = None
        self._barrier_count: Optional[int] = None

    @property
    def ops(self) -> List[ScheduledOp]:
        """Scheduled ops in order."""
        return [r.op for r in self._op_records]

    def add_op(
        self,
        op_cls: Type[Op],
        tiles_m: int,
        tiles_n: int = 1,
        tiles_l: int = 1,
        dim_names: Optional[Dict[str, str]] = None,
        **params,
    ) -> "InstructionStreamBuilder":
        """Add an operation to the stream.

        Args:
            op_cls: Operation class
            tiles_m: Number of M tiles
            tiles_n: Number of N tiles
            tiles_l: Number of L tiles
            dim_names: Maps semantic dimension names to tile axes.
                Example: {"batch": "m", "seqlen": "n"}
            **params: Operation-specific parameters
        """
        # Validate dim_names
        _valid_axes = {"m", "n", "l"}
        if dim_names:
            for dim, axis in dim_names.items():
                if axis not in _valid_axes:
                    raise ValueError(
                        f"Invalid axis '{axis}' for dim '{dim}'. "
                        f"Must be one of {_valid_axes}"
                    )
            axes = list(dim_names.values())
            if len(axes) != len(set(axes)):
                raise ValueError(
                    f"dim_names maps multiple dims to the same axis: "
                    f"{dim_names}"
                )

        op = ScheduledOp(
            op_cls=op_cls,
            tiles_m=tiles_m,
            tiles_n=tiles_n,
            tiles_l=tiles_l,
            params=params,
            dim_names=dim_names or {},
        )

        # Pre-compute flat tile list
        tiles = []
        for tile_idx in range(op.total_tiles):
            tile_l_idx = tile_idx // (tiles_m * tiles_n)
            tile_n_idx = (tile_idx // tiles_m) % tiles_n
            tile_m_idx = tile_idx % tiles_m
            tiles.append((tile_m_idx, tile_n_idx, tile_l_idx))

        record = _OpRecord(
            op_idx=len(self._op_records),
            op=op,
            tiles=tiles,
        )
        self._op_records.append(record)
        # Invalidate cache
        self._cached_formulas = None
        self._barrier_count = None
        return self

    def _has_named_buffers(self) -> bool:
        """Check if any op declares INPUTS or OUTPUTS."""
        return any(
            r.op.op_cls.INPUTS or r.op.op_cls.OUTPUTS
            for r in self._op_records
        )

    def _resolve_dep_edges(self) -> List[_DepEdge]:
        """Resolve op-level dependency edges for tile scheduling.

        Returns a list of _DepEdge with producer/consumer indices and
        dependency kind. Used by build() to determine tile emission order.
        """
        if not self._has_named_buffers():
            # Legacy linear chain: each op depends on previous, 1:1
            return [
                _DepEdge(producer_idx=i - 1, consumer_idx=i, kind="one_to_one")
                for i in range(1, len(self._op_records))
            ]

        # Named buffer dependencies
        buffer_producers: Dict[str, int] = {}
        for rec in self._op_records:
            for buf in rec.op.op_cls.OUTPUTS:
                buffer_producers[buf] = rec.op_idx

        edges: List[_DepEdge] = []
        for rec in self._op_records:
            for buf in rec.op.op_cls.INPUTS:
                if buf not in buffer_producers:
                    continue  # External input (provided by host, not by another op)
                prod_idx = buffer_producers[buf]
                if prod_idx == rec.op_idx:
                    continue  # Self-dependency (in-place op reads/writes same buffer)
                producer = self._op_records[prod_idx]

                p_dims = set(producer.op.dim_names.keys())
                c_dims = set(rec.op.dim_names.keys())
                producer_only = p_dims - c_dims
                consumer_only = c_dims - p_dims

                if producer_only and not consumer_only:
                    kind = "many_to_one"
                elif consumer_only and not producer_only:
                    kind = "one_to_many"
                else:
                    kind = "one_to_one"

                edges.append(_DepEdge(
                    producer_idx=prod_idx,
                    consumer_idx=rec.op_idx,
                    kind=kind,
                ))

        return edges

    @staticmethod
    def _producer_threshold(
        edge: _DepEdge,
        consumer_cursor: int,
        producer_total: int,
        consumer_total: int,
    ) -> int:
        """Minimum producer tiles that must be emitted before consumer tile k.

        Args:
            edge: The dependency edge
            consumer_cursor: Current consumer tile index (k)
            producer_total: Total tiles in the producer op
            consumer_total: Total tiles in the consumer op
        """
        if edge.kind == "many_to_one":
            return producer_total
        elif edge.kind == "one_to_many":
            # Consumer tile k needs producer tile floor(k * P / C)
            return min(
                (consumer_cursor * producer_total) // consumer_total + 1,
                producer_total,
            )
        else:
            # one_to_one: consumer tile k needs producer tile k
            return min(consumer_cursor + 1, producer_total)

    def _resolve(self) -> Tuple[
        Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]],
        int,
    ]:
        """Resolve dependencies into barrier formulas.

        Returns:
            (formulas, barrier_count)
            formulas: {op_idx: (wait_formulas, signal_formulas)}
        """
        if self._cached_formulas is not None:
            return self._cached_formulas, self._barrier_count

        if self._has_named_buffers():
            formulas, count = self._resolve_named_formulas()
        else:
            formulas, count = self._resolve_legacy_formulas()

        self._cached_formulas = formulas
        self._barrier_count = count
        return formulas, count

    def _resolve_legacy_formulas(self) -> Tuple[
        Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]],
        int,
    ]:
        """Compute barrier formulas for legacy linear chain.

        Each op signals its own barrier set. The next op waits on the
        previous op's barriers with 1:1 tile index mapping.
        """
        formulas: Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]] = {}
        barrier_counter = 0

        for i, rec in enumerate(self._op_records):
            op = rec.op
            signal_base = barrier_counter

            # Own linear index strides
            cm = 1
            cn = op.tiles_m
            cl = op.tiles_m * op.tiles_n

            # Signal: own barrier
            signal_formulas = [BarrierFormula(
                base=signal_base, coeff_m=cm, coeff_n=cn, coeff_l=cl,
            )]

            # Wait: previous op's barrier (if exists)
            wait_formulas: List[BarrierFormula] = []
            if i > 0:
                prev_op = self._op_records[i - 1].op
                prev_base = barrier_counter - prev_op.total_tiles
                # Guard for mismatched tile counts
                guard = (prev_op.total_tiles
                         if prev_op.total_tiles != op.total_tiles
                         else BarrierFormula.NO_GUARD)
                wait_formulas.append(BarrierFormula(
                    base=prev_base, coeff_m=cm, coeff_n=cn, coeff_l=cl,
                    expected=1, guard_max=guard,
                ))

            formulas[i] = (wait_formulas, signal_formulas)
            barrier_counter += op.total_tiles

        return formulas, barrier_counter

    def _resolve_named_formulas(self) -> Tuple[
        Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]],
        int,
    ]:
        """Compute barrier formulas from named buffer dependencies.

        For each (producer, consumer, buffer) edge:
        1. Determine barrier allocation side (target):
           - Many-to-one → target = consumer (fewer tiles)
           - One-to-many → target = producer (fewer tiles)
           - 1:1 → target = consumer
        2. Compute formula coefficients for both sides
        3. Allocate barriers
        """
        # Track buffer producers
        buffer_producers: Dict[str, int] = {}
        for rec in self._op_records:
            for buf in rec.op.op_cls.OUTPUTS:
                if buf in buffer_producers:
                    raise ValueError(
                        f"Buffer '{buf}' produced by both op {buffer_producers[buf]} "
                        f"and op {rec.op_idx}"
                    )
                buffer_producers[buf] = rec.op_idx

        # Init per-op formula lists
        formulas: Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]] = {
            i: ([], []) for i in range(len(self._op_records))
        }
        barrier_counter = 0

        # Process each dependency edge
        for rec in self._op_records:
            for buf in rec.op.op_cls.INPUTS:
                if buf not in buffer_producers:
                    continue  # External input (provided by host, not by another op)
                prod_idx = buffer_producers[buf]
                if prod_idx == rec.op_idx:
                    continue  # Self-dependency (in-place op reads/writes same buffer)
                producer = self._op_records[prod_idx]
                consumer = rec

                p_op = producer.op
                c_op = consumer.op
                p_dims = set(p_op.dim_names.keys())
                c_dims = set(c_op.dim_names.keys())
                shared_dims = p_dims & c_dims
                producer_only = p_dims - c_dims
                consumer_only = c_dims - p_dims

                if producer_only and consumer_only:
                    raise ValueError(
                        f"Unsupported dependency: producer has extra dims "
                        f"{producer_only} and consumer has extra dims "
                        f"{consumer_only}. Both-sides-extra is not supported."
                    )

                # Determine target side and compute barrier count / expected
                if producer_only and not consumer_only:
                    # Many-to-one: barriers indexed by consumer tiles
                    target_op = c_op
                    num_barriers = c_op.total_tiles
                    collapsed = 1
                    for dim in producer_only:
                        axis = p_op.dim_names[dim]
                        collapsed *= p_op.tiles_for_axis(axis)
                    expected = collapsed
                elif consumer_only and not producer_only:
                    # One-to-many: barriers indexed by producer tiles
                    target_op = p_op
                    num_barriers = p_op.total_tiles
                    expected = 1
                else:
                    # 1:1: same dims (or no dims)
                    target_op = c_op
                    num_barriers = max(p_op.total_tiles, c_op.total_tiles)
                    expected = 1

                # Compute formula coefficients
                p_cm, p_cn, p_cl = _compute_formula_coeffs(
                    p_op, target_op, shared_dims,
                )
                c_cm, c_cn, c_cl = _compute_formula_coeffs(
                    c_op, target_op, shared_dims,
                )

                # Producer signal formula
                formulas[prod_idx][1].append(BarrierFormula(
                    base=barrier_counter,
                    coeff_m=p_cm, coeff_n=p_cn, coeff_l=p_cl,
                ))

                # Consumer wait formula
                formulas[rec.op_idx][0].append(BarrierFormula(
                    base=barrier_counter,
                    coeff_m=c_cm, coeff_n=c_cn, coeff_l=c_cl,
                    expected=expected,
                ))

                barrier_counter += num_barriers

        return formulas, barrier_counter

    def build(self) -> List[TileInstruction]:
        """Build an interleaved instruction list using greedy wavefront scheduling.

        Emits tiles from multiple ops in an order that maximizes pipeline
        overlap: a consumer tile is emitted as soon as its producer
        dependencies have been emitted, rather than waiting for the entire
        producer op to finish.

        For a 2-op 1:1 chain (A→B) with 4 tiles each, the output is:
            A[0], A[1], B[0], A[2], B[1], A[3], B[2], B[3]
        instead of the flat sequential:
            A[0], A[1], A[2], A[3], B[0], B[1], B[2], B[3]
        """
        # Ensure formulas are resolved (sets _barrier_count)
        self._resolve()

        if not self._op_records:
            return [TileInstruction.end_instruction()]

        # Resolve dependency edges and group by consumer
        edges = self._resolve_dep_edges()
        consumer_deps: Dict[int, List[_DepEdge]] = {}
        for edge in edges:
            consumer_deps.setdefault(edge.consumer_idx, []).append(edge)

        # Greedy wavefront: emit tiles as soon as dependencies are met
        num_ops = len(self._op_records)
        cursors = [0] * num_ops
        total_tiles = sum(rec.op.total_tiles for rec in self._op_records)
        instructions: List[TileInstruction] = []

        while len(instructions) < total_tiles:
            progress = False
            for rec in self._op_records:
                idx = rec.op_idx
                if cursors[idx] >= rec.op.total_tiles:
                    continue

                # Check all producer dependencies for the next tile
                can_emit = True
                for edge in consumer_deps.get(idx, []):
                    prod_rec = self._op_records[edge.producer_idx]
                    needed = self._producer_threshold(
                        edge,
                        cursors[idx],
                        prod_rec.op.total_tiles,
                        rec.op.total_tiles,
                    )
                    if cursors[edge.producer_idx] < needed:
                        can_emit = False
                        break

                if can_emit:
                    tile = rec.tiles[cursors[idx]]
                    instructions.append(TileInstruction(
                        op_idx=idx,
                        tile_m=tile[0],
                        tile_n=tile[1],
                        tile_l=tile[2],
                    ))
                    cursors[idx] += 1
                    progress = True

            if not progress:
                scheduled = [
                    f"{self._op_records[i].op.op_cls.__name__}: "
                    f"{cursors[i]}/{self._op_records[i].op.total_tiles}"
                    for i in range(num_ops)
                ]
                raise RuntimeError(
                    f"Deadlock in tile scheduling. Cursors: {scheduled}"
                )

        instructions.append(TileInstruction.end_instruction())
        return instructions

    def build_tensor(self, device: str = "cuda"):
        """Build instruction stream as GPU tensor.

        Returns:
            Tensor of shape [num_instructions, INSTRUCTION_WORDS]
        """
        import torch

        instructions = self.build()
        packed = [instr.pack() for instr in instructions]
        return torch.tensor(packed, dtype=torch.int32, device=device)

    def get_op_barrier_formulas(self) -> Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]]:
        """Get per-op barrier formulas for compile-time baking.

        Returns:
            {op_idx: (wait_formulas, signal_formulas)}
            Each formula specifies how to compute a barrier index from
            tile coordinates at runtime.
        """
        formulas, _ = self._resolve()
        return formulas

    @property
    def total_tiles(self) -> int:
        """Total number of work tiles."""
        return sum(r.op.total_tiles for r in self._op_records)

    @property
    def num_barriers(self) -> int:
        """Number of barriers needed."""
        _, count = self._resolve()
        return count


__all__ = [
    # Protocol
    "Op",
    # Built-in Operations
    "NOPOp",
    # Execution Mode
    "ExecutionMode",
    # Scheduling
    "ScheduledOp",
    # Barrier Formulas
    "BarrierFormula",
    # Instruction Stream
    "INSTRUCTION_WORDS",
    "TileInstruction",
    "InstructionStreamBuilder",
]
