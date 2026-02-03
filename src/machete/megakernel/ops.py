# Copyright (c) 2025, Machete Authors
"""
GPU-Executable Operations for Megakernel.

This module defines the operation protocol for GPU execution using CuTe DSL.
Operations are templates that get inlined at compile time, enabling full
compiler optimization with no runtime dispatch overhead.

Subclass ``Op`` and declare tensors, tiling, then implement compute_forward::

    class MyOp(Op):
        reads  = {"x": (Float32, "M, D")}
        writes = {"x": (Float32, "M, D")}
        tile   = ("M",)

        @staticmethod
        def compute_forward(smem_base, config_ptr, page_ids,
                            tile_m, tile_n, tile_l, op_config_ptr):
            ...  # M is dynamic (tile dim), D is a static compile-time int

    ops = [MyOp.schedule(x=tensor)]
    kernel = Megakernel(ops)
    kernel.run()
"""

import enum
import struct
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Type

import cutlass
import torch
from cutlass import Int32, Int64


# =============================================================================
# Dtype Mapping
# =============================================================================

TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _resolve_dtype(declared_dtype, tensor_dtype):
    """Resolve a dtype declaration, allowing inference from tensor.

    Args:
        declared_dtype: The dtype declared in reads/writes. Can be:
            - A CUTLASS dtype (Float32, Float16, etc.) - used as-is
            - None - infer from tensor_dtype
        tensor_dtype: The actual PyTorch tensor dtype (e.g., torch.float16)

    Returns:
        The resolved CUTLASS dtype.
    """
    if declared_dtype is None:
        if tensor_dtype not in TORCH_TO_CUTLASS_DTYPE:
            raise ValueError(f"Unsupported tensor dtype: {tensor_dtype}")
        return TORCH_TO_CUTLASS_DTYPE[tensor_dtype]
    return declared_dtype


def _parse_dims(dim_str: str) -> List[str]:
    """Parse a dim string like "M, H, D" into ["M", "H", "D"]."""
    return [d.strip() for d in dim_str.split(",")]


def _parse_tile_spec(tile) -> Tuple[Tuple[str, ...], Dict[str, int]]:
    """Parse tile specification into dimension names and tile sizes.

    Supports two formats:
    - Simple: tile = ("M", "N") - all tile sizes default to 1
    - With sizes: tile = (("M", 2), ("N", 4)) - explicit tile sizes

    Mixed format is also supported:
    - tile = ("M", ("N", 4)) - M has size 1, N has size 4

    Returns:
        (dim_names, tile_sizes) where tile_sizes maps dim_name -> size
    """
    if isinstance(tile, str):
        tile = (tile,)

    dim_names = []
    tile_sizes = {}

    for item in tile:
        if isinstance(item, str):
            dim_names.append(item)
            tile_sizes[item] = 1
        elif isinstance(item, tuple) and len(item) == 2:
            dim_name, size = item
            dim_names.append(dim_name)
            tile_sizes[dim_name] = size
        else:
            raise ValueError(f"Invalid tile spec item: {item}. Expected str or (str, int) tuple.")

    return tuple(dim_names), tile_sizes


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

    for decls in (reads, writes):
        for name, (dtype, dim_str) in decls.items():
            if name not in seen_tensors:
                seen_tensors.add(name)
                dims = _parse_dims(dim_str)
                unique_tensors.append((name, dtype, dims))
                for i, d in enumerate(dims):
                    if d not in seen_dims:
                        seen_dims.add(d)
                        unique_dims.append((d, name, i))

    return unique_tensors, unique_dims


def _gen_init_source(
    unique_tensors,
    unique_dims,
    static_dims=None,
    tile_dim_names=None,
    dynamic_dim_overrides=None,
    kernel_config=None,
    tensor_dtypes=None,
):
    """Generate init source code with static dims as compile-time constants.

    Static dims (non-tile dimensions) are emitted as Python int literals,
    making them compile-time constants in CuTe DSL. This enables CuTe layout
    algebra, TMA descriptors, and vectorized loads for those dimensions.

    Dynamic dims (tile dimensions) are loaded from the config tensor at
    runtime via ld_global_i32.

    Args:
        unique_tensors: List of (name, dtype, dims) tuples.
        unique_dims: List of (dim_name, tensor_name, axis_idx) tuples.
        static_dims: Dict mapping static dim names to their int values.
            If None, all dims are treated as dynamic (legacy behavior).
        tile_dim_names: Set of dim names that are tile (dynamic) dimensions.
        dynamic_dim_overrides: Set of additional dim names to treat as dynamic.
        kernel_config: Dict of kernel configuration parameters to emit as
            compile-time constants (e.g., threads_per_row, tile_size_M).
    """
    lines = []
    lines.append("tidx = cute.arch.thread_idx()[0]")
    lines.append("num_threads = cute.arch.block_dim()[0]")

    # Emit kernel config parameters as compile-time constants
    if kernel_config:
        for name, value in kernel_config.items():
            lines.append(f"{name} = {value}")

    # Determine which dims are dynamic vs static
    if static_dims is None:
        # Legacy: all dims are dynamic
        dynamic_names = {d for d, _, _ in unique_dims}
    else:
        dynamic_names = set(tile_dim_names or ())
        if dynamic_dim_overrides:
            dynamic_names |= set(dynamic_dim_overrides)

    # Separate dims into dynamic (config-loaded) and static (literal)
    dynamic_dims = [(d, t, a) for d, t, a in unique_dims if d in dynamic_names]
    static_dim_list = [(d, t, a) for d, t, a in unique_dims if d not in dynamic_names]

    # Read pointers (int64, 2 int32 words each)
    for i, (name, dtype, dims) in enumerate(unique_tensors):
        lines.append(f"{name}_ptr_raw = ld_global_i64(op_config_ptr, Int32({i}))")

    # Dynamic dims: load from config (after pointers)
    dim_offset = 2 * len(unique_tensors)
    for j, (dim_name, _, _) in enumerate(dynamic_dims):
        lines.append(f"{dim_name} = ld_global_i32(op_config_ptr, Int32({dim_offset + j}))")

    # Static dims: Python int literals (compile-time constants in CuTe DSL)
    if static_dims is not None:
        for dim_name, _, _ in static_dim_list:
            lines.append(f"{dim_name} = {static_dims[dim_name]}")

    # Create CuTe tensor views and emit dtype constants
    for name, dtype, dims in unique_tensors:
        # Use overridden dtype if provided, otherwise use declared dtype
        if tensor_dtypes and name in tensor_dtypes:
            dtype = tensor_dtypes[name]
        dtype_name = dtype.__name__ if hasattr(dtype, "__name__") else str(dtype)
        lines.append(
            f"{name} = cute.make_tensor("
            f"cute.make_ptr({dtype_name}, {name}_ptr_raw, cute.AddressSpace.gmem), "
            f"cute.make_layout(_FLAT))"
        )
        # Emit dtype constant for use in compute methods (e.g., x_dtype = Float16)
        lines.append(f"{name}_dtype = {dtype_name}")

    return "\n".join(lines)


def _gen_pack_config(cls, unique_tensors, unique_dims, reads, writes, dynamic_dim_names=None):
    """Generate and attach pack_config / pack_backward_config staticmethods.

    Only packs pointers and dynamic dims into the config tensor. Static dims
    are baked into the compiled kernel as compile-time constants.
    """

    def _make_pack_fn(tensors_decl, dims_list, dyn_names):
        """Create a pack_config function for a given tensor/dim specification."""
        num_unique = len(tensors_decl)
        # Only pack dynamic dims
        dynamic_dims = [(d, t, a) for d, t, a in dims_list if d in dyn_names]
        num_dynamic = len(dynamic_dims)
        config_size = 2 * num_unique + num_dynamic

        def pack_config(**tensors):
            config = torch.zeros(config_size, dtype=torch.int32, device=next(iter(tensors.values())).device)
            # Pack pointers
            for i, (name, dtype, dims) in enumerate(tensors_decl):
                t = tensors[name]
                assert t.is_contiguous(), f"{name} must be contiguous"
                lo, hi = struct.unpack("<2i", struct.pack("<Q", t.data_ptr()))
                config[2 * i] = lo
                config[2 * i + 1] = hi
            # Pack dynamic dims only
            offset = 2 * num_unique
            for j, (dim_name, tensor_name, axis_idx) in enumerate(dynamic_dims):
                config[offset + j] = tensors[tensor_name].shape[axis_idx]
            return config

        return staticmethod(pack_config)

    # All dims are dynamic if no classification provided (legacy / NOPOp)
    if dynamic_dim_names is None:
        dynamic_dim_names = {d for d, _, _ in unique_dims}

    cls.pack_config = _make_pack_fn(unique_tensors, unique_dims, dynamic_dim_names)

    # Backward config: use backward_reads/backward_writes if defined
    bwd_reads = getattr(cls, "backward_reads", None) or reads
    bwd_writes = getattr(cls, "backward_writes", None) or writes
    if bwd_reads is reads and bwd_writes is writes:
        cls.pack_backward_config = cls.pack_config
    else:
        bwd_tensors, bwd_dims = _build_tensor_and_dim_lists(bwd_reads, bwd_writes)
        cls.pack_backward_config = _make_pack_fn(bwd_tensors, bwd_dims, dynamic_dim_names)


def _process_op_declarations(cls):
    """Process reads/writes/tile declarations on an Op subclass.

    Auto-generates: INPUTS, OUTPUTS, NUM_INPUT_PAGES, NUM_OUTPUT_PAGES,
    DIM_NAMES, compute_tiles, tiles_n, tiles_l, pack_config,
    pack_backward_config, load/store stubs, and schedule() classmethod.

    Stores tensor/dim metadata on the class for deferred init generation
    at compile time (when static dim values are known).

    Tile declaration format:
        tile = ("M",)              # Simple: tile_size defaults to 1
        tile = (("M", 2),)         # With size: M tiles process 2 items each
        tile = ("M", ("N", 4))     # Mixed: M size 1, N size 4
    """
    reads = cls.reads
    writes = cls.writes
    tile = cls.tile

    # Parse tile spec into dim names and sizes
    tile, tile_sizes = _parse_tile_spec(tile)

    # Build tensor and dim lists
    unique_tensors, unique_dims = _build_tensor_and_dim_lists(reads, writes)

    # --- Classify dims as static or dynamic ---
    # Tile dims are always dynamic (they determine grid size and change per input).
    # All other dims are static by default (model constants baked into compiled kernel).
    # Ops can override with dynamic_dims = ("S",) to keep specific dims dynamic.
    tile_dim_names = set(tile)
    dynamic_dim_overrides = set(getattr(cls, "dynamic_dims", ()))
    dynamic_dim_names = tile_dim_names | dynamic_dim_overrides
    static_dim_names = [d for d, _, _ in unique_dims if d not in dynamic_dim_names]

    # Store metadata for deferred init generation at compile time
    cls._UNIQUE_TENSORS = unique_tensors
    cls._UNIQUE_DIMS = unique_dims
    cls._TILE_DIM_NAMES = tile_dim_names
    cls._DYNAMIC_DIM_OVERRIDES = dynamic_dim_overrides
    cls._STATIC_DIM_NAMES = static_dim_names
    cls._TILE_SIZES = tile_sizes  # {dim_name: tile_size}

    # Backward tensor/dim metadata (may differ from forward)
    bwd_reads = getattr(cls, "backward_reads", None) or reads
    bwd_writes = getattr(cls, "backward_writes", None) or writes
    if bwd_reads is reads and bwd_writes is writes:
        cls._BWD_UNIQUE_TENSORS = unique_tensors
        cls._BWD_UNIQUE_DIMS = unique_dims
    else:
        cls._BWD_UNIQUE_TENSORS, cls._BWD_UNIQUE_DIMS = _build_tensor_and_dim_lists(bwd_reads, bwd_writes)

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
            raise ValueError(f"Tile dim '{t}' not found in any tensor shape declaration")
        tile_dims.append(dim_lookup[t])

    def _make_tile_fn(tensor_name, axis_idx, dim_name, tile_size):
        def tile_fn(**tensors):
            dim_size = tensors[tensor_name].shape[axis_idx]
            # Ceiling division: (dim_size + tile_size - 1) // tile_size
            return (dim_size + tile_size - 1) // tile_size

        return staticmethod(tile_fn)

    # Create tile functions with appropriate tile_size for each dimension
    tile_dim_0 = tile[0]
    cls.compute_tiles = _make_tile_fn(*tile_dims[0], tile_dim_0, tile_sizes[tile_dim_0])
    if len(tile) > 1:
        tile_dim_1 = tile[1]
        cls.tiles_n = _make_tile_fn(*tile_dims[1], tile_dim_1, tile_sizes[tile_dim_1])
    if len(tile) > 2:
        tile_dim_2 = tile[2]
        cls.tiles_l = _make_tile_fn(*tile_dims[2], tile_dim_2, tile_sizes[tile_dim_2])

    # Generate pack_config / pack_backward_config (only packs dynamic dims)
    _gen_pack_config(cls, unique_tensors, unique_dims, reads, writes, dynamic_dim_names=dynamic_dim_names)

    # Add schedule() classmethod
    @classmethod
    def schedule(cls, backward=False, **tensors):
        """Create a ScheduledOp from tensor kwargs.

        Args:
            backward: If True, use backward_reads/backward_writes tensor
                      declarations for packing and dim extraction.
            **tensors: Tensor keyword arguments matching the declared reads/writes.
        """
        tiles_m = cls.compute_tiles(**tensors)
        tiles_n = cls.tiles_n(**tensors)
        tiles_l = cls.tiles_l(**tensors)

        # Select forward or backward dim/tensor declarations
        if backward and hasattr(cls, "_BWD_UNIQUE_DIMS"):
            unique_dims = cls._BWD_UNIQUE_DIMS
            unique_tensors = cls._BWD_UNIQUE_TENSORS
            pack_fn = cls.pack_backward_config
        else:
            unique_dims = cls._UNIQUE_DIMS
            unique_tensors = cls._UNIQUE_TENSORS
            pack_fn = cls.pack_config

        # Extract static dim values from actual tensor shapes
        static_dims = {}
        for dim_name, tensor_name, axis_idx in unique_dims:
            if dim_name not in (cls._TILE_DIM_NAMES | cls._DYNAMIC_DIM_OVERRIDES):
                static_dims[dim_name] = int(tensors[tensor_name].shape[axis_idx])

        # Extract tensor dtypes when declared dtype is None (infer from tensor)
        tensor_dtypes = {}
        for name, dtype, dims in unique_tensors:
            if dtype is None:
                # Infer dtype from the actual tensor
                tensor_dtypes[name] = _resolve_dtype(None, tensors[name].dtype)
            # Note: if dtype is specified, we don't store it (use the declared one)

        config_data = pack_fn(**tensors)
        return ScheduledOp(
            op_cls=cls,
            tiles_m=tiles_m,
            tiles_n=tiles_n,
            tiles_l=tiles_l,
            config_data=config_data,
            static_dims=static_dims,
            tensor_dtypes=tensor_dtypes,
            dim_names=cls.DIM_NAMES,
            tile_sizes=cls._TILE_SIZES.copy(),  # Include tile sizes for barrier computation
        )

    cls.schedule = schedule

    # Add gen_init_source classmethod for deferred init generation
    @classmethod
    def gen_init_source(cls, static_dims, backward=False, kernel_config=None, tensor_dtypes=None):
        """Generate init source with static dims baked as compile-time constants.

        Called at kernel compile time when static dim values are known.
        Returns a source string (not a function) to be inlined by compile.py.

        Args:
            static_dims: Dict mapping static dim names to their int values.
            backward: If True, use backward tensor declarations.
            kernel_config: Dict of kernel config parameters (e.g., threads_per_row).
                Also includes tile_size_<DIM> for each tile dimension.
            tensor_dtypes: Dict mapping tensor names to CUTLASS dtypes. Used to
                override declared dtypes (for dtype inference from actual tensors).
        """
        if backward:
            tensors = cls._BWD_UNIQUE_TENSORS
            dims = cls._BWD_UNIQUE_DIMS
        else:
            tensors = cls._UNIQUE_TENSORS
            dims = cls._UNIQUE_DIMS

        # Build kernel config with tile sizes
        full_kernel_config = {}
        if kernel_config:
            full_kernel_config.update(kernel_config)

        # Add tile sizes as tile_size_<DIM> (e.g., tile_size_M = 2)
        for dim_name, size in cls._TILE_SIZES.items():
            full_kernel_config[f"tile_size_{dim_name}"] = size

        return _gen_init_source(
            tensors,
            dims,
            static_dims=static_dims,
            tile_dim_names=cls._TILE_DIM_NAMES,
            dynamic_dim_overrides=cls._DYNAMIC_DIM_OVERRIDES,
            kernel_config=full_kernel_config if full_kernel_config else None,
            tensor_dtypes=tensor_dtypes,
        )

    cls.gen_init_source = gen_init_source


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


class Op:
    """Base class for GPU-executable operations.

    Each operation implements static methods that get inlined into the
    megakernel at compile time. The methods receive raw CuTe DSL types
    (Int32 pointers and indices) so they can execute on the GPU.

    Subclasses declare tensors and tiling via class attributes:
        reads:  Dict of tensor name → (dtype, dim_string)
        writes: Dict of tensor name → (dtype, dim_string)
        tile:   Tuple of dim names that define the tile grid

    Then implement at minimum ``compute_forward``. Load/store default
    to no-ops (suitable for global-memory-only ops). Backward methods
    are optional.

    Method signature (same for all phases):
        def method(smem_base: Int32, config_ptr: Int32,
                   page_ids: tuple[Int32, ...],
                   tile_m: Int32, tile_n: Int32, tile_l: Int32,
                   op_config_ptr: Int64) -> None

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
        if hasattr(cls, "reads") and hasattr(cls, "writes"):
            _process_op_declarations(cls)

    # --- Forward pass ---

    @staticmethod
    def init_forward(
        smem_base: Int32,
        config_ptr: Int32,
        page_ids: tuple,
        tile_m: Int32,
        tile_n: Int32,
        tile_l: Int32,
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
    def load_forward(
        smem_base: Int32,
        config_ptr: Int32,
        page_ids: tuple,
        tile_m: Int32,
        tile_n: Int32,
        tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Load data from global to shared memory."""
        pass

    @staticmethod
    def compute_forward(
        smem_base: Int32,
        config_ptr: Int32,
        page_ids: tuple,
        tile_m: Int32,
        tile_n: Int32,
        tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Perform the forward computation."""
        pass

    @staticmethod
    def store_forward(
        smem_base: Int32,
        config_ptr: Int32,
        page_ids: tuple,
        tile_m: Int32,
        tile_n: Int32,
        tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Store results to global memory."""
        pass

    # --- Backward pass (optional) ---

    @staticmethod
    def init_backward(
        smem_base: Int32,
        config_ptr: Int32,
        page_ids: tuple,
        tile_m: Int32,
        tile_n: Int32,
        tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Shared setup before warp split for the backward pass (optional).

        Same role as init_forward but for gradient computation.
        """
        pass

    @staticmethod
    def load_backward(
        smem_base: Int32,
        config_ptr: Int32,
        page_ids: tuple,
        tile_m: Int32,
        tile_n: Int32,
        tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Load data/gradients from global to shared memory for backward pass."""
        pass

    @staticmethod
    def compute_backward(
        smem_base: Int32,
        config_ptr: Int32,
        page_ids: tuple,
        tile_m: Int32,
        tile_n: Int32,
        tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Compute gradients."""
        pass

    @staticmethod
    def store_backward(
        smem_base: Int32,
        config_ptr: Int32,
        page_ids: tuple,
        tile_m: Int32,
        tile_n: Int32,
        tile_l: Int32,
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
        raise NotImplementedError("Op subclasses must implement compute_tiles()")

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
    static_dims: Dict[str, int] = field(default_factory=dict)  # Compile-time dim values
    tensor_dtypes: Dict[str, Any] = field(default_factory=dict)  # Compile-time tensor dtypes
    tile_sizes: Dict[str, int] = field(default_factory=dict)  # Tile sizes per dim (e.g., {"M": 4})

    def __post_init__(self):
        if self.dim_names is None:
            self.dim_names = getattr(self.op_cls, "DIM_NAMES", {})

    @property
    def total_tiles(self) -> int:
        """Total number of tiles for this operation."""
        return self.tiles_m * self.tiles_n * self.tiles_l

    def tiles_for_axis(self, axis: str) -> int:
        """Get tile count for a given axis ("m", "n", or "l")."""
        return {"m": self.tiles_m, "n": self.tiles_n, "l": self.tiles_l}[axis]

    def tile_size_for_axis(self, axis: str) -> int:
        """Get tile size for a given axis ("m", "n", or "l"). Default is 1."""
        # Find the dim name that maps to this axis, then look up its tile size
        for dim_name, ax in self.dim_names.items():
            if ax == axis and dim_name in self.tile_sizes:
                return self.tile_sizes[dim_name]
        return 1  # Default tile size

    def axis_for_dim(self, dim_name: str) -> Optional[str]:
        """Get the tile axis for a semantic dimension name, or None."""
        return self.dim_names.get(dim_name)


# =============================================================================
# Compile-Time Barrier Formulas
# =============================================================================


@dataclass
class BarrierFormula:
    """Compile-time formula for computing a barrier index from tile coordinates.

    barrier_idx = base + (coeff_m * tile_m) // div_m + (coeff_n * tile_n) // div_n
                       + (coeff_l * tile_l) // div_l

    Used by the megakernel handler to bake barrier wait/signal calls directly
    into each op's handler at JIT compile time. No per-instruction encoding
    needed — the formula coefficients are Python-level constants captured
    by closure.

    Attributes:
        base: Barrier base offset
        coeff_m: Multiplier for tile_m
        coeff_n: Multiplier for tile_n
        coeff_l: Multiplier for tile_l
        div_m: Divisor for tile_m (for tile size ratio handling, default 1)
        div_n: Divisor for tile_n (for tile size ratio handling, default 1)
        div_l: Divisor for tile_l (for tile size ratio handling, default 1)
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
    div_m: int = 1  # Divisor for tile_m (for tile size ratios)
    div_n: int = 1  # Divisor for tile_n (for tile size ratios)
    div_l: int = 1  # Divisor for tile_l (for tile size ratios)
    expected: int = 1
    guard_max: int = NO_GUARD

    def compute_index(self, tile_m: int, tile_n: int, tile_l: int) -> int:
        """Compute barrier index for a given tile (host-side, for testing)."""
        return (
            self.base
            + (self.coeff_m * tile_m) // self.div_m
            + (self.coeff_n * tile_n) // self.div_n
            + (self.coeff_l * tile_l) // self.div_l
        )

    def is_guarded(self, tile_m: int, tile_n: int, tile_l: int) -> bool:
        """Check if the guard passes for a given tile (host-side, for testing)."""
        linear = self.coeff_m * tile_m + self.coeff_n * tile_n + self.coeff_l * tile_l
        return linear < self.guard_max

    @property
    def has_guard(self) -> bool:
        """Whether this formula has an active guard (not NO_GUARD)."""
        return self.guard_max != self.NO_GUARD

    @property
    def has_divisors(self) -> bool:
        """Whether this formula uses divisors (any div > 1)."""
        return self.div_m > 1 or self.div_n > 1 or self.div_l > 1


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
                    raise ValueError(f"Invalid axis '{axis}' for dim '{dim}'. Must be one of {_valid_axes}")
            axes = list(dim_names.values())
            if len(axes) != len(set(axes)):
                raise ValueError(f"dim_names maps multiple dims to the same axis: {dim_names}")

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
        return any(r.op.op_cls.INPUTS or r.op.op_cls.OUTPUTS for r in self._op_records)

    def _resolve_dep_edges(self) -> List[_DepEdge]:
        """Resolve op-level dependency edges for tile scheduling.

        Returns a list of _DepEdge with producer/consumer indices and
        dependency kind. Used by build() to determine tile emission order.
        """
        if not self._has_named_buffers():
            # Legacy linear chain: each op depends on previous, 1:1
            return [
                _DepEdge(producer_idx=i - 1, consumer_idx=i, kind="one_to_one") for i in range(1, len(self._op_records))
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

                edges.append(
                    _DepEdge(
                        producer_idx=prod_idx,
                        consumer_idx=rec.op_idx,
                        kind=kind,
                    )
                )

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

    def _resolve(
        self,
    ) -> Tuple[
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

    def _resolve_legacy_formulas(
        self,
    ) -> Tuple[
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
            signal_formulas = [
                BarrierFormula(
                    base=signal_base,
                    coeff_m=cm,
                    coeff_n=cn,
                    coeff_l=cl,
                )
            ]

            # Wait: previous op's barrier (if exists)
            wait_formulas: List[BarrierFormula] = []
            if i > 0:
                prev_op = self._op_records[i - 1].op
                prev_base = barrier_counter - prev_op.total_tiles
                # Guard for mismatched tile counts
                guard = prev_op.total_tiles if prev_op.total_tiles != op.total_tiles else BarrierFormula.NO_GUARD
                wait_formulas.append(
                    BarrierFormula(
                        base=prev_base,
                        coeff_m=cm,
                        coeff_n=cn,
                        coeff_l=cl,
                        expected=1,
                        guard_max=guard,
                    )
                )

            formulas[i] = (wait_formulas, signal_formulas)
            barrier_counter += op.total_tiles

        return formulas, barrier_counter

    def _resolve_named_formulas(
        self,
    ) -> Tuple[
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
                    raise ValueError(f"Buffer '{buf}' produced by both op {buffer_producers[buf]} and op {rec.op_idx}")
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
                # Also compute divisors for tile size ratio handling
                p_div_m, p_div_n, p_div_l = 1, 1, 1  # Producer divisors
                c_div_m, c_div_n, c_div_l = 1, 1, 1  # Consumer divisors
                expected = 1

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
                    # Same dims (or no dims) - check for tile size differences
                    # Compute per-axis tile size ratios for shared dimensions
                    for dim in shared_dims:
                        p_axis = p_op.dim_names[dim]
                        c_axis = c_op.dim_names[dim]
                        p_ts = p_op.tile_sizes.get(dim, 1)
                        c_ts = c_op.tile_sizes.get(dim, 1)

                        if p_ts > c_ts:
                            # Producer has coarser granularity (larger tile size)
                            # One producer tile → many consumer tiles
                            # Consumer needs divisor to map to producer's barrier
                            ratio = p_ts // c_ts
                            if c_axis == "m":
                                c_div_m = ratio
                            elif c_axis == "n":
                                c_div_n = ratio
                            elif c_axis == "l":
                                c_div_l = ratio
                        elif c_ts > p_ts:
                            # Consumer has coarser granularity (larger tile size)
                            # Many producer tiles → one consumer tile
                            # Producer needs divisor, expected increases
                            ratio = c_ts // p_ts
                            if p_axis == "m":
                                p_div_m = ratio
                            elif p_axis == "n":
                                p_div_n = ratio
                            elif p_axis == "l":
                                p_div_l = ratio
                            expected *= ratio

                    # Use min total tiles for barrier count (the coarser granularity)
                    target_op = c_op
                    num_barriers = min(p_op.total_tiles, c_op.total_tiles)

                # Compute formula coefficients
                p_cm, p_cn, p_cl = _compute_formula_coeffs(
                    p_op,
                    target_op,
                    shared_dims,
                )
                c_cm, c_cn, c_cl = _compute_formula_coeffs(
                    c_op,
                    target_op,
                    shared_dims,
                )

                # Producer signal formula (with divisors for fine-grained producer)
                formulas[prod_idx][1].append(
                    BarrierFormula(
                        base=barrier_counter,
                        coeff_m=p_cm,
                        coeff_n=p_cn,
                        coeff_l=p_cl,
                        div_m=p_div_m,
                        div_n=p_div_n,
                        div_l=p_div_l,
                    )
                )

                # Consumer wait formula (with divisors for fine-grained consumer)
                formulas[rec.op_idx][0].append(
                    BarrierFormula(
                        base=barrier_counter,
                        coeff_m=c_cm,
                        coeff_n=c_cn,
                        coeff_l=c_cl,
                        div_m=c_div_m,
                        div_n=c_div_n,
                        div_l=c_div_l,
                        expected=expected,
                    )
                )

                barrier_counter += num_barriers

        return formulas, barrier_counter

    def build(self) -> List[TileInstruction]:
        """Build an instruction list using level-batched wavefront scheduling.

        Emits tiles in dependency order, batching tiles from the same "level"
        together to naturally spread work across SMs when they fetch
        instructions with strided distribution.

        Strategy:
        1. Emit ALL tiles from source ops (no dependencies) first
        2. Then emit dependent ops using greedy wavefront

        This spreads producer tiles across SMs (better load balancing) while
        still respecting dependencies via barriers. For example:

        One-to-many (Producer 4 tiles → Consumer 16 tiles):
            P0, P1, P2, P3, C0, C1, C2, ...
        With 2 SMs: Block 0 gets P0,P2,C0,... Block 1 gets P1,P3,C1,...

        Contrast with pure greedy which interleaves P0,C0,P1,C1,... putting
        all producers on one SM.
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

        num_ops = len(self._op_records)
        cursors = [0] * num_ops
        total_tiles = sum(rec.op.total_tiles for rec in self._op_records)
        instructions: List[TileInstruction] = []

        # Phase 1: Emit ALL tiles from source ops (ops with no dependencies)
        # This spreads them evenly across SMs via strided fetch
        for rec in self._op_records:
            idx = rec.op_idx
            if idx not in consumer_deps:
                # Source op: emit all tiles now
                for tile in rec.tiles:
                    instructions.append(
                        TileInstruction(
                            op_idx=idx,
                            tile_m=tile[0],
                            tile_n=tile[1],
                            tile_l=tile[2],
                        )
                    )
                cursors[idx] = rec.op.total_tiles

        # Phase 2: Greedy wavefront for remaining ops
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
                    instructions.append(
                        TileInstruction(
                            op_idx=idx,
                            tile_m=tile[0],
                            tile_n=tile[1],
                            tile_l=tile[2],
                        )
                    )
                    cursors[idx] += 1
                    progress = True

            if not progress:
                scheduled = [
                    f"{self._op_records[i].op.op_cls.__name__}: {cursors[i]}/{self._op_records[i].op.total_tiles}"
                    for i in range(num_ops)
                ]
                raise RuntimeError(f"Deadlock in tile scheduling. Cursors: {scheduled}")

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
