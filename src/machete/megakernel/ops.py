# Copyright (c) 2025, Machete Authors
"""
GPU-Executable Operations for Megakernel.

This module defines the operation protocol for GPU execution using CuTe DSL.
Operations are templates that get inlined at compile time, enabling full
compiler optimization with no runtime dispatch overhead.

Subclass ``Op`` and declare tensors, tiling, then implement compute::

    class MyOp(Op):
        reads  = {"x": (Float32, ("M", "D"))}
        writes = {"x": (Float32, ("M", "D"))}
        tile   = ("M", "D")  # dimension names only

        @staticmethod
        def compute(page_ptr, op_config_ptr):
            ...  # tile_M, tile_D, M, D available from init_source

    # Tile sizes passed at schedule time; tile_counts deduced
    ops = MyOp.schedule(x=tensor, tile_sizes={"M": 4})
    kernel = Megakernel(ops)
    kernel.run()
"""

import math
import struct
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import cutlass
import torch


# =============================================================================
# Constants
# =============================================================================

MAX_TILE_DIMS = 5  # Matches TMA's 5D tensor capability


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


# =============================================================================
# Tensor Metadata
# =============================================================================


@dataclass
class TensorMeta:
    """Metadata for a tensor captured at schedule() time.

    Captures shape, strides, dtype, and declared dimensionality for
    validation and stride-based init source generation.
    """

    name: str  # Op-local name ("x", "weight", "q")
    declared_dims: Tuple[str, ...]  # From reads/writes: ("M", "D") or ("M", "H", "D")
    ndim: int  # Expected number of dimensions = len(declared_dims)
    shape: Tuple[int, ...]  # Actual tensor shape
    strides: Tuple[int, ...]  # Actual tensor strides (in elements)
    dtype: Any  # Resolved CUTLASS dtype
    is_contiguous: bool  # Whether tensor is contiguous
    data_ptr: int  # GPU data pointer


def _parse_dims(dims) -> List[str]:
    """Parse dimension specification into a list of dimension names.

    Accepts:
        - Tuple/list of strings: ("M", "H", "D") → ["M", "H", "D"]
        - Comma-separated string (legacy): "M, H, D" → ["M", "H", "D"]
    """
    if isinstance(dims, (tuple, list)):
        return list(dims)
    if isinstance(dims, str):
        return [d.strip() for d in dims.split(",")]
    raise TypeError(f"dims must be tuple, list, or str, got {type(dims)}")


def _parse_tile_spec(tile) -> Tuple[str, ...]:
    """Parse tile specification into dimension names.

    Format: tile = ("M", "D") — a tuple of dimension name strings.

    Returns:
        Tuple of dimension name strings.
    """
    if isinstance(tile, str):
        tile = (tile,)

    for item in tile:
        if not isinstance(item, str):
            raise ValueError(
                f"Invalid tile spec item: {item}. "
                f"Expected dimension name string. "
                f"Tile sizes are now passed at schedule() time via tile_sizes=."
            )

    return tuple(tile)


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


def _gen_pack_config(cls, unique_tensors, unique_dims, reads, writes, dynamic_dim_names=None):
    """Generate and attach pack_config staticmethod.

    Only packs pointers and dynamic dims into the config tensor. Static dims
    are baked into the compiled kernel as compile-time constants.
    """
    num_unique = len(unique_tensors)
    # Only pack dynamic dims
    if dynamic_dim_names is None:
        dynamic_dim_names = {d for d, _, _ in unique_dims}
    dynamic_dims = [(d, t, a) for d, t, a in unique_dims if d in dynamic_dim_names]
    num_dynamic = len(dynamic_dims)
    config_size = 2 * num_unique + num_dynamic

    def pack_config(**tensors):
        config = torch.zeros(config_size, dtype=torch.int32, device=next(iter(tensors.values())).device)
        # Pack pointers
        for i, (name, dtype, dims) in enumerate(unique_tensors):
            t = tensors[name]
            assert t.is_contiguous() or t.stride(-1) == 1, (
                f"{name} must be contiguous or have unit innermost stride"
            )
            lo, hi = struct.unpack("<2i", struct.pack("<Q", t.data_ptr()))
            config[2 * i] = lo
            config[2 * i + 1] = hi
        # Pack dynamic dims only
        offset = 2 * num_unique
        for j, (dim_name, tensor_name, axis_idx) in enumerate(dynamic_dims):
            config[offset + j] = tensors[tensor_name].shape[axis_idx]
        return config

    cls.pack_config = staticmethod(pack_config)


def _process_op_declarations(cls):
    """Process reads/writes/tile declarations on an Op subclass.

    Auto-generates: INPUTS, OUTPUTS,
    DIM_NAMES, pack_config, load/store stubs,
    and schedule() classmethod.

    Stores tensor/dim metadata on the class for deferred init generation
    at compile time (when static dim values are known).

    Tile declaration format (up to 5 dimensions, matching TMA):
        tile = ("M",)             # single dimension
        tile = ("M", "D")         # two dimensions
        tile = ("M", "N", "D")    # three dimensions

    Tile sizes are passed at schedule() time via tile_sizes={"M": 4}.
    Unspecified dims default to full extent (1 tile covering entire dim).
    """
    reads = cls.reads
    writes = cls.writes
    tile = cls.tile

    # Parse tile spec into dim names
    tile = _parse_tile_spec(tile)

    if len(tile) > MAX_TILE_DIMS:
        raise ValueError(f"Tile spec has {len(tile)} dims, max is {MAX_TILE_DIMS}")

    # Build tensor and dim lists
    unique_tensors, unique_dims = _build_tensor_and_dim_lists(reads, writes)

    # --- Classify dims as static or dynamic ---
    # Tile dims are static (their sizes are compile-time constants).
    # Non-tile dims are dynamic by default (packed into config tensor).
    # Ops can override with dynamic_dims = ("S",) to force specific dims dynamic.
    tile_dim_names = set(tile)
    dynamic_dim_overrides = set(getattr(cls, "dynamic_dims", ()))
    non_tile_dims = {d for d, _, _ in unique_dims if d not in tile_dim_names}
    dynamic_dim_names = non_tile_dims | dynamic_dim_overrides
    static_dim_names = [d for d, _, _ in unique_dims if d not in dynamic_dim_names]

    # Store metadata for deferred init generation at compile time
    cls._UNIQUE_TENSORS = unique_tensors
    cls._UNIQUE_DIMS = unique_dims
    cls._TILE_DIM_NAMES = tile_dim_names
    cls._DYNAMIC_DIM_OVERRIDES = dynamic_dim_overrides
    cls._STATIC_DIM_NAMES = static_dim_names
    cls._TILE_DIM_NAMES_ORDERED = tile  # ordered tuple of tile dim names

    # Set INPUTS / OUTPUTS
    cls.INPUTS = list(reads.keys())
    cls.OUTPUTS = list(writes.keys())

    # DIM_NAMES: map tile dim names to axis indices (0..4)
    cls.DIM_NAMES = {tile[i]: i for i in range(len(tile))}

    # Validate tile dims exist in tensor declarations
    dim_lookup = {d: (tname, idx) for d, tname, idx in unique_dims}
    for t in tile:
        if t not in dim_lookup:
            raise ValueError(f"Tile dim '{t}' not found in any tensor shape declaration")

    # --- Process TMA declarations ---
    # tma_loads: set of tensor names to load via TMA G2S
    # tma_stores: set of tensor names to store via TMA S2G
    # tma_reduce_stores: set of tensor names to store via TMA S2G with atomic add
    tma_loads = set(getattr(cls, "tma_loads", set()))
    tma_stores = set(getattr(cls, "tma_stores", set()))
    tma_reduce_stores = set(getattr(cls, "tma_reduce_stores", set()))

    # Validate TMA tensors exist in reads/writes.
    all_read_names = set(reads.keys())
    all_write_names = set(writes.keys())
    for name in tma_loads:
        if name not in all_read_names:
            raise ValueError(f"tma_loads tensor '{name}' not found in reads")
    for name in tma_stores:
        if name not in all_write_names:
            raise ValueError(f"tma_stores tensor '{name}' not found in writes")
    for name in tma_reduce_stores:
        if name not in all_write_names:
            raise ValueError(f"tma_reduce_stores tensor '{name}' not found in writes")

    cls._TMA_LOADS = tma_loads
    cls._TMA_STORES = tma_stores
    cls._TMA_REDUCE_STORES = tma_reduce_stores

    # --- Process peer store declarations ---
    # peer_stores: set of tensor names to send to peer GPUs via TMA S2G
    # peer_reduce_stores: set of tensor names to send via TMA S2G with atomic add
    peer_stores = set(getattr(cls, "peer_stores", set()))
    for name in peer_stores:
        if name not in all_write_names:
            raise ValueError(f"peer_stores tensor '{name}' not found in writes")
    cls._PEER_STORES = peer_stores

    peer_reduce_stores = set(getattr(cls, "peer_reduce_stores", set()))
    for name in peer_reduce_stores:
        if name not in all_write_names:
            raise ValueError(f"peer_reduce_stores tensor '{name}' not found in writes")
    cls._PEER_REDUCE_STORES = peer_reduce_stores

    # Build TMA tile shape info per tensor.
    # For each dim of a TMA tensor: use tile_size if tiled, else full extent.
    # The actual values are filled at schedule() time when static_dims are known.
    tma_tensor_dims = {}  # {tensor_name: list of dim_names}
    tensor_dims_map = {name: dims for name, _, dims in unique_tensors}
    for name in tma_loads | tma_stores | tma_reduce_stores | peer_stores | peer_reduce_stores:
        if name in tensor_dims_map:
            tma_tensor_dims[name] = tensor_dims_map[name]
    cls._TMA_TENSOR_DIMS = tma_tensor_dims

    # Generate pack_config (only packs dynamic dims)
    _gen_pack_config(cls, unique_tensors, unique_dims, reads, writes, dynamic_dim_names=dynamic_dim_names)

    # Add _schedule_single() classmethod (internal — returns one ScheduledOp)
    @classmethod
    def _schedule_single(cls, tile_sizes=None, **tensors):
        """Create a single ScheduledOp from tensor kwargs (internal).

        Args:
            tile_sizes: Dict mapping tile dim names to tile sizes.
                E.g., {"M": 4}. Unspecified dims default to full extent
                (1 tile covering entire dimension).
            **tensors: Tensor keyword arguments matching the declared reads/writes.
        """
        if tile_sizes is None:
            tile_sizes = {}

        unique_dims = cls._UNIQUE_DIMS
        unique_tensors = cls._UNIQUE_TENSORS
        pack_fn = cls.pack_config

        # --- Validate tensor ndim matches declaration ---
        for name, dtype, dims in unique_tensors:
            if name not in tensors:
                continue
            t = tensors[name]
            expected_ndim = len(dims)
            if hasattr(t, "ndim") and t.ndim != expected_ndim:
                raise ValueError(
                    f"{cls.__name__}: tensor '{name}' declared as {expected_ndim}D "
                    f"(dims={dims}) but got {t.ndim}D tensor with shape {tuple(t.shape)}"
                )

        # Extract ALL dim values from actual tensor shapes (compile-time constants).
        # The cache key includes static_dims, so different values trigger recompilation.
        static_dims = {}
        for dim_name, tensor_name, axis_idx in unique_dims:
            val = int(tensors[tensor_name].shape[axis_idx])
            # Validate dim consistency: if we already saw this dim, values must agree
            if dim_name in static_dims and static_dims[dim_name] != val:
                raise ValueError(
                    f"{cls.__name__}: dim '{dim_name}' conflict: "
                    f"expected {static_dims[dim_name]} but tensor '{tensor_name}' "
                    f"axis {axis_idx} has {val}"
                )
            static_dims[dim_name] = val

        # Compute tile_counts from tile_sizes and static_dims
        resolved_tile_sizes = {}
        tile_counts = []
        for dim_name in cls._TILE_DIM_NAMES_ORDERED:
            ts = tile_sizes.get(dim_name)
            dim_size = static_dims[dim_name]
            if ts is None:
                # Full extent: 1 tile covering entire dimension
                tile_counts.append(1)
                resolved_tile_sizes[dim_name] = dim_size
            else:
                # Clamp tile size to dimension extent — TMA descriptors
                # require box size <= tensor extent in each mode.
                clamped_ts = min(ts, dim_size)
                tile_counts.append((dim_size + clamped_ts - 1) // clamped_ts)
                resolved_tile_sizes[dim_name] = clamped_ts
        tile_counts = tuple(tile_counts)

        # Extract tensor dtypes when declared dtype is None (infer from tensor)
        tensor_dtypes = {}
        for name, dtype, dims in unique_tensors:
            if dtype is None:
                # Infer dtype from the actual tensor
                tensor_dtypes[name] = _resolve_dtype(None, tensors[name].dtype)
            # Note: if dtype is specified, we don't store it (use the declared one)

        # Capture tensor data pointers for automatic dependency detection
        # This allows the framework to detect when the same tensor is used
        # as output of one op and input of another, regardless of buffer names
        tensor_ptrs = {}
        tensor_refs = {}
        for name in tensors:
            if hasattr(tensors[name], "data_ptr"):
                tensor_ptrs[name] = tensors[name].data_ptr()
                tensor_refs[name] = tensors[name]

        # Build tensor metadata and strides
        tensor_metas = {}
        tensor_strides = {}
        for name, dtype, dims in unique_tensors:
            if name not in tensors:
                continue
            t = tensors[name]
            resolved_dtype = tensor_dtypes.get(name, dtype)
            tensor_metas[name] = TensorMeta(
                name=name,
                declared_dims=tuple(dims),
                ndim=len(dims),
                shape=tuple(t.shape),
                strides=tuple(t.stride()),
                dtype=resolved_dtype,
                is_contiguous=t.is_contiguous(),
                data_ptr=t.data_ptr(),
            )
            tensor_strides[name] = tuple(t.stride())

        config_data = pack_fn(**tensors)
        return ScheduledOp(
            op_cls=cls,
            tile_counts=tile_counts,
            config_data=config_data,
            static_dims=static_dims,
            tensor_dtypes=tensor_dtypes,
            tensor_ptrs=tensor_ptrs,
            tensor_refs=tensor_refs,
            dim_names=cls.DIM_NAMES,
            tile_sizes=resolved_tile_sizes,
            tensor_metas=tensor_metas,
            tensor_strides=tensor_strides,
        )

    cls._schedule_single = _schedule_single

    # Add schedule() classmethod (public dispatcher — returns list of ScheduledOps)
    @classmethod
    def schedule(cls, tile_sizes=None, **tensors):
        """Schedule op(s) from tensor kwargs. Returns a list of ScheduledOp.

        Checks for schedule_forward override on the subclass. If found,
        delegates to it. Otherwise wraps a single _schedule_single() call
        in a list.
        """
        if hasattr(cls, "schedule_forward"):
            return cls.schedule_forward(tile_sizes=tile_sizes, **tensors)
        return [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]

    cls.schedule = schedule

    # Add gen_tensor_param_names classmethod
    @classmethod
    def gen_tensor_param_names(cls):
        """Get the tensor parameter names for function signature in tensor mode.

        Returns the list of tensor names declared in reads/writes (deduplicated,
        in declaration order). Used to build the phase function signature.
        """
        return [name for name, _, _ in cls._UNIQUE_TENSORS]

    cls.gen_tensor_param_names = gen_tensor_param_names

    # Add gen_tma_param_names classmethod
    @classmethod
    def gen_tma_param_names(cls, phase="load"):
        """Get TMA parameter names for a given phase.

        For each TMA tensor in the given phase (load or store), returns pairs
        of local parameter names: (name_tma, name_tma_gmem).

        Args:
            phase: "load" or "store"

        Returns:
            List of (local_atom_name, local_gmem_name, tensor_name) tuples.
            Example: [("x_tma", "x_tma_gmem", "x")]
        """
        if phase == "load":
            tma_names = cls._TMA_LOADS
        elif phase == "store":
            tma_names = cls._TMA_STORES | cls._TMA_REDUCE_STORES
        else:
            raise ValueError(f"Unknown phase: {phase}")

        result = []
        for name in tma_names:
            result.append((f"{name}_tma", f"{name}_tma_gmem", name))
        return result

    cls.gen_tma_param_names = gen_tma_param_names


# =============================================================================
# Operation Protocol
# =============================================================================

DEFAULT_PAGE_SIZE: int = 49152
"""Default page size in bytes (48KB). Must match MegakernelConfig.page_size."""


class Op:
    """Base class for GPU-executable operations with pipelined execution.

    Methods are ``@cute.jit`` instance methods on a class in a real .py file.
    The framework creates an instance at compile time with config stored as
    ``self`` attributes. Methods access config via ``self.M``,
    ``self.tile_size_M``, etc. Tensors and tile indices are explicit method
    parameters.

    Declare tensors and tiling via class attributes::

        reads:  Dict of tensor name → (dtype, dim_string)
        writes: Dict of tensor name → (dtype, dim_string)
        tile:   Tuple of dim names that define the tile grid

    Execution Model:
        The megakernel uses double-buffered pipelining. Ops implement three phases:
        - load(): Load data from global to shared memory
        - compute(): Process data (main computation logic)
        - store(): Write results to global memory

        When dependencies allow, load[N+1] overlaps with compute[N].
        When dependencies block, execution gracefully degrades to sequential.

        For ops that don't use shared memory staging, put all logic in compute()
        and leave load()/store() as pass.

    Example:
        class MyOp(Op):
            reads = {"x": (Float32, ("M", "D"))}
            writes = {"y": (Float32, ("M", "D"))}
            tile = ("M", "D")

            @cute.jit
            def compute(self, page_ptr, tile_M, tile_D, x, y):
                tidx = cute.arch.thread_idx()[0]
                for i in range(tidx, self.D, self.threads_per_row):
                    y[tile_M * self.D + i] = x[tile_M * self.D + i] * 2.0

    Class attributes:
        INPUTS: Named global memory buffers this op reads from
        OUTPUTS: Named global memory buffers this op produces
    """

    # Named buffer declarations for automatic dependency inference.
    # Each string names a global memory buffer. The builder matches
    # producer OUTPUTS to consumer INPUTS to build the dependency DAG.
    INPUTS: ClassVar[List[str]] = []
    OUTPUTS: ClassVar[List[str]] = []

    def __init__(self, **config):
        """Initialize Op instance with compile-time configuration.

        For class-based Ops, the framework calls this at compile time
        with all config as keyword arguments. Config includes dim values
        (M, D), tile sizes (tile_size_M), thread config (threads_per_row),
        tensor dtypes (x_dtype) and strides (x_stride_M).

        Subclasses can override for explicit parameter validation.
        """
        for key, value in config.items():
            setattr(self, key, value)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "reads") and hasattr(cls, "writes"):
            _process_op_declarations(cls)

    # =========================================================================
    # Pipelined Execution Interface
    # =========================================================================

    def load(self, page_ptr) -> None:
        """Load data from global memory to shared memory page.

        Called by DMA warp thread 0. For TMA-based ops (async_load = True),
        add ``work_mbar`` parameter — issue
        mbarrier_arrive_expect_tx(work_mbar, nbytes) + cute.copy(..., mbar_ptr=...)
        and return immediately (async).

        Default: no-op (for ops that access global memory directly in compute).
        """
        pass

    def compute(self, page_ptr) -> None:
        """Main computation for one tile.

        This is where the op's core logic goes. Can access:
        - Shared memory via page_ptr (if load staged data there)
        - Global memory directly via tensor parameters
        - Tile indices via tile_M, tile_D, etc. parameters
        - Config via self.M, self.D, self.threads_per_row, etc.

        Called by MMA warps after DMA warp completes the load phase.
        """
        pass

    def store(self, page_ptr) -> None:
        """Store results from shared memory page to global memory.

        Called by DMA warp thread 0. For TMA-based ops, issue
        cute.copy(tma_store_atom, ...) here (S2G does not need mbarrier).

        Default: no-op (for ops that write directly to global memory in compute).
        """
        pass

    # =========================================================================
    # Multi-GPU Communication Interface
    # =========================================================================

    def communicate(self, page_ptr) -> None:
        """Send results to peer GPU buffers after local store.

        Called by store warp after store() completes. For TMA-based peer
        communication, issue cute.copy(peer_tma_atom, ...) here (S2G to
        peer memory via NVLink).

        Receives peer TMA descriptors as keyword arguments when peer_stores
        is declared on the op class.

        Default: no-op (for ops that don't need multi-GPU communication).
        """
        pass



def build_op_config(
    op: "ScheduledOp",
    kernel_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build config dict for creating a class-based Op instance.

    The config dict contains all compile-time constants that the instance
    stores as ``self`` attributes: dim values, tile sizes, thread config,
    tensor dtypes, and tensor strides.

    Args:
        op: Scheduled operation with static_dims, tile_sizes, tensor info.
        kernel_config: Kernel-level config (threads_per_row, etc.).
    """
    config: Dict[str, Any] = {}

    # Kernel config (threads_per_row, etc.)
    if kernel_config:
        config.update(kernel_config)

    # Tile sizes (tile_size_M, tile_size_D, ...)
    if op.tile_sizes:
        for dim_name, size in op.tile_sizes.items():
            config[f"tile_size_{dim_name}"] = size

    # Dim values (M, D, S, H, ...)
    if op.static_dims:
        config.update(op.static_dims)

    # Tensor dtypes (x_dtype, weight_dtype, dout_dtype, ...)
    if hasattr(op.op_cls, "_UNIQUE_TENSORS"):
        for name, dtype, dims in op.op_cls._UNIQUE_TENSORS:
            key = f"{name}_dtype"
            if key not in config:
                resolved = op.tensor_dtypes.get(name, dtype) if op.tensor_dtypes else dtype
                if resolved is not None:
                    config[key] = resolved

    # Tensor strides (x_stride_M, x_stride_D, ...)
    if op.tensor_strides and hasattr(op.op_cls, "_UNIQUE_TENSORS"):
        for name, dtype, dims in op.op_cls._UNIQUE_TENSORS:
            if name in op.tensor_strides:
                for i, dim_name in enumerate(dims):
                    config[f"{name}_stride_{dim_name}"] = op.tensor_strides[name][i]

    return config


# =============================================================================
# Scheduled Operation
# =============================================================================


@dataclass
class ScheduledOp:
    """An operation scheduled for execution with tile requirements.

    Pairs an operation class with the number of tiles to process.

    Attributes:
        op_cls: Operation class (must be Op subclass)
        tile_counts: Number of tiles per dimension (up to MAX_TILE_DIMS).
        params: Operation-specific parameters
        dim_names: Maps semantic dimension names to tile axis indices (0..4).
            Example: {"M": 0, "D": 1} means tile_0 indexes M
            and tile_1 indexes D. Used by the builder to compute tile
            mappings between ops with different grid shapes.
    """

    op_cls: Type[Op]
    tile_counts: Tuple[int, ...] = (1,)  # Variable length, up to MAX_TILE_DIMS
    params: Dict[str, Any] = field(default_factory=dict)
    dim_names: Dict[str, int] = None  # Maps dim name -> axis index (0..4)
    config_data: Any = None  # Optional torch.Tensor with per-op config in global memory
    static_dims: Dict[str, int] = field(default_factory=dict)  # Compile-time dim values
    tensor_dtypes: Dict[str, Any] = field(default_factory=dict)  # Compile-time tensor dtypes
    tile_sizes: Dict[str, int] = field(default_factory=dict)  # Tile sizes per dim (e.g., {"M": 4})
    tensor_ptrs: Dict[str, int] = field(default_factory=dict)  # Tensor data pointers for dependency matching
    tensor_refs: Dict[str, Any] = field(default_factory=dict)  # {name: torch.Tensor} for tensor param mode
    tensor_metas: Dict[str, TensorMeta] = field(default_factory=dict)  # Per-tensor metadata (shape, strides, ndim)
    tensor_strides: Dict[str, Tuple[int, ...]] = field(default_factory=dict)  # {name: strides} for stride init source
    dim_aliases: Dict[str, str] = field(default_factory=dict)  # Maps dim name → canonical for barrier matching

    def __post_init__(self):
        if self.dim_names is None:
            self.dim_names = getattr(self.op_cls, "DIM_NAMES", {})

    @property
    def ndims(self) -> int:
        """Number of tile dimensions."""
        return len(self.tile_counts)

    @property
    def total_tiles(self) -> int:
        """Total number of tiles for this operation."""
        return math.prod(self.tile_counts)

    def tiles_for_axis(self, axis: int) -> int:
        """Get tile count for a given axis index (0..4)."""
        if axis < len(self.tile_counts):
            return self.tile_counts[axis]
        return 1

    def tile_size_for_axis(self, axis: int) -> int:
        """Get tile size for a given axis index. Default is 1."""
        for dim_name, ax in self.dim_names.items():
            if ax == axis and dim_name in self.tile_sizes:
                return self.tile_sizes[dim_name]
        return 1

    def axis_for_dim(self, dim_name: str) -> Optional[int]:
        """Get the tile axis index for a semantic dimension name, or None."""
        return self.dim_names.get(dim_name)

__all__ = [
    # Constants
    "MAX_TILE_DIMS",
    "DEFAULT_PAGE_SIZE",
    "TORCH_TO_CUTLASS_DTYPE",
    # Metadata
    "TensorMeta",
    # Protocol
    "Op",
    "build_op_config",
    # Scheduling
    "ScheduledOp",
]
