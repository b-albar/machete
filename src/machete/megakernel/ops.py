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
    ops = [MyOp.schedule(x=tensor, tile_sizes={"M": 4})]
    kernel = Megakernel(ops)
    kernel.run()
"""

import math
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Type, Union

import cutlass
import torch
from cutlass import Int32, Int64


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

    name: str                       # Op-local name ("x", "weight", "q")
    declared_dims: Tuple[str, ...]  # From reads/writes: ("M", "D") or ("M", "H", "D")
    ndim: int                       # Expected number of dimensions = len(declared_dims)
    shape: Tuple[int, ...]          # Actual tensor shape
    strides: Tuple[int, ...]        # Actual tensor strides (in elements)
    dtype: Any                      # Resolved CUTLASS dtype
    is_contiguous: bool             # Whether tensor is contiguous
    data_ptr: int                   # GPU data pointer


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


def _gen_init_source(
    unique_tensors,
    unique_dims,
    tensor_param_map=None,
    static_dims=None,
    kernel_config=None,
    tensor_dtypes=None,
    tensor_strides=None,
    warp_specialized=False,
    dma_warp_mode=False,
    dim_names=None,
):
    """Generate init source for tensor parameter mode (runtime tensor params).

    Tensors are passed as function parameters (e.g., t0, t1, t2) and aliased
    to the op's local names (e.g., x, weight, y). ALL dimensions are emitted
    as compile-time Python int literals. Strides are emitted as compile-time
    constants (e.g., x_stride_M, x_stride_D) when tensor_strides is provided.
    Tile indices are mapped from positional (tile_0, tile_1, ...) to named
    variables (tile_M, tile_D, ...) based on dim_names.

    Args:
        unique_tensors: List of (name, dtype, dims) tuples.
        unique_dims: List of (dim_name, tensor_name, axis_idx) tuples.
        tensor_param_map: Dict mapping local tensor name to canonical parameter
            name. E.g., ``{'x': 't0', 'weight': 't1', 'y': 't2'}``.
        static_dims: Dict mapping ALL dim names to their int values.
        kernel_config: Dict of kernel configuration parameters.
        tensor_dtypes: Dict mapping tensor names to CUTLASS dtypes.
        tensor_strides: Dict mapping tensor names to stride tuples.
            E.g., ``{'x': (256, 1), 'weight': (1,)}``.
        warp_specialized: If True, set num_threads to threads_per_row (compute
            threads only, excluding the DMA warp) instead of block_dim.
        dma_warp_mode: If True, remap tidx for DMA warp (32 threads).
            tidx is offset so DMA warp threads become 0-31, num_threads=32.
        dim_names: Dict mapping dim name -> axis index (e.g., {"M": 0, "D": 1}).
            Used to generate named tile index aliases (tile_M = tile_0, etc.).
    """
    lines = []

    # Kernel config params first (threads_per_row, tile_size_M, etc.)
    # Must come before tidx/num_threads since warp_specialized mode references threads_per_row
    if kernel_config:
        for name, value in kernel_config.items():
            lines.append(f"{name} = {value}")

    if dma_warp_mode:
        # DMA warp: remap tidx to 0-31 (subtract compute thread count)
        compute_threads = kernel_config.get("threads_per_row", 0) if kernel_config else 0
        lines.append(f"tidx = cute.arch.thread_idx()[0] - Int32({compute_threads})")
        lines.append("num_threads = 32")
    elif warp_specialized:
        lines.append("tidx = cute.arch.thread_idx()[0]")
        # In warp-specialized mode, num_threads = compute threads only
        lines.append("num_threads = threads_per_row")
    else:
        lines.append("tidx = cute.arch.thread_idx()[0]")
        lines.append("num_threads = cute.arch.block_dim()[0]")

    # Map positional tile indices to named variables (tile_M = tile_0, etc.)
    if dim_names:
        for dim_name, axis_idx in dim_names.items():
            lines.append(f"tile_{dim_name} = tile_{axis_idx}")

    # Alias tensor params to local names. Tensors are passed as N-D
    # torch.Tensor (auto-converted by TensorAdapter). Ops that need flat 1D
    # access should create their own flat view via:
    #   x = cute.make_tensor(x.iterator, cute.make_layout(M * D))
    if tensor_param_map:
        for local_name, canonical_name in tensor_param_map.items():
            lines.append(f"{local_name} = {canonical_name}")

    # ALL dims as compile-time Python int literals
    if static_dims:
        for dim_name, _, _ in unique_dims:
            if dim_name in static_dims:
                lines.append(f"{dim_name} = {static_dims[dim_name]}")

    # Dtype constants for each tensor
    for name, dtype, dims in unique_tensors:
        if tensor_dtypes and name in tensor_dtypes:
            dtype = tensor_dtypes[name]
        dtype_name = dtype.__name__ if hasattr(dtype, "__name__") else str(dtype)
        lines.append(f"{name}_dtype = {dtype_name}")

    # Stride constants for each tensor (e.g., x_stride_M = 256, x_stride_D = 1)
    if tensor_strides:
        for name, dtype, dims in unique_tensors:
            if name in tensor_strides:
                strides = tensor_strides[name]
                for i, dim_name in enumerate(dims):
                    lines.append(f"{name}_stride_{dim_name} = {strides[i]}")

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

    # All dims are dynamic if no classification provided
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
    DIM_NAMES, pack_config, pack_backward_config, load/store stubs,
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

    # DIM_NAMES: map tile dim names to axis indices (0..4)
    cls.DIM_NAMES = {tile[i]: i for i in range(len(tile))}

    # Validate tile dims exist in tensor declarations
    dim_lookup = {d: (tname, idx) for d, tname, idx in unique_dims}
    for t in tile:
        if t not in dim_lookup:
            raise ValueError(f"Tile dim '{t}' not found in any tensor shape declaration")

    # Generate pack_config / pack_backward_config (only packs dynamic dims)
    _gen_pack_config(cls, unique_tensors, unique_dims, reads, writes, dynamic_dim_names=dynamic_dim_names)

    # Add schedule() classmethod
    @classmethod
    def schedule(cls, backward=False, tile_sizes=None, **tensors):
        """Create a ScheduledOp from tensor kwargs.

        Args:
            backward: If True, use backward_reads/backward_writes tensor
                      declarations for packing and dim extraction.
            tile_sizes: Dict mapping tile dim names to tile sizes.
                E.g., {"M": 4}. Unspecified dims default to full extent
                (1 tile covering entire dimension).
            **tensors: Tensor keyword arguments matching the declared reads/writes.
        """
        if tile_sizes is None:
            tile_sizes = {}

        # Select forward or backward dim/tensor declarations
        if backward and hasattr(cls, "_BWD_UNIQUE_DIMS"):
            unique_dims = cls._BWD_UNIQUE_DIMS
            unique_tensors = cls._BWD_UNIQUE_TENSORS
            pack_fn = cls.pack_backward_config
        else:
            unique_dims = cls._UNIQUE_DIMS
            unique_tensors = cls._UNIQUE_TENSORS
            pack_fn = cls.pack_config

        # --- Validate tensor ndim matches declaration ---
        for name, dtype, dims in unique_tensors:
            if name not in tensors:
                continue
            t = tensors[name]
            expected_ndim = len(dims)
            if hasattr(t, 'ndim') and t.ndim != expected_ndim:
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
                tile_counts.append((dim_size + ts - 1) // ts)
                resolved_tile_sizes[dim_name] = ts
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
            if hasattr(tensors[name], 'data_ptr'):
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

    cls.schedule = schedule

    # Add gen_init_source classmethod for deferred init generation
    @classmethod
    def gen_init_source(cls, static_dims, tensor_param_map=None,
                                    backward=False, kernel_config=None, tensor_dtypes=None,
                                    tensor_strides=None,
                                    warp_specialized=False, dma_warp_mode=False,
                                    tile_sizes=None):
        """Generate init source for tensor parameter mode.

        Tensors are passed as function parameters and aliased to local names.
        ALL dimensions are emitted as compile-time constants. Strides are
        emitted as compile-time constants when tensor_strides is provided.

        Args:
            static_dims: Dict mapping ALL dim names to their int values.
            tensor_param_map: Dict mapping local tensor name to canonical
                param name. E.g., ``{'x': 't0', 'weight': 't1', 'y': 't2'}``.
            backward: If True, use backward tensor declarations.
            kernel_config: Dict of kernel config parameters.
            tensor_dtypes: Dict mapping tensor names to CUTLASS dtypes.
            tensor_strides: Dict mapping tensor names to stride tuples.
                E.g., ``{'x': (256, 1), 'weight': (1,)}``.
            warp_specialized: If True, set num_threads to compute threads only.
            dma_warp_mode: If True, remap tidx for DMA warp (32 threads).
            tile_sizes: Dict mapping tile dim names to resolved tile sizes.
                E.g., ``{'M': 4, 'D': 256}``. All values must be ints
                (no None — resolve full-extent before calling).
        """
        if backward:
            tensors = cls._BWD_UNIQUE_TENSORS
            dims = cls._BWD_UNIQUE_DIMS
        else:
            tensors = cls._UNIQUE_TENSORS
            dims = cls._UNIQUE_DIMS

        full_kernel_config = {}
        if kernel_config:
            full_kernel_config.update(kernel_config)
        # Emit tile_size_X as compile-time constants
        if tile_sizes:
            for dim_name, size in tile_sizes.items():
                full_kernel_config[f"tile_size_{dim_name}"] = size

        return _gen_init_source(
            tensors,
            dims,
            tensor_param_map=tensor_param_map,
            static_dims=static_dims,
            kernel_config=full_kernel_config if full_kernel_config else None,
            tensor_dtypes=tensor_dtypes,
            tensor_strides=tensor_strides,
            warp_specialized=warp_specialized,
            dma_warp_mode=dma_warp_mode,
            dim_names=cls.DIM_NAMES,
        )

    cls.gen_init_source = gen_init_source

    # Add gen_tensor_param_names classmethod
    @classmethod
    def gen_tensor_param_names(cls, backward=False):
        """Get the tensor parameter names for function signature in tensor mode.

        Returns the list of tensor names declared in reads/writes (deduplicated,
        in declaration order). Used to build the phase function signature.

        Args:
            backward: If True, use backward tensor declarations.
        """
        tensors = cls._BWD_UNIQUE_TENSORS if backward else cls._UNIQUE_TENSORS
        return [name for name, _, _ in tensors]

    cls.gen_tensor_param_names = gen_tensor_param_names


# =============================================================================
# Operation Protocol
# =============================================================================


class Op:
    """Base class for GPU-executable operations with pipelined execution.

    Each operation implements static methods that get inlined into the
    megakernel at compile time. The methods receive raw CuTe DSL types
    (Int32 pointers and indices) so they can execute on the GPU.

    Subclasses declare tensors and tiling via class attributes:
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

    Example (direct global memory, typical pattern):
        class MyOp(Op):
            reads = {"x": (Float32, ("M", "D"))}
            writes = {"y": (Float32, ("M", "D"))}
            tile = ("M", "D")

            @staticmethod
            def compute(page_ptr, op_config_ptr):
                # tile_M, tile_D, x, y, M, D available from init_source
                for i in range(tidx, D, num_threads):
                    y[tile_M * D + i] = x[tile_M * D + i] * 2.0

    Class attributes:
        NUM_INPUT_PAGES: Number of input shared memory pages needed
        NUM_OUTPUT_PAGES: Number of output shared memory pages needed
        INPUTS: Named global memory buffers this op reads from
        OUTPUTS: Named global memory buffers this op produces
    """

    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0

    # Named buffer declarations for automatic dependency inference.
    # Each string names a global memory buffer. The builder matches
    # producer OUTPUTS to consumer INPUTS to build the dependency DAG.
    INPUTS: ClassVar[List[str]] = []
    OUTPUTS: ClassVar[List[str]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "reads") and hasattr(cls, "writes"):
            _process_op_declarations(cls)

    # =========================================================================
    # Pipelined Execution Interface
    # =========================================================================
    #
    # Op method signatures: (page_ptr, op_config_ptr)
    # Tile indices (tile_M, tile_D, etc.) and dimensions (M, D, etc.)
    # are injected via init_source as named variables.

    @staticmethod
    def load(
        page_ptr: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Load data from global memory to shared memory page.

        Called by DMA warp thread 0. For TMA-based ops (async_load = True),
        receives an additional work_mbar parameter — issue
        mbarrier_arrive_expect_tx(work_mbar, nbytes) + cute.copy(..., mbar_ptr=...)
        and return immediately (async).

        Default: no-op (for ops that access global memory directly in compute).
        """
        pass

    @staticmethod
    def compute(
        page_ptr: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Main computation for one tile.

        This is where the op's core logic goes. Can access:
        - Shared memory via page_ptr (if load staged data there)
        - Global memory directly via tensor pointers from init_source
        - Tile indices via tile_M, tile_D, etc. from init_source

        Called by MMA warps after DMA warp completes the load phase.
        """
        pass

    @staticmethod
    def store(
        page_ptr: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Store results from shared memory page to global memory.

        Called by DMA warp thread 0. For TMA-based ops, issue
        cute.copy(tma_store_atom, ...) here (S2G does not need mbarrier).

        Default: no-op (for ops that write directly to global memory in compute).
        """
        pass

    # =========================================================================
    # Backward Pass Interface
    # =========================================================================

    @staticmethod
    def backward_load(
        page_ptr: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Load data for backward pass.

        Default: no-op (for ops that don't use shared memory staging).
        """
        pass

    @staticmethod
    def backward_compute(
        page_ptr: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Backward computation for one tile.

        Default: no-op.
        """
        pass

    @staticmethod
    def backward_store(
        page_ptr: Int32,
        op_config_ptr: Int64,
    ) -> None:
        """Store backward results from shared memory to global memory.

        Default: no-op.
        """
        pass

    # =========================================================================
    # Host-side tiling (used by autograd and scheduling)
    # =========================================================================

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


# =============================================================================
# Tensor Registry (Deduplication for Tensor Parameter Mode)
# =============================================================================


@dataclass
class TensorRegistry:
    """Deduplicates tensors across ops by data_ptr() for tensor parameter mode.

    When multiple ops share the same underlying tensor (e.g., RMSNorm.y and
    Rope.q point to the same GPU buffer), this registry assigns a single
    canonical parameter name (t0, t1, ...) to avoid passing duplicate tensors.

    Usage:
        registry = TensorRegistry.from_ops(ops)
        # registry.canonical_names → ['t0', 't1', 't2', 't3', 't4']
        # registry.get_op_tensor_args(0, RMSNormOp) → ['t0', 't1', 't2']
        # registry.get_op_tensor_args(1, RopeOp) → ['t2', 't3', 't4']
    """

    # List of (canonical_name, torch.Tensor, cutlass_dtype)
    tensors: List[Tuple[str, Any, Any]]
    # Per-op mappings: {op_idx: {local_name: canonical_name}}
    op_mappings: Dict[int, Dict[str, str]]
    # Reverse mapping: canonical_name -> index in tensors list
    name_to_idx: Dict[str, int]

    @classmethod
    def from_ops(cls, ops: List[ScheduledOp], backward: bool = False) -> "TensorRegistry":
        """Build a TensorRegistry from a list of ScheduledOps.

        Iterates through each op's tensor_refs in declaration order,
        deduplicating by data_ptr(). Tensors with the same GPU address
        get the same canonical name.

        Args:
            ops: List of scheduled operations with tensor_refs populated.
            backward: If True, use backward tensor declarations for ordering.
        """
        ptr_to_canonical: Dict[int, str] = {}
        tensors: List[Tuple[str, Any, Any]] = []
        op_mappings: Dict[int, Dict[str, str]] = {}
        name_to_idx: Dict[str, int] = {}

        for i, op in enumerate(ops):
            mapping: Dict[str, str] = {}
            # Skip ops that don't have tensor declarations (e.g., simple test ops
            # like StampOp/TensorScaleOp that don't use the @op decorator)
            if not hasattr(op.op_cls, '_UNIQUE_TENSORS'):
                op_mappings[i] = mapping
                continue

            # Use the op's unique_tensors for consistent ordering
            unique_tensors = (
                op.op_cls._BWD_UNIQUE_TENSORS if backward else op.op_cls._UNIQUE_TENSORS
            )

            for name, dtype, dims in unique_tensors:
                if name not in op.tensor_refs:
                    continue
                tensor = op.tensor_refs[name]
                ptr = tensor.data_ptr()

                if ptr not in ptr_to_canonical:
                    canonical = f"t{len(tensors)}"
                    ptr_to_canonical[ptr] = canonical

                    # Resolve dtype (None means infer from tensor)
                    resolved_dtype = dtype
                    if resolved_dtype is None:
                        resolved_dtype = TORCH_TO_CUTLASS_DTYPE.get(tensor.dtype)

                    name_to_idx[canonical] = len(tensors)
                    tensors.append((canonical, tensor, resolved_dtype))

                mapping[name] = ptr_to_canonical[ptr]
            op_mappings[i] = mapping

        return cls(tensors=tensors, op_mappings=op_mappings, name_to_idx=name_to_idx)

    @property
    def canonical_names(self) -> List[str]:
        """List of canonical tensor parameter names in order."""
        return [name for name, _, _ in self.tensors]

    @property
    def num_tensors(self) -> int:
        """Number of unique tensors."""
        return len(self.tensors)

    def get_op_tensor_args(self, op_idx: int, op_cls, backward: bool = False) -> List[str]:
        """Get ordered canonical tensor names for an op's function call.

        Returns canonical names in the same order as the op's tensor
        declarations (reads then writes, deduplicated). This order matches
        the tensor parameter positions in the compiled phase function.

        Args:
            op_idx: Index of the op in the ops list.
            op_cls: The op class (for accessing _UNIQUE_TENSORS).
            backward: If True, use backward tensor declarations.
        """
        if not hasattr(op_cls, '_UNIQUE_TENSORS'):
            return []
        unique_tensors = op_cls._BWD_UNIQUE_TENSORS if backward else op_cls._UNIQUE_TENSORS
        mapping = self.op_mappings[op_idx]
        return [mapping[name] for name, _, _ in unique_tensors if name in mapping]


# =============================================================================
# Cross-Op Compatibility Validation
# =============================================================================


def validate_op_compatibility(ops: List[ScheduledOp], registry: "TensorRegistry") -> None:
    """Validate shared tensors across fused ops have compatible shapes.

    When two ops share the same underlying tensor (same data_ptr), checks that:
    1. Total element count matches (product of shapes).
    2. Shared dimension names have matching values in static_dims.

    This allows reshapes like (M, D) → (M, H, D) as long as total elements agree
    and any dimension names in common (e.g. M) have the same value.

    Args:
        ops: List of scheduled operations.
        registry: TensorRegistry with deduplication info.

    Raises:
        ValueError: If shared tensors have incompatible shapes.
    """
    # Build reverse map: data_ptr → list of (op_idx, tensor_name, TensorMeta)
    ptr_to_uses: Dict[int, List[Tuple[int, str, TensorMeta]]] = {}
    for i, op in enumerate(ops):
        for name, meta in op.tensor_metas.items():
            ptr_to_uses.setdefault(meta.data_ptr, []).append((i, name, meta))

    # Check each shared tensor (data_ptr with multiple users)
    for ptr, uses in ptr_to_uses.items():
        if len(uses) < 2:
            continue

        # Check total element count compatibility
        ref_idx, ref_name, ref_meta = uses[0]
        ref_numel = math.prod(ref_meta.shape)
        ref_op_name = ops[ref_idx].op_cls.__name__

        for other_idx, other_name, other_meta in uses[1:]:
            other_numel = math.prod(other_meta.shape)
            other_op_name = ops[other_idx].op_cls.__name__

            if ref_numel != other_numel:
                raise ValueError(
                    f"Shared tensor incompatibility: "
                    f"{ref_op_name}.{ref_name} has shape {ref_meta.shape} "
                    f"({ref_numel} elements) but "
                    f"{other_op_name}.{other_name} has shape {other_meta.shape} "
                    f"({other_numel} elements)"
                )

        # Note: We intentionally do NOT check per-dim name matching on shared
        # tensors. Different ops can reshape the same buffer (e.g., RMSNorm
        # outputs (M, D) and RoPE reads (M, H, D) from the same storage),
        # and dim names like "D" can legitimately mean different things.
        # The total element count check above is the correct constraint.


# =============================================================================
# Compile-Time Barrier Formulas
# =============================================================================


@dataclass
class BarrierFormula:
    """Compile-time formula for computing a barrier index from tile coordinates.

    barrier_idx = base + sum((coeffs[i] * tile_i) // divs[i] for i in range(ndims))

    Used by the megakernel handler to bake barrier wait/signal calls directly
    into each op's handler at JIT compile time. No per-instruction encoding
    needed — the formula coefficients are Python-level constants captured
    by closure.

    Attributes:
        base: Barrier base offset
        coeffs: Per-axis multipliers for tile indices (up to MAX_TILE_DIMS)
        divs: Per-axis divisors for tile indices (for tile size ratios, default 1)
        expected: For wait deps: how many signals to wait for (default 1)
        guard_max: Only execute when the computed linear index is less than
            this value. Defaults to NO_GUARD (always passes).
    """

    # Sentinel: guard_max value that always passes (larger than any tile count)
    NO_GUARD: ClassVar[int] = 2**30

    base: int
    coeffs: Tuple[int, ...] = (0,) * MAX_TILE_DIMS
    divs: Tuple[int, ...] = (1,) * MAX_TILE_DIMS
    expected: int = 1
    guard_max: int = NO_GUARD

    def compute_index(self, tiles: Tuple[int, ...]) -> int:
        """Compute barrier index for a given tile (host-side, for testing)."""
        padded = tuple(tiles) + (0,) * (MAX_TILE_DIMS - len(tiles))
        result = self.base
        for i in range(MAX_TILE_DIMS):
            result += (self.coeffs[i] * padded[i]) // self.divs[i]
        return result

    def is_guarded(self, tiles: Tuple[int, ...]) -> bool:
        """Check if the guard passes for a given tile (host-side, for testing)."""
        padded = tuple(tiles) + (0,) * (MAX_TILE_DIMS - len(tiles))
        linear = sum(self.coeffs[i] * padded[i] for i in range(MAX_TILE_DIMS))
        return linear < self.guard_max

    @property
    def has_guard(self) -> bool:
        """Whether this formula has an active guard (not NO_GUARD)."""
        return self.guard_max != self.NO_GUARD



# =============================================================================
# Instruction Stream (Lightweight — barriers baked into handlers)
# =============================================================================

INSTRUCTION_WORDS = 1 + MAX_TILE_DIMS  # op_idx + up to 5 tile indices = 6


@dataclass
class TileInstruction:
    """A single tile work instruction for the persistent megakernel.

    Lightweight encoding in global memory (6 x int32):
    [0]  op_idx: Which operation (indexes into op list), or -1 for end marker
    [1..5]  tile indices: Up to 5 tile dimension indices

    Barrier wait/signal logic is baked into op handlers at compile time
    via BarrierFormula, not encoded in the instruction stream.
    """

    op_idx: int
    tiles: Tuple[int, ...]  # Up to MAX_TILE_DIMS tile indices

    # Sentinel for end of stream
    END_MARKER: int = -1

    def pack(self) -> List[int]:
        """Pack into list of int32 (padded to INSTRUCTION_WORDS)."""
        padded = list(self.tiles) + [0] * (MAX_TILE_DIMS - len(self.tiles))
        return [self.op_idx] + padded

    @classmethod
    def end_instruction(cls) -> "TileInstruction":
        """Create end-of-stream marker."""
        return cls(op_idx=cls.END_MARKER, tiles=(0,) * MAX_TILE_DIMS)


# =============================================================================
# Dependency Resolution Helpers
# =============================================================================


def _linear_strides(tile_counts: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute row-major linear strides from tile counts, padded to MAX_TILE_DIMS.

    For tile_counts = (4, 8, 2): strides = (16, 2, 1, 0, 0)
    stride[i] = product of tile_counts[i+1:]
    """
    ndims = len(tile_counts)
    strides = [0] * MAX_TILE_DIMS
    stride = 1
    for i in range(ndims - 1, -1, -1):
        strides[i] = stride
        stride *= tile_counts[i]
    return tuple(strides)


@dataclass
class _OpRecord:
    """Internal record for an op added to the builder."""

    op_idx: int
    op: ScheduledOp
    # Flat list of tile coordinate tuples: [(i0, i1, ...), ...]
    tiles: List[Tuple[int, ...]]


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
) -> Tuple[int, ...]:
    """Compute coefficients for source computing target's barrier index.

    Maps shared dimension values from source tile coordinates to target's
    linear index. The target's strides are computed from tile_counts
    (row-major: stride[i] = product of tile_counts[i+1:]).

    For each shared dim, find which axis it maps to on each side, then
    accumulate the target stride onto the source axis coefficient.

    Returns:
        Tuple of MAX_TILE_DIMS coefficients.
    """
    s_dims = source_op.dim_names
    t_dims = target_op.dim_names

    # Compute target strides from tile_counts (row-major linearization)
    t_counts = target_op.tile_counts
    t_strides = {}
    stride = 1
    for i in range(len(t_counts) - 1, -1, -1):
        t_strides[i] = stride
        stride *= t_counts[i]

    coeffs = [0] * MAX_TILE_DIMS

    if shared_dims:
        for dim in shared_dims:
            s_axis = s_dims[dim]
            t_axis = t_dims[dim]
            coeffs[s_axis] += t_strides[t_axis]
    else:
        # No dim_names: use source's own linear index
        s_counts = source_op.tile_counts
        stride = 1
        for i in range(len(s_counts) - 1, -1, -1):
            coeffs[i] = stride
            stride *= s_counts[i]

    return tuple(coeffs)


# =============================================================================
# Tile Schedulers
# =============================================================================


class TileScheduler(ABC):
    """Abstract base class for tile scheduling strategies.

    A scheduler takes the dependency graph and produces an ordered list of
    TileInstructions that SMs will fetch and execute. Different strategies
    optimize for different goals (load balance, latency, overlap, etc.).

    The scheduler receives:
    - op_records: List of _OpRecord with op metadata and tile coordinates
    - consumer_deps: Dict mapping op_idx -> list of _DepEdge dependencies
    - edges: All dependency edges in the graph

    And produces a list of TileInstruction in execution order.
    """

    @abstractmethod
    def schedule(
        self,
        op_records: List["_OpRecord"],
        consumer_deps: Dict[int, List["_DepEdge"]],
        edges: List["_DepEdge"],
    ) -> List[TileInstruction]:
        """Build the instruction list from ops and dependencies.

        Args:
            op_records: List of op records with tiles
            consumer_deps: Maps consumer op_idx to its dependency edges
            edges: All dependency edges

        Returns:
            Ordered list of TileInstruction (without END marker)
        """
        pass

    @staticmethod
    def _producer_threshold(
        edge: "_DepEdge",
        consumer_cursor: int,
        producer_total: int,
        consumer_total: int,
    ) -> int:
        """Minimum producer tiles that must be emitted before consumer tile k."""
        if edge.kind == "many_to_one":
            return producer_total
        elif edge.kind == "one_to_many":
            return min(
                (consumer_cursor * producer_total) // consumer_total + 1,
                producer_total,
            )
        else:
            return min(consumer_cursor + 1, producer_total)


class LevelBatchedScheduler(TileScheduler):
    """Level-batched wavefront scheduler.

    Emits tiles in dependency order, batching tiles from the same "level"
    together to naturally spread work across SMs when they fetch
    instructions with strided distribution.

    Strategy:
    1. Emit ALL tiles from source ops (no dependencies) first
    2. Then emit dependent ops using greedy wavefront

    This spreads producer tiles across SMs (better load balancing) while
    still respecting dependencies via barriers.

    Example (one-to-many, Producer 4 tiles → Consumer 16 tiles):
        P0, P1, P2, P3, C0, C1, C2, ...
    With 2 SMs: Block 0 gets P0,P2,C0,... Block 1 gets P1,P3,C1,...
    """

    def schedule(
        self,
        op_records: List["_OpRecord"],
        consumer_deps: Dict[int, List["_DepEdge"]],
        edges: List["_DepEdge"],
    ) -> List[TileInstruction]:
        num_ops = len(op_records)
        cursors = [0] * num_ops
        total_tiles = sum(rec.op.total_tiles for rec in op_records)
        instructions: List[TileInstruction] = []

        # Phase 1: Emit ALL tiles from source ops (no dependencies)
        for rec in op_records:
            idx = rec.op_idx
            if idx not in consumer_deps:
                for tile in rec.tiles:
                    instructions.append(TileInstruction(op_idx=idx, tiles=tile))
                cursors[idx] = rec.op.total_tiles

        # Phase 2: Greedy wavefront for remaining ops
        while len(instructions) < total_tiles:
            progress = False
            for rec in op_records:
                idx = rec.op_idx
                if cursors[idx] >= rec.op.total_tiles:
                    continue

                can_emit = True
                for edge in consumer_deps.get(idx, []):
                    prod_rec = op_records[edge.producer_idx]
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
                    instructions.append(TileInstruction(op_idx=idx, tiles=tile))
                    cursors[idx] += 1
                    progress = True

            if not progress:
                scheduled = [
                    f"{op_records[i].op.op_cls.__name__}: {cursors[i]}/{op_records[i].op.total_tiles}"
                    for i in range(num_ops)
                ]
                raise RuntimeError(f"Deadlock in tile scheduling. Cursors: {scheduled}")

        return instructions


class BackwardScheduler(TileScheduler):
    """Backward scheduler optimized for latency.

    Plans execution from the final outputs backward, assigning each tile
    the latest possible "start time" that still meets dependencies. Then
    schedules tiles in order of their deadlines (earliest deadline first).

    This achieves better overlap between stages because producers complete
    "just in time" for consumers, minimizing idle waiting.

    Algorithm:
    1. Compute "depth" for each op (longest path to any sink)
    2. For each tile, compute latest_start = max_depth - depth[op]
    3. Sort tiles by (latest_start, op_idx, tile_idx) for determinism
    4. Emit in sorted order, respecting dependencies

    The depth-based ordering ensures:
    - Sink tiles (final outputs) have highest priority (depth=0, latest_start=max)
    - Source tiles (no deps) have lowest priority, scheduled last but still valid
    - Critical path tiles are interleaved optimally
    """

    def schedule(
        self,
        op_records: List["_OpRecord"],
        consumer_deps: Dict[int, List["_DepEdge"]],
        edges: List["_DepEdge"],
    ) -> List[TileInstruction]:
        num_ops = len(op_records)
        if num_ops == 0:
            return []

        # Build producer_deps: op_idx -> list of producer op indices
        producer_deps: Dict[int, List[int]] = {i: [] for i in range(num_ops)}
        for edge in edges:
            producer_deps[edge.consumer_idx].append(edge.producer_idx)

        # Compute depth: longest path FROM this op TO any sink (no outgoing edges)
        # Sinks have depth 0, their producers have depth 1, etc.
        # We compute "reverse depth" - distance to nearest sink
        depth = [-1] * num_ops

        # Find sinks (ops with no consumers)
        has_consumer = set()
        for edge in edges:
            has_consumer.add(edge.producer_idx)
        sinks = [i for i in range(num_ops) if i not in has_consumer]

        # BFS from sinks backward
        from collections import deque
        queue = deque()
        for sink_idx in sinks:
            depth[sink_idx] = 0
            queue.append(sink_idx)

        while queue:
            op_idx = queue.popleft()
            current_depth = depth[op_idx]
            # All producers of this op should have depth >= current + 1
            for prod_idx in producer_deps[op_idx]:
                if depth[prod_idx] < current_depth + 1:
                    depth[prod_idx] = current_depth + 1
                    queue.append(prod_idx)

        # Handle disconnected ops (sources with no path to sink)
        max_depth = max(depth) if depth else 0
        for i in range(num_ops):
            if depth[i] < 0:
                depth[i] = max_depth + 1

        # Compute priority for each tile: (depth, op_idx, tile_idx)
        # Lower depth = closer to output = higher priority (schedule later)
        # We want to schedule high-depth (sources) first, low-depth (sinks) last
        # But we emit in order that respects dependencies

        # Create all tile entries with priorities
        tile_entries = []
        for rec in op_records:
            op_depth = depth[rec.op_idx]
            for tile_idx, tile in enumerate(rec.tiles):
                # Priority: higher depth = earlier in schedule
                # This naturally puts producers before consumers
                priority = (-op_depth, rec.op_idx, tile_idx)
                tile_entries.append((priority, rec.op_idx, tile))

        # Sort by priority (highest depth first = sources first)
        tile_entries.sort(key=lambda x: x[0])

        # Now emit in priority order, but verify dependencies are met
        # Use cursors to track emitted tiles per op
        cursors = [0] * num_ops
        emitted = set()
        instructions: List[TileInstruction] = []

        # Track which tiles we've scheduled
        pending = list(tile_entries)

        while pending:
            progress = False
            next_pending = []

            for priority, op_idx, tile in pending:
                tile_global_idx = (op_idx, cursors[op_idx])

                # Check if dependencies are met
                can_emit = True
                for edge in consumer_deps.get(op_idx, []):
                    prod_rec = op_records[edge.producer_idx]
                    needed = self._producer_threshold(
                        edge,
                        cursors[op_idx],
                        prod_rec.op.total_tiles,
                        op_records[op_idx].op.total_tiles,
                    )
                    if cursors[edge.producer_idx] < needed:
                        can_emit = False
                        break

                if can_emit and tile_global_idx not in emitted:
                    # Verify this is the next tile for this op
                    expected_tile = op_records[op_idx].tiles[cursors[op_idx]]
                    if tile == expected_tile:
                        instructions.append(TileInstruction(op_idx=op_idx, tiles=tile))
                        emitted.add(tile_global_idx)
                        cursors[op_idx] += 1
                        progress = True
                    else:
                        next_pending.append((priority, op_idx, tile))
                else:
                    next_pending.append((priority, op_idx, tile))

            if not progress and next_pending:
                # Try emitting any ready tile (fallback to greedy)
                for i, (priority, op_idx, tile) in enumerate(next_pending):
                    if cursors[op_idx] < op_records[op_idx].op.total_tiles:
                        can_emit = True
                        for edge in consumer_deps.get(op_idx, []):
                            prod_rec = op_records[edge.producer_idx]
                            needed = self._producer_threshold(
                                edge,
                                cursors[op_idx],
                                prod_rec.op.total_tiles,
                                op_records[op_idx].op.total_tiles,
                            )
                            if cursors[edge.producer_idx] < needed:
                                can_emit = False
                                break

                        if can_emit:
                            actual_tile = op_records[op_idx].tiles[cursors[op_idx]]
                            instructions.append(TileInstruction(op_idx=op_idx, tiles=actual_tile))
                            cursors[op_idx] += 1
                            progress = True
                            break

                if not progress:
                    scheduled = [
                        f"{op_records[i].op.op_cls.__name__}: {cursors[i]}/{op_records[i].op.total_tiles}"
                        for i in range(num_ops)
                    ]
                    raise RuntimeError(f"Deadlock in backward scheduling. Cursors: {scheduled}")

            pending = next_pending

        return instructions


# Default scheduler instance
_default_scheduler: TileScheduler = LevelBatchedScheduler()


def get_default_scheduler() -> TileScheduler:
    """Get the default tile scheduler."""
    return _default_scheduler


def set_default_scheduler(scheduler: TileScheduler) -> None:
    """Set the default tile scheduler."""
    global _default_scheduler
    _default_scheduler = scheduler


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
        builder.add_op(OpA, tile_counts=(4, 32),
                       dim_names={"batch": 0, "seqlen": 1})
        builder.add_op(OpB, tile_counts=(4,),
                       dim_names={"batch": 0})
        # OpB reads OpA's output → OpB batch=0 waits for all 32 seqlen tiles

    Fallback (no INPUTS/OUTPUTS): linear chain with 1:1 tile mapping.
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
        op_or_cls: Union[ScheduledOp, Type[Op]],
        tile_counts: Optional[Tuple[int, ...]] = None,
        dim_names: Optional[Dict[str, int]] = None,
        **params,
    ) -> "InstructionStreamBuilder":
        """Add an operation to the stream.

        Can be called in two ways:
        1. With a ScheduledOp: add_op(scheduled_op)
           Preserves all fields including tensor_ptrs for dependency detection.
        2. With an Op class: add_op(OpClass, tile_counts=(4, 1), ...)
           Creates a new ScheduledOp from the parameters.

        Args:
            op_or_cls: Either a ScheduledOp instance or an Op class
            tile_counts: Tuple of tile counts per axis (required when passing Op class)
            dim_names: Maps semantic dimension names to tile axis indices.
                Example: {"M": 0, "D": 1}
            **params: Operation-specific parameters (only used with Op class)
        """
        # Handle ScheduledOp instance
        if isinstance(op_or_cls, ScheduledOp):
            op = op_or_cls
        else:
            # Handle Op class with parameters
            if tile_counts is None:
                raise ValueError("tile_counts is required when passing an Op class")
            op = ScheduledOp(
                op_cls=op_or_cls,
                tile_counts=tile_counts,
                params=params,
                dim_names=dim_names or {},
            )

        # Validate dim_names
        if op.dim_names:
            for dim, axis in op.dim_names.items():
                if not isinstance(axis, int) or axis < 0 or axis >= MAX_TILE_DIMS:
                    raise ValueError(f"Invalid axis {axis} for dim '{dim}'. Must be int in [0, {MAX_TILE_DIMS})")
            axes = list(op.dim_names.values())
            if len(axes) != len(set(axes)):
                raise ValueError(f"dim_names maps multiple dims to the same axis: {op.dim_names}")

        # Pre-compute flat tile list using row-major linearization
        tiles = []
        counts = op.tile_counts
        ndims = len(counts)
        for linear_idx in range(op.total_tiles):
            # Decompose linear index into multi-dim tile coordinates (row-major)
            tile = []
            remainder = linear_idx
            for i in range(ndims):
                # Stride for axis i = product of counts[i+1:]
                stride = math.prod(counts[i + 1:]) if i + 1 < ndims else 1
                tile.append(remainder // stride)
                remainder %= stride
            tiles.append(tuple(tile))

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
            # Linear chain fallback: each op depends on previous, 1:1
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
            formulas, count = self._resolve_linear_chain_formulas()

        self._cached_formulas = formulas
        self._barrier_count = count
        return formulas, count

    def _resolve_linear_chain_formulas(
        self,
    ) -> Tuple[
        Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]],
        int,
    ]:
        """Compute barrier formulas for linear chain (no named buffers).

        Each op signals its own barrier set. The next op waits on the
        previous op's barriers with 1:1 tile index mapping.
        """
        formulas: Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]] = {}
        barrier_counter = 0

        for i, rec in enumerate(self._op_records):
            op = rec.op
            signal_base = barrier_counter

            # Own linear index strides (row-major)
            coeffs = _linear_strides(op.tile_counts)

            # Signal: own barrier
            signal_formulas = [
                BarrierFormula(base=signal_base, coeffs=coeffs)
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
                        coeffs=coeffs,
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
        # Track buffer producers by name
        buffer_producers: Dict[str, int] = {}
        for rec in self._op_records:
            for buf in rec.op.op_cls.OUTPUTS:
                if buf in buffer_producers:
                    raise ValueError(f"Buffer '{buf}' produced by both op {buffer_producers[buf]} and op {rec.op_idx}")
                buffer_producers[buf] = rec.op_idx

        # Also track buffer producers by tensor data pointer for automatic dependency detection
        # This maps (consumer_op_idx, consumer_input_buf) -> producer_op_idx
        # when tensors share the same data pointer but have different buffer names
        tensor_ptr_deps: Dict[Tuple[int, str], int] = {}
        for cons_rec in self._op_records:
            for cons_buf in cons_rec.op.op_cls.INPUTS:
                cons_ptr = cons_rec.op.tensor_ptrs.get(cons_buf)
                if cons_ptr is None:
                    continue
                # Look for a producer with matching tensor pointer
                for prod_rec in self._op_records:
                    if prod_rec.op_idx >= cons_rec.op_idx:
                        continue  # Only look at earlier ops
                    for prod_buf in prod_rec.op.op_cls.OUTPUTS:
                        prod_ptr = prod_rec.op.tensor_ptrs.get(prod_buf)
                        if prod_ptr is not None and prod_ptr == cons_ptr:
                            # Found a match by tensor identity
                            tensor_ptr_deps[(cons_rec.op_idx, cons_buf)] = prod_rec.op_idx

        # Init per-op formula lists
        formulas: Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]] = {
            i: ([], []) for i in range(len(self._op_records))
        }
        barrier_counter = 0

        # Process each dependency edge
        for rec in self._op_records:
            for buf in rec.op.op_cls.INPUTS:
                # First try matching by buffer name
                prod_idx = buffer_producers.get(buf)

                # If no name match, or name matches self (in-place op), try tensor identity
                # This handles cases like RopeOp (q->q in-place) depending on RMSNormOp (->y)
                # where the tensor pointers match but buffer names differ
                if prod_idx is None or prod_idx == rec.op_idx:
                    ptr_prod_idx = tensor_ptr_deps.get((rec.op_idx, buf))
                    if ptr_prod_idx is not None:
                        prod_idx = ptr_prod_idx

                if prod_idx is None:
                    continue  # External input (provided by host, not by another op)
                if prod_idx == rec.op_idx:
                    continue  # True self-dependency (same op produces and consumes)
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
                p_divs = [1] * MAX_TILE_DIMS  # Producer divisors
                c_divs = [1] * MAX_TILE_DIMS  # Consumer divisors
                expected = 1

                if producer_only and not consumer_only:
                    # Many-to-one: producer has extra dims not in consumer
                    target_op = c_op

                    # Handle tile size ratios on shared dims
                    for dim in shared_dims:
                        p_axis = p_op.dim_names[dim]
                        c_axis = c_op.dim_names[dim]
                        p_ts = p_op.tile_sizes.get(dim, 1)
                        c_ts = c_op.tile_sizes.get(dim, 1)
                        if p_ts > c_ts:
                            ratio = p_ts // c_ts
                            c_divs[c_axis] = ratio
                        elif c_ts > p_ts:
                            ratio = c_ts // p_ts
                            p_divs[p_axis] = ratio
                            expected *= ratio

                    # Barrier count: min tile counts on shared dims
                    num_barriers = 1
                    for dim in shared_dims:
                        p_axis = p_op.dim_names[dim]
                        c_axis = c_op.dim_names[dim]
                        num_barriers *= min(
                            p_op.tiles_for_axis(p_axis),
                            c_op.tiles_for_axis(c_axis),
                        )

                    # Collapse producer-only dims
                    collapsed = 1
                    for dim in producer_only:
                        axis = p_op.dim_names[dim]
                        collapsed *= p_op.tiles_for_axis(axis)
                    expected *= collapsed
                elif consumer_only and not producer_only:
                    # One-to-many: producer has fewer dims
                    target_op = p_op

                    # Handle tile size ratios on shared dims
                    for dim in shared_dims:
                        p_axis = p_op.dim_names[dim]
                        c_axis = c_op.dim_names[dim]
                        p_ts = p_op.tile_sizes.get(dim, 1)
                        c_ts = c_op.tile_sizes.get(dim, 1)
                        if p_ts > c_ts:
                            ratio = p_ts // c_ts
                            c_divs[c_axis] = ratio
                        elif c_ts > p_ts:
                            ratio = c_ts // p_ts
                            p_divs[p_axis] = ratio

                    # Barrier count: min tile counts on shared dims
                    num_barriers = 1
                    for dim in shared_dims:
                        p_axis = p_op.dim_names[dim]
                        c_axis = c_op.dim_names[dim]
                        num_barriers *= min(
                            p_op.tiles_for_axis(p_axis),
                            c_op.tiles_for_axis(c_axis),
                        )
                    expected = 1
                else:
                    # Same dims (or no dims) - check for tile size differences
                    for dim in shared_dims:
                        p_axis = p_op.dim_names[dim]
                        c_axis = c_op.dim_names[dim]
                        p_ts = p_op.tile_sizes.get(dim, 1)
                        c_ts = c_op.tile_sizes.get(dim, 1)

                        if p_ts > c_ts:
                            ratio = p_ts // c_ts
                            c_divs[c_axis] = ratio
                        elif c_ts > p_ts:
                            ratio = c_ts // p_ts
                            p_divs[p_axis] = ratio
                            expected *= ratio

                    target_op = c_op
                    num_barriers = min(p_op.total_tiles, c_op.total_tiles)

                # Compute formula coefficients
                p_coeffs = _compute_formula_coeffs(p_op, target_op, shared_dims)
                c_coeffs = _compute_formula_coeffs(c_op, target_op, shared_dims)

                # Producer signal formula
                formulas[prod_idx][1].append(
                    BarrierFormula(
                        base=barrier_counter,
                        coeffs=p_coeffs,
                        divs=tuple(p_divs),
                    )
                )

                # Consumer wait formula
                formulas[rec.op_idx][0].append(
                    BarrierFormula(
                        base=barrier_counter,
                        coeffs=c_coeffs,
                        divs=tuple(c_divs),
                        expected=expected,
                    )
                )

                barrier_counter += num_barriers

        return formulas, barrier_counter

    def build(self, scheduler: Optional[TileScheduler] = None) -> List[TileInstruction]:
        """Build an instruction list using the specified scheduler.

        Args:
            scheduler: Tile scheduler to use. If None, uses the default
                scheduler (LevelBatchedScheduler by default).

        Returns:
            List of TileInstructions with END marker at the end.

        The default LevelBatchedScheduler emits tiles in dependency order,
        batching tiles from the same "level" together to naturally spread
        work across SMs when they fetch instructions with strided distribution.

        For example with one-to-many (Producer 4 tiles → Consumer 16 tiles):
            P0, P1, P2, P3, C0, C1, C2, ...
        With 2 SMs: Block 0 gets P0,P2,C0,... Block 1 gets P1,P3,C1,...
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

        # Use provided scheduler or default
        if scheduler is None:
            scheduler = get_default_scheduler()

        instructions = scheduler.schedule(self._op_records, consumer_deps, edges)
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
    # Scheduling
    "ScheduledOp",
    "TensorRegistry",
    "TileScheduler",
    "LevelBatchedScheduler",
    "BackwardScheduler",
    "get_default_scheduler",
    "set_default_scheduler",
    # Barrier Formulas
    "BarrierFormula",
    # Instruction Stream
    "INSTRUCTION_WORDS",
    "TileInstruction",
    "InstructionStreamBuilder",
]
