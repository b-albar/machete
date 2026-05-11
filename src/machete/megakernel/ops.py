# Copyright (c) 2025, Machete Authors
"""
GPU-Executable Operations for Megakernel.

This module defines the operation protocol for GPU execution using CuTe DSL.
Operations are templates that get inlined at compile time, enabling full
compiler optimization with no runtime dispatch overhead.

Subclass ``Op`` and declare tensors, tiling, then implement compute::

    class MyOp(Op):
        reads  = {"x": (Float32, ("M", "D"))}
        writes = {"y": (Float32, ("M", "D"))}
        tile   = ("M", "D")  # dimension names only

        @cute.jit
        def compute(self, page_ptr, tile_M, tile_D, op_config_ptr):
            runtime_m = config_dim_i32(op_config_ptr, "M", type(self))
            x = config_flat_tensor(op_config_ptr, "x", self.x_dtype, runtime_m * self.D, type(self))
            y = config_flat_tensor(op_config_ptr, "y", self.y_dtype, runtime_m * self.D, type(self))
            ...

    # Tile sizes passed at schedule time; tile_counts deduced
    ops = MyOp.schedule(x=tensor, tile_sizes={"M": 4})
    kernel = Megakernel(ops)
    kernel.run()

Shared config helpers intentionally mirror common CuTe DSL patterns:
    - ``config_dim_i32(op_config_ptr, "B", type(self))``
    - ``config_ptr_i64(op_config_ptr, "x", type(self))``
    - ``config_flat_tensor(op_config_ptr, "x", self.x_dtype, size, type(self))``
"""

import math
import struct
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32

from .interpreter import ld_global_i32, ld_global_i64


# =============================================================================
# Constants
# =============================================================================

MAX_TILE_DIMS = 5  # Matches TMA's 5D tensor capability

_SCHEDULING_STATIC_DIM_PREFIXES = ("barrier_", "pipeline_", "load_replay_")


@dataclass(frozen=True)
class PipelineSpec:
    """Compile-time resource and range contract for staged pipeline ops.

    This is intentionally structural metadata. Ops still implement the normal
    ``load``/``compute``/``store`` phases; this declaration only tells the
    megakernel how much instruction-local staged state to reserve and whether
    scheduler-side tile ranges can be coalesced.
    """

    page_count: int
    page_bytes: int = 0
    input_stages: int = 0
    output_stages: int = 0
    stage_pages: int = 1
    semaphore_count: int = 0
    scratch_bytes: int = 0
    range_axis: int = -1
    range_end_axis: int = -1
    range_block_size: int = 1
    coalesce_ranges: bool = False

    def __post_init__(self):
        if self.page_count < 1:
            raise ValueError("PipelineSpec.page_count must be >= 1")
        if self.page_bytes < 0:
            raise ValueError("PipelineSpec.page_bytes must be non-negative")
        if self.input_stages < 0 or self.output_stages < 0:
            raise ValueError("PipelineSpec stages must be non-negative")
        if self.stage_pages < 1:
            raise ValueError("PipelineSpec.stage_pages must be >= 1")
        if self.semaphore_count < 0:
            raise ValueError("PipelineSpec.semaphore_count must be non-negative")
        if self.scratch_bytes < 0:
            raise ValueError("PipelineSpec.scratch_bytes must be non-negative")
        if self.range_axis < -1 or self.range_axis >= MAX_TILE_DIMS:
            raise ValueError(
                f"PipelineSpec.range_axis must be -1 or in [0, {MAX_TILE_DIMS})"
            )
        if self.range_end_axis < -1 or self.range_end_axis >= MAX_TILE_DIMS:
            raise ValueError(
                f"PipelineSpec.range_end_axis must be -1 or in [0, {MAX_TILE_DIMS})"
            )
        if self.coalesce_ranges and self.range_axis < 0:
            raise ValueError("PipelineSpec.coalesce_ranges requires range_axis >= 0")
        if (
            self.coalesce_ranges
            and self.range_end_axis >= 0
            and self.range_end_axis == self.range_axis
        ):
            raise ValueError("PipelineSpec.range_end_axis must differ from range_axis")
        if self.range_block_size < 0:
            raise ValueError("PipelineSpec.range_block_size must be >= 0")

    @property
    def resource_bytes(self) -> int:
        """Minimum bytes needed inside an op page for pipeline-local state."""
        return self.page_count * self.page_bytes + self.semaphore_count * 8 + self.scratch_bytes

    def with_overrides(self, **overrides: int) -> "PipelineSpec":
        values = {
            "page_count": self.page_count,
            "page_bytes": self.page_bytes,
            "input_stages": self.input_stages,
            "output_stages": self.output_stages,
            "stage_pages": self.stage_pages,
            "semaphore_count": self.semaphore_count,
            "scratch_bytes": self.scratch_bytes,
            "range_axis": self.range_axis,
            "range_end_axis": self.range_end_axis,
            "range_block_size": self.range_block_size,
            "coalesce_ranges": self.coalesce_ranges,
        }
        values.update({k: v for k, v in overrides.items() if v is not None})
        return PipelineSpec(**values)

    @classmethod
    def range_capable(
        cls,
        *,
        page_count: int = 1,
        range_axis: int,
        range_end_axis: int = -1,
        range_block_size: int = 1,
        page_bytes: int = 0,
        semaphore_count: int = 0,
        scratch_bytes: int = 0,
    ) -> "PipelineSpec":
        """Return a pipeline contract whose range support is opt-in per op.

        Use this on an op class to declare that scheduled instances may pass a
        ``TileRange``. It does not coalesce by default; each scheduled op opts
        in with ``tile_range=TileRange.coalesced(...)``.
        """
        return cls(
            page_count=page_count,
            page_bytes=page_bytes,
            semaphore_count=semaphore_count,
            scratch_bytes=scratch_bytes,
            range_axis=range_axis,
            range_end_axis=range_end_axis,
            range_block_size=range_block_size,
            coalesce_ranges=False,
        )

    @classmethod
    def streaming(
        cls,
        *,
        input_stages: int = 3,
        output_stages: int = 3,
        stage_pages: int = 4,
        page_bytes: int = 0,
        scratch_bytes: int = 0,
        range_axis: int = -1,
        range_end_axis: int = -1,
        range_block_size: int = 1,
        coalesce_ranges: bool = False,
    ) -> "PipelineSpec":
        """Return a staged load/compute/store page-ring contract."""
        return cls(
            page_count=1 + input_stages * stage_pages,
            page_bytes=page_bytes,
            input_stages=input_stages,
            output_stages=output_stages,
            stage_pages=stage_pages,
            semaphore_count=1 + 2 * (input_stages + output_stages),
            scratch_bytes=scratch_bytes,
            range_axis=range_axis,
            range_end_axis=range_end_axis,
            range_block_size=range_block_size,
            coalesce_ranges=coalesce_ranges,
        )

    def page_protocol(self) -> "InstructionPageProtocol":
        """Return the named instruction-owned page/semaphore protocol.

        ``PipelineSpec`` is intentionally compact because it is carried on every
        scheduled op.  The named protocol is host-side structure used by region
        lowering and tests to reason about staged handlers without growing
        the device metadata stream.
        """
        staged_page_count = 1 + self.input_stages * self.stage_pages
        staged_semaphore_count = 1 + 2 * (self.input_stages + self.output_stages)
        if (
            self.page_count == staged_page_count
            and self.semaphore_count == staged_semaphore_count
        ):
            return InstructionPageProtocol.streaming(
                input_stages=self.input_stages,
                output_stages=self.output_stages,
                stage_pages=self.stage_pages,
                page_bytes=self.page_bytes,
                scratch_bytes=self.scratch_bytes,
            )
        return InstructionPageProtocol.generic(
            page_count=self.page_count,
            semaphore_count=self.semaphore_count,
            page_bytes=self.page_bytes,
            scratch_bytes=self.scratch_bytes,
        )


@dataclass(frozen=True)
class TileRange:
    """Per-scheduled-op range coalescing policy.

    ``PipelineSpec`` declares that an op class is range-capable. ``TileRange``
    is the user-facing schedule-time choice for one op instance::

        ops = MyOp.schedule(
            ...,
            tile_range=TileRange.coalesced("O", block_size=2),
        )

    The axis may be a semantic tile dimension name or a numeric tile axis.
    """

    axis: Union[int, str]
    block_size: int = 1
    end_axis: Optional[Union[int, str]] = None
    coalesce: bool = True
    stride: int = 1

    def __post_init__(self):
        if self.block_size < 0:
            raise ValueError("TileRange.block_size must be >= 0")
        if self.stride < 1:
            raise ValueError("TileRange.stride must be >= 1")

    @classmethod
    def coalesced(
        cls,
        axis: Union[int, str],
        *,
        block_size: int = 1,
        end_axis: Optional[Union[int, str]] = None,
    ) -> "TileRange":
        return cls(axis=axis, block_size=block_size, end_axis=end_axis, coalesce=True)

    @classmethod
    def strided(
        cls,
        axis: Union[int, str],
        *,
        stride: int,
        block_size: int = 1,
        end_axis: Optional[Union[int, str]] = None,
    ) -> "TileRange":
        return cls(
            axis=axis,
            block_size=block_size,
            end_axis=end_axis,
            coalesce=True,
            stride=stride,
        )

    @classmethod
    def disabled(
        cls,
        axis: Union[int, str] = 0,
        *,
        end_axis: Optional[Union[int, str]] = None,
    ) -> "TileRange":
        return cls(axis=axis, block_size=1, end_axis=end_axis, coalesce=False)

    @staticmethod
    def _resolve_axis(axis: Union[int, str], op: "ScheduledOp") -> int:
        if isinstance(axis, str):
            if axis not in op.dim_names:
                raise ValueError(
                    f"Range axis {axis!r} is not a tile dimension for "
                    f"{op.op_cls.__name__}; available: {sorted(op.dim_names)}"
                )
            return int(op.dim_names[axis])
        return int(axis)

    def apply(self, op: "ScheduledOp") -> "ScheduledOp":
        if getattr(op.op_cls, "pipeline", None) is None:
            raise ValueError(
                f"Op {op.op_cls.__name__} does not declare a PipelineSpec; "
                "tile ranges require a range-capable pipeline."
            )
        range_axis = self._resolve_axis(self.axis, op)
        range_end_axis = (
            self._resolve_axis(self.end_axis, op)
            if self.end_axis is not None
            else range_axis + 1
        )
        op.static_dims["pipeline_coalesce_ranges"] = bool(self.coalesce)
        op.static_dims["pipeline_range_axis"] = range_axis
        op.static_dims["pipeline_range_end_axis"] = range_end_axis
        op.static_dims["pipeline_range_block_size"] = int(self.block_size)
        op.static_dims["pipeline_range_stride"] = int(self.stride)
        if self.stride > 1:
            stride_axis = range_end_axis + 1
            if stride_axis >= MAX_TILE_DIMS or stride_axis == range_axis:
                raise ValueError(
                    "TileRange.strided requires one spare tile coordinate after "
                    "range_end_axis to encode the stride"
                )
            op.static_dims["pipeline_range_stride_axis"] = stride_axis
        return op


@dataclass(frozen=True)
class PageRole:
    """A contiguous set of pages owned by one specialized instruction."""

    name: str
    offset: int
    count: int

    def page(self, local_idx: int = 0) -> int:
        if local_idx < 0 or local_idx >= self.count:
            raise IndexError(
                f"page role {self.name!r} index {local_idx} out of range for {self.count} pages"
            )
        return self.offset + local_idx


@dataclass(frozen=True)
class SemaphoreRole:
    """A contiguous set of instruction-local semaphores."""

    name: str
    offset: int
    count: int
    participants: int = 1

    def semaphore(self, local_idx: int = 0) -> int:
        if local_idx < 0 or local_idx >= self.count:
            raise IndexError(
                f"semaphore role {self.name!r} index {local_idx} out of range for {self.count} semaphores"
            )
        return self.offset + local_idx


@dataclass(frozen=True)
class InstructionPageProtocol:
    """Named page/semaphore ABI for specialized staged instructions.

    This is host-side protocol metadata.  A generated persistent handler can use
    the names to emit a static page ring and semaphores, while the current replay
    backend can ignore the names and continue using the compact counts from
    ``PipelineSpec``.
    """

    page_roles: Tuple[PageRole, ...]
    semaphore_roles: Tuple[SemaphoreRole, ...] = ()
    page_bytes: int = 0
    scratch_bytes: int = 0

    def __post_init__(self):
        seen_pages = set()
        for role in self.page_roles:
            if role.count < 1:
                raise ValueError(f"page role {role.name!r} must have at least one page")
            if role.name in seen_pages:
                raise ValueError(f"duplicate page role {role.name!r}")
            seen_pages.add(role.name)
        seen_sems = set()
        for role in self.semaphore_roles:
            if role.count < 1:
                raise ValueError(f"semaphore role {role.name!r} must have at least one semaphore")
            if role.participants < 1:
                raise ValueError(f"semaphore role {role.name!r} must have participants >= 1")
            if role.name in seen_sems:
                raise ValueError(f"duplicate semaphore role {role.name!r}")
            seen_sems.add(role.name)
        if self.page_bytes < 0 or self.scratch_bytes < 0:
            raise ValueError("page_bytes and scratch_bytes must be non-negative")

    @property
    def page_count(self) -> int:
        return sum(role.count for role in self.page_roles)

    @property
    def semaphore_count(self) -> int:
        return sum(role.count for role in self.semaphore_roles)

    @property
    def resource_bytes(self) -> int:
        return self.page_count * self.page_bytes + self.semaphore_count * 8 + self.scratch_bytes

    def page_role(self, name: str) -> PageRole:
        for role in self.page_roles:
            if role.name == name:
                return role
        raise KeyError(name)

    def semaphore_role(self, name: str) -> SemaphoreRole:
        for role in self.semaphore_roles:
            if role.name == name:
                return role
        raise KeyError(name)

    def page(self, name: str, local_idx: int = 0) -> int:
        return self.page_role(name).page(local_idx)

    def semaphore(self, name: str, local_idx: int = 0) -> int:
        return self.semaphore_role(name).semaphore(local_idx)

    def semaphore_participants(self, name: str) -> int:
        return self.semaphore_role(name).participants

    @classmethod
    def streaming(
        cls,
        *,
        input_stages: int = 3,
        output_stages: int = 3,
        stage_pages: int = 4,
        page_bytes: int = 0,
        scratch_bytes: int = 0,
    ) -> "InstructionPageProtocol":
        if input_stages < 0 or output_stages < 0:
            raise ValueError("stages must be non-negative")
        if stage_pages < 1:
            raise ValueError("stage_pages must be >= 1")

        page_roles = [PageRole("activation", 0, 1)]
        page_offset = 1
        for stage in range(input_stages):
            page_roles.append(PageRole(f"input_stage_{stage}", page_offset, stage_pages))
            page_offset += stage_pages

        sem_roles = [SemaphoreRole("activations_arrived", 0, 1)]
        sem_offset = 1
        for stage in range(input_stages):
            sem_roles.append(SemaphoreRole(f"weights_arrived_{stage}", sem_offset, 1))
            sem_offset += 1
        for stage in range(input_stages):
            sem_roles.append(SemaphoreRole(f"weights_finished_{stage}", sem_offset, 1))
            sem_offset += 1
        for stage in range(output_stages):
            sem_roles.append(SemaphoreRole(f"outputs_arrived_{stage}", sem_offset, 1))
            sem_offset += 1
        for stage in range(output_stages):
            sem_roles.append(SemaphoreRole(f"outputs_finished_{stage}", sem_offset, 1))
            sem_offset += 1

        return cls(
            page_roles=tuple(page_roles),
            semaphore_roles=tuple(sem_roles),
            page_bytes=page_bytes,
            scratch_bytes=scratch_bytes,
        )

    @classmethod
    def generic(
        cls,
        *,
        page_count: int,
        semaphore_count: int = 0,
        page_bytes: int = 0,
        scratch_bytes: int = 0,
    ) -> "InstructionPageProtocol":
        page_roles = tuple(PageRole(f"page_{idx}", idx, 1) for idx in range(page_count))
        sem_roles = tuple(
            SemaphoreRole(f"semaphore_{idx}", idx, 1)
            for idx in range(semaphore_count)
        )
        return cls(
            page_roles=page_roles,
            semaphore_roles=sem_roles,
            page_bytes=page_bytes,
            scratch_bytes=scratch_bytes,
        )


@dataclass(frozen=True)
class PipelineABI:
    """Execution ABI for ops that declare ``pipeline`` resources.

    ``PipelineSpec`` describes memory/range resources. This ABI describes
    how the op consumes them.
    """

    kind: str
    execution: str

    KIND_STAGED: ClassVar[str] = "staged"
    EXECUTION_OP_OWNED: ClassVar[str] = "op_owned"

    def __post_init__(self):
        if self.kind != self.KIND_STAGED:
            raise ValueError(f"unsupported pipeline ABI kind: {self.kind!r}")
        if self.execution != self.EXECUTION_OP_OWNED:
            raise ValueError(
                f"unsupported pipeline ABI execution: {self.execution!r}"
            )

    @classmethod
    def op_owned(cls) -> "PipelineABI":
        return cls(kind=cls.KIND_STAGED, execution=cls.EXECUTION_OP_OWNED)


def is_compile_static_dim(name: str) -> bool:
    """Return whether a static dim should specialize device code.

    Some ``ScheduledOp.static_dims`` entries are metadata for the host-side
    dependency scheduler. They must stay on the op for barrier construction, but
    they should not become ``self`` attributes or compile-key entries because
    that creates duplicate handlers for identical device code.
    """
    return not name.startswith(_SCHEDULING_STATIC_DIM_PREFIXES)


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
    storage_ptr: int  # Backing storage pointer (shared across aliasing views)
    storage_offset: int  # Offset into backing storage, in elements


def _decompose_storage_offset(meta: TensorMeta) -> Optional[Tuple[int, ...]]:
    """Approximate the logical start indices of a non-overlapping view.

    This is primarily used to recognize views that share the same backing
    storage and differ only by a slice on the innermost dimension, such as a
    packed QKV buffer consumed through separate q/k/v views.
    """
    if meta.ndim == 0:
        return ()

    remaining = meta.storage_offset
    starts = []
    for axis, stride in enumerate(meta.strides):
        if stride <= 0:
            return None
        if axis == meta.ndim - 1:
            starts.append(remaining)
            break
        starts.append(remaining // stride)
        remaining %= stride
    return tuple(starts)


def is_last_dim_slice_alias(a: TensorMeta, b: TensorMeta) -> bool:
    """Return whether two views share the same outer lattice and slice dim -1."""
    if a.storage_ptr != b.storage_ptr or a.ndim != b.ndim or a.ndim == 0:
        return False
    if a.strides[-1] != 1 or b.strides[-1] != 1:
        return False
    if a.shape[:-1] != b.shape[:-1]:
        return False
    if a.strides[:-1] != b.strides[:-1]:
        return False

    a_starts = _decompose_storage_offset(a)
    b_starts = _decompose_storage_offset(b)
    if a_starts is None or b_starts is None:
        return False
    return a_starts[:-1] == b_starts[:-1]


def last_dim_slice_region(full: TensorMeta, view: TensorMeta) -> Optional[Tuple[int, int, int]]:
    """Return the half-open last-dim region for ``view`` inside ``full``.

    The returned tuple is ``(start, length, full_length)`` in elements.  This is
    stricter than ``tensor_meta_overlaps`` and is used by the dependency
    scheduler to map a full-tensor producer to a consumer reading one contiguous
    innermost-dimension slice.
    """
    if not is_last_dim_slice_alias(full, view):
        return None
    if full.shape[:-1] != view.shape[:-1]:
        return None

    full_starts = _decompose_storage_offset(full)
    view_starts = _decompose_storage_offset(view)
    if full_starts is None or view_starts is None:
        return None

    start = view_starts[-1] - full_starts[-1]
    length = view.shape[-1]
    full_length = full.shape[-1]
    if start < 0 or length < 0 or start + length > full_length:
        return None
    return start, length, full_length


def _flattened_leading_overlap(a: TensorMeta, b: TensorMeta) -> Optional[bool]:
    """Exact overlap for views that differ only by flattening the first 2 axes."""
    if a.storage_ptr != b.storage_ptr:
        return None
    if a.ndim + 1 == b.ndim:
        flat, expanded = a, b
        swapped = False
    elif b.ndim + 1 == a.ndim:
        flat, expanded = b, a
        swapped = True
    else:
        return None
    if flat.ndim < 2:
        return None
    if flat.strides != expanded.strides[1:]:
        return None
    if flat.shape[0] != expanded.shape[0] * expanded.shape[1]:
        return None

    flat_starts = _decompose_storage_offset(flat)
    expanded_starts = _decompose_storage_offset(expanded)
    if flat_starts is None or expanded_starts is None:
        return None

    expanded_flat_start = expanded_starts[0] * expanded.shape[1] + expanded_starts[1]
    flat_intervals = [(flat_starts[0], flat_starts[0] + flat.shape[0])]
    expanded_intervals = [(expanded_flat_start, expanded_flat_start + expanded.shape[0] * expanded.shape[1])]
    for axis in range(1, flat.ndim):
        flat_intervals.append((flat_starts[axis], flat_starts[axis] + flat.shape[axis]))
        expanded_axis = axis + 1
        expanded_intervals.append((
            expanded_starts[expanded_axis],
            expanded_starts[expanded_axis] + expanded.shape[expanded_axis],
        ))

    overlaps = all(
        max(f_lo, e_lo) < min(f_hi, e_hi)
        for (f_lo, f_hi), (e_lo, e_hi) in zip(flat_intervals, expanded_intervals)
    )
    return overlaps if not swapped else overlaps


def tensor_meta_overlaps(a: TensorMeta, b: TensorMeta) -> bool:
    """Return whether two tensor views overlap in device memory.

    Exact data_ptr matches remain the fast path for traditional aliasing. We
    also recognize views that share backing storage even when their ranks differ
    (for example a GEMM output scheduled as flat BSK and a later BSHD slice).
    """
    if a.data_ptr == b.data_ptr:
        return True
    if a.storage_ptr == b.storage_ptr:
        flattened_overlap = _flattened_leading_overlap(a, b)
        if flattened_overlap is not None:
            return flattened_overlap
        if a.ndim == b.ndim and a.strides == b.strides:
            a_starts = _decompose_storage_offset(a)
            b_starts = _decompose_storage_offset(b)
            if a_starts is not None and b_starts is not None:
                return all(
                    max(a_starts[axis], b_starts[axis])
                    < min(
                        a_starts[axis] + a.shape[axis],
                        b_starts[axis] + b.shape[axis],
                    )
                    for axis in range(a.ndim)
                )
        a_span = _tensor_storage_span(a)
        b_span = _tensor_storage_span(b)
        if a_span is not None and b_span is not None:
            return max(a_span[0], b_span[0]) < min(a_span[1], b_span[1])
    if not is_last_dim_slice_alias(a, b):
        return False

    a_starts = _decompose_storage_offset(a)
    b_starts = _decompose_storage_offset(b)
    assert a_starts is not None and b_starts is not None
    a_lo = a_starts[-1]
    a_hi = a_lo + a.shape[-1]
    b_lo = b_starts[-1]
    b_hi = b_lo + b.shape[-1]
    return max(a_lo, b_lo) < min(a_hi, b_hi)


def _tensor_storage_span(meta: TensorMeta) -> Optional[Tuple[int, int]]:
    """Conservative half-open storage span in elements for positive-stride views."""
    if meta.ndim == 0:
        return (meta.storage_offset, meta.storage_offset + 1)
    lo = meta.storage_offset
    hi = lo
    for size, stride in zip(meta.shape, meta.strides):
        if size <= 0:
            return (lo, lo)
        if stride < 0:
            return None
        hi += (size - 1) * stride
    return (lo, hi + 1)


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
    cls._CONFIG_PTR_I64_INDEX = {
        name: i for i, (name, _dtype, _dims) in enumerate(unique_tensors)
    }
    cls._CONFIG_PTR_I32_OFFSET = {
        name: 2 * i for i, (name, _dtype, _dims) in enumerate(unique_tensors)
    }
    # Only pack dynamic dims
    if dynamic_dim_names is None:
        dynamic_dim_names = {d for d, _, _ in unique_dims}
    dynamic_dims = [(d, t, a) for d, t, a in unique_dims if d in dynamic_dim_names]
    num_dynamic = len(dynamic_dims)
    config_size = 2 * num_unique + num_dynamic
    cls._CONFIG_DYNAMIC_I32_OFFSET = {
        dim_name: 2 * num_unique + j
        for j, (dim_name, _tensor_name, _axis_idx) in enumerate(dynamic_dims)
    }

    def pack_config(**tensors):
        """Pack tensor pointers and dynamic dimensions into the runtime config tensor."""
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


def _validate_declared_tensor_ranks(cls, unique_tensors, tensors):
    """Validate runtime tensor ranks against the op declaration."""
    for name, _dtype, dims in unique_tensors:
        if name not in tensors:
            continue
        tensor = tensors[name]
        expected_ndim = len(dims)
        if hasattr(tensor, "ndim") and tensor.ndim != expected_ndim:
            raise ValueError(
                f"{cls.__name__}: tensor '{name}' declared as {expected_ndim}D "
                f"(dims={dims}) but got {tensor.ndim}D tensor with shape {tuple(tensor.shape)}"
            )


def _extract_dim_values(cls, unique_dims, tensors):
    """Extract dimension sizes from scheduled tensors."""
    dim_values = {}
    for dim_name, tensor_name, axis_idx in unique_dims:
        value = int(tensors[tensor_name].shape[axis_idx])
        if dim_name in dim_values and dim_values[dim_name] != value:
            raise ValueError(
                f"{cls.__name__}: dim '{dim_name}' conflict: "
                f"expected {dim_values[dim_name]} but tensor '{tensor_name}' "
                f"axis {axis_idx} has {value}"
            )
        dim_values[dim_name] = value
    return dim_values


def _resolve_schedule_tile_sizes(tile_dim_names, tile_sizes, static_dims):
    """Resolve per-dimension tile sizes and derived tile counts."""
    resolved_tile_sizes = {}
    tile_counts = []
    for dim_name in tile_dim_names:
        requested_tile_size = tile_sizes.get(dim_name)
        dim_extent = static_dims[dim_name]
        if requested_tile_size is None:
            resolved_tile_sizes[dim_name] = dim_extent
            tile_counts.append(1)
            continue

        clamped_tile_size = min(requested_tile_size, dim_extent)
        resolved_tile_sizes[dim_name] = clamped_tile_size
        tile_counts.append((dim_extent + clamped_tile_size - 1) // clamped_tile_size)

    return resolved_tile_sizes, tuple(tile_counts)


def _infer_tensor_dtypes(unique_tensors, tensors):
    """Infer CUTLASS dtypes for tensors declared with dtype=None."""
    inferred_dtypes = {}
    for name, declared_dtype, _dims in unique_tensors:
        if declared_dtype is None:
            inferred_dtypes[name] = _resolve_dtype(None, tensors[name].dtype)
    return inferred_dtypes


def _capture_tensor_runtime_info(unique_tensors, tensors, tensor_dtypes):
    """Capture pointers, refs, strides, and TensorMeta for scheduled tensors."""
    tensor_ptrs = {}
    tensor_refs = {}
    tensor_metas = {}
    tensor_strides = {}

    for name in tensors:
        tensor = tensors[name]
        if hasattr(tensor, "data_ptr"):
            tensor_ptrs[name] = tensor.data_ptr()
            tensor_refs[name] = tensor

    for name, declared_dtype, dims in unique_tensors:
        if name not in tensors:
            continue
        tensor = tensors[name]
        resolved_dtype = tensor_dtypes.get(name, declared_dtype)
        tensor_metas[name] = TensorMeta(
            name=name,
            declared_dims=tuple(dims),
            ndim=len(dims),
            shape=tuple(tensor.shape),
            strides=tuple(tensor.stride()),
            dtype=resolved_dtype,
            is_contiguous=tensor.is_contiguous(),
            data_ptr=tensor.data_ptr(),
            storage_ptr=tensor.untyped_storage().data_ptr(),
            storage_offset=tensor.storage_offset(),
        )
        tensor_strides[name] = tuple(tensor.stride())

    return tensor_ptrs, tensor_refs, tensor_metas, tensor_strides


def _classify_declared_dims(cls, unique_dims, tile_dim_names):
    """Split declared dimensions into static and dynamic sets.

    Tile dimensions are compile-time constants. Non-tile dimensions are
    dynamic by default, and ops can force additional dimensions dynamic via
    ``dynamic_dims`` overrides on the class.
    """
    dynamic_dim_overrides = set(getattr(cls, "dynamic_dims", ()))
    non_tile_dims = {dim_name for dim_name, _, _ in unique_dims if dim_name not in tile_dim_names}
    dynamic_dim_names = non_tile_dims | dynamic_dim_overrides
    static_dim_names = [dim_name for dim_name, _, _ in unique_dims if dim_name not in dynamic_dim_names]
    return dynamic_dim_overrides, dynamic_dim_names, static_dim_names


def _validate_tile_dims_exist(tile_dim_names, unique_dims):
    """Ensure every tiled dimension appears in at least one tensor declaration."""
    dim_lookup = {dim_name for dim_name, _, _ in unique_dims}
    for dim_name in tile_dim_names:
        if dim_name not in dim_lookup:
            raise ValueError(f"Tile dim '{dim_name}' not found in any tensor shape declaration")


def _validate_tensor_set(tensor_set, valid_names, decl_name, io_kind):
    """Validate that a TMA or peer-store declaration references known tensors."""
    for name in tensor_set:
        if name not in valid_names:
            raise ValueError(f"{decl_name} tensor '{name}' not found in {io_kind}")


def _resolve_transfer_tensor_sets(cls, reads, writes):
    """Collect and validate TMA and peer-store tensor declarations."""
    read_names = set(reads.keys())
    write_names = set(writes.keys())

    transfer_sets = {
        "_TMA_LOADS": set(getattr(cls, "tma_loads", set())),
        "_TMA_COMPUTE_LOADS": set(getattr(cls, "tma_compute_loads", set())),
        "_TMA_STORES": set(getattr(cls, "tma_stores", set())),
        "_TMA_COMPUTE_STORES": set(getattr(cls, "tma_compute_stores", set())),
        "_TMA_REDUCE_STORES": set(getattr(cls, "tma_reduce_stores", set())),
        "_PEER_STORES": set(getattr(cls, "peer_stores", set())),
        "_PEER_REDUCE_STORES": set(getattr(cls, "peer_reduce_stores", set())),
    }

    _validate_tensor_set(transfer_sets["_TMA_LOADS"], read_names, "tma_loads", "reads")
    _validate_tensor_set(
        transfer_sets["_TMA_COMPUTE_LOADS"],
        read_names,
        "tma_compute_loads",
        "reads",
    )
    _validate_tensor_set(transfer_sets["_TMA_STORES"], write_names, "tma_stores", "writes")
    _validate_tensor_set(
        transfer_sets["_TMA_COMPUTE_STORES"],
        write_names,
        "tma_compute_stores",
        "writes",
    )
    _validate_tensor_set(
        transfer_sets["_TMA_REDUCE_STORES"],
        write_names,
        "tma_reduce_stores",
        "writes",
    )
    _validate_tensor_set(transfer_sets["_PEER_STORES"], write_names, "peer_stores", "writes")
    _validate_tensor_set(
        transfer_sets["_PEER_REDUCE_STORES"],
        write_names,
        "peer_reduce_stores",
        "writes",
    )
    return transfer_sets


def _collect_tma_tensor_dims(unique_tensors, transfer_sets):
    """Map each TMA-capable tensor name to its declared dimension list."""
    tensor_dims_map = {name: dims for name, _, dims in unique_tensors}
    tma_tensor_names = (
        transfer_sets["_TMA_LOADS"]
        | transfer_sets["_TMA_COMPUTE_LOADS"]
        | transfer_sets["_TMA_STORES"]
        | transfer_sets["_TMA_COMPUTE_STORES"]
        | transfer_sets["_TMA_REDUCE_STORES"]
        | transfer_sets["_PEER_STORES"]
        | transfer_sets["_PEER_REDUCE_STORES"]
    )
    return {
        name: tensor_dims_map[name]
        for name in tma_tensor_names
        if name in tensor_dims_map
    }


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

    tile_dim_names = set(tile)
    dynamic_dim_overrides, dynamic_dim_names, static_dim_names = _classify_declared_dims(
        cls,
        unique_dims,
        tile_dim_names,
    )

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

    _validate_tile_dims_exist(tile, unique_dims)

    transfer_sets = _resolve_transfer_tensor_sets(cls, reads, writes)
    for attr_name, value in transfer_sets.items():
        setattr(cls, attr_name, value)

    # Record the tensor dimensions that may need TMA tile-shape derivation later.
    cls._TMA_TENSOR_DIMS = _collect_tma_tensor_dims(unique_tensors, transfer_sets)

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
        tile_range = tensors.pop("tile_range", None)

        unique_dims = cls._UNIQUE_DIMS
        unique_tensors = cls._UNIQUE_TENSORS
        pack_fn = cls.pack_config

        _validate_declared_tensor_ranks(cls, unique_tensors, tensors)

        # Extract all runtime dimension values from scheduled tensors.
        # Explicit dynamic-dim overrides are excluded from static_dims so they
        # do not participate in the compile key, but still drive tile-count
        # computation and are packed into op_config.
        dim_values = _extract_dim_values(cls, unique_dims, tensors)
        static_dims = {
            dim_name: value
            for dim_name, value in dim_values.items()
            if dim_name not in cls._DYNAMIC_DIM_OVERRIDES
        }

        # Compute tile_counts from tile_sizes and full runtime dimension values.
        resolved_tile_sizes, tile_counts = _resolve_schedule_tile_sizes(
            cls._TILE_DIM_NAMES_ORDERED,
            tile_sizes,
            dim_values,
        )

        # Extract tensor dtypes when declared dtype is None (infer from tensor)
        tensor_dtypes = _infer_tensor_dtypes(unique_tensors, tensors)
        tensor_ptrs, tensor_refs, tensor_metas, tensor_strides = _capture_tensor_runtime_info(
            unique_tensors,
            tensors,
            tensor_dtypes,
        )

        config_data = pack_fn(**tensors)
        op = ScheduledOp(
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
        if tile_range is not None:
            op.with_tile_range(tile_range)
        return op

    cls._schedule_single = _schedule_single

    # Default schedule() is on Op base class. Subclasses override directly.

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
        elif phase == "compute":
            tma_names = cls._TMA_COMPUTE_LOADS | cls._TMA_COMPUTE_STORES
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


@cute.jit
def config_dim_i32(op_config_ptr, dim_name: str, cls):
    """Load one named dynamic dimension from ``op_config_ptr``.

    This mirrors typical CuTe DSL usage: pass the config pointer plus the op
    class and ask for the named dimension directly instead of hard-coding slot
    arithmetic in each op.
    """
    return ld_global_i32(
        op_config_ptr,
        Int32(cls._CONFIG_DYNAMIC_I32_OFFSET[dim_name]),
    )


@cute.jit
def config_ptr_i64(op_config_ptr, tensor_name: str, cls):
    """Load one named tensor pointer from ``op_config_ptr``."""
    return ld_global_i64(
        op_config_ptr,
        Int32(cls._CONFIG_PTR_I64_INDEX[tensor_name]),
    )


def config_flat_tensor(op_config_ptr, tensor_name: str, dtype, size: int, cls):
    """Build a flat gmem CuTe tensor view from one named packed config tensor."""
    ptr = config_ptr_i64(op_config_ptr, tensor_name, cls)
    return cute.make_tensor(
        cute.make_ptr(dtype, ptr, cute.AddressSpace.gmem, assumed_align=16),
        cute.make_layout(size),
    )


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
    pipeline: ClassVar[Optional[PipelineSpec]] = None
    pipeline_abi: ClassVar[Optional[PipelineABI]] = None
    pipeline_page_protocol: ClassVar[Optional[InstructionPageProtocol]] = None
    load_phase: ClassVar[Optional[str]] = None
    compute_phase: ClassVar[Optional[str]] = None
    store_phase: ClassVar[Optional[str]] = None
    communicate_phase: ClassVar[Optional[str]] = None
    inline_phases: ClassVar[Tuple[str, ...]] = ()
    sync_compute_warps_after_tile: ClassVar[bool] = False
    uses_smem_page: ClassVar[bool] = True

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
        self._bind_declared_phases()

    def _bind_phase(self, phase_name: str, implementation_name: Optional[str]) -> None:
        """Expose a named implementation method as a standard execution phase."""
        if implementation_name is None:
            return
        implementation = getattr(self, implementation_name)
        setattr(self, phase_name, implementation)

    def _bind_declared_phases(self) -> None:
        """Bind class-declared phase aliases on this concrete op instance."""
        self._bind_phase("load", type(self).load_phase)
        self._bind_phase("compute", type(self).compute_phase)
        self._bind_phase("store", type(self).store_phase)
        self._bind_phase("communicate", type(self).communicate_phase)

    def should_noinline_phase(self, phase_name: str) -> bool:
        """Return whether a generated wrapper for ``phase_name`` should be noinline.

        The framework default is conservative: every phase stays behind a
        noinline boundary. Ops with thin hot-path phase wrappers may opt
        specific phases into inlining via ``inline_phases``.
        """
        if phase_name in type(self).inline_phases:
            return False
        return True

    @classmethod
    def pipeline_protocol(cls) -> Optional[InstructionPageProtocol]:
        """Return the named op-local page/semaphore protocol, if any.

        Op-owned staged kernels should use this protocol instead of hard-coded
        shared-memory offsets. The megakernel materializes the corresponding
        offsets as compile-time attributes on each op instance.
        """
        explicit = getattr(cls, "pipeline_page_protocol", None)
        if explicit is not None:
            return explicit
        pipeline = getattr(cls, "pipeline", None)
        if pipeline is None:
            return None
        return pipeline.page_protocol()

    def pipeline_page_offset(self, name: str, local_idx: int = 0) -> int:
        """Return byte offset of a named op-local page within ``page_ptr``."""
        protocol = type(self).pipeline_protocol()
        if protocol is None:
            raise ValueError(f"{type(self).__name__} does not declare a pipeline")
        page_idx = protocol.page(name, local_idx)
        return getattr(self, f"machete_pipeline_page_{page_idx}_offset")

    def pipeline_semaphore_offset(self, name: str, local_idx: int = 0) -> int:
        """Return byte offset of a named op-local semaphore within ``page_ptr``."""
        protocol = type(self).pipeline_protocol()
        if protocol is None:
            raise ValueError(f"{type(self).__name__} does not declare a pipeline")
        sem_idx = protocol.semaphore(name, local_idx)
        return getattr(self, f"machete_pipeline_semaphore_{sem_idx}_offset")

    def pipeline_semaphore_participants(self, name: str) -> int:
        """Return participant count declared for a named op-local semaphore."""
        protocol = type(self).pipeline_protocol()
        if protocol is None:
            raise ValueError(f"{type(self).__name__} does not declare a pipeline")
        return protocol.semaphore_participants(name)

    def pipeline_scratch_offset(self) -> int:
        """Return byte offset of op-local scratch storage within ``page_ptr``."""
        return getattr(self, "machete_pipeline_scratch_offset")

    def __init_subclass__(cls, **kwargs):
        """Process op declarations when subclasses define reads and writes."""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "reads") and hasattr(cls, "writes"):
            _process_op_declarations(cls)

    # =========================================================================
    # Scheduling
    # =========================================================================

    @classmethod
    def schedule(cls, tile_sizes=None, **tensors):
        """Schedule op(s) from tensor kwargs. Returns a list of ScheduledOp.

        Subclasses override this for custom scheduling (auto-tiling, etc.).
        Default: wraps a single _schedule_single() call in a list.
        """
        return [cls._schedule_single(tile_sizes=tile_sizes, **tensors)]

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
        config.update(
            {
                name: value
                for name, value in op.static_dims.items()
                if is_compile_static_dim(name)
            }
        )

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
        """Populate default dim-name metadata from the op class."""
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

    def with_tile_range(self, tile_range: TileRange) -> "ScheduledOp":
        """Apply per-op range coalescing metadata and return ``self``."""
        return tile_range.apply(self)

__all__ = [
    # Constants
    "MAX_TILE_DIMS",
    "DEFAULT_PAGE_SIZE",
    "TORCH_TO_CUTLASS_DTYPE",
    # Metadata
    "InstructionPageProtocol",
    "PageRole",
    "PipelineSpec",
    "TileRange",
    "PipelineABI",
    "SemaphoreRole",
    "TensorMeta",
    # Protocol
    "Op",
    "build_op_config",
    "is_compile_static_dim",
    "last_dim_slice_region",
    # Scheduling
    "ScheduledOp",
]
