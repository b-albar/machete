# Copyright (c) 2025, Machete Authors
"""
Barrier Formulas, Tile Instructions, and Schedulers for Megakernel.

This module implements the dependency resolution and tile scheduling
infrastructure for the persistent megakernel:

- BarrierFormula: Compile-time formulas for barrier index computation
- TileInstruction: Flat-encoded work instructions for GPU execution
- TileScheduler: Abstract interface for tile ordering strategies
- InstructionStreamBuilder: Builds instruction streams with barrier DAGs

The dependency graph is resolved at build time and expressed as
BarrierFormula objects that get baked into op handlers at JIT compile time.
"""

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Type, Union

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from .ops import MAX_TILE_DIMS, Op, ScheduledOp


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
            result += self.coeffs[i] * (padded[i] // self.divs[i])
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

INSTRUCTION_WORDS = 2  # op_idx + linear_tile_idx (flat encoding)


@dataclass
class TileInstruction:
    """A single tile work instruction for the persistent megakernel.

    Flat encoding in global memory (2 x int32):
    [0]  op_idx: Which operation (indexes into op list), or -1 for end marker
    [1]  linear_tile_idx: Row-major linearized tile index

    Tile coordinates are decomposed at runtime from the linear index
    using compile-time per-op tile_counts (via decompose_tile JIT fn).

    Barrier wait/signal logic is baked into op handlers at compile time
    via BarrierFormula, not encoded in the instruction stream.
    """

    op_idx: int
    tiles: Tuple[int, ...]  # Up to MAX_TILE_DIMS tile indices

    # Sentinel for end of stream
    END_MARKER: int = -1

    def pack(self, strides: Optional[Tuple[int, ...]] = None) -> List[int]:
        """Pack into list of int32: [op_idx, linear_tile_idx].

        Args:
            strides: Row-major strides for linearization. If None (e.g., end
                marker), linear index is 0.
        """
        if self.op_idx == self.END_MARKER or strides is None:
            return [self.op_idx, 0]
        linear = sum(t * s for t, s in zip(self.tiles, strides))
        return [self.op_idx, linear]

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
    elif not s_dims and not t_dims:
        # Raw ops (no named dimensions): use source's own linear index
        s_counts = source_op.tile_counts
        stride = 1
        for i in range(len(s_counts) - 1, -1, -1):
            coeffs[i] = stride
            stride *= s_counts[i]
    # else: named ops with no shared dims — should not happen
    # (_resolve_named_formulas raises ValueError before reaching here)

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


def _compute_op_depths(num_ops: int, edges: List["_DepEdge"]) -> List[int]:
    """Compute per-op depth via BFS from sinks.

    Depth = longest path from this op to any sink. Sinks have depth 0,
    their producers have depth 1, etc. Disconnected ops get max_depth + 1.
    """
    producer_deps: Dict[int, List[int]] = {i: [] for i in range(num_ops)}
    for edge in edges:
        producer_deps[edge.consumer_idx].append(edge.producer_idx)

    depth = [-1] * num_ops
    has_consumer = set()
    for edge in edges:
        has_consumer.add(edge.producer_idx)

    queue = deque()
    for i in range(num_ops):
        if i not in has_consumer:
            depth[i] = 0
            queue.append(i)

    while queue:
        op_idx = queue.popleft()
        for prod_idx in producer_deps[op_idx]:
            new_depth = depth[op_idx] + 1
            if depth[prod_idx] < new_depth:
                depth[prod_idx] = new_depth
                queue.append(prod_idx)

    max_depth = max(depth) if depth else 0
    for i in range(num_ops):
        if depth[i] < 0:
            depth[i] = max_depth + 1

    return depth


def _depth_sorted_instructions(
    op_records: List["_OpRecord"],
    depth: List[int],
) -> List[TileInstruction]:
    """Sort tiles by (-depth, op_idx, tile_idx) and return instructions."""
    entries = []
    for rec in op_records:
        op_depth = depth[rec.op_idx]
        for tile_idx, tile in enumerate(rec.tiles):
            entries.append((-op_depth, rec.op_idx, tile_idx, tile))
    entries.sort(key=lambda x: (x[0], x[1], x[2]))
    return [
        TileInstruction(op_idx=op_idx, tiles=tile)
        for _, op_idx, _, tile in entries
    ]


class BackwardScheduler(TileScheduler):
    """Depth-first backward scheduler.

    Computes op depth (longest path to any sink) and sorts all tiles by
    (-depth, op_idx, tile_idx). This batches all tiles of the same op
    together in dependency order: sources first, then intermediate ops,
    then sinks.

    Since all producer tiles appear before their consumers in the
    instruction stream, runtime barrier waits never stall — by the time
    an SM fetches a consumer tile, all producer tiles are already in
    flight or completed.
    """

    def schedule(
        self,
        op_records: List["_OpRecord"],
        consumer_deps: Dict[int, List["_DepEdge"]],
        edges: List["_DepEdge"],
    ) -> List[TileInstruction]:
        if not op_records:
            return []
        depth = _compute_op_depths(len(op_records), edges)
        return _depth_sorted_instructions(op_records, depth)


class GroupedScheduler(TileScheduler):
    """Interleaved scheduler that groups tiles by shared dimensions.

    For ops with many-to-one dependencies (e.g., PrepOp(B,H,T) → FusedOp(B,H,V)),
    groups tiles by the shared dims (B,H) and interleaves groups:

    Instead of: all PrepOp(b,h,*) then all FusedOp(b,h,*)
    Produces:   PrepOp(0,0,*) FusedOp(0,0,*) PrepOp(0,1,*) FusedOp(0,1,*) ...

    This enables FusedOp(0,0,*) to start as soon as PrepOp(0,0,*) completes,
    while PrepOp(0,1,*) runs on other SMs — overlapping producer and consumer
    across groups.

    Falls back to BackwardScheduler ordering when no shared dimensions exist.
    """

    @staticmethod
    def _find_shared_dims(
        op_records: List["_OpRecord"],
        edges: List["_DepEdge"],
    ) -> Optional[Set[str]]:
        """Find dimensions shared across all dependency edges."""
        shared_dims: Optional[Set[str]] = None
        for edge in edges:
            p_dims = set(op_records[edge.producer_idx].op.dim_names.keys())
            c_dims = set(op_records[edge.consumer_idx].op.dim_names.keys())
            edge_shared = p_dims & c_dims
            if shared_dims is None:
                shared_dims = edge_shared
            else:
                shared_dims &= edge_shared
        return shared_dims if shared_dims else None

    def schedule(
        self,
        op_records: List["_OpRecord"],
        consumer_deps: Dict[int, List["_DepEdge"]],
        edges: List["_DepEdge"],
    ) -> List[TileInstruction]:
        if not op_records:
            return []

        depth = _compute_op_depths(len(op_records), edges)
        shared_dims = self._find_shared_dims(op_records, edges)

        if not shared_dims:
            return _depth_sorted_instructions(op_records, depth)

        # Map shared dims to tile axes for each op
        sorted_shared = sorted(shared_dims)
        op_group_axes: Dict[int, List[Optional[int]]] = {}
        for rec in op_records:
            op_group_axes[rec.op_idx] = [
                rec.op.dim_names.get(dim) for dim in sorted_shared
            ]

        # Build (group_key, -depth, op_idx, tile_idx) and sort
        entries = []
        for rec in op_records:
            axes = op_group_axes[rec.op_idx]
            op_depth = depth[rec.op_idx]
            for tile_idx, tile in enumerate(rec.tiles):
                group_key = tuple(
                    tile[ax] if ax is not None else 0 for ax in axes
                )
                entries.append(
                    (group_key, -op_depth, rec.op_idx, tile_idx, tile)
                )

        entries.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

        return [
            TileInstruction(op_idx=op_idx, tiles=tile)
            for _, _, op_idx, _, tile in entries
        ]


# Default scheduler instance
_default_scheduler: TileScheduler = BackwardScheduler()


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
                stride = math.prod(counts[i + 1 :]) if i + 1 < ndims else 1
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

    def _build_buffer_producers(self, *, strict: bool = False) -> Dict[str, int]:
        """Track latest producer op_idx per buffer name, respecting tensor identity.

        When two ops produce the same buffer name but target different tensors
        (different data_ptr), they are independent — the earlier producer is
        NOT overwritten.

        Args:
            strict: If True, raise ValueError when both ops lack tensor_ptrs
                (ambiguous duplicate). Used by _resolve_named_formulas.
        """
        buffer_producers: Dict[str, int] = {}
        for rec in self._op_records:
            for buf in rec.op.op_cls.OUTPUTS:
                if buf in buffer_producers:
                    prev_idx = buffer_producers[buf]
                    prev_ptr = self._op_records[prev_idx].op.tensor_ptrs.get(buf)
                    curr_ptr = rec.op.tensor_ptrs.get(buf)
                    if prev_ptr is not None and curr_ptr is not None and prev_ptr != curr_ptr:
                        continue  # Different tensors — independent
                    if strict and prev_ptr is None and curr_ptr is None:
                        prev_cls = self._op_records[prev_idx].op.op_cls.__name__
                        cur_cls = rec.op.op_cls.__name__
                        raise ValueError(
                            f"Buffer '{buf}' produced by both op {prev_idx} "
                            f"({prev_cls}) and op {rec.op_idx} ({cur_cls})"
                        )
                buffer_producers[buf] = rec.op_idx
        return buffer_producers

    def _build_tensor_ptr_deps(self) -> Dict[Tuple[int, str], int]:
        """Detect cross-buffer dependencies by tensor pointer identity.

        Maps (consumer_op_idx, consumer_input_buf) → producer_op_idx when
        a consumer's input shares the same data_ptr as a producer's output,
        even though their buffer names differ. E.g., RopeOp writes 'q'
        (ptr=k_flat) and PrepOp reads 'k' (ptr=k_f) where both are views
        of the same storage.
        """
        tensor_ptr_deps: Dict[Tuple[int, str], int] = {}
        for cons_rec in self._op_records:
            for cons_buf in cons_rec.op.op_cls.INPUTS:
                cons_ptr = cons_rec.op.tensor_ptrs.get(cons_buf)
                if cons_ptr is None:
                    continue
                for prod_rec in self._op_records:
                    if prod_rec.op_idx >= cons_rec.op_idx:
                        continue
                    for prod_buf in prod_rec.op.op_cls.OUTPUTS:
                        prod_ptr = prod_rec.op.tensor_ptrs.get(prod_buf)
                        if prod_ptr is not None and prod_ptr == cons_ptr:
                            tensor_ptr_deps[(cons_rec.op_idx, cons_buf)] = prod_rec.op_idx
        return tensor_ptr_deps

    def _find_producer(
        self,
        rec: "_OpRecord",
        buf: str,
        buffer_producers: Dict[str, int],
        tensor_ptr_deps: Dict[Tuple[int, str], int],
    ) -> Optional[int]:
        """Find the producer op_idx for a consumer's input buffer.

        Checks buffer name match first (skipping false matches where tensor
        ptrs differ), then falls back to tensor pointer identity.
        Returns None for external inputs or true self-dependencies.
        """
        prod_idx = buffer_producers.get(buf)
        # Skip false name matches (same name, different tensor)
        if prod_idx is not None and prod_idx != rec.op_idx:
            prod_ptr = self._op_records[prod_idx].op.tensor_ptrs.get(buf)
            cons_ptr = rec.op.tensor_ptrs.get(buf)
            if prod_ptr is not None and cons_ptr is not None and prod_ptr != cons_ptr:
                prod_idx = None
        # Fall back to tensor pointer identity for cross-buffer deps
        if prod_idx is None or prod_idx == rec.op_idx:
            ptr_prod = tensor_ptr_deps.get((rec.op_idx, buf))
            if ptr_prod is not None:
                prod_idx = ptr_prod
        if prod_idx is None or prod_idx == rec.op_idx:
            return None
        return prod_idx

    def _resolve_dep_edges(self) -> List[_DepEdge]:
        """Resolve op-level dependency edges for tile scheduling.

        Returns a list of _DepEdge with producer/consumer indices and
        dependency kind. Used by build() to determine tile emission order.
        """
        if not self._has_named_buffers():
            return [
                _DepEdge(producer_idx=i - 1, consumer_idx=i, kind="one_to_one")
                for i in range(1, len(self._op_records))
            ]

        buffer_producers = self._build_buffer_producers()
        tensor_ptr_deps = self._build_tensor_ptr_deps()

        edges: List[_DepEdge] = []
        seen: Set[Tuple[int, int]] = set()
        for rec in self._op_records:
            for buf in rec.op.op_cls.INPUTS:
                prod_idx = self._find_producer(rec, buf, buffer_producers, tensor_ptr_deps)
                if prod_idx is None:
                    continue
                pair = (prod_idx, rec.op_idx)
                if pair in seen:
                    continue
                seen.add(pair)

                producer = self._op_records[prod_idx]
                p_dims = set(producer.op.dim_names.keys())
                c_dims = set(rec.op.dim_names.keys())
                producer_only = p_dims - c_dims
                consumer_only = c_dims - p_dims

                if producer_only:
                    kind = "many_to_one"
                elif consumer_only:
                    kind = "one_to_many"
                else:
                    kind = "one_to_one"

                edges.append(_DepEdge(
                    producer_idx=prod_idx,
                    consumer_idx=rec.op_idx,
                    kind=kind,
                ))

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
            signal_formulas = [BarrierFormula(base=signal_base, coeffs=coeffs)]

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
        buffer_producers = self._build_buffer_producers(strict=True)
        tensor_ptr_deps = self._build_tensor_ptr_deps()

        formulas: Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]] = {
            i: ([], []) for i in range(len(self._op_records))
        }
        barrier_counter = 0

        for rec in self._op_records:
            for buf in rec.op.op_cls.INPUTS:
                prod_idx = self._find_producer(rec, buf, buffer_producers, tensor_ptr_deps)
                if prod_idx is None:
                    continue
                producer = self._op_records[prod_idx]
                consumer = rec

                p_op = producer.op
                c_op = consumer.op

                # Build canonical → original dim mappings (applying dim_aliases)
                # E.g., if consumer has dims {"M","H"} with alias {"M":"S"},
                # then canonical "S" maps to consumer's "M", matching producer's "S".
                p_canon_to_orig = {}
                for d in p_op.dim_names:
                    canon = p_op.dim_aliases.get(d, d)
                    p_canon_to_orig[canon] = d
                c_canon_to_orig = {}
                for d in c_op.dim_names:
                    canon = c_op.dim_aliases.get(d, d)
                    c_canon_to_orig[canon] = d

                shared_canonical = set(p_canon_to_orig.keys()) & set(c_canon_to_orig.keys())
                producer_only_canonical = set(p_canon_to_orig.keys()) - shared_canonical

                # Build shared_pairs: [(canonical, p_orig, c_orig), ...]
                shared_pairs = [
                    (canon, p_canon_to_orig[canon], c_canon_to_orig[canon])
                    for canon in shared_canonical
                ]
                producer_only_dims = [p_canon_to_orig[c] for c in producer_only_canonical]

                # Validate tile sizes on shared dims are divisible and
                # the tile-size ratio maps both sides to the same number
                # of barriers.  Non-divisible tile sizes or mismatched
                # effective tile counts cannot be expressed by
                # BarrierFormula integer division.  Move incompatible
                # dims to producer_only for conservative many-to-one
                # barriers.
                incompatible = []
                compatible_pairs = []
                for canon, p_dim, c_dim in shared_pairs:
                    p_ts = p_op.tile_sizes.get(p_dim, 1)
                    c_ts = c_op.tile_sizes.get(c_dim, 1)
                    p_axis = p_op.dim_names[p_dim]
                    c_axis = c_op.dim_names[c_dim]
                    p_tiles = p_op.tiles_for_axis(p_axis)
                    c_tiles = c_op.tiles_for_axis(c_axis)
                    if p_ts != c_ts and p_ts % c_ts != 0 and c_ts % p_ts != 0:
                        # Incompatible tile sizes (not evenly divisible)
                        incompatible.append(p_dim)
                    elif p_tiles != c_tiles and p_ts == c_ts:
                        # Same tile size but different extents → different
                        # semantics. E.g., GEMM_Q N=2048 vs GEMM_O N=1024
                        # both with tile_N=64. Consumer likely reads
                        # globally along this dim (reduction), so per-tile
                        # barriers are unsafe. Fall back to many-to-one.
                        incompatible.append(p_dim)
                    elif p_tiles != c_tiles and p_ts != c_ts:
                        # Different tile sizes AND different tile counts.
                        # Verify the tile-size ratio maps both sides to
                        # the same effective barrier count.  If not, the
                        # dim extents truly differ (e.g., QKNormRope H=2
                        # vs FA H=8) and per-tile barriers would produce
                        # out-of-bounds indices.
                        if p_ts > c_ts:
                            ratio = p_ts // c_ts
                            c_eff = (c_tiles + ratio - 1) // ratio
                            if c_eff != p_tiles:
                                incompatible.append(p_dim)
                            else:
                                compatible_pairs.append((canon, p_dim, c_dim))
                        else:
                            ratio = c_ts // p_ts
                            p_eff = (p_tiles + ratio - 1) // ratio
                            if p_eff != c_tiles:
                                incompatible.append(p_dim)
                            else:
                                compatible_pairs.append((canon, p_dim, c_dim))
                    else:
                        compatible_pairs.append((canon, p_dim, c_dim))
                shared_pairs = compatible_pairs
                producer_only_dims.extend(incompatible)

                # When named ops share no dimensions, all tiles map to the
                # same barrier (all-or-nothing). This is correct but suboptimal.

                # Determine target side and compute barrier count / expected
                # Also compute divisors for tile size ratio handling
                p_divs = [1] * MAX_TILE_DIMS  # Producer divisors
                c_divs = [1] * MAX_TILE_DIMS  # Consumer divisors
                expected = 1

                # Handle tile size ratios on shared dims
                for canon, p_dim, c_dim in shared_pairs:
                    p_axis = p_op.dim_names[p_dim]
                    c_axis = c_op.dim_names[c_dim]
                    p_ts = p_op.tile_sizes.get(p_dim, 1)
                    c_ts = c_op.tile_sizes.get(c_dim, 1)
                    if p_ts > c_ts:
                        ratio = p_ts // c_ts
                        c_divs[c_axis] = ratio
                    elif c_ts > p_ts:
                        ratio = c_ts // p_ts
                        p_divs[p_axis] = ratio
                        expected *= ratio

                # Barrier count: product of min tile counts on shared dims
                num_barriers = 1
                for canon, p_dim, c_dim in shared_pairs:
                    p_axis = p_op.dim_names[p_dim]
                    c_axis = c_op.dim_names[c_dim]
                    num_barriers *= min(
                        p_op.tiles_for_axis(p_axis),
                        c_op.tiles_for_axis(c_axis),
                    )

                if producer_only_dims:
                    # Many-to-one (or both-sides-extra): collapse
                    # producer-only dims — all producer tiles on those dims
                    # signal the same barrier, so consumer expects them all.
                    collapsed = 1
                    for dim in producer_only_dims:
                        axis = p_op.dim_names[dim]
                        collapsed *= p_op.tiles_for_axis(axis)
                    expected *= collapsed

                # Compute coefficients using dense shared strides.
                # Extra dims (producer-only / consumer-only) get coefficient 0
                # since they don't appear in shared_pairs. This correctly
                # handles different axis orderings and tile size differences.
                shared_pairs_sorted = sorted(shared_pairs, key=lambda t: t[0])
                shared_strides: Dict[str, int] = {}
                stride = 1
                for canon, p_dim, c_dim in reversed(shared_pairs_sorted):
                    shared_strides[canon] = stride
                    p_axis = p_op.dim_names[p_dim]
                    c_axis = c_op.dim_names[c_dim]
                    stride *= min(
                        p_op.tiles_for_axis(p_axis),
                        c_op.tiles_for_axis(c_axis),
                    )

                p_coeffs_list = [0] * MAX_TILE_DIMS
                for canon, p_dim, c_dim in shared_pairs:
                    p_coeffs_list[p_op.dim_names[p_dim]] = shared_strides[canon]
                p_coeffs = tuple(p_coeffs_list)

                c_coeffs_list = [0] * MAX_TILE_DIMS
                for canon, p_dim, c_dim in shared_pairs:
                    c_coeffs_list[c_op.dim_names[c_dim]] = shared_strides[canon]
                c_coeffs = tuple(c_coeffs_list)

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
                scheduler (BackwardScheduler).

        Returns:
            List of TileInstructions with END marker at the end.
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

    def build_tensor(self, device: str = "cuda", scheduler: Optional[TileScheduler] = None):
        """Build instruction stream as GPU tensor.

        Returns:
            Tensor of shape [num_instructions, INSTRUCTION_WORDS] where
            INSTRUCTION_WORDS=2 (op_idx + linear_tile_idx).
        """
        import torch

        instructions = self.build(scheduler=scheduler)
        # Precompute row-major strides per op for linearization
        strides_by_op = {
            r.op_idx: _linear_strides(r.op.tile_counts)
            for r in self._op_records
        }
        packed = [
            instr.pack(strides=strides_by_op.get(instr.op_idx))
            for instr in instructions
        ]
        return torch.tensor(packed, dtype=torch.int32, device=device)

    @property
    def max_wait_deps(self) -> int:
        """Maximum number of wait dependencies across all ops."""
        formulas, _ = self._resolve()
        return max((len(wf) for wf, _ in formulas.values()), default=0)

    def build_wait_info_tensor(self, instructions, device="cuda"):
        """Pre-compute (barrier_idx, expected) per instruction.

        Returns tensor [num_instr, max_waits * 2]:
        [wait0_barrier_idx, wait0_expected, wait1_barrier_idx, wait1_expected, ...]
        barrier_idx = -1 means skip.
        """
        import torch

        formulas, _ = self._resolve()
        max_waits = max(1, self.max_wait_deps)

        wait_data = []
        for instr in instructions:
            entry = []
            if instr.op_idx == TileInstruction.END_MARKER:
                entry = [-1, 0] * max_waits
            else:
                wait_formulas = formulas.get(instr.op_idx, ([], []))[0]
                for wf in wait_formulas:
                    if wf.has_guard and not wf.is_guarded(instr.tiles):
                        entry.extend([-1, 0])
                    else:
                        entry.extend([wf.compute_index(instr.tiles), wf.expected])
                while len(entry) < max_waits * 2:
                    entry.extend([-1, 0])
            wait_data.append(entry)

        return torch.tensor(wait_data, dtype=torch.int32, device=device)

    def build_decompose_tile_fn(self):
        """Build a @cute.jit function that decomposes (op_idx, linear_idx) → (t0..t4).

        Uses compile-time baked per-op tile_counts to perform integer
        div/mod decomposition. Pure ALU — no memory access.

        Returns:
            A @cute.jit function: decompose_tile(op_idx, linear_idx) → (t0, t1, t2, t3, t4)
        """
        import linecache
        import machete.megakernel.compile as compile_mod

        lines = []
        for rec in self._op_records:
            idx = rec.op_idx
            tc = rec.op.tile_counts
            ndims = len(tc)
            keyword = "if" if idx == 0 else "elif"

            # Build decomposition: divide linear_idx by strides
            # For tile_counts = (4, 3, 2): strides = (6, 2, 1)
            # t0 = linear_idx // 6; rem = linear_idx % 6
            # t1 = rem // 2; t2 = rem % 2
            body_lines = []
            if ndims == 1:
                body_lines.append("        t0 = _lin")
            else:
                remainder = "_lin"
                for d in range(ndims):
                    stride = 1
                    for k in range(d + 1, ndims):
                        stride *= tc[k]
                    if d == ndims - 1:
                        body_lines.append(f"        t{d} = {remainder}")
                    else:
                        body_lines.append(
                            f"        t{d} = {remainder} // Int32({stride})"
                        )
                        if d < ndims - 2:
                            body_lines.append(
                                f"        {remainder} = {remainder} % Int32({stride})"
                            )
                        else:
                            # Last remainder becomes the final dim
                            body_lines.append(
                                f"        t{d+1} = {remainder} % Int32({stride})"
                            )
                            break

            lines.append(
                f"    {keyword} op_idx == Int32({idx}):\n"
                + "\n".join(body_lines)
            )

        body = "\n".join(lines) if lines else "    pass"
        fn_source = (
            "@cute.jit\n"
            "def decompose_tile(op_idx, linear_idx):\n"
            "    t0 = Int32(0)\n"
            "    t1 = Int32(0)\n"
            "    t2 = Int32(0)\n"
            "    t3 = Int32(0)\n"
            "    t4 = Int32(0)\n"
            "    _lin = linear_idx\n"
            f"{body}\n"
            "    return t0, t1, t2, t3, t4\n"
        )

        exec_globals = {"cute": cute, "Int32": Int32}
        compile_mod._compile_counter += 1
        unique_filename = f"<decompose_tile>_{compile_mod._compile_counter}"
        linecache.cache[unique_filename] = (
            len(fn_source), None, fn_source.splitlines(True), unique_filename,
        )
        compile_mod._linecache_entries.append(unique_filename)

        code = compile(fn_source, unique_filename, "exec")
        exec(code, exec_globals)
        return exec_globals["decompose_tile"]

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
    "BarrierFormula",
    "INSTRUCTION_WORDS",
    "TileInstruction",
    "TileScheduler",
    "BackwardScheduler",
    "GroupedScheduler",
    "get_default_scheduler",
    "set_default_scheduler",
    "InstructionStreamBuilder",
]
