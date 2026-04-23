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
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Set, Tuple, Type, Union

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

INSTRUCTION_WORDS = 2 + MAX_TILE_DIMS  # op_idx + linear_tile_idx + expanded tile coords


@dataclass
class TileInstruction:
    """A single tile work instruction for the persistent megakernel.

    Flat encoding in global memory (7 x int32):
    [0]  op_idx: Which operation (indexes into op list), or -1 for end marker
    [1]  linear_tile_idx: Row-major linearized tile index
    [2:] tile coordinates padded to MAX_TILE_DIMS

    Barrier wait/signal logic is baked into op handlers at compile time
    via BarrierFormula, not encoded in the instruction stream.
    """

    op_idx: int
    tiles: Tuple[int, ...]  # Up to MAX_TILE_DIMS tile indices

    # Sentinel for end of stream
    END_MARKER: int = -1

    def pack(self, strides: Optional[Tuple[int, ...]] = None) -> List[int]:
        """Pack into list of int32: [op_idx, linear_tile_idx, tile_0..tile_4].

        Args:
            strides: Row-major strides for linearization. If None (e.g., end
                marker), linear index is 0.
        """
        if self.op_idx == self.END_MARKER or strides is None:
            return [self.op_idx, 0] + [0] * MAX_TILE_DIMS
        linear = sum(t * s for t, s in zip(self.tiles, strides))
        padded_tiles = list(self.tiles) + [0] * (MAX_TILE_DIMS - len(self.tiles))
        return [self.op_idx, linear] + padded_tiles

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


def _decompose_linear_tile(linear_idx: int, tile_counts: Tuple[int, ...]) -> Tuple[int, ...]:
    """Decompose a row-major linear tile index into tile coordinates."""
    tile = []
    remainder = linear_idx
    for axis in range(len(tile_counts)):
        stride = math.prod(tile_counts[axis + 1 :]) if axis + 1 < len(tile_counts) else 1
        tile.append(remainder // stride)
        remainder %= stride
    return tuple(tile)


def _group_consumer_deps(edges: List["_DepEdge"]) -> Dict[int, List["_DepEdge"]]:
    """Group dependency edges by consumer op index."""
    consumer_deps: Dict[int, List[_DepEdge]] = {}
    for edge in edges:
        consumer_deps.setdefault(edge.consumer_idx, []).append(edge)
    return consumer_deps


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
        """Schedule tile instructions from the dependency graph."""
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
        """Interleave tile groups that share common dependency dimensions."""
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
        """Initialize an empty builder and clear cached dependency formulas."""
        self._op_records: List[_OpRecord] = []
        self._cached_formulas: Optional[Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]]] = None
        self._barrier_count: Optional[int] = None

    def _invalidate_resolution_cache(self) -> None:
        """Clear cached formulas after the op list changes."""
        self._cached_formulas = None
        self._barrier_count = None

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

        tiles = [
            _decompose_linear_tile(linear_idx, op.tile_counts)
            for linear_idx in range(op.total_tiles)
        ]

        record = _OpRecord(
            op_idx=len(self._op_records),
            op=op,
            tiles=tiles,
        )
        self._op_records.append(record)
        self._invalidate_resolution_cache()
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
        Named-buffer matching must only consider producers that appear
        strictly before the consumer; re-used buffer names across layers
        would otherwise create future-pointing edges and dependency cycles.
        Returns None for external inputs or true self-dependencies.
        """
        prod_idx = None
        cons_ptr = rec.op.tensor_ptrs.get(buf)

        # Search backwards so we pick the nearest prior producer of this
        # named buffer, never a future writer. This is still cheap for the
        # op counts we schedule, and it preserves correct chaining when the
        # same buffer name is re-used for ping-pong activations across layers.
        for cand_idx in range(rec.op_idx - 1, -1, -1):
            cand_rec = self._op_records[cand_idx]
            if buf not in cand_rec.op.op_cls.OUTPUTS:
                continue
            prod_ptr = cand_rec.op.tensor_ptrs.get(buf)
            if prod_ptr is not None and cons_ptr is not None and prod_ptr != cons_ptr:
                continue
            prod_idx = cand_idx
            break

        # Fall back to tensor pointer identity for cross-buffer deps
        if prod_idx is None or prod_idx == rec.op_idx:
            ptr_prod = tensor_ptr_deps.get((rec.op_idx, buf))
            if ptr_prod is not None:
                prod_idx = ptr_prod
        if prod_idx is None or prod_idx == rec.op_idx:
            return None
        return prod_idx

    def _resolve_named_dep_pairs(
        self,
        buffer_producers: Dict[str, int],
        tensor_ptr_deps: Dict[Tuple[int, str], int],
    ) -> List[Tuple[int, int]]:
        """Resolve ordered op pairs for RAW plus shared-storage anti-dependencies."""
        pairs: List[Tuple[int, int]] = []
        seen: Set[Tuple[int, int]] = set()

        # RAW edges from declared inputs.
        for rec in self._op_records:
            for buf in rec.op.op_cls.INPUTS:
                prod_idx = self._find_producer(rec, buf, buffer_producers, tensor_ptr_deps)
                if prod_idx is None:
                    continue
                pair = (prod_idx, rec.op_idx)
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)

        # Shared-storage anti-dependencies: a later writer must wait until all
        # earlier readers/writers of the same underlying tensor have finished,
        # even when the logical buffer names differ (e.g. q/c/o scratch reuse
        # across different op classes and layers).
        for rec in self._op_records:
            for out_name in rec.op.op_cls.OUTPUTS:
                out_ptr = rec.op.tensor_ptrs.get(out_name)
                if out_ptr is None:
                    continue
                for cand_idx in range(rec.op_idx - 1, -1, -1):
                    cand = self._op_records[cand_idx]
                    matched = False
                    for cand_name in cand.op.op_cls.INPUTS:
                        if cand.op.tensor_ptrs.get(cand_name) == out_ptr:
                            matched = True
                            break
                    if not matched:
                        for cand_name in cand.op.op_cls.OUTPUTS:
                            if cand.op.tensor_ptrs.get(cand_name) == out_ptr:
                                matched = True
                                break
                    if matched:
                        pair = (cand_idx, rec.op_idx)
                        if pair not in seen:
                            seen.add(pair)
                            pairs.append(pair)

        return pairs

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
        for prod_idx, cons_idx in self._resolve_named_dep_pairs(buffer_producers, tensor_ptr_deps):
            producer = self._op_records[prod_idx]
            consumer = self._op_records[cons_idx]
            p_dims = set(producer.op.dim_names.keys())
            c_dims = set(consumer.op.dim_names.keys())
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
                consumer_idx=cons_idx,
                kind=kind,
            ))

        return edges

    def _resolve_edges_and_consumers(self) -> Tuple[List["_DepEdge"], Dict[int, List["_DepEdge"]]]:
        """Resolve dependency edges and the consumer-grouped view used by schedulers."""
        edges = self._resolve_dep_edges()
        return edges, _group_consumer_deps(edges)

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

    def _canonical_dim_map(self, op: ScheduledOp) -> Dict[str, str]:
        """Map canonical dimension names back to the op's original dim names."""
        canon_to_orig = {}
        for dim_name in op.dim_names:
            canonical_name = op.dim_aliases.get(dim_name, dim_name)
            canon_to_orig[canonical_name] = dim_name
        return canon_to_orig

    def _shared_dim_pairs(
        self,
        producer: ScheduledOp,
        consumer: ScheduledOp,
    ) -> Tuple[List[Tuple[str, str, str]], List[str]]:
        """Resolve shared and producer-only dimensions between two ops."""
        producer_dims = self._canonical_dim_map(producer)
        consumer_dims = self._canonical_dim_map(consumer)
        shared_canonical = set(producer_dims) & set(consumer_dims)
        producer_only = set(producer_dims) - shared_canonical
        shared_pairs = [
            (canonical_name, producer_dims[canonical_name], consumer_dims[canonical_name])
            for canonical_name in shared_canonical
        ]
        producer_only_dims = [producer_dims[canonical_name] for canonical_name in producer_only]
        return shared_pairs, producer_only_dims

    def _compatible_shared_dim_pairs(
        self,
        producer: ScheduledOp,
        consumer: ScheduledOp,
        shared_pairs: List[Tuple[str, str, str]],
    ) -> Tuple[List[Tuple[str, str, str]], List[str]]:
        """Filter shared dims down to those representable by BarrierFormula."""
        incompatible_dims = []
        compatible_pairs = []

        for canonical_name, producer_dim, consumer_dim in shared_pairs:
            producer_tile_size = producer.tile_sizes.get(producer_dim, 1)
            consumer_tile_size = consumer.tile_sizes.get(consumer_dim, 1)
            producer_axis = producer.dim_names[producer_dim]
            consumer_axis = consumer.dim_names[consumer_dim]
            producer_tiles = producer.tiles_for_axis(producer_axis)
            consumer_tiles = consumer.tiles_for_axis(consumer_axis)

            if (
                producer_tile_size != consumer_tile_size
                and producer_tile_size % consumer_tile_size != 0
                and consumer_tile_size % producer_tile_size != 0
            ):
                incompatible_dims.append(producer_dim)
            elif producer_tiles != consumer_tiles and producer_tile_size == consumer_tile_size:
                incompatible_dims.append(producer_dim)
            elif producer_tiles != consumer_tiles and producer_tile_size != consumer_tile_size:
                if producer_tile_size > consumer_tile_size:
                    ratio = producer_tile_size // consumer_tile_size
                    consumer_effective_tiles = (consumer_tiles + ratio - 1) // ratio
                    if consumer_effective_tiles != producer_tiles:
                        incompatible_dims.append(producer_dim)
                    else:
                        compatible_pairs.append((canonical_name, producer_dim, consumer_dim))
                else:
                    ratio = consumer_tile_size // producer_tile_size
                    producer_effective_tiles = (producer_tiles + ratio - 1) // ratio
                    if producer_effective_tiles != consumer_tiles:
                        incompatible_dims.append(producer_dim)
                    else:
                        compatible_pairs.append((canonical_name, producer_dim, consumer_dim))
            else:
                compatible_pairs.append((canonical_name, producer_dim, consumer_dim))

        return compatible_pairs, incompatible_dims

    def _barrier_formula_params(
        self,
        producer: ScheduledOp,
        consumer: ScheduledOp,
        shared_pairs: List[Tuple[str, str, str]],
        producer_only_dims: List[str],
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, int]:
        """Compute divisor vectors plus expected/barrier counts for a dependency edge."""
        producer_divs = [1] * MAX_TILE_DIMS
        consumer_divs = [1] * MAX_TILE_DIMS
        expected = 1

        for _canonical_name, producer_dim, consumer_dim in shared_pairs:
            producer_axis = producer.dim_names[producer_dim]
            consumer_axis = consumer.dim_names[consumer_dim]
            producer_tile_size = producer.tile_sizes.get(producer_dim, 1)
            consumer_tile_size = consumer.tile_sizes.get(consumer_dim, 1)
            if producer_tile_size > consumer_tile_size:
                consumer_divs[consumer_axis] = producer_tile_size // consumer_tile_size
            elif consumer_tile_size > producer_tile_size:
                producer_divs[producer_axis] = consumer_tile_size // producer_tile_size
                expected *= consumer_tile_size // producer_tile_size

        num_barriers = 1
        for _canonical_name, producer_dim, consumer_dim in shared_pairs:
            producer_axis = producer.dim_names[producer_dim]
            consumer_axis = consumer.dim_names[consumer_dim]
            num_barriers *= min(
                producer.tiles_for_axis(producer_axis),
                consumer.tiles_for_axis(consumer_axis),
            )

        if producer_only_dims:
            collapsed = 1
            for producer_dim in producer_only_dims:
                collapsed *= producer.tiles_for_axis(producer.dim_names[producer_dim])
            expected *= collapsed

        return tuple(producer_divs), tuple(consumer_divs), expected, num_barriers

    def _barrier_formula_coeffs(
        self,
        producer: ScheduledOp,
        consumer: ScheduledOp,
        shared_pairs: List[Tuple[str, str, str]],
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Compute producer/consumer barrier-index coefficients for shared dims."""
        shared_pairs_sorted = sorted(shared_pairs, key=lambda item: item[0])
        shared_strides: Dict[str, int] = {}
        stride = 1
        for canonical_name, producer_dim, consumer_dim in reversed(shared_pairs_sorted):
            shared_strides[canonical_name] = stride
            producer_axis = producer.dim_names[producer_dim]
            consumer_axis = consumer.dim_names[consumer_dim]
            stride *= min(
                producer.tiles_for_axis(producer_axis),
                consumer.tiles_for_axis(consumer_axis),
            )

        producer_coeffs = [0] * MAX_TILE_DIMS
        consumer_coeffs = [0] * MAX_TILE_DIMS
        for canonical_name, producer_dim, consumer_dim in shared_pairs:
            producer_coeffs[producer.dim_names[producer_dim]] = shared_strides[canonical_name]
            consumer_coeffs[consumer.dim_names[consumer_dim]] = shared_strides[canonical_name]
        return tuple(producer_coeffs), tuple(consumer_coeffs)

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

        for prod_idx, cons_idx in self._resolve_named_dep_pairs(buffer_producers, tensor_ptr_deps):
                producer = self._op_records[prod_idx]
                consumer = self._op_records[cons_idx]

                p_op = producer.op
                c_op = consumer.op

                shared_pairs, producer_only_dims = self._shared_dim_pairs(p_op, c_op)
                shared_pairs, incompatible_dims = self._compatible_shared_dim_pairs(
                    p_op, c_op, shared_pairs
                )
                producer_only_dims.extend(incompatible_dims)

                producer_divs, consumer_divs, expected, num_barriers = self._barrier_formula_params(
                    p_op,
                    c_op,
                    shared_pairs,
                    producer_only_dims,
                )
                producer_coeffs, consumer_coeffs = self._barrier_formula_coeffs(
                    p_op,
                    c_op,
                    shared_pairs,
                )

                # Producer signal formula
                formulas[prod_idx][1].append(
                    BarrierFormula(
                        base=barrier_counter,
                        coeffs=producer_coeffs,
                        divs=producer_divs,
                    )
                )

                # Consumer wait formula
                formulas[cons_idx][0].append(
                    BarrierFormula(
                        base=barrier_counter,
                        coeffs=consumer_coeffs,
                        divs=consumer_divs,
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

        edges, consumer_deps = self._resolve_edges_and_consumers()

        # Use provided scheduler or default
        if scheduler is None:
            scheduler = get_default_scheduler()

        instructions = scheduler.schedule(self._op_records, consumer_deps, edges)
        instructions.append(TileInstruction.end_instruction())
        return instructions

    def build_tensor(
        self,
        device: str = "cuda",
        scheduler: Optional[TileScheduler] = None,
        instructions: Optional[List[TileInstruction]] = None,
    ):
        """Build instruction stream as GPU tensor.

        Returns:
            Tensor of shape [num_instructions, INSTRUCTION_WORDS] where
            INSTRUCTION_WORDS=2 + MAX_TILE_DIMS.
        """
        import torch

        if instructions is None:
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

        wait_data = [
            self._build_wait_info_entry(instr, formulas, max_waits)
            for instr in instructions
        ]

        return torch.tensor(wait_data, dtype=torch.int32, device=device)

    @staticmethod
    def _build_wait_info_entry(
        instr: TileInstruction,
        formulas: Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]],
        max_waits: int,
    ) -> List[int]:
        """Pack wait metadata for one instruction into `[barrier, expected, ...]`."""
        if instr.op_idx == TileInstruction.END_MARKER:
            return [-1, 0] * max_waits

        entry: List[int] = []
        wait_formulas = formulas.get(instr.op_idx, ([], []))[0]
        for formula in wait_formulas:
            if formula.has_guard and not formula.is_guarded(instr.tiles):
                entry.extend([-1, 0])
            else:
                entry.extend([formula.compute_index(instr.tiles), formula.expected])

        while len(entry) < max_waits * 2:
            entry.extend([-1, 0])
        return entry

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
