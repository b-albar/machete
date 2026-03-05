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
    # else: named ops with no shared dims → all zeros (all tiles map to same barrier)

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

    Algorithm:
    1. Compute depth per op via BFS from sinks
    2. Sort tiles by (-depth, op_idx, tile_idx)
    3. Emit in sorted order
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

        # Compute depth: longest path FROM this op TO any sink
        # Sinks have depth 0, their producers have depth 1, etc.
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

        # Handle disconnected ops
        max_depth = max(depth) if depth else 0
        for i in range(num_ops):
            if depth[i] < 0:
                depth[i] = max_depth + 1

        # Sort all tiles by (-depth, op_idx, tile_idx)
        # Higher depth = earlier in schedule (producers before consumers)
        instructions = []
        for rec in op_records:
            op_depth = depth[rec.op_idx]
            for tile_idx, tile in enumerate(rec.tiles):
                instructions.append((-op_depth, rec.op_idx, tile_idx, tile))

        instructions.sort(key=lambda x: (x[0], x[1], x[2]))

        return [
            TileInstruction(op_idx=op_idx, tiles=tile)
            for _, op_idx, _, tile in instructions
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

        # Named buffer dependencies — track latest producer per buffer name.
        # Multiple ops may produce the same buffer name (e.g., chained GEMMs
        # all writing "c" with different or ping-pong-reused data_ptrs).
        # Validation is handled by _resolve_named_formulas; here we just
        # need the latest producer for dependency edge resolution.
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

                if producer_only:
                    # Producer has extra dims (many-to-one on shared dims).
                    # Applies whether or not consumer also has extra dims.
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
        # Track buffer producers by name (latest writer wins).
        # When two ops produce the same buffer name but target different
        # tensors (different data_ptr), they are independent — skip.
        # When same data_ptr (ping-pong reuse), update to latest producer.
        # Safety: transitive dependencies via tensor_ptr_deps guarantee
        # the earlier write completes before the later write starts.
        buffer_producers: Dict[str, int] = {}
        for rec in self._op_records:
            for buf in rec.op.op_cls.OUTPUTS:
                if buf in buffer_producers:
                    prev_idx = buffer_producers[buf]
                    prev_ptr = self._op_records[prev_idx].op.tensor_ptrs.get(buf)
                    curr_ptr = rec.op.tensor_ptrs.get(buf)
                    if prev_ptr is not None and curr_ptr is not None and prev_ptr != curr_ptr:
                        # Different tensors — independent, no conflict
                        continue
                    # Same ptr (ping-pong reuse) or both have ptrs — update
                    # to latest producer. Transitive deps guarantee ordering.
                    if prev_ptr is None and curr_ptr is None:
                        # No tensor tracking — ambiguous duplicate
                        prev_cls = self._op_records[prev_idx].op.op_cls.__name__
                        cur_cls = rec.op.op_cls.__name__
                        raise ValueError(
                            f"Buffer '{buf}' produced by both op {prev_idx} "
                            f"({prev_cls}) and op {rec.op_idx} ({cur_cls})"
                        )
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

                # Handle non-divisible tile sizes on shared dims.
                # When tile sizes don't divide evenly (e.g., producer M=4,
                # consumer M=3), the integer division in BarrierFormula can't
                # express the correct per-tile mapping. Move such dims to
                # producer_only so all producer tiles on that dim get collapsed
                # into a single barrier group (conservative but correct).
                non_divisible = set()
                for dim in shared_dims:
                    p_ts = p_op.tile_sizes.get(dim, 1)
                    c_ts = c_op.tile_sizes.get(dim, 1)
                    if p_ts != c_ts and p_ts % c_ts != 0 and c_ts % p_ts != 0:
                        non_divisible.add(dim)
                if non_divisible:
                    shared_dims = shared_dims - non_divisible
                    producer_only = producer_only | non_divisible

                # Determine target side and compute barrier count / expected
                # Also compute divisors for tile size ratio handling
                p_divs = [1] * MAX_TILE_DIMS  # Producer divisors
                c_divs = [1] * MAX_TILE_DIMS  # Consumer divisors
                expected = 1

                if producer_only:
                    # Many-to-one: producer has extra dims not in consumer.
                    # Also handles both-sides-extra: consumer extra dims are
                    # broadcast (all consumer tiles with same shared-dim values
                    # wait on the same barrier, so extra consumer dims get
                    # coefficient 0 in the formula).

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

                    if consumer_only:
                        # Both-sides-extra: compute coefficients using
                        # shared-dim-only strides (not target_op strides)
                        # so consumer extra dims get coefficient 0.
                        shared_dims_sorted = sorted(shared_dims)
                        shared_strides: Dict[str, int] = {}
                        stride = 1
                        for dim in reversed(shared_dims_sorted):
                            shared_strides[dim] = stride
                            p_axis = p_op.dim_names[dim]
                            c_axis = c_op.dim_names[dim]
                            stride *= min(
                                p_op.tiles_for_axis(p_axis),
                                c_op.tiles_for_axis(c_axis),
                            )

                        p_coeffs_list = [0] * MAX_TILE_DIMS
                        for dim in shared_dims:
                            p_coeffs_list[p_op.dim_names[dim]] = shared_strides[dim]
                        p_coeffs = tuple(p_coeffs_list)

                        c_coeffs_list = [0] * MAX_TILE_DIMS
                        for dim in shared_dims:
                            c_coeffs_list[c_op.dim_names[dim]] = shared_strides[dim]
                        c_coeffs = tuple(c_coeffs_list)

                        # Skip _compute_formula_coeffs below
                        formulas[prod_idx][1].append(
                            BarrierFormula(
                                base=barrier_counter,
                                coeffs=p_coeffs,
                                divs=tuple(p_divs),
                            )
                        )
                        formulas[rec.op_idx][0].append(
                            BarrierFormula(
                                base=barrier_counter,
                                coeffs=c_coeffs,
                                divs=tuple(c_divs),
                                expected=expected,
                            )
                        )
                        barrier_counter += num_barriers
                        continue
                    else:
                        target_op = c_op

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

    def build_tensor(self, device: str = "cuda"):
        """Build instruction stream as GPU tensor.

        Returns:
            Tensor of shape [num_instructions, INSTRUCTION_WORDS] where
            INSTRUCTION_WORDS=2 (op_idx + linear_tile_idx).
        """
        import torch

        instructions = self.build()
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
    "get_default_scheduler",
    "set_default_scheduler",
    "InstructionStreamBuilder",
]
