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

from .instruction_layout import (
    INSTR_BARRIER_META_IDX,
    INSTR_OP_IDX,
    INSTR_TILE_0,
    INSTR_TILE_1,
    INSTR_TILE_2,
    INSTR_TILE_3,
    INSTR_TILE_4,
    INSTRUCTION_WORDS,
)
from .ops import MAX_TILE_DIMS, Op, ScheduledOp, last_dim_slice_region, tensor_meta_overlaps


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
        guard_min/guard_max: Only execute when the computed linear index is
            within this half-open interval. Defaults to [0, NO_GUARD), which
            always passes for normal tile coordinates.
    """

    # Sentinel: guard_max value that always passes (larger than any tile count)
    NO_GUARD: ClassVar[int] = 2**30

    base: int
    coeffs: Tuple[int, ...] = (0,) * MAX_TILE_DIMS
    divs: Tuple[int, ...] = (1,) * MAX_TILE_DIMS
    offset: int = 0
    expected: int = 1
    guard_min: int = 0
    guard_max: int = NO_GUARD
    guard_coeffs: Optional[Tuple[int, ...]] = None

    def compute_index(self, tiles: Tuple[int, ...]) -> int:
        """Compute barrier index for a given tile (host-side, for testing)."""
        padded = tuple(tiles) + (0,) * (MAX_TILE_DIMS - len(tiles))
        result = self.base + self.offset
        for i in range(MAX_TILE_DIMS):
            result += self.coeffs[i] * (padded[i] // self.divs[i])
        return result

    def is_guarded(self, tiles: Tuple[int, ...]) -> bool:
        """Check if the guard passes for a given tile (host-side, for testing)."""
        padded = tuple(tiles) + (0,) * (MAX_TILE_DIMS - len(tiles))
        guard_coeffs = self.guard_coeffs if self.guard_coeffs is not None else self.coeffs
        linear = sum(guard_coeffs[i] * padded[i] for i in range(MAX_TILE_DIMS))
        return self.guard_min <= linear < self.guard_max

    @property
    def has_guard(self) -> bool:
        """Whether this formula has an active guard (not NO_GUARD)."""
        return self.guard_min != 0 or self.guard_max != self.NO_GUARD


# =============================================================================
# Instruction Stream (Lightweight — barriers baked into handlers)
# =============================================================================

@dataclass
class TileInstruction:
    """A single tile work instruction for the persistent megakernel.

    Flat encoding in global memory:
    [0]  op_idx: Which operation (indexes into op list), or -1 for end marker
    [1:] tile coordinates followed by the barrier metadata row index

    Barrier wait/signal formulas stay in compact side tensors. The instruction
    carries the row index into those side tensors so replay does not need to
    derive it from scheduler shape in the hot loop.
    """

    op_idx: int
    tiles: Tuple[int, ...]  # Up to MAX_TILE_DIMS tile indices

    # Sentinel for end of stream
    END_MARKER: int = -1

    def pack(
        self,
        barrier_meta_idx: int = 0,
    ) -> List[int]:
        """Pack into the replay instruction row.

        Args:
            barrier_meta_idx: Row in the wait/signal side metadata tensors.
        """
        if self.op_idx == self.END_MARKER:
            return [self.op_idx] + [0] * (INSTRUCTION_WORDS - 1)
        padded_tiles = list(self.tiles) + [0] * (MAX_TILE_DIMS - len(self.tiles))
        return [
            self.op_idx,
            *padded_tiles,
            barrier_meta_idx,
            0,
        ]

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
    producer_buffer: str = ""
    consumer_buffer: str = ""


@dataclass(frozen=True)
class _DepPair:
    """Resolved logical dependency between two scheduled ops."""

    producer_idx: int
    consumer_idx: int
    producer_buffer: str = ""
    consumer_buffer: str = ""


@dataclass(frozen=True)
class _LastDimRegionDep:
    """Producer-only tiled dimension grouped by a consumer's last-dim slice."""

    producer_dim: str
    group_index: int
    group_count: int
    group_tiles: int

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


def _coalesce_pipeline_ranges(
    instructions: List[TileInstruction],
    *,
    range_axis: int,
    range_end_axis: int,
    max_range_tiles: int = 0,
    range_stride: int = 1,
    range_stride_axis: int = -1,
) -> List[TileInstruction]:
    """Coalesce single-block instructions into contiguous or strided ranges."""
    if not instructions:
        return []
    if range_axis < 0:
        return instructions
    if range_end_axis < 0:
        range_end_axis = range_axis + 1
    if range_end_axis >= MAX_TILE_DIMS or range_end_axis == range_axis:
        return instructions
    if range_stride < 1:
        range_stride = 1
    if range_stride == 1:
        range_stride_axis = -1
    elif range_stride_axis < 0:
        range_stride_axis = range_end_axis + 1
    if range_stride > 1 and (
        range_stride_axis >= MAX_TILE_DIMS
        or range_stride_axis in (range_axis, range_end_axis)
    ):
        return instructions

    if range_stride > 1:
        groups: Dict[Tuple[int, ...], List[Tuple[int, TileInstruction]]] = {}
        for original_idx, instr in enumerate(instructions):
            tiles = list(instr.tiles) + [0] * (MAX_TILE_DIMS - len(instr.tiles))
            key = []
            for axis in range(MAX_TILE_DIMS):
                if axis in (range_axis, range_end_axis, range_stride_axis):
                    continue
                key.append(tiles[axis])
            key.append(tiles[range_axis] % range_stride)
            groups.setdefault(tuple(key), []).append((original_idx, instr))

        out_with_order: List[Tuple[int, TileInstruction]] = []
        for group in groups.values():
            group.sort(key=lambda item: (item[1].tiles[range_axis], item[0]))
            idx = 0
            while idx < len(group):
                first_order, first = group[idx]
                tiles = list(first.tiles) + [0] * (MAX_TILE_DIMS - len(first.tiles))
                start = tiles[range_axis]
                count = 1
                idx += 1
                while idx < len(group):
                    _, candidate = group[idx]
                    cand_tiles = list(candidate.tiles) + [0] * (
                        MAX_TILE_DIMS - len(candidate.tiles)
                    )
                    if max_range_tiles > 0 and count >= max_range_tiles:
                        break
                    if cand_tiles[range_axis] != start + count * range_stride:
                        break
                    count += 1
                    idx += 1
                tiles[range_end_axis] = count
                tiles[range_stride_axis] = range_stride
                out_with_order.append((first_order, TileInstruction(first.op_idx, tuple(tiles))))
        out_with_order.sort(key=lambda item: item[0])
        return [instr for _, instr in out_with_order]

    out: List[TileInstruction] = []
    idx = 0
    while idx < len(instructions):
        first = instructions[idx]
        tiles = list(first.tiles) + [0] * (MAX_TILE_DIMS - len(first.tiles))
        start = tiles[range_axis]
        existing_end = tiles[range_end_axis]
        end = existing_end if existing_end > start else start + 1
        idx += 1

        while idx < len(instructions):
            candidate = instructions[idx]
            cand_tiles = list(candidate.tiles) + [0] * (MAX_TILE_DIMS - len(candidate.tiles))
            if max_range_tiles > 0 and end - start >= max_range_tiles:
                break
            same_prefix = True
            for axis in range(MAX_TILE_DIMS):
                if axis in (range_axis, range_end_axis):
                    continue
                if cand_tiles[axis] != tiles[axis]:
                    same_prefix = False
                    break
            if not same_prefix or cand_tiles[range_axis] != end:
                break
            end += 1
            idx += 1

        tiles[range_end_axis] = end
        out.append(TileInstruction(op_idx=first.op_idx, tiles=tuple(tiles)))

    return out


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


class OverlapTileScheduler(TileScheduler):
    """Readiness-aware scheduler for load/compute/store overlap.

    ``BackwardScheduler`` emits broad op waves: all producer tiles, then their
    consumers.  That minimizes runtime waits but can leave the persistent
    loader/store/compute pipeline with long stretches of similar work.

    This scheduler simulates the compiled barrier formulas on the host and
    emits only tiles whose waits are already satisfied. It schedules ready work
    in fetch-stride-sized waves so persistent CTAs do not receive same-round
    producer/consumer dependencies, while still allowing data-movement-heavy
    ops to be prioritized within each safe wave.
    """

    def __init__(
        self,
        *,
        fetch_stride: Optional[int] = None,
        prefer_data_movement: bool = True,
        prefer_ready_consumers: bool = False,
    ):
        self.fetch_stride = fetch_stride
        self.prefer_data_movement = prefer_data_movement
        self.prefer_ready_consumers = prefer_ready_consumers

    def bind_num_blocks(self, num_blocks: int) -> None:
        """Bind the runtime instruction fetch stride used by persistent CTAs."""
        if self.fetch_stride is None:
            self.fetch_stride = max(1, int(num_blocks))

    @staticmethod
    def _phase_overridden(op_cls: Type[Op], phase_name: str) -> bool:
        if getattr(op_cls, f"{phase_name}_phase", None) is not None:
            return True
        return getattr(op_cls, phase_name, None) is not getattr(Op, phase_name, None)

    @classmethod
    def _resource_score(cls, op: ScheduledOp) -> int:
        op_cls = op.op_cls
        score = 0
        if getattr(op_cls, "pipeline", None) is not None:
            score += 3
        if cls._phase_overridden(op_cls, "load"):
            score += 2
        if cls._phase_overridden(op_cls, "store"):
            score += 1
        if cls._phase_overridden(op_cls, "communicate"):
            score += 1
        if getattr(op_cls, "_TMA_LOADS", set()) or getattr(op_cls, "_TMA_COMPUTE_LOADS", set()):
            score += 1
        if (
            getattr(op_cls, "_TMA_STORES", set())
            or getattr(op_cls, "_TMA_REDUCE_STORES", set())
            or getattr(op_cls, "_PEER_STORES", set())
            or getattr(op_cls, "_PEER_REDUCE_STORES", set())
        ):
            score += 1
        return score

    @staticmethod
    def _waits_ready(
        instr: TileInstruction,
        wait_formulas: List[BarrierFormula],
        barrier_counts: Dict[int, int],
    ) -> bool:
        for formula in wait_formulas:
            if formula.has_guard and not formula.is_guarded(instr.tiles):
                continue
            barrier_idx = formula.compute_index(instr.tiles)
            if barrier_counts.get(barrier_idx, 0) < formula.expected:
                return False
        return True

    @staticmethod
    def _signal(
        instr: TileInstruction,
        signal_formulas: List[BarrierFormula],
        barrier_counts: Dict[int, int],
    ) -> None:
        for formula in signal_formulas:
            if formula.has_guard and not formula.is_guarded(instr.tiles):
                continue
            barrier_idx = formula.compute_index(instr.tiles)
            barrier_counts[barrier_idx] = barrier_counts.get(barrier_idx, 0) + 1

    def _priority(
        self,
        *,
        op_idx: int,
        tile_idx: int,
        pos: int,
        depths: List[int],
        resource_scores: List[int],
    ) -> Tuple[int, int, int, int, int]:
        resource_score = resource_scores[op_idx] if self.prefer_data_movement else 0
        depth_score = -depths[op_idx] if self.prefer_ready_consumers else depths[op_idx]
        return (
            depth_score,
            -op_idx,
            resource_score,
            -tile_idx,
            -pos,
        )

    def _schedule_stride_waves(
        self,
        candidates: List[Tuple[int, int, TileInstruction]],
        formulas: Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]],
        depths: List[int],
        resource_scores: List[int],
        fetch_stride: int,
    ) -> List[TileInstruction]:
        scheduled: List[TileInstruction] = []
        barrier_counts: Dict[int, int] = {}

        while candidates:
            ready = []
            for pos, (op_idx, tile_idx, instr) in enumerate(candidates):
                wait_formulas = formulas.get(op_idx, ([], []))[0]
                if self._waits_ready(instr, wait_formulas, barrier_counts):
                    ready.append((
                        *self._priority(
                            op_idx=op_idx,
                            tile_idx=tile_idx,
                            pos=pos,
                            depths=depths,
                            resource_scores=resource_scores,
                        ),
                        pos,
                    ))

            if not ready:
                raise RuntimeError(
                    "OverlapTileScheduler could not find a ready tile. "
                    "This usually indicates a cyclic or unsatisfied dependency graph."
                )

            ready.sort(reverse=True)
            selected = ready[:fetch_stride]
            selected_positions = [entry[-1] for entry in selected]
            selected_records = [(pos, candidates[pos]) for pos in selected_positions]

            # Remove after recording so priority order is independent of index
            # shifts. Signals are applied after selecting the whole wave; this
            # prevents same-fetch-round dependencies from being introduced.
            for pos in sorted(selected_positions, reverse=True):
                candidates.pop(pos)

            for _pos, (op_idx, _tile_idx, instr) in selected_records:
                scheduled.append(instr)
            for _pos, (op_idx, _tile_idx, instr) in selected_records:
                signal_formulas = formulas.get(op_idx, ([], []))[1]
                self._signal(instr, signal_formulas, barrier_counts)

        return scheduled

    def schedule_with_formulas(
        self,
        op_records: List["_OpRecord"],
        edges: List["_DepEdge"],
        formulas: Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]],
    ) -> List[TileInstruction]:
        if not op_records:
            return []

        depths = _compute_op_depths(len(op_records), edges)
        resource_scores = [self._resource_score(rec.op) for rec in op_records]
        candidates: List[Tuple[int, int, TileInstruction]] = []
        for rec in op_records:
            for tile_idx, tile in enumerate(rec.tiles):
                candidates.append((rec.op_idx, tile_idx, TileInstruction(rec.op_idx, tile)))

        fetch_stride = max(1, int(self.fetch_stride or 1))
        return self._schedule_stride_waves(
            candidates,
            formulas,
            depths,
            resource_scores,
            fetch_stride,
        )

    def schedule(
        self,
        op_records: List["_OpRecord"],
        consumer_deps: Dict[int, List["_DepEdge"]],
        edges: List["_DepEdge"],
    ) -> List[TileInstruction]:
        """Fallback when barrier formulas are unavailable.

        ``InstructionStreamBuilder`` calls ``schedule_with_formulas`` for this
        scheduler.  This method keeps the abstract interface complete.
        """
        return BackwardScheduler().schedule(op_records, consumer_deps, edges)


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
        self._op_wait_counts: Dict[int, int] = {}
        self._op_signal_counts: Dict[int, int] = {}

    def _invalidate_resolution_cache(self) -> None:
        """Clear cached formulas after the op list changes."""
        self._cached_formulas = None
        self._barrier_count = None

    def _range_axis_for_op_idx(self, op_idx: int) -> Tuple[int, int]:
        pipeline = self._pipeline_for_op_idx(op_idx)
        if pipeline is None or not pipeline.coalesce_ranges or pipeline.range_axis < 0:
            return -1, -1
        end_axis = pipeline.range_end_axis
        if end_axis < 0:
            end_axis = pipeline.range_axis + 1
        if end_axis >= MAX_TILE_DIMS or end_axis == pipeline.range_axis:
            return -1, -1
        return pipeline.range_axis, end_axis

    def _range_stride_for_op_idx(self, op_idx: int) -> Tuple[int, int]:
        if op_idx < 0 or op_idx >= len(self._op_records):
            return 1, -1
        static_dims = self._op_records[op_idx].op.static_dims
        stride = int(static_dims.get("pipeline_range_stride", 1))
        stride_axis = int(static_dims.get("pipeline_range_stride_axis", -1))
        if stride < 1:
            stride = 1
        if stride == 1:
            return 1, -1
        return stride, stride_axis

    def _logical_tiles_for_instruction(self, instr: TileInstruction) -> List[Tuple[int, ...]]:
        """Return logical tiles covered by one possibly range-owned instruction."""
        if instr.op_idx == TileInstruction.END_MARKER:
            return []
        range_axis, end_axis = self._range_axis_for_op_idx(instr.op_idx)
        if range_axis < 0:
            return [instr.tiles]
        range_stride, stride_axis = self._range_stride_for_op_idx(instr.op_idx)
        tiles = list(instr.tiles) + [0] * (MAX_TILE_DIMS - len(instr.tiles))
        start = tiles[range_axis]
        if range_stride > 1:
            count = tiles[end_axis]
            if count <= 1:
                return [instr.tiles]
            end = start + count * range_stride
        else:
            end = tiles[end_axis]
            if end <= start:
                return [instr.tiles]
        out: List[Tuple[int, ...]] = []
        for axis_value in range(start, end, range_stride):
            logical = list(tiles)
            logical[range_axis] = axis_value
            logical[end_axis] = 0
            if stride_axis >= 0 and stride_axis < MAX_TILE_DIMS:
                logical[stride_axis] = 0
            out.append(tuple(logical))
        return out

    @property
    def ops(self) -> List[ScheduledOp]:
        """Scheduled ops in order."""
        return [r.op for r in self._op_records]

    def _pipeline_for_op_idx(self, op_idx: int):
        op = self._op_records[op_idx].op
        op_pipeline = getattr(op.op_cls, "pipeline", None)
        if op_pipeline is not None:
            return op_pipeline.with_overrides(
                page_count=op.static_dims.get("pipeline_page_count"),
                page_bytes=op.static_dims.get("pipeline_page_bytes"),
                semaphore_count=op.static_dims.get("pipeline_semaphore_count"),
                scratch_bytes=op.static_dims.get("pipeline_scratch_bytes"),
                input_stages=op.static_dims.get("pipeline_input_stages"),
                output_stages=op.static_dims.get("pipeline_output_stages"),
                stage_pages=op.static_dims.get("pipeline_stage_pages"),
                range_axis=op.static_dims.get("pipeline_range_axis"),
                range_end_axis=op.static_dims.get("pipeline_range_end_axis"),
                range_block_size=op.static_dims.get("pipeline_range_block_size"),
                coalesce_ranges=op.static_dims.get("pipeline_coalesce_ranges"),
            )
        return None

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
        a consumer's input overlaps a producer's output, even though their
        buffer names differ. Exact pointer matches remain supported, and we
        also handle packed-buffer aliases such as a fused QKV output consumed
        through separate q/k/v slice views.
        """
        tensor_ptr_deps: Dict[Tuple[int, str], int] = {}
        for cons_rec in self._op_records:
            for cons_buf in cons_rec.op.op_cls.INPUTS:
                cons_meta = cons_rec.op.tensor_metas.get(cons_buf)
                if cons_meta is None:
                    continue
                for prod_rec in self._op_records:
                    if prod_rec.op_idx >= cons_rec.op_idx:
                        continue
                    for prod_buf in prod_rec.op.op_cls.OUTPUTS:
                        prod_meta = prod_rec.op.tensor_metas.get(prod_buf)
                        if prod_meta is not None and tensor_meta_overlaps(prod_meta, cons_meta):
                            tensor_ptr_deps[(cons_rec.op_idx, cons_buf)] = prod_rec.op_idx
        return tensor_ptr_deps

    def _find_producer_buffer(
        self,
        prod_idx: int,
        consumer: ScheduledOp,
        consumer_buf: str,
    ) -> str:
        """Return the producer output buffer that backs a consumer input."""
        prod = self._op_records[prod_idx].op
        if consumer_buf in prod.op_cls.OUTPUTS:
            prod_ptr = prod.tensor_ptrs.get(consumer_buf)
            cons_ptr = consumer.tensor_ptrs.get(consumer_buf)
            if prod_ptr is None or cons_ptr is None or prod_ptr == cons_ptr:
                return consumer_buf

        cons_meta = consumer.tensor_metas.get(consumer_buf)
        if cons_meta is not None:
            for prod_buf in prod.op_cls.OUTPUTS:
                prod_meta = prod.tensor_metas.get(prod_buf)
                if prod_meta is not None and tensor_meta_overlaps(prod_meta, cons_meta):
                    return prod_buf

        return consumer_buf

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

    @staticmethod
    def _is_reduce_store(op: ScheduledOp, buf: str) -> bool:
        """Whether this output is accumulated atomically rather than overwritten."""
        return buf in getattr(op.op_cls, "_TMA_REDUCE_STORES", set())

    def _resolve_named_dep_pairs(
        self,
        buffer_producers: Dict[str, int],
        tensor_ptr_deps: Dict[Tuple[int, str], int],
    ) -> List[_DepPair]:
        """Resolve ordered op pairs for RAW plus shared-storage anti-dependencies."""
        pairs: List[_DepPair] = []
        seen: Set[Tuple[int, int, str, str]] = set()
        raw_pairs: Set[Tuple[int, int]] = set()

        # RAW edges from declared inputs.
        for rec in self._op_records:
            for buf in rec.op.op_cls.INPUTS:
                prod_idx = self._find_producer(rec, buf, buffer_producers, tensor_ptr_deps)
                if prod_idx is None:
                    continue
                prod_buf = self._find_producer_buffer(prod_idx, rec.op, buf)
                seen_key = (prod_idx, rec.op_idx, prod_buf, buf)
                if seen_key not in seen:
                    seen.add(seen_key)
                    raw_pairs.add((prod_idx, rec.op_idx))
                    pairs.append(
                        _DepPair(
                            producer_idx=prod_idx,
                            consumer_idx=rec.op_idx,
                            producer_buffer=prod_buf,
                            consumer_buffer=buf,
                        )
                    )

                # Split-K / row-parallel style reductions have multiple
                # associative writers for one logical tensor. The latest writer
                # is not sufficient as a RAW dependency: consumers need all
                # prior reduce stores that overlap their input.
                cons_meta = rec.op.tensor_metas.get(buf)
                if cons_meta is not None:
                    for cand_idx in range(rec.op_idx - 1, -1, -1):
                        cand = self._op_records[cand_idx]
                        added_reduce = False
                        for cand_name in cand.op.op_cls.OUTPUTS:
                            if not self._is_reduce_store(cand.op, cand_name):
                                continue
                            cand_meta = cand.op.tensor_metas.get(cand_name)
                            if cand_meta is not None and tensor_meta_overlaps(cand_meta, cons_meta):
                                seen_key = (cand_idx, rec.op_idx, cand_name, buf)
                                if seen_key not in seen:
                                    seen.add(seen_key)
                                    raw_pairs.add((cand_idx, rec.op_idx))
                                    pairs.append(
                                        _DepPair(
                                            producer_idx=cand_idx,
                                            consumer_idx=rec.op_idx,
                                            producer_buffer=cand_name,
                                            consumer_buffer=buf,
                                        )
                                    )
                                added_reduce = True
                        if added_reduce:
                            continue

        # Shared-storage anti-dependencies: a later writer must wait until all
        # earlier readers/writers of the same underlying tensor have finished,
        # even when the logical buffer names differ (e.g. q/c/o scratch reuse
        # across different op classes and layers).
        for rec in self._op_records:
            for out_name in rec.op.op_cls.OUTPUTS:
                out_meta = rec.op.tensor_metas.get(out_name)
                if out_meta is None:
                    continue
                for cand_idx in range(rec.op_idx - 1, -1, -1):
                    cand = self._op_records[cand_idx]
                    matched = False
                    cand_buf = ""
                    for cand_name in cand.op.op_cls.INPUTS:
                        cand_meta = cand.op.tensor_metas.get(cand_name)
                        if cand_meta is not None and tensor_meta_overlaps(cand_meta, out_meta):
                            matched = True
                            cand_buf = cand_name
                            break
                    if not matched:
                        for cand_name in cand.op.op_cls.OUTPUTS:
                            cand_meta = cand.op.tensor_metas.get(cand_name)
                            if cand_meta is not None and tensor_meta_overlaps(cand_meta, out_meta):
                                matched = True
                                cand_buf = cand_name
                                break
                    if matched:
                        # Atomic-reduce stores to the same output are
                        # associative and may overlap. Consumers get explicit
                        # RAW edges to all reduce writers above.
                        if any(
                            self._is_reduce_store(rec.op, out_name)
                            and self._is_reduce_store(cand.op, cand_name)
                            and rec.op.tensor_metas.get(out_name) is not None
                            and cand.op.tensor_metas.get(cand_name) is not None
                            and tensor_meta_overlaps(
                                rec.op.tensor_metas[out_name],
                                cand.op.tensor_metas[cand_name],
                            )
                            for out_name in rec.op.op_cls.OUTPUTS
                            for cand_name in cand.op.op_cls.OUTPUTS
                        ):
                            continue
                        seen_key = (cand_idx, rec.op_idx, cand_buf, out_name)
                        if seen_key not in seen:
                            seen.add(seen_key)
                            pairs.append(
                                _DepPair(
                                    producer_idx=cand_idx,
                                    consumer_idx=rec.op_idx,
                                    producer_buffer=cand_buf,
                                    consumer_buffer=out_name,
                                )
                            )

        return self._transitively_reduce_dep_pairs(pairs, preserve=raw_pairs)

    @staticmethod
    def _transitively_reduce_dep_pairs(
        pairs: List[_DepPair],
        preserve: Optional[Set[Tuple[int, int]]] = None,
    ) -> List[_DepPair]:
        """Remove dependency pairs already implied by another op-level path.

        Shared-storage anti-dependencies can connect a later writer to every
        earlier reader/writer of the same scratch tensor. In long fused graphs
        that creates O(layers^2) wait formulas even though most are implied by
        the nearest intervening dependencies. Keeping only the op-level
        transitive reduction preserves ordering while keeping per-instruction
        wait metadata proportional to the real frontier.
        """
        if len(pairs) <= 1:
            return pairs
        preserve = preserve or set()

        adjacency: Dict[int, Set[int]] = {}
        for pair in pairs:
            src, dst = pair.producer_idx, pair.consumer_idx
            adjacency.setdefault(src, set()).add(dst)

        def has_alternate_path(src: int, dst: int) -> bool:
            stack = [node for node in adjacency.get(src, ()) if node != dst]
            visited: Set[int] = set()
            while stack:
                node = stack.pop()
                if node == dst:
                    return True
                if node in visited:
                    continue
                visited.add(node)
                stack.extend(adjacency.get(node, ()))
            return False

        return [
            pair
            for pair in pairs
            if (pair.producer_idx, pair.consumer_idx) in preserve
            or not has_alternate_path(pair.producer_idx, pair.consumer_idx)
        ]

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
        for dep in self._resolve_named_dep_pairs(buffer_producers, tensor_ptr_deps):
            prod_idx = dep.producer_idx
            cons_idx = dep.consumer_idx
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
                producer_buffer=dep.producer_buffer,
                consumer_buffer=dep.consumer_buffer,
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

    @staticmethod
    def _buffer_static_keys(prefix: str, role: str, buffer_name: str, suffix: str) -> Tuple[str, ...]:
        if buffer_name:
            return (
                f"{prefix}_{role}_{buffer_name}_{suffix}",
                f"{prefix}_{buffer_name}_{role}_{suffix}",
            )
        return ()

    def _static_dim_lookup(
        self,
        op: ScheduledOp,
        *,
        prefix: str,
        role: str,
        buffer_name: str,
        suffix: str,
        fallback,
    ):
        for key in self._buffer_static_keys(prefix, role, buffer_name, suffix):
            if key in op.static_dims:
                return op.static_dims[key]
        return op.static_dims.get(f"{prefix}_{role}_{suffix}", fallback)

    def _canonical_dim_map(self, op: ScheduledOp, role: str, buffer_name: str = "") -> Dict[str, str]:
        """Map role-specific canonical dimension names to original dim names."""
        canon_to_orig = {}
        declared_dims = None
        if buffer_name:
            meta = op.tensor_metas.get(buffer_name)
            if meta is not None:
                declared_dims = set(meta.declared_dims)
        for dim_name in op.dim_names:
            has_explicit_alias = dim_name in op.dim_aliases
            suffix = f"alias_{dim_name}"
            if any(key in op.static_dims for key in self._buffer_static_keys("barrier", role, buffer_name, suffix)):
                has_explicit_alias = True
            if f"barrier_{role}_{suffix}" in op.static_dims:
                has_explicit_alias = True
            if declared_dims is not None and dim_name not in declared_dims and not has_explicit_alias:
                continue
            canonical_name = self._static_dim_lookup(
                op,
                prefix="barrier",
                role=role,
                buffer_name=buffer_name,
                suffix=f"alias_{dim_name}",
                fallback=op.dim_aliases.get(dim_name, dim_name),
            )
            canon_to_orig[canonical_name] = dim_name
        return canon_to_orig

    def _shared_dim_pairs(
        self,
        producer: ScheduledOp,
        consumer: ScheduledOp,
        producer_buffer: str = "",
        consumer_buffer: str = "",
    ) -> Tuple[List[Tuple[str, str, str]], List[str]]:
        """Resolve shared and producer-only dimensions between two ops."""
        producer_dims = self._canonical_dim_map(producer, "signal", producer_buffer)
        consumer_dims = self._canonical_dim_map(consumer, "wait", consumer_buffer)
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
        producer_buffer: str = "",
        consumer_buffer: str = "",
    ) -> Tuple[List[Tuple[str, str, str]], List[str]]:
        """Filter shared dims down to those representable by BarrierFormula."""
        incompatible_dims = []
        compatible_pairs = []

        for canonical_name, producer_dim, consumer_dim in shared_pairs:
            producer_tile_size = self._barrier_tile_size(producer, producer_dim, "signal", producer_buffer)
            consumer_tile_size = self._barrier_tile_size(consumer, consumer_dim, "wait", consumer_buffer)
            producer_axis = producer.dim_names[producer_dim]
            consumer_axis = consumer.dim_names[consumer_dim]
            producer_tiles = self._barrier_axis_tile_span(
                producer,
                producer_dim,
                "signal",
                producer_buffer,
                producer.tiles_for_axis(producer_axis),
            )
            consumer_tiles = self._barrier_axis_tile_span(
                consumer,
                consumer_dim,
                "wait",
                consumer_buffer,
                consumer.tiles_for_axis(consumer_axis),
            )
            producer_group_count = self._barrier_group_count(
                producer,
                producer_dim,
                "signal",
                producer_buffer,
            )
            consumer_offset = self._barrier_index_offset(consumer, consumer_dim, "wait", consumer_buffer)

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
                        if (
                            producer_group_count is not None
                            and producer_effective_tiles == producer_group_count
                            and consumer_offset + consumer_tiles <= producer_group_count
                        ):
                            compatible_pairs.append((canonical_name, producer_dim, consumer_dim))
                        else:
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
        producer_buffer: str = "",
        consumer_buffer: str = "",
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, int]:
        """Compute divisor vectors plus expected/barrier counts for a dependency edge."""
        producer_divs = [1] * MAX_TILE_DIMS
        consumer_divs = [1] * MAX_TILE_DIMS
        expected = 1

        for _canonical_name, producer_dim, consumer_dim in shared_pairs:
            producer_axis = producer.dim_names[producer_dim]
            consumer_axis = consumer.dim_names[consumer_dim]
            producer_tile_size = self._barrier_tile_size(producer, producer_dim, "signal", producer_buffer)
            consumer_tile_size = self._barrier_tile_size(consumer, consumer_dim, "wait", consumer_buffer)
            if producer_tile_size > consumer_tile_size:
                consumer_divs[consumer_axis] = producer_tile_size // consumer_tile_size
            elif consumer_tile_size > producer_tile_size:
                producer_divs[producer_axis] = consumer_tile_size // producer_tile_size
                expected *= consumer_tile_size // producer_tile_size

        num_barriers = 1
        for _canonical_name, producer_dim, consumer_dim in shared_pairs:
            producer_axis = producer.dim_names[producer_dim]
            consumer_axis = consumer.dim_names[consumer_dim]
            producer_group_count = self._barrier_group_count(
                producer,
                producer_dim,
                "signal",
                producer_buffer,
            )
            consumer_group_count = self._barrier_group_count(
                consumer,
                consumer_dim,
                "wait",
                consumer_buffer,
            )
            if producer_group_count is not None:
                num_barriers *= int(producer_group_count)
            elif consumer_group_count is not None:
                num_barriers *= int(consumer_group_count)
            else:
                num_barriers *= min(
                    self._barrier_axis_tile_span(
                        producer,
                        producer_dim,
                        "signal",
                        producer_buffer,
                        producer.tiles_for_axis(producer_axis),
                    ),
                    self._barrier_axis_tile_span(
                        consumer,
                        consumer_dim,
                        "wait",
                        consumer_buffer,
                        consumer.tiles_for_axis(consumer_axis),
                    ),
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

    def _barrier_tile_size(self, op: ScheduledOp, dim: str, role: str, buffer_name: str = "") -> int:
        """Return role-specific tile size used only for barrier index mapping."""
        return int(
            self._static_dim_lookup(
                op,
                prefix="barrier",
                role=role,
                buffer_name=buffer_name,
                suffix=f"tile_size_{dim}",
                fallback=op.static_dims.get(
                    f"barrier_tile_size_{dim}",
                    op.tile_sizes.get(dim, 1),
                ),
            )
        )

    def _barrier_index_offset(self, op: ScheduledOp, dim: str, role: str, buffer_name: str = "") -> int:
        """Return role-specific barrier index offset for packed/subset views."""
        return int(
            self._static_dim_lookup(
                op,
                prefix="barrier",
                role=role,
                buffer_name=buffer_name,
                suffix=f"index_offset_{dim}",
                fallback=op.static_dims.get(f"barrier_index_offset_{dim}", 0),
            )
        )

    def _barrier_group_count(self, op: ScheduledOp, dim: str, role: str, buffer_name: str = ""):
        value = self._static_dim_lookup(
            op,
            prefix="barrier",
            role=role,
            buffer_name=buffer_name,
            suffix=f"group_count_{dim}",
            fallback=op.static_dims.get(f"barrier_group_count_{dim}"),
        )
        return None if value is None else int(value)

    def _barrier_guard_bound(
        self,
        op: ScheduledOp,
        dim: str,
        role: str,
        buffer_name: str,
        bound: str,
        fallback: int,
    ) -> int:
        return int(
            self._static_dim_lookup(
                op,
                prefix="barrier",
                role=role,
                buffer_name=buffer_name,
                suffix=f"guard_{bound}_{dim}",
                fallback=fallback,
            )
        )

    def _has_barrier_guard_bound(
        self,
        op: ScheduledOp,
        dim: str,
        role: str,
        buffer_name: str,
        bound: str,
    ) -> bool:
        suffix = f"guard_{bound}_{dim}"
        if any(key in op.static_dims for key in self._buffer_static_keys("barrier", role, buffer_name, suffix)):
            return True
        return f"barrier_{role}_{suffix}" in op.static_dims

    def _barrier_axis_tile_span(
        self,
        op: ScheduledOp,
        dim: str,
        role: str,
        buffer_name: str,
        fallback: int,
    ) -> int:
        guard_min = self._barrier_guard_bound(op, dim, role, buffer_name, "min", 0)
        guard_max = self._barrier_guard_bound(op, dim, role, buffer_name, "max", BarrierFormula.NO_GUARD)
        if guard_min == 0 and guard_max == BarrierFormula.NO_GUARD:
            return fallback
        return max(0, min(guard_max, fallback) - max(0, guard_min))

    def _barrier_formula_guard(
        self,
        op: ScheduledOp,
        role: str,
        buffer_name: str,
        shared_pairs: List[Tuple[str, str, str]],
        pair_index: int,
    ) -> Tuple[int, int, Optional[Tuple[int, ...]]]:
        if not shared_pairs:
            return 0, BarrierFormula.NO_GUARD, None
        mins = []
        maxs = []
        guard_coeffs = [0] * MAX_TILE_DIMS
        has_axis_guard = False
        for _canonical_name, producer_dim, consumer_dim in shared_pairs:
            dim = producer_dim if pair_index == 1 else consumer_dim
            mins.append(self._barrier_guard_bound(op, dim, role, buffer_name, "min", 0))
            maxs.append(self._barrier_guard_bound(op, dim, role, buffer_name, "max", BarrierFormula.NO_GUARD))
            if (
                self._has_barrier_guard_bound(op, dim, role, buffer_name, "min")
                or self._has_barrier_guard_bound(op, dim, role, buffer_name, "max")
            ):
                guard_coeffs[op.dim_names[dim]] = 1
                has_axis_guard = True
        guard_min = max(mins) if mins else 0
        guard_max = min(maxs) if maxs else BarrierFormula.NO_GUARD
        return guard_min, guard_max, tuple(guard_coeffs) if has_axis_guard else None

    def _barrier_formula_offsets(
        self,
        producer: ScheduledOp,
        consumer: ScheduledOp,
        shared_pairs: List[Tuple[str, str, str]],
        producer_buffer: str = "",
        consumer_buffer: str = "",
    ) -> Tuple[int, int]:
        """Compute optional constant offsets for shared-dim barrier mappings."""
        producer_offset = 0
        consumer_offset = 0
        for _canonical_name, producer_dim, consumer_dim in shared_pairs:
            producer_offset += self._barrier_index_offset(producer, producer_dim, "signal", producer_buffer)
            consumer_offset += self._barrier_index_offset(consumer, consumer_dim, "wait", consumer_buffer)
        return producer_offset, consumer_offset

    def _last_dim_region_dep(
        self,
        producer: ScheduledOp,
        consumer: ScheduledOp,
        producer_buffer: str,
        consumer_buffer: str,
        producer_only_dims: List[str],
    ) -> Optional[_LastDimRegionDep]:
        """Detect full-producer to sliced-consumer dependencies.

        This covers the decode MLP pattern where an up/gate op writes
        ``y[B,S,O]`` and each down op reads a contiguous ``a[B,S,K]`` view of
        that storage.  The consumer has no tile axis for that input ``K`` dim,
        so ordinary shared-dim formulas collapse all producer ``O`` tiles.  If
        the slice boundaries align with producer barrier tiles, we can group
        producer signals by region and let the consumer wait on only its group.
        """
        prod_meta = producer.tensor_metas.get(producer_buffer)
        cons_meta = consumer.tensor_metas.get(consumer_buffer)
        if prod_meta is None or cons_meta is None:
            return None
        if not prod_meta.declared_dims or not cons_meta.declared_dims:
            return None

        producer_dim = prod_meta.declared_dims[-1]
        consumer_dim = cons_meta.declared_dims[-1]
        if producer_dim not in producer_only_dims:
            return None
        if producer_dim not in producer.dim_names:
            return None
        if consumer_dim in consumer.dim_names:
            return None

        region = last_dim_slice_region(prod_meta, cons_meta)
        if region is None:
            return None
        slice_start, slice_len, full_len = region
        if slice_len <= 0 or full_len <= 0 or full_len % slice_len != 0:
            return None

        producer_tile_size = self._barrier_tile_size(
            producer,
            producer_dim,
            "signal",
            producer_buffer,
        )
        if producer_tile_size <= 0:
            return None
        if slice_start % slice_len != 0:
            return None
        if slice_start % producer_tile_size != 0 or slice_len % producer_tile_size != 0:
            return None

        group_tiles = slice_len // producer_tile_size
        group_count = full_len // slice_len
        group_index = slice_start // slice_len
        producer_tiles = producer.tiles_for_axis(producer.dim_names[producer_dim])
        if producer_tiles != group_tiles * group_count:
            return None

        return _LastDimRegionDep(
            producer_dim=producer_dim,
            group_index=group_index,
            group_count=group_count,
            group_tiles=group_tiles,
        )

    def _barrier_region_formula_components(
        self,
        producer: ScheduledOp,
        consumer: ScheduledOp,
        shared_pairs: List[Tuple[str, str, str]],
        producer_only_dims: List[str],
        region: _LastDimRegionDep,
        producer_buffer: str,
        consumer_buffer: str,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], int, int, int, int]:
        """Build barrier formulas for one producer-only last-dim region edge."""
        producer_divs, consumer_divs, expected, _ = self._barrier_formula_params(
            producer,
            consumer,
            shared_pairs,
            [],
            producer_buffer,
            consumer_buffer,
        )
        producer_divs = list(producer_divs)
        producer_divs[producer.dim_names[region.producer_dim]] = region.group_tiles
        expected *= region.group_tiles

        remaining_producer_only = [
            dim for dim in producer_only_dims if dim != region.producer_dim
        ]
        for producer_dim in remaining_producer_only:
            expected *= producer.tiles_for_axis(producer.dim_names[producer_dim])

        shared_pairs_sorted = sorted(shared_pairs, key=lambda item: item[0])
        shared_strides: Dict[str, int] = {}
        stride = region.group_count
        for canonical_name, producer_dim, consumer_dim in reversed(shared_pairs_sorted):
            shared_strides[canonical_name] = stride
            producer_axis = producer.dim_names[producer_dim]
            consumer_axis = consumer.dim_names[consumer_dim]
            stride *= min(
                self._barrier_axis_tile_span(
                    producer,
                    producer_dim,
                    "signal",
                    producer_buffer,
                    producer.tiles_for_axis(producer_axis),
                ),
                self._barrier_axis_tile_span(
                    consumer,
                    consumer_dim,
                    "wait",
                    consumer_buffer,
                    consumer.tiles_for_axis(consumer_axis),
                ),
            )

        producer_coeffs = [0] * MAX_TILE_DIMS
        consumer_coeffs = [0] * MAX_TILE_DIMS
        for canonical_name, producer_dim, consumer_dim in shared_pairs:
            producer_coeffs[producer.dim_names[producer_dim]] = shared_strides[canonical_name]
            consumer_coeffs[consumer.dim_names[consumer_dim]] = shared_strides[canonical_name]
        producer_coeffs[producer.dim_names[region.producer_dim]] = 1

        producer_offset, consumer_offset = self._barrier_formula_offsets(
            producer,
            consumer,
            shared_pairs,
            producer_buffer,
            consumer_buffer,
        )
        consumer_offset += region.group_index

        num_barriers = stride
        return (
            tuple(producer_coeffs),
            tuple(consumer_coeffs),
            tuple(producer_divs),
            consumer_divs,
            expected,
            num_barriers,
            producer_offset,
            consumer_offset,
        )

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
        signal_family_cache: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...], int, int], int] = {}

        for dep in self._resolve_named_dep_pairs(buffer_producers, tensor_ptr_deps):
                prod_idx = dep.producer_idx
                cons_idx = dep.consumer_idx
                producer = self._op_records[prod_idx]
                consumer = self._op_records[cons_idx]

                p_op = producer.op
                c_op = consumer.op

                shared_pairs, producer_only_dims = self._shared_dim_pairs(
                    p_op,
                    c_op,
                    dep.producer_buffer,
                    dep.consumer_buffer,
                )
                shared_pairs, incompatible_dims = self._compatible_shared_dim_pairs(
                    p_op,
                    c_op,
                    shared_pairs,
                    dep.producer_buffer,
                    dep.consumer_buffer,
                )
                producer_only_dims.extend(incompatible_dims)

                region_dep = self._last_dim_region_dep(
                    p_op,
                    c_op,
                    dep.producer_buffer,
                    dep.consumer_buffer,
                    producer_only_dims,
                )
                if region_dep is not None:
                    (
                        producer_coeffs,
                        consumer_coeffs,
                        producer_divs,
                        consumer_divs,
                        expected,
                        num_barriers,
                        producer_offset,
                        consumer_offset,
                    ) = self._barrier_region_formula_components(
                        p_op,
                        c_op,
                        shared_pairs,
                        producer_only_dims,
                        region_dep,
                        dep.producer_buffer,
                        dep.consumer_buffer,
                    )
                else:
                    producer_divs, consumer_divs, expected, num_barriers = self._barrier_formula_params(
                        p_op,
                        c_op,
                        shared_pairs,
                        producer_only_dims,
                        dep.producer_buffer,
                        dep.consumer_buffer,
                    )
                    producer_coeffs, consumer_coeffs = self._barrier_formula_coeffs(
                        p_op,
                        c_op,
                        shared_pairs,
                    )
                    producer_offset, consumer_offset = self._barrier_formula_offsets(
                        p_op,
                        c_op,
                        shared_pairs,
                        dep.producer_buffer,
                        dep.consumer_buffer,
                    )
                producer_guard_min, producer_guard_max, producer_guard_coeffs = self._barrier_formula_guard(
                    p_op,
                    "signal",
                    dep.producer_buffer,
                    shared_pairs,
                    1,
                )
                consumer_guard_min, consumer_guard_max, consumer_guard_coeffs = self._barrier_formula_guard(
                    c_op,
                    "wait",
                    dep.consumer_buffer,
                    shared_pairs,
                    0,
                )

                # Fan-out consumers can share the same producer-side readiness
                # counter. Each consumer still keeps its own expected count.
                signal_key = (
                    prod_idx,
                    producer_coeffs,
                    producer_divs,
                    num_barriers,
                    producer_offset,
                    producer_guard_min,
                    producer_guard_max,
                    producer_guard_coeffs,
                )
                signal_base = signal_family_cache.get(signal_key)
                if signal_base is None:
                    signal_base = barrier_counter
                    signal_family_cache[signal_key] = signal_base
                    formulas[prod_idx][1].append(
                        BarrierFormula(
                            base=signal_base,
                            coeffs=producer_coeffs,
                            divs=producer_divs,
                            offset=producer_offset,
                            guard_min=producer_guard_min,
                            guard_max=producer_guard_max,
                            guard_coeffs=producer_guard_coeffs,
                        )
                    )
                    barrier_counter += num_barriers

                # Consumer wait formula
                formulas[cons_idx][0].append(
                    BarrierFormula(
                        base=signal_base,
                        coeffs=consumer_coeffs,
                        divs=consumer_divs,
                        offset=consumer_offset,
                        expected=expected,
                        guard_min=consumer_guard_min,
                        guard_max=consumer_guard_max,
                        guard_coeffs=consumer_guard_coeffs,
                    )
                )

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

        if scheduler is None:
            scheduler = BackwardScheduler()

        if hasattr(scheduler, "schedule_with_formulas"):
            formulas, _ = self._resolve()
            instructions = scheduler.schedule_with_formulas(
                self._op_records,
                edges,
                formulas,
            )
        else:
            instructions = scheduler.schedule(self._op_records, consumer_deps, edges)
        instructions.append(TileInstruction.end_instruction())
        return instructions

    def build_tensor(
        self,
        device: str = "cuda",
        scheduler: Optional[TileScheduler] = None,
        instructions: Optional[List[TileInstruction]] = None,
        barrier_meta_indices: Optional[List[int]] = None,
    ):
        """Build instruction stream as GPU tensor.

        Returns:
            Tensor of shape [num_instructions, INSTRUCTION_WORDS].
        """
        import torch

        if instructions is None:
            instructions = self.build(scheduler=scheduler)
        packed = []
        for row_idx, instr in enumerate(instructions):
            barrier_meta_idx = (
                barrier_meta_indices[row_idx]
                if barrier_meta_indices is not None
                else row_idx
            )
            packed.append(
                instr.pack(
                    barrier_meta_idx=barrier_meta_idx,
                )
            )
        return torch.tensor(packed, dtype=torch.int32, device=device)

    def expand_pipeline_instructions(
        self,
        instructions: List[TileInstruction],
    ) -> List[TileInstruction]:
        """Expand coalesced range instructions into logical per-tile rows."""
        out: List[TileInstruction] = []
        for instr in instructions:
            if instr.op_idx == TileInstruction.END_MARKER:
                out.append(instr)
                continue
            for logical_tile in self._logical_tiles_for_instruction(instr):
                out.append(TileInstruction(instr.op_idx, logical_tile))
        return out

    def pipeline_barrier_meta_indices(
        self,
        instructions: List[TileInstruction],
    ) -> Tuple[List[int], List[TileInstruction]]:
        """Return coalesced-instruction metadata row bases and expanded rows.

        The replay stream may keep one coalesced instruction while the runtime
        expands its subtiles into ring slots. Wait/signal metadata remains
        per logical subtile, so the controller can use ``base + subtile_offset``
        for each emitted ring slot.
        """
        meta_indices: List[int] = []
        expanded: List[TileInstruction] = []
        for instr in instructions:
            meta_indices.append(len(expanded))
            if instr.op_idx == TileInstruction.END_MARKER:
                expanded.append(instr)
                continue
            for logical_tile in self._logical_tiles_for_instruction(instr):
                expanded.append(TileInstruction(instr.op_idx, logical_tile))
        return meta_indices, expanded

    def coalesce_pipeline_instructions(
        self,
        instructions: List[TileInstruction],
    ) -> List[TileInstruction]:
        """Coalesce adjacent range-owned instructions in a flat stream."""
        out: List[TileInstruction] = []
        non_end = [instr for instr in instructions if instr.op_idx != TileInstruction.END_MARKER]
        idx = 0
        while idx < len(non_end):
            op_idx = non_end[idx].op_idx
            run_end = idx + 1
            while run_end < len(non_end) and non_end[run_end].op_idx == op_idx:
                run_end += 1
            run = non_end[idx:run_end]
            pipeline = self._pipeline_for_op_idx(op_idx)
            if pipeline is not None and pipeline.coalesce_ranges:
                static_dims = self._op_records[op_idx].op.static_dims
                run = _coalesce_pipeline_ranges(
                    run,
                    range_axis=pipeline.range_axis,
                    range_end_axis=pipeline.range_end_axis,
                    max_range_tiles=pipeline.range_block_size,
                    range_stride=int(static_dims.get("pipeline_range_stride", 1)),
                    range_stride_axis=int(static_dims.get("pipeline_range_stride_axis", -1)),
                )
            out.extend(run)
            idx = run_end
        out.append(TileInstruction.end_instruction())
        return out

    @property
    def max_wait_deps(self) -> int:
        """Maximum number of wait dependencies across all ops."""
        formulas, _ = self._resolve()
        return max((len(wf) for wf, _ in formulas.values()), default=0)

    @property
    def max_signal_deps(self) -> int:
        """Maximum number of signal dependencies across all ops."""
        formulas, _ = self._resolve()
        return max((len(sf) for _wf, sf in formulas.values()), default=0)

    def build_wait_info_tensor(
        self,
        instructions,
        device="cuda",
        num_blocks: Optional[int] = None,
    ):
        """Pre-compute (barrier_idx, expected) per instruction.

        Returns tensor [num_instr, max_waits * 2]:
        [wait0_barrier_idx, wait0_expected, wait1_barrier_idx, wait1_expected, ...]
        barrier_idx = -1 means skip.

        When num_blocks is provided, repeated waits along each persistent CTA's
        strided instruction stream are removed.
        """
        import torch

        formulas, _ = self._resolve()
        raw_wait_data = [
            self._build_wait_info_entry(instr, formulas)
            for instr in instructions
        ]
        max_waits = max(1, max((len(entry) // 2 for entry in raw_wait_data), default=0))
        wait_data = []
        for entry in raw_wait_data:
            padded = list(entry)
            while len(padded) < max_waits * 2:
                padded.extend([-1, 0])
            wait_data.append(padded)
        self._op_wait_counts = self._max_counts_by_op(instructions, raw_wait_data, pair_width=2)
        if num_blocks is not None and num_blocks > 0:
            empty = [-1, 0] * max_waits
            seen_waits = [dict() for _ in range(num_blocks)]
            for idx, entry in enumerate(wait_data):
                block_idx = idx % num_blocks
                if entry == empty:
                    continue
                compacted: List[int] = []
                for pair_idx in range(max_waits):
                    barrier_idx = entry[pair_idx * 2]
                    expected = entry[pair_idx * 2 + 1]
                    if barrier_idx < 0:
                        continue
                    if seen_waits[block_idx].get(barrier_idx, 0) >= expected:
                        continue
                    compacted.extend([barrier_idx, expected])
                    seen_waits[block_idx][barrier_idx] = expected
                while len(compacted) < max_waits * 2:
                    compacted.extend([-1, 0])
                wait_data[idx] = compacted

        return torch.tensor(wait_data, dtype=torch.int32, device=device)

    def build_signal_info_tensor(self, instructions, device="cuda"):
        """Pre-compute signal barrier indices per instruction.

        Returns tensor [num_instr, max_signals], where -1 means skip.
        """
        import torch

        formulas, _ = self._resolve()
        raw_signal_data = [
            self._build_signal_info_entry(instr, formulas)
            for instr in instructions
        ]
        max_signals = max(1, max((len(entry) for entry in raw_signal_data), default=0))
        signal_data = []
        for entry in raw_signal_data:
            padded = list(entry)
            while len(padded) < max_signals:
                padded.append(-1)
            signal_data.append(padded)
        self._op_signal_counts = self._max_counts_by_op(instructions, raw_signal_data, pair_width=1)

        return torch.tensor(signal_data, dtype=torch.int32, device=device)

    @staticmethod
    def _max_counts_by_op(
        instructions: List[TileInstruction],
        entries: List[List[int]],
        *,
        pair_width: int,
    ) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for instr, entry in zip(instructions, entries):
            if instr.op_idx == TileInstruction.END_MARKER:
                continue
            counts[instr.op_idx] = max(counts.get(instr.op_idx, 0), len(entry) // pair_width)
        return counts

    def _build_wait_info_entry(
        self,
        instr: TileInstruction,
        formulas: Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]],
    ) -> List[int]:
        """Pack wait metadata for one instruction into `[barrier, expected, ...]`."""
        if instr.op_idx == TileInstruction.END_MARKER:
            return []

        entry: List[int] = []
        seen = set()
        wait_formulas = formulas.get(instr.op_idx, ([], []))[0]
        for tile in self._logical_tiles_for_instruction(instr):
            for formula in wait_formulas:
                if not (formula.has_guard and not formula.is_guarded(tile)):
                    pair = (formula.compute_index(tile), formula.expected)
                    if pair not in seen:
                        seen.add(pair)
                        entry.extend(pair)
        return entry

    def _build_signal_info_entry(
        self,
        instr: TileInstruction,
        formulas: Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]],
    ) -> List[int]:
        """Pack signal metadata for one instruction into barrier indices."""
        if instr.op_idx == TileInstruction.END_MARKER:
            return []

        entry: List[int] = []
        signal_formulas = formulas.get(instr.op_idx, ([], []))[1]
        logical_tiles = self._logical_tiles_for_instruction(instr)
        range_axis, _end_axis = self._range_axis_for_op_idx(instr.op_idx)
        for formula in signal_formulas:
            seen = set()
            preserve_range_multiplicity = range_axis >= 0 and formula.coeffs[range_axis] != 0
            for tile in logical_tiles:
                if not (formula.has_guard and not formula.is_guarded(tile)):
                    barrier_idx = formula.compute_index(tile)
                    if preserve_range_multiplicity or barrier_idx not in seen:
                        seen.add(barrier_idx)
                        entry.append(barrier_idx)
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
    "INSTR_BARRIER_META_IDX",
    "INSTR_OP_IDX",
    "INSTR_TILE_0",
    "INSTR_TILE_1",
    "INSTR_TILE_2",
    "INSTR_TILE_3",
    "INSTR_TILE_4",
    "TileInstruction",
    "TileScheduler",
    "BackwardScheduler",
    "OverlapTileScheduler",
    "InstructionStreamBuilder",
]
