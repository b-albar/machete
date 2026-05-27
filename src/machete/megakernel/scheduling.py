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
import inspect
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Set, Tuple, Type, Union

from .instruction_layout import (
    INSTR_BARRIER_META_IDX,
    INSTR_OP_IDX,
    INSTR_RANGE_END,
    INSTR_RANGE_META,
    INSTR_TILE_01,
    INSTR_TILE_23,
    INSTRUCTION_WORDS,
)
from .ops import (
    MAX_TILE_DIMS,
    Op,
    ScheduledOp,
    build_op_config,
    last_dim_slice_region,
    tensor_meta_overlaps,
)


def _op_cls_phase_param_names(op_cls: Type[Op], phase_name: str) -> Tuple[str, ...]:
    """Return parameter names for the class-declared implementation of a phase."""
    implementation_name = getattr(op_cls, f"{phase_name}_phase", None)
    method = getattr(op_cls, implementation_name or phase_name, None)
    if method is None:
        return ()
    return tuple(name for name in inspect.signature(method).parameters if name != "self")


def _scheduled_op_phase_param_names(op: ScheduledOp, phase_name: str) -> Tuple[str, ...]:
    """Return parameter names for a scheduled op's concrete phase implementation."""
    try:
        instance = op.op_cls(**build_op_config(op))
    except Exception:
        return _op_cls_phase_param_names(op.op_cls, phase_name)
    method = getattr(instance, phase_name, None)
    if method is None:
        return ()
    method_target = getattr(method, "__func__", method)
    return tuple(name for name in inspect.signature(method_target).parameters if name != "self")


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
    [0]  low 16 bits: op_idx, high 16 bits: range metadata
    [1]  tile_0 | (tile_1 << 16)
    [2]  tile_2 | (tile_3 << 16)
    [3]  barrier metadata row index
    [4]  range end coordinate

    Barrier wait/signal formulas stay in compact side tensors. The instruction
    carries the row index into those side tensors so replay does not need to
    derive it from scheduler shape in the hot loop.
    """

    op_idx: int
    tiles: Tuple[int, ...]  # Up to MAX_TILE_DIMS tile indices
    range_axis: int = -1
    range_end_axis: int = -1
    range_end: int = 0

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
            return [0xFFFF] + [0] * (INSTRUCTION_WORDS - 1)
        if self.op_idx < 0 or self.op_idx > 0xFFFE:
            raise ValueError(f"Instruction op_idx must fit in uint16, got {self.op_idx}")
        if barrier_meta_idx < 0 or barrier_meta_idx > 0x7FFFFFFF:
            raise ValueError(
                f"Instruction barrier_meta_idx must fit in int32, got {barrier_meta_idx}"
            )
        padded_tiles = list(self.tiles) + [0] * (MAX_TILE_DIMS - len(self.tiles))
        if len(padded_tiles) != MAX_TILE_DIMS:
            raise ValueError(f"Instruction tile rank exceeds {MAX_TILE_DIMS}")
        for tile in padded_tiles:
            if tile < 0 or tile > 0xFFFF:
                raise ValueError(f"Instruction tile coordinate must fit in uint16, got {tile}")
        range_meta = 0
        if self.range_axis >= 0 and self.range_end_axis >= 0:
            if self.range_axis >= MAX_TILE_DIMS:
                raise ValueError(f"range_axis must be < {MAX_TILE_DIMS}, got {self.range_axis}")
            if self.range_end_axis < 0 or self.range_end_axis > 0xE:
                raise ValueError(
                    f"range_end_axis must fit in 4 encoded bits, got {self.range_end_axis}"
                )
            if self.range_end < 0 or self.range_end > 0xFFFF:
                raise ValueError(f"range_end must fit in uint16, got {self.range_end}")
            range_meta = (self.range_axis + 1) | ((self.range_end_axis + 1) << 4)
        tile_01 = padded_tiles[0] | (padded_tiles[1] << 16)
        tile_23 = padded_tiles[2] | (padded_tiles[3] << 16)
        op_range = self.op_idx | (range_meta << 16)

        def _signed_i32(word: int) -> int:
            return word if word <= 0x7FFFFFFF else word - 0x100000000

        return [
            _signed_i32(op_range),
            _signed_i32(tile_01),
            _signed_i32(tile_23),
            barrier_meta_idx,
            self.range_end,
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
class DependencyPlan:
    """Resolved dependency graph and barrier formulas for one instruction stream."""

    formulas: Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]]
    barrier_count: int
    edges: List[_DepEdge]
    consumer_deps: Dict[int, List[_DepEdge]]
    controller_wait_formulas: Dict[int, List[BarrierFormula]]
    compute_wait_formulas: Dict[int, List[BarrierFormula]]


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


def _build_linear_chain_dependency_plan(op_records: List["_OpRecord"]) -> DependencyPlan:
    """Build the legacy dependency plan for ops without named buffers.

    This fallback preserves old behavior for simple tests and ad-hoc kernels,
    but keeps the no-metadata path out of the main named-buffer resolver.
    """
    formulas: Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]] = {}
    controller_wait_formulas: Dict[int, List[BarrierFormula]] = {}
    compute_wait_formulas: Dict[int, List[BarrierFormula]] = {
        i: [] for i in range(len(op_records))
    }
    edges = [
        _DepEdge(producer_idx=i - 1, consumer_idx=i, kind="one_to_one")
        for i in range(1, len(op_records))
    ]
    barrier_counter = 0

    for i, rec in enumerate(op_records):
        op = rec.op
        signal_base = barrier_counter
        coeffs = _linear_strides(op.tile_counts)
        signal_formulas = [BarrierFormula(base=signal_base, coeffs=coeffs)]

        wait_formulas: List[BarrierFormula] = []
        if i > 0:
            prev_op = op_records[i - 1].op
            prev_base = barrier_counter - prev_op.total_tiles
            guard = (
                prev_op.total_tiles
                if prev_op.total_tiles != op.total_tiles
                else BarrierFormula.NO_GUARD
            )
            wait_formulas.append(
                BarrierFormula(
                    base=prev_base,
                    coeffs=coeffs,
                    expected=1,
                    guard_max=guard,
                )
            )

        formulas[i] = (wait_formulas, signal_formulas)
        controller_wait_formulas[i] = wait_formulas
        barrier_counter += op.total_tiles

    return DependencyPlan(
        formulas=formulas,
        barrier_count=barrier_counter,
        edges=edges,
        consumer_deps=_group_consumer_deps(edges),
        controller_wait_formulas=controller_wait_formulas,
        compute_wait_formulas=compute_wait_formulas,
    )


def _coalesce_coordinate_ranges(
    instructions: List[TileInstruction],
    *,
    tile_rank: int,
    max_range_tiles: int = 0,
    allow_range=None,
) -> List[TileInstruction]:
    """Coalesce adjacent same-op tiles that form a contiguous coordinate range."""
    if not instructions:
        return []
    if tile_rank <= 0 or tile_rank > MAX_TILE_DIMS:
        return instructions

    range_end_axis = tile_rank
    out: List[TileInstruction] = []
    idx = 0
    while idx < len(instructions):
        first = instructions[idx]
        tiles = list(first.tiles) + [0] * (MAX_TILE_DIMS - len(first.tiles))

        range_axis = -1
        if idx + 1 < len(instructions):
            next_tiles = list(instructions[idx + 1].tiles) + [0] * (
                MAX_TILE_DIMS - len(instructions[idx + 1].tiles)
            )
            for axis in range(tile_rank):
                if next_tiles[axis] != tiles[axis] + 1:
                    continue
                if all(
                    next_tiles[other] == tiles[other]
                    for other in range(tile_rank)
                    if other != axis
                ):
                    range_axis = axis
                    break

        if range_axis < 0:
            out.append(first)
            idx += 1
            continue

        start = tiles[range_axis]
        end = start + 1
        idx += 1

        while idx < len(instructions):
            candidate = instructions[idx]
            cand_tiles = list(candidate.tiles) + [0] * (MAX_TILE_DIMS - len(candidate.tiles))
            if max_range_tiles > 0 and end - start >= max_range_tiles:
                break
            same_prefix = all(
                cand_tiles[axis] == tiles[axis]
                for axis in range(tile_rank)
                if axis != range_axis
            )
            if not same_prefix or cand_tiles[range_axis] != end:
                break
            end += 1
            idx += 1

        range_instr = TileInstruction(
            op_idx=first.op_idx,
            tiles=tuple(tiles[:MAX_TILE_DIMS]),
            range_axis=range_axis,
            range_end_axis=range_end_axis,
            range_end=end,
        )
        if allow_range is not None and not allow_range(range_instr):
            out.append(first)
            idx = idx - (end - start - 1)
            continue
        out.append(range_instr)

    return out


def _instruction_range_len(instr: TileInstruction) -> int:
    """Number of logical tiles represented by one instruction row."""
    if instr.range_axis < 0 or instr.range_axis >= MAX_TILE_DIMS:
        return 1
    tiles = list(instr.tiles) + [0] * MAX_TILE_DIMS
    return max(1, instr.range_end - tiles[instr.range_axis])


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
        adaptive_fetch_stride: bool = False,
        prefer_data_movement: bool = True,
        prefer_ready_consumers: bool = False,
        prefer_ready_consumer_op_names: Optional[Set[str]] = None,
        dependency_slack_waves: int = 0,
        dependency_slack_op_names: Optional[Set[str]] = None,
        dependency_slack_op_indices: Optional[Set[int]] = None,
        dependency_slack_requires_full_op: bool = True,
        use_controller_waits_for_readiness: bool = False,
    ):
        self.fetch_stride = fetch_stride
        self.adaptive_fetch_stride = bool(adaptive_fetch_stride)
        self.prefer_data_movement = prefer_data_movement
        self.prefer_ready_consumers = prefer_ready_consumers
        self.prefer_ready_consumer_op_names = (
            set(prefer_ready_consumer_op_names)
            if prefer_ready_consumer_op_names is not None
            else None
        )
        self.dependency_slack_waves = max(0, int(dependency_slack_waves))
        self.dependency_slack_op_names = (
            set(dependency_slack_op_names)
            if dependency_slack_op_names is not None
            else None
        )
        self.dependency_slack_op_indices = (
            {int(op_idx) for op_idx in dependency_slack_op_indices}
            if dependency_slack_op_indices is not None
            else None
        )
        self.dependency_slack_requires_full_op = bool(dependency_slack_requires_full_op)
        self.use_controller_waits_for_readiness = bool(use_controller_waits_for_readiness)
        self._bound_num_blocks: Optional[int] = None

    def bind_num_blocks(self, num_blocks: int) -> None:
        """Bind the runtime instruction fetch stride used by persistent CTAs."""
        self._bound_num_blocks = max(1, int(num_blocks))
        if self.fetch_stride is None and not self.adaptive_fetch_stride:
            self.fetch_stride = max(1, int(num_blocks))

    def _resolve_fetch_stride(self, total_tiles: int) -> int:
        if self.fetch_stride is not None:
            return max(1, int(self.fetch_stride))
        num_blocks = max(1, int(self._bound_num_blocks or 1))
        if self.adaptive_fetch_stride and total_tiles <= 14 * num_blocks:
            return 2 * num_blocks
        return num_blocks

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

    def _waits_ready_with_slack(
        self,
        instr: TileInstruction,
        wait_formulas: List[BarrierFormula],
        barrier_counts: Dict[int, int],
        barrier_ready_wave: Dict[int, int],
        current_wave: int,
        op_name: str,
        op_idx: int,
    ) -> bool:
        if not self._slack_applies(op_idx, op_name):
            return True
        for formula in wait_formulas:
            if formula.has_guard and not formula.is_guarded(instr.tiles):
                continue
            barrier_idx = formula.compute_index(instr.tiles)
            if barrier_counts.get(barrier_idx, 0) < formula.expected:
                return False
            ready_wave = barrier_ready_wave.get(barrier_idx)
            if ready_wave is not None and current_wave < ready_wave + self.dependency_slack_waves + 1:
                return False
        return True

    def _slack_applies(self, op_idx: int, op_name: str) -> bool:
        if self.dependency_slack_waves <= 0:
            return False
        slack_indices = self.dependency_slack_op_indices
        if slack_indices is not None and op_idx not in slack_indices:
            return False
        slack_ops = self.dependency_slack_op_names
        if slack_ops is not None and op_name not in slack_ops:
            return False
        return True

    def _signal(
        self,
        instr: TileInstruction,
        signal_formulas: List[BarrierFormula],
        barrier_counts: Dict[int, int],
        barrier_ready_wave: Dict[int, int],
        current_wave: int,
    ) -> None:
        for formula in signal_formulas:
            if formula.has_guard and not formula.is_guarded(instr.tiles):
                continue
            barrier_idx = formula.compute_index(instr.tiles)
            before = barrier_counts.get(barrier_idx, 0)
            after = before + 1
            barrier_counts[barrier_idx] = after
            if before < formula.expected <= after and barrier_idx not in barrier_ready_wave:
                barrier_ready_wave[barrier_idx] = current_wave

    def _priority(
        self,
        *,
        op_idx: int,
        tile_idx: int,
        pos: int,
        depths: List[int],
        resource_scores: List[int],
        has_waits: List[bool],
        op_names: List[str],
    ) -> Tuple[int, int, int, int, int]:
        resource_score = resource_scores[op_idx] if self.prefer_data_movement else 0
        allowed_consumers = self.prefer_ready_consumer_op_names
        consumer_score = 1 if (
            self.prefer_ready_consumers
            and has_waits[op_idx]
            and (allowed_consumers is None or op_names[op_idx] in allowed_consumers)
        ) else 0
        return (
            consumer_score,
            depths[op_idx],
            -op_idx,
            resource_score,
            -tile_idx,
        )

    def _schedule_stride_waves(
        self,
        candidates: List[Tuple[int, int, TileInstruction]],
        formulas: Dict[int, Tuple[List[BarrierFormula], List[BarrierFormula]]],
        depths: List[int],
        resource_scores: List[int],
        has_waits: List[bool],
        op_names: List[str],
        fetch_stride: int,
    ) -> List[TileInstruction]:
        scheduled: List[TileInstruction] = []
        barrier_counts: Dict[int, int] = {}
        barrier_ready_wave: Dict[int, int] = {}
        current_wave = 0

        while candidates:
            ready = []
            fallback_ready = []
            base_ready_entries = []
            pending_by_op: Dict[int, int] = {}
            base_ready_by_op: Dict[int, int] = {}
            slack_ready_by_op: Dict[int, int] = {}
            for op_idx, _tile_idx, _instr in candidates:
                pending_by_op[op_idx] = pending_by_op.get(op_idx, 0) + 1
            for pos, (op_idx, tile_idx, instr) in enumerate(candidates):
                wait_formulas = formulas.get(op_idx, ([], []))[0]
                if self._waits_ready(instr, wait_formulas, barrier_counts):
                    base_ready_by_op[op_idx] = base_ready_by_op.get(op_idx, 0) + 1
                    entry = (
                        *self._priority(
                            op_idx=op_idx,
                            tile_idx=tile_idx,
                            pos=pos,
                            depths=depths,
                            resource_scores=resource_scores,
                            has_waits=has_waits,
                            op_names=op_names,
                        ),
                        pos,
                    )
                    base_ready_entries.append((entry, op_idx, instr, wait_formulas))
                    fallback_ready.append(entry)
                    if self._waits_ready_with_slack(
                        instr,
                        wait_formulas,
                        barrier_counts,
                        barrier_ready_wave,
                        current_wave,
                        op_names[op_idx],
                        op_idx,
                    ):
                        slack_ready_by_op[op_idx] = slack_ready_by_op.get(op_idx, 0) + 1

            for entry, op_idx, instr, wait_formulas in base_ready_entries:
                if (
                    self.dependency_slack_requires_full_op
                    and self._slack_applies(op_idx, op_names[op_idx])
                    and (
                        base_ready_by_op.get(op_idx, 0) < pending_by_op.get(op_idx, 0)
                        or slack_ready_by_op.get(op_idx, 0) < pending_by_op.get(op_idx, 0)
                    )
                ):
                    continue
                if self._waits_ready_with_slack(
                        instr,
                        wait_formulas,
                        barrier_counts,
                        barrier_ready_wave,
                        current_wave,
                        op_names[op_idx],
                        op_idx,
                ):
                    ready.append(entry)

            if not ready:
                if fallback_ready:
                    ready = fallback_ready
                else:
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
                self._signal(
                    instr,
                    signal_formulas,
                    barrier_counts,
                    barrier_ready_wave,
                    current_wave,
                )
            current_wave += 1

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
        has_waits = [bool(formulas.get(rec.op_idx, ([], []))[0]) for rec in op_records]
        op_names = [rec.op.op_cls.__name__ for rec in op_records]
        candidates: List[Tuple[int, int, TileInstruction]] = []
        for rec in op_records:
            for tile_idx, tile in enumerate(rec.tiles):
                candidates.append((rec.op_idx, tile_idx, TileInstruction(rec.op_idx, tile)))

        fetch_stride = self._resolve_fetch_stride(sum(rec.op.total_tiles for rec in op_records))
        return self._schedule_stride_waves(
            candidates,
            formulas,
            depths,
            resource_scores,
            has_waits,
            op_names,
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
        self._cached_dependency_plan: Optional[DependencyPlan] = None
        self._cached_controller_wait_formulas: Optional[Dict[int, List[BarrierFormula]]] = None
        self._cached_compute_wait_formulas: Optional[Dict[int, List[BarrierFormula]]] = None
        self._op_wait_counts: Dict[int, int] = {}
        self._op_compute_wait_counts: Dict[int, int] = {}
        self._op_signal_counts: Dict[int, int] = {}

    def _invalidate_resolution_cache(self) -> None:
        """Clear cached formulas after the op list changes."""
        self._cached_dependency_plan = None
        self._cached_controller_wait_formulas = None
        self._cached_compute_wait_formulas = None

    def _logical_tiles_for_instruction(self, instr: TileInstruction) -> List[Tuple[int, ...]]:
        """Return logical tiles covered by one possibly range-owned instruction."""
        if instr.op_idx == TileInstruction.END_MARKER:
            return []
        range_axis = instr.range_axis
        end_axis = instr.range_end_axis
        if range_axis < 0:
            return [instr.tiles]
        if end_axis < 0 or end_axis > MAX_TILE_DIMS or end_axis == range_axis:
            return [instr.tiles]
        tiles = list(instr.tiles) + [0] * (MAX_TILE_DIMS - len(instr.tiles))
        start = tiles[range_axis]
        end = instr.range_end
        if end <= start:
            return [instr.tiles]
        out: List[Tuple[int, ...]] = []
        for axis_value in range(start, end):
            logical = list(tiles)
            logical[range_axis] = axis_value
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

    def _find_producer_buffers(
        self,
        prod_idx: int,
        consumer: ScheduledOp,
        consumer_buf: str,
    ) -> List[str]:
        """Return all producer outputs that back a consumer input.

        A single op can materialize several disjoint views into one later
        logical buffer, for example a fused q/k/v projection that writes three
        last-dim slices which are then consumed as one packed qkv tensor.  The
        consumer must wait for every overlapping output region, not just the
        first matching view.
        """
        prod = self._op_records[prod_idx].op
        cons_meta = consumer.tensor_metas.get(consumer_buf)
        matches: List[str] = []

        if cons_meta is not None:
            for prod_buf in prod.op_cls.OUTPUTS:
                prod_meta = prod.tensor_metas.get(prod_buf)
                if prod_meta is not None and tensor_meta_overlaps(prod_meta, cons_meta):
                    matches.append(prod_buf)

        if matches:
            return matches
        return [self._find_producer_buffer(prod_idx, consumer, consumer_buf)]

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
            prod_meta = cand_rec.op.tensor_metas.get(buf)
            cons_meta = rec.op.tensor_metas.get(buf)
            if (
                prod_meta is not None
                and cons_meta is not None
                and not tensor_meta_overlaps(prod_meta, cons_meta)
            ):
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

    def _find_producers(
        self,
        rec: "_OpRecord",
        buf: str,
        buffer_producers: Dict[str, int],
        tensor_ptr_deps: Dict[Tuple[int, str], int],
    ) -> List[int]:
        """Find producer op indices for a consumer input.

        Most buffers are single-writer regions, where the nearest producer is
        enough. Dimension-windowed ops can materialize disjoint regions of the
        same logical tensor, so a full-region consumer must wait for every
        previous overlapping window.
        """
        has_windows = bool(rec.op.tile_origins) or any(
            prior.op.tile_origins for prior in self._op_records[: rec.op_idx]
        )
        if not has_windows:
            prod_idx = self._find_producer(rec, buf, buffer_producers, tensor_ptr_deps)
            return [] if prod_idx is None else [prod_idx]

        cons_meta = rec.op.tensor_metas.get(buf)
        producers: List[int] = []
        seen: Set[int] = set()
        for cand_idx in range(rec.op_idx - 1, -1, -1):
            cand_rec = self._op_records[cand_idx]
            for prod_buf in cand_rec.op.op_cls.OUTPUTS:
                prod_meta = cand_rec.op.tensor_metas.get(prod_buf)
                if cons_meta is not None and prod_meta is not None:
                    if not tensor_meta_overlaps(prod_meta, cons_meta):
                        continue
                elif prod_buf != buf:
                    continue
                if cand_idx not in seen:
                    producers.append(cand_idx)
                    seen.add(cand_idx)
        if producers:
            return list(reversed(producers))

        prod_idx = self._find_producer(rec, buf, buffer_producers, tensor_ptr_deps)
        return [] if prod_idx is None else [prod_idx]

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
                prod_indices = self._find_producers(rec, buf, buffer_producers, tensor_ptr_deps)
                if not prod_indices:
                    continue
                for prod_idx in prod_indices:
                    for prod_buf in self._find_producer_buffers(prod_idx, rec.op, buf):
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

    def dependency_plan(self) -> DependencyPlan:
        """Return the resolved dependency plan for this instruction stream."""
        if self._cached_dependency_plan is not None:
            return self._cached_dependency_plan

        if self._has_named_buffers():
            formulas, count = self._resolve_named_formulas()
            edges = self._resolve_dep_edges()
            controller_wait_formulas = (
                self._cached_controller_wait_formulas
                if self._cached_controller_wait_formulas is not None
                else {i: waits for i, (waits, _signals) in formulas.items()}
            )
            compute_wait_formulas = (
                self._cached_compute_wait_formulas
                if self._cached_compute_wait_formulas is not None
                else {i: [] for i in range(len(self._op_records))}
            )
            plan = DependencyPlan(
                formulas=formulas,
                barrier_count=count,
                edges=edges,
                consumer_deps=_group_consumer_deps(edges),
                controller_wait_formulas=controller_wait_formulas,
                compute_wait_formulas=compute_wait_formulas,
            )
        else:
            plan = _build_linear_chain_dependency_plan(self._op_records)

        self._cached_dependency_plan = plan
        self._cached_controller_wait_formulas = plan.controller_wait_formulas
        self._cached_compute_wait_formulas = plan.compute_wait_formulas
        return plan

    def _resolve_compute_wait_formulas(self) -> Dict[int, List[BarrierFormula]]:
        """Return dependency waits that can be delayed until compute phase."""
        return self.dependency_plan().compute_wait_formulas

    def _resolve_controller_wait_formulas(self) -> Dict[int, List[BarrierFormula]]:
        """Return dependency waits that must run before load phase."""
        return self.dependency_plan().controller_wait_formulas

    def _dep_wait_can_move_to_compute(self, dep: _DepPair) -> bool:
        """Whether a consumer dependency is needed by compute but not TMA load.

        If the dependency backs a TMA-loaded tensor, the controller must wait
        before load so TMA does not read producer output early. If it backs a
        non-TMA input, load can prefetch the op's independent TMA operands and
        compute waits immediately before consuming the dependent input.
        """
        consumer = self._op_records[dep.consumer_idx].op
        if dep.consumer_buffer not in consumer.op_cls.INPUTS:
            return False
        controller_wait_inputs = set(getattr(consumer.op_cls, "controller_wait_inputs", set()))
        if dep.consumer_buffer in controller_wait_inputs:
            return False
        tma_loads = set(getattr(consumer.op_cls, "_TMA_LOADS", set()))
        tma_compute_loads = set(getattr(consumer.op_cls, "_TMA_COMPUTE_LOADS", set()))
        if dep.consumer_buffer in tma_compute_loads:
            return True
        if dep.consumer_buffer in tma_loads:
            return False

        load_params = set(_scheduled_op_phase_param_names(consumer, "load"))
        if "op_config_ptr" in load_params:
            return False
        if dep.consumer_buffer in load_params:
            return False

        if consumer.op_cls.load is not Op.load:
            compute_params = set(_scheduled_op_phase_param_names(consumer, "compute"))
            return dep.consumer_buffer in compute_params
        return True

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
                producer_start = producer.tile_origin_for_axis(producer_axis) * producer_tile_size
                producer_end = producer_start + producer_tiles * producer_tile_size
                consumer_start = consumer.tile_origin_for_axis(consumer_axis) * consumer_tile_size
                consumer_end = consumer_start + consumer_tiles * consumer_tile_size
                overlap_start = max(producer_start, consumer_start)
                overlap_end = min(producer_end, consumer_end)
                if overlap_end > overlap_start and (overlap_end - overlap_start) % producer_tile_size == 0:
                    compatible_pairs.append((canonical_name, producer_dim, consumer_dim))
                else:
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

    @staticmethod
    def _merge_axis_guard(
        guard_min: int,
        guard_max: int,
        guard_coeffs: Optional[Tuple[int, ...]],
        *,
        axis: int,
        start: int,
        end: int,
    ) -> Tuple[int, int, Optional[Tuple[int, ...]]]:
        axis_coeffs = [0] * MAX_TILE_DIMS
        axis_coeffs[axis] = 1
        axis_coeffs_tuple = tuple(axis_coeffs)
        if guard_coeffs is not None and guard_coeffs != axis_coeffs_tuple:
            return guard_min, guard_max, guard_coeffs
        return max(guard_min, start), min(guard_max, end), axis_coeffs_tuple

    def _apply_window_overlap_to_formulas(
        self,
        producer: ScheduledOp,
        consumer: ScheduledOp,
        shared_pairs: List[Tuple[str, str, str]],
        producer_buffer: str,
        consumer_buffer: str,
        producer_coeffs: Tuple[int, ...],
        consumer_coeffs: Tuple[int, ...],
        producer_divs: Tuple[int, ...],
        consumer_divs: Tuple[int, ...],
        producer_offset: int,
        consumer_offset: int,
        producer_guard_min: int,
        producer_guard_max: int,
        producer_guard_coeffs: Optional[Tuple[int, ...]],
        consumer_guard_min: int,
        consumer_guard_max: int,
        consumer_guard_coeffs: Optional[Tuple[int, ...]],
    ) -> Tuple[int, int, int, int, Optional[Tuple[int, ...]], int, int, Optional[Tuple[int, ...]]]:
        """Restrict barrier formulas to the actual overlap of windowed tile domains.

        Tensor metadata overlap tells us that a producer and consumer touch the same
        storage, but formula coordinates are local to each scheduled op. When a
        producer is a batch window and the consumer covers the full batch, both
        sides must map only their common local tile interval onto the same compact
        barrier range.
        """
        for _canonical_name, producer_dim, consumer_dim in shared_pairs:
            producer_axis = producer.dim_names[producer_dim]
            consumer_axis = consumer.dim_names[consumer_dim]
            producer_tile_size = self._barrier_tile_size(
                producer,
                producer_dim,
                "signal",
                producer_buffer,
            )
            consumer_tile_size = self._barrier_tile_size(
                consumer,
                consumer_dim,
                "wait",
                consumer_buffer,
            )
            producer_start = producer.tile_origin_for_axis(producer_axis) * producer_tile_size
            producer_end = producer_start + producer.tiles_for_axis(producer_axis) * producer_tile_size
            consumer_start = consumer.tile_origin_for_axis(consumer_axis) * consumer_tile_size
            consumer_end = consumer_start + consumer.tiles_for_axis(consumer_axis) * consumer_tile_size
            overlap_start = max(producer_start, consumer_start)
            overlap_end = min(producer_end, consumer_end)
            if overlap_end <= overlap_start:
                producer_guard_min, producer_guard_max, producer_guard_coeffs = self._merge_axis_guard(
                    producer_guard_min,
                    producer_guard_max,
                    producer_guard_coeffs,
                    axis=producer_axis,
                    start=1,
                    end=0,
                )
                consumer_guard_min, consumer_guard_max, consumer_guard_coeffs = self._merge_axis_guard(
                    consumer_guard_min,
                    consumer_guard_max,
                    consumer_guard_coeffs,
                    axis=consumer_axis,
                    start=1,
                    end=0,
                )
                continue
            if overlap_start == producer_start and overlap_end == producer_end and overlap_start == consumer_start and overlap_end == consumer_end:
                continue
            if (
                (overlap_start - producer_start) % producer_tile_size != 0
                or (overlap_end - producer_start) % producer_tile_size != 0
                or (overlap_start - consumer_start) % consumer_tile_size != 0
                or (overlap_end - consumer_start) % consumer_tile_size != 0
            ):
                continue

            producer_local_start = (overlap_start - producer_start) // producer_tile_size
            producer_local_end = (overlap_end - producer_start) // producer_tile_size
            consumer_local_start = (overlap_start - consumer_start) // consumer_tile_size
            consumer_local_end = (overlap_end - consumer_start) // consumer_tile_size
            producer_div = producer_divs[producer_axis]
            consumer_div = consumer_divs[consumer_axis]
            if producer_local_start % producer_div != 0 or consumer_local_start % consumer_div != 0:
                continue

            producer_guard_min, producer_guard_max, producer_guard_coeffs = self._merge_axis_guard(
                producer_guard_min,
                producer_guard_max,
                producer_guard_coeffs,
                axis=producer_axis,
                start=producer_local_start,
                end=producer_local_end,
            )
            consumer_guard_min, consumer_guard_max, consumer_guard_coeffs = self._merge_axis_guard(
                consumer_guard_min,
                consumer_guard_max,
                consumer_guard_coeffs,
                axis=consumer_axis,
                start=consumer_local_start,
                end=consumer_local_end,
            )
            producer_offset -= producer_coeffs[producer_axis] * (producer_local_start // producer_div)
            consumer_offset -= consumer_coeffs[consumer_axis] * (consumer_local_start // consumer_div)

        return (
            producer_offset,
            consumer_offset,
            producer_guard_min,
            producer_guard_max,
            producer_guard_coeffs,
            consumer_guard_min,
            consumer_guard_max,
            consumer_guard_coeffs,
        )

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

        producer_meta_dim = prod_meta.declared_dims[-1]
        consumer_dim = cons_meta.declared_dims[-1]
        producer_dim = producer_meta_dim
        if producer_dim not in producer_only_dims:
            producer_dim = ""
            for dim in producer_only_dims:
                canonical_dim = self._static_dim_lookup(
                    producer,
                    prefix="barrier",
                    role="signal",
                    buffer_name=producer_buffer,
                    suffix=f"alias_{dim}",
                    fallback=producer.dim_aliases.get(dim, dim),
                )
                if canonical_dim == producer_meta_dim:
                    producer_dim = dim
                    break
            if not producer_dim:
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
        controller_wait_formulas: Dict[int, List[BarrierFormula]] = {
            i: [] for i in range(len(self._op_records))
        }
        compute_wait_formulas: Dict[int, List[BarrierFormula]] = {
            i: [] for i in range(len(self._op_records))
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
                (
                    producer_offset,
                    consumer_offset,
                    producer_guard_min,
                    producer_guard_max,
                    producer_guard_coeffs,
                    consumer_guard_min,
                    consumer_guard_max,
                    consumer_guard_coeffs,
                ) = self._apply_window_overlap_to_formulas(
                    p_op,
                    c_op,
                    shared_pairs,
                    dep.producer_buffer,
                    dep.consumer_buffer,
                    producer_coeffs,
                    consumer_coeffs,
                    producer_divs,
                    consumer_divs,
                    producer_offset,
                    consumer_offset,
                    producer_guard_min,
                    producer_guard_max,
                    producer_guard_coeffs,
                    consumer_guard_min,
                    consumer_guard_max,
                    consumer_guard_coeffs,
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

                wait_formula = BarrierFormula(
                    base=signal_base,
                    coeffs=consumer_coeffs,
                    divs=consumer_divs,
                    offset=consumer_offset,
                    expected=expected,
                    guard_min=consumer_guard_min,
                    guard_max=consumer_guard_max,
                    guard_coeffs=consumer_guard_coeffs,
                )
                if self._dep_wait_can_move_to_compute(dep):
                    formulas[cons_idx][0].append(wait_formula)
                    compute_wait_formulas[cons_idx].append(wait_formula)
                else:
                    formulas[cons_idx][0].append(wait_formula)
                    controller_wait_formulas[cons_idx].append(wait_formula)

        self._cached_controller_wait_formulas = controller_wait_formulas
        self._cached_compute_wait_formulas = compute_wait_formulas
        return formulas, barrier_counter

    def build(self, scheduler: Optional[TileScheduler] = None) -> List[TileInstruction]:
        """Build an instruction list using the specified scheduler.

        Args:
            scheduler: Tile scheduler to use. If None, uses the default
                scheduler (BackwardScheduler).

        Returns:
            List of TileInstructions with END marker at the end.
        """
        plan = self.dependency_plan()

        if not self._op_records:
            return [TileInstruction.end_instruction()]

        if scheduler is None:
            scheduler = BackwardScheduler()

        if hasattr(scheduler, "schedule_with_formulas"):
            schedule_formulas = plan.formulas
            instructions = scheduler.schedule_with_formulas(
                self._op_records,
                plan.edges,
                schedule_formulas,
            )
        else:
            instructions = scheduler.schedule(self._op_records, plan.consumer_deps, plan.edges)
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

    def pipeline_barrier_meta_indices(
        self,
        instructions: List[TileInstruction],
        *,
        expand_predicate=None,
    ) -> Tuple[List[int], List[TileInstruction]]:
        """Return coalesced-instruction metadata row bases and expanded rows.

        The replay stream may keep one coalesced instruction while the runtime
        expands its subtiles into ring slots. Wait/signal metadata remains
        per logical subtile, so the controller can use ``base + subtile_offset``
        for each emitted ring slot. Ops that own the coalesced range internally
        keep one metadata row containing all logical waits/signals.
        """
        meta_indices: List[int] = []
        expanded: List[TileInstruction] = []
        for instr in instructions:
            meta_indices.append(len(expanded))
            if instr.op_idx == TileInstruction.END_MARKER:
                expanded.append(instr)
                continue
            should_expand = True
            if expand_predicate is not None:
                should_expand = bool(expand_predicate(instr))
            if should_expand:
                for logical_tile in self._logical_tiles_for_instruction(instr):
                    expanded.append(TileInstruction(instr.op_idx, logical_tile))
            else:
                expanded.append(instr)
        return meta_indices, expanded

    def coalesce_pipeline_instructions(
        self,
        instructions: List[TileInstruction],
        *,
        num_blocks: Optional[int] = None,
        framework_expands_predicate=None,
    ) -> List[TileInstruction]:
        """Coalesce adjacent same-op instructions into coordinate ranges.

        Persistent CTAs fetch the instruction stream with a stride of
        ``num_blocks``. Framework-expanded ranges must preserve that strided
        work distribution, while op-owned ranges may be coalesced globally.
        """
        non_end = [instr for instr in instructions if instr.op_idx != TileInstruction.END_MARKER]

        def _max_range_tiles_for_op(op_idx: int) -> int:
            max_range_tiles = 0
            if num_blocks is not None and num_blocks > 0:
                total_tiles = max(1, int(self._op_records[op_idx].op.total_tiles))
                max_range_tiles = max(1, math.ceil(total_tiles / max(1, int(num_blocks))))
            op = self._op_records[op_idx].op
            explicit = op.static_dims.get(
                "max_coalesced_range_tiles",
                getattr(op.op_cls, "max_coalesced_range_tiles", None),
            )
            return int(explicit) if explicit is not None else max_range_tiles

        def _is_range(instr: TileInstruction) -> bool:
            return instr.range_axis >= 0

        def _framework_expands(instr: TileInstruction) -> bool:
            return framework_expands_predicate is not None and bool(framework_expands_predicate(instr))

        def _allow_op_owned_range(instr: TileInstruction) -> bool:
            return _is_range(instr) and not _framework_expands(instr)

        def _allow_framework_range(instr: TileInstruction) -> bool:
            return _is_range(instr) and (
                framework_expands_predicate is None or _framework_expands(instr)
            )

        def _coalesce_flat(stream: List[TileInstruction], allow_range=None) -> List[TileInstruction]:
            out: List[TileInstruction] = []
            idx = 0
            while idx < len(stream):
                if stream[idx].range_axis >= 0:
                    out.append(stream[idx])
                    idx += 1
                    continue
                op_idx = stream[idx].op_idx
                run_end = idx + 1
                while (
                    run_end < len(stream)
                    and stream[run_end].op_idx == op_idx
                    and stream[run_end].range_axis < 0
                ):
                    run_end += 1
                run = stream[idx:run_end]
                tile_rank = len(self._op_records[op_idx].op.tile_counts)
                run = _coalesce_coordinate_ranges(
                    run,
                    tile_rank=tile_rank,
                    max_range_tiles=_max_range_tiles_for_op(op_idx),
                    allow_range=allow_range,
                )
                out.extend(run)
                idx = run_end
            return out

        def _interleave_strided(coalesced: List[TileInstruction]) -> List[TileInstruction]:
            stream_count = max(1, int(num_blocks or 1))
            streams: List[List[TileInstruction]] = []
            for block_idx in range(stream_count):
                stream = [
                    coalesced[idx]
                    for idx in range(block_idx, len(coalesced), stream_count)
                ]
                streams.append(_coalesce_flat(stream, allow_range=_allow_framework_range))

            max_stream_len = max((len(stream) for stream in streams), default=0)
            strided_out: List[TileInstruction] = []
            end = TileInstruction.end_instruction()
            for stream_pos in range(max_stream_len + 1):
                for stream in streams:
                    if stream_pos < len(stream):
                        strided_out.append(stream[stream_pos])
                    else:
                        strided_out.append(end)
            return strided_out

        global_op_owned = _coalesce_flat(non_end, allow_range=_allow_op_owned_range)
        global_out = global_op_owned + [TileInstruction.end_instruction()]

        if num_blocks is None or num_blocks <= 1 or len(non_end) <= 1:
            return global_out

        def _logical_work_by_cta(instructions: List[TileInstruction]) -> List[int]:
            work = [0 for _ in range(max(1, int(num_blocks or 1)))]
            for idx, instr in enumerate(instructions):
                if instr.op_idx == TileInstruction.END_MARKER:
                    continue
                work[idx % len(work)] += _instruction_range_len(instr)
            return work

        conservative = _interleave_strided(global_op_owned)

        many_op_graph = len(self._op_records) > max(1, int(num_blocks or 1))
        if not many_op_graph:
            return conservative

        # Keep the per-CTA streams when globally coalescing framework-expanded
        # ranges would collapse work onto too few CTAs. The persistent runtime
        # fetches instruction i, i+num_blocks, ... per CTA, so preserving that
        # distribution matters more than reducing metadata rows for dense ops.
        return conservative

    @property
    def max_wait_deps(self) -> int:
        """Maximum number of wait dependencies across all ops."""
        return max(
            (len(waits) for waits in self.dependency_plan().controller_wait_formulas.values()),
            default=0,
        )

    @property
    def max_signal_deps(self) -> int:
        """Maximum number of signal dependencies across all ops."""
        formulas = self.dependency_plan().formulas
        return max((len(signals) for _waits, signals in formulas.values()), default=0)

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
        tensor, counts = self._build_wait_tensor(
            instructions,
            self._resolve_controller_wait_formulas(),
            device=device,
            num_blocks=num_blocks,
        )
        self._op_wait_counts = counts
        return tensor

    def build_compute_wait_info_tensor(
        self,
        instructions,
        device="cuda",
        num_blocks: Optional[int] = None,
    ):
        """Pre-compute waits that are delayed until compute phase."""
        tensor, counts = self._build_wait_tensor(
            instructions,
            self._resolve_compute_wait_formulas(),
            device=device,
            num_blocks=num_blocks,
        )
        self._op_compute_wait_counts = counts
        return tensor

    def _build_wait_tensor(
        self,
        instructions,
        wait_formulas: Dict[int, List[BarrierFormula]],
        *,
        device: str,
        num_blocks: Optional[int],
    ):
        """Build padded wait tensor and per-op wait counts for one wait phase."""
        import torch

        raw_wait_data = [
            self._build_wait_info_entry(
                instr,
                {instr.op_idx: (wait_formulas.get(instr.op_idx, []), [])}
                if instr.op_idx != TileInstruction.END_MARKER
                else {},
            )
            for instr in instructions
        ]
        max_waits = max(1, max((len(entry) // 2 for entry in raw_wait_data), default=0))
        wait_data = []
        for entry in raw_wait_data:
            padded = list(entry)
            while len(padded) < max_waits * 2:
                padded.extend([-1, 0])
            wait_data.append(padded)

        op_counts = self._max_counts_by_op(instructions, raw_wait_data, pair_width=2)
        empty = [-1, 0] * max_waits
        if num_blocks is not None and num_blocks > 0:
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

        return torch.tensor(wait_data, dtype=torch.int32, device=device), op_counts

    def build_signal_info_tensor(self, instructions, device="cuda"):
        """Pre-compute signal barrier indices per instruction.

        Returns tensor [num_instr, max_signals], where -1 means skip.
        """
        import torch

        formulas = self.dependency_plan().formulas
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
        range_axis = instr.range_axis
        preserve_range_multiplicity = len(logical_tiles) > 1
        for formula in signal_formulas:
            seen = set()
            preserve_formula_multiplicity = (
                preserve_range_multiplicity
                or (range_axis >= 0 and formula.coeffs[range_axis] != 0)
            )
            for tile in logical_tiles:
                if not (formula.has_guard and not formula.is_guarded(tile)):
                    barrier_idx = formula.compute_index(tile)
                    if preserve_formula_multiplicity or barrier_idx not in seen:
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
        return self.dependency_plan().formulas

    def export_dependency_graph_csv(self, op_csv: str, tile_csv: str) -> None:
        """Export op-level and tile-level dependency graphs as CSV files."""
        from .dependency_graph_export import export_dependency_graph_csv

        export_dependency_graph_csv(self, op_csv, tile_csv)

    @property
    def total_tiles(self) -> int:
        """Total number of work tiles."""
        return sum(r.op.total_tiles for r in self._op_records)

    @property
    def num_barriers(self) -> int:
        """Number of barriers needed."""
        return self.dependency_plan().barrier_count


__all__ = [
    "BarrierFormula",
    "DependencyPlan",
    "INSTRUCTION_WORDS",
    "INSTR_BARRIER_META_IDX",
    "INSTR_OP_IDX",
    "INSTR_RANGE_META",
    "INSTR_RANGE_END",
    "INSTR_TILE_01",
    "INSTR_TILE_23",
    "TileInstruction",
    "TileScheduler",
    "BackwardScheduler",
    "OverlapTileScheduler",
    "InstructionStreamBuilder",
]
