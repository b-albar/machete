# Copyright (c) 2025, Machete Authors
"""
Tests for instruction stream building and compile-time barrier formulas.

These tests verify the core scheduling logic that the megakernel relies on:
barrier formula computation, dependency chains, and instruction packing.
"""

import pytest
import torch
import cutlass.cute as cute
from machete.megakernel.backend import _build_compile_key, _phase_param_names_for_instance
from machete.megakernel.ops import MAX_TILE_DIMS, Op, PipelineSpec, ScheduledOp, build_op_config
from machete.megakernel.registries import TensorRegistry, validate_op_compatibility
from machete.megakernel.scheduling import (
    BarrierFormula,
    INSTR_BARRIER_META_IDX,
    INSTR_OP_IDX,
    INSTR_RANGE_END,
    INSTR_TILE_01,
    INSTR_TILE_23,
    INSTRUCTION_WORDS,
    InstructionStreamBuilder,
    TileInstruction,
    TileScheduler,
    BackwardScheduler,
    OverlapTileScheduler,
)
from typing import ClassVar, List


class _NOPOp(Op):
    """Test-only no-op for barrier formula and instruction stream tests."""
    pass


class _RangeCapableNOPOp(_NOPOp):
    pipeline = PipelineSpec(page_count=1)


class _UncappedRangeNOPOp(_RangeCapableNOPOp):
    max_coalesced_range_tiles = 0


class _AliasPhaseOp(Op):
    load_phase = "load_impl"
    compute_phase = "compute_impl"

    def load_impl(self, page_ptr, tile_M, x):
        pass

    def compute_impl(self, page_ptr, tile_M, y):
        pass


class _ComputeTmaLoadOp(Op):
    reads = {"x": (torch.float16, ("M", "K"))}
    writes = {"y": (torch.float16, ("M", "K"))}
    tile = ("M",)
    tma_compute_loads = {"x"}


class _SerialNonTmaLoadOp(Op):
    reads = {"x": (torch.float16, ("M",))}
    writes = {"y": (torch.float16, ("M",))}
    tile = ("M",)

    @cute.jit
    def load(self, page_ptr, tile_M, x, work_mbar):
        pass


class _CollectiveNonTmaLoadOp(_SerialNonTmaLoadOp):
    collective_non_tma_load = True


class _RangeCapableSerialNonTmaLoadOp(_SerialNonTmaLoadOp):
    pipeline = PipelineSpec(page_count=1)


def test_pipeline_spec_no_longer_declares_range_metadata():
    spec = PipelineSpec(page_count=1)

    assert spec.page_count == 1


def test_scheduler_coordinate_range_coalesces_and_expands_logical_tiles():
    op = ScheduledOp(
        _RangeCapableNOPOp,
        tile_counts=(1, 1, 10),
        dim_names={"B": 0, "S": 1, "O": 2},
    )
    builder = InstructionStreamBuilder()
    builder.add_op(op)

    instructions = builder.coalesce_pipeline_instructions(builder.build())
    non_end = [instr for instr in instructions if instr.op_idx != TileInstruction.END_MARKER]

    assert [instr.tiles for instr in non_end] == [
        (0, 0, 0, 0),
    ]
    assert [(instr.range_axis, instr.range_end_axis) for instr in non_end] == [(2, 3)]
    assert [instr.range_end for instr in non_end] == [10]
    _, expanded = builder.pipeline_barrier_meta_indices(instructions)
    assert [instr.tiles for instr in expanded if instr.op_idx != TileInstruction.END_MARKER] == [
        (0, 0, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 2, 0),
        (0, 0, 3, 0),
        (0, 0, 4, 0),
        (0, 0, 5, 0),
        (0, 0, 6, 0),
        (0, 0, 7, 0),
        (0, 0, 8, 0),
        (0, 0, 9, 0),
    ]


def test_pipeline_range_coalesces_per_cta_fetch_stream():
    """Same-op tiles consecutive per CTA should form ranges even if interleaved globally."""
    op = ScheduledOp(
        _RangeCapableNOPOp,
        tile_counts=(1, 1, 110),
        dim_names={"B": 0, "S": 1, "O": 2},
    )
    builder = InstructionStreamBuilder()
    builder.add_op(op)
    instructions = []
    for i in range(10):
        instructions.append(TileInstruction(0, (0, 0, i)))
        instructions.append(TileInstruction(0, (0, 0, 100 + i)))
    instructions.append(TileInstruction.end_instruction())

    coalesced = builder.coalesce_pipeline_instructions(instructions, num_blocks=2)

    assert [instr.tiles for instr in coalesced] == [
        (0, 0, 0, 0),
        (0, 0, 100, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    ]
    assert [instr.range_end for instr in coalesced[:2]] == [10, 110]


def test_framework_expanded_ranges_preserve_cta_distribution():
    op = ScheduledOp(
        _RangeCapableNOPOp,
        tile_counts=(1, 1, 110),
        dim_names={"B": 0, "S": 1, "O": 2},
    )
    builder = InstructionStreamBuilder()
    builder.add_op(op)
    instructions = []
    for i in range(10):
        instructions.append(TileInstruction(0, (0, 0, i)))
        instructions.append(TileInstruction(0, (0, 0, 100 + i)))
    instructions.append(TileInstruction.end_instruction())

    coalesced = builder.coalesce_pipeline_instructions(
        instructions,
        num_blocks=2,
        framework_expands_predicate=lambda instr: instr.range_axis >= 0,
    )

    assert [instr.tiles for instr in coalesced] == [
        (0, 0, 0, 0),
        (0, 0, 100, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    ]
    assert [instr.range_end for instr in coalesced[:2]] == [10, 110]


def test_op_can_request_uncapped_coordinate_range():
    op = ScheduledOp(
        _UncappedRangeNOPOp,
        tile_counts=(1, 1, 16),
        dim_names={"B": 0, "S": 1, "O": 2},
    )
    builder = InstructionStreamBuilder()
    builder.add_op(op)

    coalesced = builder.coalesce_pipeline_instructions(builder.build(), num_blocks=4)
    work = [instr for instr in coalesced if instr.op_idx != TileInstruction.END_MARKER]

    assert len(work) == 1
    assert work[0].tiles == (0, 0, 0, 0)
    assert work[0].range_end == 16


def test_phase_aliases_bind_concrete_instance_methods():
    op = _AliasPhaseOp()

    assert op.load.__func__ is _AliasPhaseOp.load_impl
    assert op.compute.__func__ is _AliasPhaseOp.compute_impl
    assert _phase_param_names_for_instance(op, "load") == ("page_ptr", "tile_M", "x")
    assert _phase_param_names_for_instance(op, "compute") == ("page_ptr", "tile_M", "y")


def test_compute_phase_tma_load_declaration_is_registered():
    assert _ComputeTmaLoadOp._TMA_COMPUTE_LOADS == {"x"}
    assert _ComputeTmaLoadOp._TMA_LOADS == set()
    assert _ComputeTmaLoadOp.gen_tma_param_names("compute") == [
        ("x_tma", "x_tma_gmem", "x")
    ]


def test_non_tma_loads_default_to_elect_one_wrapper():
    from machete.megakernel.compile import compile_phase, _linecache_entries
    import linecache

    compile_phase(_SerialNonTmaLoadOp(), "load", tensor_param_names=["x"])
    source = "".join(linecache.cache[_linecache_entries[-1]][2])

    assert "with cute.arch.elect_one():" in source


def test_collective_non_tma_load_owns_dma_warp_and_mbarrier():
    from machete.megakernel.compile import compile_phase, _linecache_entries
    import linecache

    compile_phase(_CollectiveNonTmaLoadOp(), "load", tensor_param_names=["x"])
    source = "".join(linecache.cache[_linecache_entries[-1]][2])

    assert "with cute.arch.elect_one():" not in source
    assert "_instance.load(page_ptr, tile_0, x, work_mbar)" in source


def test_barrier_static_dims_do_not_specialize_device_code():
    """Barrier metadata is scheduler-only and must not duplicate handlers."""
    op_a = ScheduledOp(
        _NOPOp,
        static_dims={"M": 16, "barrier_wait_alias_H": "layer_0"},
    )
    op_b = ScheduledOp(
        _NOPOp,
        static_dims={"M": 16, "barrier_wait_alias_H": "layer_1"},
    )

    key_a = _build_compile_key(
        op_a,
        all_local_tensor_names=(),
        local_tensor_names={phase: () for phase in ("load", "compute", "store", "communicate")},
        local_tma_args={phase: () for phase in ("load", "compute", "store", "communicate")},
        tensor_args={phase: () for phase in ("load", "compute", "store", "communicate")},
        tma_args={phase: () for phase in ("load", "compute", "store", "communicate")},
    )
    key_b = _build_compile_key(
        op_b,
        all_local_tensor_names=(),
        local_tensor_names={phase: () for phase in ("load", "compute", "store", "communicate")},
        local_tma_args={phase: () for phase in ("load", "compute", "store", "communicate")},
        tensor_args={phase: () for phase in ("load", "compute", "store", "communicate")},
        tma_args={phase: () for phase in ("load", "compute", "store", "communicate")},
    )

    assert key_a == key_b
    assert build_op_config(op_a)["M"] == 16
    assert "barrier_wait_alias_H" not in build_op_config(op_a)


def test_tile_sizes_specialize_device_code():
    """Tile sizes are used in CuTe compile-time loops and must key handlers."""
    op_a = ScheduledOp(_NOPOp, tile_sizes={"M": 16})
    op_b = ScheduledOp(_NOPOp, tile_sizes={"M": 64})

    common = {
        "all_local_tensor_names": (),
        "local_tensor_names": {phase: () for phase in ("load", "compute", "store", "communicate")},
        "local_tma_args": {phase: () for phase in ("load", "compute", "store", "communicate")},
        "tensor_args": {phase: () for phase in ("load", "compute", "store", "communicate")},
        "tma_args": {phase: () for phase in ("load", "compute", "store", "communicate")},
    }

    assert _build_compile_key(op_a, **common) != _build_compile_key(op_b, **common)
    assert build_op_config(op_a)["tile_size_M"] == 16
    assert build_op_config(op_b)["tile_size_M"] == 64


# =============================================================================
# Linear Chain Tests (no INPUTS/OUTPUTS)
# =============================================================================


class TestLinearChainFormulas:
    """Test that linear chain ops (no INPUTS/OUTPUTS) produce correct barrier formulas."""

    def test_first_op_has_no_wait_formulas(self):
        """First op should have no wait formulas."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(4,))
        formulas = builder.get_op_barrier_formulas()

        wait, signal = formulas[0]
        assert len(wait) == 0
        assert len(signal) == 1

    def test_first_op_signals_own_barrier(self):
        """First op should signal barrier = tiles[0] (base=0, coeffs[0]=1)."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(4,))
        formulas = builder.get_op_barrier_formulas()

        _, signal = formulas[0]
        sf = signal[0]
        assert sf.base == 0
        assert sf.coeffs[0] == 1
        # Higher coeffs are strides for the linear index
        # but tiles beyond dim 0 are 0 for 1D ops, so they don't affect the result

        # Verify: tiles[0]=0 → barrier 0, tiles[0]=3 → barrier 3
        assert sf.compute_index((0, 0, 0)) == 0
        assert sf.compute_index((3, 0, 0)) == 3

    def test_second_op_waits_on_first(self):
        """Second op should wait on first op's barriers with expected=1."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(4,))
        builder.add_op(_NOPOp, tile_counts=(4,))
        formulas = builder.get_op_barrier_formulas()

        wait, signal = formulas[1]
        assert len(wait) == 1
        assert len(signal) == 1

        wf = wait[0]
        assert wf.base == 0  # Wait on op0's barriers
        assert wf.coeffs[0] == 1
        assert wf.expected == 1
        assert not wf.has_guard  # Same tile count, no guard

        sf = signal[0]
        assert sf.base == 4  # Signal op1's own barriers

    def test_three_op_chain(self):
        """Three-op chain: op2 waits on op1 (not op0)."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(2,))  # barriers 0-1
        builder.add_op(_NOPOp, tile_counts=(2,))  # barriers 2-3
        builder.add_op(_NOPOp, tile_counts=(2,))  # barriers 4-5

        formulas = builder.get_op_barrier_formulas()
        assert builder.num_barriers == 6

        # Op 0: no wait, signal base=0
        assert len(formulas[0][0]) == 0
        assert formulas[0][1][0].base == 0

        # Op 1: wait base=0, signal base=2
        assert formulas[1][0][0].base == 0
        assert formulas[1][1][0].base == 2

        # Op 2: wait base=2, signal base=4
        assert formulas[2][0][0].base == 2
        assert formulas[2][1][0].base == 4

    def test_barrier_count_matches_total_tiles(self):
        """num_barriers should equal total tiles across all ops."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(16,))
        builder.add_op(_NOPOp, tile_counts=(16,))
        builder.add_op(_NOPOp, tile_counts=(16,))

        assert builder.num_barriers == 48
        assert builder.total_tiles == 48

    def test_2d_formula_strides(self):
        """2D grid should produce correct strides in formulas (row-major)."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(4, 3))
        builder.add_op(_NOPOp, tile_counts=(4, 3))
        formulas = builder.get_op_barrier_formulas()

        sf = formulas[0][1][0]
        assert sf.coeffs[0] == 3   # stride for dim 0 = tile_counts[1] = 3
        assert sf.coeffs[1] == 1   # stride for dim 1 = 1 (innermost)

        # Verify row-major: (0,0,0)→0, (1,0,0)→3, (0,1,0)→1, (3,2,0)→11
        assert sf.compute_index((0, 0, 0)) == 0
        assert sf.compute_index((1, 0, 0)) == 3
        assert sf.compute_index((0, 1, 0)) == 1
        assert sf.compute_index((3, 2, 0)) == 11

    def test_3d_formula_strides(self):
        """3D grid should produce correct strides in formulas (row-major)."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(2, 3, 2))
        formulas = builder.get_op_barrier_formulas()

        sf = formulas[0][1][0]
        assert sf.coeffs[0] == 6  # stride for dim 0 = tile_counts[1] * tile_counts[2] = 3*2
        assert sf.coeffs[1] == 2  # stride for dim 1 = tile_counts[2] = 2
        assert sf.coeffs[2] == 1  # stride for dim 2 = 1 (innermost)


class TestMismatchedTileCounts:
    """Test dependency behavior when sequential ops have different tile counts."""

    def test_fewer_tiles_in_second_op(self):
        """When op1 has fewer tiles, guard is set to prev op's total tiles."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(8,))
        builder.add_op(_NOPOp, tile_counts=(4,))
        formulas = builder.get_op_barrier_formulas()

        wf = formulas[1][0][0]
        assert wf.base == 0
        assert wf.expected == 1
        assert wf.has_guard  # Guard active because tile counts differ
        assert wf.guard_max == 8

    def test_more_tiles_in_second_op(self):
        """When op1 has more tiles, guard prevents waiting on non-existent barriers."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(4,))
        builder.add_op(_NOPOp, tile_counts=(8,))
        formulas = builder.get_op_barrier_formulas()

        wf = formulas[1][0][0]
        assert wf.base == 0
        assert wf.expected == 1
        assert wf.has_guard
        assert wf.guard_max == 4  # Only 4 barriers from op0

        # Tiles 0-3 pass guard, tiles 4-7 don't
        assert wf.is_guarded((0, 0, 0)) is True
        assert wf.is_guarded((3, 0, 0)) is True
        assert wf.is_guarded((4, 0, 0)) is False
        assert wf.is_guarded((7, 0, 0)) is False

    def test_same_tiles_no_guard(self):
        """When tile counts match, no guard is needed."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(4,))
        builder.add_op(_NOPOp, tile_counts=(4,))
        formulas = builder.get_op_barrier_formulas()

        wf = formulas[1][0][0]
        assert not wf.has_guard  # No guard needed


class TestInstructionStreamMultiDim:
    """Test instruction stream with multi-dimensional tile grids."""

    def test_2d_tile_indices(self):
        """Tiles should have correct (tiles[0], tiles[1]) indices for a 2D grid."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(4, 3))
        instructions = builder.build()[:-1]

        assert len(instructions) == 12
        indices = {(i.tiles[0], i.tiles[1]) for i in instructions}
        expected = {(m, n) for m in range(4) for n in range(3)}
        assert indices == expected

    def test_3d_tile_indices(self):
        """Tiles should have correct (tiles[0], tiles[1], tiles[2]) indices for a 3D grid."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(2, 3, 2))
        instructions = builder.build()[:-1]

        assert len(instructions) == 12
        indices = {(i.tiles[0], i.tiles[1], i.tiles[2]) for i in instructions}
        expected = {(m, n, ll) for m in range(2) for n in range(3) for ll in range(2)}
        assert indices == expected


class TestTileInstructionPacking:
    """Test that instructions pack correctly for GPU transfer."""

    def test_pack_size(self):
        """Packed instruction should be exactly INSTRUCTION_WORDS wide."""
        instr = TileInstruction(op_idx=1, tiles=(5, 2, 0))
        packed = instr.pack()
        assert len(packed) == INSTRUCTION_WORDS
        assert INSTRUCTION_WORDS == 5

    def test_pack_fields(self):
        """Fields should pack op index, tile coordinates, and barrier metadata."""
        instr = TileInstruction(op_idx=3, tiles=(7, 1, 2))
        packed = instr.pack(barrier_meta_idx=99)
        assert packed[INSTR_OP_IDX] == 3
        assert packed[INSTR_TILE_01] == 7 | (1 << 16)
        assert packed[INSTR_TILE_23] == 2
        assert packed[INSTR_BARRIER_META_IDX] == 99
        assert packed[INSTR_RANGE_END] == 0

    def test_end_marker_packs_correctly(self):
        """End marker should have op_idx == END_MARKER."""
        end = TileInstruction.end_instruction()
        packed = end.pack()
        assert packed[0] == 0xFFFF

    def test_build_tensor_shape(self):
        """GPU tensor should have shape [num_instructions, INSTRUCTION_WORDS]."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(10,))
        tensor = builder.build_tensor(device="cpu")

        # 10 tiles + 1 end marker
        assert tensor.shape == (11, INSTRUCTION_WORDS)
        assert tensor.dtype == torch.int32

    def test_build_tensor_roundtrip(self):
        """Packed tensor values should encode op index and tile coordinates."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(4,))
        builder.add_op(_NOPOp, tile_counts=(4,))
        tensor = builder.build_tensor(device="cpu")
        instructions = builder.build()

        for i, instr in enumerate(instructions):
            row = tensor[i].tolist()
            row_op = row[INSTR_OP_IDX] & 0xFFFF
            if instr.op_idx != TileInstruction.END_MARKER:
                assert row_op == instr.op_idx, f"Instruction {i} op_idx mismatch"
                assert row[INSTR_TILE_01] & 0xFFFF == instr.tiles[0], f"Instruction {i} tile mismatch"
            else:
                assert row_op == 0xFFFF, f"Instruction {i} end marker mismatch"


# =============================================================================
# Named Buffer Dependency Tests
# =============================================================================


class _ProducerOp(Op):
    """Test op that produces buffer 'x'."""
    INPUTS: ClassVar[List[str]] = []
    OUTPUTS: ClassVar[List[str]] = ["x"]


class _ConsumerOp(Op):
    """Test op that consumes buffer 'x'."""
    INPUTS: ClassVar[List[str]] = ["x"]
    OUTPUTS: ClassVar[List[str]] = []


class _ProducerY(Op):
    """Test op that produces buffer 'y'."""
    INPUTS: ClassVar[List[str]] = []
    OUTPUTS: ClassVar[List[str]] = ["y"]


class _FanInOp(Op):
    """Test op that consumes both 'x' and 'y'."""
    INPUTS: ClassVar[List[str]] = ["x", "y"]
    OUTPUTS: ClassVar[List[str]] = []


class _PackedQKVProducerOp(Op):
    reads = {}
    writes = {"qkv": (None, ("B", "S", "N"))}
    tile = ("B", "S")
    OUTPUTS: ClassVar[List[str]] = ["qkv"]


class _PackedQConsumerOp(Op):
    reads = {"q": (None, ("B", "S", "Q"))}
    writes = {}
    tile = ("B", "S")
    INPUTS: ClassVar[List[str]] = ["q"]


class _PackedKConsumerOp(Op):
    reads = {"k": (None, ("B", "S", "K"))}
    writes = {}
    tile = ("B", "S")
    INPUTS: ClassVar[List[str]] = ["k"]


class _PackedVConsumerOp(Op):
    reads = {"v": (None, ("B", "S", "V"))}
    writes = {}
    tile = ("B", "S")
    INPUTS: ClassVar[List[str]] = ["v"]


class _PackedV4DConsumerOp(Op):
    reads = {"v": (None, ("B", "S", "H", "D"))}
    writes = {}
    tile = ("B", "S")
    INPUTS: ClassVar[List[str]] = ["v"]


class _ScratchWriterOp(Op):
    reads = {}
    writes = {"x": (None, ("B", "S"))}
    tile = ("B", "S")
    OUTPUTS: ClassVar[List[str]] = ["x"]


class _ScratchReaderOp(Op):
    reads = {"x": (None, ("B", "S"))}
    writes = {}
    tile = ("B", "S")
    INPUTS: ClassVar[List[str]] = ["x"]


class TestNamedBufferDeps:
    """Test dependency resolution with named buffers and dim_names."""

    def test_one_to_one_named(self):
        """Same dims on both sides → 1:1 tile mapping, expected=1."""
        builder = InstructionStreamBuilder()
        builder.add_op(_ProducerOp, tile_counts=(4,),
                       dim_names={"batch": 0})
        builder.add_op(_ConsumerOp, tile_counts=(4,),
                       dim_names={"batch": 0})
        formulas = builder.get_op_barrier_formulas()

        # Producer signals
        p_signal = formulas[0][1][0]
        assert p_signal.base == 0
        assert p_signal.coeffs[0] == 1
        assert p_signal.compute_index((0, 0, 0)) == 0
        assert p_signal.compute_index((3, 0, 0)) == 3

        # Consumer waits
        c_wait = formulas[1][0][0]
        assert c_wait.base == 0
        assert c_wait.coeffs[0] == 1
        assert c_wait.expected == 1
        # Same barrier index for matching tile
        assert c_wait.compute_index((2, 0, 0)) == p_signal.compute_index((2, 0, 0))

    def test_many_to_one(self):
        """Producer (batch=4, seqlen=8) → Consumer (batch=4).

        Consumer batch=0 waits for all 8 seqlen tiles with expected=8.
        """
        builder = InstructionStreamBuilder()
        builder.add_op(_ProducerOp, tile_counts=(4, 8),
                       dim_names={"batch": 0, "seqlen": 1})
        builder.add_op(_ConsumerOp, tile_counts=(4,),
                       dim_names={"batch": 0})
        formulas = builder.get_op_barrier_formulas()

        assert builder.num_barriers == 4  # One per consumer tile

        # Consumer waits with expected=8 (collapsed seqlen)
        c_wait = formulas[1][0][0]
        assert c_wait.expected == 8
        assert c_wait.coeffs[0] == 1
        assert c_wait.coeffs[1] == 0  # Consumer has no dim 1
        assert c_wait.compute_index((0, 0, 0)) == 0  # batch 0 → barrier 0
        assert c_wait.compute_index((3, 0, 0)) == 3  # batch 3 → barrier 3

        # Producer signals: all seqlen tiles for batch 0 → same barrier
        p_signal = formulas[0][1][0]
        assert p_signal.coeffs[0] == 1  # batch maps to target's dim 0
        assert p_signal.coeffs[1] == 0  # seqlen collapsed (not in target)
        assert p_signal.compute_index((0, 0, 0)) == 0  # (batch=0, seqlen=0) → barrier 0
        assert p_signal.compute_index((0, 5, 0)) == 0  # (batch=0, seqlen=5) → barrier 0
        assert p_signal.compute_index((2, 3, 0)) == 2  # (batch=2, seqlen=3) → barrier 2

    def test_one_to_many(self):
        """Producer (batch=4) → Consumer (batch=4, seqlen=8).

        All 8 consumer seqlen tiles for batch=0 wait on one producer barrier.
        """
        builder = InstructionStreamBuilder()
        builder.add_op(_ProducerOp, tile_counts=(4,),
                       dim_names={"batch": 0})
        builder.add_op(_ConsumerOp, tile_counts=(4, 8),
                       dim_names={"batch": 0, "seqlen": 1})
        formulas = builder.get_op_barrier_formulas()

        assert builder.num_barriers == 4  # One per producer tile

        # Producer signals: one barrier per batch tile
        p_signal = formulas[0][1][0]
        assert p_signal.coeffs[0] == 1
        assert p_signal.compute_index((0, 0, 0)) == 0
        assert p_signal.compute_index((3, 0, 0)) == 3

        # Consumer waits: all seqlen tiles for batch 0 wait on barrier 0
        c_wait = formulas[1][0][0]
        assert c_wait.expected == 1
        assert c_wait.coeffs[0] == 1  # batch maps to producer's dim 0
        assert c_wait.coeffs[1] == 0  # seqlen not in producer (broadcast)
        assert c_wait.compute_index((0, 0, 0)) == 0  # (batch=0, seqlen=0) → barrier 0

    def test_packed_qkv_views_resolve_dependencies(self):
        """Packed producer output should feed q/k/v slice consumers."""
        qkv = torch.empty(1, 4, 12)
        q = qkv[:, :, :8]
        k = qkv[:, :, 8:10]
        v = qkv[:, :, 10:]

        prod = _PackedQKVProducerOp.schedule(
            qkv=qkv,
            tile_sizes={"B": 1, "S": 1},
        )[0]
        q_cons = _PackedQConsumerOp.schedule(
            q=q,
            tile_sizes={"B": 1, "S": 1},
        )[0]
        k_cons = _PackedKConsumerOp.schedule(
            k=k,
            tile_sizes={"B": 1, "S": 1},
        )[0]
        v_cons = _PackedVConsumerOp.schedule(
            v=v,
            tile_sizes={"B": 1, "S": 1},
        )[0]

        validate_op_compatibility([prod, q_cons, k_cons, v_cons], TensorRegistry.from_ops([prod, q_cons, k_cons, v_cons]))

        builder = InstructionStreamBuilder()
        builder.add_op(prod)
        builder.add_op(q_cons)
        builder.add_op(k_cons)
        builder.add_op(v_cons)
        formulas = builder.get_op_barrier_formulas()
        assert builder.num_barriers == 4

        for op_idx in (1, 2, 3):
            wait = formulas[op_idx][0]
            assert len(wait) == 1
            assert wait[0].expected == 1
            assert wait[0].base == formulas[0][1][0].base

    def test_packed_flat_producer_feeds_rank_changed_view(self):
        """Flat producer output should feed a BSHD slice view of same storage."""
        qkv = torch.empty(1, 4, 12)
        v_4d = qkv.view(1, 4, 3, 4)[:, :, 2:, :]

        prod = _PackedQKVProducerOp.schedule(
            qkv=qkv,
            tile_sizes={"B": 1, "S": 1},
        )[0]
        v_cons = _PackedV4DConsumerOp.schedule(
            v=v_4d,
            tile_sizes={"B": 1, "S": 1},
        )[0]

        builder = InstructionStreamBuilder()
        builder.add_op(prod)
        builder.add_op(v_cons)
        formulas = builder.get_op_barrier_formulas()

        wait = formulas[1][0]
        assert len(wait) == 1
        assert wait[0].expected == 1

    def test_fan_in(self):
        """OpA → "x", OpB → "y", OpC reads ["x", "y"]. OpC has 2 wait formulas."""
        builder = InstructionStreamBuilder()
        builder.add_op(_ProducerOp, tile_counts=(4,),
                       dim_names={"batch": 0})
        builder.add_op(_ProducerY, tile_counts=(4,),
                       dim_names={"batch": 0})
        builder.add_op(_FanInOp, tile_counts=(4,),
                       dim_names={"batch": 0})
        formulas = builder.get_op_barrier_formulas()

        # OpC waits on 2 barriers (one per input buffer)
        c_waits = formulas[2][0]
        assert len(c_waits) == 2
        for wf in c_waits:
            assert wf.expected == 1
            assert wf.coeffs[0] == 1

        # Different barrier bases for each edge
        assert c_waits[0].base != c_waits[1].base

    def test_fan_out(self):
        """OpA → "x", both OpB and OpC read "x". OpA signals once."""
        builder = InstructionStreamBuilder()
        builder.add_op(_ProducerOp, tile_counts=(4,),
                       dim_names={"batch": 0})
        builder.add_op(_ConsumerOp, tile_counts=(4,),
                       dim_names={"batch": 0})
        # Second consumer also reads "x"
        builder.add_op(_ConsumerOp, tile_counts=(4,),
                       dim_names={"batch": 0})
        formulas = builder.get_op_barrier_formulas()

        # Both consumers can wait on the same producer-side readiness signal.
        p_signals = formulas[0][1]
        assert len(p_signals) == 1

        assert formulas[1][0][0].base == p_signals[0].base
        assert formulas[2][0][0].base == p_signals[0].base

    def test_reused_scratch_antideps_do_not_accumulate_waits(self):
        """Repeated scratch reuse should keep wait formulas on the frontier."""
        scratch = torch.empty(1, 8)
        builder = InstructionStreamBuilder()
        for _ in range(8):
            builder.add_op(_ScratchWriterOp.schedule(
                x=scratch, tile_sizes={"B": 1, "S": 1},
            )[0])
            builder.add_op(_ScratchReaderOp.schedule(
                x=scratch, tile_sizes={"B": 1, "S": 1},
            )[0])

        assert builder.max_wait_deps <= 2

    def test_buffer_not_produced_is_external(self):
        """Consuming a buffer with no producer is treated as an external input."""
        builder = InstructionStreamBuilder()
        builder.add_op(_ConsumerOp, tile_counts=(4,), dim_names={"batch": 0})
        # No ValueError — unproduced INPUTS are external (host-provided) tensors
        instructions = builder.build()
        assert len(instructions) > 0

    def test_duplicate_producer_raises(self):
        """Two ops producing the same buffer name should raise."""
        builder = InstructionStreamBuilder()
        builder.add_op(_ProducerOp, tile_counts=(4,), dim_names={"batch": 0})
        builder.add_op(_ProducerOp, tile_counts=(4,), dim_names={"batch": 0})
        with pytest.raises(ValueError, match="produced by both"):
            builder.build()


# =============================================================================
# BarrierFormula Unit Tests
# =============================================================================


class TestBarrierFormula:
    """Test BarrierFormula compute_index and is_guarded methods."""

    def test_compute_index_1d(self):
        f = BarrierFormula(base=10, coeffs=(1, 0, 0, 0, 0))
        assert f.compute_index((0, 0, 0)) == 10
        assert f.compute_index((5, 0, 0)) == 15

    def test_compute_index_2d(self):
        f = BarrierFormula(base=0, coeffs=(1, 4, 0, 0, 0))
        assert f.compute_index((0, 0, 0)) == 0
        assert f.compute_index((3, 0, 0)) == 3
        assert f.compute_index((0, 2, 0)) == 8
        assert f.compute_index((3, 2, 0)) == 11

    def test_compute_index_3d(self):
        f = BarrierFormula(base=100, coeffs=(1, 2, 6, 0, 0))
        assert f.compute_index((0, 0, 0)) == 100
        assert f.compute_index((1, 2, 1)) == 100 + 1 + 4 + 6  # 111

    def test_no_guard_always_passes(self):
        f = BarrierFormula(base=0, coeffs=(1, 0, 0, 0, 0))  # default NO_GUARD
        assert f.is_guarded((100, 0, 0)) is True
        assert not f.has_guard

    def test_guard_max_blocks(self):
        f = BarrierFormula(base=0, coeffs=(1, 0, 0, 0, 0), guard_max=4)
        assert f.is_guarded((0, 0, 0)) is True
        assert f.is_guarded((3, 0, 0)) is True
        assert f.is_guarded((4, 0, 0)) is False
        assert f.is_guarded((7, 0, 0)) is False


class TestDimNamesValidation:
    """Test validation of dim_names in add_op."""

    def test_invalid_axis_rejected(self):
        builder = InstructionStreamBuilder()
        with pytest.raises(ValueError, match="Invalid axis x for dim 'batch'. Must be int in"):
            builder.add_op(_NOPOp, tile_counts=(4,), dim_names={"batch": "x"})

    def test_duplicate_axis_rejected(self):
        builder = InstructionStreamBuilder()
        with pytest.raises(ValueError, match="multiple dims to the same axis"):
            builder.add_op(
                _NOPOp, tile_counts=(4, 2),
                dim_names={"batch": 0, "tokens": 0},
            )

    def test_valid_dim_names_accepted(self):
        builder = InstructionStreamBuilder()
        builder.add_op(
            _NOPOp, tile_counts=(4, 2, 3),
            dim_names={"batch": 0, "seqlen": 1, "heads": 2},
        )
        assert len(builder.ops) == 1


class TestLinecacheCleanup:
    """Test that compile-time linecache entries can be cleaned up."""

    def test_cleanup_removes_entries(self):
        from machete.megakernel.compile import (
            compile_phase,
            cleanup_linecache,
            _linecache_entries,
        )
        import linecache

        before = len(_linecache_entries)
        compile_phase(_NOPOp(), "compute")
        assert len(_linecache_entries) > before

        # Verify entries exist in linecache
        for entry in _linecache_entries:
            assert entry in linecache.cache

        removed = cleanup_linecache()
        assert removed > 0
        assert len(_linecache_entries) == 0


# =============================================================================
# Level-Batched Tile Scheduling Tests
# =============================================================================


class TestLevelBatchedScheduling:
    """Test that build() uses level-batched scheduling for load balancing.

    Level-batched scheduling emits all tiles from source ops (no dependencies)
    first, then processes dependent ops. This spreads producer tiles across
    SMs when using strided instruction fetch.
    """

    def test_two_op_chain_batched(self):
        """A 2-op chain emits all source tiles before consumer tiles."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(4,))
        builder.add_op(_NOPOp, tile_counts=(4,))
        instructions = builder.build()[:-1]  # strip end marker

        assert len(instructions) == 8
        op_sequence = [i.op_idx for i in instructions]

        # With level-batched scheduling, all source tiles come first
        # Expected: [0, 0, 0, 0, 1, 1, 1, 1]
        assert op_sequence == [0, 0, 0, 0, 1, 1, 1, 1], f"Expected batched order but got: {op_sequence}"

    def test_three_op_chain_batched(self):
        """A 3-op chain emits source first, then wavefront for consumers."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(4,))
        builder.add_op(_NOPOp, tile_counts=(4,))
        builder.add_op(_NOPOp, tile_counts=(4,))
        instructions = builder.build()[:-1]

        assert len(instructions) == 12
        op_sequence = [i.op_idx for i in instructions]

        # Source op 0 emits all tiles first (level-batched)
        # Then ops 1 and 2 interleave via greedy wavefront
        # Expected: [0,0,0,0, 1,2,1,2,1,2,1,2]
        assert op_sequence[:4] == [0, 0, 0, 0], f"Source tiles not first: {op_sequence}"
        # Remaining tiles should have both op 1 and op 2 interleaved
        remaining = op_sequence[4:]
        assert sorted(remaining) == [1, 1, 1, 1, 2, 2, 2, 2], f"Missing tiles: {remaining}"
        # First op1 tile should appear before any op2 tiles complete
        first_op2 = remaining.index(2)
        assert remaining[0] == 1, f"Op 1 should start before op 2: {remaining}"
        assert first_op2 > 0, f"Op 2 shouldn't start first: {remaining}"

    def test_all_tiles_present(self):
        """Interleaved build must emit all tiles from all ops."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(3, 2))
        builder.add_op(_NOPOp, tile_counts=(3, 2))
        instructions = builder.build()[:-1]

        assert len(instructions) == 12

        # Check op 0 has all 6 tiles
        op0_tiles = {(i.tiles[0], i.tiles[1]) for i in instructions if i.op_idx == 0}
        assert op0_tiles == {(m, n) for m in range(3) for n in range(2)}

        # Check op 1 has all 6 tiles
        op1_tiles = {(i.tiles[0], i.tiles[1]) for i in instructions if i.op_idx == 1}
        assert op1_tiles == {(m, n) for m in range(3) for n in range(2)}

    def test_single_op_unchanged(self):
        """A single op should emit all tiles in order (no deps)."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(4,))
        instructions = builder.build()[:-1]

        assert len(instructions) == 4
        assert all(i.op_idx == 0 for i in instructions)
        assert [i.tiles[0] for i in instructions] == [0, 1, 2, 3]

    def test_dependency_order_respected(self):
        """Consumer tile k must appear after producer tile k in the stream."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(4,))
        builder.add_op(_NOPOp, tile_counts=(4,))
        instructions = builder.build()[:-1]

        # Track emission position of each (op_idx, tiles[0])
        positions = {}
        for pos, instr in enumerate(instructions):
            positions[(instr.op_idx, instr.tiles[0])] = pos

        # For 1:1 chain: consumer tile k must be after producer tile k
        for k in range(4):
            assert positions[(1, k)] > positions[(0, k)], (
                f"Consumer tile {k} at pos {positions[(1, k)]} "
                f"before producer tile {k} at pos {positions[(0, k)]}"
            )


# =============================================================================
# Scheduler API Tests
# =============================================================================


class TestSchedulerAPI:
    """Test the tile scheduler abstraction and different schedulers."""

    def test_explicit_scheduler_parameter(self):
        """Can pass scheduler directly to build()."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(4,))
        builder.add_op(_NOPOp, tile_counts=(4,))

        # Using explicit scheduler should work
        scheduler = BackwardScheduler()
        instructions = builder.build(scheduler=scheduler)
        assert len(instructions) == 9  # 8 tiles + end marker

    def test_backward_scheduler_respects_dependencies(self):
        """BackwardScheduler must still respect dependencies."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(4,))
        builder.add_op(_NOPOp, tile_counts=(4,))

        scheduler = BackwardScheduler()
        instructions = builder.build(scheduler=scheduler)[:-1]

        # Track emission position of each (op_idx, tiles[0])
        positions = {}
        for pos, instr in enumerate(instructions):
            positions[(instr.op_idx, instr.tiles[0])] = pos

        # For 1:1 chain: consumer tile k must be after producer tile k
        for k in range(4):
            assert positions[(1, k)] > positions[(0, k)], (
                f"BackwardScheduler violated deps: consumer tile {k} at pos {positions[(1, k)]} "
                f"before producer tile {k} at pos {positions[(0, k)]}"
            )

    def test_backward_scheduler_all_tiles_present(self):
        """BackwardScheduler must emit all tiles."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(3, 2))
        builder.add_op(_NOPOp, tile_counts=(3, 2))

        scheduler = BackwardScheduler()
        instructions = builder.build(scheduler=scheduler)[:-1]

        assert len(instructions) == 12

        # Check op 0 has all 6 tiles
        op0_tiles = {(i.tiles[0], i.tiles[1]) for i in instructions if i.op_idx == 0}
        assert op0_tiles == {(m, n) for m in range(3) for n in range(2)}

        # Check op 1 has all 6 tiles
        op1_tiles = {(i.tiles[0], i.tiles[1]) for i in instructions if i.op_idx == 1}
        assert op1_tiles == {(m, n) for m in range(3) for n in range(2)}

    def test_scheduler_is_abstract(self):
        """TileScheduler base class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TileScheduler()

    def test_overlap_scheduler_preserves_stride_for_ready_chain_tiles(self):
        """OverlapTileScheduler should preserve broad producer/consumer waves."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(4,))
        builder.add_op(_NOPOp, tile_counts=(4,))

        instructions = builder.build(scheduler=OverlapTileScheduler())[:-1]

        assert [(i.op_idx, i.tiles[0]) for i in instructions] == [
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 0), (1, 1), (1, 2), (1, 3),
        ]

    def test_overlap_scheduler_respects_fetch_stride_waves(self):
        """Stride-aware overlap should not create same-round dependencies."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(8,))
        builder.add_op(_NOPOp, tile_counts=(8,))

        instructions = builder.build(scheduler=OverlapTileScheduler(fetch_stride=4))[:-1]

        assert [(i.op_idx, i.tiles[0]) for i in instructions] == [
            (0, 0), (0, 1), (0, 2), (0, 3),
            (0, 4), (0, 5), (0, 6), (0, 7),
            (1, 0), (1, 1), (1, 2), (1, 3),
            (1, 4), (1, 5), (1, 6), (1, 7),
        ]

    def test_overlap_scheduler_adaptive_fetch_stride_uses_wider_small_waves(self):
        """Adaptive overlap can use two CTA waves for short instruction streams."""
        builder = InstructionStreamBuilder()
        builder.add_op(_NOPOp, tile_counts=(8,))
        builder.add_op(_NOPOp, tile_counts=(8,))
        scheduler = OverlapTileScheduler(adaptive_fetch_stride=True)
        scheduler.bind_num_blocks(4)

        instructions = builder.build(scheduler=scheduler)[:-1]

        assert [(i.op_idx, i.tiles[0]) for i in instructions[:8]] == [
            (0, 0), (0, 1), (0, 2), (0, 3),
            (0, 4), (0, 5), (0, 6), (0, 7),
        ]

    def test_overlap_scheduler_waits_for_many_to_one_dependencies(self):
        """Many-to-one consumers must wait until all producer tiles have signaled."""
        builder = InstructionStreamBuilder()
        builder.add_op(
            _ProducerOp,
            tile_counts=(2, 3),
            dim_names={"batch": 0, "seq": 1},
        )
        builder.add_op(
            _ConsumerOp,
            tile_counts=(2,),
            dim_names={"batch": 0},
        )

        instructions = builder.build(scheduler=OverlapTileScheduler())[:-1]
        positions = {(i.op_idx, i.tiles): pos for pos, i in enumerate(instructions)}

        for batch in range(2):
            consumer_pos = positions[(1, (batch,))]
            producer_positions = [
                positions[(0, (batch, seq))]
                for seq in range(3)
            ]
            assert consumer_pos > max(producer_positions)

    def test_overlap_scheduler_all_tiles_present(self):
        """OverlapTileScheduler must emit every tile exactly once."""
        builder = InstructionStreamBuilder()
        builder.add_op(_ProducerOp, tile_counts=(3, 2), dim_names={"M": 0, "N": 1})
        builder.add_op(_ConsumerOp, tile_counts=(3, 2), dim_names={"M": 0, "N": 1})

        instructions = builder.build(scheduler=OverlapTileScheduler())[:-1]

        assert len(instructions) == 12
        assert {
            (i.op_idx, i.tiles[0], i.tiles[1])
            for i in instructions
        } == {
            (op_idx, m, n)
            for op_idx in (0, 1)
            for m in range(3)
            for n in range(2)
        }

    def test_overlap_ready_does_not_prioritize_independent_sinks(self):
        """Ready-consumer mode should not move unrelated sinks ahead of the chain."""
        builder = InstructionStreamBuilder()
        builder.add_op(_ProducerOp, tile_counts=(4,))
        builder.add_op(_NOPOp, tile_counts=(4,))
        builder.add_op(_ConsumerOp, tile_counts=(4,))

        instructions = builder.build(
            scheduler=OverlapTileScheduler(
                fetch_stride=2,
                prefer_data_movement=False,
                prefer_ready_consumers=True,
            )
        )[:-1]

        assert [(i.op_idx, i.tiles[0]) for i in instructions[:2]] == [
            (0, 0), (0, 1),
        ]


# =============================================================================
# Tensor Metadata & Validation Tests
# =============================================================================


from machete.megakernel.ops import (
    TensorMeta,
    _build_tensor_and_dim_lists,
    ScheduledOp,
)
from machete.megakernel.registries import (
    TensorRegistry,
    validate_op_compatibility,
)


class _Dim2DOp(Op):
    """Test op with 2D tensor declarations."""
    reads = {"x": (None, ("M", "D"))}
    writes = {"y": (None, ("M", "D"))}
    tile = ("M",)


class _Dim3DOp(Op):
    """Test op with 3D tensor declaration."""
    reads = {"q": (None, ("M", "H", "D"))}
    writes = {"q": (None, ("M", "H", "D"))}
    tile = ("M",)


class _Dim1DOp(Op):
    """Test op with 1D tensor declarations."""
    reads = {"w": (None, ("D",))}
    writes = {"w": (None, ("D",))}
    tile = ("D",)


class TestTupleAndStringDimFormats:
    """Test that both tuple and string dim formats work."""

    def test_tuple_format_parsed(self):
        """Tuple dim format produces correct _UNIQUE_TENSORS."""
        assert _Dim2DOp._UNIQUE_TENSORS == [
            ("x", None, ["M", "D"]),
            ("y", None, ["M", "D"]),
        ]

    def test_tuple_format_unique_dims(self):
        """Tuple dim format produces correct _UNIQUE_DIMS."""
        assert _Dim2DOp._UNIQUE_DIMS == [
            ("M", "x", 0),
            ("D", "x", 1),
        ]

    def test_string_format_still_works(self):
        """Legacy string format still parses correctly."""
        tensors, dims = _build_tensor_and_dim_lists(
            {"x": (None, "M, D")}, {"y": (None, "M, D")}
        )
        assert tensors == [("x", None, ["M", "D"]), ("y", None, ["M", "D"])]
        assert dims == [("M", "x", 0), ("D", "x", 1)]


class TestNdimValidation:
    """Test that schedule() validates tensor ndim against declaration."""

    def test_ndim_mismatch_raises(self):
        """Passing 1D tensor to 2D op raises ValueError."""
        x = torch.randn(32, device="cpu")  # 1D
        y = torch.empty(32, device="cpu")  # 1D
        with pytest.raises(ValueError, match="declared as 2D"):
            _Dim2DOp.schedule(x=x, y=y)

    def test_ndim_match_passes(self):
        """Passing 2D tensor to 2D op succeeds."""
        x = torch.randn(4, 8, device="cpu")
        y = torch.empty(4, 8, device="cpu")
        [op] = _Dim2DOp.schedule(x=x, y=y)
        assert op.tensor_metas["x"].ndim == 2
        assert op.tensor_metas["x"].shape == (4, 8)

    def test_3d_ndim_match(self):
        """Passing 3D tensor to 3D op succeeds with correct metadata."""
        q = torch.randn(4, 8, 16, device="cpu")
        [op] = _Dim3DOp.schedule(q=q)
        assert op.tensor_metas["q"].ndim == 3
        assert op.tensor_metas["q"].shape == (4, 8, 16)
        assert op.tensor_metas["q"].strides == tuple(q.stride())

    def test_tensor_strides_captured(self):
        """Strides are captured in ScheduledOp.tensor_strides."""
        x = torch.randn(4, 8, device="cpu")
        y = torch.empty(4, 8, device="cpu")
        [op] = _Dim2DOp.schedule(x=x, y=y)
        assert op.tensor_strides["x"] == (8, 1)
        assert op.tensor_strides["y"] == (8, 1)


class TestDimConsistencyValidation:
    """Test that schedule() validates dim value consistency across tensors."""

    def test_dim_consistency_valid(self):
        """Tensors sharing dim name with same value passes."""
        x = torch.randn(4, 8, device="cpu")
        y = torch.empty(4, 8, device="cpu")
        [op] = _Dim2DOp.schedule(x=x, y=y)
        assert op.static_dims["M"] == 4
        assert op.static_dims["D"] == 8


class TestCrossOpCompatibility:
    """Test cross-op tensor compatibility validation."""

    def test_compatible_reshape(self):
        """Shared tensor with reshape (M,D) -> (M,H,D) passes when total elements match."""
        # Simulate RMSNorm output → RoPE input
        y = torch.randn(32, 256, device="cpu")
        q = y.view(32, 4, 64)  # Same storage, different shape

        [op_a] = _Dim2DOp.schedule(x=y, y=y)
        [op_b] = _Dim3DOp.schedule(q=q)

        registry = TensorRegistry.from_ops([op_a, op_b])
        # Should not raise
        validate_op_compatibility([op_a, op_b], registry)

    def test_reshape_with_different_dim_names_ok(self):
        """Reshape where dim names differ is allowed (only total elements checked)."""
        x = torch.randn(32, 256, device="cpu")
        q = x.view(32, 4, 64)

        [op_a] = _Dim2DOp.schedule(x=x, y=x)
        [op_b] = _Dim3DOp.schedule(q=q)

        # Total elements match (32*256 == 32*4*64), so validation passes
        # even though D=256 in op_a and D=64 in op_b
        assert op_a.static_dims["M"] == 32
        assert op_b.static_dims["M"] == 32
        registry = TensorRegistry.from_ops([op_a, op_b])
        validate_op_compatibility([op_a, op_b], registry)


class TestStrideInitSource:
    """Test that stride constants are included in op config."""

    def test_stride_constants_in_config(self):
        """build_op_config with tensor_strides produces stride constants."""
        from machete.megakernel.ops import build_op_config
        x = torch.randn(32, 64, device="cpu")
        y = torch.empty(32, 64, device="cpu")
        [op] = _Dim2DOp.schedule(x=x, y=y)
        config = build_op_config(op)
        assert config["x_stride_M"] == 64
        assert config["x_stride_D"] == 1
        assert config["y_stride_M"] == 64
        assert config["y_stride_D"] == 1

    def test_3d_stride_constants(self):
        """3D tensor strides are in config correctly."""
        from machete.megakernel.ops import build_op_config
        q = torch.randn(32, 8, 64, device="cpu")
        [op] = _Dim3DOp.schedule(q=q)
        config = build_op_config(op)
        assert config["q_stride_M"] == 512
        assert config["q_stride_H"] == 64
        assert config["q_stride_D"] == 1


class TestTensorMeta:
    """Test TensorMeta creation and properties."""

    def test_tensor_meta_from_schedule(self):
        """TensorMeta is correctly populated from schedule()."""
        x = torch.randn(4, 8, device="cpu")
        y = torch.empty(4, 8, device="cpu")
        [op] = _Dim2DOp.schedule(x=x, y=y)

        meta_x = op.tensor_metas["x"]
        assert meta_x.name == "x"
        assert meta_x.declared_dims == ("M", "D")
        assert meta_x.ndim == 2
        assert meta_x.shape == (4, 8)
        assert meta_x.strides == (8, 1)
        assert meta_x.is_contiguous is True
        assert meta_x.data_ptr == x.data_ptr()

    def test_tensor_meta_1d(self):
        """1D tensor metadata is correct."""
        w = torch.randn(16, device="cpu")
        [op] = _Dim1DOp.schedule(w=w)

        meta_w = op.tensor_metas["w"]
        assert meta_w.ndim == 1
        assert meta_w.shape == (16,)
        assert meta_w.strides == (1,)


# =============================================================================
# Tile Sizes API Tests
# =============================================================================


class TestTileSizesScheduleAPI:
    """Test the tile_sizes parameter of schedule()."""

    def test_full_extent_default(self):
        """Omitting a tile dim from tile_sizes defaults to full extent (1 tile)."""
        x = torch.randn(32, 64, device="cpu")
        y = torch.empty(32, 64, device="cpu")
        # _Dim2DOp has tile = ("M",), so M is the only tile dim.
        # Not passing tile_sizes → M defaults to full extent.
        [op] = _Dim2DOp.schedule(x=x, y=y)
        assert op.tile_counts == (1,)
        assert op.tile_sizes == {"M": 32}

    def test_explicit_tile_sizes(self):
        """Passing tile_sizes computes correct tile_counts."""
        x = torch.randn(32, 64, device="cpu")
        y = torch.empty(32, 64, device="cpu")
        [op] = _Dim2DOp.schedule(x=x, y=y, tile_sizes={"M": 8})
        assert op.tile_counts == (4,)  # 32 / 8 = 4
        assert op.tile_sizes == {"M": 8}

    def test_tile_sizes_ceiling_division(self):
        """tile_counts uses ceiling division when dim not evenly divisible."""
        x = torch.randn(30, 64, device="cpu")
        y = torch.empty(30, 64, device="cpu")
        [op] = _Dim2DOp.schedule(x=x, y=y, tile_sizes={"M": 8})
        assert op.tile_counts == (4,)  # ceil(30 / 8) = 4
        assert op.tile_sizes == {"M": 8}

    def test_tile_size_in_config(self):
        """build_op_config includes tile_size_X constants."""
        from machete.megakernel.ops import build_op_config
        x = torch.randn(32, 64, device="cpu")
        y = torch.empty(32, 64, device="cpu")
        [op] = _Dim2DOp.schedule(x=x, y=y, tile_sizes={"M": 4})
        config = build_op_config(op)
        assert config["tile_size_M"] == 4

    def test_full_extent_tile_size_in_config(self):
        """Full-extent tile size appears in config as the dim's full value."""
        from machete.megakernel.ops import build_op_config
        x = torch.randn(32, 64, device="cpu")
        y = torch.empty(32, 64, device="cpu")
        [op] = _Dim2DOp.schedule(x=x, y=y)  # No tile_sizes → full extent
        config = build_op_config(op)
        assert config["tile_size_M"] == 32


class TestParseTileSpecErrors:
    """Test that old-style tile spec format gives a clear error."""

    def test_tuple_pair_raises_with_helpful_message(self):
        """Passing (("M", 4),) as tile spec raises ValueError with tile_sizes hint."""
        with pytest.raises(ValueError, match="tile_sizes"):
            from machete.megakernel.ops import _parse_tile_spec
            _parse_tile_spec((("M", 4),))

    def test_int_item_raises(self):
        """Passing integer items in tile spec raises ValueError."""
        with pytest.raises(ValueError, match="tile_sizes"):
            from machete.megakernel.ops import _parse_tile_spec
            _parse_tile_spec((4,))

    def test_valid_string_tuple(self):
        """Valid string tuple parses correctly."""
        from machete.megakernel.ops import _parse_tile_spec
        assert _parse_tile_spec(("M", "D")) == ("M", "D")

    def test_single_string(self):
        """Single string is wrapped into a tuple."""
        from machete.megakernel.ops import _parse_tile_spec
        assert _parse_tile_spec("M") == ("M",)
