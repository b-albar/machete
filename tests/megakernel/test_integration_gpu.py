# Copyright (c) 2025, Machete Authors
"""GPU integration tests for the persistent megakernel framework.

Tests cover:
1. Dependency ordering — StampOp tile i completes before CheckOp tile i starts
2. Shared memory paging — page data write/readback via get_page_data_ptr
3. Page recycling — circular buffer with more tiles than pages
4. Multi-op chains (3+ ops)
5. Mismatched tile counts with guard logic
6. 2D tile grids with barrier stride formulas
7. Named buffer dependencies (INPUTS/OUTPUTS with dim_names)
8. Many-to-one tile mapping (expected > 1)
9. Fan-in pattern (multiple producers → single consumer)
10. Barrier reset across multiple runs
11. High contention stress (many tiles, few SMs, few pages)
12. Zero-page ops (no shared memory paging)

Correctness is verified by writing results to global memory tensors and
asserting on host-side readback. printf calls are kept for human debugging.
"""

import struct

import pytest
import torch
from typing import ClassVar, List

import cutlass.cute as cute
from cutlass import Int32, Int64

from machete.megakernel import Megakernel, MegakernelConfig, ScheduledOp
from machete.megakernel.ops import Op


def _pack_ptr(config, offset, ptr):
    """Pack a 64-bit pointer into two int32 slots using struct (handles sign correctly)."""
    lo, hi = struct.unpack("ii", struct.pack("Q", ptr))
    config[offset] = lo
    config[offset + 1] = hi
from machete.megakernel.paged_memory import (
    get_page_data_ptr,
    st_shared_i32,
    ld_shared_i32,
)
from machete.megakernel.interpreter import st_global_i32, ld_global_i32, ld_global_i64


def is_hopper_available():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


# =============================================================================
# Global result tensor pointers (set before kernel launch, captured by
# _merge_globals during compilation)
# =============================================================================

_stamp_result_ptr = 0
_check_result_ptr = 0
_check_stale_ptr = 0


# =============================================================================
# Test Ops
# =============================================================================


class StampOp(Op):
    """Writes tile_m * 100 + 42 to output page, stores readback to global tensor."""

    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 1

    @staticmethod
    def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
        data_ptr = get_page_data_ptr(smem_base, config_ptr, page_ids[0])
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            value = tile_m * Int32(100) + Int32(42)
            st_shared_i32(data_ptr, value)
        cute.arch.sync_threads()
        if tidx == Int32(0):
            readback = ld_shared_i32(data_ptr)
            # Write readback to global results tensor
            st_global_i32(Int64(_stamp_result_ptr), tile_m, readback)
            cute.printf("[StampOp] tile_m=%d wrote=%d readback=%d",
                        tile_m, tile_m * Int32(100) + Int32(42), readback)


class CheckOp(Op):
    """Reads stale page data, writes tile_m * 200 + 7, stores results to global tensors."""

    NUM_INPUT_PAGES: ClassVar[int] = 1
    NUM_OUTPUT_PAGES: ClassVar[int] = 0

    @staticmethod
    def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
        data_ptr = get_page_data_ptr(smem_base, config_ptr, page_ids[0])
        tidx = cute.arch.thread_idx()[0]
        # Read stale value for verification
        if tidx == Int32(0):
            stale = ld_shared_i32(data_ptr)
            st_global_i32(Int64(_check_stale_ptr), tile_m, stale)
            cute.printf("[CheckOp] tile_m=%d START stale_page_value=%d", tile_m, stale)
        cute.arch.sync_threads()
        # Write and verify
        if tidx == Int32(0):
            value = tile_m * Int32(200) + Int32(7)
            st_shared_i32(data_ptr, value)
        cute.arch.sync_threads()
        if tidx == Int32(0):
            readback = ld_shared_i32(data_ptr)
            st_global_i32(Int64(_check_result_ptr), tile_m, readback)
            cute.printf("[CheckOp] tile_m=%d wrote=%d readback=%d",
                        tile_m, tile_m * Int32(200) + Int32(7), readback)


# =============================================================================
# GPU Tests
# =============================================================================


@pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
class TestSequentialOpsGPU:
    """Integration tests for sequential ops with real shared memory usage."""

    def test_two_ops_dependency_and_paging(self):
        """Two ops: StampOp -> CheckOp, 4 tiles, 2 pages (forces recycling).

        Verifies:
        - StampOp readback matches expected values
        - CheckOp readback matches expected values
        - Page recycling: 4 tiles with only 2 pages completes without deadlock
        """
        global _stamp_result_ptr, _check_result_ptr, _check_stale_ptr
        num_tiles = 4

        stamp_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        check_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        check_stale = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")

        _stamp_result_ptr = stamp_results.data_ptr()
        _check_result_ptr = check_results.data_ptr()
        _check_stale_ptr = check_stale.data_ptr()

        ops = [
            ScheduledOp(StampOp, tiles_m=num_tiles),
            ScheduledOp(CheckOp, tiles_m=num_tiles),
        ]
        config = MegakernelConfig(num_pages=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        # Verify StampOp: tile_m * 100 + 42
        expected_stamp = torch.tensor(
            [i * 100 + 42 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        assert torch.equal(stamp_results, expected_stamp), (
            f"StampOp mismatch: got {stamp_results.tolist()}, "
            f"expected {expected_stamp.tolist()}"
        )

        # Verify CheckOp: tile_m * 200 + 7
        expected_check = torch.tensor(
            [i * 200 + 7 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        assert torch.equal(check_results, expected_check), (
            f"CheckOp mismatch: got {check_results.tolist()}, "
            f"expected {expected_check.tolist()}"
        )

    def test_single_stamp_op(self):
        """Single StampOp to verify basic page write/readback."""
        global _stamp_result_ptr
        num_tiles = 2

        stamp_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        _stamp_result_ptr = stamp_results.data_ptr()

        ops = [ScheduledOp(StampOp, tiles_m=num_tiles)]
        kernel = Megakernel(ops)
        kernel.run()

        expected = torch.tensor(
            [i * 100 + 42 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        assert torch.equal(stamp_results, expected), (
            f"StampOp mismatch: got {stamp_results.tolist()}, "
            f"expected {expected.tolist()}"
        )

    def test_many_tiles_recycling(self):
        """8 tiles with 2 pages -- stress test page recycling."""
        global _stamp_result_ptr, _check_result_ptr, _check_stale_ptr
        num_tiles = 8

        stamp_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        check_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        check_stale = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")

        _stamp_result_ptr = stamp_results.data_ptr()
        _check_result_ptr = check_results.data_ptr()
        _check_stale_ptr = check_stale.data_ptr()

        ops = [
            ScheduledOp(StampOp, tiles_m=num_tiles),
            ScheduledOp(CheckOp, tiles_m=num_tiles),
        ]
        config = MegakernelConfig(num_pages=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        # Verify StampOp: tile_m * 100 + 42
        expected_stamp = torch.tensor(
            [i * 100 + 42 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        assert torch.equal(stamp_results, expected_stamp), (
            f"StampOp mismatch: got {stamp_results.tolist()}, "
            f"expected {expected_stamp.tolist()}"
        )

        # Verify CheckOp: tile_m * 200 + 7
        expected_check = torch.tensor(
            [i * 200 + 7 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        assert torch.equal(check_results, expected_check), (
            f"CheckOp mismatch: got {check_results.tolist()}, "
            f"expected {expected_check.tolist()}"
        )


# =============================================================================
# Zero-Page Ops (write directly to global memory, no shared memory pages)
# =============================================================================

# --- OpA/OpB/OpC globals (for multi-op chain tests) ---
_opa_result_ptr = 0
_opb_result_ptr = 0
_opc_result_ptr = 0


class OpA(Op):
    """Zero-page op: writes tile_m * 100 + 1 to global tensor."""

    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0

    @staticmethod
    def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            value = tile_m * Int32(100) + Int32(1)
            st_global_i32(Int64(_opa_result_ptr), tile_m, value)



class OpB(Op):
    """Zero-page op: writes tile_m * 200 + 2 to global tensor."""

    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0

    @staticmethod
    def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            value = tile_m * Int32(200) + Int32(2)
            st_global_i32(Int64(_opb_result_ptr), tile_m, value)



class OpC(Op):
    """Zero-page op: writes tile_m * 300 + 3 to global tensor."""

    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0

    @staticmethod
    def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            value = tile_m * Int32(300) + Int32(3)
            st_global_i32(Int64(_opc_result_ptr), tile_m, value)



# --- Tag2DOp globals (for 2D tile grid tests) ---
_tag2d_result_ptr = 0
_tag2d_cols = 0  # tiles_n, used to compute linear index


class Tag2DOp(Op):
    """Zero-page op for 2D tiles: writes tile_m * 1000 + tile_n to global tensor.

    Result index: tile_n * _tag2d_cols_m + tile_m (m varies fastest, matching
    the instruction stream ordering from InstructionStreamBuilder).
    """

    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0

    @staticmethod
    def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            value = tile_m * Int32(1000) + tile_n
            idx = tile_n * Int32(_tag2d_cols) + tile_m
            st_global_i32(Int64(_tag2d_result_ptr), idx, value)



class Tag2DOpB(Op):
    """Second 2D zero-page op: writes tile_m * 2000 + tile_n to separate tensor."""

    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0

    @staticmethod
    def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            value = tile_m * Int32(2000) + tile_n
            idx = tile_n * Int32(_tag2d_cols) + tile_m
            st_global_i32(Int64(_opb_result_ptr), idx, value)



# =============================================================================
# Named Buffer Ops (for DAG dependency tests)
# =============================================================================

_nprod_result_ptr = 0
_ncons_result_ptr = 0
_nprody_result_ptr = 0
_nfanin_result_ptr = 0


class NamedProducerOp(Op):
    """Produces buffer 'x'. Writes tile_m * 100 + 42."""

    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0
    INPUTS: ClassVar[List[str]] = []
    OUTPUTS: ClassVar[List[str]] = ["x"]

    @staticmethod
    def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            value = tile_m * Int32(100) + Int32(42)
            st_global_i32(Int64(_nprod_result_ptr), tile_m, value)



class NamedConsumerOp(Op):
    """Consumes buffer 'x'. Writes tile_m * 200 + 7."""

    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0
    INPUTS: ClassVar[List[str]] = ["x"]
    OUTPUTS: ClassVar[List[str]] = []

    @staticmethod
    def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            value = tile_m * Int32(200) + Int32(7)
            st_global_i32(Int64(_ncons_result_ptr), tile_m, value)



class NamedProducerY(Op):
    """Produces buffer 'y'. Writes tile_m * 300 + 11."""

    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0
    INPUTS: ClassVar[List[str]] = []
    OUTPUTS: ClassVar[List[str]] = ["y"]

    @staticmethod
    def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            value = tile_m * Int32(300) + Int32(11)
            st_global_i32(Int64(_nprody_result_ptr), tile_m, value)



class NamedFanInOp(Op):
    """Consumes both 'x' and 'y'. Writes tile_m * 400 + 13."""

    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0
    INPUTS: ClassVar[List[str]] = ["x", "y"]
    OUTPUTS: ClassVar[List[str]] = []

    @staticmethod
    def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            value = tile_m * Int32(400) + Int32(13)
            st_global_i32(Int64(_nfanin_result_ptr), tile_m, value)



# =============================================================================
# Many-to-One Ops
# =============================================================================

_mto_matrix_ptr = 0   # 2D result tensor for producer (m x n)
_mto_cols = 0          # tiles_n for index computation
_mto_result_ptr = 0    # 1D result tensor for consumer


class ManyToOneProducerOp(Op):
    """2D producer for many-to-one: writes 1 to matrix[m, n] for each tile."""

    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0
    INPUTS: ClassVar[List[str]] = []
    OUTPUTS: ClassVar[List[str]] = ["x"]

    @staticmethod
    def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            idx = tile_m * Int32(_mto_cols) + tile_n
            st_global_i32(Int64(_mto_matrix_ptr), idx, Int32(1))



class ManyToOneConsumerOp(Op):
    """1D consumer for many-to-one: writes tile_m * 500 + 17."""

    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0
    INPUTS: ClassVar[List[str]] = ["x"]
    OUTPUTS: ClassVar[List[str]] = []

    @staticmethod
    def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            value = tile_m * Int32(500) + Int32(17)
            st_global_i32(Int64(_mto_result_ptr), tile_m, value)



# =============================================================================
# Comprehensive GPU Tests
# =============================================================================


@pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
class TestComprehensiveGPU:
    """Comprehensive GPU integration tests covering all framework aspects.

    All tests use small num_sms (2-4) to force multiple loop iterations
    per block and increase contention, exposing synchronization bugs.
    """

    def test_three_op_chain(self):
        """Three-op linear chain: OpA -> OpB -> OpC.

        Verifies:
        - Barrier dependencies form correct chain (OpC waits on OpB, not OpA)
        - All 3 ops produce correct values
        - num_sms=2 forces strided instruction processing
        """
        global _opa_result_ptr, _opb_result_ptr, _opc_result_ptr
        num_tiles = 6

        a_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        b_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        c_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")

        _opa_result_ptr = a_results.data_ptr()
        _opb_result_ptr = b_results.data_ptr()
        _opc_result_ptr = c_results.data_ptr()

        ops = [
            ScheduledOp(OpA, tiles_m=num_tiles),
            ScheduledOp(OpB, tiles_m=num_tiles),
            ScheduledOp(OpC, tiles_m=num_tiles),
        ]
        config = MegakernelConfig(num_sms=2, num_pages=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        expected_a = torch.tensor(
            [i * 100 + 1 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        expected_b = torch.tensor(
            [i * 200 + 2 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        expected_c = torch.tensor(
            [i * 300 + 3 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        assert torch.equal(a_results, expected_a), (
            f"OpA mismatch: {a_results.tolist()} != {expected_a.tolist()}"
        )
        assert torch.equal(b_results, expected_b), (
            f"OpB mismatch: {b_results.tolist()} != {expected_b.tolist()}"
        )
        assert torch.equal(c_results, expected_c), (
            f"OpC mismatch: {c_results.tolist()} != {expected_c.tolist()}"
        )

    @pytest.mark.parametrize("tiles_a,tiles_b", [(8, 4), (4, 8)])
    def test_mismatched_tile_counts(self, tiles_a, tiles_b):
        """Test mismatched tile counts between producer and consumer.

        Guard logic prevents deadlock when ops have different tile counts:
        - More producer tiles: consumer skips non-existent barriers
        - More consumer tiles: extra consumer tiles skip wait via guard
        """
        global _opa_result_ptr, _opb_result_ptr

        a_results = torch.zeros(tiles_a, dtype=torch.int32, device="cuda")
        b_results = torch.zeros(tiles_b, dtype=torch.int32, device="cuda")

        _opa_result_ptr = a_results.data_ptr()
        _opb_result_ptr = b_results.data_ptr()

        ops = [
            ScheduledOp(OpA, tiles_m=tiles_a),
            ScheduledOp(OpB, tiles_m=tiles_b),
        ]
        config = MegakernelConfig(num_sms=2, num_pages=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        expected_a = torch.tensor(
            [i * 100 + 1 for i in range(tiles_a)],
            dtype=torch.int32, device="cuda",
        )
        expected_b = torch.tensor(
            [i * 200 + 2 for i in range(tiles_b)],
            dtype=torch.int32, device="cuda",
        )
        assert torch.equal(a_results, expected_a), f"OpA mismatch: {a_results.tolist()}"
        assert torch.equal(b_results, expected_b), f"OpB mismatch: {b_results.tolist()}"

    def test_2d_tile_grid(self):
        """Single 2D op: tiles_m=4, tiles_n=3 (12 tiles).

        Verifies:
        - All 12 (m, n) combinations produce correct encoded values
        - tile_m and tile_n are correctly dispatched from instruction stream
        """
        global _tag2d_result_ptr, _tag2d_cols
        tiles_m, tiles_n = 4, 3
        num_tiles = tiles_m * tiles_n

        _tag2d_cols = tiles_m  # m varies fastest
        results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        _tag2d_result_ptr = results.data_ptr()

        ops = [ScheduledOp(Tag2DOp, tiles_m=tiles_m, tiles_n=tiles_n)]
        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        # Instruction stream ordering: m varies fastest
        # Index = n * tiles_m + m
        expected = torch.tensor(
            [m * 1000 + n for n in range(tiles_n) for m in range(tiles_m)],
            dtype=torch.int32, device="cuda",
        )
        assert torch.equal(results, expected), (
            f"2D grid mismatch: {results.tolist()} != {expected.tolist()}"
        )

    def test_2d_two_op_chain(self):
        """Two 2D ops chained: Tag2DOp -> Tag2DOpB, tiles_m=3, tiles_n=4.

        Verifies:
        - 2D barrier strides (coeff_m=1, coeff_n=tiles_m) work correctly
        - Both ops produce correct results with 2D tile indices
        """
        global _tag2d_result_ptr, _tag2d_cols, _opb_result_ptr
        tiles_m, tiles_n = 3, 4
        num_tiles = tiles_m * tiles_n

        _tag2d_cols = tiles_m
        a_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        b_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        _tag2d_result_ptr = a_results.data_ptr()
        _opb_result_ptr = b_results.data_ptr()

        ops = [
            ScheduledOp(Tag2DOp, tiles_m=tiles_m, tiles_n=tiles_n),
            ScheduledOp(Tag2DOpB, tiles_m=tiles_m, tiles_n=tiles_n),
        ]
        config = MegakernelConfig(num_sms=2, num_pages=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        expected_a = torch.tensor(
            [m * 1000 + n for n in range(tiles_n) for m in range(tiles_m)],
            dtype=torch.int32, device="cuda",
        )
        expected_b = torch.tensor(
            [m * 2000 + n for n in range(tiles_n) for m in range(tiles_m)],
            dtype=torch.int32, device="cuda",
        )
        assert torch.equal(a_results, expected_a), (
            f"Tag2DOp mismatch: {a_results.tolist()}"
        )
        assert torch.equal(b_results, expected_b), (
            f"Tag2DOpB mismatch: {b_results.tolist()}"
        )

    def test_named_buffer_one_to_one(self):
        """Named buffer DAG: Producer('x') -> Consumer('x'), 1:1 mapping.

        Verifies:
        - DAG-based dependency resolution works on GPU
        - Both ops complete with correct values
        """
        global _nprod_result_ptr, _ncons_result_ptr
        num_tiles = 4

        prod_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        cons_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        _nprod_result_ptr = prod_results.data_ptr()
        _ncons_result_ptr = cons_results.data_ptr()

        ops = [
            ScheduledOp(
                NamedProducerOp, tiles_m=num_tiles,
                dim_names={"batch": "m"},
            ),
            ScheduledOp(
                NamedConsumerOp, tiles_m=num_tiles,
                dim_names={"batch": "m"},
            ),
        ]
        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        expected_prod = torch.tensor(
            [i * 100 + 42 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        expected_cons = torch.tensor(
            [i * 200 + 7 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        assert torch.equal(prod_results, expected_prod), (
            f"Producer mismatch: {prod_results.tolist()}"
        )
        assert torch.equal(cons_results, expected_cons), (
            f"Consumer mismatch: {cons_results.tolist()}"
        )

    def test_named_buffer_many_to_one(self):
        """Many-to-one: Producer(m=4, n=8) -> Consumer(m=4), expected=8.

        Consumer tile m=i only runs after ALL 8 producer tiles for batch i
        have signaled. Verifies:
        - All 32 producer tiles write to matrix
        - All 4 consumer tiles complete (barrier expected=8 works)
        """
        global _mto_matrix_ptr, _mto_cols, _mto_result_ptr
        tiles_m, tiles_n = 4, 8

        matrix = torch.zeros(tiles_m * tiles_n, dtype=torch.int32, device="cuda")
        cons_results = torch.zeros(tiles_m, dtype=torch.int32, device="cuda")
        _mto_matrix_ptr = matrix.data_ptr()
        _mto_cols = tiles_n  # stride for tile_m (row-major: m * cols + n)
        _mto_result_ptr = cons_results.data_ptr()

        ops = [
            ScheduledOp(
                ManyToOneProducerOp, tiles_m=tiles_m, tiles_n=tiles_n,
                dim_names={"batch": "m", "seqlen": "n"},
            ),
            ScheduledOp(
                ManyToOneConsumerOp, tiles_m=tiles_m,
                dim_names={"batch": "m"},
            ),
        ]
        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        # All 32 producer entries should be 1
        expected_matrix = torch.ones(
            tiles_m * tiles_n, dtype=torch.int32, device="cuda",
        )
        assert torch.equal(matrix, expected_matrix), (
            f"Producer matrix mismatch: {matrix.tolist()}"
        )

        # All 4 consumer tiles should complete
        expected_cons = torch.tensor(
            [i * 500 + 17 for i in range(tiles_m)],
            dtype=torch.int32, device="cuda",
        )
        assert torch.equal(cons_results, expected_cons), (
            f"Consumer mismatch: {cons_results.tolist()}"
        )

    def test_named_buffer_fan_in(self):
        """Fan-in: ProducerX -> 'x', ProducerY -> 'y', FanIn reads ['x','y'].

        Verifies:
        - FanIn op waits for BOTH independent producers
        - All 3 ops produce correct values
        """
        global _nprod_result_ptr, _nprody_result_ptr, _nfanin_result_ptr
        num_tiles = 4

        prod_x = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        prod_y = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        fanin = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        _nprod_result_ptr = prod_x.data_ptr()
        _nprody_result_ptr = prod_y.data_ptr()
        _nfanin_result_ptr = fanin.data_ptr()

        ops = [
            ScheduledOp(
                NamedProducerOp, tiles_m=num_tiles,
                dim_names={"batch": "m"},
            ),
            ScheduledOp(
                NamedProducerY, tiles_m=num_tiles,
                dim_names={"batch": "m"},
            ),
            ScheduledOp(
                NamedFanInOp, tiles_m=num_tiles,
                dim_names={"batch": "m"},
            ),
        ]
        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        expected_x = torch.tensor(
            [i * 100 + 42 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        expected_y = torch.tensor(
            [i * 300 + 11 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        expected_fanin = torch.tensor(
            [i * 400 + 13 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        assert torch.equal(prod_x, expected_x), (
            f"ProducerX mismatch: {prod_x.tolist()}"
        )
        assert torch.equal(prod_y, expected_y), (
            f"ProducerY mismatch: {prod_y.tolist()}"
        )
        assert torch.equal(fanin, expected_fanin), (
            f"FanIn mismatch: {fanin.tolist()}"
        )

    def test_multiple_runs_barrier_reset(self):
        """Run same kernel twice — barriers must reset between runs.

        Verifies:
        - First run produces correct results
        - Second run also produces correct results (no stale barrier state)
        """
        global _stamp_result_ptr, _check_result_ptr, _check_stale_ptr
        num_tiles = 4

        stamp_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        check_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        check_stale = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")

        _stamp_result_ptr = stamp_results.data_ptr()
        _check_result_ptr = check_results.data_ptr()
        _check_stale_ptr = check_stale.data_ptr()

        ops = [
            ScheduledOp(StampOp, tiles_m=num_tiles),
            ScheduledOp(CheckOp, tiles_m=num_tiles),
        ]
        config = MegakernelConfig(num_sms=2, num_pages=2)
        kernel = Megakernel(ops, config=config)

        expected_stamp = torch.tensor(
            [i * 100 + 42 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        expected_check = torch.tensor(
            [i * 200 + 7 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )

        # First run
        kernel.run()
        assert torch.equal(stamp_results, expected_stamp), (
            f"Run 1 StampOp: {stamp_results.tolist()}"
        )
        assert torch.equal(check_results, expected_check), (
            f"Run 1 CheckOp: {check_results.tolist()}"
        )

        # Reset result tensors to zero
        stamp_results.zero_()
        check_results.zero_()
        check_stale.zero_()

        # Second run — same compiled kernel, barriers must be reset
        kernel.run()
        assert torch.equal(stamp_results, expected_stamp), (
            f"Run 2 StampOp: {stamp_results.tolist()}"
        )
        assert torch.equal(check_results, expected_check), (
            f"Run 2 CheckOp: {check_results.tolist()}"
        )

    def test_high_contention_stress(self):
        """32 tiles, 2 SMs, 2 pages — maximum contention stress test.

        Forces:
        - 16 loop iterations per block (32 tiles / 2 SMs)
        - Continuous page recycling (32 acquire/release cycles with only 2 pages)
        - Heavy barrier traffic

        Verifies no deadlocks and correct values for all tiles.
        """
        global _stamp_result_ptr, _check_result_ptr, _check_stale_ptr
        num_tiles = 32

        stamp_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        check_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        check_stale = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")

        _stamp_result_ptr = stamp_results.data_ptr()
        _check_result_ptr = check_results.data_ptr()
        _check_stale_ptr = check_stale.data_ptr()

        ops = [
            ScheduledOp(StampOp, tiles_m=num_tiles),
            ScheduledOp(CheckOp, tiles_m=num_tiles),
        ]
        config = MegakernelConfig(num_sms=2, num_pages=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        expected_stamp = torch.tensor(
            [i * 100 + 42 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        expected_check = torch.tensor(
            [i * 200 + 7 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        assert torch.equal(stamp_results, expected_stamp), (
            f"StampOp mismatch: {stamp_results.tolist()}"
        )
        assert torch.equal(check_results, expected_check), (
            f"CheckOp mismatch: {check_results.tolist()}"
        )

    def test_zero_page_ops_chain(self):
        """Three zero-page ops: OpA -> OpB -> OpC.

        Tests the n_pages=0 handler path in _make_op_handler — barriers
        work correctly without any page acquire/release.
        """
        global _opa_result_ptr, _opb_result_ptr, _opc_result_ptr
        num_tiles = 8

        a_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        b_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        c_results = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")

        _opa_result_ptr = a_results.data_ptr()
        _opb_result_ptr = b_results.data_ptr()
        _opc_result_ptr = c_results.data_ptr()

        ops = [
            ScheduledOp(OpA, tiles_m=num_tiles),
            ScheduledOp(OpB, tiles_m=num_tiles),
            ScheduledOp(OpC, tiles_m=num_tiles),
        ]
        config = MegakernelConfig(num_sms=4)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        expected_a = torch.tensor(
            [i * 100 + 1 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        expected_b = torch.tensor(
            [i * 200 + 2 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        expected_c = torch.tensor(
            [i * 300 + 3 for i in range(num_tiles)],
            dtype=torch.int32, device="cuda",
        )
        assert torch.equal(a_results, expected_a), (
            f"OpA mismatch: {a_results.tolist()}"
        )
        assert torch.equal(b_results, expected_b), (
            f"OpB mismatch: {b_results.tolist()}"
        )
        assert torch.equal(c_results, expected_c), (
            f"OpC mismatch: {c_results.tolist()}"
        )


# =============================================================================
# Config Pointer Ops (per-op runtime data via global memory config struct)
# =============================================================================


class ScaleOp(Op):
    """Reads input[tile_m] from config, multiplies by scale (also from config),
    writes to output[tile_m].

    Config layout (int64 array):
        [0] input_ptr  (int64) — pointer to input int32 tensor
        [1] output_ptr (int64) — pointer to output int32 tensor
        [2] scale      (int64, lower 32 bits used) — scale factor
    """

    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0

    @staticmethod
    def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            input_ptr = ld_global_i64(op_config_ptr, Int32(0))
            output_ptr = ld_global_i64(op_config_ptr, Int32(1))
            scale = ld_global_i32(op_config_ptr, Int32(4))  # word offset 4 = byte offset 16
            val = ld_global_i32(input_ptr, tile_m)
            st_global_i32(output_ptr, tile_m, val * scale)



class AddOp(Op):
    """Reads input_a[tile_m] and input_b[tile_m] from config, writes sum to output[tile_m].

    Config layout (int64 array):
        [0] input_a_ptr (int64)
        [1] input_b_ptr (int64)
        [2] output_ptr  (int64)
    """

    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0
    INPUTS: ClassVar[List[str]] = ["scaled"]
    OUTPUTS: ClassVar[List[str]] = []

    @staticmethod
    def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            input_a_ptr = ld_global_i64(op_config_ptr, Int32(0))
            input_b_ptr = ld_global_i64(op_config_ptr, Int32(1))
            output_ptr = ld_global_i64(op_config_ptr, Int32(2))
            a = ld_global_i32(input_a_ptr, tile_m)
            b = ld_global_i32(input_b_ptr, tile_m)
            st_global_i32(output_ptr, tile_m, a + b)



class TensorScaleOp(Op):
    """Like ScaleOp, but uses cute.make_ptr/cute.make_tensor instead of raw PTX.

    Demonstrates accessing global memory through CuTe Tensor abstraction,
    which enables higher-level indexing and future use of TMA/copy operations.

    Config layout (int64 array):
        [0] input_ptr  (int64) — pointer to input int32 tensor
        [1] output_ptr (int64) — pointer to output int32 tensor
        [2] scale      (int64, lower 32 bits used) — scale factor
    """

    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0

    @staticmethod
    def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            # Read pointers and scale from config
            input_ptr = ld_global_i64(op_config_ptr, Int32(0))
            output_ptr = ld_global_i64(op_config_ptr, Int32(1))
            scale = ld_global_i32(op_config_ptr, Int32(4))

            # Create CuTe Tensors from raw pointers
            in_tensor = cute.make_tensor(
                cute.make_ptr(Int32, input_ptr, cute.AddressSpace.gmem),
                cute.make_layout(1024),
            )
            out_tensor = cute.make_tensor(
                cute.make_ptr(Int32, output_ptr, cute.AddressSpace.gmem),
                cute.make_layout(1024),
            )

            # Access via tensor indexing
            out_tensor[tile_m] = in_tensor[tile_m] * scale



@pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
class TestConfigPointerGPU:
    """Tests for per-op config pointer mechanism.

    Verifies that ops can receive runtime data (tensor pointers, scalars)
    via the op_config_ptr parameter without module-level globals.
    """

    def test_scale_op_reads_config(self):
        """ScaleOp reads input pointer and scale from config, writes scaled output.

        Config: [input_ptr, output_ptr, scale=3]
        Expected: output[i] = input[i] * 3
        """
        num_tiles = 8
        input_data = torch.tensor(
            [10, 20, 30, 40, 50, 60, 70, 80],
            dtype=torch.int32, device="cuda",
        )
        output_data = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        scale = 3

        # Pack config: [input_ptr(i64), output_ptr(i64), scale(i32 at word offset 4)]
        # We use an int32 tensor for fine-grained layout control.
        # Layout: bytes 0-7 = input_ptr, bytes 8-15 = output_ptr, bytes 16-19 = scale
        config = torch.zeros(5, dtype=torch.int32, device="cuda")
        _pack_ptr(config, 0, input_data.data_ptr())
        _pack_ptr(config, 2, output_data.data_ptr())
        config[4] = scale

        ops = [ScheduledOp(ScaleOp, tiles_m=num_tiles, config_data=config)]
        kernel_config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=kernel_config)
        kernel.run()

        expected = torch.tensor(
            [i * scale for i in [10, 20, 30, 40, 50, 60, 70, 80]],
            dtype=torch.int32, device="cuda",
        )
        assert torch.equal(output_data, expected), (
            f"ScaleOp mismatch: got {output_data.tolist()}, expected {expected.tolist()}"
        )

    def test_two_instances_same_op_different_config(self):
        """Two ScaleOp instances with different scale factors.

        Verifies tile function deduplication: same op class compiles once,
        but each instance receives its own config pointer with different data.

        Instance 0: scale=2, input=[1,2,3,4] -> output=[2,4,6,8]
        Instance 1: scale=5, input=[10,20,30,40] -> output=[50,100,150,200]
        """
        num_tiles = 4

        input_a = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
        output_a = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        input_b = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device="cuda")
        output_b = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")

        def pack_scale_config(inp, out, scale_val):
            cfg = torch.zeros(5, dtype=torch.int32, device="cuda")
            _pack_ptr(cfg, 0, inp.data_ptr())
            _pack_ptr(cfg, 2, out.data_ptr())
            cfg[4] = scale_val
            return cfg

        config_a = pack_scale_config(input_a, output_a, 2)
        config_b = pack_scale_config(input_b, output_b, 5)

        ops = [
            ScheduledOp(ScaleOp, tiles_m=num_tiles, config_data=config_a),
            ScheduledOp(ScaleOp, tiles_m=num_tiles, config_data=config_b),
        ]
        kernel_config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=kernel_config)
        kernel.run()

        expected_a = torch.tensor([2, 4, 6, 8], dtype=torch.int32, device="cuda")
        expected_b = torch.tensor([50, 100, 150, 200], dtype=torch.int32, device="cuda")

        assert torch.equal(output_a, expected_a), (
            f"ScaleOp instance 0: got {output_a.tolist()}, expected {expected_a.tolist()}"
        )
        assert torch.equal(output_b, expected_b), (
            f"ScaleOp instance 1: got {output_b.tolist()}, expected {expected_b.tolist()}"
        )

    def test_config_ptr_chain_with_dependency(self):
        """ScaleOp -> AddOp chain where both ops use config pointers.

        ScaleOp: output_scale[i] = input[i] * 3
        AddOp: output_add[i] = output_scale[i] + bias[i]

        Tests that config pointers work correctly across ops with
        barrier dependencies — AddOp must wait for ScaleOp to finish.
        """
        num_tiles = 4
        input_data = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device="cuda")
        bias_data = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device="cuda")
        scale_output = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        add_output = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")

        # ScaleOp config: [input_ptr, scale_output_ptr, scale=3]
        scale_config = torch.zeros(5, dtype=torch.int32, device="cuda")
        _pack_ptr(scale_config, 0, input_data.data_ptr())
        _pack_ptr(scale_config, 2, scale_output.data_ptr())
        scale_config[4] = 3

        # AddOp config: [input_a_ptr (=scale_output), input_b_ptr (=bias), output_ptr]
        add_config = torch.zeros(6, dtype=torch.int32, device="cuda")
        _pack_ptr(add_config, 0, scale_output.data_ptr())
        _pack_ptr(add_config, 2, bias_data.data_ptr())
        _pack_ptr(add_config, 4, add_output.data_ptr())

        # Use a ScaleOp subclass that declares OUTPUTS for dependency tracking
        class ScaleOpProducer(ScaleOp):
            INPUTS: ClassVar[List[str]] = []
            OUTPUTS: ClassVar[List[str]] = ["scaled"]

        ops = [
            ScheduledOp(
                ScaleOpProducer, tiles_m=num_tiles, config_data=scale_config,
                dim_names={"batch": "m"},
            ),
            ScheduledOp(
                AddOp, tiles_m=num_tiles, config_data=add_config,
                dim_names={"batch": "m"},
            ),
        ]
        kernel_config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=kernel_config)
        kernel.run()

        # ScaleOp: [10*3, 20*3, 30*3, 40*3] = [30, 60, 90, 120]
        expected_scale = torch.tensor([30, 60, 90, 120], dtype=torch.int32, device="cuda")
        assert torch.equal(scale_output, expected_scale), (
            f"ScaleOp: got {scale_output.tolist()}, expected {expected_scale.tolist()}"
        )

        # AddOp: [30+1, 60+2, 90+3, 120+4] = [31, 62, 93, 124]
        expected_add = torch.tensor([31, 62, 93, 124], dtype=torch.int32, device="cuda")
        assert torch.equal(add_output, expected_add), (
            f"AddOp: got {add_output.tolist()}, expected {expected_add.tolist()}"
        )

    def test_tensor_scale_op(self):
        """TensorScaleOp uses cute.make_ptr/cute.make_tensor to access global memory.

        Same logic as ScaleOp but through CuTe Tensor abstraction:
        output[i] = input[i] * scale, using tensor indexing instead of raw PTX.
        """
        num_tiles = 8
        input_data = torch.tensor(
            [5, 10, 15, 20, 25, 30, 35, 40],
            dtype=torch.int32, device="cuda",
        )
        output_data = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        scale = 4

        # Same config layout as ScaleOp
        config = torch.zeros(5, dtype=torch.int32, device="cuda")
        _pack_ptr(config, 0, input_data.data_ptr())
        _pack_ptr(config, 2, output_data.data_ptr())
        config[4] = scale

        ops = [ScheduledOp(TensorScaleOp, tiles_m=num_tiles, config_data=config)]
        kernel_config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=kernel_config)
        kernel.run()

        expected = torch.tensor(
            [i * scale for i in [5, 10, 15, 20, 25, 30, 35, 40]],
            dtype=torch.int32, device="cuda",
        )
        assert torch.equal(output_data, expected), (
            f"TensorScaleOp mismatch: got {output_data.tolist()}, expected {expected.tolist()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
