# Copyright (c) 2025, Machete Authors
"""
Test: Chain with Mixed Tile Dimensions (GPU)

Dependency Pattern:
==================

A chain of ops where some have extra dimensions that others don't.
Tests combinations of 1D and 2D ops in a chain.

    Op A (M x N)  →  Op B (M)  →  Op C (M x N)
       16 tiles       4 tiles       16 tiles

Test Strategy:
    A: writes tile_m * 1000 + tile_n to matrix[m * N + n]
    B: reads matrix, writes m * 100 + 1 to buf[m]
    C: reads buf[m], writes buf[m] * (n + 1) to output[m * N + n]
    Verify: output[m * N + n] = (m * 100 + 1) * (n + 1)
"""

import pytest
import torch
from typing import ClassVar, List

import cutlass.cute as cute
from cutlass import Int32, Int64

from machete.megakernel import Megakernel, MegakernelConfig, ScheduledOp
from machete.megakernel.ops import Op
from machete.megakernel.interpreter import st_global_i32, ld_global_i32


def is_hopper_available():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


# =============================================================================
# Global result tensor pointers
# =============================================================================

_matrix_ptr = 0
_buf_ptr = 0
_output_ptr = 0
_num_cols = 4


# =============================================================================
# GPU Tests
# =============================================================================


@pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
class TestChainMixedDimsGPU:
    """
    GPU test for chain with mixed tile dimensions.

    Tests combinations where some ops have extra dimensions.
    """

    def test_2d_1d_2d_chain(self):
        """
        Chain: A(2D M=4, N=4) -> B(1D M=4) -> C(2D M=4, N=4)

        Op A writes to 4x4 matrix.
        Op B reads from matrix, writes to 1D buffer.
        Op C reads from buffer, writes to 4x4 output.
        """
        global _matrix_ptr, _buf_ptr, _output_ptr, _num_cols
        tiles_m = 4
        tiles_n = 4
        _num_cols = tiles_n

        class OpA(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 0
            NUM_OUTPUT_PAGES: ClassVar[int] = 0
            INPUTS: ClassVar[List[str]] = []
            OUTPUTS: ClassVar[List[str]] = ["matrix"]

            @staticmethod
            def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    idx = tile_m * Int32(_num_cols) + tile_n
                    st_global_i32(Int64(_matrix_ptr), idx, tile_m * Int32(1000) + tile_n)

        class OpB(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 0
            NUM_OUTPUT_PAGES: ClassVar[int] = 0
            INPUTS: ClassVar[List[str]] = ["matrix"]
            OUTPUTS: ClassVar[List[str]] = ["buf"]

            @staticmethod
            def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    # Read last value for this M to verify all producers completed
                    _ = ld_global_i32(Int64(_matrix_ptr), tile_m * Int32(_num_cols) + Int32(_num_cols - 1))
                    st_global_i32(Int64(_buf_ptr), tile_m, tile_m * Int32(100) + Int32(1))

        class OpC(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 0
            NUM_OUTPUT_PAGES: ClassVar[int] = 0
            INPUTS: ClassVar[List[str]] = ["buf"]
            OUTPUTS: ClassVar[List[str]] = []

            @staticmethod
            def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    val = ld_global_i32(Int64(_buf_ptr), tile_m)
                    out_idx = tile_m * Int32(_num_cols) + tile_n
                    st_global_i32(Int64(_output_ptr), out_idx, val * (tile_n + Int32(1)))

        matrix = torch.zeros(tiles_m * tiles_n, dtype=torch.int32, device="cuda")
        buf = torch.zeros(tiles_m, dtype=torch.int32, device="cuda")
        output = torch.zeros(tiles_m * tiles_n, dtype=torch.int32, device="cuda")

        _matrix_ptr = matrix.data_ptr()
        _buf_ptr = buf.data_ptr()
        _output_ptr = output.data_ptr()

        ops = [
            ScheduledOp(OpA, tiles_m=tiles_m, tiles_n=tiles_n, dim_names={"batch": "m", "seq": "n"}),
            ScheduledOp(OpB, tiles_m=tiles_m, dim_names={"batch": "m"}),
            ScheduledOp(OpC, tiles_m=tiles_m, tiles_n=tiles_n, dim_names={"batch": "m", "seq": "n"}),
        ]

        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        # Expected: (m * 100 + 1) * (n + 1)
        expected = torch.tensor([
            (m * 100 + 1) * (n + 1)
            for m in range(tiles_m)
            for n in range(tiles_n)
        ], dtype=torch.int32, device="cuda")
        assert torch.equal(output, expected), (
            f"Expected {expected.tolist()}, got {output.tolist()}"
        )

    def test_1d_2d_1d_chain(self):
        """
        Chain: A(1D M=4) -> B(2D M=4, N=4) -> C(1D M=4)

        Op A writes to 1D buffer.
        Op B reads from buffer, writes to 4x4 matrix.
        Op C reads from matrix, writes to 1D output.
        """
        global _buf1_ptr, _matrix_ptr, _output_ptr, _num_cols
        _buf1_ptr = 0
        tiles_m = 4
        tiles_n = 4
        _num_cols = tiles_n

        class OpA(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 0
            NUM_OUTPUT_PAGES: ClassVar[int] = 0
            INPUTS: ClassVar[List[str]] = []
            OUTPUTS: ClassVar[List[str]] = ["buf1"]

            @staticmethod
            def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    st_global_i32(Int64(_buf1_ptr), tile_m, tile_m * Int32(10) + Int32(1))

        class OpB(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 0
            NUM_OUTPUT_PAGES: ClassVar[int] = 0
            INPUTS: ClassVar[List[str]] = ["buf1"]
            OUTPUTS: ClassVar[List[str]] = ["matrix"]

            @staticmethod
            def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    val = ld_global_i32(Int64(_buf1_ptr), tile_m)
                    out_idx = tile_m * Int32(_num_cols) + tile_n
                    st_global_i32(Int64(_matrix_ptr), out_idx, val * (tile_n + Int32(1)))

        class OpC(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 0
            NUM_OUTPUT_PAGES: ClassVar[int] = 0
            INPUTS: ClassVar[List[str]] = ["matrix"]
            OUTPUTS: ClassVar[List[str]] = []

            @staticmethod
            def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    # Read last value for this M
                    val = ld_global_i32(Int64(_matrix_ptr), tile_m * Int32(_num_cols) + Int32(_num_cols - 1))
                    st_global_i32(Int64(_output_ptr), tile_m, val * Int32(2))

        buf1 = torch.zeros(tiles_m, dtype=torch.int32, device="cuda")
        matrix = torch.zeros(tiles_m * tiles_n, dtype=torch.int32, device="cuda")
        output = torch.zeros(tiles_m, dtype=torch.int32, device="cuda")

        _buf1_ptr = buf1.data_ptr()
        _matrix_ptr = matrix.data_ptr()
        _output_ptr = output.data_ptr()

        ops = [
            ScheduledOp(OpA, tiles_m=tiles_m, dim_names={"batch": "m"}),
            ScheduledOp(OpB, tiles_m=tiles_m, tiles_n=tiles_n, dim_names={"batch": "m", "seq": "n"}),
            ScheduledOp(OpC, tiles_m=tiles_m, dim_names={"batch": "m"}),
        ]

        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        # OpA: m * 10 + 1
        # OpB: (m * 10 + 1) * (n + 1) for n in 0..3
        # OpC reads (m * 10 + 1) * 4 (last column n=3), multiplies by 2
        # Expected: (m * 10 + 1) * 4 * 2 = (m * 10 + 1) * 8
        expected = torch.tensor([
            (m * 10 + 1) * 8 for m in range(tiles_m)
        ], dtype=torch.int32, device="cuda")
        assert torch.equal(output, expected), (
            f"Expected {expected.tolist()}, got {output.tolist()}"
        )

    def test_four_op_chain(self):
        """
        Four-op chain: A(1D) -> B(2D) -> C(2D) -> D(1D)

        Tests longer chain with dimension changes.
        """
        global _a_ptr, _b_ptr, _c_ptr, _d_ptr, _num_cols
        _a_ptr = 0
        _b_ptr = 0
        _c_ptr = 0
        _d_ptr = 0
        tiles_m = 4
        tiles_n = 4
        _num_cols = tiles_n

        class OpA(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 0
            NUM_OUTPUT_PAGES: ClassVar[int] = 0
            INPUTS: ClassVar[List[str]] = []
            OUTPUTS: ClassVar[List[str]] = ["a"]

            @staticmethod
            def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    st_global_i32(Int64(_a_ptr), tile_m, tile_m + Int32(1))

        class OpB(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 0
            NUM_OUTPUT_PAGES: ClassVar[int] = 0
            INPUTS: ClassVar[List[str]] = ["a"]
            OUTPUTS: ClassVar[List[str]] = ["b"]

            @staticmethod
            def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    val = ld_global_i32(Int64(_a_ptr), tile_m)
                    out_idx = tile_m * Int32(_num_cols) + tile_n
                    st_global_i32(Int64(_b_ptr), out_idx, val * Int32(10))

        class OpC(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 0
            NUM_OUTPUT_PAGES: ClassVar[int] = 0
            INPUTS: ClassVar[List[str]] = ["b"]
            OUTPUTS: ClassVar[List[str]] = ["c"]

            @staticmethod
            def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    in_idx = tile_m * Int32(_num_cols) + tile_n
                    val = ld_global_i32(Int64(_b_ptr), in_idx)
                    st_global_i32(Int64(_c_ptr), in_idx, val + tile_n)

        class OpD(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 0
            NUM_OUTPUT_PAGES: ClassVar[int] = 0
            INPUTS: ClassVar[List[str]] = ["c"]
            OUTPUTS: ClassVar[List[str]] = []

            @staticmethod
            def forward(smem_base, config_ptr, page_ids, tile_m, tile_n, tile_l, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    # Read last value for this M
                    val = ld_global_i32(Int64(_c_ptr), tile_m * Int32(_num_cols) + Int32(_num_cols - 1))
                    st_global_i32(Int64(_d_ptr), tile_m, val)

        a = torch.zeros(tiles_m, dtype=torch.int32, device="cuda")
        b = torch.zeros(tiles_m * tiles_n, dtype=torch.int32, device="cuda")
        c = torch.zeros(tiles_m * tiles_n, dtype=torch.int32, device="cuda")
        d = torch.zeros(tiles_m, dtype=torch.int32, device="cuda")

        _a_ptr = a.data_ptr()
        _b_ptr = b.data_ptr()
        _c_ptr = c.data_ptr()
        _d_ptr = d.data_ptr()

        ops = [
            ScheduledOp(OpA, tiles_m=tiles_m, dim_names={"batch": "m"}),
            ScheduledOp(OpB, tiles_m=tiles_m, tiles_n=tiles_n, dim_names={"batch": "m", "seq": "n"}),
            ScheduledOp(OpC, tiles_m=tiles_m, tiles_n=tiles_n, dim_names={"batch": "m", "seq": "n"}),
            ScheduledOp(OpD, tiles_m=tiles_m, dim_names={"batch": "m"}),
        ]

        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        # OpA: m + 1
        # OpB: (m + 1) * 10 for all n
        # OpC: (m + 1) * 10 + n
        # OpD reads n=3: (m + 1) * 10 + 3
        expected = torch.tensor([
            (m + 1) * 10 + 3 for m in range(tiles_m)
        ], dtype=torch.int32, device="cuda")
        assert torch.equal(d, expected), (
            f"Expected {expected.tolist()}, got {d.tolist()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
