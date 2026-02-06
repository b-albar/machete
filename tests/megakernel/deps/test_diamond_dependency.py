# Copyright (c) 2025, Machete Authors
"""
Test: Diamond Dependency Pattern (GPU)

Dependency Pattern:
==================

Multiple ops depend on the same producer, and another op depends on all of them.
Forms a diamond shape in the dependency graph.

                    ┌───────┐
                    │   A   │
                    └───┬───┘
                   ╱         ╲
                  ╱           ╲
           ┌─────▼─────┐ ┌─────▼─────┐
           │     B     │ │     C     │
           └─────┬─────┘ └─────┬─────┘
                  ╲           ╱
                   ╲         ╱
                    ▼───────▼
                    │   D   │
                    └───────┘

Barrier Mapping:
    - A → B: barrier set 1
    - A → C: barrier set 2 (same producer, different consumer)
    - B → D: barrier set 3
    - C → D: barrier set 4

Test Strategy:
    A: writes (tile_m + 1) to data[tile_m]
    B: reads data[tile_m], writes data[tile_m] * 2 to buf_b[tile_m]
    C: reads data[tile_m], writes data[tile_m] * 3 to buf_c[tile_m]
    D: reads buf_b[tile_m] and buf_c[tile_m], writes sum to output[tile_m]
    Verify: output[m] = (m+1)*2 + (m+1)*3 = (m+1)*5
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

_data_ptr = 0
_buf_b_ptr = 0
_buf_c_ptr = 0
_output_ptr = 0


# =============================================================================
# Test Ops with Real GPU Operations
# =============================================================================


class OpA(Op):
    """Root producer: writes (tile_m + 1) to data[tile_m]."""
    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0
    INPUTS: ClassVar[List[str]] = []
    OUTPUTS: ClassVar[List[str]] = ["data"]

    @staticmethod
    def compute(
        page_ptr: Int32,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            st_global_i32(Int64(_data_ptr), tile_m, tile_m + Int32(1))


class OpB(Op):
    """Branch B: reads data[tile_m], writes data[tile_m] * 2 to buf_b[tile_m]."""
    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0
    INPUTS: ClassVar[List[str]] = ["data"]
    OUTPUTS: ClassVar[List[str]] = ["buf_b"]

    @staticmethod
    def compute(
        page_ptr: Int32,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            val = ld_global_i32(Int64(_data_ptr), tile_m)
            st_global_i32(Int64(_buf_b_ptr), tile_m, val * Int32(2))


class OpC(Op):
    """Branch C: reads data[tile_m], writes data[tile_m] * 3 to buf_c[tile_m]."""
    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0
    INPUTS: ClassVar[List[str]] = ["data"]
    OUTPUTS: ClassVar[List[str]] = ["buf_c"]

    @staticmethod
    def compute(
        page_ptr: Int32,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            val = ld_global_i32(Int64(_data_ptr), tile_m)
            st_global_i32(Int64(_buf_c_ptr), tile_m, val * Int32(3))


class OpD(Op):
    """Join: reads buf_b and buf_c, writes sum to output."""
    NUM_INPUT_PAGES: ClassVar[int] = 0
    NUM_OUTPUT_PAGES: ClassVar[int] = 0
    INPUTS: ClassVar[List[str]] = ["buf_b", "buf_c"]
    OUTPUTS: ClassVar[List[str]] = []

    @staticmethod
    def compute(
        page_ptr: Int32,
        tile_m: Int32, tile_n: Int32, tile_l: Int32,
        op_config_ptr: Int64,
    ) -> None:
        tidx = cute.arch.thread_idx()[0]
        if tidx == Int32(0):
            b_val = ld_global_i32(Int64(_buf_b_ptr), tile_m)
            c_val = ld_global_i32(Int64(_buf_c_ptr), tile_m)
            st_global_i32(Int64(_output_ptr), tile_m, b_val + c_val)


# =============================================================================
# GPU Tests
# =============================================================================


@pytest.mark.skipif(not is_hopper_available(), reason="Hopper (SM90+) GPU required")
class TestDiamondDependencyGPU:
    """
    GPU test for diamond dependency pattern.

    Diagram:
                        ┌─────────┐
                        │   OpA   │  writes (m+1) to data[m]
                        └────┬────┘
                       ╱          ╲
                      ╱            ╲
               ┌─────▼─────┐  ┌─────▼─────┐
               │    OpB    │  │    OpC    │
               │  *2 path  │  │  *3 path  │
               └─────┬─────┘  └─────┬─────┘
                      ╲            ╱
                       ╲          ╱
                        ▼────────▼
                        │   OpD   │  sums both paths
                        └─────────┘

    A writes: (m+1)
    B computes: (m+1) * 2
    C computes: (m+1) * 3
    D computes: (m+1)*2 + (m+1)*3 = (m+1)*5
    """

    def test_basic_diamond(self):
        """
        Basic diamond with 4 tiles.

        A writes: [1, 2, 3, 4]
        B writes: [2, 4, 6, 8]
        C writes: [3, 6, 9, 12]
        D writes: [5, 10, 15, 20]
        """
        global _data_ptr, _buf_b_ptr, _buf_c_ptr, _output_ptr
        num_tiles = 4

        data = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        buf_b = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        buf_c = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        output = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")

        _data_ptr = data.data_ptr()
        _buf_b_ptr = buf_b.data_ptr()
        _buf_c_ptr = buf_c.data_ptr()
        _output_ptr = output.data_ptr()

        ops = [
            ScheduledOp(OpA, tiles_m=num_tiles, dim_names={"batch": "m"}),
            ScheduledOp(OpB, tiles_m=num_tiles, dim_names={"batch": "m"}),
            ScheduledOp(OpC, tiles_m=num_tiles, dim_names={"batch": "m"}),
            ScheduledOp(OpD, tiles_m=num_tiles, dim_names={"batch": "m"}),
        ]

        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        # Verify output: (m+1) * 5
        expected = torch.tensor([5, 10, 15, 20], dtype=torch.int32, device="cuda")
        assert torch.equal(output, expected), (
            f"Expected {expected.tolist()}, got {output.tolist()}"
        )

    def test_larger_diamond(self):
        """
        Diamond with 16 tiles.

        Verifies synchronization at larger scale.
        """
        global _data_ptr, _buf_b_ptr, _buf_c_ptr, _output_ptr
        num_tiles = 16

        data = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        buf_b = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        buf_c = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        output = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")

        _data_ptr = data.data_ptr()
        _buf_b_ptr = buf_b.data_ptr()
        _buf_c_ptr = buf_c.data_ptr()
        _output_ptr = output.data_ptr()

        ops = [
            ScheduledOp(OpA, tiles_m=num_tiles, dim_names={"batch": "m"}),
            ScheduledOp(OpB, tiles_m=num_tiles, dim_names={"batch": "m"}),
            ScheduledOp(OpC, tiles_m=num_tiles, dim_names={"batch": "m"}),
            ScheduledOp(OpD, tiles_m=num_tiles, dim_names={"batch": "m"}),
        ]

        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        # Expected: (m+1) * 5
        expected = torch.tensor([(m + 1) * 5 for m in range(num_tiles)], dtype=torch.int32, device="cuda")
        assert torch.equal(output, expected), (
            f"Mismatch at some indices"
        )

    def test_three_branch_diamond(self):
        """
        Diamond with three parallel branches.

                  A
                / | \
               B  C  D
                \ | /
                  E

        Tests synchronization with more than two branches.
        """
        global _a_ptr, _b_ptr, _c_ptr, _d_ptr, _e_ptr
        _a_ptr = 0
        _b_ptr = 0
        _c_ptr = 0
        _d_ptr = 0
        _e_ptr = 0
        num_tiles = 4

        class OpA3(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 0
            NUM_OUTPUT_PAGES: ClassVar[int] = 0
            INPUTS: ClassVar[List[str]] = []
            OUTPUTS: ClassVar[List[str]] = ["a"]

            @staticmethod
            def compute(page_ptr, tile_m, tile_n, tile_l, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    st_global_i32(Int64(_a_ptr), tile_m, tile_m + Int32(1))

        class OpB3(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 0
            NUM_OUTPUT_PAGES: ClassVar[int] = 0
            INPUTS: ClassVar[List[str]] = ["a"]
            OUTPUTS: ClassVar[List[str]] = ["b"]

            @staticmethod
            def compute(page_ptr, tile_m, tile_n, tile_l, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    val = ld_global_i32(Int64(_a_ptr), tile_m)
                    st_global_i32(Int64(_b_ptr), tile_m, val * Int32(2))

        class OpC3(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 0
            NUM_OUTPUT_PAGES: ClassVar[int] = 0
            INPUTS: ClassVar[List[str]] = ["a"]
            OUTPUTS: ClassVar[List[str]] = ["c"]

            @staticmethod
            def compute(page_ptr, tile_m, tile_n, tile_l, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    val = ld_global_i32(Int64(_a_ptr), tile_m)
                    st_global_i32(Int64(_c_ptr), tile_m, val * Int32(3))

        class OpD3(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 0
            NUM_OUTPUT_PAGES: ClassVar[int] = 0
            INPUTS: ClassVar[List[str]] = ["a"]
            OUTPUTS: ClassVar[List[str]] = ["d"]

            @staticmethod
            def compute(page_ptr, tile_m, tile_n, tile_l, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    val = ld_global_i32(Int64(_a_ptr), tile_m)
                    st_global_i32(Int64(_d_ptr), tile_m, val * Int32(4))

        class OpE3(Op):
            NUM_INPUT_PAGES: ClassVar[int] = 0
            NUM_OUTPUT_PAGES: ClassVar[int] = 0
            INPUTS: ClassVar[List[str]] = ["b", "c", "d"]
            OUTPUTS: ClassVar[List[str]] = []

            @staticmethod
            def compute(page_ptr, tile_m, tile_n, tile_l, op_config_ptr):
                tidx = cute.arch.thread_idx()[0]
                if tidx == Int32(0):
                    b_val = ld_global_i32(Int64(_b_ptr), tile_m)
                    c_val = ld_global_i32(Int64(_c_ptr), tile_m)
                    d_val = ld_global_i32(Int64(_d_ptr), tile_m)
                    st_global_i32(Int64(_e_ptr), tile_m, b_val + c_val + d_val)

        a = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        b = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        c = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        d = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
        e = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")

        _a_ptr = a.data_ptr()
        _b_ptr = b.data_ptr()
        _c_ptr = c.data_ptr()
        _d_ptr = d.data_ptr()
        _e_ptr = e.data_ptr()

        ops = [
            ScheduledOp(OpA3, tiles_m=num_tiles, dim_names={"batch": "m"}),
            ScheduledOp(OpB3, tiles_m=num_tiles, dim_names={"batch": "m"}),
            ScheduledOp(OpC3, tiles_m=num_tiles, dim_names={"batch": "m"}),
            ScheduledOp(OpD3, tiles_m=num_tiles, dim_names={"batch": "m"}),
            ScheduledOp(OpE3, tiles_m=num_tiles, dim_names={"batch": "m"}),
        ]

        config = MegakernelConfig(num_sms=2)
        kernel = Megakernel(ops, config=config)
        kernel.run()

        # Expected: (m+1)*2 + (m+1)*3 + (m+1)*4 = (m+1)*9
        expected = torch.tensor([(m + 1) * 9 for m in range(num_tiles)], dtype=torch.int32, device="cuda")
        assert torch.equal(e, expected), (
            f"Expected {expected.tolist()}, got {e.tolist()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
