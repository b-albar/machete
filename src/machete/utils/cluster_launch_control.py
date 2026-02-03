# Copyright (c) 2025, Machete Authors
"""Cluster Launch Control (CLC) utilities for Blackwell (SM100) architecture.

This module provides cuteDSL wrappers for NVIDIA's Cluster Launch Control PTX
instructions, enabling dynamic persistent kernel scheduling on Blackwell GPUs.

CLC allows kernels to dynamically request new work tiles from the hardware,
enabling efficient load balancing without host-side coordination.

Reference:
    https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html
"""

from dataclasses import dataclass
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass._mlir import ir


def _get_struct_type_i32x3():
    """Get LLVM struct type for 3 x i32 values."""
    i32_type = ir.IntegerType.get_signless(32)
    return llvm.StructType.get_literal([i32_type, i32_type, i32_type])


def _get_struct_type_i32x4():
    """Get LLVM struct type for 4 x i32 values."""
    i32_type = ir.IntegerType.get_signless(32)
    return llvm.StructType.get_literal([i32_type, i32_type, i32_type, i32_type])


@dataclass
class CLCResponse:
    """128-bit opaque response from Cluster Launch Control hardware.

    The CLC hardware returns a 128-bit response containing either:
    - A valid work tile with CTA coordinates (M, N, L indices)
    - A cancellation signal indicating no more work is available

    Attributes:
        data: Four 32-bit words containing the CLC response.
    """

    data0: Uint32
    data1: Uint32
    data2: Uint32
    data3: Uint32

    @staticmethod
    def zero() -> "CLCResponse":
        """Create a zero-initialized CLC response."""
        return CLCResponse(Uint32(0), Uint32(0), Uint32(0), Uint32(0))


@dataclass
class WorkTileInfo:
    """Information about a work tile obtained from CLC.

    Attributes:
        M_idx: M dimension tile index.
        N_idx: N dimension tile index.
        L_idx: L (batch) dimension tile index.
        is_valid_tile: Whether this represents a valid tile or cancellation.
    """

    M_idx: Int32
    N_idx: Int32
    L_idx: Int32
    is_valid_tile: Boolean


@dsl_user_op
def clc_try_cancel(
    result_smem_addr: Int32,
    mbarrier_smem_addr: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue an async CLC query to try to cancel (acquire) a new work tile.

    This instruction sends a request to the CLC hardware and asynchronously
    writes the 128-bit response to shared memory. The mbarrier is used for
    synchronization - it will be signaled when the response is ready.

    The instruction used is:
        clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128

    Args:
        result_smem_addr: Shared memory address (u32) where the 128-bit response will be written.
        mbarrier_smem_addr: Shared memory address (u32) of the mbarrier for completion signaling.
    """
    llvm.inline_asm(
        None,
        [result_smem_addr.ir_value(loc=loc, ip=ip), mbarrier_smem_addr.ir_value(loc=loc, ip=ip)],
        """
        {
            clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [$0], [$1];
        }
        """,
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def clc_query_is_canceled(clc_response: "CLCResponse", *, loc=None, ip=None) -> Boolean:
    """Check if the CLC response indicates a valid tile (is_canceled predicate).

    The is_canceled predicate is TRUE when a valid work tile is available.
    This may seem counterintuitive, but "canceled" refers to the kernel's
    pending cancellation being prevented by receiving new work.

    Args:
        clc_response: The 128-bit CLC response.

    Returns:
        True if a valid work tile is available, False if no more work.
    """
    result = llvm.inline_asm(
        T.i1(),
        [
            clc_response.data0.ir_value(loc=loc, ip=ip),
            clc_response.data1.ir_value(loc=loc, ip=ip),
            clc_response.data2.ir_value(loc=loc, ip=ip),
            clc_response.data3.ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .pred p1;
            .reg .b128 clc_result;
            mov.b128 clc_result, {$1, $2, $3, $4};
            clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;
            selp.u16 $0, 1, 0, p1;
        }
        """,
        "=c,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Boolean(result)


@dsl_user_op
def clc_query_get_first_ctaid(
    clc_response: "CLCResponse", *, loc=None, ip=None
) -> Tuple[Int32, Int32, Int32]:
    """Extract the CTA ID (M, N, L indices) from a CLC response.

    This should only be called when clc_query_is_canceled returns True.

    Args:
        clc_response: The 128-bit CLC response.

    Returns:
        Tuple of (M_idx, N_idx, L_idx) tile coordinates.
    """
    result = llvm.inline_asm(
        _get_struct_type_i32x3(),
        [
            clc_response.data0.ir_value(loc=loc, ip=ip),
            clc_response.data1.ir_value(loc=loc, ip=ip),
            clc_response.data2.ir_value(loc=loc, ip=ip),
            clc_response.data3.ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b128 clc_result;
            mov.b128 clc_result, {$3, $4, $5, $6};
            clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {$0, $1, $2, _}, clc_result;
        }
        """,
        "=r,=r,=r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    m_idx = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    n_idx = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    l_idx = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    return Int32(m_idx), Int32(n_idx), Int32(l_idx)


@dsl_user_op
def clc_load_response(smem_addr: Int32, *, loc=None, ip=None) -> "CLCResponse":
    """Load a 128-bit CLC response from shared memory.

    Args:
        smem_addr: Shared memory address (u32) of the 128-bit response.

    Returns:
        CLCResponse containing the four 32-bit words.
    """
    result = llvm.inline_asm(
        _get_struct_type_i32x4(),
        [smem_addr.ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    d0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    d1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    d2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    d3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)
    return CLCResponse(Uint32(d0), Uint32(d1), Uint32(d2), Uint32(d3))


@dsl_user_op
def clc_store_response(
    smem_addr: Int32, clc_response: "CLCResponse", *, loc=None, ip=None
) -> None:
    """Store a 128-bit CLC response to shared memory.

    Args:
        smem_addr: Shared memory address (u32) where to store the response.
        clc_response: The CLC response to store.
    """
    llvm.inline_asm(
        None,
        [
            smem_addr.ir_value(loc=loc, ip=ip),
            clc_response.data0.ir_value(loc=loc, ip=ip),
            clc_response.data1.ir_value(loc=loc, ip=ip),
            clc_response.data2.ir_value(loc=loc, ip=ip),
            clc_response.data3.ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def work_tile_info_from_clc_response(clc_response: "CLCResponse") -> WorkTileInfo:
    """Extract WorkTileInfo from a CLC response.

    This is a convenience function that combines clc_query_is_canceled
    and clc_query_get_first_ctaid into a single operation.

    Args:
        clc_response: The 128-bit CLC response.

    Returns:
        WorkTileInfo containing tile coordinates and validity flag.
    """
    is_valid = clc_query_is_canceled(clc_response)
    m_idx, n_idx, l_idx = clc_query_get_first_ctaid(clc_response)
    return WorkTileInfo(
        M_idx=m_idx if is_valid else Int32(0),
        N_idx=n_idx if is_valid else Int32(0),
        L_idx=l_idx if is_valid else Int32(0),
        is_valid_tile=is_valid,
    )


@dsl_user_op
def work_tile_info_from_smem(smem_addr: Int32, *, loc=None, ip=None) -> WorkTileInfo:
    """Extract WorkTileInfo directly from a shared memory CLC response.

    This is an optimized version that loads and decodes the CLC response
    in a single inline assembly block, avoiding intermediate register pressure.

    This matches the CUTLASS implementation in sm100_tile_scheduler.hpp.

    Args:
        smem_addr: Shared memory address (u32) of the 128-bit CLC response.

    Returns:
        WorkTileInfo containing tile coordinates and validity flag.
    """
    result = llvm.inline_asm(
        _get_struct_type_i32x4(),
        [smem_addr.ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred p1;
            .reg .b128 clc_result;
            ld.shared.b128 clc_result, [$4];
            clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;
            selp.u32 $3, 1, 0, p1;
            @p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {$0, $1, $2, _}, clc_result;
        }
        """,
        "=r,=r,=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    m_idx = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    n_idx = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    l_idx = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    valid = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)
    return WorkTileInfo(
        M_idx=Int32(m_idx),
        N_idx=Int32(n_idx),
        L_idx=Int32(l_idx),
        is_valid_tile=Boolean(Int32(valid) != 0),
    )


@cute.jit
def cast_smem_ptr_to_uint(smem_ptr: cute.Pointer) -> Int32:
    """Convert a shared memory pointer to a u32 address for PTX operations.

    Args:
        smem_ptr: Shared memory pointer.

    Returns:
        32-bit unsigned integer address.
    """
    return smem_ptr.toint()


# =============================================================================
# CLC Pipeline Constants
# =============================================================================

# CLC response size in bytes (128 bits = 16 bytes)
CLC_RESPONSE_BYTES: int = 16


# =============================================================================
# Mbarrier Operations for CLC Pipeline
# =============================================================================


@dsl_user_op
def mbarrier_init_for_clc(
    mbarrier_smem_addr: Int32,
    arrive_count: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Initialize an mbarrier for use with CLC.

    The mbarrier is initialized with the expected arrival count. For CLC,
    this is typically 1 (the scheduler thread) plus the transaction bytes
    from the CLC hardware.

    Args:
        mbarrier_smem_addr: Shared memory address of the mbarrier.
        arrive_count: Expected number of arrivals before the barrier trips.
    """
    llvm.inline_asm(
        None,
        [mbarrier_smem_addr.ir_value(loc=loc, ip=ip), arrive_count.ir_value(loc=loc, ip=ip)],
        "mbarrier.init.shared::cta.b64 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbarrier_arrive_expect_tx_for_clc(
    mbarrier_smem_addr: Int32,
    tx_count: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Arrive at mbarrier and set expected transaction count for CLC response.

    This is called before issuing a CLC query. It tells the mbarrier to expect
    tx_count bytes of data (the CLC response) before it can proceed.

    For CLC, tx_count should be CLC_RESPONSE_BYTES (16 bytes).

    Args:
        mbarrier_smem_addr: Shared memory address of the mbarrier.
        tx_count: Number of bytes expected from the async operation.
    """
    llvm.inline_asm(
        None,
        [mbarrier_smem_addr.ir_value(loc=loc, ip=ip), tx_count.ir_value(loc=loc, ip=ip)],
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbarrier_try_wait(
    mbarrier_smem_addr: Int32,
    phase: Int32,
    *,
    loc=None,
    ip=None,
) -> Boolean:
    """Try to wait on an mbarrier without blocking.

    Returns True if the barrier has been satisfied (all arrivals complete
    and transaction bytes received), False otherwise.

    Args:
        mbarrier_smem_addr: Shared memory address of the mbarrier.
        phase: Current phase bit (0 or 1) of the barrier.

    Returns:
        True if barrier is complete, False if still pending.
    """
    result = llvm.inline_asm(
        T.i32(),
        [mbarrier_smem_addr.ir_value(loc=loc, ip=ip), phase.ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred p1;
            mbarrier.try_wait.parity.shared::cta.b64 p1, [$0], $1;
            selp.u32 $2, 1, 0, p1;
        }
        """,
        "r,r,=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Boolean(Int32(result) != 0)


@dsl_user_op
def mbarrier_try_wait_parity_ticks(
    mbarrier_smem_addr: Int32,
    phase: Int32,
    ticks: Int32,
    *,
    loc=None,
    ip=None,
) -> Boolean:
    """Try to wait on an mbarrier with a timeout.

    Waits for up to `ticks` cycles before returning. This is useful for
    implementing spin-wait loops with bounded latency.

    Args:
        mbarrier_smem_addr: Shared memory address of the mbarrier.
        phase: Current phase bit (0 or 1) of the barrier.
        ticks: Maximum number of cycles to wait.

    Returns:
        True if barrier is complete, False if timed out.
    """
    result = llvm.inline_asm(
        T.i32(),
        [
            mbarrier_smem_addr.ir_value(loc=loc, ip=ip),
            phase.ir_value(loc=loc, ip=ip),
            ticks.ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .pred p1;
            mbarrier.try_wait.parity.shared::cta.b64 p1, [$0], $1, $2;
            selp.u32 $3, 1, 0, p1;
        }
        """,
        "r,r,r,=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Boolean(Int32(result) != 0)


@dsl_user_op
def mbarrier_arrive(
    mbarrier_smem_addr: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Signal arrival at an mbarrier.

    Args:
        mbarrier_smem_addr: Shared memory address of the mbarrier.
    """
    llvm.inline_asm(
        None,
        [mbarrier_smem_addr.ir_value(loc=loc, ip=ip)],
        "mbarrier.arrive.shared::cta.b64 _, [$0];",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbarrier_invalidate(
    mbarrier_smem_addr: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Invalidate an mbarrier.

    This should be called when the mbarrier is no longer needed,
    typically at kernel exit or when resetting for reuse.

    Args:
        mbarrier_smem_addr: Shared memory address of the mbarrier.
    """
    llvm.inline_asm(
        None,
        [mbarrier_smem_addr.ir_value(loc=loc, ip=ip)],
        "mbarrier.inval.shared::cta.b64 [$0];",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fence_mbarrier_init(*, loc=None, ip=None) -> None:
    """Issue a fence to ensure mbarrier initialization is visible.

    This fence should be called after initializing mbarriers and before
    any cluster-wide synchronization.
    """
    llvm.inline_asm(
        None,
        [],
        "fence.mbarrier_init.release.cluster;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# =============================================================================
# CLC Pipeline Helper Class
# =============================================================================


class CLCPipelineState:
    """Tracks state for a multi-stage CLC fetch pipeline.

    This class manages the bookkeeping for pipelined CLC queries, allowing
    multiple queries to be in flight simultaneously for latency hiding.

    Attributes:
        stages: Number of pipeline stages.
        index: Current stage index.
        phase: Current phase bit for mbarrier synchronization.
    """

    def __init__(self, stages: int):
        """Initialize the pipeline state.

        Args:
            stages: Number of pipeline stages (typically 2-3 for CLC).
        """
        self.stages = stages
        self.index = Int32(0)
        self.phase = Int32(0)

    @cute.jit
    def advance(self) -> None:
        """Advance to the next pipeline stage."""
        new_index = self.index + 1
        # Flip phase when wrapping around
        self.phase = self.phase ^ (new_index // self.stages)
        self.index = new_index % self.stages


@cute.jit
def clc_pipeline_issue_query(
    clc_response_smem_ptr: cute.Pointer,
    mbarrier_smem_ptr: cute.Pointer,
    stage_index: Int32,
) -> None:
    """Issue a CLC query for a specific pipeline stage.

    This function sets up the mbarrier to expect the CLC response and
    issues the async query. The response will be written to the shared
    memory location for the specified stage.

    Args:
        clc_response_smem_ptr: Base pointer to CLC response buffer in shared memory.
        mbarrier_smem_ptr: Base pointer to mbarrier array in shared memory.
        stage_index: Which pipeline stage to use (0 to stages-1).
    """
    # Calculate addresses for this stage
    response_addr = cast_smem_ptr_to_uint(clc_response_smem_ptr) + stage_index * CLC_RESPONSE_BYTES
    mbarrier_addr = cast_smem_ptr_to_uint(mbarrier_smem_ptr) + stage_index * 8  # mbarrier is 8 bytes

    # Set up mbarrier to expect CLC response bytes
    mbarrier_arrive_expect_tx_for_clc(mbarrier_addr, Int32(CLC_RESPONSE_BYTES))

    # Issue the async CLC query
    clc_try_cancel(response_addr, mbarrier_addr)


@cute.jit
def clc_pipeline_wait_and_get_work(
    clc_response_smem_ptr: cute.Pointer,
    mbarrier_smem_ptr: cute.Pointer,
    stage_index: Int32,
    phase: Int32,
) -> WorkTileInfo:
    """Wait for a CLC response and extract work tile info.

    This function waits for the CLC response at the specified stage to
    complete, then decodes it into work tile information.

    Args:
        clc_response_smem_ptr: Base pointer to CLC response buffer in shared memory.
        mbarrier_smem_ptr: Base pointer to mbarrier array in shared memory.
        stage_index: Which pipeline stage to wait on.
        phase: Current phase bit for the mbarrier.

    Returns:
        WorkTileInfo with the tile coordinates and validity flag.
    """
    mbarrier_addr = cast_smem_ptr_to_uint(mbarrier_smem_ptr) + stage_index * 8
    response_addr = cast_smem_ptr_to_uint(clc_response_smem_ptr) + stage_index * CLC_RESPONSE_BYTES

    # Spin-wait for the response
    # Note: In production, you might want to use the ticks variant for bounded waiting
    while not mbarrier_try_wait(mbarrier_addr, phase):
        pass

    # Decode the response
    return work_tile_info_from_smem(response_addr)


# =============================================================================
# Tile Scheduler Parameters (Host-side configuration)
# =============================================================================


@dataclass
class TileSchedulerParams:
    """Host-side parameters for the persistent tile scheduler.

    This structure holds all the configuration needed to launch a CLC-enabled
    kernel and schedule work tiles across the problem space.

    Attributes:
        problem_shape_m: Problem size in M dimension.
        problem_shape_n: Problem size in N dimension.
        problem_shape_l: Problem size in L (batch) dimension.
        tile_shape_m: Tile size in M dimension.
        tile_shape_n: Tile size in N dimension.
        cluster_shape_m: Cluster size in M dimension.
        cluster_shape_n: Cluster size in N dimension.
        swizzle_size: Size of swizzle pattern (power of 2, typically 1, 2, 4, or 8).
        raster_order_m_major: If True, rasterize M-major; if False, N-major.
    """

    problem_shape_m: int
    problem_shape_n: int
    problem_shape_l: int = 1
    tile_shape_m: int = 128
    tile_shape_n: int = 128
    cluster_shape_m: int = 1
    cluster_shape_n: int = 1
    swizzle_size: int = 1
    raster_order_m_major: bool = True

    def __post_init__(self):
        """Validate parameters after initialization."""
        assert self.problem_shape_m > 0, "Problem M must be positive"
        assert self.problem_shape_n > 0, "Problem N must be positive"
        assert self.problem_shape_l > 0, "Problem L must be positive"
        assert self.tile_shape_m > 0, "Tile M must be positive"
        assert self.tile_shape_n > 0, "Tile N must be positive"
        assert self.cluster_shape_m > 0, "Cluster M must be positive"
        assert self.cluster_shape_n > 0, "Cluster N must be positive"
        assert self.swizzle_size >= 1, "Swizzle size must be >= 1"
        # Swizzle size should be a power of 2
        assert (self.swizzle_size & (self.swizzle_size - 1)) == 0, "Swizzle size must be power of 2"

    @property
    def num_tiles_m(self) -> int:
        """Number of tiles in M dimension."""
        return (self.problem_shape_m + self.tile_shape_m - 1) // self.tile_shape_m

    @property
    def num_tiles_n(self) -> int:
        """Number of tiles in N dimension."""
        return (self.problem_shape_n + self.tile_shape_n - 1) // self.tile_shape_n

    @property
    def num_tiles_l(self) -> int:
        """Number of tiles in L dimension (same as problem_shape_l)."""
        return self.problem_shape_l

    @property
    def total_tiles(self) -> int:
        """Total number of tiles in the problem."""
        return self.num_tiles_m * self.num_tiles_n * self.num_tiles_l

    def get_grid_shape(self) -> Tuple[int, int, int]:
        """Compute the CUDA grid dimensions for kernel launch.

        Returns grid (x, y, z) where:
        - x = num_tiles_m rounded up to cluster_shape_m
        - y = num_tiles_n rounded up to cluster_shape_n
        - z = num_tiles_l

        Returns:
            Tuple of (grid_x, grid_y, grid_z).
        """
        # Round up to cluster boundaries
        grid_m = ((self.num_tiles_m + self.cluster_shape_m - 1) // self.cluster_shape_m) * self.cluster_shape_m
        grid_n = ((self.num_tiles_n + self.cluster_shape_n - 1) // self.cluster_shape_n) * self.cluster_shape_n

        if self.raster_order_m_major:
            return (grid_m, grid_n, self.num_tiles_l)
        else:
            return (grid_n, grid_m, self.num_tiles_l)

    def get_cluster_dims(self) -> Tuple[int, int, int]:
        """Get the cluster dimensions for kernel launch.

        Returns:
            Tuple of (cluster_x, cluster_y, cluster_z).
        """
        if self.raster_order_m_major:
            return (self.cluster_shape_m, self.cluster_shape_n, 1)
        else:
            return (self.cluster_shape_n, self.cluster_shape_m, 1)


# =============================================================================
# CLC Pipeline for Warp-Specialized Kernels
# =============================================================================


class CLCFetchPipeline:
    """Async pipeline for CLC work tile fetching.

    This class manages a multi-stage pipeline for issuing CLC queries and
    waiting for responses. It follows the producer-consumer pattern from
    CUTLASS's PipelineCLCFetchAsync.

    The pipeline allows overlapping CLC queries with computation:
    - Producer (scheduler warp) issues queries ahead of time
    - Consumer (compute warps) wait for and process responses

    Attributes:
        stages: Number of pipeline stages.
        role: Thread role (producer, consumer, or both).
    """

    # Thread role constants
    ROLE_PRODUCER = 0
    ROLE_CONSUMER = 1
    ROLE_PRODUCER_CONSUMER = 2
    ROLE_NON_PARTICIPANT = 3

    def __init__(self, stages: int = 2):
        """Initialize the CLC fetch pipeline.

        Args:
            stages: Number of pipeline stages (default 2).
        """
        self.stages = stages

    def smem_size(self) -> int:
        """Calculate required shared memory size.

        The shared memory layout is:
        - CLC response buffers: stages * 16 bytes (128 bits each)
        - Full barriers: stages * 8 bytes (mbarrier)
        - Empty barriers: stages * 8 bytes (mbarrier)

        Returns:
            Total shared memory size in bytes.
        """
        clc_response_size = self.stages * CLC_RESPONSE_BYTES
        mbarrier_size = self.stages * 8 * 2  # full + empty barriers
        return clc_response_size + mbarrier_size


class CLCPipelineProducerState:
    """State for the producer side of the CLC pipeline.

    The producer is responsible for issuing CLC queries ahead of time
    to hide latency.

    Attributes:
        stage: Current stage index.
        phase: Current phase bit for synchronization.
    """

    def __init__(self, stages: int):
        """Initialize producer state.

        Args:
            stages: Number of pipeline stages.
        """
        self.stages = stages
        self.stage = Int32(0)
        self.phase = Int32(0)

    @cute.jit
    def advance(self) -> None:
        """Advance to the next pipeline stage."""
        new_stage = self.stage + 1
        # Flip phase when wrapping around
        self.phase = self.phase ^ (new_stage // self.stages)
        self.stage = new_stage % self.stages


class CLCPipelineConsumerState:
    """State for the consumer side of the CLC pipeline.

    The consumer waits for CLC responses and processes work tiles.

    Attributes:
        stage: Current stage index.
        phase: Current phase bit for synchronization.
    """

    def __init__(self, stages: int):
        """Initialize consumer state.

        Args:
            stages: Number of pipeline stages.
        """
        self.stages = stages
        self.stage = Int32(0)
        self.phase = Int32(0)

    @cute.jit
    def advance(self) -> None:
        """Advance to the next pipeline stage."""
        new_stage = self.stage + 1
        # Flip phase when wrapping around
        self.phase = self.phase ^ (new_stage // self.stages)
        self.stage = new_stage % self.stages


# =============================================================================
# Persistent Tile Scheduler
# =============================================================================


class PersistentTileScheduler:
    """Persistent tile scheduler using Cluster Launch Control.

    This class implements the full tile scheduling infrastructure for
    CLC-enabled kernels. It manages:
    - Initial work tile assignment based on blockIdx
    - Dynamic work tile acquisition via CLC queries
    - Coordinate swizzling for L2 cache locality
    - Pipeline management for latency hiding

    Usage pattern in a kernel:
        1. Create scheduler in kernel init
        2. Call initial_work_tile_info() for first tile
        3. Loop:
           a. Process current tile
           b. Call fetch_next_work() to get next tile
           c. Call advance_to_next_work() to issue next query
           d. Exit when is_valid_tile is False

    Attributes:
        params: Host-side scheduler parameters.
        num_stages: Number of pipeline stages.
    """

    def __init__(self, params: TileSchedulerParams, num_stages: int = 2):
        """Initialize the persistent tile scheduler.

        Args:
            params: Tile scheduler configuration parameters.
            num_stages: Number of CLC pipeline stages (default 2).
        """
        self.params = params
        self.num_stages = num_stages

    def smem_size(self) -> int:
        """Calculate required shared memory for the scheduler.

        Returns:
            Shared memory size in bytes.
        """
        # CLC response buffer + mbarrier arrays (full + empty)
        clc_buffer_size = self.num_stages * CLC_RESPONSE_BYTES
        mbarrier_size = self.num_stages * 8 * 2  # full and empty barriers
        return clc_buffer_size + mbarrier_size

    def get_grid_shape(self) -> Tuple[int, int, int]:
        """Get CUDA grid dimensions for kernel launch.

        Returns:
            Tuple of (grid_x, grid_y, grid_z).
        """
        return self.params.get_grid_shape()

    def get_cluster_dims(self) -> Tuple[int, int, int]:
        """Get cluster dimensions for kernel launch.

        Returns:
            Tuple of (cluster_x, cluster_y, cluster_z).
        """
        return self.params.get_cluster_dims()


# =============================================================================
# Device-side Scheduler Functions
# =============================================================================


@dsl_user_op
def get_block_idx(*, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
    """Get the current block index (blockIdx.x, blockIdx.y, blockIdx.z).

    Returns:
        Tuple of (block_x, block_y, block_z).
    """
    result = llvm.inline_asm(
        _get_struct_type_i32x3(),
        [],
        """
        {
            mov.u32 $0, %ctaid.x;
            mov.u32 $1, %ctaid.y;
            mov.u32 $2, %ctaid.z;
        }
        """,
        "=r,=r,=r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    x = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    y = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    z = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    return Int32(x), Int32(y), Int32(z)


@dsl_user_op
def get_cluster_idx(*, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
    """Get the current cluster index.

    Returns:
        Tuple of (cluster_x, cluster_y, cluster_z).
    """
    result = llvm.inline_asm(
        _get_struct_type_i32x3(),
        [],
        """
        {
            mov.u32 $0, %clusterid.x;
            mov.u32 $1, %clusterid.y;
            mov.u32 $2, %clusterid.z;
        }
        """,
        "=r,=r,=r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    x = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    y = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    z = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    return Int32(x), Int32(y), Int32(z)


@dsl_user_op
def get_cluster_dim(*, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
    """Get the cluster dimensions.

    Returns:
        Tuple of (dim_x, dim_y, dim_z).
    """
    result = llvm.inline_asm(
        _get_struct_type_i32x3(),
        [],
        """
        {
            mov.u32 $0, %cluster_nctaid.x;
            mov.u32 $1, %cluster_nctaid.y;
            mov.u32 $2, %cluster_nctaid.z;
        }
        """,
        "=r,=r,=r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    x = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    y = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    z = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    return Int32(x), Int32(y), Int32(z)


@dsl_user_op
def get_block_idx_in_cluster(*, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
    """Get the block index within the cluster.

    Returns:
        Tuple of (block_x, block_y, block_z) within cluster.
    """
    result = llvm.inline_asm(
        _get_struct_type_i32x3(),
        [],
        """
        {
            mov.u32 $0, %cluster_ctaid.x;
            mov.u32 $1, %cluster_ctaid.y;
            mov.u32 $2, %cluster_ctaid.z;
        }
        """,
        "=r,=r,=r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    x = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    y = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    z = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    return Int32(x), Int32(y), Int32(z)


@cute.jit
def compute_initial_tile_coords(
    block_idx_x: Int32,
    block_idx_y: Int32,
    block_idx_z: Int32,
    cluster_shape_m: int,
    cluster_shape_n: int,
    raster_order_m_major: bool,
) -> WorkTileInfo:
    """Compute initial work tile coordinates from block index.

    This function converts the CUDA block index to tile coordinates,
    accounting for cluster shape and rasterization order.

    Args:
        block_idx_x: Block index X.
        block_idx_y: Block index Y.
        block_idx_z: Block index Z (batch dimension).
        cluster_shape_m: Cluster size in M.
        cluster_shape_n: Cluster size in N.
        raster_order_m_major: True for M-major rasterization.

    Returns:
        WorkTileInfo with initial tile coordinates.
    """
    if raster_order_m_major:
        m_idx = block_idx_x
        n_idx = block_idx_y
    else:
        m_idx = block_idx_y
        n_idx = block_idx_x

    return WorkTileInfo(
        M_idx=m_idx,
        N_idx=n_idx,
        L_idx=block_idx_z,
        is_valid_tile=Boolean(True),
    )


@cute.jit
def swizzle_tile_coords(
    m_idx: Int32,
    n_idx: Int32,
    l_idx: Int32,
    num_tiles_m: int,
    num_tiles_n: int,
    swizzle_size: int,
    cluster_shape_m: int,
    cluster_shape_n: int,
) -> WorkTileInfo:
    """Apply swizzle transformation to tile coordinates.

    Swizzling reorganizes the tile iteration order to improve L2 cache
    locality by processing nearby tiles together.

    The swizzle pattern divides the grid into swizzle_size x swizzle_size
    groups of clusters, then applies a serpentine pattern within each group.

    Args:
        m_idx: Original M tile index.
        n_idx: Original N tile index.
        l_idx: Batch dimension index.
        num_tiles_m: Total tiles in M dimension.
        num_tiles_n: Total tiles in N dimension.
        swizzle_size: Size of swizzle pattern.
        cluster_shape_m: Cluster size in M.
        cluster_shape_n: Cluster size in N.

    Returns:
        WorkTileInfo with swizzled coordinates.
    """
    if swizzle_size <= 1:
        # No swizzling
        return WorkTileInfo(
            M_idx=m_idx,
            N_idx=n_idx,
            L_idx=l_idx,
            is_valid_tile=Boolean(True),
        )

    # Compute cluster index
    cluster_m = m_idx // cluster_shape_m
    cluster_n = n_idx // cluster_shape_n

    # Index within cluster
    local_m = m_idx % cluster_shape_m
    local_n = n_idx % cluster_shape_n

    # Number of clusters in each dimension
    num_clusters_m = (num_tiles_m + cluster_shape_m - 1) // cluster_shape_m
    num_clusters_n = (num_tiles_n + cluster_shape_n - 1) // cluster_shape_n

    # Swizzle group index
    swizzle_group_m = cluster_m // swizzle_size
    swizzle_group_n = cluster_n // swizzle_size

    # Index within swizzle group
    swizzle_local_m = cluster_m % swizzle_size
    swizzle_local_n = cluster_n % swizzle_size

    # Apply serpentine pattern within swizzle group
    # Even rows go left-to-right, odd rows go right-to-left
    new_swizzle_local_n = swizzle_local_n
    if swizzle_local_m % 2 == 1:
        new_swizzle_local_n = swizzle_size - 1 - swizzle_local_n

    # Reconstruct cluster index
    new_cluster_m = swizzle_group_m * swizzle_size + swizzle_local_m
    new_cluster_n = swizzle_group_n * swizzle_size + new_swizzle_local_n

    # Reconstruct tile index
    new_m_idx = new_cluster_m * cluster_shape_m + local_m
    new_n_idx = new_cluster_n * cluster_shape_n + local_n

    # Check bounds
    is_valid = Boolean((new_m_idx < num_tiles_m) & (new_n_idx < num_tiles_n))

    return WorkTileInfo(
        M_idx=new_m_idx,
        N_idx=new_n_idx,
        L_idx=l_idx,
        is_valid_tile=is_valid,
    )


# =============================================================================
# Scheduler Initialization Helpers
# =============================================================================


@cute.jit
def scheduler_init_mbarriers(
    mbarrier_smem_ptr: cute.Pointer,
    num_stages: int,
    thread_idx: Int32,
) -> None:
    """Initialize mbarriers for the CLC pipeline.

    This should be called by thread 0 of the scheduler warp before any
    CLC queries are issued.

    Args:
        mbarrier_smem_ptr: Pointer to mbarrier array in shared memory.
        num_stages: Number of pipeline stages.
        thread_idx: Current thread index (only thread 0 initializes).
    """
    if thread_idx == 0:
        for stage in cutlass.range_constexpr(num_stages):
            # Full barrier (for producer -> consumer sync)
            full_mbar_addr = cast_smem_ptr_to_uint(mbarrier_smem_ptr) + stage * 8
            mbarrier_init_for_clc(full_mbar_addr, Int32(1))

            # Empty barrier (for consumer -> producer sync)
            empty_mbar_addr = full_mbar_addr + num_stages * 8
            mbarrier_init_for_clc(empty_mbar_addr, Int32(1))

        # Fence to ensure initialization is visible
        fence_mbarrier_init()


@cute.jit
def scheduler_prefill_pipeline(
    clc_response_smem_ptr: cute.Pointer,
    mbarrier_smem_ptr: cute.Pointer,
    num_stages: int,
    thread_idx: Int32,
) -> Int32:
    """Prefill the CLC pipeline with initial queries.

    Issues CLC queries for all pipeline stages to fill the pipeline.
    This should be called after mbarrier initialization.

    Args:
        clc_response_smem_ptr: Pointer to CLC response buffer.
        mbarrier_smem_ptr: Pointer to mbarrier array.
        num_stages: Number of pipeline stages.
        thread_idx: Current thread index.

    Returns:
        Initial producer stage index (wraps back to 0).
    """
    if thread_idx == 0:
        for stage in cutlass.range_constexpr(num_stages):
            clc_pipeline_issue_query(
                clc_response_smem_ptr,
                mbarrier_smem_ptr,
                Int32(stage),
            )

    return Int32(0)


# =============================================================================
# Cluster Synchronization
# =============================================================================


@dsl_user_op
def cluster_barrier_sync(*, loc=None, ip=None) -> None:
    """Synchronize all blocks in the cluster.

    This barrier ensures all CTAs in the cluster have reached this point
    before any can proceed.
    """
    llvm.inline_asm(
        None,
        [],
        "barrier.cluster.arrive.release.aligned;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    llvm.inline_asm(
        None,
        [],
        "barrier.cluster.wait.acquire.aligned;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cluster_barrier_arrive(*, loc=None, ip=None) -> None:
    """Signal arrival at cluster barrier without waiting."""
    llvm.inline_asm(
        None,
        [],
        "barrier.cluster.arrive.release.aligned;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cluster_barrier_wait(*, loc=None, ip=None) -> None:
    """Wait at cluster barrier (must have arrived first)."""
    llvm.inline_asm(
        None,
        [],
        "barrier.cluster.wait.acquire.aligned;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# =============================================================================
# Elect Leader in Cluster
# =============================================================================


@dsl_user_op
def elect_one_in_cluster(*, loc=None, ip=None) -> Boolean:
    """Elect one thread across the entire cluster.

    Returns True for exactly one thread in the cluster, False for all others.
    This is useful for designating a single "leader" thread to perform
    cluster-wide operations like CLC queries.

    Returns:
        True if this thread is elected, False otherwise.
    """
    result = llvm.inline_asm(
        T.i32(),
        [],
        """
        {
            .reg .pred p1;
            elect.sync _|p1, 0xFFFFFFFF;
            selp.u32 $0, 1, 0, p1;
        }
        """,
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Boolean(Int32(result) != 0)


@cute.jit
def is_cluster_leader() -> Boolean:
    """Check if this thread is the cluster leader.

    The cluster leader is defined as thread 0 of block 0 within the cluster.

    Returns:
        True if this is the cluster leader thread.
    """
    tidx = cute.arch.thread_idx()[0]
    block_in_cluster_x, block_in_cluster_y, block_in_cluster_z = get_block_idx_in_cluster()

    is_thread_0 = tidx == 0
    is_block_0 = (block_in_cluster_x == 0) & (block_in_cluster_y == 0) & (block_in_cluster_z == 0)

    return Boolean(is_thread_0 & is_block_0)
