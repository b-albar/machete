# Copyright (c) 2025, Machete Authors
"""Tests for Cluster Launch Control utilities.

Note: The actual CLC PTX instructions require SM100 (Blackwell) hardware.
These tests verify the module structure and can only fully run on Blackwell GPUs.
"""

import pytest
import torch


def test_imports():
    """Verify all CLC utilities can be imported."""
    from machete.megakernel.utils import (
        # Data structures
        CLCResponse,
        WorkTileInfo,
        CLCPipelineState,
        # Constants
        CLC_RESPONSE_BYTES,
        # Core CLC operations
        clc_try_cancel,
        clc_query_is_canceled,
        clc_query_get_first_ctaid,
        clc_load_response,
        clc_store_response,
        work_tile_info_from_clc_response,
        work_tile_info_from_smem,
        cast_smem_ptr_to_uint,
        # Mbarrier operations
        mbarrier_init_for_clc,
        mbarrier_arrive_expect_tx_for_clc,
        mbarrier_try_wait,
        mbarrier_try_wait_parity_ticks,
        mbarrier_arrive,
        mbarrier_invalidate,
        fence_mbarrier_init,
        # Pipeline helpers
        clc_pipeline_issue_query,
        clc_pipeline_wait_and_get_work,
        # Host-side configuration
        TileSchedulerParams,
        # CLC Pipeline classes
        CLCFetchPipeline,
        CLCPipelineProducerState,
        CLCPipelineConsumerState,
        # Persistent tile scheduler
        PersistentTileScheduler,
        # Device-side helpers
        get_block_idx,
        get_cluster_idx,
        get_cluster_dim,
        get_block_idx_in_cluster,
        compute_initial_tile_coords,
        swizzle_tile_coords,
        # Scheduler initialization
        scheduler_init_mbarriers,
        scheduler_prefill_pipeline,
        # Cluster synchronization
        cluster_barrier_sync,
        cluster_barrier_arrive,
        cluster_barrier_wait,
        # Leader election
        elect_one_in_cluster,
        is_cluster_leader,
    )

    # Verify constant value
    assert CLC_RESPONSE_BYTES == 16, "CLC response should be 128 bits = 16 bytes"


def test_clc_response_dataclass():
    """Test CLCResponse dataclass creation."""
    from machete.megakernel.utils import CLCResponse
    from cutlass import Uint32

    # Test zero initialization
    response = CLCResponse.zero()
    assert response.data0 == Uint32(0)
    assert response.data1 == Uint32(0)
    assert response.data2 == Uint32(0)
    assert response.data3 == Uint32(0)

    # Test custom initialization
    response = CLCResponse(Uint32(1), Uint32(2), Uint32(3), Uint32(4))
    assert response.data0 == Uint32(1)
    assert response.data1 == Uint32(2)
    assert response.data2 == Uint32(3)
    assert response.data3 == Uint32(4)


def test_work_tile_info_dataclass():
    """Test WorkTileInfo dataclass creation."""
    from machete.megakernel.utils import WorkTileInfo
    from cutlass import Boolean, Int32

    tile_info = WorkTileInfo(
        M_idx=Int32(10),
        N_idx=Int32(20),
        L_idx=Int32(0),
        is_valid_tile=Boolean(True),
    )

    assert tile_info.M_idx == Int32(10)
    assert tile_info.N_idx == Int32(20)
    assert tile_info.L_idx == Int32(0)
    assert tile_info.is_valid_tile == Boolean(True)


def test_clc_pipeline_state():
    """Test CLCPipelineState initialization."""
    from machete.megakernel.utils import CLCPipelineState
    from cutlass import Int32

    state = CLCPipelineState(stages=3)

    assert state.stages == 3
    assert state.index == Int32(0)
    assert state.phase == Int32(0)


def test_tile_scheduler_params():
    """Test TileSchedulerParams host-side configuration."""
    from machete.megakernel.utils import TileSchedulerParams

    # Test basic initialization
    params = TileSchedulerParams(
        problem_shape_m=1024,
        problem_shape_n=2048,
        problem_shape_l=4,
        tile_shape_m=128,
        tile_shape_n=256,
        cluster_shape_m=2,
        cluster_shape_n=1,
        swizzle_size=4,
        raster_order_m_major=True,
    )

    # Verify tile counts
    assert params.num_tiles_m == 8  # 1024 / 128
    assert params.num_tiles_n == 8  # 2048 / 256
    assert params.num_tiles_l == 4
    assert params.total_tiles == 8 * 8 * 4

    # Test grid shape computation
    grid = params.get_grid_shape()
    assert grid[0] == 8  # M tiles rounded to cluster (8 is already multiple of 2)
    assert grid[1] == 8  # N tiles rounded to cluster (8 is already multiple of 1)
    assert grid[2] == 4  # L dimension

    # Test cluster dims
    cluster_dims = params.get_cluster_dims()
    assert cluster_dims == (2, 1, 1)


def test_tile_scheduler_params_n_major():
    """Test TileSchedulerParams with N-major rasterization."""
    from machete.megakernel.utils import TileSchedulerParams

    params = TileSchedulerParams(
        problem_shape_m=512,
        problem_shape_n=1024,
        tile_shape_m=64,
        tile_shape_n=128,
        cluster_shape_m=1,
        cluster_shape_n=2,
        raster_order_m_major=False,
    )

    # Grid should be transposed for N-major
    grid = params.get_grid_shape()
    assert grid[0] == 8  # N tiles (1024/128)
    assert grid[1] == 8  # M tiles (512/64)
    assert grid[2] == 1

    # Cluster dims should be transposed
    cluster_dims = params.get_cluster_dims()
    assert cluster_dims == (2, 1, 1)


def test_tile_scheduler_params_validation():
    """Test TileSchedulerParams validation."""
    from machete.megakernel.utils import TileSchedulerParams
    import pytest

    # Non-power-of-2 swizzle should fail
    with pytest.raises(AssertionError):
        TileSchedulerParams(
            problem_shape_m=1024,
            problem_shape_n=1024,
            swizzle_size=3,  # Not a power of 2
        )


def test_clc_fetch_pipeline():
    """Test CLCFetchPipeline configuration."""
    from machete.megakernel.utils import CLCFetchPipeline, CLC_RESPONSE_BYTES

    pipeline = CLCFetchPipeline(stages=3)

    # Verify smem calculation: 3 * 16 (responses) + 3 * 8 * 2 (barriers)
    expected_smem = 3 * CLC_RESPONSE_BYTES + 3 * 8 * 2
    assert pipeline.smem_size() == expected_smem


def test_clc_pipeline_producer_consumer_state():
    """Test CLCPipelineProducerState and CLCPipelineConsumerState."""
    from machete.megakernel.utils import CLCPipelineProducerState, CLCPipelineConsumerState
    from cutlass import Int32

    # Test producer state
    producer = CLCPipelineProducerState(stages=2)
    assert producer.stages == 2
    assert producer.stage == Int32(0)
    assert producer.phase == Int32(0)

    # Test consumer state
    consumer = CLCPipelineConsumerState(stages=3)
    assert consumer.stages == 3
    assert consumer.stage == Int32(0)
    assert consumer.phase == Int32(0)


def test_persistent_tile_scheduler():
    """Test PersistentTileScheduler full configuration."""
    from machete.megakernel.utils import PersistentTileScheduler, TileSchedulerParams

    params = TileSchedulerParams(
        problem_shape_m=4096,
        problem_shape_n=4096,
        problem_shape_l=2,
        tile_shape_m=128,
        tile_shape_n=128,
        cluster_shape_m=2,
        cluster_shape_n=2,
    )

    scheduler = PersistentTileScheduler(params, num_stages=3)

    # Check grid shape
    grid = scheduler.get_grid_shape()
    assert grid[0] == 32  # 4096/128 = 32 tiles
    assert grid[1] == 32
    assert grid[2] == 2

    # Check cluster dims
    cluster_dims = scheduler.get_cluster_dims()
    assert cluster_dims == (2, 2, 1)

    # Check smem size
    # 3 stages * 16 bytes (CLC response) + 3 * 8 * 2 (barriers)
    expected_smem = 3 * 16 + 3 * 8 * 2
    assert scheduler.smem_size() == expected_smem


def is_blackwell_available():
    """Check if Blackwell (SM100) GPU is available."""
    if not torch.cuda.is_available():
        return False

    # Get compute capability
    major, minor = torch.cuda.get_device_capability()
    # Blackwell is SM100 (compute capability 10.0)
    return major >= 10


@pytest.mark.skipif(not is_blackwell_available(), reason="CLC instructions require Blackwell (SM100) GPU")
def test_clc_kernel_compilation():
    """Test that a simple CLC-using kernel can compile on Blackwell.

    This test is skipped on non-Blackwell hardware.
    """
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from machete.megakernel.utils import (
        CLCResponse,
        WorkTileInfo,
        CLC_RESPONSE_BYTES,
        clc_try_cancel,
        work_tile_info_from_smem,
        mbarrier_init_for_clc,
        mbarrier_arrive_expect_tx_for_clc,
        mbarrier_try_wait,
        cast_smem_ptr_to_uint,
    )

    class CLCTestKernel:
        """Simple test kernel that issues a CLC query."""

        def __init__(self):
            self.num_stages = 2

        @cute.jit
        def __call__(self, stream):
            self.kernel().launch(
                grid=[1, 1, 1],
                block=[32, 1, 1],
                smem=self._smem_size(),
                stream=stream,
            )

        def _smem_size(self):
            # CLC response buffer + mbarrier array
            return CLC_RESPONSE_BYTES * self.num_stages + 8 * self.num_stages

        @cute.kernel
        def kernel(self):
            tidx = cute.arch.thread_idx()[0]
            smem = cutlass.utils.SmemAllocator()

            # Allocate CLC response buffer (16 bytes per stage)
            clc_buffer = smem.allocate_array(Int32, num_elems=4 * self.num_stages)
            # Allocate mbarrier array (8 bytes per stage)
            mbar_buffer = smem.allocate_array(Int64, num_elems=self.num_stages)

            # Initialize mbarriers (only thread 0)
            if tidx == 0:
                for stage in cutlass.range_constexpr(self.num_stages):
                    mbar_addr = cast_smem_ptr_to_uint(mbar_buffer) + stage * 8
                    mbarrier_init_for_clc(mbar_addr, Int32(1))

            cute.arch.barrier()

            # Issue CLC query (only thread 0)
            if tidx == 0:
                clc_addr = cast_smem_ptr_to_uint(clc_buffer)
                mbar_addr = cast_smem_ptr_to_uint(mbar_buffer)

                # Set up mbarrier
                mbarrier_arrive_expect_tx_for_clc(mbar_addr, Int32(CLC_RESPONSE_BYTES))

                # Issue query
                clc_try_cancel(clc_addr, mbar_addr)

                # Wait for response
                phase = Int32(0)
                while not mbarrier_try_wait(mbar_addr, phase):
                    pass

                # Get work tile info
                tile_info = work_tile_info_from_smem(clc_addr)

                # The kernel would normally use tile_info.M_idx, N_idx, L_idx
                # to determine which tile to process
                cute.printf(
                    "tile_info: %d %d %d %d\n",
                    tile_info.M_idx,
                    tile_info.N_idx,
                    tile_info.L_idx,
                    tile_info.is_valid_tile,
                )

    # Create and launch the kernel
    import cuda.bindings.driver as cuda
    kernel = CLCTestKernel()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kernel(stream)
    torch.cuda.synchronize()
    print("CLC kernel launched and completed successfully!")


if __name__ == "__main__":
    print("Running CLC import tests...")
    test_imports()
    print("  PASSED: test_imports")

    test_clc_response_dataclass()
    print("  PASSED: test_clc_response_dataclass")

    test_work_tile_info_dataclass()
    print("  PASSED: test_work_tile_info_dataclass")

    test_clc_pipeline_state()
    print("  PASSED: test_clc_pipeline_state")

    print("\nRunning infrastructure tests...")
    test_tile_scheduler_params()
    print("  PASSED: test_tile_scheduler_params")

    test_tile_scheduler_params_n_major()
    print("  PASSED: test_tile_scheduler_params_n_major")

    test_tile_scheduler_params_validation()
    print("  PASSED: test_tile_scheduler_params_validation")

    test_clc_fetch_pipeline()
    print("  PASSED: test_clc_fetch_pipeline")

    test_clc_pipeline_producer_consumer_state()
    print("  PASSED: test_clc_pipeline_producer_consumer_state")

    test_persistent_tile_scheduler()
    print("  PASSED: test_persistent_tile_scheduler")

    if is_blackwell_available():
        print("\nBlackwell GPU detected, running CLC kernel compilation test...")
        test_clc_kernel_compilation()
        print("  PASSED: test_clc_kernel_compilation")
    else:
        print("\nBlackwell GPU not available, skipping CLC kernel compilation test")

    print("\nAll tests passed!")
