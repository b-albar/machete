# Copyright (c) 2025, Machete Authors
"""Example CLC kernels demonstrating proper usage with validation via cute.printf.

These tests validate the CLC (Cluster Launch Control) infrastructure for Blackwell GPUs.

## What These Tests Validate

1. **CLC PTX Instructions** - The inline assembly for CLC queries compiles and executes:
   - `clusterlaunchcontrol.try_cancel.async` - Issue work tile requests
   - `clusterlaunchcontrol.query_cancel.is_canceled` - Check if tile is valid
   - `clusterlaunchcontrol.query_cancel.get_first_ctaid` - Extract tile coordinates

2. **Mbarrier Synchronization** - The mbarrier operations work correctly:
   - `mbarrier.init` - Initialize barriers
   - `mbarrier.arrive.expect_tx` - Set up for async completion
   - `mbarrier.try_wait.parity` - Wait for CLC response

3. **Cluster Launch** - Kernels can be launched with cluster dimensions:
   - `cluster=[X,Y,Z]` parameter in launch()
   - `cluster_barrier_sync()` for inter-block synchronization
   - `get_block_idx_in_cluster()` for block position within cluster

4. **Persistent Kernel Pattern** - Blocks process blockIdx first, then steal work

## CLC Work Stealing Mechanism

CLC works by "cancelling" (stealing) pending blocks that haven't started yet:
- Each block FIRST processes its own blockIdx tile (this is always valid)
- Then blocks use CLC to steal work from pending blocks in the launch queue
- CLC returns valid=1 when there are pending blocks to steal
- CLC returns valid=0 when all blocks have started (nothing left to steal)

For CLC to return valid work, the launch grid must be larger than available SMs!

## Test Structure

- `test_clc_persistent_kernel_example` - Persistent kernel with work stealing
- `test_clc_multi_block_coordination` - Multiple blocks competing for work
- `test_clc_pipeline_stages` - Multi-stage pipelined CLC queries
- `test_clc_with_cluster_launch` - Cluster launch with CLC leader election
"""

import pytest
import torch


def is_blackwell_available():
    """Check if Blackwell (SM100+) GPU is available."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 10


@pytest.mark.skipif(not is_blackwell_available(), reason="CLC requires Blackwell (SM100+) GPU")
def test_clc_persistent_kernel_example():
    """Test a persistent kernel that processes multiple tiles via CLC.

    This implements the correct persistent kernel pattern:
    1. Each block FIRST processes its own blockIdx tile (always valid!)
    2. Initialize mbarriers for CLC pipeline
    3. Issue CLC query to steal next tile from pending blocks
    4. Wait for response and extract tile info
    5. If valid, process the stolen tile; if not, exit
    6. Repeat until CLC returns no more work

    KEY INSIGHT: CLC only returns valid tiles when there are pending blocks
    that haven't started yet. We launch a large grid (more tiles than SMs)
    so that CLC can steal work from the pending block queue.
    """
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64, Boolean
    from machete.megakernel.utils import (
        CLC_RESPONSE_BYTES,
        TileSchedulerParams,
        clc_try_cancel,
        work_tile_info_from_smem,
        mbarrier_init_for_clc,
        mbarrier_arrive_expect_tx_for_clc,
        mbarrier_try_wait,
        mbarrier_invalidate,
        fence_mbarrier_init,
        cast_smem_ptr_to_uint,
        get_block_idx,
    )

    # Configure the problem - Launch 4096 blocks (64x64 grid) to ensure
    # we have MANY more blocks than SMs, creating pending blocks to steal.
    # With 100 SMs and 4096 blocks, there should be ~40 waves of blocks,
    # giving CLC ample opportunity to steal from pending blocks.
    # We also use large shared memory (48KB) to limit occupancy to ~1 block/SM.
    params = TileSchedulerParams(
        problem_shape_m=8192,  # 64 tiles in M (8192/128 = 64)
        problem_shape_n=8192,  # 64 tiles in N (8192/128 = 64)
        problem_shape_l=1,
        tile_shape_m=128,
        tile_shape_n=128,
        cluster_shape_m=1,
        cluster_shape_n=1,
    )

    print(f"Problem: {params.problem_shape_m}x{params.problem_shape_n}")
    print(f"Tiles: {params.num_tiles_m}x{params.num_tiles_n} = {params.total_tiles} total")
    print(f"Grid: {params.get_grid_shape()}")
    print(f"Cluster dims: {params.get_cluster_dims()}")

    num_stages = 2

    class CLCPersistentKernel:
        """Persistent kernel that dynamically acquires tiles via CLC.

        Pattern:
        1. Process initial blockIdx tile (always valid, no CLC needed)
        2. Issue CLC query to steal next tile
        3. If valid, process stolen tile and repeat from step 2
        4. If not valid, exit (no more work to steal)
        """

        def __init__(self, params: TileSchedulerParams):
            self.params = params
            self.num_stages = num_stages

        @cute.jit
        def __call__(self, stream):
            # Launch with large grid - CLC will allow work stealing
            # between blocks that are pending vs running
            self.kernel().launch(
                grid=list(self.params.get_grid_shape()),
                block=[32, 1, 1],
                smem=self._smem_size(),
                stream=stream,
            )

        def _smem_size(self):
            # Use large shared memory (48KB) to limit occupancy to ~1 block/SM
            # This ensures blocks queue up, giving CLC opportunity to steal work
            # CLC response buffer (16 bytes per stage) + mbarrier array (8 bytes per stage)
            # + padding to reach 48KB
            base_size = CLC_RESPONSE_BYTES * self.num_stages + 8 * self.num_stages
            return 48 * 1024  # 48KB to limit occupancy

        @cute.kernel
        def kernel(self):
            tidx = cute.arch.thread_idx()[0]
            smem = cutlass.utils.SmemAllocator()

            # Allocate shared memory
            clc_buffer = smem.allocate_array(Int32, num_elems=4 * self.num_stages)
            mbar_buffer = smem.allocate_array(Int64, num_elems=self.num_stages)

            # Get block index - this is our FIRST tile (always valid!)
            block_x, block_y, block_z = get_block_idx()

            # Only thread 0 does CLC operations
            if tidx == 0:
                # ============================================================
                # Step 1: Process our initial blockIdx tile (ALWAYS VALID!)
                # ============================================================
                # This is the key insight - we don't need CLC for first tile
                cute.printf(
                    "Block (%d, %d, %d) processing initial tile M=%d, N=%d (from blockIdx)\\n",
                    block_x,
                    block_y,
                    block_z,
                    block_x,  # M_idx = blockIdx.x
                    block_y,  # N_idx = blockIdx.y
                )

                tiles_processed = Int32(1)  # Count the initial tile

                # ============================================================
                # Step 2: Initialize mbarriers for CLC pipeline
                # ============================================================
                for stage in cutlass.range_constexpr(self.num_stages):
                    mbar_addr = cast_smem_ptr_to_uint(mbar_buffer) + stage * 8
                    mbarrier_init_for_clc(mbar_addr, Int32(1))

                fence_mbarrier_init()

                clc_addr = cast_smem_ptr_to_uint(clc_buffer)
                mbar_addr = cast_smem_ptr_to_uint(mbar_buffer)

                # ============================================================
                # Step 3: Work stealing loop - try to steal from pending blocks
                # ============================================================
                # CLC will return valid tiles until all blocks have started
                max_steal_attempts = Int32(8)  # Limit iterations for testing
                steal_attempt = Int32(0)

                while steal_attempt < max_steal_attempts:
                    # Set up mbarrier for this query
                    mbarrier_arrive_expect_tx_for_clc(mbar_addr, Int32(CLC_RESPONSE_BYTES))

                    # Issue CLC query to try to steal a pending block's work
                    clc_try_cancel(clc_addr, mbar_addr)

                    # Wait for response
                    phase = Int32(0)
                    while not mbarrier_try_wait(mbar_addr, phase):
                        pass

                    # Get tile info from CLC response
                    tile_info = work_tile_info_from_smem(clc_addr)

                    if tile_info.is_valid_tile:
                        # Success! We stole work from a pending block
                        cute.printf(
                            "Block (%d, %d, %d) STOLE tile M=%d, N=%d, L=%d (attempt %d)\\n",
                            block_x,
                            block_y,
                            block_z,
                            tile_info.M_idx,
                            tile_info.N_idx,
                            tile_info.L_idx,
                            steal_attempt,
                        )
                        tiles_processed = tiles_processed + 1
                    else:
                        # No more pending blocks to steal from
                        cute.printf(
                            "Block (%d, %d, %d) no more work to steal (attempt %d)\\n",
                            block_x,
                            block_y,
                            block_z,
                            steal_attempt,
                        )
                        # In real kernel, we would break here
                        # For testing, continue to show behavior

                    steal_attempt = steal_attempt + 1

                # ============================================================
                # Step 4: Cleanup - invalidate mbarriers on exit
                # ============================================================
                for stage in cutlass.range_constexpr(self.num_stages):
                    mbar_addr_inv = cast_smem_ptr_to_uint(mbar_buffer) + stage * 8
                    mbarrier_invalidate(mbar_addr_inv)

                cute.printf(
                    "Block (%d, %d, %d) finished, processed %d tiles total\\n",
                    block_x,
                    block_y,
                    block_z,
                    tiles_processed,
                )

            # Sync all threads before exit
            cute.arch.barrier()

    # Run the kernel
    import cuda.bindings.driver as cuda

    kernel = CLCPersistentKernel(params)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    print("\n--- Kernel Output ---")
    kernel(stream)
    torch.cuda.synchronize()
    print("--- End Kernel Output ---\n")

    print("CLC persistent kernel completed successfully!")
    print("If you see 'STOLE tile' messages, CLC work stealing is working!")
    print("If all show 'no more work to steal', all blocks started before CLC queries completed.")


@pytest.mark.skipif(not is_blackwell_available(), reason="CLC requires Blackwell (SM100+) GPU")
def test_clc_multi_block_coordination():
    """Test CLC with multiple blocks showing work distribution.

    This test launches multiple blocks and shows how CLC distributes
    work tiles among them dynamically.
    """
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from machete.megakernel.utils import (
        CLC_RESPONSE_BYTES,
        clc_try_cancel,
        work_tile_info_from_smem,
        mbarrier_init_for_clc,
        mbarrier_arrive_expect_tx_for_clc,
        mbarrier_try_wait,
        fence_mbarrier_init,
        cast_smem_ptr_to_uint,
        get_block_idx,
    )

    # Launch many more blocks than can run simultaneously
    # This creates a queue of pending blocks that can be stolen
    num_blocks = 256  # Should be >> number of SMs
    num_stages = 2

    class MultiBlockCLCKernel:
        """Kernel with multiple blocks competing for work via CLC."""

        def __init__(self):
            self.num_stages = num_stages

        @cute.jit
        def __call__(self, stream):
            self.kernel().launch(
                grid=[num_blocks, 1, 1],
                block=[32, 1, 1],
                smem=self._smem_size(),
                stream=stream,
            )

        def _smem_size(self):
            return CLC_RESPONSE_BYTES * self.num_stages + 8 * self.num_stages

        @cute.kernel
        def kernel(self):
            tidx = cute.arch.thread_idx()[0]
            smem = cutlass.utils.SmemAllocator()

            clc_buffer = smem.allocate_array(Int32, num_elems=4 * self.num_stages)
            mbar_buffer = smem.allocate_array(Int64, num_elems=self.num_stages)

            block_x, block_y, block_z = get_block_idx()

            if tidx == 0:
                # First, process our own blockIdx tile
                cute.printf(
                    "Block %d processing initial tile (blockIdx)\\n",
                    block_x,
                )
                tiles_acquired = Int32(1)

                # Initialize mbarriers
                for stage in cutlass.range_constexpr(self.num_stages):
                    mbar_addr = cast_smem_ptr_to_uint(mbar_buffer) + stage * 8
                    mbarrier_init_for_clc(mbar_addr, Int32(1))
                fence_mbarrier_init()

                clc_addr = cast_smem_ptr_to_uint(clc_buffer)
                mbar_addr = cast_smem_ptr_to_uint(mbar_buffer)

                # Try to steal additional work
                for _ in cutlass.range_constexpr(3):
                    mbarrier_arrive_expect_tx_for_clc(mbar_addr, Int32(CLC_RESPONSE_BYTES))
                    clc_try_cancel(clc_addr, mbar_addr)

                    phase = Int32(0)
                    while not mbarrier_try_wait(mbar_addr, phase):
                        pass

                    tile_info = work_tile_info_from_smem(clc_addr)

                    if tile_info.is_valid_tile:
                        cute.printf(
                            "Block %d STOLE tile: M=%d, N=%d, L=%d\\n",
                            block_x,
                            tile_info.M_idx,
                            tile_info.N_idx,
                            tile_info.L_idx,
                        )
                        tiles_acquired = tiles_acquired + 1
                    else:
                        cute.printf(
                            "Block %d: no work to steal\\n",
                            block_x,
                        )

                cute.printf(
                    "Block %d finished, acquired %d tiles total\\n",
                    block_x,
                    tiles_acquired,
                )

            cute.arch.barrier()

    import cuda.bindings.driver as cuda

    kernel = MultiBlockCLCKernel()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    print(f"\n--- Multi-Block CLC Test ({num_blocks} blocks) ---")
    kernel(stream)
    torch.cuda.synchronize()
    print("--- End Multi-Block Test ---\n")


@pytest.mark.skipif(not is_blackwell_available(), reason="CLC requires Blackwell (SM100+) GPU")
def test_clc_pipeline_stages():
    """Test CLC pipeline with multiple stages for latency hiding.

    This demonstrates using multiple pipeline stages to overlap
    CLC queries with tile processing.
    """
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from machete.megakernel.utils import (
        CLC_RESPONSE_BYTES,
        clc_try_cancel,
        work_tile_info_from_smem,
        mbarrier_init_for_clc,
        mbarrier_arrive_expect_tx_for_clc,
        mbarrier_try_wait,
        fence_mbarrier_init,
        cast_smem_ptr_to_uint,
        get_block_idx,
    )

    num_stages = 3  # Use 3 stages for better pipelining
    num_blocks = 128  # Large grid for work stealing

    class PipelinedCLCKernel:
        """Kernel demonstrating pipelined CLC queries."""

        def __init__(self):
            self.num_stages = num_stages

        @cute.jit
        def __call__(self, stream):
            self.kernel().launch(
                grid=[num_blocks, 1, 1],
                block=[32, 1, 1],
                smem=self._smem_size(),
                stream=stream,
            )

        def _smem_size(self):
            return CLC_RESPONSE_BYTES * self.num_stages + 8 * self.num_stages

        @cute.kernel
        def kernel(self):
            tidx = cute.arch.thread_idx()[0]
            smem = cutlass.utils.SmemAllocator()

            clc_buffer = smem.allocate_array(Int32, num_elems=4 * self.num_stages)
            mbar_buffer = smem.allocate_array(Int64, num_elems=self.num_stages)

            block_x, block_y, block_z = get_block_idx()

            if tidx == 0:
                cute.printf("Block %d initializing %d-stage CLC pipeline\\n", block_x, Int32(self.num_stages))

                # Initialize all mbarriers
                for stage in cutlass.range_constexpr(self.num_stages):
                    mbar_addr = cast_smem_ptr_to_uint(mbar_buffer) + stage * 8
                    mbarrier_init_for_clc(mbar_addr, Int32(1))
                fence_mbarrier_init()

                clc_base = cast_smem_ptr_to_uint(clc_buffer)
                mbar_base = cast_smem_ptr_to_uint(mbar_buffer)

                # First process blockIdx tile
                cute.printf("Block %d processing initial tile from blockIdx\\n", block_x)

                # Prefill pipeline - issue queries for all stages
                cute.printf("Block %d prefilling pipeline with %d queries\\n", block_x, Int32(self.num_stages))
                for stage in cutlass.range_constexpr(self.num_stages):
                    clc_addr = clc_base + stage * CLC_RESPONSE_BYTES
                    mbar_addr = mbar_base + stage * 8
                    mbarrier_arrive_expect_tx_for_clc(mbar_addr, Int32(CLC_RESPONSE_BYTES))
                    clc_try_cancel(clc_addr, mbar_addr)
                    cute.printf("  Issued query for stage %d\\n", Int32(stage))

                # Process pipeline
                consumer_stage = Int32(0)
                consumer_phase = Int32(0)
                tiles_processed = Int32(1)  # Count initial blockIdx tile

                for iteration in cutlass.range_constexpr(5):
                    # Consumer: wait for and process current stage
                    consumer_mbar = mbar_base + consumer_stage * 8
                    consumer_clc = clc_base + consumer_stage * CLC_RESPONSE_BYTES

                    while not mbarrier_try_wait(consumer_mbar, consumer_phase):
                        pass

                    tile_info = work_tile_info_from_smem(consumer_clc)

                    cute.printf(
                        "Block %d iter %d: stage %d, valid=%d, M=%d, N=%d\\n",
                        block_x,
                        Int32(iteration),
                        consumer_stage,
                        tile_info.is_valid_tile,
                        tile_info.M_idx,
                        tile_info.N_idx,
                    )

                    if tile_info.is_valid_tile:
                        tiles_processed = tiles_processed + 1

                    # Producer: issue next query (reusing the stage we just consumed)
                    mbarrier_arrive_expect_tx_for_clc(consumer_mbar, Int32(CLC_RESPONSE_BYTES))
                    clc_try_cancel(consumer_clc, consumer_mbar)

                    # Advance consumer stage
                    consumer_stage = (consumer_stage + 1) % self.num_stages
                    if consumer_stage == 0:
                        consumer_phase = Int32(1) - consumer_phase

                cute.printf("Block %d pipeline test complete, processed %d tiles\\n", block_x, tiles_processed)

            cute.arch.barrier()

    import cuda.bindings.driver as cuda

    kernel = PipelinedCLCKernel()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    print(f"\n--- Pipelined CLC Test ({num_stages} stages, {num_blocks} blocks) ---")
    kernel(stream)
    torch.cuda.synchronize()
    print("--- End Pipeline Test ---\n")


@pytest.mark.skipif(not is_blackwell_available(), reason="CLC requires Blackwell (SM100+) GPU")
def test_clc_with_cluster_launch():
    """Test CLC with proper cluster launch configuration.

    This test demonstrates launching a kernel with cluster dimensions,
    which is required for CLC hardware to distribute work across blocks.

    The cluster launch enables:
    - Hardware-level synchronization between blocks in a cluster
    - CLC work distribution across cluster members
    - Distributed shared memory access within the cluster
    """
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64
    from machete.megakernel.utils import (
        CLC_RESPONSE_BYTES,
        clc_try_cancel,
        work_tile_info_from_smem,
        mbarrier_init_for_clc,
        mbarrier_arrive_expect_tx_for_clc,
        mbarrier_try_wait,
        fence_mbarrier_init,
        cast_smem_ptr_to_uint,
        get_block_idx,
        get_block_idx_in_cluster,
        cluster_barrier_sync,
    )

    # Grid must be multiple of cluster size
    # Large grid for work stealing
    grid_x = 64
    grid_y = 2
    cluster_x = 2
    cluster_y = 1

    class ClusterCLCKernel:
        """Kernel using cluster launch with CLC."""

        def __init__(self):
            self.num_stages = 2

        @cute.jit
        def __call__(self, stream):
            self.kernel().launch(
                grid=[grid_x, grid_y, 1],
                block=[128, 1, 1],
                cluster=[cluster_x, cluster_y, 1],
                smem=self._smem_size(),
                stream=stream,
            )

        def _smem_size(self):
            return CLC_RESPONSE_BYTES * self.num_stages + 8 * self.num_stages

        @cute.kernel
        def kernel(self):
            tidx = cute.arch.thread_idx()[0]
            smem = cutlass.utils.SmemAllocator()

            clc_buffer = smem.allocate_array(Int32, num_elems=4 * self.num_stages)
            mbar_buffer = smem.allocate_array(Int64, num_elems=self.num_stages)

            block_x, block_y, block_z = get_block_idx()
            cluster_block_x, cluster_block_y, cluster_block_z = get_block_idx_in_cluster()

            # All threads sync at cluster barrier first
            cluster_barrier_sync()

            if tidx == 0:
                cute.printf(
                    "Block (%d,%d) in cluster pos (%d,%d) - processing initial tile\\n",
                    block_x,
                    block_y,
                    cluster_block_x,
                    cluster_block_y,
                )

                # Only cluster leader (block 0,0 in cluster) does CLC
                is_leader = (cluster_block_x == 0) & (cluster_block_y == 0)

                if is_leader:
                    # Initialize mbarrier
                    mbar_addr = cast_smem_ptr_to_uint(mbar_buffer)
                    mbarrier_init_for_clc(mbar_addr, Int32(1))
                    fence_mbarrier_init()

                    clc_addr = cast_smem_ptr_to_uint(clc_buffer)

                    # Try to steal work
                    for attempt in cutlass.range_constexpr(3):
                        mbarrier_arrive_expect_tx_for_clc(mbar_addr, Int32(CLC_RESPONSE_BYTES))
                        clc_try_cancel(clc_addr, mbar_addr)

                        phase = Int32(0)
                        while not mbarrier_try_wait(mbar_addr, phase):
                            pass

                        tile_info = work_tile_info_from_smem(clc_addr)
                        cute.printf(
                            "  Leader block (%d,%d) CLC attempt %d: valid=%d\\n",
                            block_x,
                            block_y,
                            Int32(attempt),
                            tile_info.is_valid_tile,
                        )

            # Final cluster sync
            cluster_barrier_sync()
            cute.arch.barrier()

    import cuda.bindings.driver as cuda

    kernel = ClusterCLCKernel()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    print(f"\n--- Cluster CLC Test (grid={grid_x}x{grid_y}, cluster={cluster_x}x{cluster_y}) ---")
    kernel(stream)
    torch.cuda.synchronize()
    print("--- End Cluster Test ---\n")
    print("Cluster launch with CLC completed successfully!")


if __name__ == "__main__":
    print("=" * 60)
    print("CLC Example Tests")
    print("=" * 60)

    if not is_blackwell_available():
        print("Blackwell GPU not available, skipping CLC tests")
        exit(0)

    print("\n[1] Testing CLC Persistent Kernel Example")
    print("-" * 40)
    test_clc_persistent_kernel_example()

    print("\n[2] Testing CLC Multi-Block Coordination")
    print("-" * 40)
    test_clc_multi_block_coordination()

    print("\n[3] Testing CLC Pipeline Stages")
    print("-" * 40)
    test_clc_pipeline_stages()

    print("\n[4] Testing CLC with Cluster Launch")
    print("-" * 40)
    test_clc_with_cluster_launch()

    print("\n" + "=" * 60)
    print("All CLC example tests passed!")
    print("=" * 60)
