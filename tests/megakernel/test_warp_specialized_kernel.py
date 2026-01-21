# Copyright (c) 2025, Machete Authors
"""
Test warp-specialized execution mode for MacheteKernel.

This test demonstrates a kernel that runs in warp-specialized mode,
where different warps execute different roles (loader, consumer, storer)
concurrently with semaphore-based synchronization.
"""

import pytest
import torch
from typing import Dict, Tuple

try:
    import cutlass.cute as cute
    from cutlass import Float32, const_expr

    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False

from machete.megakernel import MacheteKernel, TensorSpec, WarpConfig


class MatMulTileKernel(MacheteKernel):
    """Tiled matrix multiplication kernel using warp specialization.

    Computes: C = A @ B (simplified tile)

    This kernel demonstrates the warp-specialized execution pattern where:
    - Loader warps: Fetch A and B tiles from global to shared memory
    - Consumer warps: Perform tile matrix multiplication
    - Storer warps: Write C tile back to global memory

    All three warp groups run concurrently with semaphore synchronization.
    """

    NUM_THREADS = 640  # 20 warps = 16 consumer + 2 loader + 1 storer + 1 controller
    NUM_STAGES = 2  # Double buffering for pipelining
    TILE_M = 64
    TILE_N = 64
    TILE_K = 32

    def __init__(self, dtype: torch.dtype, M: int, N: int, K: int):
        self.torch_dtype = dtype
        self.M = M
        self.N = N
        self.K = K
        self.element_size = 2 if dtype == torch.float16 else 4

    @property
    def smem_size_fwd(self) -> int:
        """Shared memory for A and B tiles with double buffering.

        Layout: NUM_STAGES * (A_tile + B_tile)
        A_tile: TILE_M * TILE_K
        B_tile: TILE_K * TILE_N
        """
        a_tile = self.TILE_M * self.TILE_K * self.element_size
        b_tile = self.TILE_K * self.TILE_N * self.element_size
        return self.NUM_STAGES * (a_tile + b_tile)

    def declare_tensors(self) -> Dict[str, TensorSpec]:
        """Declare input and output tensors."""
        return {
            "A": TensorSpec(
                name="A",
                dtype=None,
                shape_expr=("M", "K"),
                is_input=True,
            ),
            "B": TensorSpec(
                name="B",
                dtype=None,
                shape_expr=("K", "N"),
                is_input=True,
            ),
            "C": TensorSpec(
                name="C",
                dtype=None,
                shape_expr=("M", "N"),
                is_output=True,
            ),
        }

    def declare_scalars(self) -> Tuple[str, ...]:
        """Declare scalar parameters."""
        return ("M", "N", "K")

    def get_logical_grid_size(self, A, B, C, M, N, K) -> int:
        """One logical block per output tile."""
        tiles_m = (M + self.TILE_M - 1) // self.TILE_M
        tiles_n = (N + self.TILE_N - 1) // self.TILE_N
        return tiles_m * tiles_n

    # ========== Warp Specialization Configuration ==========

    @property
    def uses_warp_specialization(self) -> bool:
        """Enable warp-specialized execution."""
        return True

    @property
    def warp_config(self) -> WarpConfig:
        """Configure warp roles for matmul.

        - 16 consumer warps for MMA operations
        - 2 loader warps for async copy of A and B
        - 1 storer warp for writing C
        - 1 controller warp for pipeline coordination
        """
        return WarpConfig(
            num_consumer_warps=16,
            num_loader_warps=2,
            num_storer_warps=1,
            num_launcher_warps=0,
            num_controller_warps=1,
        )

    def setup_kernel(self, logical_idx, smem, A, B, C, M, N, K):
        """Setup per-block state for the tile."""
        tiles_n = (N + self.TILE_N - 1) // self.TILE_N
        self.tile_m = logical_idx // tiles_n
        self.tile_n = logical_idx % tiles_n

    def load_forward(self, logical_idx, smem, A, B, C, M, N, K):
        """Loader warps: fetch A and B tiles to shared memory.

        In warp-specialized mode, only loader warps execute this.
        Uses async copy for maximum memory bandwidth.
        """
        # Load A tile: A[tile_m*TILE_M : (tile_m+1)*TILE_M, :]
        # Load B tile: B[:, tile_n*TILE_N : (tile_n+1)*TILE_N]
        pass

    def compute_forward(self, logical_idx, smem, A, B, C, M, N, K):
        """Consumer warps: perform tile matrix multiplication.

        In warp-specialized mode, only consumer warps execute this.
        Uses tensor cores (MMA) for compute.
        """
        # Compute C_tile = A_tile @ B_tile
        pass

    def store_forward(self, logical_idx, smem, A, B, C, M, N, K):
        """Storer warps: write C tile back to global memory.

        In warp-specialized mode, only storer warps execute this.
        """
        # Store C[tile_m*TILE_M : (tile_m+1)*TILE_M,
        #         tile_n*TILE_N : (tile_n+1)*TILE_N]
        pass



class TestWarpSpecializedKernel:
    """Test suite for warp-specialized kernel execution."""

    def test_kernel_interface(self):
        """Test that the kernel interface is correctly defined."""
        kernel = MatMulTileKernel(torch.float16, M=512, N=512, K=256)

        # Check warp specialization is enabled
        assert kernel.uses_warp_specialization is True
        assert kernel.NUM_STAGES == 2  # Double buffering

        # Check tensor declarations
        tensors = kernel.declare_tensors()
        assert "A" in tensors
        assert "B" in tensors
        assert "C" in tensors
        assert tensors["A"].is_input is True
        assert tensors["C"].is_output is True

    def test_warp_config(self):
        """Test warp configuration for specialized execution."""
        kernel = MatMulTileKernel(torch.float16, M=512, N=512, K=256)
        config = kernel.warp_config

        assert config.num_consumer_warps == 16
        assert config.num_loader_warps == 2
        assert config.num_storer_warps == 1
        assert config.num_controller_warps == 1

        # Total warps should match thread count
        total_warps = (
            config.num_consumer_warps
            + config.num_loader_warps
            + config.num_storer_warps
            + config.num_launcher_warps
            + config.num_controller_warps
        )
        assert total_warps == kernel.NUM_THREADS // 32

    def test_logical_grid_size(self):
        """Test logical grid size calculation for tiled matmul."""
        kernel = MatMulTileKernel(torch.float16, M=512, N=512, K=256)

        # Grid should be (M/TILE_M) * (N/TILE_N)
        expected_tiles = (512 // 64) * (512 // 64)  # 8 * 8 = 64
        actual_tiles = kernel.get_logical_grid_size(None, None, None, 512, 512, 256)

        assert actual_tiles == expected_tiles

    def test_smem_size_double_buffering(self):
        """Test shared memory size with double buffering."""
        kernel = MatMulTileKernel(torch.float16, M=512, N=512, K=256)

        # A tile: 64 * 32 * 2 = 4096 bytes
        # B tile: 32 * 64 * 2 = 4096 bytes
        # Total per stage: 8192 bytes
        # With 2 stages: 16384 bytes
        a_tile = 64 * 32 * 2
        b_tile = 32 * 64 * 2
        expected_smem = 2 * (a_tile + b_tile)

        assert kernel.smem_size_fwd == expected_smem

    def test_warp_role_via_config(self):
        """Test warp role classification via WarpConfig.get_warp_role()."""
        from machete.megakernel.scheduler import WarpRole

        kernel = MatMulTileKernel(torch.float16, M=512, N=512, K=256)
        config = kernel.warp_config

        # Consumer warps: 0-15
        assert config.get_warp_role(0) == WarpRole.CONSUMER
        assert config.get_warp_role(15) == WarpRole.CONSUMER

        # Loader warps: 16-17
        assert config.get_warp_role(16) == WarpRole.LOADER
        assert config.get_warp_role(17) == WarpRole.LOADER

        # Storer warp: 18
        assert config.get_warp_role(18) == WarpRole.STORER

        # Controller warp: 19
        assert config.get_warp_role(19) == WarpRole.CONTROLLER

    def test_kernel_signature(self):
        """Test kernel signature generation."""
        kernel = MatMulTileKernel(torch.float16, M=512, N=512, K=256)
        sig = kernel.get_kernel_signature()

        assert len(sig.tensors) == 3
        assert len(sig.scalars) == 3
        assert "M" in sig.scalars
        assert "N" in sig.scalars
        assert "K" in sig.scalars

    def test_same_kernel_different_modes(self):
        """Test that the same kernel methods work for both execution modes.

        The key insight is that load_forward/compute_forward/store_forward
        are the same methods - only the execution pattern differs based on
        uses_warp_specialization.
        """

        class DualModeKernel(MacheteKernel):
            """Kernel that can run in either mode based on configuration."""

            def __init__(self, use_warp_spec: bool):
                self._use_warp_spec = use_warp_spec

            @property
            def uses_warp_specialization(self) -> bool:
                return self._use_warp_spec

            @property
            def smem_size_fwd(self) -> int:
                return 1024

            def load_forward(self, logical_idx, smem, *args):
                pass  # Same implementation

            def compute_forward(self, logical_idx, smem, *args):
                pass  # Same implementation

            def store_forward(self, logical_idx, smem, *args):
                pass  # Same implementation

        # Sequential mode
        seq_kernel = DualModeKernel(use_warp_spec=False)
        assert seq_kernel.uses_warp_specialization is False
        assert seq_kernel.NUM_STAGES == 1

        # Warp-specialized mode
        ws_kernel = DualModeKernel(use_warp_spec=True)
        assert ws_kernel.uses_warp_specialization is True

        # Both have the same methods
        assert hasattr(seq_kernel, "load_forward")
        assert hasattr(ws_kernel, "load_forward")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
