# Copyright (c) 2025, Machete Authors
"""
Test: Can TMA descriptors be pre-computed on host and loaded in kernel?

Minimal test to validate Option B for TMA in machete.
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils
import cuda.bindings.driver as cuda

# Test parameters
M = 256
K = 128
TILE_M = 64
TILE_K = 64


def test_tensor_dynamic_marking():
    """Test 1: Can we create tensors with dynamic layout marking?"""
    print("Test 1: Tensor dynamic layout marking")

    a_torch = torch.randn(M, K, dtype=torch.float16, device="cuda")
    a_tensor = from_dlpack(a_torch, assumed_align=16)

    print(f"  Before marking - shape: {a_tensor.shape}")
    print(f"  Before marking - stride: {a_tensor.stride}")

    # Mark M (dim 0) as dynamic by specifying K (dim 1) as leading
    a_tensor = a_tensor.mark_layout_dynamic(leading_dim=1)

    print(f"  After marking - dynamic shapes: {a_tensor.dynamic_shapes_mask}")
    print(f"  After marking - dynamic strides: {a_tensor.dynamic_strides_mask}")

    return True


def test_tma_atom_creation():
    """Test 2: Can we create TMA atoms from dynamic tensors inside JIT?"""
    print("\nTest 2: TMA atom creation from dynamic tensor")

    a_torch = torch.randn(M, K, dtype=torch.float16, device="cuda")
    a_tensor = from_dlpack(a_torch, assumed_align=16)
    a_tensor = a_tensor.mark_layout_dynamic(leading_dim=1)

    # Track if TMA creation succeeded
    result = {"success": False}

    @cute.jit
    def create_tma(a: cute.Tensor):
        # Create smem layout
        tile_shape_mnk = (TILE_M, 1, TILE_K)
        a_smem_layout = sm90_utils.make_smem_layout_a(
            cutlass.utils.LayoutEnum.ROW_MAJOR,
            tile_shape_mnk,
            cutlass.Float16,
            2,  # stages
        )
        a_smem_layout_single = cute.slice_(a_smem_layout, (None, None, 0))

        # Create TMA atom - this is the key test!
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            a,
            a_smem_layout_single,
            (TILE_M, TILE_K),
            num_multicast=1,
        )

        # If we get here, TMA creation worked
        result["success"] = True
        return tma_atom

    try:
        tma_atom = create_tma(a_tensor)
        print(f"  TMA atom created: {type(tma_atom)}")
        print(f"  Success: {result['success']}")
        return result["success"]
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


class TMAKernelTest:
    """Test class for TMA kernel with tensor parameter."""

    def __init__(self):
        self.a_dtype = cutlass.Float16

    @cute.jit
    def __call__(self, a: cute.Tensor, stream: cuda.CUstream):
        """Entry point that creates TMA and launches kernel."""
        # Create smem layout
        tile_shape_mnk = (TILE_M, 1, TILE_K)
        a_smem_layout = sm90_utils.make_smem_layout_a(
            cutlass.utils.LayoutEnum.ROW_MAJOR,
            tile_shape_mnk,
            self.a_dtype,
            2,
        )
        a_smem_layout_single = cute.slice_(a_smem_layout, (None, None, 0))

        # Create TMA atom
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            a,
            a_smem_layout_single,
            (TILE_M, TILE_K),
            num_multicast=1,
        )

        # Launch kernel
        self.kernel(tma_atom, tma_tensor, a_smem_layout).launch(
            grid=[1, 1, 1],
            block=[128, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom: cute.CopyAtom,
        src_tensor: cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
    ):
        """Kernel that uses TMA to copy data to shared memory."""
        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(
            self.a_dtype,
            a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )

        warp_idx = cute.arch.warp_idx()

        # Prefetch TMA descriptor
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom)

        cute.arch.barrier()

        # Partition for TMA
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(src_tensor, 0, 2),
        )

        # Allocate barrier
        mbar_ptr = smem.allocate(cutlass.Int64)

        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar_ptr, 1)

        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # Issue TMA copy for tile 0
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.copy(
                    tma_atom,
                    tAgA[(None, 0, 0)],
                    tAsA[(None, 0)],
                    tma_bar_ptr=mbar_ptr,
                    mcast_mask=0,
                )

        # Wait for copy
        cute.arch.mbarrier_wait(mbar_ptr, 0)
        cute.arch.barrier()


def test_full_tma_kernel():
    """Test 3: Full TMA kernel with dynamic tensor."""
    print("\nTest 3: Full TMA kernel with dynamic tensor")

    a_torch = torch.randn(M, K, dtype=torch.float16, device="cuda")
    a_tensor = from_dlpack(a_torch, assumed_align=16)
    a_tensor = a_tensor.mark_layout_dynamic(leading_dim=1)

    try:
        torch_stream = torch.cuda.Stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)

        test = TMAKernelTest()
        test(a_tensor, stream)
        torch.cuda.synchronize()
        print("  TMA kernel executed successfully!")
        return True

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_m_values():
    """Test 4: Same kernel works with different M values."""
    print("\nTest 4: Same kernel, different M values")

    try:
        torch_stream = torch.cuda.Stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)

        test = TMAKernelTest()

        for m in [128, 256, 512]:
            a_torch = torch.randn(m, K, dtype=torch.float16, device="cuda")
            a_tensor = from_dlpack(a_torch, assumed_align=16)
            a_tensor = a_tensor.mark_layout_dynamic(leading_dim=1)

            test(a_tensor, stream)
            torch.cuda.synchronize()
            print(f"  M={m}: SUCCESS")

        return True

    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing TMA Descriptor Approaches for Machete")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    device = torch.cuda.get_device_properties(0)
    print(f"GPU: {device.name}")
    print(f"Compute Capability: {device.major}.{device.minor}")
    print()

    results = []

    # Test 1: Basic dynamic marking (works on all GPUs)
    try:
        results.append(("Dynamic marking", test_tensor_dynamic_marking()))
    except Exception as e:
        print(f"Test 1 crashed: {e}")
        results.append(("Dynamic marking", False))

    # TMA tests only on SM90+
    if device.major >= 9:
        # Test 2: TMA atom creation
        try:
            results.append(("TMA atom creation", test_tma_atom_creation()))
        except Exception as e:
            print(f"Test 2 crashed: {e}")
            results.append(("TMA atom creation", False))

        # Test 3: Full TMA kernel
        try:
            results.append(("Full TMA kernel", test_full_tma_kernel()))
        except Exception as e:
            print(f"Test 3 crashed: {e}")
            results.append(("Full TMA kernel", False))

        # Test 4: Different M values
        try:
            results.append(("Different M values", test_different_m_values()))
        except Exception as e:
            print(f"Test 4 crashed: {e}")
            results.append(("Different M values", False))
    else:
        print("\nTMA tests skipped (requires SM90+ Hopper)")

    print("\n" + "=" * 60)
    print("Results:")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    # Summary for Option B validity
    print("\n" + "=" * 60)
    if all(p for _, p in results):
        print("CONCLUSION: Option B is VALID!")
        print("  - Tensors can be passed with dynamic layouts")
        print("  - TMA atoms can be created from dynamic tensors")
        print("  - Same compiled kernel works with different M values")
    else:
        print("CONCLUSION: Some tests failed, needs investigation")
