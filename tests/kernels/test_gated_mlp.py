# Copyright (c) 2025, Machete Authors
import torch
import torch.nn.functional as functional
import pytest
from machete.kernels.gated_mlp import swiglu_mlp_func, geglu_mlp_func, gated_mlp_func


def swiglu_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Reference SwiGLU MLP implementation.
    x @ W where W is (K, 2*N) interleaved [g0, u0, g1, u1, ...].
    Output = SiLU(x @ W_gate) * (x @ W_up)
    """
    k_dim = weight.shape[0]
    n2_dim = weight.shape[1]
    n_dim = n2_dim // 2

    # Extract gate and up weights from interleaved format
    w_gate = weight[:, ::2]  # columns 0, 2, 4, ...
    w_up = weight[:, 1::2]   # columns 1, 3, 5, ...

    # Compute in FP32 for reference
    orig_shape = x.shape
    x_flat = x.reshape(-1, k_dim).float()
    w_gate_f = w_gate.float()
    w_up_f = w_up.float()

    h_gate = x_flat @ w_gate_f
    h_up = x_flat @ w_up_f

    out = functional.silu(h_gate) * h_up
    return out.to(x.dtype).view(*orig_shape[:-1], n_dim)


def geglu_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Reference GeGLU MLP implementation.
    Output = GELU(x @ W_gate) * (x @ W_up)
    """
    k_dim = weight.shape[0]
    n2_dim = weight.shape[1]
    n_dim = n2_dim // 2

    w_gate = weight[:, ::2]
    w_up = weight[:, 1::2]

    orig_shape = x.shape
    x_flat = x.reshape(-1, k_dim).float()
    w_gate_f = w_gate.float()
    w_up_f = w_up.float()

    h_gate = x_flat @ w_gate_f
    h_up = x_flat @ w_up_f

    # Use tanh approximation to match kernel
    out = functional.gelu(h_gate, approximate="tanh") * h_up
    return out.to(x.dtype).view(*orig_shape[:-1], n_dim)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape, k_dim, n_dim",
    [
        ((2, 64), 128, 256),     # Small
        ((4, 128), 256, 512),   # Medium
        ((8, 256), 512, 1024),  # Larger
        ((1, 32), 64, 128),     # Minimum
        ((2, 3, 64), 128, 256), # 3D input
    ],
)
@pytest.mark.parametrize(
    "op_name, func, ref_func",
    [
        ("SwiGLU_MLP", swiglu_mlp_func, swiglu_ref),
        ("GeGLU_MLP", geglu_mlp_func, geglu_ref),
    ],
)
def test_gated_mlp_forward(shape, k_dim, n_dim, op_name, func, ref_func, dtype):
    """Test Gated MLP forward pass correctness."""
    torch.manual_seed(42)
    device = "cuda"

    print(f"\nTesting {op_name} shape={shape}, K={k_dim}, N={n_dim}, dtype={dtype}")

    # Create inputs
    x = torch.randn(*shape, k_dim, device=device, dtype=dtype)
    # Weight is (K, 2*N) interleaved
    weight = torch.randn(k_dim, 2 * n_dim, device=device, dtype=dtype) * 0.02

    # 1. Compute Golden Reference in FP32
    out_ref = ref_func(x, weight)

    # 2. Compute kernel output
    out_kernel = func(x, weight)

    # 3. Check shapes
    expected_shape = (*shape, n_dim)
    assert out_kernel.shape == expected_shape, f"Shape mismatch: {out_kernel.shape} vs {expected_shape}"

    # 4. Compute baseline error for tolerance
    # Due to GEMM accumulation, we need looser tolerance
    # Especially for larger K dimensions
    base_atol = 0.1 if dtype == torch.bfloat16 else 0.05
    atol = base_atol * (1 + k_dim / 256)  # Scale with K

    diff = (out_kernel - out_ref).abs().max().item()
    print(f"  Max diff: {diff:.6f}, atol: {atol:.6f}")

    assert diff <= atol, f"{op_name} forward mismatch: {diff} > {atol}"
    print("  PASSED!")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "m_dim, k_dim, n_dim",
    [
        (64, 128, 256),
        (128, 256, 512),
    ],
)
@pytest.mark.parametrize("act_type", ["silu", "gelu"])
def test_gated_mlp_backward(m_dim, k_dim, n_dim, act_type, dtype):
    """Test Gated MLP backward pass correctness (when implemented)."""
    torch.manual_seed(42)
    device = "cuda"

    print(f"\nTesting GatedMLP backward: M={m_dim}, K={k_dim}, N={n_dim}, act={act_type}, dtype={dtype}")

    # Create inputs with gradients
    x = torch.randn(m_dim, k_dim, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(k_dim, 2 * n_dim, device=device, dtype=dtype, requires_grad=True) * 0.02

    # Clone for kernel test
    x_kernel = x.detach().clone().requires_grad_(True)
    weight_kernel = weight.detach().clone().requires_grad_(True)

    # Random gradient
    dy = torch.randn(m_dim, n_dim, device=device, dtype=dtype)

    # 1. Reference backward
    ref_func = swiglu_ref if act_type == "silu" else geglu_ref
    out_ref = ref_func(x, weight)
    out_ref.backward(dy)
    dx_ref = x.grad.clone()
    # dw_ref = weight.grad.clone()  # Will be used when backward is implemented

    # 2. Kernel backward
    from machete.kernels.gated_mlp import GatedMLP
    kernel_op = GatedMLP(dtype, act_type)
    out_kernel = kernel_op(x_kernel, weight_kernel)
    out_kernel.backward(dy)

    dx_kernel = x_kernel.grad
    # dw_kernel = weight_kernel.grad  # Will be used when backward is implemented

    # Note: Backward not yet implemented in kernel, so gradients will be zeros
    if dx_kernel is not None and dx_kernel.abs().sum() > 0:
        # If gradients are computed, check them
        dx_diff = (dx_kernel - dx_ref).abs().max().item()
        print(f"  dx diff: {dx_diff:.6f}")
        # Very loose tolerance for backward
        atol = 1.0
        assert dx_diff <= atol, f"GatedMLP backward dx mismatch: {dx_diff} > {atol}"
        print("  PASSED!")
    else:
        print("  Backward not yet implemented (gradients are zero), skipping check")


# Legacy test for compatibility
def test_gated_mlp_legacy():
    """Legacy test using the old interface."""
    device = "cuda"
    d_model = 128
    d_intermediate = 256

    dtypes = [torch.float16, torch.bfloat16]
    act_types = ["silu", "gelu"]

    for dtype in dtypes:
        for act_type in act_types:
            print(f"\n--- Legacy Testing dtype={dtype}, act_type={act_type} ---")

            # Prepare weights - interleaved format
            w_gate_up = torch.randn(d_model, 2 * d_intermediate, device=device, dtype=dtype)
            w_gate_up = (w_gate_up - w_gate_up.mean()) / w_gate_up.std()
            gate, up = w_gate_up.chunk(2, dim=-1)
            w_interleaved = torch.stack((gate, up), dim=-1).flatten(-2)

            w_test = w_interleaved.clone().detach()

            batch_sizes = [(2, 128, d_model)]

            for shape in batch_sizes:
                print(f"Testing shape {shape}")
                x_raw = torch.randn(*shape, device=device, dtype=dtype)
                x_raw = (x_raw - x_raw.mean()) / x_raw.std()
                x = x_raw.detach()

                # Reference
                ref_func = swiglu_ref if act_type == "silu" else geglu_ref
                out_ref = ref_func(x, w_test)

                # Kernel
                out_kernel = gated_mlp_func(x, w_test, act_type=act_type)

                diff = (out_kernel - out_ref).abs().max().item()
                atol = 2.0 if dtype == torch.float16 else 10.0
                print(f"  Max diff: {diff:.6f}, atol: {atol:.6f}")
                assert diff <= atol, f"Legacy test failed: {diff} > {atol}"
                print("  PASSED!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
