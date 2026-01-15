# Copyright (c) 2025, Machete Authors
import torch
import torch.nn.functional as functional
import pytest
from machete.kernels.gated_linear import geglu_func, swiglu_func, reglu_func
from machete.utils.testing import verify_kernel


def geglu_ref(a, b):
    # GeGLU: GELU(a) * b
    # Using torch's approximate GELU (tanh) to match kernel
    return functional.gelu(a, approximate="tanh") * b


def swiglu_ref(a, b):
    # SwiGLU: SiLU(a) * b
    return functional.silu(a) * b


def reglu_ref(a, b):
    # ReGLU: ReLU(a) * b
    return functional.relu(a) * b


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "b_dim, s_dim, d_dim",
    [
        (1, 1, 1024),  # Minimum size
        (2, 64, 2048),  # Fits in one tile (TILE_N=2048)
        (1, 128, 2049),  # Spans across tiles (irregular)
        (4, 256, 4096),  # Multiple tiles
        (1, 1, 8192),  # Large N
        (2, 33, 1535),  # Irregular everything
    ],
)
@pytest.mark.parametrize(
    "op_name, func, ref_func",
    [
        ("GeGLU", geglu_func, geglu_ref),
        ("SwiGLU", swiglu_func, swiglu_ref),
        ("ReGLU", reglu_func, reglu_ref),
    ],
)
def test_gated_linear(b_dim, s_dim, d_dim, op_name, func, ref_func, dtype):
    torch.manual_seed(0)
    device = "cuda"

    print(f"\nTesting {op_name} B={b_dim}, S={s_dim}, D={d_dim}, dtype={dtype}")

    # Inputs
    a = torch.randn((b_dim, s_dim, d_dim), device=device, dtype=dtype)
    b_tensor = torch.randn((b_dim, s_dim, d_dim), device=device, dtype=dtype)

    # 1. Compute Golden Reference in FP32
    a_fp32 = a.float()
    b_fp32 = b_tensor.float()
    out_fp32 = ref_func(a_fp32, b_fp32)

    # 2. Compute Reference in Target Precision
    out_low = ref_func(a, b_tensor)

    # 3. Baseline Quantization Error
    baseline_diff = (out_fp32.to(dtype) - out_low).abs().max().item()
    print(f"  Baseline error (FP32 vs {dtype}): {baseline_diff:.6f}")

    # 4. Set tolerance: 5x baseline error (floor at 1e-3) to handle bfloat16 grad accumulation
    atol = max(5.0 * baseline_diff, 1e-3)
    print(f"  Setting verif atol = {atol:.6f}")

    # 5. Verify Kernel
    # Inputs must be new leaves for grad check
    a_in = a.clone().detach().requires_grad_(True)
    b_in = b_tensor.clone().detach().requires_grad_(True)

    verify_kernel(
        name=f"{op_name}_{dtype}_{b_dim}_{s_dim}_{d_dim}",
        func=func,
        ref_func=ref_func,
        inputs=(a_in, b_in),
        dtype=dtype,
        atol=atol,
        check_grad=True,
    )


if __name__ == "__main__":
    pytest.main([__file__])
