import torch
import torch.nn.functional as functional
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


def test_gated_linear():
    torch.manual_seed(0)
    device = "cuda"

    ops = [
        ("GeGLU", geglu_func, geglu_ref),
        ("SwiGLU", swiglu_func, swiglu_ref),
        ("ReGLU", reglu_func, reglu_ref),
    ]

    for dtype in [torch.float16, torch.bfloat16]:
        print(f"\n{'=' * 20} Testing dtype: {dtype} {'=' * 20}")
        for name, func, ref_func in ops:
            print(f"-- Protocol for {name} --")
            b_dim, s_dim, d_dim = 2, 128, 1024

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

            # 4. Set tolerance: 2x baseline error (floor at 1e-3)
            atol = max(2.0 * baseline_diff, 1e-3)
            print(f"  Setting verif atol = {atol:.6f}")

            # 5. Verify Kernel
            # Inputs must be new leaves for grad check
            a_in = a.clone().detach().requires_grad_(True)
            b_in = b_tensor.clone().detach().requires_grad_(True)

            verify_kernel(
                name=f"{name}_{dtype}",
                func=func,
                ref_func=ref_func,
                inputs=(a_in, b_in),
                dtype=dtype,
                atol=atol,
                check_grad=True,
            )


if __name__ == "__main__":
    test_gated_linear()
