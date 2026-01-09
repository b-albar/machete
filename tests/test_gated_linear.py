# Copyright (c) 2025, Machete Authors
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
        for name, func, ref_func in ops:
            b, s, d = 2, 128, 1024
            a = torch.randn((b, s, d), device=device, dtype=dtype, requires_grad=True)
            b_tensor = torch.randn((b, s, d), device=device, dtype=dtype, requires_grad=True)
            verify_kernel(name, func, ref_func, (a, b_tensor), dtype)


if __name__ == "__main__":
    test_gated_linear()
