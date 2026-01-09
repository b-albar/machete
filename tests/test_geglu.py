# Copyright (c) 2025, Machete Authors
import torch
import torch.nn.functional as F
from machete.kernels.gated_linear import geglu_func


def gelu_ref(x):
    # Liger-Kernel uses tanh approximation
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


import math


def geglu_ref(a, b):
    # Liger-Kernel GeGLU: GELU(a) * b
    # Using torch's approximate GELU
    return F.gelu(a, approximate="tanh") * b


def test_geglu():
    torch.manual_seed(0)
    device = "cuda"

    for dtype in [torch.float16, torch.bfloat16]:
        print(f"\nTesting GeGLU for dtype: {dtype}")
        B, S, D = 2, 128, 1024
        a = torch.randn((B, S, D), device=device, dtype=dtype, requires_grad=True)
        b = torch.randn((B, S, D), device=device, dtype=dtype, requires_grad=True)

        # Forward
        print("Running Ref Forward...")
        c_ref = geglu_ref(a, b)

        print("Running Kernel Forward...")
        c_out = geglu_func(a, b)

        diff_fwd = (c_out - c_ref).abs().max().item()
        print(f"Forward Max Diff: {diff_fwd}")

        # Backward
        print("Running Backward...")
        dc = torch.randn_like(c_out)

        c_ref.backward(dc)
        da_ref, db_ref = a.grad.clone(), b.grad.clone()

        a.grad, b.grad = None, None
        c_out.backward(dc)
        da_out, db_out = a.grad.clone(), b.grad.clone()

        diff_da = (da_out - da_ref).abs().max().item()
        diff_db = (db_out - db_ref).abs().max().item()

        print(f"da Max Diff: {diff_da}")
        print(f"db Max Diff: {diff_db}")

        atol = 1e-1 if dtype == torch.bfloat16 else 1e-2
        assert diff_fwd < atol, f"Forward mismatch for {dtype}"
        assert diff_da < atol, f"da mismatch for {dtype}"
        assert diff_db < atol, f"db mismatch for {dtype}"

        print(f"Passed for {dtype}!")


if __name__ == "__main__":
    test_geglu()
