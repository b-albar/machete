# Copyright (c) 2025, Machete Authors
import torch
import torch.nn.functional as F
from machete.kernels.activation import silu_func
from machete.kernels.gated_linear import swiglu_func


def test_activations():
    torch.manual_seed(0)
    device = "cuda"

    for dtype in [torch.float16, torch.bfloat16]:
        print(f"\nTesting SiLU/SwiGLU for dtype: {dtype}")
        B, S, D = 2, 128, 1024

        # Test SiLU
        print("Testing SiLU...")
        x = torch.randn((B, S, D), device=device, dtype=dtype, requires_grad=True)
        y_ref = F.silu(x)
        y_out = silu_func(x)

        diff_fwd = (y_out - y_ref).abs().max().item()
        print(f"SiLU Forward Max Diff: {diff_fwd}")

        dy = torch.randn_like(y_out)
        y_ref.backward(dy)
        dx_ref = x.grad.clone()

        x.grad = None
        y_out.backward(dy)
        dx_out = x.grad.clone()

        diff_bwd = (dx_out - dx_ref).abs().max().item()
        print(f"SiLU Backward Max Diff: {diff_bwd}")

        atol = 1e-2 if dtype == torch.float16 else 5e-2
        assert diff_fwd < atol
        assert diff_bwd < atol

        # Test SwiGLU
        print("Testing SwiGLU...")
        a = torch.randn((B, S, D), device=device, dtype=dtype, requires_grad=True)
        b = torch.randn((B, S, D), device=device, dtype=dtype, requires_grad=True)

        c_ref = F.silu(a) * b
        c_out = swiglu_func(a, b)

        diff_fwd = (c_out - c_ref).abs().max().item()
        print(f"SwiGLU Forward Max Diff: {diff_fwd}")

        dc = torch.randn_like(c_out)
        c_ref.backward(dc)
        da_ref, db_ref = a.grad.clone(), b.grad.clone()

        a.grad, b.grad = None, None
        c_out.backward(dc)
        da_out, db_out = a.grad.clone(), b.grad.clone()

        diff_da = (da_out - da_ref).abs().max().item()
        diff_db = (db_out - db_ref).abs().max().item()

        print(f"SwiGLU da Max Diff: {diff_da}")
        print(f"SwiGLU db Max Diff: {diff_db}")

        atol = 2e-2 if dtype == torch.float16 else 1e-1
        assert diff_fwd < atol
        assert diff_da < atol
        assert diff_db < atol

        print(f"Passed for {dtype}!")


if __name__ == "__main__":
    test_activations()
