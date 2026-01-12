import torch
import sys
import os

from machete.kernels import Rope


def rope_ref(Q, cos, sin):
    # Q: (B, S, H, D)
    # cos, sin: (S, D)
    # expand cos/sin
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D)
    sin = sin.unsqueeze(0).unsqueeze(2)

    half = Q.shape[-1] // 2
    RH_Q = torch.cat((-Q[..., half:], Q[..., :half]), dim=-1)

    return Q * cos + RH_Q * sin


def test_rope():
    torch.manual_seed(0)
    device = "cuda"

    for dtype in [torch.float16, torch.bfloat16]:
        print(f"\nTesting dtype: {dtype}")
        B, S, H, D = 2, 128, 4, 64
        Q = torch.randn((B, S, H, D), device=device, dtype=dtype)

        # Create symmetric Cos/Sin
        # RoPE typically uses cos/sin from unit circle. Random values can be large.
        # Let's normalize or bound them to avoid numerical instability.
        Cos = torch.randn((S, D), device=device, dtype=dtype).clamp(-1, 1)
        Sin = torch.randn((S, D), device=device, dtype=dtype).clamp(-1, 1)
        # Or even better, actual cos/sin
        # theta = torch.randn((S, D), device=device, dtype=dtype)
        # Cos = torch.cos(theta)
        # Sin = torch.sin(theta)

        half = D // 2
        Cos[:, half:] = Cos[:, :half]
        Sin[:, half:] = Sin[:, :half]

        print("Running Ref...")
        Q_ref = rope_ref(Q.clone(), Cos.clone(), Sin.clone())

        print("Running Kernel...")
        # Use Autograd-friendly calling convention
        # We need a leaf that requires grad to test backward
        Q_leaf = Q.clone().detach().requires_grad_(True)
        # The kernel modifies input in-place. To avoid "leaf variable was used in an inplace operation" error
        # (if we passed Q_leaf directly and it modified it), we pass a clone.
        # But for correctness of the *kernel* output, we just capture the return.

        # Note: Our Rope kernel implementation (via SingleKernel) modifies the first argument in-place.
        # If we pass Q_leaf.clone(), that clone is modified.
        # Autograd tracks the operation.

        Q_input = Q_leaf.clone()
        rope = Rope(dtype=dtype, head_dim=D)
        # Call without backward=False
        Q_out = rope(Q_input, Cos, Sin)

        diff = (Q_out - Q_ref).abs().max().item()
        print("Max diff FWD:", diff)

        atol = 1e-1 if dtype == torch.bfloat16 else 1e-2
        rtol = 1e-1 if dtype == torch.bfloat16 else 1e-2

        if diff > atol:
            print(f"Mismatch indices FWD for {dtype}:")
            idx = (Q_out - Q_ref).abs() > atol
            print(torch.nonzero(idx))

        assert torch.allclose(Q_out, Q_ref, atol=atol, rtol=rtol)

        # --- Backward Test ---
        print("Running Backward...")
        # --- Backward Test ---
        print("Running Backward...")
        grad_out = torch.randn_like(Q_out)
        grad_out_clone = grad_out.clone()
        Q_out.backward(grad_out)

        # Reference Backward: RoPE(grad_out, cos, -sin)
        # Note: technically we should check math derivation, but for RoPE it's known property.
        # dL/dq = RoPE(dL/dy, cos, -sin)
        Q_grad_ref = rope_ref(grad_out_clone, Cos.clone(), -Sin.clone())

        Q_grad_kernel = Q_leaf.grad

        diff_bwd = (Q_grad_kernel - Q_grad_ref).abs().max().item()
        print("Max diff BWD:", diff_bwd)

        if diff_bwd > atol:
            print(f"Mismatch indices BWD for {dtype}:")
            idx = (Q_grad_kernel - Q_grad_ref).abs() > atol
            print(torch.nonzero(idx))

        assert torch.allclose(Q_grad_kernel, Q_grad_ref, atol=atol, rtol=rtol)

        print(f"Passed for {dtype}!")


if __name__ == "__main__":
    test_rope()
