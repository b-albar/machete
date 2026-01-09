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
    dtype = torch.float16

    B, S, H, D = 2, 128, 4, 64
    Q = torch.randn((B, S, H, D), device=device, dtype=dtype)

    # Create symmetric Cos/Sin
    Cos = torch.randn((S, D), device=device, dtype=dtype)
    Sin = torch.randn((S, D), device=device, dtype=dtype)
    half = D // 2
    Cos[:, half:] = Cos[:, :half]
    Sin[:, half:] = Sin[:, :half]

    print("Running Ref...")
    Q_ref = rope_ref(Q.clone(), Cos.clone(), Sin.clone())

    print("Running Kernel...")
    Q_out = Q.clone()
    rope = Rope(dtype=dtype, head_dim=D)
    rope(Q_out, Cos, Sin, backward=False)

    diff = (Q_out - Q_ref).abs().max().item()
    print("Max diff:", diff)

    if diff > 1e-3:
        print("Mismatch indices:")
        idx = (Q_out - Q_ref).abs() > 1e-3
        print(torch.nonzero(idx))

    assert torch.allclose(Q_out, Q_ref, atol=1e-3, rtol=1e-3)
    print("Passed!")


if __name__ == "__main__":
    test_rope()
