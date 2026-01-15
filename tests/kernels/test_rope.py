import torch
import pytest
from machete.kernels import Rope
from machete.utils.testing import verify_kernel


def rope_ref(Q, cos, sin):
    # Q: (B, S, H, D)
    # cos, sin: (S, D)
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D)
    sin = sin.unsqueeze(0).unsqueeze(2)

    half = Q.shape[-1] // 2
    RH_Q = torch.cat((-Q[..., half:], Q[..., :half]), dim=-1)

    return Q * cos + RH_Q * sin


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "b, s, h, d",
    [
        (1, 128, 4, 64),
        (2, 64, 8, 128),
        (4, 1, 16, 64),  # Min sequence length
        (1, 1024, 1, 128),  # Single head
        (8, 512, 12, 96),  # Non-power-of-2 heads/dim
        (1, 2049, 8, 64),  # Irregular sequence length
        (2, 4096, 32, 128),  # Large size
    ],
)
def test_rope(b, s, h, d, dtype):
    torch.manual_seed(0)
    device = "cuda"

    print(f"\nTesting B={b}, S={s}, H={h}, D={d}, dtype={dtype}")

    # Inputs
    q = torch.randn((b, s, h, d), device=device, dtype=dtype)

    # Create symmetric Cos/Sin and normalize to avoid large values instability
    cos = torch.randn((s, d), device=device, dtype=dtype).clamp(-1, 1)
    sin = torch.randn((s, d), device=device, dtype=dtype).clamp(-1, 1)

    half = d // 2
    cos[:, half:] = cos[:, :half]
    sin[:, half:] = sin[:, :half]

    # 1. Compute Golden Reference in Float32 to determine baseline quantization error
    q_fp32 = q.float()
    cos_fp32 = cos.float()
    sin_fp32 = sin.float()
    q_ref_fp32 = rope_ref(q_fp32, cos_fp32, sin_fp32)

    # 2. Compute Reference in target dtype
    q_ref_low = rope_ref(q, cos, sin)

    # 3. Calculate baseline quantization error between FP32 and Low Precision
    baseline_diff = (q_ref_fp32.to(dtype) - q_ref_low).abs().max().item()
    print(f"  Baseline quantization error (FP32 vs {dtype}): {baseline_diff}")

    # 4. Set tolerance: Error must be at most 2x the baseline quantization error
    atol = max(2.5 * baseline_diff, 1e-3)
    print(f"  Setting verification atol = {atol}")

    # 5. Run Verification using Kernel
    rope_op = Rope(dtype=dtype, head_dim=d)

    def rope_wrapper(q_in, c, s_in):
        # Rope kernel modifies 'q' in-place.
        # We must clone it so verify_kernel logic (which compares against ref) remains valid.
        return rope_op(q_in.clone(), c, s_in)

    # We pass Q as leaf requiring grad for backward check
    inputs = (q.clone().detach().requires_grad_(True), cos.clone(), sin.clone())

    verify_kernel(
        name=f"Rope_{dtype}_{b}_{s}_{h}_{d}",
        func=rope_wrapper,
        ref_func=rope_ref,
        inputs=inputs,
        dtype=dtype,
        atol=atol,
        check_grad=True,
    )


if __name__ == "__main__":
    pytest.main([__file__])
