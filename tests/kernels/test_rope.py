import torch
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


def test_rope():
    torch.manual_seed(0)
    device = "cuda"

    for dtype in [torch.float16, torch.bfloat16]:
        print(f"\nTesting dtype: {dtype}")
        B, S, H, D = 2, 128, 4, 64

        # Inputs
        Q = torch.randn((B, S, H, D), device=device, dtype=dtype)

        # Create symmetric Cos/Sin and normalize to avoid large values instability
        Cos = torch.randn((S, D), device=device, dtype=dtype).clamp(-1, 1)
        Sin = torch.randn((S, D), device=device, dtype=dtype).clamp(-1, 1)

        half = D // 2
        Cos[:, half:] = Cos[:, :half]
        Sin[:, half:] = Sin[:, :half]

        # 1. Compute Golden Reference in Float32 to determine baseline quantization error
        Q_fp32 = Q.float()
        Cos_fp32 = Cos.float()
        Sin_fp32 = Sin.float()
        Q_ref_fp32 = rope_ref(Q_fp32, Cos_fp32, Sin_fp32)

        # 2. Compute Reference in target dtype
        Q_ref_low = rope_ref(Q, Cos, Sin)

        # 3. Calculate baseline quantization error between FP32 and Low Precision
        baseline_diff = (Q_ref_fp32.to(dtype) - Q_ref_low).abs().max().item()
        print(f"  Baseline quantization error (FP32 vs {dtype}): {baseline_diff}")

        # 4. Set tolerance: Error must be at most 2x the baseline quantization error
        # Use a small epsilon floor to prevent 0-tolerance if baseline is perfect (unlikely)
        atol = max(2.0 * baseline_diff, 1e-3)
        print(f"  Setting verification atol = {atol}")

        # 5. Run Verification using Kernel
        rope_op = Rope(dtype=dtype, head_dim=D)

        def rope_wrapper(q, c, s):
            # Rope kernel modifies 'q' in-place.
            # We must clone it so verify_kernel logic (which compares against ref) remains valid.
            # Cloning also ensures we don't modify the leaf variable Q directly.
            return rope_op(q.clone(), c, s)

        # We pass Q as leaf requiring grad for backward check
        inputs = (Q.clone().detach().requires_grad_(True), Cos.clone(), Sin.clone())

        verify_kernel(
            name=f"Rope_{dtype}",
            func=rope_wrapper,
            ref_func=rope_ref,
            inputs=inputs,
            dtype=dtype,
            atol=atol,
            check_grad=True,
        )


if __name__ == "__main__":
    test_rope()
