import torch
import torch.nn.functional as F
from machete.megakernel.core import Megakernel
from machete.kernels.gated_linear.sm80 import GatedLinearSM80
from machete.kernels.rope.sm80 import RopeSM80


def rope_ref(Q, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    half = Q.shape[-1] // 2
    RH_Q = torch.cat((-Q[..., half:], Q[..., :half]), dim=-1)
    return Q * cos + RH_Q * sin


def get_config():
    device = "cuda"
    B, S, H, D = 1, 128, 4, 64
    return device, B, S, H, D


def get_atol(baseline_diff):
    return max(2.0 * baseline_diff, 1e-3)


def test_megakernel_fusion_forward():
    print("\n--- Testing Megakernel Forward Fusion (RoPE + GatedLinear) ---")
    device, B, S, H, D = get_config()

    for dtype in [torch.float16, torch.bfloat16]:
        print(f"Testing dtype: {dtype}")

        # Inputs
        q_leaf = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=True)
        q = q_leaf.clone()
        cos = torch.randn(S, D, device=device, dtype=dtype).clamp(-1, 1)
        sin = torch.randn(S, D, device=device, dtype=dtype).clamp(-1, 1)
        half = D // 2
        cos[:, half:] = cos[:, :half]
        sin[:, half:] = sin[:, :half]

        gate_leaf = torch.randn(B, S, H * D, device=device, dtype=dtype, requires_grad=True)
        gate = gate_leaf.clone()

        out_mk = torch.empty(B, S, H * D, device=device, dtype=dtype)

        # 1. Reference (FP32 baseline calculation)
        def run_ref(q_in, g_in, c_in, s_in):
            r = rope_ref(q_in, c_in, s_in)
            r_flat = r.view(-1, H * D)
            g_flat = g_in.view(-1, H * D)
            return F.silu(r_flat) * g_flat

        ref_fp32 = run_ref(q.float(), gate.float(), cos.float(), sin.float())
        ref_low = run_ref(q, gate, cos, sin)

        baseline_diff = (ref_fp32.to(dtype) - ref_low).abs().max().item()
        atol = get_atol(baseline_diff)
        print(f"  Baseline error: {baseline_diff:.6f}, atol: {atol:.6f}")

        # 2. Megakernel
        rope_impl = RopeSM80(dtype, head_dim=D)
        gl_impl = GatedLinearSM80(dtype, act_type="silu")

        mk = Megakernel(name=f"test_fwd_fusion_{dtype}", mode="forward")

        # RoPE modifies q in-place
        # GatedLinear reads q (viewed flat), gate, writes to out_mk
        mk.add(rope_impl, q.view(-1, H, D), cos, sin, S)
        mk.add(gl_impl, q.view(-1, H * D), gate.view(-1, H * D), out_mk.view(-1, H * D), H * D)

        grid = [B * S, 1, 1]
        block = [256, 1, 1]
        # Barrier allocated internally
        mk.launch(B * S, grid, block)

        # Compare
        out_ref = ref_low.view(B, S, H * D)
        diff = (out_mk - out_ref).abs().max().item()
        print(f"  Max Diff: {diff:.6f}")
        assert diff <= atol, f"Forward mismatch: {diff} > {atol}"
        print("  Passed!")


def test_megakernel_fusion_backward():
    print("\n--- Testing Megakernel Backward Fusion (RoPE + GatedLinear) ---")
    device, B, S, H, D = get_config()

    for dtype in [torch.float16, torch.bfloat16]:
        print(f"Testing dtype: {dtype}")

        # Inputs (Simulation of Forward Output state)
        a_leaf = torch.randn(B, S, H * D, device=device, dtype=dtype, requires_grad=True)
        b_leaf = torch.randn(B, S, H * D, device=device, dtype=dtype, requires_grad=True)
        d_out = torch.randn(B, S, H * D, device=device, dtype=dtype)

        # RoPE cos/sin
        cos = torch.randn(S, D, device=device, dtype=dtype).clamp(-1, 1)
        sin = torch.randn(S, D, device=device, dtype=dtype).clamp(-1, 1)
        half = D // 2
        cos[:, half:] = cos[:, :half]
        sin[:, half:] = sin[:, :half]

        # 1. Reference
        def run_ref_bwd(a_in, b_in, dout_in, c_in, s_in):
            a_flat = a_in.view(-1, H * D)
            b_flat = b_in.view(-1, H * D)
            d_flat = dout_in.view(-1, H * D)

            # GL Backward
            # Need to recompute FWD to backprop
            fwd = F.silu(a_flat) * b_flat
            grad_a, grad_b = torch.autograd.grad(fwd, (a_flat, b_flat), d_flat)

            # RoPE Backward (on grad_a)
            grad_a_rs = grad_a.view(B, S, H, D)
            # RoPE Bwd is effectively RoPE(grad, cos, -sin)
            dq = rope_ref(grad_a_rs, c_in, -s_in)
            return grad_b.view(B, S, H * D), dq

        # FP32 Baseline
        db_fp32, dq_fp32 = run_ref_bwd(a_leaf.float(), b_leaf.float(), d_out.float(), cos.float(), sin.float())
        db_low, dq_low = run_ref_bwd(a_leaf, b_leaf, d_out, cos, sin)

        err_b = (db_fp32.to(dtype) - db_low).abs().max().item()
        err_q = (dq_fp32.to(dtype) - dq_low).abs().max().item()
        baseline_diff = max(err_b, err_q)
        atol = get_atol(baseline_diff)
        print(f"  Baseline error: {baseline_diff:.6f}, atol: {atol:.6f}")

        # 2. Megakernel
        d_a_mk = torch.empty(B, S, H * D, device=device, dtype=dtype)
        d_b_mk = torch.empty(B, S, H * D, device=device, dtype=dtype)

        mk = Megakernel(name=f"test_bwd_fusion_{dtype}", mode="backward")
        rope_impl = RopeSM80(dtype, head_dim=D)
        gl_impl = GatedLinearSM80(dtype, act_type="silu")

        # GL Backward: writes d_a_mk, d_b_mk
        mk.add(
            gl_impl,
            d_out.view(-1, H * D),
            a_leaf.view(-1, H * D),
            b_leaf.view(-1, H * D),
            d_a_mk.view(-1, H * D),
            d_b_mk.view(-1, H * D),
            H * D,
        )
        # RoPE Backward: reads d_a_mk, writes d_a_mk (in place)
        mk.add(rope_impl, d_a_mk.view(-1, H, D), cos, sin, S)

        grid = [B * S, 1, 1]
        block = [256, 1, 1]
        # Barrier allocated internally
        mk.launch(B * S, grid, block)

        # Compare
        diff_b = (d_b_mk - db_low).abs().max().item()
        d_q_mk = d_a_mk.view(B, S, H, D)
        diff_q = (d_q_mk - dq_low).abs().max().item()

        max_diff = max(diff_b, diff_q)
        print(f"  Max Diff: {max_diff:.6f}")
        assert max_diff <= atol, f"Backward mismatch: {max_diff} > {atol}"
        print("  Passed!")


if __name__ == "__main__":
    test_megakernel_fusion_forward()
    test_megakernel_fusion_backward()
