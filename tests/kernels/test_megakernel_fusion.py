import torch
from machete.megakernel.core import Megakernel
from machete.kernels.gated_linear.sm80 import GatedLinearSM80
from machete.kernels.rope.sm80 import RopeSM80
from machete.kernels.gated_linear import GatedLinear as GatedLinearOp


def rope_ref(Q, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, S, 1, D)
    sin = sin.unsqueeze(0).unsqueeze(2)
    half = Q.shape[-1] // 2
    RH_Q = torch.cat((-Q[..., half:], Q[..., :half]), dim=-1)
    return Q * cos + RH_Q * sin


def get_config():
    device = "cuda"
    dtype = torch.float16
    B, S, H, D = 1, 128, 4, 64
    return device, dtype, B, S, H, D


def test_megakernel_fusion_forward():
    print("\n--- Testing Megakernel Forward Fusion (RoPE + GatedLinear) ---")
    device, dtype, B, S, H, D = get_config()

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
    barrier = torch.zeros(1, device=device, dtype=torch.int32)

    # 1. Reference Implementation (PyTorch / Sequential)
    q_ref = q_leaf.clone()
    q_ref_rope = rope_ref(q_ref, cos, sin)
    # Reshape for GatedLinear (B, S, H, D) -> (B*S, H*D)
    q_ref_flat = q_ref_rope.view(-1, H * D)
    gate_ref_flat = gate_leaf.view(-1, H * D)

    # GatedLinear: GL(a, b). For us 'a' comes from RoPE, 'b' is gate.
    # Ref GatedLinear logic (silu): a * silu(b) ?? No, GatedLinear usually is (a * act(b)) or (act(a) * b)?
    # Machete GatedLinear(a, b) -> c.
    # Logic in kernel: gate = act(a). out = gate * b.
    # Wait, check GatedLinear logic.
    # compute_forward(ma, mb, mc).
    # val_a = ma... val_b = mb...
    # gate = act(val_a).
    # out = gate * val_b.
    # So 'a' is the gated/act input, 'b' is the other input.
    # In RoPE+GL fusion benchmark:
    # rope(q, ...).
    # gl(q_view, gate).
    # So 'a' is q_view (output of rope), 'b' is gate.
    # So out = act(RoPE(q)) * gate.

    # Ref logic:
    import torch.nn.functional as F

    act = F.silu(q_ref_flat)
    out_ref_flat = act * gate_ref_flat

    # 2. Megakernel Implementation
    rope_impl = RopeSM80(dtype, head_dim=D)
    gl_impl = GatedLinearSM80(dtype, act_type="silu")

    # We must operate on 'q' in place for RoPE.
    # But for test correctness, we pass 'q' which is clone of leaf.

    mk = Megakernel(name="test_fwd_fusion", mode="forward")
    # Add RoPE: modifies 'q' in place.
    # Arguments: q_flat, cos, sin, seq_len
    q_flat = q.view(-1, H, D)
    # Cos/Sin for RoPE kernel must be flattened?
    # RopeSM80.__call__ flattens them. But kernel takes (S, D).
    # Inspecting benchmark_megakernel.py:
    # mk.add(rope_impl, q.view(-1, h, d), cos, sin, s)
    # 'cos' is (S, D). 'sin' is (S, D). 's' is scalar.
    mk.add(rope_impl, q_flat, cos, sin, S)

    # Add GatedLinear: reads 'q', 'gate'. writes 'out'.
    # Arguments: a, b, c, n_cols
    # a = q.view(-1, H*D)
    # b = gate.view(-1, H*D)
    # c = out.view(-1, H*D)
    mk.add(gl_impl, q.view(-1, H * D), gate.view(-1, H * D), out_mk.view(-1, H * D), H * D)

    grid = [B * S, 1, 1]
    block = [256, 1, 1]  # 256 threads

    mk.launch(barrier, B * S, grid, block)

    # Compare
    # out_mk is (B, S, H*D)
    out_ref = out_ref_flat.view(B, S, H * D)

    diff = (out_mk - out_ref).abs().max().item()
    print(f"Forward Max Diff: {diff}")
    assert torch.allclose(out_mk, out_ref, atol=1e-2, rtol=1e-2)
    print("Forward Pass Passed!")


def test_megakernel_fusion_backward():
    print("\n--- Testing Megakernel Backward Fusion (RoPE + GatedLinear) ---")
    device, dtype, B, S, H, D = get_config()

    # Setup inputs
    # Need to reproduce forward state first to have 'q' modified if needed?
    # Or just setup backward inputs.

    # Backward flow:
    # dOut (gradient of output)
    # -> GatedLinear Backward -> computes dA (grad of RoPE output), dB (grad of gate)
    # -> RoPE Backward -> computes dQ (grad of RoPE input) from dA

    # We need inputs 'a' and 'b' for GatedLinear backward.
    # 'a' is the input to GatedLinear forward, which is RoPE output.
    # 'b' is the gate.

    # Let's generate random 'a' (simulating RoPE output) and 'b'.
    a_leaf = torch.randn(B, S, H * D, device=device, dtype=dtype, requires_grad=True)
    b_leaf = torch.randn(B, S, H * D, device=device, dtype=dtype, requires_grad=True)

    d_out = torch.randn(B, S, H * D, device=device, dtype=dtype)

    # Reference Backward
    # Forward: out = act(a) * b
    # Backward:
    # dout = d_out_flat
    # d_b = dout * act(a)
    # d_a = dout * b * act'(a)

    import torch.nn.functional as F

    a_flat = a_leaf.view(-1, H * D)
    b_flat = b_leaf.view(-1, H * D)
    d_out_flat = d_out.view(-1, H * D)

    with torch.enable_grad():
        out_ref = F.silu(a_flat) * b_flat
        out_ref.backward(d_out_flat)

    d_a_ref = a_leaf.grad.view(B, S, H, D)  # Gradient w.r.t GL input (RoPE output)
    d_b_ref = b_leaf.grad.view(B, S, H * D)

    # Reference RoPE Backward
    # dQ = RoPE_bwd(d_a_ref)
    # Ref logic: RoPE(d_a_ref, cos, -sin)

    cos = torch.randn(S, D, device=device, dtype=dtype).clamp(-1, 1).view(S, D)
    sin = torch.randn(S, D, device=device, dtype=dtype).clamp(-1, 1).view(S, D)
    half = D // 2
    cos[:, half:] = cos[:, :half]
    sin[:, half:] = sin[:, :half]

    d_q_ref = rope_ref(d_a_ref, cos, -sin)

    # Megakernel Backward
    # We need buffers for d_a, d_b, d_q?
    # GL Backward writes d_a and d_b.
    # RoPE Backward reads d_a (as 'mq') and updates it in-place to become d_q.

    # So we can share buffer for d_a and d_q?
    # Yes, RoPE works in place.

    d_a_mk = torch.empty(B, S, H * D, device=device, dtype=dtype)
    d_b_mk = torch.empty(B, S, H * D, device=device, dtype=dtype)

    mk = Megakernel(name="test_bwd_fusion", mode="backward")

    rope_impl = RopeSM80(dtype, head_dim=D)
    gl_impl = GatedLinearSM80(dtype, act_type="silu")

    # Add Ops in order: GL Backward then RoPE Backward

    # 1. GL Backward
    # Args: dc, a, b, da, db, n_cols
    # dc = d_out
    # a = a_leaf (RoPE output from fwd)
    # b = b_leaf
    # da = d_a_mk (output of GL bwd, input to RoPE bwd)
    # db = d_b_mk
    mk.add(
        gl_impl,
        d_out.view(-1, H * D),
        a_leaf.view(-1, H * D),
        b_leaf.view(-1, H * D),
        d_a_mk.view(-1, H * D),
        d_b_mk.view(-1, H * D),
        H * D,
    )

    # 2. RoPE Backward
    # Args: smem, mq, cos, sin, seq_len
    # mq = d_a_mk (which is now viewed as (S, H, D)?? No, (B*S, H, D))
    # It modifies mq in place to produce d_q.
    mk.add(rope_impl, d_a_mk.view(-1, H, D), cos, sin, S)

    barrier = torch.zeros(1, device=device, dtype=torch.int32)
    grid = [B * S, 1, 1]
    block = [256, 1, 1]

    mk.launch(barrier, B * S, grid, block)

    # Check results
    # d_b_mk should match d_b_ref
    diff_b = (d_b_mk - d_b_ref).abs().max().item()
    print(f"Backward GL Grad_B Max Diff: {diff_b}")
    assert torch.allclose(d_b_mk, d_b_ref, atol=1e-2, rtol=1e-2)

    # d_a_mk (which is now d_q) should match d_q_ref
    d_q_mk = d_a_mk.view(B, S, H, D)
    diff_q = (d_q_mk - d_q_ref).abs().max().item()
    print(f"Backward RoPE Grad_Q Max Diff: {diff_q}")
    assert torch.allclose(d_q_mk, d_q_ref, atol=1e-2, rtol=1e-2)

    print("Backward Pass Passed!")


if __name__ == "__main__":
    test_megakernel_fusion_forward()
    test_megakernel_fusion_backward()
