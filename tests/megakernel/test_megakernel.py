import torch
from machete.megakernel.core import Megakernel
from machete.kernels.gated_linear.sm80 import GatedLinearSM80
from machete.kernels.rope.sm80 import RopeSM80
from quack.cute_dsl_utils import torch2cute_dtype_map


# Helper refs
def swiglu_ref(a, b):
    return torch.nn.functional.silu(a) * b


def geglu_ref(a, b):
    # approximate="tanh" for consistency with typical kernels
    return torch.nn.functional.gelu(a, approximate="tanh") * b


def rope_ref(Q, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    half = Q.shape[-1] // 2
    RH_Q = torch.cat((-Q[..., half:], Q[..., :half]), dim=-1)
    return Q * cos + RH_Q * sin


def get_atol(baseline_diff):
    return max(2.0 * baseline_diff, 1e-3)


def test_megakernel_fuse_gated_linear():
    device = "cuda"
    n_rows = 128
    n_cols = 256

    for dtype in [torch.float16, torch.bfloat16]:
        print(f"\nTesting Fuse Gated Linear {dtype}")
        a1 = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
        b1 = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
        c1 = torch.empty(n_rows, n_cols, device=device, dtype=dtype)
        b2 = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
        c2 = torch.empty(n_rows, n_cols, device=device, dtype=dtype)

        # 1. FP32 Baseline
        a1_f = a1.float()
        b1_f = b1.float()
        b2_f = b2.float()
        ref_c1_f = swiglu_ref(a1_f, b1_f)
        ref_c2_f = geglu_ref(ref_c1_f, b2_f)

        # 2. Target dtype ref
        ref_c1 = swiglu_ref(a1, b1)
        ref_c2 = geglu_ref(ref_c1, b2)

        baseline_diff = (ref_c2_f.to(dtype) - ref_c2).abs().max().item()
        atol = get_atol(baseline_diff)
        print(f"  Baseline diff: {baseline_diff:.6f}, atol: {atol:.6f}")

        # 3. Kernel
        megakernel = Megakernel(f"forward_fuse_{dtype}", mode="forward")
        op1_impl = GatedLinearSM80(dtype, "silu")
        op2_impl = GatedLinearSM80(dtype, "gelu")
        megakernel.add(op1_impl, a1, b1, c1, n_cols)
        megakernel.add(op2_impl, c1, b2, c2, n_cols)

        # Must match grid/block requirements. GatedLinear uses TILE_N typically.
        # But here manual grid launch used in test.
        # GatedLinearSM80.grid_fn does calculations.
        # We manually launch here for simplicity?
        # Re-using test's manual grid.
        grid = [n_rows, 1, 1]
        block = [128, 1, 1]
        # Barrier internal
        megakernel.launch(grid[0], grid, block)

        diff = (c2 - ref_c2).abs().max().item()
        print(f"  Kernel diff: {diff:.6f}")
        assert diff <= atol, f"Mismatch: {diff} > {atol}"
        print("  Passed!")


def test_megakernel_with_rope():
    device = "cuda"
    b_sz, s_sz, h_sz, d_sz = 2, 64, 8, 128

    for dtype in [torch.float16, torch.bfloat16]:
        print(f"\nTesting Fuse RoPE + Gated {dtype}")
        q_tensor = torch.randn(b_sz, s_sz, h_sz, d_sz, device=device, dtype=dtype)
        cos_tensor = torch.randn(s_sz, d_sz, device=device, dtype=dtype).clamp(-1, 1)
        sin_tensor = torch.randn(s_sz, d_sz, device=device, dtype=dtype).clamp(-1, 1)
        half = d_sz // 2
        cos_tensor[:, half:] = cos_tensor[:, :half]
        sin_tensor[:, half:] = sin_tensor[:, :half]

        flat_q = q_tensor.view(b_sz * s_sz, h_sz * d_sz)
        gate = torch.randn(b_sz * s_sz, h_sz * d_sz, device=device, dtype=dtype)
        out = torch.empty(b_sz * s_sz, h_sz * d_sz, device=device, dtype=dtype)

        # 1. FP32 Ref
        q_f = q_tensor.float()
        cos_f = cos_tensor.float()
        sin_f = sin_tensor.float()
        gate_f = gate.float()

        # RoPE
        rope_out_f = rope_ref(q_f, cos_f, sin_f)
        # Flatten
        rope_flat_f = rope_out_f.view(b_sz * s_sz, h_sz * d_sz)
        # Gated Linear (SiLU * Gate)
        final_out_f = swiglu_ref(rope_flat_f, gate_f)

        # 2. Target Ref
        q_ref = q_tensor.clone()
        rope_out_ref = rope_ref(q_ref, cos_tensor, sin_tensor)
        rope_flat_ref = rope_out_ref.view(b_sz * s_sz, h_sz * d_sz)
        final_out_ref = swiglu_ref(rope_flat_ref, gate)

        baseline_diff = (final_out_f.to(dtype) - final_out_ref).abs().max().item()
        atol = get_atol(baseline_diff)
        print(f"  Baseline diff: {baseline_diff:.6f}, atol: {atol:.6f}")

        # 3. Kernel
        megakernel = Megakernel(f"rope_fuse_{dtype}", mode="forward")
        rope_impl = RopeSM80(dtype, d_sz)
        gl_impl = GatedLinearSM80(dtype, "silu")

        # Kernel modifies q_input in place
        q_input = q_tensor.clone()
        flat_q_in = q_input.view(b_sz * s_sz, h_sz * d_sz)

        megakernel.add(rope_impl, q_input.view(b_sz * s_sz, h_sz, d_sz), cos_tensor, sin_tensor, s_sz)
        megakernel.add(gl_impl, flat_q_in, gate, out, h_sz * d_sz)

        grid = [b_sz * s_sz, 1, 1]
        block = [256, 1, 1]
        # Barrier internal
        megakernel.launch(grid[0], grid, block)

        diff = (out - final_out_ref).abs().max().item()
        print(f"  Kernel diff: {diff:.6f}")
        assert diff <= atol, f"Mismatch: {diff} > {atol}"
        print("  Passed!")


def test_megakernel_backward():
    device = "cuda"
    n_rows = 128
    n_cols = 256

    for dtype in [torch.float16, torch.bfloat16]:
        print(f"\nTesting Backward {dtype}")
        dc_grad = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
        a_in = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
        b_in = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
        da_out = torch.empty(n_rows, n_cols, device=device, dtype=dtype)
        db_out = torch.empty(n_rows, n_cols, device=device, dtype=dtype)

        # 1. FP32 Ref
        dc_f = dc_grad.float()
        a_f = a_in.float().requires_grad_(True)
        b_f = b_in.float().requires_grad_(True)
        out_f = swiglu_ref(a_f, b_f)
        out_f.backward(dc_f)
        ref_da_f = a_f.grad
        ref_db_f = b_f.grad

        # 2. Target Ref
        a_t = a_in.clone().detach().requires_grad_(True)
        b_t = b_in.clone().detach().requires_grad_(True)
        out_t = swiglu_ref(a_t, b_t)
        out_t.backward(dc_grad)
        ref_da_t = a_t.grad
        ref_db_t = b_t.grad

        base_da = (ref_da_f.to(dtype) - ref_da_t).abs().max().item()
        base_db = (ref_db_f.to(dtype) - ref_db_t).abs().max().item()
        baseline_diff = max(base_da, base_db)
        atol = get_atol(baseline_diff)
        print(f"  Baseline diff: {baseline_diff:.6f}, atol: {atol:.6f}")

        # 3. Kernel
        megakernel = Megakernel(f"backward_fuse_{dtype}", mode="backward")
        op_impl = GatedLinearSM80(dtype, "silu")
        megakernel.add(op_impl, dc_grad, a_in, b_in, da_out, db_out, n_cols)

        grid = [n_rows, 1, 1]
        block = [128, 1, 1]
        # Barrier internal
        megakernel.launch(grid[0], grid, block)

        diff_da = (da_out - ref_da_t).abs().max().item()
        diff_db = (db_out - ref_db_t).abs().max().item()
        diff = max(diff_da, diff_db)
        print(f"  Kernel diff: {diff:.6f}")
        assert diff <= atol, f"Mismatch: {diff} > {atol}"
        print("  Passed!")


if __name__ == "__main__":
    test_megakernel_fuse_gated_linear()
    test_megakernel_with_rope()
    test_megakernel_backward()
