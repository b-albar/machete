import torch
from machete.megakernel.core import Megakernel
from machete.kernels.gated_linear.sm80 import GatedLinearSM80
from machete.kernels.rope.sm80 import RopeSM80
from quack.cute_dsl_utils import torch2cute_dtype_map


def test_megakernel_fuse_gated_linear():
    device = "cuda"
    dtype = torch.float16
    n_rows = 128
    n_cols = 256

    a1 = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
    b1 = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
    c1 = torch.empty(n_rows, n_cols, device=device, dtype=dtype)

    b2 = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
    c2 = torch.empty(n_rows, n_cols, device=device, dtype=dtype)

    barrier = torch.zeros(1, device=device, dtype=torch.int32)

    megakernel = Megakernel("forward_fuse", mode="forward")
    cute_dtype = torch2cute_dtype_map[dtype]

    op1_impl = GatedLinearSM80(dtype, "silu")
    op2_impl = GatedLinearSM80(dtype, "gelu")

    megakernel.add(op1_impl, a1, b1, c1, n_cols)
    megakernel.add(op2_impl, c1, b2, c2, n_cols)

    grid = [n_rows, 1, 1]
    block = [128, 1, 1]
    megakernel.launch(barrier, grid[0], grid, block)

    ref_c1 = torch.nn.functional.silu(a1.float()) * b1.float()
    ref_c2 = torch.nn.functional.gelu(ref_c1, approximate="tanh") * b2.float()

    torch.testing.assert_close(c2.float(), ref_c2, atol=1e-3, rtol=1e-3)
    print("Megakernel Forward fusion test passed!")


def test_megakernel_with_rope():
    device = "cuda"
    dtype = torch.float16
    b_sz, s_sz, h_sz, d_sz = 2, 64, 8, 128

    q_tensor = torch.randn(b_sz, s_sz, h_sz, d_sz, device=device, dtype=dtype)
    cos_tensor = torch.randn(s_sz, d_sz, device=device, dtype=dtype)
    sin_tensor = torch.randn(s_sz, d_sz, device=device, dtype=dtype)

    # We apply RoPE then a GatedLinear (contrived but tests fusion)
    # GatedLinear will act on the flattened B*S, H*D
    flat_q = q_tensor.view(b_sz * s_sz, h_sz * d_sz)
    gate = torch.randn(b_sz * s_sz, h_sz * d_sz, device=device, dtype=dtype)
    out = torch.empty(b_sz * s_sz, h_sz * d_sz, device=device, dtype=dtype)

    barrier = torch.zeros(1, device=device, dtype=torch.int32)
    megakernel = Megakernel("rope_fuse", mode="forward")
    cute_dtype = torch2cute_dtype_map[dtype]

    rope_impl = RopeSM80(dtype, d_sz)
    gl_impl = GatedLinearSM80(dtype, "silu")

    megakernel.add(rope_impl, q_tensor.view(b_sz * s_sz, h_sz, d_sz), cos_tensor, sin_tensor, s_sz)
    megakernel.add(gl_impl, flat_q, gate, out, h_sz * d_sz)

    grid = [b_sz * s_sz, 1, 1]
    block = [256, 1, 1]
    megakernel.launch(barrier, grid[0], grid, block)

    print("Megakernel with RoPE fusion test passed!")


def test_megakernel_backward():
    device = "cuda"
    dtype = torch.float16
    n_rows = 128
    n_cols = 256

    dc_grad = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
    a_in = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
    b_in = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
    da_out = torch.empty(n_rows, n_cols, device=device, dtype=dtype)
    db_out = torch.empty(n_rows, n_cols, device=device, dtype=dtype)

    barrier = torch.zeros(1, device=device, dtype=torch.int32)
    megakernel = Megakernel("backward_fuse", mode="backward")
    cute_dtype = torch2cute_dtype_map[dtype]

    op_impl = GatedLinearSM80(dtype, "silu")
    megakernel.add(op_impl, dc_grad, a_in, b_in, da_out, db_out, n_cols)

    grid = [n_rows, 1, 1]
    block = [128, 1, 1]
    megakernel.launch(barrier, grid[0], grid, block)

    # Reference
    with torch.enable_grad():
        a_reg = a_in.float().clone().detach().requires_grad_(True)
        b_reg = b_in.float().clone().detach().requires_grad_(True)
        out_fwd = torch.nn.functional.silu(a_reg) * b_reg
        out_fwd.backward(dc_grad.float())
        ref_da = a_reg.grad
        ref_db = b_reg.grad

    torch.testing.assert_close(da_out.float(), ref_da, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(db_out.float(), ref_db, atol=1e-3, rtol=1e-3)
    print("Megakernel Backward test passed!")


if __name__ == "__main__":
    test_megakernel_fuse_gated_linear()
    test_megakernel_with_rope()
    test_megakernel_backward()
