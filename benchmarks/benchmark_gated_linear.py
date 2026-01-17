# Copyright (c) 2025, Machete Authors
import torch
import torch.nn.functional as functional
import triton
import triton.language as tl
from machete.kernels.gated_linear import swiglu_func
from machete.utils.testing import benchmark_op

# --- Triton Reference Kernels ---


@triton.jit
def _swiglu_forward_kernel(a, b, c, stride, n_cols: tl.constexpr, block_size: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)
    row_offset = program_id * stride
    a += row_offset
    b += row_offset
    c += row_offset

    col_offsets = tl.arange(0, block_size)
    mask = col_offsets < n_cols
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0).to(tl.float32)

    # SiLU(a) * b
    c_row = (tl.sigmoid(a_row) * a_row) * b_row
    tl.store(c + col_offsets, c_row.to(c.dtype.element_ty), mask=mask)


def swiglu_triton_fwd(a, b):
    n_rows, n_cols = a.shape[0], a.shape[1]
    c = torch.empty_like(a)
    block_size = triton.next_power_of_2(n_cols)
    _swiglu_forward_kernel[(n_rows,)](a, b, c, a.stride(0), n_cols, block_size)
    return c


@triton.jit
def _swiglu_backward_kernel(da, db, dc, a, b, stride, n_cols: tl.constexpr, block_size: tl.constexpr):
    program_id = tl.program_id(0).to(tl.int64)
    row_offset = program_id * stride
    da += row_offset
    db += row_offset
    dc += row_offset
    a += row_offset
    b += row_offset

    col_offsets = tl.arange(0, block_size)
    mask = col_offsets < n_cols

    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0).to(tl.float32)
    dc_row = tl.load(dc + col_offsets, mask=mask, other=0).to(tl.float32)

    # SiLU(a) = a * sigmoid(a)
    sig_a = tl.sigmoid(a_row)
    silu_a = a_row * sig_a

    # db = dc * silu(a)
    # da = dc * b * d/da(silu(a))
    # d/da(silu(a)) = sigmoid(a) * (1 + a * (1 - sigmoid(a)))
    db_row = dc_row * silu_a
    da_row = dc_row * b_row * (sig_a * (1.0 + a_row * (1.0 - sig_a)))

    tl.store(da + col_offsets, da_row.to(da.dtype.element_ty), mask=mask)
    tl.store(db + col_offsets, db_row.to(db.dtype.element_ty), mask=mask)


def swiglu_triton_bwd(a, b, dy):
    n_rows, n_cols = a.shape[0], a.shape[1]
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    block_size = triton.next_power_of_2(n_cols)
    _swiglu_backward_kernel[(n_rows,)](da, db, dy, a, b, a.stride(0), n_cols, block_size)
    return da, db


# --- Benchmarking Logic ---


def main():
    device = "cuda"
    dtype = torch.float16

    # --- Gated Benchmark (SwiGLU) ---
    def get_configs():
        batch_sizes_k = [1, 2, 4, 8, 16, 32, 64, 128]
        hidden_dims = [4096, 8192]

        for dim in hidden_dims:
            dim_name = f"{dim // 1024}k"
            for k in batch_sizes_k:
                batch_size = k * 1024
                name = f"{k}k x {dim_name}"
                try:
                    a = torch.randn(batch_size, dim, device=device, dtype=dtype, requires_grad=True)
                    b = torch.randn(batch_size, dim, device=device, dtype=dtype, requires_grad=True)
                    dy = torch.randn(batch_size, dim, device=device, dtype=dtype)
                    yield name, (a, b, dy)
                    # Explicit cleanup to help memory
                    del a, b, dy
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    yield name, None
                    torch.cuda.empty_cache()
                    break
                except Exception as e:
                    print(f"Skipping {name} due to error: {e}")
                    break

    # Forward
    forward_op_map = {
        "PyTorch": lambda a, b, dy: functional.silu(a) * b,
        "Triton": lambda a, b, dy: swiglu_triton_fwd(a, b),
        "cuteDSL": lambda a, b, dy: swiglu_func(a, b),
    }

    def fwd_numel_provider(args):
        # Forward transfers: a, b (in) -> c (out)
        return args[0].numel() * 3

    benchmark_op("SwiGLU Forward", get_configs(), forward_op_map, fwd_numel_provider)

    # Clear kernel caches between forward and backward benchmarks to free GPU memory
    from machete.utils.testing import clear_kernel_caches
    clear_kernel_caches()

    # Backward
    def pytorch_bwd(a, b, dy):
        a.grad = b.grad = None
        c = functional.silu(a) * b
        c.backward(dy)
        return a.grad, b.grad

    def cutedsl_bwd(a, b, dy):
        # Fair benchmark: Execute backward kernel directly without Forward recompute
        # Mock Context
        class MockCtx:
            def __init__(self, saved):
                self.saved_tensors = saved
                self.others = []
                self.layout = []

        # Get kernel from global LRU cache
        from machete.kernels.gated_linear import _get_kernel

        n_cols = a.shape[-1]
        a_flat = a.view(-1, n_cols)
        b_flat = b.view(-1, n_cols)
        n_rows = a_flat.shape[0]

        kernel = _get_kernel(a.dtype, "silu", n_rows, n_cols)

        ctx = MockCtx((a_flat, b_flat))
        kernel.run_backward(ctx, dy.view(-1, n_cols))
        return None

    backward_op_map = {
        "PyTorch": pytorch_bwd,
        "Triton": swiglu_triton_bwd,
        "cuteDSL": cutedsl_bwd,
    }

    def bwd_numel_provider(args):
        # Backward transfers: a, b, dy (in) -> da, db (out)
        return args[0].numel() * 5

    benchmark_op("SwiGLU Backward", get_configs(), backward_op_map, bwd_numel_provider)


if __name__ == "__main__":
    main()
