# Copyright (c) 2025, Machete Authors
import torch
import torch.nn.functional as F
from machete.utils.testing import benchmark_op, clear_kernel_caches
from machete.kernels.gated_mlp.triton_impl import gated_mlp_triton
from machete.kernels.gated_mlp import swiglu_mlp_func, geglu_mlp_func


def swiglu_pytorch(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """PyTorch reference SwiGLU MLP."""
    k_dim = weight.shape[0]
    n2_dim = weight.shape[1]
    n_dim = n2_dim // 2

    w_gate = weight[:, ::2]
    w_up = weight[:, 1::2]

    orig_shape = x.shape
    x_flat = x.reshape(-1, k_dim)

    h_gate = x_flat @ w_gate
    h_up = x_flat @ w_up

    out = F.silu(h_gate) * h_up
    return out.view(*orig_shape[:-1], n_dim)


def geglu_pytorch(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """PyTorch reference GeGLU MLP."""
    k_dim = weight.shape[0]
    n2_dim = weight.shape[1]
    n_dim = n2_dim // 2

    w_gate = weight[:, ::2]
    w_up = weight[:, 1::2]

    orig_shape = x.shape
    x_flat = x.reshape(-1, k_dim)

    h_gate = x_flat @ w_gate
    h_up = x_flat @ w_up

    out = F.gelu(h_gate, approximate="tanh") * h_up
    return out.view(*orig_shape[:-1], n_dim)


def main():
    device = "cuda"
    dtype = torch.float16

    # --- SwiGLU Benchmark ---
    def get_configs():
        # (M, K, N) configurations
        # M = batch * seq, K = d_model, N = d_intermediate (output is N, weight is K x 2N)
        configs = [
            (1024, 1024, 2048),    # Small model
            (2048, 2048, 4096),    # Medium model
            (4096, 4096, 8192),    # Large model (LLaMA-7B like)
            (8192, 4096, 11008),   # LLaMA-7B exact
            (16384, 4096, 11008),  # LLaMA-7B with larger batch
        ]

        for m_dim, k_dim, n_dim in configs:
            name = f"M={m_dim} K={k_dim} N={n_dim}"
            try:
                x = torch.randn(m_dim, k_dim, device=device, dtype=dtype)
                # Weight is (K, 2*N) interleaved
                weight = torch.randn(k_dim, 2 * n_dim, device=device, dtype=dtype) * 0.02
                yield name, (x, weight)
                del x, weight
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                yield name, None
                torch.cuda.empty_cache()
                break
            except Exception as e:
                print(f"Skipping {name} due to error: {e}")
                break

    # SwiGLU Forward
    swiglu_op_map = {
        "PyTorch": lambda x, w: swiglu_pytorch(x, w),
        "Triton": lambda x, w: gated_mlp_triton(x, w, activation="silu"),
        "cuteDSL": lambda x, w: swiglu_mlp_func(x, w),
    }

    def numel_provider(args):
        x, weight = args
        m_dim = x.shape[0]
        k_dim = x.shape[1]
        n_dim = weight.shape[1] // 2
        # Memory transfers: x (M*K) + weight (K*2N) + output (M*N)
        return m_dim * k_dim + k_dim * 2 * n_dim + m_dim * n_dim

    benchmark_op("SwiGLU MLP Forward", get_configs(), swiglu_op_map, numel_provider)

    clear_kernel_caches()

    # GeGLU Forward
    geglu_op_map = {
        "PyTorch": lambda x, w: geglu_pytorch(x, w),
        "Triton": lambda x, w: gated_mlp_triton(x, w, activation="gelu"),
        "cuteDSL": lambda x, w: geglu_mlp_func(x, w),
    }

    benchmark_op("GeGLU MLP Forward", get_configs(), geglu_op_map, numel_provider)

    clear_kernel_caches()


if __name__ == "__main__":
    main()
