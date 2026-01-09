# Copyright (c) 2025, Machete Authors
import torch
import torch.nn as nn
import torch.nn.functional as F
from machete.utils.testing import benchmark_op, verify_kernel
from machete.kernels.gated_mlp.triton_impl import gated_mlp_triton
from machete.kernels.gated_mlp.sm80 import gated_mlp_sm80


class ReferenceGatedMLP(nn.Module):
    def __init__(self, d_model, d_intermediate, act_type="silu"):
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, 2 * d_intermediate, bias=False)
        self.act_type = act_type

    def forward(self, x):
        gate_up = self.w_gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        if self.act_type == "silu":
            return F.silu(gate) * up
        elif self.act_type == "gelu":
            return F.gelu(gate, approximate="tanh") * up
        return gate * up


def functional_gated_mlp(x, weight, act_type="silu"):
    # weight is (K, 2N)
    # x is (..., K)
    # x @ weight -> (..., 2N)
    gate_up = torch.matmul(x, weight)
    gate, up = gate_up.chunk(2, dim=-1)
    if act_type == "silu":
        return F.silu(gate) * up
    elif act_type == "gelu":
        return F.gelu(gate, approximate="tanh") * up
    return gate * up


def main():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    d_model = 4096
    d_intermediate = 11008

    ref_mlp = ReferenceGatedMLP(d_model, d_intermediate).to(device, dtype)
    # Prepare weight for kernel: (K, 2N)
    # nn.Linear weights are (Out, In) -> (2N, K)
    # We transpose to (K, 2N) and make contiguous
    weight = ref_mlp.w_gate_up.weight.detach().T.contiguous()

    configs = {
        "BS=1, S=2048": (torch.randn(1, 2048, d_model, device=device, dtype=dtype), weight),
        "BS=4, S=4096": (torch.randn(4, 4096, d_model, device=device, dtype=dtype), weight),
    }

    # Verify correctness first
    print("Verifying correctness...")
    x_test = torch.randn(2, 128, d_model, device=device, dtype=dtype)
    out_ref = ref_mlp(x_test)
    out_triton = gated_mlp_triton(x_test, weight)
    # Check max diff
    diff = (out_ref - out_triton).abs().max().item()
    print(f"Max difference (PyTorch vs Triton): {diff}")
    assert diff < 1e-2, "Triton implementation verification failed!"
    print("Verification passed!")

    op_map = {
        "PyTorch": lambda x, w: functional_gated_mlp(x, w),
        "Triton": gated_mlp_triton,
        "Ampere (SM80)": gated_mlp_sm80,
    }

    def numel_provider(args):
        x = args[0]
        # X: M*K, W: K*2N, Output: M*N
        # args[0] is x, args[1] is w
        M = x.numel() // x.shape[-1]
        K = x.shape[-1]
        N = d_intermediate
        # Read X (M*K), Read W (K*2N), Write Y (M*N)
        return M * K + K * 2 * N + M * N

    benchmark_op("GatedMLP Forward", configs, op_map, numel_provider)


if __name__ == "__main__":
    main()
