import sys
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from machete.utils.testing import verify_kernel
from machete.kernels.gated_mlp import gated_mlp_func


def test_gated_mlp():
    device = "cuda"
    d_model = 128
    d_intermediate = 256

    dtypes = [torch.float16, torch.bfloat16]
    act_types = ["silu", "gelu"]

    for dtype in dtypes:
        for act_type in act_types:
            print(f"\n--- Testing dtype={dtype}, act_type={act_type} ---")

            # Prepare weights
            # For correctness test, we want to ensure the weight layout matches what the kernel expects for the "interleaved" logic
            # The kernel expects (K, 2N) where adjacent columns are (gate, up).
            # We generate random weights same as before.
            w_gate_up = torch.randn(d_model, 2 * d_intermediate, device=device, dtype=dtype)
            # Normalize to mean 0, std 1
            w_gate_up = (w_gate_up - w_gate_up.mean()) / w_gate_up.std()
            gate, up = w_gate_up.chunk(2, dim=-1)
            # Make interleaved: [g0, u0, g1, u1, ...]
            # stack gives (K, N, 2), flatten gives (K, 2N)
            w_interleaved = torch.stack((gate, up), dim=-1).flatten(-2)

            w_test = w_interleaved.clone().detach().requires_grad_(True)

            batch_sizes = [(2, 128, d_model)]

            for shape in batch_sizes:
                print(f"Testing shape {shape}")
                # Generate x, then normalize to mean 0 std 1
                x_raw = torch.randn(*shape, device=device, dtype=dtype)
                x_raw = (x_raw - x_raw.mean()) / x_raw.std()
                x = x_raw.detach().requires_grad_(True)

                # Define specialized closures for the current loop iteration
                # We need to bind act_type immediately

                def make_ref(act):
                    def ref_func(x, w):
                        # Computations in FP32 for higher precision reference
                        x_32 = x.float()
                        w_32 = w.float()
                        gu = x_32 @ w_32
                        gu = gu.reshape(*gu.shape[:-1], -1, 2)
                        gate = gu[..., 0]
                        up = gu[..., 1]
                        if act == "silu":
                            return torch.nn.functional.silu(gate) * up
                        elif act == "gelu":
                            # Use tanh approximation to match widespread "GeGLU" definition in LLMs/Triton
                            return torch.nn.functional.gelu(gate, approximate="tanh") * up
                        return gate * up

                    return ref_func

                def make_kernel(act):
                    return lambda x, w: gated_mlp_func(x, w, act_type=act)

                # Rely on gated_mlp_func to dispatch to the correct kernel for the local GPU
                print(f"Testing GatedMLP on {torch.cuda.get_device_name(0)}")

                verify_kernel(
                    f"GatedMLP_{act_type}_{dtype}",
                    make_kernel(act_type),
                    make_ref(act_type),
                    (x, w_test),
                    dtype,
                    atol=2.0 if dtype == torch.float16 else 10.0,  # BF16 needs loose tolerance
                    check_grad=True,
                )


if __name__ == "__main__":
    test_gated_mlp()
