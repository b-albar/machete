# Copyright (c) 2025, Machete Authors
import argparse
import torch
import torch.nn.functional as F
from machete.kernels.rms_norm.sm120 import RMSNormSM120, rms_norm_sm120
from machete.utils.testing import benchmark_op

try:
    from quack.rmsnorm import rmsnorm as quack_rmsnorm
    from quack.rmsnorm import rmsnorm_bwd as quack_rmsnorm_bwd
    from quack.rmsnorm import rmsnorm_fwd as quack_rmsnorm_fwd

    QUACK_AVAILABLE = True
except ImportError:
    QUACK_AVAILABLE = False


def rmsnorm_pytorch(x, weight, eps=1e-6):
    return F.rms_norm(x, (x.shape[-1],), weight, eps=eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="fwd", choices=["fwd", "bwd"])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    device = "cuda"
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    mode = args.mode

    def get_configs():
        # Batch sizes (M) and Hidden dims (N)
        Ms = [1, 32, 128, 512, 1024, 2048, 4096, 8192]
        Ns = [128, 768, 1024, 2048, 4096, 8192, 16384]

        for N in Ns:
            for M in Ms:
                name = f"M={M}, N={N}"
                try:
                    x = torch.randn(M, N, device=device, dtype=dtype)
                    weight = torch.randn(N, device=device, dtype=dtype)
                    eps = 1e-6
                    yield name, (x, weight, eps)
                    del x, weight
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    yield name, None
                    torch.cuda.empty_cache()
                    break
                except Exception as e:
                    print(f"Skipping {name} due to error: {e}")
                    torch.cuda.empty_cache()
                    continue

    if mode == "fwd":

        def machete_fwd(x, weight, eps):
            return rms_norm_sm120(x, weight, eps)

        def quack_fwd_wrapper(x, weight, eps):
            return quack_rmsnorm(x, weight, eps=eps)

        def numel_provider(args):
            M, N = args[0].shape
            # Read x, Read weight, Write out
            return 2 * M * N + N

        ops = {
            "PyTorch": rmsnorm_pytorch,
            "Machete": machete_fwd,
        }
        if QUACK_AVAILABLE:
            ops["Quack"] = quack_fwd_wrapper

        benchmark_op(
            f"RMSNorm Forward ({args.dtype})",
            get_configs(),
            ops,
            numel_provider,
        )

    else:  # Backward
        # We need to wrap backward because benchmark_op expects a function that takes inputs
        # and we need to do the forward pass and dout generation inside or outside.
        # benchmark_op's loop will call the function with args from get_configs.

        # To avoid re-running forward too many times in do_bench:
        # Actually benchmark_op passes the SAME args tuple to all providers.

        machete_kernels = {}  # Cache kernel instances by N

        def machete_bwd(x, weight, eps):
            N = x.shape[-1]
            if N not in machete_kernels:
                machete_kernels[N] = RMSNormSM120(dtype, N)

            # Need dout and potentially intermediate states
            # Since we want to benchmark the RAW kernel performance, we should ideally
            # have dout pre-generated.
            # But benchmark_op doesn't easily allow passing extra state per call without
            # modifying get_configs.

            # For backward, we'll generate dout once per config if we were smart.
            # Here we just generate it inside for simplicity, but it might add overhead.
            # Or we can modify get_configs to yield (x, weight, eps, dout).
            pass

        # Let's redefine get_configs for backward
        def get_configs_bwd():
            Ms = [1, 32, 128, 512, 1024, 2048, 4096, 8192]
            Ns = [128, 768, 1024, 2048, 4096, 8192, 16384]

            for N in Ns:
                for M in Ms:
                    name = f"M={M}, N={N}"
                    try:
                        x = torch.randn(M, N, device=device, dtype=dtype)
                        weight = torch.randn(N, device=device, dtype=dtype)
                        eps = 1e-6
                        dout = torch.randn(M, N, device=device, dtype=dtype)

                        # For Quack we also need rstd
                        rstd = None
                        if QUACK_AVAILABLE:
                            _, _, rstd = quack_rmsnorm_fwd(x, weight, eps=eps, store_rstd=True)

                        yield name, (x, weight, eps, dout, rstd)
                        del x, weight, dout, rstd
                        torch.cuda.empty_cache()
                    except torch.cuda.OutOfMemoryError:
                        yield name, None
                        torch.cuda.empty_cache()
                        break
                    except Exception:
                        torch.cuda.empty_cache()
                        continue

        def machete_bwd_final(x, weight, eps, dout, rstd):
            N = x.shape[-1]
            if N not in machete_kernels:
                machete_kernels[N] = RMSNormSM120(dtype, N)
            return machete_kernels[N].backward(dout, x, weight, eps)

        def pytorch_bwd_final(x, weight, eps, dout, rstd):
            # We must use F.rms_norm and then backward
            x_req = x.detach().requires_grad_(True)
            w_req = weight.detach().requires_grad_(True) if weight is not None else None
            out = F.rms_norm(x_req, (N,), w_req, eps=eps)
            out.backward(dout)
            return x_req.grad

        def quack_bwd_final(x, weight, eps, dout, rstd):
            return quack_rmsnorm_bwd(x, weight, dout, rstd)

        def numel_provider_bwd(args):
            M, N = args[0].shape
            # Read dout, x, weight, rstd(M). Write dx, dw.
            return 3 * M * N + M + 2 * N

        ops_bwd = {
            "PyTorch": pytorch_bwd_final,
            "Machete": machete_bwd_final,
        }
        if QUACK_AVAILABLE:
            ops_bwd["Quack"] = quack_bwd_final

        benchmark_op(
            f"RMSNorm Backward ({args.dtype})",
            get_configs_bwd(),
            ops_bwd,
            numel_provider_bwd,
        )


if __name__ == "__main__":
    main()
