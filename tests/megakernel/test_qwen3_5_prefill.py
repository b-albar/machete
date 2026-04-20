import pytest
import torch

from benchmarks.kernels.benchmark_qwen3_5_decode import HIDDEN, allocate_model_weights
from benchmarks.kernels.benchmark_qwen3_5_layer import megakernel_forward_build, sequential_forward
from benchmarks.kernels.benchmark_qwen3_5_prefill import megakernel_prefill_build


def _single_kernel_layer_chain_prefill(x, residual, weights, num_layers):
    """Reference full-prefill composition using existing single-layer megakernels."""
    x_cur = x
    res_cur = residual
    for i in range(num_layers):
        pfx = f"layer.{i}"
        args = (
            x_cur,
            res_cur,
            weights[f"{pfx}.attn_norm"],
            weights[f"{pfx}.W_q"],
            weights[f"{pfx}.W_k"],
            weights[f"{pfx}.W_v"],
            weights[f"{pfx}.w_q_norm"],
            weights[f"{pfx}.w_k_norm"],
            weights["cos"],
            weights["sin"],
            weights[f"{pfx}.W_o"],
            weights[f"{pfx}.mlp_norm"],
            weights[f"{pfx}.W_gate_up"],
            weights[f"{pfx}.W_down"],
        )
        _, _, x_cur, res_cur = megakernel_forward_build(
            x.shape[0], x.shape[1], *args, page_size=32768
        )

    residual_final = x_cur + res_cur
    variance = residual_final.float().pow(2).mean(-1, keepdim=True)
    h_final = (
        residual_final.float()
        * torch.rsqrt(variance + 1e-6)
        * weights["final_norm"].float()
    ).to(residual_final.dtype)
    logits = torch.matmul(h_final, weights["lm_head"].t())
    return logits, residual_final


def _sequential_full_prefill(x, residual, weights, num_layers):
    x_cur = x
    res_cur = residual
    for i in range(num_layers):
        pfx = f"layer.{i}"
        x_cur, res_cur = sequential_forward(
            x_cur,
            res_cur,
            weights[f"{pfx}.attn_norm"],
            weights[f"{pfx}.W_q"],
            weights[f"{pfx}.W_k"],
            weights[f"{pfx}.W_v"],
            weights[f"{pfx}.w_q_norm"],
            weights[f"{pfx}.w_k_norm"],
            weights["cos"],
            weights["sin"],
            weights[f"{pfx}.W_o"],
            weights[f"{pfx}.mlp_norm"],
            weights[f"{pfx}.W_gate_up"],
            weights[f"{pfx}.W_down"],
        )

    residual_final = x_cur + res_cur
    variance = residual_final.float().pow(2).mean(-1, keepdim=True)
    h_final = (
        residual_final.float()
        * torch.rsqrt(variance + 1e-6)
        * weights["final_norm"].float()
    ).to(residual_final.dtype)
    logits = torch.matmul(h_final, weights["lm_head"].t())
    return logits, residual_final


@torch.inference_mode()
def test_full_prefill_matches_single_kernel_layer_chain_1layer():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    device = "cuda"
    batch, seq_len, num_layers = 1, 128, 1

    weights = allocate_model_weights(dtype=dtype, device=device)
    x = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    residual = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device)

    _, logits_mk, residual_mk = megakernel_prefill_build(
        batch, seq_len, x, residual, weights, page_size=32768, num_layers=num_layers
    )
    logits_ref, residual_ref = _sequential_full_prefill(
        x, residual, weights, num_layers
    )

    torch.testing.assert_close(residual_mk.float(), residual_ref.float(), atol=3e-1, rtol=2e-2)
    torch.testing.assert_close(logits_mk.float(), logits_ref.float(), atol=2e-1, rtol=2e-2)


@torch.inference_mode()
def test_full_prefill_matches_single_kernel_layer_chain_2layers():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    device = "cuda"
    batch, seq_len, num_layers = 1, 128, 2

    weights = allocate_model_weights(dtype=dtype, device=device)
    x = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device)
    residual = torch.randn(batch, seq_len, HIDDEN, dtype=dtype, device=device)

    _, logits_mk, residual_mk = megakernel_prefill_build(
        batch, seq_len, x, residual, weights, page_size=32768, num_layers=num_layers
    )
    logits_ref, residual_ref = _sequential_full_prefill(
        x, residual, weights, num_layers
    )

    torch.testing.assert_close(residual_mk.float(), residual_ref.float(), atol=3e-1, rtol=2e-2)
    torch.testing.assert_close(logits_mk.float(), logits_ref.float(), atol=2e-1, rtol=2e-2)
