#!/usr/bin/env python
"""Check Machete Qwen3.5 decode against Luce from the same prefill state."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks.kernels import benchmark_qwen3_5_mxfp4_simt_decode as machete_decode
from machete.kernels.qwen_3_5 import QWEN3_5_LAYER_TYPES


def build_exact_prompt_ids(tokenizer, target_tokens: int) -> list[int]:
    text = (
        "Explain in great detail the history of artificial intelligence, "
        "machine learning, deep learning, and neural networks. "
    )
    ids: list[int] = []
    repeat = 1
    while len(ids) < target_tokens:
        ids = tokenizer.encode(text * repeat, add_special_tokens=False)
        repeat *= 2
    return ids[:target_tokens]


def make_machete_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        context_len=args.context_len,
        page_size=args.page_size,
        num_pages=args.num_pages,
        threads=args.threads,
        mma_reg_count=args.mma_reg_count,
        matvec_block=args.matvec_block,
        group_size=args.group_size,
        warmup=args.warmup,
        rep=args.rep,
        top_partitions=args.top_partitions,
        fa_num_splits=0,
        prefetch_gate_up=args.prefetch_gate_up,
        scheduler=args.scheduler,
        controller_waits=args.controller_waits,
        dummy_weights=False,
        compile_only=False,
        no_final=False,
        final_only=False,
        split_final=args.split_final,
        attention_buffer_dtype="fp32",
        max_layers=args.max_layers,
        trace=None,
        trace_perfetto=None,
    )


def snapshot_luce_state(decoder) -> dict[str, torch.Tensor]:
    return {
        "embed_weight": decoder._embed_weight,
        "fa_k_cache": decoder._fa_k_cache.clone(),
        "fa_v_cache": decoder._fa_v_cache.clone(),
        "dn_states": decoder._dn_states.clone(),
        "conv_bufs": decoder._conv_bufs.clone(),
    }


def copy_luce_state_to_machete(state: dict[str, torch.Tensor], buffers, first_token: int, context_len: int) -> None:
    (
        x,
        residual,
        k_cache,
        v_cache,
        q_buf,
        q_raw,
        kv_raw,
        q_gate,
        attn_out,
        norm,
        qkv,
        z,
        beta,
        alpha,
        dn_out,
        mlp,
        dn_state,
        conv,
        top_values,
        top_indices,
        top_partial_values,
        top_partial_indices,
    ) = buffers

    for tensor in x:
        tensor.zero_()
    for tensor in residual:
        tensor.zero_()
    for group in (q_buf, q_raw, kv_raw, q_gate, attn_out, norm, qkv, z, beta, alpha, dn_out, mlp):
        for tensor in group:
            tensor.zero_()
    top_values.zero_()
    top_indices.zero_()
    if top_partial_values is not None:
        top_partial_values.zero_()
    if top_partial_indices is not None:
        top_partial_indices.zero_()

    x[0][0, 0].copy_(state["embed_weight"][int(first_token)])

    fa_slot = 0
    dn_slot = 0
    for layer_idx, layer_type in enumerate(QWEN3_5_LAYER_TYPES):
        if layer_type == "full_attention":
            k_cache[layer_idx].zero_()
            v_cache[layer_idx].zero_()
            k_cache[layer_idx][0, :context_len].copy_(state["fa_k_cache"][fa_slot, :, :context_len, :].permute(1, 0, 2))
            v_cache[layer_idx][0, :context_len].copy_(state["fa_v_cache"][fa_slot, :, :context_len, :].permute(1, 0, 2))
            fa_slot += 1
        else:
            dn_state[dn_slot][0].copy_(state["dn_states"][dn_slot])
            conv[dn_slot][0].copy_(state["conv_bufs"][dn_slot])
            dn_slot += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--luce-dir", type=Path, default=Path("/home/elentir/Projets/lucebox-hub"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=32768)
    parser.add_argument("--num-pages", type=int, default=3)
    parser.add_argument("--threads", type=int, default=512)
    parser.add_argument("--mma-reg-count", type=int, default=96)
    parser.add_argument("--matvec-block", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--scheduler", choices=("default", "overlap", "overlap-adaptive"), default="overlap-adaptive")
    parser.add_argument(
        "--controller-waits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use controller wait formulas when overlap scheduling computes readiness.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rep", type=int, default=5)
    parser.add_argument(
        "--top-partitions",
        type=int,
        default=0,
        help="0 uses one partial top-1 partition per SM; negative disables partial top-1.",
    )
    parser.add_argument(
        "--prefetch-gate-up",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--initial-token", type=int, default=None, help="Skip prefill and decode this token from zero state.")
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--split-final", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.luce_dir / "megakernel"))
    from model_nvfp4 import Decoder

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    decoder = Decoder(model_name=args.model, backend="nvfp4", verbose=False)
    if args.initial_token is None:
        prompt_ids = build_exact_prompt_ids(tokenizer, args.context_len)
        ids_t = torch.tensor(prompt_ids, dtype=torch.int32, device="cuda")
        first_token = int(decoder.prefill_tokens(ids_t))
        pre_decode_state = snapshot_luce_state(decoder)
    else:
        prompt_ids = []
        first_token = int(args.initial_token)
        pre_decode_state = snapshot_luce_state(decoder)
    luce_next = int(decoder.step(first_token))
    torch.cuda.synchronize()

    machete_args = make_machete_args(args)
    if args.split_final:
        body_kernel, final_kernel = machete_decode.build_split_final_kernels(machete_args)
        machete_weights = body_kernel._keep_alive[0]
        buffers = body_kernel._keep_alive[1]
        kernels = (body_kernel, final_kernel)
    else:
        kernel = machete_decode.build_kernel(machete_args)
        machete_weights = kernel._keep_alive[0]
        buffers = kernel._keep_alive[1]
        kernels = (kernel,)
    copy_luce_state_to_machete(pre_decode_state, buffers, first_token, args.context_len)
    for kernel in kernels:
        kernel.compile()
    for kernel in kernels:
        kernel.run(validate=False)
    torch.cuda.synchronize()
    machete_next = int(buffers[19].cpu().item())

    matched = machete_next == luce_next
    print(f"prompt_tokens={len(prompt_ids)}")
    print(f"first_token={first_token}")
    print(f"luce_next={luce_next}")
    print(f"machete_next={machete_next}")
    print(f"match={matched}")
    print(f"top_partitions={args.top_partitions}")
    if args.debug:
        x, residual = buffers[0], buffers[1]
        top_values, top_indices = buffers[18], buffers[19]
        top_partial_values, top_partial_indices = buffers[20], buffers[21]
        print(f"machete_x0_norm={float(x[0].float().norm().cpu()):.6f}")
        print(f"machete_x24_norm={float(x[-1].float().norm().cpu()):.6f}")
        print(f"machete_residual24_norm={float(residual[-1].float().norm().cpu()):.6f}")
        print(f"machete_x24_nan={bool(torch.isnan(x[-1].float()).any().cpu())}")
        print(f"machete_residual24_nan={bool(torch.isnan(residual[-1].float()).any().cpu())}")
        print(f"machete_top_value={float(top_values.float().cpu().item()):.6f}")
        print(f"machete_top_index_raw={int(top_indices.cpu().item())}")
        if top_partial_values is not None:
            pv = top_partial_values[0, 0].float().detach().cpu()
            pi = top_partial_indices[0, 0].detach().cpu()
            best_p = int(torch.argmax(pv).item())
            preview = ",".join(f"{i}:{float(pv[i]):.4g}/{int(pi[i])}" for i in range(min(12, pv.numel())))
            print(f"machete_partial_best={best_p}:{float(pv[best_p]):.6f}/{int(pi[best_p])}")
            print(f"machete_partial_preview={preview}")
        beta, alpha = buffers[12], buffers[13]
        norm, qkv, z, dn_out = buffers[9], buffers[10], buffers[11], buffers[14]
        print("machete_beta_norms=" + ",".join(f"{i}:{float(beta[i].float().norm().cpu()):.4g}" for i in range(min(4, len(beta)))))
        print("machete_alpha_norms=" + ",".join(f"{i}:{float(alpha[i].float().norm().cpu()):.4g}" for i in range(min(4, len(alpha)))))
        print("machete_beta_maxabs=" + ",".join(f"{i}:{float(beta[i].float().abs().max().cpu()):.4g}" for i in range(min(4, len(beta)))))
        print("machete_alpha_maxabs=" + ",".join(f"{i}:{float(alpha[i].float().abs().max().cpu()):.4g}" for i in range(min(4, len(alpha)))))
        print("machete_dn_out_norms=" + ",".join(f"{i}:{float(dn_out[i].float().norm().cpu()):.4g}" for i in range(min(4, len(dn_out)))))
        print("machete_dn_out_maxabs=" + ",".join(f"{i}:{float(dn_out[i].float().abs().max().cpu()):.4g}" for i in range(min(4, len(dn_out)))))
        print("machete_qkv_maxabs=" + ",".join(f"{i}:{float(qkv[i].float().abs().max().cpu()):.4g}" for i in range(min(4, len(qkv)))))
        print("machete_z_maxabs=" + ",".join(f"{i}:{float(z[i].float().abs().max().cpu()):.4g}" for i in range(min(4, len(z)))))
        print("machete_norm_maxabs=" + ",".join(f"{i}:{float(norm[i].float().abs().max().cpu()):.4g}" for i in range(min(6, len(norm)))))
        for name in ("layer.0.W_qkv_mxfp4", "layer.1.W_qkv_mxfp4", "layer.2.W_qkv_mxfp4"):
            qw = machete_weights[name]
            print(f"{name}.scale_max={float(qw.scales.float().abs().max().cpu()):.6g} packed_max={int(qw.packed.max().cpu())}")
        print("luce_dn_state_norms=" + ",".join(f"{i}:{float(pre_decode_state['dn_states'][i].float().norm().cpu()):.4g}" for i in range(4)))
        print("luce_dn_state_maxabs=" + ",".join(f"{i}:{float(pre_decode_state['dn_states'][i].float().abs().max().cpu()):.4g}" for i in range(4)))
        print("luce_conv_norms=" + ",".join(f"{i}:{float(pre_decode_state['conv_bufs'][i].float().norm().cpu()):.4g}" for i in range(4)))
        print("luce_conv_maxabs=" + ",".join(f"{i}:{float(pre_decode_state['conv_bufs'][i].float().abs().max().cpu()):.4g}" for i in range(4)))
        bad_layers = []
        for idx, tensor in enumerate(residual):
            f = tensor.float()
            if bool(torch.isinf(f).any().cpu()) or bool(torch.isnan(f).any().cpu()):
                bad_layers.append(idx)
        print(f"machete_bad_residual_layers={bad_layers}")
        print("machete_layer_norms=" + ",".join(f"{i}:{float(x[i].float().norm().cpu()):.4g}/{float(residual[i].float().norm().cpu()):.4g}" for i in range(len(x))))
        final_hidden = x[-1].float() + residual[-1].float()
        luce_hidden = decoder._hidden.float().view_as(final_hidden)
        hidden_diff = final_hidden - luce_hidden
        hidden_cos = torch.nn.functional.cosine_similarity(
            final_hidden.flatten(), luce_hidden.flatten(), dim=0
        )
        print(f"machete_luce_final_hidden_l2={float(hidden_diff.norm().cpu()):.6f}")
        print(f"machete_luce_final_hidden_maxabs={float(hidden_diff.abs().max().cpu()):.6f}")
        print(f"machete_luce_final_hidden_cos={float(hidden_cos.cpu()):.6f}")
        final_rstd = torch.rsqrt(final_hidden.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        final_normed = final_hidden * final_rstd * machete_weights["final_norm"].float().view(1, 1, -1)
        bf16_logits = torch.matmul(final_normed[0, 0].to(torch.bfloat16).float(), pre_decode_state["embed_weight"].float().t())
        bf16_top = int(torch.argmax(bf16_logits).cpu())
        print(f"machete_hidden_bf16_lm_top={bf16_top}")
        print(f"machete_hidden_bf16_lm_top_text={tokenizer.decode([bf16_top])!r}")
        print(f"luce_next_text={tokenizer.decode([luce_next])!r}")
        print(f"machete_next_text={tokenizer.decode([machete_next])!r}")

    copy_luce_state_to_machete(pre_decode_state, buffers, first_token, args.context_len)
    for _ in range(args.warmup):
        for kernel in kernels:
            kernel.run(validate=False)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.rep):
        copy_luce_state_to_machete(pre_decode_state, buffers, first_token, args.context_len)
        for kernel in kernels:
            kernel.run(validate=False)
    end.record()
    torch.cuda.synchronize()
    ms = float(start.elapsed_time(end) / args.rep)
    print(f"machete_prefilled_decode: {ms:.3f} ms/token, {1000.0 / ms:.1f} tok/s")

    if not matched:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
