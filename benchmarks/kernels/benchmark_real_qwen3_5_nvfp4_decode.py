#!/usr/bin/env python
"""Benchmark the composable Machete real-Qwen3.5 NVFP4 decode graph.

This benchmark targets the same model shape as Luce's Qwen/Qwen3.5-0.8B path:
24 hybrid layers, 18 DeltaNet layers, 6 full-attention layers, hidden=1024,
and vocab=248320.  It measures one composed Machete megakernel decode step.

Use ``--dummy-weights`` for fast framework/runtime iteration.  Omit it to load
HF weights and quantize them into Machete's NVFP4 row layout before timing.
"""

from __future__ import annotations

import argparse
import time

import torch

from machete.megakernel import Megakernel, MegakernelConfig
from machete.quantization import NVFP4Tensor, quantize_nvfp4_weight
from machete.kernels.qwen_3_5 import (
    QWEN3_5_REAL_DN_CONV_CHANNELS,
    QWEN3_5_REAL_DN_CONV_KERNEL,
    QWEN3_5_REAL_DN_NUM_HEADS,
    QWEN3_5_REAL_DN_VALUE_DIM,
    QWEN3_5_REAL_DN_V_SIZE,
    QWEN3_5_REAL_HEAD_DIM,
    QWEN3_5_REAL_HIDDEN,
    QWEN3_5_REAL_INTERMEDIATE,
    QWEN3_5_REAL_KV_DIM,
    QWEN3_5_REAL_LAYER_TYPES,
    QWEN3_5_REAL_NUM_KV_HEADS,
    QWEN3_5_REAL_NUM_LAYERS,
    QWEN3_5_REAL_Q_DIM,
    QWEN3_5_REAL_Q_RAW_DIM,
    QWEN3_5_REAL_ROTARY_D2,
    QWEN3_5_REAL_VOCAB,
    schedule_qwen3_5_final_nvfp4_sm120,
    schedule_qwen3_5_real_nvfp4_decode_sm120,
)
from machete.kernels.qwen_3_5.real_nvfp4_ops import QWEN3_5_REAL_DN_KEY_DIM


def _qweight_empty(rows: int, cols: int, group_size: int = 32) -> NVFP4Tensor:
    packed = torch.empty(rows, cols // 2, device="cuda", dtype=torch.uint8)
    scales = torch.empty(rows, cols // group_size, device="cuda", dtype=torch.float16)
    return NVFP4Tensor(packed, scales, group_size=group_size, rows=rows, cols=cols)


def _qweight_real(weight: torch.Tensor, group_size: int = 32) -> NVFP4Tensor:
    return quantize_nvfp4_weight(weight.contiguous(), group_size=group_size)


def _qwen_rms_weight(weight: torch.Tensor) -> torch.Tensor:
    return (weight.float() + 1.0).to(weight.dtype).contiguous()


def _make_dummy_weights(context_len: int, dtype: torch.dtype, group_size: int) -> dict:
    weights = {
        "cos": torch.ones(context_len + 1, QWEN3_5_REAL_ROTARY_D2, device="cuda", dtype=dtype),
        "sin": torch.zeros(context_len + 1, QWEN3_5_REAL_ROTARY_D2, device="cuda", dtype=dtype),
        "final_norm": torch.ones(QWEN3_5_REAL_HIDDEN, device="cuda", dtype=dtype),
    }
    for layer_idx, layer_type in enumerate(QWEN3_5_REAL_LAYER_TYPES):
        pfx = f"layer.{layer_idx}"
        weights[f"{pfx}.attn_norm"] = torch.ones(QWEN3_5_REAL_HIDDEN, device="cuda", dtype=dtype)
        weights[f"{pfx}.mlp_norm"] = torch.ones(QWEN3_5_REAL_HIDDEN, device="cuda", dtype=dtype)
        weights[f"{pfx}.W_gate_nvfp4"] = _qweight_empty(QWEN3_5_REAL_INTERMEDIATE, QWEN3_5_REAL_HIDDEN, group_size)
        weights[f"{pfx}.W_up_nvfp4"] = _qweight_empty(QWEN3_5_REAL_INTERMEDIATE, QWEN3_5_REAL_HIDDEN, group_size)
        weights[f"{pfx}.W_down_nvfp4"] = _qweight_empty(QWEN3_5_REAL_HIDDEN, QWEN3_5_REAL_INTERMEDIATE, group_size)
        if layer_type == "full_attention":
            weights[f"{pfx}.q_norm"] = torch.ones(QWEN3_5_REAL_HEAD_DIM, device="cuda", dtype=dtype)
            weights[f"{pfx}.k_norm"] = torch.ones(QWEN3_5_REAL_HEAD_DIM, device="cuda", dtype=dtype)
            weights[f"{pfx}.W_q_nvfp4"] = _qweight_empty(QWEN3_5_REAL_Q_RAW_DIM, QWEN3_5_REAL_HIDDEN, group_size)
            weights[f"{pfx}.W_k_nvfp4"] = _qweight_empty(QWEN3_5_REAL_KV_DIM, QWEN3_5_REAL_HIDDEN, group_size)
            weights[f"{pfx}.W_v_nvfp4"] = _qweight_empty(QWEN3_5_REAL_KV_DIM, QWEN3_5_REAL_HIDDEN, group_size)
            weights[f"{pfx}.W_o_nvfp4"] = _qweight_empty(QWEN3_5_REAL_HIDDEN, QWEN3_5_REAL_Q_DIM, group_size)
        else:
            weights[f"{pfx}.linear_norm"] = torch.ones(QWEN3_5_REAL_DN_VALUE_DIM, device="cuda", dtype=dtype)
            weights[f"{pfx}.W_qkv_nvfp4"] = _qweight_empty(QWEN3_5_REAL_DN_CONV_CHANNELS, QWEN3_5_REAL_HIDDEN, group_size)
            weights[f"{pfx}.W_z_nvfp4"] = _qweight_empty(QWEN3_5_REAL_DN_V_SIZE, QWEN3_5_REAL_HIDDEN, group_size)
            weights[f"{pfx}.W_beta_nvfp4"] = _qweight_empty(QWEN3_5_REAL_DN_NUM_HEADS, QWEN3_5_REAL_HIDDEN, group_size)
            weights[f"{pfx}.W_alpha_nvfp4"] = _qweight_empty(QWEN3_5_REAL_DN_NUM_HEADS, QWEN3_5_REAL_HIDDEN, group_size)
            weights[f"{pfx}.conv_weight"] = torch.zeros(
                QWEN3_5_REAL_DN_CONV_CHANNELS,
                QWEN3_5_REAL_DN_CONV_KERNEL,
                device="cuda",
                dtype=dtype,
            )
            weights[f"{pfx}.a_log"] = torch.zeros(QWEN3_5_REAL_DN_NUM_HEADS, device="cuda", dtype=dtype)
            weights[f"{pfx}.dt_bias"] = torch.zeros(QWEN3_5_REAL_DN_NUM_HEADS, device="cuda", dtype=dtype)
            weights[f"{pfx}.W_out_nvfp4"] = _qweight_empty(QWEN3_5_REAL_HIDDEN, QWEN3_5_REAL_DN_V_SIZE, group_size)
    weights["lm_head_nvfp4"] = _qweight_empty(QWEN3_5_REAL_VOCAB, QWEN3_5_REAL_HIDDEN, group_size)
    return weights


def _load_real_weights(model_name: str, context_len: int, dtype: torch.dtype, group_size: int) -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name)
    config = getattr(config, "text_config", config)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        dtype=dtype,
        device_map="cuda",
    ).eval()
    state = model.state_dict()
    weights = {
        "cos": torch.ones(context_len + 1, QWEN3_5_REAL_ROTARY_D2, device="cuda", dtype=dtype),
        "sin": torch.zeros(context_len + 1, QWEN3_5_REAL_ROTARY_D2, device="cuda", dtype=dtype),
        "final_norm": _qwen_rms_weight(state["model.norm.weight"]),
    }
    for layer_idx, layer_type in enumerate(QWEN3_5_REAL_LAYER_TYPES):
        pfx = f"layer.{layer_idx}"
        hf = f"model.layers.{layer_idx}"
        weights[f"{pfx}.attn_norm"] = _qwen_rms_weight(state[f"{hf}.input_layernorm.weight"])
        weights[f"{pfx}.mlp_norm"] = _qwen_rms_weight(state[f"{hf}.post_attention_layernorm.weight"])
        weights[f"{pfx}.W_gate_nvfp4"] = _qweight_real(state[f"{hf}.mlp.gate_proj.weight"], group_size)
        weights[f"{pfx}.W_up_nvfp4"] = _qweight_real(state[f"{hf}.mlp.up_proj.weight"], group_size)
        weights[f"{pfx}.W_down_nvfp4"] = _qweight_real(state[f"{hf}.mlp.down_proj.weight"], group_size)
        if layer_type == "full_attention":
            weights[f"{pfx}.q_norm"] = _qwen_rms_weight(state[f"{hf}.self_attn.q_norm.weight"])
            weights[f"{pfx}.k_norm"] = _qwen_rms_weight(state[f"{hf}.self_attn.k_norm.weight"])
            weights[f"{pfx}.W_q_nvfp4"] = _qweight_real(state[f"{hf}.self_attn.q_proj.weight"], group_size)
            weights[f"{pfx}.W_k_nvfp4"] = _qweight_real(state[f"{hf}.self_attn.k_proj.weight"], group_size)
            weights[f"{pfx}.W_v_nvfp4"] = _qweight_real(state[f"{hf}.self_attn.v_proj.weight"], group_size)
            weights[f"{pfx}.W_o_nvfp4"] = _qweight_real(state[f"{hf}.self_attn.o_proj.weight"], group_size)
        else:
            weights[f"{pfx}.linear_norm"] = state[f"{hf}.linear_attn.norm.weight"].contiguous()
            weights[f"{pfx}.W_qkv_nvfp4"] = _qweight_real(state[f"{hf}.linear_attn.in_proj_qkv.weight"], group_size)
            weights[f"{pfx}.W_z_nvfp4"] = _qweight_real(state[f"{hf}.linear_attn.in_proj_z.weight"], group_size)
            weights[f"{pfx}.W_beta_nvfp4"] = _qweight_real(state[f"{hf}.linear_attn.in_proj_b.weight"], group_size)
            weights[f"{pfx}.W_alpha_nvfp4"] = _qweight_real(state[f"{hf}.linear_attn.in_proj_a.weight"], group_size)
            weights[f"{pfx}.conv_weight"] = state[f"{hf}.linear_attn.conv1d.weight"].squeeze(1).contiguous()
            weights[f"{pfx}.a_log"] = state[f"{hf}.linear_attn.A_log"].contiguous()
            weights[f"{pfx}.dt_bias"] = state[f"{hf}.linear_attn.dt_bias"].contiguous()
            weights[f"{pfx}.W_out_nvfp4"] = _qweight_real(state[f"{hf}.linear_attn.out_proj.weight"], group_size)
    lm_head = state.get("lm_head.weight", state["model.embed_tokens.weight"]).contiguous()
    weights["lm_head_nvfp4"] = _qweight_real(lm_head, group_size)
    del model
    torch.cuda.empty_cache()
    return weights


def _make_buffers(context_len: int, dtype: torch.dtype, top_partitions: int):
    batch = 1
    seq_len = 1
    x = [torch.zeros(batch, seq_len, QWEN3_5_REAL_HIDDEN, device="cuda", dtype=dtype) for _ in range(QWEN3_5_REAL_NUM_LAYERS + 1)]
    residual = [torch.zeros_like(x[0]) for _ in range(QWEN3_5_REAL_NUM_LAYERS + 1)]
    k_cache = [
        torch.zeros(batch, context_len + 1, QWEN3_5_REAL_NUM_KV_HEADS, QWEN3_5_REAL_HEAD_DIM, device="cuda", dtype=dtype)
        for _ in range(QWEN3_5_REAL_NUM_LAYERS)
    ]
    v_cache = [torch.zeros_like(k) for k in k_cache]
    q_raw = [torch.empty(batch, seq_len, QWEN3_5_REAL_Q_RAW_DIM, device="cuda", dtype=dtype) for _ in range(QWEN3_5_REAL_NUM_LAYERS)]
    kv_raw = [torch.empty(batch, seq_len, 2 * QWEN3_5_REAL_KV_DIM, device="cuda", dtype=dtype) for _ in range(QWEN3_5_REAL_NUM_LAYERS)]
    q_gate = [torch.empty(batch, seq_len, QWEN3_5_REAL_Q_DIM, device="cuda", dtype=dtype) for _ in range(QWEN3_5_REAL_NUM_LAYERS)]
    q_buf = [torch.empty(batch, seq_len, QWEN3_5_REAL_Q_DIM, device="cuda", dtype=dtype) for _ in range(QWEN3_5_REAL_NUM_LAYERS)]
    attn_out = [torch.empty_like(q) for q in q_buf]
    norm = [torch.empty(batch, seq_len, QWEN3_5_REAL_HIDDEN, device="cuda", dtype=dtype) for _ in range(QWEN3_5_REAL_NUM_LAYERS)]
    qkv = [torch.empty(batch, seq_len, QWEN3_5_REAL_DN_CONV_CHANNELS, device="cuda", dtype=dtype) for _ in range(18)]
    z = [torch.empty(batch, seq_len, QWEN3_5_REAL_DN_V_SIZE, device="cuda", dtype=dtype) for _ in range(18)]
    beta = [torch.empty(batch, seq_len, QWEN3_5_REAL_DN_NUM_HEADS, device="cuda", dtype=dtype) for _ in range(18)]
    alpha = [torch.empty_like(b) for b in beta]
    dn_out = [torch.empty_like(v) for v in z]
    mlp = [torch.empty(batch, seq_len, QWEN3_5_REAL_INTERMEDIATE, device="cuda", dtype=dtype) for _ in range(QWEN3_5_REAL_NUM_LAYERS)]
    dn_state = [
        torch.zeros(batch, QWEN3_5_REAL_DN_NUM_HEADS, QWEN3_5_REAL_DN_KEY_DIM, QWEN3_5_REAL_DN_VALUE_DIM, device="cuda")
        for _ in range(18)
    ]
    conv = [
        torch.zeros(batch, QWEN3_5_REAL_DN_CONV_CHANNELS, QWEN3_5_REAL_DN_CONV_KERNEL, device="cuda")
        for _ in range(18)
    ]
    top_values = torch.empty(batch, seq_len, device="cuda", dtype=torch.float32)
    top_indices = torch.empty(batch, seq_len, device="cuda", dtype=torch.int32)
    top_partial_values = None
    top_partial_indices = None
    if top_partitions > 0:
        top_partial_values = torch.empty(batch, seq_len, top_partitions, device="cuda", dtype=torch.float32)
        top_partial_indices = torch.empty(batch, seq_len, top_partitions, device="cuda", dtype=torch.int32)
    return (
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
    )


def _schedule_body(args, weights, buffers):
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
    schedule = schedule_qwen3_5_real_nvfp4_decode_sm120(
        batch=1,
        seq_len=1,
        cache_pos=args.context_len,
        weights=weights,
        x_buffers=x,
        residual_buffers=residual,
        k_cache=k_cache,
        v_cache=v_cache,
        q_buf=q_buf,
        q_raw_buf=q_raw,
        kv_raw_buf=kv_raw,
        q_gate_buf=q_gate,
        attn_out_buf=attn_out,
        norm_buf=norm,
        qkv_buf=qkv,
        z_buf=z,
        beta_buf=beta,
        alpha_buf=alpha,
        dn_out_buf=dn_out,
        mlp_h_buf=mlp,
        dn_state=dn_state,
        conv_buf=conv,
        final_norm=None if args.no_final else weights["final_norm"],
        lm_head_nvfp4=None if args.no_final else weights["lm_head_nvfp4"],
        top_values=top_values,
        top_indices=top_indices,
        top_partial_values=top_partial_values,
        top_partial_indices=top_partial_indices,
        page_size=args.page_size,
        group_size=args.group_size,
        fa_num_splits=args.fa_num_splits,
        use_flash_attention=args.use_flash_attention,
        matvec_block=args.matvec_block,
        final_head_range_block=args.final_head_range_block,
    )
    return schedule


def _make_replay_kernel(args, ops, keep_alive):
    kernel = Megakernel(
        ops,
        config=MegakernelConfig(
            num_pages=args.num_pages,
            page_size=args.page_size,
            threads_per_block=args.threads,
        ),
    )
    kernel._keep_alive = keep_alive
    return kernel


def build_kernel(args):
    dtype = torch.bfloat16
    top_partitions = args.top_partitions
    if top_partitions <= 0:
        sm_count = torch.cuda.get_device_properties(0).multi_processor_count
        top_partitions = sm_count
    weights = (
        _make_dummy_weights(args.context_len, dtype, args.group_size)
        if args.dummy_weights
        else _load_real_weights(args.model, args.context_len, dtype, args.group_size)
    )
    buffers = _make_buffers(args.context_len, dtype, top_partitions)
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
    if args.final_only:
        schedule_ops = schedule_qwen3_5_final_nvfp4_sm120(
            x=x[-1],
            residual_in=residual[-1],
            residual_out=residual[-1],
            final_norm=weights["final_norm"],
            lm_head_nvfp4=weights["lm_head_nvfp4"],
            top_values=top_values,
            top_indices=top_indices,
            top_partial_values=top_partial_values,
            top_partial_indices=top_partial_indices,
            seq_len=1,
            page_size=args.page_size,
            group_size=args.group_size,
            final_head_range_block=args.final_head_range_block,
        )
        keep_alive = []
    else:
        schedule = _schedule_body(args, weights, buffers)
        schedule_ops = schedule.ops
        keep_alive = schedule.keep_alive
    return _make_replay_kernel(args, schedule_ops, [weights, buffers, keep_alive])


def time_kernel(kernel: Megakernel, warmup: int, rep: int) -> float:
    kernel.compile()
    for _ in range(warmup):
        kernel.run(validate=False)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        kernel.run(validate=False)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=32768)
    parser.add_argument("--num-pages", type=int, default=2)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--matvec-block", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rep", type=int, default=5)
    parser.add_argument("--top-partitions", type=int, default=0)
    parser.add_argument("--final-head-range-block", type=int, default=1)
    parser.add_argument("--fa-num-splits", type=int, default=0)
    parser.add_argument("--use-flash-attention", action="store_true")
    parser.add_argument("--dummy-weights", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--no-final", action="store_true")
    parser.add_argument("--final-only", action="store_true")
    args = parser.parse_args()

    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, SMs={props.multi_processor_count}", flush=True)
    print(
        f"model={args.model}, ctx={args.context_len}, dummy={args.dummy_weights}, "
        f"group_size={args.group_size}, "
        f"top_partitions={'sms' if args.top_partitions <= 0 else args.top_partitions}, "
        f"final_head_range_block={args.final_head_range_block}, "
        f"no_final={args.no_final}, "
        f"final_only={args.final_only}, "
        f"use_flash_attention={args.use_flash_attention}, fa_num_splits={args.fa_num_splits}",
        flush=True,
    )
    t0 = time.perf_counter()
    print("building schedule...", flush=True)
    kernel = build_kernel(args)
    print(f"build_host_s={time.perf_counter() - t0:.3f}", flush=True)
    if args.compile_only:
        kernel.compile()
        print(
            f"compiled: ops={len(kernel.ops)}, tiles={kernel.total_tiles}, "
            f"instructions={kernel._num_instructions}",
            flush=True,
        )
        return
    print("timing hot launches...", flush=True)
    ms = time_kernel(kernel, args.warmup, args.rep)
    print(
        f"machete_real_qwen_nvfp4_decode: {ms:.3f} ms/token, "
        f"{1000.0 / ms:.1f} tok/s, ops={len(kernel.ops)}, "
        f"tiles={kernel.total_tiles}, instructions={kernel._num_instructions}, "
        f"page_size={args.page_size}, num_pages={args.num_pages}"
        ,
        flush=True,
    )


if __name__ == "__main__":
    main()
