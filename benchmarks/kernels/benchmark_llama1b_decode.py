#!/usr/bin/env python
# Copyright (c) 2025, Machete Authors
"""Benchmark a Llama-3.2-1B style decode pass as one Machete megakernel.

This mirrors a low-latency Llama decode setup:
    - 16 layers, hidden=2048, intermediate=8192
    - 32 query heads, 8 KV heads, head_dim=64
    - matvec/decode tile of 16 rows, matching their 16-element block design

The fused implementation defines larger instructions such as RMS+QKV+RoPE+
KV-append and RMS+up/gate+SiLU.  In this repo we express the same forward pass
with the existing Machete op vocabulary inside a single persistent megakernel,
preserving the important pieces for B200 benchmarking: one launch, paged
shared-memory reuse, and fine-grained instruction dependencies.

Usage:
    python benchmarks/kernels/benchmark_llama1b_decode.py --context-len 2048 --num-pages 3
"""

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from machete.utils.benchmark import Benchmark

try:
    import cutlass  # noqa: F401

    CUTLASS_AVAILABLE = True
except ImportError:
    CUTLASS_AVAILABLE = False


NUM_LAYERS = 16
HIDDEN = 2048
INTERMEDIATE = 8192
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 64
ROTARY_DIM = 64
D2 = ROTARY_DIM // 2
Q_DIM = NUM_Q_HEADS * HEAD_DIM
KV_DIM = NUM_KV_HEADS * HEAD_DIM
KV_GROUP_SIZE = NUM_Q_HEADS // NUM_KV_HEADS
VOCAB_SIZE = 128256
EPS = 1e-5
DECODE_S = 16


try:
    from machete.kernels.gemm import GemmOp as _BaseGemmOp
except ImportError:
    _BaseGemmOp = None


if _BaseGemmOp is not None:

    class LmHeadGemmOp(_BaseGemmOp):
        """Separate handler family for the final LM head."""


else:

    class LmHeadGemmOp:
        """Import-safe placeholder used only when Machete is unavailable."""


def is_sm90_or_newer():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _correctness_status(actual, expected, rtol=2e-2, atol=2e-1):
    actual_f = actual.float()
    expected_f = expected.float()
    if not torch.isfinite(actual_f).all():
        bad = int((~torch.isfinite(actual_f)).sum().item())
        return f"BAD {100.0 * bad / max(1, actual.numel()):.1f}%/nonfinite"
    diff = (actual_f - expected_f).abs()
    max_abs = float(diff.max().item())
    bad = diff > (atol + rtol * expected_f.abs())
    bad_count = int(bad.sum().item())
    if bad_count == 0:
        return f"OK/{max_abs:.3g}"
    return f"BAD {100.0 * bad_count / max(1, actual.numel()):.1f}%/{max_abs:.3g}"


def _rmsnorm(x, weight, eps=EPS):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def _apply_rope(x, cos, sin, num_heads, pos):
    b, s, _ = x.shape
    x = x.view(b, s, num_heads, HEAD_DIM)
    x0 = x[..., :D2]
    x1 = x[..., D2:ROTARY_DIM]
    cos_s = cos[pos : pos + s].view(1, s, 1, D2)
    sin_s = sin[pos : pos + s].view(1, s, 1, D2)
    out = x.clone()
    out[..., :D2] = x0.float() * cos_s.float() - x1.float() * sin_s.float()
    out[..., D2:ROTARY_DIM] = x1.float() * cos_s.float() + x0.float() * sin_s.float()
    return out.reshape(b, s, num_heads * HEAD_DIM).to(x.dtype)


def allocate_model_weights(dtype=torch.bfloat16, device="cuda", max_seq_len=8192):
    """Allocate synthetic Llama-3.2-1B weights in Machete's expected layouts."""
    weights = {}
    for i in range(NUM_LAYERS):
        pfx = f"layer.{i}"
        weights[f"{pfx}.attn_norm"] = torch.ones(HIDDEN, dtype=dtype, device=device)
        weights[f"{pfx}.W_q"] = torch.randn(Q_DIM, HIDDEN, dtype=dtype, device=device) * 0.02
        weights[f"{pfx}.W_k"] = torch.randn(KV_DIM, HIDDEN, dtype=dtype, device=device) * 0.02
        weights[f"{pfx}.W_v"] = torch.randn(KV_DIM, HIDDEN, dtype=dtype, device=device) * 0.02
        weights[f"{pfx}.W_o"] = torch.randn(HIDDEN, Q_DIM, dtype=dtype, device=device) * 0.02
        weights[f"{pfx}.mlp_norm"] = torch.ones(HIDDEN, dtype=dtype, device=device)
        weights[f"{pfx}.W_gate"] = torch.randn(INTERMEDIATE, HIDDEN, dtype=dtype, device=device) * 0.02
        weights[f"{pfx}.W_up"] = torch.randn(INTERMEDIATE, HIDDEN, dtype=dtype, device=device) * 0.02
        weights[f"{pfx}.W_gate_up"] = torch.cat(
            [weights[f"{pfx}.W_gate"], weights[f"{pfx}.W_up"]],
            dim=0,
        ).contiguous()
        weights[f"{pfx}.W_down"] = torch.randn(HIDDEN, INTERMEDIATE, dtype=dtype, device=device) * 0.02
    weights["final_norm"] = torch.ones(HIDDEN, dtype=dtype, device=device)
    weights["lm_head"] = torch.randn(VOCAB_SIZE, HIDDEN, dtype=dtype, device=device) * 0.02
    theta = 1.0 / (500000.0 ** (torch.arange(0, ROTARY_DIM, 2, device=device).float() / ROTARY_DIM))
    idx = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(idx, theta)
    weights["cos"] = freqs.cos().to(dtype)
    weights["sin"] = freqs.sin().to(dtype)
    return weights


def allocate_kv_cache(batch, max_seq_len, dtype=torch.bfloat16, device="cuda"):
    k_caches = [
        torch.zeros(batch, max_seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
        for _ in range(NUM_LAYERS)
    ]
    v_caches = [
        torch.zeros(batch, max_seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
        for _ in range(NUM_LAYERS)
    ]
    return k_caches, v_caches


def sequential_decode_step(x, residual, pos, k_caches, v_caches, weights, num_layers=NUM_LAYERS):
    b, s, _ = x.shape
    for i in range(num_layers):
        pfx = f"layer.{i}"
        residual = x + residual
        h = _rmsnorm(residual, weights[f"{pfx}.attn_norm"])

        q = torch.matmul(h, weights[f"{pfx}.W_q"].t())
        k = torch.matmul(h, weights[f"{pfx}.W_k"].t())
        v = torch.matmul(h, weights[f"{pfx}.W_v"].t())
        q = _apply_rope(q, weights["cos"], weights["sin"], NUM_Q_HEADS, pos)
        k = _apply_rope(k, weights["cos"], weights["sin"], NUM_KV_HEADS, pos)

        k_caches[i][:, pos : pos + s] = k.view(b, s, NUM_KV_HEADS, HEAD_DIM)
        v_caches[i][:, pos : pos + s] = v.view(b, s, NUM_KV_HEADS, HEAD_DIM)

        q_4d = q.view(b, s, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
        k_full = k_caches[i][:, : pos + s].transpose(1, 2).repeat_interleave(KV_GROUP_SIZE, dim=1)
        v_full = v_caches[i][:, : pos + s].transpose(1, 2).repeat_interleave(KV_GROUP_SIZE, dim=1)
        attn = F.scaled_dot_product_attention(q_4d, k_full, v_full, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(b, s, Q_DIM)

        proj = torch.matmul(attn, weights[f"{pfx}.W_o"].t())
        residual = proj + residual
        h2 = _rmsnorm(residual, weights[f"{pfx}.mlp_norm"])
        gate = torch.matmul(h2, weights[f"{pfx}.W_gate"].t())
        up = torch.matmul(h2, weights[f"{pfx}.W_up"].t())
        x = torch.matmul(F.silu(gate) * up, weights[f"{pfx}.W_down"].t())

    residual = x + residual
    h_final = _rmsnorm(residual, weights["final_norm"])
    logits = torch.matmul(h_final, weights["lm_head"].t())
    return logits, residual


def maybe_compiled_forward(fn, example_args):
    try:
        compiled = torch.compile(fn, mode="reduce-overhead", fullgraph=True)
        compiled(*example_args)
        compiled(*example_args)
        torch.cuda.synchronize()
        return compiled
    except Exception as exc:
        print(f"  torch.compile failed: {exc}")
        return None


def _schedule_layer_ops(
    i,
    batch,
    pos,
    seq_s,
    k_caches,
    v_caches,
    weights,
    page_size,
    x_in,
    res_in,
    x_out,
    res_out,
    h_buf,
    q_buf,
    k_buf,
    v_buf,
    attn_out_buf,
    proj_buf,
    h2_buf,
    gate_up_buf,
    mlp_h_buf,
    use_sm100_ops=False,
):
    if use_sm100_ops:
        decode_arch = os.environ.get("LLAMA1B_DECODE_ARCH", "sm100").lower()
        if decode_arch == "sm120":
            from machete.kernels.decode_matvec.sm120 import (
                schedule_decode_layer_sm120 as schedule_decode_layer,
            )
        elif decode_arch == "sm100":
            from machete.kernels.decode_matvec.sm100 import (
                schedule_decode_layer_sm100 as schedule_decode_layer,
            )
        else:
            raise ValueError(f"Unsupported LLAMA1B_DECODE_ARCH={decode_arch!r}")

        layer = schedule_decode_layer(
            layer_idx=i,
            batch=batch,
            seq_len=seq_s,
            cache_pos=pos,
            weights=weights,
            k_cache=k_caches[i],
            v_cache=v_caches[i],
            x_in=x_in,
            residual_in=res_in,
            x_out=x_out,
            residual_out=res_out,
            q_buf=q_buf,
            attn_out_buf=attn_out_buf,
            mlp_h_buf=mlp_h_buf,
            page_size=page_size,
            eps=EPS,
            fa_num_splits=int(os.environ.get("LLAMA1B_DECODE_FA_SPLITS", "0")),
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            num_q_heads=NUM_Q_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            kv_group_size=KV_GROUP_SIZE,
        )
        return layer.ops, layer.attention_config, layer.keep_alive

    from benchmarks.kernels.benchmark_qwen3_5_decode import VCacheStoreOp
    from machete.kernels.attention import FlashAttentionOp
    from machete.kernels.attention.flash_decoding import flash_decoding_schedule
    from machete.kernels.gemm import GemmOp
    from machete.kernels.glu import GLUOp
    from machete.kernels.rms_norm import RMSNormOp
    from machete.kernels.rope import RopeOp

    pfx = f"layer.{i}"
    cos = weights["cos"][pos : pos + seq_s]
    sin = weights["sin"][pos : pos + seq_s]
    rope_tile_nh = 4 if page_size <= 16 * 1024 else 8

    q_4d = q_buf.view(batch, seq_s, NUM_Q_HEADS, HEAD_DIM)
    k_4d = k_buf.view(batch, seq_s, NUM_KV_HEADS, HEAD_DIM)
    k_fd = k_caches[i][:, : pos + seq_s]
    v_fd = v_caches[i][:, : pos + seq_s]
    o_fd = attn_out_buf.view(batch, seq_s, NUM_Q_HEADS, HEAD_DIM)

    ops = []
    ops += RMSNormOp.schedule(
        x=x_in,
        weight=weights[f"{pfx}.attn_norm"],
        y=h_buf,
        residual_in=res_in,
        residual_out=res_out,
        tile_sizes={"S": int(os.environ.get("LLAMA1B_DECODE_RMS_TILE_S", "1"))},
        page_size=page_size,
    )
    ops += GemmOp.schedule(a=h_buf, b=weights[f"{pfx}.W_q"], c=q_buf, page_size=page_size)
    ops += GemmOp.schedule(a=h_buf, b=weights[f"{pfx}.W_k"], c=k_buf, page_size=page_size)
    ops += GemmOp.schedule(a=h_buf, b=weights[f"{pfx}.W_v"], c=v_buf, page_size=page_size)
    ops += RopeOp.schedule(q=q_4d, cos=cos, sin=sin, tile_sizes={"S": seq_s, "NH": rope_tile_nh}, page_size=page_size)
    ops += RopeOp.schedule(q=k_4d, cos=cos, sin=sin, tile_sizes={"S": seq_s, "NH": rope_tile_nh}, page_size=page_size)
    ops += VCacheStoreOp.schedule(src_v=k_4d, dst_v=k_fd, cache_pos=pos, tile_sizes={"S": seq_s, "H": 1})
    ops += VCacheStoreOp.schedule(src_v=v_buf.view(batch, seq_s, NUM_KV_HEADS, HEAD_DIM), dst_v=v_fd, cache_pos=pos, tile_sizes={"S": seq_s, "H": 1})

    if k_fd.shape[1] <= 256:
        fa_ops = FlashAttentionOp.schedule(
            q=q_4d,
            k=k_fd,
            v=v_fd,
            o=o_fd,
            kv_group_size=KV_GROUP_SIZE,
            page_size=page_size,
        )
        fa_config = FlashAttentionOp.kernel_config(fa_ops)
    else:
        fa_ops, fa_config = flash_decoding_schedule(
            q=q_4d,
            k=k_fd,
            v=v_fd,
            o=o_fd,
            kv_group_size=KV_GROUP_SIZE,
            page_size=page_size,
            num_splits=int(os.environ.get("LLAMA1B_DECODE_FA_SPLITS", "0")),
        )
    ops += fa_ops
    ops += GemmOp.schedule(a=attn_out_buf, b=weights[f"{pfx}.W_o"], c=proj_buf, page_size=page_size)
    ops += RMSNormOp.schedule(
        x=proj_buf,
        weight=weights[f"{pfx}.mlp_norm"],
        y=h2_buf,
        residual_in=res_out,
        residual_out=res_out,
        tile_sizes={"S": int(os.environ.get("LLAMA1B_DECODE_RMS_TILE_S", "1"))},
        page_size=page_size,
    )
    ops += GemmOp.schedule(
        a=h2_buf,
        b=weights[f"{pfx}.W_gate_up"],
        c=gate_up_buf,
        tile_sizes={"S": seq_s, "N": 128, "K": 32},
        page_size=page_size,
    )
    for op in ops[-1:]:
        op.dim_aliases["N"] = f"mlp_chunk_{i}"
        op.static_dims["barrier_group_count_N"] = 4
    ops += GLUOp.schedule(x=gate_up_buf, y=mlp_h_buf, activation="silu", tile_sizes={"S": 1}, page_size=page_size)
    ops += GemmOp.schedule(a=mlp_h_buf, b=weights[f"{pfx}.W_down"], c=x_out, page_size=page_size)

    return ops, fa_config, [cos, sin, q_4d, k_4d, k_fd, v_fd, o_fd]


def megakernel_decode_build(
    batch,
    pos,
    k_caches,
    v_caches,
    weights,
    x_init,
    residual_init,
    page_size=32768,
    num_pages=3,
    torch_lm_head=True,
    num_layers=NUM_LAYERS,
    use_sm100_ops=False,
):
    from machete.kernels.gemm import GemmOp
    from machete.kernels.rms_norm import RMSNormOp
    from machete.megakernel import Megakernel, MegakernelConfig

    dtype = x_init.dtype
    device = x_init.device
    seq_s = x_init.shape[1]

    x_buf = [torch.empty(batch, seq_s, HIDDEN, dtype=dtype, device=device) for _ in range(2)]
    res_buf = [torch.empty(batch, seq_s, HIDDEN, dtype=dtype, device=device) for _ in range(2)]
    x_buf[0].copy_(x_init)
    res_buf[0].copy_(residual_init)

    h_buf = torch.empty(batch, seq_s, HIDDEN, dtype=dtype, device=device)
    q_buf = torch.empty(batch, seq_s, Q_DIM, dtype=dtype, device=device)
    k_buf = torch.empty(batch, seq_s, KV_DIM, dtype=dtype, device=device)
    v_buf = torch.empty(batch, seq_s, KV_DIM, dtype=dtype, device=device)
    attn_out_buf = torch.empty(batch, seq_s, Q_DIM, dtype=dtype, device=device)
    proj_buf = torch.empty(batch, seq_s, HIDDEN, dtype=dtype, device=device)
    h2_buf = torch.empty(batch, seq_s, HIDDEN, dtype=dtype, device=device)
    gate_up_buf = torch.empty(batch, seq_s, 2 * INTERMEDIATE, dtype=dtype, device=device)
    mlp_h_buf = torch.empty(batch, seq_s, INTERMEDIATE, dtype=dtype, device=device)
    h_final_buf = torch.empty(batch, seq_s, HIDDEN, dtype=dtype, device=device)
    res_final = torch.empty(batch, seq_s, HIDDEN, dtype=dtype, device=device)
    logits_buf = torch.empty(batch, seq_s, VOCAB_SIZE, dtype=dtype, device=device)

    keep_alive = list(weights.values()) + k_caches + v_caches + [
        *x_buf,
        *res_buf,
        h_buf,
        q_buf,
        k_buf,
        v_buf,
        attn_out_buf,
        proj_buf,
        h2_buf,
        gate_up_buf,
        mlp_h_buf,
        h_final_buf,
        res_final,
        logits_buf,
    ]

    all_ops = []
    max_fa_tpb = 0
    for i in range(num_layers):
        layer_ops, fa_config, extra_keep = _schedule_layer_ops(
            i,
            batch,
            pos,
            seq_s,
            k_caches,
            v_caches,
            weights,
            page_size,
            x_buf[i % 2],
            res_buf[i % 2],
            x_buf[(i + 1) % 2],
            res_buf[(i + 1) % 2],
            h_buf,
            q_buf,
            k_buf,
            v_buf,
            attn_out_buf,
            proj_buf,
            h2_buf,
            gate_up_buf,
            mlp_h_buf,
            use_sm100_ops=use_sm100_ops,
        )
        all_ops += layer_ops
        max_fa_tpb = max(max_fa_tpb, fa_config.threads_per_block)
        keep_alive += extra_keep

    if use_sm100_ops:
        decode_arch = os.environ.get("LLAMA1B_DECODE_ARCH", "sm100").lower()
        if decode_arch == "sm120":
            from machete.kernels.decode_matvec.sm120 import (
                schedule_final_sm120 as schedule_final,
            )
        elif decode_arch == "sm100":
            from machete.kernels.decode_matvec.sm100 import (
                schedule_final_sm100 as schedule_final,
            )
        else:
            raise ValueError(f"Unsupported LLAMA1B_DECODE_ARCH={decode_arch!r}")

        all_ops += schedule_final(
            x=x_buf[num_layers % 2],
            residual_in=res_buf[num_layers % 2],
            residual_out=res_final,
            final_norm=weights["final_norm"],
            lm_head=None if torch_lm_head else weights["lm_head"],
            logits=None if torch_lm_head else logits_buf,
            seq_len=seq_s,
            page_size=page_size,
            eps=EPS,
        )
    else:
        all_ops += RMSNormOp.schedule(
            x=x_buf[num_layers % 2],
            weight=weights["final_norm"],
            y=h_final_buf,
            residual_in=res_buf[num_layers % 2],
            residual_out=res_final,
            tile_sizes={"S": int(os.environ.get("LLAMA1B_DECODE_RMS_TILE_S", "1"))},
            page_size=page_size,
        )
    if not torch_lm_head and not use_sm100_ops:
        all_ops += LmHeadGemmOp.schedule(a=h_final_buf, b=weights["lm_head"], c=logits_buf, page_size=page_size)

    gemm_config = GemmOp.kernel_config([op for op in all_ops if op.tile_sizes.get("S") is not None] or all_ops)
    props = torch.cuda.get_device_properties(device)
    config = MegakernelConfig(
        num_sms=int(os.environ.get("LLAMA1B_DECODE_NUM_SMS", props.multi_processor_count)),
        threads_per_block=int(os.environ.get("LLAMA1B_DECODE_THREADS", str(max(224, gemm_config.threads_per_block, max_fa_tpb)))),
        page_size=max(op.static_dims.get("page_size", page_size) for op in all_ops),
        num_pages=num_pages,
        sync_compute_warps_after_tile=False,
        inline_thin_phases=os.environ.get("LLAMA1B_DECODE_INLINE_THIN", "1") != "0",
        loader_idle_sleep_ns=int(os.environ.get("LLAMA1B_DECODE_LOADER_SLEEP", "0")),
        relaxed_global_barriers=os.environ.get("LLAMA1B_DECODE_RELAXED_BARRIER", "1") == "1",
        global_barrier_sleep_ns=int(os.environ.get("LLAMA1B_DECODE_BARRIER_SLEEP", "0")),
        opt_level=int(os.environ.get("LLAMA1B_DECODE_OPT_LEVEL", "2")),
    )

    kernel = Megakernel(all_ops, config=config)
    spec = kernel.bench_spec(keep_alive=keep_alive)
    if torch_lm_head:
        from machete.utils.benchmark_utils import KernelBenchSpec

        core_spec = spec
        bench_stream, cu_stream = core_spec.stream

        def _setup():
            if core_spec.setup_fn is not None:
                core_spec.setup_fn()

        def _launch():
            core_spec.launch_fn()
            with torch.cuda.stream(bench_stream):
                if use_sm100_ops:
                    h_tmp = _rmsnorm(res_final, weights["final_norm"])
                    torch.matmul(h_tmp, weights["lm_head"].t(), out=logits_buf)
                else:
                    torch.matmul(h_final_buf, weights["lm_head"].t(), out=logits_buf)

        spec = KernelBenchSpec(
            launch_fn=_launch,
            setup_fn=_setup,
            stream=(bench_stream, cu_stream),
            use_host_timer=core_spec.use_host_timer,
            _keep_alive=(core_spec, keep_alive, h_final_buf, res_final, logits_buf, weights["lm_head"]),
        )
    return spec, logits_buf, res_final


def _total_weight_bytes():
    elem = 2
    per_layer = (
        Q_DIM * HIDDEN
        + 2 * KV_DIM * HIDDEN
        + HIDDEN * Q_DIM
        + 2 * INTERMEDIATE * HIDDEN
        + HIDDEN * INTERMEDIATE
        + 2 * HIDDEN
    )
    return (NUM_LAYERS * per_layer + VOCAB_SIZE * HIDDEN + HIDDEN) * elem


def _total_bytes_decode(context_len):
    elem = 2
    kv_bytes = NUM_LAYERS * 2 * NUM_KV_HEADS * (context_len + DECODE_S) * HEAD_DIM * elem
    return _total_weight_bytes() + kv_bytes


def _total_flops_decode(context_len):
    per_layer = (
        2 * Q_DIM * HIDDEN
        + 2 * KV_DIM * HIDDEN * 2
        + 2 * HIDDEN * Q_DIM
        + 2 * INTERMEDIATE * HIDDEN * 2
        + 2 * HIDDEN * INTERMEDIATE
        + NUM_Q_HEADS * 2 * (context_len + DECODE_S) * HEAD_DIM
    )
    return NUM_LAYERS * per_layer + 2 * VOCAB_SIZE * HIDDEN


@Benchmark.parametrize("batch", [1])
@Benchmark.parametrize("context_len", [128, 512, 1024, 2048, 4096])
@Benchmark.parametrize("num_pages", [2, 3, 4])
@Benchmark.parametrize("page_size", [16384, 32768, 49152])
def bench_llama1b_decode(context_len, batch, num_pages, page_size):
    torch.manual_seed(123)
    dtype = torch.bfloat16
    device = "cuda"

    weights = allocate_model_weights(dtype=dtype, device=device, max_seq_len=context_len + DECODE_S + 1)
    k_caches, v_caches = allocate_kv_cache(batch, context_len + DECODE_S, dtype=dtype, device=device)
    pos = context_len
    for i in range(NUM_LAYERS):
        k_caches[i][:, :pos].normal_()
        v_caches[i][:, :pos].normal_()
    x = torch.randn(batch, DECODE_S, HIDDEN, dtype=dtype, device=device)
    residual = torch.zeros(batch, DECODE_S, HIDDEN, dtype=dtype, device=device)

    funcs = {}
    sequential_decode_step(x, residual.clone(), pos, k_caches, v_caches, weights, num_layers=1)
    torch.cuda.synchronize()
    funcs["sequential_1layer"] = lambda: sequential_decode_step(
        x, residual.clone(), pos, k_caches, v_caches, weights, num_layers=1
    )

    def full_decode(x_arg, residual_arg):
        return sequential_decode_step(x_arg, residual_arg, pos, k_caches, v_caches, weights)

    if os.environ.get("LLAMA1B_DECODE_COMPILE", "0") == "1":
        compiled = maybe_compiled_forward(full_decode, (x, residual.clone()))
        if compiled is not None:
            funcs["torch_compile"] = lambda: compiled(x, residual.clone())

    if is_sm90_or_newer() and CUTLASS_AVAILABLE:
        try:
            ref_logits = ref_residual = None
            if os.environ.get("LLAMA1B_DECODE_CHECK", "0") == "1":
                ref_logits, ref_residual = sequential_decode_step(
                    x,
                    residual.clone(),
                    pos,
                    [c.clone() for c in k_caches],
                    [c.clone() for c in v_caches],
                    weights,
                )
                torch.cuda.synchronize()
            spec, logits_mk, residual_mk = megakernel_decode_build(
                batch,
                pos,
                [c.clone() for c in k_caches],
                [c.clone() for c in v_caches],
                weights,
                x_init=x,
                residual_init=residual.clone(),
                page_size=page_size,
                num_pages=num_pages,
                torch_lm_head=os.environ.get("LLAMA1B_DECODE_TORCH_LM_HEAD", "1") == "1",
                use_sm100_ops=os.environ.get("LLAMA1B_DECODE_SM100_OPS", "1") != "0",
            )
            if ref_logits is not None:
                if spec.setup_fn is not None:
                    spec.setup_fn()
                spec.launch_fn()
                torch.cuda.synchronize()
                l_stat = _correctness_status(logits_mk, ref_logits, rtol=2e-2, atol=3e-1)
                r_stat = _correctness_status(residual_mk, ref_residual, rtol=2e-2, atol=4e-1)
                spec.metadata = f"L {l_stat} R {r_stat}"
            funcs["mega"] = spec
        except Exception as exc:
            import traceback

            traceback.print_exc()
            print(f"  megakernel build failed: {exc}")

    return funcs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-len", type=int, action="append")
    parser.add_argument("--batch", type=int, action="append")
    parser.add_argument("--page-size", type=int, action="append")
    parser.add_argument("--num-pages", type=int, action="append")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    args = parser.parse_args()

    params = bench_llama1b_decode._benchmark.parameters
    if args.context_len is not None:
        params["context_len"] = args.context_len
    if args.batch is not None:
        params["batch"] = args.batch
    if args.page_size is not None:
        params["page_size"] = args.page_size
    if args.num_pages is not None:
        params["num_pages"] = args.num_pages

    print("=" * 100)
    print("Llama-3.2-1B Full-Model Decode Benchmark (Machete Megakernel)")
    print(f"  {NUM_LAYERS} layers, hidden={HIDDEN}, intermediate={INTERMEDIATE}")
    print(f"  Q heads={NUM_Q_HEADS}, KV heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}")
    print(f"  vocab={VOCAB_SIZE}, decode tile S={DECODE_S}")
    print("=" * 100)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.multi_processor_count} SMs)")
        print(f"SM90+: {is_sm90_or_newer()}")
    print(f"CUTLASS: {CUTLASS_AVAILABLE}")
    print(f"Fused decode matvec ops: {os.environ.get('LLAMA1B_DECODE_SM100_OPS', '1') != '0'}")
    print(f"Decode matvec arch: {os.environ.get('LLAMA1B_DECODE_ARCH', 'sm100')}")

    weight_bytes = _total_weight_bytes()
    print(f"\nModel weights: {weight_bytes / 1e9:.2f} GB bf16")
    for ctx in [128, 512, 1024, 2048, 4096]:
        total_bytes = _total_bytes_decode(ctx)
        total_flops = _total_flops_decode(ctx)
        print(
            f"  ctx={ctx:5d}: total={total_bytes / 1e9:.2f} GB, "
            f"AI={total_flops / total_bytes:.2f} FLOP/byte, "
            f"B200 memory limit={total_bytes / 8.0e12 * 1e6:.0f} us"
        )
    print(f"  Reference B200 target: under 680 us, theoretical around 3000 forwards/s.")

    bench_llama1b_decode._benchmark.run(mode="kernel", warmup=args.warmup, rep=args.rep)
