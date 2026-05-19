# Copyright (c) 2025, Machete Authors
"""Focused tests for Llama-1B SM120 decode helpers."""

import pytest
import torch

from machete.kernels.llama1b import (
    LLAMA1B_HIDDEN,
    LLAMA1B_SM120_FINAL_MATVEC_BLOCK,
    LLAMA1B_SM120_QKV_HEAD_BLOCK,
    LLAMA1B_SM120_THREADS_PER_BLOCK,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _require_sm120():
    major, _minor = torch.cuda.get_device_capability()
    if major < 12:
        pytest.skip("SM120 required")


def _run_ops(ops, page_size=32768):
    from machete.megakernel import Megakernel, MegakernelConfig

    kernel = Megakernel(
        ops,
        config=MegakernelConfig(
            threads_per_block=LLAMA1B_SM120_THREADS_PER_BLOCK,
            page_size=page_size,
            loader_idle_sleep_ns=0,
            mma_reg_count=96,
        ),
    )
    kernel.run(validate=False)
    torch.cuda.synchronize()


def _top1_buffers(batch, seq_len, vocab, device="cuda"):
    vocab_tiles = (vocab + LLAMA1B_SM120_FINAL_MATVEC_BLOCK - 1) // LLAMA1B_SM120_FINAL_MATVEC_BLOCK
    partial_values = torch.empty(batch, seq_len, vocab_tiles, device=device, dtype=torch.float32)
    partial_indices = torch.empty_like(partial_values, dtype=torch.int32)
    top_values = torch.empty(batch, seq_len, device=device, dtype=torch.float32)
    top_indices = torch.empty(batch, seq_len, device=device, dtype=torch.int32)
    return partial_values, partial_indices, top_values, top_indices


def test_llama1b_sm120_fused_upgate_matches_reference():
    from machete.kernels.llama1b import Llama1BRmsUpGateSiluSm120Op
    from machete.megakernel import Megakernel, MegakernelConfig

    major, _minor = torch.cuda.get_device_capability()
    if major < 12:
        pytest.skip("SM120 required")

    torch.manual_seed(1)
    dtype = torch.bfloat16
    batch, seq_len, hidden, intermediate = 1, 1, 512, 16
    x = torch.randn(batch, seq_len, hidden, device="cuda", dtype=dtype)
    norm_weight = torch.randn(hidden, device="cuda", dtype=dtype)
    up_weight = torch.randn(intermediate, hidden, device="cuda", dtype=dtype) * 0.02
    gate_weight = torch.randn(intermediate, hidden, device="cuda", dtype=dtype) * 0.02
    y = torch.empty(batch, seq_len, intermediate, device="cuda", dtype=dtype)

    ops = Llama1BRmsUpGateSiluSm120Op.schedule(
        x=x,
        norm_weight=norm_weight,
        up_weight=up_weight,
        gate_weight=gate_weight,
        y=y,
        tile_sizes={"S": 1, "O": 8},
        page_size=32768,
        eps=1e-5,
    )
    assert ops[0].tile_sizes["O"] == 8

    kernel = Megakernel(
        ops,
        config=MegakernelConfig(
            threads_per_block=LLAMA1B_SM120_THREADS_PER_BLOCK,
            page_size=32768,
            num_pages=3,
            loader_idle_sleep_ns=0,
            mma_reg_count=96,
        ),
    )
    assert kernel._layout.num_slots > kernel._layout.num_pages
    kernel.run(validate=False)
    torch.cuda.synchronize()

    x_norm = torch.nn.functional.rms_norm(
        x.float(),
        (hidden,),
        norm_weight.float(),
        1e-5,
    )
    ref = (
        torch.nn.functional.silu(torch.matmul(x_norm, gate_weight.float().t()))
        * torch.matmul(x_norm, up_weight.float().t())
    ).to(dtype)
    torch.testing.assert_close(y, ref, atol=1e-3, rtol=1e-3)


def test_llama1b_sm120_grouped_attention_matches_reference():
    from machete.kernels.llama1b import Llama1BDecodeAttentionSm120Op
    from machete.megakernel import Megakernel, MegakernelConfig

    major, _minor = torch.cuda.get_device_capability()
    if major < 12:
        pytest.skip("SM120 required")

    torch.manual_seed(2)
    dtype = torch.bfloat16
    batch, seq_len, q_heads, kv_heads, head_dim = 1, 1, 8, 2, 64
    cache_len = 17
    kv_group_size = q_heads // kv_heads
    q = torch.randn(batch, seq_len, q_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, cache_len, kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, cache_len, kv_heads, head_dim, device="cuda", dtype=dtype)
    o = torch.empty(batch, seq_len, q_heads, head_dim, device="cuda", dtype=dtype)

    ops = Llama1BDecodeAttentionSm120Op.schedule(
        q=q,
        k=k,
        v=v,
        o=o,
        kv_group_size=kv_group_size,
    )
    assert len(ops) == 1
    assert ops[0].tile_sizes["H_kv"] == 1

    kernel = Megakernel(
        ops,
        config=MegakernelConfig(
            threads_per_block=LLAMA1B_SM120_THREADS_PER_BLOCK,
            page_size=32768,
            loader_idle_sleep_ns=0,
            mma_reg_count=96,
        ),
    )
    kernel.run(validate=False)
    torch.cuda.synchronize()

    q_ref = q.float().squeeze(1)
    k_ref = k.float().repeat_interleave(kv_group_size, dim=2).squeeze(0).transpose(0, 1)
    v_ref = v.float().repeat_interleave(kv_group_size, dim=2).squeeze(0).transpose(0, 1)
    scores = torch.einsum("hd,hnd->hn", q_ref[0], k_ref) * (head_dim ** -0.5)
    probs = torch.softmax(scores, dim=-1)
    ref = torch.einsum("hn,hnd->hd", probs, v_ref).view(batch, seq_len, q_heads, head_dim).to(dtype)
    torch.testing.assert_close(o, ref, atol=1e-2, rtol=1e-2)


def test_llama1b_sm120_split_attention_matches_reference():
    from machete.kernels.llama1b import schedule_llama1b_decode_attention_sm120
    from machete.megakernel import Megakernel, MegakernelConfig

    major, _minor = torch.cuda.get_device_capability()
    if major < 12:
        pytest.skip("SM120 required")

    torch.manual_seed(3)
    dtype = torch.bfloat16
    batch, seq_len, q_heads, kv_heads, head_dim = 1, 1, 8, 2, 64
    cache_len = 33
    kv_group_size = q_heads // kv_heads
    q = torch.randn(batch, seq_len, q_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, cache_len, kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, cache_len, kv_heads, head_dim, device="cuda", dtype=dtype)
    o = torch.empty(batch, seq_len, q_heads, head_dim, device="cuda", dtype=dtype)

    ops, keep_alive = schedule_llama1b_decode_attention_sm120(
        q=q,
        k=k,
        v=v,
        o=o,
        kv_group_size=kv_group_size,
        num_splits=4,
    )
    assert len(ops) == 2

    kernel = Megakernel(
        ops,
        config=MegakernelConfig(
            threads_per_block=LLAMA1B_SM120_THREADS_PER_BLOCK,
            page_size=32768,
            loader_idle_sleep_ns=0,
            mma_reg_count=96,
        ),
    )
    keep_alive.append(kernel)
    kernel.run(validate=False)
    torch.cuda.synchronize()

    q_ref = q.float().squeeze(1)
    k_ref = k.float().repeat_interleave(kv_group_size, dim=2).squeeze(0).transpose(0, 1)
    v_ref = v.float().repeat_interleave(kv_group_size, dim=2).squeeze(0).transpose(0, 1)
    scores = torch.einsum("hd,hnd->hn", q_ref[0], k_ref) * (head_dim ** -0.5)
    probs = torch.softmax(scores, dim=-1)
    ref = torch.einsum("hn,hnd->hd", probs, v_ref).view(batch, seq_len, q_heads, head_dim).to(dtype)
    torch.testing.assert_close(o, ref, atol=1e-2, rtol=1e-2)


def test_llama1b_sm120_fused_kv_cache_matches_split_ops():
    from machete.kernels.llama1b import (
        Llama1BRmsKCacheSm120Op,
        Llama1BRmsKVCacheSm120Op,
        Llama1BRmsVCacheSm120Op,
    )
    from machete.megakernel import Megakernel, MegakernelConfig

    major, _minor = torch.cuda.get_device_capability()
    if major < 12:
        pytest.skip("SM120 required")

    torch.manual_seed(4)
    dtype = torch.bfloat16
    batch, seq_len, hidden, kv_dim, kv_heads, head_dim = 1, 1, 512, 128, 2, 64
    cache_len = 5
    cache_pos = 2
    x = torch.randn(batch, seq_len, hidden, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    norm_weight = torch.randn(hidden, device="cuda", dtype=dtype)
    k_weight = torch.randn(kv_dim, hidden, device="cuda", dtype=dtype) * 0.02
    v_weight = torch.randn(kv_dim, hidden, device="cuda", dtype=dtype) * 0.02
    cos = torch.randn(seq_len, head_dim, device="cuda", dtype=dtype)
    sin = torch.randn(seq_len, head_dim, device="cuda", dtype=dtype)
    k_fused = torch.empty(batch, cache_len, kv_heads, head_dim, device="cuda", dtype=dtype)
    v_fused = torch.empty_like(k_fused)
    k_split = torch.empty_like(k_fused)
    v_split = torch.empty_like(k_fused)

    fused_ops = Llama1BRmsKVCacheSm120Op.schedule(
        x=x,
        residual_in=residual,
        norm_weight=norm_weight,
        k_weight=k_weight,
        v_weight=v_weight,
        cos=cos,
        sin=sin,
        k_cache=k_fused,
        v_cache=v_fused,
        cache_pos=cache_pos,
        tile_sizes={"S": 1, "O": LLAMA1B_SM120_QKV_HEAD_BLOCK},
        page_size=32768,
    )
    fused = Megakernel(
        fused_ops,
        config=MegakernelConfig(threads_per_block=LLAMA1B_SM120_THREADS_PER_BLOCK, page_size=32768, num_pages=3, loader_idle_sleep_ns=0, mma_reg_count=96),
    )
    fused.run(validate=False)

    split_ops = []
    split_ops += Llama1BRmsKCacheSm120Op.schedule(
        x=x,
        residual_in=residual,
        norm_weight=norm_weight,
        weight=k_weight,
        cos=cos,
        sin=sin,
        dst_cache=k_split,
        cache_pos=cache_pos,
        tile_sizes={"S": 1, "O": LLAMA1B_SM120_QKV_HEAD_BLOCK},
        page_size=32768,
    )
    split_ops += Llama1BRmsVCacheSm120Op.schedule(
        x=x,
        residual_in=residual,
        norm_weight=norm_weight,
        weight=v_weight,
        cos=cos,
        sin=sin,
        dst_cache=v_split,
        cache_pos=cache_pos,
        tile_sizes={"S": 1, "O": LLAMA1B_SM120_QKV_HEAD_BLOCK},
        page_size=32768,
    )
    split = Megakernel(
        split_ops,
        config=MegakernelConfig(threads_per_block=LLAMA1B_SM120_THREADS_PER_BLOCK, page_size=32768, num_pages=3, loader_idle_sleep_ns=0, mma_reg_count=96),
    )
    split.run(validate=False)
    torch.cuda.synchronize()

    torch.testing.assert_close(k_fused[:, cache_pos], k_split[:, cache_pos], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(v_fused[:, cache_pos], v_split[:, cache_pos], atol=1e-3, rtol=1e-3)


def test_llama1b_sm120_kstream_final_head_matches_reference():
    from machete.kernels.llama1b import LLAMA1B_HIDDEN, Llama1BFinalRmsLmHeadKStreamSm120Op
    from machete.megakernel import Megakernel, MegakernelConfig

    major, _minor = torch.cuda.get_device_capability()
    if major < 12:
        pytest.skip("SM120 required")

    torch.manual_seed(5)
    dtype = torch.bfloat16
    batch, seq_len, vocab = 1, 1, 128
    x = torch.randn(batch, seq_len, LLAMA1B_HIDDEN, device="cuda", dtype=dtype)
    norm_weight = torch.randn(LLAMA1B_HIDDEN, device="cuda", dtype=dtype)
    weight = torch.randn(vocab, LLAMA1B_HIDDEN, device="cuda", dtype=dtype) * 0.02
    logits = torch.empty(batch, seq_len, vocab, device="cuda", dtype=dtype)

    ops = Llama1BFinalRmsLmHeadKStreamSm120Op.schedule(
        x=x,
        norm_weight=norm_weight,
        weight=weight,
        logits=logits,
        tile_sizes={"S": 1, "O": 16},
        page_size=32768,
    )
    page_size = max(op.static_dims.get("page_size", 32768) for op in ops)
    kernel = Megakernel(
        ops,
        config=MegakernelConfig(
            threads_per_block=LLAMA1B_SM120_THREADS_PER_BLOCK,
            page_size=page_size,
            loader_idle_sleep_ns=0,
            mma_reg_count=96,
        ),
    )
    kernel.run(validate=False)
    torch.cuda.synchronize()

    ref = torch.nn.functional.rms_norm(
        x.float(),
        (LLAMA1B_HIDDEN,),
        norm_weight.float(),
        1e-5,
    ) @ weight.float().t()
    torch.testing.assert_close(logits.float(), ref, atol=3e-2, rtol=3e-2)


def test_llama1b_sm120_kstream_final_head_top1_matches_reference():
    from machete.kernels.llama1b import (
        Llama1BFinalRmsTop1PartialLmHeadKStreamSm120Op,
        Llama1BReduceTop1PartialsSm120Op,
    )

    _require_sm120()

    torch.manual_seed(6)
    dtype = torch.bfloat16
    batch, seq_len, vocab = 1, 1, 128
    x = torch.randn(batch, seq_len, LLAMA1B_HIDDEN, device="cuda", dtype=dtype)
    norm_weight = torch.randn(LLAMA1B_HIDDEN, device="cuda", dtype=dtype)
    weight = torch.randn(vocab, LLAMA1B_HIDDEN, device="cuda", dtype=dtype) * 0.02
    partial_values, partial_indices, top_values, top_indices = _top1_buffers(batch, seq_len, vocab)

    ops = Llama1BFinalRmsTop1PartialLmHeadKStreamSm120Op.schedule(
        x=x,
        norm_weight=norm_weight,
        weight=weight,
        partial_values=partial_values,
        partial_indices=partial_indices,
        tile_sizes={"S": 1, "P": 1},
        page_size=32768,
        vocab_block=LLAMA1B_SM120_FINAL_MATVEC_BLOCK,
    )
    ops += Llama1BReduceTop1PartialsSm120Op.schedule(
        partial_values=partial_values,
        partial_indices=partial_indices,
        top_values=top_values,
        top_indices=top_indices,
        tile_sizes={"S": 1},
        page_size=32768,
    )
    _run_ops(ops)

    ref = torch.nn.functional.rms_norm(
        x.float(),
        (LLAMA1B_HIDDEN,),
        norm_weight.float(),
        1e-5,
    ) @ weight.float().t()
    ref_values, ref_indices = ref.max(dim=-1)
    torch.testing.assert_close(top_values, ref_values, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(top_indices, ref_indices.to(torch.int32), atol=0, rtol=0)


def test_llama1b_sm120_kstream_final_add_head_top1_matches_reference():
    from machete.kernels.llama1b import (
        Llama1BFinalAddRmsTop1PartialLmHeadKStreamSm120Op,
        Llama1BReduceTop1PartialsSm120Op,
    )

    _require_sm120()

    torch.manual_seed(7)
    dtype = torch.bfloat16
    batch, seq_len, vocab = 1, 1, 128
    x = torch.randn(batch, seq_len, LLAMA1B_HIDDEN, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    norm_weight = torch.randn(LLAMA1B_HIDDEN, device="cuda", dtype=dtype)
    weight = torch.randn(vocab, LLAMA1B_HIDDEN, device="cuda", dtype=dtype) * 0.02
    partial_values, partial_indices, top_values, top_indices = _top1_buffers(batch, seq_len, vocab)

    ops = Llama1BFinalAddRmsTop1PartialLmHeadKStreamSm120Op.schedule(
        x=x,
        residual_in=residual,
        norm_weight=norm_weight,
        weight=weight,
        partial_values=partial_values,
        partial_indices=partial_indices,
        tile_sizes={"S": 1, "P": 1},
        page_size=32768,
        vocab_block=LLAMA1B_SM120_FINAL_MATVEC_BLOCK,
    )
    ops += Llama1BReduceTop1PartialsSm120Op.schedule(
        partial_values=partial_values,
        partial_indices=partial_indices,
        top_values=top_values,
        top_indices=top_indices,
        tile_sizes={"S": 1},
        page_size=32768,
    )
    _run_ops(ops)

    ref = torch.nn.functional.rms_norm(
        (x + residual).float(),
        (LLAMA1B_HIDDEN,),
        norm_weight.float(),
        1e-5,
    ) @ weight.float().t()
    ref_values, ref_indices = ref.max(dim=-1)
    torch.testing.assert_close(top_values, ref_values, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(top_indices, ref_indices.to(torch.int32), atol=0, rtol=0)


def test_llama1b_sm120_fused_qkv_cache_matches_split_ops():
    from machete.kernels.llama1b import (
        Llama1BRmsKCacheSm120Op,
        Llama1BRmsQKVCacheSm120Op,
        Llama1BRmsQSm120Op,
        Llama1BRmsVCacheSm120Op,
    )
    from machete.megakernel import Megakernel, MegakernelConfig

    major, _minor = torch.cuda.get_device_capability()
    if major < 12:
        pytest.skip("SM120 required")

    torch.manual_seed(5)
    dtype = torch.bfloat16
    batch, seq_len, hidden, q_dim, kv_dim, kv_heads, head_dim = 1, 1, 512, 256, 128, 2, 64
    cache_len = 5
    cache_pos = 2
    x = torch.randn(batch, seq_len, hidden, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    norm_weight = torch.randn(hidden, device="cuda", dtype=dtype)
    q_weight = torch.randn(q_dim, hidden, device="cuda", dtype=dtype) * 0.02
    k_weight = torch.randn(kv_dim, hidden, device="cuda", dtype=dtype) * 0.02
    v_weight = torch.randn(kv_dim, hidden, device="cuda", dtype=dtype) * 0.02
    kv_group_size = q_dim // kv_dim
    qkv_weight = torch.cat(
        [
            part
            for kv_head in range(kv_heads)
            for part in (
                q_weight[kv_head * kv_group_size * head_dim : (kv_head + 1) * kv_group_size * head_dim],
                k_weight[kv_head * head_dim : (kv_head + 1) * head_dim],
                v_weight[kv_head * head_dim : (kv_head + 1) * head_dim],
            )
        ],
        dim=0,
    )
    cos = torch.randn(seq_len, head_dim, device="cuda", dtype=dtype)
    sin = torch.randn(seq_len, head_dim, device="cuda", dtype=dtype)
    q_fused = torch.empty(batch, seq_len, q_dim, device="cuda", dtype=dtype)
    q_split = torch.empty_like(q_fused)
    residual_fused = torch.empty_like(x)
    residual_split = torch.empty_like(x)
    k_fused = torch.empty(batch, cache_len, kv_heads, head_dim, device="cuda", dtype=dtype)
    v_fused = torch.empty_like(k_fused)
    k_split = torch.empty_like(k_fused)
    v_split = torch.empty_like(k_fused)

    fused_ops = Llama1BRmsQKVCacheSm120Op.schedule(
        x=x,
        residual_in=residual,
        norm_weight=norm_weight,
        weight=qkv_weight,
        cos=cos,
        sin=sin,
        residual_out=residual_fused,
        q=q_fused,
        k_cache=k_fused,
        v_cache=v_fused,
        cache_pos=cache_pos,
        tile_sizes={"S": 1, "O": LLAMA1B_SM120_QKV_HEAD_BLOCK},
        kv_group_size=kv_group_size,
        page_size=32768,
    )
    fused = Megakernel(
        fused_ops,
        config=MegakernelConfig(threads_per_block=LLAMA1B_SM120_THREADS_PER_BLOCK, page_size=32768, num_pages=3, loader_idle_sleep_ns=0, mma_reg_count=96),
    )
    fused.run(validate=False)

    split_ops = []
    split_ops += Llama1BRmsQSm120Op.schedule(
        x=x,
        residual_in=residual,
        norm_weight=norm_weight,
        weight=q_weight,
        cos=cos,
        sin=sin,
        residual_out=residual_split,
        q=q_split,
        tile_sizes={"S": 1, "O": LLAMA1B_SM120_QKV_HEAD_BLOCK},
        page_size=32768,
    )
    split_ops += Llama1BRmsKCacheSm120Op.schedule(
        x=x,
        residual_in=residual,
        norm_weight=norm_weight,
        weight=k_weight,
        cos=cos,
        sin=sin,
        dst_cache=k_split,
        cache_pos=cache_pos,
        tile_sizes={"S": 1, "O": LLAMA1B_SM120_QKV_HEAD_BLOCK},
        page_size=32768,
    )
    split_ops += Llama1BRmsVCacheSm120Op.schedule(
        x=x,
        residual_in=residual,
        norm_weight=norm_weight,
        weight=v_weight,
        cos=cos,
        sin=sin,
        dst_cache=v_split,
        cache_pos=cache_pos,
        tile_sizes={"S": 1, "O": LLAMA1B_SM120_QKV_HEAD_BLOCK},
        page_size=32768,
    )
    split = Megakernel(
        split_ops,
        config=MegakernelConfig(threads_per_block=LLAMA1B_SM120_THREADS_PER_BLOCK, page_size=32768, num_pages=3, loader_idle_sleep_ns=0, mma_reg_count=96),
    )
    split.run(validate=False)
    torch.cuda.synchronize()

    torch.testing.assert_close(q_fused, q_split, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(residual_fused, residual_split, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(k_fused[:, cache_pos], k_split[:, cache_pos], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(v_fused[:, cache_pos], v_split[:, cache_pos], atol=1e-3, rtol=1e-3)
