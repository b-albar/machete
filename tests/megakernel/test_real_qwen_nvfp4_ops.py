import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _qweight(rows, cols, group_size=32):
    from machete.quantization import NVFP4Tensor

    return NVFP4Tensor(
        packed=torch.empty(rows, cols // 2, device="cuda", dtype=torch.uint8),
        scales=torch.empty(rows, cols // group_size, device="cuda", dtype=torch.float16),
        group_size=group_size,
        rows=rows,
        cols=cols,
    )


def test_real_qwen_full_attention_nvfp4_schedule_uses_native_ops():
    from machete.kernels.decode_matvec import (
        MatvecNvfp4Sm120Op,
        MatvecResidualNvfp4Sm120Op,
        RmsAddNormSm120Op,
        RmsGateUpSiluNvfp4Sm120Op,
    )
    from machete.kernels.qwen3_5_sm120 import (
        Qwen3_5QkvRopeCacheSm120Op,
        QWEN3_5_REAL_HEAD_DIM,
        QWEN3_5_REAL_HIDDEN,
        QWEN3_5_REAL_INTERMEDIATE,
        QWEN3_5_REAL_KV_DIM,
        QWEN3_5_REAL_NUM_KV_HEADS,
        QWEN3_5_REAL_Q_DIM,
        QWEN3_5_REAL_ROTARY_D2,
        schedule_qwen3_5_full_attention_nvfp4_sm120,
    )

    dtype = torch.bfloat16
    batch, seq_len, cache_pos = 1, 1, 8
    layer_idx = 3
    pfx = f"layer.{layer_idx}"
    weights = {
        "cos": torch.empty(128, QWEN3_5_REAL_ROTARY_D2, device="cuda", dtype=dtype),
        "sin": torch.empty(128, QWEN3_5_REAL_ROTARY_D2, device="cuda", dtype=dtype),
        f"{pfx}.attn_norm": torch.empty(QWEN3_5_REAL_HIDDEN, device="cuda", dtype=dtype),
        f"{pfx}.mlp_norm": torch.empty(QWEN3_5_REAL_HIDDEN, device="cuda", dtype=dtype),
        f"{pfx}.W_q_nvfp4": _qweight(QWEN3_5_REAL_Q_DIM, QWEN3_5_REAL_HIDDEN),
        f"{pfx}.W_k_nvfp4": _qweight(QWEN3_5_REAL_KV_DIM, QWEN3_5_REAL_HIDDEN),
        f"{pfx}.W_v_nvfp4": _qweight(QWEN3_5_REAL_KV_DIM, QWEN3_5_REAL_HIDDEN),
        f"{pfx}.W_o_nvfp4": _qweight(QWEN3_5_REAL_HIDDEN, QWEN3_5_REAL_Q_DIM),
        f"{pfx}.W_gate_nvfp4": _qweight(QWEN3_5_REAL_INTERMEDIATE, QWEN3_5_REAL_HIDDEN),
        f"{pfx}.W_up_nvfp4": _qweight(QWEN3_5_REAL_INTERMEDIATE, QWEN3_5_REAL_HIDDEN),
        f"{pfx}.W_down_nvfp4": _qweight(QWEN3_5_REAL_HIDDEN, QWEN3_5_REAL_INTERMEDIATE),
    }
    x = torch.empty(batch, seq_len, QWEN3_5_REAL_HIDDEN, device="cuda", dtype=dtype)
    residual = torch.empty_like(x)
    x_out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    q_buf = torch.empty(batch, seq_len, QWEN3_5_REAL_Q_DIM, device="cuda", dtype=dtype)
    attn_out = torch.empty_like(q_buf)
    norm_buf = torch.empty(batch, seq_len, QWEN3_5_REAL_HIDDEN, device="cuda", dtype=dtype)
    mlp_h = torch.empty(batch, seq_len, QWEN3_5_REAL_INTERMEDIATE, device="cuda", dtype=dtype)
    k_cache = torch.empty(
        batch,
        128,
        QWEN3_5_REAL_NUM_KV_HEADS,
        QWEN3_5_REAL_HEAD_DIM,
        device="cuda",
        dtype=dtype,
    )
    v_cache = torch.empty_like(k_cache)

    layer = schedule_qwen3_5_full_attention_nvfp4_sm120(
        layer_idx=layer_idx,
        batch=batch,
        seq_len=seq_len,
        cache_pos=cache_pos,
        weights=weights,
        k_cache=k_cache,
        v_cache=v_cache,
        x_in=x,
        residual_in=residual,
        x_out=x_out,
        residual_out=residual_out,
        norm_buf=norm_buf,
        q_buf=q_buf,
        attn_out_buf=attn_out,
        mlp_h_buf=mlp_h,
    )

    op_classes = [op.op_cls for op in layer.ops]
    assert op_classes[0] is RmsAddNormSm120Op
    assert op_classes.count(MatvecNvfp4Sm120Op) >= 4
    assert Qwen3_5QkvRopeCacheSm120Op in op_classes
    assert MatvecResidualNvfp4Sm120Op in op_classes
    assert RmsGateUpSiluNvfp4Sm120Op in op_classes
    post = layer.ops[4]
    assert post.op_cls is Qwen3_5QkvRopeCacheSm120Op
    assert post.static_dims["head_dim"] == QWEN3_5_REAL_HEAD_DIM


def test_real_qwen_qkv_rope_cache_postprocess_matches_reference():
    from machete.kernels.qwen3_5_sm120 import Qwen3_5QkvRopeCacheSm120Op
    from machete.megakernel import Megakernel, MegakernelConfig

    torch.manual_seed(31)
    dtype = torch.bfloat16
    batch, seq, q_heads, kv_heads, head_dim, d2 = 1, 1, 2, 1, 8, 2
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    cache_pos = 3

    q_raw = torch.randn(batch, seq, q_dim, device="cuda", dtype=dtype)
    kv_raw = torch.randn(batch, seq, q_dim, device="cuda", dtype=dtype)
    cos = torch.randn(seq, d2, device="cuda", dtype=dtype)
    sin = torch.randn(seq, d2, device="cuda", dtype=dtype)
    q = torch.empty_like(q_raw)
    k_cache = torch.zeros(batch, 8, kv_heads, head_dim, device="cuda", dtype=dtype)
    v_cache = torch.zeros_like(k_cache)

    kernel = Megakernel(
        Qwen3_5QkvRopeCacheSm120Op.schedule(
            q_raw=q_raw,
            kv_raw=kv_raw,
            cos=cos,
            sin=sin,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_pos=cache_pos,
            tile_sizes={"S": 1},
            page_size=32768,
        ),
        config=MegakernelConfig(num_sms=1, page_size=32768, threads_per_block=128),
    )
    kernel.run()
    torch.cuda.synchronize()

    q_ref = q_raw.float().clone()
    for head in range(q_heads):
        base = head * head_dim
        low = q_raw[0, 0, base : base + d2].float()
        high = q_raw[0, 0, base + d2 : base + 2 * d2].float()
        q_ref[0, 0, base : base + d2] = low * cos.float()[0] - high * sin.float()[0]
        q_ref[0, 0, base + d2 : base + 2 * d2] = high * cos.float()[0] + low * sin.float()[0]

    k_ref = kv_raw[0, 0, :kv_dim].float().clone()
    low = kv_raw[0, 0, :d2].float()
    high = kv_raw[0, 0, d2 : 2 * d2].float()
    k_ref[:d2] = low * cos.float()[0] - high * sin.float()[0]
    k_ref[d2 : 2 * d2] = high * cos.float()[0] + low * sin.float()[0]
    v_ref = kv_raw[0, 0, kv_dim : 2 * kv_dim].float()

    torch.testing.assert_close(q.float(), q_ref.to(dtype).float(), rtol=0, atol=0)
    torch.testing.assert_close(k_cache[0, cache_pos, 0].float(), k_ref.to(dtype).float(), rtol=0, atol=0)
    torch.testing.assert_close(v_cache[0, cache_pos, 0].float(), v_ref.to(dtype).float(), rtol=0, atol=0)


def test_real_qwen_single_token_attention_matches_reference():
    from machete.kernels.qwen3_5_sm120 import Qwen3_5SingleTokenAttentionSm120Op
    from machete.megakernel import Megakernel, MegakernelConfig

    torch.manual_seed(19)
    dtype = torch.bfloat16
    batch, seq, q_heads, kv_heads, head_dim, context = 1, 1, 4, 2, 16, 13
    kv_group_size = q_heads // kv_heads
    q = torch.randn(batch, seq, q_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, context, kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    o = torch.empty(batch, seq, q_heads, head_dim, device="cuda", dtype=dtype)

    kernel = Megakernel(
        Qwen3_5SingleTokenAttentionSm120Op.schedule(
            q=q,
            k=k,
            v=v,
            o=o,
            kv_group_size=kv_group_size,
            page_size=32768,
        ),
        config=MegakernelConfig(num_sms=1, page_size=32768, threads_per_block=128),
    )
    kernel.run()
    torch.cuda.synchronize()

    ref = torch.empty_like(o, dtype=torch.float32)
    for qh in range(q_heads):
        kvh = qh // kv_group_size
        scores = torch.einsum(
            "d,td->t",
            q[0, 0, qh].float(),
            k[0, :, kvh].float(),
        ) * 0.0625
        probs = torch.softmax(scores, dim=0)
        ref[0, 0, qh] = torch.einsum("t,td->d", probs, v[0, :, kvh].float())

    torch.testing.assert_close(o.float(), ref, rtol=2e-2, atol=2e-2)


def test_real_qwen_flash_decode_attention_matches_reference():
    from machete.kernels.qwen3_5_sm120 import schedule_qwen3_5_flash_decode_attention_sm120
    from machete.megakernel import Megakernel, MegakernelConfig

    torch.manual_seed(23)
    dtype = torch.bfloat16
    batch, seq, q_heads, kv_heads, head_dim, context = 1, 1, 8, 2, 256, 128
    kv_group_size = q_heads // kv_heads
    q = torch.randn(batch, seq, q_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, context, kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    o = torch.empty(batch, seq, q_heads, head_dim, device="cuda", dtype=dtype)

    ops, keep = schedule_qwen3_5_flash_decode_attention_sm120(
        q=q,
        k=k,
        v=v,
        o=o,
        kv_group_size=kv_group_size,
        page_size=32768,
    )
    kernel = Megakernel(
        ops,
        config=MegakernelConfig(num_sms=4, page_size=32768, threads_per_block=128),
    )
    kernel._keep_alive = keep
    kernel.run()
    torch.cuda.synchronize()

    ref = torch.empty_like(o, dtype=torch.float32)
    for qh in range(q_heads):
        kvh = qh // kv_group_size
        scores = torch.einsum(
            "d,td->t",
            q[0, 0, qh].float(),
            k[0, :, kvh].float(),
        ) * (1 / (head_dim**0.5))
        probs = torch.softmax(scores, dim=0)
        ref[0, 0, qh] = torch.einsum("t,td->d", probs, v[0, :, kvh].float())

    torch.testing.assert_close(o.float(), ref, rtol=2e-2, atol=2e-2)


def test_real_qwen_deltanet_nvfp4_schedule_uses_native_ops():
    from machete.kernels.decode_matvec import (
        MatvecNvfp4Sm120Op,
        MatvecResidualNvfp4Sm120Op,
        RmsAddNormSm120Op,
        RmsGateUpSiluNvfp4Sm120Op,
        RmsMatvecNvfp4Sm120Op,
        RmsReadMatvecNvfp4Sm120Op,
    )
    from machete.kernels.qwen3_5_sm120 import (
        Qwen3_5DeltaNetCoreSm120Op,
        QWEN3_5_REAL_HIDDEN,
        QWEN3_5_REAL_INTERMEDIATE,
        QWEN3_5_REAL_ROTARY_D2,
        schedule_qwen3_5_deltanet_nvfp4_sm120,
    )

    dtype = torch.bfloat16
    batch, seq_len = 1, 1
    layer_idx = 0
    pfx = f"layer.{layer_idx}"
    qk_size = 16 * 128
    v_size = 16 * 128
    conv_channels = 2 * qk_size + v_size
    weights = {
        "cos": torch.empty(128, QWEN3_5_REAL_ROTARY_D2, device="cuda", dtype=dtype),
        "sin": torch.empty(128, QWEN3_5_REAL_ROTARY_D2, device="cuda", dtype=dtype),
        f"{pfx}.attn_norm": torch.empty(QWEN3_5_REAL_HIDDEN, device="cuda", dtype=dtype),
        f"{pfx}.linear_norm": torch.empty(128, device="cuda", dtype=dtype),
        f"{pfx}.mlp_norm": torch.empty(QWEN3_5_REAL_HIDDEN, device="cuda", dtype=dtype),
        f"{pfx}.W_qkv_nvfp4": _qweight(conv_channels, QWEN3_5_REAL_HIDDEN),
        f"{pfx}.W_z_nvfp4": _qweight(v_size, QWEN3_5_REAL_HIDDEN),
        f"{pfx}.W_beta_nvfp4": _qweight(16, QWEN3_5_REAL_HIDDEN),
        f"{pfx}.W_alpha_nvfp4": _qweight(16, QWEN3_5_REAL_HIDDEN),
        f"{pfx}.conv_weight": torch.empty(conv_channels, 4, device="cuda", dtype=dtype),
        f"{pfx}.a_log": torch.empty(16, device="cuda", dtype=dtype),
        f"{pfx}.dt_bias": torch.empty(16, device="cuda", dtype=dtype),
        f"{pfx}.W_out_nvfp4": _qweight(QWEN3_5_REAL_HIDDEN, v_size),
        f"{pfx}.W_gate_nvfp4": _qweight(QWEN3_5_REAL_INTERMEDIATE, QWEN3_5_REAL_HIDDEN),
        f"{pfx}.W_up_nvfp4": _qweight(QWEN3_5_REAL_INTERMEDIATE, QWEN3_5_REAL_HIDDEN),
        f"{pfx}.W_down_nvfp4": _qweight(QWEN3_5_REAL_HIDDEN, QWEN3_5_REAL_INTERMEDIATE),
    }
    x = torch.empty(batch, seq_len, QWEN3_5_REAL_HIDDEN, device="cuda", dtype=dtype)
    residual = torch.empty_like(x)
    x_out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    qkv_buf = torch.empty(batch, seq_len, conv_channels, device="cuda", dtype=dtype)
    norm_buf = torch.empty(batch, seq_len, QWEN3_5_REAL_HIDDEN, device="cuda", dtype=dtype)
    z_buf = torch.empty(batch, seq_len, v_size, device="cuda", dtype=dtype)
    beta_buf = torch.empty(batch, seq_len, 16, device="cuda", dtype=dtype)
    alpha_buf = torch.empty_like(beta_buf)
    dn_out = torch.empty(batch, seq_len, v_size, device="cuda", dtype=dtype)
    mlp_h = torch.empty(batch, seq_len, QWEN3_5_REAL_INTERMEDIATE, device="cuda", dtype=dtype)
    dn_state = torch.empty(batch, 16, 128, 128, device="cuda", dtype=torch.float32)
    conv_buf = torch.empty(batch, conv_channels, 4, device="cuda", dtype=torch.float32)

    layer = schedule_qwen3_5_deltanet_nvfp4_sm120(
        layer_idx=layer_idx,
        batch=batch,
        seq_len=seq_len,
        weights=weights,
        x_in=x,
        residual_in=residual,
        x_out=x_out,
        residual_out=residual_out,
        norm_buf=norm_buf,
        qkv_buf=qkv_buf,
        z_buf=z_buf,
        beta_buf=beta_buf,
        alpha_buf=alpha_buf,
        dn_out_buf=dn_out,
        mlp_h_buf=mlp_h,
        dn_state=dn_state,
        conv_buf=conv_buf,
    )

    op_classes = [op.op_cls for op in layer.ops]
    assert op_classes[0] is RmsAddNormSm120Op
    assert op_classes.count(RmsMatvecNvfp4Sm120Op) == 0
    assert op_classes.count(RmsReadMatvecNvfp4Sm120Op) == 0
    assert op_classes.count(MatvecNvfp4Sm120Op) >= 5
    assert Qwen3_5DeltaNetCoreSm120Op in op_classes
    assert MatvecResidualNvfp4Sm120Op in op_classes
    assert RmsGateUpSiluNvfp4Sm120Op in op_classes
    assert op_classes[-1] is MatvecNvfp4Sm120Op
    assert layer.ops[5].op_cls is Qwen3_5DeltaNetCoreSm120Op


def test_real_qwen_deltanet_core_matches_reference():
    from machete.kernels.qwen3_5_sm120 import Qwen3_5DeltaNetCoreSm120Op
    from machete.megakernel import Megakernel, MegakernelConfig

    major, _minor = torch.cuda.get_device_capability()
    if major < 12:
        pytest.skip("SM120 required")

    torch.manual_seed(3)
    batch = seq = 1
    heads, dim = 16, 128
    value_size = heads * dim
    conv_channels = 3 * value_size
    qkv = torch.randn(batch, seq, conv_channels, device="cuda", dtype=torch.bfloat16) * 0.01
    z = torch.randn(batch, seq, value_size, device="cuda", dtype=torch.bfloat16) * 0.01
    beta = torch.randn(batch, seq, heads, device="cuda", dtype=torch.bfloat16) * 0.01
    alpha = torch.randn_like(beta)
    conv_weight = torch.randn(conv_channels, 4, device="cuda", dtype=torch.bfloat16) * 0.01
    a_log = torch.randn(heads, device="cuda", dtype=torch.bfloat16) * 0.01
    dt_bias = torch.randn(heads, device="cuda", dtype=torch.bfloat16) * 0.01
    norm_weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16) * 0.01 + 1
    dn_state = torch.randn(batch, heads, dim, dim, device="cuda", dtype=torch.float32) * 0.001
    conv_buf = torch.randn(batch, conv_channels, 4, device="cuda", dtype=torch.float32) * 0.001
    y = torch.empty(batch, seq, value_size, device="cuda", dtype=torch.bfloat16)

    ref_state = dn_state.clone()
    ref_conv = conv_buf.clone()
    ref_y = torch.empty_like(y, dtype=torch.float32)

    qkv_f = qkv.float()
    z_f = z.float()
    beta_f = beta.float()
    alpha_f = alpha.float()
    cw_f = conv_weight.float()
    a_log_f = a_log.float()
    dt_f = dt_bias.float()
    nw_f = norm_weight.float()
    for h in range(heads):
        parts = []
        for region in range(3):
            offset = region * value_size + h * dim
            vals = []
            for j in range(dim):
                ch = offset + j
                old = ref_conv[0, ch].clone()
                new_val = qkv_f[0, 0, ch]
                ref_conv[0, ch, 0] = old[1]
                ref_conv[0, ch, 1] = old[2]
                ref_conv[0, ch, 2] = old[3]
                ref_conv[0, ch, 3] = new_val
                co = old[1] * cw_f[ch, 0] + old[2] * cw_f[ch, 1] + old[3] * cw_f[ch, 2] + new_val * cw_f[ch, 3]
                vals.append(F.silu(co))
            parts.append(torch.stack(vals))
        q = parts[0] * torch.rsqrt(parts[0].square().sum() + 1e-6) * (dim ** -0.5)
        k = parts[1] * torch.rsqrt(parts[1].square().sum() + 1e-6)
        v = parts[2]
        beta_h = torch.sigmoid(beta_f[0, 0, h])
        softplus = F.softplus(alpha_f[0, 0, h] + dt_f[h])
        decay = torch.exp(-torch.exp(a_log_f[h]) * softplus)
        kq = torch.dot(k, q)
        head_y = []
        for j in range(dim):
            st = ref_state[0, h, j].clone()
            stk = torch.dot(st, k)
            sqv = torch.dot(st, q)
            err = (v[j] - stk) * beta_h
            head_y.append(decay * sqv + err * kq)
            ref_state[0, h, j] = st * decay + k * err
        head_y = torch.stack(head_y)
        rstd = torch.rsqrt(head_y.square().mean() + 1e-6)
        gate = F.silu(z_f[0, 0, h * dim : (h + 1) * dim])
        ref_y[0, 0, h * dim : (h + 1) * dim] = head_y * rstd * nw_f * gate

    ops = Qwen3_5DeltaNetCoreSm120Op.schedule(
        qkv=qkv,
        z=z,
        beta=beta,
        alpha=alpha,
        conv_weight=conv_weight,
        a_log=a_log,
        dt_bias=dt_bias,
        norm_weight=norm_weight,
        dn_state=dn_state,
        conv_buf=conv_buf,
        y=y,
        page_size=49152,
    )
    kernel = Megakernel(
        ops,
        config=MegakernelConfig(threads_per_block=128, page_size=49152, num_pages=1),
    )
    kernel.run()
    torch.cuda.synchronize()

    torch.testing.assert_close(y.float(), ref_y, atol=1e-4, rtol=2e-2)
    torch.testing.assert_close(conv_buf, ref_conv, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(dn_state, ref_state, atol=5e-4, rtol=5e-2)


def test_final_nvfp4_top1_lm_head_matches_dequantized_reference():
    from machete.kernels.decode_matvec import FinalRmsTop1LmHeadNvfp4Sm120Op
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.quantization import dequantize_nvfp4_weight, quantize_nvfp4_weight

    torch.manual_seed(11)
    batch, seq, hidden, vocab = 1, 1, 64, 256
    x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.bfloat16)
    norm_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    qweight = quantize_nvfp4_weight(weight, group_size=32)
    top_values = torch.empty(batch, seq, device="cuda", dtype=torch.float32)
    top_indices = torch.empty(batch, seq, device="cuda", dtype=torch.int32)

    ops = FinalRmsTop1LmHeadNvfp4Sm120Op.schedule(
        x=x,
        norm_weight=norm_weight,
        weight_packed=qweight.packed,
        weight_scales=qweight.scales,
        top_values=top_values,
        top_indices=top_indices,
        tile_sizes={"S": seq, "V": 16},
        page_size=8192,
        group_size=32,
    )
    kernel = Megakernel(
        ops,
        config=MegakernelConfig(
            num_sms=4,
            threads_per_block=128,
            page_size=8192,
            num_pages=1,
            per_sm_instruction_queues=True,
        ),
    )
    kernel.run()
    torch.cuda.synchronize()

    rstd = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-5)
    h = x.float() * rstd * norm_weight.float()
    ref = torch.matmul(h, dequantize_nvfp4_weight(qweight).float().t())
    ref_values, ref_indices = ref.max(dim=-1)
    torch.testing.assert_close(top_values, ref_values, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(top_indices, ref_indices.to(torch.int32), atol=0, rtol=0)


def test_real_qwen_full_decode_entry_is_explicit_about_required_buffers():
    from machete.kernels.qwen3_5_sm120 import schedule_qwen3_5_real_nvfp4_decode_sm120

    with pytest.raises(TypeError):
        schedule_qwen3_5_real_nvfp4_decode_sm120()
