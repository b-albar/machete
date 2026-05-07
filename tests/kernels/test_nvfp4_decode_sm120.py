import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_nvfp4_quantize_dequantize_shapes_and_error():
    from machete.quantization import dequantize_nvfp4_weight, quantize_nvfp4_weight

    torch.manual_seed(0)
    weight = torch.randn(17, 64, device="cuda", dtype=torch.bfloat16)
    qweight = quantize_nvfp4_weight(weight, group_size=32)

    assert qweight.packed.shape == (17, 32)
    assert qweight.packed.dtype == torch.uint8
    assert qweight.scales.shape == (17, 2)
    assert qweight.scales.dtype == torch.float16

    deq = dequantize_nvfp4_weight(qweight)
    assert deq.shape == weight.shape
    assert torch.isfinite(deq).all()
    assert (deq - weight.float()).abs().max() < 0.8


def test_sm120_nvfp4_matvec_matches_dequantized_reference():
    from machete.kernels.decode_matvec import MatvecNvfp4Sm120Op
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.quantization import dequantize_nvfp4_weight, quantize_nvfp4_weight

    major, _minor = torch.cuda.get_device_capability()
    if major < 12:
        pytest.skip("SM120 required")

    torch.manual_seed(1)
    batch, seq, k_dim, out_dim = 1, 16, 64, 16
    a = torch.randn(batch, seq, k_dim, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(out_dim, k_dim, device="cuda", dtype=torch.bfloat16) * 0.1
    qweight = quantize_nvfp4_weight(weight, group_size=32)
    y = torch.empty(batch, seq, out_dim, device="cuda", dtype=torch.bfloat16)

    ops = MatvecNvfp4Sm120Op.schedule(
        a=a,
        weight_packed=qweight.packed,
        weight_scales=qweight.scales,
        y=y,
        tile_sizes={"S": seq, "O": 16},
        page_size=49152,
        group_size=32,
    )
    kernel = Megakernel(
        ops,
        config=MegakernelConfig(
            threads_per_block=128,
            page_size=49152,
            num_pages=1,
        ),
    )
    kernel.run()
    torch.cuda.synchronize()

    ref = torch.matmul(a.float(), dequantize_nvfp4_weight(qweight).float().t())
    torch.testing.assert_close(y.float(), ref, atol=5e-3, rtol=5e-3)


def test_sm120_nvfp4_pair_matvec_matches_dequantized_reference():
    from machete.kernels.decode_matvec import MatvecPairNvfp4Sm120Op
    from machete.megakernel import Megakernel, MegakernelConfig, TileRange
    from machete.quantization import dequantize_nvfp4_weight, quantize_nvfp4_weight

    major, _minor = torch.cuda.get_device_capability()
    if major < 12:
        pytest.skip("SM120 required")

    torch.manual_seed(17)
    batch, seq, k_dim, out_dim = 1, 1, 64, 32
    a = torch.randn(batch, seq, k_dim, device="cuda", dtype=torch.bfloat16)
    weight0 = torch.randn(out_dim, k_dim, device="cuda", dtype=torch.bfloat16) * 0.1
    weight1 = torch.randn(out_dim, k_dim, device="cuda", dtype=torch.bfloat16) * 0.1
    qweight0 = quantize_nvfp4_weight(weight0, group_size=32)
    qweight1 = quantize_nvfp4_weight(weight1, group_size=32)
    y0 = torch.empty(batch, seq, out_dim, device="cuda", dtype=torch.bfloat16)
    y1 = torch.empty_like(y0)

    ops = MatvecPairNvfp4Sm120Op.schedule(
        a=a,
        weight0_packed=qweight0.packed,
        weight0_scales=qweight0.scales,
        weight1_packed=qweight1.packed,
        weight1_scales=qweight1.scales,
        y0=y0,
        y1=y1,
        tile_sizes={"S": seq, "O": 16},
        page_size=49152,
        group_size=32,
        tile_range=TileRange.coalesced("O", block_size=2),
    )
    assert ops[0].static_dims["pipeline_coalesce_ranges"] is True
    assert ops[0].static_dims["pipeline_range_axis"] == ops[0].dim_names["O"]
    assert ops[0].static_dims["pipeline_range_block_size"] == 2
    kernel = Megakernel(
        ops,
        config=MegakernelConfig(
            threads_per_block=128,
            page_size=49152,
            num_pages=1,
        ),
    )
    kernel.run()
    torch.cuda.synchronize()

    ref0 = torch.matmul(a.float(), dequantize_nvfp4_weight(qweight0).float().t())
    ref1 = torch.matmul(a.float(), dequantize_nvfp4_weight(qweight1).float().t())
    torch.testing.assert_close(y0.float(), ref0, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(y1.float(), ref1, atol=5e-3, rtol=5e-3)


def test_sm120_nvfp4_quad_matvec_matches_dequantized_reference():
    from machete.kernels.decode_matvec import MatvecQuadNvfp4Sm120Op
    from machete.megakernel import Megakernel, MegakernelConfig
    from machete.quantization import dequantize_nvfp4_weight, quantize_nvfp4_weight

    major, _minor = torch.cuda.get_device_capability()
    if major < 12:
        pytest.skip("SM120 required")

    torch.manual_seed(19)
    batch, seq, k_dim, out_dim = 1, 1, 64, 32
    a = torch.randn(batch, seq, k_dim, device="cuda", dtype=torch.bfloat16)
    weights = [torch.randn(out_dim, k_dim, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(4)]
    qweights = [quantize_nvfp4_weight(w, group_size=32) for w in weights]
    ys = [torch.empty(batch, seq, out_dim, device="cuda", dtype=torch.bfloat16) for _ in range(4)]

    ops = MatvecQuadNvfp4Sm120Op.schedule(
        a=a,
        weight0_packed=qweights[0].packed,
        weight0_scales=qweights[0].scales,
        weight1_packed=qweights[1].packed,
        weight1_scales=qweights[1].scales,
        weight2_packed=qweights[2].packed,
        weight2_scales=qweights[2].scales,
        weight3_packed=qweights[3].packed,
        weight3_scales=qweights[3].scales,
        y0=ys[0],
        y1=ys[1],
        y2=ys[2],
        y3=ys[3],
        tile_sizes={"S": seq, "O": 16},
        page_size=49152,
        group_size=32,
    )
    kernel = Megakernel(
        ops,
        config=MegakernelConfig(
            threads_per_block=128,
            page_size=49152,
            num_pages=1,
        ),
    )
    kernel.run()
    torch.cuda.synchronize()

    for y, weight, qweight in zip(ys, weights, qweights):
        ref = torch.matmul(a.float(), dequantize_nvfp4_weight(qweight).float().t())
        torch.testing.assert_close(y.float(), ref, atol=5e-3, rtol=5e-3)


def test_direct_cute_nvfp4_final_head_matches_dequantized_reference():
    from machete.kernels.decode_matvec.direct_cute_lm_head import get_direct_cute_nvfp4_final
    from machete.quantization import dequantize_nvfp4_weight, quantize_nvfp4_weight

    torch.manual_seed(7)
    hidden = 64
    vocab = 256
    blocks = 4
    x = torch.randn(1, 1, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    norm_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    qweight = quantize_nvfp4_weight(weight, group_size=32)
    block_values = torch.empty(blocks, device="cuda", dtype=torch.float32)
    block_indices = torch.empty(blocks, device="cuda", dtype=torch.int32)
    top_values = torch.empty(1, 1, device="cuda", dtype=torch.float32)
    top_indices = torch.empty(1, 1, device="cuda", dtype=torch.int32)

    kernel = get_direct_cute_nvfp4_final(
        hidden_size=hidden,
        vocab_size=vocab,
        group_size=32,
        blocks=blocks,
        threads=128,
    )
    kernel(
        x,
        residual,
        norm_weight,
        qweight.packed,
        qweight.scales,
        block_values,
        block_indices,
        top_values,
        top_indices,
    )
    torch.cuda.synchronize()

    summed = x.float() + residual.float()
    h = summed * torch.rsqrt(summed.square().mean(dim=-1, keepdim=True) + 1e-5) * norm_weight.float()
    ref = torch.matmul(h, dequantize_nvfp4_weight(qweight).float().t())
    ref_values, ref_indices = ref.max(dim=-1)
    torch.testing.assert_close(top_values, ref_values, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(top_indices, ref_indices.to(torch.int32), atol=0, rtol=0)
