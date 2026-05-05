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
