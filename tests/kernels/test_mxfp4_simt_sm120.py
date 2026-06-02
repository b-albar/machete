import torch


def test_mxfp4_simt_weight_uses_e8m0_scales():
    from machete.kernels.qwen_3_5.mxfp4_ops import empty_mxfp4_simt_weight

    q = empty_mxfp4_simt_weight(16, 64)

    assert q.packed.dtype is torch.uint8
    assert q.scales.dtype is torch.uint8
    assert q.packed.shape == (16, 32)
    assert q.scales.shape == (16, 2)


def test_mxfp4_simt_ops_are_explicit_mxfp4_classes():
    import cutlass
    from machete.kernels.decode_matvec.sm120 import MatvecNvfp4Sm120Op
    from machete.kernels.qwen_3_5.mxfp4_ops import (
        MatvecMxfp4SimtSm120Op,
        QWEN3_5_MXFP4_SIMT_OPS,
    )

    assert MatvecMxfp4SimtSm120Op is not MatvecNvfp4Sm120Op
    assert issubclass(MatvecMxfp4SimtSm120Op, MatvecNvfp4Sm120Op)
    assert MatvecMxfp4SimtSm120Op.reads["weight_scales"][0] is cutlass.Uint8
    assert QWEN3_5_MXFP4_SIMT_OPS.matvec is MatvecMxfp4SimtSm120Op
