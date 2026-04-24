# Copyright (c) 2025, Machete Authors
"""Layout regressions for the Qwen 3.5 decode benchmark."""

import torch

from benchmarks.kernels import benchmark_qwen3_5_decode as qwen_decode


def test_kv_cache_is_physical_bshd():
    batch = 2
    max_seq = 9
    k_caches, v_caches = qwen_decode.allocate_kv_cache(
        batch, max_seq, dtype=torch.float32, device="cpu"
    )

    expected_shape = (batch, max_seq, qwen_decode.NUM_KV_HEADS, qwen_decode.HEAD_DIM)
    expected_stride = (
        max_seq * qwen_decode.NUM_KV_HEADS * qwen_decode.HEAD_DIM,
        qwen_decode.NUM_KV_HEADS * qwen_decode.HEAD_DIM,
        qwen_decode.HEAD_DIM,
        1,
    )

    assert k_caches[0].shape == expected_shape
    assert v_caches[0].shape == expected_shape
    assert k_caches[0].stride() == expected_stride
    assert v_caches[0].stride() == expected_stride


def test_decode_correctness_status_uses_model_level_tolerance():
    expected = torch.zeros(4, dtype=torch.float32)
    actual = torch.full_like(expected, 0.07)

    assert qwen_decode._correctness_status(actual, expected, atol=0.2, rtol=0.02).startswith("OK/")
    assert qwen_decode._correctness_status(actual, expected, atol=0.02, rtol=0.02).startswith("BAD ")
