# Copyright (c) 2025, Machete Authors
"""Llama-1B convenience aliases for reusable SM100 decode matvec kernels."""

from machete.kernels.decode_matvec.sm100 import (
    DecodeLayerSchedule as Llama1BLayerSchedule,
    FinalRmsLmHeadSm100Op as Llama1BFinalRmsLmHeadSm100Op,
    MatvecResidualSm100Op as Llama1BMatvecResidualSm100Op,
    MatvecSm100Op as Llama1BDownMatvecSm100Op,
    ResidualAddSm100Op as Llama1BResidualAddSm100Op,
    RmsGateUpSiluSm100Op as Llama1BRmsGateUpSiluSm100Op,
    RmsKMatvecRopeCacheSm100Op as Llama1BRmsKCacheSm100Op,
    RmsQMatvecRopeSm100Op as Llama1BRmsQSm100Op,
    RmsVMatvecCacheSm100Op as Llama1BRmsVCacheSm100Op,
    schedule_decode_layer_sm100,
    schedule_final_sm100,
)

LLAMA1B_HIDDEN = 2048
LLAMA1B_HEAD_DIM = 64
LLAMA1B_ROTARY_D2 = 32
LLAMA1B_Q_DIM = 2048
LLAMA1B_KV_DIM = 512
LLAMA1B_INTERMEDIATE = 8192
LLAMA1B_VOCAB = 128256
LLAMA1B_MATVEC_BLOCK = 16
LLAMA1B_REDUCTION_DIM_PER_WARP = 512
LLAMA1B_CONSUMER_WARPS = 4

__all__ = [
    "LLAMA1B_HIDDEN",
    "LLAMA1B_HEAD_DIM",
    "LLAMA1B_ROTARY_D2",
    "LLAMA1B_Q_DIM",
    "LLAMA1B_KV_DIM",
    "LLAMA1B_INTERMEDIATE",
    "LLAMA1B_VOCAB",
    "LLAMA1B_MATVEC_BLOCK",
    "LLAMA1B_REDUCTION_DIM_PER_WARP",
    "LLAMA1B_CONSUMER_WARPS",
    "Llama1BDownMatvecSm100Op",
    "Llama1BFinalRmsLmHeadSm100Op",
    "Llama1BMatvecResidualSm100Op",
    "Llama1BResidualAddSm100Op",
    "Llama1BRmsGateUpSiluSm100Op",
    "Llama1BRmsKCacheSm100Op",
    "Llama1BRmsQSm100Op",
    "Llama1BRmsVCacheSm100Op",
    "Llama1BLayerSchedule",
    "schedule_decode_layer_sm100",
    "schedule_final_sm100",
]
