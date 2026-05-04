# Copyright (c) 2025, Machete Authors
"""Reusable decode matvec kernels, grouped by target architecture."""

from .sm100 import (
    DecodeLayerSchedule as DecodeLayerScheduleSm100,
    FinalRmsLmHeadSm100Op,
    MatvecResidualSm100Op,
    MatvecSm100Op,
    ResidualAddSm100Op,
    RmsGateUpSiluSm100Op,
    RmsKMatvecRopeCacheSm100Op,
    RmsQMatvecRopeSm100Op,
    RmsVMatvecCacheSm100Op,
    schedule_decode_layer_sm100,
    schedule_final_sm100,
)
from .sm120 import (
    DecodeLayerSchedule as DecodeLayerScheduleSm120,
    FinalRmsLmHeadSm120Op,
    MatvecResidualSm120Op,
    MatvecSm120Op,
    ResidualAddSm120Op,
    RmsGateUpSiluSm120Op,
    RmsKMatvecRopeCacheSm120Op,
    RmsQMatvecRopeSm120Op,
    RmsVMatvecCacheSm120Op,
    schedule_decode_layer_sm120,
    schedule_final_sm120,
)

__all__ = [
    "DecodeLayerScheduleSm100",
    "DecodeLayerScheduleSm120",
    "FinalRmsLmHeadSm100Op",
    "FinalRmsLmHeadSm120Op",
    "MatvecResidualSm100Op",
    "MatvecResidualSm120Op",
    "MatvecSm100Op",
    "MatvecSm120Op",
    "ResidualAddSm100Op",
    "ResidualAddSm120Op",
    "RmsGateUpSiluSm100Op",
    "RmsGateUpSiluSm120Op",
    "RmsKMatvecRopeCacheSm100Op",
    "RmsKMatvecRopeCacheSm120Op",
    "RmsQMatvecRopeSm100Op",
    "RmsQMatvecRopeSm120Op",
    "RmsVMatvecCacheSm100Op",
    "RmsVMatvecCacheSm120Op",
    "schedule_decode_layer_sm100",
    "schedule_decode_layer_sm120",
    "schedule_final_sm100",
    "schedule_final_sm120",
]
