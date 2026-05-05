# Copyright (c) 2025, Machete Authors
"""Packed NVFP4 E2M1 weight helpers.

The layout matches the decode convention used by Luce:

* two 4-bit E2M1 values per byte, low nibble first;
* one fp16 scale per row and K group;
* scale = amax / 6 for each group, so code 7 represents +amax.

This module is intentionally inference-only. It does not define autograd
behavior and should not be used for training paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_E2M1_THRESHOLDS = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class NVFP4Tensor:
    """Packed row-major NVFP4 matrix plus per-group scales."""

    packed: torch.Tensor
    scales: torch.Tensor
    group_size: int = 32
    rows: int | None = None
    cols: int | None = None

    def __post_init__(self):
        if self.packed.dtype != torch.uint8:
            raise TypeError(f"packed must be torch.uint8, got {self.packed.dtype}")
        if self.scales.dtype != torch.float16:
            raise TypeError(f"scales must be torch.float16, got {self.scales.dtype}")
        if self.packed.dim() != 2 or self.scales.dim() != 2:
            raise ValueError("packed and scales must be rank-2 tensors")
        rows = self.packed.shape[0] if self.rows is None else self.rows
        cols = self.packed.shape[1] * 2 if self.cols is None else self.cols
        if rows != self.packed.shape[0]:
            raise ValueError("rows must match packed.shape[0]")
        if cols != self.packed.shape[1] * 2:
            raise ValueError("cols must equal packed.shape[1] * 2")
        if cols % self.group_size != 0:
            raise ValueError("cols must be divisible by group_size")
        expected_scales = (rows, cols // self.group_size)
        if tuple(self.scales.shape) != expected_scales:
            raise ValueError(f"scales shape must be {expected_scales}, got {tuple(self.scales.shape)}")
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "cols", cols)


def _check_weight(weight: torch.Tensor, group_size: int) -> None:
    if weight.dim() != 2:
        raise ValueError(f"weight must be rank-2, got shape {tuple(weight.shape)}")
    if weight.shape[1] % 2 != 0:
        raise ValueError("weight cols must be divisible by 2")
    if weight.shape[1] % group_size != 0:
        raise ValueError("weight cols must be divisible by group_size")
    if group_size % 2 != 0:
        raise ValueError("group_size must be even")
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported weight dtype {weight.dtype}")


@torch.no_grad()
def quantize_nvfp4_weight(weight: torch.Tensor, group_size: int = 32) -> NVFP4Tensor:
    """Quantize a row-major matrix to packed E2M1 NVFP4.

    Args:
        weight: ``(rows, cols)`` fp16/bf16/fp32 matrix.
        group_size: K elements per scale. Luce uses 32 for decode matvecs.
    """

    _check_weight(weight, group_size)
    rows, cols = weight.shape
    device = weight.device
    thresholds = _E2M1_THRESHOLDS.to(device=device)

    w = weight.float().reshape(rows, cols // group_size, group_size)
    amax = w.abs().amax(dim=-1)
    scales_f = torch.where(amax > 0, amax / 6.0, torch.ones_like(amax))
    normalized = w / scales_f.unsqueeze(-1)

    mag = torch.bucketize(normalized.abs().contiguous(), thresholds)
    codes = mag.to(torch.uint8)
    codes = codes | ((normalized < 0).to(torch.uint8) << 3)
    codes = codes.reshape(rows, cols)

    lo = codes[:, 0::2]
    hi = codes[:, 1::2]
    packed = (lo | (hi << 4)).contiguous()
    scales = scales_f.to(torch.float16).contiguous()
    return NVFP4Tensor(packed=packed, scales=scales, group_size=group_size, rows=rows, cols=cols)


@torch.no_grad()
def dequantize_nvfp4_weight(qweight: NVFP4Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Dequantize a packed NVFP4 matrix for validation/reference code."""

    lut = _E2M1_LUT.to(device=qweight.packed.device)
    packed = qweight.packed
    lo = packed & 0x0F
    hi = packed >> 4
    codes = torch.empty((qweight.rows, qweight.cols), dtype=torch.long, device=packed.device)
    codes[:, 0::2] = lo.long()
    codes[:, 1::2] = hi.long()
    vals = lut[codes]
    group_vals = vals.reshape(qweight.rows, qweight.cols // qweight.group_size, qweight.group_size)
    out = group_vals * qweight.scales.float().unsqueeze(-1)
    return out.reshape(qweight.rows, qweight.cols).to(dtype)
