# Copyright (c) 2025, Machete Authors
"""MXFP4 (OCP microscaling fp4) weight/activation helpers for the SM120 block-scaled MMA.

Format: E2M1 4-bit values, group of 32 sharing one **E8M0** (power-of-two) scale byte
(UE8M0: value = 2**(byte-127)). Matches ``GGML_TYPE_MXFP4`` and the warp
``MmaMXF4Op`` (sf_vec_size=32). Decode is ``E2M1_ABS[code&7]*sign * 2**(e8m0-127)``;
the tensor-core MMA matches this exactly, so only the decode must be consistent.

The scale factors must be laid out in the byte arrangement the warp MMA's SFA/SFB
operand partition reads (``build_sfa``/``build_sfb``). Inference-only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# MMA instruction shape + scale-group constants (MmaMXF4Op).
MMA_M, MMA_N, MMA_K = 16, 8, 64
SF_VEC = 32
GROUPS_PER_MMA_K = MMA_K // SF_VEC          # 2 scale groups per 64-wide K-tile
SFA_TILE_SPAN = 80                          # bytes of SFA per K-tile (16 rows)
SFB_TILE_SPAN = 16                          # bytes of SFB per K-tile (8 cols)
# The warp MMA's SFA/SFB operand partition reads a fixed-size footprint per K-tile
# that runs a few bytes past the logical per-tile span (a benign read-ahead — the
# extra lanes are unused). For every tile but the last that read-ahead lands in the
# next tile's bytes; at the last K-tile it would run past the buffer. A trailing
# guard of zero bytes keeps that final read in-bounds for users of the MMA SFA/SFB
# layout helpers.
SF_READ_GUARD = 16                          # trailing guard bytes on every SF buffer

_E2M1_ABS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)
_E2M1_THR = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], np.float32)


@dataclass(frozen=True)
class MXFP4Tensor:
    """Packed row-major MXFP4 matrix + E8M0 scale bytes + the MMA SFA buffer."""

    packed: torch.Tensor       # (rows, cols//2) uint8, low nibble even-k
    scales_e8m0: torch.Tensor  # (rows, cols//32) uint8 (UE8M0 bytes)
    sfa: torch.Tensor          # (rows//16 * SFA_TILE_SPAN * n_ktiles + SF_READ_GUARD,) uint8 — MMA SFA layout
    rows: int
    cols: int


def _quantize_codes_e8m0(W: np.ndarray):
    """Return (codes uint8 [rows,cols], e8m0 uint8 [rows,cols//32], deq f32)."""
    rows, K = W.shape
    Wg = W.reshape(rows, K // SF_VEC, SF_VEC).astype(np.float32)
    amax = np.abs(Wg).max(-1)
    e = np.clip(np.floor(np.log2(np.maximum(amax, 1e-30))) - 2.0, -127, 127)
    scale = (2.0 ** e).astype(np.float32)
    e8m0 = (e + 127).astype(np.uint8)
    Wn = Wg / scale[..., None]
    mag = np.digitize(np.abs(Wn), _E2M1_THR).astype(np.uint8)            # 0..7
    codes = (mag | ((Wn < 0).astype(np.uint8) << 3)).reshape(rows, K)
    deq = (_E2M1_ABS[mag] * np.where(Wn < 0, -1.0, 1.0) * scale[..., None]).reshape(rows, K)
    return codes, e8m0, deq.astype(np.float32)


def _pack(codes: np.ndarray) -> np.ndarray:
    return ((codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)).astype(np.uint8)


def build_sfa(e8m0: np.ndarray) -> np.ndarray:
    """E8M0 weight scales (rows, K//32), rows multiple of 16 -> flat SFA buffer
    in the warp MMA's SFA partition layout (rows 0-7 at base, 8-15 at base+64;
    byte(m,kg)=(m%8)+8*kg or 64+(m-8)+8*kg), tiles of 16 rows concatenated."""
    rows, G = e8m0.shape
    n_ktiles = G // GROUPS_PER_MMA_K
    n_tiles = rows // MMA_M
    s = e8m0.reshape(n_tiles, MMA_M, n_ktiles, GROUPS_PER_MMA_K)
    buf = np.zeros((n_tiles, n_ktiles, SFA_TILE_SPAN), np.uint8)
    for m in range(MMA_M):
        for kg in range(GROUPS_PER_MMA_K):
            off = (m % 8) + 8 * kg if m < 8 else 64 + (m - 8) + 8 * kg
            buf[:, :, off] = s[:, m, :, kg]
    return np.concatenate([buf.reshape(-1), np.zeros(SF_READ_GUARD, np.uint8)])


def build_sfb(e8m0: np.ndarray) -> np.ndarray:
    """E8M0 activation scales (8, K//32) -> flat SFB buffer (byte(n,kg)=2*n+kg)."""
    rows, G = e8m0.shape
    n_ktiles = G // GROUPS_PER_MMA_K
    s = e8m0.reshape(rows, n_ktiles, GROUPS_PER_MMA_K)
    buf = np.zeros((n_ktiles, SFB_TILE_SPAN), np.uint8)
    for n in range(rows):
        for kg in range(GROUPS_PER_MMA_K):
            buf[:, 2 * n + kg] = s[n, :, kg]
    return np.concatenate([buf.reshape(-1), np.zeros(SF_READ_GUARD, np.uint8)])


@torch.no_grad()
def quantize_mxfp4_weight(weight: torch.Tensor) -> MXFP4Tensor:
    """Quantize a row-major (rows, cols) matrix to MXFP4 + build the MMA SFA buffer.
    ``rows`` must be a multiple of 16 and ``cols`` a multiple of 64."""
    if weight.dim() != 2:
        raise ValueError("weight must be rank-2")
    rows, cols = weight.shape
    if rows % MMA_M or cols % MMA_K:
        raise ValueError(f"rows%16 and cols%64 required, got ({rows},{cols})")
    W = weight.detach().float().cpu().numpy()
    codes, e8m0, _ = _quantize_codes_e8m0(W)
    dev = weight.device
    return MXFP4Tensor(
        packed=torch.from_numpy(_pack(codes)).to(dev),
        scales_e8m0=torch.from_numpy(e8m0).to(dev),
        sfa=torch.from_numpy(build_sfa(e8m0)).to(dev),
        rows=rows, cols=cols,
    )


@torch.no_grad()
def quantize_mxfp4_activation(x: torch.Tensor):
    """Quantize a length-K activation token to MXFP4, padded to the MMA's N=8 rows.
    Returns (packed (8, K//2) uint8, sfb flat uint8). Rows 1-7 are zero."""
    K = x.numel()
    a = np.zeros((MMA_N, K), np.float32)
    a[0] = x.detach().float().cpu().numpy().reshape(K)
    codes, e8m0, _ = _quantize_codes_e8m0(a)
    dev = x.device
    return (torch.from_numpy(_pack(codes)).to(dev), torch.from_numpy(build_sfb(e8m0)).to(dev))


@torch.no_grad()
def dequantize_mxfp4_weight(q: MXFP4Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    lo = (q.packed & 0xF).to(torch.long)
    hi = (q.packed >> 4).to(torch.long)
    codes = torch.empty((q.rows, q.cols), dtype=torch.long, device=q.packed.device)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    lut = torch.tensor(
        [0.0, .5, 1., 1.5, 2., 3., 4., 6., -0.0, -.5, -1., -1.5, -2., -3., -4., -6.],
        dtype=torch.float32, device=q.packed.device)
    vals = lut[codes].reshape(q.rows, q.cols // SF_VEC, SF_VEC)
    scale = torch.pow(2.0, q.scales_e8m0.float() - 127.0).unsqueeze(-1)
    return (vals * scale).reshape(q.rows, q.cols).to(dtype)
