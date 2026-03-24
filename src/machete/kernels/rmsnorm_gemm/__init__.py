# Copyright (c) 2025, Machete Authors
"""Fused RMSNorm + GEMM kernel for the megakernel framework."""

from .rmsnorm_gemm import RMSNormGemmOp

__all__ = ["RMSNormGemmOp"]
