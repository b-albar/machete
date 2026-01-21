# Copyright (c) 2025, Machete Authors
"""
Kernel Code Templates for Machete Megakernels.

This package provides templates for generating different types of fused kernels:
- SingleKernelTemplate: Single operation kernels
- SequentialTemplate: Sequential fusing of multiple operations
- WarpSpecializedTemplate: Warp-specialized producer-consumer pattern
"""

from .common import KernelTemplate, TemplateContext
from .single import SingleKernelTemplate
from .sequential import SequentialTemplate
from .warp_spec import WarpSpecializedTemplate

__all__ = [
    "KernelTemplate",
    "TemplateContext",
    "SingleKernelTemplate",
    "SequentialTemplate",
    "WarpSpecializedTemplate",
]
