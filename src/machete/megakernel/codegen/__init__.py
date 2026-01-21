# Copyright (c) 2025, Machete Authors
"""
Code Generation System for Machete Megakernels.

This package provides infrastructure for extracting Python source code
from kernel methods and generating optimized fused kernels via inlining.
"""

from .extractor import MethodExtractor
from .transformer import (
    SelfReferenceReplacer,
    DecoratorRemover,
    ArgumentRenamer,
    LocalVariablePrefixer,
)
from .inliner import CodeInliner

__all__ = [
    "MethodExtractor",
    "SelfReferenceReplacer",
    "DecoratorRemover",
    "ArgumentRenamer",
    "LocalVariablePrefixer",
    "CodeInliner",
    "select_template",
]


def select_template(ctx):
    """Select the appropriate template based on kernel configuration.

    Args:
        ctx: EmitterContext with kernel information

    Returns:
        KernelTemplate instance appropriate for the configuration
    """
    from .templates import SingleKernelTemplate, SequentialTemplate, WarpSpecializedTemplate

    num_ops = len(ctx.graph.nodes)
    has_warp_spec = any(n.uses_warp_specialization for n in ctx.graph.nodes.values())

    if num_ops == 1 and not has_warp_spec:
        return SingleKernelTemplate()
    elif has_warp_spec:
        return WarpSpecializedTemplate()
    else:
        return SequentialTemplate()
