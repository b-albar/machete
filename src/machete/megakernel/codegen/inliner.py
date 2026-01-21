# Copyright (c) 2025, Machete Authors
"""
Code Inlining Pipeline for Kernel Methods.

This module provides the CodeInliner class that orchestrates the extraction,
transformation, and inlining of kernel method code into generated megakernels.
"""

import ast
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from .extractor import MethodExtractor, extract_method_info
from .transformer import (
    transform_for_inlining,
    SelfReferenceReplacer,
    DecoratorRemover,
    ArgumentRenamer,
    LocalVariablePrefixer,
    PassRemover,
    ReturnRemover,
)


@dataclass
class InlinedMethod:
    """Result of inlining a single method.

    Attributes:
        name: Original method name
        body: List of AST statement nodes (the inlined code)
        captured: Dict of captured complex values from self.xxx
        source_code: Generated Python source code string
        is_empty: True if the method body was just `pass`
    """
    name: str
    body: List[ast.stmt]
    captured: Dict[str, Any]
    source_code: str
    is_empty: bool = False


@dataclass
class InlinedKernel:
    """Result of inlining all methods from a kernel.

    Attributes:
        setup_kernel: Inlined setup_kernel method (or None)
        load_forward: Inlined load_forward method
        compute_forward: Inlined compute_forward method
        store_forward: Inlined store_forward method
        load_backward: Inlined load_backward method (or None)
        compute_backward: Inlined compute_backward method (or None)
        store_backward: Inlined store_backward method (or None)
        all_captured: Merged dict of all captured values
    """
    setup_kernel: Optional[InlinedMethod] = None
    load_forward: Optional[InlinedMethod] = None
    compute_forward: Optional[InlinedMethod] = None
    store_forward: Optional[InlinedMethod] = None
    load_backward: Optional[InlinedMethod] = None
    compute_backward: Optional[InlinedMethod] = None
    store_backward: Optional[InlinedMethod] = None
    all_captured: Dict[str, Any] = field(default_factory=dict)


class CodeInliner:
    """Orchestrates extraction and inlining of kernel method code.

    This class provides the main API for converting kernel methods into
    inlined code suitable for inclusion in generated megakernels.

    Example:
        inliner = CodeInliner()

        # Inline a single method
        result = inliner.inline_method(
            kernel.load_forward,
            kernel,
            arg_mapping={"q_ptr": "q"},
            prefix="op0_load_"
        )

        # Generate source code
        code = inliner.to_source(result.body, indent=8)

        # Inline all methods from a kernel
        inlined = inliner.inline_kernel(kernel, op_idx=0)
    """

    def __init__(self):
        """Initialize the inliner."""
        self.extractor = MethodExtractor()

    def inline_method(
        self,
        method: Callable,
        kernel_instance: Any,
        arg_mapping: Optional[Dict[str, str]] = None,
        prefix: str = "",
        exclude_locals: Optional[Set[str]] = None,
    ) -> InlinedMethod:
        """Inline a single kernel method.

        Args:
            method: The method to inline (e.g., kernel.load_forward)
            kernel_instance: The kernel instance for resolving self.xxx
            arg_mapping: Optional mapping of argument names (e.g., q_ptr -> q)
            prefix: Prefix for local variables to avoid collisions
            exclude_locals: Names to exclude from prefixing

        Returns:
            InlinedMethod with transformed body and captured values
        """
        # Extract source and parse to AST
        source = self.extractor.extract_source(method)
        func_def = self.extractor.parse_to_ast(source)

        # Check if method is just `pass`
        body_stmts = self.extractor.get_body_without_docstring(func_def)
        is_empty = (len(body_stmts) == 1 and isinstance(body_stmts[0], ast.Pass))

        if is_empty:
            return InlinedMethod(
                name=func_def.name,
                body=[],
                captured={},
                source_code="",
                is_empty=True
            )

        # Transform for inlining
        body, captured = transform_for_inlining(
            func_def,
            kernel_instance,
            arg_mapping=arg_mapping,
            local_prefix=prefix,
            exclude_locals=exclude_locals,
        )

        # Generate source code
        source_code = self.to_source(body)

        return InlinedMethod(
            name=func_def.name,
            body=body,
            captured=captured,
            source_code=source_code,
            is_empty=False
        )

    def inline_kernel(
        self,
        kernel: Any,
        op_idx: int = 0,
        mode: str = "forward",
        arg_mapping: Optional[Dict[str, str]] = None,
    ) -> InlinedKernel:
        """Inline all relevant methods from a kernel.

        Args:
            kernel: The kernel instance
            op_idx: Operation index for variable prefixing
            mode: "forward" or "backward"
            arg_mapping: Optional argument name mapping

        Returns:
            InlinedKernel with all inlined methods
        """
        prefix_base = f"op{op_idx}_"
        result = InlinedKernel()

        # setup_kernel (if exists)
        if hasattr(kernel, 'setup_kernel'):
            setup = self.inline_method(
                kernel.setup_kernel,
                kernel,
                arg_mapping=arg_mapping,
                prefix=f"{prefix_base}setup_",
            )
            result.setup_kernel = setup
            result.all_captured.update(setup.captured)

        # Forward pass methods
        if mode == "forward" or mode == "both":
            if hasattr(kernel, 'load_forward'):
                load = self.inline_method(
                    kernel.load_forward,
                    kernel,
                    arg_mapping=arg_mapping,
                    prefix=f"{prefix_base}load_",
                )
                result.load_forward = load
                result.all_captured.update(load.captured)

            if hasattr(kernel, 'compute_forward'):
                compute = self.inline_method(
                    kernel.compute_forward,
                    kernel,
                    arg_mapping=arg_mapping,
                    prefix=f"{prefix_base}compute_",
                )
                result.compute_forward = compute
                result.all_captured.update(compute.captured)

            if hasattr(kernel, 'store_forward'):
                store = self.inline_method(
                    kernel.store_forward,
                    kernel,
                    arg_mapping=arg_mapping,
                    prefix=f"{prefix_base}store_",
                )
                result.store_forward = store
                result.all_captured.update(store.captured)

        # Backward pass methods
        if mode == "backward" or mode == "both":
            if hasattr(kernel, 'load_backward'):
                load = self.inline_method(
                    kernel.load_backward,
                    kernel,
                    arg_mapping=arg_mapping,
                    prefix=f"{prefix_base}bwd_load_",
                )
                result.load_backward = load
                result.all_captured.update(load.captured)

            if hasattr(kernel, 'compute_backward'):
                compute = self.inline_method(
                    kernel.compute_backward,
                    kernel,
                    arg_mapping=arg_mapping,
                    prefix=f"{prefix_base}bwd_compute_",
                )
                result.compute_backward = compute
                result.all_captured.update(compute.captured)

            if hasattr(kernel, 'store_backward'):
                store = self.inline_method(
                    kernel.store_backward,
                    kernel,
                    arg_mapping=arg_mapping,
                    prefix=f"{prefix_base}bwd_store_",
                )
                result.store_backward = store
                result.all_captured.update(store.captured)

        return result

    def to_source(
        self,
        statements: List[ast.stmt],
        indent: int = 0,
    ) -> str:
        """Convert AST statements back to Python source code.

        Args:
            statements: List of AST statement nodes
            indent: Number of spaces to indent each line

        Returns:
            Python source code string
        """
        if not statements:
            return ""

        # Filter out None values (from removed statements)
        statements = [s for s in statements if s is not None]

        if not statements:
            return ""

        # Create a module to hold the statements
        module = ast.Module(body=statements, type_ignores=[])
        ast.fix_missing_locations(module)

        # Use ast.unparse (Python 3.9+)
        try:
            code = ast.unparse(module)
        except AttributeError:
            # Fallback for older Python versions
            import astor
            code = astor.to_source(module)

        # Add indentation
        if indent > 0:
            indent_str = ' ' * indent
            lines = code.split('\n')
            code = '\n'.join(indent_str + line if line.strip() else line for line in lines)

        return code

    def generate_captured_bindings(
        self,
        captured: Dict[str, Any],
        prefix: str = "_k_",
        indent: int = 0,
    ) -> str:
        """Generate code to bind captured values.

        Args:
            captured: Dict of captured attribute names to values
            prefix: Prefix used in variable names
            indent: Number of spaces to indent

        Returns:
            Python code string with assignment statements
        """
        if not captured:
            return ""

        indent_str = ' ' * indent
        lines = []
        for attr_name in captured:
            var_name = f"{prefix}{attr_name}"
            lines.append(f"{indent_str}{var_name} = _captured['{attr_name}']")

        return '\n'.join(lines)


def inline_lcs_methods(
    kernel: Any,
    op_idx: int,
    mode: str,
    arg_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Convenience function to inline L/C/S methods and return source code.

    Args:
        kernel: The kernel instance
        op_idx: Operation index for prefixing
        mode: "forward" or "backward"
        arg_mapping: Optional argument name mapping

    Returns:
        Dict with keys "setup", "load", "compute", "store" mapped to source code strings
    """
    inliner = CodeInliner()
    result = inliner.inline_kernel(kernel, op_idx, mode, arg_mapping)

    code_dict = {}

    if result.setup_kernel and not result.setup_kernel.is_empty:
        code_dict["setup"] = result.setup_kernel.source_code
    else:
        code_dict["setup"] = ""

    if mode == "forward":
        code_dict["load"] = result.load_forward.source_code if result.load_forward and not result.load_forward.is_empty else ""
        code_dict["compute"] = result.compute_forward.source_code if result.compute_forward and not result.compute_forward.is_empty else ""
        code_dict["store"] = result.store_forward.source_code if result.store_forward and not result.store_forward.is_empty else ""
    else:
        code_dict["load"] = result.load_backward.source_code if result.load_backward and not result.load_backward.is_empty else ""
        code_dict["compute"] = result.compute_backward.source_code if result.compute_backward and not result.compute_backward.is_empty else ""
        code_dict["store"] = result.store_backward.source_code if result.store_backward and not result.store_backward.is_empty else ""

    return code_dict
