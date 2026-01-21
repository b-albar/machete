# Copyright (c) 2025, Machete Authors
"""
AST Transformers for Kernel Code Generation.

This module provides AST NodeTransformer classes that transform kernel method
code for inlining into generated megakernels.
"""

import ast
from typing import Any, Dict, Set, Optional, List
import copy


class SelfReferenceReplacer(ast.NodeTransformer):
    """Replace self.xxx attribute accesses with inlined values or captured references.

    For simple types (int, float, bool), replaces self.xxx with const_expr(value).
    For complex types, captures the reference and replaces with a variable name.

    Example:
        transformer = SelfReferenceReplacer(kernel_instance)
        new_tree = transformer.visit(func_def)
        captured_values = transformer.captured

    Attributes:
        kernel: The kernel instance to read attribute values from
        captured: Dict of attribute name -> value for complex captured types
        const_types: Tuple of types that should be inlined as const_expr
    """

    def __init__(self, kernel_instance: Any, prefix: str = "_k_"):
        """Initialize the transformer.

        Args:
            kernel_instance: The kernel instance to read values from
            prefix: Prefix for captured variable names
        """
        super().__init__()
        self.kernel = kernel_instance
        self.prefix = prefix
        self.captured: Dict[str, Any] = {}
        self.const_types = (int, float, bool)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Transform self.xxx attribute access.

        - Simple types -> const_expr(value)
        - Complex types -> _k_attrname reference
        """
        # Only transform self.xxx
        if not (isinstance(node.value, ast.Name) and node.value.id == 'self'):
            return self.generic_visit(node)

        attr_name = node.attr
        value = getattr(self.kernel, attr_name, None)

        if value is None:
            # Leave as-is if attribute doesn't exist (might be set at runtime)
            return self.generic_visit(node)

        if isinstance(value, self.const_types):
            # Inline as const_expr(value)
            return ast.Call(
                func=ast.Name(id='const_expr', ctx=ast.Load()),
                args=[ast.Constant(value=value)],
                keywords=[]
            )
        else:
            # Capture and reference with prefix
            var_name = f"{self.prefix}{attr_name}"
            self.captured[attr_name] = value
            return ast.Name(id=var_name, ctx=node.ctx)

    def get_captured_bindings(self) -> List[str]:
        """Generate Python code lines for binding captured values.

        Returns:
            List of assignment statements like "_k_dtype = captured['dtype']"
        """
        lines = []
        for attr_name in self.captured:
            var_name = f"{self.prefix}{attr_name}"
            lines.append(f"{var_name} = _captured['{attr_name}']")
        return lines


class DecoratorRemover(ast.NodeTransformer):
    """Remove all decorators from function definitions.

    This is used to strip @cute.jit and other decorators from methods
    before inlining their code.
    """

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Remove decorators from a function definition."""
        node.decorator_list = []
        return self.generic_visit(node)


class ArgumentRenamer(ast.NodeTransformer):
    """Rename function arguments according to a mapping.

    Used to transform pointer arguments to tensor arguments, e.g.,
    q_ptr -> q when the calling context provides CuTe tensors.

    Example:
        renamer = ArgumentRenamer({"q_ptr": "q", "cos_ptr": "cos"})
        new_tree = renamer.visit(func_def)
    """

    def __init__(self, mapping: Dict[str, str]):
        """Initialize with a name mapping.

        Args:
            mapping: Dict from old names to new names
        """
        super().__init__()
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Rename Name nodes according to mapping."""
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        """Rename function argument names."""
        if node.arg in self.mapping:
            node.arg = self.mapping[node.arg]
        return node


class LocalVariablePrefixer(ast.NodeTransformer):
    """Add prefix to local variables to avoid name collisions when inlining.

    When inlining multiple methods, their local variables might collide.
    This transformer adds a prefix to all local variable names.

    Example:
        prefixer = LocalVariablePrefixer("op0_load_", locals={"i", "val"})
        new_tree = prefixer.visit(func_def)
    """

    def __init__(self, prefix: str, local_names: Set[str], exclude: Optional[Set[str]] = None):
        """Initialize the prefixer.

        Args:
            prefix: Prefix to add to local variable names
            local_names: Set of local variable names to prefix
            exclude: Optional set of names to NOT prefix (e.g., builtins)
        """
        super().__init__()
        self.prefix = prefix
        self.local_names = local_names
        self.exclude = exclude or set()

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Add prefix to local variable names."""
        if node.id in self.local_names and node.id not in self.exclude:
            node.id = f"{self.prefix}{node.id}"
        return node


class MethodCallInliner(ast.NodeTransformer):
    """Replace method calls with inlined code.

    This transformer replaces calls to self.method_name(...) with
    the inlined body of that method.

    Note: This is a complex transformation and should be used carefully.
    """

    def __init__(self, method_bodies: Dict[str, List[ast.stmt]]):
        """Initialize with method bodies to inline.

        Args:
            method_bodies: Dict from method name to list of body statements
        """
        super().__init__()
        self.method_bodies = method_bodies

    def visit_Expr(self, node: ast.Expr) -> Any:
        """Check if this is a method call that should be inlined."""
        if isinstance(node.value, ast.Call):
            call = node.value
            # Check for self.method_name(...)
            if (isinstance(call.func, ast.Attribute) and
                isinstance(call.func.value, ast.Name) and
                call.func.value.id == 'self'):

                method_name = call.func.attr
                if method_name in self.method_bodies:
                    # Return the inlined body statements
                    return copy.deepcopy(self.method_bodies[method_name])

        return node


class ConstExprWrapper(ast.NodeTransformer):
    """Wrap compile-time constant values with const_expr().

    This transformer identifies values that should be compile-time constants
    and wraps them with const_expr() calls.
    """

    def __init__(self, const_names: Set[str]):
        """Initialize with names that should be wrapped.

        Args:
            const_names: Set of variable names that are compile-time constants
        """
        super().__init__()
        self.const_names = const_names

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Wrap constant names with const_expr()."""
        if node.id in self.const_names and isinstance(node.ctx, ast.Load):
            return ast.Call(
                func=ast.Name(id='const_expr', ctx=ast.Load()),
                args=[node],
                keywords=[]
            )
        return node


class PassRemover(ast.NodeTransformer):
    """Remove `pass` statements from function bodies.

    When a method body is just `pass` (no-op), we want to remove it
    entirely when inlining.
    """

    def visit_Pass(self, node: ast.Pass) -> Optional[ast.AST]:
        """Remove pass statements."""
        return None


class ReturnRemover(ast.NodeTransformer):
    """Remove or transform return statements.

    For L/C/S methods that shouldn't have return values when inlined,
    this removes the return statements.

    For methods that return values, optionally captures the return value
    into a named variable.
    """

    def __init__(self, capture_var: Optional[str] = None):
        """Initialize the transformer.

        Args:
            capture_var: If set, convert `return x` to `capture_var = x`
        """
        super().__init__()
        self.capture_var = capture_var

    def visit_Return(self, node: ast.Return) -> Optional[ast.AST]:
        """Transform or remove return statements."""
        if node.value is None:
            # `return` with no value - remove
            return None

        if self.capture_var:
            # Convert `return x` to `capture_var = x`
            return ast.Assign(
                targets=[ast.Name(id=self.capture_var, ctx=ast.Store())],
                value=node.value
            )
        else:
            # Remove the return, keep just the expression (if it has side effects)
            return ast.Expr(value=node.value)


def apply_transformers(tree: ast.AST, *transformers: ast.NodeTransformer) -> ast.AST:
    """Apply multiple transformers in sequence.

    Args:
        tree: The AST to transform
        *transformers: Transformer instances to apply in order

    Returns:
        The transformed AST
    """
    result = tree
    for transformer in transformers:
        result = transformer.visit(result)
        ast.fix_missing_locations(result)
    return result


def transform_for_inlining(
    func_def: ast.FunctionDef,
    kernel_instance: Any,
    arg_mapping: Optional[Dict[str, str]] = None,
    local_prefix: str = "",
    exclude_locals: Optional[Set[str]] = None,
) -> tuple:
    """Apply standard transformations for inlining a method.

    This applies the common transformations needed to prepare a method
    for inlining into a generated kernel.

    Args:
        func_def: The function definition AST
        kernel_instance: The kernel instance for self.xxx resolution
        arg_mapping: Optional mapping of argument names
        local_prefix: Prefix for local variables
        exclude_locals: Names to exclude from prefixing

    Returns:
        Tuple of (transformed body statements, captured values dict)
    """
    from .extractor import MethodExtractor

    # Create fresh copy to avoid modifying original
    func_def = copy.deepcopy(func_def)

    extractor = MethodExtractor()

    # Find local variables for prefixing
    locals_set = extractor.find_local_assignments(func_def)

    # Build transformers
    transformers = [
        DecoratorRemover(),
        PassRemover(),
    ]

    # Self reference replacement
    self_replacer = SelfReferenceReplacer(kernel_instance)
    transformers.append(self_replacer)

    # Argument renaming if provided
    if arg_mapping:
        transformers.append(ArgumentRenamer(arg_mapping))

    # Local variable prefixing if prefix provided
    if local_prefix and locals_set:
        exclude = exclude_locals or set()
        transformers.append(LocalVariablePrefixer(local_prefix, locals_set, exclude))

    # Apply transformations
    for transformer in transformers:
        func_def = transformer.visit(func_def)
        ast.fix_missing_locations(func_def)

    # Get body without docstring
    body = extractor.get_body_without_docstring(func_def)

    return body, self_replacer.captured
