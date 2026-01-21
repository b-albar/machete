# Copyright (c) 2025, Machete Authors
"""
Source Code Extraction for Kernel Methods.

This module provides utilities for extracting Python source code from
kernel methods using the inspect module and parsing it into AST.
"""

import ast
import inspect
import textwrap
from typing import Callable, List, Set, Tuple, Optional, Any


class MethodExtractor:
    """Extract and parse source code from kernel methods.

    This class handles the extraction of Python source code from methods,
    removing decorators, and parsing into AST for further transformation.

    Example:
        extractor = MethodExtractor()
        source = extractor.extract_source(kernel.load_forward)
        func_def = extractor.parse_to_ast(source)
        body = extractor.extract_body(func_def)
    """

    def extract_source(self, method: Callable) -> str:
        """Extract dedented source code from a method.

        Args:
            method: The method to extract source from

        Returns:
            Dedented source code string

        Raises:
            OSError: If source code cannot be retrieved
        """
        source = inspect.getsource(method)
        return textwrap.dedent(source)

    def parse_to_ast(self, source: str) -> ast.FunctionDef:
        """Parse source code string into an AST FunctionDef node.

        Args:
            source: Python source code string

        Returns:
            The FunctionDef AST node

        Raises:
            SyntaxError: If source code is invalid Python
            ValueError: If source doesn't contain a function definition
        """
        tree = ast.parse(source)
        if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError("Source does not contain a function definition")
        return tree.body[0]

    def extract_body(self, func_def: ast.FunctionDef) -> List[ast.stmt]:
        """Extract the body statements from a function definition.

        This returns just the body statements without the function signature
        or decorators, suitable for inlining into another function.

        Args:
            func_def: The FunctionDef AST node

        Returns:
            List of AST statement nodes from the function body
        """
        return func_def.body

    def extract_and_parse(self, method: Callable) -> Tuple[ast.FunctionDef, str]:
        """Convenience method to extract source and parse to AST.

        Args:
            method: The method to extract

        Returns:
            Tuple of (FunctionDef node, original source string)
        """
        source = self.extract_source(method)
        func_def = self.parse_to_ast(source)
        return func_def, source

    def get_parameter_names(self, func_def: ast.FunctionDef) -> List[str]:
        """Get the parameter names from a function definition.

        Args:
            func_def: The FunctionDef AST node

        Returns:
            List of parameter names (excluding 'self')
        """
        params = []
        for arg in func_def.args.args:
            if arg.arg != 'self':
                params.append(arg.arg)
        return params

    def get_docstring(self, func_def: ast.FunctionDef) -> Optional[str]:
        """Extract the docstring from a function definition.

        Args:
            func_def: The FunctionDef AST node

        Returns:
            The docstring, or None if not present
        """
        return ast.get_docstring(func_def)

    def get_body_without_docstring(self, func_def: ast.FunctionDef) -> List[ast.stmt]:
        """Get body statements excluding the docstring.

        Args:
            func_def: The FunctionDef AST node

        Returns:
            List of statement nodes, excluding docstring if present
        """
        body = func_def.body
        if body and isinstance(body[0], ast.Expr):
            if isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                return body[1:]
        return body

    def find_references(self, tree: ast.AST, target: str) -> List[ast.AST]:
        """Find all references to a name in an AST.

        Args:
            tree: The AST to search
            target: The name to find

        Returns:
            List of AST nodes that reference the target name
        """
        refs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == target:
                refs.append(node)
        return refs

    def find_self_attributes(self, tree: ast.AST) -> Set[str]:
        """Find all self.xxx attribute accesses in an AST.

        Args:
            tree: The AST to search

        Returns:
            Set of attribute names accessed on self
        """
        attrs = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == 'self':
                    attrs.add(node.attr)
        return attrs

    def find_local_assignments(self, func_def: ast.FunctionDef) -> Set[str]:
        """Find all local variable names assigned in a function.

        Args:
            func_def: The FunctionDef AST node

        Returns:
            Set of variable names that are assigned to
        """
        locals_set = set()

        for node in ast.walk(func_def):
            # Simple assignments: x = ...
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        locals_set.add(target.id)
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                locals_set.add(elt.id)

            # Annotated assignments: x: int = ...
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    locals_set.add(node.target.id)

            # For loop targets: for x in ...
            elif isinstance(node, ast.For):
                if isinstance(node.target, ast.Name):
                    locals_set.add(node.target.id)
                elif isinstance(node.target, ast.Tuple):
                    for elt in node.target.elts:
                        if isinstance(elt, ast.Name):
                            locals_set.add(elt.id)

            # Walrus operator: (x := ...)
            elif isinstance(node, ast.NamedExpr):
                if isinstance(node.target, ast.Name):
                    locals_set.add(node.target.id)

        return locals_set

    def has_return_statement(self, func_def: ast.FunctionDef) -> bool:
        """Check if a function has any return statements.

        Args:
            func_def: The FunctionDef AST node

        Returns:
            True if the function contains return statements
        """
        for node in ast.walk(func_def):
            if isinstance(node, ast.Return):
                return True
        return False

    def find_function_calls(self, tree: ast.AST) -> Set[str]:
        """Find all function/method names called in an AST.

        Args:
            tree: The AST to search

        Returns:
            Set of called function names
        """
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        return calls


def extract_method_info(method: Callable) -> dict:
    """Extract comprehensive information about a method.

    This is a convenience function that extracts all relevant information
    about a method in one call.

    Args:
        method: The method to analyze

    Returns:
        Dict with keys:
            - source: Original source code
            - func_def: AST FunctionDef node
            - params: List of parameter names
            - self_attrs: Set of self.xxx attributes used
            - locals: Set of local variable names
            - calls: Set of function calls made
            - has_return: Whether function has return statements
            - docstring: The docstring or None
    """
    extractor = MethodExtractor()
    source = extractor.extract_source(method)
    func_def = extractor.parse_to_ast(source)

    return {
        'source': source,
        'func_def': func_def,
        'params': extractor.get_parameter_names(func_def),
        'self_attrs': extractor.find_self_attributes(func_def),
        'locals': extractor.find_local_assignments(func_def),
        'calls': extractor.find_function_calls(func_def),
        'has_return': extractor.has_return_statement(func_def),
        'docstring': extractor.get_docstring(func_def),
    }
