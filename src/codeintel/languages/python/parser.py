"""Tree-sitter wrapper for Python source parsing."""

from __future__ import annotations

import tree_sitter_python
from tree_sitter import Language, Parser, Tree


def create_python_parser() -> Parser:
    """Construct a Tree-sitter parser configured for Python."""
    language = Language(tree_sitter_python.language())
    return Parser(language)


def parse_python(source: bytes, parser: Parser | None = None) -> Tree:
    """Parse Python source bytes into a Tree-sitter syntax tree."""
    active_parser = parser if parser is not None else create_python_parser()
    return active_parser.parse(source)
