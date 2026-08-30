"""Tree-sitter wrapper for C++ source parsing."""

from __future__ import annotations

import tree_sitter_cpp
from tree_sitter import Language, Parser, Tree


def create_cpp_parser() -> Parser:
    """Construct a Tree-sitter parser configured for C++."""
    language = Language(tree_sitter_cpp.language())
    return Parser(language)


def parse_cpp(source: bytes, parser: Parser | None = None) -> Tree:
    """Parse C++ source bytes into a Tree-sitter syntax tree."""
    active_parser = parser if parser is not None else create_cpp_parser()
    return active_parser.parse(source)
