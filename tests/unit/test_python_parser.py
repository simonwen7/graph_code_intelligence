"""Tests for the Python Tree-sitter parser wrapper."""

from codeintel.languages.python.parser import create_python_parser, parse_python


def test_parse_python_smoke() -> None:
    """The parser should produce a module tree without syntax errors."""
    tree = parse_python(b"def example():\n    pass\n", parser=create_python_parser())

    assert tree.root_node.type == "module"
    assert not tree.root_node.has_error
