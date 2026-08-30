"""Unit tests for C++ parameter-type canonicalization and naming helpers."""

from __future__ import annotations

import tree_sitter_cpp
from tree_sitter import Language, Node, Parser

from codeintel.languages.cpp.naming import (
    canonicalize_parameter_list,
    file_module_qname,
    filelocal_prefix,
    namespace_container_qname,
    trailing_method_qualifiers,
)


def _parser() -> Parser:
    return Parser(Language(tree_sitter_cpp.language()))


def _find(node: Node, typ: str) -> Node | None:
    if node.type == typ:
        return node
    for child in node.children:
        found = _find(child, typ)
        if found is not None:
            return found
    return None


def _parameter_list(source: str) -> tuple[bytes, Node, Node]:
    data = source.encode("utf-8")
    root = _parser().parse(data).root_node
    params = _find(root, "parameter_list")
    declarator = _find(root, "function_declarator")
    assert params is not None
    assert declarator is not None
    return data, params, declarator


def test_parameter_names_do_not_affect_identity() -> None:
    data_a, params_a, _ = _parameter_list("void foo(int x) {}")
    data_b, params_b, _ = _parameter_list("void foo(int value) {}")
    assert canonicalize_parameter_list(params_a, data_a) == canonicalize_parameter_list(
        params_b, data_b
    )
    assert canonicalize_parameter_list(params_a, data_a) == "(int)"


def test_const_ref_and_pointer_canonicalization() -> None:
    data, params, _ = _parameter_list(
        "void f(const std::string& value, Widget* widget, std::string&& other) {}"
    )
    assert (
        canonicalize_parameter_list(params, data) == "(const std::string&, Widget*, std::string&&)"
    )


def test_defaults_excluded_from_identity() -> None:
    data, params, _ = _parameter_list("void f(int value = 3) {}")
    text = canonicalize_parameter_list(params, data)
    assert "3" not in text
    assert "int" in text


def test_trailing_cv_ref_qualifiers() -> None:
    data, _, declarator = _parameter_list("class F { int value() const & { return 1; } };")
    assert trailing_method_qualifiers(declarator, data) == " const &"


def test_naming_prefixes() -> None:
    assert file_module_qname("src/a.cpp") == "@file:src/a.cpp"
    assert filelocal_prefix("src/a.cpp") == "@filelocal:src/a.cpp::"
    assert namespace_container_qname("src/a.cpp", 2, "pricing") == "@namespace:src/a.cpp:2:pricing"
