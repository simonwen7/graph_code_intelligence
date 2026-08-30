"""Deterministic C++ qualified-name and parameter-type canonicalization."""

from __future__ import annotations

from tree_sitter import Node

FILE_MODULE_PREFIX = "@file:"
FILELOCAL_PREFIX = "@filelocal:"
NAMESPACE_PREFIX = "@namespace:"
ANONYMOUS_NAMESPACE_NAME = "<anonymous>"


def file_module_qname(relative_path: str) -> str:
    """Return the synthetic MODULE qname for a repository-relative C++ path."""
    return f"{FILE_MODULE_PREFIX}{relative_path}"


def filelocal_prefix(relative_path: str) -> str:
    """Return the file-local semantic identity prefix for a C++ path."""
    return f"{FILELOCAL_PREFIX}{relative_path}::"


def namespace_container_qname(
    relative_path: str,
    ordinal: int,
    semantic_namespace_path: str,
) -> str:
    """Return a file-local syntactic namespace container identity.

    Format:
    ``@namespace:<relative-path>:<source-order-ordinal>:<semantic-namespace-path>``

    Anonymous namespaces use semantic path component ``<anonymous>`` (or nested
    ``pricing::<anonymous>``).
    """
    return f"{NAMESPACE_PREFIX}{relative_path}:{ordinal}:{semantic_namespace_path}"


def join_semantic_scope(*parts: str) -> str:
    """Join non-empty C++ semantic scope parts with ``::``."""
    return "::".join(part for part in parts if part)


def node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")


def collapse_ws(text: str) -> str:
    return " ".join(text.split())


def canonicalize_parameter_list(parameter_list: Node | None, source_bytes: bytes) -> str:
    """Return compact ``(type, type, ...)`` text without parameter names/defaults."""
    if parameter_list is None:
        return "()"
    parts: list[str] = []
    for child in parameter_list.named_children:
        if child.type == "parameter_declaration":
            parts.append(canonicalize_parameter_declaration(child, source_bytes))
        elif child.type == "variadic_parameter_declaration":
            parts.append("...")
        elif child.type == "optional_parameter_declaration":
            parts.append(canonicalize_parameter_declaration(child, source_bytes))
        else:
            parts.append(collapse_ws(node_text(child, source_bytes)))
    return f"({', '.join(parts)})"


def canonicalize_parameter_declaration(node: Node, source_bytes: bytes) -> str:
    """Canonicalize one parameter's type/declarator shape without the identifier."""
    type_parts: list[str] = []
    declarator: Node | None = None
    for index in range(node.child_count):
        child = node.child(index)
        if child is None or not child.is_named:
            continue
        field = node.field_name_for_child(index)
        if field == "declarator":
            declarator = child
            continue
        if child.type in {
            "type_qualifier",
            "primitive_type",
            "type_identifier",
            "qualified_identifier",
            "sized_type_specifier",
            "struct_specifier",
            "class_specifier",
            "enum_specifier",
            "union_specifier",
            "template_type",
            "dependent_type",
            "placeholder_type_specifier",
        }:
            type_parts.append(_canonicalize_type_node(child, source_bytes))
        elif field == "type":
            type_parts.append(_canonicalize_type_node(child, source_bytes))
    type_text = collapse_ws(" ".join(part for part in type_parts if part))
    if declarator is None:
        return type_text or collapse_ws(node_text(node, source_bytes))
    return _apply_declarator(type_text, declarator, source_bytes)


def _canonicalize_type_node(node: Node, source_bytes: bytes) -> str:
    if node.type == "type_qualifier":
        return collapse_ws(node_text(node, source_bytes))
    if node.type == "qualified_identifier":
        return _canonicalize_qualified_identifier(node, source_bytes)
    if node.type == "template_type":
        return _canonicalize_template_type(node, source_bytes)
    return collapse_ws(node_text(node, source_bytes))


def _canonicalize_qualified_identifier(node: Node, source_bytes: bytes) -> str:
    scope = node.child_by_field_name("scope")
    name = node.child_by_field_name("name")
    if scope is None or name is None:
        return collapse_ws(node_text(node, source_bytes))
    return (
        f"{_canonicalize_type_node(scope, source_bytes)}"
        f"::{_canonicalize_type_node(name, source_bytes)}"
    )


def _canonicalize_template_type(node: Node, source_bytes: bytes) -> str:
    name = node.child_by_field_name("name")
    arguments = node.child_by_field_name("arguments")
    if name is None:
        return collapse_ws(node_text(node, source_bytes))
    name_text = _canonicalize_type_node(name, source_bytes)
    if arguments is None:
        return name_text
    args: list[str] = []
    for child in arguments.named_children:
        if child.type == "type_descriptor":
            args.append(_canonicalize_type_descriptor(child, source_bytes))
        else:
            args.append(collapse_ws(node_text(child, source_bytes)))
    return f"{name_text}<{', '.join(args)}>"


def _canonicalize_type_descriptor(node: Node, source_bytes: bytes) -> str:
    parts: list[str] = []
    for child in node.named_children:
        parts.append(_canonicalize_type_node(child, source_bytes))
    return collapse_ws(" ".join(parts))


def _apply_declarator(type_text: str, declarator: Node, source_bytes: bytes) -> str:
    """Fold pointer/reference/array declarators onto the type; drop the identifier."""
    if declarator.type == "identifier":
        return type_text
    if declarator.type == "pointer_declarator":
        inner = declarator.child_by_field_name("declarator")
        quals = [
            collapse_ws(node_text(child, source_bytes))
            for child in declarator.named_children
            if child.type == "type_qualifier"
        ]
        prefix = "*" + "".join(f" {q}" for q in quals)
        base = f"{type_text}{prefix}" if type_text else prefix
        if inner is None:
            return base
        return _apply_declarator(base, inner, source_bytes)
    if declarator.type == "reference_declarator":
        text = node_text(declarator, source_bytes)
        ref = "&&" if "&&" in text.split(")", 1)[0] else "&"
        # Find nested named declarator (identifier or deeper).
        inner = next((child for child in declarator.named_children if child.is_named), None)
        base = f"{type_text}{ref}" if type_text else ref
        if inner is None or inner.type == "identifier":
            return base
        return _apply_declarator(base, inner, source_bytes)
    if declarator.type == "array_declarator":
        inner = declarator.child_by_field_name("declarator")
        size = declarator.child_by_field_name("size")
        size_text = collapse_ws(node_text(size, source_bytes)) if size is not None else ""
        suffix = f"[{size_text}]"
        if inner is None or inner.type == "identifier":
            return f"{type_text}{suffix}"
        return _apply_declarator(f"{type_text}{suffix}", inner, source_bytes)
    if declarator.type == "function_declarator":
        # Function-pointer parameters: keep a conservative collapsed form.
        return collapse_ws(f"{type_text} {node_text(declarator, source_bytes)}")
    if declarator.type == "parenthesized_declarator":
        inner = next((child for child in declarator.named_children if child.is_named), None)
        if inner is None:
            return type_text
        return _apply_declarator(type_text, inner, source_bytes)
    if declarator.type == "abstract_pointer_declarator":
        return f"{type_text}*" if type_text else "*"
    if declarator.type == "abstract_reference_declarator":
        text = node_text(declarator, source_bytes)
        ref = "&&" if "&&" in text else "&"
        return f"{type_text}{ref}" if type_text else ref
    if declarator.type == "abstract_array_declarator":
        return f"{type_text}[]" if type_text else "[]"
    if declarator.type == "abstract_function_declarator":
        return collapse_ws(f"{type_text}{node_text(declarator, source_bytes)}")
    # Conservative fallback: strip trailing identifier-like token if present.
    raw = collapse_ws(node_text(declarator, source_bytes))
    if type_text:
        return collapse_ws(f"{type_text} {raw}")
    return raw


def trailing_method_qualifiers(function_declarator: Node, source_bytes: bytes) -> str:
    """Return identity-relevant trailing qualifiers: const/volatile/&/&& after params."""
    ordered: list[str] = []
    after = False
    for child in function_declarator.children:
        if child.type == "parameter_list":
            after = True
            continue
        if not after:
            continue
        if child.type == "type_qualifier":
            text = collapse_ws(node_text(child, source_bytes))
            if text in {"const", "volatile"}:
                ordered.append(text)
        elif child.type == "ref_qualifier":
            text = collapse_ws(node_text(child, source_bytes))
            if text in {"&", "&&"}:
                ordered.append(text)
        elif child.type in {"&", "&&"}:
            ordered.append(child.type)
    if not ordered:
        return ""
    return " " + " ".join(ordered)


def callable_stem_and_params(
    *,
    semantic_scope: str,
    callable_name: str,
    parameter_list: Node | None,
    function_declarator: Node | None,
    source_bytes: bytes,
    file_local: bool,
    relative_path: str,
) -> str:
    """Build overload-safe semantic callable qname."""
    params = canonicalize_parameter_list(parameter_list, source_bytes)
    quals = ""
    if function_declarator is not None:
        quals = trailing_method_qualifiers(function_declarator, source_bytes)
    local = f"{callable_name}{params}{quals}"
    scoped = join_semantic_scope(semantic_scope, local) if semantic_scope else local
    if file_local:
        return f"{filelocal_prefix(relative_path)}{scoped}"
    return scoped
