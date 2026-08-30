"""C++ language adapter for Tree-sitter-backed semantic extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Node

from codeintel.languages.base import LanguageAdapter
from codeintel.languages.cpp.naming import (
    ANONYMOUS_NAMESPACE_NAME,
    callable_stem_and_params,
    file_module_qname,
    filelocal_prefix,
    join_semantic_scope,
    namespace_container_qname,
    node_text,
)
from codeintel.languages.cpp.parser import create_cpp_parser, parse_cpp
from codeintel.models import AnalysisResult, CodeUnit, SourceSpan, Symbol, SymbolKind

_CPP_EXTENSIONS = frozenset({".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh", ".hxx"})
_MEMORY_MODULE_NAME = "<memory>"


@dataclass
class _ScopeFrame:
    kind: SymbolKind
    parent_qname: str
    semantic_scope: str
    file_local: bool = False


@dataclass
class _ExtractionState:
    relative_path: str
    source_bytes: bytes
    symbols: list[Symbol] = field(default_factory=list)
    code_units: list[CodeUnit] = field(default_factory=list)
    namespace_ordinal: int = 0
    class_semantic_qnames: set[str] = field(default_factory=set)


class CppAdapter(LanguageAdapter):
    """Extract language-neutral Symbols and CodeUnits from C++ source."""

    def __init__(self) -> None:
        self._parser = create_cpp_parser()

    @property
    def language_id(self) -> str:
        return "cpp"

    @property
    def file_extensions(self) -> frozenset[str]:
        return _CPP_EXTENSIONS

    def analyze_file(
        self,
        path: Path,
        *,
        repository_root: Path | None = None,
    ) -> AnalysisResult:
        source_bytes = path.read_bytes()
        module_name = derive_module_name(path, repository_root=repository_root)
        return self.analyze_source(
            source_bytes,
            module_name=module_name,
            path=path,
        )

    def analyze_source(
        self,
        source: str | bytes,
        *,
        module_name: str | None = None,
        path: Path | None = None,
    ) -> AnalysisResult:
        source_bytes = source.encode("utf-8") if isinstance(source, str) else source
        resolved_module_name = module_name if module_name is not None else _MEMORY_MODULE_NAME
        tree = parse_cpp(source_bytes, parser=self._parser)
        root = tree.root_node

        file_qname = file_module_qname(resolved_module_name)
        state = _ExtractionState(relative_path=resolved_module_name, source_bytes=source_bytes)
        state.symbols.append(
            Symbol(
                name=resolved_module_name,
                qualified_name=file_qname,
                kind=SymbolKind.MODULE,
                span=_span_from_node(root),
                signature=None,
                parent_qualified_name=None,
            )
        )

        root_scope = _ScopeFrame(SymbolKind.MODULE, file_qname, "", False)
        self._collect_class_names(root, state=state, scope=root_scope)
        self._extract(root, state=state, scope=root_scope)

        return AnalysisResult(
            path=path,
            language_id=self.language_id,
            module_name=resolved_module_name,
            symbols=tuple(state.symbols),
            code_units=tuple(state.code_units),
            has_syntax_errors=root.has_error,
        )

    def _collect_class_names(
        self, node: Node, *, state: _ExtractionState, scope: _ScopeFrame
    ) -> None:
        if node.type in {
            "enum_specifier",
            "union_specifier",
            "lambda_expression",
            "template_declaration",
        }:
            return
        if node.type == "namespace_definition":
            child_scope = self._preview_namespace_scope(node, state=state, scope=scope)
            body = node.child_by_field_name("body")
            if body is not None and child_scope is not None:
                for item in body.named_children:
                    self._collect_class_names(item, state=state, scope=child_scope)
            return
        if node.type in {"class_specifier", "struct_specifier"}:
            name_node = node.child_by_field_name("name")
            body = node.child_by_field_name("body")
            # Forward declarations (no body) are not CLASS definitions.
            if name_node is None or body is None:
                return
            name = node_text(name_node, state.source_bytes)
            semantic = join_semantic_scope(scope.semantic_scope, name)
            state.class_semantic_qnames.add(semantic)
            nested = _ScopeFrame(SymbolKind.CLASS, semantic, semantic, scope.file_local)
            for item in body.named_children:
                self._collect_class_names(item, state=state, scope=nested)
            return
        for child in node.named_children:
            self._collect_class_names(child, state=state, scope=scope)

    def _preview_namespace_scope(
        self, node: Node, *, state: _ExtractionState, scope: _ScopeFrame
    ) -> _ScopeFrame | None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            semantic = join_semantic_scope(scope.semantic_scope, ANONYMOUS_NAMESPACE_NAME)
            return _ScopeFrame(SymbolKind.NAMESPACE, scope.parent_qname, semantic, True)
        local_name = node_text(name_node, state.source_bytes)
        semantic = join_semantic_scope(scope.semantic_scope, local_name)
        return _ScopeFrame(SymbolKind.NAMESPACE, scope.parent_qname, semantic, scope.file_local)

    def _extract(self, node: Node, *, state: _ExtractionState, scope: _ScopeFrame) -> None:
        if node.type in {
            "enum_specifier",
            "union_specifier",
            "lambda_expression",
            "template_declaration",
        }:
            return
        if node.type == "namespace_definition":
            self._handle_namespace(node, state=state, scope=scope)
            return
        if node.type in {"class_specifier", "struct_specifier"}:
            self._handle_class(node, state=state, scope=scope)
            return
        if node.type == "function_definition":
            self._handle_function_definition(node, state=state, scope=scope)
            return
        for child in node.named_children:
            self._extract(child, state=state, scope=scope)

    def _handle_namespace(self, node: Node, *, state: _ExtractionState, scope: _ScopeFrame) -> None:
        body = node.child_by_field_name("body")
        if body is None:
            return
        state.namespace_ordinal += 1
        ordinal = state.namespace_ordinal
        name_node = node.child_by_field_name("name")
        if name_node is None:
            local_name = ANONYMOUS_NAMESPACE_NAME
            # Ordinary child symbols include the reserved <anonymous> semantic marker.
            # Reopened anonymous blocks share this same semantic path (no occurrence ordinal).
            semantic_path = join_semantic_scope(scope.semantic_scope, local_name)
            file_local = True
            child_semantic = semantic_path
        else:
            local_name = node_text(name_node, state.source_bytes)
            if not local_name:
                return
            semantic_path = join_semantic_scope(scope.semantic_scope, local_name)
            file_local = scope.file_local
            child_semantic = semantic_path

        container_qname = namespace_container_qname(state.relative_path, ordinal, semantic_path)
        state.symbols.append(
            Symbol(
                name=local_name,
                qualified_name=container_qname,
                kind=SymbolKind.NAMESPACE,
                span=_span_from_node(node),
                signature=None,
                parent_qualified_name=scope.parent_qname,
            )
        )
        child_scope = _ScopeFrame(
            SymbolKind.NAMESPACE,
            container_qname,
            child_semantic,
            file_local,
        )
        for child in body.named_children:
            self._extract(child, state=state, scope=child_scope)

    def _handle_class(self, node: Node, *, state: _ExtractionState, scope: _ScopeFrame) -> None:
        name_node = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        # Require a definition body; skip forward declarations such as `class Foo;`.
        if name_node is None or body is None:
            return
        name = node_text(name_node, state.source_bytes)
        if not name:
            return
        semantic = join_semantic_scope(scope.semantic_scope, name)
        qname = (
            f"{filelocal_prefix(state.relative_path)}{semantic}" if scope.file_local else semantic
        )
        span = _span_from_node(node)
        state.symbols.append(
            Symbol(
                name=name,
                qualified_name=qname,
                kind=SymbolKind.CLASS,
                span=span,
                signature=_header_before_body(node, state.source_bytes),
                parent_qualified_name=scope.parent_qname,
            )
        )
        state.code_units.append(
            CodeUnit(
                symbol_qualified_name=qname,
                kind=SymbolKind.CLASS,
                source_text=node_text(node, state.source_bytes),
                span=span,
            )
        )
        child_scope = _ScopeFrame(SymbolKind.CLASS, qname, semantic, scope.file_local)
        for child in body.named_children:
            self._extract(child, state=state, scope=child_scope)

    def _handle_function_definition(
        self, node: Node, *, state: _ExtractionState, scope: _ScopeFrame
    ) -> None:
        body = node.child_by_field_name("body")
        if body is None:
            return
        declarator = node.child_by_field_name("declarator")
        if declarator is None:
            return
        function_declarator = _find_function_declarator(declarator)
        if function_declarator is None:
            return
        parameter_list = function_declarator.child_by_field_name("parameters")
        name_info = _callable_name_info(function_declarator, state.source_bytes)
        if name_info is None:
            return
        name, qualifier_parts = name_info
        if not name:
            return

        file_local = scope.file_local
        if scope.kind is not SymbolKind.CLASS and _is_static_definition(node):
            file_local = True

        if scope.kind is SymbolKind.CLASS:
            qname = callable_stem_and_params(
                semantic_scope=scope.semantic_scope,
                callable_name=name,
                parameter_list=parameter_list,
                function_declarator=function_declarator,
                source_bytes=state.source_bytes,
                file_local=file_local,
                relative_path=state.relative_path,
            )
            self._emit_callable(
                state=state,
                name=name,
                qname=qname,
                kind=SymbolKind.METHOD,
                parent_qname=scope.parent_qname,
                node=node,
            )
            return

        if qualifier_parts:
            class_semantic = join_semantic_scope(scope.semantic_scope, *qualifier_parts)
            matched = _resolve_same_file_class_qname(class_semantic, state)
            params_suffix = _params_suffix(parameter_list, function_declarator, state.source_bytes)
            if matched is not None:
                # Method qname uses semantic class path + overload identity.
                semantic_callable = f"{class_semantic}::{name}{params_suffix}"
                qname = (
                    f"{filelocal_prefix(state.relative_path)}{semantic_callable}"
                    if matched.startswith("@filelocal:")
                    else semantic_callable
                )
                self._emit_callable(
                    state=state,
                    name=name,
                    qname=qname,
                    kind=SymbolKind.METHOD,
                    parent_qname=matched,
                    node=node,
                )
                return

            # Cross-file / unknown class: keep qualified semantic qname as FUNCTION.
            qualified_stem = join_semantic_scope(scope.semantic_scope, *qualifier_parts, name)
            qname = f"{qualified_stem}{params_suffix}"
            if file_local:
                qname = f"{filelocal_prefix(state.relative_path)}{qname}"
            self._emit_callable(
                state=state,
                name=name,
                qname=qname,
                kind=SymbolKind.FUNCTION,
                parent_qname=scope.parent_qname,
                node=node,
            )
            return

        qname = callable_stem_and_params(
            semantic_scope=scope.semantic_scope,
            callable_name=name,
            parameter_list=parameter_list,
            function_declarator=function_declarator,
            source_bytes=state.source_bytes,
            file_local=file_local,
            relative_path=state.relative_path,
        )
        self._emit_callable(
            state=state,
            name=name,
            qname=qname,
            kind=SymbolKind.FUNCTION,
            parent_qname=scope.parent_qname,
            node=node,
        )

    def _emit_callable(
        self,
        *,
        state: _ExtractionState,
        name: str,
        qname: str,
        kind: SymbolKind,
        parent_qname: str,
        node: Node,
    ) -> None:
        if any(symbol.qualified_name == qname for symbol in state.symbols):
            return
        span = _span_from_node(node)
        state.symbols.append(
            Symbol(
                name=name,
                qualified_name=qname,
                kind=kind,
                span=span,
                signature=_header_before_body(node, state.source_bytes),
                parent_qualified_name=parent_qname,
            )
        )
        state.code_units.append(
            CodeUnit(
                symbol_qualified_name=qname,
                kind=kind,
                source_text=node_text(node, state.source_bytes),
                span=span,
            )
        )


def derive_module_name(path: Path, *, repository_root: Path | None = None) -> str:
    """Derive C++ module_name as repository-relative POSIX path including extension."""
    if repository_root is not None:
        try:
            relative = path.resolve().relative_to(repository_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Path {path} is outside repository root {repository_root}") from exc
        return relative.as_posix()
    return path.name


def _resolve_same_file_class_qname(class_semantic: str, state: _ExtractionState) -> str | None:
    if class_semantic not in state.class_semantic_qnames:
        return None
    prefix = filelocal_prefix(state.relative_path)
    for symbol in state.symbols:
        if symbol.kind is not SymbolKind.CLASS:
            continue
        semantic = (
            symbol.qualified_name[len(prefix) :]
            if symbol.qualified_name.startswith(prefix)
            else symbol.qualified_name
        )
        if semantic == class_semantic:
            return symbol.qualified_name
    # Class body not yet emitted (should be rare with pass ordering).
    return class_semantic


def _params_suffix(
    parameter_list: Node | None,
    function_declarator: Node | None,
    source_bytes: bytes,
) -> str:
    from codeintel.languages.cpp.naming import (
        canonicalize_parameter_list,
        trailing_method_qualifiers,
    )

    params = canonicalize_parameter_list(parameter_list, source_bytes)
    quals = (
        trailing_method_qualifiers(function_declarator, source_bytes)
        if function_declarator is not None
        else ""
    )
    return f"{params}{quals}"


def _find_function_declarator(node: Node) -> Node | None:
    current: Node | None = node
    while current is not None:
        if current.type == "function_declarator":
            return current
        nested = current.child_by_field_name("declarator")
        if nested is not None:
            current = nested
            continue
        for child in current.named_children:
            if child.type == "function_declarator":
                return child
        return None
    return None


def _callable_name_info(
    function_declarator: Node,
    source_bytes: bytes,
) -> tuple[str, list[str]] | None:
    declarator = function_declarator.child_by_field_name("declarator")
    if declarator is None:
        return None
    if declarator.type in {"identifier", "field_identifier", "destructor_name", "operator_name"}:
        return node_text(declarator, source_bytes), []
    if declarator.type == "qualified_identifier":
        parts: list[str] = []
        current: Node | None = declarator
        while current is not None and current.type == "qualified_identifier":
            scope = current.child_by_field_name("scope")
            name = current.child_by_field_name("name")
            if scope is not None:
                parts.append(node_text(scope, source_bytes))
            if name is None:
                return None
            if name.type == "qualified_identifier":
                current = name
                continue
            return node_text(name, source_bytes), parts
    return None


def _is_static_definition(node: Node) -> bool:
    for child in node.children:
        if child.type != "storage_class_specifier":
            continue
        text = (
            child.text.decode("utf-8")
            if isinstance(child.text, (bytes, bytearray))
            else str(child.text)
        )
        return text == "static"
    return False


def _header_before_body(node: Node, source_bytes: bytes) -> str:
    body = node.child_by_field_name("body")
    if body is None:
        return node_text(node, source_bytes).rstrip()
    return source_bytes[node.start_byte : body.start_byte].decode("utf-8").rstrip()


def _span_from_node(node: Node) -> SourceSpan:
    start_line = node.start_point[0] + 1
    end_row, end_column = node.end_point
    end_line = end_row if end_column == 0 and end_row > 0 else end_row + 1
    if end_line < start_line:
        end_line = start_line
    return SourceSpan(
        start_line=start_line,
        end_line=end_line,
        start_byte=node.start_byte,
        end_byte=node.end_byte,
    )
