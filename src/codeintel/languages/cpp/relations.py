"""C++-specific static relationship extraction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Node

from codeintel.languages.cpp.naming import (
    ANONYMOUS_NAMESPACE_NAME,
    file_module_qname,
    join_semantic_scope,
    node_text,
)
from codeintel.languages.cpp.parser import parse_cpp
from codeintel.models import (
    AnalysisResult,
    Relation,
    RelationKind,
    ResolutionStatus,
    SourceSpan,
    Symbol,
    SymbolKind,
)


class CppRelationExtractor:
    """Extract IMPORTS, CALLS, and INHERITS from C++ source (no REFERENCES)."""

    def extract_relations(
        self,
        path: Path,
        *,
        repository_root: Path,
        analysis: AnalysisResult,
        symbols_by_qualified_name: Mapping[str, Symbol],
    ) -> tuple[Relation, ...]:
        source_bytes = path.read_bytes()
        tree = parse_cpp(source_bytes)
        extractor = _FileExtractor(
            path=path,
            repository_root=repository_root,
            source_bytes=source_bytes,
            analysis=analysis,
            symbols=symbols_by_qualified_name,
        )
        extractor.visit(tree.root_node)
        return tuple(extractor.relations)


@dataclass
class _Scope:
    enclosing_qname: str
    enclosing_kind: SymbolKind
    semantic_scope: str
    parent: _Scope | None
    locals: set[str] = field(default_factory=set)

    def class_semantic(self) -> str | None:
        current: _Scope | None = self
        while current is not None:
            if current.enclosing_kind is SymbolKind.CLASS:
                return current.semantic_scope
            current = current.parent
        return None


class _FileExtractor:
    def __init__(
        self,
        *,
        path: Path,
        repository_root: Path,
        source_bytes: bytes,
        analysis: AnalysisResult,
        symbols: Mapping[str, Symbol],
    ) -> None:
        self.path = path
        self.repository_root = repository_root.resolve()
        self.source_bytes = source_bytes
        self.analysis = analysis
        self.symbols = symbols
        self.relations: list[Relation] = []
        self.module_qname = file_module_qname(analysis.module_name)
        self.relative_path = analysis.module_name
        self._file_modules = {
            symbol.qualified_name[len("@file:") :]
            for symbol in symbols.values()
            if symbol.kind is SymbolKind.MODULE and symbol.qualified_name.startswith("@file:")
        }
        self._callables_by_stem = _index_callables(symbols)

    def visit(self, root: Node) -> None:
        scope = _Scope(self.module_qname, SymbolKind.MODULE, "", None)
        self._visit(root, scope)

    def _visit(self, node: Node, scope: _Scope) -> None:
        if node.type == "preproc_include":
            self._handle_include(node)
            return
        if node.type == "namespace_definition":
            self._visit_namespace(node, scope)
            return
        if node.type in {"class_specifier", "struct_specifier"}:
            self._visit_class(node, scope)
            return
        if node.type == "function_definition":
            self._visit_function(node, scope)
            return
        if node.type == "call_expression":
            self._handle_call(node, scope)
            # Still visit children for nested calls.
            for child in node.named_children:
                if child is node.child_by_field_name("function"):
                    continue
                self._visit(child, scope)
            return
        if node.type == "declaration":
            self._bind_locals_from_declaration(node, scope)
        for child in node.named_children:
            self._visit(child, scope)

    def _visit_namespace(self, node: Node, parent: _Scope) -> None:
        name_node = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        if body is None:
            return
        # Find matching namespace symbol by span proximity / parent.
        container = self._namespace_symbol_for_node(node)
        qname = container.qualified_name if container is not None else parent.enclosing_qname
        if name_node is None:
            semantic = join_semantic_scope(parent.semantic_scope, ANONYMOUS_NAMESPACE_NAME)
        else:
            local = node_text(name_node, self.source_bytes)
            semantic = join_semantic_scope(parent.semantic_scope, local)
        child = _Scope(qname, SymbolKind.NAMESPACE, semantic, parent)
        for item in body.named_children:
            self._visit(item, child)

    def _visit_class(self, node: Node, parent: _Scope) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = node_text(name_node, self.source_bytes)
        semantic = f"{parent.semantic_scope}::{name}" if parent.semantic_scope else name
        class_symbol = self._class_symbol_for_semantic(semantic)
        if class_symbol is None:
            return
        self._handle_bases(node, parent, class_symbol.qualified_name)
        body = node.child_by_field_name("body")
        if body is None:
            return
        child = _Scope(class_symbol.qualified_name, SymbolKind.CLASS, semantic, parent)
        for item in body.named_children:
            self._visit(item, child)

    def _visit_function(self, node: Node, parent: _Scope) -> None:
        body = node.child_by_field_name("body")
        if body is None:
            return
        symbol = self._callable_symbol_for_definition(node, parent)
        if symbol is None:
            # Still walk body under parent scope for calls outside indexed defs.
            for item in body.named_children:
                self._visit(item, parent)
            return
        child = _Scope(symbol.qualified_name, symbol.kind, parent.semantic_scope, parent)
        # Bind parameters as locals.
        declarator = node.child_by_field_name("declarator")
        if declarator is not None:
            function_declarator = _find_function_declarator(declarator)
            if function_declarator is not None:
                params = function_declarator.child_by_field_name("parameters")
                if params is not None:
                    for name in _parameter_names(params, self.source_bytes):
                        child.locals.add(name)
        for item in body.named_children:
            self._visit(item, child)

    def _handle_include(self, node: Node) -> None:
        path_node = node.child_by_field_name("path")
        if path_node is None:
            return
        target_text = node_text(path_node, self.source_bytes)
        if path_node.type == "system_lib_string":
            self.relations.append(
                Relation(
                    kind=RelationKind.IMPORTS,
                    source_qualified_name=self.module_qname,
                    target_qualified_name=None,
                    target_text=target_text,
                    resolution=ResolutionStatus.UNRESOLVED,
                    path=self.path,
                    span=_span_from_node(path_node),
                )
            )
            return
        # Quoted include.
        inner = target_text.strip().strip('"')
        resolved = self._resolve_quoted_include(inner)
        self.relations.append(
            Relation(
                kind=RelationKind.IMPORTS,
                source_qualified_name=self.module_qname,
                target_qualified_name=resolved,
                target_text=target_text,
                resolution=(
                    ResolutionStatus.RESOLVED
                    if resolved is not None
                    else ResolutionStatus.UNRESOLVED
                ),
                path=self.path,
                span=_span_from_node(path_node),
            )
        )

    def _resolve_quoted_include(self, include_path: str) -> str | None:
        candidates: list[str] = []
        # 1) relative to including file directory
        local = (self.path.parent / include_path).resolve()
        try:
            rel_local = local.relative_to(self.repository_root).as_posix()
        except ValueError:
            rel_local = None
        if rel_local is not None and rel_local in self._file_modules:
            candidates.append(rel_local)
        # 2) relative to repository root
        root_candidate = include_path.lstrip("./")
        if root_candidate in self._file_modules and root_candidate not in candidates:
            candidates.append(root_candidate)
        if len(candidates) == 1:
            return file_module_qname(candidates[0])
        return None

    def _handle_bases(self, class_node: Node, scope: _Scope, class_qname: str) -> None:
        bases = class_node.child_by_field_name("base_class_clause") or next(
            (child for child in class_node.named_children if child.type == "base_class_clause"),
            None,
        )
        # Field may not be named in all versions; search children.
        if bases is None:
            for child in class_node.children:
                if child.type == "base_class_clause":
                    bases = child
                    break
        if bases is None:
            return
        for child in bases.named_children:
            if child.type == "access_specifier":
                continue
            if child.type not in {"type_identifier", "qualified_identifier", "template_type"}:
                continue
            target_text = node_text(child, self.source_bytes)
            target = None
            resolution = ResolutionStatus.UNRESOLVED
            if child.type == "type_identifier":
                target = self._resolve_unique_class(target_text, scope.semantic_scope)
                if target is not None:
                    resolution = ResolutionStatus.RESOLVED
            elif child.type == "qualified_identifier":
                target = self._resolve_unique_class_qname(target_text)
                if target is not None:
                    resolution = ResolutionStatus.RESOLVED
            self.relations.append(
                Relation(
                    kind=RelationKind.INHERITS,
                    source_qualified_name=class_qname,
                    target_qualified_name=target,
                    target_text=target_text,
                    resolution=resolution,
                    path=self.path,
                    span=_span_from_node(child),
                )
            )

    def _handle_call(self, node: Node, scope: _Scope) -> None:
        function = node.child_by_field_name("function")
        if function is None:
            return
        target_text = node_text(function, self.source_bytes)
        target, resolution = self._resolve_call(function, scope)
        self.relations.append(
            Relation(
                kind=RelationKind.CALLS,
                source_qualified_name=scope.enclosing_qname,
                target_qualified_name=target,
                target_text=target_text,
                resolution=resolution,
                path=self.path,
                span=_span_from_node(function),
            )
        )

    def _resolve_call(self, function: Node, scope: _Scope) -> tuple[str | None, ResolutionStatus]:
        if function.type == "identifier":
            name = node_text(function, self.source_bytes)
            if name in scope.locals or self._is_local_shadowed(scope, name):
                return None, ResolutionStatus.UNRESOLVED
            return self._resolve_bare_callable(name, scope.semantic_scope)
        if function.type == "qualified_identifier":
            text = node_text(function, self.source_bytes)
            # Class-qualified vs namespace-qualified: if leading part is a CLASS → PROBABLE
            parts = text.split("::")
            if len(parts) >= 2:
                head = "::".join(parts[:-1])
                method = parts[-1]
                class_qname = self._resolve_unique_class_qname(head)
                if class_qname is None and scope.semantic_scope:
                    class_qname = self._resolve_unique_class(head, scope.semantic_scope)
                if class_qname is not None:
                    return self._resolve_class_method(class_qname, method, probable=True)
                return self._resolve_qualified_callable(text)
            return self._resolve_qualified_callable(text)
        if function.type == "field_expression":
            return self._resolve_field_call(function, scope)
        return None, ResolutionStatus.UNRESOLVED

    def _resolve_field_call(self, node: Node, scope: _Scope) -> tuple[str | None, ResolutionStatus]:
        argument = node.child_by_field_name("argument")
        field = node.child_by_field_name("field")
        if argument is None or field is None:
            return None, ResolutionStatus.UNRESOLVED
        field_name = node_text(field, self.source_bytes)
        arg_text = node_text(argument, self.source_bytes)
        if arg_text == "this":
            class_semantic = scope.class_semantic()
            if class_semantic is None:
                return None, ResolutionStatus.UNRESOLVED
            class_qname = self._class_symbol_for_semantic(class_semantic)
            if class_qname is None:
                return None, ResolutionStatus.UNRESOLVED
            return self._resolve_class_method(class_qname.qualified_name, field_name, probable=True)
        # obj.bar / ptr->bar without types → UNRESOLVED
        return None, ResolutionStatus.UNRESOLVED

    def _resolve_bare_callable(
        self, name: str, semantic_scope: str
    ) -> tuple[str | None, ResolutionStatus]:
        candidates = self._callables_matching_stem(name, semantic_scope)
        if len(candidates) == 1:
            return candidates[0], ResolutionStatus.RESOLVED
        return None, ResolutionStatus.UNRESOLVED

    def _resolve_qualified_callable(
        self, qualified_stem: str
    ) -> tuple[str | None, ResolutionStatus]:
        # Stem without params: match callables whose qname startswith stem + '('
        # or equal semantic path before '('.
        candidates = [
            qname
            for qname in self._callables_by_stem.get(qualified_stem.split("::")[-1], ())
            if _callable_semantic_stem(qname) == qualified_stem
            or qname.startswith(f"{qualified_stem}(")
        ]
        # Also direct stem index by full qualified name without params.
        extra = [
            qname
            for qname, symbol in self.symbols.items()
            if symbol.kind in {SymbolKind.FUNCTION, SymbolKind.METHOD}
            and _callable_semantic_stem(qname) == qualified_stem
        ]
        merged = sorted(set(candidates) | set(extra))
        if len(merged) == 1:
            return merged[0], ResolutionStatus.RESOLVED
        return None, ResolutionStatus.UNRESOLVED

    def _resolve_class_method(
        self, class_qname: str, method_name: str, *, probable: bool
    ) -> tuple[str | None, ResolutionStatus]:
        class_symbol = self.symbols.get(class_qname)
        if class_symbol is None or class_symbol.kind is not SymbolKind.CLASS:
            return None, ResolutionStatus.UNRESOLVED
        semantic = _strip_filelocal(class_qname)
        candidates = [
            qname
            for qname, symbol in self.symbols.items()
            if symbol.kind in {SymbolKind.FUNCTION, SymbolKind.METHOD}
            and symbol.parent_qualified_name == class_qname
            and symbol.name == method_name
        ]
        if not candidates:
            # Also match by semantic stem Class::method
            prefix = f"{semantic}::{method_name}"
            candidates = [
                qname
                for qname, symbol in self.symbols.items()
                if symbol.kind in {SymbolKind.FUNCTION, SymbolKind.METHOD}
                and (
                    _callable_semantic_stem(qname) == prefix
                    or qname.startswith(f"{prefix}(")
                    or (
                        qname.startswith("@filelocal:") and _callable_semantic_stem(qname) == prefix
                    )
                )
            ]
        candidates = sorted(set(candidates))
        if len(candidates) == 1:
            status = ResolutionStatus.PROBABLE if probable else ResolutionStatus.RESOLVED
            return candidates[0], status
        return None, ResolutionStatus.UNRESOLVED

    def _callables_matching_stem(self, name: str, semantic_scope: str) -> list[str]:
        results: list[str] = []
        for qname in self._callables_by_stem.get(name, ()):
            stem = _callable_semantic_stem(qname)
            # Prefer same semantic scope; allow global if scope empty.
            if semantic_scope:
                if stem == f"{semantic_scope}::{name}" or stem.endswith(f"::{name}"):
                    # Require exact scope match when scoped.
                    if stem == f"{semantic_scope}::{name}":
                        results.append(qname)
                elif stem == name and not semantic_scope:
                    results.append(qname)
            else:
                if stem == name:
                    results.append(qname)
        # If scoped search empty, also try exact global name in scope chain walk-up.
        if not results and semantic_scope:
            parts = semantic_scope.split("::")
            for index in range(len(parts), -1, -1):
                prefix = "::".join(parts[:index])
                expected = f"{prefix}::{name}" if prefix else name
                for qname in self._callables_by_stem.get(name, ()):
                    if _callable_semantic_stem(qname) == expected:
                        results.append(qname)
                if results:
                    break
        return sorted(set(results))

    def _resolve_unique_class(self, name: str, semantic_scope: str) -> str | None:
        candidates: list[str] = []
        parts = semantic_scope.split("::") if semantic_scope else []
        for index in range(len(parts), -1, -1):
            prefix = "::".join(parts[:index])
            expected = f"{prefix}::{name}" if prefix else name
            for symbol in self.symbols.values():
                if symbol.kind is not SymbolKind.CLASS:
                    continue
                if _strip_filelocal(symbol.qualified_name) == expected:
                    candidates.append(symbol.qualified_name)
            if candidates:
                break
        candidates = sorted(set(candidates))
        return candidates[0] if len(candidates) == 1 else None

    def _resolve_unique_class_qname(self, text: str) -> str | None:
        matches = [
            symbol.qualified_name
            for symbol in self.symbols.values()
            if symbol.kind is SymbolKind.CLASS and _strip_filelocal(symbol.qualified_name) == text
        ]
        matches = sorted(set(matches))
        return matches[0] if len(matches) == 1 else None

    def _class_symbol_for_semantic(self, semantic: str) -> Symbol | None:
        qname = self._resolve_unique_class_qname(semantic)
        if qname is None:
            return None
        return self.symbols.get(qname)

    def _namespace_symbol_for_node(self, node: Node) -> Symbol | None:
        for symbol in self.analysis.symbols:
            if symbol.kind is SymbolKind.NAMESPACE and symbol.span.start_byte == node.start_byte:
                return self.symbols.get(symbol.qualified_name)
        return None

    def _callable_symbol_for_definition(self, node: Node, scope: _Scope) -> Symbol | None:
        for symbol in self.analysis.symbols:
            if symbol.kind not in {SymbolKind.FUNCTION, SymbolKind.METHOD}:
                continue
            if symbol.span.start_byte == node.start_byte and symbol.span.end_byte == node.end_byte:
                return self.symbols.get(symbol.qualified_name)
        return None

    def _bind_locals_from_declaration(self, node: Node, scope: _Scope) -> None:
        for name in _declaration_local_names(node, self.source_bytes):
            scope.locals.add(name)

    def _is_local_shadowed(self, scope: _Scope, name: str) -> bool:
        current: _Scope | None = scope
        while current is not None:
            if name in current.locals:
                return True
            current = current.parent
        return False


def _index_callables(symbols: Mapping[str, Symbol]) -> dict[str, tuple[str, ...]]:
    buckets: dict[str, list[str]] = {}
    for qname, symbol in symbols.items():
        if symbol.kind not in {SymbolKind.FUNCTION, SymbolKind.METHOD}:
            continue
        buckets.setdefault(symbol.name, []).append(qname)
    return {key: tuple(sorted(values)) for key, values in buckets.items()}


def _callable_semantic_stem(qname: str) -> str:
    """Strip filelocal prefix and parameter list / trailing quals from a callable qname."""
    text = _strip_filelocal(qname)
    paren = text.find("(")
    if paren == -1:
        return text
    return text[:paren]


def _strip_filelocal(qname: str) -> str:
    if not qname.startswith("@filelocal:"):
        return qname
    # @filelocal:<path>::rest
    rest = qname[len("@filelocal:") :]
    marker = "::"
    index = rest.find(marker)
    if index == -1:
        return qname
    return rest[index + len(marker) :]


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


def _parameter_names(parameters: Node, source_bytes: bytes) -> list[str]:
    names: list[str] = []
    for child in parameters.named_children:
        if child.type not in {"parameter_declaration", "optional_parameter_declaration"}:
            continue
        declarator = child.child_by_field_name("declarator")
        if declarator is None:
            continue
        ident = _declarator_identifier(declarator)
        if ident is not None:
            names.append(node_text(ident, source_bytes))
    return names


def _declarator_identifier(node: Node) -> Node | None:
    current: Node | None = node
    while current is not None:
        if current.type == "identifier":
            return current
        nested = current.child_by_field_name("declarator")
        if nested is not None:
            current = nested
            continue
        named = [child for child in current.named_children if child.type == "identifier"]
        return named[0] if named else None
    return None


def _declaration_local_names(node: Node, source_bytes: bytes) -> list[str]:
    names: list[str] = []
    for child in node.named_children:
        if child.type in {"init_declarator", "identifier"}:
            if child.type == "identifier":
                names.append(node_text(child, source_bytes))
            else:
                declarator = child.child_by_field_name("declarator")
                if declarator is None:
                    declarator = child.named_children[0] if child.named_children else None
                if declarator is not None:
                    ident = _declarator_identifier(declarator)
                    if ident is not None:
                        names.append(node_text(ident, source_bytes))
        elif child.type.endswith("_declarator"):
            ident = _declarator_identifier(child)
            if ident is not None:
                names.append(node_text(ident, source_bytes))
    return names


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
