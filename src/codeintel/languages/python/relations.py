"""Python-specific static relationship extraction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from tree_sitter import Node

from codeintel.languages.python.parser import parse_python
from codeintel.models import (
    AnalysisResult,
    Relation,
    RelationKind,
    ResolutionStatus,
    SourceSpan,
    Symbol,
    SymbolKind,
)

_DEFINITION_TYPES = frozenset({"function_definition", "class_definition"})


class _BindingKind(StrEnum):
    SYMBOL = "symbol"
    MODULE = "module"
    LOCAL = "local"


@dataclass
class _NameBinding:
    kind: _BindingKind
    qualified_name: str | None = None


@dataclass
class _LexicalScope:
    enclosing_qname: str
    enclosing_kind: SymbolKind
    parent: _LexicalScope | None
    bindings: dict[str, _NameBinding] = field(default_factory=dict)

    def class_qname(self) -> str | None:
        current: _LexicalScope | None = self
        while current is not None:
            if current.enclosing_kind == SymbolKind.CLASS:
                return current.enclosing_qname
            current = current.parent
        return None


class PythonRelationExtractor:
    """Extract IMPORTS, CALLS, REFERENCES, and INHERITS from Python source."""

    def extract_relations(
        self,
        path: Path,
        *,
        repository_root: Path,
        analysis: AnalysisResult,
        symbols_by_qualified_name: Mapping[str, Symbol],
    ) -> tuple[Relation, ...]:
        del repository_root
        source_bytes = path.read_bytes()
        tree = parse_python(source_bytes)
        extractor = _FileExtractor(
            path=path,
            source_bytes=source_bytes,
            analysis=analysis,
            symbols=symbols_by_qualified_name,
        )
        extractor.collect_and_visit(tree.root_node)
        return tuple(extractor.relations)


class _FileExtractor:
    def __init__(
        self,
        *,
        path: Path,
        source_bytes: bytes,
        analysis: AnalysisResult,
        symbols: Mapping[str, Symbol],
    ) -> None:
        self.path = path
        self.source_bytes = source_bytes
        self.analysis = analysis
        self.symbols = symbols
        self.relations: list[Relation] = []
        self.module_qname = analysis.module_name
        self.is_package_module = path.name == "__init__.py"

    def collect_and_visit(self, root: Node) -> None:
        module_scope = _LexicalScope(self.module_qname, SymbolKind.MODULE, None)
        self._bind_child_symbols(module_scope)
        self._collect_bindings(root, module_scope)
        self._visit(root, module_scope)

    def _bind_child_symbols(self, scope: _LexicalScope) -> None:
        prefix = f"{scope.enclosing_qname}."
        for symbol in self.symbols.values():
            if symbol.parent_qualified_name != scope.enclosing_qname:
                continue
            name = (
                symbol.qualified_name[len(prefix) :]
                if symbol.qualified_name.startswith(prefix)
                else symbol.name
            )
            if "." in name:
                continue
            if symbol.kind == SymbolKind.MODULE:
                kind = _BindingKind.MODULE
            else:
                kind = _BindingKind.SYMBOL
            scope.bindings[symbol.name] = _NameBinding(kind, symbol.qualified_name)

    def _collect_bindings(self, node: Node, scope: _LexicalScope) -> None:
        node_type = node.type
        if node_type in _DEFINITION_TYPES or node_type == "decorated_definition":
            return
        if node_type == "import_statement":
            self._handle_import_statement(node, scope)
            return
        if node_type == "import_from_statement":
            self._handle_import_from_statement(node, scope)
            return
        if node_type == "assignment":
            for name in _assignment_target_names(node, self.source_bytes):
                scope.bindings[name] = _NameBinding(_BindingKind.LOCAL)
            return
        if node_type == "for_statement":
            left = node.child_by_field_name("left")
            if left is not None:
                for name in _identifier_names(left, self.source_bytes):
                    scope.bindings[name] = _NameBinding(_BindingKind.LOCAL)
            body = node.child_by_field_name("body")
            if body is not None:
                self._collect_bindings(body, scope)
            return
        for child in node.children:
            self._collect_bindings(child, scope)

    def _visit(self, node: Node, scope: _LexicalScope) -> None:
        node_type = node.type
        if node_type == "decorated_definition":
            definition = _find_definition_child(node)
            if definition is not None:
                self._visit_definition(definition, scope)
            return
        if node_type in _DEFINITION_TYPES:
            self._visit_definition(node, scope)
            return
        if node_type in {"import_statement", "import_from_statement"}:
            return
        if node_type == "call":
            self._handle_call(node, scope)
            return
        if node_type == "assignment":
            right = node.child_by_field_name("right")
            if right is not None:
                self._visit(right, scope)
            return
        if node_type == "identifier":
            self._handle_reference_identifier(node, scope)
            return
        if node_type == "attribute":
            self._handle_reference_attribute(node, scope)
            return
        for child in node.children:
            self._visit(child, scope)

    def _visit_definition(self, node: Node, parent_scope: _LexicalScope) -> None:
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = _node_text(name_node, self.source_bytes)
        qname = f"{parent_scope.enclosing_qname}.{name}"
        symbol = self.symbols.get(qname)
        if symbol is None:
            return
        if node.type == "class_definition":
            self._handle_bases(node, parent_scope, symbol.qualified_name)
        child_scope = _LexicalScope(symbol.qualified_name, symbol.kind, parent_scope)
        self._bind_child_symbols(child_scope)
        if node.type == "function_definition":
            self._bind_parameters(node, child_scope)
        body = node.child_by_field_name("body")
        if body is not None:
            self._collect_bindings(body, child_scope)
            self._visit(body, child_scope)

    def _bind_parameters(self, function: Node, scope: _LexicalScope) -> None:
        parameters = function.child_by_field_name("parameters")
        if parameters is None:
            return
        for name in _parameter_names(parameters, self.source_bytes):
            existing = scope.bindings.get(name)
            if existing is not None and existing.kind != _BindingKind.LOCAL:
                continue
            scope.bindings[name] = _NameBinding(_BindingKind.LOCAL)

    def _lookup(self, scope: _LexicalScope, name: str) -> _NameBinding | None:
        current: _LexicalScope | None = scope
        start_kind = scope.enclosing_kind
        while current is not None:
            skip_class = (
                current.enclosing_kind == SymbolKind.CLASS
                and current is not scope
                and start_kind in {SymbolKind.FUNCTION, SymbolKind.METHOD}
            )
            if not skip_class and name in current.bindings:
                return current.bindings[name]
            current = current.parent
        return None

    def _handle_import_statement(self, node: Node, scope: _LexicalScope) -> None:
        for child in node.children:
            if child.type == "dotted_name":
                dotted = _node_text(child, self.source_bytes)
                self._emit_module_import(
                    scope,
                    dotted,
                    bind_name=dotted.split(".", 1)[0],
                    span_node=child,
                    aliased=False,
                )
            elif child.type == "aliased_import":
                dotted_node = _first_child_of_type(child, "dotted_name")
                alias_node = _last_child_of_type(child, "identifier")
                if dotted_node is None or alias_node is None:
                    continue
                dotted = _node_text(dotted_node, self.source_bytes)
                alias = _node_text(alias_node, self.source_bytes)
                self._emit_module_import(
                    scope, dotted, bind_name=alias, span_node=child, aliased=True
                )

    def _emit_module_import(
        self,
        scope: _LexicalScope,
        dotted: str,
        *,
        bind_name: str,
        span_node: Node,
        aliased: bool,
    ) -> None:
        target = dotted if dotted in self.symbols else None
        self._emit_import_relation(dotted, target, span_node)
        binding_target = target
        if not aliased and "." in dotted:
            top = dotted.split(".", 1)[0]
            bind_name = top
            binding_target = top if top in self.symbols else None
        if binding_target is not None:
            kind = (
                _BindingKind.MODULE
                if self.symbols[binding_target].kind == SymbolKind.MODULE
                else _BindingKind.SYMBOL
            )
            scope.bindings[bind_name] = _NameBinding(kind, binding_target)
        else:
            scope.bindings[bind_name] = _NameBinding(_BindingKind.LOCAL)

    def _handle_import_from_statement(self, node: Node, scope: _LexicalScope) -> None:
        module_text, dots, remainder = _from_module_parts(node, self.source_bytes)
        if any(child.type == "wildcard_import" for child in node.children):
            syntactic = _join_import_text(module_text, "*") if module_text else "*"
            self._emit_import_relation(syntactic, None, node)
            return
        resolved_module = self._resolve_from_module(dots, remainder)
        for imported_name, alias, item_node in _from_import_items(node, self.source_bytes):
            syntactic = _join_import_text(module_text, imported_name)
            target: str | None = None
            if resolved_module is not None:
                candidate = f"{resolved_module}.{imported_name}"
                if candidate in self.symbols:
                    target = candidate
            self._emit_import_relation(syntactic, target, item_node)
            bind_name = alias if alias is not None else imported_name
            if target is not None:
                kind = (
                    _BindingKind.MODULE
                    if self.symbols[target].kind == SymbolKind.MODULE
                    else _BindingKind.SYMBOL
                )
                scope.bindings[bind_name] = _NameBinding(kind, target)
            else:
                scope.bindings[bind_name] = _NameBinding(_BindingKind.LOCAL)

    def _resolve_from_module(self, dots: int, remainder: str | None) -> str | None:
        if dots == 0:
            return remainder
        parts = self.module_qname.split(".") if self.module_qname else []
        package = parts if self.is_package_module else parts[:-1]
        up = dots - 1
        if up > len(package):
            return None
        base = package[: len(package) - up]
        if remainder:
            return ".".join((*base, remainder)) if base else remainder
        return ".".join(base) if base else None

    def _emit_import_relation(self, target_text: str, target: str | None, span_node: Node) -> None:
        resolution = (
            ResolutionStatus.RESOLVED if target is not None else ResolutionStatus.UNRESOLVED
        )
        self.relations.append(
            Relation(
                kind=RelationKind.IMPORTS,
                source_qualified_name=self.module_qname,
                target_qualified_name=target,
                target_text=target_text,
                resolution=resolution,
                path=self.path,
                span=_span_from_node(span_node),
            )
        )

    def _handle_bases(
        self, class_node: Node, lookup_scope: _LexicalScope, class_qname: str
    ) -> None:
        bases = class_node.child_by_field_name("superclasses")
        if bases is None:
            return
        for child in bases.children:
            if child.type in {"(", ")", ",", "comment"}:
                continue
            target_text = _node_text(child, self.source_bytes)
            if child.type in {"identifier", "attribute"}:
                target, resolution = self._resolve_name_expression(child, lookup_scope)
                if target is not None and self.symbols[target].kind != SymbolKind.CLASS:
                    target = None
                    resolution = ResolutionStatus.UNRESOLVED
            else:
                target = None
                resolution = ResolutionStatus.UNRESOLVED
            if target is None:
                resolution = ResolutionStatus.UNRESOLVED
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

    def _handle_call(self, node: Node, scope: _LexicalScope) -> None:
        function = node.child_by_field_name("function")
        if function is None:
            return
        target_text = _node_text(function, self.source_bytes)
        target, resolution = self._resolve_call_callee(function, scope)
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

    def _resolve_call_callee(
        self, function: Node, scope: _LexicalScope
    ) -> tuple[str | None, ResolutionStatus]:
        if function.type == "identifier":
            name = _node_text(function, self.source_bytes)
            binding = self._lookup(scope, name)
            if binding is None or binding.kind == _BindingKind.LOCAL:
                return None, ResolutionStatus.UNRESOLVED
            if binding.qualified_name is not None and binding.qualified_name in self.symbols:
                return binding.qualified_name, ResolutionStatus.RESOLVED
            return None, ResolutionStatus.UNRESOLVED
        if function.type == "attribute":
            return self._resolve_attribute_call(function, scope)
        return None, ResolutionStatus.UNRESOLVED

    def _resolve_attribute_call(
        self, node: Node, scope: _LexicalScope
    ) -> tuple[str | None, ResolutionStatus]:
        obj = node.child_by_field_name("object")
        attr = node.child_by_field_name("attribute")
        if obj is None or attr is None:
            return None, ResolutionStatus.UNRESOLVED
        attr_name = _node_text(attr, self.source_bytes)
        if obj.type != "identifier":
            return None, ResolutionStatus.UNRESOLVED
        obj_name = _node_text(obj, self.source_bytes)
        class_qname = scope.class_qname()
        if obj_name in {"self", "cls"} and class_qname is not None:
            candidate = f"{class_qname}.{attr_name}"
            if candidate in self.symbols:
                return candidate, ResolutionStatus.PROBABLE
            return None, ResolutionStatus.UNRESOLVED
        binding = self._lookup(scope, obj_name)
        if binding is None or binding.kind == _BindingKind.LOCAL or binding.qualified_name is None:
            return None, ResolutionStatus.UNRESOLVED
        obj_symbol = self.symbols.get(binding.qualified_name)
        if obj_symbol is None:
            return None, ResolutionStatus.UNRESOLVED
        candidate = f"{binding.qualified_name}.{attr_name}"
        if candidate not in self.symbols:
            return None, ResolutionStatus.UNRESOLVED
        if obj_symbol.kind == SymbolKind.MODULE:
            return candidate, ResolutionStatus.RESOLVED
        if obj_symbol.kind == SymbolKind.CLASS:
            return candidate, ResolutionStatus.PROBABLE
        return None, ResolutionStatus.UNRESOLVED

    def _handle_reference_identifier(self, node: Node, scope: _LexicalScope) -> None:
        name = _node_text(node, self.source_bytes)
        binding = self._lookup(scope, name)
        if binding is None or binding.kind == _BindingKind.LOCAL:
            return
        if binding.qualified_name is None or binding.qualified_name not in self.symbols:
            return
        self.relations.append(
            Relation(
                kind=RelationKind.REFERENCES,
                source_qualified_name=scope.enclosing_qname,
                target_qualified_name=binding.qualified_name,
                target_text=name,
                resolution=ResolutionStatus.RESOLVED,
                path=self.path,
                span=_span_from_node(node),
            )
        )

    def _handle_reference_attribute(self, node: Node, scope: _LexicalScope) -> None:
        target, resolution = self._resolve_attribute_use(node, scope)
        if target is None or resolution is ResolutionStatus.UNRESOLVED:
            return
        self.relations.append(
            Relation(
                kind=RelationKind.REFERENCES,
                source_qualified_name=scope.enclosing_qname,
                target_qualified_name=target,
                target_text=_node_text(node, self.source_bytes),
                resolution=resolution,
                path=self.path,
                span=_span_from_node(node),
            )
        )

    def _resolve_attribute_use(
        self, node: Node, scope: _LexicalScope
    ) -> tuple[str | None, ResolutionStatus]:
        obj = node.child_by_field_name("object")
        attr = node.child_by_field_name("attribute")
        if obj is None or attr is None or obj.type != "identifier":
            return None, ResolutionStatus.UNRESOLVED
        obj_name = _node_text(obj, self.source_bytes)
        attr_name = _node_text(attr, self.source_bytes)
        class_qname = scope.class_qname()
        if obj_name in {"self", "cls"} and class_qname is not None:
            candidate = f"{class_qname}.{attr_name}"
            if candidate in self.symbols:
                return candidate, ResolutionStatus.PROBABLE
            return None, ResolutionStatus.UNRESOLVED
        binding = self._lookup(scope, obj_name)
        if binding is None or binding.kind == _BindingKind.LOCAL or binding.qualified_name is None:
            return None, ResolutionStatus.UNRESOLVED
        candidate = f"{binding.qualified_name}.{attr_name}"
        if candidate not in self.symbols:
            return None, ResolutionStatus.UNRESOLVED
        obj_symbol = self.symbols[binding.qualified_name]
        if obj_symbol.kind == SymbolKind.CLASS:
            return candidate, ResolutionStatus.PROBABLE
        return candidate, ResolutionStatus.RESOLVED

    def _resolve_name_expression(
        self, node: Node, scope: _LexicalScope
    ) -> tuple[str | None, ResolutionStatus]:
        if node.type == "identifier":
            binding = self._lookup(scope, _node_text(node, self.source_bytes))
            if (
                binding is None
                or binding.kind == _BindingKind.LOCAL
                or binding.qualified_name is None
            ):
                return None, ResolutionStatus.UNRESOLVED
            return binding.qualified_name, ResolutionStatus.RESOLVED
        if node.type == "attribute":
            return self._resolve_attribute_use(node, scope)
        return None, ResolutionStatus.UNRESOLVED


def _find_definition_child(node: Node) -> Node | None:
    for child in node.children:
        if child.type in _DEFINITION_TYPES:
            return child
    return None


def _first_child_of_type(node: Node, node_type: str) -> Node | None:
    for child in node.children:
        if child.type == node_type:
            return child
    return None


def _last_child_of_type(node: Node, node_type: str) -> Node | None:
    for child in reversed(node.children):
        if child.type == node_type:
            return child
    return None


def _join_import_text(module_text: str, imported_name: str) -> str:
    if not module_text:
        return imported_name
    if module_text.endswith("."):
        return f"{module_text}{imported_name}"
    return f"{module_text}.{imported_name}"


def _from_module_parts(node: Node, source_bytes: bytes) -> tuple[str, int, str | None]:
    relative = _first_child_of_type(node, "relative_import")
    if relative is not None:
        prefix = _first_child_of_type(relative, "import_prefix")
        dots = 0
        if prefix is not None:
            dots = sum(1 for child in prefix.children if child.type == ".")
        dotted = _first_child_of_type(relative, "dotted_name")
        remainder = _node_text(dotted, source_bytes) if dotted is not None else None
        return _node_text(relative, source_bytes), dots, remainder
    dotted = _first_child_of_type(node, "dotted_name")
    if dotted is not None:
        remainder = _node_text(dotted, source_bytes)
        return remainder, 0, remainder
    return "", 0, None


def _from_import_items(node: Node, source_bytes: bytes) -> list[tuple[str, str | None, Node]]:
    items: list[tuple[str, str | None, Node]] = []
    skip_module = _first_child_of_type(node, "relative_import") is None
    skipped = not skip_module
    for child in node.children:
        if child.type == "relative_import":
            skipped = True
            continue
        if child.type == "dotted_name" and not skipped:
            skipped = True
            continue
        if child.type == "dotted_name":
            items.append((_node_text(child, source_bytes), None, child))
        elif child.type == "aliased_import":
            dotted = _first_child_of_type(child, "dotted_name")
            alias = _last_child_of_type(child, "identifier")
            if dotted is None:
                continue
            alias_name = _node_text(alias, source_bytes) if alias is not None else None
            items.append((_node_text(dotted, source_bytes), alias_name, child))
    return items


def _assignment_target_names(node: Node, source_bytes: bytes) -> list[str]:
    left = node.child_by_field_name("left")
    if left is None:
        return []
    return _identifier_names(left, source_bytes)


def _identifier_names(node: Node, source_bytes: bytes) -> list[str]:
    if node.type == "identifier":
        return [_node_text(node, source_bytes)]
    names: list[str] = []
    for child in node.children:
        if child.type == "identifier":
            names.append(_node_text(child, source_bytes))
        elif child.type in {"pattern_list", "tuple_pattern", "list_pattern", "tuple", "list"}:
            names.extend(_identifier_names(child, source_bytes))
    return names


def _parameter_names(parameters: Node, source_bytes: bytes) -> list[str]:
    names: list[str] = []
    for child in parameters.children:
        if child.type == "identifier":
            names.append(_node_text(child, source_bytes))
        elif child.type in {
            "typed_parameter",
            "default_parameter",
            "typed_default_parameter",
            "list_splat_pattern",
            "dictionary_splat_pattern",
        }:
            ident = child.child_by_field_name("name")
            if ident is None:
                ident = _first_child_of_type(child, "identifier")
            if ident is not None:
                names.append(_node_text(ident, source_bytes))
    return names


def _node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")


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
