"""Python language adapter for Tree-sitter-backed semantic extraction."""

from __future__ import annotations

from pathlib import Path

from tree_sitter import Node

from codeintel.languages.base import LanguageAdapter
from codeintel.languages.python.parser import create_python_parser, parse_python
from codeintel.models import AnalysisResult, CodeUnit, SourceSpan, Symbol, SymbolKind

_MEMORY_MODULE_NAME = "<memory>"


class PythonAdapter(LanguageAdapter):
    """Extract language-neutral Symbols and CodeUnits from Python source."""

    def __init__(self) -> None:
        self._parser = create_python_parser()

    @property
    def language_id(self) -> str:
        return "python"

    @property
    def file_extensions(self) -> frozenset[str]:
        return frozenset({".py"})

    def analyze_file(
        self,
        path: Path,
        *,
        repository_root: Path | None = None,
    ) -> AnalysisResult:
        """Analyze a Python source file on disk."""
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
        """Analyze Python source text or bytes without requiring a filesystem path."""
        source_bytes = source.encode("utf-8") if isinstance(source, str) else source
        resolved_module_name = module_name if module_name is not None else _MEMORY_MODULE_NAME
        tree = parse_python(source_bytes, parser=self._parser)
        root = tree.root_node

        symbols: list[Symbol] = []
        code_units: list[CodeUnit] = []

        module_span = _span_from_node(root)
        symbols.append(
            Symbol(
                name=_final_component(resolved_module_name),
                qualified_name=resolved_module_name,
                kind=SymbolKind.MODULE,
                span=module_span,
                signature=None,
                parent_qualified_name=None,
            )
        )

        self._extract_definitions(
            root,
            source_bytes=source_bytes,
            parent_qualified_name=resolved_module_name,
            enclosing_kind=SymbolKind.MODULE,
            symbols=symbols,
            code_units=code_units,
        )

        return AnalysisResult(
            path=path,
            language_id=self.language_id,
            module_name=resolved_module_name,
            symbols=tuple(symbols),
            code_units=tuple(code_units),
            has_syntax_errors=root.has_error,
        )

    def _extract_definitions(
        self,
        node: Node,
        *,
        source_bytes: bytes,
        parent_qualified_name: str,
        enclosing_kind: SymbolKind,
        symbols: list[Symbol],
        code_units: list[CodeUnit],
    ) -> None:
        index = 0
        children = node.children
        while index < len(children):
            child = children[index]
            if child.type == "decorated_definition":
                self._handle_decorated_definition(
                    child,
                    source_bytes=source_bytes,
                    parent_qualified_name=parent_qualified_name,
                    enclosing_kind=enclosing_kind,
                    symbols=symbols,
                    code_units=code_units,
                )
            elif child.type == "class_definition":
                self._handle_class_definition(
                    child,
                    span_node=child,
                    source_bytes=source_bytes,
                    parent_qualified_name=parent_qualified_name,
                    symbols=symbols,
                    code_units=code_units,
                )
            elif child.type == "function_definition":
                self._handle_function_definition(
                    child,
                    span_node=child,
                    source_bytes=source_bytes,
                    parent_qualified_name=parent_qualified_name,
                    enclosing_kind=enclosing_kind,
                    symbols=symbols,
                    code_units=code_units,
                )
            else:
                self._extract_definitions(
                    child,
                    source_bytes=source_bytes,
                    parent_qualified_name=parent_qualified_name,
                    enclosing_kind=enclosing_kind,
                    symbols=symbols,
                    code_units=code_units,
                )
            index += 1

    def _handle_decorated_definition(
        self,
        node: Node,
        *,
        source_bytes: bytes,
        parent_qualified_name: str,
        enclosing_kind: SymbolKind,
        symbols: list[Symbol],
        code_units: list[CodeUnit],
    ) -> None:
        definition = _find_definition_child(node)
        if definition is None:
            return
        if definition.type == "class_definition":
            self._handle_class_definition(
                definition,
                span_node=node,
                source_bytes=source_bytes,
                parent_qualified_name=parent_qualified_name,
                symbols=symbols,
                code_units=code_units,
            )
        elif definition.type == "function_definition":
            self._handle_function_definition(
                definition,
                span_node=node,
                source_bytes=source_bytes,
                parent_qualified_name=parent_qualified_name,
                enclosing_kind=enclosing_kind,
                symbols=symbols,
                code_units=code_units,
            )

    def _handle_class_definition(
        self,
        definition: Node,
        *,
        span_node: Node,
        source_bytes: bytes,
        parent_qualified_name: str,
        symbols: list[Symbol],
        code_units: list[CodeUnit],
    ) -> None:
        name_node = definition.child_by_field_name("name")
        if name_node is None:
            return
        name = _node_text(name_node, source_bytes)
        qualified_name = f"{parent_qualified_name}.{name}"
        span = _span_from_node(span_node)
        symbols.append(
            Symbol(
                name=name,
                qualified_name=qualified_name,
                kind=SymbolKind.CLASS,
                span=span,
                signature=_extract_signature(definition, source_bytes),
                parent_qualified_name=parent_qualified_name,
            )
        )
        code_units.append(
            CodeUnit(
                symbol_qualified_name=qualified_name,
                kind=SymbolKind.CLASS,
                source_text=_slice_source(source_bytes, span_node),
                span=span,
            )
        )

        body = definition.child_by_field_name("body")
        if body is not None:
            self._extract_definitions(
                body,
                source_bytes=source_bytes,
                parent_qualified_name=qualified_name,
                enclosing_kind=SymbolKind.CLASS,
                symbols=symbols,
                code_units=code_units,
            )

    def _handle_function_definition(
        self,
        definition: Node,
        *,
        span_node: Node,
        source_bytes: bytes,
        parent_qualified_name: str,
        enclosing_kind: SymbolKind,
        symbols: list[Symbol],
        code_units: list[CodeUnit],
    ) -> None:
        name_node = definition.child_by_field_name("name")
        if name_node is None:
            return
        name = _node_text(name_node, source_bytes)
        kind = SymbolKind.METHOD if enclosing_kind == SymbolKind.CLASS else SymbolKind.FUNCTION
        qualified_name = f"{parent_qualified_name}.{name}"
        span = _span_from_node(span_node)
        symbols.append(
            Symbol(
                name=name,
                qualified_name=qualified_name,
                kind=kind,
                span=span,
                signature=_extract_signature(definition, source_bytes),
                parent_qualified_name=parent_qualified_name,
            )
        )
        code_units.append(
            CodeUnit(
                symbol_qualified_name=qualified_name,
                kind=kind,
                source_text=_slice_source(source_bytes, span_node),
                span=span,
            )
        )

        body = definition.child_by_field_name("body")
        if body is not None:
            self._extract_definitions(
                body,
                source_bytes=source_bytes,
                parent_qualified_name=qualified_name,
                enclosing_kind=kind,
                symbols=symbols,
                code_units=code_units,
            )


def derive_module_name(path: Path, *, repository_root: Path | None = None) -> str:
    """Derive a dotted module name from a filesystem path."""
    if repository_root is not None:
        try:
            relative = path.resolve().relative_to(repository_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Path {path} is outside repository root {repository_root}") from exc
    else:
        relative = Path(path.name)

    parts = list(relative.parts)
    if not parts:
        return path.stem

    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]

    if parts[-1] == "__init__":
        parts = parts[:-1]

    if not parts:
        return path.parent.name or path.stem

    return ".".join(parts)


def _final_component(module_name: str) -> str:
    if module_name == _MEMORY_MODULE_NAME:
        return module_name
    return module_name.rsplit(".", 1)[-1]


def _find_definition_child(node: Node) -> Node | None:
    for child in node.children:
        if child.type in {"class_definition", "function_definition"}:
            return child
    return None


def _span_from_node(node: Node) -> SourceSpan:
    start_line = node.start_point[0] + 1
    end_row, end_column = node.end_point
    # Tree-sitter end points are exclusive; an end column of 0 means the span
    # ended at the newline concluding the previous line.
    end_line = end_row if end_column == 0 and end_row > 0 else end_row + 1
    if end_line < start_line:
        end_line = start_line
    return SourceSpan(
        start_line=start_line,
        end_line=end_line,
        start_byte=node.start_byte,
        end_byte=node.end_byte,
    )


def _node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")


def _slice_source(source_bytes: bytes, node: Node) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")


def _extract_signature(definition: Node, source_bytes: bytes) -> str:
    body = definition.child_by_field_name("body")
    if body is None:
        text = _node_text(definition, source_bytes)
    else:
        text = source_bytes[definition.start_byte : body.start_byte].decode("utf-8")
    return text.rstrip().removesuffix(":").rstrip()
