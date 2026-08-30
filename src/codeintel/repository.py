"""Repository-level orchestration for symbols, relations, and the code graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from codeintel.discovery import discover_source_files
from codeintel.graph import CodeGraph
from codeintel.languages.base import LanguageAdapter
from codeintel.models import (
    AnalysisResult,
    Relation,
    RelationKind,
    ResolutionStatus,
    Symbol,
)


class RelationExtractor(Protocol):
    """Language-specific extraction of non-structural relations."""

    def extract_relations(
        self,
        path: Path,
        *,
        repository_root: Path,
        analysis: AnalysisResult,
        symbols_by_qualified_name: Mapping[str, Symbol],
    ) -> tuple[Relation, ...]:
        """Return language-specific relations for one already-analyzed file."""


@dataclass(frozen=True, slots=True)
class RepositoryAnalysis:
    """Complete in-memory analysis of a source repository."""

    root: Path
    files: tuple[AnalysisResult, ...]
    symbols: tuple[Symbol, ...]
    relations: tuple[Relation, ...]
    graph: CodeGraph


def analyze_repository(
    root: Path,
    adapter: LanguageAdapter,
    relation_extractor: RelationExtractor,
) -> RepositoryAnalysis:
    """Discover, analyze, and relate all supported files under ``root``."""
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Repository analysis requires a directory: {root}")

    discovered = discover_source_files(root, adapter)
    files = tuple(adapter.analyze_file(path, repository_root=root) for path in discovered)

    symbols_by_qualified_name, symbol_paths = _build_symbol_index(files)
    symbols = tuple(symbols_by_qualified_name[name] for name in sorted(symbols_by_qualified_name))

    contains = _derive_contains_relations(files, symbol_paths)
    language_relations: list[Relation] = []
    for analysis in files:
        if analysis.path is None:
            continue
        language_relations.extend(
            relation_extractor.extract_relations(
                analysis.path,
                repository_root=root,
                analysis=analysis,
                symbols_by_qualified_name=symbols_by_qualified_name,
            )
        )

    relations = _dedupe_relations((*contains, *language_relations))
    graph = CodeGraph(symbols, relations)
    return RepositoryAnalysis(
        root=root,
        files=files,
        symbols=symbols,
        relations=relations,
        graph=graph,
    )


def _build_symbol_index(
    files: tuple[AnalysisResult, ...],
) -> tuple[dict[str, Symbol], dict[str, Path]]:
    index: dict[str, Symbol] = {}
    paths: dict[str, Path] = {}
    for analysis in files:
        analysis_path = analysis.path
        for symbol in analysis.symbols:
            previous = index.get(symbol.qualified_name)
            if previous is not None:
                previous_path = paths.get(symbol.qualified_name)
                raise ValueError(
                    "Duplicate qualified name "
                    f"{symbol.qualified_name!r} in {previous_path} and {analysis_path}"
                )
            index[symbol.qualified_name] = symbol
            if analysis_path is not None:
                paths[symbol.qualified_name] = analysis_path
    return index, paths


def _derive_contains_relations(
    files: tuple[AnalysisResult, ...],
    symbol_paths: dict[str, Path],
) -> tuple[Relation, ...]:
    relations: list[Relation] = []
    seen: set[tuple[str, str]] = set()
    for analysis in files:
        for symbol in analysis.symbols:
            parent = symbol.parent_qualified_name
            if parent is None:
                continue
            pair = (parent, symbol.qualified_name)
            if pair in seen:
                continue
            seen.add(pair)
            path = (
                analysis.path
                if analysis.path is not None
                else symbol_paths.get(symbol.qualified_name)
            )
            if path is None:
                continue
            relations.append(
                Relation(
                    kind=RelationKind.CONTAINS,
                    source_qualified_name=parent,
                    target_qualified_name=symbol.qualified_name,
                    target_text=symbol.qualified_name,
                    resolution=ResolutionStatus.RESOLVED,
                    path=path,
                    span=symbol.span,
                )
            )
    return tuple(relations)


def _dedupe_relations(relations: tuple[Relation, ...]) -> tuple[Relation, ...]:
    unique: dict[tuple[object, ...], Relation] = {}
    for relation in relations:
        key = (
            relation.kind,
            relation.source_qualified_name,
            relation.target_qualified_name,
            relation.target_text,
            relation.resolution,
            str(relation.path),
            relation.span.start_byte if relation.span is not None else None,
            relation.span.end_byte if relation.span is not None else None,
        )
        unique[key] = relation
    return tuple(unique.values())
