"""Tests for repository-level analysis orchestration."""

from pathlib import Path

import pytest

from codeintel.languages.python import PythonAdapter, PythonRelationExtractor
from codeintel.models import RelationKind, ResolutionStatus
from codeintel.repository import analyze_repository

GRAPH_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "python_graph"


def test_analyze_repository_indexes_symbols_before_relations() -> None:
    analysis = analyze_repository(GRAPH_ROOT, PythonAdapter(), PythonRelationExtractor())

    assert analysis.root == GRAPH_ROOT
    assert len(analysis.files) >= 6
    names = {symbol.qualified_name for symbol in analysis.symbols}
    assert "helpers.helper" in names
    assert "service.Service" in names
    contains = [rel for rel in analysis.relations if rel.kind is RelationKind.CONTAINS]
    assert any(rel.target_qualified_name == "helpers.helper" for rel in contains)
    assert analysis.graph.has_symbol("helpers.helper")
    consumer = GRAPH_ROOT / "consumer.py"
    helpers = GRAPH_ROOT / "helpers.py"
    assert consumer.name < helpers.name
    imported = [
        rel
        for rel in analysis.relations
        if rel.kind is RelationKind.IMPORTS
        and rel.source_qualified_name == "consumer"
        and rel.target_qualified_name == "helpers.helper"
    ]
    assert imported
    calls = [
        rel
        for rel in analysis.relations
        if rel.kind is RelationKind.CALLS
        and rel.source_qualified_name == "consumer.use_calls"
        and rel.target_qualified_name == "helpers.helper"
    ]
    assert calls


def test_empty_directory_returns_empty_analysis(tmp_path: Path) -> None:
    analysis = analyze_repository(tmp_path, PythonAdapter(), PythonRelationExtractor())

    assert analysis.files == ()
    assert analysis.symbols == ()
    assert analysis.relations == ()
    assert analysis.graph.symbols == ()


def test_duplicate_qualified_name_is_detected(tmp_path: Path) -> None:
    (tmp_path / "foo.py").write_text("x = 1\n", encoding="utf-8")
    package = tmp_path / "foo"
    package.mkdir()
    (package / "__init__.py").write_text("x = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate qualified name 'foo'"):
        analyze_repository(tmp_path, PythonAdapter(), PythonRelationExtractor())


def test_contains_has_no_duplicates() -> None:
    analysis = analyze_repository(GRAPH_ROOT, PythonAdapter(), PythonRelationExtractor())
    pairs = [
        (rel.source_qualified_name, rel.target_qualified_name)
        for rel in analysis.relations
        if rel.kind is RelationKind.CONTAINS
    ]
    assert len(pairs) == len(set(pairs))
    contains = [rel for rel in analysis.relations if rel.kind is RelationKind.CONTAINS]
    assert all(rel.resolution is ResolutionStatus.RESOLVED for rel in contains)
