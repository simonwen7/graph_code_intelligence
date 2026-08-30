"""Unit tests for C++ relation extraction."""

from __future__ import annotations

from pathlib import Path

from codeintel.languages.cpp import CppAdapter, CppRelationExtractor
from codeintel.models import RelationKind, ResolutionStatus
from codeintel.repository import RepositoryAnalysis, analyze_repository

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "cpp_graph"


def _analysis() -> RepositoryAnalysis:
    return analyze_repository(FIXTURE, CppAdapter(), CppRelationExtractor())


def test_contains_derived_for_namespaces_and_classes() -> None:
    analysis = _analysis()
    contains = [rel for rel in analysis.relations if rel.kind is RelationKind.CONTAINS]
    assert any(rel.target_qualified_name == "tools::unique_helper()" for rel in contains)
    assert any(rel.target_qualified_name == "tools::Derived" for rel in contains)
    assert all(rel.resolution is ResolutionStatus.RESOLVED for rel in contains)


def test_no_cpp_references() -> None:
    analysis = _analysis()
    assert all(rel.kind is not RelationKind.REFERENCES for rel in analysis.relations)


def test_quoted_and_system_includes() -> None:
    analysis = _analysis()
    imports = [rel for rel in analysis.relations if rel.kind is RelationKind.IMPORTS]
    quoted = [rel for rel in imports if '"alpha.hpp"' in rel.target_text]
    system = [rel for rel in imports if "<vector>" in rel.target_text]
    assert len(quoted) == 1
    assert quoted[0].resolution is ResolutionStatus.RESOLVED
    assert quoted[0].target_qualified_name == "@file:alpha.hpp"
    assert len(system) == 1
    assert system[0].resolution is ResolutionStatus.UNRESOLVED
    assert system[0].target_qualified_name is None


def test_bare_unique_and_overload_calls() -> None:
    analysis = _analysis()
    calls = [rel for rel in analysis.relations if rel.kind is RelationKind.CALLS]
    unique = [rel for rel in calls if rel.target_text == "unique_helper"]
    overload = [rel for rel in calls if rel.target_text == "overload"]
    assert len(unique) == 1
    assert unique[0].resolution is ResolutionStatus.RESOLVED
    assert unique[0].target_qualified_name == "tools::unique_helper()"
    assert len(overload) == 1
    assert overload[0].resolution is ResolutionStatus.UNRESOLVED
    assert overload[0].target_qualified_name is None


def test_this_and_object_member_calls() -> None:
    analysis = _analysis()
    calls = [rel for rel in analysis.relations if rel.kind is RelationKind.CALLS]
    this_call = [rel for rel in calls if rel.target_text == "this->run"]
    obj_call = [rel for rel in calls if rel.target_text == "obj.missing"]
    ptr_call = [rel for rel in calls if rel.target_text == "ptr->missing"]
    assert len(this_call) == 1
    assert this_call[0].resolution is ResolutionStatus.PROBABLE
    assert this_call[0].target_qualified_name == "tools::Derived::run()"
    assert len(obj_call) == 1 and obj_call[0].resolution is ResolutionStatus.UNRESOLVED
    assert len(ptr_call) == 1 and ptr_call[0].resolution is ResolutionStatus.UNRESOLVED


def test_class_qualified_call_probable() -> None:
    analysis = _analysis()
    calls = [rel for rel in analysis.relations if rel.kind is RelationKind.CALLS]
    alpha = [rel for rel in calls if rel.target_text == "Alpha::act"]
    assert len(alpha) == 1
    assert alpha[0].resolution is ResolutionStatus.PROBABLE
    assert alpha[0].target_qualified_name == "Alpha::act()"


def test_inherits_multiple_and_unknown() -> None:
    analysis = _analysis()
    inherits = [rel for rel in analysis.relations if rel.kind is RelationKind.INHERITS]
    multi = [rel for rel in inherits if rel.source_qualified_name == "tools::Multi"]
    assert len(multi) == 2
    assert {rel.target_text for rel in multi} == {"Base", "Derived"}
    assert all(rel.resolution is ResolutionStatus.RESOLVED for rel in multi)
    unknown = [rel for rel in inherits if rel.source_qualified_name == "tools::UnknownChild"]
    assert len(unknown) == 1
    assert unknown[0].resolution is ResolutionStatus.UNRESOLVED
    assert unknown[0].target_qualified_name is None


def test_anonymous_namespace_internal_call_resolves(tmp_path: Path) -> None:
    (tmp_path / "anon.cpp").write_text(
        "namespace {\nvoid helper() {}\nvoid caller() { helper(); }\n}\n",
        encoding="utf-8",
    )
    analysis = analyze_repository(tmp_path, CppAdapter(), CppRelationExtractor())
    calls = [rel for rel in analysis.relations if rel.kind is RelationKind.CALLS]
    assert len(calls) == 1
    assert calls[0].resolution is ResolutionStatus.RESOLVED
    assert calls[0].target_qualified_name == "@filelocal:anon.cpp::<anonymous>::helper()"
    assert calls[0].source_qualified_name == "@filelocal:anon.cpp::<anonymous>::caller()"
