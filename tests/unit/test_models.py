"""Tests for language-neutral relation model invariants."""

from pathlib import Path

import pytest

from codeintel.models import Relation, RelationKind, ResolutionStatus, SourceSpan


def _relation(**overrides: object) -> Relation:
    values: dict[str, object] = {
        "kind": RelationKind.CALLS,
        "source_qualified_name": "mod.fn",
        "target_qualified_name": "mod.other",
        "target_text": "other",
        "resolution": ResolutionStatus.RESOLVED,
        "path": Path("mod.py"),
        "span": SourceSpan(1, 1, 0, 4),
    }
    values.update(overrides)
    return Relation(**values)  # type: ignore[arg-type]


def test_resolved_and_probable_require_target_qname() -> None:
    with pytest.raises(ValueError, match="require target_qualified_name"):
        _relation(resolution=ResolutionStatus.RESOLVED, target_qualified_name=None)
    with pytest.raises(ValueError, match="require target_qualified_name"):
        _relation(resolution=ResolutionStatus.PROBABLE, target_qualified_name=None)


def test_unresolved_forbids_target_qname() -> None:
    with pytest.raises(ValueError, match="must not fabricate"):
        _relation(resolution=ResolutionStatus.UNRESOLVED, target_qualified_name="mod.other")


def test_relationkind_values_are_frozen_m2_set() -> None:
    assert {kind.value for kind in RelationKind} == {
        "contains",
        "imports",
        "references",
        "calls",
        "inherits",
    }
    assert "defines" not in {kind.value for kind in RelationKind}


def test_resolutionstatus_values_are_frozen_m2_set() -> None:
    assert {status.value for status in ResolutionStatus} == {
        "resolved",
        "probable",
        "unresolved",
    }
