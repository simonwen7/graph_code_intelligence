"""Independent unit tests for the in-memory CodeGraph."""

from pathlib import Path

import pytest

from codeintel.graph import CodeGraph, TraversalDirection
from codeintel.models import (
    Relation,
    RelationKind,
    ResolutionStatus,
    SourceSpan,
    Symbol,
    SymbolKind,
)

SPAN = SourceSpan(1, 1, 0, 1)
PATH = Path("mod.py")


def _symbol(
    qname: str, kind: SymbolKind = SymbolKind.FUNCTION, parent: str | None = "mod"
) -> Symbol:
    name = qname.rsplit(".", 1)[-1]
    return Symbol(
        name=name,
        qualified_name=qname,
        kind=kind,
        span=SPAN,
        signature=None,
        parent_qualified_name=parent,
    )


def _rel(
    kind: RelationKind,
    source: str,
    target: str | None,
    *,
    resolution: ResolutionStatus,
    text: str | None = None,
) -> Relation:
    return Relation(
        kind=kind,
        source_qualified_name=source,
        target_qualified_name=target,
        target_text=text if text is not None else (target or "unknown"),
        resolution=resolution,
        path=PATH,
        span=SPAN,
    )


def _graph() -> CodeGraph:
    symbols = (
        _symbol("mod", SymbolKind.MODULE, None),
        _symbol("mod.a"),
        _symbol("mod.b"),
        _symbol("mod.c"),
    )
    relations = (
        _rel(RelationKind.CONTAINS, "mod", "mod.a", resolution=ResolutionStatus.RESOLVED),
        _rel(RelationKind.CALLS, "mod.a", "mod.b", resolution=ResolutionStatus.RESOLVED),
        _rel(RelationKind.CALLS, "mod.b", "mod.c", resolution=ResolutionStatus.PROBABLE),
        _rel(
            RelationKind.CALLS,
            "mod.a",
            None,
            resolution=ResolutionStatus.UNRESOLVED,
            text="obj.dyn",
        ),
    )
    return CodeGraph(symbols, relations)


def test_get_symbol_and_unknown_node() -> None:
    graph = _graph()
    assert graph.get_symbol("mod.a").qualified_name == "mod.a"
    with pytest.raises(KeyError, match="Unknown symbol"):
        graph.get_symbol("missing")


def test_outgoing_includes_unresolved_but_incoming_does_not_invent_targets() -> None:
    graph = _graph()
    outgoing = graph.outgoing("mod.a")
    unresolved_text = [
        rel.target_text for rel in outgoing if rel.resolution is ResolutionStatus.UNRESOLVED
    ]
    assert unresolved_text == ["obj.dyn"]
    assert graph.has_symbol("obj.dyn") is False
    assert all(rel.target_qualified_name is not None for rel in graph.incoming("mod.b"))


def test_kind_and_resolution_filters() -> None:
    graph = _graph()
    calls = graph.outgoing(
        "mod.a", kinds=[RelationKind.CALLS], resolutions=[ResolutionStatus.RESOLVED]
    )
    assert len(calls) == 1
    assert calls[0].target_qualified_name == "mod.b"
    unresolved = graph.outgoing("mod.a", resolutions=[ResolutionStatus.UNRESOLVED])
    assert len(unresolved) == 1


def test_neighbors_ignore_unresolved_and_are_deterministic() -> None:
    graph = _graph()
    outgoing = [symbol.qualified_name for symbol in graph.neighbors("mod.a")]
    assert outgoing == ["mod.b"]
    incoming = [
        symbol.qualified_name
        for symbol in graph.neighbors("mod.b", direction=TraversalDirection.INCOMING)
    ]
    assert incoming == ["mod.a"]
    contains_children = [symbol.qualified_name for symbol in graph.neighbors("mod")]
    assert contains_children == ["mod.a"]
    parent = [
        symbol.qualified_name
        for symbol in graph.neighbors("mod.a", direction=TraversalDirection.INCOMING)
    ]
    assert parent == ["mod"]


def test_bounded_neighborhood_and_shortest_distance() -> None:
    graph = _graph()
    assert graph.bounded_neighborhood("mod.a", max_depth=0) == (("mod.a", 0),)
    neighborhood = graph.bounded_neighborhood("mod.a", max_depth=2)
    assert neighborhood[0] == ("mod.a", 0)
    distances = dict(neighborhood)
    assert distances["mod.b"] == 1
    assert distances["mod.c"] == 2
    assert graph.shortest_distance("mod.a", "mod.a") == 0
    assert graph.shortest_distance("mod.a", "mod.c") == 2
    assert graph.shortest_distance("mod.c", "mod.a") is None
    assert graph.shortest_distance("mod.a", "mod.c", max_depth=1) is None
    assert graph.shortest_distance("mod", "mod.a", kinds=[RelationKind.CONTAINS]) == 1
    assert graph.shortest_distance("mod.a", "mod.c", kinds=[RelationKind.CONTAINS]) is None
    assert (
        graph.shortest_distance("mod.a", "mod.c", resolutions=[ResolutionStatus.RESOLVED]) is None
    )


def test_cycles_do_not_loop_and_keep_minimum_distance() -> None:
    symbols = (
        _symbol("mod", SymbolKind.MODULE, None),
        _symbol("mod.a"),
        _symbol("mod.b"),
    )
    relations = (
        _rel(RelationKind.CALLS, "mod.a", "mod.b", resolution=ResolutionStatus.RESOLVED),
        _rel(RelationKind.CALLS, "mod.b", "mod.a", resolution=ResolutionStatus.RESOLVED),
    )
    graph = CodeGraph(symbols, relations)
    neighborhood = dict(graph.bounded_neighborhood("mod.a", max_depth=5))
    assert neighborhood == {"mod.a": 0, "mod.b": 1}
    assert graph.shortest_distance("mod.a", "mod.b") == 1


def test_resolved_missing_target_does_not_create_a_node() -> None:
    symbols = (
        _symbol("mod", SymbolKind.MODULE, None),
        _symbol("mod.a"),
    )
    relations = (
        _rel(
            RelationKind.CALLS,
            "mod.a",
            "ghost.fn",
            resolution=ResolutionStatus.RESOLVED,
            text="ghost.fn",
        ),
    )
    graph = CodeGraph(symbols, relations)
    assert graph.has_symbol("ghost.fn") is False
    outgoing = graph.outgoing("mod.a")
    assert outgoing[0].target_qualified_name == "ghost.fn"
    assert graph.neighbors("mod.a") == ()
    assert dict(graph.bounded_neighborhood("mod.a", max_depth=2)) == {"mod.a": 0}


def test_duplicate_symbol_identity_rejected() -> None:
    symbols = (
        _symbol("mod.a"),
        Symbol(
            name="a",
            qualified_name="mod.a",
            kind=SymbolKind.FUNCTION,
            span=SourceSpan(2, 2, 4, 8),
            signature=None,
            parent_qualified_name="mod",
        ),
    )
    with pytest.raises(ValueError, match="Duplicate qualified name"):
        CodeGraph(symbols, ())
