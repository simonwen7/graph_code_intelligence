"""Unit tests for graph-augmented hybrid retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers.fake_embeddings import FakeEmbeddingProvider

from codeintel.dense import (
    DenseIndexMismatchError,
    build_dense_index,
    format_dense_document,
    load_dense_documents,
)
from codeintel.graph import CodeGraph
from codeintel.graph_retrieval import (
    FINAL_GRAPH_RRF_K,
    GRAPH_MAX_DEPTH,
    GRAPH_RELATION_KINDS,
    GRAPH_RRF_K,
    GRAPH_SEED_COUNT,
    _collect_graph_supports,
    _direct_neighbors,
    _GraphCandidateSupport,
    _structural_ranked_results,
    graph_base_pool_limit,
    search_graph_augmented,
)
from codeintel.hybrid import fuse_rrf, search_hybrid
from codeintel.languages.python import PythonAdapter, PythonRelationExtractor
from codeintel.models import (
    Relation,
    RelationKind,
    ResolutionStatus,
    SearchResult,
    SourceSpan,
    Symbol,
    SymbolKind,
)
from codeintel.repository import analyze_repository
from codeintel.storage import IndexDatabase, PersistedCodeUnitView

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "python_graph_search"
SPAN = SourceSpan(1, 1, 0, 1)


def _sym(qname: str, kind: SymbolKind = SymbolKind.FUNCTION) -> Symbol:
    name = qname.rsplit(".", 1)[-1]
    return Symbol(name, qname, kind, SPAN, None, None)


def _rel(
    kind: RelationKind,
    source: str,
    target: str | None,
    *,
    resolution: ResolutionStatus = ResolutionStatus.RESOLVED,
    path: str = "a.py",
) -> Relation:
    return Relation(
        kind=kind,
        source_qualified_name=source,
        target_qualified_name=target,
        target_text=target or "missing",
        resolution=resolution,
        path=Path(path),
        span=SPAN,
    )


def _unit(qname: str, path: str = "a.py") -> PersistedCodeUnitView:
    return PersistedCodeUnitView(
        symbol_qualified_name=qname,
        kind=SymbolKind.FUNCTION,
        path=Path(path),
        span=SPAN,
        signature=None,
        source_text="pass",
    )


def _hit(qname: str, score: float, path: str = "a.py") -> SearchResult:
    return SearchResult(
        symbol_qualified_name=qname,
        kind=SymbolKind.FUNCTION,
        path=Path(path),
        span=SPAN,
        signature=None,
        source_text="pass",
        score=score,
    )


def test_constants_and_base_pool() -> None:
    assert GRAPH_SEED_COUNT == 10
    assert GRAPH_MAX_DEPTH == 1
    assert GRAPH_RRF_K == 60
    assert FINAL_GRAPH_RRF_K == 60
    assert RelationKind.IMPORTS not in GRAPH_RELATION_KINDS
    assert graph_base_pool_limit(1) == 10
    assert graph_base_pool_limit(3) == 10
    assert graph_base_pool_limit(10) == 10
    assert graph_base_pool_limit(11) == 11
    assert graph_base_pool_limit(20) == 20
    with pytest.raises(ValueError):
        graph_base_pool_limit(0)


def test_direct_neighbors_policy() -> None:
    symbols = [
        _sym("seed"),
        _sym("via_calls"),
        _sym("via_refs"),
        _sym("via_inherits", SymbolKind.CLASS),
        _sym("via_contains", SymbolKind.METHOD),
        _sym("via_imports"),
        _sym("via_probable"),
        _sym("mod", SymbolKind.MODULE),
    ]
    relations = [
        _rel(RelationKind.CALLS, "seed", "via_calls"),
        _rel(RelationKind.REFERENCES, "seed", "via_refs"),
        _rel(RelationKind.INHERITS, "seed", "via_inherits"),
        _rel(RelationKind.CONTAINS, "seed", "via_contains"),
        _rel(RelationKind.IMPORTS, "seed", "via_imports"),
        _rel(
            RelationKind.CALLS,
            "seed",
            "via_probable",
            resolution=ResolutionStatus.PROBABLE,
        ),
        _rel(
            RelationKind.CALLS,
            "seed",
            None,
            resolution=ResolutionStatus.UNRESOLVED,
        ),
        _rel(RelationKind.CALLS, "via_calls", "seed"),  # incoming both-direction
    ]
    graph = CodeGraph(symbols, relations)
    neighbors = _direct_neighbors(graph, "seed")
    assert neighbors == {"via_calls", "via_refs", "via_inherits", "via_contains"}


@pytest.mark.parametrize(
    ("kind", "source", "target", "source_kind", "target_kind"),
    [
        (RelationKind.CALLS, "caller", "callee", SymbolKind.FUNCTION, SymbolKind.FUNCTION),
        (
            RelationKind.REFERENCES,
            "referrer",
            "referenced",
            SymbolKind.FUNCTION,
            SymbolKind.FUNCTION,
        ),
        (RelationKind.INHERITS, "child", "base", SymbolKind.CLASS, SymbolKind.CLASS),
        (
            RelationKind.CONTAINS,
            "parent",
            "parent.method",
            SymbolKind.CLASS,
            SymbolKind.METHOD,
        ),
    ],
)
def test_both_directions_per_included_kind(
    kind: RelationKind,
    source: str,
    target: str,
    source_kind: SymbolKind,
    target_kind: SymbolKind,
) -> None:
    graph = CodeGraph(
        [_sym(source, source_kind), _sym(target, target_kind)],
        [_rel(kind, source, target)],
    )
    assert target in _direct_neighbors(graph, source)
    assert source in _direct_neighbors(graph, target)


def test_imports_probable_unresolved_do_not_create_candidates() -> None:
    symbols = [_sym("seed"), _sym("via_imports"), _sym("via_probable")]
    relations = [
        _rel(RelationKind.IMPORTS, "seed", "via_imports"),
        _rel(
            RelationKind.CALLS,
            "seed",
            "via_probable",
            resolution=ResolutionStatus.PROBABLE,
        ),
        _rel(
            RelationKind.CALLS,
            "seed",
            None,
            resolution=ResolutionStatus.UNRESOLVED,
        ),
    ]
    graph = CodeGraph(symbols, relations)
    returnable = {
        "via_imports": _unit("via_imports"),
        "via_probable": _unit("via_probable"),
    }
    supports = _collect_graph_supports(
        graph,
        (_hit("seed", 1.0),),
        returnable=returnable,
        kind=None,
        path_prefix=None,
    )
    assert supports == {}


def test_depth_one_only() -> None:
    symbols = [_sym("a"), _sym("b"), _sym("c")]
    relations = [
        _rel(RelationKind.CALLS, "a", "b"),
        _rel(RelationKind.CALLS, "b", "c"),
    ]
    graph = CodeGraph(symbols, relations)
    assert _direct_neighbors(graph, "a") == {"b"}
    assert "c" not in _direct_neighbors(graph, "a")


def test_class_contains_no_sibling_depth2() -> None:
    symbols = [
        _sym("Cls", SymbolKind.CLASS),
        _sym("Cls.m1", SymbolKind.METHOD),
        _sym("Cls.m2", SymbolKind.METHOD),
    ]
    relations = [
        _rel(RelationKind.CONTAINS, "Cls", "Cls.m1"),
        _rel(RelationKind.CONTAINS, "Cls", "Cls.m2"),
    ]
    graph = CodeGraph(symbols, relations)
    returnable = {
        "Cls": PersistedCodeUnitView(
            "Cls",
            SymbolKind.CLASS,
            Path("c.py"),
            SPAN,
            None,
            "class Cls: ...",
        ),
        "Cls.m1": PersistedCodeUnitView(
            "Cls.m1",
            SymbolKind.METHOD,
            Path("c.py"),
            SPAN,
            None,
            "def m1(self): ...",
        ),
        "Cls.m2": PersistedCodeUnitView(
            "Cls.m2",
            SymbolKind.METHOD,
            Path("c.py"),
            SPAN,
            None,
            "def m2(self): ...",
        ),
    }
    supports = _collect_graph_supports(
        graph,
        (_hit("Cls.m1", 1.0),),
        returnable=returnable,
        kind=None,
        path_prefix=None,
    )
    assert "Cls" in supports
    assert "Cls.m2" not in supports


def test_unique_seed_and_multi_seed_support() -> None:
    support = _GraphCandidateSupport("x")
    support.supporting_seed_ranks.add(1)
    support.supporting_seed_ranks.add(1)  # duplicate seed ignored by set
    support.supporting_seed_ranks.add(3)
    assert support.graph_support() == pytest.approx(1 / 61 + 1 / 63)


def test_multiple_edge_kinds_dedup_one_seed() -> None:
    symbols = [_sym("seed"), _sym("x")]
    relations = [
        _rel(RelationKind.CALLS, "seed", "x"),
        _rel(RelationKind.REFERENCES, "seed", "x"),
    ]
    graph = CodeGraph(symbols, relations)
    returnable = {"x": _unit("x", "x.py")}
    seeds = (_hit("seed", 1.0),)
    supports = _collect_graph_supports(
        graph,
        seeds,
        returnable=returnable,
        kind=None,
        path_prefix=None,
    )
    assert supports["x"].supporting_seed_ranks == {1}
    assert supports["x"].graph_support() == pytest.approx(1 / 61)


def test_module_not_returned_and_no_depth2_bridge() -> None:
    symbols = [
        _sym("seed"),
        _sym("pkg", SymbolKind.MODULE),
        _sym("pkg.sibling"),
    ]
    relations = [
        _rel(RelationKind.CONTAINS, "pkg", "seed"),
        _rel(RelationKind.CONTAINS, "pkg", "pkg.sibling"),
    ]
    graph = CodeGraph(symbols, relations)
    returnable = {
        "seed": _unit("seed"),
        "pkg.sibling": _unit("pkg.sibling", "b.py"),
    }
    supports = _collect_graph_supports(
        graph,
        (_hit("seed", 1.0),),
        returnable=returnable,
        kind=None,
        path_prefix=None,
    )
    assert "pkg" not in supports
    assert "pkg.sibling" not in supports


def test_structural_order_and_final_rrf_math() -> None:
    returnable = {
        "a": _unit("a", "a.py"),
        "b": _unit("b", "b.py"),
        "c": _unit("c", "c.py"),
    }
    supports = {
        "b": _GraphCandidateSupport("b", {1}),
        "a": _GraphCandidateSupport("a", {2}),
        "c": _GraphCandidateSupport("c", {1, 2}),
    }
    ranked = _structural_ranked_results(supports, returnable)
    assert [item.symbol_qualified_name for item in ranked] == ["c", "b", "a"]

    hybrid = (_hit("a", 9.0), _hit("b", 8.0))
    graph_list = (_hit("c", 1.0), _hit("a", 0.5))
    fused = fuse_rrf(hybrid, graph_list, limit=10, k=FINAL_GRAPH_RRF_K)
    by_name = {item.symbol_qualified_name: item.score for item in fused}
    assert by_name["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert by_name["b"] == pytest.approx(1 / 62)
    assert by_name["c"] == pytest.approx(1 / 61)


def _index_and_embed(
    tmp_path: Path,
    root: Path = FIXTURE,
) -> tuple[Path, Path, FakeEmbeddingProvider]:
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    analysis = analyze_repository(root, PythonAdapter(), PythonRelationExtractor())
    with IndexDatabase(db_path) as database:
        database.rebuild(analysis)
        documents = load_dense_documents(database)
        query = "authorize payment checkout"
        document_vectors: dict[str, list[float]] = {}
        for document in documents:
            text = document.document_text
            qname = document.qualified_name
            if "handle_payment_checkout" in qname or "summarize_payment_checkout" in qname:
                vector = [1.0, 0.0, 0.0, 0.0]
            elif "authorize_payment_filler" in qname:
                vector = [0.9, 0.1, 0.0, 0.0]
            elif "verify_basket" in qname or "recount_basket" in qname:
                vector = [0.0, 1.0, 0.0, 0.0]
            else:
                vector = [0.2, 0.2, 0.2, 0.2]
            document_vectors[text] = vector
        provider = FakeEmbeddingProvider(
            dimension=4,
            document_vectors=document_vectors,
            query_vectors={query: [1.0, 0.0, 0.0, 0.0]},
            default_document=[0.1, 0.1, 0.1, 0.1],
            default_query=[0.1, 0.1, 0.1, 0.1],
        )
        build_dense_index(database, provider, artifact_dir=artifact_dir)
    return db_path, artifact_dir, provider


def test_uses_hybrid_seeds_and_base_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path, artifact_dir, provider = _index_and_embed(tmp_path)
    calls: list[dict[str, object]] = []

    real_hybrid = search_hybrid

    def tracking_hybrid(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(
            {
                "limit": int(kwargs["limit"]),
                "kind": kwargs.get("kind"),
                "path_prefix": kwargs.get("path_prefix"),
            }
        )
        return real_hybrid(*args, **kwargs)

    monkeypatch.setattr("codeintel.graph_retrieval.search_hybrid", tracking_hybrid)
    with IndexDatabase(db_path, create=False) as database:
        results = search_graph_augmented(
            database,
            provider,
            "authorize payment checkout",
            artifact_dir=artifact_dir,
            limit=3,
            kind=SymbolKind.FUNCTION,
            path_prefix="checkout_",
        )
    assert calls == [{"limit": 10, "kind": SymbolKind.FUNCTION, "path_prefix": "checkout_"}]
    assert results
    assert len(results) <= 3
    assert all(item.kind is SymbolKind.FUNCTION for item in results)
    assert all(str(item.path).startswith("checkout_") for item in results)


def test_seed_count_capped_at_ten(monkeypatch: pytest.MonkeyPatch) -> None:
    hybrid = tuple(_hit(f"seed_{index}", float(20 - index)) for index in range(1, 16))
    monkeypatch.setattr("codeintel.graph_retrieval.search_hybrid", lambda *a, **k: hybrid)

    captured: list[tuple[SearchResult, ...]] = []

    def fake_collect(graph, seeds, **kwargs):  # type: ignore[no-untyped-def]
        captured.append(seeds)
        return {}

    monkeypatch.setattr("codeintel.graph_retrieval._collect_graph_supports", fake_collect)
    monkeypatch.setattr(
        "codeintel.graph_retrieval.CodeGraph",
        lambda symbols, relations: object(),
    )

    class _FakeDb:
        def load_persisted_code_units(self) -> dict[str, PersistedCodeUnitView]:
            return {}

        def load_symbols(self) -> tuple[Symbol, ...]:
            return ()

        def load_relations(self) -> tuple[Relation, ...]:
            return ()

    results = search_graph_augmented(
        _FakeDb(),  # type: ignore[arg-type]
        FakeEmbeddingProvider(dimension=2),
        "query",
        artifact_dir=Path("unused"),
        limit=5,
    )
    assert len(captured) == 1
    assert len(captured[0]) == GRAPH_SEED_COUNT
    assert [item.symbol_qualified_name for item in captured[0]] == [
        f"seed_{index}" for index in range(1, 11)
    ]
    assert [item.symbol_qualified_name for item in results] == [
        f"seed_{index}" for index in range(1, 6)
    ]


def test_empty_structural_preserves_hybrid_order() -> None:
    hybrid = (_hit("a", 9.0), _hit("b", 8.0), _hit("c", 7.0))
    fused = fuse_rrf(hybrid, (), limit=3, k=FINAL_GRAPH_RRF_K)
    assert [item.symbol_qualified_name for item in fused] == ["a", "b", "c"]
    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[1].score == pytest.approx(1 / 62)
    assert fused[2].score == pytest.approx(1 / 63)


def test_stale_dense_rejects_graph_mode(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "mod.py"
    source.write_text(
        "def seed_fn() -> None:\n    helper_fn()\n\ndef helper_fn() -> None:\n    return\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    analysis = analyze_repository(root, PythonAdapter(), PythonRelationExtractor())
    with IndexDatabase(db_path) as database:
        database.rebuild(analysis)
        documents = load_dense_documents(database)
        provider = FakeEmbeddingProvider(
            dimension=2,
            document_vectors={
                doc.document_text: ([1.0, 0.0] if "seed_fn" in doc.qualified_name else [0.0, 1.0])
                for doc in documents
            },
            query_vectors={"seed": [1.0, 0.0]},
        )
        build_dense_index(database, provider, artifact_dir=artifact_dir)

    source.write_text(
        "def seed_fn() -> None:\n    return 1\n\ndef helper_fn() -> None:\n    return 2\n"
        "\ndef extra_fn() -> None:\n    return 3\n",
        encoding="utf-8",
    )
    rebuilt = analyze_repository(root, PythonAdapter(), PythonRelationExtractor())
    with IndexDatabase(db_path, create=False) as database:
        database.rebuild(rebuilt)
        with pytest.raises(DenseIndexMismatchError, match="stale|fingerprint"):
            search_graph_augmented(
                database,
                provider,
                "seed",
                artifact_dir=artifact_dir,
                limit=5,
            )


def test_graph_only_candidate_via_calls(tmp_path: Path) -> None:
    db_path, artifact_dir, provider = _index_and_embed(tmp_path)
    query = "authorize payment checkout"
    with IndexDatabase(db_path, create=False) as database:
        hybrid = search_hybrid(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=10,
        )
        hybrid_names = {item.symbol_qualified_name for item in hybrid}
        # Basket helper should be absent from Hybrid top-10 under fake vectors.
        assert "cart_rules.verify_basket_line_items" not in hybrid_names
        graph_results = search_graph_augmented(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=20,
        )
        graph_names = {item.symbol_qualified_name for item in graph_results}
        assert "cart_rules.verify_basket_line_items" in graph_names
        assert "checkout_handler.handle_payment_checkout" in graph_names
        basket = next(
            item
            for item in graph_results
            if item.symbol_qualified_name == "cart_rules.verify_basket_line_items"
        )
        assert basket.path.as_posix() == "cart_rules.py"
        assert "shopping basket" in basket.source_text
        assert "payment" not in basket.source_text.lower()


def test_hybrid_unchanged_regression(tmp_path: Path) -> None:
    db_path, artifact_dir, provider = _index_and_embed(tmp_path)
    query = "authorize payment checkout"
    with IndexDatabase(db_path, create=False) as database:
        first = search_hybrid(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=5,
        )
        _ = search_graph_augmented(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=5,
        )
        second = search_hybrid(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=5,
        )
    assert [item.symbol_qualified_name for item in first] == [
        item.symbol_qualified_name for item in second
    ]
    assert [item.score for item in first] == [item.score for item in second]


def test_filters_and_determinism(tmp_path: Path) -> None:
    db_path, artifact_dir, provider = _index_and_embed(tmp_path)
    query = "authorize payment checkout"
    with IndexDatabase(db_path, create=False) as database:
        functions = search_graph_augmented(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=20,
            kind=SymbolKind.FUNCTION,
        )
        assert all(item.kind is SymbolKind.FUNCTION for item in functions)
        prefixed = search_graph_augmented(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=20,
            path_prefix="cart_",
        )
        assert all(str(item.path).startswith("cart_") for item in prefixed)
        first = search_graph_augmented(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=10,
        )
        second = search_graph_augmented(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=10,
        )
        assert [(r.symbol_qualified_name, r.score) for r in first] == [
            (r.symbol_qualified_name, r.score) for r in second
        ]


def test_snapshot_independence_from_live_source(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "mod.py"
    source.write_text(
        "def seed_fn() -> None:\n    helper_fn()\n\ndef helper_fn() -> None:\n    return\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    analysis = analyze_repository(root, PythonAdapter(), PythonRelationExtractor())
    with IndexDatabase(db_path) as database:
        database.rebuild(analysis)
        documents = load_dense_documents(database)
        provider = FakeEmbeddingProvider(
            dimension=2,
            document_vectors={
                doc.document_text: ([1.0, 0.0] if "seed_fn" in doc.qualified_name else [0.0, 1.0])
                for doc in documents
            },
            query_vectors={"seed": [1.0, 0.0]},
        )
        build_dense_index(database, provider, artifact_dir=artifact_dir)
        before = search_graph_augmented(
            database,
            provider,
            "seed",
            artifact_dir=artifact_dir,
            limit=5,
        )
        source.write_text("def seed_fn() -> None:\n    return 'MUTATED'\n", encoding="utf-8")
        after = search_graph_augmented(
            database,
            provider,
            "seed",
            artifact_dir=artifact_dir,
            limit=5,
        )
    assert before
    assert after
    assert all("MUTATED" not in item.source_text for item in after)
    assert {item.symbol_qualified_name for item in before} == {
        item.symbol_qualified_name for item in after
    }


def test_empty_query_and_empty_base(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path, artifact_dir, provider = _index_and_embed(tmp_path)
    with IndexDatabase(db_path, create=False) as database:
        assert (
            search_graph_augmented(
                database,
                provider,
                "   ",
                artifact_dir=artifact_dir,
            )
            == ()
        )

    monkeypatch.setattr("codeintel.graph_retrieval.search_hybrid", lambda *a, **k: ())
    with IndexDatabase(db_path, create=False) as database:
        assert (
            search_graph_augmented(
                database,
                provider,
                "authorize payment checkout",
                artifact_dir=artifact_dir,
            )
            == ()
        )


def test_format_helpers_unused_import_guard() -> None:
    # Keep fixture document formatter reachable for embedding key construction.
    assert "symbol:" in format_dense_document("a.b", None, "pass")
