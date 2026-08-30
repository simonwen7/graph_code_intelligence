"""Unit tests for structured Graph candidate reranking."""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers.fake_embeddings import FakeEmbeddingProvider

from codeintel.dense import DenseIndexMismatchError, build_dense_index, load_dense_documents
from codeintel.graph import CodeGraph
from codeintel.graph_retrieval import search_graph_augmented
from codeintel.hybrid import search_hybrid
from codeintel.languages.python import PythonAdapter, PythonRelationExtractor
from codeintel.models import (
    ContributionSource,
    EvidenceDirection,
    Relation,
    RelationEvidence,
    RelationKind,
    ResolutionStatus,
    SearchResult,
    SourceSpan,
    Symbol,
    SymbolKind,
)
from codeintel.repository import analyze_repository
from codeintel.reranking import (
    RELATION_SUPPORT_RRF_K,
    RERANK_RELATION_KINDS,
    RERANK_RRF_K,
    _build_evidence_rankings,
    _collect_relation_evidence,
    _evidence_between,
    _fuse_and_explain,
    relation_support,
    rerank_candidate_depth,
    search_reranked,
)
from codeintel.storage import IndexDatabase

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "python_rerank"
SPAN = SourceSpan(1, 1, 0, 1)


def _sym(qname: str, kind: SymbolKind = SymbolKind.FUNCTION) -> Symbol:
    return Symbol(qname.rsplit(".", 1)[-1], qname, kind, SPAN, None, None)


def _rel(
    kind: RelationKind,
    source: str,
    target: str | None,
    *,
    resolution: ResolutionStatus = ResolutionStatus.RESOLVED,
) -> Relation:
    return Relation(
        kind=kind,
        source_qualified_name=source,
        target_qualified_name=target,
        target_text=target or "missing",
        resolution=resolution,
        path=Path("a.py"),
        span=SPAN,
    )


def _hit(qname: str, score: float = 1.0, path: str = "a.py") -> SearchResult:
    return SearchResult(
        symbol_qualified_name=qname,
        kind=SymbolKind.FUNCTION,
        path=Path(path),
        span=SPAN,
        signature=None,
        source_text="pass",
        score=score,
    )


def test_candidate_depth_and_constants() -> None:
    assert RELATION_SUPPORT_RRF_K == 60
    assert RERANK_RRF_K == 60
    assert RelationKind.IMPORTS not in RERANK_RELATION_KINDS
    assert rerank_candidate_depth(1) == 20
    assert rerank_candidate_depth(3) == 20
    assert rerank_candidate_depth(4) == 20
    assert rerank_candidate_depth(10) == 50
    assert rerank_candidate_depth(20) == 100
    with pytest.raises(ValueError):
        rerank_candidate_depth(0)


def test_direction_and_self_exclusion() -> None:
    graph = CodeGraph(
        [_sym("a"), _sym("b")],
        [
            _rel(RelationKind.CALLS, "a", "b"),
            _rel(RelationKind.CALLS, "a", "a"),
        ],
    )
    forward = _evidence_between(graph, "a", 1, "b")
    assert len(forward) == 1
    assert forward[0].direction is EvidenceDirection.SEED_TO_CANDIDATE
    reverse = _evidence_between(graph, "b", 2, "a")
    assert len(reverse) == 1
    assert reverse[0].direction is EvidenceDirection.CANDIDATE_TO_SEED
    assert _evidence_between(graph, "a", 1, "a") == ()
    collected = _collect_relation_evidence(
        graph,
        (_hit("a"),),
        candidate_qnames=frozenset({"a", "b"}),
    )
    assert "a" not in collected


@pytest.mark.parametrize(
    "kind",
    [
        RelationKind.CALLS,
        RelationKind.REFERENCES,
        RelationKind.INHERITS,
        RelationKind.CONTAINS,
    ],
)
def test_direction_generic_across_kinds(kind: RelationKind) -> None:
    if kind in {RelationKind.INHERITS, RelationKind.CONTAINS}:
        source_kind = SymbolKind.CLASS
    else:
        source_kind = SymbolKind.FUNCTION
    if kind is RelationKind.INHERITS:
        target_kind = SymbolKind.CLASS
    elif kind is RelationKind.CONTAINS:
        target_kind = SymbolKind.METHOD
    else:
        target_kind = SymbolKind.FUNCTION
    graph = CodeGraph(
        [_sym("src", source_kind), _sym("tgt", target_kind)],
        [_rel(kind, "src", "tgt")],
    )
    forward = _evidence_between(graph, "src", 1, "tgt")[0].direction
    reverse = _evidence_between(graph, "tgt", 1, "src")[0].direction
    assert forward is EvidenceDirection.SEED_TO_CANDIDATE
    assert reverse is EvidenceDirection.CANDIDATE_TO_SEED


def test_relation_policy_excludes_imports_probable_unresolved() -> None:
    graph = CodeGraph(
        [_sym("seed"), _sym("x"), _sym("y")],
        [
            _rel(RelationKind.IMPORTS, "seed", "x"),
            _rel(RelationKind.CALLS, "seed", "y", resolution=ResolutionStatus.PROBABLE),
            _rel(RelationKind.CALLS, "seed", None, resolution=ResolutionStatus.UNRESOLVED),
            _rel(RelationKind.CALLS, "seed", "x"),
        ],
    )
    seeds = (_hit("seed"),)
    evidence = _collect_relation_evidence(
        graph,
        seeds,
        candidate_qnames=frozenset({"x", "y"}),
    )
    assert [item.relation_kind for item in evidence["x"]] == [RelationKind.CALLS]
    assert "y" not in evidence


def test_evidence_dedup_and_support_formulas() -> None:
    graph = CodeGraph(
        [_sym("seed"), _sym("x")],
        [
            _rel(RelationKind.CALLS, "seed", "x"),
            _rel(RelationKind.CALLS, "seed", "x"),
            _rel(RelationKind.REFERENCES, "seed", "x"),
        ],
    )
    seeds = (_hit("seed"),)
    evidence = _collect_relation_evidence(
        graph,
        seeds,
        candidate_qnames=frozenset({"x"}),
    )
    records = evidence["x"]
    assert len(records) == 2
    assert relation_support(records, relation_kind=RelationKind.CALLS) == pytest.approx(1 / 61)
    assert relation_support(records, relation_kind=RelationKind.REFERENCES) == pytest.approx(1 / 61)

    multi = (
        _hit("s1"),
        _hit("s2"),
        _hit("s3"),
    )
    graph2 = CodeGraph(
        [_sym("s1"), _sym("s2"), _sym("s3"), _sym("x")],
        [
            _rel(RelationKind.CALLS, "s1", "x"),
            _rel(RelationKind.CALLS, "s3", "x"),
        ],
    )
    evidence2 = _collect_relation_evidence(
        graph2,
        multi,
        candidate_qnames=frozenset({"x"}),
    )
    assert relation_support(
        evidence2["x"],
        relation_kind=RelationKind.CALLS,
    ) == pytest.approx(1 / 61 + 1 / 63)


def test_no_new_candidates_outside_graph_set() -> None:
    graph = CodeGraph(
        [_sym("seed"), _sym("a"), _sym("d")],
        [_rel(RelationKind.CALLS, "seed", "d")],
    )
    evidence = _collect_relation_evidence(
        graph,
        (_hit("seed"),),
        candidate_qnames=frozenset({"a"}),
    )
    assert evidence == {}


def test_evidence_rankings_and_final_rrf_math() -> None:
    graph_candidates = (_hit("a", path="a.py"), _hit("b", path="b.py"), _hit("c", path="c.py"))
    evidence_by_candidate = {
        "b": (
            RelationEvidence(
                "s2",
                2,
                RelationKind.CALLS,
                EvidenceDirection.SEED_TO_CANDIDATE,
            ),
        ),
        "c": (
            RelationEvidence(
                "s1",
                1,
                RelationKind.CALLS,
                EvidenceDirection.SEED_TO_CANDIDATE,
            ),
            RelationEvidence(
                "s1",
                1,
                RelationKind.INHERITS,
                EvidenceDirection.SEED_TO_CANDIDATE,
            ),
        ),
    }
    rankings = _build_evidence_rankings(
        evidence_by_candidate,
        graph_candidates=graph_candidates,
    )
    assert [item.symbol_qualified_name for item in rankings[RelationKind.CALLS]] == ["c", "b"]
    assert [item.symbol_qualified_name for item in rankings[RelationKind.INHERITS]] == ["c"]

    fused = _fuse_and_explain(
        [
            (ContributionSource.GRAPH_BASE, graph_candidates),
            (ContributionSource.CALLS, rankings[RelationKind.CALLS]),
            (ContributionSource.INHERITS, rankings[RelationKind.INHERITS]),
        ],
        original_ranks={"a": 1, "b": 2, "c": 3},
        evidence_by_candidate=evidence_by_candidate,
        exemplars=graph_candidates,
        limit=10,
        k=60,
    )
    by_name = {item.result.symbol_qualified_name: item for item in fused}
    assert by_name["a"].result.score == pytest.approx(1 / 61)
    assert by_name["b"].result.score == pytest.approx(1 / 62 + 1 / 62)
    assert by_name["c"].result.score == pytest.approx(1 / 63 + 1 / 61 + 1 / 61)
    assert [item.result.symbol_qualified_name for item in fused] == ["c", "b", "a"]
    assert by_name["c"].explanation.original_rank == 3
    assert by_name["c"].explanation.final_rank == 1
    assert by_name["c"].explanation.rank_delta == 2
    for item in fused:
        total = sum(c.rrf_contribution for c in item.explanation.contributions)
        assert total == pytest.approx(item.result.score)


def test_empty_evidence_preserves_graph_order() -> None:
    graph_candidates = (_hit("a"), _hit("b"), _hit("c"))
    fused = _fuse_and_explain(
        [(ContributionSource.GRAPH_BASE, graph_candidates)],
        original_ranks={"a": 1, "b": 2, "c": 3},
        evidence_by_candidate={},
        exemplars=graph_candidates,
        limit=10,
        k=60,
    )
    assert [item.result.symbol_qualified_name for item in fused] == ["a", "b", "c"]
    assert all(item.explanation.relation_evidence == () for item in fused)
    assert fused[0].explanation.rank_delta == 0
    assert fused[0].result.score == pytest.approx(1 / 61)
    assert fused[1].result.score == pytest.approx(1 / 62)
    assert fused[2].result.score == pytest.approx(1 / 63)


def test_seed_rank_beyond_ten_never_contributes() -> None:
    seeds = tuple(_hit(f"s{index}") for index in range(1, 16))
    symbols = [_sym(f"s{index}") for index in range(1, 16)] + [_sym("x")]
    relations = [_rel(RelationKind.CALLS, f"s{index}", "x") for index in range(1, 16)]
    graph = CodeGraph(symbols, relations)
    evidence = _collect_relation_evidence(
        graph,
        seeds[:10],
        candidate_qnames=frozenset({"x"}),
    )
    ranks = {item.seed_rank for item in evidence["x"]}
    assert ranks == set(range(1, 11))
    assert 11 not in ranks
    assert relation_support(evidence["x"], relation_kind=RelationKind.CALLS) == pytest.approx(
        sum(1 / (60 + rank) for rank in range(1, 11))
    )


def test_support_magnitude_and_graph_score_excluded_from_final() -> None:
    # Large RelationSupport still contributes only list-rank RRF, not support magnitude.
    high_support = tuple(
        RelationEvidence(
            f"s{index}",
            index,
            RelationKind.CALLS,
            EvidenceDirection.SEED_TO_CANDIDATE,
        )
        for index in range(1, 6)
    )
    low_support = (
        RelationEvidence(
            "s10",
            10,
            RelationKind.CALLS,
            EvidenceDirection.SEED_TO_CANDIDATE,
        ),
    )
    assert relation_support(high_support, relation_kind=RelationKind.CALLS) > relation_support(
        low_support,
        relation_kind=RelationKind.CALLS,
    )
    candidates = (
        _hit("x", score=999.0, path="x.py"),
        _hit("y", score=0.001, path="y.py"),
    )
    evidence: dict[str, tuple[RelationEvidence, ...]] = {
        "x": high_support,
        "y": low_support,
    }
    rankings = _build_evidence_rankings(evidence, graph_candidates=candidates)
    assert [item.symbol_qualified_name for item in rankings[RelationKind.CALLS]] == ["x", "y"]
    fused = _fuse_and_explain(
        [
            (ContributionSource.GRAPH_BASE, candidates),
            (ContributionSource.CALLS, rankings[RelationKind.CALLS]),
        ],
        original_ranks={"x": 1, "y": 2},
        evidence_by_candidate=evidence,
        exemplars=candidates,
        limit=10,
        k=60,
    )
    by_name = {item.result.symbol_qualified_name: item for item in fused}
    assert by_name["x"].result.score == pytest.approx(1 / 61 + 1 / 61)
    assert by_name["y"].result.score == pytest.approx(1 / 62 + 1 / 62)
    assert by_name["x"].result.score != pytest.approx(
        relation_support(high_support, relation_kind=RelationKind.CALLS)
    )
    # GRAPH_BASE uses ranks only — Graph SearchResult.score is ignored.
    base_only = _fuse_and_explain(
        [(ContributionSource.GRAPH_BASE, candidates)],
        original_ranks={"x": 1, "y": 2},
        evidence_by_candidate={},
        exemplars=candidates,
        limit=10,
        k=60,
    )
    assert base_only[0].result.score == pytest.approx(1 / 61)
    assert base_only[1].result.score == pytest.approx(1 / 62)


def test_rank_delta_sign_and_evidence_consistency() -> None:
    graph_candidates = (_hit("a"), _hit("b"), _hit("c"))
    evidence_by_candidate: dict[str, tuple[RelationEvidence, ...]] = {
        "a": (
            RelationEvidence(
                "s1",
                1,
                RelationKind.CALLS,
                EvidenceDirection.SEED_TO_CANDIDATE,
            ),
        ),
    }
    rankings = _build_evidence_rankings(
        evidence_by_candidate,
        graph_candidates=graph_candidates,
    )
    # Boost a so it stays #1 (delta 0); c stays last without evidence.
    fused = _fuse_and_explain(
        [
            (ContributionSource.GRAPH_BASE, graph_candidates),
            (ContributionSource.CALLS, rankings[RelationKind.CALLS]),
        ],
        original_ranks={"a": 1, "b": 2, "c": 3},
        evidence_by_candidate=evidence_by_candidate,
        exemplars=graph_candidates,
        limit=10,
        k=60,
    )
    by_name = {item.result.symbol_qualified_name: item for item in fused}
    assert by_name["a"].explanation.rank_delta == 0
    # Move c from 3 to 1 via strong evidence in a separate scenario.
    strong_c: dict[str, tuple[RelationEvidence, ...]] = {
        "c": (
            RelationEvidence(
                "s1",
                1,
                RelationKind.CALLS,
                EvidenceDirection.SEED_TO_CANDIDATE,
            ),
            RelationEvidence(
                "s1",
                1,
                RelationKind.REFERENCES,
                EvidenceDirection.SEED_TO_CANDIDATE,
            ),
        ),
    }
    ranks_c = _build_evidence_rankings(strong_c, graph_candidates=graph_candidates)
    moved = _fuse_and_explain(
        [
            (ContributionSource.GRAPH_BASE, graph_candidates),
            (ContributionSource.CALLS, ranks_c[RelationKind.CALLS]),
            (ContributionSource.REFERENCES, ranks_c[RelationKind.REFERENCES]),
        ],
        original_ranks={"a": 1, "b": 2, "c": 3},
        evidence_by_candidate=strong_c,
        exemplars=graph_candidates,
        limit=10,
        k=60,
    )
    c_item = next(item for item in moved if item.result.symbol_qualified_name == "c")
    assert c_item.explanation.original_rank == 3
    assert c_item.explanation.final_rank == 1
    assert c_item.explanation.rank_delta == 2
    a_item = next(item for item in moved if item.result.symbol_qualified_name == "a")
    assert a_item.explanation.rank_delta < 0  # moved down
    for item in moved:
        sources = {contrib.source for contrib in item.explanation.contributions}
        kinds = {evidence.relation_kind for evidence in item.explanation.relation_evidence}
        for source, kind in (
            (ContributionSource.CALLS, RelationKind.CALLS),
            (ContributionSource.REFERENCES, RelationKind.REFERENCES),
            (ContributionSource.INHERITS, RelationKind.INHERITS),
            (ContributionSource.CONTAINS, RelationKind.CONTAINS),
        ):
            if source in sources:
                assert kind in kinds
        total = sum(contrib.rrf_contribution for contrib in item.explanation.contributions)
        assert total == pytest.approx(item.result.score)


def test_original_rank_follows_graph_return_order_not_score() -> None:
    # Same Graph scores, distinct return order must define original_rank.
    graph_candidates = (
        _hit("b", score=1.0, path="b.py"),
        _hit("a", score=1.0, path="a.py"),
    )
    fused = _fuse_and_explain(
        [(ContributionSource.GRAPH_BASE, graph_candidates)],
        original_ranks={"b": 1, "a": 2},
        evidence_by_candidate={},
        exemplars=graph_candidates,
        limit=10,
        k=60,
    )
    assert fused[0].result.symbol_qualified_name == "b"
    assert fused[0].explanation.original_rank == 1
    assert fused[1].result.symbol_qualified_name == "a"
    assert fused[1].explanation.original_rank == 2


def _index_and_embed(tmp_path: Path) -> tuple[Path, Path, FakeEmbeddingProvider]:
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    analysis = analyze_repository(FIXTURE, PythonAdapter(), PythonRelationExtractor())
    query = "authorize payment transfer"
    with IndexDatabase(db_path) as database:
        database.rebuild(analysis)
        documents = load_dense_documents(database)
        document_vectors: dict[str, list[float]] = {}
        for document in documents:
            qname = document.qualified_name
            if "authorize_payment_transfer" in qname or "describe_payment_transfer" in qname:
                vector = [1.0, 0.0, 0.0, 0.0]
            elif "authorize_payment_filler" in qname:
                vector = [0.85, 0.15, 0.0, 0.0]
            elif "verify_line_bundle" in qname or "recount_sku" in qname:
                vector = [0.0, 1.0, 0.0, 0.0]
            else:
                vector = [0.2, 0.2, 0.2, 0.2]
            document_vectors[document.document_text] = vector
        provider = FakeEmbeddingProvider(
            dimension=4,
            document_vectors=document_vectors,
            query_vectors={query: [1.0, 0.0, 0.0, 0.0]},
            default_document=[0.1, 0.1, 0.1, 0.1],
            default_query=[0.1, 0.1, 0.1, 0.1],
        )
        build_dense_index(database, provider, artifact_dir=artifact_dir)
    return db_path, artifact_dir, provider


def test_uses_graph_source_and_depth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path, artifact_dir, provider = _index_and_embed(tmp_path)
    calls: list[int] = []
    real_graph = search_graph_augmented

    def tracking_graph(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(int(kwargs["limit"]))
        return real_graph(*args, **kwargs)

    monkeypatch.setattr("codeintel.reranking.search_graph_augmented", tracking_graph)
    with IndexDatabase(db_path, create=False) as database:
        results = search_reranked(
            database,
            provider,
            "authorize payment transfer",
            artifact_dir=artifact_dir,
            limit=3,
        )
    assert calls == [20]
    assert len(results) <= 3


def test_fixture_rerank_introduces_structural_boost(tmp_path: Path) -> None:
    db_path, artifact_dir, provider = _index_and_embed(tmp_path)
    query = "authorize payment transfer"
    with IndexDatabase(db_path, create=False) as database:
        graph = search_graph_augmented(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=20,
        )
        reranked = search_reranked(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=20,
        )
        assert any(item.symbol_qualified_name == "line_checks.verify_line_bundle" for item in graph)
        assert any(
            item.result.symbol_qualified_name == "line_checks.verify_line_bundle"
            for item in reranked
        )
        bundle = next(
            item
            for item in reranked
            if item.result.symbol_qualified_name == "line_checks.verify_line_bundle"
        )
        assert any(c.source is ContributionSource.CALLS for c in bundle.explanation.contributions)
        assert any(
            e.relation_kind is RelationKind.CALLS for e in bundle.explanation.relation_evidence
        )
        total = sum(c.rrf_contribution for c in bundle.explanation.contributions)
        assert total == pytest.approx(bundle.result.score)


def test_hybrid_seeds_forward_filters(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path, artifact_dir, provider = _index_and_embed(tmp_path)
    hybrid_kwargs: list[dict[str, object]] = []
    real_hybrid = search_hybrid

    def tracking_hybrid(*args, **kwargs):  # type: ignore[no-untyped-def]
        hybrid_kwargs.append(
            {
                "limit": kwargs["limit"],
                "kind": kwargs.get("kind"),
                "path_prefix": kwargs.get("path_prefix"),
            }
        )
        return real_hybrid(*args, **kwargs)

    monkeypatch.setattr("codeintel.reranking.search_hybrid", tracking_hybrid)
    with IndexDatabase(db_path, create=False) as database:
        results = search_reranked(
            database,
            provider,
            "authorize payment transfer",
            artifact_dir=artifact_dir,
            limit=2,
            kind=SymbolKind.FUNCTION,
            path_prefix="payment_",
        )
    assert hybrid_kwargs
    assert hybrid_kwargs[0]["limit"] == 20
    assert hybrid_kwargs[0]["kind"] is SymbolKind.FUNCTION
    assert hybrid_kwargs[0]["path_prefix"] == "payment_"
    assert all(item.result.kind is SymbolKind.FUNCTION for item in results)
    assert all(str(item.result.path).startswith("payment_") for item in results)


def test_graph_baseline_unchanged(tmp_path: Path) -> None:
    db_path, artifact_dir, provider = _index_and_embed(tmp_path)
    query = "authorize payment transfer"
    with IndexDatabase(db_path, create=False) as database:
        first = search_graph_augmented(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=5,
        )
        _ = search_reranked(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=5,
        )
        second = search_graph_augmented(
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


def test_determinism_and_snapshot(tmp_path: Path) -> None:
    db_path, artifact_dir, provider = _index_and_embed(tmp_path)
    query = "authorize payment transfer"
    with IndexDatabase(db_path, create=False) as database:
        first = search_reranked(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=10,
        )
        second = search_reranked(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=10,
        )
    assert [
        (
            item.result.symbol_qualified_name,
            item.result.score,
            item.explanation.original_rank,
            item.explanation.final_rank,
            item.explanation.contributions,
            item.explanation.relation_evidence,
        )
        for item in first
    ] == [
        (
            item.result.symbol_qualified_name,
            item.result.score,
            item.explanation.original_rank,
            item.explanation.final_rank,
            item.explanation.contributions,
            item.explanation.relation_evidence,
        )
        for item in second
    ]


def test_snapshot_independence_and_stale_dense(tmp_path: Path) -> None:
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
        before = search_reranked(
            database,
            provider,
            "seed",
            artifact_dir=artifact_dir,
            limit=5,
        )
        source.write_text("def seed_fn() -> None:\n    return 'MUTATED'\n", encoding="utf-8")
        after = search_reranked(
            database,
            provider,
            "seed",
            artifact_dir=artifact_dir,
            limit=5,
        )
    assert before
    assert all("MUTATED" not in item.result.source_text for item in after)

    source.write_text(
        "def seed_fn() -> None:\n    return 1\n\ndef helper_fn() -> None:\n    return 2\n"
        "\ndef extra_fn() -> None:\n    return 3\n",
        encoding="utf-8",
    )
    rebuilt = analyze_repository(root, PythonAdapter(), PythonRelationExtractor())
    with IndexDatabase(db_path, create=False) as database:
        database.rebuild(rebuilt)
        with pytest.raises(DenseIndexMismatchError):
            search_reranked(
                database,
                provider,
                "seed",
                artifact_dir=artifact_dir,
                limit=5,
            )


def test_empty_query_and_empty_graph(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path, artifact_dir, provider = _index_and_embed(tmp_path)
    with IndexDatabase(db_path, create=False) as database:
        assert (
            search_reranked(
                database,
                provider,
                "  ",
                artifact_dir=artifact_dir,
            )
            == ()
        )
    monkeypatch.setattr("codeintel.reranking.search_graph_augmented", lambda *a, **k: ())
    with IndexDatabase(db_path, create=False) as database:
        assert (
            search_reranked(
                database,
                provider,
                "authorize payment transfer",
                artifact_dir=artifact_dir,
            )
            == ()
        )
