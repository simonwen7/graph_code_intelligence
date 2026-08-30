"""Exact Reciprocal Rank Fusion tests for hybrid retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers.fake_embeddings import FakeEmbeddingProvider

from codeintel.dense import build_dense_index, format_dense_document, load_dense_documents
from codeintel.hybrid import RRF_K, fuse_rrf, hybrid_candidate_depth, search_hybrid
from codeintel.languages.python import PythonAdapter, PythonRelationExtractor
from codeintel.models import SearchResult, SourceSpan, SymbolKind
from codeintel.repository import analyze_repository
from codeintel.storage import IndexDatabase

DENSE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "python_dense"


def _result(qname: str, score: float, path: str = "a.py") -> SearchResult:
    return SearchResult(
        symbol_qualified_name=qname,
        kind=SymbolKind.FUNCTION,
        path=Path(path),
        span=SourceSpan(1, 1, 0, 1),
        signature=None,
        source_text="pass",
        score=score,
    )


def test_rrf_constants_and_candidate_depth() -> None:
    assert RRF_K == 60
    assert hybrid_candidate_depth(1) == 50
    assert hybrid_candidate_depth(10) == 50
    assert hybrid_candidate_depth(11) == 55
    assert hybrid_candidate_depth(20) == 100
    with pytest.raises(ValueError):
        hybrid_candidate_depth(0)


def test_rrf_exact_example_a_and_b() -> None:
    lexical = (_result("a", 9.0), _result("b", 8.0))
    dense = (_result("x", 0.95), _result("a", 0.9, "a.py"))
    fused = fuse_rrf(lexical, dense, limit=10)
    by_name = {item.symbol_qualified_name: item.score for item in fused}
    assert by_name["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert by_name["b"] == pytest.approx(1 / 62)


def test_rrf_math_both_lists_and_missing() -> None:
    lexical = (_result("a", 9.0), _result("b", 8.0), _result("c", 7.0))
    dense = (_result("b", 0.9), _result("d", 0.8), _result("a", 0.7))
    fused = fuse_rrf(lexical, dense, limit=10)

    expected = {
        "a": 1 / (60 + 1) + 1 / (60 + 3),
        "b": 1 / (60 + 2) + 1 / (60 + 1),
        "c": 1 / (60 + 3),
        "d": 1 / (60 + 2),
    }
    by_name = {item.symbol_qualified_name: item.score for item in fused}
    assert by_name == pytest.approx(expected)
    assert [item.symbol_qualified_name for item in fused] == sorted(
        expected,
        key=lambda name: (-expected[name], name),
    )


def test_rrf_dedup_limit_and_ties() -> None:
    lexical = (_result("b", 1.0, "b.py"), _result("a", 1.0, "a.py"))
    dense = (_result("a", 1.0, "a.py"), _result("b", 1.0, "b.py"))
    fused = fuse_rrf(lexical, dense, limit=1)
    assert len(fused) == 1
    # Equal RRF contributions; tie-break by qname ASC => "a"
    assert fused[0].symbol_qualified_name == "a"
    assert fused[0].score == pytest.approx(1 / (60 + 2) + 1 / (60 + 1))


def test_search_hybrid_end_to_end(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    analysis = analyze_repository(DENSE_ROOT, PythonAdapter(), PythonRelationExtractor())
    with IndexDatabase(db_path) as database:
        database.rebuild(analysis)
        documents = load_dense_documents(database)
        preferred = "auth_session.apply_login_rate_limit"
        preferred_doc = next(doc for doc in documents if doc.qualified_name == preferred)
        document_vectors = {
            doc.document_text: [1.0 if doc.qualified_name == preferred else 0.01 for _ in range(4)]
            for doc in documents
        }
        # Make preferred distinctly aligned on axis 0.
        for doc in documents:
            vector = [0.0, 0.0, 0.0, 0.0]
            if doc.qualified_name == preferred:
                vector[0] = 1.0
            else:
                vector[1] = 1.0
            document_vectors[doc.document_text] = vector
        query = "prevent too many authentication attempts"
        provider = FakeEmbeddingProvider(
            dimension=4,
            document_vectors=document_vectors,
            query_vectors={query: [1.0, 0.0, 0.0, 0.0]},
        )
        build_dense_index(database, provider, artifact_dir=artifact_dir)
        results = search_hybrid(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=3,
        )
        assert preferred_doc.qualified_name in {result.symbol_qualified_name for result in results}
        assert results == tuple(
            sorted(
                results,
                key=lambda item: (-item.score, item.symbol_qualified_name, item.path.as_posix()),
            )
        )
        filtered = search_hybrid(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=5,
            path_prefix="auth_",
        )
        assert all(str(result.path).startswith("auth_") for result in filtered)
        assert (
            format_dense_document(
                preferred_doc.qualified_name,
                preferred_doc.signature,
                preferred_doc.source_text,
            )
            == preferred_doc.document_text
        )
