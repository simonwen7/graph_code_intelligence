"""Unit tests for frozen retrieval-benchmark evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from codeintel.evaluation import (
    BENCHMARK_TOP_K,
    EXPECTED_SHOWCASE_COUNT,
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkDefinition,
    BenchmarkError,
    ModeMetrics,
    QueryEvaluation,
    compare_ranks,
    compute_mode_metrics,
    first_relevant_rank,
    load_benchmark_definition,
    pairwise_comparison,
    ranked_qnames,
    validate_gold_qnames,
)
from codeintel.models import (
    RerankedResult,
    RerankExplanation,
    SearchResult,
    SourceSpan,
    SymbolKind,
)

FROZEN_SHA256 = "5125c8facaa3344417ca5ea31958f2f6d1a1393a3675963c8e2cbf8d611ec2de"
BENCHMARK_JSON = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "python_retrieval_v1" / "benchmark.json"
)


def _span() -> SourceSpan:
    return SourceSpan(start_line=1, end_line=1, start_byte=0, end_byte=1)


def _search(qname: str) -> SearchResult:
    return SearchResult(
        symbol_qualified_name=qname,
        kind=SymbolKind.FUNCTION,
        path=Path("x.py"),
        span=_span(),
        signature=None,
        source_text="pass",
        score=1.0,
    )


def _minimal_payload(**overrides: object) -> dict[str, Any]:
    queries: list[dict[str, Any]] = []
    categories = ["lexical", "behavioral", "calls", "inheritance"]
    for category in categories:
        for index in range(1, 7):
            queries.append(
                {
                    "id": f"{category}-{index}",
                    "category": category,
                    "query": f"query {category} {index}",
                    "relevant_qnames": [f"mod.{category}_{index}"],
                    "showcase": category == "lexical" and index <= 2,
                    "notes": "",
                }
            )
    payload: dict[str, Any] = {
        "benchmark_version": 1,
        "benchmark_id": "python-structural-retrieval-v1",
        "language": "python",
        "queries": queries,
    }
    payload.update(overrides)
    return payload


def _write_payload(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_committed_benchmark_freeze_hash() -> None:
    definition = load_benchmark_definition(BENCHMARK_JSON)
    assert definition.benchmark_version == 1
    assert definition.benchmark_id == "python-structural-retrieval-v1"
    assert definition.source_sha256 == FROZEN_SHA256
    assert len(definition.queries) == 24
    assert sum(1 for case in definition.queries if case.showcase) == EXPECTED_SHOWCASE_COUNT


def test_bad_version(tmp_path: Path) -> None:
    path = _write_payload(tmp_path, _minimal_payload(benchmark_version=2))
    with pytest.raises(BenchmarkError, match="benchmark_version"):
        load_benchmark_definition(path)


def test_duplicate_query_ids(tmp_path: Path) -> None:
    payload = _minimal_payload()
    queries = payload["queries"]
    queries[1]["id"] = queries[0]["id"]
    path = _write_payload(tmp_path, payload)
    with pytest.raises(BenchmarkError, match="Duplicate query id"):
        load_benchmark_definition(path)


def test_bad_category(tmp_path: Path) -> None:
    payload = _minimal_payload()
    payload["queries"][0]["category"] = "graph"
    path = _write_payload(tmp_path, payload)
    with pytest.raises(BenchmarkError, match="Invalid category"):
        load_benchmark_definition(path)


def test_wrong_category_counts(tmp_path: Path) -> None:
    payload = _minimal_payload()
    payload["queries"][0]["category"] = "behavioral"
    path = _write_payload(tmp_path, payload)
    with pytest.raises(BenchmarkError, match="Category lexical"):
        load_benchmark_definition(path)


def test_empty_query(tmp_path: Path) -> None:
    payload = _minimal_payload()
    payload["queries"][0]["query"] = "   "
    path = _write_payload(tmp_path, payload)
    with pytest.raises(BenchmarkError, match="empty query"):
        load_benchmark_definition(path)


def test_empty_and_duplicate_gold(tmp_path: Path) -> None:
    payload = _minimal_payload()
    payload["queries"][0]["relevant_qnames"] = []
    path = _write_payload(tmp_path, payload)
    with pytest.raises(BenchmarkError, match="non-empty relevant_qnames"):
        load_benchmark_definition(path)

    payload = _minimal_payload()
    payload["queries"][0]["relevant_qnames"] = ["a.b", "a.b"]
    path = _write_payload(tmp_path, payload)
    with pytest.raises(BenchmarkError, match="duplicate relevant_qnames"):
        load_benchmark_definition(path)


def test_wrong_showcase_count(tmp_path: Path) -> None:
    payload = _minimal_payload()
    payload["queries"][2]["showcase"] = True
    path = _write_payload(tmp_path, payload)
    with pytest.raises(BenchmarkError, match="showcase"):
        load_benchmark_definition(path)


def test_missing_gold_qname() -> None:
    definition = BenchmarkDefinition(
        benchmark_version=1,
        benchmark_id="x",
        language="python",
        queries=(
            BenchmarkCase(
                id="q1",
                category=BenchmarkCategory.LEXICAL,
                query="x",
                relevant_qnames=("missing.fn",),
            ),
        ),
        source_sha256="abc",
    )
    with pytest.raises(BenchmarkError, match="missing from indexed"):
        validate_gold_qnames(definition, {"present.fn"})


def test_metric_exactness_and_multi_gold() -> None:
    ranks = [1, 5, None, 10]
    metrics = compute_mode_metrics(ranks)
    assert metrics == ModeMetrics(
        hit_at_1=0.25,
        hit_at_5=0.5,
        hit_at_10=0.75,
        mrr_at_10=(1.0 + 0.2 + 0.0 + 0.1) / 4.0,
        query_count=4,
    )
    assert first_relevant_rank(["a", "b", "gold"], ["gold", "other"]) == 3
    assert first_relevant_rank(["a", "b"], ["gold"], top_k=BENCHMARK_TOP_K) is None


def test_pairwise_and_missing_policy() -> None:
    assert compare_ranks(1, 3) == "win"
    assert compare_ranks(3, 1) == "loss"
    assert compare_ranks(2, 2) == "tie"
    assert compare_ranks(None, 4) == "loss"
    assert compare_ranks(4, None) == "win"
    assert compare_ranks(None, None) == "tie"

    case = BenchmarkCase(
        id="q",
        category=BenchmarkCategory.CALLS,
        query="q",
        relevant_qnames=("a.b",),
    )
    evaluations = (
        QueryEvaluation(case=case, ranks={"graph": 1, "hybrid": 4}),
        QueryEvaluation(case=case, ranks={"graph": 2, "hybrid": 2}),
        QueryEvaluation(case=case, ranks={"graph": None, "hybrid": 3}),
        QueryEvaluation(case=case, ranks={"graph": None, "hybrid": None}),
    )
    pair = pairwise_comparison("graph", "hybrid", evaluations)
    assert (pair.wins, pair.ties, pair.losses) == (1, 2, 1)


def test_result_normalization_and_duplicate_detection() -> None:
    ranked = ranked_qnames([_search("a.b"), _search("c.d")])
    assert ranked == ("a.b", "c.d")
    explanation = RerankExplanation(
        original_rank=1,
        final_rank=1,
        rank_delta=0,
        contributions=(),
        relation_evidence=(),
    )
    reranked = ranked_qnames([RerankedResult(result=_search("e.f"), explanation=explanation)])
    assert reranked == ("e.f",)
    with pytest.raises(BenchmarkError, match="Duplicate qname"):
        ranked_qnames([_search("a.b"), _search("a.b")])
