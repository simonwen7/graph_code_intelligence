"""Frozen retrieval-benchmark evaluation over persisted index and dense artifacts.

Evaluates the existing search ladder without duplicating ranking logic.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from codeintel import __version__
from codeintel.dense import (
    DenseIndexError,
    load_and_validate_dense_artifact,
    search_dense,
)
from codeintel.embeddings import EmbeddingProvider
from codeintel.graph_retrieval import search_graph_augmented
from codeintel.hybrid import search_hybrid
from codeintel.lexical import search_code_units
from codeintel.models import RerankedResult, SearchResult
from codeintel.reranking import search_reranked
from codeintel.storage.database import IndexDatabase

BENCHMARK_TOP_K = 10
EXPECTED_QUERY_COUNT = 24
EXPECTED_CATEGORY_COUNT = 6
EXPECTED_SHOWCASE_COUNT = 2
EVALUATION_MODES: tuple[str, ...] = ("lexical", "dense", "hybrid", "graph", "reranked")
BENCHMARK_CATEGORIES: tuple[str, ...] = ("lexical", "behavioral", "calls", "inheritance")


class BenchmarkError(ValueError):
    """Raised when a benchmark definition or evaluation setup is invalid."""


class BenchmarkCategory(StrEnum):
    """Stable benchmark query categories."""

    LEXICAL = "lexical"
    BEHAVIORAL = "behavioral"
    CALLS = "calls"
    INHERITANCE = "inheritance"


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One labeled retrieval query."""

    id: str
    category: BenchmarkCategory
    query: str
    relevant_qnames: tuple[str, ...]
    showcase: bool = False
    notes: str = ""


@dataclass(frozen=True, slots=True)
class BenchmarkDefinition:
    """Frozen labeled benchmark envelope."""

    benchmark_version: int
    benchmark_id: str
    language: str
    queries: tuple[BenchmarkCase, ...]
    source_path: Path | None = None
    source_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ModeQueryResult:
    """First-relevant rank for one mode on one query (None = miss in top-K)."""

    mode: str
    first_relevant_rank: int | None


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    """Per-query ranks across frozen evaluation modes."""

    case: BenchmarkCase
    ranks: Mapping[str, int | None]


@dataclass(frozen=True, slots=True)
class ModeMetrics:
    """Aggregate ranking metrics for one mode over a query set."""

    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    mrr_at_10: float
    query_count: int


@dataclass(frozen=True, slots=True)
class PairwiseComparison:
    """Per-query win/tie/loss between two modes on first-relevant rank."""

    left_mode: str
    right_mode: str
    wins: int
    ties: int
    losses: int


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Complete deterministic evaluation result."""

    benchmark_id: str
    benchmark_version: int
    benchmark_sha256: str
    query_count: int
    top_k: int
    provider_id: str
    model_id: str
    python_version: str
    platform: str
    engine_version: str
    corpus_fingerprint: str
    aggregate: Mapping[str, ModeMetrics]
    by_category: Mapping[str, Mapping[str, ModeMetrics]]
    pairwise: tuple[PairwiseComparison, ...]
    queries: tuple[QueryEvaluation, ...]


def sha256_file(path: Path) -> str:
    """Return lowercase hex SHA-256 of a file's raw bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_benchmark_definition(path: Path) -> BenchmarkDefinition:
    """Load and structurally validate a benchmark JSON file."""
    text = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"Invalid benchmark JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkError("Benchmark root must be a JSON object")

    version = payload.get("benchmark_version")
    benchmark_id = payload.get("benchmark_id")
    language = payload.get("language")
    raw_queries = payload.get("queries")
    if version != 1:
        raise BenchmarkError(f"Unsupported benchmark_version: {version!r}")
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise BenchmarkError("benchmark_id must be a non-empty string")
    if language != "python":
        raise BenchmarkError(f"Unsupported benchmark language: {language!r}")
    if not isinstance(raw_queries, list):
        raise BenchmarkError("queries must be a JSON array")
    if len(raw_queries) != EXPECTED_QUERY_COUNT:
        raise BenchmarkError(
            f"Expected exactly {EXPECTED_QUERY_COUNT} queries, found {len(raw_queries)}"
        )

    cases: list[BenchmarkCase] = []
    seen_ids: set[str] = set()
    category_counts = {category: 0 for category in BENCHMARK_CATEGORIES}
    showcase_count = 0

    for index, item in enumerate(raw_queries):
        if not isinstance(item, dict):
            raise BenchmarkError(f"queries[{index}] must be an object")
        case_id = item.get("id")
        category_raw = item.get("category")
        query = item.get("query")
        relevant = item.get("relevant_qnames")
        showcase = item.get("showcase", False)
        notes = item.get("notes", "")
        if not isinstance(case_id, str) or not case_id.strip():
            raise BenchmarkError(f"queries[{index}].id must be a non-empty string")
        if case_id in seen_ids:
            raise BenchmarkError(f"Duplicate query id: {case_id}")
        seen_ids.add(case_id)
        if category_raw not in BENCHMARK_CATEGORIES:
            raise BenchmarkError(f"Invalid category for {case_id}: {category_raw!r}")
        if not isinstance(query, str) or not query.strip():
            raise BenchmarkError(f"Query {case_id} has empty query text")
        if not isinstance(relevant, list) or not relevant:
            raise BenchmarkError(f"Query {case_id} requires non-empty relevant_qnames")
        if not all(isinstance(name, str) and name.strip() for name in relevant):
            raise BenchmarkError(f"Query {case_id} has invalid relevant_qnames entries")
        if len(set(relevant)) != len(relevant):
            raise BenchmarkError(f"Query {case_id} has duplicate relevant_qnames")
        if not isinstance(showcase, bool):
            raise BenchmarkError(f"Query {case_id} showcase must be a boolean")
        if not isinstance(notes, str):
            raise BenchmarkError(f"Query {case_id} notes must be a string")
        if showcase:
            showcase_count += 1
        category = BenchmarkCategory(category_raw)
        category_counts[category.value] += 1
        cases.append(
            BenchmarkCase(
                id=case_id,
                category=category,
                query=query,
                relevant_qnames=tuple(relevant),
                showcase=showcase,
                notes=notes,
            )
        )

    for category_name, count in category_counts.items():
        if count != EXPECTED_CATEGORY_COUNT:
            raise BenchmarkError(
                f"Category {category_name} must have exactly "
                f"{EXPECTED_CATEGORY_COUNT} queries, found {count}"
            )
    if showcase_count != EXPECTED_SHOWCASE_COUNT:
        raise BenchmarkError(
            f"Expected exactly {EXPECTED_SHOWCASE_COUNT} showcase queries, found {showcase_count}"
        )

    return BenchmarkDefinition(
        benchmark_version=1,
        benchmark_id=benchmark_id,
        language="python",
        queries=tuple(cases),
        source_path=path.resolve(),
        source_sha256=digest,
    )


def validate_gold_qnames(
    definition: BenchmarkDefinition,
    code_unit_qnames: set[str],
) -> None:
    """Fail if any gold qname is missing from indexed CodeUnits."""
    missing: list[str] = []
    for case in definition.queries:
        for qname in case.relevant_qnames:
            if qname not in code_unit_qnames:
                missing.append(f"{case.id}:{qname}")
    if missing:
        joined = ", ".join(missing)
        raise BenchmarkError(f"Gold qnames missing from indexed CodeUnits: {joined}")


def ranked_qnames(results: Sequence[SearchResult | RerankedResult]) -> tuple[str, ...]:
    """Extract 1-based ranked qnames; reject duplicates."""
    ordered: list[str] = []
    seen: set[str] = set()
    for item in results:
        if isinstance(item, RerankedResult):
            qname = item.result.symbol_qualified_name
        elif isinstance(item, SearchResult):
            qname = item.symbol_qualified_name
        else:
            raise BenchmarkError(f"Unsupported result type: {type(item)!r}")
        if qname in seen:
            raise BenchmarkError(f"Duplicate qname in retrieval results: {qname}")
        seen.add(qname)
        ordered.append(qname)
    return tuple(ordered)


def first_relevant_rank(
    ranked: Sequence[str],
    relevant_qnames: Sequence[str],
    *,
    top_k: int = BENCHMARK_TOP_K,
) -> int | None:
    """Return 1-based rank of the first relevant hit within ``top_k``, else None."""
    relevant = set(relevant_qnames)
    for index, qname in enumerate(ranked[:top_k], start=1):
        if qname in relevant:
            return index
    return None


def compute_mode_metrics(
    ranks: Sequence[int | None],
    *,
    top_k: int = BENCHMARK_TOP_K,
) -> ModeMetrics:
    """Compute Hit@1/5/10 and MRR@10 from first-relevant ranks."""
    if top_k != BENCHMARK_TOP_K:
        raise BenchmarkError(f"TOP_K is frozen at {BENCHMARK_TOP_K}")
    if not ranks:
        raise BenchmarkError("Cannot compute metrics over an empty rank list")
    n = len(ranks)
    hit1 = sum(1 for rank in ranks if rank is not None and rank <= 1) / n
    hit5 = sum(1 for rank in ranks if rank is not None and rank <= 5) / n
    hit10 = sum(1 for rank in ranks if rank is not None and rank <= 10) / n
    mrr = sum((1.0 / rank) if rank is not None else 0.0 for rank in ranks) / n
    return ModeMetrics(
        hit_at_1=hit1,
        hit_at_5=hit5,
        hit_at_10=hit10,
        mrr_at_10=mrr,
        query_count=n,
    )


def compare_ranks(left: int | None, right: int | None) -> str:
    """Compare first-relevant ranks; missing is worse than any 1..10.

    Returns ``win`` if left is better than right, ``loss`` if worse, else ``tie``.
    """
    if left is None and right is None:
        return "tie"
    if left is None:
        return "loss"
    if right is None:
        return "win"
    if left < right:
        return "win"
    if left > right:
        return "loss"
    return "tie"


def pairwise_comparison(
    left_mode: str,
    right_mode: str,
    evaluations: Sequence[QueryEvaluation],
) -> PairwiseComparison:
    """Aggregate wins/ties/losses for left vs right on first-relevant ranks."""
    wins = ties = losses = 0
    for evaluation in evaluations:
        outcome = compare_ranks(evaluation.ranks[left_mode], evaluation.ranks[right_mode])
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        else:
            ties += 1
    return PairwiseComparison(
        left_mode=left_mode,
        right_mode=right_mode,
        wins=wins,
        ties=ties,
        losses=losses,
    )


def _search_mode(
    mode: str,
    *,
    database: IndexDatabase,
    dense_dir: Path,
    provider: EmbeddingProvider,
    query: str,
    limit: int,
) -> Sequence[SearchResult | RerankedResult]:
    if mode == "lexical":
        return search_code_units(database, query, limit=limit)
    if mode == "dense":
        return search_dense(database, provider, query, artifact_dir=dense_dir, limit=limit)
    if mode == "hybrid":
        return search_hybrid(database, provider, query, artifact_dir=dense_dir, limit=limit)
    if mode == "graph":
        return search_graph_augmented(
            database, provider, query, artifact_dir=dense_dir, limit=limit
        )
    if mode == "reranked":
        return search_reranked(database, provider, query, artifact_dir=dense_dir, limit=limit)
    raise BenchmarkError(f"Unknown evaluation mode: {mode}")


def run_benchmark(
    definition: BenchmarkDefinition,
    *,
    database: IndexDatabase,
    dense_dir: Path,
    provider: EmbeddingProvider,
    top_k: int = BENCHMARK_TOP_K,
) -> BenchmarkResult:
    """Evaluate all modes for every query against an existing index + dense artifact."""
    if top_k != BENCHMARK_TOP_K:
        raise BenchmarkError(f"TOP_K is frozen at {BENCHMARK_TOP_K}")
    if definition.source_sha256 is None:
        raise BenchmarkError("Benchmark definition is missing source_sha256")

    code_unit_qnames = {qname for qname, _unit in database.load_code_units()}
    validate_gold_qnames(definition, code_unit_qnames)

    try:
        metadata, _index = load_and_validate_dense_artifact(
            database,
            provider,
            artifact_dir=dense_dir,
        )
    except DenseIndexError as exc:
        raise BenchmarkError(str(exc)) from exc
    corpus_fingerprint = str(metadata["corpus_fingerprint"])

    evaluations: list[QueryEvaluation] = []
    for case in definition.queries:
        ranks: dict[str, int | None] = {}
        for mode in EVALUATION_MODES:
            results = _search_mode(
                mode,
                database=database,
                dense_dir=dense_dir,
                provider=provider,
                query=case.query,
                limit=top_k,
            )
            ranked = ranked_qnames(results)
            ranks[mode] = first_relevant_rank(ranked, case.relevant_qnames, top_k=top_k)
        evaluations.append(QueryEvaluation(case=case, ranks=ranks))

    evaluation_tuple = tuple(evaluations)
    aggregate = {
        mode: compute_mode_metrics([item.ranks[mode] for item in evaluation_tuple], top_k=top_k)
        for mode in EVALUATION_MODES
    }
    by_category: dict[str, dict[str, ModeMetrics]] = {}
    for category in BENCHMARK_CATEGORIES:
        subset = [item for item in evaluation_tuple if item.case.category.value == category]
        by_category[category] = {
            mode: compute_mode_metrics([item.ranks[mode] for item in subset], top_k=top_k)
            for mode in EVALUATION_MODES
        }

    pairwise = (
        pairwise_comparison("graph", "hybrid", evaluation_tuple),
        pairwise_comparison("reranked", "graph", evaluation_tuple),
    )

    return BenchmarkResult(
        benchmark_id=definition.benchmark_id,
        benchmark_version=definition.benchmark_version,
        benchmark_sha256=definition.source_sha256,
        query_count=len(definition.queries),
        top_k=top_k,
        provider_id=provider.provider_id,
        model_id=provider.model_id,
        python_version=platform.python_version(),
        platform=sys.platform,
        engine_version=__version__,
        corpus_fingerprint=corpus_fingerprint,
        aggregate=aggregate,
        by_category=by_category,
        pairwise=pairwise,
        queries=evaluation_tuple,
    )


def _metrics_dict(metrics: ModeMetrics) -> dict[str, float | int]:
    return {
        "hit_at_1": metrics.hit_at_1,
        "hit_at_5": metrics.hit_at_5,
        "hit_at_10": metrics.hit_at_10,
        "mrr_at_10": metrics.mrr_at_10,
        "query_count": metrics.query_count,
    }


def benchmark_result_to_dict(result: BenchmarkResult) -> dict[str, Any]:
    """Deterministic JSON-serializable mapping for a benchmark result."""
    return {
        "benchmark_id": result.benchmark_id,
        "benchmark_version": result.benchmark_version,
        "benchmark_sha256": result.benchmark_sha256,
        "query_count": result.query_count,
        "top_k": result.top_k,
        "provider_id": result.provider_id,
        "model_id": result.model_id,
        "python_version": result.python_version,
        "platform": result.platform,
        "engine_version": result.engine_version,
        "corpus_fingerprint": result.corpus_fingerprint,
        "aggregate": {mode: _metrics_dict(result.aggregate[mode]) for mode in EVALUATION_MODES},
        "by_category": {
            category: {
                mode: _metrics_dict(result.by_category[category][mode]) for mode in EVALUATION_MODES
            }
            for category in BENCHMARK_CATEGORIES
        },
        "pairwise": [asdict(item) for item in result.pairwise],
        "queries": [
            {
                "id": item.case.id,
                "category": item.case.category.value,
                "query": item.case.query,
                "relevant_qnames": list(item.case.relevant_qnames),
                "showcase": item.case.showcase,
                "notes": item.case.notes,
                "ranks": {mode: item.ranks[mode] for mode in EVALUATION_MODES},
                "hit_at_1": {mode: _hit_at(item.ranks[mode], 1) for mode in EVALUATION_MODES},
                "hit_at_5": {mode: _hit_at(item.ranks[mode], 5) for mode in EVALUATION_MODES},
                "hit_at_10": {mode: _hit_at(item.ranks[mode], 10) for mode in EVALUATION_MODES},
            }
            for item in result.queries
        ],
    }


def _hit_at(rank: int | None, threshold: int) -> bool:
    return rank is not None and rank <= threshold


def write_benchmark_json(result: BenchmarkResult, path: Path) -> None:
    """Write deterministic pretty-printed JSON result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = benchmark_result_to_dict(result)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _fmt_metric(value: float) -> str:
    return f"{value:.4f}"


def _metrics_row(mode: str, metrics: ModeMetrics) -> str:
    return (
        f"| {mode} | {_fmt_metric(metrics.hit_at_1)} | {_fmt_metric(metrics.hit_at_5)} | "
        f"{_fmt_metric(metrics.hit_at_10)} | {_fmt_metric(metrics.mrr_at_10)} |"
    )


def render_benchmark_markdown(result: BenchmarkResult) -> str:
    """Render recruiter-readable Markdown from a BenchmarkResult."""
    lines: list[str] = [
        f"# Benchmark Results: {result.benchmark_id}",
        "",
        "## Methodology",
        "",
        "- Language: Python only",
        f"- Queries: {result.query_count} frozen labeled cases (6 lexical / 6 behavioral / "
        "6 calls / 6 inheritance)",
        f"- Top-K: {result.top_k}",
        "- Metrics: Hit@1, Hit@5, Hit@10, MRR@10 (ranking only; scores not compared across modes)",
        "- Modes (fixed order): lexical, dense, hybrid, graph, reranked",
        "- No ranking-constant tuning on this evaluation set",
        f"- Benchmark SHA-256: `{result.benchmark_sha256}`",
        "",
        "## Setup",
        "",
        f"- Engine version: `{result.engine_version}`",
        f"- Python: `{result.python_version}`",
        f"- Platform: `{result.platform}`",
        f"- Provider: `{result.provider_id}`",
        f"- Model: `{result.model_id}`",
        f"- Corpus fingerprint: `{result.corpus_fingerprint}`",
        "",
        "## Aggregate metrics",
        "",
        "| Mode | Hit@1 | Hit@5 | Hit@10 | MRR@10 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in EVALUATION_MODES:
        lines.append(_metrics_row(mode, result.aggregate[mode]))

    lines.extend(["", "## Category breakdown", ""])
    for category in BENCHMARK_CATEGORIES:
        lines.extend(
            [
                f"### {category}",
                "",
                "| Mode | Hit@1 | Hit@5 | Hit@10 | MRR@10 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for mode in EVALUATION_MODES:
            lines.append(_metrics_row(mode, result.by_category[category][mode]))
        lines.append("")

    lines.extend(["## Pairwise comparisons", ""])
    for pair in result.pairwise:
        lines.append(
            f"- **{pair.left_mode} vs {pair.right_mode}**: "
            f"wins={pair.wins}, ties={pair.ties}, losses={pair.losses}"
        )
    lines.append("")
    lines.append(
        "Missing ranks (no relevant hit in top-10) are treated as worse than any rank 1..10; "
        "both missing counts as a tie."
    )
    lines.extend(
        [
            "",
            "## Per-query first-relevant ranks",
            "",
            "| ID | Category | Lexical | Dense | Hybrid | Graph | Reranked | Gold |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in result.queries:
        gold = ", ".join(f"`{name}`" for name in item.case.relevant_qnames)
        lines.append(
            "| {id} | {cat} | {lex} | {den} | {hyb} | {gra} | {rer} | {gold} |".format(
                id=item.case.id,
                cat=item.case.category.value,
                lex=_rank_cell(item.ranks["lexical"]),
                den=_rank_cell(item.ranks["dense"]),
                hyb=_rank_cell(item.ranks["hybrid"]),
                gra=_rank_cell(item.ranks["graph"]),
                rer=_rank_cell(item.ranks["reranked"]),
                gold=gold,
            )
        )

    showcases = [item for item in result.queries if item.case.showcase]
    lines.extend(["", "## Showcase cases (preselected before first real run)", ""])
    for item in showcases:
        lines.extend(
            [
                f"### `{item.case.id}` ({item.case.category.value})",
                "",
                f"- Query: {item.case.query}",
                f"- Gold: {', '.join(f'`{name}`' for name in item.case.relevant_qnames)}",
                f"- Hybrid rank: {_rank_cell(item.ranks['hybrid'])}",
                f"- Graph rank: {_rank_cell(item.ranks['graph'])}",
                f"- Reranked rank: {_rank_cell(item.ranks['reranked'])}",
                f"- Notes: {item.case.notes or '(none)'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Limitations",
            "",
            "- Small synthetic committed corpus; not a claim about all codebases.",
            "- Hand-authored queries and labels; author familiarity bias is possible.",
            "- MiniLM is a general text embedding model, not code-specialized.",
            "- Results are scoped to this benchmark only.",
            "",
        ]
    )
    return "\n".join(lines)


def _rank_cell(rank: int | None) -> str:
    return "—" if rank is None else str(rank)


def write_benchmark_markdown(result: BenchmarkResult, path: Path) -> None:
    """Write Markdown report derived from BenchmarkResult."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_benchmark_markdown(result), encoding="utf-8")
