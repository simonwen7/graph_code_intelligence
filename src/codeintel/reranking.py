"""Structured deterministic reranking over Graph-Augmented candidates."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from codeintel.embeddings import EmbeddingProvider
from codeintel.graph import CodeGraph
from codeintel.graph_retrieval import GRAPH_SEED_COUNT, search_graph_augmented
from codeintel.hybrid import search_hybrid
from codeintel.models import (
    ContributionSource,
    EvidenceDirection,
    RankContribution,
    RelationEvidence,
    RelationKind,
    RerankedResult,
    RerankExplanation,
    ResolutionStatus,
    SearchResult,
    SymbolKind,
)
from codeintel.storage.database import IndexDatabase

RELATION_SUPPORT_RRF_K = 60
RERANK_RRF_K = 60

RERANK_RELATION_KINDS: tuple[RelationKind, ...] = (
    RelationKind.CALLS,
    RelationKind.REFERENCES,
    RelationKind.INHERITS,
    RelationKind.CONTAINS,
)

_RESOLVED_ONLY: tuple[ResolutionStatus, ...] = (ResolutionStatus.RESOLVED,)

_KIND_TO_SOURCE: dict[RelationKind, ContributionSource] = {
    RelationKind.CALLS: ContributionSource.CALLS,
    RelationKind.REFERENCES: ContributionSource.REFERENCES,
    RelationKind.INHERITS: ContributionSource.INHERITS,
    RelationKind.CONTAINS: ContributionSource.CONTAINS,
}


def rerank_candidate_depth(limit: int) -> int:
    """Return Graph candidate pool size for a requested final ``limit``."""
    if limit <= 0:
        raise ValueError("limit must be > 0")
    return max(20, 5 * limit)


def search_reranked(
    database: IndexDatabase,
    provider: EmbeddingProvider,
    query: str,
    *,
    artifact_dir: Path,
    limit: int = 10,
    kind: SymbolKind | None = None,
    path_prefix: str | None = None,
) -> tuple[RerankedResult, ...]:
    """Rerank Graph-Augmented candidates with relation-specific evidence RRF.

    Does not expand beyond the Graph candidate set. ``SearchResult.score`` is the
    final equal-weight RRF score (higher is better) and is not comparable across
    retrieval modes.
    """
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if not query.strip():
        return ()

    depth = rerank_candidate_depth(limit)
    graph_candidates = search_graph_augmented(
        database,
        provider,
        query,
        artifact_dir=artifact_dir,
        limit=depth,
        kind=kind,
        path_prefix=path_prefix,
    )
    if not graph_candidates:
        return ()

    original_ranks = {
        item.symbol_qualified_name: rank for rank, item in enumerate(graph_candidates, start=1)
    }
    candidate_qnames = frozenset(original_ranks)

    hybrid = search_hybrid(
        database,
        provider,
        query,
        artifact_dir=artifact_dir,
        limit=depth,
        kind=kind,
        path_prefix=path_prefix,
    )
    seeds = hybrid[: min(GRAPH_SEED_COUNT, len(hybrid))]
    code_graph = CodeGraph(database.load_symbols(), database.load_relations())

    evidence_by_candidate = _collect_relation_evidence(
        code_graph,
        seeds,
        candidate_qnames=candidate_qnames,
    )
    evidence_lists = _build_evidence_rankings(
        evidence_by_candidate,
        graph_candidates=graph_candidates,
    )

    ranked_lists: list[tuple[ContributionSource, tuple[SearchResult, ...]]] = [
        (ContributionSource.GRAPH_BASE, graph_candidates),
    ]
    for relation_kind in RERANK_RELATION_KINDS:
        ranked = evidence_lists.get(relation_kind, ())
        if ranked:
            ranked_lists.append((_KIND_TO_SOURCE[relation_kind], ranked))

    return _fuse_and_explain(
        ranked_lists,
        original_ranks=original_ranks,
        evidence_by_candidate=evidence_by_candidate,
        exemplars=graph_candidates,
        limit=limit,
        k=RERANK_RRF_K,
    )


def _collect_relation_evidence(
    graph: CodeGraph,
    seeds: tuple[SearchResult, ...],
    *,
    candidate_qnames: frozenset[str],
) -> dict[str, tuple[RelationEvidence, ...]]:
    collected: dict[str, dict[tuple[str, RelationKind, EvidenceDirection], RelationEvidence]] = (
        defaultdict(dict)
    )
    for seed_rank, seed in enumerate(seeds, start=1):
        seed_qname = seed.symbol_qualified_name
        if not graph.has_symbol(seed_qname):
            continue
        for candidate_qname in candidate_qnames:
            if candidate_qname == seed_qname:
                continue
            if not graph.has_symbol(candidate_qname):
                continue
            for evidence in _evidence_between(graph, seed_qname, seed_rank, candidate_qname):
                key = (
                    evidence.seed_qualified_name,
                    evidence.relation_kind,
                    evidence.direction,
                )
                collected[candidate_qname].setdefault(key, evidence)

    return {
        qname: tuple(
            sorted(
                items.values(),
                key=lambda item: (
                    item.seed_rank,
                    item.relation_kind.value,
                    item.direction.value,
                    item.seed_qualified_name,
                ),
            )
        )
        for qname, items in collected.items()
    }


def _evidence_between(
    graph: CodeGraph,
    seed_qname: str,
    seed_rank: int,
    candidate_qname: str,
) -> tuple[RelationEvidence, ...]:
    if seed_qname == candidate_qname:
        return ()
    records: list[RelationEvidence] = []
    outgoing = graph.outgoing(
        seed_qname,
        kinds=RERANK_RELATION_KINDS,
        resolutions=_RESOLVED_ONLY,
    )
    for relation in outgoing:
        if relation.target_qualified_name == candidate_qname:
            records.append(
                RelationEvidence(
                    seed_qualified_name=seed_qname,
                    seed_rank=seed_rank,
                    relation_kind=relation.kind,
                    direction=EvidenceDirection.SEED_TO_CANDIDATE,
                )
            )
    incoming = graph.incoming(
        seed_qname,
        kinds=RERANK_RELATION_KINDS,
        resolutions=_RESOLVED_ONLY,
    )
    for relation in incoming:
        if relation.source_qualified_name == candidate_qname:
            records.append(
                RelationEvidence(
                    seed_qualified_name=seed_qname,
                    seed_rank=seed_rank,
                    relation_kind=relation.kind,
                    direction=EvidenceDirection.CANDIDATE_TO_SEED,
                )
            )
    return tuple(records)


def relation_support(
    evidence: tuple[RelationEvidence, ...],
    *,
    relation_kind: RelationKind,
    k: int = RELATION_SUPPORT_RRF_K,
) -> float:
    """Compute unique-seed RelationSupport for one relation channel."""
    ranks = {item.seed_rank for item in evidence if item.relation_kind is relation_kind}
    return sum(1.0 / (k + rank) for rank in ranks)


def _build_evidence_rankings(
    evidence_by_candidate: dict[str, tuple[RelationEvidence, ...]],
    *,
    graph_candidates: tuple[SearchResult, ...],
) -> dict[RelationKind, tuple[SearchResult, ...]]:
    by_qname = {item.symbol_qualified_name: item for item in graph_candidates}
    rankings: dict[RelationKind, tuple[SearchResult, ...]] = {}
    for relation_kind in RERANK_RELATION_KINDS:
        scored: list[tuple[float, SearchResult]] = []
        for qname, evidence in evidence_by_candidate.items():
            support = relation_support(evidence, relation_kind=relation_kind)
            if support <= 0:
                continue
            base = by_qname.get(qname)
            if base is None:
                continue
            scored.append((support, base))
        scored.sort(
            key=lambda item: (
                -item[0],
                item[1].symbol_qualified_name,
                item[1].path.as_posix(),
            )
        )
        rankings[relation_kind] = tuple(result for _, result in scored)
    return rankings


def _fuse_and_explain(
    ranked_lists: list[tuple[ContributionSource, tuple[SearchResult, ...]]],
    *,
    original_ranks: dict[str, int],
    evidence_by_candidate: dict[str, tuple[RelationEvidence, ...]],
    exemplars: tuple[SearchResult, ...],
    limit: int,
    k: int,
) -> tuple[RerankedResult, ...]:
    scores: dict[str, float] = {}
    contributions: dict[str, list[RankContribution]] = defaultdict(list)
    by_qname = {item.symbol_qualified_name: item for item in exemplars}

    for source, ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            qname = result.symbol_qualified_name
            if qname not in original_ranks:
                continue
            contrib = 1.0 / (k + rank)
            scores[qname] = scores.get(qname, 0.0) + contrib
            contributions[qname].append(
                RankContribution(source=source, rank=rank, rrf_contribution=contrib)
            )

    ordered = sorted(
        scores.items(),
        key=lambda item: (
            -item[1],
            original_ranks[item[0]],
            item[0],
            by_qname[item[0]].path.as_posix(),
        ),
    )[:limit]

    results: list[RerankedResult] = []
    for final_rank, (qname, score) in enumerate(ordered, start=1):
        base = by_qname[qname]
        original_rank = original_ranks[qname]
        contrib_tuple = tuple(contributions[qname])
        has_structural = any(
            item.source is not ContributionSource.GRAPH_BASE for item in contrib_tuple
        )
        display_evidence = evidence_by_candidate.get(qname, ()) if has_structural else ()

        results.append(
            RerankedResult(
                result=SearchResult(
                    symbol_qualified_name=base.symbol_qualified_name,
                    kind=base.kind,
                    path=base.path,
                    span=base.span,
                    signature=base.signature,
                    source_text=base.source_text,
                    score=score,
                ),
                explanation=RerankExplanation(
                    original_rank=original_rank,
                    final_rank=final_rank,
                    rank_delta=original_rank - final_rank,
                    contributions=contrib_tuple,
                    relation_evidence=display_evidence,
                ),
            )
        )
    return tuple(results)
