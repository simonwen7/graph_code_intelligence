"""Graph-augmented hybrid retrieval over a persisted CodeGraph snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from codeintel.embeddings import EmbeddingProvider
from codeintel.graph import CodeGraph
from codeintel.hybrid import fuse_rrf, search_hybrid
from codeintel.models import (
    RelationKind,
    ResolutionStatus,
    SearchResult,
    SymbolKind,
)
from codeintel.storage.database import IndexDatabase, PersistedCodeUnitView

GRAPH_SEED_COUNT = 10
GRAPH_MAX_DEPTH = 1
GRAPH_RRF_K = 60
FINAL_GRAPH_RRF_K = 60

GRAPH_RELATION_KINDS: tuple[RelationKind, ...] = (
    RelationKind.CALLS,
    RelationKind.REFERENCES,
    RelationKind.INHERITS,
    RelationKind.CONTAINS,
)

_RESOLVED_ONLY: tuple[ResolutionStatus, ...] = (ResolutionStatus.RESOLVED,)


@dataclass
class _GraphCandidateSupport:
    """Private structural support accumulator (not a public API)."""

    qualified_name: str
    supporting_seed_ranks: set[int] = field(default_factory=set)

    def graph_support(self, *, k: int = GRAPH_RRF_K) -> float:
        return sum(1.0 / (k + rank) for rank in self.supporting_seed_ranks)


def graph_base_pool_limit(limit: int) -> int:
    """Return Hybrid pool size for a requested final ``limit``."""
    if limit <= 0:
        raise ValueError("limit must be > 0")
    return max(10, limit)


def search_graph_augmented(
    database: IndexDatabase,
    provider: EmbeddingProvider,
    query: str,
    *,
    artifact_dir: Path,
    limit: int = 10,
    kind: SymbolKind | None = None,
    path_prefix: str | None = None,
) -> tuple[SearchResult, ...]:
    """Graph-augmented Hybrid retrieval using direct RESOLVED structural neighbors.

    Reconstructs ``CodeGraph`` from the SQLite snapshot (no live reparse).
    ``SearchResult.score`` is the final Hybrid+Graph RRF score (higher is better)
    and is not comparable to pure lexical/dense/hybrid scores.
    """
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if not query.strip():
        return ()

    base_pool = graph_base_pool_limit(limit)
    hybrid = search_hybrid(
        database,
        provider,
        query,
        artifact_dir=artifact_dir,
        limit=base_pool,
        kind=kind,
        path_prefix=path_prefix,
    )
    if not hybrid:
        return ()

    seeds = hybrid[: min(GRAPH_SEED_COUNT, len(hybrid))]
    returnable = database.load_persisted_code_units()
    graph = CodeGraph(database.load_symbols(), database.load_relations())
    supports = _collect_graph_supports(
        graph,
        seeds,
        returnable=returnable,
        kind=kind,
        path_prefix=path_prefix,
    )
    graph_ranked = _structural_ranked_results(supports, returnable)
    return fuse_rrf(hybrid, graph_ranked, limit=limit, k=FINAL_GRAPH_RRF_K)


def _collect_graph_supports(
    graph: CodeGraph,
    seeds: tuple[SearchResult, ...],
    *,
    returnable: dict[str, PersistedCodeUnitView],
    kind: SymbolKind | None,
    path_prefix: str | None,
) -> dict[str, _GraphCandidateSupport]:
    supports: dict[str, _GraphCandidateSupport] = {}
    for seed_rank, seed in enumerate(seeds, start=1):
        seed_qname = seed.symbol_qualified_name
        if not graph.has_symbol(seed_qname):
            continue
        neighbors = _direct_neighbors(graph, seed_qname)
        for neighbor in neighbors:
            if neighbor == seed_qname:
                continue
            if not graph.has_symbol(neighbor):
                continue
            symbol = graph.get_symbol(neighbor)
            if symbol.kind is SymbolKind.MODULE or symbol.kind is SymbolKind.NAMESPACE:
                continue
            unit = returnable.get(neighbor)
            if unit is None:
                continue
            if kind is not None and unit.kind is not kind:
                continue
            if path_prefix is not None and not _path_matches_prefix(
                unit.path.as_posix(),
                path_prefix,
            ):
                continue
            entry = supports.get(neighbor)
            if entry is None:
                entry = _GraphCandidateSupport(qualified_name=neighbor)
                supports[neighbor] = entry
            entry.supporting_seed_ranks.add(seed_rank)
    return supports


def _direct_neighbors(graph: CodeGraph, qualified_name: str) -> set[str]:
    """Return direct BOTH-direction neighbors for included RESOLVED relation kinds."""
    names: set[str] = set()
    outgoing = graph.outgoing(
        qualified_name,
        kinds=GRAPH_RELATION_KINDS,
        resolutions=_RESOLVED_ONLY,
    )
    incoming = graph.incoming(
        qualified_name,
        kinds=GRAPH_RELATION_KINDS,
        resolutions=_RESOLVED_ONLY,
    )
    for relation in (*outgoing, *incoming):
        if relation.source_qualified_name == qualified_name:
            target = relation.target_qualified_name
            if target is not None:
                names.add(target)
        elif relation.target_qualified_name == qualified_name:
            names.add(relation.source_qualified_name)
    return names


def _structural_ranked_results(
    supports: dict[str, _GraphCandidateSupport],
    returnable: dict[str, PersistedCodeUnitView],
) -> tuple[SearchResult, ...]:
    ranked = sorted(
        supports.values(),
        key=lambda item: (
            -item.graph_support(),
            item.qualified_name,
            returnable[item.qualified_name].path.as_posix(),
        ),
    )
    results: list[SearchResult] = []
    for item in ranked:
        unit = returnable[item.qualified_name]
        results.append(
            SearchResult(
                symbol_qualified_name=unit.symbol_qualified_name,
                kind=unit.kind,
                path=unit.path,
                span=unit.span,
                signature=unit.signature,
                source_text=unit.source_text,
                score=item.graph_support(),
            )
        )
    return tuple(results)


def _path_matches_prefix(path: str, path_prefix: str) -> bool:
    normalized_path = path.replace("\\", "/")
    normalized_prefix = path_prefix.replace("\\", "/")
    return normalized_path.startswith(normalized_prefix)
