"""Hybrid retrieval via Reciprocal Rank Fusion of lexical and dense rankings."""

from __future__ import annotations

from pathlib import Path

from codeintel.dense import search_dense
from codeintel.embeddings import EmbeddingProvider
from codeintel.lexical import search_code_units
from codeintel.models import SearchResult, SymbolKind
from codeintel.storage.database import IndexDatabase

RRF_K = 60


def hybrid_candidate_depth(limit: int) -> int:
    """Return per-retriever candidate depth for a final ``limit``."""
    if limit <= 0:
        raise ValueError("limit must be > 0")
    return max(50, 5 * limit)


def search_hybrid(
    database: IndexDatabase,
    provider: EmbeddingProvider,
    query: str,
    *,
    artifact_dir: Path,
    limit: int = 10,
    kind: SymbolKind | None = None,
    path_prefix: str | None = None,
) -> tuple[SearchResult, ...]:
    """Fuse lexical BM25 and dense cosine rankings with equal-weight RRF.

    Graph relationships are intentionally unused. ``SearchResult.score`` is the
    RRF score (higher is better). Scores are not comparable to pure lexical or
    dense scores.
    """
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if not query.strip():
        return ()

    depth = hybrid_candidate_depth(limit)
    lexical = search_code_units(
        database,
        query,
        limit=depth,
        kind=kind,
        path_prefix=path_prefix,
    )
    dense = search_dense(
        database,
        provider,
        query,
        artifact_dir=artifact_dir,
        limit=depth,
        kind=kind,
        path_prefix=path_prefix,
    )
    return fuse_rrf(lexical, dense, limit=limit)


def fuse_rrf(
    lexical: tuple[SearchResult, ...],
    dense: tuple[SearchResult, ...],
    *,
    limit: int,
    k: int = RRF_K,
) -> tuple[SearchResult, ...]:
    """Fuse two ranked lists with Reciprocal Rank Fusion.

    Ranks are 1-based. Missing documents contribute only from lists where they
    appear. Candidates are deduplicated by ``symbol_qualified_name``.
    """
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if k <= 0:
        raise ValueError("k must be > 0")

    scores: dict[str, float] = {}
    exemplars: dict[str, SearchResult] = {}

    for rank, result in enumerate(lexical, start=1):
        qname = result.symbol_qualified_name
        scores[qname] = scores.get(qname, 0.0) + 1.0 / (k + rank)
        exemplars.setdefault(qname, result)

    for rank, result in enumerate(dense, start=1):
        qname = result.symbol_qualified_name
        scores[qname] = scores.get(qname, 0.0) + 1.0 / (k + rank)
        exemplars.setdefault(qname, result)

    ordered = sorted(
        scores.items(),
        key=lambda item: (
            -item[1],
            item[0],
            exemplars[item[0]].path.as_posix(),
        ),
    )
    results: list[SearchResult] = []
    for qname, score in ordered[:limit]:
        base = exemplars[qname]
        results.append(
            SearchResult(
                symbol_qualified_name=base.symbol_qualified_name,
                kind=base.kind,
                path=base.path,
                span=base.span,
                signature=base.signature,
                source_text=base.source_text,
                score=score,
            )
        )
    return tuple(results)
