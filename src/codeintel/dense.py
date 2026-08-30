"""Dense retrieval over persisted CodeUnits using FAISS."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from codeintel.embeddings import EmbeddingProvider
from codeintel.models import SearchResult, SourceSpan, SymbolKind
from codeintel.storage.database import IndexDatabase
from codeintel.storage.schema import DEFAULT_INDEX_DIRNAME
from codeintel.vector_index import FaissVectorIndex, VectorIndexError, l2_normalize

ARTIFACT_VERSION = 1
DOCUMENT_TEXT_FORMAT_VERSION = 1
DENSE_METRIC = "cosine_ip"
DENSE_DIRNAME = "dense"
FAISS_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"


class DenseIndexError(RuntimeError):
    """Base error for dense index operations."""


class DenseIndexMissingError(DenseIndexError):
    """Dense artifact files are missing."""


class DenseIndexMismatchError(DenseIndexError):
    """Dense artifact is incompatible with the current index or provider."""


@dataclass(frozen=True, slots=True)
class DenseIndexStats:
    """Summary of a dense index build."""

    document_count: int
    dimension: int
    provider_id: str
    model_id: str
    artifact_dir: Path
    corpus_fingerprint: str
    vectors_reused: int
    vectors_embedded: int
    rewritten: bool


@dataclass(frozen=True, slots=True)
class _DenseDocument:
    qualified_name: str
    kind: SymbolKind
    path: str
    signature: str | None
    source_text: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int

    @property
    def document_text(self) -> str:
        return format_dense_document(
            self.qualified_name,
            self.signature,
            self.source_text,
        )


def default_dense_dir(repository_root: Path) -> Path:
    """Return the default dense artifact directory for a repository root."""
    return repository_root / DEFAULT_INDEX_DIRNAME / DENSE_DIRNAME


def format_dense_document(
    qualified_name: str,
    signature: str | None,
    source_text: str,
) -> str:
    """Build the frozen M4 dense document text for one CodeUnit."""
    signature_value = signature if signature is not None else ""
    return f"symbol: {qualified_name}\nsignature: {signature_value}\ncode:\n{source_text}"


def dense_document_fingerprint(document_text: str) -> str:
    """SHA-256 of UTF-8 dense document text sent to the embedding provider."""
    return hashlib.sha256(document_text.encode("utf-8")).hexdigest()


def compute_corpus_fingerprint(documents: list[_DenseDocument]) -> str:
    """Compute a deterministic SHA-256 fingerprint over ordered dense inputs."""
    digest = hashlib.sha256()
    for document in documents:
        signature = document.signature if document.signature is not None else ""
        for field in (
            document.qualified_name,
            document.kind.value,
            document.path,
            signature,
            document.source_text,
        ):
            encoded = field.encode("utf-8")
            digest.update(f"{len(encoded)}:".encode("ascii"))
            digest.update(encoded)
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()


def load_dense_documents(database: IndexDatabase) -> list[_DenseDocument]:
    """Load CodeUnits for dense indexing in qualified_name ascending order."""
    try:
        rows = database.connection.execute(
            """
            SELECT
                s.qualified_name AS qualified_name,
                s.kind AS kind,
                f.path AS path,
                s.signature AS signature,
                c.source_text AS source_text,
                c.start_line AS start_line,
                c.end_line AS end_line,
                c.start_byte AS start_byte,
                c.end_byte AS end_byte
            FROM code_units AS c
            JOIN symbols AS s ON s.id = c.symbol_id
            JOIN files AS f ON f.id = s.file_id
            ORDER BY s.qualified_name ASC
            """
        ).fetchall()
    except sqlite3.Error as exc:
        raise DenseIndexError("Failed to load CodeUnits for dense indexing") from exc

    documents: list[_DenseDocument] = []
    for row in rows:
        signature = row["signature"]
        documents.append(
            _DenseDocument(
                qualified_name=str(row["qualified_name"]),
                kind=SymbolKind(str(row["kind"])),
                path=str(row["path"]),
                signature=None if signature is None else str(signature),
                source_text=str(row["source_text"]),
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                start_byte=int(row["start_byte"]),
                end_byte=int(row["end_byte"]),
            )
        )
    return documents


def build_dense_index(
    database: IndexDatabase,
    provider: EmbeddingProvider,
    *,
    artifact_dir: Path,
    full: bool = False,
) -> DenseIndexStats:
    """Build a dense FAISS artifact from the persisted SQLite CodeUnit snapshot."""
    documents = load_dense_documents(database)
    texts = [document.document_text for document in documents]
    fingerprints = [dense_document_fingerprint(text) for text in texts]
    qualified_names = [document.qualified_name for document in documents]
    if len(set(qualified_names)) != len(qualified_names):
        raise DenseIndexError("Dense corpus contains duplicate qualified_names")
    fingerprint = compute_corpus_fingerprint(documents)

    reuse_source = None if full else _load_reuse_source(artifact_dir, provider)
    if (
        reuse_source is not None
        and reuse_source.corpus_fingerprint == fingerprint
        and reuse_source.document_fingerprints == fingerprints
        and reuse_source.qualified_names == qualified_names
    ):
        return DenseIndexStats(
            document_count=len(documents),
            dimension=provider.dimension,
            provider_id=provider.provider_id,
            model_id=provider.model_id,
            artifact_dir=artifact_dir,
            corpus_fingerprint=fingerprint,
            vectors_reused=len(documents),
            vectors_embedded=0,
            rewritten=False,
        )

    vectors = np.zeros((len(documents), provider.dimension), dtype=np.float32)
    reuse_mask = [False] * len(documents)
    embed_indices: list[int] = []
    embed_texts: list[str] = []

    old_by_qname: dict[str, tuple[str, NDArray[np.floating]]] = {}
    if reuse_source is not None:
        for ordinal, qname in enumerate(reuse_source.qualified_names):
            old_by_qname[qname] = (
                reuse_source.document_fingerprints[ordinal],
                reuse_source.vectors[ordinal],
            )

    for doc_index, (qname, text, doc_fp) in enumerate(
        zip(qualified_names, texts, fingerprints, strict=True)
    ):
        previous = old_by_qname.get(qname)
        if previous is not None and previous[0] == doc_fp:
            vector = np.asarray(previous[1], dtype=np.float32)
            if vector.shape != (provider.dimension,) or not np.all(np.isfinite(vector)):
                embed_indices.append(doc_index)
                embed_texts.append(text)
                continue
            vectors[doc_index] = vector
            reuse_mask[doc_index] = True
        else:
            embed_indices.append(doc_index)
            embed_texts.append(text)

    if embed_texts:
        embedded = provider.embed_documents(embed_texts)
        embedded = np.asarray(embedded, dtype=np.float32)
        if embedded.ndim != 2 or embedded.shape[0] != len(embed_texts):
            raise DenseIndexError(
                f"Embedding provider returned unexpected shape {embedded.shape} "
                f"for {len(embed_texts)} documents"
            )
        if embedded.shape[1] != provider.dimension:
            raise DenseIndexError(
                f"Embedding dimension mismatch: provider reports {provider.dimension}, "
                f"got {embedded.shape[1]}"
            )
        if not np.all(np.isfinite(embedded)):
            raise DenseIndexError("Embedding provider returned non-finite vectors")
        normalized_new = l2_normalize(embedded) if embedded.shape[0] else embedded
        for offset, doc_index in enumerate(embed_indices):
            vectors[doc_index] = normalized_new[offset]

    reused_count = sum(1 for flag in reuse_mask if flag)
    embedded_count = len(embed_indices)

    try:
        index = FaissVectorIndex(provider.dimension)
        if len(documents):
            # Vectors are already L2-normalized; add() re-normalizes idempotently.
            index.add(vectors)
    except VectorIndexError as exc:
        raise DenseIndexError(str(exc)) from exc

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "provider_id": provider.provider_id,
        "model_id": provider.model_id,
        "dimension": provider.dimension,
        "metric": DENSE_METRIC,
        "normalized": True,
        "document_count": len(documents),
        "document_text_format_version": DOCUMENT_TEXT_FORMAT_VERSION,
        "corpus_fingerprint": fingerprint,
        "qualified_names": qualified_names,
        "document_fingerprints": fingerprints,
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = artifact_dir / FAISS_FILENAME
    metadata_path = artifact_dir / METADATA_FILENAME

    with tempfile.TemporaryDirectory(dir=artifact_dir) as tmp_name:
        tmp_dir = Path(tmp_name)
        tmp_faiss = tmp_dir / FAISS_FILENAME
        tmp_metadata = tmp_dir / METADATA_FILENAME
        try:
            index.save(tmp_faiss)
            tmp_metadata.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _validate_written_pair(tmp_faiss, tmp_metadata, provider, fingerprint)
            tmp_faiss.replace(faiss_path)
            tmp_metadata.replace(metadata_path)
        except Exception:
            raise

    return DenseIndexStats(
        document_count=len(documents),
        dimension=provider.dimension,
        provider_id=provider.provider_id,
        model_id=provider.model_id,
        artifact_dir=artifact_dir,
        corpus_fingerprint=fingerprint,
        vectors_reused=reused_count,
        vectors_embedded=embedded_count,
        rewritten=True,
    )


@dataclass(frozen=True, slots=True)
class _ReuseSource:
    qualified_names: list[str]
    document_fingerprints: list[str]
    vectors: NDArray[np.floating]
    corpus_fingerprint: str


def _load_reuse_source(
    artifact_dir: Path,
    provider: EmbeddingProvider,
) -> _ReuseSource | None:
    """Load an old artifact for vector reuse without requiring SQLite fingerprint match.

    Returns ``None`` when no reusable fingerprint metadata is available (legacy artifact).
    Raises ``DenseIndexError`` when an artifact exists but is internally inconsistent.
    """
    faiss_path = artifact_dir / FAISS_FILENAME
    metadata_path = artifact_dir / METADATA_FILENAME
    if not metadata_path.is_file() or not faiss_path.is_file():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DenseIndexError(
            f"Corrupt dense metadata at {metadata_path}. Rebuild with `aicode embed PATH --full`."
        ) from exc
    if not isinstance(metadata, dict):
        raise DenseIndexError(
            f"Corrupt dense metadata at {metadata_path}. Rebuild with `aicode embed PATH --full`."
        )

    try:
        _require_metadata_fields(metadata)
        if int(metadata["artifact_version"]) != ARTIFACT_VERSION:
            raise DenseIndexError("unsupported artifact_version")
        if int(metadata["document_text_format_version"]) != DOCUMENT_TEXT_FORMAT_VERSION:
            raise DenseIndexError("unsupported document_text_format_version")
        if metadata["metric"] != DENSE_METRIC:
            raise DenseIndexError("metric mismatch")
        if metadata["normalized"] is not True:
            raise DenseIndexError("normalization mismatch")
        if metadata["provider_id"] != provider.provider_id:
            return None
        if metadata["model_id"] != provider.model_id:
            return None
        if int(metadata["dimension"]) != provider.dimension:
            return None
        qualified_names = metadata["qualified_names"]
        if not isinstance(qualified_names, list) or not all(
            isinstance(name, str) for name in qualified_names
        ):
            raise DenseIndexError("qualified_names invalid")
        if len(qualified_names) != int(metadata["document_count"]):
            raise DenseIndexError("qualified_names length mismatch")
        if len(set(qualified_names)) != len(qualified_names):
            raise DenseIndexError("duplicate qualified_names")
        fingerprints = metadata.get("document_fingerprints")
        if fingerprints is None:
            return None
        if not isinstance(fingerprints, list) or not all(
            isinstance(item, str) for item in fingerprints
        ):
            raise DenseIndexError("document_fingerprints invalid")
        if len(fingerprints) != len(qualified_names):
            raise DenseIndexError("document_fingerprints length mismatch")
        index = FaissVectorIndex.load(faiss_path)
        if index.dimension != int(metadata["dimension"]):
            raise DenseIndexError("FAISS dimension mismatch")
        if index.size != int(metadata["document_count"]):
            raise DenseIndexError("FAISS size mismatch")
        if index.size == 0:
            vectors = np.zeros((0, provider.dimension), dtype=np.float32)
        else:
            vectors = index.reconstruct_n(0, index.size)
    except DenseIndexError:
        raise
    except Exception as exc:
        raise DenseIndexError(
            f"Corrupt dense reuse artifact under {artifact_dir}. "
            "Rebuild with `aicode embed PATH --full`."
        ) from exc

    return _ReuseSource(
        qualified_names=list(qualified_names),
        document_fingerprints=list(fingerprints),
        vectors=vectors,
        corpus_fingerprint=str(metadata["corpus_fingerprint"]),
    )


def search_dense(
    database: IndexDatabase,
    provider: EmbeddingProvider,
    query: str,
    *,
    artifact_dir: Path,
    limit: int = 10,
    kind: SymbolKind | None = None,
    path_prefix: str | None = None,
) -> tuple[SearchResult, ...]:
    """Search the dense artifact with cosine similarity (higher score is better)."""
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if not query.strip():
        return ()

    metadata, index = load_and_validate_dense_artifact(
        database,
        provider,
        artifact_dir=artifact_dir,
    )
    documents = load_dense_documents(database)
    by_qname = {document.qualified_name: document for document in documents}

    try:
        query_vector = provider.embed_query(query)
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        scores, indices = index.search_all(query_vector)
    except VectorIndexError as exc:
        raise DenseIndexError(str(exc)) from exc
    except Exception as exc:
        raise DenseIndexError(f"Dense query embedding failed: {exc}") from exc

    qualified_names: list[str] = list(metadata["qualified_names"])
    candidates: list[tuple[float, _DenseDocument]] = []
    for score, ordinal in zip(scores.tolist(), indices.tolist(), strict=True):
        if ordinal < 0:
            continue
        if ordinal >= len(qualified_names):
            raise DenseIndexMismatchError(
                f"FAISS returned out-of-range ordinal {ordinal} for "
                f"{len(qualified_names)} documents"
            )
        qname = qualified_names[ordinal]
        document = by_qname.get(qname)
        if document is None:
            raise DenseIndexMismatchError(
                f"Dense artifact qname {qname!r} is missing from the SQLite snapshot"
            )
        if kind is not None and document.kind is not kind:
            continue
        if path_prefix is not None and not _path_matches_prefix(document.path, path_prefix):
            continue
        candidates.append((float(score), document))

    candidates.sort(key=lambda item: (-item[0], item[1].qualified_name, item[1].path))
    selected = candidates[:limit]
    return tuple(_to_search_result(score, document) for score, document in selected)


def load_and_validate_dense_artifact(
    database: IndexDatabase,
    provider: EmbeddingProvider,
    *,
    artifact_dir: Path,
) -> tuple[dict[str, Any], FaissVectorIndex]:
    """Load dense metadata/FAISS and validate against DB + provider."""
    faiss_path = artifact_dir / FAISS_FILENAME
    metadata_path = artifact_dir / METADATA_FILENAME
    if not metadata_path.is_file() or not faiss_path.is_file():
        raise DenseIndexMissingError(
            f"Dense artifact is missing under {artifact_dir}. Run `aicode embed` to build it."
        )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DenseIndexError(f"Corrupt dense metadata at {metadata_path}") from exc

    if not isinstance(metadata, dict):
        raise DenseIndexError(f"Corrupt dense metadata at {metadata_path}")

    _require_metadata_fields(metadata)

    if int(metadata["artifact_version"]) != ARTIFACT_VERSION:
        raise DenseIndexMismatchError(
            f"Unsupported dense artifact_version {metadata['artifact_version']}; "
            f"expected {ARTIFACT_VERSION}. Rebuild with `aicode embed`."
        )
    if int(metadata["document_text_format_version"]) != DOCUMENT_TEXT_FORMAT_VERSION:
        raise DenseIndexMismatchError(
            f"Unsupported dense document_text_format_version "
            f"{metadata['document_text_format_version']}; "
            f"expected {DOCUMENT_TEXT_FORMAT_VERSION}. Rebuild with `aicode embed`."
        )
    if metadata["metric"] != DENSE_METRIC:
        raise DenseIndexMismatchError(
            f"Dense artifact metric mismatch: {metadata['metric']!r} != {DENSE_METRIC!r}"
        )
    if metadata["normalized"] is not True:
        raise DenseIndexMismatchError("Dense artifact is not marked as L2-normalized")
    if not isinstance(metadata["dimension"], int) or int(metadata["dimension"]) <= 0:
        raise DenseIndexError("Dense metadata dimension must be a positive integer")
    if not isinstance(metadata["document_count"], int) or int(metadata["document_count"]) < 0:
        raise DenseIndexError("Dense metadata document_count must be a non-negative integer")
    if metadata["provider_id"] != provider.provider_id:
        raise DenseIndexMismatchError(
            f"Dense provider mismatch: artifact {metadata['provider_id']!r}, "
            f"requested {provider.provider_id!r}. Rebuild with `aicode embed`."
        )
    if metadata["model_id"] != provider.model_id:
        raise DenseIndexMismatchError(
            f"Dense model mismatch: artifact {metadata['model_id']!r}, "
            f"requested {provider.model_id!r}. Rebuild with `aicode embed`."
        )
    if int(metadata["dimension"]) != provider.dimension:
        raise DenseIndexMismatchError(
            f"Dense dimension mismatch: artifact {metadata['dimension']}, "
            f"provider {provider.dimension}. Rebuild with `aicode embed`."
        )

    documents = load_dense_documents(database)
    expected_fingerprint = compute_corpus_fingerprint(documents)
    if metadata["corpus_fingerprint"] != expected_fingerprint:
        raise DenseIndexMismatchError(
            "Dense artifact is stale relative to the SQLite index "
            "(corpus fingerprint mismatch). Rebuild with `aicode embed`."
        )
    if int(metadata["document_count"]) != len(documents):
        raise DenseIndexMismatchError(
            f"Dense document_count mismatch: artifact {metadata['document_count']}, "
            f"SQLite {len(documents)}. Rebuild with `aicode embed`."
        )

    qualified_names = metadata["qualified_names"]
    if not isinstance(qualified_names, list) or not all(
        isinstance(name, str) for name in qualified_names
    ):
        raise DenseIndexError("Dense metadata qualified_names must be a list of strings")
    if len(qualified_names) != int(metadata["document_count"]):
        raise DenseIndexMismatchError(
            "Dense metadata qualified_names length does not match document_count"
        )
    if len(set(qualified_names)) != len(qualified_names):
        raise DenseIndexError("Dense metadata qualified_names must not contain duplicates")
    expected_names = [document.qualified_name for document in documents]
    if qualified_names != expected_names:
        raise DenseIndexMismatchError(
            "Dense qualified_names mapping does not match the SQLite snapshot. "
            "Rebuild with `aicode embed`."
        )

    try:
        index = FaissVectorIndex.load(faiss_path)
    except VectorIndexError as exc:
        raise DenseIndexError(str(exc)) from exc

    if index.dimension != int(metadata["dimension"]):
        raise DenseIndexMismatchError(
            f"FAISS dimension {index.dimension} does not match metadata {metadata['dimension']}"
        )
    if index.size != int(metadata["document_count"]):
        raise DenseIndexMismatchError(
            f"FAISS size {index.size} does not match metadata document_count "
            f"{metadata['document_count']}"
        )
    return metadata, index


def _validate_written_pair(
    faiss_path: Path,
    metadata_path: Path,
    provider: EmbeddingProvider,
    fingerprint: str,
) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["corpus_fingerprint"] != fingerprint:
        raise DenseIndexError("Internal error: fingerprint mismatch while writing artifact")
    if metadata["provider_id"] != provider.provider_id:
        raise DenseIndexError("Internal error: provider_id mismatch while writing artifact")
    index = FaissVectorIndex.load(faiss_path)
    if index.dimension != provider.dimension:
        raise DenseIndexError("Internal error: FAISS dimension mismatch while writing artifact")
    if index.size != int(metadata["document_count"]):
        raise DenseIndexError("Internal error: FAISS size mismatch while writing artifact")


def _require_metadata_fields(metadata: dict[str, Any]) -> None:
    required = {
        "artifact_version",
        "provider_id",
        "model_id",
        "dimension",
        "metric",
        "normalized",
        "document_count",
        "document_text_format_version",
        "corpus_fingerprint",
        "qualified_names",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise DenseIndexError(f"Dense metadata missing fields: {', '.join(missing)}")


def _path_matches_prefix(path: str, path_prefix: str) -> bool:
    normalized_path = path.replace("\\", "/")
    normalized_prefix = path_prefix.replace("\\", "/")
    return normalized_path.startswith(normalized_prefix)


def _to_search_result(score: float, document: _DenseDocument) -> SearchResult:
    return SearchResult(
        symbol_qualified_name=document.qualified_name,
        kind=document.kind,
        path=Path(document.path),
        span=SourceSpan(
            start_line=document.start_line,
            end_line=document.end_line,
            start_byte=document.start_byte,
            end_byte=document.end_byte,
        ),
        signature=document.signature,
        source_text=document.source_text,
        score=score,
    )
