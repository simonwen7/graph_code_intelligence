"""Exact FAISS vector index with L2-normalized inner-product (cosine) search."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray


class VectorIndexError(ValueError):
    """Raised for invalid vector-index operations."""


def l2_normalize(vectors: NDArray[np.floating]) -> NDArray[np.float32]:
    """Return L2-normalized float32 vectors.

    Accepts shape ``(dimension,)`` or ``(N, dimension)``. Zero-norm vectors raise
    :class:`VectorIndexError` instead of producing NaNs.
    """
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        norm = float(np.linalg.norm(array))
        if norm == 0.0 or not np.isfinite(norm):
            raise VectorIndexError("Cannot L2-normalize a zero or non-finite vector")
        return np.asarray(array / np.float32(norm), dtype=np.float32)

    if array.ndim != 2:
        raise VectorIndexError(f"Expected 1-D or 2-D vectors, got shape {array.shape}")

    if array.shape[0] == 0:
        return np.asarray(array, dtype=np.float32)

    norms = np.linalg.norm(array, axis=1)
    if not np.all(np.isfinite(norms)):
        raise VectorIndexError("Cannot L2-normalize vectors containing non-finite values")
    if np.any(norms == 0.0):
        raise VectorIndexError("Cannot L2-normalize a zero vector")
    return np.asarray(array / norms[:, np.newaxis], dtype=np.float32)


class FaissVectorIndex:
    """Exact dense retrieval index using FAISS ``IndexFlatIP`` on L2-normalized vectors.

    Cosine similarity is computed as inner product over normalized vectors.
    Higher scores are better. This wrapper does not know about SQLite, CodeUnits,
    CLI, hybrid fusion, or graphs.
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise VectorIndexError("dimension must be > 0")
        self._dimension = dimension
        self._index: faiss.Index = faiss.IndexFlatIP(dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    def add(self, vectors: NDArray[np.floating]) -> None:
        """L2-normalize and add document vectors."""
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2:
            raise VectorIndexError(f"Expected 2-D document matrix, got shape {array.shape}")
        if array.shape[1] != self._dimension:
            raise VectorIndexError(
                f"Document dimension mismatch: expected {self._dimension}, got {array.shape[1]}"
            )
        if not np.all(np.isfinite(array)):
            raise VectorIndexError("Document vectors must be finite")
        if array.shape[0] == 0:
            return
        normalized = l2_normalize(array)
        self._index.add(normalized)

    def search(
        self,
        query: NDArray[np.floating],
        *,
        limit: int,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Search with a raw query vector.

        Returns ``(scores, indices)`` each of shape ``(limit,)`` after L2-normalizing
        the query. Missing slots (limit > corpus) use FAISS sentinel ``-1`` indices.
        """
        if limit <= 0:
            raise VectorIndexError("limit must be > 0")
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if vector.shape != (self._dimension,):
            raise VectorIndexError(
                f"Query dimension mismatch: expected {self._dimension}, got {vector.shape[0]}"
            )
        if not np.all(np.isfinite(vector)):
            raise VectorIndexError("Query vector must be finite")

        if self.size == 0:
            scores = np.full(limit, -np.inf, dtype=np.float32)
            indices = np.full(limit, -1, dtype=np.int64)
            return scores, indices

        normalized = l2_normalize(vector).reshape(1, -1)
        k = min(limit, self.size)
        scores, indices = self._index.search(normalized, k)
        out_scores = np.full(limit, -np.inf, dtype=np.float32)
        out_indices = np.full(limit, -1, dtype=np.int64)
        out_scores[:k] = scores[0]
        out_indices[:k] = indices[0]
        return out_scores, out_indices

    def search_all(
        self,
        query: NDArray[np.floating],
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Return scores/indices for the entire corpus (exact Flat search)."""
        if self.size == 0:
            return (
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.int64),
            )
        return self.search(query, limit=self.size)

    def reconstruct(self, ordinal: int) -> NDArray[np.float32]:
        """Return the stored L2-normalized vector at ``ordinal``."""
        if ordinal < 0 or ordinal >= self.size:
            raise VectorIndexError(
                f"FAISS reconstruct ordinal {ordinal} out of range for size {self.size}"
            )
        vector = np.asarray(self._index.reconstruct(ordinal), dtype=np.float32)
        if vector.shape != (self._dimension,):
            raise VectorIndexError(
                f"Reconstructed vector has shape {vector.shape}, expected ({self._dimension},)"
            )
        if not np.all(np.isfinite(vector)):
            raise VectorIndexError("Reconstructed vector contains non-finite values")
        return vector

    def reconstruct_n(self, start: int, count: int) -> NDArray[np.float32]:
        """Return ``count`` consecutive reconstructed vectors starting at ``start``."""
        if start < 0 or count < 0 or start + count > self.size:
            raise VectorIndexError(
                f"FAISS reconstruct_n range [{start}, {start + count}) invalid for size {self.size}"
            )
        if count == 0:
            return np.zeros((0, self._dimension), dtype=np.float32)
        matrix = np.asarray(self._index.reconstruct_n(start, count), dtype=np.float32)
        if matrix.shape != (count, self._dimension):
            raise VectorIndexError(
                f"reconstruct_n returned shape {matrix.shape}, "
                f"expected ({count}, {self._dimension})"
            )
        if not np.all(np.isfinite(matrix)):
            raise VectorIndexError("Reconstructed vectors contain non-finite values")
        return matrix

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Path) -> FaissVectorIndex:
        if not path.is_file():
            raise VectorIndexError(f"FAISS index file does not exist: {path}")
        try:
            index = faiss.read_index(str(path))
        except Exception as exc:  # noqa: BLE001 - FAISS raises varied native errors
            raise VectorIndexError(f"Failed to load FAISS index: {path}") from exc
        if not hasattr(index, "d") or int(index.d) <= 0:
            raise VectorIndexError(f"Invalid FAISS index at {path}")
        metric_type = getattr(index, "metric_type", None)
        if metric_type is not None and int(metric_type) != int(faiss.METRIC_INNER_PRODUCT):
            raise VectorIndexError(
                f"Unsupported FAISS metric_type at {path}; expected IndexFlatIP "
                f"(inner product), got {metric_type}"
            )
        wrapper = cls.__new__(cls)
        wrapper._dimension = int(index.d)
        wrapper._index = index
        return wrapper
