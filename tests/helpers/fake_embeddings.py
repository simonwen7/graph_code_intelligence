"""Deterministic fake embedding provider for offline tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


class FakeEmbeddingProvider:
    """Deterministic EmbeddingProvider used only in tests."""

    def __init__(
        self,
        *,
        dimension: int = 8,
        provider_id: str = "fake",
        model_id: str = "fake-model",
        document_vectors: Mapping[str, Sequence[float]] | None = None,
        query_vectors: Mapping[str, Sequence[float]] | None = None,
        default_document: Sequence[float] | None = None,
        default_query: Sequence[float] | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be > 0")
        self._dimension = dimension
        self._provider_id = provider_id
        self._model_id = model_id
        self._document_vectors = {
            key: self._as_vector(value) for key, value in (document_vectors or {}).items()
        }
        self._query_vectors = {
            key: self._as_vector(value) for key, value in (query_vectors or {}).items()
        }
        self._default_document = (
            self._as_vector(default_document) if default_document is not None else self._unit(0)
        )
        self._default_query = (
            self._as_vector(default_query) if default_query is not None else self._unit(0)
        )
        self.document_embed_calls = 0
        self.documents_embedded = 0

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.floating]:
        self.document_embed_calls += 1
        self.documents_embedded += len(texts)
        rows = [self._document_vectors.get(text, self._default_document) for text in texts]
        if not rows:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.stack(rows, axis=0)

    def embed_query(self, text: str) -> NDArray[np.floating]:
        return self._query_vectors.get(text, self._default_query).copy()

    def _as_vector(self, values: Sequence[float]) -> NDArray[np.float32]:
        array = np.asarray(list(values), dtype=np.float32)
        if array.shape != (self._dimension,):
            raise ValueError(
                f"Expected vector of length {self._dimension}, got shape {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("Vectors must be finite")
        return array

    def _unit(self, axis: int) -> NDArray[np.float32]:
        vector = np.zeros(self._dimension, dtype=np.float32)
        vector[axis % self._dimension] = 1.0
        return vector
