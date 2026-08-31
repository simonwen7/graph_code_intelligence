"""Embedding providers for dense retrieval."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_TRANSFORMER_PROVIDER_ID = "sentence-transformers"


class EmbeddingDependencyError(RuntimeError):
    """Raised when the optional embeddings extra is not installed."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Language-neutral embedding interface.

    Providers return raw float32 vectors. L2 normalization is owned by the
    vector-index layer, not by providers.
    """

    @property
    def provider_id(self) -> str:
        """Stable provider identifier."""

    @property
    def model_id(self) -> str:
        """Model identifier used for embeddings."""

    @property
    def dimension(self) -> int:
        """Embedding dimensionality."""

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.floating]:
        """Embed documents into a float32 matrix of shape ``(N, dimension)``."""

    def embed_query(self, text: str) -> NDArray[np.floating]:
        """Embed a query into a float32 vector of shape ``(dimension,)``."""


class SentenceTransformerEmbeddingProvider:
    """Local Sentence Transformers embedding provider (CPU, optional extra).

    Requires the project ``embeddings`` optional dependency. Vectors are raw
    (not L2-normalized). The default MiniLM model may truncate long inputs
    according to its own maximum sequence length; M4 does not chunk CodeUnits.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID) -> None:
        # Set thread-env guards BEFORE importing Sentence Transformers / Torch.
        # setdefault preserves deliberate user overrides.
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("OMP_NUM_THREADS", "1")

        try:
            from sentence_transformers import (  # type: ignore[import-not-found,unused-ignore]
                SentenceTransformer,
            )
        except ImportError as exc:
            raise EmbeddingDependencyError(
                "Optional embedding dependency is not installed. "
                "Install/sync the project with the 'embeddings' extra, for example: "
                "`uv sync --extra embeddings`."
            ) from exc

        # CPU-only baseline for reproducibility. Do not enable trust_remote_code.
        self._model = SentenceTransformer(model_id, device="cpu")
        self._model_id = model_id
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        if callable(get_dim):
            self._dimension = int(get_dim())
        else:
            dimension = self._model.get_sentence_embedding_dimension()
            if dimension is None:
                raise EmbeddingDependencyError(
                    f"Embedding model {model_id!r} did not report a dimension"
                )
            self._dimension = int(dimension)

    @property
    def provider_id(self) -> str:
        return SENTENCE_TRANSFORMER_PROVIDER_ID

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.floating]:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        array = np.array(vectors, dtype=np.float32, copy=True)
        if array.ndim != 2 or array.shape[1] != self._dimension:
            raise ValueError(
                f"Unexpected document embedding shape {array.shape}; "
                f"expected (N, {self._dimension})"
            )
        return array

    def embed_query(self, text: str) -> NDArray[np.floating]:
        vector = self._model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        array = np.array(vector, dtype=np.float32, copy=True).reshape(-1)
        if array.shape != (self._dimension,):
            raise ValueError(
                f"Unexpected query embedding shape {array.shape}; expected ({self._dimension},)"
            )
        return array


def create_embedding_provider(model_id: str = DEFAULT_MODEL_ID) -> EmbeddingProvider:
    """Construct the real Sentence Transformers provider.

    Tests may monkeypatch this factory to inject a deterministic fake provider.
    """
    return SentenceTransformerEmbeddingProvider(model_id)
