"""EmbeddingProvider contract tests using the offline fake provider."""

from __future__ import annotations

import numpy as np
import pytest
from helpers.fake_embeddings import FakeEmbeddingProvider


def test_fake_provider_shapes_and_identity() -> None:
    provider = FakeEmbeddingProvider(dimension=4, model_id="fake-v1")
    assert provider.provider_id == "fake"
    assert provider.model_id == "fake-v1"
    assert provider.dimension == 4

    docs = provider.embed_documents(["a", "b"])
    assert docs.shape == (2, 4)
    assert docs.dtype == np.float32
    query = provider.embed_query("a")
    assert query.shape == (4,)
    assert query.dtype == np.float32


def test_fake_provider_deterministic_and_rejects_bad_vectors() -> None:
    provider = FakeEmbeddingProvider(
        dimension=3,
        document_vectors={"doc": [1.0, 0.0, 0.0]},
        query_vectors={"q": [0.0, 1.0, 0.0]},
    )
    first = provider.embed_documents(["doc"])
    second = provider.embed_documents(["doc"])
    assert np.array_equal(first, second)
    assert np.array_equal(provider.embed_query("q"), np.asarray([0.0, 1.0, 0.0], dtype=np.float32))

    with pytest.raises(ValueError, match="length"):
        FakeEmbeddingProvider(dimension=2, document_vectors={"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="finite"):
        FakeEmbeddingProvider(dimension=2, query_vectors={"x": [1.0, float("nan")]})


def test_empty_document_batch_shape() -> None:
    provider = FakeEmbeddingProvider(dimension=5)
    empty = provider.embed_documents([])
    assert empty.shape == (0, 5)


def test_env_guards_precede_sentence_transformers_import() -> None:
    import ast
    from pathlib import Path

    source = Path("src/codeintel/embeddings.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    init = next(
        item
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SentenceTransformerEmbeddingProvider"
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "__init__"
    )
    texts = [ast.unparse(stmt) for stmt in init.body]
    guard_indexes = [
        index
        for index, text in enumerate(texts)
        if "TOKENIZERS_PARALLELISM" in text or "OMP_NUM_THREADS" in text
    ]
    import_indexes = [index for index, text in enumerate(texts) if "sentence_transformers" in text]
    assert guard_indexes
    assert import_indexes
    assert max(guard_indexes) < min(import_indexes)
