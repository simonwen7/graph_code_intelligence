"""Unit tests for FaissVectorIndex normalization and exact search."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from codeintel.vector_index import FaissVectorIndex, VectorIndexError, l2_normalize


def test_l2_normalize_rejects_zero_vector() -> None:
    with pytest.raises(VectorIndexError, match="zero"):
        l2_normalize(np.zeros(4, dtype=np.float32))


def test_l2_normalize_matrix_and_query() -> None:
    matrix = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    normalized = l2_normalize(matrix)
    assert normalized.dtype == np.float32
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
    query = l2_normalize(np.asarray([3.0, 4.0], dtype=np.float32))
    assert np.allclose(query, np.asarray([0.6, 0.8], dtype=np.float32))


def test_build_search_higher_is_better_and_cosine(tmp_path: Path) -> None:
    index = FaissVectorIndex(2)
    docs = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    index.add(docs)
    assert index.size == 3

    scores, indices = index.search(np.asarray([1.0, 0.0], dtype=np.float32), limit=3)
    assert indices[0] == 0
    assert scores[0] > scores[1]
    assert scores[0] == pytest.approx(1.0, abs=1e-5)

    path = tmp_path / "index.faiss"
    index.save(path)
    loaded = FaissVectorIndex.load(path)
    assert loaded.dimension == 2
    assert loaded.size == 3
    loaded_scores, loaded_indices = loaded.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        limit=2,
    )
    assert loaded_indices.tolist() == indices[:2].tolist()
    assert np.allclose(loaded_scores[:2], scores[:2])


def test_dimension_mismatch_and_limit_gt_corpus() -> None:
    index = FaissVectorIndex(3)
    index.add(np.eye(3, dtype=np.float32))
    with pytest.raises(VectorIndexError, match="dimension"):
        index.add(np.eye(2, dtype=np.float32))
    with pytest.raises(VectorIndexError, match="dimension"):
        index.search(np.ones(2, dtype=np.float32), limit=1)

    scores, indices = index.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), limit=10)
    assert scores.shape == (10,)
    assert indices.shape == (10,)
    assert list(indices[:3]) == [0, 1, 2] or indices[0] == 0
    assert indices[3] == -1


def test_empty_index_search_and_save_load(tmp_path: Path) -> None:
    index = FaissVectorIndex(4)
    assert index.size == 0
    scores, indices = index.search(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32), limit=3)
    assert indices.tolist() == [-1, -1, -1]
    path = tmp_path / "empty.faiss"
    index.save(path)
    loaded = FaissVectorIndex.load(path)
    assert loaded.size == 0


def test_add_rejects_zero_vector() -> None:
    index = FaissVectorIndex(2)
    with pytest.raises(VectorIndexError, match="zero"):
        index.add(np.asarray([[0.0, 0.0]], dtype=np.float32))


def test_add_and_search_reject_nonfinite_vectors() -> None:
    index = FaissVectorIndex(2)
    with pytest.raises(VectorIndexError, match="finite"):
        index.add(np.asarray([[1.0, float("nan")]], dtype=np.float32))
    with pytest.raises(VectorIndexError, match="finite"):
        index.add(np.asarray([[1.0, float("inf")]], dtype=np.float32))
    index.add(np.asarray([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(VectorIndexError, match="finite"):
        index.search(np.asarray([float("nan"), 0.0], dtype=np.float32), limit=1)
    with pytest.raises(VectorIndexError, match="zero|non-finite|finite"):
        index.search(np.zeros(2, dtype=np.float32), limit=1)


def test_load_rejects_l2_metric_index(tmp_path: Path) -> None:
    import faiss

    path = tmp_path / "l2.faiss"
    index = faiss.IndexFlatL2(2)
    index.add(np.asarray([[1.0, 0.0]], dtype=np.float32))
    faiss.write_index(index, str(path))
    with pytest.raises(VectorIndexError, match="metric_type|inner product"):
        FaissVectorIndex.load(path)


def test_corrupt_faiss_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.faiss"
    path.write_bytes(b"not-a-faiss-index")
    with pytest.raises(VectorIndexError, match="Failed to load"):
        FaissVectorIndex.load(path)
