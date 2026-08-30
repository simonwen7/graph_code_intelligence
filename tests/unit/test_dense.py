"""Dense indexing and search tests (offline fake embeddings)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers.fake_embeddings import FakeEmbeddingProvider

from codeintel.dense import (
    ARTIFACT_VERSION,
    DenseIndexError,
    DenseIndexMismatchError,
    DenseIndexMissingError,
    build_dense_index,
    compute_corpus_fingerprint,
    format_dense_document,
    load_and_validate_dense_artifact,
    load_dense_documents,
    search_dense,
)
from codeintel.languages.python import PythonAdapter, PythonRelationExtractor
from codeintel.models import SymbolKind
from codeintel.repository import analyze_repository
from codeintel.storage import IndexDatabase

DENSE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "python_dense"


def _index_fixture(db_path: Path, root: Path = DENSE_ROOT) -> None:
    analysis = analyze_repository(root, PythonAdapter(), PythonRelationExtractor())
    with IndexDatabase(db_path) as database:
        database.rebuild(analysis)


def _provider_for_documents(
    database: IndexDatabase,
    *,
    query: str,
    preferred_qname: str,
    dimension: int = 8,
) -> FakeEmbeddingProvider:
    documents = load_dense_documents(database)
    document_vectors: dict[str, list[float]] = {}
    for offset, document in enumerate(documents):
        vector = [0.0] * dimension
        vector[offset % dimension] = 1.0
        document_vectors[document.document_text] = vector

    preferred = next(doc for doc in documents if doc.qualified_name == preferred_qname)
    query_vectors = {query: document_vectors[preferred.document_text]}
    return FakeEmbeddingProvider(
        dimension=dimension,
        document_vectors=document_vectors,
        query_vectors=query_vectors,
        default_document=[0.01] * dimension,
        default_query=[0.01] * dimension,
    )


def test_format_dense_document() -> None:
    text = format_dense_document("mod.fn", "def fn() -> None", "def fn() -> None:\n    return\n")
    assert text.startswith("symbol: mod.fn\n")
    assert "signature: def fn() -> None\n" in text
    assert text.endswith("code:\ndef fn() -> None:\n    return\n")


def test_build_artifact_metadata_and_qname_mapping(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    _index_fixture(db_path)
    with IndexDatabase(db_path, create=False) as database:
        documents = load_dense_documents(database)
        assert documents == sorted(documents, key=lambda item: item.qualified_name)
        provider = FakeEmbeddingProvider(
            dimension=4,
            default_document=[1.0, 0.0, 0.0, 0.0],
            default_query=[1.0, 0.0, 0.0, 0.0],
        )
        stats = build_dense_index(database, provider, artifact_dir=artifact_dir)

    assert stats.document_count == len(documents)
    assert (artifact_dir / "index.faiss").is_file()
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["artifact_version"] == ARTIFACT_VERSION
    assert metadata["provider_id"] == "fake"
    assert metadata["model_id"] == "fake-model"
    assert metadata["dimension"] == 4
    assert metadata["metric"] == "cosine_ip"
    assert metadata["normalized"] is True
    assert metadata["document_count"] == len(documents)
    assert metadata["qualified_names"] == [doc.qualified_name for doc in documents]
    assert "id" not in metadata
    assert metadata["corpus_fingerprint"] == compute_corpus_fingerprint(documents)


def test_fingerprint_changes_when_source_changes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "mod.py"
    source.write_text("def alpha() -> int:\n    return 1\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    _index_fixture(db_path, root)
    with IndexDatabase(db_path, create=False) as database:
        first = compute_corpus_fingerprint(load_dense_documents(database))

    source.write_text("def alpha() -> int:\n    return 2\n", encoding="utf-8")
    _index_fixture(db_path, root)
    with IndexDatabase(db_path, create=False) as database:
        second = compute_corpus_fingerprint(load_dense_documents(database))
    assert first != second


def test_dense_search_ranking_filters_and_snapshot(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    _index_fixture(db_path)
    query = "check whether a login session is still valid"
    preferred = "auth_session.refresh_access_token"

    with IndexDatabase(db_path, create=False) as database:
        provider = _provider_for_documents(database, query=query, preferred_qname=preferred)
        build_dense_index(database, provider, artifact_dir=artifact_dir)
        results = search_dense(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=3,
        )
        assert results[0].symbol_qualified_name == preferred
        assert results[0].score >= results[1].score
        assert "ttl_seconds" in results[0].source_text

        methods = search_dense(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=10,
            kind=SymbolKind.FUNCTION,
        )
        assert all(result.kind is SymbolKind.FUNCTION for result in methods)

        prefixed = search_dense(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=10,
            path_prefix="auth_",
        )
        assert all(str(result.path).startswith("auth_") for result in prefixed)
        assert search_dense(database, provider, "   ", artifact_dir=artifact_dir) == ()


def test_dense_does_not_read_live_source(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "mod.py"
    source.write_text(
        "def authorize_payment(amount: int) -> bool:\n    return amount > 0\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    _index_fixture(db_path, root)
    with IndexDatabase(db_path, create=False) as database:
        provider = FakeEmbeddingProvider(
            dimension=2,
            default_document=[1.0, 0.0],
            default_query=[1.0, 0.0],
        )
        build_dense_index(database, provider, artifact_dir=artifact_dir)
        source.write_text(
            "def authorize_payment(amount: int) -> bool:\n    return 'MUTATED'\n",
            encoding="utf-8",
        )
        results = search_dense(
            database,
            provider,
            "authorize",
            artifact_dir=artifact_dir,
            limit=1,
        )
    assert "MUTATED" not in results[0].source_text
    assert "amount > 0" in results[0].source_text


def test_stale_artifact_rejected(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "mod.py"
    source.write_text("def alpha() -> int:\n    return 1\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    _index_fixture(db_path, root)
    with IndexDatabase(db_path, create=False) as database:
        provider = FakeEmbeddingProvider(
            dimension=2,
            default_document=[1.0, 0.0],
            default_query=[1.0, 0.0],
        )
        build_dense_index(database, provider, artifact_dir=artifact_dir)

    source.write_text("def alpha() -> int:\n    return 99\n", encoding="utf-8")
    _index_fixture(db_path, root)
    with IndexDatabase(db_path, create=False) as database:
        provider = FakeEmbeddingProvider(
            dimension=2,
            default_document=[1.0, 0.0],
            default_query=[1.0, 0.0],
        )
        with pytest.raises(DenseIndexMismatchError, match="stale|fingerprint"):
            search_dense(database, provider, "alpha", artifact_dir=artifact_dir)


def test_missing_corrupt_version_model_mismatch(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    _index_fixture(db_path)
    with IndexDatabase(db_path, create=False) as database:
        provider = FakeEmbeddingProvider(
            dimension=3,
            default_document=[1.0, 0.0, 0.0],
            default_query=[1.0, 0.0, 0.0],
        )
        with pytest.raises(DenseIndexMissingError):
            search_dense(database, provider, "x", artifact_dir=artifact_dir)

        build_dense_index(database, provider, artifact_dir=artifact_dir)
        metadata_path = artifact_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        metadata["artifact_version"] = 999
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(DenseIndexMismatchError, match="artifact_version"):
            load_and_validate_dense_artifact(database, provider, artifact_dir=artifact_dir)

        metadata["artifact_version"] = ARTIFACT_VERSION
        metadata["model_id"] = "other-model"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(DenseIndexMismatchError, match="model"):
            load_and_validate_dense_artifact(database, provider, artifact_dir=artifact_dir)

        metadata["model_id"] = provider.model_id
        metadata["dimension"] = 99
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(DenseIndexMismatchError, match="dimension"):
            load_and_validate_dense_artifact(database, provider, artifact_dir=artifact_dir)

        metadata_path.write_text("{not-json", encoding="utf-8")
        with pytest.raises(Exception, match="Corrupt|JSON"):
            load_and_validate_dense_artifact(database, provider, artifact_dir=artifact_dir)

        # restore valid metadata then corrupt FAISS
        build_dense_index(database, provider, artifact_dir=artifact_dir, full=True)
        (artifact_dir / "index.faiss").write_bytes(b"corrupt")
        with pytest.raises(Exception, match="FAISS|Failed|Corrupt|load"):
            search_dense(database, provider, "x", artifact_dir=artifact_dir)


def test_qname_identity_not_row_ids(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    _index_fixture(db_path)
    with IndexDatabase(db_path, create=False) as database:
        docs = load_dense_documents(database)
        # Force document vectors keyed by text; ordinal order follows qname ASC.
        provider = FakeEmbeddingProvider(
            dimension=len(docs) + 1,
            document_vectors={
                doc.document_text: [1.0 if i == offset else 0.0 for i in range(len(docs) + 1)]
                for offset, doc in enumerate(docs)
            },
            query_vectors={
                "q": [1.0 if i == 0 else 0.0 for i in range(len(docs) + 1)],
            },
        )
        build_dense_index(database, provider, artifact_dir=artifact_dir)
        metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        row_ids = [
            int(row[0])
            for row in database.connection.execute(
                "SELECT id FROM code_units ORDER BY id"
            ).fetchall()
        ]
        assert metadata["qualified_names"][0] == docs[0].qualified_name
        # Mapping is by qname list, not by asserting equality with row ids.
        assert metadata["qualified_names"] != [str(row_id) for row_id in row_ids]
        results = search_dense(database, provider, "q", artifact_dir=artifact_dir, limit=1)
        assert results[0].symbol_qualified_name == docs[0].qualified_name


def test_empty_corpus_dense(tmp_path: Path) -> None:
    root = tmp_path / "empty_repo"
    root.mkdir()
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    analysis = analyze_repository(root, PythonAdapter(), PythonRelationExtractor())
    with IndexDatabase(db_path) as database:
        database.rebuild(analysis)
        provider = FakeEmbeddingProvider(dimension=2)
        stats = build_dense_index(database, provider, artifact_dir=artifact_dir)
        assert stats.document_count == 0
        assert search_dense(database, provider, "anything", artifact_dir=artifact_dir) == ()


def test_deterministic_tie_break(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def zebra() -> None:\n    return\n", encoding="utf-8")
    (root / "b.py").write_text("def apple() -> None:\n    return\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    _index_fixture(db_path, root)
    identical = [1.0, 0.0]
    with IndexDatabase(db_path, create=False) as database:
        provider = FakeEmbeddingProvider(
            dimension=2,
            default_document=identical,
            default_query=identical,
        )
        build_dense_index(database, provider, artifact_dir=artifact_dir)
        results = search_dense(database, provider, "tie", artifact_dir=artifact_dir, limit=2)
        names = [result.symbol_qualified_name for result in results]
        assert names == sorted(names)
        assert results[0].score == pytest.approx(results[1].score)


def test_fingerprint_ignores_live_source_until_reindex(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "mod.py"
    source.write_text("def alpha() -> int:\n    return 1\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    _index_fixture(db_path, root)
    with IndexDatabase(db_path, create=False) as database:
        provider = FakeEmbeddingProvider(
            dimension=2,
            default_document=[1.0, 0.0],
            default_query=[1.0, 0.0],
        )
        build_dense_index(database, provider, artifact_dir=artifact_dir)
        source.write_text("def alpha() -> int:\n    return 999\n", encoding="utf-8")
        # Live source changed but SQLite snapshot unchanged → dense still valid.
        results = search_dense(database, provider, "alpha", artifact_dir=artifact_dir, limit=1)
        assert results[0].symbol_qualified_name == "mod.alpha"
        assert "return 1" in results[0].source_text
        assert "999" not in results[0].source_text


def test_document_format_version_and_duplicate_qnames_rejected(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    _index_fixture(db_path)
    with IndexDatabase(db_path, create=False) as database:
        provider = FakeEmbeddingProvider(
            dimension=2,
            default_document=[1.0, 0.0],
            default_query=[1.0, 0.0],
        )
        build_dense_index(database, provider, artifact_dir=artifact_dir)
        metadata_path = artifact_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        metadata["document_text_format_version"] = 999
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(DenseIndexMismatchError, match="document_text_format_version"):
            load_and_validate_dense_artifact(database, provider, artifact_dir=artifact_dir)

        metadata["document_text_format_version"] = 1
        names = list(metadata["qualified_names"])
        assert len(names) >= 2
        names[-1] = names[0]
        metadata["qualified_names"] = names
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(DenseIndexError, match="duplicate"):
            load_and_validate_dense_artifact(database, provider, artifact_dir=artifact_dir)


def test_failed_rebuild_preserves_previous_artifact(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    _index_fixture(db_path)
    with IndexDatabase(db_path, create=False) as database:
        provider = FakeEmbeddingProvider(
            dimension=2,
            default_document=[1.0, 0.0],
            default_query=[1.0, 0.0],
        )
        build_dense_index(database, provider, artifact_dir=artifact_dir)
        original_meta = (artifact_dir / "metadata.json").read_text(encoding="utf-8")
        original_faiss = (artifact_dir / "index.faiss").read_bytes()

        class BoomProvider(FakeEmbeddingProvider):
            def embed_documents(self, texts):  # type: ignore[no-untyped-def]
                raise RuntimeError("boom during embed")

        with pytest.raises(RuntimeError, match="boom"):
            build_dense_index(
                database,
                BoomProvider(dimension=2, default_document=[1.0, 0.0], default_query=[1.0, 0.0]),
                artifact_dir=artifact_dir,
                full=True,
            )
        assert (artifact_dir / "metadata.json").read_text(encoding="utf-8") == original_meta
        assert (artifact_dir / "index.faiss").read_bytes() == original_faiss
        # Previous artifact still searchable.
        assert search_dense(database, provider, "anything", artifact_dir=artifact_dir)


def test_filter_can_surface_candidate_outside_unfiltered_topk(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "keep.py").write_text(
        "def keep_me() -> None:\n    return\n",
        encoding="utf-8",
    )
    for index in range(5):
        (root / f"other_{index}.py").write_text(
            f"def other_{index}() -> None:\n    return\n",
            encoding="utf-8",
        )
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    _index_fixture(db_path, root)
    with IndexDatabase(db_path, create=False) as database:
        documents = load_dense_documents(database)
        document_vectors: dict[str, list[float]] = {}
        for document in documents:
            # Keep path matches filter but is least similar unfiltered.
            if document.qualified_name == "keep.keep_me":
                document_vectors[document.document_text] = [0.01, 1.0]
            else:
                document_vectors[document.document_text] = [1.0, 0.0]
        provider = FakeEmbeddingProvider(
            dimension=2,
            document_vectors=document_vectors,
            query_vectors={"q": [1.0, 0.0]},
            default_document=[1.0, 0.0],
            default_query=[1.0, 0.0],
        )
        build_dense_index(database, provider, artifact_dir=artifact_dir)
        unfiltered = search_dense(database, provider, "q", artifact_dir=artifact_dir, limit=2)
        assert all(result.symbol_qualified_name != "keep.keep_me" for result in unfiltered)
        filtered = search_dense(
            database,
            provider,
            "q",
            artifact_dir=artifact_dir,
            limit=1,
            path_prefix="keep",
        )
        assert filtered[0].symbol_qualified_name == "keep.keep_me"


def test_fingerprint_length_prefix_avoids_boundary_ambiguity() -> None:
    from codeintel.dense import _DenseDocument

    left = [
        _DenseDocument("ab", SymbolKind.FUNCTION, "p.py", None, "c", 1, 1, 0, 1),
    ]
    right = [
        _DenseDocument("a", SymbolKind.FUNCTION, "p.py", None, "bc", 1, 1, 0, 1),
    ]
    # Different field boundaries must not collide under length-prefixed hashing.
    assert compute_corpus_fingerprint(left) != compute_corpus_fingerprint(right)
