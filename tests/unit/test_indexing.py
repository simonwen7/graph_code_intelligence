"""Tests for Milestone 8 incremental indexing and selective dense reuse."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from helpers.fake_embeddings import FakeEmbeddingProvider

from codeintel.dense import (
    ARTIFACT_VERSION,
    build_dense_index,
    dense_document_fingerprint,
    format_dense_document,
    load_and_validate_dense_artifact,
    search_dense,
)
from codeintel.indexing import (
    IndexWorkStats,
    compute_changeset,
    hash_bytes,
    hash_file_bytes,
    index_repository,
)
from codeintel.languages.python import PythonAdapter, PythonRelationExtractor
from codeintel.lexical import search_code_units
from codeintel.models import AnalysisResult, ResolutionStatus
from codeintel.storage import SCHEMA_VERSION, IndexDatabase, SchemaVersionError
from codeintel.storage.schema import SCHEMA_SQL
from codeintel.vector_index import FaissVectorIndex


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _index(root: Path, db_path: Path, *, full: bool = False) -> IndexWorkStats:
    return index_repository(
        root,
        PythonAdapter(),
        PythonRelationExtractor(),
        database_path=db_path,
        full=full,
    )


def test_hash_raw_bytes_semantics(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    path.write_bytes(b"abc")
    assert hash_file_bytes(path) == hash_bytes(b"abc")
    assert hash_bytes(b"abc") == hash_bytes(b"abc")
    assert hash_bytes(b"abc") != hash_bytes(b"abd")
    assert hash_bytes(b"a\nb") != hash_bytes(b"a\r\nb")
    assert hash_bytes("café".encode()) == hash_bytes("café".encode())


def test_changeset_classification() -> None:
    changeset = compute_changeset(
        current_hashes={"a.py": "1", "b.py": "2", "c.py": "3"},
        persisted_hashes={"b.py": "2", "c.py": "9", "d.py": "4"},
    )
    assert changeset.added == ("a.py",)
    assert changeset.changed == ("c.py",)
    assert changeset.deleted == ("d.py",)
    assert changeset.unchanged == ("b.py",)


def test_noop_preserves_ids_and_skips_work(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def alpha() -> int:\n    return 1\n")
    _write(root / "b.py", "def beta() -> int:\n    return 2\n")
    db_path = tmp_path / "index.db"
    first = _index(root, db_path)
    assert first.mode == "full"
    with IndexDatabase(db_path, create=False) as database:
        file_ids = database.file_id_map()
        unit_ids = database.code_unit_id_map()
        relations = database.load_relations()
        symbols = database.load_symbols()
        fts_count = database.counts().fts_documents
    second = _index(root, db_path)
    assert second.mode == "noop"
    assert second.files_analyzed == 0
    assert second.relation_files_recomputed == 0
    with IndexDatabase(db_path, create=False) as database:
        assert database.file_id_map() == file_ids
        assert database.code_unit_id_map() == unit_ids
        assert database.load_relations() == relations
        assert database.load_symbols() == symbols
        assert database.counts().fts_documents == fts_count


def test_body_edit_local_relation_refresh(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def alpha() -> int:\n    return 1\n")
    _write(root / "b.py", "def beta() -> int:\n    return alpha()\n")
    db_path = tmp_path / "index.db"
    _index(root, db_path)
    with IndexDatabase(db_path, create=False) as database:
        before_ids = database.code_unit_id_map()
        b_file_id = database.file_id_map()["b.py"]
    _write(root / "a.py", "def alpha() -> int:\n    return 99\n")
    stats = _index(root, db_path)
    assert stats.mode == "incremental"
    assert stats.files_changed == 1
    assert stats.files_analyzed == 1
    assert stats.relation_files_recomputed == 1
    with IndexDatabase(db_path, create=False) as database:
        assert database.file_id_map()["b.py"] == b_file_id
        assert database.code_unit_id_map()["b.beta"] == before_ids["b.beta"]
        assert database.code_unit_id_map()["a.alpha"] != before_ids["a.alpha"]
        units = dict(database.load_code_units())
        assert "return 99" in units["a.alpha"].source_text
        assert "return alpha()" in units["b.beta"].source_text


def test_signature_edit_does_not_force_global_refresh(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def alpha() -> int:\n    return 1\n")
    _write(root / "b.py", "def beta() -> int:\n    return 2\n")
    db_path = tmp_path / "index.db"
    _index(root, db_path)
    _write(root / "a.py", "def alpha() -> str:\n    return 'x'\n")
    stats = _index(root, db_path)
    assert stats.relation_files_recomputed == 1
    assert stats.files_unchanged == 1


def test_symbol_rename_global_refresh_clears_stale_targets(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def alpha() -> int:\n    return 1\n")
    _write(root / "b.py", "from a import alpha\n\ndef beta() -> int:\n    return alpha()\n")
    db_path = tmp_path / "index.db"
    _index(root, db_path)
    _write(root / "a.py", "def gamma() -> int:\n    return 1\n")
    stats = _index(root, db_path)
    assert stats.relation_files_recomputed == 2
    with IndexDatabase(db_path, create=False) as database:
        qnames = {symbol.qualified_name for symbol in database.load_symbols()}
        assert "a.alpha" not in qnames
        assert "a.gamma" in qnames
        targets = {
            relation.target_qualified_name
            for relation in database.load_relations()
            if relation.target_qualified_name is not None
        }
        assert "a.alpha" not in targets


def test_new_target_appears_global_refresh(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(
        root / "b.py",
        "from a import helper\n\ndef beta() -> int:\n    return helper()\n",
    )
    db_path = tmp_path / "index.db"
    _index(root, db_path)
    with IndexDatabase(db_path, create=False) as database:
        imports = [
            relation for relation in database.load_relations() if relation.kind.value == "imports"
        ]
        assert imports
        assert all(relation.resolution is ResolutionStatus.UNRESOLVED for relation in imports)
    _write(root / "a.py", "def helper() -> int:\n    return 1\n")
    stats = _index(root, db_path)
    assert stats.files_added == 1
    assert stats.relation_files_recomputed == 2
    with IndexDatabase(db_path, create=False) as database:
        imports = [
            relation for relation in database.load_relations() if relation.kind.value == "imports"
        ]
        assert any(relation.resolution is ResolutionStatus.RESOLVED for relation in imports)


def test_file_delete_and_add_and_rename(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def alpha() -> int:\n    return 1\n")
    _write(root / "b.py", "def beta() -> int:\n    return 2\n")
    db_path = tmp_path / "index.db"
    _index(root, db_path)
    (root / "b.py").unlink()
    deleted = _index(root, db_path)
    assert deleted.files_deleted == 1
    with IndexDatabase(db_path, create=False) as database:
        assert "b.py" not in database.file_id_map()
        assert all(not qname.startswith("b.") for qname in database.code_unit_id_map())
    _write(root / "c.py", "def charlie() -> int:\n    return 3\n")
    added = _index(root, db_path)
    assert added.files_added == 1
    (root / "a.py").rename(root / "d.py")
    renamed = _index(root, db_path)
    assert renamed.files_deleted == 1
    assert renamed.files_added == 1
    with IndexDatabase(db_path, create=False) as database:
        paths = set(database.file_id_map())
        assert "a.py" not in paths
        assert "d.py" in paths


def test_collision_preflight_from_duplicate_modules(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "pkg" / "__init__.py", "VALUE = 1\n")
    db_path = tmp_path / "index.db"
    _index(root, db_path)
    with IndexDatabase(db_path, create=False) as database:
        before_files = database.load_files()
        before_counts = database.counts()
    # Second __init__ path that resolves to same module name is difficult; use nested package
    # trick from existing tests: write two files with identical module via same relative stem
    # under mirrored layout used in test_repository.
    _write(root / "pkg.py", "def collide() -> None:\n    return\n")
    # pkg.py module is `pkg`, pkg/__init__.py module is also `pkg` → duplicate MODULE symbol.
    with pytest.raises(ValueError, match="Duplicate qualified name"):
        _index(root, db_path)
    with IndexDatabase(db_path, create=False) as database:
        assert database.load_files() == before_files
        assert database.counts() == before_counts


def test_syntax_error_replaces_current_truth(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def alpha() -> int:\n    return 1\n")
    db_path = tmp_path / "index.db"
    _index(root, db_path)
    _write(root / "a.py", "def alpha( -> int:\n    return 1\n")
    stats = _index(root, db_path)
    assert stats.files_changed == 1
    with IndexDatabase(db_path, create=False) as database:
        files = database.load_files()
        assert files[0][3] is True  # has_syntax_errors


def test_transaction_rollback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def alpha() -> int:\n    return 1\n")
    db_path = tmp_path / "index.db"
    _index(root, db_path)
    with IndexDatabase(db_path, create=False) as database:
        before_files = database.load_files()
        before_counts = database.counts()
        before_fts = list(
            database.connection.execute(
                "SELECT rowid, qualified_name, source_text FROM code_units_fts ORDER BY rowid"
            )
        )
    _write(root / "a.py", "def alpha() -> int:\n    return 2\n")

    def boom(self, paths, file_id_by_path):  # type: ignore[no-untyped-def]
        raise sqlite3.Error("boom")

    # Fail after FTS delete / before symbol deletion completes the mutation.
    monkeypatch.setattr(IndexDatabase, "_delete_symbols_for_paths", boom)
    with pytest.raises(Exception, match="boom|Failed"):
        _index(root, db_path)
    with IndexDatabase(db_path, create=False) as database:
        assert database.load_files() == before_files
        assert database.counts() == before_counts
        after_fts = list(
            database.connection.execute(
                "SELECT rowid, qualified_name, source_text FROM code_units_fts ORDER BY rowid"
            )
        )
        assert after_fts == before_fts


def test_schema_v1_requires_full_rebuild(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def alpha() -> int:\n    return 1\n")
    db_path = tmp_path / "index.db"
    # Create a v1-shaped database without content_sha256.
    v1_sql = """
    CREATE TABLE files (
        id INTEGER PRIMARY KEY,
        path TEXT NOT NULL UNIQUE,
        language_id TEXT NOT NULL,
        module_name TEXT NOT NULL,
        has_syntax_errors INTEGER NOT NULL CHECK (has_syntax_errors IN (0, 1))
    );
    CREATE TABLE symbols (
        id INTEGER PRIMARY KEY,
        file_id INTEGER NOT NULL REFERENCES files(id),
        name TEXT NOT NULL,
        qualified_name TEXT NOT NULL UNIQUE,
        kind TEXT NOT NULL,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        start_byte INTEGER NOT NULL,
        end_byte INTEGER NOT NULL,
        signature TEXT NULL,
        parent_qualified_name TEXT NULL
    );
    CREATE TABLE code_units (
        id INTEGER PRIMARY KEY,
        symbol_id INTEGER NOT NULL UNIQUE REFERENCES symbols(id),
        kind TEXT NOT NULL,
        source_text TEXT NOT NULL,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        start_byte INTEGER NOT NULL,
        end_byte INTEGER NOT NULL
    );
    CREATE TABLE relations (
        id INTEGER PRIMARY KEY,
        source_qualified_name TEXT NOT NULL REFERENCES symbols(qualified_name),
        target_qualified_name TEXT NULL,
        target_text TEXT NOT NULL,
        kind TEXT NOT NULL,
        resolution TEXT NOT NULL,
        file_id INTEGER NOT NULL REFERENCES files(id),
        start_line INTEGER NULL,
        end_line INTEGER NULL,
        start_byte INTEGER NULL,
        end_byte INTEGER NULL
    );
    CREATE VIRTUAL TABLE code_units_fts USING fts5(
        qualified_name, name, signature, source_text, module_name, path, kind,
        tokenize = 'unicode61'
    );
    """
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(v1_sql)
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(SchemaVersionError, match="--full"):
        IndexDatabase(db_path, create=False).open()
    # Ensure v1 file was not mutated.
    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        columns = {row[1] for row in connection.execute("PRAGMA table_info(files)")}
        assert "content_sha256" not in columns
    finally:
        connection.close()

    stats = _index(root, db_path, full=True)
    assert stats.mode == "full"
    with IndexDatabase(db_path, create=False) as database:
        assert database.schema_version() == SCHEMA_VERSION
        assert database.load_files()[0][4]


def test_full_failure_preserves_old_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def alpha() -> int:\n    return 1\n")
    db_path = tmp_path / "index.db"
    _index(root, db_path)
    original = db_path.read_bytes()

    def boom(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("analyze boom")

    monkeypatch.setattr("codeintel.indexing.analyze_repository", boom)
    with pytest.raises(RuntimeError, match="analyze boom"):
        _index(root, db_path, full=True)
    assert db_path.read_bytes() == original


def test_dense_selective_reuse_and_full(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def alpha() -> int:\n    return 1\n")
    _write(root / "b.py", "def beta() -> int:\n    return 2\n")
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    _index(root, db_path)
    with IndexDatabase(db_path, create=False) as database:
        provider = FakeEmbeddingProvider(dimension=4, default_document=[1.0, 0.0, 0.0, 0.0])
        first = build_dense_index(database, provider, artifact_dir=artifact_dir, full=True)
        assert first.vectors_embedded == first.document_count
        assert first.vectors_reused == 0
        metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["artifact_version"] == ARTIFACT_VERSION
        assert "document_fingerprints" in metadata
        old_index = FaissVectorIndex.load(artifact_dir / "index.faiss")
        alpha_ordinal = metadata["qualified_names"].index("a.alpha")
        alpha_vector = old_index.reconstruct(alpha_ordinal)

    _write(root / "b.py", "def beta() -> int:\n    return 99\n")
    _index(root, db_path)
    with IndexDatabase(db_path, create=False) as database:
        with pytest.raises(Exception, match="stale|fingerprint"):
            search_dense(
                database,
                FakeEmbeddingProvider(dimension=4, default_document=[1.0, 0.0, 0.0, 0.0]),
                "alpha",
                artifact_dir=artifact_dir,
            )
        provider2 = FakeEmbeddingProvider(dimension=4, default_document=[0.0, 1.0, 0.0, 0.0])
        second = build_dense_index(database, provider2, artifact_dir=artifact_dir)
        assert second.vectors_reused >= 1
        assert second.vectors_embedded >= 1
        assert provider2.documents_embedded == second.vectors_embedded
        metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
        new_index = FaissVectorIndex.load(artifact_dir / "index.faiss")
        new_alpha = new_index.reconstruct(metadata["qualified_names"].index("a.alpha"))
        assert np.allclose(new_alpha, alpha_vector)
        load_and_validate_dense_artifact(database, provider2, artifact_dir=artifact_dir)

        provider3 = FakeEmbeddingProvider(dimension=4, default_document=[0.0, 0.0, 1.0, 0.0])
        full = build_dense_index(database, provider3, artifact_dir=artifact_dir, full=True)
        assert full.vectors_reused == 0
        assert full.vectors_embedded == full.document_count
        assert provider3.documents_embedded == full.document_count


def test_qname_rename_forces_reembed(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def alpha() -> int:\n    return 1\n")
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    _index(root, db_path)
    with IndexDatabase(db_path, create=False) as database:
        provider = FakeEmbeddingProvider(dimension=2, default_document=[1.0, 0.0])
        build_dense_index(database, provider, artifact_dir=artifact_dir, full=True)
    _write(root / "a.py", "def gamma() -> int:\n    return 1\n")
    _index(root, db_path)
    with IndexDatabase(db_path, create=False) as database:
        provider2 = FakeEmbeddingProvider(dimension=2, default_document=[0.0, 1.0])
        stats = build_dense_index(database, provider2, artifact_dir=artifact_dir)
        assert stats.vectors_reused == 0
        assert stats.vectors_embedded == stats.document_count


def test_legacy_artifact_no_fingerprint_reuse(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def alpha() -> int:\n    return 1\n")
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    _index(root, db_path)
    with IndexDatabase(db_path, create=False) as database:
        provider = FakeEmbeddingProvider(dimension=2, default_document=[1.0, 0.0])
        build_dense_index(database, provider, artifact_dir=artifact_dir, full=True)
        metadata_path = artifact_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.pop("document_fingerprints")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        # Search still works without fingerprints when corpus matches.
        load_and_validate_dense_artifact(database, provider, artifact_dir=artifact_dir)
        provider2 = FakeEmbeddingProvider(dimension=2, default_document=[0.0, 1.0])
        stats = build_dense_index(database, provider2, artifact_dir=artifact_dir)
        assert stats.vectors_reused == 0
        assert stats.vectors_embedded == stats.document_count
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert "document_fingerprints" in metadata


def test_dense_document_fingerprint_includes_qname() -> None:
    left = format_dense_document("a.alpha", "def alpha() -> int", "return 1\n")
    right = format_dense_document("a.gamma", "def alpha() -> int", "return 1\n")
    assert dense_document_fingerprint(left) != dense_document_fingerprint(right)


def test_schema_version_is_two() -> None:
    assert SCHEMA_VERSION == 2
    assert "content_sha256" in SCHEMA_SQL


def _semantic_snapshot(database: IndexDatabase) -> object:
    files = [
        (path, language_id, module_name, has_errors, digest)
        for path, language_id, module_name, has_errors, digest in database.load_files()
    ]
    symbols = [
        (
            symbol.qualified_name,
            symbol.kind,
            symbol.signature,
            symbol.parent_qualified_name,
            symbol.span,
        )
        for symbol in database.load_symbols()
    ]
    units = [
        (qname, unit.kind, unit.source_text, unit.span)
        for qname, unit in database.load_code_units()
    ]
    relations = sorted(
        (
            relation.kind,
            relation.source_qualified_name,
            relation.target_qualified_name,
            relation.target_text,
            relation.resolution,
            relation.path.name,
            relation.span,
        )
        for relation in database.load_relations()
    )
    return (files, symbols, units, relations)


def test_incremental_vs_full_semantic_equivalence(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def alpha() -> int:\n    return 1\n")
    _write(root / "b.py", "def beta() -> int:\n    return 2\n")
    db_inc = tmp_path / "inc.db"
    _index(root, db_inc)
    _write(root / "a.py", "def alpha() -> int:\n    return 99\n")
    _index(root, db_inc)
    _write(root / "a.py", "def gamma() -> int:\n    return 99\n")
    _write(root / "c.py", "def charlie() -> int:\n    return 3\n")
    (root / "b.py").unlink()
    _index(root, db_inc)

    db_full = tmp_path / "full.db"
    _index(root, db_full, full=True)
    with IndexDatabase(db_inc, create=False) as left:
        with IndexDatabase(db_full, create=False) as right:
            assert _semantic_snapshot(left) == _semantic_snapshot(right)


def test_analyze_only_changed_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def alpha() -> int:\n    return 1\n")
    _write(root / "b.py", "def beta() -> int:\n    return 2\n")
    db_path = tmp_path / "index.db"
    _index(root, db_path)
    seen: list[str] = []
    real = PythonAdapter.analyze_file

    def spy(
        self: PythonAdapter,
        path: Path,
        *,
        repository_root: Path | None = None,
    ) -> AnalysisResult:
        seen.append(path.name)
        return real(self, path, repository_root=repository_root)

    monkeypatch.setattr(PythonAdapter, "analyze_file", spy)
    _write(root / "a.py", "def alpha() -> int:\n    return 2\n")
    _index(root, db_path)
    assert seen == ["a.py"]
    seen.clear()
    _index(root, db_path)
    assert seen == []


def test_body_relation_edit_stays_local(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def helper() -> int:\n    return 1\n")
    _write(
        root / "b.py",
        "from a import helper\n\ndef beta() -> int:\n    return helper()\n",
    )
    db_path = tmp_path / "index.db"
    _index(root, db_path)
    _write(
        root / "b.py",
        "from a import helper\n\ndef beta() -> int:\n    value = helper()\n    return value\n",
    )
    stats = _index(root, db_path)
    assert stats.relation_files_recomputed == 1


def test_target_disappears_clears_resolved_edge(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def helper() -> int:\n    return 1\n")
    _write(
        root / "b.py",
        "from a import helper\n\ndef beta() -> int:\n    return helper()\n",
    )
    db_path = tmp_path / "index.db"
    _index(root, db_path)
    with IndexDatabase(db_path, create=False) as database:
        assert any(
            relation.resolution is ResolutionStatus.RESOLVED
            and relation.target_qualified_name == "a.helper"
            for relation in database.load_relations()
        )
    _write(root / "a.py", "def other() -> int:\n    return 1\n")
    stats = _index(root, db_path)
    assert stats.relation_files_recomputed == 2
    with IndexDatabase(db_path, create=False) as database:
        targets = {
            relation.target_qualified_name
            for relation in database.load_relations()
            if relation.target_qualified_name is not None
        }
        assert "a.helper" not in targets


def test_fts_unchanged_rowids_and_stale_text(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def alpha() -> int:\n    return TOKEN_OLD\n")
    _write(root / "b.py", "def beta() -> int:\n    return KEEP_ME\n")
    db_path = tmp_path / "index.db"
    _index(root, db_path)
    with IndexDatabase(db_path, create=False) as database:
        before = database.code_unit_id_map()
    _write(root / "a.py", "def alpha() -> int:\n    return TOKEN_NEW\n")
    stats = _index(root, db_path)
    assert stats.relation_files_recomputed == 1
    with IndexDatabase(db_path, create=False) as database:
        assert database.code_unit_id_map()["b.beta"] == before["b.beta"]
        assert database.code_unit_id_map()["a.alpha"] != before["a.alpha"]
        assert search_code_units(database, "TOKEN_NEW")
        assert not any(
            result.symbol_qualified_name == "a.alpha"
            for result in search_code_units(database, "TOKEN_OLD")
        )
        assert search_code_units(database, "KEEP_ME")


def test_provider_mismatch_prevents_reuse(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def alpha() -> int:\n    return 1\n")
    db_path = tmp_path / "index.db"
    artifact_dir = tmp_path / "dense"
    _index(root, db_path)
    with IndexDatabase(db_path, create=False) as database:
        build_dense_index(
            database,
            FakeEmbeddingProvider(
                dimension=4,
                provider_id="p1",
                model_id="m1",
                default_document=[1.0, 0.0, 0.0, 0.0],
            ),
            artifact_dir=artifact_dir,
            full=True,
        )
        stats = build_dense_index(
            database,
            FakeEmbeddingProvider(
                dimension=4,
                provider_id="p2",
                model_id="m1",
                default_document=[1.0, 0.0, 0.0, 0.0],
            ),
            artifact_dir=artifact_dir,
        )
        assert stats.vectors_reused == 0
        assert stats.vectors_embedded == stats.document_count


def test_dense_selective_vs_full_equivalence(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def alpha() -> int:\n    return 1\n")
    _write(root / "b.py", "def beta() -> int:\n    return 2\n")
    _write(root / "c.py", "def charlie() -> int:\n    return 3\n")
    db_path = tmp_path / "index.db"
    selective_dir = tmp_path / "selective"
    full_dir = tmp_path / "full"
    _index(root, db_path)

    class Det(FakeEmbeddingProvider):
        def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            self.document_embed_calls += 1
            self.documents_embedded += len(texts)
            rows = []
            for text in texts:
                vector = np.zeros(self.dimension, dtype=np.float32)
                vector[int(dense_document_fingerprint(text)[:8], 16) % self.dimension] = 1.0
                rows.append(vector)
            return np.stack(rows) if rows else np.zeros((0, self.dimension), dtype=np.float32)

    with IndexDatabase(db_path, create=False) as database:
        build_dense_index(database, Det(dimension=8), artifact_dir=selective_dir, full=True)
    _write(root / "b.py", "def beta() -> int:\n    return 99\n")
    _index(root, db_path)
    with IndexDatabase(db_path, create=False) as database:
        selective = build_dense_index(database, Det(dimension=8), artifact_dir=selective_dir)
        assert selective.vectors_embedded == 1
        assert selective.vectors_reused == selective.document_count - 1
        full = build_dense_index(database, Det(dimension=8), artifact_dir=full_dir, full=True)
        assert full.vectors_reused == 0
        left = json.loads((selective_dir / "metadata.json").read_text(encoding="utf-8"))
        right = json.loads((full_dir / "metadata.json").read_text(encoding="utf-8"))
        assert left["qualified_names"] == right["qualified_names"]
        assert left["document_fingerprints"] == right["document_fingerprints"]
        assert left["corpus_fingerprint"] == right["corpus_fingerprint"]
        index_left = FaissVectorIndex.load(selective_dir / "index.faiss")
        index_right = FaissVectorIndex.load(full_dir / "index.faiss")
        for ordinal in range(index_left.size):
            assert np.allclose(index_left.reconstruct(ordinal), index_right.reconstruct(ordinal))
        unchanged = build_dense_index(database, Det(dimension=8), artifact_dir=selective_dir)
        assert unchanged.vectors_embedded == 0
        assert unchanged.rewritten is False
