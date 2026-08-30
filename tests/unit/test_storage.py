"""Tests for SQLite index schema lifecycle and persistence roundtrips."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from codeintel.graph import CodeGraph
from codeintel.languages.python import PythonAdapter, PythonRelationExtractor
from codeintel.lexical import search_code_units
from codeintel.models import (
    AnalysisResult,
    RelationKind,
    ResolutionStatus,
    SourceSpan,
    Symbol,
    SymbolKind,
)
from codeintel.repository import RepositoryAnalysis, analyze_repository
from codeintel.storage import (
    SCHEMA_VERSION,
    IndexDatabase,
    IndexDatabaseError,
    SchemaVersionError,
    default_index_path,
)

GRAPH_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "python_graph"
SEARCH_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "python_search"


def _analyze(root: Path = GRAPH_ROOT) -> RepositoryAnalysis:
    return analyze_repository(root, PythonAdapter(), PythonRelationExtractor())


def test_default_index_path() -> None:
    assert default_index_path(Path("/tmp/repo")) == Path("/tmp/repo/.codeintel/index.db")


def test_schema_initialization_and_foreign_keys(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    with IndexDatabase(db_path) as database:
        assert database.schema_version() == SCHEMA_VERSION
        assert database.foreign_keys_enabled() is True
        names = {
            row[0]
            for row in database.connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        assert {"files", "symbols", "code_units", "relations", "code_units_fts"} <= names


def test_unsupported_schema_version_is_rejected_without_reset(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    with IndexDatabase(db_path) as database:
        database.connection.execute("PRAGMA user_version = 999")
        database.connection.commit()
    with pytest.raises(SchemaVersionError, match="Unsupported index schema version 999"):
        IndexDatabase(db_path).open()
    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 999
    finally:
        connection.close()


def test_roundtrip_files_symbols_code_units_relations(tmp_path: Path) -> None:
    analysis = _analyze()
    db_path = tmp_path / "index.db"
    with IndexDatabase(db_path) as database:
        stats = database.rebuild(analysis)
        assert stats.files == len(analysis.files)
        assert stats.symbols == len(analysis.symbols)
        assert stats.code_units == sum(len(item.code_units) for item in analysis.files)
        assert stats.relations == len(analysis.relations)
        assert stats.fts_documents == stats.code_units

        files = database.load_files()
        assert all(not Path(path).is_absolute() for path, *_ in files)
        assert all("\\" not in path for path, *_ in files)
        assert {path for path, *_ in files} == {
            item.path.resolve().relative_to(analysis.root.resolve()).as_posix()
            for item in analysis.files
            if item.path is not None
        }

        loaded_symbols = database.load_symbols()
        assert [symbol.qualified_name for symbol in loaded_symbols] == [
            symbol.qualified_name for symbol in analysis.symbols
        ]
        for original, loaded in zip(analysis.symbols, loaded_symbols, strict=True):
            assert loaded.kind is original.kind
            assert loaded.span == original.span
            assert loaded.signature == original.signature
            assert loaded.parent_qualified_name == original.parent_qualified_name

        loaded_units = dict(database.load_code_units())
        expected_units = {
            unit.symbol_qualified_name: unit for item in analysis.files for unit in item.code_units
        }
        assert set(loaded_units) == set(expected_units)
        for qname, expected_unit in expected_units.items():
            loaded_unit = loaded_units[qname]
            assert loaded_unit.kind is expected_unit.kind
            assert loaded_unit.source_text == expected_unit.source_text
            assert loaded_unit.span == expected_unit.span
            assert expected_unit.kind is not SymbolKind.MODULE

        loaded_relations = database.load_relations()
        assert len(loaded_relations) == len(analysis.relations)
        unresolved = [
            relation
            for relation in loaded_relations
            if relation.resolution is ResolutionStatus.UNRESOLVED
        ]
        assert unresolved
        assert all(relation.target_qualified_name is None for relation in unresolved)
        assert all(isinstance(relation.path, Path) for relation in loaded_relations)
        assert all(not relation.path.is_absolute() for relation in loaded_relations)


def test_second_rebuild_replaces_stale_rows(tmp_path: Path) -> None:
    analysis = _analyze(SEARCH_ROOT)
    db_path = tmp_path / "index.db"
    with IndexDatabase(db_path) as database:
        first = database.rebuild(analysis)
        second = database.rebuild(analysis)
        assert second == first
        assert database.counts() == first


def test_failed_rebuild_rolls_back_previous_index(tmp_path: Path) -> None:
    analysis = _analyze(SEARCH_ROOT)
    db_path = tmp_path / "index.db"
    with IndexDatabase(db_path) as database:
        original = database.rebuild(analysis)
        before_files = database.load_files()
        before_symbols = [symbol.qualified_name for symbol in database.load_symbols()]
        before_fts = database.connection.execute(
            "SELECT qualified_name, source_text FROM code_units_fts ORDER BY qualified_name"
        ).fetchall()

        def boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("inject failure")

        database._insert_fts = boom  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="inject failure"):
            database.rebuild(analysis)
        assert database.load_files() == before_files
        assert [symbol.qualified_name for symbol in database.load_symbols()] == before_symbols
        after_fts = database.connection.execute(
            "SELECT qualified_name, source_text FROM code_units_fts ORDER BY qualified_name"
        ).fetchall()
        assert after_fts == before_fts
        assert database.counts() == original


def test_codegraph_reconstructs_from_persisted_symbols_and_relations(tmp_path: Path) -> None:
    analysis = _analyze()
    db_path = tmp_path / "index.db"
    with IndexDatabase(db_path) as database:
        database.rebuild(analysis)
        graph = CodeGraph(database.load_symbols(), database.load_relations())

    assert graph.has_symbol("helpers.helper")
    assert graph.has_symbol("service.Service")
    outgoing = graph.outgoing("service.Service")
    assert any(relation.kind is RelationKind.INHERITS for relation in outgoing)
    neighbors = [symbol.qualified_name for symbol in graph.neighbors("helpers")]
    assert "helpers.helper" in neighbors


def test_open_missing_database_without_create_fails(tmp_path: Path) -> None:
    missing = tmp_path / "missing.db"
    with pytest.raises(IndexDatabaseError, match="does not exist"):
        IndexDatabase(missing, create=False).open(create=False)


def test_foreign_sqlite_database_is_not_silently_converted(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.db"
    connection = sqlite3.connect(foreign)
    connection.execute("CREATE TABLE app_data(id INTEGER PRIMARY KEY, payload TEXT)")
    connection.execute("INSERT INTO app_data(payload) VALUES ('secret')")
    connection.commit()
    connection.close()

    with pytest.raises(IndexDatabaseError, match="not a compatible codeintel index"):
        IndexDatabase(foreign).open()

    verify = sqlite3.connect(foreign)
    try:
        assert verify.execute("PRAGMA user_version").fetchone()[0] == 0
        names = {
            row[0] for row in verify.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert names == {"app_data"}
        assert verify.execute("SELECT payload FROM app_data").fetchone()[0] == "secret"
    finally:
        verify.close()


def test_non_sqlite_file_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "not-a-db.db"
    path.write_text("this is not sqlite", encoding="utf-8")
    with pytest.raises(IndexDatabaseError, match="Failed to open index database"):
        IndexDatabase(path).open()


def test_second_rebuild_removes_stale_fts_terms(tmp_path: Path) -> None:
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()
    (repo_a / "alpha.py").write_text(
        "def unique_alpha_token() -> int:\n    return 1\n",
        encoding="utf-8",
    )
    (repo_b / "beta.py").write_text(
        "def unique_beta_token() -> int:\n    return 2\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "index.db"
    with IndexDatabase(db_path) as database:
        database.rebuild(_analyze(repo_a))
        assert search_code_units(database, "unique_alpha_token")
        database.rebuild(_analyze(repo_b))
        assert search_code_units(database, "unique_alpha_token") == ()
        beta = search_code_units(database, "unique_beta_token")
        assert beta
        assert beta[0].symbol_qualified_name == "beta.unique_beta_token"
        assert database.counts().fts_documents == database.counts().code_units == 1


def test_foreign_key_constraint_is_enforced(tmp_path: Path) -> None:
    analysis = _analyze(SEARCH_ROOT)
    db_path = tmp_path / "index.db"
    with IndexDatabase(db_path) as database:
        database.rebuild(analysis)
        assert database.foreign_keys_enabled() is True
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            database.connection.execute(
                """
                INSERT INTO symbols(
                    file_id, name, qualified_name, kind,
                    start_line, end_line, start_byte, end_byte
                ) VALUES (999, 'x', 'ghost.x', 'function', 1, 1, 0, 1)
                """
            )
        database.connection.rollback()


def test_codeunit_symbol_id_uniqueness_enforced(tmp_path: Path) -> None:
    analysis = _analyze(SEARCH_ROOT)
    db_path = tmp_path / "index.db"
    with IndexDatabase(db_path) as database:
        database.rebuild(analysis)
        symbol_id = database.connection.execute(
            "SELECT id FROM symbols WHERE qualified_name = ?",
            ("payment_gateway.authorize_payment",),
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            database.connection.execute(
                """
                INSERT INTO code_units(
                    symbol_id, kind, source_text,
                    start_line, end_line, start_byte, end_byte
                ) VALUES (?, 'function', 'dup', 1, 1, 0, 1)
                """,
                (symbol_id,),
            )
        database.connection.rollback()


def test_outside_repository_path_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    span = SourceSpan(1, 1, 0, 1)
    module = Symbol("mod", "mod", SymbolKind.MODULE, span, None, None)
    analysis = RepositoryAnalysis(
        root=root,
        files=(
            AnalysisResult(
                path=tmp_path / "outside.py",
                language_id="python",
                module_name="outside",
                symbols=(module,),
                code_units=(),
                has_syntax_errors=False,
            ),
        ),
        symbols=(module,),
        relations=(),
        graph=CodeGraph((module,), ()),
    )
    with IndexDatabase(tmp_path / "index.db") as database:
        with pytest.raises(IndexDatabaseError, match="outside repository root"):
            database.rebuild(analysis)


def test_gitignore_covers_default_index_location() -> None:
    root = Path(__file__).resolve().parents[2]
    ignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert ".codeintel/" in ignore
