"""Incremental indexing and language-switch tests for C++."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.indexing import IndexLanguageError, index_repository
from codeintel.languages.cpp import CppAdapter, CppRelationExtractor
from codeintel.languages.python import PythonAdapter, PythonRelationExtractor
from codeintel.storage import IndexDatabase


def test_cpp_noop_preserves_ids(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.cpp").write_text("int add(int a, int b) { return a + b; }\n", encoding="utf-8")
    db = tmp_path / "index.db"
    first = index_repository(root, CppAdapter(), CppRelationExtractor(), database_path=db)
    with IndexDatabase(db, create=False) as database:
        before = database.load_symbols()
        before_ids = {
            symbol.qualified_name: _symbol_row_id(database, symbol.qualified_name)
            for symbol in before
        }
    second = index_repository(root, CppAdapter(), CppRelationExtractor(), database_path=db)
    assert second.mode == "noop"
    assert second.files_analyzed == 0
    assert second.relation_files_recomputed == 0
    with IndexDatabase(db, create=False) as database:
        after_ids = {
            symbol.qualified_name: _symbol_row_id(database, symbol.qualified_name)
            for symbol in database.load_symbols()
        }
    assert before_ids == after_ids
    assert first.symbols == second.symbols


def test_cpp_body_edit_local_refresh(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "main.cpp"
    target.write_text("int add(int a, int b) { return a + b; }\n", encoding="utf-8")
    db = tmp_path / "index.db"
    index_repository(root, CppAdapter(), CppRelationExtractor(), database_path=db)
    with IndexDatabase(db, create=False) as database:
        unchanged_qname = "add(int, int)"
        kept_id = _symbol_row_id(database, unchanged_qname)
    target.write_text("int add(int a, int b) { return a + b + 1; }\n", encoding="utf-8")
    stats = index_repository(root, CppAdapter(), CppRelationExtractor(), database_path=db)
    assert stats.mode == "incremental"
    assert stats.files_changed == 1
    assert stats.files_analyzed == 1
    assert stats.relation_files_recomputed == 1
    with IndexDatabase(db, create=False) as database:
        assert _symbol_row_id(database, unchanged_qname) == kept_id
        unit = next(
            code_unit for qname, code_unit in database.load_code_units() if qname == unchanged_qname
        )
        assert "+ 1" in unit.source_text


def test_cpp_rename_global_refresh(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "main.cpp"
    target.write_text(
        "int add(int a, int b) { return a + b; }\nint use() { return add(1, 2); }\n",
        encoding="utf-8",
    )
    db = tmp_path / "index.db"
    index_repository(root, CppAdapter(), CppRelationExtractor(), database_path=db)
    target.write_text(
        "int sum(int a, int b) { return a + b; }\nint use() { return sum(1, 2); }\n",
        encoding="utf-8",
    )
    stats = index_repository(root, CppAdapter(), CppRelationExtractor(), database_path=db)
    assert stats.relation_files_recomputed == 1
    with IndexDatabase(db, create=False) as database:
        qnames = {symbol.qualified_name for symbol in database.load_symbols()}
        assert "add(int, int)" not in qnames
        assert "sum(int, int)" in qnames
        relations = database.load_relations()
        assert all(rel.target_qualified_name != "add(int, int)" for rel in relations)


def test_language_mismatch_blocks_incremental(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("def hello() -> None:\n    pass\n", encoding="utf-8")
    (root / "main.cpp").write_text("int hello() { return 1; }\n", encoding="utf-8")
    db = tmp_path / "index.db"
    index_repository(root, PythonAdapter(), PythonRelationExtractor(), database_path=db)
    with IndexDatabase(db, create=False) as database:
        before = database.counts()
    with pytest.raises(IndexLanguageError, match="--full"):
        index_repository(root, CppAdapter(), CppRelationExtractor(), database_path=db)
    with IndexDatabase(db, create=False) as database:
        after = database.counts()
    assert before == after

    replaced = index_repository(
        root,
        CppAdapter(),
        CppRelationExtractor(),
        database_path=db,
        full=True,
    )
    assert replaced.mode == "full"
    with IndexDatabase(db, create=False) as database:
        languages = {language for _, language, _, _, _ in database.load_files()}
        assert languages == {"cpp"}
        assert all(language == "cpp" for _, language, _, _, _ in database.load_files())


def _symbol_row_id(database: IndexDatabase, qname: str) -> int:
    row = database.connection.execute(
        "SELECT id FROM symbols WHERE qualified_name = ?",
        (qname,),
    ).fetchone()
    assert row is not None
    return int(row["id"])
