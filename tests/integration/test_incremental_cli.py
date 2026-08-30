"""Offline CLI tests for incremental index and selective embed."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from helpers.fake_embeddings import FakeEmbeddingProvider
from typer.testing import CliRunner

from codeintel.cli import app

runner = CliRunner()


def test_index_incremental_and_full_cli(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def alpha() -> int:\n    return 1\n", encoding="utf-8")
    db_path = tmp_path / "index.db"

    first = runner.invoke(app, ["index", str(root), "--db", str(db_path)])
    assert first.exit_code == 0, first.stdout + first.stderr
    assert "mode: full" in first.stdout
    assert "analyzed=" in first.stdout

    noop = runner.invoke(app, ["index", str(root), "--db", str(db_path)])
    assert noop.exit_code == 0, noop.stdout + noop.stderr
    assert "mode: noop" in noop.stdout
    assert "analyzed=0" in noop.stdout

    (root / "a.py").write_text("def alpha() -> int:\n    return 2\n", encoding="utf-8")
    changed = runner.invoke(app, ["index", str(root), "--db", str(db_path)])
    assert changed.exit_code == 0, changed.stdout + changed.stderr
    assert "mode: incremental" in changed.stdout
    assert "changed=1" in changed.stdout

    full = runner.invoke(app, ["index", str(root), "--db", str(db_path), "--full"])
    assert full.exit_code == 0, full.stdout + full.stderr
    assert "mode: full" in full.stdout


def test_index_v1_requires_full(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def alpha() -> int:\n    return 1\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE files (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                language_id TEXT NOT NULL,
                module_name TEXT NOT NULL,
                has_syntax_errors INTEGER NOT NULL
            )
            """
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    finally:
        connection.close()

    blocked = runner.invoke(app, ["index", str(root), "--db", str(db_path)])
    assert blocked.exit_code == 1
    assert "--full" in f"{blocked.stdout}\n{blocked.stderr}"

    recovered = runner.invoke(app, ["index", str(root), "--db", str(db_path), "--full"])
    assert recovered.exit_code == 0, recovered.stdout + recovered.stderr
    assert "mode: full" in recovered.stdout


def test_embed_selective_stats_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("def alpha() -> int:\n    return 1\n", encoding="utf-8")
    (root / "b.py").write_text("def beta() -> int:\n    return 2\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    dense_dir = tmp_path / "dense"
    assert runner.invoke(app, ["index", str(root), "--db", str(db_path)]).exit_code == 0

    monkeypatch.setattr(
        "codeintel.cli.create_embedding_provider",
        lambda model_id="fake-model": FakeEmbeddingProvider(
            dimension=4,
            model_id=model_id,
            default_document=[1.0, 0.0, 0.0, 0.0],
            default_query=[1.0, 0.0, 0.0, 0.0],
        ),
    )
    first = runner.invoke(
        app,
        ["embed", str(root), "--db", str(db_path), "--dense-dir", str(dense_dir)],
    )
    assert first.exit_code == 0, first.stdout + first.stderr
    assert "documents_total:" in first.stdout
    assert "vectors_embedded:" in first.stdout

    (root / "b.py").write_text("def beta() -> int:\n    return 9\n", encoding="utf-8")
    assert runner.invoke(app, ["index", str(root), "--db", str(db_path)]).exit_code == 0
    second = runner.invoke(
        app,
        ["embed", str(root), "--db", str(db_path), "--dense-dir", str(dense_dir)],
    )
    assert second.exit_code == 0, second.stdout + second.stderr
    assert "vectors_reused:" in second.stdout
    assert "vectors_embedded:" in second.stdout

    full = runner.invoke(
        app,
        ["embed", str(root), "--db", str(db_path), "--dense-dir", str(dense_dir), "--full"],
    )
    assert full.exit_code == 0, full.stdout + full.stderr
    assert "vectors_reused: 0" in full.stdout
