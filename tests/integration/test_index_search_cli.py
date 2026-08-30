"""Integration tests for index and search CLI commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from codeintel.cli import app

SEARCH_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "python_search"
GRAPH_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "python_graph"
REPO_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "python_repo"
runner = CliRunner()


def test_help_lists_index_and_search() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("version", "inspect", "graph", "index", "embed", "search"):
        assert command in result.stdout


def test_index_and_search_with_temp_db(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    indexed = runner.invoke(app, ["index", str(SEARCH_ROOT), "--db", str(db_path)])
    assert indexed.exit_code == 0
    assert f"database: {db_path}" in indexed.stdout
    assert "files:" in indexed.stdout
    assert "fts_documents:" in indexed.stdout
    assert db_path.exists()

    searched = runner.invoke(
        app,
        ["search", str(SEARCH_ROOT), "authorize_payment", "--db", str(db_path)],
    )
    assert searched.exit_code == 0
    assert "payment_gateway.authorize_payment" in searched.stdout
    assert "score=" in searched.stdout


def test_search_limit_kind_and_path_prefix(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    assert runner.invoke(app, ["index", str(SEARCH_ROOT), "--db", str(db_path)]).exit_code == 0

    limited = runner.invoke(
        app,
        ["search", str(SEARCH_ROOT), "token", "--db", str(db_path), "--limit", "1"],
    )
    assert limited.exit_code == 0
    assert limited.stdout.count("score=") == 1

    kinded = runner.invoke(
        app,
        [
            "search",
            str(SEARCH_ROOT),
            "cache",
            "--db",
            str(db_path),
            "--kind",
            "method",
        ],
    )
    assert kinded.exit_code == 0
    assert "method" in kinded.stdout

    prefixed = runner.invoke(
        app,
        [
            "search",
            str(SEARCH_ROOT),
            "export",
            "--db",
            str(db_path),
            "--path-prefix",
            "report_",
        ],
    )
    assert prefixed.exit_code == 0
    assert "report_export" in prefixed.stdout


def test_search_before_index_fails_cleanly(tmp_path: Path) -> None:
    missing = tmp_path / "missing.db"
    result = runner.invoke(
        app,
        ["search", str(SEARCH_ROOT), "payment", "--db", str(missing)],
    )
    assert result.exit_code == 1
    combined = f"{result.stdout}\n{result.stderr}"
    assert "index database does not exist" in combined
    assert "aicode index" in combined


def test_default_mode_matches_explicit_lexical(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    assert runner.invoke(app, ["index", str(SEARCH_ROOT), "--db", str(db_path)]).exit_code == 0
    default = runner.invoke(
        app,
        ["search", str(SEARCH_ROOT), "authorize_payment", "--db", str(db_path)],
    )
    explicit = runner.invoke(
        app,
        [
            "search",
            str(SEARCH_ROOT),
            "authorize_payment",
            "--db",
            str(db_path),
            "--mode",
            "lexical",
        ],
    )
    assert default.exit_code == 0
    assert explicit.exit_code == 0
    assert default.stdout == explicit.stdout


def test_index_rejects_missing_and_file_paths(tmp_path: Path) -> None:
    missing = runner.invoke(app, ["index", str(tmp_path / "nope")])
    assert missing.exit_code == 1
    assert "does not exist" in missing.stdout or "does not exist" in missing.stderr

    file_path = SEARCH_ROOT / "payment_gateway.py"
    as_file = runner.invoke(app, ["index", str(file_path)])
    assert as_file.exit_code == 1
    assert "directory" in as_file.stdout or "directory" in as_file.stderr


def test_search_empty_query_and_no_results(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    assert runner.invoke(app, ["index", str(SEARCH_ROOT), "--db", str(db_path)]).exit_code == 0

    empty = runner.invoke(app, ["search", str(SEARCH_ROOT), "   ", "--db", str(db_path)])
    assert empty.exit_code == 0
    assert "No query terms" in empty.stdout

    none = runner.invoke(
        app,
        ["search", str(SEARCH_ROOT), "zzzz_no_such_term_qqq", "--db", str(db_path)],
    )
    assert none.exit_code == 0
    assert "No matching code units" in none.stdout


def test_m1_inspect_and_m2_graph_still_work() -> None:
    inspect_result = runner.invoke(app, ["inspect", str(REPO_ROOT / "simple.py")])
    assert inspect_result.exit_code == 0
    assert "simple.greet" in inspect_result.stdout

    graph_result = runner.invoke(app, ["graph", str(GRAPH_ROOT), "--symbol", "service.Service"])
    assert graph_result.exit_code == 0
    assert "symbol: service.Service" in graph_result.stdout
