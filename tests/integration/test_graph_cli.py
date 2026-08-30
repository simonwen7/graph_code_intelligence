"""Integration tests for the graph CLI command."""

from pathlib import Path

from typer.testing import CliRunner

from codeintel.cli import app

GRAPH_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "python_graph"
runner = CliRunner()


def test_help_lists_graph_command() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "graph" in result.stdout
    assert "inspect" in result.stdout
    assert "version" in result.stdout


def test_graph_directory_summary() -> None:
    result = runner.invoke(app, ["graph", str(GRAPH_ROOT)])

    assert result.exit_code == 0
    assert "repository:" in result.stdout
    assert "by_kind:" in result.stdout
    assert "contains:" in result.stdout
    assert "--calls[" in result.stdout


def test_graph_symbol_view() -> None:
    result = runner.invoke(app, ["graph", str(GRAPH_ROOT), "--symbol", "service.Service"])

    assert result.exit_code == 0
    assert "symbol: service.Service" in result.stdout
    assert "outgoing:" in result.stdout
    assert "incoming:" in result.stdout


def test_graph_unknown_symbol() -> None:
    result = runner.invoke(app, ["graph", str(GRAPH_ROOT), "--symbol", "missing.Name"])

    assert result.exit_code == 1
    assert "unknown symbol" in result.stdout or "unknown symbol" in result.stderr


def test_graph_rejects_file_path() -> None:
    result = runner.invoke(app, ["graph", str(GRAPH_ROOT / "helpers.py")])

    assert result.exit_code == 1
    assert "repository directory" in result.stdout or "repository directory" in result.stderr


def test_graph_missing_path() -> None:
    result = runner.invoke(app, ["graph", str(GRAPH_ROOT / "does-not-exist")])

    assert result.exit_code == 1
    assert "does not exist" in result.stdout or "does not exist" in result.stderr


def test_graph_empty_directory(tmp_path: Path) -> None:
    result = runner.invoke(app, ["graph", str(tmp_path)])

    assert result.exit_code == 0
    assert "No supported Python source files found under:" in result.stdout
