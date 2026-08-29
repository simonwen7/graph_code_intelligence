"""Integration tests for the inspect CLI command."""

from pathlib import Path

from typer.testing import CliRunner

from codeintel.cli import app

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "python_repo"
runner = CliRunner()


def test_version_command_still_works() -> None:
    """Existing version behavior must remain intact after adding inspect."""
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "aicode 0.1.0" in result.stdout


def test_inspect_single_file() -> None:
    """Inspecting a Python file should report module symbols."""
    result = runner.invoke(app, ["inspect", str(FIXTURE_ROOT / "simple.py")])

    assert result.exit_code == 0
    assert "module: simple" in result.stdout
    assert "function simple.greet" in result.stdout
    assert "syntax_errors: false" in result.stdout


def test_inspect_directory() -> None:
    """Inspecting a directory should analyze discovered Python files."""
    result = runner.invoke(app, ["inspect", str(FIXTURE_ROOT)])

    assert result.exit_code == 0
    assert "module: package" in result.stdout
    assert "module: package.service" in result.stdout
    assert "module: nested" in result.stdout
    assert "class    nested.Service" in result.stdout


def test_inspect_missing_path() -> None:
    """A missing path should produce a clean non-zero CLI failure."""
    result = runner.invoke(app, ["inspect", str(FIXTURE_ROOT / "does-not-exist.py")])

    assert result.exit_code == 1
    assert "does not exist" in result.stdout or "does not exist" in result.stderr
