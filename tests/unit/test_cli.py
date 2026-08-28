"""Tests for the command-line interface."""

from typer.testing import CliRunner

from codeintel.cli import app

runner = CliRunner()


def test_version_command() -> None:
    """The CLI should report the current application version."""
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "aicode 0.1.0" in result.stdout
