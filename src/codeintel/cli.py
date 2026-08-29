"""Command-line interface for the code intelligence engine."""

from __future__ import annotations

from pathlib import Path

import typer

from codeintel import __version__
from codeintel.discovery import discover_source_files
from codeintel.languages.python import PythonAdapter
from codeintel.models import AnalysisResult

app = typer.Typer(
    name="aicode",
    help="Graph-Augmented Code Intelligence Engine.",
    no_args_is_help=True,
)


@app.callback()
def callback() -> None:
    """Graph-Augmented Code Intelligence Engine."""


@app.command()
def version() -> None:
    """Display the installed project version."""
    typer.echo(f"aicode {__version__}")


@app.command("inspect")
def inspect_command(path: Path) -> None:
    """Inspect Symbols and CodeUnits extracted from Python source."""
    if not path.exists():
        typer.echo(f"Error: path does not exist: {path}", err=True)
        raise typer.Exit(code=1)

    adapter = PythonAdapter()
    try:
        files = discover_source_files(path, adapter)
    except OSError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not files:
        typer.echo(f"No supported Python source files found under: {path}")
        raise typer.Exit(code=0)

    repository_root = path if path.is_dir() else None
    for file_path in files:
        result = adapter.analyze_file(file_path, repository_root=repository_root)
        _print_analysis_result(result)


def _print_analysis_result(result: AnalysisResult) -> None:
    location = str(result.path) if result.path is not None else "<memory>"
    typer.echo(f"file: {location}")
    typer.echo(f"module: {result.module_name}")
    typer.echo(f"syntax_errors: {str(result.has_syntax_errors).lower()}")
    typer.echo(f"symbols: {len(result.symbols)}")
    for symbol in result.symbols:
        signature = symbol.signature if symbol.signature is not None else "-"
        typer.echo(
            "  "
            f"{symbol.kind.value:8} "
            f"{symbol.qualified_name} "
            f"L{symbol.span.start_line}-{symbol.span.end_line} "
            f"| {signature}"
        )
    typer.echo(f"code_units: {len(result.code_units)}")
    for unit in result.code_units:
        typer.echo(
            "  "
            f"{unit.kind.value:8} "
            f"{unit.symbol_qualified_name} "
            f"L{unit.span.start_line}-{unit.span.end_line} "
            f"({len(unit.source_text.encode('utf-8'))} bytes)"
        )
    typer.echo("")


def main() -> None:
    """Run the command-line application."""
    app()


if __name__ == "__main__":
    main()
