"""Command-line interface for the code intelligence engine."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer

from codeintel import __version__
from codeintel.discovery import discover_source_files
from codeintel.graph import CodeGraph
from codeintel.languages.python import PythonAdapter, PythonRelationExtractor
from codeintel.models import AnalysisResult, Relation
from codeintel.repository import RepositoryAnalysis, analyze_repository

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


@app.command("graph")
def graph_command(
    path: Path,
    symbol: str | None = typer.Option(
        None,
        "--symbol",
        help="Show incoming and outgoing relations for one qualified name.",
    ),
) -> None:
    """Inspect the in-memory code graph for a repository directory."""
    if not path.exists():
        typer.echo(f"Error: path does not exist: {path}", err=True)
        raise typer.Exit(code=1)
    if path.is_file():
        typer.echo("Error: graph analysis expects a repository directory, not a file.", err=True)
        raise typer.Exit(code=1)

    try:
        analysis = analyze_repository(path, PythonAdapter(), PythonRelationExtractor())
    except (OSError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not analysis.files:
        typer.echo(f"No supported Python source files found under: {path}")
        raise typer.Exit(code=0)

    if symbol is not None:
        _print_symbol_view(analysis.graph, symbol)
        return

    _print_graph_summary(analysis)


def _print_analysis_result(result: AnalysisResult) -> None:
    location = str(result.path) if result.path is not None else "<memory>"
    typer.echo(f"file: {location}")
    typer.echo(f"module: {result.module_name}")
    typer.echo(f"syntax_errors: {str(result.has_syntax_errors).lower()}")
    typer.echo(f"symbols: {len(result.symbols)}")
    for symbol_item in result.symbols:
        signature = symbol_item.signature if symbol_item.signature is not None else "-"
        typer.echo(
            "  "
            f"{symbol_item.kind.value:8} "
            f"{symbol_item.qualified_name} "
            f"L{symbol_item.span.start_line}-{symbol_item.span.end_line} "
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


def _print_graph_summary(analysis: RepositoryAnalysis) -> None:
    graph = analysis.graph
    typer.echo(f"repository: {analysis.root}")
    typer.echo(f"files: {len(analysis.files)}")
    typer.echo(f"symbols: {len(graph.symbols)}")
    typer.echo(f"relations: {len(graph.relations)}")
    kind_counts = Counter(relation.kind.value for relation in graph.relations)
    status_counts = Counter(relation.resolution.value for relation in graph.relations)
    typer.echo("by_kind:")
    for kind in ("contains", "imports", "references", "calls", "inherits"):
        typer.echo(f"  {kind}: {kind_counts.get(kind, 0)}")
    typer.echo("by_resolution:")
    for status in ("resolved", "probable", "unresolved"):
        typer.echo(f"  {status}: {status_counts.get(status, 0)}")
    typer.echo("relations:")
    for relation in graph.relations:
        typer.echo(f"  {_format_relation(relation)}")


def _print_symbol_view(graph: CodeGraph, qualified_name: str) -> None:
    if not graph.has_symbol(qualified_name):
        typer.echo(f"Error: unknown symbol: {qualified_name}", err=True)
        raise typer.Exit(code=1)
    symbol = graph.get_symbol(qualified_name)
    typer.echo(f"symbol: {symbol.qualified_name}")
    typer.echo(f"kind: {symbol.kind.value}")
    typer.echo("outgoing:")
    outgoing = graph.outgoing(qualified_name)
    if not outgoing:
        typer.echo("  (none)")
    for relation in outgoing:
        typer.echo(f"  {_format_relation(relation)}")
    typer.echo("incoming:")
    incoming = graph.incoming(qualified_name)
    if not incoming:
        typer.echo("  (none)")
    for relation in incoming:
        typer.echo(f"  {_format_relation(relation)}")


def _format_relation(relation: Relation) -> str:
    if relation.target_qualified_name is not None:
        target = relation.target_qualified_name
    else:
        target = relation.target_text
    return (
        f"{relation.source_qualified_name} "
        f"--{relation.kind.value}[{relation.resolution.value}]--> "
        f"{target}"
    )


def main() -> None:
    """Run the command-line application."""
    app()


if __name__ == "__main__":
    main()
