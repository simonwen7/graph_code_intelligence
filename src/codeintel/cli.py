"""Command-line interface for the code intelligence engine."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from codeintel import __version__
from codeintel.discovery import discover_source_files
from codeintel.graph import CodeGraph
from codeintel.languages.python import PythonAdapter, PythonRelationExtractor
from codeintel.lexical import search_code_units
from codeintel.models import AnalysisResult, Relation, SearchResult, SymbolKind
from codeintel.repository import RepositoryAnalysis, analyze_repository
from codeintel.storage import (
    IndexDatabase,
    IndexDatabaseError,
    SchemaVersionError,
    default_index_path,
)

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
    symbol: Annotated[
        str | None,
        typer.Option(
            "--symbol",
            help="Show incoming and outgoing relations for one qualified name.",
        ),
    ] = None,
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


@app.command("index")
def index_command(
    path: Path,
    db: Annotated[
        Path | None,
        typer.Option("--db", help="SQLite index path (default: PATH/.codeintel/index.db)."),
    ] = None,
) -> None:
    """Build a persistent SQLite lexical index for a repository directory."""
    if not path.exists():
        typer.echo(f"Error: path does not exist: {path}", err=True)
        raise typer.Exit(code=1)
    if not path.is_dir():
        typer.echo("Error: index expects a repository directory, not a file.", err=True)
        raise typer.Exit(code=1)

    database_path = db if db is not None else default_index_path(path)
    try:
        analysis = analyze_repository(path, PythonAdapter(), PythonRelationExtractor())
        with IndexDatabase(database_path) as database:
            stats = database.rebuild(analysis)
    except (OSError, ValueError, IndexDatabaseError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"database: {database_path}")
    typer.echo(f"files: {stats.files}")
    typer.echo(f"symbols: {stats.symbols}")
    typer.echo(f"code_units: {stats.code_units}")
    typer.echo(f"relations: {stats.relations}")
    typer.echo(f"fts_documents: {stats.fts_documents}")


@app.command("search")
def search_command(
    path: Path,
    query: str,
    db: Annotated[
        Path | None,
        typer.Option("--db", help="SQLite index path (default: PATH/.codeintel/index.db)."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Maximum number of results.")] = 10,
    kind: Annotated[
        str | None,
        typer.Option(
            "--kind",
            help="Optional SymbolKind filter (module, class, function, method).",
        ),
    ] = None,
    path_prefix: Annotated[
        str | None,
        typer.Option("--path-prefix", help="Optional repository-relative path prefix filter."),
    ] = None,
) -> None:
    """Search a previously built persistent CodeUnit index."""
    if not path.exists():
        typer.echo(f"Error: path does not exist: {path}", err=True)
        raise typer.Exit(code=1)
    if not path.is_dir():
        typer.echo("Error: search expects a repository directory, not a file.", err=True)
        raise typer.Exit(code=1)

    if not query.strip():
        typer.echo("No query terms provided.")
        raise typer.Exit(code=0)

    database_path = db if db is not None else default_index_path(path)
    if not database_path.exists():
        typer.echo(
            f"Error: index database does not exist: {database_path}\n"
            f"Run `aicode index {path}` first.",
            err=True,
        )
        raise typer.Exit(code=1)

    symbol_kind: SymbolKind | None = None
    if kind is not None:
        try:
            symbol_kind = SymbolKind(kind)
        except ValueError as exc:
            typer.echo(
                "Error: --kind must be one of: module, class, function, method.",
                err=True,
            )
            raise typer.Exit(code=1) from exc

    try:
        with IndexDatabase(database_path, create=False) as database:
            results = search_code_units(
                database,
                query,
                limit=limit,
                kind=symbol_kind,
                path_prefix=path_prefix,
            )
    except SchemaVersionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except IndexDatabaseError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not results:
        typer.echo("No matching code units.")
        raise typer.Exit(code=0)

    for rank, result in enumerate(results, start=1):
        _print_search_result(rank, result)


def _print_search_result(rank: int, result: SearchResult) -> None:
    signature = result.signature if result.signature is not None else "-"
    preview = _source_preview(result.source_text)
    typer.echo(f"{rank}. score={result.score:.6f}")
    typer.echo(f"   {result.kind.value} {result.symbol_qualified_name}")
    typer.echo(f"   {result.path.as_posix()}:L{result.span.start_line}-{result.span.end_line}")
    typer.echo(f"   signature: {signature}")
    typer.echo(f"   preview: {preview}")


def _source_preview(source_text: str, *, max_chars: int = 120) -> str:
    first_line = source_text.splitlines()[0] if source_text else ""
    compact = " ".join(first_line.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


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
