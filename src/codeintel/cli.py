"""Command-line interface for the code intelligence engine."""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from codeintel import __version__
from codeintel.dense import (
    DenseIndexError,
    DenseIndexMismatchError,
    DenseIndexMissingError,
    build_dense_index,
    default_dense_dir,
)
from codeintel.discovery import discover_source_files
from codeintel.embeddings import (
    DEFAULT_MODEL_ID,
    EmbeddingDependencyError,
    create_embedding_provider,
)
from codeintel.graph import CodeGraph
from codeintel.hybrid import search_hybrid
from codeintel.languages.selection import SourceLanguage, create_language_tools
from codeintel.lexical import search_code_units
from codeintel.models import (
    AnalysisResult,
    Relation,
    RerankedResult,
    RerankExplanation,
    SearchResult,
    SymbolKind,
)
from codeintel.repository import RepositoryAnalysis, analyze_repository
from codeintel.storage import (
    IndexDatabase,
    IndexDatabaseError,
    SchemaVersionError,
    default_index_path,
)

_KIND_HELP = "Optional SymbolKind filter (module, namespace, class, function, method)."
_KIND_ERROR = "Error: --kind must be one of: module, namespace, class, function, method."
_LANGUAGE_HELP = "Source language for analysis (default: python). One language per index."


app = typer.Typer(
    name="aicode",
    help="Graph-Augmented Code Intelligence Engine.",
    no_args_is_help=True,
)


class SearchMode(StrEnum):
    """Retrieval mode for ``aicode search``."""

    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID = "hybrid"
    GRAPH = "graph"
    RERANKED = "reranked"


@app.callback()
def callback() -> None:
    """Graph-Augmented Code Intelligence Engine."""


@app.command()
def version() -> None:
    """Display the installed project version."""
    typer.echo(f"aicode {__version__}")


@app.command("inspect")
def inspect_command(
    path: Path,
    language: Annotated[
        SourceLanguage,
        typer.Option("--language", help=_LANGUAGE_HELP),
    ] = SourceLanguage.PYTHON,
) -> None:
    """Inspect Symbols and CodeUnits extracted from source files."""
    if not path.exists():
        typer.echo(f"Error: path does not exist: {path}", err=True)
        raise typer.Exit(code=1)

    adapter, _extractor = create_language_tools(language)
    try:
        files = discover_source_files(path, adapter)
    except OSError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not files:
        label = "Python" if language is SourceLanguage.PYTHON else "C++"
        typer.echo(f"No supported {label} source files found under: {path}")
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
    language: Annotated[
        SourceLanguage,
        typer.Option("--language", help=_LANGUAGE_HELP),
    ] = SourceLanguage.PYTHON,
) -> None:
    """Inspect the in-memory code graph for a repository directory."""
    if not path.exists():
        typer.echo(f"Error: path does not exist: {path}", err=True)
        raise typer.Exit(code=1)
    if path.is_file():
        typer.echo("Error: graph analysis expects a repository directory, not a file.", err=True)
        raise typer.Exit(code=1)

    adapter, extractor = create_language_tools(language)
    try:
        analysis = analyze_repository(path, adapter, extractor)
    except (OSError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not analysis.files:
        label = "Python" if language is SourceLanguage.PYTHON else "C++"
        typer.echo(f"No supported {label} source files found under: {path}")
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
    full: Annotated[
        bool,
        typer.Option(
            "--full",
            help=(
                "Force a full schema-v2 rebuild (required to upgrade unsupported schema versions "
                "or intentionally switch index language)."
            ),
        ),
    ] = False,
    language: Annotated[
        SourceLanguage,
        typer.Option("--language", help=_LANGUAGE_HELP),
    ] = SourceLanguage.PYTHON,
) -> None:
    """Build or incrementally update a persistent SQLite lexical index."""
    from codeintel.indexing import IndexLanguageError, index_repository

    if not path.exists():
        typer.echo(f"Error: path does not exist: {path}", err=True)
        raise typer.Exit(code=1)
    if not path.is_dir():
        typer.echo("Error: index expects a repository directory, not a file.", err=True)
        raise typer.Exit(code=1)

    adapter, extractor = create_language_tools(language)
    database_path = db if db is not None else default_index_path(path)
    try:
        stats = index_repository(
            path,
            adapter,
            extractor,
            database_path=database_path,
            full=full,
        )
    except SchemaVersionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except IndexLanguageError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except (OSError, ValueError, IndexDatabaseError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"database: {database_path}")
    typer.echo(f"mode: {stats.mode}")
    typer.echo(
        "files "
        f"added={stats.files_added} changed={stats.files_changed} "
        f"deleted={stats.files_deleted} unchanged={stats.files_unchanged}"
    )
    typer.echo(f"analyzed={stats.files_analyzed}")
    typer.echo(f"relation_files_recomputed={stats.relation_files_recomputed}")
    typer.echo(f"symbols_rewritten={stats.symbols_rewritten}")
    typer.echo(f"code_units_rewritten={stats.code_units_rewritten}")
    typer.echo(f"files: {stats.files}")
    typer.echo(f"symbols: {stats.symbols}")
    typer.echo(f"code_units: {stats.code_units}")
    typer.echo(f"relations: {stats.relations}")
    typer.echo(f"fts_documents: {stats.fts_documents}")


@app.command("embed")
def embed_command(
    path: Path,
    db: Annotated[
        Path | None,
        typer.Option("--db", help="SQLite index path (default: PATH/.codeintel/index.db)."),
    ] = None,
    dense_dir: Annotated[
        Path | None,
        typer.Option(
            "--dense-dir",
            help="Dense artifact directory (default: PATH/.codeintel/dense/).",
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", help="Embedding model id for the Sentence Transformers provider."),
    ] = DEFAULT_MODEL_ID,
    full: Annotated[
        bool,
        typer.Option(
            "--full",
            help="Ignore vector reuse and re-embed every current dense document.",
        ),
    ] = False,
) -> None:
    """Build a dense FAISS artifact from an existing SQLite index."""
    if not path.exists():
        typer.echo(f"Error: path does not exist: {path}", err=True)
        raise typer.Exit(code=1)
    if not path.is_dir():
        typer.echo("Error: embed expects a repository directory, not a file.", err=True)
        raise typer.Exit(code=1)

    database_path = db if db is not None else default_index_path(path)
    artifact_dir = dense_dir if dense_dir is not None else default_dense_dir(path)
    if not database_path.exists():
        typer.echo(
            f"Error: index database does not exist: {database_path}\n"
            f"Run `aicode index {path}` first.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        provider = create_embedding_provider(model)
        with IndexDatabase(database_path, create=False) as database:
            stats = build_dense_index(
                database,
                provider,
                artifact_dir=artifact_dir,
                full=full,
            )
    except EmbeddingDependencyError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except SchemaVersionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except (IndexDatabaseError, DenseIndexError, OSError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"database: {database_path}")
    typer.echo(f"dense_dir: {stats.artifact_dir}")
    typer.echo(f"provider: {stats.provider_id}")
    typer.echo(f"model: {stats.model_id}")
    typer.echo(f"documents_total: {stats.document_count}")
    typer.echo(f"vectors_reused: {stats.vectors_reused}")
    typer.echo(f"vectors_embedded: {stats.vectors_embedded}")
    typer.echo(f"dimension: {stats.dimension}")
    typer.echo(f"corpus_fingerprint: {stats.corpus_fingerprint}")
    if not stats.rewritten:
        typer.echo("artifact: unchanged (all vectors reusable)")


@app.command("search")
def search_command(
    path: Path,
    query: str,
    db: Annotated[
        Path | None,
        typer.Option("--db", help="SQLite index path (default: PATH/.codeintel/index.db)."),
    ] = None,
    dense_dir: Annotated[
        Path | None,
        typer.Option(
            "--dense-dir",
            help="Dense artifact directory (default: PATH/.codeintel/dense/).",
        ),
    ] = None,
    mode: Annotated[
        SearchMode,
        typer.Option(
            "--mode",
            help=(
                "Retrieval mode: lexical, dense, hybrid, graph "
                "(graph-augmented hybrid), or reranked."
            ),
        ),
    ] = SearchMode.LEXICAL,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Maximum number of results.")] = 10,
    kind: Annotated[
        str | None,
        typer.Option(
            "--kind",
            help=_KIND_HELP,
        ),
    ] = None,
    path_prefix: Annotated[
        str | None,
        typer.Option("--path-prefix", help="Optional repository-relative path prefix filter."),
    ] = None,
    explain: Annotated[
        bool,
        typer.Option(
            "--explain",
            help="Print structured rerank explanations (requires --mode reranked).",
        ),
    ] = False,
) -> None:
    """Search a previously built persistent CodeUnit index."""
    if not path.exists():
        typer.echo(f"Error: path does not exist: {path}", err=True)
        raise typer.Exit(code=1)
    if not path.is_dir():
        typer.echo("Error: search expects a repository directory, not a file.", err=True)
        raise typer.Exit(code=1)

    if explain and mode is not SearchMode.RERANKED:
        typer.echo(
            "Error: --explain is only supported with --mode reranked.",
            err=True,
        )
        raise typer.Exit(code=1)

    if not query.strip():
        typer.echo("No query terms provided.")
        raise typer.Exit(code=0)

    database_path = db if db is not None else default_index_path(path)
    artifact_dir = dense_dir if dense_dir is not None else default_dense_dir(path)
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
                _KIND_ERROR,
                err=True,
            )
            raise typer.Exit(code=1) from exc

    try:
        with IndexDatabase(database_path, create=False) as database:
            if mode is SearchMode.LEXICAL:
                results = search_code_units(
                    database,
                    query,
                    limit=limit,
                    kind=symbol_kind,
                    path_prefix=path_prefix,
                )
                if not results:
                    typer.echo("No matching code units.")
                    raise typer.Exit(code=0)
                for rank, result in enumerate(results, start=1):
                    _print_search_result(rank, result)
                return

            if mode is SearchMode.RERANKED:
                reranked = _search_reranked(
                    database,
                    query,
                    artifact_dir=artifact_dir,
                    limit=limit,
                    kind=symbol_kind,
                    path_prefix=path_prefix,
                )
                if not reranked:
                    typer.echo("No matching code units.")
                    raise typer.Exit(code=0)
                for rank, item in enumerate(reranked, start=1):
                    _print_search_result(rank, item.result)
                    if explain:
                        _print_rerank_explanation(item.explanation)
                return

            results = _search_with_embeddings(
                database,
                query,
                mode=mode,
                artifact_dir=artifact_dir,
                limit=limit,
                kind=symbol_kind,
                path_prefix=path_prefix,
            )
    except EmbeddingDependencyError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except DenseIndexMissingError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except DenseIndexMismatchError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except DenseIndexError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
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


def _search_with_embeddings(
    database: IndexDatabase,
    query: str,
    *,
    mode: SearchMode,
    artifact_dir: Path,
    limit: int,
    kind: SymbolKind | None,
    path_prefix: str | None,
) -> tuple[SearchResult, ...]:
    import json

    from codeintel.dense import search_dense as dense_search

    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.is_file():
        raise DenseIndexMissingError(
            f"Dense artifact is missing under {artifact_dir}. Run `aicode embed` to build it."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DenseIndexError(f"Corrupt dense metadata at {metadata_path}") from exc
    if not isinstance(metadata, dict) or "model_id" not in metadata:
        raise DenseIndexError(f"Corrupt dense metadata at {metadata_path}")

    provider = create_embedding_provider(str(metadata["model_id"]))
    if mode is SearchMode.DENSE:
        return dense_search(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=limit,
            kind=kind,
            path_prefix=path_prefix,
        )
    if mode is SearchMode.HYBRID:
        return search_hybrid(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=limit,
            kind=kind,
            path_prefix=path_prefix,
        )
    if mode is SearchMode.GRAPH:
        from codeintel.graph_retrieval import search_graph_augmented

        return search_graph_augmented(
            database,
            provider,
            query,
            artifact_dir=artifact_dir,
            limit=limit,
            kind=kind,
            path_prefix=path_prefix,
        )
    raise ValueError(f"Unsupported embedding search mode: {mode.value}")


def _search_reranked(
    database: IndexDatabase,
    query: str,
    *,
    artifact_dir: Path,
    limit: int,
    kind: SymbolKind | None,
    path_prefix: str | None,
) -> tuple[RerankedResult, ...]:
    import json

    from codeintel.reranking import search_reranked

    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.is_file():
        raise DenseIndexMissingError(
            f"Dense artifact is missing under {artifact_dir}. Run `aicode embed` to build it."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DenseIndexError(f"Corrupt dense metadata at {metadata_path}") from exc
    if not isinstance(metadata, dict) or "model_id" not in metadata:
        raise DenseIndexError(f"Corrupt dense metadata at {metadata_path}")

    provider = create_embedding_provider(str(metadata["model_id"]))
    return search_reranked(
        database,
        provider,
        query,
        artifact_dir=artifact_dir,
        limit=limit,
        kind=kind,
        path_prefix=path_prefix,
    )


def _print_search_result(rank: int, result: SearchResult) -> None:
    signature = result.signature if result.signature is not None else "-"
    preview = _source_preview(result.source_text)
    typer.echo(f"{rank}. score={result.score:.6f}")
    typer.echo(f"   {result.kind.value} {result.symbol_qualified_name}")
    typer.echo(f"   {result.path.as_posix()}:L{result.span.start_line}-{result.span.end_line}")
    typer.echo(f"   signature: {signature}")
    typer.echo(f"   preview: {preview}")


def _print_rerank_explanation(explanation: RerankExplanation) -> None:
    delta = explanation.rank_delta
    delta_text = f"+{delta}" if delta > 0 else str(delta)
    typer.echo(
        f"   rerank: #{explanation.original_rank} -> #{explanation.final_rank} (delta {delta_text})"
    )
    typer.echo("   contributions:")
    for contribution in explanation.contributions:
        typer.echo(
            "     "
            f"{contribution.source.value} rank={contribution.rank} "
            f"rrf={contribution.rrf_contribution:.6f}"
        )
    if explanation.relation_evidence:
        typer.echo("   evidence:")
        for evidence in explanation.relation_evidence:
            typer.echo(
                "     "
                f"{evidence.seed_qualified_name} [rank {evidence.seed_rank}] "
                f"{evidence.relation_kind.value} {evidence.direction.value}"
            )


@app.command("context")
def context_command(
    path: Path,
    query: str,
    budget: Annotated[
        int,
        typer.Option(
            "--budget",
            help=(
                "Required estimated token budget (simple-lexical-v1 units; "
                "not a vendor LLM tokenizer)."
            ),
        ),
    ],
    db: Annotated[
        Path | None,
        typer.Option("--db", help="SQLite index path (default: PATH/.codeintel/index.db)."),
    ] = None,
    dense_dir: Annotated[
        Path | None,
        typer.Option(
            "--dense-dir",
            help="Dense artifact directory (default: PATH/.codeintel/dense/).",
        ),
    ] = None,
    kind: Annotated[
        str | None,
        typer.Option(
            "--kind",
            help=_KIND_HELP,
        ),
    ] = None,
    path_prefix: Annotated[
        str | None,
        typer.Option("--path-prefix", help="Optional repository-relative path prefix filter."),
    ] = None,
) -> None:
    """Compile a token-budgeted context from reranked CodeUnits."""
    from codeintel.context import CONTEXT_CANDIDATE_LIMIT, SimpleTokenCounter, compile_context

    if not path.exists():
        typer.echo(f"Error: path does not exist: {path}", err=True)
        raise typer.Exit(code=1)
    if not path.is_dir():
        typer.echo("Error: context expects a repository directory, not a file.", err=True)
        raise typer.Exit(code=1)
    if budget < 0:
        typer.echo("Error: --budget must be >= 0.", err=True)
        raise typer.Exit(code=1)
    if not query.strip():
        typer.echo("No query terms provided.")
        raise typer.Exit(code=0)

    database_path = db if db is not None else default_index_path(path)
    artifact_dir = dense_dir if dense_dir is not None else default_dense_dir(path)
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
                _KIND_ERROR,
                err=True,
            )
            raise typer.Exit(code=1) from exc

    try:
        with IndexDatabase(database_path, create=False) as database:
            reranked = _search_reranked(
                database,
                query,
                artifact_dir=artifact_dir,
                limit=CONTEXT_CANDIDATE_LIMIT,
                kind=symbol_kind,
                path_prefix=path_prefix,
            )
            compiled = compile_context(
                reranked,
                token_budget=budget,
                token_counter=SimpleTokenCounter(),
            )
    except EmbeddingDependencyError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except DenseIndexMissingError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except DenseIndexMismatchError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except DenseIndexError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except SchemaVersionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except IndexDatabaseError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if compiled.text:
        typer.echo(compiled.text, nl=False)
        if not compiled.text.endswith("\n"):
            typer.echo("")
        typer.echo("")
    else:
        typer.echo("Empty compiled context.")
        typer.echo("")

    typer.echo(
        "summary: "
        f"used={compiled.used_tokens}/{compiled.token_budget} estimated tokens "
        f"({compiled.token_counter_id}); "
        f"selected={compiled.selected_count}/{compiled.candidate_count}; "
        f"omitted={compiled.omitted_count}"
    )


def _source_preview(source_text: str, *, max_chars: int = 120) -> str:
    first_line = source_text.splitlines()[0] if source_text else ""
    compact = " ".join(first_line.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _print_analysis_result(result: AnalysisResult) -> None:
    location = str(result.path) if result.path is not None else "<memory>"
    typer.echo(f"file: {location}")
    typer.echo(f"language: {result.language_id}")
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
