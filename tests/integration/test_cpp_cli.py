"""CLI and retrieval integration tests for C++ indexes."""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers.fake_embeddings import FakeEmbeddingProvider
from typer.testing import CliRunner

from codeintel.cli import app
from codeintel.dense import load_dense_documents
from codeintel.storage import IndexDatabase

CPP_REPO = Path(__file__).resolve().parents[1] / "fixtures" / "cpp_repo"
CPP_GRAPH = Path(__file__).resolve().parents[1] / "fixtures" / "cpp_graph"
runner = CliRunner()


def test_inspect_and_graph_cli_language_cpp() -> None:
    inspect_result = runner.invoke(
        app, ["inspect", str(CPP_REPO / "pricing.cpp"), "--language", "cpp"]
    )
    assert inspect_result.exit_code == 0, inspect_result.stdout + inspect_result.stderr
    assert "language: cpp" in inspect_result.stdout
    assert "pricing::calculate(int)" in inspect_result.stdout

    graph_result = runner.invoke(app, ["graph", str(CPP_GRAPH), "--language", "cpp"])
    assert graph_result.exit_code == 0, graph_result.stdout + graph_result.stderr
    assert "imports" in graph_result.stdout
    assert "calls" in graph_result.stdout

    symbol_result = runner.invoke(
        app,
        ["graph", str(CPP_GRAPH), "--language", "cpp", "--symbol", "tools::Derived::run()"],
    )
    assert symbol_result.exit_code == 0, symbol_result.stdout + symbol_result.stderr
    assert "symbol: tools::Derived::run()" in symbol_result.stdout


def test_default_inspect_still_python() -> None:
    result = runner.invoke(
        app, ["inspect", str(Path(__file__).resolve().parents[1] / "fixtures" / "python_repo")]
    )
    assert result.exit_code == 0
    assert "language: python" in result.stdout


def test_cpp_index_embed_search_context_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "index.db"
    dense_dir = tmp_path / "dense"
    index = runner.invoke(
        app,
        ["index", str(CPP_REPO), "--language", "cpp", "--db", str(db_path)],
    )
    assert index.exit_code == 0, index.stdout + index.stderr
    assert "files:" in index.stdout

    with IndexDatabase(db_path, create=False) as database:
        documents = load_dense_documents(database)
    assert documents
    document_vectors = {document.document_text: [1.0, 0.0, 0.0, 0.0] for document in documents}
    query = "calculate quantity"
    provider = FakeEmbeddingProvider(
        dimension=4,
        model_id="fake-model",
        document_vectors=document_vectors,
        query_vectors={query: [1.0, 0.0, 0.0, 0.0]},
        default_document=[0.1, 0.1, 0.1, 0.1],
        default_query=[0.1, 0.1, 0.1, 0.1],
    )

    def factory(model_id: str = "fake-model") -> FakeEmbeddingProvider:
        return FakeEmbeddingProvider(
            dimension=provider.dimension,
            provider_id=provider.provider_id,
            model_id=model_id,
            document_vectors={
                text: vector.tolist()
                for text, vector in provider._document_vectors.items()  # noqa: SLF001
            },
            query_vectors={
                text: vector.tolist()
                for text, vector in provider._query_vectors.items()  # noqa: SLF001
            },
            default_document=provider._default_document.tolist(),  # noqa: SLF001
            default_query=provider._default_query.tolist(),  # noqa: SLF001
        )

    monkeypatch.setattr("codeintel.cli.create_embedding_provider", factory)
    embedded = runner.invoke(
        app,
        [
            "embed",
            str(CPP_REPO),
            "--db",
            str(db_path),
            "--dense-dir",
            str(dense_dir),
            "--model",
            "fake-model",
        ],
    )
    assert embedded.exit_code == 0, embedded.stdout + embedded.stderr

    lexical = runner.invoke(
        app,
        ["search", str(CPP_REPO), "calculate", "--db", str(db_path), "--limit", "5"],
    )
    assert lexical.exit_code == 0, lexical.stdout + lexical.stderr
    assert "calculate" in lexical.stdout

    for mode in ("dense", "hybrid", "graph", "reranked"):
        result = runner.invoke(
            app,
            [
                "search",
                str(CPP_REPO),
                query,
                "--db",
                str(db_path),
                "--dense-dir",
                str(dense_dir),
                "--mode",
                mode,
                "--limit",
                "5",
            ],
        )
        assert result.exit_code == 0, mode + result.stdout + result.stderr

    context = runner.invoke(
        app,
        [
            "context",
            str(CPP_REPO),
            query,
            "--budget",
            "400",
            "--db",
            str(db_path),
            "--dense-dir",
            str(dense_dir),
        ],
    )
    assert context.exit_code == 0, context.stdout + context.stderr
    assert "CODE UNIT" in context.stdout
