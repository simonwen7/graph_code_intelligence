"""Offline CLI tests for structured reranked search mode."""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers.fake_embeddings import FakeEmbeddingProvider
from typer._click.utils import strip_ansi  # type: ignore[attr-defined]
from typer.testing import CliRunner

from codeintel.cli import app
from codeintel.dense import load_dense_documents
from codeintel.storage import IndexDatabase

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "python_rerank"
runner = CliRunner()


def _prepare_fake_provider(db_path: Path) -> FakeEmbeddingProvider:
    with IndexDatabase(db_path, create=False) as database:
        documents = load_dense_documents(database)
    document_vectors: dict[str, list[float]] = {}
    for document in documents:
        qname = document.qualified_name
        if "authorize_payment_transfer" in qname or "describe_payment_transfer" in qname:
            vector = [1.0, 0.0, 0.0, 0.0]
        elif "authorize_payment_filler" in qname:
            vector = [0.85, 0.15, 0.0, 0.0]
        elif "verify_line_bundle" in qname:
            vector = [0.0, 1.0, 0.0, 0.0]
        else:
            vector = [0.2, 0.2, 0.2, 0.2]
        document_vectors[document.document_text] = vector
    query = "authorize payment transfer"
    return FakeEmbeddingProvider(
        dimension=4,
        model_id="fake-model",
        document_vectors=document_vectors,
        query_vectors={query: [1.0, 0.0, 0.0, 0.0]},
        default_document=[0.1, 0.1, 0.1, 0.1],
        default_query=[0.1, 0.1, 0.1, 0.1],
    )


def test_reranked_cli_and_explain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "index.db"
    dense_dir = tmp_path / "dense"
    assert runner.invoke(app, ["index", str(FIXTURE), "--db", str(db_path)]).exit_code == 0
    base = _prepare_fake_provider(db_path)

    def factory(model_id: str = "fake-model") -> FakeEmbeddingProvider:
        return FakeEmbeddingProvider(
            dimension=base.dimension,
            provider_id=base.provider_id,
            model_id=model_id,
            document_vectors={
                text: vector.tolist()
                for text, vector in base._document_vectors.items()  # noqa: SLF001
            },
            query_vectors={
                text: vector.tolist()
                for text, vector in base._query_vectors.items()  # noqa: SLF001
            },
            default_document=base._default_document.tolist(),  # noqa: SLF001
            default_query=base._default_query.tolist(),  # noqa: SLF001
        )

    monkeypatch.setattr("codeintel.cli.create_embedding_provider", factory)
    embedded = runner.invoke(
        app,
        [
            "embed",
            str(FIXTURE),
            "--db",
            str(db_path),
            "--dense-dir",
            str(dense_dir),
            "--model",
            "fake-model",
        ],
    )
    assert embedded.exit_code == 0, embedded.stdout + embedded.stderr

    help_result = runner.invoke(app, ["search", "--help"], color=False)
    assert help_result.exit_code == 0
    help_text = strip_ansi(help_result.stdout)
    assert "reranked" in help_text
    assert "--explain" in help_text

    query = "authorize payment transfer"
    compact = runner.invoke(
        app,
        [
            "search",
            str(FIXTURE),
            query,
            "--db",
            str(db_path),
            "--dense-dir",
            str(dense_dir),
            "--mode",
            "reranked",
            "--limit",
            "10",
        ],
    )
    assert compact.exit_code == 0, compact.stdout + compact.stderr
    assert "authorize_payment_transfer" in compact.stdout
    assert "rerank:" not in compact.stdout

    explained = runner.invoke(
        app,
        [
            "search",
            str(FIXTURE),
            query,
            "--db",
            str(db_path),
            "--dense-dir",
            str(dense_dir),
            "--mode",
            "reranked",
            "--explain",
            "--limit",
            "10",
        ],
    )
    assert explained.exit_code == 0, explained.stdout + explained.stderr
    assert "rerank:" in explained.stdout
    assert "contributions:" in explained.stdout
    assert "graph_base" in explained.stdout

    for mode in ("lexical", "dense", "hybrid", "graph"):
        bad = runner.invoke(
            app,
            [
                "search",
                str(FIXTURE),
                query,
                "--db",
                str(db_path),
                "--dense-dir",
                str(dense_dir),
                "--mode",
                mode,
                "--explain",
            ],
        )
        assert bad.exit_code == 1
        assert "reranked" in f"{bad.stdout}\n{bad.stderr}".lower()

    missing = runner.invoke(
        app,
        [
            "search",
            str(FIXTURE),
            query,
            "--db",
            str(db_path),
            "--dense-dir",
            str(tmp_path / "missing"),
            "--mode",
            "reranked",
        ],
    )
    assert missing.exit_code == 1
    assert "embed" in f"{missing.stdout}\n{missing.stderr}".lower()

    inspect_graph = runner.invoke(app, ["graph", str(FIXTURE)])
    assert inspect_graph.exit_code == 0
    assert "relations:" in inspect_graph.stdout
