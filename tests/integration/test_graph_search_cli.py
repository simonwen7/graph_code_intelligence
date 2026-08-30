"""Offline CLI tests for graph-augmented search mode."""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers.fake_embeddings import FakeEmbeddingProvider
from typer.testing import CliRunner

from codeintel.cli import app
from codeintel.dense import load_dense_documents
from codeintel.storage import IndexDatabase

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "python_graph_search"
runner = CliRunner()


def test_graph_mode_cli_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "index.db"
    dense_dir = tmp_path / "dense"
    assert runner.invoke(app, ["index", str(FIXTURE), "--db", str(db_path)]).exit_code == 0

    with IndexDatabase(db_path, create=False) as database:
        documents = load_dense_documents(database)
    document_vectors: dict[str, list[float]] = {}
    for document in documents:
        if "handle_payment_checkout" in document.qualified_name:
            vector = [1.0, 0.0, 0.0, 0.0]
        elif "authorize_payment_filler" in document.qualified_name:
            vector = [0.9, 0.1, 0.0, 0.0]
        elif "verify_basket" in document.qualified_name:
            vector = [0.0, 1.0, 0.0, 0.0]
        else:
            vector = [0.2, 0.2, 0.2, 0.2]
        document_vectors[document.document_text] = vector
    query = "authorize payment checkout"
    base = FakeEmbeddingProvider(
        dimension=4,
        model_id="fake-model",
        document_vectors=document_vectors,
        query_vectors={query: [1.0, 0.0, 0.0, 0.0]},
        default_document=[0.1, 0.1, 0.1, 0.1],
        default_query=[0.1, 0.1, 0.1, 0.1],
    )

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

    graph = runner.invoke(
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
            "graph",
            "--limit",
            "15",
        ],
    )
    assert graph.exit_code == 0, graph.stdout + graph.stderr
    assert "checkout_handler.handle_payment_checkout" in graph.stdout
    assert "cart_rules.verify_basket_line_items" in graph.stdout

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
            "graph",
        ],
    )
    assert missing.exit_code == 1
    assert "embed" in f"{missing.stdout}\n{missing.stderr}".lower()

    # Existing inspect-style graph command still works and is distinct.
    inspect_graph = runner.invoke(app, ["graph", str(FIXTURE)])
    assert inspect_graph.exit_code == 0
    assert "relations:" in inspect_graph.stdout
