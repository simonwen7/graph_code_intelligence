"""Offline CLI tests for token-budget context compilation."""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers.fake_embeddings import FakeEmbeddingProvider
from typer.testing import CliRunner

from codeintel.cli import app
from codeintel.dense import load_dense_documents
from codeintel.storage import IndexDatabase

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "python_context"
runner = CliRunner()


def _prepare_provider(db_path: Path) -> FakeEmbeddingProvider:
    with IndexDatabase(db_path, create=False) as database:
        documents = load_dense_documents(database)
    document_vectors: dict[str, list[float]] = {}
    for document in documents:
        if "authorize_payment_checkout" in document.qualified_name:
            vector = [1.0, 0.0, 0.0, 0.0]
        elif "huge_payment" in document.qualified_name:
            vector = [0.7, 0.3, 0.0, 0.0]
        else:
            vector = [0.2, 0.2, 0.2, 0.2]
        document_vectors[document.document_text] = vector
    query = "authorize payment checkout"
    return FakeEmbeddingProvider(
        dimension=4,
        model_id="fake-model",
        document_vectors=document_vectors,
        query_vectors={query: [1.0, 0.0, 0.0, 0.0]},
        default_document=[0.1, 0.1, 0.1, 0.1],
        default_query=[0.1, 0.1, 0.1, 0.1],
    )


def test_context_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "index.db"
    dense_dir = tmp_path / "dense"
    assert runner.invoke(app, ["index", str(FIXTURE), "--db", str(db_path)]).exit_code == 0
    base = _prepare_provider(db_path)

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

    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "context" in help_result.stdout

    missing_budget = runner.invoke(
        app,
        ["context", str(FIXTURE), "authorize payment checkout", "--db", str(db_path)],
    )
    assert missing_budget.exit_code != 0

    negative = runner.invoke(
        app,
        [
            "context",
            str(FIXTURE),
            "authorize payment checkout",
            "--budget",
            "-1",
            "--db",
            str(db_path),
            "--dense-dir",
            str(dense_dir),
        ],
    )
    assert negative.exit_code == 1
    assert "budget" in f"{negative.stdout}\n{negative.stderr}".lower()

    zero = runner.invoke(
        app,
        [
            "context",
            str(FIXTURE),
            "authorize payment checkout",
            "--budget",
            "0",
            "--db",
            str(db_path),
            "--dense-dir",
            str(dense_dir),
        ],
    )
    assert zero.exit_code == 0, zero.stdout + zero.stderr
    assert "Empty compiled context" in zero.stdout
    assert "simple-lexical-v1" in zero.stdout
    assert "used=0/0" in zero.stdout

    ok = runner.invoke(
        app,
        [
            "context",
            str(FIXTURE),
            "authorize payment checkout",
            "--budget",
            "5000",
            "--db",
            str(db_path),
            "--dense-dir",
            str(dense_dir),
            "--kind",
            "function",
            "--path-prefix",
            "seed_",
        ],
    )
    assert ok.exit_code == 0, ok.stdout + ok.stderr
    assert "=== CODE UNIT ===" in ok.stdout
    assert "authorize_payment_checkout" in ok.stdout
    assert "simple-lexical-v1" in ok.stdout
    assert "summary:" in ok.stdout
    assert "selected=" in ok.stdout

    missing_dense = runner.invoke(
        app,
        [
            "context",
            str(FIXTURE),
            "authorize payment checkout",
            "--budget",
            "100",
            "--db",
            str(db_path),
            "--dense-dir",
            str(tmp_path / "missing"),
        ],
    )
    assert missing_dense.exit_code == 1
    assert "embed" in f"{missing_dense.stdout}\n{missing_dense.stderr}".lower()

    missing_db = runner.invoke(
        app,
        [
            "context",
            str(FIXTURE),
            "authorize payment checkout",
            "--budget",
            "100",
            "--db",
            str(tmp_path / "no.db"),
            "--dense-dir",
            str(dense_dir),
        ],
    )
    assert missing_db.exit_code == 1
    assert "index" in f"{missing_db.stdout}\n{missing_db.stderr}".lower()

    search = runner.invoke(
        app,
        [
            "search",
            str(FIXTURE),
            "authorize payment checkout",
            "--db",
            str(db_path),
            "--dense-dir",
            str(dense_dir),
            "--mode",
            "lexical",
            "--limit",
            "3",
        ],
    )
    assert search.exit_code == 0

    from codeintel.context import CONTEXT_CANDIDATE_LIMIT

    assert CONTEXT_CANDIDATE_LIMIT == 20

    # Candidate pool is fixed at 20; assert CLI wiring uses the constant.
    captured: dict[str, object] = {}

    def fake_search_reranked(*_args: object, **kwargs: object) -> list[object]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr("codeintel.cli._search_reranked", fake_search_reranked)
    limited = runner.invoke(
        app,
        [
            "context",
            str(FIXTURE),
            "authorize payment checkout",
            "--budget",
            "100",
            "--db",
            str(db_path),
            "--dense-dir",
            str(dense_dir),
        ],
    )
    assert limited.exit_code == 0, limited.stdout + limited.stderr
    assert captured.get("limit") == CONTEXT_CANDIDATE_LIMIT
    assert "estimated tokens" in limited.stdout
    assert "simple-lexical-v1" in limited.stdout
