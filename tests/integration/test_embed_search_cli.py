"""Offline CLI tests for embed and dense/hybrid search modes."""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers.fake_embeddings import FakeEmbeddingProvider
from typer.testing import CliRunner

from codeintel.cli import app
from codeintel.dense import load_dense_documents
from codeintel.storage import IndexDatabase

DENSE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "python_dense"
runner = CliRunner()


def _provider_matching_db(db_path: Path, query: str, preferred_qname: str) -> FakeEmbeddingProvider:
    with IndexDatabase(db_path, create=False) as database:
        documents = load_dense_documents(database)
    document_vectors: dict[str, list[float]] = {}
    for offset, document in enumerate(documents):
        vector = [0.0] * 8
        vector[offset % 8] = 1.0
        document_vectors[document.document_text] = vector
    preferred = next(doc for doc in documents if doc.qualified_name == preferred_qname)
    return FakeEmbeddingProvider(
        dimension=8,
        model_id="fake-model",
        document_vectors=document_vectors,
        query_vectors={query: document_vectors[preferred.document_text]},
        default_document=[0.01] * 8,
        default_query=[0.01] * 8,
    )


def test_embed_and_mode_search_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "index.db"
    dense_dir = tmp_path / "dense"
    assert runner.invoke(app, ["index", str(DENSE_ROOT), "--db", str(db_path)]).exit_code == 0

    query = "check whether a login session is still valid"
    preferred = "auth_session.refresh_access_token"
    provider = _provider_matching_db(db_path, query, preferred)

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

    missing_embed = runner.invoke(
        app,
        [
            "embed",
            str(DENSE_ROOT),
            "--db",
            str(tmp_path / "missing.db"),
            "--dense-dir",
            str(dense_dir),
        ],
    )
    assert missing_embed.exit_code == 1
    assert "index database does not exist" in f"{missing_embed.stdout}\n{missing_embed.stderr}"

    embedded = runner.invoke(
        app,
        [
            "embed",
            str(DENSE_ROOT),
            "--db",
            str(db_path),
            "--dense-dir",
            str(dense_dir),
            "--model",
            "fake-model",
        ],
    )
    assert embedded.exit_code == 0, embedded.stdout + embedded.stderr
    assert (dense_dir / "index.faiss").is_file()
    assert (dense_dir / "metadata.json").is_file()
    assert "documents_total:" in embedded.stdout

    lexical = runner.invoke(
        app,
        [
            "search",
            str(DENSE_ROOT),
            "refresh_access_token",
            "--db",
            str(db_path),
            "--mode",
            "lexical",
        ],
    )
    assert lexical.exit_code == 0
    assert preferred in lexical.stdout

    default_mode = runner.invoke(
        app,
        ["search", str(DENSE_ROOT), "refresh_access_token", "--db", str(db_path)],
    )
    assert default_mode.exit_code == 0
    assert preferred in default_mode.stdout

    dense = runner.invoke(
        app,
        [
            "search",
            str(DENSE_ROOT),
            query,
            "--db",
            str(db_path),
            "--dense-dir",
            str(dense_dir),
            "--mode",
            "dense",
        ],
    )
    assert dense.exit_code == 0, dense.stdout + dense.stderr
    assert preferred in dense.stdout

    hybrid = runner.invoke(
        app,
        [
            "search",
            str(DENSE_ROOT),
            query,
            "--db",
            str(db_path),
            "--dense-dir",
            str(dense_dir),
            "--mode",
            "hybrid",
        ],
    )
    assert hybrid.exit_code == 0, hybrid.stdout + hybrid.stderr
    assert preferred in hybrid.stdout

    invalid = runner.invoke(
        app,
        ["search", str(DENSE_ROOT), "x", "--db", str(db_path), "--mode", "nope"],
    )
    assert invalid.exit_code != 0

    missing_dense = runner.invoke(
        app,
        [
            "search",
            str(DENSE_ROOT),
            query,
            "--db",
            str(db_path),
            "--dense-dir",
            str(tmp_path / "missing-dense"),
            "--mode",
            "dense",
        ],
    )
    assert missing_dense.exit_code == 1
    assert "Dense artifact is missing" in f"{missing_dense.stdout}\n{missing_dense.stderr}"


def test_stale_dense_cli_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "mod.py"
    source.write_text("def alpha() -> int:\n    return 1\n", encoding="utf-8")
    db_path = tmp_path / "index.db"
    dense_dir = tmp_path / "dense"
    assert runner.invoke(app, ["index", str(root), "--db", str(db_path)]).exit_code == 0

    def factory(model_id: str = "fake-model") -> FakeEmbeddingProvider:
        return FakeEmbeddingProvider(
            dimension=2,
            model_id=model_id,
            default_document=[1.0, 0.0],
            default_query=[1.0, 0.0],
        )

    monkeypatch.setattr("codeintel.cli.create_embedding_provider", factory)
    assert (
        runner.invoke(
            app,
            ["embed", str(root), "--db", str(db_path), "--dense-dir", str(dense_dir)],
        ).exit_code
        == 0
    )

    source.write_text("def alpha() -> int:\n    return 2\n", encoding="utf-8")
    assert runner.invoke(app, ["index", str(root), "--db", str(db_path)]).exit_code == 0
    stale = runner.invoke(
        app,
        [
            "search",
            str(root),
            "alpha",
            "--db",
            str(db_path),
            "--dense-dir",
            str(dense_dir),
            "--mode",
            "dense",
        ],
    )
    assert stale.exit_code == 1
    combined = f"{stale.stdout}\n{stale.stderr}"
    assert "stale" in combined.lower() or "fingerprint" in combined.lower()
    assert "aicode embed" in combined
