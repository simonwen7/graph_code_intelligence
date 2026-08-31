"""CLI and end-to-end offline tests for ``aicode benchmark``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from helpers.fake_embeddings import FakeEmbeddingProvider
from typer.testing import CliRunner

from codeintel.cli import app
from codeintel.dense import build_dense_index, load_dense_documents
from codeintel.evaluation import load_benchmark_definition
from codeintel.indexing import index_repository
from codeintel.languages.python import PythonAdapter, PythonRelationExtractor
from codeintel.storage import IndexDatabase

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / "benchmarks" / "python_retrieval_v1" / "corpus"
BENCHMARK_JSON = REPO_ROOT / "benchmarks" / "python_retrieval_v1" / "benchmark.json"
TINY = REPO_ROOT / "tests" / "fixtures" / "python_search"


def test_committed_benchmark_gold_qnames_exist(tmp_path: Path) -> None:
    db_path = tmp_path / "index.db"
    index_repository(
        CORPUS,
        PythonAdapter(),
        PythonRelationExtractor(),
        database_path=db_path,
        full=True,
    )
    definition = load_benchmark_definition(BENCHMARK_JSON)
    with IndexDatabase(db_path, create=False) as database:
        qnames = {qname for qname, _unit in database.load_code_units()}
    missing = [
        f"{case.id}:{name}"
        for case in definition.queries
        for name in case.relevant_qnames
        if name not in qnames
    ]
    assert not missing
    assert len(definition.queries) == 24


def test_benchmark_cli_fake_provider_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "index.db"
    dense_dir = tmp_path / "dense"
    json_out = tmp_path / "out.json"
    md_out = tmp_path / "out.md"

    index_repository(
        TINY,
        PythonAdapter(),
        PythonRelationExtractor(),
        database_path=db_path,
        full=True,
    )
    with IndexDatabase(db_path, create=False) as database:
        units = database.load_code_units()
        assert units
        gold = units[0][0]
        documents = load_dense_documents(database)

    queries: list[dict[str, Any]] = []
    categories = ["lexical", "behavioral", "calls", "inheritance"]
    for category in categories:
        for index in range(1, 7):
            queries.append(
                {
                    "id": f"{category}-{index}",
                    "category": category,
                    "query": f"find {gold.split('.')[-1]} {category} {index}",
                    "relevant_qnames": [gold],
                    "showcase": category == "lexical" and index <= 2,
                    "notes": "",
                }
            )
    payload: dict[str, Any] = {
        "benchmark_version": 1,
        "benchmark_id": "python-structural-retrieval-v1",
        "language": "python",
        "queries": queries,
    }
    bench_path = tmp_path / "bench.json"
    bench_path.write_text(json.dumps(payload), encoding="utf-8")

    document_vectors = {document.document_text: [1.0, 0.0, 0.0, 0.0] for document in documents}
    query_vectors = {case["query"]: [1.0, 0.0, 0.0, 0.0] for case in queries}
    provider = FakeEmbeddingProvider(
        dimension=4,
        model_id="fake-model",
        document_vectors=document_vectors,
        query_vectors=query_vectors,
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
    with IndexDatabase(db_path, create=False) as database:
        build_dense_index(database, factory(), artifact_dir=dense_dir)

    result = runner.invoke(
        app,
        [
            "benchmark",
            str(TINY),
            str(bench_path),
            "--db",
            str(db_path),
            "--dense-dir",
            str(dense_dir),
            "--model",
            "fake-model",
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(md_out),
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "benchmark_id:" in result.stdout
    assert "graph_vs_hybrid:" in result.stdout
    assert json_out.is_file()
    assert md_out.is_file()
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["query_count"] == 24
    assert list(data["aggregate"].keys()) == [
        "lexical",
        "dense",
        "hybrid",
        "graph",
        "reranked",
    ]
    assert [row["id"] for row in data["queries"]] == [q["id"] for q in queries]


def test_benchmark_cli_missing_index_fails(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "benchmark",
            str(TINY),
            str(BENCHMARK_JSON),
            "--db",
            str(tmp_path / "missing.db"),
        ],
    )
    assert result.exit_code != 0
    assert "index database does not exist" in result.stdout + result.stderr
