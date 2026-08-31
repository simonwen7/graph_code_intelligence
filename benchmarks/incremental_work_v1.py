#!/usr/bin/env python3
"""Deterministic incremental vs full work-count evaluation for Milestone 10.

Uses a temporary copy of the committed Python retrieval corpus. Does not mutate
committed sources. Dense work counts use a fake embedding provider (work only).
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from codeintel import __version__
from codeintel.dense import build_dense_index, load_dense_documents
from codeintel.indexing import index_repository
from codeintel.languages.python import PythonAdapter, PythonRelationExtractor
from codeintel.storage import IndexDatabase

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_SRC = REPO_ROOT / "benchmarks" / "python_retrieval_v1" / "corpus"
RESULTS_JSON = REPO_ROOT / "benchmarks" / "results" / "incremental_work_v1.json"
RESULTS_MD = REPO_ROOT / "benchmarks" / "results" / "incremental_work_v1.md"


def _fake_embedding_provider_class() -> type:
    """Load FakeEmbeddingProvider without making benchmarks/ depend on import order."""
    tests_path = str(REPO_ROOT / "tests")
    if tests_path not in sys.path:
        sys.path.insert(0, tests_path)
    from helpers.fake_embeddings import FakeEmbeddingProvider

    return FakeEmbeddingProvider


@dataclass(frozen=True, slots=True)
class IndexCounts:
    mode: str
    files_added: int
    files_changed: int
    files_deleted: int
    files_unchanged: int
    files_analyzed: int
    relation_files_recomputed: int
    symbols_rewritten: int
    code_units_rewritten: int
    files: int
    symbols: int
    code_units: int
    relations: int


@dataclass(frozen=True, slots=True)
class DenseCounts:
    provider_id: str
    model_id: str
    document_count: int
    vectors_reused: int
    vectors_embedded: int
    rewritten: bool


def _index_counts(stats: object) -> IndexCounts:
    return IndexCounts(
        mode=str(stats.mode),
        files_added=int(stats.files_added),
        files_changed=int(stats.files_changed),
        files_deleted=int(stats.files_deleted),
        files_unchanged=int(stats.files_unchanged),
        files_analyzed=int(stats.files_analyzed),
        relation_files_recomputed=int(stats.relation_files_recomputed),
        symbols_rewritten=int(stats.symbols_rewritten),
        code_units_rewritten=int(stats.code_units_rewritten),
        files=int(stats.files),
        symbols=int(stats.symbols),
        code_units=int(stats.code_units),
        relations=int(stats.relations),
    )


def _snapshot_qnames(db_path: Path) -> set[str]:
    with IndexDatabase(db_path, create=False) as database:
        return {qname for qname, _unit in database.load_code_units()}


def _fake_provider_for_db(db_path: Path) -> object:
    with IndexDatabase(db_path, create=False) as database:
        documents = load_dense_documents(database)
    provider_cls = _fake_embedding_provider_class()
    return provider_cls(
        dimension=4,
        provider_id="fake",
        model_id="fake-incremental-work",
        document_vectors={document.document_text: [1.0, 0.0, 0.0, 0.0] for document in documents},
        default_document=[0.25, 0.25, 0.25, 0.25],
        default_query=[1.0, 0.0, 0.0, 0.0],
    )


def _dense_counts(db_path: Path, dense_dir: Path, *, full: bool) -> DenseCounts:
    provider = _fake_provider_for_db(db_path)
    with IndexDatabase(db_path, create=False) as database:
        stats = build_dense_index(database, provider, artifact_dir=dense_dir, full=full)
    return DenseCounts(
        provider_id=stats.provider_id,
        model_id=stats.model_id,
        document_count=stats.document_count,
        vectors_reused=stats.vectors_reused,
        vectors_embedded=stats.vectors_embedded,
        rewritten=stats.rewritten,
    )


def _copy_corpus(destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(CORPUS_SRC, destination)
    return destination


def run_evaluation() -> dict[str, object]:
    adapter = PythonAdapter()
    extractor = PythonRelationExtractor()
    scenarios: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="m10-incr-") as tmp:
        root = Path(tmp)
        work = _copy_corpus(root / "repo")
        db = root / "index.db"
        dense = root / "dense"

        baseline = index_repository(work, adapter, extractor, database_path=db, full=True)
        baseline_qnames = _snapshot_qnames(db)
        initial_dense = _dense_counts(db, dense, full=True)

        # 1) no-op incremental vs full rebuild of same state
        noop_inc = index_repository(work, adapter, extractor, database_path=db)
        full_db = root / "full_noop.db"
        noop_full = index_repository(work, adapter, extractor, database_path=full_db, full=True)
        assert _snapshot_qnames(db) == baseline_qnames == _snapshot_qnames(full_db)
        noop_dense_reuse = _dense_counts(db, dense, full=False)
        noop_dense_full = _dense_counts(db, root / "dense_noop_full", full=True)
        scenarios.append(
            {
                "id": "no-op",
                "description": "Unchanged corpus after baseline index.",
                "incremental": asdict(_index_counts(noop_inc)),
                "full": asdict(_index_counts(noop_full)),
                "dense_selective": asdict(noop_dense_reuse),
                "dense_full": asdict(noop_dense_full),
                "semantic_equivalent": True,
            }
        )

        # 2) body-only edit
        target = work / "pricing.py"
        original = target.read_text(encoding="utf-8")
        target.write_text(
            original.replace(
                "return unit_cents * quantity",
                "return unit_cents * quantity  # body-only edit",
            ),
            encoding="utf-8",
        )
        body_inc = index_repository(work, adapter, extractor, database_path=db)
        body_full_db = root / "full_body.db"
        body_full = index_repository(
            work, adapter, extractor, database_path=body_full_db, full=True
        )
        assert _snapshot_qnames(db) == _snapshot_qnames(body_full_db)
        body_dense_sel = _dense_counts(db, dense, full=False)
        body_dense_full = _dense_counts(db, root / "dense_body_full", full=True)
        scenarios.append(
            {
                "id": "body-edit",
                "description": "Comment-only body change in pricing.line_total_cents.",
                "incremental": asdict(_index_counts(body_inc)),
                "full": asdict(_index_counts(body_full)),
                "dense_selective": asdict(body_dense_sel),
                "dense_full": asdict(body_dense_full),
                "semantic_equivalent": True,
            }
        )

        # 3) symbol rename
        target.write_text(
            target.read_text(encoding="utf-8").replace(
                "def line_total_cents",
                "def line_amount_cents",
            ),
            encoding="utf-8",
        )
        # Keep call sites compiling for relation extraction honesty.
        discounts = work / "discounts.py"
        discounts.write_text(
            discounts.read_text(encoding="utf-8").replace(
                "line_total_cents",
                "line_amount_cents",
            ),
            encoding="utf-8",
        )
        rename_inc = index_repository(work, adapter, extractor, database_path=db)
        rename_full_db = root / "full_rename.db"
        rename_full = index_repository(
            work, adapter, extractor, database_path=rename_full_db, full=True
        )
        assert _snapshot_qnames(db) == _snapshot_qnames(rename_full_db)
        assert "pricing.line_total_cents" not in _snapshot_qnames(db)
        assert "pricing.line_amount_cents" in _snapshot_qnames(db)
        rename_dense_sel = _dense_counts(db, dense, full=False)
        rename_dense_full = _dense_counts(db, root / "dense_rename_full", full=True)
        scenarios.append(
            {
                "id": "symbol-rename",
                "description": "Rename pricing.line_total_cents → pricing.line_amount_cents.",
                "incremental": asdict(_index_counts(rename_inc)),
                "full": asdict(_index_counts(rename_full)),
                "dense_selective": asdict(rename_dense_sel),
                "dense_full": asdict(rename_dense_full),
                "semantic_equivalent": True,
            }
        )

        # 4) file add + delete
        added = work / "audit.py"
        added.write_text(
            '"""Tiny audit helper."""\n\ndef record_event(name: str) -> str:\n    return f"event:{name}"\n',
            encoding="utf-8",
        )
        (work / "payments.py").unlink()
        add_del_inc = index_repository(work, adapter, extractor, database_path=db)
        add_del_full_db = root / "full_add_del.db"
        add_del_full = index_repository(
            work, adapter, extractor, database_path=add_del_full_db, full=True
        )
        assert _snapshot_qnames(db) == _snapshot_qnames(add_del_full_db)
        assert "audit.record_event" in _snapshot_qnames(db)
        assert "payments.authorize_card" not in _snapshot_qnames(db)
        add_del_dense_sel = _dense_counts(db, dense, full=False)
        add_del_dense_full = _dense_counts(db, root / "dense_add_del_full", full=True)
        scenarios.append(
            {
                "id": "add-delete",
                "description": "Add audit.py and delete payments.py.",
                "incremental": asdict(_index_counts(add_del_inc)),
                "full": asdict(_index_counts(add_del_full)),
                "dense_selective": asdict(add_del_dense_sel),
                "dense_full": asdict(add_del_dense_full),
                "semantic_equivalent": True,
            }
        )

        payload = {
            "evaluation_id": "incremental-work-v1",
            "engine_version": __version__,
            "corpus_source": "benchmarks/python_retrieval_v1/corpus",
            "baseline": {
                "index": asdict(_index_counts(baseline)),
                "dense_initial_full": asdict(initial_dense),
            },
            "dense_provider_note": (
                "Dense work counts use FakeEmbeddingProvider (fake-incremental-work). "
                "They measure reuse/embed work, not retrieval quality."
            ),
            "scenarios": scenarios,
        }
    return payload


def render_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Incremental Work Evaluation v1",
        "",
        "Exact M8 work counters comparing incremental transitions vs full rebuilds on a",
        "temporary copy of `benchmarks/python_retrieval_v1/corpus`.",
        "",
        f"- Engine version: `{payload['engine_version']}`",
        f"- Corpus source: `{payload['corpus_source']}`",
        f"- Dense provider note: {payload['dense_provider_note']}",
        "",
        "No wall-clock timing claims.",
        "",
    ]
    for scenario in payload["scenarios"]:  # type: ignore[index]
        assert isinstance(scenario, dict)
        inc = scenario["incremental"]
        full = scenario["full"]
        dense_sel = scenario["dense_selective"]
        dense_full = scenario["dense_full"]
        assert isinstance(inc, dict)
        assert isinstance(full, dict)
        assert isinstance(dense_sel, dict)
        assert isinstance(dense_full, dict)
        lines.extend(
            [
                f"## {scenario['id']}",
                "",
                str(scenario["description"]),
                "",
                "| Counter | Incremental | Full |",
                "| --- | ---: | ---: |",
                f"| mode | {inc['mode']} | {full['mode']} |",
                f"| files_analyzed | {inc['files_analyzed']} | {full['files_analyzed']} |",
                f"| relation_files_recomputed | {inc['relation_files_recomputed']} | {full['relation_files_recomputed']} |",
                f"| symbols_rewritten | {inc['symbols_rewritten']} | {full['symbols_rewritten']} |",
                f"| code_units_rewritten | {inc['code_units_rewritten']} | {full['code_units_rewritten']} |",
                f"| files_added | {inc['files_added']} | {full['files_added']} |",
                f"| files_changed | {inc['files_changed']} | {full['files_changed']} |",
                f"| files_deleted | {inc['files_deleted']} | {full['files_deleted']} |",
                f"| files_unchanged | {inc['files_unchanged']} | {full['files_unchanged']} |",
                "",
                "| Dense counter | Selective embed | Full embed |",
                "| --- | ---: | ---: |",
                f"| documents_total | {dense_sel['document_count']} | {dense_full['document_count']} |",
                f"| vectors_reused | {dense_sel['vectors_reused']} | {dense_full['vectors_reused']} |",
                f"| vectors_embedded | {dense_sel['vectors_embedded']} | {dense_full['vectors_embedded']} |",
                "",
                f"Semantic equivalence of final CodeUnit qnames: `{scenario['semantic_equivalent']}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    payload = run_evaluation()
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    RESULTS_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {RESULTS_JSON}")
    print(f"wrote {RESULTS_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
