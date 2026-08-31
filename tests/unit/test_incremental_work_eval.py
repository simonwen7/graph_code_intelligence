"""Offline tests for incremental work evaluation harness."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "benchmarks" / "incremental_work_v1.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("incremental_work_v1", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_incremental_work_scenarios_deterministic() -> None:
    module = _load_module()
    payload = module.run_evaluation()
    assert payload["evaluation_id"] == "incremental-work-v1"
    scenarios = payload["scenarios"]
    assert isinstance(scenarios, list)
    assert [item["id"] for item in scenarios] == [
        "no-op",
        "body-edit",
        "symbol-rename",
        "add-delete",
    ]
    noop = scenarios[0]
    assert noop["incremental"]["files_analyzed"] == 0
    assert noop["incremental"]["relation_files_recomputed"] == 0
    assert noop["full"]["files_analyzed"] > 0
    assert noop["semantic_equivalent"] is True

    body = scenarios[1]
    assert body["incremental"]["files_changed"] == 1
    assert body["incremental"]["files_analyzed"] == 1
    assert body["semantic_equivalent"] is True

    rename = scenarios[2]
    assert rename["incremental"]["files_analyzed"] >= 1
    assert rename["semantic_equivalent"] is True

    add_delete = scenarios[3]
    assert add_delete["incremental"]["files_added"] == 1
    assert add_delete["incremental"]["files_deleted"] == 1
    assert add_delete["semantic_equivalent"] is True

    for scenario in scenarios:
        assert "dense_selective" in scenario
        assert "dense_full" in scenario
        assert scenario["dense_selective"]["provider_id"] == "fake"
