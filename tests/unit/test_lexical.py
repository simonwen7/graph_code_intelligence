"""Tests for FTS5 lexical retrieval and safe query construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.languages.python import PythonAdapter, PythonRelationExtractor
from codeintel.lexical import build_fts_query, search_code_units
from codeintel.models import SymbolKind
from codeintel.repository import analyze_repository
from codeintel.storage import IndexDatabase

SEARCH_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "python_search"


def _index(tmp_path: Path, root: Path = SEARCH_ROOT) -> IndexDatabase:
    analysis = analyze_repository(root, PythonAdapter(), PythonRelationExtractor())
    database = IndexDatabase(tmp_path / "index.db")
    database.open()
    database.rebuild(analysis)
    return database


def test_build_fts_query_quotes_tokens_and_joins_with_or() -> None:
    assert build_fts_query("payment authorization") == '"payment" OR "authorization"'
    assert build_fts_query('say "hello"') == '"say" OR "hello"'
    assert build_fts_query("helpers.helper") == '"helpers.helper"'
    assert build_fts_query("payment AND fraud") == '"payment" OR "AND" OR "fraud"'
    assert build_fts_query("foo(bar)") == '"foo(bar)"'
    assert build_fts_query("path/to/file") == '"path/to/file"'
    assert build_fts_query("a:b") == '"a:b"'
    assert build_fts_query("a*b") == '"a*b"'
    assert build_fts_query("dash-term") == '"dash-term"'
    assert build_fts_query("NOT OR AND") == '"NOT" OR "OR" OR "AND"'
    assert build_fts_query("multiple     whitespace") == '"multiple" OR "whitespace"'
    assert build_fts_query("\t\n  ") is None
    assert build_fts_query("") is None


def test_fts_document_count_matches_code_units(tmp_path: Path) -> None:
    database = _index(tmp_path)
    try:
        counts = database.counts()
        assert counts.fts_documents == counts.code_units
        assert counts.code_units > 0
    finally:
        database.close()


def test_distinctive_term_ranks_intended_code_unit(tmp_path: Path) -> None:
    database = _index(tmp_path)
    try:
        results = search_code_units(database, "authorize_payment")
        assert results
        assert results[0].symbol_qualified_name == "payment_gateway.authorize_payment"
        assert results[0].score > 0
        if len(results) > 1:
            assert results[0].score >= results[1].score
    finally:
        database.close()


def test_source_signature_and_qualified_name_search(tmp_path: Path) -> None:
    database = _index(tmp_path)
    try:
        source_hits = search_code_units(database, "fraud_threshold")
        assert source_hits[0].symbol_qualified_name == "payment_gateway.fraud_threshold"

        signature_hits = search_code_units(database, "csv_export")
        assert signature_hits[0].symbol_qualified_name == "report_export.csv_export"

        qname_hits = search_code_units(database, "cache_policy.CachePolicy")
        assert any(hit.symbol_qualified_name == "cache_policy.CachePolicy" for hit in qname_hits)
    finally:
        database.close()


def test_limit_kind_and_path_prefix_filters(tmp_path: Path) -> None:
    database = _index(tmp_path)
    try:
        limited = search_code_units(database, "token", limit=1)
        assert len(limited) == 1

        methods = search_code_units(database, "cache", kind=SymbolKind.METHOD)
        assert methods
        assert all(result.kind is SymbolKind.METHOD for result in methods)

        prefixed = search_code_units(database, "export", path_prefix="report_")
        assert prefixed
        assert all(str(result.path).startswith("report_") for result in prefixed)

        escaped = search_code_units(database, "payment", path_prefix="payment%")
        assert escaped == ()
    finally:
        database.close()


def test_empty_and_no_hit_queries(tmp_path: Path) -> None:
    database = _index(tmp_path)
    try:
        assert search_code_units(database, "   ") == ()
        assert search_code_units(database, "zzzz_no_such_term_qqq") == ()
    finally:
        database.close()


def test_special_character_queries_do_not_raise(tmp_path: Path) -> None:
    database = _index(tmp_path)
    try:
        for query in [
            "helpers.helper",
            "payment AND fraud",
            "OR",
            "NOT",
            '"unterminated',
            '"a quote"',
            "foo(bar)",
            "path/to/file",
            "a:b",
            "a*b",
            "dash-term",
        ]:
            search_code_units(database, query)
    finally:
        database.close()


def test_search_uses_persisted_snapshot_not_live_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "payment_gateway.py"
    source.write_text(
        "def authorize_payment(amount: int) -> bool:\n    return amount > 0\n",
        encoding="utf-8",
    )
    analysis = analyze_repository(repo, PythonAdapter(), PythonRelationExtractor())
    db_path = tmp_path / "index.db"
    with IndexDatabase(db_path) as database:
        database.rebuild(analysis)
        source.write_text(
            "def authorize_payment(amount: int) -> bool:\n    return 'MUTATED_LIVE_SOURCE'\n",
            encoding="utf-8",
        )
        results = search_code_units(database, "authorize_payment")
        assert results
        assert "MUTATED_LIVE_SOURCE" not in results[0].source_text
        assert "amount > 0" in results[0].source_text


def test_higher_score_means_better_result(tmp_path: Path) -> None:
    database = _index(tmp_path)
    try:
        results = search_code_units(database, "payment fraud_threshold")
        assert len(results) >= 2
        scores = [result.score for result in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].symbol_qualified_name == "payment_gateway.fraud_threshold"
    finally:
        database.close()


def test_limit_must_be_positive(tmp_path: Path) -> None:
    database = _index(tmp_path)
    try:
        with pytest.raises(ValueError, match="limit must be > 0"):
            search_code_units(database, "payment", limit=0)
    finally:
        database.close()


def test_deterministic_tie_ordering(tmp_path: Path) -> None:
    database = _index(tmp_path)
    try:
        first = [result.symbol_qualified_name for result in search_code_units(database, "token")]
        second = [result.symbol_qualified_name for result in search_code_units(database, "token")]
        assert first == second
        assert first == sorted(first) or len(set(first)) == len(first)
    finally:
        database.close()
