"""Lexical BM25 retrieval over a persisted CodeUnit FTS index."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from codeintel.models import SearchResult, SourceSpan, SymbolKind
from codeintel.storage.database import IndexDatabase, IndexDatabaseError


def build_fts_query(query: str) -> str | None:
    """Build a safe ordinary-text FTS5 MATCH expression.

    Whitespace-separated chunks become quoted literal tokens joined by OR.
    Empty/whitespace-only input returns ``None`` (caller issues no MATCH).
    """
    chunks = query.split()
    if not chunks:
        return None
    quoted = [f'"{_escape_fts_token(chunk)}"' for chunk in chunks]
    return " OR ".join(quoted)


def search_code_units(
    database: IndexDatabase,
    query: str,
    *,
    limit: int = 10,
    kind: SymbolKind | None = None,
    path_prefix: str | None = None,
) -> tuple[SearchResult, ...]:
    """Search persisted CodeUnits with BM25 ranking.

    Results are ordered by raw SQLite BM25 ascending (best first).
    ``SearchResult.score`` exposes ``-raw_bm25`` so higher is better.
    """
    if limit <= 0:
        raise ValueError("limit must be > 0")

    match_expr = build_fts_query(query)
    if match_expr is None:
        return ()

    clauses = ["code_units_fts MATCH ?"]
    params: list[object] = [match_expr]

    if kind is not None:
        clauses.append("s.kind = ?")
        params.append(kind.value)

    if path_prefix is not None:
        clauses.append("f.path LIKE ? ESCAPE '\\'")
        params.append(_like_prefix(path_prefix))

    params.append(limit)
    where = " AND ".join(clauses)
    sql = f"""
        SELECT
            s.qualified_name AS symbol_qualified_name,
            s.kind AS kind,
            f.path AS path,
            c.start_line AS start_line,
            c.end_line AS end_line,
            c.start_byte AS start_byte,
            c.end_byte AS end_byte,
            s.signature AS signature,
            c.source_text AS source_text,
            c.id AS code_unit_id,
            bm25(code_units_fts) AS raw_bm25
        FROM code_units_fts
        JOIN code_units AS c ON c.id = code_units_fts.rowid
        JOIN symbols AS s ON s.id = c.symbol_id
        JOIN files AS f ON f.id = s.file_id
        WHERE {where}
        ORDER BY raw_bm25 ASC, s.qualified_name ASC, f.path ASC, c.id ASC
        LIMIT ?
    """
    try:
        rows = database.connection.execute(sql, params).fetchall()
    except sqlite3.Error as exc:
        raise IndexDatabaseError("Lexical search query failed") from exc

    results: list[SearchResult] = []
    for row in rows:
        signature = row["signature"]
        results.append(
            SearchResult(
                symbol_qualified_name=str(row["symbol_qualified_name"]),
                kind=SymbolKind(str(row["kind"])),
                path=Path(str(row["path"])),
                span=SourceSpan(
                    start_line=int(row["start_line"]),
                    end_line=int(row["end_line"]),
                    start_byte=int(row["start_byte"]),
                    end_byte=int(row["end_byte"]),
                ),
                signature=None if signature is None else str(signature),
                source_text=str(row["source_text"]),
                score=-float(row["raw_bm25"]),
            )
        )
    return tuple(results)


def _escape_fts_token(token: str) -> str:
    if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
        token = token[1:-1]
    return token.replace('"', '""')


def _like_prefix(path_prefix: str) -> str:
    normalized = path_prefix.replace("\\", "/")
    escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}%"
