"""SQLite schema definitions for the persistent repository index."""

from __future__ import annotations

from pathlib import Path

SCHEMA_VERSION = 2

DEFAULT_INDEX_DIRNAME = ".codeintel"
DEFAULT_INDEX_FILENAME = "index.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    language_id TEXT NOT NULL,
    module_name TEXT NOT NULL,
    has_syntax_errors INTEGER NOT NULL CHECK (has_syntax_errors IN (0, 1)),
    content_sha256 TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_files_module_name ON files(module_name);

CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id),
    name TEXT NOT NULL,
    qualified_name TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    start_byte INTEGER NOT NULL,
    end_byte INTEGER NOT NULL,
    signature TEXT NULL,
    parent_qualified_name TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_symbols_file_id ON symbols(file_id);
CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind);
CREATE INDEX IF NOT EXISTS idx_symbols_parent ON symbols(parent_qualified_name);

CREATE TABLE IF NOT EXISTS code_units (
    id INTEGER PRIMARY KEY,
    symbol_id INTEGER NOT NULL UNIQUE REFERENCES symbols(id),
    kind TEXT NOT NULL,
    source_text TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    start_byte INTEGER NOT NULL,
    end_byte INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY,
    source_qualified_name TEXT NOT NULL REFERENCES symbols(qualified_name),
    target_qualified_name TEXT NULL,
    target_text TEXT NOT NULL,
    kind TEXT NOT NULL,
    resolution TEXT NOT NULL,
    file_id INTEGER NOT NULL REFERENCES files(id),
    start_line INTEGER NULL,
    end_line INTEGER NULL,
    start_byte INTEGER NULL,
    end_byte INTEGER NULL
);

CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_qualified_name);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_qualified_name);
CREATE INDEX IF NOT EXISTS idx_relations_kind ON relations(kind);
CREATE INDEX IF NOT EXISTS idx_relations_file_id ON relations(file_id);

CREATE VIRTUAL TABLE IF NOT EXISTS code_units_fts USING fts5(
    qualified_name,
    name,
    signature,
    source_text,
    module_name,
    path,
    kind,
    tokenize = 'unicode61'
);
"""


def default_index_path(repository_root: Path) -> Path:
    """Return the default on-disk index path for a repository root."""
    return repository_root / DEFAULT_INDEX_DIRNAME / DEFAULT_INDEX_FILENAME
