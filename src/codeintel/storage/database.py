"""SQLite lifecycle and persistence for repository analysis snapshots."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from codeintel.models import (
    AnalysisResult,
    CodeUnit,
    Relation,
    RelationKind,
    ResolutionStatus,
    SourceSpan,
    Symbol,
    SymbolKind,
)
from codeintel.repository import RepositoryAnalysis
from codeintel.storage.schema import SCHEMA_SQL, SCHEMA_VERSION


class IndexDatabaseError(Exception):
    """Raised for persistent-index failures."""


class SchemaVersionError(IndexDatabaseError):
    """Raised when an on-disk index uses an unsupported schema version."""


@dataclass(frozen=True, slots=True)
class IndexStats:
    """Counts written by a full index rebuild."""

    files: int
    symbols: int
    code_units: int
    relations: int
    fts_documents: int


@dataclass(frozen=True, slots=True)
class PersistedCodeUnitView:
    """Returnable CodeUnit snapshot fields for retrieval (includes file path)."""

    symbol_qualified_name: str
    kind: SymbolKind
    path: Path
    span: SourceSpan
    signature: str | None
    source_text: str


class IndexDatabase:
    """Small SQLite-backed repository index with full-rebuild semantics."""

    def __init__(self, path: Path, *, create: bool = True) -> None:
        self.path = Path(path)
        self._create = create
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> IndexDatabase:
        self.open(create=self._create)
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise IndexDatabaseError("Database is not open")
        return self._connection

    def open(self, *, create: bool = True) -> None:
        """Open the database, optionally creating a missing file for indexing."""
        if self._connection is not None:
            return
        if not create and not self.path.exists():
            raise IndexDatabaseError(f"Index database does not exist: {self.path}")
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            connection = sqlite3.connect(self.path)
            connection.row_factory = sqlite3.Row
            self._connection = connection
            self._enable_foreign_keys()
            self._initialize_or_validate_schema()
        except sqlite3.Error as exc:
            self.close()
            raise IndexDatabaseError(f"Failed to open index database: {self.path}") from exc

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def foreign_keys_enabled(self) -> bool:
        row = self.connection.execute("PRAGMA foreign_keys").fetchone()
        return bool(row[0]) if row is not None else False

    def schema_version(self) -> int:
        row = self.connection.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row is not None else 0

    def rebuild(self, analysis: RepositoryAnalysis) -> IndexStats:
        """Replace the entire index snapshot from ``analysis`` in one transaction."""
        prepared = _PreparedSnapshot.from_analysis(analysis)
        try:
            with self._write_transaction():
                self.connection.execute("DELETE FROM code_units_fts")
                self.connection.execute("DELETE FROM relations")
                self.connection.execute("DELETE FROM code_units")
                self.connection.execute("DELETE FROM symbols")
                self.connection.execute("DELETE FROM files")
                file_ids = self._insert_files(prepared.files)
                symbol_ids = self._insert_symbols(prepared.symbols, file_ids)
                code_unit_ids = self._insert_code_units(prepared.code_units, symbol_ids)
                self._insert_relations(prepared.relations, file_ids)
                self._insert_fts(prepared.fts_rows, code_unit_ids)
        except sqlite3.Error as exc:
            raise IndexDatabaseError(f"Failed to rebuild index database: {self.path}") from exc
        return IndexStats(
            files=len(prepared.files),
            symbols=len(prepared.symbols),
            code_units=len(prepared.code_units),
            relations=len(prepared.relations),
            fts_documents=len(prepared.fts_rows),
        )

    def counts(self) -> IndexStats:
        conn = self.connection
        return IndexStats(
            files=int(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]),
            symbols=int(conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]),
            code_units=int(conn.execute("SELECT COUNT(*) FROM code_units").fetchone()[0]),
            relations=int(conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]),
            fts_documents=int(conn.execute("SELECT COUNT(*) FROM code_units_fts").fetchone()[0]),
        )

    def load_symbols(self) -> tuple[Symbol, ...]:
        rows = self.connection.execute(
            """
            SELECT name, qualified_name, kind, start_line, end_line, start_byte, end_byte,
                   signature, parent_qualified_name
            FROM symbols
            ORDER BY qualified_name
            """
        ).fetchall()
        return tuple(_symbol_from_row(row) for row in rows)

    def load_relations(self) -> tuple[Relation, ...]:
        rows = self.connection.execute(
            """
            SELECT r.source_qualified_name, r.target_qualified_name, r.target_text,
                   r.kind, r.resolution, f.path,
                   r.start_line, r.end_line, r.start_byte, r.end_byte
            FROM relations AS r
            JOIN files AS f ON f.id = r.file_id
            ORDER BY r.kind, r.source_qualified_name,
                     COALESCE(r.target_qualified_name, ''),
                     r.target_text, r.resolution, f.path,
                     COALESCE(r.start_byte, -1), COALESCE(r.end_byte, -1)
            """
        ).fetchall()
        return tuple(_relation_from_row(row) for row in rows)

    def load_code_units(self) -> tuple[tuple[str, CodeUnit], ...]:
        """Return ``(symbol_qualified_name, CodeUnit)`` pairs in qname order."""
        rows = self.connection.execute(
            """
            SELECT s.qualified_name, c.kind, c.source_text,
                   c.start_line, c.end_line, c.start_byte, c.end_byte
            FROM code_units AS c
            JOIN symbols AS s ON s.id = c.symbol_id
            ORDER BY s.qualified_name
            """
        ).fetchall()
        return tuple(
            (
                str(row["qualified_name"]),
                CodeUnit(
                    symbol_qualified_name=str(row["qualified_name"]),
                    kind=SymbolKind(str(row["kind"])),
                    source_text=str(row["source_text"]),
                    span=_span_from_row(row),
                ),
            )
            for row in rows
        )

    def load_persisted_code_units(self) -> dict[str, PersistedCodeUnitView]:
        """Return returnable CodeUnits keyed by ``symbol_qualified_name``.

        MODULE symbols are absent because they have no CodeUnit rows.
        """
        rows = self.connection.execute(
            """
            SELECT
                s.qualified_name AS qualified_name,
                s.kind AS kind,
                f.path AS path,
                s.signature AS signature,
                c.source_text AS source_text,
                c.start_line AS start_line,
                c.end_line AS end_line,
                c.start_byte AS start_byte,
                c.end_byte AS end_byte
            FROM code_units AS c
            JOIN symbols AS s ON s.id = c.symbol_id
            JOIN files AS f ON f.id = s.file_id
            ORDER BY s.qualified_name ASC
            """
        ).fetchall()
        units: dict[str, PersistedCodeUnitView] = {}
        for row in rows:
            signature = row["signature"]
            qname = str(row["qualified_name"])
            units[qname] = PersistedCodeUnitView(
                symbol_qualified_name=qname,
                kind=SymbolKind(str(row["kind"])),
                path=Path(str(row["path"])),
                span=_span_from_row(row),
                signature=None if signature is None else str(signature),
                source_text=str(row["source_text"]),
            )
        return units

    def load_files(self) -> tuple[tuple[str, str, str, bool], ...]:
        """Return ``(path, language_id, module_name, has_syntax_errors)``."""
        rows = self.connection.execute(
            """
            SELECT path, language_id, module_name, has_syntax_errors
            FROM files
            ORDER BY path
            """
        ).fetchall()
        return tuple(
            (
                str(row["path"]),
                str(row["language_id"]),
                str(row["module_name"]),
                bool(row["has_syntax_errors"]),
            )
            for row in rows
        )

    def _enable_foreign_keys(self) -> None:
        self.connection.execute("PRAGMA foreign_keys = ON")
        if not self.foreign_keys_enabled():
            raise IndexDatabaseError("Failed to enable SQLite foreign_keys")

    def _initialize_or_validate_schema(self) -> None:
        version = self.schema_version()
        tables = self._application_table_names()
        core_tables = {name for name in tables if not name.startswith("code_units_fts_")}
        required = {"files", "symbols", "code_units", "relations", "code_units_fts"}

        if version == 0:
            if not core_tables:
                self.connection.executescript(SCHEMA_SQL)
                self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                self.connection.commit()
                return
            unexpected = core_tables - required
            if unexpected:
                raise IndexDatabaseError(
                    f"Path is not a compatible codeintel index database: {self.path}"
                )
            missing = required - core_tables
            if missing:
                raise IndexDatabaseError(
                    f"Index database is missing required tables: {', '.join(sorted(missing))}"
                )
            self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self.connection.commit()
            return

        if version != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Unsupported index schema version {version}; expected {SCHEMA_VERSION}"
            )
        missing = required - core_tables
        if missing:
            raise IndexDatabaseError(
                f"Index database is missing required tables: {', '.join(sorted(missing))}"
            )

    def _application_table_names(self) -> set[str]:
        rows = self.connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
        return {str(row[0]) for row in rows}

    @contextmanager
    def _write_transaction(self) -> Iterator[None]:
        connection = self.connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def _insert_files(self, files: list[_FileRow]) -> dict[str, int]:
        file_ids: dict[str, int] = {}
        for row in files:
            cursor = self.connection.execute(
                """
                INSERT INTO files(path, language_id, module_name, has_syntax_errors)
                VALUES (?, ?, ?, ?)
                """,
                (row.path, row.language_id, row.module_name, int(row.has_syntax_errors)),
            )
            assert cursor.lastrowid is not None
            file_ids[row.path] = int(cursor.lastrowid)
        return file_ids

    def _insert_symbols(
        self, symbols: list[_SymbolRow], file_ids: dict[str, int]
    ) -> dict[str, int]:
        symbol_ids: dict[str, int] = {}
        for row in symbols:
            cursor = self.connection.execute(
                """
                INSERT INTO symbols(
                    file_id, name, qualified_name, kind,
                    start_line, end_line, start_byte, end_byte,
                    signature, parent_qualified_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_ids[row.relative_path],
                    row.name,
                    row.qualified_name,
                    row.kind,
                    row.start_line,
                    row.end_line,
                    row.start_byte,
                    row.end_byte,
                    row.signature,
                    row.parent_qualified_name,
                ),
            )
            assert cursor.lastrowid is not None
            symbol_ids[row.qualified_name] = int(cursor.lastrowid)
        return symbol_ids

    def _insert_code_units(
        self, units: list[_CodeUnitRow], symbol_ids: dict[str, int]
    ) -> dict[str, int]:
        code_unit_ids: dict[str, int] = {}
        for row in units:
            cursor = self.connection.execute(
                """
                INSERT INTO code_units(
                    symbol_id, kind, source_text,
                    start_line, end_line, start_byte, end_byte
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol_ids[row.symbol_qualified_name],
                    row.kind,
                    row.source_text,
                    row.start_line,
                    row.end_line,
                    row.start_byte,
                    row.end_byte,
                ),
            )
            assert cursor.lastrowid is not None
            code_unit_ids[row.symbol_qualified_name] = int(cursor.lastrowid)
        return code_unit_ids

    def _insert_relations(self, relations: list[_RelationRow], file_ids: dict[str, int]) -> None:
        for row in relations:
            self.connection.execute(
                """
                INSERT INTO relations(
                    source_qualified_name, target_qualified_name, target_text,
                    kind, resolution, file_id,
                    start_line, end_line, start_byte, end_byte
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.source_qualified_name,
                    row.target_qualified_name,
                    row.target_text,
                    row.kind,
                    row.resolution,
                    file_ids[row.relative_path],
                    row.start_line,
                    row.end_line,
                    row.start_byte,
                    row.end_byte,
                ),
            )

    def _insert_fts(self, rows: list[_FtsRow], code_unit_ids: dict[str, int]) -> None:
        for row in rows:
            rowid = code_unit_ids[row.symbol_qualified_name]
            self.connection.execute(
                """
                INSERT INTO code_units_fts(
                    rowid, qualified_name, name, signature, source_text,
                    module_name, path, kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rowid,
                    row.qualified_name,
                    row.name,
                    row.signature,
                    row.source_text,
                    row.module_name,
                    row.path,
                    row.kind,
                ),
            )


@dataclass(frozen=True, slots=True)
class _FileRow:
    path: str
    language_id: str
    module_name: str
    has_syntax_errors: bool


@dataclass(frozen=True, slots=True)
class _SymbolRow:
    relative_path: str
    name: str
    qualified_name: str
    kind: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    signature: str | None
    parent_qualified_name: str | None


@dataclass(frozen=True, slots=True)
class _CodeUnitRow:
    symbol_qualified_name: str
    kind: str
    source_text: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int


@dataclass(frozen=True, slots=True)
class _RelationRow:
    source_qualified_name: str
    target_qualified_name: str | None
    target_text: str
    kind: str
    resolution: str
    relative_path: str
    start_line: int | None
    end_line: int | None
    start_byte: int | None
    end_byte: int | None


@dataclass(frozen=True, slots=True)
class _FtsRow:
    symbol_qualified_name: str
    qualified_name: str
    name: str
    signature: str
    source_text: str
    module_name: str
    path: str
    kind: str


@dataclass(frozen=True, slots=True)
class _PreparedSnapshot:
    files: list[_FileRow]
    symbols: list[_SymbolRow]
    code_units: list[_CodeUnitRow]
    relations: list[_RelationRow]
    fts_rows: list[_FtsRow]

    @classmethod
    def from_analysis(cls, analysis: RepositoryAnalysis) -> _PreparedSnapshot:
        root = analysis.root.resolve()
        files: list[_FileRow] = []
        symbols: list[_SymbolRow] = []
        code_units: list[_CodeUnitRow] = []
        symbol_meta: dict[str, tuple[str, str, str, str | None]] = {}

        file_entries: list[tuple[str, AnalysisResult]] = []
        for result in analysis.files:
            if result.path is None:
                raise IndexDatabaseError("AnalysisResult.path is required for indexing")
            relative = _relative_posix_path(result.path, root)
            file_entries.append((relative, result))
        file_entries.sort(key=lambda item: item[0])

        for relative, result in file_entries:
            files.append(
                _FileRow(
                    path=relative,
                    language_id=result.language_id,
                    module_name=result.module_name,
                    has_syntax_errors=result.has_syntax_errors,
                )
            )
            for symbol in sorted(result.symbols, key=lambda item: item.qualified_name):
                symbols.append(
                    _SymbolRow(
                        relative_path=relative,
                        name=symbol.name,
                        qualified_name=symbol.qualified_name,
                        kind=symbol.kind.value,
                        start_line=symbol.span.start_line,
                        end_line=symbol.span.end_line,
                        start_byte=symbol.span.start_byte,
                        end_byte=symbol.span.end_byte,
                        signature=symbol.signature,
                        parent_qualified_name=symbol.parent_qualified_name,
                    )
                )
                symbol_meta[symbol.qualified_name] = (
                    relative,
                    result.module_name,
                    symbol.name,
                    symbol.signature,
                )
            for unit in sorted(result.code_units, key=lambda item: item.symbol_qualified_name):
                code_units.append(
                    _CodeUnitRow(
                        symbol_qualified_name=unit.symbol_qualified_name,
                        kind=unit.kind.value,
                        source_text=unit.source_text,
                        start_line=unit.span.start_line,
                        end_line=unit.span.end_line,
                        start_byte=unit.span.start_byte,
                        end_byte=unit.span.end_byte,
                    )
                )

        symbols.sort(key=lambda row: row.qualified_name)
        code_units.sort(key=lambda row: row.symbol_qualified_name)

        relations: list[_RelationRow] = []
        for relation in analysis.relations:
            relative = _relative_posix_path(relation.path, root)
            span = relation.span
            relations.append(
                _RelationRow(
                    source_qualified_name=relation.source_qualified_name,
                    target_qualified_name=relation.target_qualified_name,
                    target_text=relation.target_text,
                    kind=relation.kind.value,
                    resolution=relation.resolution.value,
                    relative_path=relative,
                    start_line=span.start_line if span is not None else None,
                    end_line=span.end_line if span is not None else None,
                    start_byte=span.start_byte if span is not None else None,
                    end_byte=span.end_byte if span is not None else None,
                )
            )
        relations.sort(
            key=lambda row: (
                row.kind,
                row.source_qualified_name,
                row.target_qualified_name or "",
                row.target_text,
                row.resolution,
                row.relative_path,
                row.start_byte if row.start_byte is not None else -1,
                row.end_byte if row.end_byte is not None else -1,
            )
        )

        fts_rows: list[_FtsRow] = []
        for unit_row in code_units:
            relative, module_name, name, signature = symbol_meta[unit_row.symbol_qualified_name]
            fts_rows.append(
                _FtsRow(
                    symbol_qualified_name=unit_row.symbol_qualified_name,
                    qualified_name=unit_row.symbol_qualified_name,
                    name=name,
                    signature=signature or "",
                    source_text=unit_row.source_text,
                    module_name=module_name,
                    path=relative,
                    kind=unit_row.kind,
                )
            )
        fts_rows.sort(key=lambda row: row.symbol_qualified_name)
        return cls(
            files=files,
            symbols=symbols,
            code_units=code_units,
            relations=relations,
            fts_rows=fts_rows,
        )


def _relative_posix_path(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root)
    except ValueError as exc:
        raise IndexDatabaseError(f"Path {path} is outside repository root {root}") from exc
    return relative.as_posix()


def _span_from_row(row: sqlite3.Row) -> SourceSpan:
    return SourceSpan(
        start_line=int(row["start_line"]),
        end_line=int(row["end_line"]),
        start_byte=int(row["start_byte"]),
        end_byte=int(row["end_byte"]),
    )


def _symbol_from_row(row: sqlite3.Row) -> Symbol:
    return Symbol(
        name=str(row["name"]),
        qualified_name=str(row["qualified_name"]),
        kind=SymbolKind(str(row["kind"])),
        span=_span_from_row(row),
        signature=None if row["signature"] is None else str(row["signature"]),
        parent_qualified_name=(
            None if row["parent_qualified_name"] is None else str(row["parent_qualified_name"])
        ),
    )


def _relation_from_row(row: sqlite3.Row) -> Relation:
    span: SourceSpan | None
    if row["start_line"] is None:
        span = None
    else:
        span = SourceSpan(
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
            start_byte=int(row["start_byte"]),
            end_byte=int(row["end_byte"]),
        )
    return Relation(
        kind=RelationKind(str(row["kind"])),
        source_qualified_name=str(row["source_qualified_name"]),
        target_qualified_name=(
            None if row["target_qualified_name"] is None else str(row["target_qualified_name"])
        ),
        target_text=str(row["target_text"]),
        resolution=ResolutionStatus(str(row["resolution"])),
        path=Path(str(row["path"])),
        span=span,
    )
