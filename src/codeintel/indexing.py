"""Incremental repository indexing: hashing, changesets, and update planning."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from codeintel.discovery import discover_source_files
from codeintel.languages.base import LanguageAdapter
from codeintel.models import AnalysisResult, Relation, Symbol
from codeintel.repository import (
    RelationExtractor,
    _build_symbol_index,
    _dedupe_relations,
    _derive_contains_relations,
    analyze_repository,
)
from codeintel.storage.database import (
    FileAnalysisView,
    IndexDatabase,
    IndexDatabaseError,
    IndexStats,
    SchemaVersionError,
)
from codeintel.storage.schema import SCHEMA_VERSION


def hash_file_bytes(path: Path) -> str:
    """Return lowercase hex SHA-256 of raw file bytes."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_bytes(data: bytes) -> str:
    """Return lowercase hex SHA-256 of ``data``."""
    import hashlib

    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class FileChangeSet:
    """Deterministic language-neutral file change classification."""

    added: tuple[str, ...]
    changed: tuple[str, ...]
    deleted: tuple[str, ...]
    unchanged: tuple[str, ...]

    @property
    def is_noop(self) -> bool:
        return not self.added and not self.changed and not self.deleted


@dataclass(frozen=True, slots=True)
class IndexWorkStats:
    """Exact incremental/full indexing work counters."""

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
    fts_documents: int


@dataclass(frozen=True, slots=True)
class ResolutionSurfaceEntry:
    """Symbol fields consulted for cross-file relation resolution."""

    qualified_name: str
    name: str
    kind: str
    parent_qualified_name: str | None


def resolution_surface(symbols: Sequence[Symbol]) -> frozenset[ResolutionSurfaceEntry]:
    """Build the deterministic global relation-resolution surface."""
    return frozenset(
        ResolutionSurfaceEntry(
            qualified_name=symbol.qualified_name,
            name=symbol.name,
            kind=symbol.kind.value,
            parent_qualified_name=symbol.parent_qualified_name,
        )
        for symbol in symbols
    )


def compute_changeset(
    *,
    current_hashes: Mapping[str, str],
    persisted_hashes: Mapping[str, str],
) -> FileChangeSet:
    """Classify paths by comparing current vs persisted content hashes."""
    current_paths = set(current_hashes)
    persisted_paths = set(persisted_hashes)
    added = tuple(sorted(current_paths - persisted_paths))
    deleted = tuple(sorted(persisted_paths - current_paths))
    shared = current_paths & persisted_paths
    changed = tuple(
        sorted(path for path in shared if current_hashes[path] != persisted_hashes[path])
    )
    unchanged = tuple(
        sorted(path for path in shared if current_hashes[path] == persisted_hashes[path])
    )
    return FileChangeSet(
        added=added,
        changed=changed,
        deleted=deleted,
        unchanged=unchanged,
    )


def discover_and_hash(
    root: Path,
    adapter: LanguageAdapter,
) -> tuple[tuple[Path, ...], dict[str, str]]:
    """Discover supported files and hash raw bytes for each relative path."""
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Repository indexing requires a directory: {root}")
    discovered = discover_source_files(root, adapter)
    resolved_root = root.resolve()
    hashes: dict[str, str] = {}
    for path in discovered:
        relative = path.resolve().relative_to(resolved_root).as_posix()
        hashes[relative] = hash_file_bytes(path)
    return discovered, hashes


def index_repository(
    root: Path,
    adapter: LanguageAdapter,
    relation_extractor: RelationExtractor,
    *,
    database_path: Path,
    full: bool = False,
) -> IndexWorkStats:
    """Build or incrementally update a schema-v2 SQLite index."""
    if full or not database_path.exists():
        return _full_index(
            root,
            adapter,
            relation_extractor,
            database_path=database_path,
            replace_existing=database_path.exists(),
        )
    return _incremental_index(
        root,
        adapter,
        relation_extractor,
        database_path=database_path,
    )


def _full_index(
    root: Path,
    adapter: LanguageAdapter,
    relation_extractor: RelationExtractor,
    *,
    database_path: Path,
    replace_existing: bool,
) -> IndexWorkStats:
    analysis = analyze_repository(root, adapter, relation_extractor)
    _, hashes = discover_and_hash(root, adapter)
    if not replace_existing:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with IndexDatabase(database_path, create=True) as database:
            stats = database.rebuild(analysis, content_hashes=hashes)
        return _stats_from_full(stats, hashes)

    database_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{database_path.name}.",
        suffix=".tmp.db",
        dir=str(database_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with IndexDatabase(tmp_path, create=True) as database:
            stats = database.rebuild(analysis, content_hashes=hashes)
            if database.schema_version() != SCHEMA_VERSION:
                raise IndexDatabaseError(
                    f"Internal error: temporary index schema version "
                    f"{database.schema_version()} != {SCHEMA_VERSION}"
                )
        os.replace(tmp_path, database_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return _stats_from_full(stats, hashes)


def _stats_from_full(stats: IndexStats, hashes: Mapping[str, str]) -> IndexWorkStats:
    file_count = len(hashes)
    return IndexWorkStats(
        mode="full",
        files_added=file_count,
        files_changed=0,
        files_deleted=0,
        files_unchanged=0,
        files_analyzed=file_count,
        relation_files_recomputed=file_count,
        symbols_rewritten=stats.symbols,
        code_units_rewritten=stats.code_units,
        files=stats.files,
        symbols=stats.symbols,
        code_units=stats.code_units,
        relations=stats.relations,
        fts_documents=stats.fts_documents,
    )


def _incremental_index(
    root: Path,
    adapter: LanguageAdapter,
    relation_extractor: RelationExtractor,
    *,
    database_path: Path,
) -> IndexWorkStats:
    with IndexDatabase(database_path, create=False) as database:
        if database.schema_version() != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Unsupported index schema version {database.schema_version()}; "
                f"expected {SCHEMA_VERSION}. Run `aicode index {root} --full`."
            )
        persisted_hashes = {
            path: content_sha256 for path, _, _, _, content_sha256 in database.load_files()
        }
        old_symbols = database.load_symbols()
        unchanged_views = database.load_file_analysis_views()

    _discovered, current_hashes = discover_and_hash(root, adapter)
    del _discovered
    changeset = compute_changeset(
        current_hashes=current_hashes,
        persisted_hashes=persisted_hashes,
    )
    if changeset.is_noop:
        with IndexDatabase(database_path, create=False) as database:
            counts = database.counts()
        return IndexWorkStats(
            mode="noop",
            files_added=0,
            files_changed=0,
            files_deleted=0,
            files_unchanged=len(changeset.unchanged),
            files_analyzed=0,
            relation_files_recomputed=0,
            symbols_rewritten=0,
            code_units_rewritten=0,
            files=counts.files,
            symbols=counts.symbols,
            code_units=counts.code_units,
            relations=counts.relations,
            fts_documents=counts.fts_documents,
        )

    resolved_root = root.resolve()
    analyzed: dict[str, AnalysisResult] = {}
    for relative in (*changeset.added, *changeset.changed):
        path = resolved_root / relative
        analyzed[relative] = adapter.analyze_file(path, repository_root=root)

    proposed_files = tuple(
        analyzed[rel]
        if rel in analyzed
        else _analysis_from_view(unchanged_views[rel], resolved_root / rel)
        for rel in sorted({*changeset.unchanged, *changeset.added, *changeset.changed})
    )
    proposed_symbols_map, symbol_paths = _build_symbol_index(proposed_files)
    proposed_symbols = tuple(proposed_symbols_map[name] for name in sorted(proposed_symbols_map))

    old_surface = resolution_surface(old_symbols)
    new_surface = resolution_surface(proposed_symbols)
    surface_changed = old_surface != new_surface

    if surface_changed:
        relation_targets = tuple(sorted(current_hashes))
        relations = _extract_all_relations(
            root=root,
            resolved_root=resolved_root,
            relative_paths=relation_targets,
            analyzed=analyzed,
            unchanged_views=unchanged_views,
            symbols_by_qualified_name=proposed_symbols_map,
            symbol_paths=symbol_paths,
            relation_extractor=relation_extractor,
            files_for_contains=proposed_files,
        )
    else:
        relation_targets = tuple(sorted((*changeset.added, *changeset.changed)))
        language_relations = _extract_local_relations(
            root=root,
            resolved_root=resolved_root,
            relative_paths=relation_targets,
            analyzed=analyzed,
            symbols_by_qualified_name=proposed_symbols_map,
            relation_extractor=relation_extractor,
        )
        contains = _derive_contains_relations(proposed_files, symbol_paths)
        refreshed_paths = {(resolved_root / rel).resolve() for rel in relation_targets}
        local_contains = tuple(
            relation for relation in contains if relation.path.resolve() in refreshed_paths
        )
        relations = _dedupe_relations((*local_contains, *language_relations))

    symbols_rewritten = sum(
        len(analyzed[rel].symbols) for rel in (*changeset.added, *changeset.changed)
    )
    code_units_rewritten = sum(
        len(analyzed[rel].code_units) for rel in (*changeset.added, *changeset.changed)
    )

    with IndexDatabase(database_path, create=False) as database:
        counts = database.apply_incremental_update(
            root=resolved_root,
            changeset=changeset,
            current_hashes=current_hashes,
            analyzed=analyzed,
            relations=relations,
            global_relation_refresh=surface_changed,
        )

    return IndexWorkStats(
        mode="incremental",
        files_added=len(changeset.added),
        files_changed=len(changeset.changed),
        files_deleted=len(changeset.deleted),
        files_unchanged=len(changeset.unchanged),
        files_analyzed=len(changeset.added) + len(changeset.changed),
        relation_files_recomputed=len(relation_targets),
        symbols_rewritten=symbols_rewritten,
        code_units_rewritten=code_units_rewritten,
        files=counts.files,
        symbols=counts.symbols,
        code_units=counts.code_units,
        relations=counts.relations,
        fts_documents=counts.fts_documents,
    )


def _analysis_from_view(view: FileAnalysisView, path: Path) -> AnalysisResult:
    return AnalysisResult(
        path=path,
        language_id=view.language_id,
        module_name=view.module_name,
        symbols=view.symbols,
        code_units=view.code_units,
        has_syntax_errors=view.has_syntax_errors,
    )


def _extract_local_relations(
    *,
    root: Path,
    resolved_root: Path,
    relative_paths: Sequence[str],
    analyzed: Mapping[str, AnalysisResult],
    symbols_by_qualified_name: Mapping[str, Symbol],
    relation_extractor: RelationExtractor,
) -> tuple[Relation, ...]:
    relations: list[Relation] = []
    for relative in relative_paths:
        analysis = analyzed[relative]
        path = analysis.path if analysis.path is not None else resolved_root / relative
        relations.extend(
            relation_extractor.extract_relations(
                path,
                repository_root=root,
                analysis=analysis,
                symbols_by_qualified_name=symbols_by_qualified_name,
            )
        )
    return tuple(relations)


def _extract_all_relations(
    *,
    root: Path,
    resolved_root: Path,
    relative_paths: Sequence[str],
    analyzed: Mapping[str, AnalysisResult],
    unchanged_views: Mapping[str, FileAnalysisView],
    symbols_by_qualified_name: Mapping[str, Symbol],
    symbol_paths: Mapping[str, Path],
    relation_extractor: RelationExtractor,
    files_for_contains: Sequence[AnalysisResult],
) -> tuple[Relation, ...]:
    language_relations: list[Relation] = []
    for relative in relative_paths:
        if relative in analyzed:
            analysis = analyzed[relative]
        else:
            analysis = _analysis_from_view(unchanged_views[relative], resolved_root / relative)
        path = analysis.path if analysis.path is not None else resolved_root / relative
        language_relations.extend(
            relation_extractor.extract_relations(
                path,
                repository_root=root,
                analysis=analysis,
                symbols_by_qualified_name=symbols_by_qualified_name,
            )
        )
    contains = _derive_contains_relations(tuple(files_for_contains), dict(symbol_paths))
    return _dedupe_relations((*contains, *language_relations))
