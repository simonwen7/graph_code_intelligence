"""Language-neutral semantic program representations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class SymbolKind(StrEnum):
    """Kinds of extracted program symbols."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Byte- and line-bounded location within a source file.

    Line numbers are 1-based and inclusive.
    Byte offsets are zero-based with an inclusive start and exclusive end.
    """

    start_line: int
    end_line: int
    start_byte: int
    end_byte: int

    def __post_init__(self) -> None:
        if self.start_line < 1 or self.end_line < 1:
            raise ValueError("line numbers must be 1-based and positive")
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        if self.start_byte < 0 or self.end_byte < 0:
            raise ValueError("byte offsets must be non-negative")
        if self.end_byte < self.start_byte:
            raise ValueError("end_byte must be >= start_byte")


@dataclass(frozen=True, slots=True)
class Symbol:
    """A named semantic entity extracted from source code."""

    name: str
    qualified_name: str
    kind: SymbolKind
    span: SourceSpan
    signature: str | None
    parent_qualified_name: str | None


@dataclass(frozen=True, slots=True)
class CodeUnit:
    """A contiguous source fragment associated with a non-module symbol."""

    symbol_qualified_name: str
    kind: SymbolKind
    source_text: str
    span: SourceSpan


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Structured analysis output for a single source file or buffer."""

    path: Path | None
    language_id: str
    module_name: str
    symbols: tuple[Symbol, ...]
    code_units: tuple[CodeUnit, ...]
    has_syntax_errors: bool


class RelationKind(StrEnum):
    """Kinds of static relationships between program symbols."""

    CONTAINS = "contains"
    IMPORTS = "imports"
    REFERENCES = "references"
    CALLS = "calls"
    INHERITS = "inherits"


class ResolutionStatus(StrEnum):
    """Honesty level for static resolution of a relationship target."""

    RESOLVED = "resolved"
    PROBABLE = "probable"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class Relation:
    """A directed static relationship originating in a source file."""

    kind: RelationKind
    source_qualified_name: str
    target_qualified_name: str | None
    target_text: str
    resolution: ResolutionStatus
    path: Path
    span: SourceSpan | None

    def __post_init__(self) -> None:
        if self.resolution in {ResolutionStatus.RESOLVED, ResolutionStatus.PROBABLE}:
            if self.target_qualified_name is None:
                raise ValueError(f"{self.resolution.value} relations require target_qualified_name")
        elif self.resolution is ResolutionStatus.UNRESOLVED:
            if self.target_qualified_name is not None:
                raise ValueError("unresolved relations must not fabricate a target qualified name")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A retrieval hit over a persisted CodeUnit snapshot.

    ``score`` is always higher-is-better within the retrieval method that
    produced the result (lexical BM25 uses ``-raw_sqlite_bm25``, dense uses
    cosine similarity, hybrid uses RRF). Scores are not probabilities and are
    not comparable across retrieval modes.
    """

    symbol_qualified_name: str
    kind: SymbolKind
    path: Path
    span: SourceSpan
    signature: str | None
    source_text: str
    score: float
