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
