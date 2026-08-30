"""Token-budget context compiler over ordered RerankedResult sequences."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from codeintel.models import RerankedResult, SourceSpan, SymbolKind

CONTEXT_CANDIDATE_LIMIT = 20

_SIMPLE_TOKEN_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|[^\s]",
)


@runtime_checkable
class TokenCounter(Protocol):
    """Language-neutral token / budget-unit counter."""

    @property
    def counter_id(self) -> str:
        """Stable identifier for the counting rule."""

    def count(self, text: str) -> int:
        """Return a non-negative deterministic count for ``text``."""


@dataclass(frozen=True, slots=True)
class SimpleTokenCounter:
    """Deterministic model-independent lexical token estimate.

    Rule ``simple-lexical-v1``:

    - ASCII identifier/word sequences count as one unit
    - numeric sequences count as one unit
    - every other non-whitespace character counts as one unit

    This is **not** OpenAI/Anthropic/Gemini/MiniLM tokenization.
    """

    @property
    def counter_id(self) -> str:
        return "simple-lexical-v1"

    def count(self, text: str) -> int:
        return len(_SIMPLE_TOKEN_RE.findall(text))


class ContextOmissionReason(StrEnum):
    """Why a candidate was not selected into compiled context."""

    DUPLICATE = "duplicate"
    OVERLAP = "overlap"
    OVERSIZED = "oversized"
    BUDGET = "budget"


@dataclass(frozen=True, slots=True)
class ContextOmission:
    """Structured omission diagnostic for one input candidate."""

    symbol_qualified_name: str
    source_rank: int
    reason: ContextOmissionReason

    def __post_init__(self) -> None:
        if self.source_rank < 1:
            raise ValueError("source_rank must be >= 1")


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """One selected whole-CodeUnit context block."""

    symbol_qualified_name: str
    kind: SymbolKind
    path: Path
    span: SourceSpan
    source_rank: int
    source_text: str
    rendered_text: str
    standalone_tokens: int


@dataclass(frozen=True, slots=True)
class CompiledContext:
    """Budget-validated compiled context over selected CodeUnits."""

    text: str
    blocks: tuple[ContextBlock, ...]
    omissions: tuple[ContextOmission, ...]
    token_budget: int
    used_tokens: int
    remaining_tokens: int
    token_counter_id: str
    candidate_count: int
    selected_count: int
    omitted_count: int


def render_context_block(
    *,
    symbol_qualified_name: str,
    kind: SymbolKind,
    path: Path,
    span: SourceSpan,
    source_text: str,
) -> str:
    """Render one deterministic language-neutral context block (ends with newline)."""
    location = f"{path.as_posix()}:L{span.start_line}-{span.end_line}"
    return (
        "=== CODE UNIT ===\n"
        f"symbol: {symbol_qualified_name}\n"
        f"kind: {kind.value}\n"
        f"location: {location}\n"
        "code:\n"
        f"{source_text}\n"
        "=== END CODE UNIT ===\n"
    )


def spans_overlap(left: SourceSpan, right: SourceSpan) -> bool:
    """Return True when two spans on the same file would overlap (caller checks path)."""
    return left.start_byte < right.end_byte and right.start_byte < left.end_byte


def _require_nonnegative_count(value: object) -> int:
    """Reject broken TokenCounter outputs that would invalidate budget arithmetic."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("token_counter.count must return a non-negative int")
    return value


def _join_rendered_blocks(left: str, right: str) -> str:
    """Join two rendered blocks with exactly one blank line (``\\n\\n``).

    Each block already ends with a trailing newline, so a single extra ``\\n``
    between them yields the canonical ``\\n\\n`` separator without a triple newline.
    """
    return f"{left}\n{right}"


def compile_context(
    results: Sequence[RerankedResult],
    *,
    token_budget: int,
    token_counter: TokenCounter,
) -> CompiledContext:
    """Pack whole CodeUnits into a token-budgeted context without retrieval.

    Candidates are processed in input order. Selection is greedy skip-to-fit with
    qname dedup and byte-span overlap suppression against **selected** blocks only.
    """
    if token_budget < 0:
        raise ValueError("token_budget must be >= 0")

    if not results:
        return CompiledContext(
            text="",
            blocks=(),
            omissions=(),
            token_budget=token_budget,
            used_tokens=0,
            remaining_tokens=token_budget,
            token_counter_id=token_counter.counter_id,
            candidate_count=0,
            selected_count=0,
            omitted_count=0,
        )

    selected: list[ContextBlock] = []
    omissions: list[ContextOmission] = []
    seen_qnames: set[str] = set()
    current_text = ""

    for source_rank, item in enumerate(results, start=1):
        result = item.result
        qname = result.symbol_qualified_name

        if qname in seen_qnames:
            omissions.append(
                ContextOmission(
                    symbol_qualified_name=qname,
                    source_rank=source_rank,
                    reason=ContextOmissionReason.DUPLICATE,
                )
            )
            continue
        seen_qnames.add(qname)

        overlaps = False
        for block in selected:
            if block.path.as_posix() == result.path.as_posix() and spans_overlap(
                block.span,
                result.span,
            ):
                overlaps = True
                break
        if overlaps:
            omissions.append(
                ContextOmission(
                    symbol_qualified_name=qname,
                    source_rank=source_rank,
                    reason=ContextOmissionReason.OVERLAP,
                )
            )
            continue

        rendered = render_context_block(
            symbol_qualified_name=qname,
            kind=result.kind,
            path=result.path,
            span=result.span,
            source_text=result.source_text,
        )
        standalone = _require_nonnegative_count(token_counter.count(rendered))
        if standalone > token_budget:
            omissions.append(
                ContextOmission(
                    symbol_qualified_name=qname,
                    source_rank=source_rank,
                    reason=ContextOmissionReason.OVERSIZED,
                )
            )
            continue

        if current_text:
            prospective = _join_rendered_blocks(current_text, rendered)
        else:
            prospective = rendered
        if _require_nonnegative_count(token_counter.count(prospective)) > token_budget:
            omissions.append(
                ContextOmission(
                    symbol_qualified_name=qname,
                    source_rank=source_rank,
                    reason=ContextOmissionReason.BUDGET,
                )
            )
            continue

        selected.append(
            ContextBlock(
                symbol_qualified_name=qname,
                kind=result.kind,
                path=result.path,
                span=result.span,
                source_rank=source_rank,
                source_text=result.source_text,
                rendered_text=rendered,
                standalone_tokens=standalone,
            )
        )
        current_text = prospective

    used = _require_nonnegative_count(token_counter.count(current_text))
    if used > token_budget:
        raise RuntimeError("internal error: compiled context exceeds token budget")

    return CompiledContext(
        text=current_text,
        blocks=tuple(selected),
        omissions=tuple(omissions),
        token_budget=token_budget,
        used_tokens=used,
        remaining_tokens=token_budget - used,
        token_counter_id=token_counter.counter_id,
        candidate_count=len(results),
        selected_count=len(selected),
        omitted_count=len(omissions),
    )
