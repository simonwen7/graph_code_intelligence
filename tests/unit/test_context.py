"""Unit tests for token-budget context compilation."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.context import (
    CONTEXT_CANDIDATE_LIMIT,
    ContextOmissionReason,
    SimpleTokenCounter,
    compile_context,
    render_context_block,
    spans_overlap,
)
from codeintel.models import (
    ContributionSource,
    RankContribution,
    RerankedResult,
    RerankExplanation,
    SearchResult,
    SourceSpan,
    SymbolKind,
)


def _span(start_byte: int, end_byte: int, start_line: int = 1, end_line: int = 1) -> SourceSpan:
    return SourceSpan(start_line, end_line, start_byte, end_byte)


def _hit(
    qname: str,
    source_text: str,
    *,
    path: str = "a.py",
    kind: SymbolKind = SymbolKind.FUNCTION,
    span: SourceSpan | None = None,
    score: float = 1.0,
) -> SearchResult:
    resolved = span if span is not None else _span(0, max(1, len(source_text.encode("utf-8"))))
    return SearchResult(
        symbol_qualified_name=qname,
        kind=kind,
        path=Path(path),
        span=resolved,
        signature=None,
        source_text=source_text,
        score=score,
    )


def _reranked(result: SearchResult, *, final_rank: int = 1) -> RerankedResult:
    return RerankedResult(
        result=result,
        explanation=RerankExplanation(
            original_rank=final_rank,
            final_rank=final_rank,
            rank_delta=0,
            contributions=(
                RankContribution(
                    source=ContributionSource.GRAPH_BASE,
                    rank=final_rank,
                    rrf_contribution=1.0 / (60 + final_rank),
                ),
            ),
            relation_evidence=(),
        ),
    )


class FakeTokenCounter:
    """Deterministic per-string cost map for boundary tests."""

    def __init__(self, costs: dict[str, int] | None = None, *, default: int = 1) -> None:
        self._costs = costs or {}
        self._default = default

    @property
    def counter_id(self) -> str:
        return "fake-counter-v1"

    def count(self, text: str) -> int:
        if text in self._costs:
            return self._costs[text]
        return self._default if text else 0


def test_simple_token_counter_contract() -> None:
    counter = SimpleTokenCounter()
    assert counter.counter_id == "simple-lexical-v1"
    assert counter.count("") == 0
    assert counter.count("hello") == 1
    assert counter.count("hello world") == 2
    assert counter.count("foo_bar") == 1
    assert counter.count("camelCase") == 1
    assert counter.count("foo123") == 1
    assert counter.count("123 45") == 2
    assert counter.count("123.45") == 3  # 123 + . + 45
    assert counter.count("a+b") == 3
    assert counter.count("::") == 2
    assert counter.count("==") == 2
    assert counter.count("  \n\t") == 0
    assert counter.count("你好") == 2
    assert counter.count("🚀") == 1
    assert counter.count("café") == 2  # caf + é
    assert counter.count("π") == 1
    assert counter.count("x") == counter.count("x")
    assert counter.count("hello") >= 0


def test_budget_validation_and_empty() -> None:
    counter = SimpleTokenCounter()
    with pytest.raises(ValueError, match="token_budget"):
        compile_context((), token_budget=-1, token_counter=counter)
    empty = compile_context((), token_budget=10, token_counter=counter)
    assert empty.text == ""
    assert empty.used_tokens == 0
    assert empty.remaining_tokens == 10
    assert empty.candidate_count == 0


def test_render_and_source_preservation() -> None:
    source = "def greet():\n    return 'café 你好 🚀'\n"
    rendered = render_context_block(
        symbol_qualified_name="mod.greet",
        kind=SymbolKind.FUNCTION,
        path=Path("unicode_unit.py"),
        span=_span(0, 10, 1, 2),
        source_text=source,
    )
    assert rendered.endswith("\n")
    assert "signature:" not in rendered
    assert "score:" not in rendered
    assert "code:\n" + source + "\n=== END CODE UNIT ===\n" in rendered
    assert "café 你好 🚀" in rendered


def test_greedy_skip_and_no_truncation() -> None:
    small = "def tiny() -> None:\n    return\n"
    huge = " ".join(["pay"] * 400)
    r1 = _reranked(_hit("huge", huge, path="big.py"), final_rank=1)
    r2 = _reranked(_hit("tiny", small, path="small.py"), final_rank=2)
    counter = SimpleTokenCounter()
    budget = counter.count(
        render_context_block(
            symbol_qualified_name="tiny",
            kind=SymbolKind.FUNCTION,
            path=Path("small.py"),
            span=_span(0, 10),
            source_text=small,
        )
    )
    compiled = compile_context((r1, r2), token_budget=budget, token_counter=counter)
    assert [block.symbol_qualified_name for block in compiled.blocks] == ["tiny"]
    assert compiled.omissions[0].reason is ContextOmissionReason.OVERSIZED
    assert huge not in compiled.text
    assert "pay pay pay" not in compiled.text
    assert small in compiled.blocks[0].rendered_text
    assert counter.count(compiled.text) == compiled.used_tokens <= budget


def test_exact_budget_boundaries_with_fake_counter() -> None:
    source = "def a():\n    return 1\n"
    rendered = render_context_block(
        symbol_qualified_name="a",
        kind=SymbolKind.FUNCTION,
        path=Path("a.py"),
        span=_span(0, 5),
        source_text=source,
    )
    # Exact standalone fit.
    counter = FakeTokenCounter({rendered: 10, "": 0}, default=99)
    item = _reranked(_hit("a", source))
    ok = compile_context((item,), token_budget=10, token_counter=counter)
    assert ok.selected_count == 1
    # Oversize by one.
    fail = compile_context((item,), token_budget=9, token_counter=counter)
    assert fail.omissions[0].reason is ContextOmissionReason.OVERSIZED

    # Second block: prospective final exceeds by separator cost.
    source_b = "def b():\n    return 2\n"
    rendered_b = render_context_block(
        symbol_qualified_name="b",
        kind=SymbolKind.FUNCTION,
        path=Path("b.py"),
        span=_span(0, 5),
        source_text=source_b,
    )
    joined = f"{rendered}\n{rendered_b}"
    counter2 = FakeTokenCounter(
        {rendered: 5, rendered_b: 5, joined: 11, "": 0},
        default=99,
    )
    two = compile_context(
        (_reranked(_hit("a", source)), _reranked(_hit("b", source_b, path="b.py"))),
        token_budget=10,
        token_counter=counter2,
    )
    assert [block.symbol_qualified_name for block in two.blocks] == ["a"]
    assert two.omissions[0].reason is ContextOmissionReason.BUDGET


def test_qname_dedup_and_order() -> None:
    counter = SimpleTokenCounter()
    a1 = _reranked(_hit("dup", "def one():\n    return 1\n", path="a.py"), final_rank=1)
    a2 = _reranked(
        _hit("dup", "def one():\n    return 999\n", path="a.py", score=99.0),
        final_rank=2,
    )
    b = _reranked(_hit("other", "def two():\n    return 2\n", path="b.py"), final_rank=3)
    budget = 10_000
    compiled = compile_context((a1, a2, b), token_budget=budget, token_counter=counter)
    assert [block.symbol_qualified_name for block in compiled.blocks] == ["dup", "other"]
    assert "return 1" in compiled.text
    assert "return 999" not in compiled.text
    assert compiled.omissions[0].reason is ContextOmissionReason.DUPLICATE
    assert compiled.candidate_count == 3
    assert compiled.selected_count + compiled.omitted_count == 3


def test_overlap_policies() -> None:
    counter = SimpleTokenCounter()
    parent_src = "class A:\n    def m(self):\n        return 1\n"
    child_src = "    def m(self):\n        return 1\n"
    parent = _reranked(
        _hit(
            "mod.A",
            parent_src,
            kind=SymbolKind.CLASS,
            path="t.py",
            span=_span(0, 40, 1, 3),
        ),
        final_rank=1,
    )
    child = _reranked(
        _hit(
            "mod.A.m",
            child_src,
            kind=SymbolKind.METHOD,
            path="t.py",
            span=_span(10, 40, 2, 3),
        ),
        final_rank=2,
    )
    assert spans_overlap(parent.result.span, child.result.span)

    selected_parent = compile_context((parent, child), token_budget=10_000, token_counter=counter)
    assert [block.symbol_qualified_name for block in selected_parent.blocks] == ["mod.A"]
    assert selected_parent.omissions[0].reason is ContextOmissionReason.OVERLAP

    selected_child = compile_context((child, parent), token_budget=10_000, token_counter=counter)
    assert [block.symbol_qualified_name for block in selected_child.blocks] == ["mod.A.m"]
    assert selected_child.omissions[0].reason is ContextOmissionReason.OVERLAP

    # Omitted oversized parent does not block child.
    huge_parent = _reranked(
        _hit(
            "mod.Huge",
            " ".join(["CLASS"] * 400),
            kind=SymbolKind.CLASS,
            path="t.py",
            span=_span(0, 2000, 1, 50),
        ),
        final_rank=1,
    )
    child_fit = _reranked(
        _hit(
            "mod.Huge.m",
            "def m(self):\n    return 1\n",
            kind=SymbolKind.METHOD,
            path="t.py",
            span=_span(10, 40, 2, 3),
        ),
        final_rank=2,
    )
    # Use enough budget for child block but not huge parent.
    child_rendered = render_context_block(
        symbol_qualified_name="mod.Huge.m",
        kind=SymbolKind.METHOD,
        path=Path("t.py"),
        span=child_fit.result.span,
        source_text=child_fit.result.source_text,
    )
    budget = counter.count(child_rendered)
    packed = compile_context((huge_parent, child_fit), token_budget=budget, token_counter=counter)
    assert packed.omissions[0].reason is ContextOmissionReason.OVERSIZED
    assert [block.symbol_qualified_name for block in packed.blocks] == ["mod.Huge.m"]


def test_same_file_disjoint_and_cross_file() -> None:
    counter = SimpleTokenCounter()
    left = _reranked(_hit("a", "def a():\n    return 1\n", path="t.py", span=_span(0, 20)))
    right = _reranked(_hit("b", "def b():\n    return 2\n", path="t.py", span=_span(50, 70)))
    compiled = compile_context((left, right), token_budget=10_000, token_counter=counter)
    assert compiled.selected_count == 2

    twin = _reranked(
        _hit("c", "def c():\n    return 3\n", path="other.py", span=_span(0, 20)),
    )
    cross = compile_context((left, twin), token_budget=10_000, token_counter=counter)
    assert cross.selected_count == 2


def test_zero_budget_and_invariants() -> None:
    counter = SimpleTokenCounter()
    item = _reranked(_hit("a", "def a():\n    return 1\n"))
    compiled = compile_context((item,), token_budget=0, token_counter=counter)
    assert compiled.blocks == ()
    assert compiled.omissions[0].reason is ContextOmissionReason.OVERSIZED
    assert compiled.used_tokens == 0
    assert compiled.remaining_tokens == 0
    assert counter.count(compiled.text) == compiled.used_tokens
    assert compiled.candidate_count == compiled.selected_count + compiled.omitted_count


def test_standalone_counts_are_diagnostic_only() -> None:
    counter = SimpleTokenCounter()
    a = _reranked(_hit("a", "def a():\n    return 1\n", path="a.py"))
    b = _reranked(_hit("b", "def b():\n    return 2\n", path="b.py"))
    compiled = compile_context((a, b), token_budget=10_000, token_counter=counter)
    assert len(compiled.blocks) == 2
    assert "\n\n" in compiled.text
    assert "=== END CODE UNIT ===\n\n\n=== CODE UNIT ===" not in compiled.text
    assert "=== END CODE UNIT ===\n\n=== CODE UNIT ===" in compiled.text
    for block in compiled.blocks:
        assert block.standalone_tokens == counter.count(block.rendered_text)
    # Authoritative invariant uses final text, not sum(standalone_tokens).
    # (simple-lexical-v1 ignores whitespace separators, so sums may coincide.)
    assert counter.count(compiled.text) == compiled.used_tokens
    assert compiled.used_tokens <= compiled.token_budget

    # With a counter that prices separators, standalone sums diverge from used.
    rendered_a = compiled.blocks[0].rendered_text
    rendered_b = compiled.blocks[1].rendered_text
    joined = f"{rendered_a}\n{rendered_b}"
    priced = FakeTokenCounter(
        {rendered_a: 5, rendered_b: 5, joined: 12, "": 0},
        default=99,
    )
    priced_compiled = compile_context((a, b), token_budget=100, token_counter=priced)
    assert sum(block.standalone_tokens for block in priced_compiled.blocks) == 10
    assert priced_compiled.used_tokens == 12
    assert priced_compiled.token_counter_id == "fake-counter-v1"


def test_determinism_and_purity() -> None:
    counter = SimpleTokenCounter()
    items = (
        _reranked(_hit("a", "def a():\n    return 1\n", path="a.py")),
        _reranked(_hit("b", "def b():\n    return 2\n", path="b.py")),
    )
    first = compile_context(items, token_budget=10_000, token_counter=counter)
    second = compile_context(items, token_budget=10_000, token_counter=counter)
    assert first == second
    # Pure: no db/repo needed — this call is the proof.
    assert first.token_counter_id == "simple-lexical-v1"


def test_touching_spans_are_not_overlap() -> None:
    assert not spans_overlap(_span(0, 10), _span(10, 20))
    assert spans_overlap(_span(0, 10), _span(9, 20))
    counter = SimpleTokenCounter()
    left = _reranked(_hit("a", "def a():\n    return 1\n", path="t.py", span=_span(0, 10)))
    right = _reranked(_hit("b", "def b():\n    return 2\n", path="t.py", span=_span(10, 20)))
    compiled = compile_context((left, right), token_budget=10_000, token_counter=counter)
    assert compiled.selected_count == 2


def test_duplicate_after_omitted_first_still_duplicate() -> None:
    """First qname occurrence may be OVERSIZED; later same qname is still DUPLICATE."""
    counter = SimpleTokenCounter()
    huge = " ".join(["pay"] * 400)
    first = _reranked(_hit("dup", huge, path="big.py"), final_rank=1)
    second = _reranked(_hit("dup", "def tiny():\n    return\n", path="small.py"), final_rank=2)
    third = _reranked(_hit("other", "def other():\n    return\n", path="o.py"), final_rank=3)
    other_rendered = render_context_block(
        symbol_qualified_name="other",
        kind=SymbolKind.FUNCTION,
        path=Path("o.py"),
        span=_span(0, 10),
        source_text="def other():\n    return\n",
    )
    budget = counter.count(other_rendered)
    compiled = compile_context((first, second, third), token_budget=budget, token_counter=counter)
    assert [o.reason for o in compiled.omissions] == [
        ContextOmissionReason.OVERSIZED,
        ContextOmissionReason.DUPLICATE,
    ]
    assert [block.symbol_qualified_name for block in compiled.blocks] == ["other"]
    assert compiled.candidate_count == 3


def test_budget_omitted_parent_does_not_block_child() -> None:
    """Parent fits total budget alone but not after prior selection → BUDGET; child may fit."""
    filler_src = "def filler():\n    return 1\n"
    parent_src = "class Parent:\n    def m(self):\n        return 1\n"
    child_src = "    def m(self):\n        return 1\n"
    filler = _reranked(_hit("filler", filler_src, path="t.py", span=_span(0, 20)), final_rank=1)
    parent = _reranked(
        _hit(
            "mod.Parent",
            parent_src,
            kind=SymbolKind.CLASS,
            path="t.py",
            span=_span(100, 200, 10, 20),
        ),
        final_rank=2,
    )
    child = _reranked(
        _hit(
            "mod.Parent.m",
            child_src,
            kind=SymbolKind.METHOD,
            path="t.py",
            span=_span(120, 160, 11, 13),
        ),
        final_rank=3,
    )
    rendered_filler = render_context_block(
        symbol_qualified_name="filler",
        kind=SymbolKind.FUNCTION,
        path=Path("t.py"),
        span=_span(0, 20),
        source_text=filler_src,
    )
    rendered_parent = render_context_block(
        symbol_qualified_name="mod.Parent",
        kind=SymbolKind.CLASS,
        path=Path("t.py"),
        span=parent.result.span,
        source_text=parent_src,
    )
    rendered_child = render_context_block(
        symbol_qualified_name="mod.Parent.m",
        kind=SymbolKind.METHOD,
        path=Path("t.py"),
        span=child.result.span,
        source_text=child_src,
    )
    joined_filler_parent = f"{rendered_filler}\n{rendered_parent}"
    joined_filler_child = f"{rendered_filler}\n{rendered_child}"
    # Parent standalone fits budget; with filler it exceeds; child + filler fits.
    counter = FakeTokenCounter(
        {
            rendered_filler: 40,
            rendered_parent: 50,
            rendered_child: 20,
            joined_filler_parent: 101,
            joined_filler_child: 70,
            "": 0,
        },
        default=99,
    )
    compiled = compile_context(
        (filler, parent, child),
        token_budget=100,
        token_counter=counter,
    )
    assert compiled.omissions[0].reason is ContextOmissionReason.BUDGET
    assert compiled.omissions[0].symbol_qualified_name == "mod.Parent"
    assert [block.symbol_qualified_name for block in compiled.blocks] == ["filler", "mod.Parent.m"]
    assert counter.count(compiled.text) == compiled.used_tokens <= 100


def test_greedy_continue_after_budget_omission() -> None:
    a_src = "def a():\n    return 1\n"
    b_src = "def b():\n    return 2\n"
    c_src = "def c():\n    return 3\n"
    rendered_a = render_context_block(
        symbol_qualified_name="a",
        kind=SymbolKind.FUNCTION,
        path=Path("a.py"),
        span=_span(0, 5),
        source_text=a_src,
    )
    rendered_b = render_context_block(
        symbol_qualified_name="b",
        kind=SymbolKind.FUNCTION,
        path=Path("b.py"),
        span=_span(0, 5),
        source_text=b_src,
    )
    rendered_c = render_context_block(
        symbol_qualified_name="c",
        kind=SymbolKind.FUNCTION,
        path=Path("c.py"),
        span=_span(0, 5),
        source_text=c_src,
    )
    joined_ab = f"{rendered_a}\n{rendered_b}"
    joined_ac = f"{rendered_a}\n{rendered_c}"
    counter = FakeTokenCounter(
        {
            rendered_a: 40,
            rendered_b: 40,
            rendered_c: 20,
            joined_ab: 90,
            joined_ac: 70,
            "": 0,
        },
        default=99,
    )
    compiled = compile_context(
        (
            _reranked(_hit("a", a_src, path="a.py")),
            _reranked(_hit("b", b_src, path="b.py")),
            _reranked(_hit("c", c_src, path="c.py")),
        ),
        token_budget=80,
        token_counter=counter,
    )
    assert [block.symbol_qualified_name for block in compiled.blocks] == ["a", "c"]
    assert compiled.omissions[0].reason is ContextOmissionReason.BUDGET
    assert counter.count(compiled.text) == compiled.used_tokens == 70


def test_non_additive_counter_enforces_final_budget() -> None:
    """Prospective full-text counting must not assume count(A)+count(B)==count(A+sep+B)."""

    class NonAdditiveCounter:
        @property
        def counter_id(self) -> str:
            return "non-additive-v1"

        def count(self, text: str) -> int:
            # Superadditive: concatenation costs more than the sum of parts.
            parts = [part for part in text.split("\n\n") if part]
            base = sum(max(1, len(part) // 10) for part in parts) if parts else 0
            if len(parts) > 1:
                return base + 5 * (len(parts) - 1)
            return base

    a = _reranked(_hit("a", "def a():\n    return 1\n", path="a.py"))
    b = _reranked(_hit("b", "def b():\n    return 2\n", path="b.py"))
    counter = NonAdditiveCounter()
    rendered_a = render_context_block(
        symbol_qualified_name="a",
        kind=SymbolKind.FUNCTION,
        path=Path("a.py"),
        span=_span(0, max(1, len(a.result.source_text.encode()))),
        source_text=a.result.source_text,
    )
    rendered_b = render_context_block(
        symbol_qualified_name="b",
        kind=SymbolKind.FUNCTION,
        path=Path("b.py"),
        span=_span(0, max(1, len(b.result.source_text.encode()))),
        source_text=b.result.source_text,
    )
    standalone_sum = counter.count(rendered_a) + counter.count(rendered_b)
    joined = f"{rendered_a}\n{rendered_b}"
    assert counter.count(joined) != standalone_sum

    budget = counter.count(joined) - 1
    compiled = compile_context((a, b), token_budget=budget, token_counter=counter)
    assert compiled.selected_count == 1
    assert compiled.omissions[0].reason is ContextOmissionReason.BUDGET
    assert counter.count(compiled.text) == compiled.used_tokens <= budget


def test_negative_custom_counter_rejected() -> None:
    class NegativeCounter:
        @property
        def counter_id(self) -> str:
            return "negative-v1"

        def count(self, text: str) -> int:
            return -1 if text else 0

    item = _reranked(_hit("a", "def a():\n    return 1\n"))
    with pytest.raises(ValueError, match="non-negative"):
        compile_context((item,), token_budget=10, token_counter=NegativeCounter())


def test_zero_cost_custom_counter_allowed() -> None:
    class ZeroCounter:
        @property
        def counter_id(self) -> str:
            return "zero-v1"

        def count(self, text: str) -> int:
            return 0

    item = _reranked(_hit("a", "def a():\n    return 1\n"))
    compiled = compile_context((item,), token_budget=0, token_counter=ZeroCounter())
    assert compiled.selected_count == 1
    assert compiled.used_tokens == 0
    assert compiled.token_counter_id == "zero-v1"


def test_source_without_trailing_newline_preserved() -> None:
    source = "\tdef f():\n\t\treturn 'π'"
    assert not source.endswith("\n")
    rendered = render_context_block(
        symbol_qualified_name="mod.f",
        kind=SymbolKind.FUNCTION,
        path=Path("u.py"),
        span=_span(0, 10, 1, 2),
        source_text=source,
    )
    assert f"code:\n{source}\n=== END CODE UNIT ===\n" in rendered
    compiled = compile_context(
        (_reranked(_hit("mod.f", source, path="u.py")),),
        token_budget=10_000,
        token_counter=SimpleTokenCounter(),
    )
    assert compiled.blocks[0].source_text == source
    assert source in compiled.text


def test_context_candidate_limit_constant() -> None:
    assert CONTEXT_CANDIDATE_LIMIT == 20
