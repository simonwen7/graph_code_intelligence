"""Unit tests for the C++ semantic adapter."""

from __future__ import annotations

from pathlib import Path

from codeintel.languages.cpp import CppAdapter
from codeintel.models import Symbol, SymbolKind

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "cpp_repo"


def _symbols(path: Path) -> dict[str, Symbol]:
    result = CppAdapter().analyze_file(path, repository_root=FIXTURE)
    return {symbol.qualified_name: symbol for symbol in result.symbols}


def test_file_root_module_and_module_name() -> None:
    path = FIXTURE / "pricing.cpp"
    result = CppAdapter().analyze_file(path, repository_root=FIXTURE)
    assert result.language_id == "cpp"
    assert result.module_name == "pricing.cpp"
    module = result.symbols[0]
    assert module.kind is SymbolKind.MODULE
    assert module.qualified_name == "@file:pricing.cpp"
    assert all(unit.kind is not SymbolKind.MODULE for unit in result.code_units)
    assert all(unit.kind is not SymbolKind.NAMESPACE for unit in result.code_units)


def test_overloads_coexist() -> None:
    symbols = _symbols(FIXTURE / "pricing.cpp")
    assert "pricing::calculate(int)" in symbols
    assert "pricing::calculate(double)" in symbols
    assert symbols["pricing::calculate(int)"].kind is SymbolKind.FUNCTION


def test_reopened_namespace_containers_are_distinct() -> None:
    symbols = _symbols(FIXTURE / "pricing.cpp")
    namespaces = [q for q, s in symbols.items() if s.kind is SymbolKind.NAMESPACE]
    assert "@namespace:pricing.cpp:1:pricing" in namespaces
    assert "@namespace:pricing.cpp:2:pricing" in namespaces
    assert "pricing::tally()" in symbols


def test_nested_namespace_function() -> None:
    symbols = _symbols(FIXTURE / "pricing.cpp")
    assert "a::b::nested_foo()" in symbols


def test_struct_maps_to_class() -> None:
    symbols = _symbols(FIXTURE / "widget.cpp")
    assert symbols["ui::Point"].kind is SymbolKind.CLASS


def test_member_cv_ref_overload_identity() -> None:
    symbols = _symbols(FIXTURE / "widget.cpp")
    assert "ui::Widget::value()" in symbols
    assert "ui::Widget::value() const" in symbols
    assert "ui::Widget::value() &" in symbols
    assert "ui::Widget::value() &&" in symbols


def test_ctor_dtor_operator() -> None:
    symbols = _symbols(FIXTURE / "widget.cpp")
    assert symbols["ui::Widget::Widget(int)"].kind is SymbolKind.METHOD
    assert symbols["ui::Widget::~Widget()"].kind is SymbolKind.METHOD
    assert symbols["ui::Widget::~Widget()"].name == "~Widget"
    assert "ui::Widget::operator==(const Widget&) const" in symbols
    assert "ui::Widget::operator[](int) const" in symbols


def test_same_file_out_of_class_method() -> None:
    symbols = _symbols(FIXTURE / "widget.cpp")
    touch = symbols["ui::Widget::touch()"]
    assert touch.kind is SymbolKind.METHOD
    assert touch.parent_qualified_name == "ui::Widget"


def test_cross_file_out_of_class_is_function() -> None:
    symbols = _symbols(FIXTURE / "process.cpp")
    run = symbols["Service::run()"]
    assert run.kind is SymbolKind.FUNCTION


def test_prototypes_skipped_definitions_kept() -> None:
    header = _symbols(FIXTURE / "process.hpp")
    source = _symbols(FIXTURE / "process.cpp")
    assert "process(int)" not in header
    assert "process(int)" in source
    assert source["process(int)"].kind is SymbolKind.FUNCTION
    assert "Service::run()" not in header
    assert "Service::inline_run()" in header
    assert header["Service::inline_run()"].kind is SymbolKind.METHOD


def test_static_and_anonymous_file_local_across_files() -> None:
    a = _symbols(FIXTURE / "local_a.cpp")
    b = _symbols(FIXTURE / "local_b.cpp")
    assert "@filelocal:local_a.cpp::<anonymous>::helper()" in a
    assert "@filelocal:local_b.cpp::<anonymous>::helper()" in b
    assert "@filelocal:local_a.cpp::helper_static()" in a
    assert "@filelocal:local_b.cpp::helper_static()" in b


def test_same_file_static_and_anonymous_helpers_coexist() -> None:
    result = CppAdapter().analyze_source(
        "static void helper() {}\n\nnamespace {\nvoid helper() {}\n}\n",
        module_name="same.cpp",
    )
    qnames = {symbol.qualified_name for symbol in result.symbols}
    assert "@filelocal:same.cpp::helper()" in qnames
    assert "@filelocal:same.cpp::<anonymous>::helper()" in qnames
    units = {unit.symbol_qualified_name for unit in result.code_units}
    assert "@filelocal:same.cpp::helper()" in units
    assert "@filelocal:same.cpp::<anonymous>::helper()" in units


def test_reopened_anonymous_namespace_shares_semantic_scope() -> None:
    result = CppAdapter().analyze_source(
        "namespace {\nvoid first() {}\n}\n\nnamespace {\nvoid second() {}\n}\n",
        module_name="reopen.cpp",
    )
    qnames = {symbol.qualified_name for symbol in result.symbols}
    assert "@filelocal:reopen.cpp::<anonymous>::first()" in qnames
    assert "@filelocal:reopen.cpp::<anonymous>::second()" in qnames
    namespaces = [
        symbol.qualified_name for symbol in result.symbols if symbol.kind is SymbolKind.NAMESPACE
    ]
    assert "@namespace:reopen.cpp:1:<anonymous>" in namespaces
    assert "@namespace:reopen.cpp:2:<anonymous>" in namespaces
    assert not any("<anonymous:1>" in name or "<anonymous:2>" in name for name in qnames)


def test_nested_named_plus_anonymous_scope() -> None:
    result = CppAdapter().analyze_source(
        "namespace pricing {\nnamespace {\nvoid helper() {}\n}\n}\n",
        module_name="nested.cpp",
    )
    qnames = {symbol.qualified_name for symbol in result.symbols}
    assert "@filelocal:nested.cpp::pricing::<anonymous>::helper()" in qnames


def test_anonymous_overloads_coexist() -> None:
    result = CppAdapter().analyze_source(
        "namespace {\nvoid helper(int) {}\nvoid helper(double) {}\n}\n",
        module_name="over.cpp",
    )
    qnames = {symbol.qualified_name for symbol in result.symbols}
    assert "@filelocal:over.cpp::<anonymous>::helper(int)" in qnames
    assert "@filelocal:over.cpp::<anonymous>::helper(double)" in qnames


def test_anonymous_ordinary_qname_stable_under_earlier_namespace_insert() -> None:
    before = CppAdapter().analyze_source(
        "namespace {\nvoid helper() {}\n}\n",
        module_name="stable.cpp",
    )
    after_source = (
        "namespace pricing {\nint unrelated() { return 1; }\n}\n\n"
        "namespace {\nvoid helper() {}\n}\n"
    )
    after = CppAdapter().analyze_source(after_source, module_name="stable.cpp")
    helper_before = next(
        symbol.qualified_name for symbol in before.symbols if symbol.name == "helper"
    )
    helper_after = next(
        symbol.qualified_name for symbol in after.symbols if symbol.name == "helper"
    )
    assert helper_before == helper_after == "@filelocal:stable.cpp::<anonymous>::helper()"


def test_templates_enum_union_lambda_deferred() -> None:
    result = CppAdapter().analyze_file(FIXTURE / "deferred.cpp", repository_root=FIXTURE)
    qnames = {symbol.qualified_name for symbol in result.symbols}
    assert not any("max_value" in name for name in qnames)
    assert not any("Box" in name for name in qnames)
    assert not any("Color" in name for name in qnames)
    assert not any("Packet" in name for name in qnames)
    assert result.has_syntax_errors is True


def test_unicode_span_round_trip() -> None:
    path = FIXTURE / "unicode.cpp"
    result = CppAdapter().analyze_file(path, repository_root=FIXTURE)
    greet = next(unit for unit in result.code_units if "greet" in unit.symbol_qualified_name)
    raw = path.read_bytes()
    assert raw[greet.span.start_byte : greet.span.end_byte].decode("utf-8") == greet.source_text
    assert "こんにちは" in greet.source_text


def test_c_extension_not_supported() -> None:
    adapter = CppAdapter()
    assert adapter.supports_file(Path("x.cpp"))
    assert adapter.supports_file(Path("x.hpp"))
    assert not adapter.supports_file(Path("x.c"))
    assert not adapter.supports_file(Path("x.py"))


def test_class_forward_declaration_skipped() -> None:
    result = CppAdapter().analyze_source(
        "class Foo;\nstruct Bar;\nclass Ready { int x; };\nclass Empty {};\n",
        module_name="fwd.hpp",
    )
    qnames = {symbol.qualified_name for symbol in result.symbols}
    assert "Foo" not in qnames
    assert "Bar" not in qnames
    assert "Ready" in qnames
    assert "Empty" in qnames
    units = {unit.symbol_qualified_name for unit in result.code_units}
    assert units == {"Ready", "Empty"}


def test_forward_declaration_does_not_collide_with_definition() -> None:
    result = CppAdapter().analyze_source(
        "class Service;\nclass Service { void run() {} };\n",
        module_name="svc.cpp",
    )
    classes = [symbol for symbol in result.symbols if symbol.kind is SymbolKind.CLASS]
    assert len(classes) == 1
    assert classes[0].qualified_name == "Service"
    assert "Service::run()" in {symbol.qualified_name for symbol in result.symbols}
