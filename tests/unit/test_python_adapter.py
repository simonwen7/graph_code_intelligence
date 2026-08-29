"""Tests for PythonAdapter semantic extraction."""

from pathlib import Path

from codeintel.languages.python import PythonAdapter
from codeintel.models import SymbolKind

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "python_repo"


def test_python_adapter_metadata() -> None:
    """PythonAdapter should identify Python files by extension."""
    adapter = PythonAdapter()

    assert adapter.language_id == "python"
    assert adapter.file_extensions == frozenset({".py"})
    assert adapter.supports_file(Path("module.py"))
    assert not adapter.supports_file(Path("module.txt"))


def test_module_symbol_extraction() -> None:
    """Analyze should always emit a MODULE symbol for the derived module name."""
    adapter = PythonAdapter()
    path = FIXTURE_ROOT / "simple.py"

    result = adapter.analyze_file(path, repository_root=FIXTURE_ROOT)

    module = result.symbols[0]
    assert module.kind == SymbolKind.MODULE
    assert module.qualified_name == "simple"
    assert module.signature is None
    assert module.parent_qualified_name is None
    assert all(unit.kind != SymbolKind.MODULE for unit in result.code_units)


def test_simple_function_symbol_and_code_unit() -> None:
    """A top-level function should produce matching Symbol and CodeUnit data."""
    adapter = PythonAdapter()
    path = FIXTURE_ROOT / "simple.py"
    source = path.read_text(encoding="utf-8")

    result = adapter.analyze_file(path, repository_root=FIXTURE_ROOT)

    function = next(symbol for symbol in result.symbols if symbol.kind == SymbolKind.FUNCTION)
    assert function.name == "greet"
    assert function.qualified_name == "simple.greet"
    assert function.signature == "def greet(name: str) -> str"
    assert function.span.start_line == 1
    assert function.span.end_line >= 1

    unit = next(unit for unit in result.code_units if unit.symbol_qualified_name == "simple.greet")
    encoded = source.encode("utf-8")
    assert unit.kind == SymbolKind.FUNCTION
    assert unit.source_text == encoded[unit.span.start_byte : unit.span.end_byte].decode("utf-8")
    assert unit.source_text.startswith("def greet(name: str) -> str:")
    assert 'return f"hello {name}"' in unit.source_text


def test_class_and_method_extraction() -> None:
    """Classes and direct methods should use CLASS and METHOD kinds."""
    adapter = PythonAdapter()
    path = FIXTURE_ROOT / "nested.py"

    result = adapter.analyze_file(path, repository_root=FIXTURE_ROOT)
    qualified = {symbol.qualified_name: symbol for symbol in result.symbols}

    assert qualified["nested.Service"].kind == SymbolKind.CLASS
    assert qualified["nested.Service"].signature == "class Service"
    assert qualified["nested.Service.run"].kind == SymbolKind.METHOD
    assert qualified["nested.Service.run"].signature == "def run(self) -> None"

    unit_names = {unit.symbol_qualified_name for unit in result.code_units}
    assert "nested.Service" in unit_names
    assert "nested.Service.run" in unit_names


def test_nested_function_classified_as_function() -> None:
    """A function nested inside a method should remain FUNCTION, not METHOD."""
    adapter = PythonAdapter()
    path = FIXTURE_ROOT / "nested.py"

    result = adapter.analyze_file(path, repository_root=FIXTURE_ROOT)
    helper = next(symbol for symbol in result.symbols if symbol.qualified_name.endswith(".helper"))

    assert helper.kind == SymbolKind.FUNCTION
    assert helper.qualified_name == "nested.Service.run.helper"
    assert helper.parent_qualified_name == "nested.Service.run"


def test_decorated_definitions() -> None:
    """Decorators should be included in CodeUnits but excluded from signatures."""
    adapter = PythonAdapter()
    path = FIXTURE_ROOT / "decorated.py"

    result = adapter.analyze_file(path, repository_root=FIXTURE_ROOT)
    symbols = {symbol.qualified_name: symbol for symbol in result.symbols}
    units = {unit.symbol_qualified_name: unit for unit in result.code_units}

    assert "decorated.decorated_function" in symbols
    assert symbols["decorated.decorated_function"].signature == (
        "def decorated_function(value: int) -> int"
    )
    assert units["decorated.decorated_function"].source_text.startswith("@decorator")

    assert "decorated.DecoratedClass" in symbols
    assert symbols["decorated.DecoratedClass"].signature == "class DecoratedClass"
    assert units["decorated.DecoratedClass"].source_text.startswith("@decorator")

    assert symbols["decorated.DecoratedClass.build"].kind == SymbolKind.METHOD
    assert symbols["decorated.DecoratedClass.build"].signature == (
        'def build(cls) -> "DecoratedClass"'
    )
    assert units["decorated.DecoratedClass.build"].source_text.startswith("@classmethod")


def test_async_function_signature() -> None:
    """Async functions should remain FUNCTION and preserve async in signatures."""
    adapter = PythonAdapter()
    path = FIXTURE_ROOT / "async_fn.py"

    result = adapter.analyze_file(path, repository_root=FIXTURE_ROOT)
    function = next(symbol for symbol in result.symbols if symbol.kind == SymbolKind.FUNCTION)

    assert function.qualified_name == "async_fn.fetch_data"
    assert function.signature == "async def fetch_data(url: str) -> str"


def test_package_module_name_derivation() -> None:
    """Package paths should derive dotted names and strip __init__."""
    adapter = PythonAdapter()

    init_result = adapter.analyze_file(
        FIXTURE_ROOT / "package" / "__init__.py",
        repository_root=FIXTURE_ROOT,
    )
    service_result = adapter.analyze_file(
        FIXTURE_ROOT / "package" / "service.py",
        repository_root=FIXTURE_ROOT,
    )

    assert init_result.module_name == "package"
    assert init_result.symbols[0].qualified_name == "package"
    assert service_result.module_name == "package.service"
    assert service_result.symbols[0].qualified_name == "package.service"
    assert any(
        symbol.qualified_name == "package.service.PaymentService"
        for symbol in service_result.symbols
    )


def test_malformed_python_does_not_crash() -> None:
    """Malformed Python should return a flagged AnalysisResult without crashing."""
    adapter = PythonAdapter()
    path = FIXTURE_ROOT / "malformed.py"

    result = adapter.analyze_file(path, repository_root=FIXTURE_ROOT)

    assert result.has_syntax_errors is True
    assert result.symbols[0].kind == SymbolKind.MODULE
    assert result.module_name == "malformed"


def test_utf8_byte_spans_preserve_exact_source() -> None:
    """CodeUnit text must slice UTF-8 bytes, not Unicode code-point indices."""
    adapter = PythonAdapter()
    source = 'MSG = "你好 café 🚀"\n\ndef greet(name: str) -> str:\n    return name\n'

    result = adapter.analyze_source(source, module_name="unicode_mod")
    unit = next(unit for unit in result.code_units if unit.symbol_qualified_name.endswith(".greet"))
    encoded = source.encode("utf-8")

    assert unit.source_text == encoded[unit.span.start_byte : unit.span.end_byte].decode("utf-8")
    assert unit.source_text.startswith("def greet(name: str) -> str:")
    assert 'MSG = "你好 café 🚀"' not in unit.source_text


def test_decorated_definitions_are_not_duplicated() -> None:
    """Decorated wrappers must emit one symbol per definition, not duplicates."""
    adapter = PythonAdapter()
    path = FIXTURE_ROOT / "decorated.py"

    result = adapter.analyze_file(path, repository_root=FIXTURE_ROOT)
    qualified_names = [symbol.qualified_name for symbol in result.symbols]

    assert len(qualified_names) == len(set(qualified_names))
    assert qualified_names.count("decorated.decorated_function") == 1
    assert qualified_names.count("decorated.DecoratedClass") == 1


def test_multiline_signature_extraction() -> None:
    """Signatures should retain multi-line parameter lists without the body."""
    adapter = PythonAdapter()
    source = "def multi(\n    a: int,\n    b: str,\n) -> bool:\n    return True\n"

    result = adapter.analyze_source(source, module_name="sigmod")
    function = next(symbol for symbol in result.symbols if symbol.kind == SymbolKind.FUNCTION)

    assert function.signature == "def multi(\n    a: int,\n    b: str,\n) -> bool"


def test_path_outside_repository_root_raises_clear_error(tmp_path: Path) -> None:
    """analyze_file must not invent a module name for paths outside the root."""
    adapter = PythonAdapter()
    repository_root = tmp_path / "repo"
    repository_root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def orphan() -> None:\n    pass\n", encoding="utf-8")

    try:
        adapter.analyze_file(outside, repository_root=repository_root)
    except ValueError as exc:
        message = str(exc)
        assert "outside repository root" in message
        assert str(outside) in message or outside.name in message
    else:
        raise AssertionError("expected ValueError for path outside repository_root")
