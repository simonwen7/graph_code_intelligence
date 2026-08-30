"""Tests for Python static relationship extraction."""

from pathlib import Path

from codeintel.languages.python import PythonAdapter, PythonRelationExtractor
from codeintel.models import Relation, RelationKind, ResolutionStatus
from codeintel.repository import RepositoryAnalysis, analyze_repository

GRAPH_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "python_graph"


def _analyze() -> RepositoryAnalysis:
    return analyze_repository(GRAPH_ROOT, PythonAdapter(), PythonRelationExtractor())


def _rels(
    analysis: RepositoryAnalysis,
    kind: RelationKind,
    *,
    source: str | None = None,
    target: str | None = None,
    text: str | None = None,
) -> list[Relation]:
    selected = [rel for rel in analysis.relations if rel.kind is kind]
    if source is not None:
        selected = [rel for rel in selected if rel.source_qualified_name == source]
    if target is not None:
        selected = [rel for rel in selected if rel.target_qualified_name == target]
    if text is not None:
        selected = [rel for rel in selected if rel.target_text == text]
    return selected


def _span_key(rel: Relation) -> tuple[Path, int | None, int | None]:
    span = rel.span
    if span is None:
        return (rel.path, None, None)
    return (rel.path, span.start_byte, span.end_byte)


def test_contains_structural_edges() -> None:
    analysis = _analyze()
    assert _rels(analysis, RelationKind.CONTAINS, source="helpers", target="helpers.helper")
    assert _rels(analysis, RelationKind.CONTAINS, source="helpers", target="helpers.Base")
    assert _rels(
        analysis,
        RelationKind.CONTAINS,
        source="service.Service",
        target="service.Service.run",
    )
    assert _rels(
        analysis,
        RelationKind.CONTAINS,
        source="service.Service.validate",
        target="service.Service.validate.inner",
    )


def test_imports_local_alias_relative_and_external() -> None:
    analysis = _analyze()
    module_import = _rels(analysis, RelationKind.IMPORTS, source="consumer", target="helpers")
    assert any(
        rel.target_text == "helpers" and rel.resolution is ResolutionStatus.RESOLVED
        for rel in module_import
    )

    alias_import = _rels(analysis, RelationKind.IMPORTS, source="consumer", text="helpers")
    assert any(rel.target_qualified_name == "helpers" for rel in alias_import)

    from_helper = _rels(analysis, RelationKind.IMPORTS, source="consumer", target="helpers.helper")
    assert from_helper
    assert all(rel.resolution is ResolutionStatus.RESOLVED for rel in from_helper)

    relative = _rels(
        analysis,
        RelationKind.IMPORTS,
        source="package.child",
        target="package.base.PackageBase",
    )
    assert relative
    relative_module = _rels(
        analysis, RelationKind.IMPORTS, source="package.child", target="package.base"
    )
    assert relative_module
    up_import = _rels(
        analysis, RelationKind.IMPORTS, source="package.child", target="helpers.helper"
    )
    assert up_import

    external = _rels(analysis, RelationKind.IMPORTS, source="consumer", text="pathlib")
    assert external
    assert external[0].resolution is ResolutionStatus.UNRESOLVED
    assert external[0].target_qualified_name is None

    star = [
        rel
        for rel in analysis.relations
        if rel.kind is RelationKind.IMPORTS and rel.target_text.endswith(".*")
    ]
    assert star
    assert all(rel.resolution is ResolutionStatus.UNRESOLVED for rel in star)


def test_calls_resolution_and_shadowing() -> None:
    analysis = _analyze()
    same_module = _rels(
        analysis, RelationKind.CALLS, source="helpers.caller", target="helpers.helper"
    )
    assert same_module[0].resolution is ResolutionStatus.RESOLVED

    imported = _rels(
        analysis, RelationKind.CALLS, source="consumer.use_calls", target="helpers.helper"
    )
    assert {rel.target_text for rel in imported} >= {
        "helper",
        "aliased",
        "hmod.helper",
        "helpers.helper",
    }
    assert all(rel.resolution is ResolutionStatus.RESOLVED for rel in imported)

    self_call = _rels(
        analysis,
        RelationKind.CALLS,
        source="service.Service.run",
        target="service.Service.validate",
    )
    assert self_call[0].resolution is ResolutionStatus.PROBABLE
    assert self_call[0].target_text == "self.validate"

    cls_call = _rels(
        analysis,
        RelationKind.CALLS,
        source="service.Service.build",
        target="service.Service.run",
    )
    assert cls_call[0].resolution is ResolutionStatus.PROBABLE
    assert cls_call[0].target_text == "cls.run"

    nested = _rels(
        analysis,
        RelationKind.CALLS,
        source="service.Service.validate",
        target="service.Service.validate.inner",
    )
    assert nested[0].resolution is ResolutionStatus.RESOLVED

    dynamic = _rels(
        analysis, RelationKind.CALLS, source="consumer.call_dynamic", text="obj.dynamic"
    )
    assert dynamic[0].resolution is ResolutionStatus.UNRESOLVED
    assert dynamic[0].target_qualified_name is None

    shadowed = _rels(analysis, RelationKind.CALLS, source="consumer.shadow")
    assert all(rel.resolution is ResolutionStatus.UNRESOLVED for rel in shadowed)
    local = _rels(analysis, RelationKind.CALLS, source="consumer.local_shadow")
    assert all(rel.resolution is ResolutionStatus.UNRESOLVED for rel in local)


def test_references_are_conservative() -> None:
    analysis = _analyze()
    refs = _rels(
        analysis,
        RelationKind.REFERENCES,
        source="consumer.use_reference",
        target="helpers.unused_ref",
    )
    assert refs
    assert refs[0].resolution is ResolutionStatus.RESOLVED
    assert refs[0].path.name == "consumer.py"
    assert refs[0].span is not None

    call_spans = {_span_key(rel) for rel in analysis.relations if rel.kind is RelationKind.CALLS}
    ref_spans = {
        _span_key(rel) for rel in analysis.relations if rel.kind is RelationKind.REFERENCES
    }
    inherit_spans = {
        _span_key(rel) for rel in analysis.relations if rel.kind is RelationKind.INHERITS
    }
    import_spans = {
        _span_key(rel) for rel in analysis.relations if rel.kind is RelationKind.IMPORTS
    }
    assert call_spans.isdisjoint(ref_spans)
    assert inherit_spans.isdisjoint(ref_spans)
    assert import_spans.isdisjoint(ref_spans)

    noisy = [
        rel
        for rel in analysis.relations
        if rel.kind is RelationKind.REFERENCES and rel.target_qualified_name is None
    ]
    assert noisy == []
    local_refs = [
        rel
        for rel in analysis.relations
        if rel.kind is RelationKind.REFERENCES
        and rel.source_qualified_name == "consumer.local_shadow"
    ]
    assert local_refs == []


def test_inherits_same_module_imported_multiple_and_unknown() -> None:
    analysis = _analyze()
    gamma_alpha = _rels(
        analysis, RelationKind.INHERITS, source="models.Gamma", target="models.Alpha"
    )
    gamma_beta = _rels(analysis, RelationKind.INHERITS, source="models.Gamma", target="models.Beta")
    assert gamma_alpha[0].resolution is ResolutionStatus.RESOLVED
    assert gamma_beta[0].resolution is ResolutionStatus.RESOLVED

    service = _rels(
        analysis, RelationKind.INHERITS, source="service.Service", target="helpers.Base"
    )
    assert service[0].resolution is ResolutionStatus.RESOLVED

    child = _rels(
        analysis,
        RelationKind.INHERITS,
        source="package.child.Child",
        target="package.base.PackageBase",
    )
    assert child[0].resolution is ResolutionStatus.RESOLVED

    unknown = _rels(analysis, RelationKind.INHERITS, source="models.UnknownBase", text="Missing")
    assert unknown[0].resolution is ResolutionStatus.UNRESOLVED
    assert unknown[0].target_qualified_name is None


def _analyze_dir(root: Path) -> RepositoryAnalysis:
    return analyze_repository(root, PythonAdapter(), PythonRelationExtractor())


def test_dotted_import_binds_package_not_submodule(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (pkg / "base.py").write_text("def helper():\n    return 2\n", encoding="utf-8")
    (tmp_path / "user.py").write_text(
        "import pkg.base\n\ndef use() -> None:\n    pkg.helper()\n",
        encoding="utf-8",
    )
    analysis = _analyze_dir(tmp_path)
    imports = _rels(analysis, RelationKind.IMPORTS, source="user", target="pkg.base")
    assert imports
    assert imports[0].resolution is ResolutionStatus.RESOLVED
    calls = _rels(analysis, RelationKind.CALLS, source="user.use")
    assert len(calls) == 1
    assert calls[0].target_qualified_name == "pkg.helper"
    assert calls[0].resolution is ResolutionStatus.RESOLVED


def test_nested_package_relative_imports(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    sub = pkg / "sub"
    sub.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (sub / "__init__.py").write_text("", encoding="utf-8")
    (sub / "helper.py").write_text("def function() -> int:\n    return 1\n", encoding="utf-8")
    (sub / "mod.py").write_text(
        "from . import helper\n"
        "from .helper import function\n"
        "\n"
        "def use() -> None:\n"
        "    function()\n",
        encoding="utf-8",
    )
    analysis = _analyze_dir(tmp_path)
    dotted = _rels(analysis, RelationKind.IMPORTS, source="pkg.sub.mod", target="pkg.sub.helper")
    assert dotted
    assert any(rel.target_text == ".helper" for rel in dotted)
    named = _rels(
        analysis,
        RelationKind.IMPORTS,
        source="pkg.sub.mod",
        target="pkg.sub.helper.function",
    )
    assert named
    assert named[0].target_text == ".helper.function"
    calls = _rels(
        analysis, RelationKind.CALLS, source="pkg.sub.mod.use", target="pkg.sub.helper.function"
    )
    assert calls[0].resolution is ResolutionStatus.RESOLVED


def test_import_inside_function_uses_module_source(tmp_path: Path) -> None:
    (tmp_path / "helpers.py").write_text("def helper() -> int:\n    return 1\n", encoding="utf-8")
    (tmp_path / "use.py").write_text(
        "def use() -> None:\n    from helpers import helper\n    helper()\n",
        encoding="utf-8",
    )
    analysis = _analyze_dir(tmp_path)
    imports = _rels(analysis, RelationKind.IMPORTS, source="use", target="helpers.helper")
    assert imports
    assert imports[0].resolution is ResolutionStatus.RESOLVED
    assert not _rels(analysis, RelationKind.IMPORTS, source="use.use")
    calls = _rels(analysis, RelationKind.CALLS, source="use.use", target="helpers.helper")
    assert calls[0].resolution is ResolutionStatus.RESOLVED


def test_assignment_after_use_is_unresolved(tmp_path: Path) -> None:
    (tmp_path / "helpers.py").write_text("def helper() -> int:\n    return 1\n", encoding="utf-8")
    (tmp_path / "use.py").write_text(
        "from helpers import helper\n\ndef use() -> None:\n    helper()\n    helper = 1\n",
        encoding="utf-8",
    )
    analysis = _analyze_dir(tmp_path)
    calls = _rels(analysis, RelationKind.CALLS, source="use.use")
    assert calls
    assert all(rel.resolution is ResolutionStatus.UNRESOLVED for rel in calls)
    assert all(rel.target_qualified_name is None for rel in calls)


def test_utf8_call_span_uses_byte_offsets(tmp_path: Path) -> None:
    (tmp_path / "helpers.py").write_text("def helper() -> int:\n    return 1\n", encoding="utf-8")
    source = 'from helpers import helper\nMSG = "你好 🚀"\n\ndef use() -> None:\n    helper()\n'
    (tmp_path / "use.py").write_text(source, encoding="utf-8")
    analysis = _analyze_dir(tmp_path)
    calls = _rels(analysis, RelationKind.CALLS, source="use.use", target="helpers.helper")
    assert calls
    span = calls[0].span
    assert span is not None
    encoded = source.encode("utf-8")
    sliced = encoded[span.start_byte : span.end_byte].decode("utf-8")
    assert sliced == "helper"
    assert sliced == calls[0].target_text
    assert span.start_line == 5


def test_self_inherited_method_stays_unresolved(tmp_path: Path) -> None:
    (tmp_path / "base.py").write_text(
        "class Base:\n    def shared(self) -> None:\n        pass\n",
        encoding="utf-8",
    )
    (tmp_path / "child.py").write_text(
        "from base import Base\n"
        "\n"
        "class Child(Base):\n"
        "    def run(self) -> None:\n"
        "        self.shared()\n",
        encoding="utf-8",
    )
    analysis = _analyze_dir(tmp_path)
    calls = _rels(analysis, RelationKind.CALLS, source="child.Child.run")
    assert calls
    assert calls[0].target_text == "self.shared"
    assert calls[0].resolution is ResolutionStatus.UNRESOLVED
    assert calls[0].target_qualified_name is None


def test_known_class_attribute_call_is_probable(tmp_path: Path) -> None:
    (tmp_path / "svc.py").write_text(
        "class Service:\n"
        "    def validate(self) -> None:\n"
        "        pass\n"
        "    def run(self) -> None:\n"
        "        Service.validate(self)\n",
        encoding="utf-8",
    )
    analysis = _analyze_dir(tmp_path)
    calls = _rels(
        analysis,
        RelationKind.CALLS,
        source="svc.Service.run",
        target="svc.Service.validate",
    )
    assert calls[0].resolution is ResolutionStatus.PROBABLE
    assert calls[0].target_text == "Service.validate"


def test_value_reference_and_call_are_not_duplicates(tmp_path: Path) -> None:
    (tmp_path / "helpers.py").write_text("def helper() -> int:\n    return 1\n", encoding="utf-8")
    (tmp_path / "use.py").write_text(
        "from helpers import helper\n\ndef use() -> None:\n    items = [helper]\n    helper()\n",
        encoding="utf-8",
    )
    analysis = _analyze_dir(tmp_path)
    refs = _rels(analysis, RelationKind.REFERENCES, source="use.use", target="helpers.helper")
    calls = _rels(analysis, RelationKind.CALLS, source="use.use", target="helpers.helper")
    assert refs
    assert calls
    assert refs[0].span != calls[0].span


def test_unrelated_modules_do_not_share_simple_names(tmp_path: Path) -> None:
    (tmp_path / "module_a.py").write_text("def helper() -> int:\n    return 1\n", encoding="utf-8")
    (tmp_path / "module_b.py").write_text("def helper() -> int:\n    return 2\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("def use() -> None:\n    helper()\n", encoding="utf-8")
    analysis = _analyze_dir(tmp_path)
    calls = _rels(analysis, RelationKind.CALLS, source="other.use")
    assert calls
    assert all(rel.resolution is ResolutionStatus.UNRESOLVED for rel in calls)
    assert all(rel.target_qualified_name is None for rel in calls)


def test_dynamic_base_emits_unresolved_inherits(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text(
        "def get_base() -> type:\n    return object\n\nclass Child(get_base()):\n    pass\n",
        encoding="utf-8",
    )
    analysis = _analyze_dir(tmp_path)
    inherits = _rels(analysis, RelationKind.INHERITS, source="mod.Child")
    assert len(inherits) == 1
    assert inherits[0].resolution is ResolutionStatus.UNRESOLVED
    assert inherits[0].target_qualified_name is None
    assert inherits[0].target_text == "get_base()"


def test_from_package_import_module_and_alias(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod.py").write_text("def helper() -> int:\n    return 1\n", encoding="utf-8")
    (tmp_path / "user.py").write_text(
        "from pkg import mod\nimport pkg.mod as alias\n\n"
        "def use() -> None:\n    mod.helper()\n    alias.helper()\n",
        encoding="utf-8",
    )
    analysis = _analyze_dir(tmp_path)
    imported = _rels(analysis, RelationKind.IMPORTS, source="user", target="pkg.mod")
    assert len(imported) == 2
    calls = _rels(analysis, RelationKind.CALLS, source="user.use", target="pkg.mod.helper")
    assert {rel.target_text for rel in calls} == {"mod.helper", "alias.helper"}
    assert all(rel.resolution is ResolutionStatus.RESOLVED for rel in calls)
