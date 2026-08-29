"""Tests for the base language-adapter interface."""

from pathlib import Path

from codeintel.languages.base import LanguageAdapter
from codeintel.models import AnalysisResult


class ExampleLanguageAdapter(LanguageAdapter):
    """Minimal concrete adapter used only by this unit test."""

    @property
    def language_id(self) -> str:
        return "example"

    @property
    def file_extensions(self) -> frozenset[str]:
        return frozenset({".example"})

    def analyze_file(
        self,
        path: Path,
        *,
        repository_root: Path | None = None,
    ) -> AnalysisResult:
        del repository_root
        return AnalysisResult(
            path=path,
            language_id=self.language_id,
            module_name=path.stem,
            symbols=(),
            code_units=(),
            has_syntax_errors=False,
        )


def test_supports_known_file_extension() -> None:
    """An adapter should recognize one of its configured source extensions."""
    adapter = ExampleLanguageAdapter()

    assert adapter.supports_file(Path("sample.example"))


def test_rejects_unknown_file_extension() -> None:
    """An adapter should reject unsupported source extensions."""
    adapter = ExampleLanguageAdapter()

    assert not adapter.supports_file(Path("sample.py"))
