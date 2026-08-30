"""Small deterministic language tool selection helpers."""

from __future__ import annotations

from enum import StrEnum

from codeintel.languages.base import LanguageAdapter
from codeintel.languages.cpp import CppAdapter, CppRelationExtractor
from codeintel.languages.python import PythonAdapter, PythonRelationExtractor
from codeintel.repository import RelationExtractor


class SourceLanguage(StrEnum):
    """Supported single-language index analysis modes."""

    PYTHON = "python"
    CPP = "cpp"


def create_language_tools(language: SourceLanguage) -> tuple[LanguageAdapter, RelationExtractor]:
    """Return the adapter/extractor pair for ``language``."""
    if language is SourceLanguage.PYTHON:
        return PythonAdapter(), PythonRelationExtractor()
    if language is SourceLanguage.CPP:
        return CppAdapter(), CppRelationExtractor()
    raise ValueError(f"Unsupported language: {language}")
