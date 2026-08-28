"""Base interface for language-specific source-code adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class LanguageAdapter(ABC):
    """Common interface implemented by each supported programming language."""

    @property
    @abstractmethod
    def language_id(self) -> str:
        """Return the canonical identifier for this programming language."""

    @property
    @abstractmethod
    def file_extensions(self) -> frozenset[str]:
        """Return the source-file extensions supported by this adapter."""

    def supports_file(self, path: Path) -> bool:
        """Return whether this adapter supports the supplied source file."""
        return path.suffix.lower() in self.file_extensions
