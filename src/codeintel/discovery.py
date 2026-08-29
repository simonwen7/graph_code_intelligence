"""Language-neutral source-file discovery."""

from __future__ import annotations

from pathlib import Path

from codeintel.languages.base import LanguageAdapter

_SKIP_DIR_NAMES = frozenset({".git", ".venv", "__pycache__", "build", "dist"})


def discover_source_files(root: Path, adapter: LanguageAdapter) -> tuple[Path, ...]:
    """Return supported source files under ``root`` in deterministic order.

    If ``root`` is a supported file, return that single path.
    If ``root`` is an unsupported file, return an empty tuple.
    If ``root`` is a directory, recursively collect supported files while
    skipping well-known non-source directories.
    """
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    if root.is_file():
        if adapter.supports_file(root):
            return (root,)
        return ()

    if not root.is_dir():
        raise NotADirectoryError(f"Path is neither a file nor a directory: {root}")

    discovered: list[Path] = []
    for current_root, dir_names, file_names in root.walk():
        dir_names[:] = sorted(name for name in dir_names if name not in _SKIP_DIR_NAMES)
        for file_name in sorted(file_names):
            candidate = current_root / file_name
            if adapter.supports_file(candidate):
                discovered.append(candidate)

    return tuple(sorted(discovered))
