"""Tests for language-neutral source discovery."""

from pathlib import Path

from codeintel.discovery import discover_source_files
from codeintel.languages.python import PythonAdapter

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "python_repo"


def test_discover_source_files_recursive_and_sorted() -> None:
    """Discovery should recursively find Python files in sorted order."""
    adapter = PythonAdapter()

    discovered = discover_source_files(FIXTURE_ROOT, adapter)

    relative = tuple(path.relative_to(FIXTURE_ROOT).as_posix() for path in discovered)
    assert relative == tuple(sorted(relative))
    assert "simple.py" in relative
    assert "package/service.py" in relative
    assert "package/__init__.py" in relative


def test_discover_source_files_skips_ignored_directories(tmp_path: Path) -> None:
    """Discovery should skip exact ignored directory names."""
    adapter = PythonAdapter()
    keep = tmp_path / "keep.py"
    keep.write_text("def ok() -> None:\n    pass\n", encoding="utf-8")
    for ignored in (".git", ".venv", "__pycache__", "build", "dist"):
        hidden_dir = tmp_path / ignored
        hidden_dir.mkdir()
        (hidden_dir / "hidden.py").write_text("def hidden() -> None:\n    pass\n", encoding="utf-8")

    discovered = discover_source_files(tmp_path, adapter)

    assert discovered == (keep,)


def test_discover_single_supported_file() -> None:
    """A supported file root should return that file alone."""
    adapter = PythonAdapter()
    target = FIXTURE_ROOT / "simple.py"

    assert discover_source_files(target, adapter) == (target,)


def test_discover_unsupported_file_returns_empty(tmp_path: Path) -> None:
    """An unsupported file should yield no discovered sources."""
    adapter = PythonAdapter()
    other = tmp_path / "notes.txt"
    other.write_text("not python\n", encoding="utf-8")

    assert discover_source_files(other, adapter) == ()
