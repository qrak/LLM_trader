"""Unit tests for the launcher's version-aware requirements check."""
from pathlib import Path

from scripts.check_requirements import find_missing


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "requirements.txt"
    path.write_text(content, encoding="utf-8")
    return path


def test_satisfied_requirements_produce_no_output(tmp_path: Path) -> None:
    path = _write(tmp_path, "pytest>=1\n# comment\n\npackaging\n")
    assert find_missing(path) == []


def test_missing_package_is_reported(tmp_path: Path) -> None:
    path = _write(tmp_path, "definitely-not-installed-xyz==1.0\n")
    assert find_missing(path) == ["definitely-not-installed-xyz==1.0"]


def test_unsatisfied_version_floor_is_reported(tmp_path: Path) -> None:
    path = _write(tmp_path, "pytest>=9999\n")
    assert find_missing(path) == ["pytest>=9999"]


def test_unsatisfied_range_upper_bound_is_reported(tmp_path: Path) -> None:
    path = _write(tmp_path, "pytest<1\n")
    assert find_missing(path) == ["pytest<1"]
