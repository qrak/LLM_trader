"""The loose-helper gate: a class-only helper must live in its class, and no def sits between classes."""

from __future__ import annotations

from pathlib import Path

from scripts.check_loose_helpers import LooseHelperChecker

CLASS_ONLY = '''
class Owner:
    def use(self):
        return _helper()


def _helper():
    return 1


class Other:
    pass
'''

NUMBA_KERNEL = '''
import numba


class Owner:
    def use(self):
        return _kernel(1)


@numba.njit(cache=True)
def _kernel(value):
    return value
'''

SHARED_BY_TWO_CLASSES = '''
class First:
    def use(self):
        return _shared()


class Second:
    def use(self):
        return _shared()


def _shared():
    return 1
'''

CALLED_BY_A_FUNCTION = '''
class Owner:
    pass


def _helper():
    return 1


def public_api():
    return _helper()
'''


def _scan(root: Path, name: str, source: str) -> list[str]:
    package = root / "src" / "pkg"
    package.mkdir(parents=True, exist_ok=True)
    (package / f"{name}.py").write_text(source, encoding="utf-8")
    return LooseHelperChecker(root).check()


def test_the_repository_has_no_loose_helpers():
    assert LooseHelperChecker().check() == []


def test_a_class_only_helper_is_reported(tmp_path):
    violations = _scan(tmp_path, "class_only", CLASS_ONLY)

    assert any("_helper -> Owner" in line for line in violations)
    assert any("_helper sits between class definitions" in line for line in violations)


def test_a_numba_kernel_is_exempt(tmp_path):
    assert "numba" in NUMBA_KERNEL

    assert _scan(tmp_path, "numba_kernel", NUMBA_KERNEL) == []


def test_a_helper_two_classes_share_is_not_a_violation(tmp_path):
    assert _scan(tmp_path, "shared", SHARED_BY_TWO_CLASSES) == []


def test_a_helper_of_a_module_function_is_not_a_violation(tmp_path):
    assert _scan(tmp_path, "function_user", CALLED_BY_A_FUNCTION) == []


def test_a_helper_imported_elsewhere_stays_free(tmp_path):
    violations = _scan(tmp_path, "imported", CLASS_ONLY.replace("class Other:\n    pass\n", ""))
    (tmp_path / "src" / "pkg" / "user.py").write_text(
        "from src.pkg.imported import _helper\n", encoding="utf-8"
    )

    assert LooseHelperChecker(tmp_path).check() == []
    assert any("_helper" in line for line in violations), "without the import the helper is loose"
