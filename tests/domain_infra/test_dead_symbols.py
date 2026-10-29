from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_dead_symbols import (
    ALLOWLIST,
    KNOWN_USED,
    DeadSymbolChecker,
    DeadSymbolScannerError,
)


def _write(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _check(root: Path) -> list[str]:
    return DeadSymbolChecker(root=root, allowlist={}).check()


def test_the_repository_has_no_dead_symbols() -> None:
    assert DeadSymbolChecker().check() == []


def test_known_live_symbols_are_not_reported() -> None:
    reported = {
        violation.name
        for violation in DeadSymbolChecker().scan()
        if violation.category in {"DEAD", "TEST-ONLY"}
    }

    assert reported.isdisjoint(KNOWN_USED)


def test_an_orphan_function_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "src/pkg/module.py", "def orphan():\n    return 1\n")

    violations = _check(tmp_path)

    assert len(violations) == 1
    assert violations[0].startswith("DEAD src/pkg/module.py:1 function orphan")


def test_a_function_called_only_from_tests_is_test_only(tmp_path: Path) -> None:
    _write(tmp_path, "src/pkg/module.py", "def orphan():\n    return 1\n")
    _write(
        tmp_path,
        "tests/test_module.py",
        "from src.pkg.module import orphan\n\n\ndef test_it():\n    assert orphan() == 1\n",
    )

    violations = _check(tmp_path)

    assert len(violations) == 1
    assert violations[0].startswith("TEST-ONLY src/pkg/module.py:1 function orphan")


def test_a_function_used_by_a_script_is_live(tmp_path: Path) -> None:
    _write(tmp_path, "src/pkg/module.py", "def tool():\n    return 1\n")
    _write(tmp_path, "scripts/tool.py", "from src.pkg.module import tool\n\ntool()\n")

    assert _check(tmp_path) == []


def test_a_package_init_export_alone_does_not_make_a_symbol_live(tmp_path: Path) -> None:
    _write(tmp_path, "src/pkg/module.py", "def exported():\n    return 1\n")
    _write(tmp_path, "src/pkg/__init__.py", "from .module import exported\n\n__all__ = [\"exported\"]\n")

    violations = _check(tmp_path)

    assert len(violations) == 1
    assert "exported" in violations[0]
    assert violations[0].startswith("DEAD ")


def test_a_fastapi_route_is_exempt(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/pkg/routes.py",
        "class Routes:\n"
        "    def __init__(self, router):\n"
        "        self.router = router\n"
        "\n"
        "    @self.router.get(\"/status\")\n"
        "    async def status(self):\n"
        "        return {}\n"
        "\n"
        "ROUTES = Routes\n",
    )

    assert _check(tmp_path) == []


def test_a_pydantic_validator_is_exempt(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/pkg/models.py",
        "from pydantic import BaseModel, field_validator\n"
        "\n"
        "class Model(BaseModel):\n"
        "    value: int\n"
        "\n"
        "    @field_validator(\"value\")\n"
        "    @classmethod\n"
        "    def check(cls, value):\n"
        "        return value\n"
        "\n"
        "MODEL = Model\n",
    )

    assert _check(tmp_path) == []


def test_a_numba_kernel_is_exempt(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/pkg/kernels.py",
        "from numba import njit\n"
        "\n"
        "@njit(cache=True)\n"
        "def kernel(value):\n"
        "    return value\n",
    )

    assert _check(tmp_path) == []


def test_an_htmlparser_override_is_exempt(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/pkg/parser.py",
        "from html.parser import HTMLParser\n"
        "\n"
        "class Parser(HTMLParser):\n"
        "    def handle_starttag(self, tag, attrs):\n"
        "        return None\n"
        "\n"
        "PARSER = Parser\n",
    )

    assert _check(tmp_path) == []


def test_a_base_http_middleware_dispatch_is_exempt(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/pkg/middleware.py",
        "from starlette.middleware.base import BaseHTTPMiddleware\n"
        "\n"
        "class Middleware(BaseHTTPMiddleware):\n"
        "    async def dispatch(self, request, call_next):\n"
        "        return await call_next(request)\n"
        "\n"
        "MIDDLEWARE = Middleware\n",
    )

    assert _check(tmp_path) == []


def test_a_method_called_on_self_is_live(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/pkg/owner.py",
        "class Owner:\n"
        "    def run(self):\n"
        "        return self.helper()\n"
        "\n"
        "    def helper(self):\n"
        "        return 1\n",
    )
    _write(tmp_path, "tests/test_owner.py", "from src.pkg.owner import Owner\n\n\ndef test_run():\n    assert Owner().run() == 1\n")

    rendered = " ".join(_check(tmp_path))

    assert "Owner.helper" not in rendered


def test_a_dunder_method_is_exempt(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/pkg/resource.py",
        "class Resource:\n"
        "    def __enter__(self):\n"
        "        return self\n"
        "\n"
        "    def __exit__(self, exc_type, exc, tb):\n"
        "        return False\n"
        "\n"
        "RESOURCE = Resource\n",
    )
    _write(tmp_path, "tests/test_resource.py", "from src.pkg.resource import Resource\n\n\ndef test_enter():\n    with Resource():\n        pass\n")

    assert _check(tmp_path) == []


def test_a_getattr_string_keeps_a_symbol_live(tmp_path: Path) -> None:
    _write(tmp_path, "src/pkg/module.py", "def handler():\n    return 1\n")
    _write(tmp_path, "src/pkg/user.py", "def call():\n    return getattr(object(), \"handler\")\n")

    rendered = " ".join(_check(tmp_path))

    assert "handler" not in rendered


def test_a_deprecation_warning_shim_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/pkg/module.py",
        "import warnings\n"
        "\n"
        "def old():\n"
        "    warnings.warn(\"old\", DeprecationWarning)\n",
    )

    violations = _check(tmp_path)

    assert any(violation.startswith("COMPATIBILITY src/pkg/module.py:4") for violation in violations)


def test_a_stale_allowlist_entry_fails(tmp_path: Path) -> None:
    checker = DeadSymbolChecker(root=tmp_path, allowlist={"src/pkg/module.py::gone": "no such symbol"})

    violations = checker.check()

    assert any(violation == "STALE-ALLOWLIST src/pkg/module.py::gone" for violation in violations)


def test_every_allowlist_entry_names_a_reason() -> None:
    for key, reason in ALLOWLIST.items():
        assert "::" in key
        assert len(reason) >= 20


def test_references_mode_lists_call_sites(tmp_path: Path) -> None:
    _write(tmp_path, "src/pkg/module.py", "def orphan():\n    return 1\n")
    _write(tmp_path, "tests/test_module.py", "from src.pkg.module import orphan\n\n\ndef test_it():\n    orphan()\n")

    references = DeadSymbolChecker(root=tmp_path, allowlist={}).references_to("orphan")

    assert any(reference.startswith("tests/test_module.py:") for reference in references)


def test_the_scanner_aborts_when_a_canary_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write(tmp_path, "src/pkg/module.py", "def orb():\n    return 1\n")
    monkeypatch.setattr("scripts.check_dead_symbols.KNOWN_USED", ("orb",))

    with pytest.raises(DeadSymbolScannerError, match="orb"):
        _check(tmp_path)
