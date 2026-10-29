from __future__ import annotations

from pathlib import Path

from scripts.check_comment_policy import CommentPolicyChecker


def _write(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _check(root: Path) -> list[str]:
    return CommentPolicyChecker(root=root).check()


def test_the_repository_has_no_comments_or_non_class_docstrings() -> None:
    assert CommentPolicyChecker().check() == []


def test_a_prose_hash_comment_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "src/pkg/module.py", "VALUE = 1  # explain the value\n")

    violations = _check(tmp_path)

    assert any(violation.startswith("COMMENT src/pkg/module.py:1") for violation in violations)


def test_a_machine_directive_is_allowed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/pkg/module.py",
        "VALUE = 1  # noqa: E501\nOTHER = 2  # type: ignore[assignment]\n",
    )

    assert _check(tmp_path) == []


def test_a_module_docstring_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "src/pkg/module.py", '"""Module documentation."""\n\nVALUE = 1\n')

    violations = _check(tmp_path)

    assert any(violation.startswith("DOCSTRING src/pkg/module.py:1 module") for violation in violations)


def test_a_function_docstring_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "src/pkg/module.py", 'def helper():\n    """Explain the helper."""\n    return 1\n')

    violations = _check(tmp_path)

    assert any(violation.startswith("DOCSTRING src/pkg/module.py:2 function helper") for violation in violations)


def test_a_method_docstring_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/pkg/module.py",
        'class Owner:\n    def helper(self):\n        """Explain the helper."""\n        return 1\n',
    )

    violations = _check(tmp_path)

    assert any(violation.startswith("DOCSTRING src/pkg/module.py:3 function helper") for violation in violations)


def test_a_class_docstring_is_allowed(tmp_path: Path) -> None:
    _write(tmp_path, "src/pkg/module.py", 'class Owner:\n    """Owns the value."""\n\n    value = 1\n')

    assert _check(tmp_path) == []


def test_a_full_line_javascript_comment_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "src/dashboard/static/modules/panel.js", "// explain the panel\nexport const value = 1;\n")

    violations = _check(tmp_path)

    assert any(violation.startswith("WEB-COMMENT src/dashboard/static/modules/panel.js:1") for violation in violations)


def test_an_inline_javascript_comment_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "src/dashboard/static/modules/panel.js", "const value = 1; // explain the panel\n")

    violations = _check(tmp_path)

    assert any(violation.startswith("WEB-COMMENT src/dashboard/static/modules/panel.js:1") for violation in violations)


def test_a_url_in_a_javascript_string_is_not_a_comment(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/dashboard/static/modules/panel.js",
        'const url = "https://example.com/path";\nconst other = `wss://example.com/socket`;\n',
    )

    assert _check(tmp_path) == []


def test_a_css_block_comment_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "src/dashboard/static/css/panel.css", "/* explain the panel */\n.panel { color: red; }\n")

    violations = _check(tmp_path)

    assert any(violation.startswith("WEB-COMMENT src/dashboard/static/css/panel.css:1") for violation in violations)


def test_a_css_directive_is_allowed(tmp_path: Path) -> None:
    _write(tmp_path, "src/dashboard/static/css/panel.css", "/* stylelint-disable */\n.panel { color: red; }\n")

    assert _check(tmp_path) == []


def test_an_html_comment_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "src/dashboard/static/index.html", "<!-- explain the page -->\n<div></div>\n")

    violations = _check(tmp_path)

    assert any(violation.startswith("WEB-COMMENT src/dashboard/static/index.html:1") for violation in violations)


def test_yaml_and_docs_are_out_of_scope(tmp_path: Path) -> None:
    _write(tmp_path, ".github/workflows/tests.yml", "# operational comment\nname: tests\n")
    _write(tmp_path, "docs/NOTES.md", "<!-- reader documentation -->\n")

    assert _check(tmp_path) == []
