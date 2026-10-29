from __future__ import annotations

import argparse
import ast
import io
import tokenize
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_DIRECTORIES = ("src", "tests", "scripts")
PYTHON_DIRECTIVES = (
    "!",
    "coding:",
    "fmt:",
    "isort:",
    "mypy:",
    "noqa",
    "nosec",
    "pragma:",
    "pylint:",
    "pyright:",
    "ruff:",
    "type:",
)
WEB_DIRECTIVES = (
    "eslint-disable",
    "eslint-enable",
    "istanbul ignore",
    "prettier-ignore",
    "stylelint-disable",
    "stylelint-enable",
)


class CommentPolicyError(RuntimeError):
    """A file could not be parsed or tokenized."""


class CommentPolicyChecker:
    """Fail on prose comments and on docstrings outside a class.

    Python is read with ``tokenize`` for comments (a ``#`` inside a string is not a
    comment) and with ``ast`` for docstrings: a module, function or method docstring is
    a violation, a class docstring is allowed. Dashboard JavaScript, CSS and HTML are
    read line-first for ``//`` comments and block comments; a machine directive
    (``eslint-disable``, ``stylelint-disable``, ``prettier-ignore``) survives.

    YAML, ``config.ini``, ``requirements*.txt``, ``docs/`` and shell/PowerShell start
    scripts are deliberately out of scope: those files carry operational documentation
    that the repository's own rules require.
    """

    def __init__(self, root: Path = PROJECT_ROOT) -> None:
        self.root = root.resolve()
        self.python_files = self._python_files()
        self.web_files = self._web_files()

    def _python_files(self) -> list[Path]:
        files: list[Path] = []
        for directory in PYTHON_DIRECTORIES:
            path = self.root / directory
            if path.is_dir():
                files.extend(sorted(path.rglob("*.py")))
        start_file = self.root / "start.py"
        if start_file.is_file():
            files.append(start_file)
        return files

    def _web_files(self) -> list[Path]:
        static = self.root / "src" / "dashboard" / "static"
        if not static.is_dir():
            return []
        return sorted(
            path
            for extension in ("*.js", "*.css", "*.html")
            for path in static.rglob(extension)
        )

    def _relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()

    @staticmethod
    def _is_directive(comment: str) -> bool:
        body = comment.lstrip("#").strip()
        return body.startswith(PYTHON_DIRECTIVES)

    def _python_violations(self, path: Path) -> list[str]:
        try:
            text = path.read_text(encoding="utf-8")
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
            tree = ast.parse(text)
        except (OSError, SyntaxError, tokenize.TokenError) as exc:
            raise CommentPolicyError(f"{path}: {exc}") from exc
        relative = self._relative(path)
        violations: list[str] = []
        for token in tokens:
            if token.type == tokenize.COMMENT and not self._is_directive(token.string):
                violations.append(f"COMMENT {relative}:{token.start[0]} {token.string.strip()}")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node, clean=False)
                if docstring is not None:
                    violations.append(f"DOCSTRING {relative}:{node.body[0].lineno} function {node.name}")
            elif isinstance(node, ast.Module):
                docstring = ast.get_docstring(node, clean=False)
                if docstring is not None:
                    violations.append(f"DOCSTRING {relative}:1 module")
        return violations

    @staticmethod
    def _web_directive(text: str) -> bool:
        body = text.lstrip("/*<!- \t").strip()
        return body.startswith(WEB_DIRECTIVES)

    @staticmethod
    def _slash_comment_index(line: str) -> int | None:
        quote = ""
        escaped = False
        for index, character in enumerate(line):
            if escaped:
                escaped = False
                continue
            if character == "\\":
                escaped = True
                continue
            if quote:
                if character == quote:
                    quote = ""
                continue
            if character in "'\"`":
                quote = character
                continue
            if character == "/" and index + 1 < len(line) and line[index + 1] == "/":
                if index > 0 and line[index - 1] == ":":
                    continue
                return index
        return None

    def _web_violations(self, path: Path) -> list[str]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise CommentPolicyError(f"{path}: {exc}") from exc
        relative = self._relative(path)
        violations: list[str] = []
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if self._slash_comment_index(line) is not None:
                violations.append(f"WEB-COMMENT {relative}:{lineno} {stripped}")
            for marker in ("/*", "<!--"):
                if marker not in line:
                    continue
                if not self._web_directive(line[line.index(marker) + len(marker):]):
                    violations.append(f"WEB-COMMENT {relative}:{lineno} {stripped}")
                break
        return violations

    def check(self) -> list[str]:
        violations: list[str] = []
        for path in self.python_files:
            violations.extend(self._python_violations(path))
        for path in self.web_files:
            violations.extend(self._web_violations(path))
        return sorted(violations)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail on prose comments and on module, function or method docstrings; a class docstring is allowed."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="repository root to scan")
    arguments = parser.parse_args()
    checker = CommentPolicyChecker(root=arguments.root)
    try:
        violations = checker.check()
    except CommentPolicyError as exc:
        print(f"Comment-policy gate is broken: {exc}")
        return 2
    if not violations:
        print("Comments: none outside machine directives. Docstrings: class-only.")
        return 0
    print(f"Comment and docstring violations: {len(violations)}")
    for violation in violations:
        print(f"  {violation}")
    print("Delete the comment or docstring; a class may carry one docstring, nothing else may.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
