"""Gate: a private helper belongs to the class that uses it, and no free function sits between classes.

A module-level function that only one class in the same module calls is a helper that never
made it into its owner: it reads as a loose step in the file, and the next patch grows it
without the class's invariants. Numba kernels are exempt because numba does not compile
methods, and ``ALLOWLIST`` carries any accepted exception together with its reason (empty
today: every helper this check can prove class-bound has been moved to its class).

The second rule is layout: a ``def`` between two ``class`` statements splits a file into two
islands and hides which class owns what.
"""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
NUMBA_DECORATORS = ("njit", "jit")

ALLOWLIST: dict[str, str] = {}


def _names_used(node: ast.AST) -> set[str]:
    """Every plain name the subtree reads, attribute names excluded."""
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


class LooseHelperChecker:
    """Find private helpers that hang beside the class that calls them."""

    def __init__(self, root: Path = PROJECT_ROOT) -> None:
        self.root = root
        self.src_dir = root / "src"
        self.scan_dirs = (self.src_dir, root / "tests", root / "scripts")
        self._trees: dict[Path, ast.Module] = {}
        self._foreign_imports: dict[Path, set[str]] = {}

    def _import_path(self, path: Path) -> str:
        """Import path of a scanned file, empty when it is not importable as a module."""
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            return ""
        if not relative.parts or relative.parts[0] not in ("src", "tests"):
            return ""
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    def _parse(self, path: Path) -> ast.Module | None:
        if path not in self._trees:
            try:
                self._trees[path] = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                return None
        return self._trees[path]

    def _collect_foreign_imports(self) -> None:
        """Names imported into a file from another file of this repository."""
        for directory in self.scan_dirs:
            for path in directory.rglob("*.py"):
                tree = self._parse(path)
                if tree is None:
                    continue
                package = self._import_path(path)
                if not package:
                    continue
                for node in ast.walk(tree):
                    if not isinstance(node, ast.ImportFrom):
                        continue
                    dotted = self._resolved_module(node, package)
                    if not dotted.startswith("src"):
                        continue
                    target = self._source_file(dotted)
                    if target is None:
                        continue
                    for alias in node.names:
                        self._foreign_imports.setdefault(target, set()).add(alias.name)

    @staticmethod
    def _resolved_module(node: ast.ImportFrom, package: str) -> str:
        """Absolute module of an import, relative ones resolved against the importing package."""
        if not node.level:
            return node.module or ""
        parts = package.split(".")[:-1]
        if node.level > 1:
            parts = parts[: max(len(parts) - (node.level - 1), 0)]
        return ".".join([*parts, node.module] if node.module else parts)

    def _source_file(self, dotted: str) -> Path | None:
        candidate = self.root / Path(*dotted.split("."))
        for path in (candidate.with_suffix(".py"), candidate / "__init__.py"):
            if path.is_file():
                return path
        return None

    @staticmethod
    def _is_numba_kernel(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
            if name in NUMBA_DECORATORS:
                return True
        return False

    def _violations_of(self, path: Path, tree: ast.Module) -> list[str]:
        relative = path.relative_to(self.root).as_posix()
        body = tree.body
        classes = [node for node in body if isinstance(node, ast.ClassDef)]
        functions = [node for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        imported = self._foreign_imports.get(path, set())

        module_scope_names: set[str] = set()
        for statement in body:
            if isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            module_scope_names |= _names_used(statement)

        function_scope_names: dict[str, set[str]] = {}
        for node in functions:
            function_scope_names[node.name] = _names_used(node) - {node.name}

        violations: list[str] = []
        for node in functions:
            if not node.name.startswith("_") or self._is_numba_kernel(node):
                continue
            owners = [cls.name for cls in classes if node.name in _names_used(cls)]
            if len(owners) != 1:
                continue
            if node.name in imported or node.name in module_scope_names:
                continue
            if any(node.name in used for used in function_scope_names.values()):
                continue
            key = f"{relative}:{node.name}"
            reason = ALLOWLIST.get(key)
            if reason:
                continue
            violations.append(f"{relative}:{node.lineno} {node.name} -> {', '.join(owners)}")

        positions = [index for index, node in enumerate(body) if isinstance(node, ast.ClassDef)]
        if len(positions) > 1:
            for index in range(positions[0] + 1, positions[-1]):
                statement = body[index]
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    violations.append(
                        f"{relative}:{statement.lineno} {statement.name} sits between class definitions"
                    )
        return violations

    def check(self) -> list[str]:
        self._collect_foreign_imports()
        violations: list[str] = []
        for path in sorted(self.src_dir.rglob("*.py")):
            tree = self._parse(path)
            if tree is not None:
                violations.extend(self._violations_of(path, tree))
        return violations


def main() -> int:
    checker = LooseHelperChecker()
    violations = checker.check()
    if not violations:
        print("Loose helpers: none. Every private helper lives in the class that calls it.")
        return 0
    print(f"Loose helpers: {len(violations)}")
    for violation in violations:
        print(f"  {violation}")
    print("Move each helper onto the class that calls it, or document why in ALLOWLIST.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
