from __future__ import annotations

import argparse
import ast
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
START_FILE = "start.py"
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

FRAMEWORK_DECORATORS = frozenset({
    "abstractmethod",
    "api_route",
    "computed_field",
    "delete",
    "exception_handler",
    "field_validator",
    "get",
    "jit",
    "middleware",
    "model_validator",
    "njit",
    "on_event",
    "overload",
    "patch",
    "post",
    "put",
    "root_validator",
    "route",
    "tool",
    "validator",
    "vectorize",
    "websocket",
})

FRAMEWORK_BASES = frozenset({
    "BaseHTTPMiddleware",
    "BaseModel",
    "BaseSettings",
    "Enum",
    "Exception",
    "Formatter",
    "HTMLParser",
    "IntEnum",
    "NamedTuple",
    "Protocol",
    "StrEnum",
    "TypedDict",
})

DYNAMIC_CALLS = frozenset({
    "getattr",
    "hasattr",
    "import_module",
    "patch",
    "patch.object",
    "setattr",
})

ALLOWLIST: dict[str, str] = {
    "src/dashboard/auth.py::hash_password": (
        "Documented generation command in keys.env.example; production only verifies the stored hash."
    ),
    "src/utils/time_utils.py::reset_timezone_cache": (
        "Test seam that clears the module timezone cache between cases."
    ),
}

KNOWN_USED: tuple[str, ...] = (
    "AnalysisEngine",
    "Config",
    "DashboardServer",
    "RagEngine",
    "RiskManager",
    "TokenCounter",
    "TradingStrategy",
    "UnifiedParser",
)


@dataclass(frozen=True, slots=True)
class Definition:
    path: Path
    lineno: int
    kind: str
    name: str
    qualified: str


@dataclass(frozen=True, slots=True)
class ReferenceEvent:
    bucket: str
    name: str
    path: Path
    lineno: int
    kind: str


@dataclass(frozen=True, slots=True)
class Violation:
    category: str
    path: str
    lineno: int
    kind: str
    qualified: str
    name: str

    def render(self) -> str:
        if self.category == "STALE-ALLOWLIST":
            return f"STALE-ALLOWLIST {self.qualified}"
        return f"{self.category} {self.path}:{self.lineno} {self.kind} {self.qualified}"


class DeadSymbolScannerError(RuntimeError):
    """The scanner itself is wrong: a symbol proven live was reported dead."""


class DeadSymbolChecker:
    """Fail when a function, method or class in ``src/`` has no consumer.

    Three reference buckets keep the verdict honest. ``runtime`` is ``src/`` plus
    ``start.py``; ``tooling`` is ``scripts/``; ``tests`` is ``tests/``. A symbol
    referenced by runtime or tooling is live. A symbol referenced only by tests is
    ``TEST-ONLY``: the classic additive-bias leftover kept green by its own unit test,
    which must be deleted with the test or justified in ``ALLOWLIST``. A symbol with
    no reference at all is ``DEAD``.

    Imports inside ``__init__.py`` are exports, not uses: a class that only appears in
    ``__all__`` and a package re-export is still an orphan export. Framework callbacks
    are exempt through their decorator (FastAPI routes, Pydantic validators, numba
    kernels) or their base class (``HTMLParser``, ``BaseHTTPMiddleware``, Pydantic
    models, protocols). Dynamic names count: strings passed to ``getattr``, ``patch``
    and friends, plus exact string constants, keep a symbol alive.

    Names are counted globally, so a local variable with the same name can hide a dead
    definition. That false negative is deliberate; the ``KNOWN_USED`` canaries abort the
    run when the scanner starts reporting proven-live symbols, and ``--symbol`` prints
    every call site before a deletion.
    """

    def __init__(self, root: Path = PROJECT_ROOT, allowlist: dict[str, str] | None = None) -> None:
        self.root = root.resolve()
        self.allowlist = dict(ALLOWLIST if allowlist is None else allowlist)
        self.src_dir = self.root / "src"
        self.src_files = self._python_files(self.src_dir)
        start_file = self.root / START_FILE
        if start_file.is_file():
            self.src_files.append(start_file)
        self.all_files = sorted({
            *self.src_files,
            *self._python_files(self.root / "tests"),
            *self._python_files(self.root / "scripts"),
        })
        self._trees: dict[Path, ast.Module] = {}
        self._events: list[ReferenceEvent] | None = None

    @staticmethod
    def _python_files(directory: Path) -> list[Path]:
        if not directory.is_dir():
            return []
        return sorted(directory.rglob("*.py"))

    def _parse(self, path: Path) -> ast.Module:
        if path not in self._trees:
            try:
                self._trees[path] = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError) as exc:
                raise DeadSymbolScannerError(f"{path}: {exc}") from exc
        return self._trees[path]

    def _relative(self, path: Path) -> str:
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return path.as_posix()

    def _bucket(self, path: Path) -> str:
        parts = path.relative_to(self.root).parts
        if parts and parts[0] == "tests":
            return "tests"
        if parts and parts[0] == "scripts":
            return "tooling"
        return "runtime"

    @staticmethod
    def _is_dunder(name: str) -> bool:
        return name.startswith("__") and name.endswith("__")

    @staticmethod
    def _call_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return ast.unparse(node)
        return ""

    @staticmethod
    def _decorator_name(node: ast.expr) -> str:
        target = node.func if isinstance(node, ast.Call) else node
        if isinstance(target, ast.Attribute):
            return target.attr
        if isinstance(target, ast.Name):
            return target.id
        return ""

    @staticmethod
    def _base_name(node: ast.expr) -> str:
        if isinstance(node, ast.Subscript):
            node = node.value
        return ast.unparse(node).split(".")[-1]

    def _decorator_names(self, node: ast.AST) -> set[str]:
        decorators = getattr(node, "decorator_list", ())
        return {self._decorator_name(decorator) for decorator in decorators}

    def _is_exempt(self, node: ast.AST, parents: list[ast.expr]) -> bool:
        if self._decorator_names(node) & FRAMEWORK_DECORATORS:
            return True
        return any(self._base_name(base) in FRAMEWORK_BASES for base in parents)

    def _definitions(self) -> list[Definition]:
        definitions: list[Definition] = []
        for path in self.src_files:
            tree = self._parse(path)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if self._is_dunder(node.name) or node.name == "main":
                        continue
                    if self._is_exempt(node, []):
                        continue
                    definitions.append(Definition(path, node.lineno, "function", node.name, node.name))
                    continue
                if not isinstance(node, ast.ClassDef):
                    continue
                if not self._is_dunder(node.name) and not self._is_exempt(node, []):
                    definitions.append(Definition(path, node.lineno, "class", node.name, node.name))
                for member in node.body:
                    if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if self._is_dunder(member.name):
                        continue
                    if self._is_exempt(member, node.bases):
                        continue
                    kind = "property" if "property" in self._decorator_names(member) else "method"
                    qualified = f"{node.name}.{member.name}"
                    definitions.append(Definition(path, member.lineno, kind, member.name, qualified))
        return definitions

    def _docstring_ids(self, tree: ast.Module) -> set[int]:
        identifiers: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.body:
                continue
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                identifiers.add(id(first.value))
        return identifiers

    def _dunder_all_ids(self, tree: ast.Module) -> set[int]:
        identifiers: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                continue
            for child in ast.walk(node.value):
                if isinstance(child, ast.Constant):
                    identifiers.add(id(child))
        return identifiers

    def _string_references(self, tree: ast.Module) -> list[ReferenceEvent]:
        events: list[ReferenceEvent] = []
        skipped = self._docstring_ids(tree) | self._dunder_all_ids(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skipped:
                for token in IDENTIFIER.findall(node.value):
                    events.append(ReferenceEvent("runtime", token, Path(), node.lineno, "string"))
            if not isinstance(node, ast.Call):
                continue
            if self._call_name(node.func) not in DYNAMIC_CALLS:
                continue
            arguments: list[ast.expr] = [*node.args, *(keyword.value for keyword in node.keywords)]
            for argument in arguments:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    for token in IDENTIFIER.findall(argument.value):
                        events.append(ReferenceEvent("runtime", token, Path(), argument.lineno, "dynamic"))
        return events

    def _collect_events(self) -> list[ReferenceEvent]:
        events: list[ReferenceEvent] = []
        for path in self.all_files:
            tree = self._parse(path)
            bucket = self._bucket(path)
            is_init = path.name == "__init__.py"
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    events.append(ReferenceEvent(bucket, node.id, path, node.lineno, "name"))
                elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    events.append(ReferenceEvent(bucket, node.attr, path, node.lineno, "attribute"))
                elif isinstance(node, ast.ImportFrom) and not is_init:
                    for alias in node.names:
                        events.append(ReferenceEvent(bucket, alias.name, path, node.lineno, "import"))
                elif isinstance(node, ast.Import) and not is_init:
                    for alias in node.names:
                        events.append(ReferenceEvent(bucket, alias.asname or alias.name.split(".")[0], path, node.lineno, "import"))
            if bucket == "runtime":
                for event in self._string_references(tree):
                    events.append(ReferenceEvent(bucket, event.name, path, event.lineno, event.kind))
        return events

    def reference_events(self) -> list[ReferenceEvent]:
        if self._events is None:
            self._events = self._collect_events()
        return self._events

    def _reference_counts(self) -> dict[str, Counter[str]]:
        counts = {"runtime": Counter(), "tooling": Counter(), "tests": Counter()}
        for event in self.reference_events():
            counts[event.bucket][event.name] += 1
        return counts

    def references_to(self, symbol: str) -> list[str]:
        target = symbol.split(".")[-1]
        matches = {
            f"{self._relative(event.path)}:{event.lineno} {event.kind}"
            for event in self.reference_events()
            if event.name == target
        }
        return sorted(matches)

    def stale_allowlist_entries(self) -> list[str]:
        keys = {f"{self._relative(definition.path)}::{definition.qualified}" for definition in self._definitions()}
        return sorted(set(self.allowlist) - keys)

    def _compatibility_violations(self) -> list[Violation]:
        violations: list[Violation] = []
        for path in self.src_files:
            tree = self._parse(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and self._call_name(node.func) in {"warnings.warn", "warn"}:
                    category = node.args[1] if len(node.args) > 1 else None
                    for keyword in node.keywords:
                        if keyword.arg == "category":
                            category = keyword.value
                    category_name = self._call_name(category).split(".")[-1] if isinstance(category, ast.expr) else ""
                    if category_name in {"DeprecationWarning", "PendingDeprecationWarning"}:
                        violations.append(
                            Violation(
                                "COMPATIBILITY",
                                self._relative(path),
                                node.lineno,
                                "call",
                                "warnings.warn",
                                "warnings.warn",
                            )
                        )
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and "deprecated" in self._decorator_names(node):
                    violations.append(
                        Violation(
                            "COMPATIBILITY",
                            self._relative(path),
                            node.lineno,
                            "definition",
                            node.name,
                            node.name,
                            )
                        )
        return violations

    def scan(self) -> list[Violation]:
        counts = self._reference_counts()
        violations: list[Violation] = []
        known_keys: set[str] = set()
        for definition in self._definitions():
            key = f"{self._relative(definition.path)}::{definition.qualified}"
            known_keys.add(key)
            if key in self.allowlist:
                continue
            runtime = counts["runtime"][definition.name]
            tooling = counts["tooling"][definition.name]
            if runtime or tooling:
                continue
            category = "TEST-ONLY" if counts["tests"][definition.name] else "DEAD"
            violations.append(
                Violation(
                    category,
                    self._relative(definition.path),
                    definition.lineno,
                    definition.kind,
                    definition.qualified,
                    definition.name,
                )
            )
        for key in sorted(set(self.allowlist) - known_keys):
            violations.append(Violation("STALE-ALLOWLIST", "", 0, "allowlist", key, key))
        violations.extend(self._compatibility_violations())
        canaries = {violation.name for violation in violations if violation.category in {"DEAD", "TEST-ONLY"}}
        broken = sorted(canaries & set(KNOWN_USED))
        if broken:
            raise DeadSymbolScannerError(
                "proven-live symbols reported dead: " + ", ".join(broken) + "; fix the scanner or update KNOWN_USED"
            )
        return sorted(
            violations,
            key=lambda violation: (violation.category, violation.path, violation.lineno, violation.qualified),
        )

    def check(self) -> list[str]:
        return [violation.render() for violation in self.scan()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail on functions, methods and classes that no runtime, test or tooling code references."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="repository root to scan")
    parser.add_argument("--symbol", help="print every reference to a symbol instead of scanning the tree")
    arguments = parser.parse_args()
    checker = DeadSymbolChecker(root=arguments.root)
    if arguments.symbol:
        references = checker.references_to(arguments.symbol)
        if not references:
            print(f"{arguments.symbol}: 0 references - delete it, do not keep it beside a replacement.")
            return 1
        print(f"{arguments.symbol}: {len(references)} references")
        for reference in references:
            print(f"  {reference}")
        return 0
    try:
        violations = checker.check()
    except DeadSymbolScannerError as exc:
        print(f"Dead-symbol gate is broken: {exc}")
        return 2
    if not violations:
        print("Dead symbols: none. Every definition in src/ is referenced by runtime, tests or scripts.")
        return 0
    dead = sum(violation.startswith("DEAD ") for violation in violations)
    test_only = sum(violation.startswith("TEST-ONLY ") for violation in violations)
    compatibility = sum(violation.startswith("COMPATIBILITY ") for violation in violations)
    stale = sum(violation.startswith("STALE-ALLOWLIST ") for violation in violations)
    print(
        f"Dead symbols: {len(violations)} "
        f"(dead={dead}, test-only={test_only}, compatibility={compatibility}, stale-allowlist={stale})"
    )
    for violation in violations:
        print(f"  {violation}")
    print("Delete each one in the change that superseded it, or add a reasoned ALLOWLIST entry.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
