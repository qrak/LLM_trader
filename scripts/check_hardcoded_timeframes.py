

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TIMEFRAMES = ("5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w")
ALLOWLIST: dict[str, str] = {
    "src/utils/timeframe_validator.py": "defines the supported frames and their minute values",
    "src/analyzer/data_fetcher.py": "TREND_TIMEFRAME and WEEK_TIMEFRAME name the macro frames the fetcher requests",
    "src/analyzer/prompts/prompt_builder.py": "multi-timeframe summary labels are periods, not the analysis frame",
}


class HardcodedTimeframeChecker:
    """Find timeframe literals outside the module that defines them."""

    def __init__(self, root: Path = PROJECT_ROOT) -> None:
        self.root = root
        self.src_dir = root / "src"

    def _docstring_constants(self, tree: ast.Module) -> set[int]:
        constants: set[int] = set()
        holders: list[ast.AST] = [tree]
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                holders.append(node)
        for holder in holders:
            body = getattr(holder, "body", [])
            if not body:
                continue
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                constants.add(id(first.value))
        return constants

    def check(self) -> list[str]:

        violations: list[str] = []
        for path in sorted(self.src_dir.rglob("*.py")):
            relative = path.relative_to(self.root).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if relative in ALLOWLIST:
                continue
            docstrings = self._docstring_constants(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant):
                    continue
                if not isinstance(node.value, str) or node.value not in TIMEFRAMES:
                    continue
                if id(node) in docstrings:
                    continue
                violations.append(
                    f"{relative}:{node.lineno}: timeframe literal {node.value!r} "
                    "must come from Config.TIMEFRAME"
                )
        return violations


def main() -> int:
    violations = HardcodedTimeframeChecker().check()
    if not violations:
        print("no hardcoded timeframes outside the allowlist")
        return 0
    for violation in violations:
        print(violation)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
