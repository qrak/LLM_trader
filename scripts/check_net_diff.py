from __future__ import annotations

import argparse
import fnmatch
import subprocess
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_PREFIXES = ("src/", "scripts/")
REFACTOR_LABELS = frozenset({"cleanup", "dead-code", "refactor", "retirement"})
REFACTOR_TITLE_PREFIXES = ("refactor", "cleanup", "retire", "chore(cleanup)")
DEFAULT_EXCLUSIONS = (
    "CHANGELOG.md",
    "README.md",
    "cache/*",
    "data/*",
    "docs/*",
    "logs/*",
    "package-lock.json",
    "pnpm-lock.yaml",
    "tests/fixtures/*",
    "*.gif",
    "*.ico",
    "*.jpeg",
    "*.jpg",
    "*.lock",
    "*.map",
    "*.png",
    "*.svg",
    "*.webp",
    "*.woff",
    "*.woff2",
)


@dataclass(frozen=True, slots=True)
class FileDelta:
    path: str
    added: int
    deleted: int

    @property
    def net(self) -> int:
        return self.added - self.deleted


@dataclass(frozen=True, slots=True)
class DiffBudget:
    files: tuple[FileDelta, ...]
    scope: str

    @classmethod
    def from_numstat(
        cls,
        numstat: str,
        scope: str = "code",
        exclusions: tuple[str, ...] = DEFAULT_EXCLUSIONS,
    ) -> DiffBudget:
        files: list[FileDelta] = []
        for line in numstat.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t", 2)
            if len(parts) != 3:
                raise ValueError(f"unexpected numstat line: {line!r}")
            added_text, deleted_text, path = parts
            added = 0 if added_text == "-" else int(added_text)
            deleted = 0 if deleted_text == "-" else int(deleted_text)
            if not cls._included(path, scope, exclusions):
                continue
            files.append(FileDelta(path, added, deleted))
        return cls(tuple(files), scope)

    @staticmethod
    def _included(path: str, scope: str, exclusions: tuple[str, ...]) -> bool:
        if any(fnmatch.fnmatch(path, pattern) for pattern in exclusions):
            return False
        if scope == "total":
            return True
        return path == "start.py" or path.startswith(CODE_PREFIXES)

    @property
    def added(self) -> int:
        return sum(file.added for file in self.files)

    @property
    def deleted(self) -> int:
        return sum(file.deleted for file in self.files)

    @property
    def net(self) -> int:
        return self.added - self.deleted

    def offenders(self) -> tuple[FileDelta, ...]:
        return tuple(sorted((file for file in self.files if file.net > 0), key=lambda file: (-file.net, file.path)))

    def summary(self) -> str:
        return (
            f"Net diff ({self.scope}): +{self.added} / -{self.deleted} = {self.net:+d} lines "
            f"across {len(self.files)} files"
        )

    def violations(self, quota: int) -> list[str]:
        if self.net <= quota:
            return []
        lines = [f"Budget violation: net {self.net:+d} is above the quota {quota:+d}."]
        for file in self.offenders()[:10]:
            lines.append(f"  adds {file.net:+d}: {file.path}")
        return lines


class NetDiffError(RuntimeError):
    """The git range could not be measured."""


class NetDiffChecker:
    """Turn a refactor's diff into a budget: the production surface must shrink.

    ``code`` scope is ``src/``, ``scripts/`` and ``start.py``; docs, changelog, fixtures
    and binary assets are excluded. ``total`` scope measures everything that is left.
    A pull request is treated as a refactor when it carries one of ``REFACTOR_LABELS``
    or its title starts with a refactor prefix; otherwise the checker only reports the
    number and exits zero, so a feature pull request is never failed by this gate.
    """

    def __init__(
        self,
        root: Path = PROJECT_ROOT,
        base: str = "origin/master",
        head: str = "HEAD",
        scope: str = "code",
        staged: bool = False,
        worktree: bool = False,
        exclusions: tuple[str, ...] = DEFAULT_EXCLUSIONS,
    ) -> None:
        self.root = root.resolve()
        self.base = base
        self.head = head
        self.scope = scope
        self.staged = staged
        self.worktree = worktree
        self.exclusions = exclusions

    def _git_numstat(self) -> str:
        command = ["git", "diff", "--numstat"]
        if self.worktree:
            command.append(self.base)
        elif self.staged:
            command.extend(["--cached", self.base])
        else:
            command.append(f"{self.base}...{self.head}")
        result = subprocess.run(
            command,
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip() or "git returned no detail"
            raise NetDiffError(f"{' '.join(command)} failed: {detail}")
        return result.stdout

    @staticmethod
    def is_refactor(labels: str = "", title: str = "") -> bool:
        names = {label.strip().casefold() for label in labels.split(",") if label.strip()}
        if names & REFACTOR_LABELS:
            return True
        normalized = title.strip().casefold()
        return any(normalized.startswith(prefix) for prefix in REFACTOR_TITLE_PREFIXES)

    def budget(self, numstat: str | None = None) -> DiffBudget:
        return DiffBudget.from_numstat(
            self._git_numstat() if numstat is None else numstat,
            scope=self.scope,
            exclusions=self.exclusions,
        )

    def check(
        self,
        labels: str = "",
        title: str = "",
        require_negative: bool = False,
        quota: int = -1,
        numstat: str | None = None,
    ) -> tuple[int, str]:
        budget = self.budget(numstat)
        lines = [budget.summary()]
        offenders = budget.offenders()
        if offenders:
            lines.append("Files adding lines: " + ", ".join(f"{file.path} ({file.net:+d})" for file in offenders[:5]))
        if not (require_negative or self.is_refactor(labels, title)):
            lines.append("REPORT ONLY: label the pull request refactor/cleanup or pass --require-negative to enforce the budget.")
            return 0, "\n".join(lines)
        violations = budget.violations(quota)
        if violations:
            lines.extend(violations)
            lines.append(f"FAIL: the production diff must be at most {quota:+d} lines net.")
            return 1, "\n".join(lines)
        lines.append(f"PASS: the production diff is net-negative (quota {quota:+d}).")
        return 0, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail a refactor pull request whose production diff does not remove more lines than it adds."
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="repository root")
    parser.add_argument("--base", default="origin/master", help="base ref or commit")
    parser.add_argument("--head", default="HEAD", help="head ref or commit")
    parser.add_argument("--labels", default="", help="comma-separated pull request labels")
    parser.add_argument("--title", default="", help="pull request title")
    parser.add_argument("--scope", choices=("code", "total"), default="code")
    parser.add_argument("--quota", type=int, default=-1, help="maximum allowed net lines (default -1)")
    parser.add_argument("--require-negative", action="store_true", help="enforce the budget even without a refactor label")
    parser.add_argument("--staged", action="store_true", help="measure staged changes against --base instead of a committed range")
    parser.add_argument("--worktree", action="store_true", help="measure the working tree against --base before staging")
    arguments = parser.parse_args()
    checker = NetDiffChecker(
        root=arguments.root,
        base=arguments.base,
        head=arguments.head,
        scope=arguments.scope,
        staged=arguments.staged,
        worktree=arguments.worktree,
    )
    try:
        code, output = checker.check(
            labels=arguments.labels,
            title=arguments.title,
            require_negative=arguments.require_negative,
            quota=arguments.quota,
        )
    except NetDiffError as exc:
        print(f"Net-diff gate is broken: {exc}")
        return 2
    print(output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
