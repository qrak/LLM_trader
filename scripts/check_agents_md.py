"""Check that every repository path quoted in the AGENTS files and the reader docs exists.

The layered AGENTS.md files are only useful if the paths they name are real, and a rename
turns "read `src/x.py` first" into a dead end just as quietly in the pages a reader lands on.
This script extracts backticked repository paths and markdown link targets from the root
AGENTS.md, every package AGENTS.md, README.md, CONTRIBUTING.md and docs/*.md, and compares
them with the tree. Changelog archives are history and are not scanned.

Usage:
    python scripts/check_agents_md.py
    python scripts/check_agents_md.py --quiet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PREFIXES = ("src/", "tests/", "scripts/", "docs/", "config/", ".github/")
EXACT = {"start.py", "AGENTS.md", "README.md", "CHANGELOG.md", "CONTRIBUTING.md", "LICENSE.md"}
# Paths the docs name on purpose even though git does not carry them: local configuration and
# secrets are created from their templates, so a fresh checkout legitimately lacks them.
ALLOWED_MISSING = {
    "config/config.ini",
    "config/peak_rates.json",
    "config/keys.env",
    "keys.env",
}
ARCHIVE = re.compile(r"changelog-archive-\d{4}\.md$")
TOKEN = re.compile(r"`([^`\n]+)`")
LINK = re.compile(r"\]\(([^)\s]+)")


def agent_files(root: Path) -> list[Path]:
    files = [root / "AGENTS.md"]
    files.extend(sorted((root / "src").rglob("AGENTS.md")))
    return [f for f in files if f.exists()]


def doc_files(root: Path) -> list[Path]:
    candidates = [root / "README.md", root / "CONTRIBUTING.md"]
    candidates.extend(sorted((root / "docs").glob("*.md")))
    return [f for f in candidates if f.exists() and not ARCHIVE.search(f.name)]


def scanned_files(root: Path) -> list[Path]:
    return agent_files(root) + doc_files(root)


def quoted_paths(line: str) -> list[str]:
    return TOKEN.findall(line) + LINK.findall(line)


def candidate(raw: str) -> str | None:
    token = raw.strip().strip("`").rstrip(".,;:)").removeprefix("./")
    token = token.split("#", 1)[0].split("?", 1)[0]
    if not token or "://" in token or token.startswith(("#", "mailto:", "<")):
        return None
    if token in EXACT:
        return token
    if not token.startswith(PREFIXES):
        return None
    if any(ch in token for ch in "*{}<>|") or " " in token:
        return None
    return token


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    missing: dict[str, list[tuple[str, int]]] = {}
    checked = 0
    files = scanned_files(root)
    for file in files:
        rel = file.relative_to(root).as_posix()
        for lineno, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            for raw in quoted_paths(line):
                token = candidate(raw)
                if token is None or token in ALLOWED_MISSING:
                    continue
                checked += 1
                if not (root / token).exists():
                    missing.setdefault(token, []).append((rel, lineno))

    if not args.quiet:
        print(f"checked {checked} path references in {len(files)} files")
    if not missing:
        print("all referenced paths exist")
        return 0
    for token, places in sorted(missing.items()):
        where = ", ".join(f"{name}:{line}" for name, line in places)
        print(f"missing: {token}  <- {where}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
