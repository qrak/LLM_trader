"""Check that every repository path quoted in AGENTS.md files still exists.

The layered AGENTS.md files are only useful if the paths they name are real; a rename
silently turns "read `src/x.py` first" into a dead end. This script extracts backticked
repository paths from the root file and every package file and compares them with the tree.

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
TOKEN = re.compile(r"`([^`\n]+)`")

def agent_files(root: Path) -> list[Path]:
    files = [root / "AGENTS.md"]
    files.extend(sorted((root / "src").rglob("AGENTS.md")))
    return [f for f in files if f.exists()]

def candidate(raw: str) -> str | None:
    token = raw.strip().rstrip(".,;:)").removeprefix("./")
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
    for file in agent_files(root):
        rel = file.relative_to(root).as_posix()
        for lineno, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            for match in TOKEN.finditer(line):
                token = candidate(match.group(1))
                if token is None:
                    continue
                checked += 1
                if not (root / token).exists():
                    missing.setdefault(token, []).append((rel, lineno))

    if not args.quiet:
        print(f"checked {checked} path references in {len(agent_files(root))} AGENTS.md files")
    if not missing:
        print("all referenced paths exist")
        return 0
    for token, places in sorted(missing.items()):
        where = ", ".join(f"{name}:{line}" for name, line in places)
        print(f"missing: {token}  <- {where}")
    return 1

if __name__ == "__main__":
    sys.exit(main())
