#!/usr/bin/env python3
"""Post-process Codacy SARIF output for GitHub Code Scanning upload.

Codacy CLI (codacy-analysis-cli) sometimes emits absolute paths or paths
prefixed with ``/src/`` (its Docker mount point). GitHub Code Scanning
requires paths relative to the repository root. This script:

1. Finds the most recent ``.sarif`` file (or accepts one as argument).
2. Strips known path prefixes so artifact locations are repo-relative.
3. Writes the fixed output to ``fixed.sarif`` (or ``--output PATH``).

Usage::

    python3 scripts/fix_sarif.py [input.sarif] [--output fixed.sarif]

If no input is given, looks for ``codacy.sarif`` then any ``*.sarif`` file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent

# Prefixes that Codacy's Docker container may prepend to paths.
# Ordered from most-specific to least.
STRIP_PREFIXES: list[str] = [
    "/src/",                     # Codacy Docker default mount
    "/home/runner/work/LLM_trader/LLM_trader/",  # GitHub Actions runner
]


def find_sarif() -> Path | None:
    """Locate the most recent SARIF file in the repo root or CWD."""
    candidates = sorted(
        [p for p in Path.cwd().glob("*.sarif")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Prefer codacy.sarif if it exists
    for c in candidates:
        if c.name == "codacy.sarif":
            return c
    return candidates[0] if candidates else None


def fix_uri(uri: str) -> str:
    """Strip known path prefixes from a SARIF artifact URI."""
    for prefix in STRIP_PREFIXES:
        if uri.startswith(prefix):
            return uri[len(prefix):]
    return uri


def fix_sarif(data: dict[str, Any]) -> dict[str, Any]:
    """Walk SARIF runs and fix artifact locations."""
    for run in data.get("runs", []):
        # Fix tool.driver.name for Codacy
        tool = run.get("tool", {})
        driver = tool.get("driver", {})
        if not driver.get("name"):
            driver["name"] = "Codacy"

        for result in run.get("results", []):
            # Fix primary location
            for loc in result.get("locations", []):
                phys = loc.get("physicalLocation", {})
                art_loc = phys.get("artifactLocation", {})
                uri = art_loc.get("uri", "")
                if uri:
                    art_loc["uri"] = fix_uri(uri)

            # Fix related locations
            for rel_loc in result.get("relatedLocations", []):
                phys = rel_loc.get("physicalLocation", {})
                art_loc = phys.get("artifactLocation", {})
                uri = art_loc.get("uri", "")
                if uri:
                    art_loc["uri"] = fix_uri(uri)

    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix SARIF paths for GitHub Code Scanning")
    parser.add_argument("input", nargs="?", help="Input SARIF file (auto-detected if omitted)")
    parser.add_argument("--output", "-o", default="fixed.sarif", help="Output file (default: fixed.sarif)")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else find_sarif()

    if input_path is None or not input_path.exists():
        print(f"❌ No SARIF file found. Looked in: {Path.cwd()}", file=sys.stderr)
        return 1

    print(f"📄 Reading: {input_path}")
    with open(input_path, encoding="utf-8") as fh:
        data = json.load(fh)

    results_before = sum(len(run.get("results", [])) for run in data.get("runs", []))
    data = fix_sarif(data)

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    print(f"✅ Fixed {results_before} results → {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
