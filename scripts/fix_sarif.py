#!/usr/bin/env python3
"""Post-process Codacy SARIF output for GitHub Code Scanning upload.

Codacy CLI (codacy-analysis-cli) sometimes emits absolute paths or paths
prefixed with ``/src/`` (its Docker mount point), null ``rules`` arrays,
or invalid ``level`` enum values. GitHub Code Scanning requires:
1. Relative file paths.
2. Valid ``tool.driver.rules`` array (not null/None).
3. Valid result levels ("none", "note", "warning", "error").

This script fixes those schema violations so upload-sarif succeeds.
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
STRIP_PREFIXES: list[str] = [
    "/src/",                     # Codacy Docker default mount
    "/home/runner/work/LLM_trader/LLM_trader/",  # GitHub Actions runner
]

VALID_LEVELS = {"none", "note", "warning", "error"}

LEVEL_MAP = {
    "info": "note",
    "informational": "note",
    "debug": "note",
    "trace": "note",
    "low": "note",
    "notice": "note",
    "style": "note",
    "convention": "note",
    "medium": "warning",
    "warn": "warning",
    "high": "error",
    "critical": "error",
    "fatal": "error",
    "off": "none",
}


def find_sarif() -> Path | None:
    """Locate the most recent SARIF file in the repo root or CWD."""
    candidates = sorted(
        [p for p in Path.cwd().glob("*.sarif")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    # Prefer results.sarif then codacy.sarif then any *.sarif file
    for name in ("results.sarif", "codacy.sarif"):
        for c in candidates:
            if c.name == name:
                return c
    return candidates[0] if candidates else None


def fix_uri(uri: str) -> str:
    """Strip known path prefixes from a SARIF artifact URI."""
    for prefix in STRIP_PREFIXES:
        if uri.startswith(prefix):
            return uri[len(prefix):]
    return uri


def sanitize_level(level: Any) -> str:
    """Ensure level is a valid SARIF level enum ('none', 'note', 'warning', 'error')."""
    if level is None:
        return "warning"
    lvl_str = str(level).lower().strip()
    if lvl_str in VALID_LEVELS:
        return lvl_str
    if lvl_str in LEVEL_MAP:
        return LEVEL_MAP[lvl_str]
    return "warning"


def fix_sarif(data: dict[str, Any]) -> dict[str, Any]:
    """Walk SARIF runs and fix artifact locations, rules array, and result levels."""
    if not isinstance(data, dict):
        return data

    runs = data.get("runs")
    if not isinstance(runs, list):
        return data

    for run in runs:
        if not isinstance(run, dict):
            continue

        tool = run.get("tool")
        if not isinstance(tool, dict):
            tool = {}
            run["tool"] = tool

        driver = tool.get("driver")
        if not isinstance(driver, dict):
            driver = {}
            tool["driver"] = driver

        if not driver.get("name"):
            driver["name"] = "Codacy"

        # Fix tool.driver.rules: MUST be an array (list), not null/None or non-list
        rules = driver.get("rules")
        if rules is None or not isinstance(rules, list):
            driver["rules"] = []
        else:
            for rule in driver["rules"]:
                if isinstance(rule, dict):
                    default_config = rule.get("defaultConfiguration")
                    if isinstance(default_config, dict) and "level" in default_config:
                        default_config["level"] = sanitize_level(default_config.get("level"))

        results = run.get("results")
        if results is None or not isinstance(results, list):
            run["results"] = []
            continue

        for result in results:
            if not isinstance(result, dict):
                continue

            # Fix result level if present
            if "level" in result:
                result["level"] = sanitize_level(result.get("level"))

            # Fix primary locations
            locations = result.get("locations")
            if isinstance(locations, list):
                for loc in locations:
                    if not isinstance(loc, dict):
                        continue
                    phys = loc.get("physicalLocation")
                    if isinstance(phys, dict):
                        art_loc = phys.get("artifactLocation")
                        if isinstance(art_loc, dict):
                            uri = art_loc.get("uri")
                            if isinstance(uri, str) and uri:
                                art_loc["uri"] = fix_uri(uri)

            # Fix related locations
            related_locations = result.get("relatedLocations")
            if isinstance(related_locations, list):
                for rel_loc in related_locations:
                    if not isinstance(rel_loc, dict):
                        continue
                    phys = rel_loc.get("physicalLocation")
                    if isinstance(phys, dict):
                        art_loc = phys.get("artifactLocation")
                        if isinstance(art_loc, dict):
                            uri = art_loc.get("uri")
                            if isinstance(uri, str) and uri:
                                art_loc["uri"] = fix_uri(uri)

    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix SARIF paths, rules, and levels for GitHub Code Scanning")
    parser.add_argument("input", nargs="?", help="Input SARIF file (auto-detected if omitted)")
    parser.add_argument("--output", "-o", default=None, help="Output file (default: same as input file)")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else find_sarif()

    if input_path is None or not input_path.exists():
        print(f"❌ No SARIF file found. Looked in: {Path.cwd()}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else input_path

    print(f"📄 Reading: {input_path}")
    with open(input_path, encoding="utf-8") as fh:
        data = json.load(fh)

    results_before = sum(len(run.get("results", [])) for run in data.get("runs", []) if isinstance(run, dict))
    data = fix_sarif(data)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    print(f"✅ Fixed {results_before} results → {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

