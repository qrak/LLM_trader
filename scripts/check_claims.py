from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_FILES = (
    "README.md",
    "docs/FEATURES.md",
    "docs/SETUP.md",
    "docs/CONFIGURATION.md",
    "CONTRIBUTING.md",
)
PCT_FIELDS = {
    "return_pct",
    "win_rate_pct",
    "max_drawdown_pct",
    "best_trade_pct",
    "worst_trade_pct",
}
USD_FIELDS = {"return_usd", "inference_usd", "net_after_inference_usd"}
DECIMAL_FIELDS = {"profit_factor", "sharpe", "sortino"}
ROW_LABELS = {"Closed trades": "closed_trades", "Analyses": "cycles"}
COLLECTED_RE = re.compile(r"([\d,]{3,})\s+collected")
RUN_RE = re.compile(r"([\d,]{3,})\s+(?:run|runnable)")
BADGE_RE = re.compile(r"tests-([\d%%,A-Fa-f]+)-brightgreen")
TESTS_TOTAL_RE = re.compile(r"TESTS_TOTAL\s*=\s*'([\d,]+)'")
TESTS_PROSE_RE = re.compile(r"([\d,]{3,})\s+automated\s+tests")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def number(text: str) -> int:
    return int(text.replace(",", ""))


def superseded_tokens(claims: dict) -> list[str]:
    tokens: list[str] = []
    for entry in claims.get("superseded", []):
        field = entry.get("field")
        value = entry.get("value")
        if value is None:
            continue
        if field in PCT_FIELDS:
            tokens += [f"{value:+.2f}%", f"{value:.2f}%", f"{value:.1f}%"]
        elif field in USD_FIELDS:
            tokens.append(f"${value:,.2f}")
        elif field in DECIMAL_FIELDS:
            tokens.append(f"{value:.2f}")
        elif field == "closed_trades":
            tokens.append(f"{value} closed trades")
        elif field == "cycles":
            tokens += [f"{value} cycles", f"{value} analyses"]
        elif field == "collected_tests":
            tokens.append(f"{value:,} collected")
        elif field == "runnable_tests":
            tokens.append(f"{value:,} runnable")
    unique: list[str] = []
    for token in tokens:
        if token not in unique:
            unique.append(token)
    return unique


def current_numbers(claims: dict) -> list[str]:
    tests = claims.get("tests", {})
    tokens: list[str] = []
    if isinstance(tests.get("collected"), int):
        tokens.append(f"{tests['collected']:,}")
    if isinstance(tests.get("runnable"), int):
        tokens.append(f"{tests['runnable']:,}")
    return tokens


def check_rows(body: str, claims: dict, relative: str) -> list[str]:
    problems: list[str] = []
    ledger = claims.get("ledger", {})
    for label, field in ROW_LABELS.items():
        expected = ledger.get(field)
        line = next((item for item in body.splitlines() if item.startswith(f"| {label} |")), None)
        if line is None or not isinstance(expected, int):
            continue
        parts = line.split("|")
        match = re.search(r"(\d[\d,]*)", parts[2]) if len(parts) > 2 else None
        if match is None:
            continue
        actual = number(match.group(1))
        if actual != expected:
            problems.append(f"{relative}: row {label!r} says {actual}, claims.json says {expected}")
    return problems


def check_texts(root: Path, claims: dict, files: tuple[str, ...]) -> list[str]:
    problems: list[str] = []
    stale = superseded_tokens(claims)
    required = current_numbers(claims)
    for relative in files:
        path = root / relative
        body = read_text(path)
        if not body:
            continue
        for token in stale:
            if token in body:
                problems.append(f"{relative}: stale value {token!r} — run scripts/claims_report.py and update the text")
        if relative == "README.md":
            problems += check_rows(body, claims, relative)
            for match in re.finditer(r"every (?:four|4) hours", body, re.IGNORECASE):
                line = body.count("\n", 0, match.start()) + 1
                problems.append(
                    f"{relative}:{line}: hardcoded four-hour cycle — the cadence is the configured `timeframe`"
                )
            for token in required:
                if token not in body:
                    problems.append(f"{relative}: missing current number {token!r}")
            for match in BADGE_RE.finditer(body):
                token = match.group(1).replace("%2C", ",").replace("%2c", ",")
                if token.isdigit() or "," in token:
                    actual = number(token)
                    expected = claims.get("tests", {}).get("collected")
                    if isinstance(expected, int) and actual != expected:
                        problems.append(f"{relative}: badge says {actual:,} collected, claims.json says {expected:,}")
            for match in COLLECTED_RE.finditer(body):
                actual = number(match.group(1))
                expected = claims.get("tests", {}).get("collected")
                if isinstance(expected, int) and actual != expected:
                    problems.append(f"{relative}: text says {actual:,} collected, claims.json says {expected:,}")
            for match in RUN_RE.finditer(body):
                actual = number(match.group(1))
                expected = claims.get("tests", {}).get("runnable")
                if isinstance(expected, int) and actual != expected:
                    problems.append(f"{relative}: text says {actual:,} runnable, claims.json says {expected:,}")
    return problems


def check_pytest(root: Path, claims: dict) -> list[str]:
    expected = claims.get("tests", {}).get("collected")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"(\d+)\s+tests? collected", completed.stdout)
    if not match:
        return ["pytest: could not read the collected count"]
    actual = int(match.group(1))
    if isinstance(expected, int) and actual != expected:
        return [f"pytest: {actual:,} tests collected, claims.json says {expected:,} — run scripts/claims_report.py"]
    return []


def check_git(root: Path, claims: dict) -> list[str]:
    expected = claims.get("repo", {}).get("commits")
    completed = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=str(root), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return ["git: could not count commits"]
    actual = int(completed.stdout.strip())
    if isinstance(expected, int) and actual != expected:
        return [f"git: {actual} commits on HEAD, claims.json says {expected} — run scripts/claims_report.py"]
    return []


def check_site(site: Path, claims: dict) -> list[str]:
    problems: list[str] = []
    project = read_json(site / "public" / "project.json")
    project_block = project.get("project", {})
    collected = claims.get("tests", {}).get("collected")
    committed_tests = project_block.get("tests", {}).get("collected")
    if isinstance(collected, int) and committed_tests != collected:
        problems.append(f"site project.json: tests {committed_tests}, claims.json says {collected}")
    commits = claims.get("repo", {}).get("commits")
    if isinstance(commits, int) and project_block.get("commits") != commits:
        problems.append(f"site project.json: commits {project_block.get('commits')}, claims.json says {commits}")
    if isinstance(collected, int):
        for name in ("index.astro", "story.astro", "about.astro"):
            body = read_text(site / "src" / "pages" / name)
            for match in TESTS_TOTAL_RE.finditer(body):
                if number(match.group(1)) != collected:
                    problems.append(f"site {name}: TESTS_TOTAL {match.group(1)}, claims.json says {collected:,}")
            for match in TESTS_PROSE_RE.finditer(body):
                if number(match.group(1)) != collected:
                    problems.append(f"site {name}: prose says {match.group(1)} tests, claims.json says {collected:,}")
    return problems


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Fail when a public text disagrees with claims.json.")
    parser.add_argument("--root", default=str(ROOT), type=Path)
    parser.add_argument("--claims", default=None, type=Path)
    parser.add_argument("--pytest", action="store_true", help="re-collect the test suite and compare")
    parser.add_argument("--git", action="store_true", help="compare the commit count with HEAD")
    parser.add_argument("--site-root", default=None, type=Path)
    parser.add_argument("--with-posts", action="store_true", help="also check the private .hermes/posts drafts")
    args = parser.parse_args()

    root: Path = args.root.resolve()
    claims_path = args.claims or root / "claims.json"
    claims = read_json(claims_path)
    if not claims:
        print(f"missing claims file: {claims_path}")
        return 1

    problems = check_texts(root, claims, TEXT_FILES)
    if args.with_posts:
        posts = sorted((root / ".hermes" / "posts").glob("*.md"))
        if posts:
            problems += check_texts(root, claims, tuple(str(p.relative_to(root)) for p in posts))
    if args.pytest:
        problems += check_pytest(root, claims)
    if args.git:
        problems += check_git(root, claims)
    if args.site_root:
        problems += check_site(args.site_root.resolve(), claims)

    if problems:
        print(f"{len(problems)} claim problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"claims consistent: {claims.get('tests', {}).get('collected'):,} tests, "
          f"ledger snapshot {claims.get('ledger', {}).get('snapshot_at')} ({claims_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
