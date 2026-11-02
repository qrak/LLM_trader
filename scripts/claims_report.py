from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ledger_facts import ledger_facts

ROOT = Path(__file__).resolve().parents[1]
CLAIMS_PATH = ROOT / "claims.json"
NUMBERS_PATH = ROOT / ".hermes" / "posts" / "_current_numbers.md"
REPO_SLUG = "qrak/LLM_trader"

SUPERSEDED_FIELDS = (
    "return_pct",
    "return_usd",
    "win_rate_pct",
    "profit_factor",
    "sharpe",
    "sortino",
    "max_drawdown_pct",
    "best_trade_pct",
    "worst_trade_pct",
    "inference_usd",
    "net_after_inference_usd",
    "closed_trades",
    "cycles",
)


def run(args: list[str], cwd: Path = ROOT) -> str:
    completed = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, check=True)
    return completed.stdout.strip()


def git_commit_count() -> int | None:
    try:
        return int(run(["git", "rev-list", "--count", "HEAD"]))
    except (subprocess.CalledProcessError, ValueError):
        return None


def gh_json(path: str, jq: str) -> object | None:
    try:
        return json.loads(run(["gh", "api", path, "--jq", jq]))
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return None


def repo_facts(previous: dict) -> dict:
    payload = {
        "full_name": REPO_SLUG,
        "url": f"https://github.com/{REPO_SLUG}",
        "homepage": "https://semanticsignal.qrak.org",
        "created": None,
        "stars": None,
        "forks": None,
        "open_issues": None,
        "commits": git_commit_count(),
        "releases": None,
        "measured_at": datetime.now(timezone.utc).date().isoformat(),
    }
    info = gh_json(f"repos/{REPO_SLUG}", '{"created": .created_at, "stars": .stargazers_count, "forks": .forks_count, "issues": .open_issues_count}')
    if isinstance(info, dict):
        payload["created"] = str(info.get("created") or "")[:10] or None
        payload["stars"] = info.get("stars")
        payload["forks"] = info.get("forks")
        payload["open_issues"] = info.get("issues")
    releases = gh_json(f"repos/{REPO_SLUG}/releases", "length")
    if isinstance(releases, int):
        payload["releases"] = releases
    carried = {key: previous.get(key) for key in ("created", "stars", "forks", "open_issues", "releases")}
    for key, value in carried.items():
        if payload[key] is None and value is not None:
            payload[key] = value
    payload["source"] = "gh" if info is not None else "carried from previous claims.json"
    return payload


def pytest_collect(arguments: list[str]) -> int | None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", *arguments],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    selected = re.search(r"(\d+)\s*/\s*(\d+)\s+tests? collected", completed.stdout)
    if selected:
        return int(selected.group(1))
    collected = re.search(r"(\d+)\s+tests? collected", completed.stdout)
    return int(collected.group(1)) if collected else None


def test_facts(previous: dict, measure: bool, suite_seconds: int | None) -> dict:
    collected = pytest_collect([]) if measure else None
    runnable = pytest_collect(["-m", "not integration"]) if measure else None
    if collected is None:
        collected = previous.get("collected")
    if runnable is None:
        runnable = previous.get("runnable")
    deselected = collected - runnable if collected is not None and runnable is not None else previous.get("deselected")
    return {
        "collected": collected,
        "runnable": runnable,
        "deselected": deselected,
        "suite_seconds": suite_seconds or previous.get("suite_seconds"),
        "command_full": "python -m pytest tests -q",
        "command_ci": 'python -m pytest tests -q -m "not integration"',
        "measured_at": datetime.now(timezone.utc).date().isoformat() if measure else previous.get("measured_at"),
    }


def superseded(previous: dict, ledger: dict, tests: dict) -> list[dict]:
    entries: list[dict] = list(previous.get("superseded", []))
    old_ledger = previous.get("ledger", {})
    old_tests = previous.get("tests", {})
    seen = {(entry.get("field"), entry.get("value")) for entry in entries}
    candidates = [
        ("return_pct", old_ledger.get("return_pct"), ledger.get("return_pct")),
        ("return_usd", old_ledger.get("return_usd"), ledger.get("return_usd")),
        ("win_rate_pct", old_ledger.get("win_rate_pct"), ledger.get("win_rate_pct")),
        ("profit_factor", old_ledger.get("profit_factor"), ledger.get("profit_factor")),
        ("inference_usd", old_ledger.get("inference_usd"), ledger.get("inference_usd")),
        ("net_after_inference_usd", old_ledger.get("net_after_inference_usd"), ledger.get("net_after_inference_usd")),
        ("closed_trades", old_ledger.get("closed_trades"), ledger.get("closed_trades")),
        ("cycles", old_ledger.get("cycles"), ledger.get("cycles")),
        ("collected_tests", old_tests.get("collected"), tests.get("collected")),
        ("runnable_tests", old_tests.get("runnable"), tests.get("runnable")),
    ]
    for field, old, new in candidates:
        if old is None or new is None or old == new:
            continue
        key = (field, old)
        if key in seen:
            continue
        entries.append({"field": field, "value": old})
        seen.add(key)
    return entries


def ledger_block(previous: dict, data_dir: Path, with_benchmark: bool) -> dict:
    measured = ledger_facts(data_dir, with_benchmark=with_benchmark)
    if not measured.get("available"):
        return previous.get("ledger", {})
    old = previous.get("ledger", {})
    same_window = old.get("window_start") == measured.get("window_start")
    if measured.get("benchmark_pct") is None and same_window:
        for key in ("benchmark_pct", "benchmark_start", "benchmark_end", "benchmark_bars"):
            measured[key] = old.get(key)
    return measured


def build_claims(previous: dict, data_dir: Path, measure_tests: bool, with_benchmark: bool, suite_seconds: int | None) -> dict:
    tests = test_facts(previous.get("tests", {}), measure_tests, suite_seconds)
    ledger = ledger_block(previous, data_dir, with_benchmark)
    return {
        "schema": 1,
        "generated": datetime.now(timezone.utc).date().isoformat(),
        "repo": repo_facts(previous.get("repo", {})),
        "tests": tests,
        "ledger": ledger,
        "superseded": superseded(previous, ledger, tests),
        "sources": {
            "ledger": "python scripts/results_report.py",
            "tests": "python -m pytest tests --collect-only -q",
            "repo": "gh api repos/qrak/LLM_trader",
        },
    }


def signed_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%"


def usd(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def dollars(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.4f}"


def capital(value: float | None) -> str:
    if not value:
        return "n/a"
    return f"{int(value):,}"


def render_numbers(claims: dict) -> str:
    ledger = claims["ledger"]
    tests = claims["tests"]
    repo = claims["repo"]
    lines = [
        f"# Current, verified numbers ({claims['generated']})",
        "",
        "Generated by `python scripts/claims_report.py` — do not edit by hand.",
        "Everything here is reproducible from the repo; anything not listed is a story, not a measurement.",
        "",
        "## The repo",
        "",
        "| item | value | how to check |",
        "|---|---|---|",
        f"| Stars / forks | {repo.get('stars')} / {repo.get('forks')} | GitHub |",
        f"| Commits on master | {repo.get('commits')} | `git rev-list --count HEAD` |",
        f"| Releases | {repo.get('releases')} | GitHub → Releases |",
        (
            f"| Tests | **{tests.get('collected'):,} collected** — {tests.get('runnable'):,} runnable, "
            f"{tests.get('deselected')} deselected (`integration`) | `{tests.get('command_full')}` |"
        ),
        f"| Repo created | {repo.get('created')} | GitHub |",
        "",
        "## Paper trading (the only performance claim you can make)",
        "",
        "| item | value |",
        "|---|---|",
        (
            f"| Window | {ledger.get('window_start')} → {ledger.get('window_end')} ({ledger.get('window_days')} days), "
            f"{ledger.get('pair')}, {ledger.get('timeframe')} |"
        ),
        f"| Capital | {capital(ledger.get('capital_usdc'))} USDC simulated |",
        (
            f"| Closed trades | {ledger.get('closed_trades')} ({ledger.get('counted_trades')} counted, "
            f"{ledger.get('excluded_trades')} excluded for fill divergence) |"
        ),
        f"| Win rate | {ledger.get('win_rate_pct')}% |",
        f"| Return | **{signed_pct(ledger.get('return_pct'))}** ({usd(ledger.get('return_usd'))}) |",
        (
            f"| Buy & hold BTC | **{signed_pct(ledger.get('benchmark_pct'))}** "
            f"({ledger.get('benchmark_start')} → {ledger.get('benchmark_end')}) |"
        ),
        f"| Profit factor | {ledger.get('profit_factor')} |",
        f"| Sharpe / Sortino | {ledger.get('sharpe')} / {ledger.get('sortino')} |",
        f"| Max drawdown | {ledger.get('max_drawdown_pct')}% |",
        f"| Best / worst trade | {signed_pct(ledger.get('best_trade_pct'))} / {signed_pct(ledger.get('worst_trade_pct'))} |",
        (
            f"| Inference cost | **{usd(ledger.get('inference_usd'))}** "
            f"({ledger.get('tokens_in'):,} in / {ledger.get('tokens_out'):,} out tokens) |"
        ),
        (
            f"| Net after inference | **{signed_pct((ledger.get('net_after_inference_usd') or 0.0) / (ledger.get('capital_usdc') or 10000) * 100)}** "
            f"({usd(ledger.get('net_after_inference_usd'))}) |"
        ),
        (
            f"| Cycles | {ledger.get('cycles')}, median {ledger.get('cycle_latency_median_s')} s, "
            f"{dollars(ledger.get('cycle_cost_usd'))} each |"
        ),
        f"| Snapshot | {ledger.get('snapshot_at')} |",
        "",
        "Reproduce: `python scripts/results_report.py`.",
        "",
        "## Safe claims",
        "",
        "- \"79 Numba-compiled indicator functions\" (`@njit` count in `src/indicators`).",
        "- \"1920×1080 chart rendered per cycle and sent to a multimodal model\".",
        f"- \"{tests.get('collected'):,} collected tests, {tests.get('runnable'):,} runnable, ruff + pyright clean in CI\".",
        "- \"beats buy-and-hold **in this window**\" — always attach the window and the trade count.",
        "- \"inference cost is measured per cycle and enters the EV threshold\".",
        "",
        "## Claims to avoid",
        "",
        "- Any edge, profitability or \"it works\" claim. The sample is small and the paper book records zero fees.",
        "- Any month count of development. The repo was created 2025-02-28; the first commit is 2025-12-21.",
        "- Anything about the private executor beyond \"decisions may be forwarded to a separate, private service\".",
        "- \"fees included\" — the paper book records `fee = 0.0`; a 0.15% round trip would change the number.",
        "",
        "## Links",
        "",
        f"- Repo: {repo.get('url')}",
        f"- Dashboard: {repo.get('homepage')}",
        "- Results page: https://qrak.org/results/",
        "- Site: https://qrak.org",
        "- Changelog: https://github.com/qrak/LLM_trader/blob/master/CHANGELOG.md",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Write claims.json and the generated numbers sheet used by every public text.")
    parser.add_argument("--data-dir", default="data/trading", type=Path)
    parser.add_argument("--skip-tests", action="store_true", help="carry the previous test counts instead of re-collecting")
    parser.add_argument("--benchmark", action="store_true", help="fetch the buy-and-hold number from Binance")
    parser.add_argument("--claims", default=str(CLAIMS_PATH), type=Path)
    parser.add_argument("--numbers", default=str(NUMBERS_PATH), type=Path)
    parser.add_argument("--suite-seconds", type=int, default=None, help="duration of the last full pytest run")
    parser.add_argument("--print", dest="print_only", action="store_true")
    args = parser.parse_args()

    previous = json.loads(args.claims.read_text(encoding="utf-8")) if args.claims.exists() else {}
    claims = build_claims(previous, args.data_dir, not args.skip_tests, args.benchmark, args.suite_seconds)
    text = json.dumps(claims, indent=2, ensure_ascii=False) + "\n"
    numbers = render_numbers(claims)
    if args.print_only:
        print(text)
        return 0
    args.claims.write_text(text, encoding="utf-8")
    if args.numbers.parent.exists():
        args.numbers.write_text(numbers, encoding="utf-8")
    print(f"wrote {args.claims}")
    print(numbers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
