from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from claims_report import build_claims, render_numbers
from ledger_facts import ledger_facts, load_json, load_trades
from snapshot_results import LedgerSnapshot

MARKER_START = "<!-- numbers:start -->"
MARKER_END = "<!-- numbers:end -->"
SITE_LEDGER = Path("src") / "data" / "ledger.json"
SITE_CLAIMS = Path("public") / "claims.json"
SITE_PROJECT = Path("public") / "project.json"
BLOCK_TEMPLATE = """{start}
**Window:** {window_start} → {window_end} ({window_days} days), {pair}, {timeframe} candles, {capital} USDC simulated capital.
**Source:** the bot's own ledger, frozen at [`results/{snapshot_at}/`](results/{snapshot_at}); `python scripts/results_report.py --data-dir results/{snapshot_at}` reprints every number below.

| metric | bot | buy & hold |
|---|---|---|
| Return | **{return_pct}** ({return_usd}) | **{benchmark_pct}** |
| Closed trades | {closed_trades} ({counted_trades} counted, {excluded_trades} excluded for fill divergence) | — |
| Win rate | {win_rate_pct}% ({winners}W / {losers}L) | — |
| Profit factor | {profit_factor} | — |
| Sharpe / Sortino | {sharpe} / {sortino} | — |
| Max drawdown | {max_drawdown_pct} | — |
| Best / worst trade | {best_trade_pct} / {worst_trade_pct} | — |
| Inference cost | {inference_usd} ({tokens_in} in / {tokens_out} out tokens) | — |
| Net after inference | {net_usd} | — |
| Analyses | {cycles} cycles, median {median_seconds} s, {cycle_usd} each | — |

![Bot vs buy-and-hold over the same window](docs/assets/results-vs-btc.png)

Over those {window_days} days the bot returned {return_pct} while the underlying fell {hold_fall}. That is
{gap_points} percentage points on {counted_words} counted trades — not evidence of an edge.
The paper book records zero fees, so the number is gross of trading costs; at a 0.15% round trip the
result would be clearly negative. The statistics exclude {excluded_words} trades because their
sandbox fills diverged from the real market. The numbers above come from that frozen snapshot,
not from the live ledger, so they do not move with every cycle the bot runs.
Method, per-trade table and the reproduction command:
`python scripts/results_report.py --data-dir results/{snapshot_at}`.
{end}"""
WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen",
    17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
}


class PublicNumbersPublisher:
    def __init__(self, root: Path, site: Path, dry_run: bool = False) -> None:
        self.root = root
        self.site = site
        self.dry_run = dry_run
        self.today = datetime.now(timezone.utc).date().isoformat()

    def run(self, force: bool = False) -> int:
        snapshot_dir = self.root / "results" / self.today
        if not self.dry_run and (problem := self._dirty()):
            print(f"abort: uncommitted changes in {problem} — commit or stash them first")
            return 1
        if snapshot_dir.exists() and not force:
            print(f"nothing to do: results/{self.today}/ already holds today's publication")
            return 0
        if self.dry_run:
            print(f"would snapshot into {snapshot_dir}")
            print("would run the suite, then regenerate claims.json, the chart, the README block and the site")
            print(f"would commit and push {self.root} and {self.site}, then rebuild and deploy the site")
            return 0

        passed, skipped, seconds = self._suite()
        self._snapshot(snapshot_dir, force)
        self._chart(snapshot_dir)
        claims = self._claims(snapshot_dir, seconds)
        self._readme(claims, snapshot_dir, ledger_facts(snapshot_dir, "BTCUSDC"))
        self._gates()
        commit = self._commit(self.root, ["README.md", "claims.json", "docs/assets/results-vs-btc.png",
                                          f"results/{self.today}"], f"chore(numbers): publish the ledger of {self.today}")
        self._sync_site(claims, snapshot_dir, commit, passed, skipped)
        self._gates(site=True)
        self._build_and_deploy()
        self._commit(self.site, ["public", "src"], f"chore(numbers): follow the bot's publication of {self.today}")
        ledger = claims["ledger"]
        print(f"published {self.today}: {passed} passed, {ledger['closed_trades']} closed trades, "
              f"{ledger['cycles']} cycles, return {ledger['return_pct']:+.2f}%")
        return 0

    def _dirty(self) -> str:
        for repo in (self.root, self.site):
            if self._git(["status", "--porcelain"], repo).strip():
                return str(repo)
        return ""

    def _suite(self) -> tuple[int, int, int]:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
            cwd=str(self.root), capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"test suite failed, nothing published:\n{completed.stdout[-3000:]}")
        match = re.search(r"(\d+) passed(?:, (\d+) skipped)?.*? in ([\d.]+)s", completed.stdout)
        if match is None:
            raise RuntimeError("could not read the suite summary")
        return int(match.group(1)), int(match.group(2) or 0), int(float(match.group(3)))

    def _snapshot(self, snapshot_dir: Path, force: bool) -> None:
        if force and snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        LedgerSnapshot(self.root / "data" / "trading", snapshot_dir).write()

    def _chart(self, snapshot_dir: Path) -> None:
        target = self.root / "docs" / "assets" / "results-vs-btc.png"
        completed = subprocess.run(
            [sys.executable, "scripts/results_chart.py", "--data-dir", str(snapshot_dir), "--out", str(target)],
            cwd=str(self.root), capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"chart render failed: {completed.stdout[-800:]}{completed.stderr[-800:]}")
        shutil.copy2(target, self.site / "public" / "results-vs-btc.png")

    def _claims(self, snapshot_dir: Path, seconds: int) -> dict:
        claims_path = self.root / "claims.json"
        previous = json.loads(claims_path.read_text(encoding="utf-8")) if claims_path.exists() else {}
        claims = build_claims(
            previous, snapshot_dir, measure_tests=True, with_benchmark=True, suite_seconds=seconds
        )
        claims_path.write_text(json.dumps(claims, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        numbers = self.root / ".hermes" / "posts" / "_current_numbers.md"
        if numbers.parent.exists():
            numbers.write_text(render_numbers(claims), encoding="utf-8")
        return claims

    def _readme(self, claims: dict, snapshot_dir: Path, facts: dict) -> None:
        path = self.root / "README.md"
        body = path.read_text(encoding="utf-8")
        if MARKER_START not in body or MARKER_END not in body:
            raise RuntimeError("README has no numbers markers")
        start = body.index(MARKER_START)
        end = body.index(MARKER_END) + len(MARKER_END)
        block = self._block(claims, facts, snapshot_dir)
        body = body[:start] + block + body[end:]
        path.write_text(self._refresh_test_facts(body, claims), encoding="utf-8")

    def _refresh_test_facts(self, body: str, claims: dict) -> str:
        collected = claims["tests"]["collected"]
        seconds = claims["tests"]["suite_seconds"]
        badge = f"tests-{collected:,}-brightgreen".replace(",", "%2C")
        body = re.sub(r"tests-[0-9%2C,]+-brightgreen", badge, body)
        body = re.sub(r"\| Tests \| [\d,]+ collected", f"| Tests | {collected:,} collected", body)
        return re.sub(r"# [\d,]+ tests, ~\d+ s", f"# {collected:,} tests, ~{seconds} s", body)

    def _block(self, claims: dict, facts: dict, snapshot_dir: Path) -> str:
        ledger = claims["ledger"]
        statistics = load_json(snapshot_dir / "statistics.json")
        net = ledger["net_after_inference_usd"]
        return BLOCK_TEMPLATE.format(
            start=MARKER_START,
            end=MARKER_END,
            window_start=ledger["window_start"],
            window_end=ledger["window_end"],
            window_days=ledger["window_days"],
            pair=ledger["pair"],
            timeframe=ledger["timeframe"],
            capital=f"{int(ledger['capital_usdc']):,}".replace(",", " "),
            snapshot_at=ledger["snapshot_at"],
            return_pct=f"{ledger['return_pct']:+.2f}%",
            return_usd=f"{'+' if ledger['return_usd'] >= 0 else '-'}${abs(ledger['return_usd']):.2f}",
            benchmark_pct=f"{ledger['benchmark_pct']:+.2f}%",
            closed_trades=ledger["closed_trades"],
            counted_trades=ledger["counted_trades"],
            excluded_trades=ledger["excluded_trades"],
            win_rate_pct=f"{ledger['win_rate_pct']:.1f}",
            winners=int(statistics.get("winning_trades", 0)),
            losers=int(statistics.get("losing_trades", 0)),
            profit_factor=f"{ledger['profit_factor']:.2f}",
            sharpe=f"{ledger['sharpe']:.2f}",
            sortino=f"{ledger['sortino']:.2f}",
            max_drawdown_pct=f"{ledger['max_drawdown_pct']:.2f}%",
            best_trade_pct=f"{ledger['best_trade_pct']:+.2f}%",
            worst_trade_pct=f"{ledger['worst_trade_pct']:+.2f}%",
            inference_usd=f"${ledger['inference_usd']:.2f}",
            tokens_in=f"{facts['tokens_in'] / 10 ** 6:.1f}M",
            tokens_out=f"{facts['tokens_out'] / 10 ** 6:.1f}M",
            net_usd=f"-${abs(net):.2f}" if net < 0 else f"+${net:.2f}",
            cycles=ledger["cycles"],
            median_seconds=round(ledger["cycle_latency_median_s"]),
            cycle_usd=f"${ledger['cycle_cost_usd']:.4f}",
            hold_fall=f"{abs(ledger['benchmark_pct']):.2f}%",
            gap_points=f"{abs(ledger['return_pct'] - ledger['benchmark_pct']):.1f}",
            counted_words=WORDS.get(ledger["counted_trades"], str(ledger["counted_trades"])),
            excluded_words=WORDS.get(ledger["excluded_trades"], str(ledger["excluded_trades"])),
        )

    def _sync_site(self, claims: dict, snapshot_dir: Path, commit: str, passed: int, skipped: int) -> None:
        ledger = ledger_facts(snapshot_dir, "BTCUSDC")
        statistics = load_json(snapshot_dir / "statistics.json")
        ledger["trades"] = load_trades(snapshot_dir)
        ledger["winners"] = int(statistics.get("winning_trades", 0))
        ledger["losers"] = int(statistics.get("losing_trades", 0))
        (self.site / SITE_LEDGER).write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        shutil.copy2(self.root / "claims.json", self.site / SITE_CLAIMS)
        shutil.copy2(self.root / "claims.json", self.site / SITE_LEDGER.with_name("claims.json"))
        project_path = self.site / SITE_PROJECT
        project = json.loads(project_path.read_text(encoding="utf-8"))
        tests = project["project"]["tests"]
        tests["collected"] = claims["tests"]["collected"]
        tests["passing"] = passed
        tests["skipped"] = skipped
        tests["duration_seconds"] = claims["tests"]["suite_seconds"]
        tests["measured"] = self.today
        tests["measured_at_commit"] = commit
        project["project"]["commits"] = claims["repo"]["commits"]
        project_path.write_text(json.dumps(project, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _gates(self, site: bool = False) -> None:
        commands = [
            [sys.executable, "scripts/check_claims.py", "--pytest"],
            [sys.executable, "scripts/check_agents_md.py"],
            [sys.executable, "-m", "ruff", "check", "src", "scripts", "start.py"],
        ]
        if site:
            commands.append([sys.executable, "scripts/check_claims.py", "--site-root", str(self.site)])
        for command in commands:
            completed = subprocess.run(command, cwd=str(self.root), capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"gate failed: {' '.join(command)}\n{completed.stdout[-2000:]}")

    def _build_and_deploy(self) -> None:
        for command in (
            ["npm", "run", "build"],
            ["npx", "--yes", "wrangler@4", "pages", "deploy", "dist",
             "--project-name", "semantic-signal-landing", "--branch", "main"],
        ):
            completed = subprocess.run(command, cwd=str(self.site), shell=True, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"{' '.join(command)} failed:\n{completed.stdout[-1500:]}{completed.stderr[-1500:]}")

    def _commit(self, repo: Path, paths: list[str], message: str) -> str:
        self._git(["add", *paths], repo)
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=str(repo), capture_output=True, text=True, check=False
        )
        if staged.returncode == 0:
            return self._git(["rev-parse", "HEAD"], repo).strip()
        self._git(["commit", "-m", message], repo)
        self._git(["push", "origin", "HEAD:master"], repo)
        return self._git(["rev-parse", "HEAD"], repo).strip()

    def _git(self, args: list[str], repo: Path) -> str:
        completed = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed in {repo}:\n{completed.stdout}{completed.stderr}")
        return completed.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Snapshot the ledger, regenerate every published number and publish it."
    )
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--site", default=None, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    site = (args.site or root.parent / "semantic-signal-site").resolve()
    try:
        return PublicNumbersPublisher(root, site, args.dry_run).run(force=args.force)
    except RuntimeError as error:
        print(f"publish aborted: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
