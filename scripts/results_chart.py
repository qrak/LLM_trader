"""Render ``results-vs-btc.png`` for the showcase site from the same runtime files.

The results page quotes the ledger next to a picture, so the picture is generated from the
same inputs as ``results_report.py`` (statistics.json + one public Binance klines request)
instead of being drawn once by hand and left to rot.

Rendering uses the Playwright Chromium build that the integration tests already require
(``playwright install chromium``); headless Chrome's ``--screenshot`` stopped producing
files on this machine.

Usage:
    python scripts/results_chart.py --out ../semantic-signal-site/public/results-vs-btc.png
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from results_report import benchmark, load_json, load_period

HTML = """<!doctype html>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; width:1200px; height:520px; background:#0b0f19; color:#e5e7eb;
         font-family:'Inter',system-ui,sans-serif; padding:30px 44px; }}
  h1 {{ font-size:24px; margin:0; font-weight:700; }}
  .sub {{ color:#94a3b8; font-size:15px; margin:6px 0 0; }}
  .chart {{ margin-top:44px; display:flex; flex-direction:column; gap:26px; }}
  .row {{ display:flex; align-items:center; gap:18px; }}
  .name {{ width:170px; text-align:right; font-size:16px; color:#e5e7eb; }}
  .name.muted {{ color:#94a3b8; }}
  .track {{ flex:1; height:46px; border-left:2px solid #64748b; background:
            linear-gradient(to bottom, transparent 22px, #1f2937 22px, #1f2937 24px, transparent 24px); }}
  .bar {{ height:46px; border-radius:0 4px 4px 0; }}
  .bot {{ background:#10b981; }}
  .btc {{ background:#ef4444; }}
  .val {{ font-size:24px; font-weight:700; width:150px; }}
  .up {{ color:#10b981; }}
  .down {{ color:#ef4444; }}
  .foot {{ margin-top:52px; color:#94a3b8; font-size:14px; }}
</style>
<h1>Live paper trading, {window} ({pair})</h1>
<p class="sub">bot result vs buy-and-hold, same window</p>
<div class="chart">
  <div class="row">
    <span class="name">Semantic Signal</span>
    <div class="track"><div class="bar bot" style="width:{bot_w}px"></div></div>
    <span class="val {bot_class}">{bot}%</span>
  </div>
  <div class="row">
    <span class="name muted">BTC buy &amp; hold</span>
    <div class="track"><div class="bar btc" style="width:{btc_w}px"></div></div>
    <span class="val {btc_class}">{btc}%</span>
  </div>
</div>
<p class="foot">{trades} closed trades · {win:.1f}% win rate · max drawdown {dd:.2f}% ·
BTC {start:,.0f} → {end:,.0f} (buy-and-hold {btc_pct}%)</p>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/trading", type=Path)
    parser.add_argument("--symbol", default="BTCUSDC")
    parser.add_argument("--out", default="docs/assets/results-vs-btc.png", type=Path)
    args = parser.parse_args()

    stats = load_json(args.data_dir / "statistics.json")
    first_day, last_day = load_period(args.data_dir)
    if not first_day or not last_day:
        print("no trades in trade_history.db")
        return 1

    first = datetime.fromisoformat(first_day).replace(tzinfo=timezone.utc)
    last = datetime.fromisoformat(last_day).replace(tzinfo=timezone.utc)
    bench = benchmark(args.symbol, first, last)
    if "error" in bench:
        print(f"benchmark failed: {bench['error']}")
        return 1

    bot = float(stats.get("total_pnl_pct", 0.0))
    btc = float(bench["change_pct"])
    scale = max(abs(bot), abs(btc)) or 1.0
    max_width = 700.0
    bot_w = max(6.0, abs(bot) / scale * max_width)
    btc_w = max(6.0, abs(btc) / scale * max_width)

    html = HTML.format(
        window=f"{first:%d %b} – {last:%d %b %Y}",
        pair=args.symbol.replace("USDC", "/USDC"),
        bot_w=bot_w,
        btc_w=btc_w,
        bot=f"{bot:+.2f}",
        btc=f"{btc:+.2f}",
        btc_pct=f"{btc:+.2f}",
        bot_class="up" if bot >= 0 else "down",
        btc_class="down" if btc < 0 else "up",
        trades=int(stats.get("total_trades", 0)),
        win=float(stats.get("win_rate", 0.0)),
        dd=float(stats.get("max_drawdown_pct", 0.0)),
        start=bench["start"],
        end=bench["end"],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 520})
        page.set_content(html, wait_until="load")
        page.screenshot(path=str(args.out.resolve()), full_page=False)
        browser.close()

    print(f"{args.out}: bot {bot:+.2f}% vs buy-and-hold {btc:+.2f}% ({first:%d %b} – {last:%d %b %Y})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
