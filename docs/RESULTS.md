# Results — paper trading ledger

Everything on this page is read from the bot's own runtime files in `data/trading/`
(`statistics.json`, `api_costs.json`, `trade_history.db`, `metrics/cycles.jsonl`) plus one
public Binance klines request for the buy-and-hold benchmark. Nothing here is a hand-written
number: `python scripts/results_report.py` prints the blocks below, so a reader can re-run it
on the same checkout.

## Window

| item | value |
|---|---|
| Ledger window | 2026-09-20 → 2026-10-26 (36 days) |
| Instrument | BTC/USDC, 4h candles |
| Mode | Paper book (`demo_quote_capital = 10000` USDC), decisions forwarded to a testnet executor |
| Closed trades | 12 |
| Analysis cycles | 48 (recorded 2026-10-19 → 2026-10-26) |

## Ledger

| metric | value |
|---|---|
| Closed trades | 12 |
| Winners / losers | 5 / 5 |
| Win rate | 41.67% |
| P&L | +0.14% (+13.95 USDC on 10 000) |
| Profit factor | 1.41 |
| Sharpe | 0.19 |
| Sortino | 0.36 |
| Max drawdown | −0.25% |
| Best trade | +4.06% |
| Worst trade | −2.60% |

## Inference cost

| item | value |
|---|---|
| Total | $6.00 |
| Google (Gemini, chart vision) | $3.88 |
| DeepSeek (official API) | $2.12 |
| Input / output tokens | 8 073 341 / 3 834 551 |
| Net after inference | +$7.95 |

Per analysis cycle (48 recorded cycles, 2026-10-19 → 2026-10-26): mean **$0.0121**, median
latency **266 s** — one full analysis includes a 1920×1080 chart render, RAG news, brain
retrieval and a multimodal LLM call.

## Closed trades

| closed | direction | P&L % | close reason |
|---|---|---|---|
| 2026-09-24 | LONG | −2.60 | closed |
| 2026-09-24 | LONG | −2.06 | stop loss |
| 2026-09-28 | LONG | −0.28 | closed |
| 2026-10-02 | LONG | −0.99 | closed |
| 2026-10-05 | LONG | −1.12 | stop loss |
| 2026-10-10 | SHORT | +3.45 | take profit |
| 2026-10-12 | SHORT | +2.04 | take profit |
| 2026-10-14 | SHORT | +0.07 | stop loss |
| 2026-10-15 | SHORT | — | excluded (sandbox fill divergence) |
| 2026-10-17 | SHORT | — | excluded (sandbox fill divergence) |
| 2026-10-26 | SHORT | +1.93 | take profit |

## Buy-and-hold benchmark

| item | value |
|---|---|
| BTC/USDC | 81 245.48 → 79 758.34 |
| Buy-and-hold, same window | **−1.83%** |

The bot finished +0.14% while the underlying fell 1.83%. That is a ~2 percentage point
difference over five weeks — not evidence of an edge.

## Limitations

- **The sample is tiny.** Twelve closed trades cannot separate skill from noise.
- **Recorded fees are zero.** The paper book stores `fee = 0.0` on every row. At a 0.15%
  round-trip fee on each of the 12 trades (8% of capital per position, ~$1.20 per trade) the
  +$13.95 gain would mostly disappear. The ledger is gross of trading costs.
- **Two trades are excluded on purpose.** Both sandbox exits printed below the real market low
  in the same minute (exit fill divergence), so they are kept in execution history but dropped
  from statistics and from the brain's learning — see `src/trading/post_mortem.py` and the
  venue-truth rule in `AGENTS.md`.
- **Profitability is unproven.** This is a paper book. The executor can reach a testnet, not
  live funds.

## Reproduce

```bash
python scripts/results_report.py
```

Recomputes every table above from `data/trading/` and one Binance request. Costs are the
provider rate-table estimates the bot records (`config/model_pricing.json`,
`config/peak_rates.json`), not invoice reconciliation.
