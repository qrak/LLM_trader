# Features in detail

The README keeps one line per feature; this page is the full description, written as
implementation notes. Every number here is checkable in the file named next to it.

## Brain with memory

`src/trading/brain.py`, `src/trading/vector_memory.py`, `src/trading/vector_memory_rules.py`

ChromaDB stores closed-trade experiences, semantic rules and blocked-trade friction. Before
each decision the bot retrieves — and injects into the prompt — a bounded slice:

- the **3 nearest past trades**,
- win rate and average P&L over the **20 nearest trades**,
- the **3 highest-ranked learned rules** (ranked by a blend, see below),
- up to **5 recent post-mortem lessons**,
- the confidence-calibration and direction-bias summaries, the active risk profile, the
  per-regime performance block and up to **5 recent blocked-entry rejections** with their
  thesis text.

No block grows with the size of the store; nothing else from the database is injected.

Closed trades are embedded locally with `BAAI/bge-base-en-v1.5` (768 dimensions), so no
embedding API is called and no trade data leaves the machine.

The historical MAE reference is the largest observed drawdown of at least five winning trades
with valid drawdown metadata; its sample count travels with it. It is not a guaranteed safe
limit, not an automatic SL floor and not a reason to widen a stop. Missing or invalid
observations never count as a zero drawdown.

Prompt size is bounded by `[prompt_budget]`: 100,000 estimated tokens per request, a 50,000
token brain context and a 10% model-window safety margin, counted with the shared `TokenCounter`
(`cl100k_base` estimates). Oversized contexts drop whole
lower-priority blocks while keeping the current position, risk instructions, response format and
recent lessons; if protected material still cannot fit, no request is sent.

## Raw OHLCV in the prompt

`src/analyzer/formatters/ohlcv_formatter.py`

Every prompt carries the analysis frame, the daily frame and the weekly frame as a plain CSV block
(`time,open,high,low,close,volume`, oldest first, UTC). The analysis frame and the daily frame are
capped at **183 closed candles** — 30.5 days at 4h and 183 days at 1d, which is exactly the `30D` and
`180D` rows of `## Period Metrics`, so the deepest range of each frame stays checkable against the
rows themselves. The weekly frame is capped at **104 closed candles (two years)**, the shortest
window that still holds the 2025-10 all-time high; everything older is already stated as text by the
200W SMA block and the 365D macro. The daily table is skipped when the analysis frame is daily.
Indicators still run on the full 999/365/300 candles; only the table is capped.

Why it is there: on 2026-11-22 an 8-cycle reading test asked the model for the last close, the
180-day high and low, the position inside that range and the 30-candle direction. With the numeric
prompt alone it scored **47%** of fields and **0/8** on the 180-day high; with the tables it scored
**92%** and **8/8**. That test measured whether the model can answer from data it was given, not
whether any decision rule needs it — no rule does, and a 13-cycle decision ablation found the tables
change the signal no more often than sending the same prompt twice. They are a hedge against
questions nobody precomputed, not a fix for a missing input.

The tables cost ~16.5k prompt tokens per cycle (~$0.0025 off-peak) and are the **first** section the
prompt budget drops when a request would not fit (`PromptBudgetPolicy.OPTIONAL_SECTIONS`).

Console and dashboard show the section as `<truncated>` (`truncate_ohlcv_section`): the model gets
every row, the human gets the rest of the prompt. `logs/Bot/<date>/Bot.log` keeps the full text.

## Long-term range in Period Metrics

`src/analyzer/market_metrics_calculator.py`

`## Period Metrics` carries a `180D` row built from the daily candles, not from the analysis frame:
999 4h candles cover only ~166 days, so a 4h slice could not honestly be labelled 180 days. The row
is labelled by its real coverage (`90D` on a token with 90 days of history) and is computed in
Python, so it survives the prompt budget dropping the raw tables. Every period row also carries
`RangePos:NN%` — where the last close sits between that period's low and high.

## Swing structure in the prompt

`src/analyzer/pattern_engine/indicator_patterns/structure_patterns.py`

With no picture in the prompt, "lower highs" has to arrive as text. The engine now reports a
`structure` category from swing pivots (a pivot is a bar that is the extreme of the ±5 bars around
it): `structure_uptrend` (higher high and higher low), `structure_downtrend` (lower high and lower
low), `structure_range` (the two disagree), `break_of_structure_bullish` / `_bearish` (the last close
beyond the most recent pivot) and `double_top` / `double_bottom` (two pivots within half a percent). Each line
carries the levels it compared, so the model can cite them.

Candlestick patterns were measured and **not** shipped. `scripts/pattern_edge_report.py` replays
8,108 BTC/USDC 4h candles through the TA-Lib-defined detectors and compares each pattern's forward
6/12/24-bar hit rate with the unconditional base rate (one-sided binomial test, Bonferroni over 13
patterns × 3 horizons, threshold p < 0.00128). No pattern passed: the best was `hanging_man` at 6 bars
(p = 0.0102, edge +8.7 pp). The detectors and the script stay in the repository as the evidence, and
nothing in the prompt claims a candlestick edge.

## Chart for humans, numbers for the model

`src/analyzer/pattern_engine/chart_generator.py`

The bot renders a **1920×1080 PNG** candlestick chart with indicators for the dashboard and the
Discord post. The model never receives it: every request is text, carrying the computed indicator
values (65 Numba-compiled kernels live in `src/indicators`), the price action and the pattern list.

That is a measured decision, not a preference. On 2026-11-22 a 16-cycle, four-arm ablation replayed
the logged cycle prompts text-only twice and with the chart. The same prompt sent twice changed the
signal in 5 of 16 cycles; adding the chart changed it in 3 (old render) and 5 (current render), with
a smaller mean confidence gap than the repeat. A separate 8-point reading test showed the model does
read the image accurately (91% of fields, price to the cent), so the image was redundant rather than
unreadable: its "P1–P4" citations restated numbers already in the numeric block.

The render runs as its own `asyncio` task, started before the model call and awaited after it, so its
~12 s overlaps the analysis instead of delaying it. Export runs in a thread with a hard timeout and
**three attempts (two retries)**; a failed render keeps the previous dashboard image and never
affects the cycle.

## Reflection engine

`src/trading/brain.py`, `src/trading/post_mortem.py`

After every *N* closed trades the system synthesises best-practice, anti-pattern and corrective
rules annotated with a **surprise ratio** — `|realized − expected| / max(|expected|, 0.01)`, where
expected is the take-profit distance — so the model can discount lucky or unlucky outcomes. Rules persist in vector memory and are ranked by a weighted
blend — semantic similarity 55%, evidence 25%, timeframe-aware freshness 15% minus a
contradiction penalty 5% — so a stale rule can lose its slot to a fresher, slightly less similar
one.

## Claim validation

`src/analyzer/trend_validator.py`, `src/analyzer/pattern_quality_scorer.py`, `src/trading/rr_policy.py`

Model claims are cross-checked against computed indicators: reported trend strength is compared
with the actual ADX, and pattern quality is replaced by a deterministic scorer — the corrected
score is written both into the returned analysis and into the `confluence_factors` entry the
closed trade teaches the brain. An **expected-value framework** — fees, the position cap, current
capital and the rolling inference cost — is injected into every prompt as a decision rule, next to
the closed-trade performance statistics already in the context, and the entry gate enforces the
stricter of the configured R/R floor and the brain's learned threshold.

`resolve_entry_rr_floor(config, thresholds)` in `rr_policy.py` is the single policy for that
floor; the brain can only raise it, never lower it.

## RAG news engine

`src/rag/`

Free RSS feeds (CoinDesk, CoinTelegraph, Decrypt, CryptoSlate) parsed with `feedparser` and
`defusedxml`, enriched over plain HTTP with trafilatura text extraction, plus market
data from CoinGecko, DeFiLlama and Alternative.me. Relevance scoring runs locally; the news
block is capped by the prompt budget like every other section.

## Live dashboard

`src/dashboard/server.py`, `src/dashboard/static/`

FastAPI + WebSocket SPA served on `0.0.0.0:8000` (or the public instance at
[semanticsignal.qrak.org](https://semanticsignal.qrak.org)) with eight tabs: Overview, Brain,
Console, Prompt, Response, Stats, News and Market.

The colour system is one place: every colour in `static/css/base.css` is a custom property
(`--accent-primary` for anything interactive, `--accent-success/-warning/-danger` for meaning,
neutrals for everything else). A panel never hardcodes a hue, and headings are sentence case; the
all-caps micro-label is reserved for values under 12 px.

The Console tab streams the bot log over a WebSocket with an all/error/warning/info level filter,
a find bar (Ctrl+F, F3, Enter/Shift+Enter) that highlights matches in place, counts them and steps
through them without hiding any line, auto-scroll paused while a query is active, and a Clear button
that asks before wiping the view while the stream keeps running
(`src/dashboard/static/modules/console_panel.js`).

The Overview also carries a **lifecycle strip** fed by `GET /api/status/state`: the current analysis
cycle (including `skipped` with its reason), the derived position phase (`flat`, `entering`,
`closing`, `divergence`, `open_verified`/`open_reported`/`open_unverified`) and the execution proof.
The position card shows the proof line and a notice while an entry or a close is unproven. Labels
and severity come from the API, so the UI and the trading prompt use the same vocabulary;
`docs/STATE_MACHINES.md` maps the evidence hierarchy.

The Brain tab holds every decision input. **Last decision** is a verdict line (signal, confidence,
trend, position) plus the market context the model saw (classifier chips and ADX/RSI) from
`GET /api/brain/decision-summary`. **Memory** lists the closed trades the bot remembers, ranked by
how many entry conditions match the market the last decision saw; five rows by default, a button
opens the whole store with sortable columns and expandable stored context, and the win-rate
breakdown by confidence, ADX, sentiment and volatility sits behind one disclosure. **Lessons** is
the post-mortem archive from
`GET /api/brain/post-mortems` — one card per closed trade (verdict, symbol, P&L, close reason, the
lesson and a full-analysis disclosure) with an FTS5 search box; the newest five are the ones the
prompt carries. Below it sit **Active rules** (the learned rule cards from `/api/brain/rules`, each
with the direction, regime and ADX bucket it was learned in, and a `scope mismatch` chip when the
live snapshot falls outside that scope), **Blocked attempts** (guard, direction, R:R, when) and the
**analysis chart** — the price-plus-indicators image rendered for humans; the model reads numbers, not this picture. The dashboard has per-client
rate limits and a loopback-only trust rule for `CF-Connecting-IP`.

The Overview's **Performance Metrics** panel plots the gain since the first recorded point **in
percent**: a $10k account swinging ±half a percent is invisible on a currency axis, so the same curve read as
dollars looked flat while exaggerating itself on an auto-scaled Y axis. Every trade opened is a
hollow dot and every close a filled one coloured by whether it moved the account up or down (grey
when it closed flat), with a one-line legend under the chart. The hover card names its numbers: an
entry reads `OPEN_LONG`/`OPEN_SHORT` (the position the bot booked, whichever signal the model sent),
a close keeps `CLOSE_LONG`/`CLOSE_SHORT`, the big percentage is the account gain since the first
trade, and a close adds its own account change (`0%` on a grey dot). A **7D / 30D / All** control
replaces the chart-library toolbar. The 30 s poll only re-renders the series when the fetched points
changed, so a chosen range survives the refresh. On phones (≤768 px) the KPI grid is 2×2, the
experience table renders as one card per row with labelled fields instead of a sideways scroll, and
the console keeps its day buttons on a single scrolling strip with 36 px tap targets.

## Local settings desk

`src/dashboard/static/admin/`, `src/dashboard/routers/settings.py`, `src/config/settings_editor.py`,
`src/dashboard/docs.py`

`/admin/` is the operator's localhost desk. The **Settings** tab lists every key present in
`config.ini`, with the file's comment block as help and loader-validated enums as selects; the
candidate is checked with `Config(config_path=...)` and written atomically, preserving comments,
order and line endings. The **API keys** tab shows only `set`/`source` for `keys.env` and never
returns a stored value. The **Documentation** tab renders the allowlisted Markdown pages with
`markdown-it-py` and sanitizes the HTML before it reaches the page. Reads and writes of the two
files require a loopback client, a local `Host` and a local `Origin`; tunnel and LAN requests get
`403`. No login, session or hot reload: every save is `restart`-scoped. Full contract:
[SETTINGS_PANEL.md](SETTINGS_PANEL.md).

## Risk pipeline

`src/trading/guards/`, `src/managers/risk_manager.py`, `src/trading/rr_policy.py`

Pre-execution guard pipeline (configured symbol, position-size cap), an entry R/R floor resolved
by `rr_policy.py` (the stricter of `min_rr_entry` and the brain's learned threshold) and dynamic
SL/TP scaling. `min_rr_entry = 0.0` starts without a fixed R/R floor; the brain can raise it
after enough losing low-R/R trades. Soft exits at candle close, hard exits polled at a
configurable interval against the live ticker.

When the executor is live, the stop that actually runs is the bot's own check against the **real
market** (`reference_exit_check_seconds`, default 10 s); the exchange leg is a catastrophe
backstop at 2.5× the stop distance. Once price covers `sl_breakeven_progress` (default 0.5) of the
entry-to-TP distance the bot moves the stop to the entry itself and never widens it back, whichever
decision the model sends. The floor is evaluated only during a successful analysis cycle, against the
ticker price fetched immediately before the decision, so an analysis that failed or a price that could
not be refreshed leaves the active stop alone. A worse proposal is clamped to the floor, and a stop
tighter than the floor still has to pass the tightening policy — if that policy rejects it, the floor
is applied anyway. `0` disables the automatic move but not the ban on widening a stop that already
sits at or beyond the entry. A forced move is not silent: the next prompt's
`### Last Action Outcome` says the bot, not the analysis, set the level, and the
decision history carries the same sentence.

## Decisions gate

`src/analyzer/decision_gate.py`, `scripts/decision_gate_report.py`

A second, independent model scores the decision after the Python validators and before the response
is rendered: a `score` for the entry location, a `choice` for the next three candles and a `noul`
expectancy bit. Only the score is spent, and only on size — 0–1 halves the position, 2 takes three
quarters, 3 leaves it alone, a disagreeing direction multiplies by 0.75 again, the floor is 0.5 and
nothing ever grows. `mode = shadow` makes the same call and writes the journal without touching the
decision, which is how the gate earns the right to move size. It never removes an entry: the bot
trades as often as before, at a smaller size when the second opinion disagrees.

## Provider routing and retries

`src/managers/provider_orchestrator.py`, `src/platforms/ai_providers/`, `src/utils/decorators.py`

One provider serves every request, chosen by `provider`: Google Gemini, DeepSeek, OpenRouter,
OpenCode Go, Command Code, or `custom`. Every provider except Google runs on one
`OpenAICompatibleClient` (OpenRouter and DeepSeek subclass it for their cost lookup and JSON-mode
retry); `custom` uses it too, or `AnthropicCompatibleClient` with `custom_protocol = anthropic`, so a
new OpenAI-style provider is a base URL plus a model in the config. OpenCode Go and Command Code are
flat subscriptions whose per-model monthly allowances are tracked as notional cost. There is no
fallback to another provider or model: a provider that does not answer is retried 5 times with a
growing wait whose last attempt lands about 15 minutes before the next candle, and a cycle that still
has no answer becomes HOLD rather than an exception.

## Read-only MCP server

`src/mcp_tools/history_server.py`, `requirements-mcp.txt` (optional)

An MCP server over the bot's own ledger for agent clients. `recent` returns the newest
`trade_history` rows (timestamp, symbol, action, prices, size, fee), `stats` returns decision counts
per action for a window plus its coverage. Both open the database with `mode=ro` and
`PRAGMA query_only`, so a write fails inside SQLite; there is no order, config or write tool, and
`limit` is capped at 200 rows. The bot never imports it, so `pip install -r requirements-mcp.txt` is
needed only where a client runs. Tool contract and client JSON: [MCP.md](MCP.md).

## Regression tests

`tests/` — the collected count and the measured duration live in [`claims.json`](../claims.json),
so this page never repeats them. No test in this tree carries the `integration` marker, so the CI
command's `-m "not integration"` filters nothing; the MCP history test skips when the optional
`mcp` package from `requirements-mcp.txt` is not installed. The suite covers LLM output corruption,
async races, rate-limit backoff, closed-loop feedback and spot/futures wiring. Real temporary
SQLite and ChromaDB databases are exercised, without live model or exchange calls.

## Layered agent documentation

`AGENTS.md` plus one `AGENTS.md` per package (8 files): root holds the shared policy, each
package file holds what only that package knows. A coding model reads one package instead of the
whole manual.

## Key files

| Path | Role |
|------|------|
| `start.py` | Entry point — 9-stage dependency provisioning, ChromaDB, CoinGecko cache |
| `src/app.py` | Main async loop, ticker fetch, analysis orchestration, unconfirmed-command alerts |
| `src/trading/brain.py` | `TradingBrainService` — context assembly, experience recording, reflection triggers |
| `src/trading/vector_memory.py` | ChromaDB interface — trade experiences, semantic rules, blocked trades, embedding cache |
| `src/trading/vector_memory_rules.py` | `VectorMemoryRulesMixin` — rule lifecycle: decay scoring, evidence ranking, surprise ratio |
| `src/trading/executor_handler.py` | Forwards decisions to the executor over HTTP, JSON file fallback, dead-letter queue |
| `src/trading/executor_reconciliation.py` | Compares bot state with the executor's, resolves positions and late receipts |
| `src/analyzer/analysis_engine.py` | Market analysis orchestration — indicators, RAG, LLM call, dashboard chart |
| `src/managers/provider_orchestrator.py` | Single-provider routing, prompt budget and response validation |
| `src/managers/risk_manager.py` | Dynamic SL/TP, position sizing, friction tracking |
| `src/managers/post_mortem_repository.py` | Stores model-written post-mortems; the brain includes up to five recent ones |
| `src/trading/trading_strategy.py` | Position lifecycle, guard enforcement, exit monitoring |
| `src/analyzer/prompts/template_manager.py` | System prompt construction with a falsification-based invalidation step |
| `src/analyzer/trend_validator.py` | Cross-checks model-reported trend strength against the computed ADX |
| `src/analyzer/pattern_quality_scorer.py` | Deterministic pattern quality scoring replacing the model's self-reported score |
| `src/notifiers/notifier.py` | Discord notifications with message-expiration tracking |
| `src/config/settings_editor.py` | `SettingsEditor` — comment-preserving INI/env edits, etag, loader validation, atomic writes |
| `src/dashboard/routers/settings.py` | `/api/settings/*` — loopback guard, config/keys reads and writes |
| `src/dashboard/docs.py` | `DocsLibrary` — allowlisted Markdown rendering and link/asset rewriting |
| `src/dashboard/static/admin/` | The `/admin/` page: Settings, API keys and Documentation |
