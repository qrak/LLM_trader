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
token brain context, a 10% model-window safety margin and a 16,384-token image reserve, counted
with the shared `TokenCounter` (`cl100k_base` estimates). Oversized contexts drop whole
lower-priority blocks while keeping the current position, risk instructions, response format and
recent lessons; if protected material still cannot fit, no request is sent.

## Vision chart analysis

`src/analyzer/pattern_engine/chart_generator.py`

The bot renders a **1920×1080 PNG** candlestick chart with indicators and sends it to a
multimodal model for visual pattern recognition. The chart is one input next to the numeric
indicator stack, not a replacement for it: the model gets 79 Numba-compiled indicator functions
as text and the picture as a second view of the same market.

Chart export runs in a thread with a hard timeout and **three attempts (two retries)**; a failed
render degrades to text-only analysis instead of failing the cycle.

## Reflection engine

`src/trading/brain.py`, `src/trading/post_mortem.py`

After every *N* closed trades the system synthesises best-practice, anti-pattern and corrective
rules annotated with a **surprise ratio** — `|realized − expected| / expected` — so the model can
discount lucky or unlucky outcomes. Rules persist in vector memory and are ranked by a weighted
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
`defusedxml`, optionally enriched through Crawl4AI with trafilatura text extraction, plus market
data from CoinGecko, DeFiLlama and Alternative.me. Relevance scoring runs locally; the news
block is capped by the prompt budget like every other section.

## Live dashboard

`src/dashboard/server.py`, `src/dashboard/static/`

FastAPI + WebSocket SPA served on `0.0.0.0:8000` (or the public instance at
[semanticsignal.qrak.org](https://semanticsignal.qrak.org)) with eleven tabs: Overview, Brain,
Console, Prompt, Response, Stats, News, Activity, Market, Memory and Journal.

The **GitHub Activity** tab echoes the account's public event stream in GitHub's own wording
(`qrak contributed to qrak/LLM_trader`), so it mirrors github.com instead of keeping a second,
differently worded activity log. `src/dashboard/github_activity.py` performs the single network
call and turns the events into rows (unknown event types are skipped, never rendered blank);
`GET /api/monitor/github_activity` caches the page for 5 minutes, serves the last good page as
`stale` for a day when GitHub is unreachable, and answers with `error` only when nothing is cached.
`GITHUB_USER` selects the account and `GITHUB_TOKEN` raises the unauthenticated 60 requests/hour to
5000; without `GITHUB_USER` the tab stays hidden behind a configuration hint.

The Brain tab draws the decision graph (memory hits, rules, post-mortems, blocked attempts) that
produced the current signal. The dashboard has per-client rate limits and a loopback-only trust
rule for `CF-Connecting-IP`.

## Risk pipeline

`src/trading/guards/`, `src/managers/risk_manager.py`, `src/trading/rr_policy.py`

Pre-execution guard pipeline (configured symbol, position-size cap), an entry R/R floor resolved
by `rr_policy.py` (the stricter of `min_rr_entry` and the brain's learned threshold) and dynamic
SL/TP scaling. `min_rr_entry = 0.0` starts without a fixed R/R floor; the brain can raise it
after enough losing low-R/R trades. Soft exits at candle close, hard exits polled at a
configurable interval against the live ticker.

When the executor is live, the stop that actually runs is the bot's own check against the **real
market** (`reference_exit_check_seconds`, default 10 s); the exchange leg is a catastrophe
backstop at 2.5× the stop distance.

## Multi-provider routing

`src/managers/provider_orchestrator.py`, `src/config/loader.py`

Primary: DeepSeek official API (`deepseek-flash`, accepts charts natively) — that is the
`provider = deepseek` deployment. `provider = all` walks the chain in order, Google Gemini →
DeepSeek → LM Studio → OpenRouter → BlockRun for text and Google → DeepSeek → OpenRouter →
BlockRun when a chart is attached; every provider that supports images gets the chart. A failed
provider degrades to the next one inside the same cycle; a total failure becomes HOLD rather than
an exception.

## Regression tests

`tests/` — 2,689 collected tests (2,655 run outside the `integration` marker; measured on
2026-10-28: ~126 s warm, ~167 s on a freshly installed environment), covering LLM output
corruption, async races, rate-limit backoff, closed-loop feedback and spot/futures wiring. Real
temporary SQLite and ChromaDB databases are exercised, without live model or exchange calls.

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
| `src/analyzer/analysis_engine.py` | Market analysis orchestration — indicators, chart, RAG, LLM call |
| `src/managers/provider_orchestrator.py` | AI provider fallback chain with retry logic and cost tracking |
| `src/managers/risk_manager.py` | Dynamic SL/TP, position sizing, friction tracking |
| `src/managers/post_mortem_repository.py` | Stores model-written post-mortems; the brain includes up to five recent ones |
| `src/trading/trading_strategy.py` | Position lifecycle, guard enforcement, exit monitoring |
| `src/analyzer/prompts/template_manager.py` | System prompt construction with a falsification-based invalidation step |
| `src/analyzer/trend_validator.py` | Cross-checks model-reported trend strength against the computed ADX |
| `src/analyzer/pattern_quality_scorer.py` | Deterministic pattern quality scoring replacing the model's self-reported score |
| `src/notifiers/notifier.py` | Discord notifications with message-expiration tracking |
