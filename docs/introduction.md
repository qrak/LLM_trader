---

**An autonomous AI trading agent that reads charts, questions its own conclusions, and sharpens its strategy over time — all running on Google's free API tier.**

---

Every trading bot I've seen falls into the same trap: it dumps raw numbers into an LLM and hopes for magic. "RSI is 30, MACD crossed, therefore BUY." But markets don't work that way. A human trader reads the shape of a wedge, checks whether the news confirms the pattern, and remembers the last time this setup failed.

I wanted to build something that could do all three.

What emerged is **LLM_Trader** — an open-source, asyncio-first trading engine that doesn't just calculate indicators. It *sees* charts through vision AI, *remembers* past trades with vector similarity, *writes its own rules* via reflection loops, and *validates its own conclusions* against computed data.

It runs for free on Google Gemini's API tier, passes 913 tests, and has been running continuously on my home server since December 2025.

This is the full story — what it does, how it works, and what I learned building it.

---

## The Core Pipeline

Every analysis cycle follows the same path:

**Market Data → Indicators + Chart → RAG News → Brain Context → AI Provider → Validation → Execution**

### 1. Market Data

The bot aggregates data from 5 exchanges via CCXT (Binance, KuCoin, Gate.io, MEXC, Hyperliquid) — but it doesn't stop at price. Every cycle collects:

- OHLCV candles (125 for chart, 999 for calculation)
- Order book depth and spread
- Trade flow (buy/sell ratio, velocity)
- Funding rates (perp futures)
- 7 timeframes of macro data (4H through 365D)

### 2. Technical Indicators

I built the indicator engine from scratch using NumPy + Numba JIT. No pandas-ta, no TA-Lib. The calculations run in microseconds — a full suite of 40+ indicators (RSI, MACD, ADX, Stochastic, Bollinger Bands, TTM Squeeze, 11 moving average types, and more) completes faster than a single network request.

Why custom? Standard libraries are bloated and hard to optimize. My `@njit(cache=True)` implementations for EMA and SMA compile to machine code on first call and cache the result.

### 3. Chart Generation (Vision AI)

This is the feature that makes the bot unique. Every cycle, it generates a 4K PNG candlestick chart with SMA, RSI, Volume, CMF, and OBV annotations. This image is sent directly to a multimodal AI model (Google Gemini 3.5 Flash) for visual pattern recognition.

I originally wrote code to detect head-and-shoulders, wedges, and trendlines programmatically. I deleted it because the AI was better at reading charts visually than any deterministic algorithm I could write. A chart pattern is worth a thousand RSI readings.

### 4. AI Provider Routing

The bot can be configured to use a single provider or a fallback chain via `config.ini`:

| Provider | Role | Cost |
|----------|------|------|
| Google Gemini 3.5 Flash | Primary (text + vision) | Free tier |
| OpenRouter | Secondary (configurable model) | Pay-per-use |
| LM Studio | Local offline fallback | Free (your hardware) |

Configure `provider = googleai`, `openrouter`, `local`, or `all` for automatic fallback. In fallback mode, the chain adapts to the request type: text-only queries route through Google → LM Studio → OpenRouter, while chart analysis (which requires vision support) goes Google → OpenRouter only — local models don't handle images. The response format is normalized before it reaches the trading logic, so the strategy layer never knows which provider served the analysis.

### 5. Validation (Don't Trust the LLM)

Every AI response goes through a validation pipeline before it's used for trading decisions:

**TrendValidator** cross-checks every LLM-reported `strength_4h` and `strength_daily` against the actual ADX computed from raw OHLCV. If the discrepancy exceeds 15 ADX points, the computed value overwrites the AI's claim.

**PatternQualityScorer** replaces the LLM's self-reported pattern quality score with a deterministic 0-100 score computed from actual pattern detection output (quantity, confirmation, recency, indicator alignment).

**Falsification Check** — the prompt now includes a dedicated step where the model must name a specific price level or indicator condition that would prove its signal wrong. If it can't, the signal is rejected and defaults to HOLD. This single change improved signal quality more than any prompt tweak I've made.

---

## The Brain: Memory That Actually Learns

The original memory system stored lessons in JSON files. It worked for simple win/loss tracking, but it couldn't answer the most important question: *"Show me trades that looked like this one."*

The current system uses **ChromaDB**, a local vector database. Every closed trade is embedded with 15+ metadata fields:

```json
{
  "rsi_at_entry": 42.3,
  "adx_at_entry": 28.7,
  "market_regime": "BULLISH",
  "trend_strength": "HIGH",
  "confluence_factors": ["volume_support", "trend_alignment"],
  "outcome": "WIN",
  "pnl_pct": 3.4,
  "timestamp": "2026-01-06T12:00:00Z"
}
```

When the bot evaluates a new setup, it performs a semantic similarity search: "Find me trades where ADX was high, trend was bullish, and we had volume support." The top 5 most similar past trades are injected into the LLM prompt alongside current market data.

### Three ChromaDB Collections

| Collection | Stores | Retention |
|------------|--------|-----------|
| `trading_experiences` | Every closed trade with full metadata | ~168 days (4h timeframe) |
| `semantic_rules` | Learned trading rules | Active rules preserved |
| `system_constraints_rejections` | Blocked/rejected trades | ~168 days |

### Time-Aware Decay

Markets change. A winning pattern from two months ago might lose money today. Every retrieval applies exponential recency decay:

```
decay = exp(-age_days * ln(2) / half_life_days)
```

For a 4h timeframe, the half-life is ~14 days. A trade from last week carries significantly more weight than one from two months ago, even if the market conditions look similar.

---

## The Reflection Engine: The Bot Writes Its Own Rules

After every N closed trades (configurable per timeframe — every 5 trades on 4h), the bot runs an automated **reflection loop** that analyzes recent outcomes and synthesizes persistent rules.

Three types of rules are generated:

### Best-Practice Rules
> *"LONG trades perform well in BULLISH markets with ADX > 25. Win rate: 78%, Avg P&L: +3.2%, 12 validated trades"*

### Anti-Pattern Rules
> *"SHORT positions opened during LOW volatility after a volume spike tend to reverse. 4 out of 6 closed at a loss. Skip short entries in low-volatility conditions."*

### AI-Mistake Rules
> *"Previous signal ignored bearish divergence on the daily timeframe. A cross-timeframe check is required before committing to a short."*

### Surprise Ratio

Every closed trade now computes a **surprise ratio**: how far the actual outcome deviated from the expected P&L at entry.

```
surprise_ratio = |realized_pnl - expected_pnl| / |expected_pnl|
```

A trade that hits take profit as predicted has `surprise_ratio ≈ 0`. A trade that goes the other direction has a high ratio. Rules with average `surprise_ratio > 1.5` are flagged as "high surprise" — the LLM can distinguish thesis-validated wins from lucky outcomes.

### Rule Lifecycle

Rules aren't permanent. They're ranked by:
- Semantic similarity to current conditions
- Evidence quality (win/loss split, profit factor)
- Recency freshness
- Contradiction penalty (trades that disagree with the rule)

Old rules that haven't been recently validated are labeled `maturing` → `stale` → `legacy` and eventually deactivated. Active rules that accumulate contradictions lose influence over time.

---

## News Integration (RAG Engine)

The bot ingests news from free RSS feeds — CoinDesk, CoinTelegraph, Decrypt, CryptoSlate — with optional content enrichment via Crawl4AI. Each article is scored for relevance to the trading pair, and the top 5 are included in the AI's context.

Articles are processed through a dedicated article processor that:
- Extracts numerical data (prices, percentages, volumes)
- Detects mentioned coins/tickers
- Filters clickbait and non-substantive content
- Rates relevance before including in the prompt

This gives the LLM grounded, current-events context alongside the technical analysis — not just "Bitcoin is up," but structured data about what's happening and why.

---

## Closed-Loop Feedback: When the Bot Rejects Itself

When the LLM proposes a trade that gets blocked by the risk manager, that rejection doesn't disappear. It's stored in ChromaDB with a structured **friction report**:

```
Guard type: min_rr
Detail: R/R was 1.10, minimum is 1.50
Direction: LONG
```

On every subsequent analysis, the brain retrieves recent rejections and formats them as system feedback in the prompt:

> *"Your last LONG signal was rejected. R/R was 1.10, minimum is 1.50. Adjust your SL/TP targets accordingly."*

This closed the loop between the execution layer and the reasoning layer. The LLM learns what the risk manager will and won't accept — without any hardcoded rules about what to tell the AI.

The six guard types in the pipeline:

| Guard | Checks |
|-------|--------|
| Symbol whitelist | Is this pair configured? |
| Max position size | Is requested size within limits? |
| Cooldown | Has enough time passed since last trade? |
| Min R:R | Is risk/reward ≥ 1.5? |
| SL/TP validity | Are levels on the correct side of entry? |
| Wrong-side SL/TP | Stop loss actually above entry for longs? |

---

## Hard Exit Monitoring

Exits used to happen only at candle close. If the market moved against a position at minute 3 of a 4-hour candle, the bot wouldn't react until the candle closed.

Now it supports two exit modes:

| Mode | Checked | What it does |
|------|---------|-------------|
| **Soft** | At candle close | Strategy evaluation in the main analysis loop |
| **Hard** | Every 15 minutes (configurable) | Background async loop, checks live ticker price against SL/TP |

Hard exits run in a dedicated `PositionStatusMonitor` loop — completely independent of the main analysis cycle. The bot also sends periodic status updates to Discord with current P&L, distance to stop/target, and time held.

---

## The Brain Dashboard

The real-time web dashboard started as a debug tool and became one of the most used features. It's a FastAPI + WebSocket application at `localhost:8000` (or live at [semanticsignal.qrak.org](https://semanticsignal.qrak.org)).

**What you see — 8 tabs:**

- **Overview** — Position state, P&L, equity curve, system status at a glance
- **Brain Activity** — Neural network status, synaptic pathway graph (Vis.js), and visual cortex input (charts the bot sent to the AI)
- **Last Prompt** — The full trading context sent to the LLM
- **Last Response** — The raw LLM analysis output
- **Statistics** — Performance history, win/loss ratios, trade metrics
- **Latest News** — Curated crypto news feed ingested via RSS
- **Market Data** — Real-time indicators, order book, funding rates, active position
- **Memory Bank** — Vector database browser — inspect ChromaDB contents and similarity scores

All panels are collapsible, fullscreen-able, and update via WebSocket. Zero build step — vanilla JavaScript, Vis.js for the synapse graph, ApexCharts for performance charts.

---

## Persistence Architecture

| Data | Storage | Why |
|------|---------|-----|
| Trade history | SQLite (WAL mode) | O(1) append, indexed queries |
| Trading experiences | ChromaDB | Vector similarity search |
| Semantic rules | ChromaDB | Semantic retrieval with scoring |
| Position state | JSON (atomic write) | Crash-safe via `os.replace()` |
| CoinGecko cache | SQLite (TTL-bound) | 24-hour expiry, prevents unbounded growth |
| Logs | Rotated daily files | `logs/Bot/YYYY_MM_DD/Bot.log` + `errors.log` |

The JSON trade history was fully migrated to SQLite. The old files were renamed to `.migrated` as backup. Runtime code has zero JSON read/write paths — it fails loudly on SQLite failure.

---

## Testing Philosophy

The test suite was built alongside the code from day one. All tests are **fully mocked** — no real I/O, no network calls, no ChromaDB, no LLM API calls. This means they run in ~60 seconds and are reliable across environments.

The suite covers:

| Area | What's tested |
|------|---------------|
| LLM output corruption | Malformed JSON, missing fields, hallucinated values |
| Async races | Concurrent analysis + position close, double signals |
| Rate limiting | Exponential backoff, retry exhaustion, non-retryable errors |
| Vector DB boundaries | Empty collections, malformed embeddings, context poisoning |
| Friction reporting | All 6 guard types, error paths, edge cases |
| Closed-loop feedback | Injection format, saturation, boundary values |
| Ticker retry | Network timeout → retry → success, exhaustion → None |

913 tests across 63 files. Run with `pytest tests/ -q`.

---

## The Numbers

| Metric | Current |
|--------|---------|
| AI provider | Google Gemini 3.5 Flash (free tier) |
| Cost per analysis | $0 |
| Simulated capital | $10,000 |
| Default timeframe | 4h |
| Test suite | 913 tests, 63 files |
| Memory half-life | ~14 days (4h) |
| Reflection interval | Every 5 trades (4h) |
| Python | 3.13 asyncio-first |
| Dashboard | FastAPI + WebSocket at localhost:8000 |

---

## What's NOT Included (Honest)

- **Real exchange execution** — Paper trading only. No `create_order` code exists.
- **Multi-model consensus** — On the roadmap, not built yet.
- **Trading personalities** — Planned but not delivered.
- **Portfolio management** — Single-pair only (configurable, but one at a time).
- **React/Next.js frontend** — The dashboard is vanilla JavaScript intentionally. No build step, no npm install.

---

## Quick Start

```bash
git clone https://github.com/qrak/LLM_trader.git && cd LLM_trader
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp keys.env.example keys.env   # add GOOGLE_STUDIO_API_KEY (free)
python start.py                # dashboard at http://localhost:8000
```

Runtime controls: `a` = force analysis, `d` = toggle dashboard, `q` = quit graceful shutdown.

---

## Repository

The full source code: **[github.com/qrak/LLM_trader](https://github.com/qrak/LLM_trader)**

Includes:
- AGENTS.md — complete architecture blueprint
- 913 passing tests across 63 files
- Cross-platform startup scripts (Windows, Linux, macOS)
- Full CHANGELOG with every change dated
- MIT license

---

## Disclaimer

**Research and educational purposes only.** Paper trading only. No real financial transactions are executed. The authors are not responsible for any financial decisions made based on this software.

---

*If you found this interesting, give it a clap 👏, star the repo on GitHub, and join the [Discord community](https://discord.gg/ZC48aTTqR2) for development chat and updates.*
