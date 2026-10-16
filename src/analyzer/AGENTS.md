# AGENTS.md — Analysis Engine

Scope: `src/analyzer/`. Root [`AGENTS.md`](../../AGENTS.md) stays canonical for system-wide policy; this file adds package-specific detail and never overrides it.

## Analysis Engine

### 🔬 Analysis Engine Agent — Technical Analysis & Pattern Recognition

> **Module path:** `src/analyzer/analysis_engine.py` (orchestrator) + collaborators in `src/analyzer/`
> **Type:** Market Data Collection → Technical Indicators → Chart Pattern Recognition → AI Signal Generation
> **Core Model:** Google Gemini 3.6 Flash (multimodal for chart image analysis)

---

#### Agent Persona & Role

The Analysis Engine is the **primary market sensing subsystem** — it transforms raw OHLCV data into a structured, multi-dimensional market assessment that the Brain Agent and Trading Strategy use to make decisions.

It performs four distinct analytical passes:
1. **Technical Calculation** — 40+ indicator arrays across momentum, trend, volatility, volume, statistical, and support/resistance categories
2. **Pattern Recognition** — Chart patterns (head & shoulders, double tops, triangles, wedges, channels) + indicator patterns (RSI, MACD, divergence, volume, stochastic, volatility)
3. **Chart Visualization** — 4K candlestick charts with SMA overlays, RSI, volume, CMF/OBV, swing annotations — passed to the LLM for visual pattern analysis
4. **LLM-Powered Signal Synthesis** — Combines all technical data + chart image + RAG context + brain context → structured BUY/SELL/HOLD signal

---

#### Inputs

##### From DataFetcher (via Exchange/CCXT)
- `ohlcv_data: np.ndarray` (columns: timestamp, open, high, low, close, volume) — primary timeframe (4h, up to 999 candles)
- Daily historical (365 days) and weekly macro (300 weeks) data
- Order book depth — multi-level depth buckets, near-mid liquidity, largest wall detection
- Recent trade flow — trade size distribution, buy/sell ratio

##### From External Providers
- CoinGecko — market-wide metrics (dominance, volume, sentiment)
- Alternative.me — Fear & Greed Index
- DeFiLlama — fundamentals (TVL, protocol metrics via RAG pipeline)

##### Reddit community context
- `RedditSentimentAnalyst.fetch_posts()` reads the four public Atom feeds; no extra LLM call or embedding/index dependency.
- `build_context()` supplies up to 15 distinct titles published in the last 48 hours, capped at 600 locally estimated tokens including instructions and UTC dates. Organizational threads, undated/future posts and duplicates are excluded; titles are never cut to fit.
- Titles are untrusted, non-representative evidence. The trading LLM interprets price expectations for the traded asset, respects negation and distinguishes wallet-security concerns from price outlook.
- `analysis.reddit_sentiment` contains `outlook` (BULLISH/BEARISH/MIXED/INSUFFICIENT_EVIDENCE), `evidence` (distinct supplied title IDs) and `reason` (1–240 characters). No eligible titles means null; malformed/missing assessments and out-of-sample IDs are discarded, never replaced with a keyword verdict.
- The validated outlook is stored in `MarketConditions.social_sentiment_reddit` for entry-time learning. Missing data remains empty; insufficient evidence is not embedded as a directional signal. Retired keyword scoring, trending-word extraction, old output keys and placeholder vote/comment counts are deleted.

##### From RAG Engine
- Market context: recent news summaries, relevant articles filtered by taxonomy
- Clear `PromptBuilder.custom_instructions` in `analyze_market`'s `finally` block,
  including failed, cancelled and early-returned requests. Appending a fresh RAG
  snapshot must not retain snapshots from previous failed cycles. Preserve normal
  HOLD/error results and cancellation propagation; do not make repair model calls.

##### From Brain Agent
- `brain_service.get_context()` — historical trade outcomes, learned rules, confidence calibration
- `brain_service.get_dynamic_thresholds()` — learned SL/TP/RR thresholds
- Whole-prompt preflight uses the configurable estimated-token request ceiling,
  not the retired 20,000-token warning. Provider-bound checks separately account
  for the actual model and output/image/safety reserves. Built prompt state may
  precede emergency reduction; the provider-bound `Full prompt content` log is the
  evidence for what was sent, particularly on a smaller fallback model.

##### Configuration (from `config/config.ini`)
- Pair: BTC/USDC, Timeframe: 4h, Candles: 999 (125 for AI chart)
- AI_CHART_CANDLE_LIMIT: 125 (max candles rendered in chart image)

---

#### Outputs

##### `analyze_market()` → Structured dict containing:
| Field | Description |
|-------|-------------|
| `analysis` | Signal (BUY/SELL/HOLD/CLOSE), confidence (0–100), trend direction + strength |
| `raw_response` | Raw LLM text output with reasoning |
| `technical_data` | All computed indicator values formatted for prompt injection |
| `sentiment` | Fear & Greed + market sentiment data |
| `market_microstructure` | Order book depth, trade flow, spread analysis with delta-from-previous |
| `chart_analysis` | Image-based pattern analysis (if chart generation succeeded) |
| `prompt_metadata` | Token counts, sections present, configuration at decision time |
| `prompt_lint` | Pre-flight linting results (missing sections, stale prompt rules) |

##### Validation Overrides (deterministic — always overwrite LLM claims)
- `TrendValidator` — cross-checks LLM ADX claims (15-point warning threshold), always uses computed ADX without changing the signal. The prompt requires copying ADX readings rather than inventing subjective strength scores.
- `PatternQualityScorer` — deterministic 0–100 score from 4 components (30% quantity, 30% confirmation, 20% recency, 20% indicator alignment), flags >25-point divergence from LLM
- `RiskRewardValidator` — checks entry/SL/TP and tagged `RR_CHECK` narrative examples with the same Numba arithmetic as `RiskManager`; an inconsistent entry becomes HOLD before strategy/executor, and a Python correction is saved in `raw_response` for the next cycle's `## PREVIOUS ANALYSIS CONTEXT`, for both JSON-object and narrative-plus-fenced-JSON provider replies. The execution layer still recalculates final R/R after SL/TP normalization.

---

#### Prompting Strategy

##### System Prompt Construction
The `PromptBuilder` composes the system prompt from these sections:
1. **Trading Context** — pair, timeframe, position state, performance metrics
2. **Market Data** — current price, volume, OHLC summary statistics
3. **Technical Analysis** — formatted indicator values with trend direction labels
4. **Period Metrics** — 1D/2D/3D/7D/30D change, volatility, S/R levels
5. **Previous Indicators Comparison** — snapshot delta for trending comparisons
6. **Long-Term/Macro** — daily SMA sets, weekly 200W SMA methodology, golden/death crosses
7. **Market Sentiment** — Fear & Greed, market-wide overview
8. **Trading Brain Context** — injected by BrainAgent. Includes:
   - Confidence calibration by level (win rate, trade count, avg P&L)
   - Direction bias check (long vs short count, "LIMITED DATA" warning)
   - **Blocked-trade feedback** (`get_blocked_trade_feedback()`) — rejected trades from past 168h formatted as `## CRITICAL FEEDBACK: System Rejections` with R:R gap, SL/TP details, and a pre-flight checklist
   - Vector-retrieved similar past experiences (top-3 semantic similarity search)
   - CoT Step 6 — Historical Evidence instructions
   - Learned trading rules matched to current conditions (similarity %, freshness, evidence score)
   - Trade journal: recent post-mortem lessons from closed trades
9. **Previous Analysis Context** (`## PREVIOUS ANALYSIS CONTEXT`) — injected when a previous response exists:
   - Decision snapshot: prior signal, confidence, entry/SL/TP/R:R levels, position size
   - Raw reasoning text (JSON/schema/instruction artifacts stripped; no separate length cap, independently of verbosity)
   - Time check: previous reasoning must be verified against current time/data
   - If the strategy vetoed the previous BUY/SELL, the saved response was patched before persisting — the LLM sees `signal: "HOLD"` with a `⚠️ REJECTED` note instead of a misleading BUY
10. **RAG Context** — news summaries, fundamentals (if available)

##### Report Detail
- `model_verbosity` controls new narrative detail and approximate output-token targets: low 400 / medium 1000 / high 2000 / full 8000 (default). These exclude hidden reasoning and decision JSON and never change execution fields or hard provider caps.
- `full` uses Markdown research sections and developed interpretations. Final `11) RISK/REWARD:`, `12) DECISION:` and `13) EXECUTION NOTE:` labels stay unheaded at column zero for the Python validator. Never copy fabricated historical analogues/probabilities from a reference report.
- DeepSeek keeps one JSON object with a Markdown `narrative` string; other providers keep narrative + a trailing JSON block. The shared result processor renders both for existing consumers.
- Previous reasoning has no separate token/character limit. Prefer the structured thesis when available; otherwise retain the sanitized narrative without length-based truncation. PromptBudgetPolicy checks the whole request immediately before every provider/model call (including fallbacks), using the configured input ceiling and output/image/safety reserves. It can omit optional previous analysis, but a section containing Python R/R feedback stays protected; if protected content cannot fit, refuse the request. TemplateManager no longer needs a token-counter dependency. Local cl100k counts remain estimates; response validation is not a context-length guard.

##### User Prompt Strategy
- Concise instruction asking for structured JSON output
- Includes optional chart image as base64-encoded PNG (4K resolution, 4-row layout)
- Provider-emitted `<think>...</think>` sections are stripped by `AnalysisResultProcessor._clean_response()` before JSON parsing
- Response schema: `TradingAnalysisResponseModel` with validated `TradingAnalysisModel`

##### Response Parsing
`UnifiedParser` handles:
- JSON extraction from ` ```json ` code blocks
- Raw JSON extraction via `json.JSONDecoder.raw_decode()`
- Fallback response if both fail (HOLD, neutral confidence)
- Pydantic validation via `TradingAnalysisResponseModel`
- Signal validation: BUY/SELL requires entry_price, stop_loss, take_profit, risk_reward_ratio, position_size

---

#### Subsystems Detail

##### TechnicalCalculator (`technical_calculator.py`)
40+ indicator arrays computed fresh each cycle:

| Category | Indicators |
|----------|-----------|
| **Volume** | VWAP, TWAP, MFI, OBV, CMF, Force Index, CCI, PVT, A/D Line |
| **Momentum** | RSI (14), Stochastic (14,3,3), Williams %R, UO, TSI, RMI, PPO, Coppock, KST, ROC, MACD (12,26,9) |
| **Volatility** | ATR (20), Bollinger Bands (20,2), %B, Keltner (20,2), Donchian (20), Chandelier Exit (20,3), Choppiness (14) |
| **Trend** | ADX (14), +DI/-DI, TRIX, PFE, TD Setup, Parabolic SAR, Supertrend (20,3), Ichimoku (9,26,52), Vortex, SMAs (20/50/200) |
| **S/R** | Kurtosis, Z-score, Hurst, Entropy, Skewness, Variance, LinReg slope/r², rolling S/R, retested S/R (3+ confirmed swing touches within +/-0.5% over 120 candles), Pivot Points, Fibonacci Pivots |

**Weekly Macro** uses 200W SMA methodology: 5 bullish/bearish criteria scored for cycle phase confidence.

##### PatternEngine (`pattern_engine/`)

All deterministic indicator-pattern detection is Numba `@njit(cache=True)` compiled for performance.

**Chart Patterns** are visually detected by the LLM from the chart image (via `ChartGenerator`, processed through `analysis_result_processor.py`). The `PatternAnalyzer` (`pattern_analyzer.py`) orchestrates indicator pattern detection only, delegating to `IndicatorPatternEngine` (`indicator_pattern_engine.py`).

**Indicator Patterns** (via `pattern_engine/indicator_patterns/indicator_pattern_engine.py`):
7 categories — RSI (oversold/overbought, W-bottom, M-top), MACD (crossovers, histogram), Divergence (bull/bear with 5-candle min spacing), Volume (spike, climax, dry-up, accumulation/distribution), Stochastic (oversold/overbought, crossovers), MA Crossovers (golden/death, alignments), Volatility (ATR spike, BB squeeze, TTM squeeze).

Conventions (`volatility_patterns.py`): the BB squeeze compares the current band width against the lowest 20% of the last **120** candles (not 20 — a 20-candle window fired on ~37% of candles and ignored real history) and reports `width_percentile` (share of the window narrower than now, 0–100); the engine turns that into `confidence = 100 - width_percentile`, so it stays inside 0–100 and never scales a price width. Both numbers are relative to the window, not absolute volatility.

**TTM squeeze** is Carter's published pairing — BB(20, 2σ) against KC(20, **1.5 × Wilder ATR**). The channel is built once in `TechnicalCalculator._calculate_volatility_indicators` (`keltner_channels(length=20, multiplier=1.5, atr_mamode="rma")`: EMA(20) middle, Wilder-smoothed ATR, the same smoothing as the printed `ATR:` line) and it also feeds the prompt's `KC: U/M/L` line, so there is one channel definition, not two. The 2.0 × ATR channel some variants use is the loose Squeeze Pro tier: measured on 999 closed BTC/USDC 4h bars (2026-05-02 → 2026-10-15) it read "in squeeze" on **42%** of bars against **15%** for 1.5, so a 2.0 channel must not be labelled extreme volatility. The squeeze is a **state**, not an event: `detect_keltner_squeeze_numba` returns `(in_squeeze, bars_in_squeeze, bars_since_release)`, the engine emits `ttm_squeeze` while the bands sit inside the channel (the description carries the run length, e.g. `in compression for 3 bars`) and `ttm_squeeze_fired` once they have left, so a multi-day compression no longer looks like a fresh setup every cycle.

**Every state pattern reports its duration.** `macd_histogram_trend_streak_numba`, `volatility_trend_streak_numba`, `bb_squeeze_streak_numba` and `accumulation_distribution_streak_numba` each count the consecutive bars the same label has held — they re-evaluate the owning detector's own rule per bar through a shared `_*_at` helper, so the streak and the label can never drift apart — and the engine renders `for N bars`. This matters because those four states sit in the prompt on **100%** (MACD histogram direction), 96% (Stochastic cross), 85% (20/50 SMA cross), 77% (MACD cross), 64% (volatility trend) and 62% (accumulation/distribution) of measured cycles; without a duration the model reads an old condition as a fresh signal. Descriptions carry **no grading adjectives** — "momentum shift", "breakout imminent" and "strong signal" were removed because each claimed more than its relative threshold proves.

**The prompt's `## Key Terminology` legend is the single home for indicator provenance and pattern thresholds** (`src/analyzer/prompts/template_manager.py`): fixed calculator windows (RSI 14 Wilder, MACD 12/26/9 EMA, Stochastic 14/3/3, Bollinger 20/2 population σ, ATR 20 Wilder, KC EMA(20) ± 1.5 × Wilder ATR(20), …) and every detector threshold (RSI 30/70, Stochastic 20/80 with crosses inside 30/70, volume spike 2.5× / dry-up 0.5× / climax 3×, ATR spike 1.5×, BB squeeze lowest 20% of 120 bars, MACD histogram majority of 3 changes, volatility trend 6 of 9 ATR changes, accumulation/distribution 60%/40% with `strength = |share − 0.5| × 2`, volume-price divergence ±2%/−15%, divergence spacing 5 bars). Adding an indicator or a detector threshold without a legend line leaves the model assuming textbook defaults, which is exactly how a printed Wilder ATR came to look like the KC's EMA-ATR.

##### ChartGenerator (`pattern_engine/chart_generator.py`)
- Resolution: 3840×2160 (4K)
- Layout: 4 rows — Candlestick + SMA (55%), RSI (15%), Volume (15%), CMF + OBV (15%)
- AI-optimized: black background, high-contrast colors, swing point annotations, global max/min labels
- Resilience: 30s timeout per export, up to 3 retries with exponential backoff
- Format: PNG via Plotly + Kaleido

---

#### Model Configuration (from `config.ini` `[model_config]`)

| Parameter | Config Key | Description |
|-----------|-----------|-------------|
| `temperature` | `temperature` | Sampling temperature (loaded dynamically, not hardcoded) |
| `top_p` | `top_p` | Nucleus sampling parameter |
| `frequency_penalty` | `frequency_penalty` (fallback: `freq_penalty`) | Reduces repetition |
| `presence_penalty` | `presence_penalty` (fallback: `pres_penalty`) | Encourages new topics |
| `max_tokens` | `max_tokens` | **Required** — response token limit for all providers |
| `google_max_tokens` | `google_max_tokens` | **Required** — Google-specific token limit |
| `google_thinking_level` | `google_thinking_level` | Google thinking depth (default: `"high"`) |
| `google_code_execution` | `google_code_execution` | Enable code execution (default: `false`) |

Parameters known to be unsupported by some providers are pre-emptively filtered by the shared provider-client retry path before each API call: `thinking_budget`, `thinking_config`, `top_k`, `freq_penalty`, `pres_penalty`.

---

#### Edge Cases & Guardrails

| Scenario | Handling |
|----------|----------|
| **Data fetch failure** | Returns `{"error": "Failed to collect market data", "recommendation": "HOLD"}` |
| **Exchange doesn't support timeframe** | Logs warning, proceeds with available granularity |
| **New token / insufficient history** | Sets `is_new_token` flag, uses fallback defaults for long-term/macro |
| **Chart generation fails** | Falls back to text-only AI analysis, logs warning |
| **RAG engine unavailable** | Logs warning, continues with empty market context |
| **AI response unparseable** | Returns fallback HOLD with raw response attached |
| **Invalid JSON from LLM** | `UnifiedParser` attempts codeblock extraction → raw_decode → fallback |
| **Missing execution fields for BUY/SELL** | `TradingAnalysisModel` validation raises ValueError |
| **ADX validity — LLM overstates trend** | `TrendValidator` always overwrites with computed value |
| **Pattern quality — LLM diverges >25 points** | Flagged in analysis output, always overwritten |
| **Chart export hangs** | Daemon thread with 30s timeout, retry 3× |
| **Microstructure comparisons** | Previous snapshot tracking is scoped per symbol |
| **Incomplete candle in dataset** | DataFetcher automatically excludes the last (incomplete) candle |
| **Indicator array mismatch** | Sliced to match displayed candle count for chart rendering |
