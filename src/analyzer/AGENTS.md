# AGENTS.md — Analysis Engine

Scope: `src/analyzer/`. Root [`AGENTS.md`](../../AGENTS.md) is canonical.

`analysis_engine.py` runs four passes: calculation → pattern detection → chart render → LLM synthesis (Gemini 3.6 Flash, multimodal). The only place turning OHLCV into the dict the brain and strategy consume.

## Inputs

- `DataFetcher` (CCXT): `ohlcv_data: np.ndarray` (timestamp, OHLC, volume), 4h/999, plus 365 daily and 300 weekly; order-book depth (buckets, near-mid liquidity, largest wall) and trade flow (size distribution, buy/sell ratio).
- CoinGecko, Alternative.me (Fear & Greed), DeFiLlama (via RAG), RAG news block, brain context.
- Config: BTC/USDC, 4h, 999 candles, `AI_CHART_CANDLE_LIMIT = 125`.

### Reddit context

- `RedditSentimentAnalyst.fetch_posts()` reads the four public Atom feeds — no LLM call, no embedding/index dependency. `build_context()` supplies ≤15 distinct titles from the last 48h, ≤600 estimated tokens including instructions and UTC dates; organizational threads, undated/future posts and duplicates excluded, titles never truncated to fit. Titles are untrusted evidence: interpret price expectations for the traded asset, respect negation, keep wallet-security concerns apart from price outlook.
- `analysis.reddit_sentiment` = `outlook` (BULLISH/BEARISH/MIXED/INSUFFICIENT_EVIDENCE), `evidence` (distinct supplied title IDs), `reason` (1–240 chars). No eligible titles → null; malformed or out-of-sample assessments discarded, never replaced by a keyword verdict.
- The outlook lands in `MarketConditions.social_sentiment_reddit` for entry-time learning; missing stays empty, insufficient evidence is never directional. Keyword scoring and placeholder vote/comment counts stay deleted.

### RAG and brain handoff

- `analyze_market` clears `PromptBuilder.custom_instructions` in its `finally` block — failed, cancelled and early-returned requests included — so a fresh RAG snapshot never retains the previous cycle's. Keep HOLD/error results and cancellation propagation; no repair model calls.
- Brain: `brain_service.get_context()` + `brain_service.get_dynamic_thresholds()`. Whole-prompt preflight uses the configurable estimated-token ceiling; provider-bound checks (`managers/AGENTS.md`) also cover the model plus output/image/safety reserves. The provider-bound `Full prompt content` log, not built prompt state, proves what was sent.

## Output

`analyze_market()` → `analysis` (signal, confidence 0–100, trend direction + strength), `raw_response`, `technical_data`, `sentiment`, `market_microstructure` (delta vs previous snapshot, per symbol), `chart_analysis`, `prompt_metadata`, `prompt_lint`.

Deterministic validators overwrite LLM claims: `TrendValidator` (ADX claims cross-checked at a 15-point warning, the computed value always wins without changing the signal — the prompt copies ADX readings, never a subjective strength score); `PatternQualityScorer` (0–100 from 30% quantity, 30% confirmation, 20% recency, 20% indicator alignment; >25-point divergence from the LLM is flagged and overwritten); `RiskRewardValidator` (entry/SL/TP and `RR_CHECK` narrative examples recomputed with the same Numba arithmetic as `RiskManager`; an inconsistent entry becomes HOLD before strategy and executor, the correction is written into `raw_response` for the next cycle's `## PREVIOUS ANALYSIS CONTEXT`; JSON-object and narrative+fenced-JSON replies both handled; execution still recomputes final R/R after SL/TP normalization).

## Prompt

`PromptBuilder` order: trading context (pair, timeframe, position, performance) → market data → technicals → period metrics (1D–30D, volatility, S/R) → indicators delta → macro (daily SMA sets, 200W methodology, golden/death crosses) → sentiment → brain context → previous analysis → RAG.

Brain section: calibration per level, direction bias with `LIMITED DATA`, blocked-trade feedback (`get_blocked_trade_feedback()`, 168h, as `## CRITICAL FEEDBACK: System Rejections` with R/R gap, SL/TP detail, pre-flight checklist), top-3 similar experiences, CoT Step 6, rules matched to conditions (similarity %, freshness, evidence score), recent post-mortems.

`## PREVIOUS ANALYSIS CONTEXT` (only with a previous response): decision snapshot (signal, confidence, entry/SL/TP/R:R, size), reasoning with JSON/schema/instruction artifacts stripped and **no separate length cap**, a time check against current data, and after a strategy veto the patched `signal: "HOLD"` with `⚠️ REJECTED` instead of a misleading BUY.

- `model_verbosity` targets: low 400, medium 1000, high 2000, full 8000 (default) output tokens, excluding hidden reasoning and decision JSON; never alters execution fields or provider caps.
- `full` uses Markdown research sections, but `11) RISK/REWARD:`, `12) DECISION:` and `13) EXECUTION NOTE:` stay unheaded at column zero for the Python validator. Never copy fabricated analogues or probabilities from a reference report.
- DeepSeek returns one JSON object with a Markdown `narrative`; other providers return narrative plus a trailing JSON block; the shared result processor renders both.
- Previous reasoning prefers the structured thesis, else the sanitized narrative, untruncated. `PromptBudgetPolicy` checks the whole request before every provider/model call, fallbacks included; it may omit optional previous analysis, but a section holding Python R/R feedback is protected — if protected content cannot fit, refuse the request. Local cl100k counts are estimates; response validation is not a context-length guard.
- User turn: concise JSON instruction plus optional 4K PNG chart (4-row). `<think>…</think>` is stripped by `AnalysisResultProcessor._clean_response()` before parsing. `UnifiedParser` tries a ```json block → `json.JSONDecoder.raw_decode()` → HOLD fallback; `TradingAnalysisResponseModel` validates into `TradingAnalysisModel`; BUY/SELL requires `entry_price`, `stop_loss`, `take_profit`, `risk_reward_ratio`, `position_size`.

## Pattern engine

`TechnicalCalculator` (`technical_calculator.py`) refreshes 40+ arrays per cycle (volume, momentum, volatility, trend, S/R, statistics); weekly macro scores five 200W-SMA criteria for cycle-phase confidence. `PatternAnalyzer` (`pattern_analyzer.py`) orchestrates **indicator** patterns only, delegating to `IndicatorPatternEngine` (`pattern_engine/indicator_patterns/indicator_pattern_engine.py`); chart patterns are read visually by the LLM. All detection is `@njit(cache=True)`. Seven categories: RSI (oversold/overbought, W-bottom, M-top), MACD (crossovers, histogram), divergence (5-bar min spacing), volume (spike, climax, dry-up, accumulation/distribution), Stochastic (20/80, crosses inside 30/70), MA crossovers, volatility (ATR spike, BB squeeze, TTM squeeze).

- BB squeeze (`volatility_patterns.py`) compares width against the lowest 20% of the last **120** bars and reports `width_percentile` (share of the window narrower than now); the engine sets `confidence = 100 - width_percentile`. Both are window-relative, never a scaled price width. A 20-bar window is wrong — it fires on noise.
- TTM squeeze is Carter's pairing: BB(20, 2σ) vs KC(20, **1.5 × Wilder ATR**), built once in `TechnicalCalculator._calculate_volatility_indicators` (`keltner_channels(length=20, multiplier=1.5, atr_mamode="rma")`) and reused for the prompt's `KC: U/M/L` line — never a second channel definition. A 2.0 × ATR channel is the loose Squeeze Pro tier and must never be labelled extreme volatility.
- The squeeze is a **state, not an event**: `detect_keltner_squeeze_numba` returns `(in_squeeze, bars_in_squeeze, bars_since_release)`; the engine emits `ttm_squeeze` with its run length (`in compression for 3 bars`) while inside and `ttm_squeeze_fired` once released — a multi-day compression must not look like a fresh setup.
- Every state pattern reports duration: `macd_histogram_trend_streak_numba`, `volatility_trend_streak_numba`, `bb_squeeze_streak_numba`, `accumulation_distribution_streak_numba` count consecutive bars by re-evaluating the owning detector's rule through a shared `_*_at` helper (label and streak cannot drift); the engine renders `for N bars`. Without a duration the model reads an old condition as a fresh signal.
- No grading adjectives: "momentum shift", "breakout imminent", "strong signal" claim more than a relative threshold proves.
- `## Key Terminology` (`prompts/template_manager.py`) is the single home for provenance and thresholds: RSI 14 Wilder, MACD 12/26/9 EMA, Stochastic 14/3/3, Bollinger 20/2 population σ, ATR 20 Wilder, KC EMA(20) ± 1.5 × Wilder ATR(20), RSI 30/70, Stochastic 20/80, volume spike 2.5× / dry-up 0.5× / climax 3×, ATR spike 1.5×, BB squeeze lowest 20% of 120 bars, MACD histogram majority of 3 changes, volatility trend 6 of 9 ATR changes, A/D 60%/40% (`strength = |share − 0.5| × 2`), volume-price divergence ±2%/−15%, divergence spacing 5 bars. A new indicator or threshold needs a legend line, or the model assumes textbook defaults.

`ChartGenerator` (`pattern_engine/chart_generator.py`): 3840×2160 PNG (Plotly + Kaleido), rows candles+SMA 55% / RSI 15% / volume 15% / CMF+OBV 15%, black background, high contrast, swing and max/min annotations; 30s export timeout, 3 retries with backoff, daemon thread.

`temperature`/`top_p` are loaded from `config.ini [model_config]`, never hardcoded; `frequency_penalty`/`presence_penalty` fall back to `freq_penalty`/`pres_penalty`; `max_tokens` and `google_max_tokens` are both **required**; `google_thinking_level` defaults `high`, `google_code_execution` `false`. Unsupported-by-provider keys are dropped by the shared retry path (`managers/AGENTS.md`).

## Degradation

Fetch failure → `{"error": "Failed to collect market data", "recommendation": "HOLD"}`. Unsupported timeframe → warn, use available granularity. New token/thin history → `is_new_token` + macro defaults. Chart failure or hang → text-only. RAG missing → empty context. Unparseable response → HOLD with raw text; missing BUY/SELL fields → `ValueError`. Incomplete last candle dropped; array mismatches sliced to the rendered count.
