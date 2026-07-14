# 🔬 Analysis Engine Agent — Technical Analysis & Pattern Recognition

> **Module path:** `src/analyzer/analysis_engine.py` (orchestrator) + collaborators in `src/analyzer/`
> **Type:** Market Data Collection → Technical Indicators → Chart Pattern Recognition → AI Signal Generation
> **Core Model:** Google Gemini 3.5 Flash (multimodal for chart image analysis)

---

## Agent Persona & Role

The Analysis Engine is the **primary market sensing subsystem** — it transforms raw OHLCV data into a structured, multi-dimensional market assessment that the Brain Agent and Trading Strategy use to make decisions.

It performs four distinct analytical passes:
1. **Technical Calculation** — 40+ indicator arrays across momentum, trend, volatility, volume, statistical, and support/resistance categories
2. **Pattern Recognition** — Chart patterns (head & shoulders, double tops, triangles, wedges, channels) + indicator patterns (RSI, MACD, divergence, volume, stochastic, volatility)
3. **Chart Visualization** — 4K candlestick charts with SMA overlays, RSI, volume, CMF/OBV, swing annotations — passed to the LLM for visual pattern analysis
4. **LLM-Powered Signal Synthesis** — Combines all technical data + chart image + RAG context + brain context → structured BUY/SELL/HOLD signal

---

## Inputs

### From DataFetcher (via Exchange/CCXT)
- `ohlcv_data: np.ndarray` (columns: timestamp, open, high, low, close, volume) — primary timeframe (4h, up to 999 candles)
- Daily historical (365 days) and weekly macro (300 weeks) data
- Order book depth — multi-level depth buckets, near-mid liquidity, largest wall detection
- Recent trade flow — trade size distribution, buy/sell ratio

### From External Providers
- CoinGecko — market-wide metrics (dominance, volume, sentiment)
- Alternative.me — Fear & Greed Index
- DeFiLlama — fundamentals (TVL, protocol metrics via RAG pipeline)

### From RAG Engine
- Market context: recent news summaries, relevant articles filtered by taxonomy

### From Brain Agent
- `brain_service.get_context()` — historical trade outcomes, learned rules, confidence calibration
- `brain_service.get_dynamic_thresholds()` — learned SL/TP/RR thresholds

### Configuration (from `config/config.ini`)
- Pair: BTC/USDC, Timeframe: 4h, Candles: 999 (125 for AI chart)
- AI_CHART_CANDLE_LIMIT: 125 (max candles rendered in chart image)

---

## Outputs

### `analyze_market()` → Structured dict containing:
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

### Validation Overrides (deterministic — always overwrite LLM claims)
- `TrendValidator` — cross-checks LLM ADX claims (±15 delta threshold), always uses computed value
- `PatternQualityScorer` — deterministic 0–100 score from 4 components (30% quantity, 30% confirmation, 20% recency, 20% indicator alignment), flags >25-point divergence from LLM

---

## Prompting Strategy

### System Prompt Construction
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
   - Raw reasoning text (JSON-stripped, truncated per verbosity setting)
   - Time check: previous reasoning must be verified against current time/data
   - If the strategy vetoed the previous BUY/SELL, the saved response was patched before persisting — the LLM sees `signal: "HOLD"` with a `⚠️ REJECTED` note instead of a misleading BUY
10. **RAG Context** — news summaries, fundamentals (if available)

### User Prompt Strategy
- Concise instruction asking for structured JSON output
- Includes optional chart image as base64-encoded PNG (4K resolution, 4-row layout)
- Provider-emitted `<think>...</think>` sections are stripped by `AnalysisResultProcessor._clean_response()` before JSON parsing
- Response schema: `TradingAnalysisResponseModel` with validated `TradingAnalysisModel`

### Response Parsing
`UnifiedParser` handles:
- JSON extraction from ` ```json ` code blocks
- Raw JSON extraction via `json.JSONDecoder.raw_decode()`
- Fallback response if both fail (HOLD, neutral confidence)
- Pydantic validation via `TradingAnalysisResponseModel`
- Signal validation: BUY/SELL requires entry_price, stop_loss, take_profit, risk_reward_ratio, position_size

---

## Subsystems Detail

### TechnicalCalculator (`technical_calculator.py`)
40+ indicator arrays computed fresh each cycle:

| Category | Indicators |
|----------|-----------|
| **Volume** | VWAP, TWAP, MFI, OBV, CMF, Force Index, CCI, PVT, A/D Line |
| **Momentum** | RSI (14), Stochastic (14,3,3), Williams %R, UO, TSI, RMI, PPO, Coppock, KST, ROC, MACD (12,26,9) |
| **Volatility** | ATR (20), Bollinger Bands (20,2), %B, Keltner (20,2), Donchian (20), Chandelier Exit (20,3), Choppiness (14) |
| **Trend** | ADX (14), +DI/-DI, TRIX, PFE, TD Sequential, Parabolic SAR, Supertrend (20,3), Ichimoku (9,26,52), Vortex, SMAs (20/50/200) |
| **S/R** | Kurtosis, Z-score, Hurst, Entropy, Skewness, Variance, LinReg slope/r², basic S/R, advanced S/R, Pivot Points, Fibonacci Pivots |

**Weekly Macro** uses 200W SMA methodology: 5 bullish/bearish criteria scored for cycle phase confidence.

### PatternEngine (`pattern_engine/`)

All deterministic indicator-pattern detection is Numba `@njit(cache=True)` compiled for performance.

**Chart Patterns** are visually detected by the LLM from the chart image (via `ChartGenerator`, processed through `analysis_result_processor.py`). The `PatternAnalyzer` (`pattern_analyzer.py`) orchestrates indicator pattern detection only, delegating to `IndicatorPatternEngine` (`indicator_pattern_engine.py`).

**Indicator Patterns** (via `pattern_engine/indicator_patterns/indicator_pattern_engine.py`):
7 categories — RSI (oversold/overbought, W-bottom, M-top), MACD (crossovers, histogram), Divergence (bull/bear with 5-candle min spacing), Volume (spike, climax, dry-up, accumulation/distribution), Stochastic (oversold/overbought, crossovers), MA Crossovers (golden/death, alignments), Volatility (ATR spike, BB squeeze, TTM squeeze).

### ChartGenerator (`pattern_engine/chart_generator.py`)
- Resolution: 3840×2160 (4K)
- Layout: 4 rows — Candlestick + SMA (55%), RSI (15%), Volume (15%), CMF + OBV (15%)
- AI-optimized: black background, high-contrast colors, swing point annotations, global max/min labels
- Resilience: 30s timeout per export, up to 3 retries with exponential backoff
- Format: PNG via Plotly + Kaleido

---

## Model Configuration (from `config.ini` `[model_config]`)

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

## Edge Cases & Guardrails

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
