# 🧠 Brain Agent — TradingBrainService

> **Module path:** `src/trading/brain.py` (facade) + 5 collaborators in `src/trading/`
> **Type:** Central LLM Decision Engine & Outcome-Aware Learning Loop
> **Core Model:** No direct LLM calls; deterministic/vector-memory service whose context is injected into AnalysisEngine prompts routed by ProviderOrchestrator

---

## Agent Persona & Role

The Brain Agent is the **central reasoning and learning subsystem** of LLM Trader. It acts as both:

- **Decision Enricher:** Injects historical trade outcomes, confidence calibration, and learned rules into every LLM prompt so the model makes context-aware decisions.
- **Autonomous Learner:** After each closed trade, records the outcome into ChromaDB vector memory, and periodically reflects on trade clusters to synthesize semantic rules (best-practices, anti-patterns, corrective measures, AI-mistake patterns).

The Brain does **not** execute trades or calculate indicators — it operates purely on metadata produced by the AnalysisEngine and TradingStrategy.

### Key Collaborators

| Module | File | Responsibility |
|--------|------|----------------|
| `BrainContextProvider` | `brain_context.py` | Assembles the "Trading Brain" context block injected into every LLM decision prompt |
| `BrainExperienceRecorder` | `brain_experience.py` | Translates closed-trade data + market conditions into structured vector-memory experiences |
| `ExitProfileResolver` | `brain_exit_profiles.py` | Single source of truth for SL/TP execution profiles serialization and rule rendering |
| `TradePatternAnalyzer` | `brain_patterns.py` | Statistical analysis engine — win/loss grouping, failure diagnostics, AI mistake classification |
| `BrainReflectionEngine` | `brain_reflection.py` | Synthesizes learned semantic rules from trade metadata clusters via periodic reflection loops |

---

## Inputs

### From TradingStrategy (via `close_position()`)
- `Position` — entry/exit/PNL, confidence, confluence factors, drawdown metrics
- `close_price` — exit price at time of close
- `close_reason` — stop_loss / take_profit / analysis_signal
- `entry_decision` — original entry TradeDecision (retrieved from SQLite trade history through `PersistenceManager` for reasoning context)
- `market_conditions` — MarketConditions at close time (or from entry if preferred)

### From VectorMemoryService (ChromaDB)
- Trade experiences: 20+ metadata fields per trade (entry confidence, AI reasoning, ADX/RSI/ATR at entry, volatility, SL/TP distances, RR ratio, max drawdown/profit, fear & greed, market regime, confluence count, timeframe alignment, exit execution context, factor scores)
- Semantic rules: best_practice, anti_pattern, corrective, ai_mistake types
- Blocked-trade feedback: guard-type grouped rejection history
- Confidence calibration stats: win rate by confidence level, direction bias, ADX performance, factor performance

---

## Outputs

### `get_context()` — Formatted text block for LLM prompt injection
- Confidence calibration by level (HIGH/MEDIUM/LOW with win rate, trade count, avg P&L)
- Direction bias check (long vs short count, "LIMITED DATA" warning)
- Blocked-trade feedback (recent 5, ≤ 168h)
- Vector-retrieved similar past experiences (top-5 semantic similarity search)
- CoT Step 6 — Historical Evidence instructions (win-rate < 50% → reduce confidence, anti-pattern matching, AI-mistake memory, exit-execution memory)
- Learned trading rules matched to current conditions (similarity %, timeframe freshness, evidence score, tagged by type)

### `get_dynamic_thresholds()` — ~15 learned parameters
- R/R minimum thresholds, ADX thresholds, SL tightening progress thresholds
- Position size limits, confluence count minimums, timeframe alignment reduction coefficients

### `get_vector_context()` — Vector-similarity search results + contextual stats

### Semantic Rules (stored to ChromaDB)
- **Best-practice rules:** ≥5 wins total, ≥3 same-pattern, ≥60% win rate → `rule_best_*`
- **Anti-pattern rules:** ≥3 losses total, ≥2 same-pattern, ≥60% loss rate → ⚠️ `rule_anti_*` with failure_reason + recommended_adjustment
- **Corrective rules:** ≥3 losses, ≥2 same-pattern, <60% loss rate → ⚡ `rule_corrective_*`
- **AI-mistake rules:** ≥2 mistakes classified → `rule_ai_mistake_*` with failed_assumption, mistake_type

---

## Prompting Strategy

### Reflection Cadence (timeframe-adaptive)

| Timeframe Bucket | Trade Interval |
|-----------------|----------------|
| Scalping (≤ 30 min) | Every 10 trades |
| Intraday (60–239 min) | Every 7 trades |
| Swing (240–1439 min) | Every 5 trades |
| Position (≥ 1440 min) | Every 3 trades |

On each reflection tick, three reflection loops fire **sequentially** (not parallel):
1. `trigger_reflection()` — best-practice rules from winning clusters
2. `trigger_loss_reflection()` — anti-pattern / corrective rules from losing clusters
3. `trigger_ai_mistake_reflection()` — AI mistake pattern detection

### Context Injection Strategy

The brain context is injected as a structured section in the LLM prompt **after** technical analysis but **before** the final decision instruction:

```
--- Trading Brain Context ---
[Confidence Calibration by Level]
[Direction Bias Warning]
[Blocked Trade Feedback]
[Top-5 Similar Past Experiences]
[Active Semantic Rules Matched to Current Conditions]
[Historical Evidence Instructions (CoT Step 6)]
```

### Embedding Strategy
- `build_rich_context_string()` — categorical labels for storage/retrieval
- `build_query_document()` — embedding query mirrors stored format but includes raw numeric values, reducing embedding asymmetry
- Hybrid retrieval: 70% similarity + 30% recency decay

### Semantic Rule Influence Scoring

Active semantic rules are durable learned policy, not raw trade examples. They are preserved while active, but their prompt influence is soft-ranked by:

- Semantic similarity to the current market context
- Evidence quality (`wins`, `losses`, `source_trades`, `expectancy_pct`, `profit_factor`)
- Timeframe-aware freshness using the same half-life model as trade experiences
- Contradiction penalty from matched closed trades that disagree with the rule

For the default 4h timeframe, the rule freshness half-life is about 14 days. Older active rules are not deleted automatically; they receive lower freshness labels (`maturing`, `stale`, `legacy`) unless recently validated by matching outcomes.

### Confidence Threshold — Adaptive Learning

The brain learns a `confidence_threshold` dynamically from historical trade data in `VectorMemoryAnalytics`:

| Condition | Threshold Set |
|-----------|--------------|
| HIGH confidence win rate > 70% | 65 (relaxed — HIGH confidence is reliable) |
| HIGH confidence win rate < 55% | 75 (tightened — HIGH confidence underperforming) |
| Default (insufficient data) | 70 |

This threshold is injected into LLM prompts to guide the model's self-assessed confidence level, but does **not** directly gate position sizing — position sizing uses confidence as a string label ("HIGH"/"MEDIUM"/"LOW") against configured fallback size percentages.

---

## SL Tightening Policy

The `StopLossTighteningPolicy` enforces a **price-progress gate** before allowing the LLM to tighten an open position's stop loss. This prevents premature SL adjustments that would turn small pullbacks into exits.

### Timeframe-Adaptive Base Thresholds

| Timeframe | Min Progress to TP | Meaning |
|-----------|-------------------|---------|
| Scalping (<1h) | 25% | Price must travel 25% toward TP before SL can tighten |
| Intraday (1h–4h) | 20% | |
| Swing (4h–1d) | 15% | |
| Position (>1d) | 10% | |

### Blending with Brain-Learned Thresholds

The effective threshold is resolved by `_resolve_effective_threshold()`:
1. Start with the timeframe base threshold
2. If a brain-learned SL tightening threshold exists with enough samples, use the learned value clamped to the configured floor/ceiling
3. Expose the result via `get_dynamic_thresholds()` as `sl_tightening_pct`, `sl_tightening_source`, and the nested `sl_tightening` payload

### Position Update Gating

`TradingStrategy` enforces a **timeframe-adaptive minimum interval** between successive position parameter updates. This prevents the LLM from "over-managing" open positions with continuous micro-adjustments:

| Timeframe | Multiplier | Effect at 4h |
|-----------|-----------|-------------|
| Scalping (<60 min) | 4× | — |
| Intraday (60–239 min) | 3× | — |
| Swing (240–1439 min) | 2× | Update every 8h |
| Position (≥1440 min) | 1× | Update every 24h |

The gate lives in `TradingStrategy._handle_existing_position()`:
```python
if hours_since_last < self._min_update_interval_hours:
    # REJECTED UPDATE — letting trade breathe
    return None
```

---

## Edge Cases & Guardrails

| Scenario | Handling |
|----------|----------|
| **Empty SQLite trade history** | `get_context()` returns empty string — brain section skipped entirely in prompt |
| **Insufficient data for reflection** | <5 wins → skip best-practice; <3 losses → skip loss reflection; <2 mistakes → skip AI mistake reflection |
| **Low win rate (<60%)** | Best-practice rule rejected |
| **Single-occurrence patterns** | Skipped (need ≥3 wins / ≥2 losses / ≥2 mistakes for pattern) |
| **Unknown exit profiles** | `UNKNOWN_EXIT_PROFILE` sentinel used as default; `refresh_semantic_rules_if_stale()` migrates legacy rules |
| **ChromaDB unavailable** | All operations gracefully degrade — `get_context()` returns empty, reflection skipped with warning log |
| **Reflection failure** | All reflection wrapped in try/except with warning log — never crashes the trading loop |
| **Blocked trade feedback stall** | Failures silently caught/passed |
| **timeframe_minutes ≤ 0** | Defaults to 240 minutes |
| **Cross-restart trade count** | `trade_count` re-read from `vector_memory.trade_count` on init |
| **Stats cache staleness** | Auto-invalidated when vector-memory experience count changes |
| **Unknown profile in vector context** | Replaced with resolved rule defaults in `get_context()` |
| **"LIMITED DATA" flag detected** | Swaps full CoT instructions for "rely on standard TA" note |

### Vector Memory Maintenance

- `VectorMemoryService.prune_aged_documents()` may prune stale experiences and blocked-trade feedback beyond the relevance window.
- Active semantic rules (`active=True`) are preserved even when old; do not delete active learned rules by age alone.
- Active semantic rules are ranked with timeframe-aware freshness/evidence scoring before prompt injection; age lowers influence but does not physically delete a rule.
- Matched closed trades update semantic-rule `validation_hit_count`, `last_validated_at`, `contradiction_count`, and `last_contradicted_at` metadata.
- Timestamp parsing must be datetime-aware so malformed timestamps are skipped safely instead of corrupting prune decisions.

### Trade Persistence Contract

- Trading code records decisions through `PersistenceManager.async_save_trade_decision()` and must not write trade-history files directly.
- Trade history is SQLite-only (`trade_history.db`). Do not reintroduce `trade_history.json` readers, fallback writes, or auto-migration paths.
- Entry-decision recovery for close-time brain learning uses SQLite timestamp-window lookup with optional symbol filtering.

---

## Data Flow

```
Market Data + Indicators
    ↓
AnalysisEngine ──→ Position + MarketConditions
    ↓
TradingStrategy ──→ closed trade
    ├── PersistenceManager.get_entry_decision_for_position()
    │     └── SQLite trade_history lookup by timestamp + symbol
    ↓
TradingBrainService.update_from_closed_trade()
    ├── BrainExperienceRecorder.record_closed_trade()
    │     └── VectorMemoryService.store_experience()
    │           (trade metadata + vector embedding)
    ├── trade_count++
    └── if count % reflection_interval == 0:
          ├── BrainReflectionEngine.trigger_reflection()       → best-practice rules
          ├── trigger_loss_reflection()                        → anti-pattern/corrective rules
          └── trigger_ai_mistake_reflection()                  → AI mistake rules

Before next LLM decision:
TradingBrainService.get_context()
    └── BrainContextProvider.get_context()
          ├── VectorMemoryService (confidence stats, direction bias, blocked feedback)
          ├── get_vector_context() (semantic similarity search)
          └── VectorMemoryService.get_relevant_rules()
                ↓
    Injected into LLM prompt as "Trading Brain Context" section
```
