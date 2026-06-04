# ⚙️ Risk Manager, Persistence Manager & Provider Orchestrator

> **Module path:** `src/managers/risk_manager.py` + `src/managers/persistence_manager.py` + `src/managers/sqlite_trade_history.py` + `src/managers/provider_orchestrator.py`
> **Type:** Signal Execution Safety + SQLite Persistence + AI Provider Fallback Chain

---

## 1. Risk Manager Agent

> **File:** `src/managers/risk_manager.py`

### Agent Persona & Role

The Risk Manager is the **safety layer between the LLM-generated trading signal and actual order execution.** It converts raw AI signals into a validated `RiskAssessment` by applying dynamic position sizing, SL/TP scaling, and consistency checks.

It is **not** a full risk management system — real exchange order execution is not yet implemented. The Risk Manager operates in paper-trading simulation mode.

### Inputs

- `TradingAnalysisModel` from `UnifiedParser` — signal, confidence, entry_price, stop_loss, take_profit, position_size
- `MarketConditions` — current price, ATR, volatility indicators
- `Position` (existing) — for UPDATE/CLOSE signal handling
- **Brain-learned thresholds** — dynamic SL/TP/RR/confluence thresholds from `get_dynamic_thresholds()`

### Outputs

| Field | Description |
|-------|-------------|
| `RiskAssessment` | Validated signal (BUY/SELL/HOLD/CLOSE/UPDATE) |
| `entry_price` | Price level for execution |
| `stop_loss` | Fixed or dynamic SL level |
| `take_profit` | Fixed or dynamic TP level |
| `position_size` | Size as fraction of portfolio (0–1) |
| `risk_reward_ratio` | Computed R:R from SL/TP |
| `confidence` | Pass-through from LLM signal |
| `reasoning` | Risk-adjusted rationale |
| `frictions` | List of blocking reasons (if signal rejected) |

### Core Logic: Dynamic SL/TP Scaling

```
Default SL = ATR × 2 (tight 2:1 R/R baseline)
Default TP = ATR × 4
AI-provided SL/TP → validated and used if reasonable
Circuit breakers:
  - SL clamped to [1%–10%] of entry price
  - SL consistency: SL must be below entry (LONG) / above entry (SHORT)
  - TP consistency: TP must be above entry (LONG) / below entry (SHORT)
  - Violations → friction recorded, dynamic default substituted
ATR fallback: if ATR unavailable → 2% of current price

Volatility classification (embedded in RiskAssessment + friction metadata):
  - ATR > 3% → HIGH
  - ATR < 1.5% → LOW
  - 1.5%–3% → MEDIUM
```

### R:R Enforcement — Two-Layer Defense

RiskManager computes `rr_ratio` = TP distance / SL distance but does **NOT** reject on it.
TradingStrategy enforces brain-learned `rr_borderline_min` (default **1.5**, adaptively learned from historical R:R performance):

```python
if rr_ratio < brain_thresholds.get("rr_borderline_min", 1.5):
    # Blocked as guard_type="rr_minimum" → stored as blocked-trade feedback
```

### Friction Lifecycle

1. `RiskManager.calculate_entry_parameters()` → accumulates frictions in `_last_frictions` list during SL/TP clamping
2. `TradingStrategy._open_new_position()` → calls `get_and_clear_frictions()` after RiskAssessment
3. Each friction → `vector_memory.store_blocked_trade()` → Brain Agent learns from clamping events

### Position Sizing

| Confidence Level | Fallback Size |
|-----------------|---------------|
| HIGH (string; numeric extractor maps ≥70) | 3% |
| MEDIUM (numeric extractor maps 50–69) | 2% |
| LOW (numeric extractor maps <50) | 1% |
| Max position | 10% of portfolio (configurable) |

### Edge Cases & Guardrails

| Scenario | Handling |
|----------|----------|
| **AI signal missing required fields** | Returns `RiskAssessment` with frictions list, blocked-trade feedback |
| **SL/TP outside clamped range** | Clamped to [1%–10%], friction logged |
| **No position for UPDATE signal** | Friction: "No existing position to update" |
| **ATR unavailable** | Falls back to percentage-based SL (2% of current price) |
| **Brain thresholds unavailable** | Uses config defaults from `config.ini` |
| **Invalid configured fallback size** | Falls back to configured MEDIUM size and logs warning |
| **R:R below minimum** | Enforced in TradingStrategy (not RiskManager) — blocked as `guard_type="rr_minimum"` with brain-learned default 1.5 |
| **SL on wrong side of entry** | SL above entry for BUY / below entry for SELL → dynamic SL substituted, friction logged |

---

## 2. Persistence Manager Agent

> **Files:** `src/managers/persistence_manager.py`, `src/managers/sqlite_trade_history.py`

### Agent Persona & Role

Persistence Manager is the **single persistence facade for trading runtime state.** It owns positions, statistics, monitor state, previous/last analysis snapshots, and trade-history access. Trade history is SQLite-only and must not fall back to legacy JSON files.

### SQLite Trade History Contract

| Responsibility | Owner | Rule |
|----------------|-------|------|
| Trade append | `PersistenceManager.save_trade_decision()` → `SQLiteTradeHistory.insert()` | SQLite-only; raise if persistence fails |
| Full history export | `PersistenceManager.load_trade_history()` | Export from SQLite only |
| Entry-decision lookup | `get_entry_decision_for_position()` | Query SQLite by timestamp window and optional symbol |
| Cooldown timestamp | `get_last_execution_timestamp()` | Query newest BUY/SELL timestamp from SQLite |
| Dashboard history | Dashboard routers via injected persistence | No direct file reads |

### Non-Negotiable Rules

- Do not reintroduce `trade_history.json` runtime reads, writes, fallback paths, or auto-migration.
- Do not pass `json_path` into `SQLiteTradeHistory`; its constructor accepts only `logger` and `db_path`.
- Historical `.json.migrated` files are backups only and are not runtime inputs.
- SQLite write failure is a hard persistence failure; do not silently continue with an alternate store.
- Keep all service dependencies injected from the composition root; do not construct persistence dependencies inside trading, dashboard, or guard classes.

### SQLite Store Behavior

- WAL journal mode and `synchronous=NORMAL` are enabled per connection.
- Inserts coerce `TradeDecision` fields into stable SQLite column types.
- Queries validate sort direction (`ASC`/`DESC`) and clamp pagination.
- `get_stats()` provides aggregate dashboard data without scanning JSON files.

### Edge Cases

| Scenario | Handling |
|----------|----------|
| **No trade history rows** | Returns empty list / `None` timestamp; callers decide no-history behavior |
| **Invalid query order** | Raises `ValueError` before SQL generation |
| **SQLite insert returns no row id** | `PersistenceManager.save_trade_decision()` raises `RuntimeError` |
| **Entry timestamp not found** | Returns `None` and logs warning |
| **Malformed persisted timestamp** | Entry lookup skips malformed candidate rows |

---

## 3. Provider Orchestrator Agent

> **File:** `src/managers/provider_orchestrator.py`

### Agent Persona & Role

The Provider Orchestrator manages the **AI provider lifecycle and fallback chain** — routing analysis requests to the best available LLM provider with automatic degradation when the primary provider is unavailable, rate-limited, or returning errors.

### Provider Registry

| Name | Client | Default Model | Chart Support | Fallback Model |
|------|--------|---------------|---------------|----------------|
| `googleai` | `GoogleAIClient` | Google Gemini 3.5 Flash | ✅ Yes | Google paid tier |
| `openrouter` | `OpenRouterClient` | Configurable base model (`google/gemini-3-flash-preview` by default) | ✅ Yes | `deepseek/deepseek-r1:free` by default |
| `local` | `LMStudioClient` | LM Studio model | Disabled in orchestrator | — |

### Fallback Chain Strategy

**Text requests:**
```
Primary: googleai (Gemini 3.5 Flash)
  → Rate limited / overloaded? → googleai paid tier (auth errors fall through to next provider)
  → Still failing? → local (LM Studio, text-only path)
  → Still failing? → openrouter (configured base model)
  → Still failing? → openrouter fallback model
  → All providers failed? → HOLD with error
```

**Chart (multimodal) requests:**
```
Primary: googleai (Gemini 3.5 Flash)
  → Rate limited / overloaded? → googleai paid tier (auth errors fall through to next provider)
  → Still failing? → openrouter (configured base model)
  → Still failing? → openrouter fallback model
  → All providers failed? → HOLD with error
```
Note: `local` is skipped in the orchestrator's chart fallback chain because provider metadata marks it as chart-unsupported, even though the LM Studio client has a chart-analysis method.

### Key Behaviors

- **Parameter auto-retry:** provider clients use `_execute_with_param_retry()` to catch unsupported parameter errors, strip the offending parameter, and retry (up to 3×)
- **Known unsupported params** filtered pre-emptively: `thinking_budget`, `thinking_config`, `top_k`, `freq_penalty`, `pres_penalty`
- **Chart analysis routing:** Only `googleai` and `openrouter` are enabled for orchestrated multimodal (text + image) requests; explicit `local` chart requests return an error before fallback
- **Cost tracking via OpenRouter** `get_generation_cost()` — async query after response for token + cost data
- **Error classification:** quota, auth, timeout, overloaded, connection — each maps to a specific error response the caller uses for fallback decisions
- **API key redaction:** `_sanitize_error_message()` redacts API keys from all error logs

### Client Implementations

| Client | Transport | Image Support | Retry |
|--------|-----------|---------------|-------|
| `GoogleAIClient` | Official `google.genai` SDK | Base64-encoded inline | `@retry_api_call` |
| `OpenRouterClient` | Official `openrouter` SDK (`OpenRouter`) | Base64 data URI | `@retry_api_call` + param retry |
| `LMStudioClient` | Official LM Studio Python SDK (`lmstudio.AsyncClient`) | Available in client, but disabled by orchestrator metadata and skipped by the chart fallback chain | `@retry_api_call` |

### Edge Cases

| Scenario | Handling |
|----------|----------|
| **Rate limited** → retry with backoff | Exponential backoff: 1s → 2s → 4s → ... → 30s max, 3 retries |
| **Unsupported parameter** → retry without it | Detected via error message regex, stripped from config, retried |
| **SDK version mismatch** | OpenRouter: falls back to `OpenRouter(api_key=...)` without `server_url` |
| **All providers fail** | Returns HOLD signal via fallback response |
| **Chart provider disabled for images** | Orchestrator returns an error for `local`; chart fallback chain only includes `googleai` and `openrouter` |
| **Non-text response parts** (Google) | Silently filtered, text parts extracted |
| **Provider not in registry** | `get_metadata()` returns None → caller uses provider not found error |
