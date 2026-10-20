# AGENTS.md — Managers

Scope: `src/managers/`. Root [`AGENTS.md`](../../AGENTS.md) is canonical. Three services: `risk_manager.py`, `persistence_manager.py` + `sqlite_trade_history.py`, `provider_orchestrator.py`. All dependencies arrive from the composition root — never construct one inside trading, dashboard or guard code.

## Risk Manager

Signal safety layer: raw LLM signal → validated `RiskAssessment`. Execution is delegated: the bot forwards the decision to `llm_trader_executor` (`[executor_api] enabled`), which places real orders via CCXT; with `enabled = false` nothing leaves the process. Never describe this layer as paper-only.

In: `TradingAnalysisModel` (signal, confidence, `entry_price`, `stop_loss`, `take_profit`, `position_size`), `MarketConditions` (price, ATR, volatility), the existing `Position` for UPDATE/CLOSE, brain thresholds from `get_dynamic_thresholds()`. Out: `RiskAssessment` carrying the validated signal, `entry_price`, `stop_loss`, `take_profit`, `position_size` (0–1 portfolio fraction), `risk_reward_ratio`, `confidence`, `reasoning` and `frictions` (blocking reasons when rejected).

```
SL default = ATR × 2, TP default = ATR × 4; AI-provided levels are used when reasonable
SL clamped to [1%, 10%] of entry; TP distance clamped to 50%
SL below entry (LONG) / above (SHORT); TP above (LONG) / below (SHORT)
violation → friction recorded + dynamic default substituted
ATR unavailable → 2% of price
ATR > 3% → HIGH volatility; < 1.5% → LOW; else MEDIUM (kept in friction metadata)
```

A regime profile cap may only tighten `MAX_POSITION_SIZE`, never raise it; an invalid cap raises `ValueError`.

R:R is computed, never rejected here. The floor is `max(MIN_RR_ENTRY, brain rr_borderline_min)`, both default 0.0 (`rr_policy.py`); the brain may raise the configured floor, never lower it, and only negative expectancy over ≥10 closed trades below 1.0 R/R can push it to 0.5 or 1.0. Enforcement happens in `TradingStrategy`, which records `guard_type="rr_minimum"` as blocked-trade feedback.

Friction lifecycle: `RiskManager.calculate_entry_parameters()` accumulates into `_last_frictions`, `TradingStrategy._open_new_position()` calls `get_and_clear_frictions()`, each friction becomes `vector_memory.store_blocked_trade()` so the brain learns from clamping.

Sizing: HIGH (≥70) 3%, MEDIUM (50–69) 2%, LOW (<50) 1%, hard cap 10%. An invalid configured fallback size falls back to the MEDIUM value and logs a warning. Missing fields → frictions, no exception. UPDATE with no position → "No existing position to update". Brain thresholds missing → `config.ini` defaults.

## Persistence Manager

Single persistence facade: positions, statistics, monitor state, previous/last analysis snapshots, trade history. Trade history is SQLite-only, no legacy JSON path.

- Append: `save_trade_decision()` → `SQLiteTradeHistory.insert()`; failure raises (`RuntimeError` when no row id) — never continue on another store.
- `load_trade_history()` exports from SQLite; `get_entry_decision_for_position()` queries by timestamp window plus optional symbol, skips malformed rows, returns `None` + warning when absent.
- Dashboard routes reach history only through the injected `PersistenceManager`.
- Do not reintroduce `trade_history.json` reads, writes, fallbacks or auto-migration; `*.json.migrated` files are backups, not inputs. Do not pass `json_path` — the constructor takes `logger` and `db_path` only.
- WAL + `synchronous=NORMAL` per connection; inserts coerce `TradeDecision` fields to stable column types; queries validate `ASC`/`DESC` and clamp pagination (`ValueError` before any SQL on a bad direction).

## Provider Orchestrator (`provider_orchestrator.py`)

`PromptBudgetPolicy` is built once (`_provision_utilities`) with the shared `TokenCounter` and injected into the brain and `ProviderOrchestrator`. Before **every** client call — model override, provider fallback, OpenRouter internal fallback, local streaming — budget the actual model against its exact configured context-window key: reserve output, framing, chart tokens and safety margin, and enforce the whole-request ceiling. Unknown capacity or a missing/invalid output reserve fails **without an API call**. Reduce only complete optional sections with unambiguous owned headers; duplicate or forged headers must not delete protected position/risk/schema/lesson sections; never mutate messages shared with another fallback model. Log `Full prompt content` at the prepared-request boundary, after reduction.

| Provider | Client | Default model | Chart |
|---|---|---|---|
| `googleai` | `GoogleAIClient` | Gemini 3.6 Flash | yes, paid-tier fallback |
| `openrouter` | `OpenRouterClient` | `google/gemini-3-flash-preview` | yes, fallback `deepseek/deepseek-r1:free` |
| `local` | `LMStudioClient` | LM Studio model | none — skipped by the chart chain |

Text: googleai → googleai paid tier → local → openrouter base → openrouter fallback → HOLD. Chart: googleai → paid tier → openrouter base → fallback → HOLD (auth errors fall through to the next provider; local is skipped because metadata marks it chart-unsupported even though the client can send images). An explicit `local` chart request returns an error before fallback.

Clients: `GoogleAIClient` (official `google.genai` SDK, base64 inline images), `OpenRouterClient` (official `openrouter` SDK, data-URI images), `LMStudioClient` (`lmstudio.AsyncClient`, image-capable but chart-disabled by metadata) — all `@retry_api_call`, OpenRouter also param-retry.

- `_execute_with_param_retry()` strips an unsupported parameter and retries up to 3× (backoff 1→2→4s, 30s cap). Filtered pre-emptively: `thinking_budget`, `thinking_config`, `top_k`, `freq_penalty`, `pres_penalty`.
- Error classification: quota, auth, timeout, overloaded, connection — each maps to a distinct response the caller uses for fallback. `_sanitize_error_message()` redacts API keys in every error path.
- Cost via OpenRouter `get_generation_cost()` after the response. Google non-text parts are filtered silently. `get_metadata()` returns `None` for an unregistered provider → caller emits provider-not-found. All providers down → HOLD.
