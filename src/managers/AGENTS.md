# AGENTS.md — Managers

Scope: `src/managers/`. Root [`AGENTS.md`](../../AGENTS.md) is canonical. Services: `risk_manager.py`, `persistence_manager.py` + `sqlite_trade_history.py`, `provider_orchestrator.py` + `model_manager.py` + `provider_types.py`, `post_mortem_repository.py`. `cycle_metrics.py` is the append-only observability journal. All dependencies arrive from the composition root — never construct one inside trading, dashboard or guard code.

## Risk Manager

Signal safety layer: raw LLM signal → validated `RiskAssessment`. Execution is delegated: when `[executor_api] enabled = true` the bot forwards the decision to the separate executor service, which is not part of this repository and runs against an exchange testnet in the public deployment. With `enabled = false` nothing leaves the process and the book is simulated. Describe the mode from the config value, not from this file.

In: `TradingAnalysisModel` (signal, confidence, `entry_price`, `stop_loss`, `take_profit`, `position_size`), `MarketConditions` (price, ATR, volatility), the existing `Position` for UPDATE/CLOSE, brain thresholds from `get_dynamic_thresholds()`. Out: `RiskAssessment` carrying the validated signal, `entry_price`, `stop_loss`, `take_profit`, `position_size` (0–1 portfolio fraction), `risk_reward_ratio`, `confidence`, `reasoning` and `frictions` (blocking reasons when rejected).

```
SL default = ATR × 2, TP default = ATR × 4 (`risk_manager.py`); AI-provided levels are used when reasonable
SL clamped to [1%, 10%] of entry; TP distance clamped to 50%
SL below entry (LONG) / above (SHORT); TP above (LONG) / below (SHORT)
violation → friction recorded + dynamic default substituted
ATR unavailable → 2% of price
ATR > 3% → HIGH volatility; < 1.5% → LOW; else MEDIUM (kept in friction metadata)
```

A regime profile cap may only tighten `MAX_POSITION_SIZE`, never raise it; an invalid cap raises `ValueError`.

R:R is computed, never rejected here. The floor is `max(MIN_RR_ENTRY, brain rr_borderline_min)`, both default 0.0 (`rr_policy.py`); the brain may raise the configured floor, never lower it, and only negative expectancy over ≥10 closed trades below 1.0 R/R can push it to 0.5 or 1.0. Enforcement happens in `TradingStrategy`, which records `guard_type="rr_minimum"` as blocked-trade feedback.

Friction lifecycle: `RiskManager.calculate_entry_parameters()` accumulates into `_last_frictions`, `TradingStrategy._open_new_position()` calls `get_and_clear_frictions()`, each friction becomes `vector_memory.store_blocked_trade()` so the brain learns from clamping.

Sizing fallback when the model gives no `position_size`: by confidence, `position_size_fallback_high/medium/low` (example config: 3% / 2% / 1%), capped by `max_position_size` (example config: 10%). An invalid configured fallback size falls back to the MEDIUM value and logs a warning. Missing fields → frictions, no exception. UPDATE with no position → "No existing position to update". Brain thresholds missing → `config.ini` defaults.

## Persistence Manager

Single persistence facade: positions, statistics, monitor state, previous/last analysis snapshots, trade history. Trade history is SQLite-only, no legacy JSON path.

- Append: `save_trade_decision()` → `SQLiteTradeHistory.insert()`; failure raises (`RuntimeError` when no row id) — never continue on another store.
- `load_trade_history()` exports from SQLite; `get_entry_decision_for_position()` queries by timestamp window plus optional symbol, skips malformed rows, returns `None` + warning when absent.
- Dashboard routes reach history only through the injected `PersistenceManager`.
- Do not reintroduce `trade_history.json` reads, writes, fallbacks or auto-migration; `*.json.migrated` files are backups, not inputs. Do not pass `json_path` — the constructor takes `logger` and `db_path` only.
- WAL + `synchronous=NORMAL` per connection; inserts coerce `TradeDecision` fields to stable column types; queries validate `ASC`/`DESC` and clamp pagination (`ValueError` before any SQL on a bad direction).
- `PostMortemRepository` keeps its own connection and an FTS5 virtual table beside the trade tables; existing tables are never altered, only new ones added.

## Cycle Metrics (`cycle_metrics.py`)

- `append_cycle()` writes one JSONL line per analysis cycle; the schema is open, and `load_cycles()` skips a torn last line instead of breaking the bot.
- Cost fields are recorded when the provider layer measured them: `inference_cost_usd` (delta of `CostStorage.snapshot()` around the cycle), `prompt_tokens`, `completion_tokens`, `provider`, `model`. A missing cost stays absent or `None` — never invented.
- `summarize()` returns `inference_cost` with `analyses`, `total_usd`, `per_analysis_usd` and `coverage`; non-numeric, bool, NaN, Inf and negative costs are ignored.
- Every analysed cycle also carries `confidence` and `confluence_factors` (declared model names, 0–100, missing or invalid values dropped). They are the model's opinions after parsing and the pattern-quality overwrite — never a calibrated `P(win)` and never a promise about the sample of closed trades. `scripts/snapshot_results.py` strips both from the frozen public copy.
- `CostStorage.snapshot()` is the only cumulative read used for cost accounting; `record_usage()`, `save()` and `_load_or_create()` keep their existing semantics.

## Provider Orchestrator (`provider_orchestrator.py`)

`PromptBudgetPolicy` is built once (`_provision_utilities`) with the shared `TokenCounter` and injected into the brain and `ProviderOrchestrator`. Before the client call — including a model override — budget the actual model against its exact configured context-window key: reserve output, framing and safety margin, and enforce the whole-request ceiling. Unknown capacity or a missing/invalid output reserve fails **without an API call**. Reduce only complete optional sections with unambiguous owned headers; duplicate or forged headers must not delete protected position/risk/schema/lesson sections; never mutate the caller's messages. Log `Full prompt content` at the prepared-request boundary, after reduction, tagged `skip_live_stream`: the terminal and `Bot.log` keep it, the dashboard live console does not.

| Provider | Client | Default model |
|---|---|---|
| `googleai` | `GoogleAIClient` | `google_studio_model` (default `gemini-3.8-flash`) |
| `deepseek` | `DeepSeekClient` | `deepseek_model` (default `deepseek-flash`) |
| `openrouter` | `OpenRouterClient` | `openrouter_base_model` (default `google/gemini-3-flash-preview`) |
| `opencode` | `OpenAICompatibleClient` | `opencode_model` (default `deepseek-v4.1-flash`) |
| `commandcode` | `OpenAICompatibleClient` | `commandcode_model` (default `deepseek/deepseek-v4.1-flash`) |
| `custom` | `OpenAICompatibleClient` or `AnthropicCompatibleClient` (`custom_protocol`) | `custom_model` |

Every request is text-only: the chart image is rendered for the dashboard and Discord and never reaches a provider. Exactly one provider is built — the composition root constructs only the client `provider` names — and there is no fallback to another provider, another model or a paid tier: a failed cycle becomes HOLD.

Clients: `GoogleAIClient` (official `google.genai` SDK) and `OpenAICompatibleClient` (`AsyncOpenAI` with `max_retries=0` and per-provider `default_headers`) for everything else. OpenCode Go gets `User-Agent: llm-trader/1.0` plus a process-stable `x-opencode-session` (the provider asks for both), Command Code and an openai-protocol `custom` endpoint only the `User-Agent`. `OpenRouterClient` subclasses it for `extra_body.reasoning` and the generation-cost lookup, `DeepSeekClient` for the token log and `retry_on_empty`. `AnthropicCompatibleClient` posts the Messages API over `httpx` (system messages joined into `system`, `stop_reason` mapped to `finish_reason`). Calls use the shared `@retry_api_call` policy. The OpenAI-style path also uses the unsupported-parameter retry; the Anthropic client builds a Messages API body and posts it directly. Adding an OpenAI-style provider = one config block plus one branch in `ProvisioningMixin._provider_client`.

- `_execute_with_param_retry()` strips an unsupported parameter and retries up to 3× with no sleep of its own. Filtered pre-emptively: `thinking_budget`, `thinking_config`, `top_k`, `freq_penalty`, `pres_penalty` (`OpenAICompatibleClient` also filters `openrouter_reasoning_effort`; `OpenRouterClient` keeps that key and filters `presence_penalty`).
- `@retry_api_call` default: 5 retries. A transient reply (`connection:`, `timeout:`, `overloaded:`, `rate_limit:`) waits 60 s × 4ⁿ, never past the deadline, and the fifth wait runs exactly to it; the deadline is 15 min before the next candle of the client's `retry_timeframe` (a quarter of the frame below 1 h), and no retry starts after it. Content rejections (empty, prose, truncated JSON) keep the short 1→2→4 s backoff. `authentication:` is never retried.
- Error classification: quota, auth, timeout, overloaded, connection — each maps to a distinct response the caller uses for fallback. `_sanitize_error_message()` redacts API keys in every error path.
- Cost via OpenRouter `get_generation_cost()` after the response; OpenCode Go and Command Code have no per-request billing, so their cost is the notional rate table (`model_pricing.json` + `peak_rates.py`) and tracks allowance consumption. Google non-text parts are filtered silently. `get_metadata()` returns `None` for an unregistered provider → caller emits provider-not-found. All providers down → HOLD.
