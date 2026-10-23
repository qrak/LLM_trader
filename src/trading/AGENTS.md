# AGENTS.md — Trading Brain & Strategy

Scope: `src/trading/`. Root [`AGENTS.md`](../../AGENTS.md) is canonical.

## Execution quality and sandbox isolation

- Account for proven venue executions even when market-data verification is missing or inconsistent; never invent a real-market exit to replace a sandbox fill. Learning needs verified exit evidence with execution-time reference provenance **and** a verified executor entry — disabled, unavailable, legacy and status-only evidence cannot authorize a lesson.
- Persist entry verification in `Position.entry_execution_quality` and structured `TradeDecision.execution_quality` in SQLite, kept through restart and entry settlement. Statistics, prompt totals and equity curves read that same structured eligibility, never caller reason text; a quarantined post-mortem must not advertise a successful plan or strategy P&L.
- Command references are persisted for replay; unknown execution time is not detection/creation time. Historical repairs need a fingerprinted dry-run, backup, stopped processes and a verified second-run no-op; retain raw executions and reset untraceable rule-validation feedback explicitly.

## Exit ownership: real price, venue backstop

- The bot reads the REAL market (`[exchanges] sandbox = false`) while the executor's protective orders rest on the testnet book. With the executor live the exchange leg is a CATASTROPHE backstop (executor `PROTECTION_BACKSTOP_MULTIPLIER`, default 2.5x the stop distance, floor 1.5% of entry; the target leg is untouched) — the stop that actually runs is `PositionStatusMonitor.check_reference_exits()` on `[risk_management] reference_exit_check_seconds` (default 10 s, `0` disables, otherwise 2–120 s). A hit is forwarded to the executor at once through `CryptoTradingBot.forward_local_exit_request()`, retried every 60 s with the same `order_id` (the executor dedups on it), and alerted once per (position, reason).
- `TradingStrategy.check_reference_exit()` is side-effect free by contract: no live metrics update, no position persist, no booking — it registers the same CLOSE intent as every other local exit condition and returns None (Wave 3). A check every few seconds must never become a write every few seconds, and only fill evidence closes the trade.
- No real price → no check: warn once per 5 min and leave the position to the exchange backstop. The two switches are a pair — disabling this check while the executor still places a widened leg leaves the position with a backstop and no stop (ADR 0005).
- `[exchanges] sandbox = true` makes this "real" check measure the testnet book; `start.py` warns at startup when that is combined with a live executor.

## Brain (`brain.py` facade)

No LLM calls: a deterministic vector-memory service whose context is injected into the analysis prompt. It never executes trades or computes indicators.

Collaborators: `BrainContextProvider` (prompt block + blocked feedback) · `BrainExperienceRecorder` (trade → experience) · `ExitProfileResolver` (sole SL/TP profile serializer) · `TradePatternAnalyzer` (grouping, diagnostics, AI-mistake classes) · `BrainReflectionEngine` (rule synthesis) · `ExecutorHandler` (payload + forward), files `brain_context.py`, `brain_experience.py`, `brain_exit_profiles.py`, `brain_patterns.py`, `brain_reflection.py`, `executor_handler.py`.

In: `Position` (entry/exit/P&L, confidence, confluence, drawdown), `close_price`, `close_reason`, the entry `TradeDecision` from SQLite via `PersistenceManager`, `MarketConditions`. ChromaDB stores 20+ fields per trade (confidence, reasoning, indicators, volatility, SL/TP distance, R:R, drawdown/profit, sentiment, regime, confluence, alignment, exit context, factor scores) plus rules, blocked feedback and calibration.

`get_context()` renders calibration by level, bias with `LIMITED DATA`, the last 5 blocked trades (≤168h), the **top-3** similar experiences, CoT Step 6 (win rate <50% → lower confidence, anti-pattern/AI-mistake/exit memory), rules matched to conditions (similarity %, freshness, evidence, type), and up to five post-mortems from the shared `PostMortemRepository` — `_provision_trading_layer` must inject the *same* repository into `TradingBrainService` and `PostMortemService`; test real provisioning and prompt assembly, not an isolated constructor.

Context rebuilds per request on the shared `PromptBudgetPolicy` + `TokenCounter`; the brain's default emergency budget is 50,000 estimated tokens, not a character cutoff (provider-side mechanics: `managers/AGENTS.md`). Never trim or partially cut recent lessons, rejection feedback or risk guidance — raise `PromptBudgetExceeded` before the model call instead, and log estimated tokens on every reduction. Keep 3 experiences, 3 rules, 5 lessons, 5 rejection samples; stats never enumerate history. Test with >64,000 characters under budget plus full lesson retention.

`get_dynamic_thresholds()` exposes ~15 learned parameters: R/R floor, ADX thresholds, SL tightening progress, size limits, confluence minimums, timeframe-alignment coefficients.

- `safe_mae_pct` = largest signed `max_drawdown_pct` magnitude among WIN experiences, percentage points converted to a fraction. Require ≥5 valid wins by default; exclude missing, malformed, non-finite, boolean and positive drawdowns, LOSS and UPDATE entries. Zeros count — a learned zero is not insufficient data. Carry `safe_mae_sample_count`; label the value historical evidence, never a guaranteed drawdown or hard SL floor.
- When extracting `MarketConditions`, pass the already-read `_portfolio_pnl_pct` to the constructor and verify positive/negative/zero/absent through Position serialization, real isolated Chroma storage and prompt rendering. Portfolio return and the trade's `pnl_pct` stay separate; never overwrite historical snapshots with current equity.
- Closed experiences persist `position.trend_direction_at_entry` apart from exit-time `market_regime`. Dashboard S/R stats read `support_resistance_strength_score` while keeping API bucket names and older `support_resistance_score` rows; do not migrate live memories or infer entry trends from exit conditions.

Rule creation: best-practice ≥5 wins, ≥3 same-pattern, ≥60% win rate → `rule_best_*`; anti-pattern ≥3 losses, ≥2 same-pattern, ≥60% loss rate → `rule_anti_*` with failure_reason + recommended_adjustment; corrective ≥3 losses, ≥2 same-pattern, <60% loss rate → `rule_corrective_*`; AI-mistake ≥2 classified → `rule_ai_mistake_*` with failed_assumption + mistake_type.

Reflection cadence: scalping (≤30 min) every 10 trades, intraday (60–239) 7, swing (240–1439) 5, position (≥1440) 3. Three loops then run **sequentially**: `trigger_reflection()`, `trigger_loss_reflection()`, `trigger_ai_mistake_reflection()`.

Injected after technicals, before the final decision instruction: calibration → bias → blocked feedback → top-3 experiences → matched rules → CoT Step 6. Embedding: `build_rich_context_string()` for storage, `build_query_document()` for queries (same shape plus raw numbers, less asymmetry), hybrid retrieval 70% similarity + 30% recency decay.

Rules are durable policy, not examples: kept while active but soft-ranked by similarity, evidence quality (`wins`, `losses`, `source_trades`, `expectancy_pct`, `profit_factor`), freshness (~14-day half-life at 4h) and a contradiction penalty. Old rules are never auto-deleted; they degrade to `maturing`, `stale`, `legacy` unless revalidated.

`confidence_threshold` is learned in `VectorMemoryAnalytics`: HIGH win rate >70% → 65, <55% → 75, else 70. It shapes self-assessment only — sizing uses the HIGH/MEDIUM/LOW string against configured fallbacks.

## SL tightening

`StopLossTighteningPolicy` gates tightening behind progress toward TP: scalping (<1h) 25%, intraday (1h–4h) 20%, swing (4h–1d) 15%, position (>1d) 10%. `resolve_effective_threshold()` starts from that base, uses a learned threshold with enough samples (clamped to the configured floor/ceiling), and exposes `sl_tightening_pct`, `sl_tightening_source` and the nested payload.

Learning is recency-weighted, not causal: one observation per closed position (smallest progress, canonicalized through its close record), never counting UPDATE aliases as trades. Age it from that UPDATE's timestamp; require a valid non-future UPDATE followed by a valid close. Apply the timeframe half-life and max age in fractional days (4h: 14/56), excluding invalid, non-finite, boolean and expired evidence. Report raw and effective counts `(sum(w) ** 2) / sum(w ** 2)`; weight expectancy, win rate and conditional win/loss means consistently. Activation needs both counts ≥ `SL_TIGHTENING_MIN_SAMPLES`; missing effective evidence means configuration, not an old learned value. `BrainContextProvider` invalidates stats after 300 monotonic seconds even when the experience count is unchanged; `TradingBrainService` copies the nested payload before adding display fields. Regression coverage needs real isolated Chroma storage, time-only expiry and the resulting LONG/SHORT prompt. No intrabar replay, alternative-exit comparison, cost-aware optimization or out-of-sample validation exists yet.

No minimum interval between position updates: each valid UPDATE may be processed immediately, consecutive cycles included. Executor-state checks, tightening/widening policy and receipts are unchanged.

## Updates, refusals, reconciliation

- Executor connection failures and timeouts are availability warnings, not `CRITICAL`. `_executor_has_position()` returns `None`, clears stale query evidence and keeps the local position; reconciliation stays unverified and UPDATE/CLOSE still need confirmation. No traceback on this expected offline path, no hiding real payload failures behind the same warning.
- Every refused command — a blocked UPDATE (tightening gate, widening cap) or an UPDATE/CLOSE skipped for unverified executor state — is recorded: `PositionManagementMixin._record_refused_command()` writes a position intent with `state=refused`, `evidence=bot_policy` in `data/trading/bot_position_intents.jsonl` and queues a one-shot alert (`take_rejected_intent_alert()` → `CryptoTradingBot._report_rejected_intent()`). `TradingStrategy.get_position_context()` renders the newest intent as `### Last Action Outcome`, so the next analysis (one per timeframe, the only place a refusal can change a verdict) sees the requested level, the level in force, and that nothing was sent.
- UPDATE/CLOSE forwarding allows 30s for queue delay plus execution; a missing verdict is UNKNOWN, not success. Reconciliation and the pre-analysis hook revisit pending/unknown UPDATE receipts without resending; match the exact order and trade, persist corrected levels on success or refusal, never let an older receipt overwrite a newer UPDATE. Prompt outcomes are scoped to the current trade and ordered by command creation, not receipt arrival; a confirmation supersedes an earlier missing-confirmation alert.
- A queued CLOSE may settle after the bot's journal read, and a market-order reply may prove quantity while omitting average price. The executor re-reads that order before writing the verdict; `book_executor_side_exit()` revisits a matching pending/unknown CLOSE during reconciliation and books only a fully validated fill receipt. No price or quantity → keep the local position and alert; never infer P&L from entry price or an `executed` flag.

## Executor forward pipeline

`TradingStrategy.process_analysis()` runs guards, `RiskManager` and the R/R gate; a block stores a blocked-trade friction (`system_constraints_rejections`) and returns `TradeDecision(action="HOLD")`. `CryptoTradingBot._execute_trading_check()` then patches the saved response (`_patch_rejected_signal_in_response()`) so the next prompt sees the `HOLD` instead of a fill that never happened, saves via `_save_analysis_data()`, and calls `await self.executor_handler.handle(analysis, decision, symbol)`.

`ExecutorHandler(persistence, config, logger)` builds the payload (`None` on HOLD veto), persists `latest_decision.json` atomically, HTTP-POSTs to `llm_trader_executor` (`[executor_api] enabled`, `url = http://127.0.0.1:9199/decision`). Wired once in `start.py build_dependencies()`, not in `app.py` constructors. Register `ExecutorHandler.close` in `CryptoTradingBot.initialize` *after* the shutdown callback so the HTTP client closes on normal exit and in-place reload; test the callback itself, including an earlier callback failure, an unused handler and repeated shutdown.

Blocked trades: BUY below the R/R floor → strategy rejects → `store_blocked_trade(guard_type="rr_minimum", …)` → HOLD → no forward; next cycle `get_blocked_trade_feedback(n=5, max_age_hours=168)` renders `## CRITICAL FEEDBACK: System Rejections`, alongside the model's own patched HOLD.

`prune_aged_documents()` may drop stale experiences and blocked feedback, but active rules (`active=True`) are never deleted by age — only down-ranked. Matching closed trades update `validation_hit_count`, `last_validated_at`, `contradiction_count`, `last_contradicted_at`. Timestamp parsing must be datetime-aware so malformed values are skipped instead of corrupting prune decisions.

Persistence: decisions go through `PersistenceManager.async_save_trade_decision()`, never trade-history files; `trade_history.db` is the only store — no `trade_history.json` readers, fallback writes or auto-migration. Close-time learning recovers the entry decision by SQLite timestamp window with optional symbol.

## Degradation

ChromaDB unavailable → `get_context()` returns an empty string (brain section skipped) and reflection is skipped with a warning; neither ever crashes the trading loop, and blocked-feedback failures are caught and passed. Reflection also skips when there is not enough data (<5 wins, <3 losses, <2 mistakes) and a best-practice rule needs ≥60% win rate. `timeframe_minutes ≤ 0` → 240. The trade counter is re-read from vector memory on init (cross-restart); stats invalidate on count change or 300 monotonic seconds. Unknown exit profiles → `UNKNOWN_EXIT_PROFILE` sentinel default. `get_vector_context()` serves similarity results plus contextual stats; `close_position()` is the learning entry point.

## Regression contracts

- Existing-position handling rechecks `current_position` after executor verification and after an awaited protection update; a closed local position produces no stale CLOSE/UPDATE decision, intent, history row or learning event.
- `PositionManagementMixin` declares its sibling-method callable contracts; the shared static commission formatter lives there, stays inherited by `TradingStrategy`, never duplicated or wrapped.
- `executor_handler.py` changes need thorough tests: an UPDATE carrying `quantity=0.0` must use the tracker's real quantity; market orders carry `entry_price=None`; a spot SELL is blocked while `SPOT_SHORT_ALLOWED=False` and that rejection is logged, never silent.
- Verify read/write round-trips after any data-model or persistence change.
- The executor keeps its file fallback — that is what makes a restart reliable even when HTTP works.
- Decision semantics stay put: `SafetyGuard` deduplicates by `order_id`, the executor executes, positions persist.
