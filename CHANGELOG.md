# Changelog

## 2026-05-16 - News Token Budget Contract Alignment

### Changed

- **config/config.ini** and **config/config.ini.example**: Set `[rag] article_max_tokens` to `1000` for runtime and example defaults.
- **src/config/loader.py**: Updated `RAG_ARTICLE_MAX_TOKENS` fallback default from `256` to `1000` so missing-config behavior matches runtime target.
- **src/rag/context_builder.py**: Removed hardcoded total-context default (`2000`) from `build_context()`. When `max_tokens` is omitted, it now resolves from config as `RAG_ARTICLE_MAX_TOKENS * RAG_NEWS_LIMIT`, while preserving strict per-article limiting via `RAG_ARTICLE_MAX_TOKENS`.
- **src/rag/article_processor.py**: Removed hardcoded `[:10000]` body truncation in `detect_coins_in_article()`, so coin detection scans full article body instead of silently clipping long articles.
- **src/rag/rag_engine.py**: Clarified retrieval-limit docstring contract that default total budget is derived from per-article config and article count.

### Added

- **tests/test_article_processor_contract.py**: Regression test proving late-body ticker mentions (beyond 10k chars) are still detected.
- **tests/test_rag_engine_retrieval_contract.py**: Contract test for config-derived default retrieval budget.
- **tests/test_rag_context_builder_contract.py**: Contract test verifying `build_context()` default budget resolves from config and passes per-article cap consistently.

## 2026-05-16 - Hybrid Stop-Loss Tightening Policy

### Added

- **src/trading/stop_loss_tightening_policy.py** (new): `StopLossTighteningPolicy` — single authoritative gate for SL tightening decisions. Replaces inline executor logic with a deterministic, side-effect-free policy class. Evaluates tightening eligibility from per-timeframe config thresholds (scalping/intraday/swing/position), with optional brain override clamped within a configurable `[floor, ceiling]` range. Returns a `TighteningEvaluation` dataclass with full decision metadata.
- **config/config.ini.example**: Seven new keys in `[risk_management]` documenting `sl_tightening_scalping`, `sl_tightening_intraday`, `sl_tightening_swing`, `sl_tightening_position`, `sl_tightening_floor`, `sl_tightening_ceiling`, and `sl_tightening_min_samples`.
- **tests/test_stop_loss_tightening_policy.py** (new, 25 tests): Covers timeframe bucket defaults, `from_config()`, LONG/SHORT tightening allow/reject, zero TP distance, missing current price, brain override with sufficient/insufficient samples, floor/ceiling clamping, and all fallback paths.

### Changed

- **src/config/loader.py** and **src/config/protocol.py**: Seven new typed properties for `SL_TIGHTENING_*` config keys.
- **start.py**: Single `StopLossTighteningPolicy` instance provisioned in `_provision_trading_layer()` and injected into both `TradingBrainService` and `TradingStrategy`.
- **src/trading/trading_strategy.py**: Replaced hardcoded inline tightening progress guard with `policy.evaluate_update()`; blocked events recorded via `vector_memory.store_blocked_trade(guard_type="sl_tightening")`; blocked events now include position identity metadata (`position_id`, `position_entry_trade_id`, `position_entry_timestamp`); `_last_sl_tightening_evaluation` instance attribute captures the accepted evaluation and forwards it to `brain_service.track_position_update()`; `get_position_context()` now renders an `## SL Tightening Policy` section showing effective threshold, current progress, and eligibility.
- **src/trading/brain.py**: `track_position_update()` accepts `tightening_evaluation: TighteningEvaluation | None` and `timeframe_minutes` and forwards them to the recorder. `get_dynamic_thresholds()` now populates a nested `sl_tightening` dict with `base_threshold`, `effective_threshold`, `effective_threshold_pct`, and `source` alongside existing flat keys.
- **src/trading/brain_experience.py**: `track_position_update()` stores position identity (`position_id`, `position_entry_trade_id`, `position_entry_timestamp`), SL/TP values, distance percentages, timeframe bucket, and full policy evaluation metadata. `record_closed_trade()` stores position identity fields to enable update–close pairing. Added `_timeframe_bucket()` static helper.
- **src/trading/brain_context.py**: `get_dynamic_thresholds()` now preserves the nested `sl_tightening` payload from `compute_optimal_thresholds()`.
- **src/trading/vector_memory_analytics.py**: `compute_optimal_thresholds()` passes the full raw Chroma snapshot to the new `_learn_sl_tightening_threshold()` method. Added `_learn_sl_tightening_threshold()`: pairs accepted SL-tightening UPDATE records with their eventual WIN/LOSS close outcomes, deduplicates to one update per position, scans candidates `[0.05 … 0.40]`, selects the lowest with positive expectancy, and stores the result under `thresholds["sl_tightening"]`.
- **src/analyzer/prompts/template_manager.py**: Replaced two hardcoded `"50%+"` thresholds in the system prompt and response template with dynamic rendering from `dynamic_thresholds["sl_tightening_pct"]`; falls back to generic guidance when the key is absent.
- **src/analyzer/prompts/prompt_builder.py**: `build_system_prompt()` now receives `dynamic_thresholds` so the effective tightening threshold is visible in the system prompt.
- **tests/test_brain_integration.py**: Added `TestTrackPositionUpdateWithPolicy` (recorder forwarding) and extended `TestGetDynamicThresholds` with nested `sl_tightening` payload tests (config-source fallback, brain-source override, flat/nested key parity).
- **tests/test_vector_memory.py**: Added four tests under `TestAnalyticsAndThresholds` for `_learn_sl_tightening_threshold`: paired wins learn threshold, below min-samples no emission, no close pairs no emission, all-loss expectancy no emission.
- **tests/test_trading_strategy_branches.py**: Updated four SL tightening tests to inject `StopLossTighteningPolicy` instances; renamed `test_not_is_tightening_but_generic_change` → `test_sl_tightening_no_current_price_rejected` to match new safety-reject behavior.
- **tests/test_prompt_consistency.py**: Updated `test_update_progress_rule_has_no_competing_percentage_thresholds` to assert the static `50%+` string is gone and `"hybrid tightening policy"` is present.

## 2026-05-16 - Trading Brain Close Lifecycle and Vector Memory Hardening

### Changed

- **trading_strategy.py**: Centralized trade-decision recording so saved BUY/SELL/UPDATE/CLOSE decisions also refresh short-term trading memory.
- **trading_strategy.py**: Close-time brain learning now runs through a thread offload after the close decision, statistics rebuild, and position clear are persisted.
- **trading_strategy.py** and **dashboard_state.py**: Trade-close brain learning now emits dashboard lifecycle notifications and invalidates brain, position, rule, memory, vector, performance, and statistics caches.
- **vector_memory.py**: Added serialized embedding-model access and stricter Chroma metadata sanitation for non-finite and unsupported values.
- **dashboard/routers/brain.py**: Moved vector detail retrieval, stats, and sorting work off the async route thread and made legacy numeric metadata sorting safe.
- **dashboard/routers/brain.py**: Added brain lifecycle and refresh endpoints plus structured risk-management data in the position payload while preserving existing flat fields.
- **dashboard/static**: Added overview SL/TP execution policy badges, brain lifecycle status, vector freshness, trade-friction rendering, and safer legacy metadata formatting.
- **persistence_manager.py**: Entry-decision lookup now filters by symbol when available and chooses the nearest timestamp match; position cache updates only after successful disk writes.
- **indicator_classifier.py** and **brain.py**: Corrected `ExitExecutionContext` type contracts used by trading brain context building.
- **brain.py**: Split the monolithic trading brain into focused collaborator classes while keeping `TradingBrainService` as the stable public facade and preserving the public `vector_memory` access path.
- **brain_context.py**, **brain_experience.py**, **brain_exit_profiles.py**, **brain_patterns.py**, and **brain_reflection.py**: Extracted prompt-context/threshold lookup, experience recording, SL/TP exit-profile normalization, trade-pattern diagnostics, and semantic-rule rebuild logic into cohesive modules.
- **indicator_classifier.py**: Preserved direct mandatory config attribute access for SL/TP execution settings; tests now supply required config fields instead of relying on missing-attribute fallbacks.
- **.github/skills/refactor/SKILL.md**: Clarified refactor validation guidance to avoid repeatedly rerunning already-green suites unless relevant code changed or a final validation boundary is reached.
- **data_utils.py**: `SerializableMixin.from_dict()` now restores tuple fields from serialized lists.
- **prompt_builder.py**: Added `_previous_context_contains_stale_prompt_rules()` to detect when old prompt contract text leaks into continuity context, protecting against schema/format instructions from corrupting future analyses.

### Added

- **test_brain_integration.py** (+86): Brain lifecycle, refresh endpoints, cache bypassing, and memory refresh after saved decisions.
- **test_dashboard_brain_router.py** (+118): Risk payload, lifecycle data, confidence parsing, brain status/memory extraction from JSON snapshots.
- **test_dashboard_server_cache.py** (+33 new): Cache headers, server cache invalidation, blocked-trade routing paths.
- **test_dashboard_static_bindings.py** (+29 new): Static HTML/JavaScript bindings, SL/TP badges, lifecycle status, vector freshness rendering.
- **test_data_utils_serialization.py** (+16 new): Tuple field serialization/deserialization in `SerializableMixin.from_dict()`.
- **test_edge_cases_feedback.py** (+99): Updated to use `MarketConditions` data model for market context across edge case scenarios.
- **test_indicator_classifier.py** (+25): Mandatory config attribute access for SL/TP execution settings.
- **test_position_persistence.py** (+60 new): Entry-decision lookup with symbol filtering, cache updates after successful disk writes, write failure handling.
- **test_prompt_consistency.py** (+137 new): Comprehensive coverage for sanitization of continuity context, stale prompt-section removal, wording consistency, HOLD/UPDATE semantics.
- **test_prompt_linting.py** (+22 new): Stale prompt-rules detection and sanitization coverage.
- **test_risk_manager_frictions.py** (+23): Guard types friction generation, delta calculations, mandatory key validation.
- **test_trading_strategy_branches.py** (+111): Market conditions validation, decision-making processes, friction capture integration.
- **test_trading_strategy_frictions.py** (+4): Trading strategy friction reporting and persistence paths.
- **test_trading_strategy_process_analysis.py** (+18): Analysis processing and market condition handling.
- **test_vector_memory.py** (+47): Non-finite metadata sanitation, complex metadata handling, Chroma store validation.
- Regression coverage and fixture updates for mandatory SL/TP config contracts in indicator classification and trading-strategy friction capture.
- Dashboard regression tests for lifecycle/risk payloads, brain refresh cache invalidation, cache headers, blocked-trade routing, and static frontend bindings.

## 2026-05-15 - System-Side ADX Trend Validation and Pattern Quality Scoring

### Added

- **trend_validator.py**: New module that cross-checks LLM-reported `strength_4h`/`daily` against computed ADX indicators.
- **pattern_quality_scorer.py**: New deterministic pattern quality scoring module using 4-component scoring from actual pattern detection results.
- Integrated trend/pattern validation into `AnalysisResultProcessor._validate_llm_claims()` so validation runs on every analysis cycle.
- 38 TrendValidator tests (100% coverage) validating ADX cross-checks and strength assertions.
- 34 PatternQualityScorer tests (90% coverage) validating deterministic scoring logic.
- 36 TradingStrategy branch tests (+22pp coverage increase to 73% overall).

## 2026-05-15 - RiskManager Friction Reporting and TradingStrategy Integration Tests

### Added

- **tests/test_risk_manager_frictions.py**: Comprehensive friction reporting tests validating generation of friction reports across guard types (max_position_size, max_concurrent_positions, etc.) with correct delta calculations.
- **tests/test_trading_strategy_friction_capture.py**: Friction capture and persistence tests ensuring frictions from RiskManager are correctly stored, retrieved, and handled on storage failures.
- Enhanced existing test fixtures by adding `scoring_policy` and `enricher` mocks where necessary to support new test requirements.

### Changed

- **RiskManager**: Verified all mandatory keys are present in friction reports and that parameters are correctly propagated during friction storage.
- **TradingStrategy**: Ensured frictions are persisted correctly and failures in storage are handled gracefully with appropriate logging.

## 2026-05-15 - Prompt Consistency and Continuity Sanitization

### Changed

- **template_manager.py**: Sanitized previous-analysis continuity text so old response-format/schema instructions cannot be reintroduced under `PREVIOUS ANALYSIS CONTEXT`.
- **template_manager.py**: Previous decision snapshots now use the last valid fenced `analysis` JSON block, avoiding schema/example JSON when a response contains multiple fenced blocks.
- **template_manager.py**: Clarified that markdown-heading restrictions apply to the model output only, and made HOLD/open-position and UPDATE semantics more explicit.
- **prompt_builder.py**: Added preflight warning detection for stale prompt instructions leaking into previous-analysis context and made untrusted-context linting robust to wording variants.

### Added

- **test_prompt_consistency.py**: Regression coverage for previous-context sanitization, stale prompt-section removal, output-heading wording, HOLD semantics, and UPDATE progress wording.

## 2026-05-12 - Gemini Flash Model Profitability: SL Tightening Guards and Prompt Simplification

### Changed

- **template_manager.py**: Compressed response template by ~60 lines through markdown tables, reduced conditional JSON rules, simplified trend/position/risk/conflict sections, and streamlined output format.
- **prompt_builder.py**: Removed all-caps directives (CRITICAL→directive, MUST→must) and reduced system prompt by ~30 lines for Flash model clarity.
- **trading_strategy.py**: Added three critical code guards to prevent SL tightening death spiral: (1) reject SL tightening until price moves 15%+ toward TP (was: immediate tightening), (2) reject entries with R/R < 1.5, (3) cap UPDATE frequency: 1 per 8h for position trades, 2 per swing, 4 per scalping (was: every 4h candle).
- **risk_manager.py**: Increased minimum SL from 0.5% to 1.0% (below 4H ATR moving average) to reduce micro-losses and preserve account.
- **Config (untracked)**: Brain learning reset from 20 consecutive losing trades; soft exit policy enabled; max position size capped at 10%; increased thinking depth for Flash model.

### Fixed

- **Root Cause**: SL tightening on every update eliminated TP hits (0/20 in previous cycle, 4.2 updates/trade on average).
- **Solution Impact**: Code guards now prevent premature SL moves; UPDATE frequency cap reduces noise; R/R check prevents underwater entries; simplified prompts reduce token waste.

### Tests

- Updated `test_template_manager.py` (+61 lines) with coverage for compressed template formats, markdown table rendering, and simplified rule sections.

### Verified

- All 56 tests pass after template simplification and guard implementation.
- No regressions in entry/exit logic validation.
- Prompt simplifications reduce token usage for more efficient model processing.

## 2026-05-14 - Code Quality: Eliminate Dynamic Attribute Access and Improve Type Safety

### Changed — Dynamic Attribute Access (Phase 1)

- **prompt_builder.py**: Removed `_detect_minimal_context()` and `_minimal_context` flag entirely. All prompt sections now render unconditionally — the feature was dead code (MINIMAL_CONTEXT config key never existed). Removed `getattr`/`hasattr` calls on config.
- **notifier.py**: Replaced `getattr(exc, "status", None)` with `isinstance(exc, discord.HTTPException)` type-narrowing.
- **crawl4ai_enricher.py**: Replaced all 6 `getattr()` calls with direct attribute access. Removed `_markdown_value()` helper. Crawl4AI objects have a documented API contract.
- **dashboard/server.py**: Replaced `getattr(response, "body", b"")` with try/except AttributeError.
- **dashboard/routers/brain.py**: Replaced `getattr(self.config, "TIMEFRAME", "unknown")` with direct access + try/except fallback (2 sites).
- **rss_primitives.py**: Removed 14 decorative section separator headers.
- **crawl4ai_enricher.py**: Removed 4 decorative section separator headers.
- **test_discord_notifier_rate_limit.py**: `FakeDiscordHTTPError` now inherits from `discord.HTTPException`, matching production hierarchy.

### Changed — Modern Typing (Phase 2)

- **config/protocol.py**: `Dict[str, str]` → `dict[str, str]`, `Dict[str, Any]` → `dict[str, Any]`. Removed unused `Dict` import.
- **contracts/model_contract.py**: Full modernization. `Optional[str]` → `str | None`, `List[Dict[str, str]]` → `list[dict[str, str]]`, `Union[io.BytesIO, bytes, str]` → `io.BytesIO | bytes | str`, `Tuple[str, str]` → `tuple[str, str]`. Removed `Optional`, `Union`, `Tuple`, `Dict`, `List` imports.
- **config/loader.py**: `Dict[str, str]` → `dict[str, str]`, `Dict[str, Any]` → `dict[str, Any]`. Removed unused `Dict` import.
- **src/app.py**: All 11 `Optional[X]` → `X | None`, all `Dict[str, Any]` → `dict[str, Any]`. Removed `Optional` and `Dict` imports.

### Removed

- `_detect_minimal_context()` and `_minimal_context` from `PromptBuilder`.
- `_markdown_value()` from `crawl4ai_enricher.py`.
- 18 decorative section separator headers across 2 files.

### Changed — Dependency Injection (Phase 3)

- **context_builder.py**: `ArticleScoringPolicy` is now injected via constructor (`scoring_policy` parameter) instead of being constructed internally from `config`. The composition root (`start.py`) now creates and wires it.
- **rss_provider.py**: `Crawl4AIEnricher` is now injected via constructor (`enricher` parameter) instead of being constructed internally from `config` values. The composition root now creates and wires it.
- **start.py**: Added construction of `ArticleScoringPolicy` and `Crawl4AIEnricher` in the DI wiring layer, passed to `ContextBuilder` and `RSSCrawl4AINewsProvider` respectively. Removed stale duplicate import.
- **test_rss_provider_contract.py**: Added `enricher=MagicMock()` to `_make_provider()` fixture.
- **test_rag_context_builder_contract.py**: Added `scoring_policy=MagicMock()` to `_builder()` fixture and inline construction site.

### Changed — Modern Typing: Full Codebase Sweep (Phase 5)

- **src/platforms/** (12 files): `Dict`/`List`/`Optional`/`Tuple` → `dict`/`list`/`| None`/`tuple` in all AI provider clients, exchange wrappers, and API clients.
- **src/trading/** (12 files): Full modernization of trading layer — brain, strategy, memory, exit monitor, position management, statistics, vector memory.
- **src/managers/** (5 files): Model, persistence, provider orchestrator, types, risk manager. Added `from __future__ import annotations` where forward refs with `|` needed.
- **src/notifiers/** (5 files): Base, console, Discord, file handler, and components.
- **src/parsing/** (1 file): Unified parser.
- **src/analyzer/** (19 files): Analysis engine, context, result processor, data fetcher, formatters, pattern engine, prompt builder, template manager, context builder, technical calculator. Added `from __future__ import annotations` for forward ref support.
- **src/dashboard/** (6 files): State, routers (brain, monitor, performance, visuals, websocket).
- **src/factories/** (2 files): Position and provider factories.
- **src/indicators/** (4 files): Base indicators, technical indicators, support/resistance, volatility.
- **src/rag/** (15 files): Article processor, category processor, collision resolver, context builder, file handler, index manager, market components, news manager, RAG engine, scoring policy, ticker manager. Added `from __future__ import annotations` where needed.
- **src/contracts/** (1 file): Risk contract.
- All bare `Dict`/`List`/`Tuple` type references (without subscript) converted to `dict`/`list`/`tuple`.

### Verified

- Entire codebase imports cleanly.
- 400 of 406 tests pass; 6 failures are pre-existing (confirmed via git stash).
- No `getattr()`/`hasattr()`/`setattr()` in application code.
- No Pydantic v1 patterns.
- No unused imports.
- Old-style typing (`Optional[X]`, `Dict[K,V]`, `List[X]`, `Tuple[X]`, bare `Dict`/`List`/`Tuple`) eliminated from all source files.
- `from __future__ import annotations` added to 5 files where forward references required it.

### Verified

- 0 `getattr()`/`hasattr()`/`setattr()` in application code.
- 400 of 406 tests pass; 6 failures are pre-existing.
- No Pydantic v1 patterns found.
- No unused imports.
- 33 files modified across all 4 phases, 0 regressions.

### Changed — Redundant isinstance Cleanup (Phase 6)

- **unified_parser.py**: Removed 2 defensive `isinstance(data, dict)` guards in `_normalize_numeric_fields()` and `_attach_response_validation()`. Both functions are typed as `dict[str, Any]` — the isinstance checks were redundant with the type contract and silently masked type errors instead of failing fast.

### Verified

- 400 of 406 tests pass; 6 failures are pre-existing.
- No regressions from isinstance removal.
- All 88 remaining isinstance calls in the codebase are legitimate: type-narrowing on `Any`, polymorphic Union dispatch, exception classification, or untrusted-data validation.

## 2026-05-09 - Non-Blocking News Refresh and Enrichment Timeouts

### Changed

- Added bounded timeout budgets for the market-knowledge refresh path so trading checks continue to analysis even when RSS fetch or article enrichment stalls.
- Added new `[rag]` configuration keys in both config files: `rag_update_timeout_seconds`, `news_fetch_total_timeout_seconds`, and `news_enrichment_timeout_seconds`.
- Increased default `news_crawl_timeout` to 120s and kept it fully configurable.
- `src/app.py` now wraps market-knowledge refresh with timeout handling and explicit fallback logs, preventing pre-analysis stalls.
- `src/rag/news_ingestion/rss_provider.py` now applies outer timeouts to RSS fetch and enrichment stages, logs stage durations, and skips stalled enrichment without blocking execution.
- `src/rag/news_ingestion/crawl4ai_enricher.py` now applies an explicit batch timeout around Crawl4AI runs and falls back to aiohttp with clear warning telemetry.
- `src/rag/news_ingestion/schema_mapper.py` now removes high-confidence Decrypt-style market ticker prefixes before article bodies reach prompts or Discord notifications.
- `src/notifiers/notifier.py` now retries transient Discord 5xx send failures (including 503 no healthy upstream) with bounded backoff for text, embed, and chart sends.

### Added

- **test_discord_notifier_rate_limit.py** (+40 new): Transient Discord 5xx retry behavior, retry backoff logic, non-transient failure handling.
- **test_news_ingestion.py** (+48 new): RSS fetch timeout, enrichment timeout skip, non-blocking execution paths.
- **test_rss_provider_contract.py** (+33 new): RSS provider contract coverage for fetch-stage timeout and enrichment-timeout skip behavior, stage duration logging.

### Verified

- All 56+ tests pass with timeout and retry integration.
- Non-blocking execution confirmed: trading analysis proceeds even when RSS/enrichment stalls.
- Discord retry backoff working correctly for transient 5xx failures.

## 2026-05-08 - Prompt Reasoning and Observability Foundation

### Changed

- Added a model-facing decision reasoning protocol and decision gate for regime classification, conflict resolution, HOLD discipline, UPDATE/CLOSE gating, and explicit invalidation checks.
- Added backend prompt metadata and dashboard metadata fields so prompt behavior can be attributed by prompt version, response-contract version, and prompt variant without injecting version text into the LLM prompt.
- Added non-blocking prompt preflight linting for critical response-format, analysis-step, analysis-time, JSON-example, token-count, and untrusted-context guardrails.
- Added Pydantic-based validation metadata for trading analysis responses while preserving the existing JSON parser and fallback behavior.
- **src/parsing/unified_parser.py**: Extended with response validation schema, decision gate logic, and error recovery paths.
- **src/platforms/ai_providers/response_models.py**: Added prompt metadata fields (version, variant, contract_version) and comprehensive validation models.

### Added

- **test_prompt_linting.py** (+65 new): Preflight linting coverage for response format, analysis steps, JSON examples, token counts, and untrusted-context detection.
- **test_response_validation.py** (+62 new): Pydantic validation metadata, decision reasoning protocol, conflict resolution gate, HOLD/UPDATE/CLOSE gating logic.
- **test_template_manager.py** (+24): Metadata field integration and template rendering with observability features.

### Verified

- All 440+ line additions validated; zero regressions in response parsing.
- Preflight linting successfully detects common response-format violations before LLM execution.

## 2026-05-07 - Documentation Discoverability and Risk Messaging Refresh

### Changed

- Updated `README.md` top-level status messaging to remove outdated branch/news notes and clearly state demo-account/paper-trading scope, with no real exchange execution implemented in the public branch.
- Added a compact `Latest Changes (May 2026)` section in `README.md` to make recent behavior updates easier to discover from GitHub and LLM retrieval workflows.
- Refreshed `README.md` feature and configuration guidance to reflect timeframe-aware memory windows, semantic-rule evolution, and current risk-management guardrails (`max_position_size` and fallback sizing tiers).
- Rewrote `README.md` disclaimer language to emphasize not-financial-advice boundaries, user responsibility for local regulatory compliance, and warranty/liability consistency with `LICENSE.md`.
- Expanded `CONTRIBUTING.md` with local setup, testing expectations, changelog/documentation update requirements, and repository convention references.
- Added a documentation navigation hub for faster contributor and LLM orientation.
- Refreshed documentation-plan metadata and completion notes to include May 6-7 updates and discoverability changes.
- Added GitHub workflow scaffolding with `.github/pull_request_template.md` and issue templates under `.github/ISSUE_TEMPLATE/`.

## 2026-05-07 - Prompt Contract Hardening

### Changed

- Trading prompts now provide a parser-safe JSON example without pseudo-values, inline comments, or placeholder ranges, reducing invalid AI response risk for compact decision output.
- Response-format guidance now maps analysis steps to the existing five compact narrative lines, uses chart validation instructions only when chart analysis is available, and clarifies BUY/SELL, HOLD, UPDATE, and CLOSE JSON field semantics for open-position and no-position states.
- Confidence guidance now defers BUY/SELL entry thresholds to the dynamic response template and allows high-confidence HOLD decisions when staying out or maintaining a position is strongly justified.
- RAG/news/custom prompt snippets are now explicitly wrapped as untrusted market evidence so embedded article text cannot override system instructions, response format, risk rules, or trading policy.

## 2026-05-07 - Timeframe-Aware Memory Relevance and Prompt Horizon

### Changed

- `VectorMemoryService` now derives recency-decay defaults from the active timeframe at startup. For `4h`, this yields a 14-day half-life and a 56-day hard relevance window; lower timeframes use shorter windows, and higher timeframes use longer windows (capped).
- `TradingBrainService` now derives its closed-trade reflection scan cadence from the active timeframe: lower/noisier timeframes wait for more closed trades, `4h` keeps the existing 5-trade baseline, and daily/weekly timeframes scan sooner while preserving semantic-rule sample gates.
- `retrieve_similar_experiences` now over-fetches vector candidates before ranking, applies a hard age cutoff for prompt relevance, and then hybrid-ranks fresh candidates by similarity and recency. Older entries remain stored but are no longer used as fallback filler when fresh candidates are insufficient.
- Vector prompt context headers now include the active relevance window (`active window: last N days`) so the model can reason with explicit memory freshness constraints.
- System prompts now include a new `Trading Style & Horizon` section that adapts guidance to the active timeframe (scalping, intraday swing, swing, or position context), including expected hold horizon, noise tolerance, and news relevance window.

## 2026-05-06 - Internal Contract Cleanup

### Changed

- Removed redundant `hasattr()`, `getattr()`, and `isinstance()` checks from typed production paths, including config access, dashboard brain status/memory handling, prompt snapshot formatting, RAG retrieval, exit monitoring, retry logging, and AI provider response conversion.
- Tightened internal contracts so services now use direct attribute and method access for known collaborators such as config objects, strategy exit checks, logger-bearing instances, semantic-rule memory, and canonical article/analysis payloads.
- Config loader now normalizes admin IDs, dashboard CORS origins, and RAG news sources at load time so downstream services receive concrete lists instead of re-checking input shapes.
- Removed verified unused internal code, including stale trading statistics dataclasses, token request tracking state, obsolete dashboard state broadcast helpers, an obsolete CCXT multi-price fallback, a dead CoinGecko coin-data fetcher, unused provider metadata, and unused helper methods/constants.
- Fixed dashboard countdown datetime arithmetic and cache eviction typing surfaced during cleanup validation.

## 2026-05-06 - Latest Change Audit Fixes

### Fixed

- Semantic-rule refresh now checks all stored semantic rules for legacy unknown exit profiles and retires both stale loss-rule prefixes when a refreshed loss pattern changes between anti-pattern and corrective classification.
- Dashboard brain helpers now ignore malformed non-object JSON snapshots instead of relying on broad exception handling when building market context, brain status, memory, or current-position fallback price data.

## 2026-05-06 - Semantic Rule Exit Profile Repair

### Fixed

- `src/trading/brain.py` now fills missing semantic-rule SL/TP execution metadata from the configured exit profile before grouping trades, storing new rules, or rendering prompt context. This prevents legacy closed trades from producing `SL unknown/unknown | TP unknown/unknown` when the bot has configured hard/soft exit settings.
- `src/trading/vector_memory_rules.py` now supports targeted semantic-rule deactivation so refreshed resolved-profile rules can retire only their matching stale unknown-profile predecessors.
- `start.py` now injects config-derived exit execution defaults into the trading brain and triggers a one-shot refresh when active semantic rules still contain unknown exit profiles.
- `src/dashboard/routers/brain.py` now exposes dominant SL/TP intervals and rewrites legacy unknown exit-profile text for the Semantic Rules dashboard response.

## 2026-05-06 - Position Size Guardrails

### Changed

- LLM `position_size` parsing now treats explicit percent strings correctly, including values below 1% such as `0.5%`.
- RiskManager position-size fallbacks for LOW/MEDIUM/HIGH confidence are now configurable under `[risk_management]` and are used only when the AI omits or returns an invalid `position_size`.
- Brain and prompt defaults now keep `max_position_size` as a hard cap without nudging normal entries up to the cap as a minimum.

## 2026-05-06 - Trading Performance Summary P&L Percent Fix

### Fixed

- Discord and console performance summaries now calculate `Total P&L (%)` from realized quote P&L divided by configured demo quote capital, matching persisted trading statistics. Previously the notifier summed per-trade percentages, so variable position sizes could show a positive total percent while quote P&L was negative.
- Trading memory context now uses the same capital-based total P&L percentage when configured, while preserving average per-trade percentage separately.

## 2026-05-02 - Previous Analysis Context: Structured Decision Snapshot

### Changed

- `src/analyzer/prompts/template_manager.py` — `## PREVIOUS ANALYSIS CONTEXT` system-prompt section now includes a compact structured snapshot extracted from the previous AI response JSON block. The snapshot surfaces prior signal, confidence, entry/SL/TP/R/R/position size, trend direction and strength, confluence factor scores, and up to two key support/resistance levels alongside the existing narrative reasoning. Falls back to narrative-only when the JSON block is absent or malformed.
- `src/analyzer/prompts/template_manager.py` — A JSON-only previous response (no narrative text) now correctly produces a `## PREVIOUS ANALYSIS CONTEXT` section; previously it was silently skipped.
- `src/analyzer/prompts/template_manager.py` — `"reasoning"` field instruction in the response template now asks for 3-4 sentences covering decision thesis, market regime, key invalidation trigger, and next watch condition, improving data quality for vector-memory similarity and subsequent analysis cycles.
- `src/trading/trading_strategy.py` now uses the shared indicator classifiers when converting analysis results into entry market conditions, so real `technical_data` keys such as `atr_percent`, `macd_line`/`macd_signal`, `obv_slope`, and Bollinger Band levels are preserved for trading-brain similarity storage.
- `src/trading/data_models.py`, `src/factories/position_factory.py`, and `src/managers/persistence_manager.py` now persist `order_book_bias_at_entry`, allowing stop-loss/take-profit exits to reconstruct the same order-book context used by vector similarity queries.

## 2026-05-02 - Test Stability Fixes: Template Config Fallback and Short Article Extraction

### Fixed

- `src/analyzer/prompts/template_manager.py` now handles lightweight config objects safely when building the response template. If `MAX_POSITION_SIZE` is missing or invalid, it falls back to `0.10` instead of raising `AttributeError`.
- `src/rag/news_ingestion/rss_primitives.py` now accepts short but valid content from `article`/article-content/main selectors, avoiding fallback parser output that could include unrelated sidebar text.

## 2026-05-02 - Position Sizing: 10% Hard Cap and Corrected Prompt Formula

### Fixed

- **Prompt formula corrected**: The AI was instructed to use `position_size = confidence / 100`, turning a 50% confidence into a 50% capital allocation. Formula is now `(confidence / 100) × max_position_size` (default 0.10), so confidence 75 → 0.075 (7.5% of capital).
- **Hard cap added in `RiskManager`**: `calculate_entry_parameters` now clamps any AI-provided `position_size` to `config.MAX_POSITION_SIZE`. A warning is logged when clamping occurs.
- **`min_pos_size` floor corrected**: The brain-learned floor for position sizing now defaults to `0.02` (was `0.10`) and is capped at `max_position_size` to prevent floor > cap.
- **New config property `MAX_POSITION_SIZE`**: Added to `src/config/loader.py`, reading `[risk_management] max_position_size` from `config.ini` with a default of `0.10`.
- **`config.ini` / `config.ini.example`**: Added `max_position_size = 0.10` under `[risk_management]`.

## 2026-05-02 - Dashboard Confidence Parsing Regression Fix

### Fixed

- Restored dashboard Confidence KPI updates in `src/dashboard/routers/brain.py` by safely parsing `analysis.confidence` from JSON blocks in `response.text_analysis` even when `technical_data` is present.
- This fixes the regression introduced after the Apr 30 confidence-path fix, where confidence fell back to `--` when LLM output used JSON-only confidence (for example `"confidence": 75`) instead of a `Confidence: 75%` text line.
- Added/updated coverage in `tests/test_dashboard_brain_router.py` to ensure confidence is extracted from both regex fallback text and JSON analysis payloads when indicator fields are present.

## 2026-05-02 - Hard Exit Monitor Duplicate Position Save Fix

### Fixed

- Updated `src/trading/exit_monitor.py` so when both hard stop-loss and hard take-profit checks are due in the same tick, the monitor performs a single combined strategy check instead of two separate checks.
- This prevents duplicate `Saved position` writes/logs for unchanged open positions while preserving the same SL-first/TP-second evaluation semantics and timestamp persistence.

## 2026-05-01 - Security: Dependency Upgrades and requirements.txt Cleanup

### Security

- Upgraded `aiohttp` from `3.13.3` to `3.13.4` (CVE-2026-34513–34520, CVE-2026-22815 — 10 CVEs fixed).
- Upgraded `cryptography` to `≥46.0.7` (CVE-2026-26007, CVE-2026-34073, CVE-2026-39892).
- Upgraded `lxml` to `≥6.1.0` (CVE-2026-41066).
- Upgraded `orjson` to `≥3.11.6` (CVE-2025-67221).
- Upgraded `protobuf` to `≥6.33.5` (CVE-2026-0994).
- Upgraded `pyasn1` to `≥0.6.3` (CVE-2026-23490, CVE-2026-30922).
- Upgraded `pygments` to `≥2.20.0` (CVE-2026-4539).
- Upgraded `pytest` to `≥9.0.3` (CVE-2025-71176) — dev dependency.
- Upgraded `python-dotenv` to `≥1.2.2` (CVE-2026-28684).
- Upgraded `requests` to `≥2.33.0` (CVE-2026-25645).
- Upgraded `transformers` to `≥4.53.0` (CVE-2025-3777, CVE-2025-3933, CVE-2025-5197, CVE-2025-6051, CVE-2025-6638, CVE-2025-6921).
- Remaining unfixed: `pip==26.0.1` (CVE-2026-3219) — no patched version published by PyPA yet.

### Changed

- `start.py` now reads optional `HF_TOKEN` from `keys.env` and exports it to process environment (`HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN`) before `SentenceTransformer` initialization. This removes repeated unauthenticated Hugging Face Hub warnings when a token is configured and enables higher rate limits.
- `keys.env.example` now documents optional `HF_TOKEN` setup.

## 2026-05-01 - Outcome-Aware Semantic Rules and AI Mistake Learning

### Changed

- `src/trading/brain.py` — `_trigger_reflection()` now evaluates all trades matching a pattern (wins and losses) via a shared `_compute_group_metrics()` helper. The win gate was lowered from 10 to 5 and rule IDs are now deterministic (`rule_best_{pattern_key}`) to prevent duplicate rules across repeated reflections. Stored `best_practice` metadata now includes `wins`, `losses`, `win_rate`, `loss_rate`, `avg_pnl_pct`, `profit_factor`, `expectancy_pct`, `avg_mae_pct`, and `avg_mfe_pct`. Rules are only stored when win rate is ≥ 60%.
- `src/trading/brain.py` — `_trigger_loss_reflection()` fully overhauled: loss gate lowered from 5 to 3, occurrence gate from 3 to 2. Uses `_compute_group_metrics`, `_derive_failure_reason`, and `_derive_recommended_adjustment`. Stores `rule_type` of either `anti_pattern` (loss rate ≥ 60%) or `corrective` (mixed). Deterministic IDs prevent duplicate rules.
- `src/trading/brain.py` — Added `_trigger_ai_mistake_reflection()` so repeated HIGH-confidence failures, sideways/choppy failures, and reasoning/premise mismatches produce `ai_mistake` semantic rules. These rules store `mistake_type`, `entry_confidence`, `failed_assumption`, `failure_reason`, `recommended_adjustment`, and win/loss/P&L metrics.
- `src/trading/brain.py` — Reflection keys now include hard/soft SL/TP execution profile. Stop-like close reasons such as `hard_stop`, `hard_stop_loss`, `soft_stop_loss`, and `stop_loss_hit` are normalized to the stop-loss diagnostic bucket while preserving the exact dominant exit profile in metadata.
- `src/trading/brain.py` — `get_context()` now renders rule type tags (`[⚠️ AVOID]`, `[⚡ IMPROVE]`, `[🧠 AI MISTAKE]`) and a recommended adjustment line for each learned rule. Apply Insights prompt now explicitly instructs the LLM to balance wins/losses, compare current reasoning to failed AI assumptions, and account for hard/soft SL/TP execution settings.
- `src/trading/vector_memory_rules.py` — `get_anti_patterns_for_prompt()` now fetches `corrective` and `ai_mistake` rules in addition to `anti_pattern` rules and appends `→ Why:` / `→ Fix:` lines when diagnostic metadata is present.
- `src/dashboard/routers/brain.py` — `get_active_rules` endpoint now maps all new metadata fields: `rule_type`, `loss_rate`, `wins`, `losses`, `avg_pnl_pct`, `profit_factor`, `expectancy_pct`, `failure_reason`, `recommended_adjustment`, `mistake_type`, `entry_confidence`, `failed_assumption`, `dominant_close_reason`, `dominant_exit_profile`, `dominant_stop_loss_type`, and `dominant_take_profit_type`.
- `src/dashboard/routers/brain.py` and `src/utils/indicator_classifier.py` — Dashboard brain status and exit-execution context building now tolerate lightweight config objects by falling back to `unknown`/timeframe defaults when optional SL/TP execution attributes are absent.
- `src/dashboard/static/modules/vector_panel.js` — `renderSemanticRules()` updated: empty-state text no longer references "wins only"; rule cards now show a color-coded type badge (Best Practice / Avoid / Corrective / AI Mistake), wins/losses split, avg P&L, profit factor, exit profile, mistake type, failure reason, and an optional recommended adjustment line.
- `src/trading/brain.py`, `src/trading/vector_memory.py`, and `src/trading/vector_memory_context.py` — Brain reflection now runs every 5 closed trades instead of 10, and recency weighting for similar experiences now uses a 30-day half-life instead of 90 days so fresh market behavior affects prompts and semantic-rule synthesis faster.

### Added

- `src/trading/brain.py` — New private helpers compute combined win/loss statistics, normalize stop-like close reasons, preserve hard/soft exit profiles, detect sideways AI overconfidence, derive failed AI assumptions from stored reasoning, and map those diagnoses to concrete LLM-actionable instructions. Shared metadata and loss-diagnostic helpers keep reflection rules DRY across best-practice, loss, and AI-mistake paths.

## 2026-05-01 - VS Code Python Terminal Startup Workaround

### Changed

- Updated workspace settings in `.vscode/settings.json` to disable `python.terminal.activateEnvironment` and `python.terminal.activateEnvInCurrentTerminal`, and set `python-envs.terminal.autoActivationType` to `command`. This avoids Python Envs shell-startup profile injection failures (`EPERM` during `mkdir`) observed on some Windows user-profile paths with non-ASCII characters, while keeping interpreter selection pinned to `./.venv/Scripts/python.exe`.

## 2026-04-30 - RSS News Pipeline, Exit Monitoring, Security Fixes, and Public Cleanup

Primary commits: `926056e`, `349d49a`, `d6a72cc`, `551f67a`, `1e244da`, `be5e190`

### Added

- Replaced the old CryptoCompare-driven news flow with a full RSS ingestion stack under `src/rag/news_ingestion/`, including `rss_provider.py`, `rss_primitives.py`, `schema_mapper.py`, and optional Crawl4AI enrichment via `crawl4ai_enricher.py`.
- Added `src/rag/scoring_policy.py`, `src/rag/news_repository.py`, and `src/rag/local_taxonomy.py` to support article scoring, storage, and local category handling without CryptoCompare category dependencies.
- Started tracking `data/categories.json` in Git as the local category/taxonomy snapshot used by the new RSS-driven news flow.
- Added `src/platforms/ccxt_market_api.py` to formalize exchange market access outside the removed CryptoCompare market path.
- Added `src/trading/exit_monitor.py` and `src/trading/position_status_monitor.py` so stop-loss and take-profit execution can be checked on bot-side intervals instead of waiting only for candle-close logic.
- Added two debugging scripts, `scripts/fetch_free_news_preview.py` and `scripts/compare_news_body_quality.py`, to inspect raw RSS ingestion and enrichment quality.
- Added broad contract coverage for the new pipeline, including tests for news ingestion, RSS provider behavior, RAG retrieval/scoring contracts, exit monitoring, notifier rate limits, and CCXT migration.

### Changed

- Reworked the configuration contract for news ingestion and exit handling. `config/config.ini.example` gained RSS source whitelisting, per-source feed URLs, page-enrichment controls, density/co-occurrence scoring knobs, lower-timeframe guidance, and the new `[risk_management]` hard-exit interval settings. Existing local `config/config.ini` files need the same keys added manually because `config/config.ini` is intentionally ignored.
- Removed the obsolete `CRYPTOCOMPARE_API_KEY` entry from `keys.env.example` to match the news migration away from CryptoCompare.
- Narrowed the `data/` ignore rule in `.gitignore` so the repo now keeps `data/categories.json` under version control while still ignoring the rest of the local runtime data directory.
- Updated the bot runtime in `src/app.py`, `start.py`, `src/config/loader.py`, and `src/config/protocol.py` to wire the new RSS/news services, low-timeframe scheduling, exit-monitor settings, and richer config contracts.
- Extended `src/dashboard/routers/brain.py` so dashboard state is derived from parsed analysis JSON and richer live-trading context instead of fragile regex-only extraction. The Apr 30 diff also shows position SL/TP distance fields being carried through the API more explicitly.
- Reworked `src/rag/news_manager.py` and `src/rag/context_builder.py` so RAG assembly now pulls from the RSS repository, local taxonomy, and scoring policy instead of the previous CryptoCompare-oriented path.
- Enhanced notifier output in `src/notifiers/base_notifier.py`, `src/notifiers/console_notifier.py`, and `src/notifiers/notifier.py` so trade alerts can include chart images and last-trade context.
- Broadened `src/utils/indicator_classifier.py` and `src/utils/timeframe_validator.py` to support exit-execution context, lower timeframes, and more explicit runtime scheduling rules.

### Performance

- Landed a major O(N) optimization wave across indicator code. The release touched momentum, stochastic, moving-average crossover, support/resistance, and classifier-related paths, while the release notes and file diffs align on sliding-window rewrites for RSI, MACD, CCI, VHF, MAD, MFI, Z-score, Donchian, Ichimoku, Fibonacci Bollinger Bands, Vortex, Choppiness, and rolling volume filters.
- Tightened market data and persistence code around `src/analyzer/analysis_engine.py`, `src/analyzer/data_fetcher.py`, `src/analyzer/market_data_collector.py`, and `src/managers/persistence_manager.py` so the new ingestion and exit-monitor flows fit into the existing analysis loop.

### Security and Fixes

- Fixed a critical path traversal issue in `src/rag/file_handler.py` and shipped a cluster of front-end hardening changes across dashboard modules for DOM XSS, mutation-XSS, clickjacking, and template interpolation bugs.
- Fixed the dashboard confidence KPI path by moving the brain router toward parsed JSON extraction rather than depending on a `Confidence: XX%` regex shape that no longer matched the prompt contract.
- Fixed position sizing for values greater than 1% and hardened vector-threshold learning and reflection-key parsing in trading logic.

### Removed

- Deleted `src/platforms/cryptocompare/categories_api.py`, `src/platforms/cryptocompare/data_processor.py`, `src/platforms/cryptocompare/market_api.py`, `src/rag/category_fetcher.py`, and `src/rag/news_category_analyzer.py` in the main release.
- Deleted the remaining `src/platforms/cryptocompare/news_client.py` and package stub in follow-up commit `1e244da`, completing the migration away from CryptoCompare-backed news.
- Deleted the public `data_template/` tree in `be5e190` after it became clear that the directory no longer matched the private mainline.

### Notes

- The Apr 30 release was published in two steps: the main feature sync landed first, then same-day cleanup commits removed merge-conflict markers from `README.md`, source files, and `config/config.ini.example`, and finished deleting leftover CryptoCompare and `data_template/` artifacts. This section describes the final cleaned-up public state.

## 2026-04-14 - Dependency Maintenance

Primary commit: `554a507` (Dependabot merge of `8319f44`)

- Bumped Pillow from `12.1.1` to `12.2.0` on the public branch. This was later carried forward into the Apr 30 release notes as the image-processing/security maintenance update.

## 2026-03-28 - Release Sync: Caching, Vector Memory, Dashboard and Trading Improvements

Primary commit: `80389c1`

### Added

- Added `tests/test_dashboard_server_cache.py` and expanded dashboard brain router coverage to validate new cache and state behavior.

### Changed

- Implemented endpoint-aware dashboard caching with ETag/header strategy in `src/dashboard/server.py`.
- Updated dashboard client/runtime flow in `src/dashboard/static/main.js` and `src/dashboard/routers/brain.py` to align vector queries and state handling with richer trading context.
- Added fallback recomputation for SL/TP distance percentages in dashboard position output when persisted values are zero or missing.

## 2026-03-27 - Release Sync: Dashboard, Indicators, and Vector-Memory Overhaul

Primary commit: `3493598`

### Added

- Split dashboard styling into modular CSS files under `src/dashboard/static/css/`.
- Added `src/trading/vector_memory_analytics.py`, `src/trading/vector_memory_context.py`, and `src/trading/vector_memory_rules.py`.
- Added `src/utils/indicator_classifier.py` and new focused tests for context builder, dashboard routes, template manager, indicator classifier, and vector memory.

### Changed

- Reworked dashboard layout and panel modules in `index.html` and `main.js`.
- Updated analyzer, formatter, and indicator modules across momentum, sentiment, statistical, trend, volatility, support/resistance, and volume packages.
- Refined RAG and market-data processing paths and aligned dashboard/trading context behavior with vector-memory logic.

### Removed

- Removed the old monolithic `src/dashboard/static/style.css`.
- Removed many internal diagnostic scripts and broad internal verification tests from the public release snapshot.

## 2026-03-04 - News Architecture Simplification, Public Packaging Pass, and README/License Cleanup

Primary commits: `f03304b`, `8621585`, `5ba8169`

### Changed

- Simplified the CryptoCompare news stack by collapsing the earlier multi-file `news_components` structure into a single `src/platforms/cryptocompare/news_client.py` paired with a leaner `src/rag/news_manager.py`.
- Expanded dashboard behavior again in `src/dashboard/routers/brain.py`, `monitor.py`, and `performance.py`, while also improving logging and monitor-route behavior.
- Updated technical-calculation and risk-manager behavior alongside the dashboard and news refactor.
- Reworked public-facing documentation and licensing so the Mar 4 release shipped with a cleaned-up README structure and MIT-oriented release files.

### Added

- Added `tests/benchmark_monitor_news.py` during the refactor phase to benchmark news monitoring before the public release tree was trimmed back.

### Removed

- Removed the older `src/platforms/cryptocompare/news_api.py` and legacy `news_components/` modules after the refactor consolidated news fetching into a single client.
- Removed additional dashboard, RAG, start-up, and test code in the public release commit to produce a smaller public release snapshot.
- Removed the remaining Polyform references from public documentation in the follow-up README cleanup commit.

## 2026-03-02 - Major Update: Performance, Security, and Quality Enhancements

Primary commit: `2f1a6ad`

### Changed

- Improved dashboard server binding/runtime handling and startup orchestration in `start.py`.
- Updated market/platform integrations around Alternative.me, CoinGecko, and CryptoCompare API handling.
- Extended configuration behavior and associated test coverage (including dashboard config/cors paths).

## 2026-02-24 - Data-Model and Provider Infrastructure Refactor

Primary commit: `d89a1f6`

### Changed

- Renamed `src/trading/dataclasses.py` to `src/trading/data_models.py` and expanded trade/position snapshots for richer brain and vector-memory context.
- Strengthened AI provider base clients and retry/error-classification behavior across provider adapters.
- Added additional optimized statistical-indicator work (including kurtosis/skewness paths) with related regression coverage.

## 2026-02-22 - Major Update: Performance, Security, and Quality Enhancements

Primary commit: `576ddf8`

### Changed

- Improved dashboard routing/state behavior and refined server/static module implementations.
- Continued O(N)-oriented indicator and analyzer optimization pass.
- Expanded validation and security-related test coverage, including websocket origin and dashboard route checks.

## 2026-02-21 - Comprehensive System Overhaul, UI Enhancements, and Performance Optimizations

Primary commits: `a6e6af4`, `9195a34`

### Added

- Added `.pylintrc`, broad regression/integration test expansion, and refreshed dashboard image assets.
- Added a `data_template/` snapshot for the public release packaging at that time.

### Changed

- Overhauled dashboard UI/UX into a richer multi-panel surface with improved accessibility and interaction behavior.
- Reworked analyzer/trading/runtime layers for non-blocking I/O paths, stronger context handling, and improved orchestration.
- Applied multiple indicator/runtime correctness and performance improvements across trend, statistical, and momentum paths.

### Security and Fixes

- Hardened dashboard security paths (including XSS/DoS-related fixes and websocket safeguards).
- Fixed a duplicate live dashboard link in README (`9195a34`).

### Removed

- Removed older agent/workflow artifacts such as `AGENTS.md`, `CONTRIBUTING.md`, and obsolete Jules prompt files from the release snapshot.

## 2026-01-25

Primary commits: `a60020d`, `04ff96d`, `a2ac568`

- Moved runtime wiring toward a composition-root architecture in `start.py` and tightened dependency boundaries across analyzer, RAG, and trading services.
- Added `PositionFactory` and `DataFetcherFactory`, improved graceful shutdown and position lifecycle handling, and cleaned imports/readability across touched modules.
- Improved technical-analysis formatting, dashboard accessibility (aria labels and focus indicators), `UnifiedParser` behavior, and LM Studio model caching.

## 2026-01-24

Primary commit: `59b6b71`

- Added dashboard full-stack caching (HTTP cache headers plus RAM state cache) and refactored the brain router for high-traffic endpoint efficiency.
- Added cache verification tooling for headers and dashboard state TTL behavior.

## 2026-01-23

Primary commits: `bab3f48`, `df589da`, `f18e98d`, `967b71a`, `5ce1f67`, `81a9c01`, `77f935a`, `5276279`, `ae2f983`

- Delivered a broad analyzer/prompt/chart quality pass: pattern deduplication, markdown/prompt cleanup, chart visibility/tooltips, and indicator-tracking improvements.
- Added performance-focused refactors in market metrics and statistical indicators (including linear regression and O(N) optimizations).

## 2026-01-21

Primary commit: `ecd3eab`

- Introduced the first usable dashboard layer with brain/performance monitoring views, dashboard state, and core UI modules.

## 2026-01-20

Primary commit: `f877508`

- Added an initial integrated web dashboard build (UI, styling, and backend server wiring).

## 2026-01-19

Primary commits: `6256e60` plus merged PR batch (`500e7e3`, `151b710`, `7a0547e`, `f63d742`, `9264bb9`, `08f9fa6`, `de1f3a0`)

- Added comprehensive trade-lifecycle regression coverage and merged a broad optimization/fix batch across indicators, async testing, and vector-memory integration paths.

## 2026-01-18

Primary commits: `593c61b`, `c8ecc6f`, `65d0bda`, `1cb788d` plus merged PR batch (`80671e9`, `61a05d4`, `cdb72b0`)

- Added DefiLlama fundamentals integration and expanded market-overview/RAG market-data building.
- Merged security/accessibility/performance PRs and removed outdated AI agent guideline docs.

## 2026-01-17

Primary commits: `fe454f3`, `b6374e8`, `ce36abb`, `f9797fe` plus merges (`b1c1920`, `6c22ccc`)

- Expanded dashboard capabilities (statistics/news/positions panels), added dashboard security hardening and Cloudflare Tunnel support, and refreshed dependencies.

## 2026-01-16

Primary commits: `2858c28`, `036954a`, `81f5c64`, `47c9dec` plus merged PRs (`71d83a6`, `fb26593`, `c64edfe`)

- Unified AI provider responses with Pydantic models and reduced redundant indicator calculations.
- Expanded AI chart generation and dashboard visual-analysis support while merging security/performance/accessibility PRs.

## 2026-01-15

Primary commits: `d12bdfe`, `9684675`, `c31d755`, `4c034bc` (plus related updates)

- Improved LONG-bias mitigation logic and upgraded security dependencies (aiohttp) with safer dashboard error handling.
- Enforced UTC consistency in temporal/trading operations and tightened BlockRun provider compatibility.

## 2026-01-14

Primary commits: `f0bf950`, `1d73168`, `66450ce`, `4532bdf`, `09f868a` (plus refactor/cleanup updates)

- Added provider orchestration, stronger typed analysis context, expanded provider test coverage, and major dashboard feature growth (including new panels and synapse-style views).
- Continued large refactor wave across retrieval, parsing, and prompt/brain paths.

## 2026-01-13

Primary commits: `614f685`, `cf168d1`, `8e6d1f8`, `e24a04c`, `bdaba75`, `a67b9fc`, `17a9f93`, `a3b551e`, `ff97d8e`, `1e578d1`, `c4dafeb`, `9a26fcd`

- Major refactor/bugfix sweep across brain context, provider SDK integration, and performance tuning.

## 2026-01-12

Primary commit: `8feb100`

- Applied a focused bugfix/refactor pass across in-flight analyzer and trading logic.

## 2026-01-11

Primary commit: `1d0141a`

- Added non-breaking dynamic threshold learning enhancements in trading brain/vector memory using optional metadata and sample-size safeguards.

## 2026-01-10

Primary commits: `a061786`, `a30c942`, `b7b6730`

- Delivered bugfix/refactor updates and richer brain context extraction (weekend awareness, sentiment/order-book pressure, and serialization cleanup).

## 2026-01-09

Primary commit: `0271f7f`

- Applied dashboard bugfixes after early rollout.

## 2026-01-07

Primary commit: `f2062e7`

- Added core position-management strategy logic and dashboard routes for brain and position panels.

## 2026-01-06

Primary commits: `450bce6`, `3871980`, `3e9d0a8`, `7f26be1`, `a5c9955`, `e94ce36`

- Added vector-memory service foundations and integrated analysis-engine/config-loader improvements.
- Expanded real-time dashboard capabilities with websocket-driven monitoring and chart management endpoints.

## 2026-01-05

Primary commits: `6e8b0d0`, `6a69c17`, `029db43`, `538ffd7`, `85c884d` (plus related updates)

- Added single-instance protection and shutdown confirmation flow.
- Improved prompt-building with position context, strengthened statistics update ordering, and refined trading insights handling.

## 2026-01-04

Primary commits: large foundation batch including `dcb6244`, `4844ff9`, `5bb354a`, `301527b` and related trading/notifier refactors

- Added foundational trading modules for memory, persistence, statistics, strategy execution, and brain context.
- Added vector-memory integration and application composition-root wiring in `app.py`/`start.py`.
- Refactored notifier/formatting and logger internals while cleaning circular-import pressure.

## 2026-01-01

Primary commits: `3f08d9f`, `cff20ce`

- Enhanced trading behavior with dynamic parameters and improved data handling.
- Updated ignore rules for research-paper artifacts.

## 2025-12-31

Primary commits: `f9d4dcd`, `eb4b7ac`, `0082b19`, `354058e`, `15c9695`

- Continued large refactor passes and refreshed README execution/notifier architecture documentation.

## 2025-12-30

Primary commits: `2665dad`, `aeebd23`, `6cb02a4`, `e9244ac`, `d84c267` plus related docs/CI updates

- Switched configuration workflow to `config/config.ini.example` and ignored local `config/config.ini`.
- Added Google/AI chart-generation foundations, early analysis-engine wiring, trading brain dataclass work, and context-builder expansion.

## 2025-12-29

Primary commits: broad infrastructure batch including `35c3a24`, `45490e5`, `72f4f71`, `dba5720`, `0f9be19`

- Added dependency-injected app initialization, notifier abstractions, technical indicator coverage, persistence/config defaults, and licensing/readme updates.
- Cleaned repository tracking for agent-related folders.

## 2025-12-28

Primary commits: `5b14c43`, `6e24348`, `3cf41f7`, `3d650fc`, `d701a80`, `30a596b`, `7946feb` and related fixes/tests

- Added foundational RAG engine improvements, OpenRouter multimodal integration, CoinGecko/CryptoCompare market feeds, and core bot orchestration/prompt modularization.

## 2025-12-26

Primary commits: `969aaa9`, `660b55e`, `922b4b2`, `cdae3af`, `4ce9b80`

- Added trading-brain factor learning, RAG context/profiling utilities, provider-factory and exchange-manager restructuring, periodic Discord trade-status notifications, and v2 documentation updates.

## 2025-12-24

Primary commits: `ed2c5a7`, `0feb6b6`

- Fixed indicator crossover detection for historical context and refactored core analysis/strategy/logging infrastructure.

## 2025-12-23

Primary commit: `05dad4c`

- Added a mock AI provider path for offline testing and analysis flows.

## 2025-12-22

Primary commit: `c849ab7`

- Added AI model management protocols and abstraction foundations.

## 2025-12-21

Primary commits: `726676e`, `e61d869`, `ccf1207`, `1e17af3`

- Initial public project bootstrap: analyzer/RAG architecture, trading persistence foundations, and core bot functionality/error-handling scaffolding.