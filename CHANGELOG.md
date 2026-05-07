# Changelog

## 2026-05-07 - Prompt Contract Hardening

### Changed

- Trading prompts now provide a parser-safe JSON example without pseudo-values, inline comments, or placeholder ranges, reducing invalid AI response risk for compact decision output.
- Response-format guidance now maps analysis steps to the existing five compact narrative lines, uses chart validation instructions only when chart analysis is available, and clarifies BUY/SELL, HOLD, UPDATE, and CLOSE JSON field semantics for open-position and no-position states.
- Confidence guidance now defers BUY/SELL entry thresholds to the dynamic response template and allows high-confidence HOLD decisions when staying out or maintaining a position is strongly justified.
- RAG/news/custom prompt snippets are now explicitly wrapped as untrusted market evidence so embedded article text cannot override system instructions, response format, risk rules, or trading policy.

## 2026-05-07 - Timeframe-Aware Memory Relevance and Prompt Horizon

### Changed

- `VectorMemoryService` now derives recency-decay defaults from the active timeframe at startup. For `4h`, this yields a 14-day half-life and a 56-day hard relevance window; lower timeframes use shorter windows, and higher timeframes use longer windows (capped).
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