# LLM Trader — Master Agent Instructions

> Consolidated from all instruction files, skills, playbooks, and documentation in the LLM Trader repository. This single file replaces fragmented `.github/`, `docs/`, `.cursorrules`, `.windsurfrules`, and template files as the authoritative guide for any agent or developer working on this codebase.

---

## 1. System Overview & Architecture

### 1.1 What Is LLM Trader

**SEMANTIC SIGNAL LLM (LLM Trader)** is a BETA / Research Edition autonomous, asyncio-first trading bot. It converts market data, news (via RAG), and chart context into structured BUY / SELL / HOLD decisions via large language models. The bot runs in **demo-account and paper-trading mode** — real exchange order execution is not yet implemented.

- **Repository:** https://github.com/qrak/LLM_trader.git
- **Python:** 3.13, Linux (bash), virtual environment at `.venv/`
- **Entry point:** `python start.py` or cross-platform scripts in `scripts/`
- **Live dashboard:** https://semanticsignal.qrak.org
- **License:** MIT

### 1.2 Application Lifecycle

```
start.py (CompositionRoot)
  -> SingleInstanceLock
  -> Event loop setup with GracefulShutdownManager
  -> 8-stage dependency provisioning:
       1. _provision_infrastructure   (logger, config, persistence)
       2. _provision_utilities        (token counter, profiler, keyboard)
       3. _provision_platforms        (exchanges, CCXT, CoinGecko, Alternative.me, DeFiLlama)
       4. _provision_rag              (RAG engine, news manager, taxonomy)
       5. _provision_model            (AI providers with fallback chain)
       6. _provision_analyzer         (AnalysisEngine, data fetcher, indicators)
       7. _provision_trading          (TradingStrategy, ExitMonitor, VectorMemory, Statistics)
       8. _provision_notifiers        (Discord, console, file notifier)
  -> App loop: AnalysisEngine -> TradingBrainService -> TradingStrategy -> ExitMonitor
```

### 1.3 Dependency Injection

All services are instantiated in the **composition layer** (`start.py` / `app.py`) and injected via constructor parameters. **Never** construct service dependencies inside other service classes. This is the single most important architectural convention.

### 1.4 Core Module Map

| Layer | Key Modules | Responsibility |
|-------|-------------|----------------|
| **Brain** | `trading/brain.py` (facade) + 5 collaborators | LLM-driven decision engine |
| | `brain_context.py` | Context assembly for prompts |
| | `brain_experience.py` | Outcome-aware memory with recency decay |
| | `brain_exit_profiles.py` | Exit strategy profiles |
| | `brain_patterns.py` | Chart pattern integration |
| | `brain_reflection.py` | Semantic rule learning via reflection loops |
| **Analyzer** | `AnalysisEngine`, `TechnicalCalculator`, `PatternAnalyzer` | Market data analysis, indicator calculation, pattern detection |
| | `analysis_context.py` | Context aggregation for analysis pipeline |
| | `analysis_result_processor.py` | Post-analysis result processing |
| | `market_data_collector.py` | Market data aggregation |
| | `market_metrics_calculator.py` | Market metrics computation |
| | `pattern_quality_scorer.py` | Pattern quality scoring |
| | `trend_validator.py` | Trend validation logic |
| | `formatters/` | Technical, market, overview, period, long-term formatters |
| | `prompts/` | Prompt builder, context builder, template manager |
| | `pattern_engine/` | Chart, swing, trendline patterns |
| | `indicator_patterns/` | RSI, MACD, MA crossover, volume, stochastic, divergence, volatility patterns |
| **Trading** | `TradingStrategy`, `ExitMonitor`, `PositionStatusMonitor` | Strategy execution, exit monitoring, position tracking |
| | `VectorMemoryService` (+ context, rules, analytics) | ChromaDB-based vector memory (ChromaDB client injected via DI from `start.py`) |
| | `Statistics`, `StatisticsCalculator` | Trade statistics, performance tracking |
| | `PositionExtractor`, `StopLossTighteningPolicy` | Position parsing, SL policy |
| | `audit.py` | Trade audit logging |
| | `data_models.py` | Shared trading data models |
| | `memory.py` | Memory abstraction layer |
| | `order_lifecycle.py` | Order lifecycle management |
| | `guards/` | Order governance pipeline (`configured_symbol.py`, `cooldown_window.py`, `max_position_size.py`, `pipeline.py`) |
| **RAG** | `RagEngine`, `NewsManager`, `NewsRepository` | Retrieval-Augmented Generation |
| | `news_ingestion/` | RSS provider, Crawl4AI enricher, schema mapper |
| | `market_components/` | Market data cache, fetcher, processor, overview builder |
| | `ScoringPolicy`, `LocalTaxonomy`, `TickerManager` | Scoring, categorization, ticker management |
| | `article_processor.py`, `category_processor.py` | Article processing & categorization |
| | `collision_resolver.py`, `context_builder.py` | Cache collision resolution, RAG context building |
| | `file_handler.py`, `index_manager.py` | File-based storage & index management |
| | `market_data_manager.py` | Market data lifecycle management |
| **Indicators** | `indicators/` (20+ modules) | Full technical indicator library (momentum, overlap, price, trend, volatility, volume, statistical, support/resistance, sentiment) |
| **Platforms** | `CCXtMarketApi`, `CoinGecko`, `AlternativeMe`, `DeFillama` | External data providers |
| | `ExchangeManager` | Multi-exchange connection management |
| | `ai_providers/` | Google AI, OpenRouter, LM Studio, BlockRun, Mock — with fallback chain |
| | `cryptocompare/` | CryptoCompare market/news API |
| | `free_news/` | Free news source integration |
| **Evals** | `evals/` | Evaluation framework (`baselines.py`, `prompt_response_scoring.py`, `replay_fixture.py`) |
| **Dashboard** | `DashboardServer` (FastAPI) + 5 routers | Real-time web dashboard with WebSocket streams |
| | `static/` | HTML/CSS/JS frontend |
| **Managers** | `ModelManager`, `PersistenceManager`, `RiskManager`, `ProviderOrchestrator` | Model lifecycle, state persistence, risk management, provider orchestration |
| | `provider_types.py` | Provider type definitions |
| **Config** | `config/loader.py`, `config/protocol.py`, `contracts/` | Configuration loading, model/risk contracts |
| **Factories** | 4 factory modules | Data fetcher, position, provider, technical indicators factories |
| **Utils** | `profiler.py`, `token_counter.py`, `graceful_shutdown_manager.py`, `keyboard_handler.py`, `decorators.py`, `timeframe_validator.py`, `indicator_classifier.py`, `data_utils.py`, `format_utils.py` | Cross-cutting utilities |

### 1.5 Active Platform Integrations

- **Exchanges:** Binance, KuCoin, Gate.io, MEXC, Hyperliquid (via CCXT)
- **Market Data:** CoinGecko, Alternative.me, DeFiLlama, CryptoCompare
- **AI Providers:** Google AI (primary — Gemini 3.5 Flash), OpenRouter (fallback — DeepSeek), BlockRun.AI, LM Studio (local), Mock
- **News Sources:** CoinDesk, CoinTelegraph, Decrypt, CryptoSlate, CryptoCompare, free news feed (RSS + Crawl4AI enrichment)

### 1.6 Configuration

Active config at `config/config.ini`. Key settings:

- **Pair:** BTC/USDC, **Timeframe:** 4h, **Candles:** 999 (125 for AI chart)
- **Capital:** $10,000 simulated, **Fee:** 0.075%
- **Max Position:** 10%, **Fallback sizes:** 1% / 2% / 3%
- **News update:** every 4 hours, 5 articles max
- **Model:** Google Gemini 3.5 Flash (temperature 1.0), OpenRouter fallback
- **Dashboard:** 0.0.0.0:8000

---

## 2. Core Coding Style & Standards

### 2.1 Language & Documentation

- **English only** for code, comments, docstrings, and documentation.
- Use **docstrings** for documentation. Keep `#` comments rare — only for logic that is not self-evident.
- **No decorative section headers** made from `#` and `=` in Python files.
- **Never** more than one consecutive blank line. Avoid extra horizontal spacing.
- Update `CHANGELOG.md` whenever a change materially affects behavior, configuration, dependencies, public APIs, user workflows, or release packaging.

### 2.2 Python Typing

- **Explicit type hints** on class attributes, function signatures, and return values.
- **Modern Python 3.10+ syntax:** `list[str]`, `dict[str, int]`, `str | None`, `type[X]`.
- Investigate call sites and object construction before defining or assuming types.
- For known contracts, use **direct attribute access** and `None` checks instead of `hasattr()` or `getattr()`.
- Avoid redundant `isinstance()` checks when type hints or interfaces already define the contract. Prefer `typing.Protocol` where appropriate.
- Do not add delegation methods that only forward to a member object. Expose or use the member directly.
- Prefer **class-owned behavior** over new module-level helper functions. Keep standalone functions only for established stateless utility modules.

### 2.3 Design Principles

- **DRY is good, but not at the cost of readability.** Avoid excessive abstraction or indirection.
- **Simple, idiomatic solutions** — prefer standard libraries or established packages over custom parsing or over-engineered helpers.
- **Dependency Injection:** Instantiate services in the composition/app wiring layer. Do not construct service dependencies inside other service classes.

### 2.4 Pydantic v2 Modeling

- Use **Pydantic v2** for schemas, API responses, and configuration objects. Use standard `dataclass` only for simple internal state without validation needs.
- Configure with `model_config = ConfigDict(...)` — **not** `class Config:`.
- Use `@field_validator` and `@model_validator` — **not** legacy `validator()` / `root_validator()`.
- Use `model_dump()` and `model_validate()` — **not** `dict()` / `parse_obj()`.
- Prefer **strict types** or strict model configuration where silent coercion would be risky.
- Use **attribute access** (`response.content`) — **not** dictionary-style lookups.
- Reference implementation: `src/platforms/ai_providers/response_models.py`.

### 2.5 Linting & Code Quality

- Initialize attributes in `__init__`.
- **No** undefined variables, unused imports, unused variables, or unused arguments.
- Imports at the top of the module. Standard library before third-party ordering.
- Tools: `ruff` (lint), `pylint`, `mypy` (type check), `pytest` (tests).

### 2.6 Refactoring Approach

**Golden rules:**
1. Behavior is preserved — refactoring changes how, not what.
2. Small steps — test after each change.
3. Version control — commit before and after each safe state.
4. Tests are essential — without tests, you are editing, not refactoring.
5. One thing at a time — do not mix refactoring with feature changes.
6. Avoid redundant test loops — after a focused suite passes, do not rerun it unless the changed slice can affect it.

**When NOT to refactor:**
- Code that works and will not change again.
- Critical production code without tests (add tests first).
- When under a tight deadline.

---

## 3. Operational Rules & Playbooks

### 3.1 Python Execution

- **Always** use the local virtual environment:
  ```bash
  source .venv/bin/activate
  python start.py
  ```
- **Do not use global system Python.** The interpreter must resolve inside `.venv/`.
- Use **Linux bash syntax** — never PowerShell or `.venv/Scripts/` paths.
- For tool-driven terminal calls, use `.venv/bin/python` and `git` without path anchors — Hermes agents set their working directory to the project root automatically.

### 3.2 Testing

```bash
source .venv/bin/activate
python -m pytest tests/                         # Full suite
python -m pytest tests/test_vector_memory.py -q  # Focused file
python -m pytest tests/test_vector_memory.py -k fallback -q  # Specific test
```

- **55 test files** in `tests/` (`test_*.py`) + `conftest.py`, `__init__.py`.
- Redirect structured output to a temporary file, then read the file:
  ```bash
  python -m pytest tests/test_indicator_classifier.py -q > temp_pytest.txt
  cat temp_pytest.txt
  ```
- `tests/conftest.py` monkeypatches `src.config.loader` — config behavior under tests may differ from normal runtime.
- Keep unrelated pre-existing failures out of scope unless they block the current task.
- Prefer validating the exact file or test name affected by the change.

### 3.3 Data Directory

- **`data/` is local-only state.** Do not commit `data/` runtime files.
- `data/trading/` contains: API costs, brain vector DBs (ChromaDB), positions, statistics, trade history.
- `data/news_cache/` contains: recent news JSON.
- `data/news_fetch_preview/` contains: per-source raw and normalized JSON artifacts.
- `data/market_data/` contains: CoinGecko global JSON.

### 3.4 Cloudflare Free Cache Playbook

**Goal:** Increase edge cache hit ratio while preserving data freshness for the live dashboard.

**6 Cache Rules (ordered top to bottom):**
1. **Bypass** `/api/brain/refresh-price` — volatile price endpoint
2. **Bypass** `/api/brain/vectors?query=*` — high-cardinality search
3. **Cache** `/api/status/countdown`
4. **Cache** `/api/*` — safe GET traffic
5. **Cache** HTML shell pages
6. **Cache** static assets (or skip, using CF defaults)

**Deployment modes:** Conservative (default) and Public High-Offload profiles.

**Origin contract:** Versioned static assets, HTML shell, API defaults, explicit no-store bypasses.

**Validation:** Check `CF-Cache-Status` headers, review CF analytics.

**Rollback:** Revert rules to previous configuration via CF dashboard.

### 3.5 CI/CD Pipeline

**Workflow:** `.github/workflows/compatibility_manual_main.yml` (manual trigger via `workflow_dispatch`)

| Job | OS | Python | Scope |
|-----|----|--------|-------|
| `guard-main` | — | — | Branch guard (only `main`) |
| `linux-full` | Ubuntu | 3.11 | Full `pytest tests/` |
| `windows-medium` | Windows | 3.11 | `ruff check` + `compileall` |
| `macos-smoke` | macOS | 3.11 | Syntax compilation check |

### 3.6 Branch & Git Strategy

- **Primary development branch** varies by release cycle — check `git branch --show-current` before starting work.
- **Feature branches** branch from and PR back to the active development branch (not `main`).
- **Other branches:** `main`, `develop`, `master_public`.
- **Remote:** single `origin` at `https://github.com/qrak/LLM_trader.git`

### 3.7 Private-to-Public Sync

**Procedure:**
1. Start from `master_public`, squash private changes into a local sync branch.
2. Create sync branch: `git checkout -B sync/private-to-public master_public`
3. Squash merge: `git merge --squash release/v1.0-rc2 --allow-unrelated-histories`
4. Safety gate: `git diff --cached --stat` — verify no secrets or private assets.
5. Commit: `git commit -m "chore(sync): publish private changes to public as single squashed commit"`
6. Push: `git push public sync/private-to-public:master`
7. Verify parity: `git diff --stat public/master..release/v1.0-rc2`

**Important:** Do not use raw `git rev-list public/master...release/v1.0-rc2` counts in release summaries — public syncs are squashed, so ancestry counts overstate the real delta.

### 3.8 PR Process

**Review workflow:**
1. Fetch PR metadata and changed files/diff.
2. Review for: logic regressions, type hints, edge cases, security, performance.
3. Run targeted tests for touched areas. Add regression tests if coverage is missing.
4. Fix issues on PR branch, commit, push.
5. Merge only if code is correct and tests are green locally.
6. Post-merge: `git fetch --prune origin`, switch to target branch, `git pull --ff-only origin`.

---

## 4. Guardrails & Deployment Restrictions

### 4.1 Trading Safety

- **Paper trading only.** Real exchange order execution is not yet implemented.
- **Soft exits** enabled. Exit monitor checks every 15 minutes.
- **Max position size:** 10% of portfolio.
- **Fallback position sizes:** 1% / 2% / 3%.
- **Simulated capital:** $10,000 with 0.075% fee model.

### 4.2 Agent Governance

- Treat **trading actions, external provider calls, and persistent memory writes** as high-risk operations.
- Prefer **append-only logs** or traceable audit events for decisions affecting positions, signals, or external notifications.
- **Fail-closed behavior** if governance logic cannot decide safely.
- Keep governance configuration **declarative** so it can be reviewed without reading code paths.
- Order governance pipeline at `src/trading/guards/` provides configurable pre-execution checks (symbol whitelist, cooldown windows, position size limits).

### 4.3 LLM Output Quality (Agentic Eval)

**When to evaluate:**
- Prompt or output quality issues are recurring.
- Explicit pass criteria needed for LLM responses.
- Comparing variants instead of relying on intuition.

**Evaluation criteria for trading decisions:**
- Structured output validity
- Consistency with indicators
- Risk framing accuracy
- Fallback robustness
- Schema compliance

**Procedure:**
1. Define a small rubric before changing the prompt or logic.
2. Generate one baseline output.
3. Evaluate against explicit criteria.
4. Refine prompt, parser, or post-processing only for failed dimensions.
5. Re-run and compare. Stop after a small number of iterations or once improvement stalls.

### 4.4 Safety Checklists

**Before every push:**
- [ ] No secrets or credentials added
- [ ] Type hints and docstrings updated where needed
- [ ] Changes follow existing repository conventions
- [ ] `CHANGELOG.md` updated (if behavior/config/API/workflow changed)
- [ ] `README.md` updated (if user-facing behavior changed)
- [ ] Tests pass locally

### 4.5 Context Mapping (Before Changes)

**Procedure for any task:**
1. Start from the most concrete anchor: file, symbol, test, or failing behavior.
2. Identify the controlling code path — not just a wrapper or registration layer.
3. Find the smallest set of related files:
   - Owning implementation
   - Direct dependencies
   - Nearest tests
   - One similar pattern if needed
4. Stop once there is one falsifiable hypothesis and one cheap validating check.
5. Do not widen exploration until that check is tried.

---

## 5. Project Structure Reference

```
LLM_trader/
  start.py                         # Entry point + CompositionRoot
  AGENTS.md                        # This file — master agent instructions
  CLAUDE.md                        # Pointer to AGENTS.md
  README.md                        # Project overview, setup, roadmap
  CONTRIBUTING.md                  # Contribution guidelines
  CHANGELOG.md                     # Version history
  requirements.txt / requirements-dev.txt
  keys.env / keys.env.example      # Secrets
  .pylintrc / pytest.ini           # Lint & test config
  .cursorrules / .windsurfrules    # Agent entry points (delegate here)
  .github/
    copilot-instructions.md        # GitHub Copilot instructions
    workflows/compatibility_manual_main.yml    # CI/CD
  config/
    config.ini / config.ini.example
    model_pricing.json
    rag_priorities.json
  src/
    app.py                     # Main application module
    analyzer/                  # AnalysisEngine, TechnicalCalculator, PatternAnalyzer,
                               #   analysis_context, analysis_result_processor,
                               #   market_data_collector, market_metrics_calculator,
                               #   pattern_quality_scorer, trend_validator,
                               #   formatters, prompts, pattern_engine
    config/                    # loader.py, protocol.py
    contracts/                 # model_contract.py, risk_contract.py
    dashboard/                 # FastAPI server, 5 routers, static frontend
    evals/                     # Evaluation framework (baselines, scoring, replay)
    factories/                 # 4 factory modules
    indicators/                # 8 sub-packages (base, momentum, overlap, price, trend,
                               #   volatility, volume, statistical, support/resistance, sentiment)
    logger/                    # logger.py
    managers/                  # ModelManager, PersistenceManager, RiskManager,
                               #   ProviderOrchestrator, provider_types
    notifiers/                 # Base, console, file notifier + components, notifier.py
    parsing/                   # unified_parser.py
    platforms/                 # AI providers, CCXT, CoinGecko, Alternative.me,
                               #   DeFiLlama, ExchangeManager, CryptoCompare, free_news
    rag/                       # RagEngine, news ingestion (RSS, Crawl4AI),
                               #   market components, scoring, taxonomy,
                               #   article_processor, context_builder, index_manager
    trading/                   # TradingBrainService + 5 collaborators, strategy,
                               #   memory, statistics, monitors, audit, data_models,
                               #   order_lifecycle, guards (governance pipeline)
    utils/                     # Profiler, token counter, graceful shutdown, decorators, etc.
  tests/                       # 55 test_*.py files + conftest.py
  docs/
    plans/                     # Planning documents
  scripts/                     # Cross-platform startup scripts (Linux, macOS, Windows)
  data/                        # Runtime state (not committed)
  website/                     # Astro 5 + Tailwind CSS 3 landing page (separate from live dashboard)
                               #   - Framework: Astro 5 (static output), TypeScript, PostCSS
                               #   - Styling: Tailwind CSS 3 (`tailwind.config.mjs`),
                               #     Autoprefixer, custom shell/signal color palette
                               #   - Entry: `src/pages/index.astro` — imports all components
                               #   - Components: Hero, LiveTelemetry, ArchitectureBento,
                               #     DirectoryDive, EdgeInfrastructure, RiskFooter
                               #   - Layout: `src/layouts/BaseLayout.astro` — dark theme,
                               #     Space Grotesk font, `.ambient-grid` background overlay
                               #   - Styles: `src/styles/global.css` — Tailwind directives,
                               #     dark scheme, body gradient, ambient grid, panel/mono
                               #     utilities, custom scrollbars, reduced-motion support
                               #   - Data: `src/data/site.ts` — copy/text constants
                               #   - Build: npm run build (generates static dist/)
  img/                         # Dashboard screenshots
  logs/                        # Bot logs by date
```

---

## 6. Quick Reference — Command Cheat Sheet

```bash
# Start the bot
source .venv/bin/activate && python start.py

# Run full test suite
.venv/bin/python -m pytest tests/ > temp_pytest.txt

# Run focused test
.venv/bin/python -m pytest tests/test_vector_memory.py -k fallback -q

# Lint check
.venv/bin/python -m ruff check src tests start.py

# Type check
.venv/bin/python -m mypy src/

# Git status
git status
```
