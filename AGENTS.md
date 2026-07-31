# LLM Trader — Master Architecture Blueprint

> **Repository:** [https://github.com/qrak/LLM_trader.git](https://github.com/qrak/LLM_trader.git)
> **Python:** 3.13, `.venv/`, `python start.py`
> **Status:** BETA / Research Edition — paper-trading mode only
> **Live Dashboard:** [https://semanticsignal.qrak.org](https://semanticsignal.qrak.org)

---

## 0. Instruction Authority

Root `AGENTS.md` is the single instruction source of truth in this repository across all IDEs, agents, and harnesses.

- Root `AGENTS.md` is canonical for system-wide rules, architecture, coding standards, testing, terminal behavior, and governance.
- IDE-specific instruction files are non-authoritative and should not contain policy that is missing from `AGENTS.md`.
- `.github/workflows/*` defines CI execution behavior, not instruction authority.

---

## 1. System Overview

**SEMANTIC SIGNAL LLM (LLM Trader)** is an autonomous, asyncio-first trading bot that converts market data, news (via RAG), and chart images into structured BUY / SELL / HOLD decisions via large language models. The system operates a **distributed multi-agent intelligence architecture**: specialized agents for technical analysis, pattern recognition, news retrieval, risk validation, outcome-aware learning, and reflection-based rule synthesis — all coordinated through a central trading loop.

```mermaid
flowchart TB
    subgraph External["External Layer"]
        EX["Exchanges<br/>(Binance, KuCoin, Gate.io,<br/>MEXC, Hyperliquid)<br/>&#8209; CCXT"]
        CG["CoinGecko<br/>DeFiLlama"]
        ALT["Alternative.me<br/>(Fear & Greed)"]
        RSS["RSS Feeds<br/>(CoinDesk, CoinTelegraph,<br/>Decrypt, CryptoSlate)"]
        AI_PROV["AI Providers<br/>Google Gemini (primary)<br/>LM Studio (local text fallback)<br/>OpenRouter (secondary configurable provider)"]
    end

    subgraph DataIngestion["Data Ingestion Layer"]
        DF["DataFetcher<br/>OHLCV + Order Book + Trade Flow"]
        RAG["RAG Engine Agent<br/>News + Fundamentals"]
    end

    subgraph AnalysisLayer["Analysis Layer"]
        TA["Analysis Engine Agent<br/>Technical Calculator<br/>50+ Indicators"]
        PE["Pattern Engine<br/>Deterministic Indicator<br/>Pattern Detection<br/>Numba JIT compiled"]
        CGEN["Chart Generator<br/>4K PNG Candlestick<br/>SMA/RSI/Volume/CMF+OBV"]
    end

    subgraph BrainLayer["Learning & Memory Layer"]
        BRAIN["🧠 Brain Agent<br/>TradingBrainService"]
        VM["Vector Memory<br/>ChromaDB<br/>Trade Experiences<br/>Semantic Rules<br/>Confidence Stats"]
        REFL["Reflection Engine<br/>Best‑practice Rules<br/>Anti‑patterns<br/>AI Mistake Rules"]
    end

    subgraph RiskLayer["Risk & Execution Layer"]
        RP["Risk Manager<br/>Dynamic SL/TP<br/>Position Sizing"]
        GP["Order Governance Pipeline<br/>Symbol Guard<br/>Max Size Guard<br/>Cooldown Guard"]
        STRAT["Trading Strategy<br/>Exit Monitor<br/>Position Status Monitor"]
    end

    subgraph Output["Output Layer"]
        DASH["📊 Dashboard<br/>FastAPI + WebSocket"]
        LOGS["Audit Trail<br/>Position Logs<br/>SQLite Trade History"]
    end

    subgraph Providers["Provider Orchestration"]
        PO["Provider Orchestrator<br/>Fallback Chain"]
    end

    %% Data Flow
    EX --> DF
    RSS --> RAG
    CG --> RAG
    ALT --> TA
    DF --> TA
    TA --> PE
    TA --> CGEN
    
    RAG --> TA
    TA --> PO
    PO --> AI_PROV
    
    AI_PROV -->|"Structured Signal"| RP
    RP --> GP
    GP --> STRAT
    
    STRAT -->|"Closed Trade"| BRAIN
    BRAIN --> VM
    VM -->|"Reflection Loop"| REFL
    REFL -->|"Rules"| VM
    VM -->|"Context Injection"| BRAIN
    BRAIN -->|"Confidence + Rules"| TA
    
    TA --> DASH
    STRAT --> DASH
    STRAT --> LOGS
```

---

## 2. Agent Inventory

| # | Agent Name | Primary Responsibility | Core Model | Core Implementation |
|---|------------|----------------------|------------|---------------------|
| 1 | **🧠 Brain Agent** (TradingBrainService) | Outcome-aware decision enricher, semantic rule learning via reflection loops, confidence calibration | Deterministic/vector memory; context is injected into provider-routed LLM prompts | [`src/trading/brain.py`](./src/trading/brain.py) |
| 2 | **🔬 Analysis Engine Agent** | Market data collection, 50+ technical indicators, pattern recognition, chart generation, AI signal synthesis | Gemini 3.6 Flash (multimodal) | [`src/analyzer/analysis_engine.py`](./src/analyzer/analysis_engine.py) |
| 3 | **📰 RAG Engine Agent** | News aggregation (RSS + Crawl4AI), fundamentals (DeFiLlama), relevance scoring, context retrieval | Deterministic (no LLM) | [`src/rag/rag_engine.py`](./src/rag/rag_engine.py) |
| 4 | **⚙️ Risk Manager Agent** | Dynamic SL/TP scaling, position sizing, signal validation, circuit breakers | Deterministic | [`src/managers/risk_manager.py`](./src/managers/risk_manager.py) |
| 5 | **☁️ Provider Orchestrator** | AI provider lifecycle, multi-provider fallback chain, parameter negotiation | — | [`src/managers/provider_orchestrator.py`](./src/managers/provider_orchestrator.py) |
| 6 | **🛡️ Governance Pipeline** | Pre-execution guard chain: symbol whitelist, max position size, cooldown | Deterministic | [`src/trading/guards/pipeline.py`](./src/trading/guards/pipeline.py) |
| 7 | **📊 Dashboard Agent** | Real-time FastAPI + WebSocket monitoring, performance analytics, brain state inspection | — | [`src/dashboard/server.py`](./src/dashboard/server.py) |

---

## 3. Application Lifecycle

### 3.1 Startup (CompositionRoot)

`start.py` → `SingleInstanceLock` → Event loop with `GracefulShutdownManager` → 9-stage dependency provisioning:

| Stage | Provisioner | Dependencies Created |
|-------|------------|---------------------|
| 1 | `_provision_infrastructure` | ExchangeManager, aiohttp session, KeyboardHandler |
| 2 | `_provision_utilities` | FormatUtils, UnifiedParser, TokenCounter, TimeframeValidator, CategoryCollisionResolver |
| 3 | `_provision_platforms` | CCXTMarketAPI, CoinGecko, Alternative.me, DeFiLlama, RSS/Crawl4AI news client |
| 4 | `_provision_rag_layer` | RagEngine, NewsManager, LocalTaxonomyProvider, TickerManager |
| 5 | `_provision_model_layer` | AI provider clients, ProviderOrchestrator, ModelManager |
| 6 | `_provision_analyzer_layer` | AnalysisEngine, MarketDataCollector, TechnicalCalculator, PatternAnalyzer |
| 7 | `_provision_trading_layer` | TradingStrategy, ExitMonitor, VectorMemoryService, TradingStatisticsService, TradingBrainService |
| 8 | `_provision_notifiers` | Discord notifier with DiscordFileHandler, or console fallback notifier |
| 9 | `_provision_dashboard_layer` | DashboardServer, DashboardState, force_analysis_event |

**Architectural invariant:** All services are instantiated in the composition layer and injected via constructor parameters. **Never** construct service dependencies inside other service classes, and **never** use in-function lazy imports (`Pylint C0415`) to resolve circular dependency cycles—refactor constructor parameter injection at the CompositionRoot (`start.py`) instead.

### 3.2 Main Loop

```
AnalysisEngine.analyze_market()
  ├── MarketDataCollector → DataFetcher (OHLCV + order book + trade flow)
  ├── TechnicalCalculator (50+ indicators) + LongTerm data + Weekly macro
  ├── PatternAnalyzer → IndicatorPatternEngine (deterministic indicator-pattern kernels)
  ├── ChartGenerator (4K PNG) → LLM visual chart-pattern analysis (via analysis_result_processor.py)
  ├── RAG context retrieval
  ├── Brain context injection (confidence + rules similar to current conditions)
  ├── AI provider call → TradingAnalysisResponseModel (prompt includes step 5.5 invalidation check:
  │      model must name a specific invalidation trigger or HOLD)
  └── Structured dict returned to TradingStrategy
       ↓
TradingStrategy.process_analysis()
    ├── PositionExtractor + UnifiedParser → extract and validate signal
    ├── GuardPipeline (symbol → max size → cooldown)
    ├── RiskManager → RiskAssessment (SL/TP scaling, computes R:R)
    ├── TradingStrategy → R:R minimum check against brain-learned threshold (default 1.5)
    ├── OrderLifecycle → INTENT → READY_FOR_REVIEW → EXECUTED (or REJECTED)
    ├── Approval is recorded as an audit event, not as an OrderLifecycle state
    ├── PersistenceManager → SQLite-only trade_history.db append (no JSON fallback/migration)
    ├── RiskManager friction drain → store_blocked_trade feedback for brain learning
    └── ExitMonitor (dual-mode: soft at candle close; hard at configurable interval per SL/TP type)
       └── PositionStatusMonitor → background asyncio loop with dynamic rescheduling
       ↓
BrainAgent.update_from_closed_trade()
  ├── BrainExperienceRecorder → store vector memory
  ├── trade_count++ → schedule reflection if interval reached
  └── ReflectionEngine → sequential: best-practice → anti-pattern → AI-mistake rules
```

### 3.3 Shutdown

`GracefulShutdownManager` handles:
- SIGINT/SIGTERM → drain active analysis → persist state → close providers → flush logs
- Keyboard handler → manual stop with state preservation

---

## 4. Core Data Flow

### 4.1 Decision Cycle

```
┌──────────────┐    ┌──────────────────────┐    ┌───────────────────┐
│  DataFetcher  │───▶│   AnalysisEngine     │───▶│  ProviderOrch.    │
│  (CCXT/API)   │    │  TechCalc + Pattern  │    │  (Fallback Chain) │
└──────────────┘    │  Chart + RAG + Brain  │    └────────┬──────────┘
                    └──────────────────────┘             │
                                    ▲                    ▼
                                    │           ┌──────────────────┐
                                    │           │   UnifiedParser   │
                                    │           │  → TradingSignal  │
                                    │           └────────┬──────────┘
                                    │                    ▼
                                    │           ┌──────────────────┐
                                    │           │  GuardPipeline   │
                                    │           │  3 Guards (pass?)│
                                    │           └────────┬──────────┘
                                    │                    ▼
                                    │           ┌──────────────────┐
                                    │           │   RiskManager    │
                                    │           │  SL/TP/Size/R:R  │
                                    │           └────────┬──────────┘
                                    │                    │
                                    │                    ▼
                                    │           ┌──────────────────────┐
                                    │           │ TradingStrategy      │
                                    │           │ R:R check (min 1.5)  │
                                    │           │ + ExitMonitor        │
                                    │           └────────┬─────────────┘
                                    │                    │
                                    │                    ▼ (on close)
                                    │           ┌──────────────────────┐
                                    └───────────│   BrainAgent         │
                                                │  Experience +        │
                                                │  Reflection + Rules  │
                                                └──────────────────────┘
```

### 4.2 Learning Loop

```
Closed Trade ──▶ BrainExperienceRecorder ──▶ ChromaDB (vector memory)
                                                   │
                                                   ├── Update matched semantic-rule validation/contradiction counters
                                                   │
                          trade_count % interval == 0
                                                   │
                                                   ▼
                                          ReflectionEngine
                                          ├── Best-practice rules
                                          ├── Anti-pattern rules
                                          └── AI-mistake rules
                                                   │
                                                   ▼
                                          Next Cycle: BrainContextProvider
                                          queries ChromaDB for:
                                          - Similar past trades (top-5)
                                                                                    - Relevant rules (matched to conditions,
                                                                                        scored by similarity + evidence + timeframe freshness)
                                          - Confidence stats by level
                                          - Blocked trade feedback
                                                   │
                                                   ▼
                                          Injected into LLM prompt
```

Semantic-rule policy:
- Active semantic rules are durable learned policy and are not deleted by age-only pruning.
- Rule influence is soft-ranked by semantic similarity, evidence quality, timeframe-aware freshness, contradiction count, and **surprise ratio** (see below).
- Closed trades that match active rules update validation or contradiction metadata for later ranking.
- **Surprise ratio** (`|realized P&L - expected P&L| / expected P&L`) is computed at trade close. A high surprise ratio (>1.5) means the outcome was driven by factors outside the entry thesis — the trade won despite flawed reasoning (or lost despite good reasoning). Rules derived from high-surprise trades carry a `⚠️ high surprise` annotation in their rule text, allowing the LLM to discount lucky outcomes when forming policy.
- Inactive old rules may be physically pruned as storage maintenance; active rules should be deactivated by evidence, not age.

### 4.3 Trade Persistence

- Trade history is SQLite-only at `data/trading/trade_history.db` via `SQLiteTradeHistory` and `PersistenceManager`.
- Runtime code must not read, write, or auto-migrate `trade_history.json`.
- `PersistenceManager.save_trade_decision()` fails loudly if SQLite persistence fails; do not add JSON fallback paths.
- Dashboard, cooldown guards, brain entry-decision lookup, and query scripts must consume trade history through injected persistence or SQLite APIs.
- Historical `.json.migrated` files are backups only, not runtime inputs.
- **Zero Backward Compatibility & Startup Clutter Policy**:
  - Runtime code in `src/` and `start.py` must remain 100% clean, canonical, and clutter-free.
  - Never introduce inline schema migration `ALTER TABLE` statements, legacy unit conversion methods, rule migration hooks (`refresh_semantic_rules_if_stale`), or `try/except` fallback paths into runtime service initialization.
  - Never add `sys.path.insert(0, ...)` manipulation hacks into `start.py`.
  - Never include startup auto-rehydration loops in `start.py`. Any database or vector storage rehydrations/conversions MUST be executed explicitly via standalone CLI scripts (e.g., in `scripts/`), after which the script is executed once and deleted.
  - **Classes Only in `src/utils/`**: All utility concerns across `src/utils/` and `app.py` must be encapsulated as a Class (e.g., `JournalRotator`, `TokenCounter`). Standalone utility functions are strictly forbidden.

---

## 5. Configuration

Active config at `config/config.ini`. Key settings:

| Setting | Value |
|---------|-------|
| **Pair** | BTC/USDC (USD Coin) |
| **Timeframe** | 4h |
| **Candles** | 999 (125 for AI chart) |
| **Capital** | $10,000 simulated |
| **Fee** | 0.075% |
| **Max Position** | 10% of portfolio |
| **Fallback sizes** | 1% / 2% / 3% (LOW/MEDIUM/HIGH confidence) |
| **News update** | Every 4 hours, 5 articles max |
| **Model** | Google Gemini 3.6 Flash (provider=`googleai`), OpenRouter base model `google/gemini-3-flash-preview`, OpenRouter fallback `deepseek/deepseek-r1:free` |
| **Dashboard** | 0.0.0.0:8000 |

---

## 6. Project Structure Reference

```
LLM_trader/
├── start.py                     # Entry point + CompositionRoot
├── AGENTS.md                    # THIS FILE — single master architecture blueprint & rules
├── README.md                    # Project overview, setup, roadmap
├── CHANGELOG.md                 # Version history
├── requirements.txt / -dev.txt
├── keys.env / keys.env.example  # Secrets
├── config/
│   ├── config.ini               # Active configuration
│   ├── model_pricing.json       # Per-model cost data
│   └── rag_priorities.json      # Category/generic RAG priority config (important_categories + generic_priorities)
├── src/
│   ├── app.py                   # Main application wiring
│   ├── trading/                 # 🧠 Brain Agent + Strategy + Monitors
│   │   ├── brain.py             # TradingBrainService (facade)
│   │   ├── brain_*.py           # 5 collaborators
│   │   ├── trading_strategy.py  # Strategy orchestration
│   │   ├── exit_monitor.py      # Hard/soft exit checks
│   │   ├── vector_memory.py     # ChromaDB interface
│   │   ├── vector_memory_*.py   # Analytics, rules, context (3 collaborators)
│   │   ├── regime_risk_profile.py # Regime-aware risk profile selector (Risk Manager)
│   │   ├── rl_policy.py         # RL policy network (experimental)
│   │   ├── statistics.py        # P&L tracking
│   │   └── guards/              # 🛡️ Governance Pipeline
│   ├── analyzer/                # 🔬 Analysis Engine
│   │   ├── analysis_engine.py   # Orchestrator
│   │   ├── technical_calculator.py # 50+ indicators
│   │   ├── pattern_engine/      # Chart + indicator patterns
│   │   ├── prompts/             # System prompt construction
│   │   ├── formatters/          # Context formatting (4 non-init source modules)
│   │   ├── data_fetcher.py      # Exchange data abstraction
│   │   └── ...                  # 15+ supporting modules
│   ├── rag/                     # 📰 RAG Engine
│   │   ├── rag_engine.py        # Orchestrator
│   │   ├── news_manager.py      # News lifecycle
│   │   ├── news_ingestion/      # RSS + Crawl4AI
│   │   └── ...                  # 15+ supporting modules
│   ├── managers/                # ⚙️ Risk Manager + ☁️ Provider Orchestrator
│   │   ├── risk_manager.py      # Signal safety layer
│   │   ├── persistence_manager.py # Position/state facade + SQLite trade history access
│   │   ├── sqlite_trade_history.py # SQLite-only trade history store
│   │   ├── provider_orchestrator.py  # AI fallback chain
│   │   └── model_manager.py     # Model lifecycle
│   ├── dashboard/               # 📊 Dashboard
│   │   ├── server.py            # FastAPI app
│   │   └── routers/             # 7 API routers
│   ├── indicators/              # Indicator library — 50+ Numba functions
│   ├── platforms/               # AI providers + exchange APIs
│   ├── parsing/                 # UnifiedParser
│   ├── logger/                  # Structured logging
│   ├── notifiers/               # Discord, console, file
│   └── utils/                   # Profiler, token counter, etc.
├── tests/                       # 89 test_*.py files + conftest.py
├── data/                        # Runtime state (not committed)
├── logs/                        # Rotated daily log output
│   └── Bot/                     # Logger name (defined in logger init)
│       └── YYYY_MM_DD/          # One folder per day
│           ├── Bot.log          # Full structured log (all levels)
│           └── errors.log       # Error-level only log
├── scripts/                     # Cross-platform startup scripts
│   └── install_agent_terminal_guard.ps1 # Optional session-local PowerShell literal ^U guard
├── docs/
├── .ai/
│   └── plans/                   # AI-generated planning documents (gitignored)
```

---

## 7. Active Platform Integrations

- **Exchanges:** Binance, KuCoin, Gate.io, MEXC, Hyperliquid (via CCXT)
- **Market Data:** CoinGecko, Alternative.me, DeFiLlama, CCXT exchange market data
- **AI Providers:** Google AI (primary — Gemini 3.6 Flash), LM Studio (local text fallback), OpenRouter (secondary provider with configurable base + fallback models)
- **News Sources:** CoinDesk, CoinTelegraph, Decrypt, CryptoSlate, RSS feeds with Crawl4AI enrichment

---

## 8. Operational Rules

Use this root `AGENTS.md` as the canonical source for all global standards and agent policies.

### Terminal Guardrails (All Agents)

- Send one terminal command per tool call.
- Never include control-key text in commands (for example `^U`, `^C`, `^[`).
- On Windows/PowerShell in VS Code, prompt-edit control text such as `^U` is sent literally and becomes part of the command name. Do not assume Linux/readline behavior.
- Never send terminal follow-up probes or marker echoes (for example `Write-Output $LASTEXITCODE`, `echo DONE`, or a "flush" command) to recover hidden or truncated validation output.
- If validation output is incomplete, either trust the user's visible terminal output when provided or rerun the exact validation command once with a generous timeout.
- `scripts/install_agent_terminal_guard.ps1` can be dot-sourced as a session-local safety net for accidental literal `^U` prefixes; it is not a substitute for clean commands.
- Never chain validation commands with `;`, `&&`, variable assignment, redirect/capture, and readback in one line.
- For pytest validation, trust only raw output from a direct pytest command.
- If terminal output is empty or malformed, do not claim success.
- Never infer pass/fail from a trailing `PYTEST_EXIT` marker when earlier commands in that same line failed.

### Operator Commands

Keep platform-specific setup, startup, test, lint, and type-check commands in `README.md`.
This file documents agent architecture and execution policy only.

### Safety

- **Paper trading only** — real exchange order execution not implemented
- **Hard SL/TP exits** are configured at 15-minute intervals; soft candle-close exits are supported by ExitMonitor
- **Max position:** 10% of portfolio
- **Simulated capital:** $10,000 with 0.075% fee model
- **Fail-closed behavior** if governance/risk validation cannot decide safely

### Codebase Vector Search (All Agents)

- Before performing architectural edits, cross-module refactoring, or searching for implementations across the codebase, query the codebase vector index:
  ```
  python scripts/query_codebase.py "<natural language query>"
  ```
- Use the returned semantic snippets (file paths + line ranges + relevance scores) to navigate directly to the right code.
- Prefer this over blind grep for architectural and "where does X happen?" questions.
- The index auto-updates on bot startup after all provisioning stages succeed. For manual refresh: `--reindex` flag.
- See the `codebase-vector-search` skill for full CLI reference.

---

## 9. Documentation Governance

### AGENTS-Only Policy Checklist

Use this checklist for every documentation or tooling-policy PR:

1. All behavioral policy changes are documented in root `AGENTS.md`.
2. Do not introduce IDE-specific policy files (for example Copilot, Claude, or Windsurf instruction docs) as authoritative guidance.
3. `.github/workflows/*` may define CI execution logic only; workflow comments must not replace policy documentation in `AGENTS.md`.
4. If a command, validation rule, or safety guard changes, update the related AGENTS section in the same PR.
5. Before merge, run a repository search to ensure no stale references point to removed tool-specific instruction files.

### Drift Prevention Rule

- Any new tool-specific instruction file must be a non-authoritative pointer to `AGENTS.md`; if it contains independent policy, it should be rejected in review.

---

## 10. AI Worker Agents & Multi-Agent Pipeline (.ai/)

This repository employs an 8-agent specialized AI developer roster coordinated by **Supervisor** ([`.ai/supervisor.md`](./.ai/supervisor.md)) for automated refactoring, performance tuning, security hardening, UI updates, and test verification.

### 10.1 AI Agent Roster

| # | Agent | Emoji | Prompt File | Primary Scope | Journal File |
|---|---|---|---|---|---|
| 1 | **Supervisor** | 🧠 | [`.ai/supervisor.md`](./.ai/supervisor.md) | Routes work, scans codebase via vector search, orchestrates pipelines | — |
| 2 | **Bolt** | ⚡ | [`.ai/bolt.md`](./.ai/bolt.md) | **Performance** — caching, async patterns, I/O, serialization, numpy, hot paths | [`.ai/journal.md`](./.ai/journal.md) |
| 3 | **Palette** | 🎨 | [`.ai/palette.md`](./.ai/palette.md) | **UX & Accessibility** — dashboard HTML/CSS/JS, ARIA, keyboard nav, responsive | [`.ai/palette-journal.md`](./.ai/palette-journal.md) |
| 4 | **Sentinel** | 🛡️ | [`.ai/sentinel.md`](./.ai/sentinel.md) | **Security** — auth, CSP headers, rate limiting, input validation, secret handling | [`.ai/sentinel-journal.md`](./.ai/sentinel-journal.md) |
| 5 | **Refactor** | ✨ | [`.ai/refactor.md`](./.ai/refactor.md) | **Clean Code** — isinstance chains, DRY violations, DI pattern enforcement | [`.ai/refactor-journal.md`](./.ai/refactor-journal.md) |
| 6 | **Concise** | ✂️ | [`.ai/concise.md`](./.ai/concise.md) | **Code Line Reduction** — DRY abstractions, mixins, dispatch tables | [`.ai/concise-journal.md`](./.ai/concise-journal.md) |
| 7 | **Smoke Tests** | 🔥 | [`.ai/smoketest.md`](./.ai/smoketest.md) | **Fast Pre-Flight & Health** — syntax compilation, targeted unit tests, linter gates (< 5s) | [`.ai/smoketest-journal.md`](./.ai/smoketest-journal.md) |
| 8 | **Bugfixer** | 🐛 | [`.ai/bugfixing.md`](./.ai/bugfixing.md) | **Bugs & Regressions** — audit changes, verify zero regressions, reads all journals | [`.ai/bugfixing-journal.md`](./.ai/bugfixing-journal.md) |

### 10.2 Master Multi-Agent Pipeline Execution Order

When executing comprehensive codebase upgrades, multi-domain enhancements, or end-to-end features, agents execute in strict dependency order:

```
Phase 0: 🔍 Vector Search Scan (python scripts/query_codebase.py "<query>")
   │
   ├─ Stage 1: ⚡ Bolt — Performance & Optimization (caching, async I/O, serialization, hot paths)
   ├─ Stage 2: 🎨 Palette — UX & Accessibility (dashboard HTML/CSS/JS, ARIA, responsive, DOM)
   ├─ Stage 3: 🛡️ Sentinel — Security & Hardening (auth, CSP headers, rate limiting, input validation)
   ├─ Stage 4: ✨ Refactor — Clean Code (DI enforcement, isinstance reduction, type clarity - AFTER features)
   ├─ Stage 5: ✂️ Concise — Code Line Reduction (if LOC reduction is possible: mixins, dispatch tables)
   ├─ Stage 6: 🔥 Smoke Tests — Rapid Pre-Flight Pass (< 5s compilation, ruff gate, startup sanity)
   └─ Stage 7: 🐛 Bugfixer — Regression Verification & Audit (run full test suite, verify journals)
```

### 10.3 Agent Workflow & Journaling Protocol

1. **Mandatory Journaling:** Every worker agent **must** append a summary entry to its corresponding `.ai/<name>-journal.md` file before completing a turn.
2. **Context Continuity:** Supervisor and worker agents must read relevant journals before initiating work to prevent regressions or duplicate changes.
3. **Bugfixer Final Gate:** Always run **Bugfixer** 🐛 as the final verification stage after any agent modifies source files to run full tests and verify project health.
4. **Autonomous Vector Hunt Queries:** Use domain-specific vector queries (`python scripts/query_codebase.py "<natural language query>"`) to target weakspots autonomously.


## 11. Workspace Customizations (.agents/)

### Workspace Customizations & Imported Memory (from Hermes Agent)

#### Developer & System Context
- **Repositories:** `LLM_trader` (decision engine), `llm_trader_executor` (optional order execution service)


#### Core Code Conventions

1. **Explicit Consent for Git Operations:**
   - Always require explicit user consent before running `git commit` or `git push`. Never commit or push without clear user approval.
   - Run tests before committing. Only commit when there are 0 failures. Commit all session changes together.
   - Lint first: `ruff check src/` before any commit.

2. **Dependency Injection & Architectural Invariants:**
   - **DI Only:** Codebase uses DI CompositionRoot pattern. All classes must receive dependencies (`self.logger`, `config`, etc.) via `__init__`, never construct their own.
   - **Logger Property:** Use public `self.logger` (never `self._logger`), because `@retry_async` decorator checks `instance.logger`.
   - **Zero Standalone Functions in `app.py`:** Every concern gets its own class (e.g. `ExecutorHandler`) wired once from `start.py` composition root. No redundant standalone files (e.g. `decision_payload.py`).
   - **No Type Introspection on Known Types:** `TradeDecision` is `@dataclass(slots=True)` with fixed fields. `Position` is `@dataclass(slots=True)`. Never use `hasattr`, `getattr`, or `isinstance` on known types. Access attributes directly or use `.get()` on raw dicts.

3. **Retry Decorators (Do NOT Hand-Roll Retry Loops):**
   - `@retry_async(max_retries=3, ...)` from `src/utils/decorators.py` for network and executor HTTP calls.
   - `@retry_api_call(max_retries=3, ...)` from `src/utils/decorators.py` for AI provider API calls.

4. **`TradeDecision` Field Contract:**
   - No `order_type` or `reduce_only` on the dataclass — those come from config `ENTRY_ORDER_TYPE` and `analysis.get("reduce_only", False)`.

5. **Front-End & Backwards Compatibility:**
   - **Cloudflare Assets:** JS/CSS changes require bumping `?v=` version tags in HTML `src` and `main.js` imports (1-year immutable cache).
   - **Zero Compatibility Leftovers:** Delete dead endpoints, cache prefixes, CSS, DOM IDs, and branding immediately. Never leave legacy code for "backwards compatibility".

6. **Test Suite Isolation:**
   - Delete `position_state.json` and `position_state.json.tmp` when running tests or set unique `state_path` to avoid stale state pollution.
   - Known pre-existing failures: 3 admin Playwright tests + 8 collection errors (NumPy/Numba constraint, stale import). These are not caused by code changes.

8. **Vector Database Search First Policy:**
   - Before making architectural edits, code refactoring, performance tuning, or security updates, ALL agents must execute `.venv\Scripts\python.exe scripts/query_codebase.py "<natural language query>"` to pinpoint exact bottleneck locations, symbol dependencies, and optimal code update targets across the repository.

---

#### AI Agents (in `.ai/`)

Eight specialized agents handle different aspects of the codebase. Load any prompt with `view_file(AbsolutePath=".ai/<name>.md")` for its full instructions.

| # | Agent | File | Focus | Journal |
|---|---|---|---|---|
| 🧠 | **Supervisor** | `.ai/supervisor.md` | Routes work, scans codebase via vector search, orchestrates full multi-agent pipelines | — |
| ⚡ | **Bolt** | `.ai/bolt.md` | Performance: caching, async patterns, I/O, serialization, numpy, hot paths | `.ai/journal.md` |
| 🎨 | **Palette** | `.ai/palette.md` | UX & Accessibility: dashboard HTML/CSS/JS, ARIA, keyboard nav, copy buttons, responsive | `.ai/palette-journal.md` |
| 🛡️ | **Sentinel** | `.ai/sentinel.md` | Security: auth, CSP, rate limiting, input validation, secret handling, XSS, CORS | `.ai/sentinel-journal.md` |
| ✨ | **Refactor** | `.ai/refactor.md` | Clean Code: isinstance chains, DRY violations, DI enforcement, exception handling | `.ai/refactor-journal.md` |
| ✂️ | **Concise** | `.ai/concise.md` | Code Line Reduction: DRY abstractions, mixins, inheritance, dispatch tables, expression conciseness | `.ai/concise-journal.md` |
| 🔥 | **Smoke Tests** | `.ai/smoketest.md` | Fast Pre-Flight & Health: syntax compilation, targeted unit tests, linter gates, startup sanity | `.ai/smoketest-journal.md` |
| 🐛 | **Bugfixer** | `.ai/bugfixing.md` | Bugs & Regressions: finding bugs, verifying other agents' changes, reads all journals | `.ai/bugfixing-journal.md` |

##### Master Multi-Agent Pipeline Execution Order

When executing comprehensive codebase upgrades, multi-domain enhancements, or end-to-end features, Supervisor orchestrates all specialized agents in strict dependency order:

```
Phase 0: Vector Search Scan (python scripts/query_codebase.py "<query>")
   │
   ├─ Stage 1: ⚡ Bolt — Performance (caching, async I/O, serialization, hot paths)
   ├─ Stage 2: 🎨 Palette — UX & Accessibility (dashboard UI, ARIA, responsive, DOM)
   ├─ Stage 3: 🛡️ Sentinel — Security & Hardening (auth, CSP headers, rate limits, validation)
   ├─ Stage 4: ✨ Refactor — Clean Code & DI Enforcement (after feature/security implementations)
   ├─ Stage 5: ✂️ Concise — Code Reduction (if LOC reduction is possible: mixins, dispatch tables)
   ├─ Stage 6: 🔥 Smoke Tests — Fast Pre-Flight Verification (< 5s compilation, linter gate, startup sanity)
   └─ Stage 7: 🐛 Bugfixer — Regression Verification & Test Suite Audit (run tests, log journals)
```

**Agent workflow rules:**
- Start with **Supervisor** for ambiguous requests or multi-domain work.
- Each agent works in isolation — one PR, one concern.
- Every agent **must** write a journal entry before creating a PR (MANDATORY, not optional).
- Always run **Smoke Tests** before full testing, and **Bugfixer last** after any agent makes changes — verifies zero regressions.
- Agents know about each other via "Companion Agents" sections in their prompts — they consult each other's scopes.

**Journal system:**
- 7 journal files track every change made by the worker agents.
- Bugfixer reads all journals before verifying to understand what changed and why.
- Use `cat .ai/<name>-journal.md` to see an agent's history.


## 12.1 Analysis Engine

### 🔬 Analysis Engine Agent — Technical Analysis & Pattern Recognition

> **Module path:** `src/analyzer/analysis_engine.py` (orchestrator) + collaborators in `src/analyzer/`
> **Type:** Market Data Collection → Technical Indicators → Chart Pattern Recognition → AI Signal Generation
> **Core Model:** Google Gemini 3.6 Flash (multimodal for chart image analysis)

---

#### Agent Persona & Role

The Analysis Engine is the **primary market sensing subsystem** — it transforms raw OHLCV data into a structured, multi-dimensional market assessment that the Brain Agent and Trading Strategy use to make decisions.

It performs four distinct analytical passes:
1. **Technical Calculation** — 40+ indicator arrays across momentum, trend, volatility, volume, statistical, and support/resistance categories
2. **Pattern Recognition** — Chart patterns (head & shoulders, double tops, triangles, wedges, channels) + indicator patterns (RSI, MACD, divergence, volume, stochastic, volatility)
3. **Chart Visualization** — 4K candlestick charts with SMA overlays, RSI, volume, CMF/OBV, swing annotations — passed to the LLM for visual pattern analysis
4. **LLM-Powered Signal Synthesis** — Combines all technical data + chart image + RAG context + brain context → structured BUY/SELL/HOLD signal

---

#### Inputs

##### From DataFetcher (via Exchange/CCXT)
- `ohlcv_data: np.ndarray` (columns: timestamp, open, high, low, close, volume) — primary timeframe (4h, up to 999 candles)
- Daily historical (365 days) and weekly macro (300 weeks) data
- Order book depth — multi-level depth buckets, near-mid liquidity, largest wall detection
- Recent trade flow — trade size distribution, buy/sell ratio

##### From External Providers
- CoinGecko — market-wide metrics (dominance, volume, sentiment)
- Alternative.me — Fear & Greed Index
- DeFiLlama — fundamentals (TVL, protocol metrics via RAG pipeline)

##### From RAG Engine
- Market context: recent news summaries, relevant articles filtered by taxonomy

##### From Brain Agent
- `brain_service.get_context()` — historical trade outcomes, learned rules, confidence calibration
- `brain_service.get_dynamic_thresholds()` — learned SL/TP/RR thresholds

##### Configuration (from `config/config.ini`)
- Pair: BTC/USDC, Timeframe: 4h, Candles: 999 (125 for AI chart)
- AI_CHART_CANDLE_LIMIT: 125 (max candles rendered in chart image)

---

#### Outputs

##### `analyze_market()` → Structured dict containing:
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

##### Validation Overrides (deterministic — always overwrite LLM claims)
- `TrendValidator` — cross-checks LLM ADX claims (±15 delta threshold), always uses computed value
- `PatternQualityScorer` — deterministic 0–100 score from 4 components (30% quantity, 30% confirmation, 20% recency, 20% indicator alignment), flags >25-point divergence from LLM

---

#### Prompting Strategy

##### System Prompt Construction
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

##### User Prompt Strategy
- Concise instruction asking for structured JSON output
- Includes optional chart image as base64-encoded PNG (4K resolution, 4-row layout)
- Provider-emitted `<think>...</think>` sections are stripped by `AnalysisResultProcessor._clean_response()` before JSON parsing
- Response schema: `TradingAnalysisResponseModel` with validated `TradingAnalysisModel`

##### Response Parsing
`UnifiedParser` handles:
- JSON extraction from ` ```json ` code blocks
- Raw JSON extraction via `json.JSONDecoder.raw_decode()`
- Fallback response if both fail (HOLD, neutral confidence)
- Pydantic validation via `TradingAnalysisResponseModel`
- Signal validation: BUY/SELL requires entry_price, stop_loss, take_profit, risk_reward_ratio, position_size

---

#### Subsystems Detail

##### TechnicalCalculator (`technical_calculator.py`)
40+ indicator arrays computed fresh each cycle:

| Category | Indicators |
|----------|-----------|
| **Volume** | VWAP, TWAP, MFI, OBV, CMF, Force Index, CCI, PVT, A/D Line |
| **Momentum** | RSI (14), Stochastic (14,3,3), Williams %R, UO, TSI, RMI, PPO, Coppock, KST, ROC, MACD (12,26,9) |
| **Volatility** | ATR (20), Bollinger Bands (20,2), %B, Keltner (20,2), Donchian (20), Chandelier Exit (20,3), Choppiness (14) |
| **Trend** | ADX (14), +DI/-DI, TRIX, PFE, TD Sequential, Parabolic SAR, Supertrend (20,3), Ichimoku (9,26,52), Vortex, SMAs (20/50/200) |
| **S/R** | Kurtosis, Z-score, Hurst, Entropy, Skewness, Variance, LinReg slope/r², basic S/R, advanced S/R, Pivot Points, Fibonacci Pivots |

**Weekly Macro** uses 200W SMA methodology: 5 bullish/bearish criteria scored for cycle phase confidence.

##### PatternEngine (`pattern_engine/`)

All deterministic indicator-pattern detection is Numba `@njit(cache=True)` compiled for performance.

**Chart Patterns** are visually detected by the LLM from the chart image (via `ChartGenerator`, processed through `analysis_result_processor.py`). The `PatternAnalyzer` (`pattern_analyzer.py`) orchestrates indicator pattern detection only, delegating to `IndicatorPatternEngine` (`indicator_pattern_engine.py`).

**Indicator Patterns** (via `pattern_engine/indicator_patterns/indicator_pattern_engine.py`):
7 categories — RSI (oversold/overbought, W-bottom, M-top), MACD (crossovers, histogram), Divergence (bull/bear with 5-candle min spacing), Volume (spike, climax, dry-up, accumulation/distribution), Stochastic (oversold/overbought, crossovers), MA Crossovers (golden/death, alignments), Volatility (ATR spike, BB squeeze, TTM squeeze).

##### ChartGenerator (`pattern_engine/chart_generator.py`)
- Resolution: 3840×2160 (4K)
- Layout: 4 rows — Candlestick + SMA (55%), RSI (15%), Volume (15%), CMF + OBV (15%)
- AI-optimized: black background, high-contrast colors, swing point annotations, global max/min labels
- Resilience: 30s timeout per export, up to 3 retries with exponential backoff
- Format: PNG via Plotly + Kaleido

---

#### Model Configuration (from `config.ini` `[model_config]`)

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

#### Edge Cases & Guardrails

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


## 12.2 Dashboard

### 📊 Dashboard Agent — Real-Time Web UI

> **Module path:** `src/dashboard/server.py` (FastAPI) + 5 routers in `src/dashboard/`
> **Type:** FastAPI + WebSocket streaming dashboard
> **Live URL:** [https://semanticsignal.qrak.org](https://semanticsignal.qrak.org)

---

#### Agent Persona & Role

The Dashboard is the **real-time monitoring and visualization interface** for the LLM Trader system. It provides:

- **Live decision streaming** — WebSocket push of each analysis cycle result
- **Performance analytics** — SQLite-backed trade history, P&L curves, win rates
- **Brain state inspection** — vector memory contents, learned rules, confidence stats
- **System health monitoring** — provider status, token costs, cycle timing

---

#### Architecture

##### Server (`server.py`)
Injected via DI from `start.py`:
```
DashboardServer(
  brain_service, vector_memory, analysis_engine,
  config, logger, unified_parser, persistence, exchange_manager
)
```

##### 5 API Routers

| Router | File | Endpoints |
|--------|------|-----------|
| **Brain** | `routers/brain.py` | `/api/brain/*` — decisions, signals, context |
| **Monitor** | `routers/monitor.py` | `/api/monitor/*` — system health, provider status |
| **Performance** | `routers/performance.py` | `/api/performance/*` — P&L, statistics, trade history |
| **Visuals** | `routers/visuals.py` | `/api/visuals/*` — chart data, indicator plots |
| **WebSocket** | `routers/ws_router.py` | `/ws/*` — real-time streaming |
| **Admin** | `routers/admin.py` | `/api/admin/*` — config CRUD, system control, log streaming, auth |

##### Admin Console (`/admin`)
- **Auth:** HMAC-SHA256 cookie sessions. Credentials from `keys.env` (`ADMIN_USERNAME`, `ADMIN_PASSWORD_HASH`, `ADMIN_SIGNING_KEY`)
- **Config write:** `WritableConfig` (async atomic INI writes via `os.replace()`). Hot-reload signal via `asyncio.Event`
- **Control:** Force analysis (`POST /api/admin/system/trigger-analysis`), toggle feed (`POST /api/admin/system/toggle-feed`)
- **Human input:** `POST /api/admin/system/human-input` — consumed by bot on next cycle
- **Log streaming:** `LogStreamHandler` → subscriber `asyncio.Queue`s. WS at `/api/admin/logs/stream?token=...`
- **Console WS:** Bidirectional at `/api/admin/console?token=...` — accepts `force_analysis`, `toggle_feed`, `human_input`, `get_status`
- **Frontend:** Vanilla JS + Tailwind CDN at `src/dashboard/static/admin/index.html`

##### Static Frontend
- `src/dashboard/static/` — HTML, Vanilla JS, Vis.js, ApexCharts
- Dark theme, auto-refreshing WebSocket data
- Real-time candlestick charts + indicator overlays

---

#### Key Behaviors

- **Startup:** Bound to `0.0.0.0:8000` (configurable)
- **CORS:** Disabled by default; enabled only when `config.ini` sets `dashboard.enable_cors = true`
- **GZip:** Compression enabled for API responses
- **Static files:** Mounted from `static/` directory
- **Lifecycle:** Managed via FastAPI lifespan context manager
- **State:** Shared via `dashboard_state.py` singleton
- **Trade history:** Performance and brain endpoints read via injected `PersistenceManager`; direct `trade_history.json` reads are forbidden

#### Persistence Contract

- `PerformanceRouter` receives `persistence` from `DashboardServer` and calls persistence-backed history APIs.
- Brain/vector endpoints use `persistence.load_trade_history()` when they need trade-history context.
- Dashboard routes must not open runtime files directly for trade history; SQLite access stays behind `PersistenceManager` / `SQLiteTradeHistory`.
- Empty SQLite history should render empty dashboard state gracefully, not as an error.

---

#### Cache Strategy (Cloudflare)

| Rule | Target | Reason |
|------|--------|--------|
| **Bypass** | `/api/brain/refresh-price` | Volatile price endpoint |
| **Bypass** | `/api/brain/vectors?query=*` | High-cardinality search |
| **Cache** | `/api/status/countdown` | Static countdown data |
| **Cache** | `/api/*` | Safe GET traffic |
| **Cache** | HTML shell pages | Static shell |
| **Cache** | Static assets | Versioned assets |

---

#### Edge Cases

| Scenario | Handling |
|----------|----------|
| **WebSocket disconnect** | Client auto-reconnects, resends subscription |
| **No data yet** | Returns empty state gracefully |
| **No SQLite trade history rows** | Performance and brain routes return empty history/state gracefully |
| **Dashboard server toggle/restart in-process** | DashboardState retains last-known values while the Python process remains alive |
| **Large vector memory queries** | Truncated/paginated API responses |
| **Concurrent dashboard access** | FastAPI async handles concurrent requests |


## 12.3 Indicators

### 📊 Numba JIT Indicator Library — 50+ Home-Grown Technical Indicators

> **Module path:** `src/indicators/` (9 category directories + `base/`)
> **Type:** Self-contained technical analysis library, Numba JIT-compiled
> **Size:** ~96 `@njit(cache=True)` functions across 13 Python files
> **Design Decision:** Built from scratch instead of using TA-Lib / pandas-ta — 0 external TA dependencies

---

#### Why A Custom Indicator Library?

LLM Trader does **not** use TA-Lib or pandas-ta. Every indicator is implemented from scratch in pure Python + Numba `@njit` for JIT compilation. This was a deliberate architectural choice:

| Concern | TA-Lib | This Library |
|---------|--------|-------------|
| **Dependencies** | C library, platform-specific builds, `.dll`/`.so` hell | Pure Python + Numba (already a dependency) |
| **Multi-timeframe isolation** | Single global state | `TechnicalCalculator` creates fresh `TechnicalIndicators` instances per timeframe |
| **NaN handling** | Inconsistent per indicator | Uniform: pre-fill with `np.nan`, first valid at index `length` |
| **Rolling window bugs** | None | Several found and fixed: CCI O(N×L)→O(N), stochastic NaN bleed-through, MACD 0.0 sentinel |
| **Customization** | Fork or wrapper layer | Direct: `@njit` your own variant alongside originals |
| **Modularity** | One giant DLL | 9 category modules, tree-shakeable imports |

---

#### Architecture

##### Layer Diagram

```
TechnicalIndicators (facade — 862 lines, methods directly on class)
  └── IndicatorBase (data holder — 160 lines)
        ├── get_data(OHLCV) → numpy arrays
        └── calculate_indicator(func, *args) → timing + CSV logging wrapper

Category Modules (each a single .py file)
  ├── momentum_indicators.py     — 14 @njit functions (RSI, MACD, Stochastic, ...)
  ├── trend_indicators.py        — 8 @njit + 8 trend_calculation_utils + 4 sar_utils
  ├── volume_indicators.py       — 13 @njit (MFI, OBV, CMF, CCI, VWAP, TWAP, ...)
  ├── volatility_indicators.py   — 8 @njit (ATR, Bollinger, Keltner, Choppiness, ...)
  ├── statistical_indicators.py  — 14 @njit + 4 correlation + 2 DSP filters
  ├── support_resistance_indicators.py — 9 @njit (pivot points, Fibonacci, floating levels)
  ├── overlap_indicators.py      — 3 @njit (SMA, EMA, EWMA)
  ├── price_transform_indicators.py — 3 @njit (log return, % return, price distribution)
  └── sentiment_indicators.py    — 6 @njit (Fear & Greed Index variants)
```

##### Import Structure

All category functions are re-exported at the module level via `__init__.py` imports. The `TechnicalIndicators` class in `base/technical_indicators.py` imports them all and exposes each as a direct method:

```python
ti = TechnicalIndicators()
ti.get_data(ohlcv_array)
rsi_values = ti.rsi(length=14)         # Direct — no delegation
macd_line, signal, hist = ti.macd()
```

There is **no delegation layer** — the category sub-object pattern was eliminated in a refactor. Every indicator is a one-line direct method call.

---

#### Numba JIT Compilation Pattern

##### Standard Template

Every indicator function follows this pattern:

```python
@njit(cache=True)
def rsi_numba(close: np.ndarray, length: int) -> np.ndarray:
    n = len(close)
    result = np.full(n, np.nan)       # Pre-fill NaN

    # Sliding window computation — no Python list allocations
    for i in range(length, n):
        ...

    return result
```

Key rules:
- **`@njit(cache=True)`** — JIT compiled, cached to disk after first call (avoids recompilation on restart)
- **Input is always `np.ndarray`** — float64 preferred, int64 auto-converted
- **Output is always `np.ndarray`** — pre-allocated with `np.full(n, np.nan)`
- **No Python objects in hot loops** — lists, dicts, and function calls inside loops are forbidden by Numba
- **`math` module** — `math.nan`, `math.isnan()`, `math.inf` are allowed (compiled to C)
- **Return type** — scalar (single value), 1D array, or tuple of arrays

##### Performance Baseline

| Indicator | 1000 candles | 100k candles | Speedup vs vanilla Python |
|-----------|-------------|--------------|--------------------------|
| RSI(14) | ~0.0001s | ~0.002s | ~50× |
| MACD(12,26,9) | ~0.0002s | ~0.005s | ~80× |
| CCI(14) | ~0.0003s | ~0.004s | ~6× (fixed from bug) |
| Bollinger (20,2) | ~0.0002s | ~0.003s | ~60× |
| ADX(14) | ~0.0005s | ~0.008s | ~40× |

**Real-world workload:** Computing all 50+ indicators across 999 candles takes ~15-25ms total.

---

#### Indicator Inventory (Complete)

##### Momentum (14 functions)
RSI, MACD (line/signal/histogram), Stochastic (%K/%D), ROC, Momentum, Williams %R, TSI, RMI, PPO, Coppock Curve, Ultimate Oscillator, KST, Relative Strength calculation, RSI divergence detection

##### Trend (8 functions + 12 utility functions)
ADX (+DI/-DI), Supertrend, Ichimoku Cloud (tenkan/kijun/senkou/chikou), Parabolic SAR, Vortex (+VI/-VI), TRIX, PFE, TD Sequential (setup/countdown)

**Utility files:**
- `trend/trend_calculation_utils.py` — true range, ATR helper, directional movement, rolling true range sum
- `trend/sar_utils.py` — acceleration factor logic, SAR point stepping

##### Volume (13 functions)
MFI, OBV, OBV Slope, PVT, Chaikin Money Flow, Accumulation/Distribution Line, Force Index, Ease of Movement, Volume Profile, Rolling VWAP, TWAP, Average Quote Volume, CCI

##### Volatility (8 functions)
ATR, Bollinger Bands (upper/lower/%B/width), Chandelier Exit (long/short), VHF, EBSW, Keltner Channels (upper/lower), Donchian Channels (upper/lower), Choppiness Index

##### Statistical (14 functions + 4 correlation + 2 DSP)
Kurtosis, Skewness, Standard Deviation, Variance, Z-Score, MAD, Quantile, Entropy, Hurst Exponent, Linear Regression (slope/r²/intercept), APA Adaptive EOT, EOT Calculation

**Utility sub-package** `statistical/utils/`:
- `statistical/utils/correlation_analysis.py` — 4 functions: autocorrelation, rolling correlation, cross-correlation, Spearman rank
- `statistical/utils/dsp_filters.py` — 2 functions: low-pass filter, high-pass filter (basic IIR-style)

##### Support/Resistance (9 functions)
Support & Resistance (basic), Find S/R (swing-point based), Advanced S/R (cluster detection), Pivot Points (classic), Fibonacci Pivot Points, Fibonacci Retracement, Floating Levels, Fibonacci Bollinger Bands

##### Overlap (3 functions)
SMA, EMA, EWMA (all support array inputs for vectorized calculation)

##### Price Transforms (3 functions)
Log Return, Percent Return, Price Distribution

##### Sentiment (6 functions)
Fear & Greed Index (5 market-based variants), Fear & Greed with configurable thresholds

---

#### Multi-Timeframe Isolation

Each analysis cycle creates **three isolated `TechnicalIndicators` instances** via `TechnicalCalculator`:

| Instance | Candle Data | Purpose |
|----------|-------------|---------|
| Current timeframe indicators | 999 candles (4h) | Primary cycle analysis |
| Long-term indicators | 365 daily candles | Long-term trend context |
| Weekly macro indicators | 300 weekly candles | Macro cycle phase |

Each instance holds its own `open/high/low/close/volume` numpy arrays. State interference between timeframes is impossible by construction.

---

#### NaN Propagation Convention

Every indicator uses a **uniform NaN strategy**:

1. **Output array**: Pre-filled with `np.nan`
2. **First valid index**: Indicator-specific, typically based on each function's `required_length` (for example RSI(14) starts at index 14, while some multi-parameter indicators can start earlier or later)
3. **Insufficient data**: If `len(data) < length`, entire output is NaN
4. **Division by zero**: Checked with `if avg_loss == 0` guards; result set to 100 (RSI), 0 (CCI), etc.
5. **NaN in input**: Not handled explicitly — caller guarantees clean data (DataFetcher excludes incomplete candles upstream)

This convention means downstream consumers (TechnicalCalculator, prompt formatters) must handle NaN at array boundaries — which they do by slicing to the visible candle count.

---

#### Correctness History

Several bugs were found and fixed during development that would have been invisible with TA-Lib:

| Bug | Symptom | Root Cause | Fix |
|-----|---------|-----------|-----|
| **CCI drifting** | Values diverged from reference as data grew | `np.roll()` allocating new arrays each iteration → O(N×L) with accumulation error | Single-pass sliding sum → O(N) |
| **Stochastic NaN bleed** | First %K value was 0.0 instead of NaN | Missing NaN assignment before first valid period | Pre-fill with `np.full(n, np.nan)` |
| **MACD 0.0 sentinel** | First histogram values were literal 0.0 | MACD line started at idx 25 but histogram used 0.0 as unset sentinel | Explicit NaN pre-fill |
| **Bollinger %B** | Out-of-bounds values (>1.0 or <0.0) on very first valid candle | Division using incomplete rolling window | Skip %B calculation until `length` candles into window |

---

#### Edge Cases & Guardrails

| Scenario | Handling |
|----------|----------|
| **Zero volume array** | Division by zero guards in VWAP, MFI, CMF — result `np.nan` with downstream fallback |
| **Flat price (all identical)** | RSI = 100 (overbought/no-loss endpoint — average loss is zero), ADX = 0, volatility = 0 |
| **Single candle** | All indicators return all-NaN — `required_length` validation catches this upstream |
| **Non-float64 input** | `get_data()` normalizes to `np.float64` — int/timestamps auto-converted |
| **Extreme values (>1e10)** | Floating point saturation possible — no explicit clamp (BTC/USDC at <10⁶ is safe) |
| **Memory fragmentation** | Each cycle creates 3× indicator instances (~15MB of arrays) — GC collects between cycles |


## 12.4 Managers

### ⚙️ Risk Manager, Persistence Manager & Provider Orchestrator

> **Module path:** `src/managers/risk_manager.py` + `src/managers/persistence_manager.py` + `src/managers/sqlite_trade_history.py` + `src/managers/provider_orchestrator.py`
> **Type:** Signal Execution Safety + SQLite Persistence + AI Provider Fallback Chain

---

#### 1. Risk Manager Agent

> **File:** `src/managers/risk_manager.py`

##### Agent Persona & Role

The Risk Manager is the **safety layer between the LLM-generated trading signal and actual order execution.** It converts raw AI signals into a validated `RiskAssessment` by applying dynamic position sizing, SL/TP scaling, and consistency checks.

It is **not** a full risk management system — real exchange order execution is not yet implemented. The Risk Manager operates in paper-trading simulation mode.

##### Inputs

- `TradingAnalysisModel` from `UnifiedParser` — signal, confidence, entry_price, stop_loss, take_profit, position_size
- `MarketConditions` — current price, ATR, volatility indicators
- `Position` (existing) — for UPDATE/CLOSE signal handling
- **Brain-learned thresholds** — dynamic SL/TP/RR/confluence thresholds from `get_dynamic_thresholds()`

##### Outputs

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

##### Core Logic: Dynamic SL/TP Scaling

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

##### R:R Enforcement — Two-Layer Defense

RiskManager computes `rr_ratio` = TP distance / SL distance but does **NOT** reject on it.
TradingStrategy enforces brain-learned `rr_borderline_min` (default **1.5**, adaptively learned from historical R:R performance):

```python
if rr_ratio < brain_thresholds.get("rr_borderline_min", 1.5):
    # Blocked as guard_type="rr_minimum" → stored as blocked-trade feedback
```

##### Friction Lifecycle

1. `RiskManager.calculate_entry_parameters()` → accumulates frictions in `_last_frictions` list during SL/TP clamping
2. `TradingStrategy._open_new_position()` → calls `get_and_clear_frictions()` after RiskAssessment
3. Each friction → `vector_memory.store_blocked_trade()` → Brain Agent learns from clamping events

##### Position Sizing

| Confidence Level | Fallback Size |
|-----------------|---------------|
| HIGH (string; numeric extractor maps ≥70) | 3% |
| MEDIUM (numeric extractor maps 50–69) | 2% |
| LOW (numeric extractor maps <50) | 1% |
| Max position | 10% of portfolio (configurable) |

##### Edge Cases & Guardrails

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

#### 2. Persistence Manager Agent

> **Files:** `src/managers/persistence_manager.py`, `src/managers/sqlite_trade_history.py`

##### Persistence Manager Persona & Role

Persistence Manager is the **single persistence facade for trading runtime state.** It owns positions, statistics, monitor state, previous/last analysis snapshots, and trade-history access. Trade history is SQLite-only and must not fall back to legacy JSON files.

##### SQLite Trade History Contract

| Responsibility | Owner | Rule |
|----------------|-------|------|
| Trade append | `PersistenceManager.save_trade_decision()` → `SQLiteTradeHistory.insert()` | SQLite-only; raise if persistence fails |
| Full history export | `PersistenceManager.load_trade_history()` | Export from SQLite only |
| Entry-decision lookup | `get_entry_decision_for_position()` | Query SQLite by timestamp window and optional symbol |
| Cooldown timestamp | `get_last_execution_timestamp()` | Query newest BUY/SELL timestamp from SQLite |
| Dashboard history | Dashboard routers via injected persistence | No direct file reads |

##### Non-Negotiable Rules

- Do not reintroduce `trade_history.json` runtime reads, writes, fallback paths, or auto-migration.
- Do not pass `json_path` into `SQLiteTradeHistory`; its constructor accepts only `logger` and `db_path`.
- Historical `.json.migrated` files are backups only and are not runtime inputs.
- SQLite write failure is a hard persistence failure; do not silently continue with an alternate store.
- Keep all service dependencies injected from the composition root; do not construct persistence dependencies inside trading, dashboard, or guard classes.

##### SQLite Store Behavior

- WAL journal mode and `synchronous=NORMAL` are enabled per connection.
- Inserts coerce `TradeDecision` fields into stable SQLite column types.
- Queries validate sort direction (`ASC`/`DESC`) and clamp pagination.
- `get_stats()` provides aggregate dashboard data without scanning JSON files.

##### Edge Cases

| Scenario | Handling |
|----------|----------|
| **No trade history rows** | Returns empty list / `None` timestamp; callers decide no-history behavior |
| **Invalid query order** | Raises `ValueError` before SQL generation |
| **SQLite insert returns no row id** | `PersistenceManager.save_trade_decision()` raises `RuntimeError` |
| **Entry timestamp not found** | Returns `None` and logs warning |
| **Malformed persisted timestamp** | Entry lookup skips malformed candidate rows |

---

#### 3. Provider Orchestrator Agent

> **File:** `src/managers/provider_orchestrator.py`

##### Agent Persona & Role

The Provider Orchestrator manages the **AI provider lifecycle and fallback chain** — routing analysis requests to the best available LLM provider with automatic degradation when the primary provider is unavailable, rate-limited, or returning errors.

##### Provider Registry

| Name | Client | Default Model | Chart Support | Fallback Model |
|------|--------|---------------|---------------|----------------|
| `googleai` | `GoogleAIClient` | Google Gemini 3.6 Flash | ✅ Yes | Google paid tier |
| `openrouter` | `OpenRouterClient` | Configurable base model (`google/gemini-3-flash-preview` by default) | ✅ Yes | `deepseek/deepseek-r1:free` by default |
| `local` | `LMStudioClient` | LM Studio model | Disabled in orchestrator | — |

##### Fallback Chain Strategy

**Text requests:**
```
Primary: googleai (Gemini 3.6 Flash)
  → Rate limited / overloaded? → googleai paid tier (auth errors fall through to next provider)
  → Still failing? → local (LM Studio, text-only path)
  → Still failing? → openrouter (configured base model)
  → Still failing? → openrouter fallback model
  → All providers failed? → HOLD with error
```

**Chart (multimodal) requests:**
```
Primary: googleai (Gemini 3.6 Flash)
  → Rate limited / overloaded? → googleai paid tier (auth errors fall through to next provider)
  → Still failing? → openrouter (configured base model)
  → Still failing? → openrouter fallback model
  → All providers failed? → HOLD with error
```
Note: `local` is skipped in the orchestrator's chart fallback chain because provider metadata marks it as chart-unsupported, even though the LM Studio client has a chart-analysis method.

##### Key Behaviors

- **Parameter auto-retry:** provider clients use `_execute_with_param_retry()` to catch unsupported parameter errors, strip the offending parameter, and retry (up to 3×)
- **Known unsupported params** filtered pre-emptively: `thinking_budget`, `thinking_config`, `top_k`, `freq_penalty`, `pres_penalty`
- **Chart analysis routing:** Only `googleai` and `openrouter` are enabled for orchestrated multimodal (text + image) requests; explicit `local` chart requests return an error before fallback
- **Cost tracking via OpenRouter** `get_generation_cost()` — async query after response for token + cost data
- **Error classification:** quota, auth, timeout, overloaded, connection — each maps to a specific error response the caller uses for fallback decisions
- **API key redaction:** `_sanitize_error_message()` redacts API keys from all error logs

##### Client Implementations

| Client | Transport | Image Support | Retry |
|--------|-----------|---------------|-------|
| `GoogleAIClient` | Official `google.genai` SDK | Base64-encoded inline | `@retry_api_call` |
| `OpenRouterClient` | Official `openrouter` SDK (`OpenRouter`) | Base64 data URI | `@retry_api_call` + param retry |
| `LMStudioClient` | Official LM Studio Python SDK (`lmstudio.AsyncClient`) | Available in client, but disabled by orchestrator metadata and skipped by the chart fallback chain | `@retry_api_call` |

##### Provider Orchestrator Edge Cases

| Scenario | Handling |
|----------|----------|
| **Rate limited** → retry with backoff | Exponential backoff: 1s → 2s → 4s → ... → 30s max, 3 retries |
| **Unsupported parameter** → retry without it | Detected via error message regex, stripped from config, retried |
| **SDK version mismatch** | OpenRouter: falls back to `OpenRouter(api_key=...)` without `server_url` |
| **All providers fail** | Returns HOLD signal via fallback response |
| **Chart provider disabled for images** | Orchestrator returns an error for `local`; chart fallback chain only includes `googleai` and `openrouter` |
| **Non-text response parts** (Google) | Silently filtered, text parts extracted |
| **Provider not in registry** | `get_metadata()` returns None → caller uses provider not found error |


## 12.5 RAG Engine

### 📰 RAG Engine Agent — News Aggregation & Market Fundamentals

> **Module path:** `src/rag/rag_engine.py` (orchestrator) + 15+ collaborators in `src/rag/`
> **Type:** Retrieval-Augmented Generation pipeline for news and market fundamentals
> **Sources:** CoinDesk, CoinTelegraph, Decrypt, CryptoSlate, RSS feeds + Crawl4AI enrichment + DefiLlama

---

#### Agent Persona & Role

The RAG Engine is the **news and fundamentals intelligence subsystem**. It aggregates, enriches, categorizes, and indexes crypto market news and fundamentals data into a context window that the AnalysisEngine injects into every LLM decision prompt.

Unlike the Reading Agent, the RAG Engine does **not** send queries to an LLM — it performs deterministic information retrieval over a curated, time-decayed local index. The output is a text block of recent, relevant articles and fundamental metrics formatted for prompt injection.

##### Key Collaborators

| Module | File | Responsibility |
|--------|------|----------------|
| `NewsManager` | `news_manager.py` | Fetch → deduplicate → persist news lifecycle |
| `NewsRepository` | `news_repository.py` | Read/write interface for news storage |
| `ContextBuilder` | `context_builder.py` | Keyword search + token-limited context formatting |
| `ArticleScoringPolicy` | `scoring_policy.py` | 5-factor relevance scoring for news articles |
| `LocalTaxonomy` | `local_taxonomy.py` | Domain-specific crypto category hierarchy |
| `TickerManager` | `ticker_manager.py` | Coin/ticker ↔ name mapping for symbol detection |
| `ArticleProcessor` | `article_processor.py` | Normalization, body extraction, boilerplate stripping |
| `CategoryProcessor` | `category_processor.py` | News → taxonomy category classification |
| `IndexManager` | `index_manager.py` | 4 in-memory indices: category, tag, coin, keyword |
| `RSSProvider` | `news_ingestion/rss_provider.py` | RSS feed polling from configured sources |
| `Crawl4AIEnricher` | `news_ingestion/crawl4ai_enricher.py` | Web-page enrichment via Crawl4AI (optional) |
| `SchemaMapper` | `news_ingestion/schema_mapper.py` | Maps source-specific schemas to unified format |
| `MarketDataManager` | `market_data_manager.py` | Lifecycle management for market data |
| `MarketDataCache` | `market_components/market_data_cache.py` | Caching layer for market data fetches |
| `MarketDataFetcher` | `market_components/market_data_fetcher.py` | Fetches market data from CoinGecko/DeFiLlama |
| `MarketOverviewBuilder` | `market_components/market_overview_builder.py` | Builds market overview text from raw data |

---

#### Inputs

##### From RSS/Web
- RSS feed XML from configured crypto news sources
- Raw HTML pages (via Crawl4AI enrichment fallback chain)
- Per-source raw and normalized JSON artifacts (saved to `data/news_fetch_preview/`)

##### From External APIs
- CoinGecko — market-wide stats (dominance, volume, BTC dominance)
- DeFiLlama — TVL data, protocol fundamentals

##### From Configuration
- `config/config.ini`: news update interval (4h), max articles (5)
- `rag_priorities.json`: important categories and generic category priorities (not per-ticker weights)
- `LocalTaxonomy`: predefined category hierarchy for news classification

##### From Query (Consumer-side — AnalysisEngine)
- Keyword search terms (derived from active trading pair, trending coins)
- Token limit for context window truncation

---

#### Outputs

##### RAG Context Block (for LLM prompt injection)
- Formatted string: "--- Market News & Fundamentals ---\n[Article 1 summary]\n[Article 2 summary]\n..."
- Token-limited to fit within model context window
- Timestamp-stamped for freshness awareness

##### Persisted State
- `data/news_cache/` — recent news JSON (per-source, categorized)
- `data/news_fetch_preview/` — pre-enrichment raw + normalized artifacts
- `data/backup/news_cache/` — cold storage for historical retrieval

##### Indexed Artifacts (via IndexManager)
4 in-memory indices maintained in sync:
- **category_index** — taxonomy category → article indices
- **tag_index** — article tag → article indices
- **coin_index** — detected coin/ticker → article indices
- **keyword_index** — category and title keywords → article indices

---

#### Pipeline: News Ingestion Flow

```
RSS Provider polls configured RSS sources (4 by default, every 4h)
    ↓
Crawl4AI Enricher (optional — degrades to aiohttp on failure)
    ↓
SchemaMapper → unified format
    ↓
ArticleProcessor (dedup, normalize, strip boilerplate)
    ↓
CategoryProcessor → LocalTaxonomy mapping
    ↓
ArticleScoringPolicy (5-factor relevance scoring)
    ↓
CategoryCollisionResolver (category-word priority collisions in collision_resolver.py)
    ↓
NewsRepository → IndexManager (category/tag/coin/keyword)
    ↓
ContextBuilder → token-limited formatted context block
```

##### Scoring Policy — Scoring Factors

Current scoring uses keyword, category, coin, importance, density, cooccurrence, coin-relevance, and recency logic (no source-authority weighting).

1. **Recency** — newer articles score higher
2. **Ticker relevance** — coin/ticker mention count
3. **Category match** — alignment with trading pair category
4. **Body length / completeness** — penalizes truncated or thin articles

##### Enrichment Fallback Chain
```
Crawl4AI (primary) → aiohttp direct fetch (degradation) → store raw RSS text
```

---

#### Edge Cases & Guardrails

| Scenario | Handling |
|----------|----------|
| **Crawl4AI unavailable** | Degrades gracefully to aiohttp direct fetch |
| **RSS feed down** | Skips source, logs warning, continues with other sources |
| **Duplicate article detected** | Article deduplication is URL-first via `news_manager.py` and `rss_primitives.py` (no fuzzy matching) |
| **Empty news cycle** | Returns empty context — AnalysisEngine continues without news |
| **Provider rate-limited** | Cache fallback — serves stale data from `news_cache/` |
| **Body too short / low density** | Density penalty defaults to 300 in `loader.py`; body re-enrichment threshold mirrors 400 in `news_manager.py` |
| **Token limit exceeded** | `ContextBuilder` truncates to token budget, keeps highest-scored articles |
| **Index corruption** | `file_handler.py` provides file-based fallback storage |
| **New ticker not in TickerManager** | Dynamic symbol detection with name→ticker resolution |
| **Market data fetch fails** | `MarketDataFetcher` falls back to cache, stale data flagged |
| **Boilerplate in article body** | `ArticleProcessor` strips common boilerplate patterns |
| **Schema mismatch per source** | `SchemaMapper` handles per-source field name variants |
| **Concurrent fetch cycles** | `NewsManager` serializes via lock on update interval |


## 12.6 Trading Brain

### 🧠 Brain Agent — TradingBrainService

> **Module path:** `src/trading/brain.py` (facade) + collaborators in `src/trading/`
> **Type:** Central LLM Decision Engine & Outcome-Aware Learning Loop
> **Core Model:** No direct LLM calls; deterministic/vector-memory service whose context is injected into AnalysisEngine prompts routed by ProviderOrchestrator

---

#### Agent Persona & Role

The Brain Agent is the **central reasoning and learning subsystem** of LLM Trader. It acts as both:

- **Decision Enricher:** Injects historical trade outcomes, confidence calibration, blocked-trade feedback, and learned rules into every LLM prompt so the model makes context-aware decisions.
- **Autonomous Learner:** After each closed trade, records the outcome into ChromaDB vector memory, and periodically reflects on trade clusters to synthesize semantic rules (best-practices, anti-patterns, corrective measures, AI-mistake patterns).

The Brain does **not** execute trades or calculate indicators — it operates purely on metadata produced by the AnalysisEngine and TradingStrategy.

##### Key Collaborators

| Module | File | Responsibility |
|--------|------|----------------|
| `BrainContextProvider` | `brain_context.py` | Assembles the "Trading Brain" context block injected into every LLM decision prompt — includes blocked-trade feedback from ChromaDB |
| `BrainExperienceRecorder` | `brain_experience.py` | Translates closed-trade data + market conditions into structured vector-memory experiences |
| `ExitProfileResolver` | `brain_exit_profiles.py` | Single source of truth for SL/TP execution profiles serialization and rule rendering |
| `TradePatternAnalyzer` | `brain_patterns.py` | Statistical analysis engine — win/loss grouping, failure diagnostics, AI mistake classification |
| `BrainReflectionEngine` | `brain_reflection.py` | Synthesizes learned semantic rules from trade metadata clusters via periodic reflection loops |
| `ExecutorHandler` | `executor_handler.py` | Builds executor payload from analysis + strategy decision, persists to `latest_decision.json`, HTTP-forwards to `llm_trader_executor` |

---

#### Inputs

##### From TradingStrategy (via `close_position()`)
- `Position` — entry/exit/PNL, confidence, confluence factors, drawdown metrics
- `close_price` — exit price at time of close
- `close_reason` — stop_loss / take_profit / analysis_signal
- `entry_decision` — original entry TradeDecision (retrieved from SQLite trade history through `PersistenceManager` for reasoning context)
- `market_conditions` — MarketConditions at close time (or from entry if preferred)

##### From VectorMemoryService (ChromaDB)
- Trade experiences: 20+ metadata fields per trade (entry confidence, AI reasoning, ADX/RSI/ATR at entry, volatility, SL/TP distances, RR ratio, max drawdown/profit, fear & greed, market regime, confluence count, timeframe alignment, exit execution context, factor scores)
- Semantic rules: best_practice, anti_pattern, corrective, ai_mistake types
- Blocked-trade feedback: guard-type grouped rejection history
- Confidence calibration stats: win rate by confidence level, direction bias, ADX performance, factor performance

---

#### Outputs

##### `get_context()` — Formatted text block for LLM prompt injection
- Confidence calibration by level (HIGH/MEDIUM/LOW with win rate, trade count, avg P&L)
- Direction bias check (long vs short count, "LIMITED DATA" warning)
- Blocked-trade feedback (recent 5, ≤ 168h)
- Vector-retrieved similar past experiences (top-5 semantic similarity search)
- CoT Step 6 — Historical Evidence instructions (win-rate < 50% → reduce confidence, anti-pattern matching, AI-mistake memory, exit-execution memory)
- Learned trading rules matched to current conditions (similarity %, timeframe freshness, evidence score, tagged by type)

##### `get_dynamic_thresholds()` — ~15 learned parameters
- R/R minimum thresholds, ADX thresholds, SL tightening progress thresholds
- Position size limits, confluence count minimums, timeframe alignment reduction coefficients

##### `get_vector_context()` — Vector-similarity search results + contextual stats

##### Semantic Rules (stored to ChromaDB)
- **Best-practice rules:** ≥5 wins total, ≥3 same-pattern, ≥60% win rate → `rule_best_*`
- **Anti-pattern rules:** ≥3 losses total, ≥2 same-pattern, ≥60% loss rate → ⚠️ `rule_anti_*` with failure_reason + recommended_adjustment
- **Corrective rules:** ≥3 losses, ≥2 same-pattern, <60% loss rate → ⚡ `rule_corrective_*`
- **AI-mistake rules:** ≥2 mistakes classified → `rule_ai_mistake_*` with failed_assumption, mistake_type

---

#### Prompting Strategy

##### Reflection Cadence (timeframe-adaptive)

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

##### Context Injection Strategy

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

##### Embedding Strategy
- `build_rich_context_string()` — categorical labels for storage/retrieval
- `build_query_document()` — embedding query mirrors stored format but includes raw numeric values, reducing embedding asymmetry
- Hybrid retrieval: 70% similarity + 30% recency decay

##### Semantic Rule Influence Scoring

Active semantic rules are durable learned policy, not raw trade examples. They are preserved while active, but their prompt influence is soft-ranked by:

- Semantic similarity to the current market context
- Evidence quality (`wins`, `losses`, `source_trades`, `expectancy_pct`, `profit_factor`)
- Timeframe-aware freshness using the same half-life model as trade experiences
- Contradiction penalty from matched closed trades that disagree with the rule

For the default 4h timeframe, the rule freshness half-life is about 14 days. Older active rules are not deleted automatically; they receive lower freshness labels (`maturing`, `stale`, `legacy`) unless recently validated by matching outcomes.

##### Confidence Threshold — Adaptive Learning

The brain learns a `confidence_threshold` dynamically from historical trade data in `VectorMemoryAnalytics`:

| Condition | Threshold Set |
|-----------|--------------|
| HIGH confidence win rate > 70% | 65 (relaxed — HIGH confidence is reliable) |
| HIGH confidence win rate < 55% | 75 (tightened — HIGH confidence underperforming) |
| Default (insufficient data) | 70 |

This threshold is injected into LLM prompts to guide the model's self-assessed confidence level, but does **not** directly gate position sizing — position sizing uses confidence as a string label ("HIGH"/"MEDIUM"/"LOW") against configured fallback size percentages.

---

#### SL Tightening Policy

The `StopLossTighteningPolicy` enforces a **price-progress gate** before allowing the LLM to tighten an open position's stop loss. This prevents premature SL adjustments that would turn small pullbacks into exits.

##### Timeframe-Adaptive Base Thresholds

| Timeframe | Min Progress to TP | Meaning |
|-----------|-------------------|---------|
| Scalping (<1h) | 25% | Price must travel 25% toward TP before SL can tighten |
| Intraday (1h–4h) | 20% | |
| Swing (4h–1d) | 15% | |
| Position (>1d) | 10% | |

##### Blending with Brain-Learned Thresholds

The effective threshold is resolved by `_resolve_effective_threshold()`:
1. Start with the timeframe base threshold
2. If a brain-learned SL tightening threshold exists with enough samples, use the learned value clamped to the configured floor/ceiling
3. Expose the result via `get_dynamic_thresholds()` as `sl_tightening_pct`, `sl_tightening_source`, and the nested `sl_tightening` payload

##### Position Update Gating

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

#### Edge Cases & Guardrails

| Scenario | Handling |
|----------|----------|
| **No prior trade data** | `get_context()` builds context from vector memory (blocked-trade feedback, similar experiences, semantic rules); if all are empty, it returns an empty string and the brain section is skipped |
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

---

#### Executor Forward Pipeline

When the LLM outputs a BUY/SELL, the decision flows through two stages before reaching the exchange:

##### Stage 1: Strategy Decision (TradingStrategy)
`process_analysis()` runs the guard pipeline, risk manager, and R:R check. If the trade is blocked (e.g. R:R < 1.50), it stores a **blocked-trade friction** in ChromaDB (`system_constraints_rejections` collection) and returns `TradeDecision(action="HOLD")`.

##### Stage 2: Context Patching + Executor Handler

```python
### In CryptoTradingBot._execute_trading_check():

### 1. If strategy vetoed the BUY/SELL, rewrite the saved previous response
###    so the LLM sees "HOLD" next cycle (not "BUY" with no position):
if decision is not None and decision.action == "HOLD" and result.get("analysis"):
    self._patch_rejected_signal_in_response(result, decision)

### 2. Save analysis data (possibly patched) for next-cycle LLM context
self._save_analysis_data(result)

### 3. Delegate executor pipeline to ExecutorHandler
analysis = result.get("analysis")
if self.executor_handler is not None and analysis:
    await self.executor_handler.handle(analysis, decision, self.current_symbol)
```

##### ExecutorHandler (`src/trading/executor_handler.py`)

```python
class ExecutorHandler:
    # Constructor injection — no isinstance/getattr, known types only
    def __init__(self, persistence, config, logger): ...

    async def handle(self, analysis, strategy_decision, symbol):
        payload = self._build(analysis, strategy_decision, symbol)  # None on HOLD veto
        if payload is None:
            return
        self._persist(payload)     # atomic write to latest_decision.json
        await self._forward(payload)  # HTTP POST to llm_trader_executor
```

Wired once in `start.py` `build_dependencies()` — not in app.py constructors.

##### Blocked-Trade Feedback Loop (ChromaDB → LLM Prompt)

```
LLM outputs BUY with R:R 0.62
  → TradingStrategy rejects (R:R < 1.50)
  → store_blocked_trade(guard_type="rr_minimum", ...) → ChromaDB
  → Returns TradeDecision(action="HOLD")
  → _patch_rejected_signal_in_response()  → rewrites raw_response "BUY" → "HOLD"
  → _save_analysis_data()                 → persists patched response
  → ExecutorHandler._build() sees action="HOLD" → returns None (no executor forward)

Next analysis cycle:
  → BrainContextProvider.get_context()
    → vector_memory.get_blocked_trade_feedback(n=5, max_age_hours=168)
    → Injects "## CRITICAL FEEDBACK: System Rejections" into prompt
      "Your R/R: 0.62 | Required: 1.50 (gap: -0.88)"
  → LLM also sees its own previous response as "HOLD" (patched), not "BUY"
```

##### Response Parameters to `config.ini` (for `[executor_api]`)

```ini
[executor_api]
enabled = true
url = http://127.0.0.1:9199/decision
```

##### Vector Memory Maintenance

- `VectorMemoryService.prune_aged_documents()` may prune stale experiences and blocked-trade feedback beyond the relevance window.
- Active semantic rules (`active=True`) are preserved even when old; do not delete active learned rules by age alone.
- Active semantic rules are ranked with timeframe-aware freshness/evidence scoring before prompt injection; age lowers influence but does not physically delete a rule.
- Matched closed trades update semantic-rule `validation_hit_count`, `last_validated_at`, `contradiction_count`, and `last_contradicted_at` metadata.
- Timestamp parsing must be datetime-aware so malformed timestamps are skipped safely instead of corrupting prune decisions.

##### Trade Persistence Contract

- Trading code records decisions through `PersistenceManager.async_save_trade_decision()` and must not write trade-history files directly.
- Trade history is SQLite-only (`trade_history.db`). Do not reintroduce `trade_history.json` readers, fallback writes, or auto-migration paths.
- Entry-decision recovery for close-time brain learning uses SQLite timestamp-window lookup with optional symbol filtering.

---

#### Data Flow

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


## 12.7 Trading Guards

### 🛡️ Order Governance Pipeline

> **Module path:** `src/trading/guards/`
> **Type:** Pre-execution guard chain for trading signal validation

---

#### Agent Persona & Role

The Governance Pipeline is the **last line of defense before any order signal reaches the simulated market.** It enforces declarative, configurable rules that every trading signal must pass before execution: symbol whitelist checks, cooldown windows, and position size limits.

The pipeline follows the **Chain of Responsibility pattern** — guards run sequentially and fail fast. If any guard fails, the order is blocked and an audit rejection is recorded.

---

#### Pipeline Architecture

```
TradingStrategy → GuardPipeline.evaluate(intent, capital, config)
    ├── ConfiguredSymbolGuard     (whitelist check)
    ├── MaxPositionSizeGuard       (explicit requested size cap)
    └── CooldownWindowGuard        (time since last SQLite-recorded BUY/SELL)
         ↓
    Result: pass → TradingStrategy proceeds
            fail → audit rejection recorded
```

##### GuardPipeline (`pipeline.py`)
- Runs guards in order and stops at the first failure
- Returns `list[GuardResult]` for the guards that were evaluated
- All guards must pass for order execution
- First failure short-circuits remaining guards (fail-fast)

---

#### Guard: ConfiguredSymbolGuard (`configured_symbol.py`)

**Purpose:** Ensures the trading signal targets a configured symbol.

**Logic:**
- Signal must reference `config.CRYPTO_PAIR`
- Prevents phantom pairs or misconfigured symbols

**Edge Cases:**
- Unknown symbol → blocked with reason "does not match configured trading pair"

---

#### Guard: CooldownWindowGuard (`cooldown_window.py`)

**Purpose:** Prevents rapid-fire trading by enforcing a minimum time window between consecutive executed BUY/SELL trades.

**Logic:**
- Reads the most recent executed BUY/SELL timestamp through injected `PersistenceManager.get_last_execution_timestamp()`
- `PersistenceManager` queries SQLite `trade_history.db`; the guard must not read `trade_history.json` or any file path directly
- Cooldown is derived from `config.TIMEFRAME`: <1h → 4× timeframe, 1h–3h → 3×, 4h–23h → 2×, daily+ → 1×
- If elapsed < cooldown → block

**Edge Cases:**
- No prior trade → immediately passes
- Cooldown applies uniformly after the most recent BUY/SELL; direction is not treated specially
- Cooldown = 0 → disabled effectively
- Missing persistence injection → fail closed
- Persistence/SQLite read failure → fail closed so execution cannot bypass cooldown due to storage errors

---

#### Guard: MaxPositionSizeGuard (`max_position_size.py`)

**Purpose:** Rejects an explicitly requested position size that exceeds the configured cap.

**Logic:**
- Reads `MAX_POSITION_SIZE` from config and validates it is positive and finite
- If AI provides a positive finite `position_size`, it must be ≤ `MAX_POSITION_SIZE` (default 10%)
- Missing, non-finite, or non-positive requested sizes pass through so `RiskManager` can apply fallback sizing

**Edge Cases:**
- Missing `position_size` → passes with reason that RiskManager fallback sizing will apply
- Non-finite requested size → passes with reason that RiskManager fallback sizing will apply
- Invalid `MAX_POSITION_SIZE` config → fails closed

---

#### Friction Recording

Risk and strategy-level blocked trade feedback is recorded through vector memory:

```
VectorMemoryService.store_blocked_trade(...)
```

This is currently used for RiskManager frictions, R:R minimum blocks, and premature SL-tightening blocks. Guard-pipeline failures are audit-recorded before risk calculation and do not call vector memory directly.

Stored blocked-trade feedback feeds the Brain Agent's `get_context()` which shows the LLM:
- Recent blocked trades (last 5, max 168h old)
- Guard type + reason for each block
- Enables the LLM to understand why previous similar signals were rejected

---

#### Configuration

All guard parameters are set in `config/config.ini`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_position_size` | 0.10 (10%) | Maximum explicit/requested position size; RiskManager also clamps fallback sizing to this cap |
| `timeframe` | 4h | Cooldown duration source |
| `crypto_pair` | BTC/USDC | Single configured trading pair |

Guards are **declarative** — they can be reviewed and modified without reading any code paths.
