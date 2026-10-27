# LLM Trader — Master Architecture Blueprint

> **Repository:** [https://github.com/qrak/LLM_trader.git](https://github.com/qrak/LLM_trader.git)
> **Python:** 3.13, `.venv/`, `python start.py`
> **Execution:** decisions are forwarded to the external executor service (`[executor_api]`, `enabled = true`); with `enabled = false` the bot is pure paper/sim
> **Live Dashboard:** [https://semanticsignal.qrak.org](https://semanticsignal.qrak.org)

---

## 0. Instruction Authority

Root `AGENTS.md` is the single instruction source of truth in this repository across all IDEs, agents, and harnesses.

- Root `AGENTS.md` is canonical for system-wide rules, architecture, coding standards, testing, terminal behavior, and governance.
- IDE-specific instruction files are non-authoritative and should not contain policy that is missing from `AGENTS.md`.
- `.github/workflows/*` defines CI execution behavior, not instruction authority.

---

## 1. System Overview

Asyncio-first trading bot: market data, news (RAG) and chart images become structured BUY / SELL / HOLD decisions from an LLM. A central trading loop coordinates data ingestion, technical analysis (Numba indicator kernels), RAG retrieval, risk validation, an outcome-aware brain (vector memory + reflection), and an order-governance chain. Filled orders are executed by a separate service; the bot keeps the decision loop and books what the venue actually did.

```
DataFetcher ─▶ AnalysisEngine ─▶ ProviderOrchestrator ─▶ UnifiedParser
   (CCXT)       TechCalc+Pattern+Chart+RAG+Brain        (fallback chain)
                                                              │
   BrainContextProvider ◀── vector memory ◀── closed trade     ▼
        (confidence + rules into next prompt)          GuardPipeline ─▶ RiskManager
                                                              │            (SL/TP, size)
                                                              ▼                │
                                                     TradingStrategy ◀─────────┘
                                                     (R:R gate, OrderLifecycle, ExitMonitor)
                                                              │
                                                        executor_api ─▶ venue, verdict journal ─▶ booking
```

---

## 2. Agent Inventory

| # | Agent Name | Primary Responsibility | Core Model | Core Implementation |
|---|------------|----------------------|------------|---------------------|
| 1 | **🧠 Brain Agent** (TradingBrainService) | Outcome-aware decision enricher, semantic rule learning via reflection loops, confidence calibration | Deterministic/vector memory; context is injected into provider-routed LLM prompts | [`src/trading/brain.py`](./src/trading/brain.py) |
| 2 | **🔬 Analysis Engine Agent** | Market data collection, 50+ technical indicators, pattern recognition, chart generation, AI signal synthesis | Provider-routed (`provider` in `config/config.ini`); the chart-vision step needs a multimodal model | [`src/analyzer/analysis_engine.py`](./src/analyzer/analysis_engine.py) |
| 3 | **📰 RAG Engine Agent** | News aggregation (RSS + optional Crawl4AI enrichment), fundamentals (DeFiLlama), relevance scoring, context retrieval | Deterministic (no LLM) | [`src/rag/rag_engine.py`](./src/rag/rag_engine.py) |
| 4 | **⚙️ Risk Manager Agent** | Dynamic SL/TP scaling, position sizing, signal validation, circuit breakers | Deterministic | [`src/managers/risk_manager.py`](./src/managers/risk_manager.py) |
| 5 | **☁️ Provider Orchestrator** | AI provider lifecycle, multi-provider fallback chain, parameter negotiation | — | [`src/managers/provider_orchestrator.py`](./src/managers/provider_orchestrator.py) |
| 6 | **🛡️ Governance Pipeline** | Pre-execution guard chain: symbol whitelist, max position size | Deterministic | [`src/trading/guards/pipeline.py`](./src/trading/guards/pipeline.py) |
| 7 | **📊 Dashboard Agent** | Real-time FastAPI + WebSocket monitoring, performance analytics, brain state inspection | — | [`src/dashboard/server.py`](./src/dashboard/server.py) |

---

## 3. Application Lifecycle

### 3.1 Startup (CompositionRoot)

`start.py` → `SingleInstanceLock` → Event loop with `GracefulShutdownManager` → 9-stage dependency provisioning (`src/composition/provisioners.py`):

| Stage | Provisioner | Dependencies Created |
|-------|------------|---------------------|
| 1 | `_provision_infrastructure` | ExchangeManager, aiohttp session, KeyboardHandler |
| 2 | `_provision_utilities` | FormatUtils, UnifiedParser, TokenCounter, TimeframeValidator, CategoryCollisionResolver |
| 3 | `_provision_platforms` | CCXTMarketAPI, CoinGecko, Alternative.me, DeFiLlama, RSS news client + Crawl4AI enricher |
| 4 | `_provision_rag_layer` | RagEngine, NewsManager, LocalTaxonomyProvider, TickerManager |
| 5 | `_provision_model_layer` | AI provider clients, ProviderOrchestrator, ModelManager |
| 6 | `_provision_analyzer_layer` | AnalysisEngine, MarketDataCollector, TechnicalCalculator, PatternAnalyzer |
| 7 | `_provision_trading_layer` | TradingStrategy, ExitMonitor, VectorMemoryService, TradingStatisticsService, TradingBrainService |
| 8 | `_provision_notifiers` | Discord notifier with file handler, or console fallback notifier |
| 9 | `_provision_dashboard_layer` | DashboardServer, DashboardState, force_analysis_event |

**Architectural invariant:** All services are instantiated in the composition layer and injected via constructor parameters. **Never** construct service dependencies inside other service classes, and **never** use in-function lazy imports (`Pylint C0415`) to resolve circular dependency cycles — refactor constructor parameter injection at the CompositionRoot (`start.py`) instead.

### 3.2 Main Loop

```
AnalysisEngine.analyze_market()
  ├── MarketDataCollector → DataFetcher (OHLCV + order book + trade flow)
  ├── TechnicalCalculator (50+ indicators) + LongTerm data + Weekly macro
  ├── PatternAnalyzer → IndicatorPatternEngine (deterministic indicator-pattern kernels)
  ├── ChartGenerator (1920×1080 PNG) → LLM visual chart-pattern analysis (analysis_result_processor.py)
  ├── RAG context retrieval
  ├── Brain context injection (confidence + rules similar to current conditions)
  ├── AI provider call → TradingAnalysisResponseModel (prompt step 5.5: the model must
  │      name a specific invalidation trigger or HOLD)
  └── Structured dict returned to TradingStrategy
       ↓
TradingStrategy.process_analysis()
    ├── PositionExtractor + UnifiedParser → extract and validate signal
    ├── GuardPipeline (symbol → max size)
    ├── RiskManager → RiskAssessment (SL/TP scaling, computes R:R)
    ├── R:R gate: max(config `min_rr_entry`, brain floor) — both 0.0 by default,
    │      so the gate is inert unless a floor is configured or learned
    ├── OrderLifecycle → INTENT → READY_FOR_REVIEW → EXECUTED (or REJECTED);
    │      approval is an audit event, not a lifecycle state
    ├── executor_api forward (order_id keyed) when `[executor_api] enabled = true`
    ├── PersistenceManager → SQLite-only trade_history.db append (no JSON fallback)
    ├── RiskManager friction drain → store_blocked_trade feedback for brain learning
    └── ExitMonitor (soft at candle close; hard at a configurable interval per SL/TP type)
       └── PositionStatusMonitor → background asyncio loop with dynamic rescheduling
       ↓
BrainAgent.update_from_closed_trade()
  ├── BrainExperienceRecorder → store vector memory
  ├── trade_count++ → schedule reflection if interval reached
  └── ReflectionEngine → sequential: best-practice → anti-pattern → AI-mistake rules
```

Per cycle the bot snapshots `CostStorage` before and after the LLM work (chart, text,
contract retry and any awaited post-mortem) and appends the delta to
`data/trading/metrics/cycles.jsonl`. The last 20 measured costs feed the EV framework,
whose 1.5× threshold covers the round-trip fee plus inference cost; with no measured
cost the EV prompt stays fee-only.

### 3.3 Shutdown

`GracefulShutdownManager`: SIGINT/SIGTERM → drain active analysis → persist state → close providers → flush logs; the keyboard handler performs a manual stop with state preservation.

---

## 4. Core Data Flow

### 4.1 Learning Loop

```
Closed trade ─▶ BrainExperienceRecorder ─▶ ChromaDB (vector memory)
                     ├── update matched semantic-rule validation / contradiction counters
                     ├── trade_count % interval == 0 → ReflectionEngine
                     │      (best-practice → anti-pattern → AI-mistake rules)
                     └── next cycle: BrainContextProvider queries ChromaDB for
                            similar past trades (top-5), rules scored by similarity +
                            evidence + timeframe freshness, confidence stats by level,
                            blocked-trade feedback → injected into the LLM prompt
```

Semantic-rule policy:

- Active semantic rules are durable learned policy and are not deleted by age-only pruning; they are deactivated by evidence. Inactive old rules may be pruned as storage maintenance.
- Rule influence is soft-ranked by semantic similarity, evidence quality, timeframe-aware freshness, contradiction count, and **surprise ratio**.
- Closed trades that match active rules update validation or contradiction metadata used by that ranking.
- **Surprise ratio** (`|realized P&L - expected P&L| / expected P&L`) is computed at trade close. A high ratio (>1.5) means the outcome came from factors outside the entry thesis — the trade won despite flawed reasoning (or lost despite good reasoning). Rules derived from such trades carry a `⚠️ high surprise` annotation so the LLM can discount lucky outcomes.

### 4.2 Trade Persistence

- Trade history is SQLite-only at `data/trading/trade_history.db` via `SQLiteTradeHistory` and `PersistenceManager`.
- Runtime code must not read, write, or auto-migrate `trade_history.json`; historical `.json.migrated` files are backups, not runtime inputs.
- `PersistenceManager.save_trade_decision()` fails loudly if SQLite persistence fails — do not add JSON fallback paths.
- Dashboard, brain entry-decision lookup, and query scripts consume trade history through injected persistence or SQLite APIs.
- **Zero Backward Compatibility & Startup Clutter Policy**:
  - Runtime code in `src/` and `start.py` stays canonical and clutter-free.
  - Never introduce inline schema migration `ALTER TABLE` statements, legacy unit conversion methods, rule migration hooks (`refresh_semantic_rules_if_stale`), or `try/except` fallback paths into runtime service initialization.
  - Never add `sys.path.insert(0, ...)` manipulation hacks to `start.py`, and never add startup auto-rehydration loops there. Database or vector-storage rehydrations run as standalone CLI scripts in `scripts/`, executed once and deleted.
  - **Classes Only in `src/utils/`**: every utility concern is encapsulated as a class (`JournalRotator`, `TokenCounter`); standalone utility functions are forbidden.

---

## 5. Configuration

Active config at `config/config.ini`. Key settings:

| Setting | Value |
|---------|-------|
| **Pair** | BTC/USDC |
| **Timeframe** | 4h |
| **Candles** | 999 (125 for the AI chart) |
| **Capital** | $10,000 simulated (`demo_quote_capital`) |
| **Fee** | 0.075% (`transaction_fee_percent`) |
| **Inference cost** | `CostStorage` delta per cycle; rolling 20-analysis average in the EV prompt; no config key |
| **Max Position** | 10% of portfolio (`max_position_size`) |
| **Fallback sizes** | 1% / 2% / 3% (LOW/MEDIUM/HIGH confidence) |
| **R:R entry floor** | `min_rr_entry = 0.0` (gate inert until set) |
| **News update** | Every 4 hours, 5 articles max |
| **Provider** | `provider = deepseek`, `deepseek_model = deepseek-flash`; OpenRouter base `google/gemini-3-flash-preview`, fallback `deepseek/deepseek-v4.1-flash` |
| **Executor** | `[executor_api] enabled = true`, `url = http://127.0.0.1:9199/decision` |
| **Market data source** | `[exchanges] sandbox = false` → real exchange book; `true` → testnet book |
| **Timezone** | `[general] timezone = Europe/Warsaw` — human-readable timestamps only (logs, notifications, check headers, chart labels); trading stays on UTC candles |
| **Real-market stop** | `[risk_management] reference_exit_check_seconds = 10` (0 = off; pairs with the executor's `PROTECTION_BACKSTOP_MULTIPLIER`) |
| **Dashboard** | 0.0.0.0:8000 |

---

## 6. Project Structure Reference

```
LLM_trader/
├── start.py                     # Entry point + CompositionRoot + SingleInstanceLock
├── AGENTS.md                    # THIS FILE — master rules; package detail lives in src/*/AGENTS.md
├── README.md / CHANGELOG.md     # Overview & setup / version history (newest first)
├── docs/                        # changelog-archive-{2025,2026}.md, adr/
├── requirements.txt / -dev.txt / constraints.txt
├── keys.env / keys.env.example  # Secrets (gitignored / template)
├── config/
│   ├── config.ini               # Active configuration
│   ├── model_pricing.json       # Per-model cost data (BASE rates)
│   ├── peak_rates.example.json  # Optional peak/off-peak billing windows
│   └── rag_priorities.json      # RAG priorities (important_categories + generic_priorities)
├── src/
│   ├── app.py                   # Main application wiring (ApplicationServices)
│   ├── composition/             # provisioners.py, startup_support.py — DI wiring
│   ├── trading/                 # 🧠 Brain + Strategy + Monitors
│   │   ├── brain.py             # TradingBrainService (facade) + brain_*.py collaborators
│   │   ├── trading_strategy.py  # Strategy orchestration
│   │   ├── exit_monitor.py      # Hard/soft exit checks
│   │   ├── vector_memory.py     # ChromaDB interface + vector_memory_*.py collaborators
│   │   ├── executor_reconciliation.py  # tri-state position truth + intent journal
│   │   ├── regime_risk_profile.py      # Regime-aware risk profile (Risk Manager)
│   │   ├── statistics.py        # P&L tracking
│   │   └── guards/              # 🛡️ Governance Pipeline (2 guards + pipeline.py)
│   ├── analyzer/                # 🔬 Analysis Engine (analysis_engine, technical_calculator,
│   │                            #    pattern_engine/, prompts/, formatters/, data_fetcher)
│   ├── rag/                     # 📰 RAG Engine (rag_engine, news_manager, news_ingestion/)
│   ├── managers/                # ⚙️ Risk Manager, persistence_manager, sqlite_trade_history,
│   │                            #    provider_orchestrator, model_manager
│   ├── dashboard/               # 📊 server.py + routers/ (6 REST routers + ws_router)
│   ├── indicators/              # 79 Numba-compiled indicator functions
│   ├── platforms/               # AI provider clients + exchange APIs
│   ├── parsing/                 # UnifiedParser
│   ├── logger/                  # Structured logging
│   ├── notifiers/               # Discord, console, file handler
│   └── utils/                   # Profiler, token counter, decorators, time_utils (classes only)
├── tests/                       # 52 test_*.py in domain_{trading,brain,rag_indicators,infra}
│                                #   + test_admin_playwright.py (marker: integration) + conftest.py
├── data/                        # Runtime state (not committed)
├── logs/Bot/YYYY_MM_DD/         # Bot.log (all levels) + errors.log (error level)
└── scripts/                     # Cross-platform startup scripts (Windows .ps1 are gitignored)
```

---

## 7. Active Platform Integrations

- **Exchanges:** Binance, KuCoin, MEXC, Hyperliquid via CCXT (`gateio` is not a valid CCXT id — the ccxt name is `gate`, so that entry never loads)
- **Market Data:** CoinGecko, Alternative.me, DeFiLlama, CCXT exchange market data
- **AI Providers:** DeepSeek (active `provider`), OpenRouter (base + fallback models), Google AI and LM Studio as configurable alternatives
- **Execution:** external `llm_trader_executor` service over HTTP (`127.0.0.1:9199`), which places real exchange orders via CCXT and writes the verdict/exit journals the bot reads
- **News Sources:** CoinDesk, CoinTelegraph, Decrypt, CryptoSlate RSS feeds, with optional Crawl4AI page enrichment (`src/rag/news_ingestion/crawl4ai_enricher.py`)

---

## 8. Operational Rules

### Terminal Guardrails (All Agents)

- Send one terminal command per tool call.
- Never include control-key text in commands (for example `^U`, `^C`, `^[`).
- On Windows/PowerShell in VS Code, prompt-edit control text such as `^U` is sent literally and becomes part of the command name. Do not assume Linux/readline behavior.
- Never send terminal follow-up probes or marker echoes (for example `Write-Output $LASTEXITCODE`, `echo DONE`, or a "flush" command) to recover hidden or truncated validation output.
- If validation output is incomplete, either trust the user's visible terminal output when provided or rerun the exact validation command once with a generous timeout.
- Never chain validation commands with `;`, `&&`, variable assignment, redirect/capture, and readback in one line.
- For pytest validation, trust only raw output from a direct pytest command.
- If terminal output is empty or malformed, do not claim success, and never infer pass/fail from a trailing exit-code marker.

### Operator Commands

Keep platform-specific setup, startup, test, lint, and type-check commands in `README.md`. This file documents agent architecture and execution policy only.

### Safety

- **Execution is delegated:** with `[executor_api] enabled = true` the bot forwards intents and the executor places them on the venue; disabling it makes the bot pure paper/simulation. Never assume "paper only".
- **Hard SL/TP exits** run at 15-minute intervals; soft candle-close exits are supported by `ExitMonitor`.
- **Max position:** 10% of portfolio; **simulated capital:** $10,000 with a 0.075% fee model.
- **Fail-closed behavior** if governance/risk validation cannot decide safely.

---

## 9. Documentation Governance

### AGENTS-Only Policy Checklist

Use this checklist for every documentation or tooling-policy PR:

1. All behavioral policy changes are documented in root `AGENTS.md`.
2. Do not introduce IDE-specific policy files (Copilot, Claude, Windsurf instruction docs) as authoritative guidance.
3. `.github/workflows/*` may define CI execution logic only; workflow comments must not replace policy documentation in `AGENTS.md`.
4. If a command, validation rule, or safety guard changes, update the related AGENTS section in the same PR.
5. Before merge, search the repository for stale references to removed tool-specific instruction files.

### Drift Prevention Rule

Any new tool-specific instruction file must be a non-authoritative pointer to `AGENTS.md`; if it contains independent policy, reject it in review.

### Documentation map

`README.md` is the landing page; the long-form pages live in `docs/` so the README stays
scannable. A fact belongs in exactly one of them and is linked from the others, never copied.

| File | Scope |
|---|---|
| `README.md` | What the project is, how it performs, how it compares, quick start |
| `AGENTS.md` + package files | Policy, architecture, conventions, verification gates |
| `docs/FEATURES.md` | Every feature in detail, with the implementing file |
| `docs/SETUP.md` | Requirements, platform scripts, secrets, where the website lives now |
| `docs/CONFIGURATION.md` | Every config key, prompt budget, billing windows, executor handoff |
| `docs/RESULTS.md` | Paper-trading ledger, benchmark, caveats, reproduction command |
| `CHANGELOG.md` | Dated changes with the run that verified them |

`python scripts/check_agents_md.py` fails when a path quoted in an AGENTS file no longer exists —
run it after any rename.

---

## 10. Code Conventions (all areas)

**No comments in code — docstrings only.** Reasoning belongs in the commit message, the change description, or this file; a `#` comment in `src/`, `tests/` or `scripts/` is deleted on sight. The only tolerated comments are machine directives (`# noqa`, `# type: ignore`). Code must explain itself: precise names, small functions, real types.

### Architecture
- **Dependency injection only.** Every class receives its dependencies (`logger`, `config`, collaborators) through `__init__`; the single composition root is `start.py`. Never construct a collaborator inside a method.
- **`self.logger`, never `self._logger`** — `@retry_async` reads `instance.logger`.
- **No standalone functions in `src/app.py`**, and no loose helper modules: every concern is a class wired once from `start.py`. Utility code lives in `src/utils/` as classes.
- **Top-level imports only.** An in-function import added to dodge a circular import means the wiring is wrong — fix it in the composition root.
- **No `sys.path` manipulation and no startup auto-rehydration loops** in `start.py`; one-off data conversions run as a script that is executed once and deleted.

### Types and data
- `TradeDecision` and `Position` are `@dataclass(slots=True)` with fixed fields: access attributes directly. Never `hasattr` / `getattr` / `isinstance` on them.
- The strategy's public surface is a contract, not a probe target: call `position_snapshot_token()`, `protection_snapshot()`, `protection_loosening_reason()`, `reconcile_local_position()`, `resolve_position_intents_after_forward()` and the one-shot `take_*()` readers directly. Wiring a double means implementing the method, not making the caller duck-type it.
- Every timestamp printed for a human goes through `src/utils/time_utils.py`: `utc_stamp()` for UTC-only text, `utc_and_local()` for a paired label, `local_stamp()`/`day_stamp()` for log lines and folders. Trading arithmetic, persistence and TTLs stay on UTC epoch or aware UTC datetimes — never on the wall clock.
- Raw payloads from LLM output or API responses are `dict | None`: read them with `.get()`, and keep the guards that protect genuinely unknown data.
- `TradeDecision` carries no `order_type` / `reduce_only`; those come from `ENTRY_ORDER_TYPE` and `analysis.get("reduce_only", False)`.
- Validate numbers at the boundary: NaN/Inf are rejected where they enter (`_parse_finite_number`, `_parse_finite_float`, `math.isfinite` for timestamps), and `serialize_for_json` maps NaN/Inf to `None`.
- `SerializableMixin.from_dict(to_dict(obj))` must round-trip nested tuples and datetimes.

### Errors and retries
- Narrow exception handling. A bare `except:` never exists; a broad `except Exception` is a deliberate fail-open or fail-closed decision that the docstring explains. Never fix a bug by catching and ignoring.
- Every queue eviction is logged with context; every `asyncio.gather` handles per-task exceptions explicitly instead of swallowing them.
- No hand-rolled retry loops: `@retry_async` (`src/utils/decorators.py`) for network and exchange calls, `@retry_api_call` for AI provider calls.
- `time.sleep` has no place in async code; the only acceptable sleeper is `_interruptible_sleep`.

### Performance
- Optimize only against measured evidence — never prematurely, never at the cost of readability: a clear 10% win beats an opaque 15% win.
- Keep an optimization self-contained (well under 50 lines), behavior-preserving, and safe under the asyncio event loop and thread concurrency.
- Ask before adding a dependency, changing queue sizes or thread boundaries, touching the core trading loop (`src/app.py`, `start.py`), or changing ChromaDB collection schemas and embedding logic.

### Dead code
- **Zero compatibility leftovers.** Delete dead endpoints, cache prefixes, CSS, DOM ids, aliases and retired guards immediately; git history is the archive.
- The decision wire format (signal names, payload fields) is a contract: never change it silently.

---

## 11. Security Conventions

- Secrets come from OS/service environment variables (recommended) or optional `keys.env` / `.env` (gitignored); process variables take precedence: never in `.py`, never in a log. Follow the redaction pattern (`_redact_private_key()`); never log credentials, private keys or full secret payloads.
- Any `eval` / `exec` / `subprocess(shell=True)` / `os.system` / `pickle.loads` / `yaml.load` hit in `src/` is CRITICAL until proven otherwise.
- Validate input with FastAPI/Pydantic models (`Field`, `Path`, `Query`) instead of hand-parsing dicts; errors must never leak sensitive data.
- Use the established patterns: PBKDF2-SHA256 password hashing, HMAC-SHA256 signed cookies, `hmac.compare_digest` for comparisons, constant-time anti-enumeration on login, `/admin` and `/api/admin` gated to LAN behind admin auth, DOMPurify for every dynamic HTML string.
- A security change is done only when you have exercised the attack vector and shown it fails; report severity, the vulnerable location, the concrete impact, the fix and the proof.
- Ask before changing authentication or authorization logic, session-cookie formats, admin middleware, or rate limiting on `/api/admin/login`.

---

## 12. Change Workflow & Verification Gates

1. **Reproduce before fixing.** A bug you cannot reproduce is a bug you cannot verify.
2. **One change, one purpose.** Never refactor and fix in the same change; keep the diff minimal and behavior identical beyond the stated fix.
3. **Fast gate first:** `py_compile` the touched files, `ruff check src/`, `python -c "import start"`, then the targeted `pytest tests/<matching>.py -x -q`.
4. **Full gate before claiming done:** `pytest tests/ -q` with 0 failures and 0 skips, and `pyright src start.py` with no new errors.
5. **Test isolation:** a test double must never default to a live state file, a live journal path (`data/trading/*.jsonl`) or a live brain database; point it at a temp directory.
6. **A bug fix ships with a regression test**, added to the existing test file for that module rather than a new one.
7. **Read the diff after every edit:** a patch that matched the wrong anchor relocates or truncates code silently.
8. **Never commit or push without explicit user consent.** Run the suite first; commit only with 0 failures.
9. **Do not touch CI configuration** (`.github/workflows/`, test/lint settings) unless asked.
10. Known pre-existing noise, never caused by your change: 9 `BLE001` hits in `tests/test_admin_playwright.py` (the only browser suite, excluded from CI by the `integration` marker) — `ruff check src start.py scripts` is the gate that must be clean.
11. **Documentation drift:** after moving or renaming a file, run `python scripts/check_agents_md.py`; after changing the ledger shape, run `python scripts/results_report.py` and update `docs/RESULTS.md` in the same change.

---

## 13. Package Maps (read on demand)

Root `AGENTS.md` is canonical for system-wide rules. Package detail lives next to the code so a task reads only the map it needs:

| Area | Detail file | Covers |
|---|---|---|
| Cross-cutting `src/` contracts (parsing, utils, notifiers, platforms, logger) | [`src/AGENTS.md`](./src/AGENTS.md) | Module regression contracts for packages without their own file |
| Trading brain, strategy, monitors, position management | [`src/trading/AGENTS.md`](./src/trading/AGENTS.md) | TradingBrainService, vector memory, strategy flow, exits |
| Order governance pipeline (guards) | [`src/trading/guards/AGENTS.md`](./src/trading/guards/AGENTS.md) | Guard chain, fail-closed policy |
| Analysis engine | [`src/analyzer/AGENTS.md`](./src/analyzer/AGENTS.md) | Data collection, indicators, prompts, chart vision |
| Indicator library | [`src/indicators/AGENTS.md`](./src/indicators/AGENTS.md) | Numba JIT functions |
| Managers | [`src/managers/AGENTS.md`](./src/managers/AGENTS.md) | Risk, persistence, SQLite history, providers |
| RAG engine | [`src/rag/AGENTS.md`](./src/rag/AGENTS.md) | News ingestion, retrieval, priorities |
| Dashboard | [`src/dashboard/AGENTS.md`](./src/dashboard/AGENTS.md) | FastAPI app, routers, static UI |

When a change touches one of these areas, read that file before editing; when the rule you need is missing there, add it to the area file rather than growing this one.
