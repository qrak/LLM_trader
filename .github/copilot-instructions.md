# LLM Trader — Master Agent Instructions

> Consolidated from all instruction files, skills, playbooks, and documentation in the LLM Trader repository. This single file replaces fragmented `.github/instructions/`, `.github/skills/`, `docs/`, `.clinerules`, and template files as the authoritative guide for any agent or developer working on this codebase.

---

## 1. System Overview & Architecture

### 1.1 What Is LLM Trader

**SEMANTIC SIGNAL LLM (LLM Trader)** is a BETA / Research Edition autonomous, asyncio-first trading bot. It converts market data, news (via RAG), and chart context into structured BUY / SELL / HOLD decisions via large language models. The bot runs in **demo-account and paper-trading mode** — real exchange order execution is not yet implemented.

- **Repository:** `/home/qrak/LLM_trader` (private) + GitHub public mirror
- **Python:** 3.13+, Linux (bash), virtual environment at `.venv/`
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
| | `formatters/` | Technical, market, overview, period, long-term formatters |
| | `prompts/` | Prompt builder, context builder, template manager |
| | `pattern_engine/` | Chart, swing, trendline patterns |
| | `indicator_patterns/` | RSI, MACD, MA crossover, volume, stochastic patterns |
| **Trading** | `TradingStrategy`, `ExitMonitor`, `PositionStatusMonitor` | Strategy execution, exit monitoring, position tracking |
| | `VectorMemoryService` (+ context, rules, analytics) | ChromaDB-based vector memory (ChromaDB client injected via DI from `start.py`) |
| | `Statistics`, `StatisticsCalculator` | Trade statistics, performance tracking |
| | `PositionExtractor`, `StopLossTighteningPolicy` | Position parsing, SL policy |
| **RAG** | `RagEngine`, `NewsManager`, `NewsRepository` | Retrieval-Augmented Generation |
| | `news_ingestion/` | RSS provider, Crawl4AI enricher, schema mapper |
| | `market_components/` | Market data cache, fetcher, processor, overview builder |
| | `ScoringPolicy`, `LocalTaxonomy`, `TickerManager` | Scoring, categorization, ticker management |
| **Indicators** | `indicators/` (20+ modules) | Full technical indicator library (momentum, overlap, price, trend, volatility, volume, statistical, support/resistance, sentiment) |
| **Platforms** | `CCXtMarketApi`, `CoinGecko`, `AlternativeMe`, `DeFillama` | External data providers |
| | `ai_providers/` | Google AI, OpenRouter, LM Studio, BlockRun, Mock — with fallback chain |
| **Dashboard** | `DashboardServer` (FastAPI) + 5 routers | Real-time web dashboard with WebSocket streams |
| | `static/` | HTML/CSS/JS frontend (7 CSS files, 9 JS modules) |
| **Managers** | `ModelManager`, `PersistenceManager`, `RiskManager`, `ProviderOrchestrator` | Model lifecycle, state persistence, risk management, provider orchestration |
| **Config** | `config/loader.py`, `config/protocol.py`, `contracts/` | Configuration loading, model/risk contracts |
| **Factories** | 4 factory modules | Data fetcher, position, provider, technical indicators factories |
| **Utils** | `profiler.py`, `token_counter.py`, `graceful_shutdown_manager.py`, `keyboard_handler.py`, `decorators.py`, `timeframe_validator.py`, `indicator_classifier.py`, `data_utils.py`, `format_utils.py` | Cross-cutting utilities |

### 1.5 Active Platform Integrations

- **Exchanges:** Binance, KuCoin, Gate.io, MEXC, Hyperliquid (via CCXT)
- **Market Data:** CoinGecko, Alternative.me, DeFiLlama
- **AI Providers:** Google AI (primary — Gemini 3 Flash Preview), OpenRouter (fallback — DeepSeek), BlockRun.AI, LM Studio (local), Mock
- **News Sources:** CoinDesk, CoinTelegraph, Decrypt, CryptoSlate (RSS + Crawl4AI enrichment)

### 1.6 Configuration

Active config at `config/config.ini` (202 lines). Key settings:

- **Pair:** BTC/USDC, **Timeframe:** 4h, **Candles:** 999 (125 for AI chart)
- **Capital:** $10,000 simulated, **Fee:** 0.075%
- **Max Position:** 10%, **Fallback sizes:** 1% / 2% / 3%
- **News update:** every 4 hours, 5 articles max
- **Model:** temperature 0.7 (Google: 1.0), top_p 0.9, max_tokens 32768
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

**Refactoring priority** (per current plan):
1. **DONE** — `TradingBrainService` (1276 lines -> 5 collaborator modules)
2. **Next** — `Config` (596 lines, 60+ property getters -> `ConfigSection` helper, ~250 lines)
3. **Then** — `AnalysisEngine` (798 lines -> extract `OrderBookAnalyzer` + `AnalysisDashboardState`, ~450 lines)

**Refactoring removal order for LLM Trader:**
1. `getattr`/`hasattr`/`setattr` removal first
2. Decorative headers + dead code
3. Typing modernization (protocol -> contract -> config -> app -> leaf modules)
4. DI violations

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
- For tool-driven terminal calls where cwd may drift, use path-anchored invocations:
  ```bash
  git -C /home/qrak/LLM_trader status
  /home/qrak/LLM_trader/.venv/bin/python -m pytest tests/
  ```

### 3.2 Testing

```bash
source .venv/bin/activate
python -m pytest tests/                         # Full suite
python -m pytest tests/test_vector_memory.py -q  # Focused file
python -m pytest tests/test_vector_memory.py -k fallback -q  # Specific test
```

- **51 test files** in `tests/` (`test_*.py`).
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
- Treat `data_template/` as the safe reference structure.
- `data/trading/` contains: API costs, brain vector DBs (ChromaDB), positions, statistics, trade history.
- `data/news_cache/` contains: recent news JSON.
- `data/news_fetch_preview/` contains: per-source raw and normalized JSON artifacts.
- `data/market_data/` contains: CoinGecko global JSON.

### 3.4 News Ingestion Pipeline

**Debug scripts:**
```bash
# Fetch and normalize news preview
source .venv/bin/activate
python scripts/fetch_free_news_preview.py --enrich-from-article-pages > temp_news_preview.txt

# With cryptocurrency.cv source
python scripts/fetch_free_news_preview.py --enrich-from-article-pages --include-cryptocurrency-cv > temp_news_preview_extended.txt

# Compare cache vs fresh body quality
python scripts/compare_news_body_quality.py > temp_news_body_quality.txt
```

**Artifacts:**
- `data/news_fetch_preview/news_raw_*.json` — per-source raw payload
- `data/news_fetch_preview/news_normalized_*.json` — deduped normalized items
- `data/news_fetch_preview/body_quality_*.json` — cache-vs-fresh quality comparison
- `data/news_cache/recent_news.json` — cached recent news

**Debug workflow:**
1. Identify failure class: endpoint access, normalized quality, body text completeness, cache mismatch.
2. Run relevant script, capture output to file.
3. Inspect generated JSON artifacts.
4. Only after artifact confirms the issue should you change code.

### 3.5 Cloudflare Free Cache Playbook

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

### 3.6 CI/CD Pipeline

**Workflow:** `compatibility_manual_main.yml` (manual trigger via `workflow_dispatch`)

| Job | OS | Python | Scope |
|-----|----|--------|-------|
| `guard-main` | — | — | Branch guard (only `main`) |
| `linux-full` | Ubuntu | 3.11 | Full `pytest tests/` |
| `windows-medium` | Windows | 3.11 | `ruff check` + `compileall` |
| `macos-smoke` | macOS | 3.11 | Syntax compilation check |

### 3.7 Branch & Git Strategy

- **Main development branch:** `main`
- **Release branch:** `release/v1.0-rc1`
- **Feature branches** from `main`, PR back to `main`.
- Existing branches: `master`, `release`, `release/v1.0-rc1`, `test`

### 3.8 Private-to-Public Sync

**Procedure:**
1. Start from `public/master`, squash private `main` into a local sync branch.
2. Create sync branch: `git checkout -B sync/private-to-public public/master`
3. Squash merge: `git merge --squash main --allow-unrelated-histories`
4. Safety gate: `git diff --cached --stat` — verify no secrets or private assets.
5. Commit: `git commit -m "chore(sync): publish private changes to public as single squashed commit"`
6. Push: `git push public sync/private-to-public:master`
7. Verify parity: `git diff --stat public/master..main`

**Important:** Do not use raw `git rev-list public/master...main` counts in release summaries — public syncs are squashed, so ancestry counts overstate the real delta.

### 3.9 PR Process

**Template fields:** Summary, Related Issues, Change Type (bug/feature/refactor/docs/test-only), Validation, Documentation Checklist, Safety Checklist.

**Review workflow:**
1. Fetch PR metadata and changed files/diff.
2. Review for: logic regressions, type hints, edge cases, security, performance.
3. Run targeted tests for touched areas. Add regression tests if coverage is missing.
4. Fix issues on PR branch, commit, push.
5. Merge only if code is correct and tests are green locally.
6. Post-merge: `git fetch --prune origin`, switch to `main`, `git pull --ff-only origin main`.

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

## 5. Editor & IDE Configuration (Windows — VS Code)

These settings live in `.vscode/settings.json` on the Windows machine and enforce consistent code quality:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
  "python.terminal.activateEnvironment": false,
  "editor.formatOnSave": true,
  "python.analysis.typeCheckingMode": "basic",
  "[python]": {
    "editor.defaultFormatter": "ms-python.python",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll.ruff": "explicit",
      "source.organizeImports.ruff": "explicit"
    }
  },
  "terminal.integrated.automationProfile.windows": {
    "path": "powershell.exe",
    "args": ["-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass"]
  },
  "chat.tools.terminal.autoApprove": {
    "git remote": true,
    "&": true
  }
}
```

- **Formatter:** Ruff (fix all + organize imports on save)
- **Type checking:** basic mode via Pylance
- **Terminal:** PowerShell with bypass execution policy
- **Interpreter:** `.venv/Scripts/python.exe` (Windows layout)

---

## 6. Project Structure Reference

```
LLM_trader/
  start.py                         # Entry point + CompositionRoot
  README.md                        # Project overview, setup, roadmap
  CONTRIBUTING.md                  # Contribution guidelines
  CHANGELOG.md                     # Version history
  requirements.txt / requirements-dev.txt
  keys.env / keys.env.example      # Secrets
  .pylintrc / pytest.ini           # Lint & test config
  .clinerules                      # Agent entry point (delegates here)

  .github/
    instructions/llm-trader.instructions.md   # Project guidelines (source of this file)
    skills/                                    # 10 reusable agent skills (source of this file)
    prompts/review-single-pr.prompt.md         # PR review template
    ISSUE_TEMPLATE/                            # Bug report, feature request
    pull_request_template.md
    workflows/compatibility_manual_main.yml    # CI/CD

  config/
    config.ini / config.ini.example
    model_pricing.json
    rag_priorities.json

  src/
    app.py                     # Main application module
    analyzer/                  # AnalysisEngine, TechnicalCalculator, PatternAnalyzer, formatters, prompts, pattern_engine
    config/                    # loader.py, protocol.py
    contracts/                 # model_contract.py, risk_contract.py
    dashboard/                 # FastAPI server, routers, static frontend
    factories/                 # 4 factory modules
    indicators/                # 20+ indicator modules (base, momentum, overlap, trend, volatility, etc.)
    logger/                    # logger.py
    managers/                  # ModelManager, PersistenceManager, RiskManager, ProviderOrchestrator
    notifiers/                 # Base, console, file notifier + components
    parsing/                   # unified_parser.py
    platforms/                 # AI providers, CCXT, CoinGecko, Alternative.me, DeFiLlama
    rag/                       # RagEngine, news ingestion, market components, scoring, taxonomy
    trading/                   # TradingBrainService + 5 collaborators, strategy, memory, statistics, monitors
    utils/                     # Profiler, token counter, graceful shutdown, decorators, etc.

  tests/                       # 59 test files, conftest.py

  docs/
    INDEX.md
    llm_agent_documentation.md
    detailed_file_documentation.md
    cloudflare_free_cache_playbook.md
    documentation_plan.md
    refactoring_plan.md
    plans/                     # Dated planning documents

  scripts/                     # Cross-platform startup + news scripts
  data/                        # Runtime state (not committed)
  website/                     # Astro-based project website
  img/                         # Dashboard screenshots
  logs/                        # Bot logs by date
```

---

## 7. Quick Reference — Command Cheat Sheet

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

# News preview
python scripts/fetch_free_news_preview.py --enrich-from-article-pages

# News body quality comparison
python scripts/compare_news_body_quality.py

# Git status (path-anchored)
git -C /home/qrak/LLM_trader status

# Measure private-to-public divergence
git rev-list --left-right --count public/master...main
```

---

## 8. File Discovery Summary

The following files were discovered and integrated into this master document:

| Source File | Location | Content |
|-------------|----------|---------|
| `llm-trader.instructions.md` | `.github/instructions/` | Core project guidelines (env, style, typing, Pydantic v2, tooling, sync, linting) — 75 lines |
| `.clinerules` | root | Agent entry point, delegates to instructions + skills — 30 lines |
| `agent-governance/SKILL.md` | `.github/skills/` | Safety boundaries, tool restrictions, auditability for autonomous trading — 36 lines |
| `agentic-eval/SKILL.md` | `.github/skills/` | Rubric-based LLM output evaluation for trading quality — 33 lines |
| `context-map/SKILL.md` | `.github/skills/` | Smallest-slice codebase mapping before changes — 48 lines |
| `mentoring-juniors/SKILL.md` | `.github/skills/` | Socratic mentoring methodology, PEAR loop, progressive clues — 309 lines |
| `news-pipeline-debugging/SKILL.md` | `.github/skills/` | News ingestion debugging workflow with scripts and artifacts — 72 lines |
| `private-public-sync/SKILL.md` | `.github/skills/` | 6-step private-to-public sync procedure with safety gates — 124 lines |
| `pydantic-v2-modeling/SKILL.md` | `.github/skills/` | Pydantic v2 patterns, validators, strict config — 34 lines |
| `refactor/SKILL.md` | `.github/skills/` | Code smells, design patterns, refactoring checklist — 645 lines |
| `repo-python-execution/SKILL.md` | `.github/skills/` | Python execution patterns for this repo (venv, bash, file-first) — 53 lines |
| `targeted-pytest-debugging/SKILL.md` | `.github/skills/` | Focused test debugging with captured output — 49 lines |
| `review-single-pr.prompt.md` | `.github/prompts/` | End-to-end PR review workflow with hard rules — 73 lines |
| `pull_request_template.md` | `.github/` | PR template with summary, validation, safety checklist — 34 lines |
| `bug_report.md` | `.github/ISSUE_TEMPLATE/` | Bug report template with reproduction steps — 37 lines |
| `feature_request.md` | `.github/ISSUE_TEMPLATE/` | Feature request with problem statement, scope, test strategy — 27 lines |
| `compatibility_manual_main.yml` | `.github/workflows/` | CI/CD pipeline (Linux full, Windows medium, macOS smoke) — 89 lines |
| `INDEX.md` | `docs/` | Documentation navigation hub — 26 lines |
| `llm_agent_documentation.md` | `docs/` | Architecture overview (lifecycle, DI, brain, dashboard, platforms) — 107 lines |
| `detailed_file_documentation.md` | `docs/` | File-by-file responsibility mapping — 101 lines |
| `cloudflare_free_cache_playbook.md` | `docs/` | CF cache rules, deployment modes, validation, guardrails — 267 lines |
| `documentation_plan.md` | `docs/` | Doc status, automation, completed updates, gaps — 60 lines |
| `refactoring_plan.md` | `docs/` | Refactoring status (P1 done, P2 Config, P3 AnalysisEngine) — 286 lines |
| `README.md` | root | Project overview, features, setup, roadmap, controls — 407 lines |
| `CONTRIBUTING.md` | root | Contribution guidelines, code standards, testing, docs — 67 lines |
| `config.ini` | `config/` | Active bot configuration (pair, timeframe, AI, RAG, risk, dashboard) — 202 lines |
| `.vscode/settings.json` | `.vscode/` (Windows) | IDE config: Python interpreter, Ruff formatter, PowerShell terminal profile |

**Total files integrated: 26 files across 8 directories.**

---

*Consolidated on 2026-05-23 from the LLM Trader repository at `/home/qrak/LLM_trader`.*
