## Instruction Hierarchy & Precedence

This repository uses a **layered instruction system** for AI agents:

1. **Root `/AGENTS.md`** (this file): Global project context, universal rules, architecture overview
2. **Nested `/package/AGENTS.md`**: Subsystem-specific context that **extends or overrides** root rules
3. **`.github/instructions/*.instructions.md`**: GitHub Copilot-specific behaviors (uses `applyTo` patterns)

**Precedence Rule**: Nested instructions override root instructions when conflicts arise. Always check the most specific AGENTS.md file for the code you're working on.

---

## Project Overview

**Crypto Trading Bot** is a Python 3.11+ asyncio-first console application that performs **automated cryptocurrency trading** using AI (Google AI/OpenRouter/LM Studio). The system collects market data, calculates technical indicators, detects patterns, and makes trading decisions (BUY/SELL/HOLD) with position management.

**IMPORTANT: This is a TRADING BOT, not an analysis tool. There is NO analysis mode.**

### Key Features
- **Automated Trading**: AI-powered trading decisions with stop-loss and take-profit management
- **Continuous Mode**: Periodic checks every timeframe candle with position monitoring
- **Rolling Memory**: Maintains context of last N trading decisions for better AI analysis
- **Multi-Exchange Support**: Binance, KuCoin, GateIO, MEXC, Hyperliquid
- **Streamlined Output**: JSON-only trading signals, no human-readable analysis

### Tech Stack
- **Core**: Python 3.11+, asyncio
- **Data**: pandas 2.3.2, numpy 2.2.3, ccxt 4.5.3 (multi-exchange)
- **AI**: google-genai, OpenRouter API, LM Studio (local)
- **APIs**: CoinGecko, CryptoCompare, Alternative.me (Fear & Greed)
- **Storage**: JSON files (`data/`, `trading_data/`), SQLite cache (`cache/coingecko_cache.db`)
- **Output**: Console only (no Discord, no HTML)

---

## Architecture: Trading Module

### Components (`src/trading/`)

| Component | File | Responsibility |
|-----------|------|----------------|
| `Position` | `dataclasses.py` | Immutable position data (entry, SL, TP, direction) |
| `TradeDecision` | `dataclasses.py` | Trading decision with timestamp, action, reasoning |
| `TradingMemory` | `dataclasses.py` | Rolling memory of last N decisions |
| `TradingBrain` | `dataclasses.py` | Bounded memory system for distilled trading insights |
| `TradingInsight` | `dataclasses.py` | Single distilled trading lesson from closed trades |
| `ConfidenceStats` | `dataclasses.py` | Win/loss statistics per confidence level (HIGH/MEDIUM/LOW) |
| `DataPersistence` | `data_persistence.py` | Save/load positions, trade history, memory, brain |
| `PositionExtractor` | `position_extractor.py` | Extract BUY/SELL/HOLD from AI response |
| `TradingStrategy` | `trading_strategy.py` | Position management, SL/TP checks, decision execution |

### Trading Brain System

The Trading Brain is a bounded, self-updating knowledge structure that stores **distilled trading insights** (not raw history) to help the AI avoid repeated mistakes and reinforce successful patterns.

**Key Features:**
- Fixed-size insight storage (max 10 insights) with FIFO eviction and category balancing
- Confidence calibration tracking (win rate per HIGH/MEDIUM/LOW confidence)
- Rule-based insight extraction from closed trades (no AI calls needed)
- Categories: STOP_LOSS, ENTRY_TIMING, RISK_MANAGEMENT, MARKET_REGIME

**Brain Context in Prompts:**
The brain context is injected into the system prompt (after performance_context, before previous_response) containing:
1. Confidence calibration stats (win rate per confidence level)
2. Distilled insights organized by category
3. Recommendations based on performance patterns

**Data Flow for Brain Updates:**
1. Position closes (SL/TP hit or signal close)
2. `TradingStrategy.close_position()` extracts market conditions
3. `DataPersistence.update_brain_from_closed_trade()` applies rule-based extraction
4. New insights added to brain with FIFO eviction
5. Brain saved to `trading_brain.json`

### Prompt Engine (`src/analyzer/prompts/`)

| Component | File | Responsibility |
|-----------|------|----------------|
| `TemplateManager` | `template_manager.py` | System prompts for trading decisions (JSON-only output) |
| `PromptBuilder` | `prompt_builder.py` | Builds prompts with market data |
| `ContextBuilder` | `context_builder.py` | Builds context sections (market data, sentiment) |

**NOTE**: Modify existing prompt engine files. Do NOT create new template files.

### Data Flow

```
start.py
    ↓
CryptoTradingBot.run()
    ↓
┌─────────────────────────────────────────────────────┐
│                 PERIODIC LOOP                        │
│                                                      │
│  1. Check existing position (SL/TP hit?)            │
│     └─ If closed: Update trading brain              │
│  2. Get brain_context from DataPersistence          │
│  3. Run AnalysisEngine.analyze_market()             │
│     └─ brain_context injected into system prompt    │
│  4. PositionExtractor.extract_trading_info()        │
│  5. TradingStrategy.process_analysis()              │
│  6. DataPersistence.save_trade_decision()           │
│  7. Wait for next timeframe candle                  │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Persistence Files (`trading_data/`)

| File | Purpose |
|------|---------|
| `positions.json` | Current open position |
| `trade_history.json` | Full history of all trading decisions |
| `trading_memory.json` | Rolling memory of last N decisions for context |
| `previous_response.json` | Last AI response for continuity |
| `trading_brain.json` | Bounded memory of distilled trading insights |

---

## Universal Coding Guidelines

### Async & Initialization
- **Asyncio-first**: Use `async def` and `await` everywhere. Mirror `start.py` event-loop setup (WindowsSelectorEventLoopPolicy on Windows)
- **Initialization order matters**: Components initialized in `CryptoTradingBot.initialize()`, torn down in reverse in `shutdown()`
- **Inject dependencies**: Pass initialized API clients/managers into constructors. Never construct services inside other classes
- **Await readiness**: `symbol_manager.initialize()` must be awaited before use

### Type Safety & Error Handling
- **No defensive checks**: Do NOT use `isinstance`, `hasattr`, `getattr`. Correct types must be passed from init
- **Pass correct objects**: If uncertain about types, write tests that assert types instead of adding runtime guards
- **Let errors surface**: Use proper exception handling, not defensive programming

### Code Style
- **Comments**: Minimal hash comments (`# short note`). Prefer expressive function names and docstrings
- **No backward-compat shims**: When refactoring, update all call sites. Delete unused files immediately
- **Small focused edits**: Follow existing patterns. Keep PRs small and targeted

### OOP & Design Patterns
- **OOP best practices**: Prefer encapsulation, clear class responsibilities, and proper class design. Use inheritance and polymorphism only when they model the domain effectively; prefer composition over inheritance for flexibility and testability.
- **Design patterns**: Apply common patterns where they improve clarity and reuse — Factory, Singleton (sparingly), Strategy, Observer, Decorator. Use `@property` for computed attributes when appropriate.
- **Abstraction**: Use abstract base classes and abstract properties to define contracts for subsystems; keep implementations injectable for testing.

### Dataclasses
- **Data classes for DTOs**: Use `@dataclass` for simple typed data carriers (Position, TradeDecision, TradingMemory). Prefer `frozen=True` for immutable value objects and `slots=True` for lower memory overhead when many instances are created.
- **Defaults**: Avoid mutable default arguments; use `field(default_factory=...)` for lists/dicts.
- **When to avoid**: For bulk numeric operations prefer numpy arrays / pandas DataFrame; dataclasses are for small structured objects passed between components.

---

**For subsystem-specific rules**, always consult the relevant `AGENTS.md` file in that directory.

---

## Build & Run Commands

### Local Development (Windows PowerShell)
```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start trading (default: BTC/USDT)
python start.py

# Trade specific symbol
python start.py ETH/USDT

# Trade with specific timeframe
python start.py BTC/USDT -t 4h

# Trade different symbol with timeframe
python start.py SOL/USDT -t 1h
```

### Testing
```powershell
# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

### Configuration
1. Copy `keys.env.example` to `keys.env`
2. Fill in Discord tokens and API keys
3. Edit `config/config.ini` for public settings

---

## Investigation & Code Change Workflow (REQUIRED)

**Always follow this workflow when investigating bugs or making changes:**

1. **Read Architecture Documentation FIRST**
   - Read the relevant `AGENTS.md` file for the subsystem you're investigating
   - Example: Investigating analyzer issues → Read `src/analyzer/AGENTS.md`
   - AGENTS.md documents component responsibilities, data flow, and methods
   - This establishes architectural context BEFORE diving into code

2. **Map Components to Source Files**
   - Use AGENTS.md to identify which component does what
   - Example: Find "ContextBuilder" → Located in `src/analyzer/prompts/context_builder.py`
   - Understand dependencies and how components interact

3. **Investigate the Code**
   - Now search/read the specific source file identified in step 2
   - Use grep to find relevant methods or calculations
   - Trace data flow through the component

4. **Create Test/Proof of Concept**
   - Write a test script to reproduce the issue
   - Compare old behavior vs expected behavior
   - Document the root cause

5. **Fix the Code**
   - Make targeted, focused changes
   - Update all call sites if changing a method signature
   - Test the fix with your test script

6. **Update AGENTS.md Documentation**
   - Document the fix in the AGENTS.md file for that component
   - Include what was fixed and why (especially for subtle bugs)
   - Update method descriptions if behavior changed
   - Example: Document off-by-one fixes, algorithm changes, etc.

7. **Verify Synchronization**
   - Ensure AGENTS.md reflects the actual code implementation
   - Check that all cross-referenced AGENTS.md files are consistent

**Example Workflow (Timeframe Bug Fix)**:
- ✅ Read `src/analyzer/AGENTS.md` → Found ContextBuilder responsibility
- ✅ Located `src/analyzer/prompts/context_builder.py` → Found build_market_data_section()
- ✅ Traced bug in indexing logic
- ✅ Created `test_timeframe_fix.py` → Proved off-by-one error
- ✅ Fixed `context_builder.py` line 158 → Changed index from `[-candle_count, 4]` to `[-(candle_count + 1), 4]`
- ✅ Updated `src/analyzer/AGENTS.md` → Documented bug and fix logic
- ✅ Verified both files synchronized

---

## Security & Secrets Management

- **Never commit** `keys.env` or any file containing API keys/tokens
- **Use environment variables** for all sensitive configuration
- **Validate inputs** from Discord commands before processing
- **Rate limit** API calls to external services

---

## Recommended MCP Tools

- **pylance**: Python analysis, imports, syntax errors
- **tavily**: Web search for crypto/library info
- **github_repo**: Search the remote `qrak/DiscordCryptoAnalyzer` GitHub repository for code examples, historical implementations, PRs, forks, or cross-repo usage; prefer workspace tools (`mcp_pylance_*` or editor search) for local development.
- **context7 mcp**: Retrieve up-to-date library documentation and code examples (useful for implementing or updating integrations and API usage).

---

## Common Pitfalls

- Forgetting to `await` initialization (`wait_until_ready()`, `initialize()`)
- Re-initializing components passed between subsystems (pass initialized clients)
- Changing provider fallback semantics without updating `config.ini` docs
- Not respecting message tracking (use `send_message()` for auto-tracking)

---

## Summary Documents Policy

**Do not create summary documents** in `.md` format or any format. All documentation should be maintained in the appropriate `AGENTS.md` file.

---

## Getting Help

- **Subsystem details**: Check the relevant `src/*/AGENTS.md` file
- **Configuration**: See `config/config.ini` and `keys.env.example`
- **Troubleshooting**: Each `AGENTS.md` has a troubleshooting section
- **Contributing**: See `CONTRIBUTING.md`
