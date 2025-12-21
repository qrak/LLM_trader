## Instruction Hierarchy & Precedence

This repository uses a **layered instruction system** for AI agents:

1. **Root `/AGENTS.md`** (this file): Global project context, universal rules, architecture overview
2. **Nested `/package/AGENTS.md`**: Subsystem-specific context that **extends or overrides** root rules
3. **`.github/instructions/*.instructions.md`**: GitHub Copilot-specific behaviors (uses `applyTo` patterns)

**Precedence Rule**: Nested instructions override root instructions when conflicts arise. Always check the most specific AGENTS.md file for the code you're working on.

---

## Project Overview

**Discord Crypto Analyzer** is a Python 3.11+ asyncio-first system that analyzes cryptocurrency markets using AI (Google AI/OpenRouter/LM Studio), publishes HTML reports to Discord automatically, and manages message lifecycle with automatic cleanup. Manual user command support has been deprecated in favor of scheduled or event-driven automated workflows.

### Tech Stack
- **Core**: Python 3.11+, asyncio, discord.py 2.6.3
- **Data**: pandas 2.3.2, numpy 2.2.3, ccxt 4.5.3 (multi-exchange)
- **Visualization**: plotly 6.3.0, kaleido (chart PNG export)
- **AI**: google-genai, OpenRouter API, LM Studio (local)
- **APIs**: CoinGecko, CryptoCompare, Alternative.me (Fear & Greed)
- **Storage**: JSON files (`data/`), SQLite cache (`cache/coingecko_cache.db`)

---

## Universal Coding Guidelines

### Async & Initialization
- **Asyncio-first**: Use `async def` and `await` everywhere. Mirror `start.py` event-loop setup (WindowsSelectorEventLoopPolicy on Windows)
- **Initialization order matters**: Components initialized in `DiscordCryptoBot.initialize()`, torn down in reverse in `shutdown()`
- **Inject dependencies**: Pass initialized API clients/managers into constructors. Never construct services inside other classes
- **Await readiness**: `discord_notifier.wait_until_ready()`, `symbol_manager.initialize()` must be awaited before use

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
- **Data classes for DTOs**: Use `@dataclass` for simple typed data carriers (Candle/OHLCV, IndicatorResult, PatternDetection, AnalysisResult). Prefer `frozen=True` for immutable value objects and `slots=True` for lower memory overhead when many instances are created.
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

# Run bot
python start.py
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
