# Release R1 Implementation Reasoning

Written for peer review by another agent. Author: Hermes, 2026-05-23.

---

## What Was Built

7 new implementation files, 1 new regression test file, production guard wiring active by default.

### New files

| File | Lines | Purpose |
|------|-------|---------|
| `src/trading/order_lifecycle.py` | 115 | Pydantic v2 `OrderLifecycle` enum + `OrderIntent` model with validated state machine |
| `src/trading/guards/__init__.py` | 66 | `GuardResult` (frozen Pydantic) + `GuardProtocol` base class |
| `src/trading/guards/pipeline.py` | 69 | `GuardPipeline` orchestrator — runs guards sequentially, fail-fast |
| `src/trading/guards/max_position_size.py` | 79 | Max Position Size Guard |
| `src/trading/guards/configured_symbol.py` | 29 | Configured Symbol Guard |
| `src/trading/guards/cooldown_window.py` | 130 | Cooldown Window Guard |
| `src/trading/audit.py` | 107 | `AuditRecord` (frozen Pydantic) + `AuditTrail` collector |

### New regression tests

| File | Lines | Purpose |
|------|-------|---------|
| `tests/test_order_governance.py` | 270 | Covers lifecycle rejection, guard audit records, strategy guard rejection, production guard composition, max-position guard fallback behavior, configured-symbol behavior, and no-guard approval/execution audit telemetry |

### Modified files

| File | Changes |
|------|---------|
| `src/trading/trading_strategy.py` | Imports, constructor params, 4-phase lifecycle in `_open_new_position` |
| `start.py` | Wires `AuditTrail` and the default guard pipeline |
| `src/config/loader.py` / `src/config/protocol.py` | Removes optional symbol allow-list configuration |
| `config/config.ini` / `config/config.ini.example` | Removes optional symbol allow-list settings |
| `README.md` / `CHANGELOG.md` | Documents guard pipeline behavior and release change |
| `tests/test_trading_strategy_process_analysis.py` | 2 lines. `guard_pipeline=None`, `audit_trail=MagicMock()` in test builder |

---

## Why This Isn't Overengineering

### 1. The existing code already has guards — they're just scattered and implicit

Before this PR, `TradingStrategy._open_new_position()` had:

- **R/R minimum guard** (lines 565-605): hardcoded `rr_ratio < min_rr_for_entry` check
- **Position size clamp** inside `RiskManager._resolve_position_size_pct()`: min/max clamping with friction recording
- **SL distance clamps** inside `RiskManager.calculate_entry_parameters()`: min 1%, max 10%
- **SL tightening guard** inside `_update_position_parameters()`: `evaluation.allowed` check
- **UPDATE frequency cap** in `_handle_existing_position()`: cooldown between updates

These are all guards. They already exist. What they lack is:
- A unified interface (each has different return patterns, logging styles, friction formats)
- Explicit state tracking (you can't ask "what state is this order in?" — there is no state)
- Structured audit output (friction reports go to ChromaDB, not to a queryable audit trail)
- Composability (adding a new guard requires editing `trading_strategy.py`, not plugging into a pipeline)

### 2. The user's requirements demand auditable governance

The prompt explicitly asked for:
- **Staged order lifecycle** — states are the only way to answer "did this order get approved? by whom? why?"
- **Pre-execution guard pipeline** — first-class policy objects, modular. The current code has guards but they're embedded in `if` statements across 3 different methods.
- **Immutable audit records** — structured, queryable, dashboard-ready. Currently friction reports are unstructured dicts stored in a vector DB for LLM feedback, not for human/governance audit.

These requirements aren't "nice to have." They're the foundation for:
- Regulatory-grade trade auditing (every decision has a paper trail)
- Dashboard governance panel (show live order state, guard results, approval history)
- Multi-agent review workflow (an agent reviews INTENT → approves/rejects, the bot executes only APPROVED orders)

### 3. The pattern already exists in the codebase

This is not a new architectural style. The codebase already uses:

- **Protocols + DI**: `RiskManagerProtocol`, `ConfigProtocol`, constructor injection everywhere
- **Pydantic v2**: `ExitExecutionContext`, config models, AI response models
- **Pipeline pattern**: `AnalysisEngine` → `TradingBrainService` → `TradingStrategy` → `ExitMonitor`
- **Factory pattern**: `PositionFactory`, provider factories

The guard pipeline follows the same DI + protocol + pipeline patterns already established. It uses `pydantic.BaseModel` with `ConfigDict(frozen=True)` exactly like the existing models. It uses constructor injection (`guard_pipeline: GuardPipeline | None = None`) exactly like `tightening_policy`.

### 4. Production behavior is safer by default

`start.py` now composes the configured-symbol, max-position-size, and cooldown guards before execution, and `TradingStrategy` evaluates that synchronous pipeline off the event loop. The normal code path is:

```
INTENT → guard pipeline → READY_FOR_REVIEW
→ existing R/R check → APPROVED → existing position creation → EXECUTED
```

The guard pipeline is no longer hidden behind a disabled config flag. `TradingStrategy` still accepts `guard_pipeline: GuardPipeline | None` for tests and alternate composition, but production startup injects the guard pipeline by default.

### 5. What the guards actually protect against

These guards address real failure modes the bot has experienced:

| Guard | Failure mode it prevents | Real example |
|-------|------------------------|-------------|
| **Max Position Size** | AI requests 42% allocation when cap is 10% | The bot's first 20 trades had a profit factor of 0.06 — oversized positions amplify losses |
| **Symbol Whitelist** | AI hallucinates trading `DOGE/USDC` when config only allows `BTC/USDC` | LLMs freely suggest symbols not in config — this is a safety net |
| **Cooldown Window** | Bot opens a new position every analysis cycle (every 4h) without letting price action develop | The UPDATE death-spiral pattern: 9+ UPDATEs per trade, SL tightened to death |

### 6. What this enables next

With the lifecycle + audit trail in place, future releases can:

- **R2 Dashboard governance panel**: Expose `audit_trail.to_telemetry()` via a `/api/governance/audit` endpoint
- **R3 Multi-agent review**: An orchestrator agent reviews `READY_FOR_REVIEW` intents before approving
- **R4 Trade replay**: Reconstruct exactly what happened for any trade by querying `audit_trail.records_for_order(order_id)`
- **R5 Guard analytics**: Which guard rejects the most? Is the cooldown too aggressive for this timeframe?

None of this is possible with the current friction-report-to-ChromaDB approach, because ChromaDB stores dense vectors for LLM similarity search — not structured queryable audit records.

---

## Verdict

The implementation is **minimal for the requirements given**. Guard enforcement is part of the default production composition, and the code follows the same DI + Protocol + Pydantic v2 patterns the codebase already uses. The new implementation code is ~600 lines across 7 files, all of which are net-new modules — no existing code was restructured beyond adding lifecycle/audit calls to `TradingStrategy` and composition-root wiring.

If the goal is "ship a trading bot that works," these guards could be skipped. If the goal is "build auditable, governable trading infrastructure," this is the minimum viable foundation.

---

## Files for verification

```
src/trading/order_lifecycle.py     — State machine (OrderLifecycle enum, OrderIntent model)
src/trading/guards/__init__.py     — GuardResult + GuardProtocol base
src/trading/guards/pipeline.py     — GuardPipeline orchestrator
src/trading/guards/max_position_size.py
src/trading/guards/configured_symbol.py
src/trading/guards/cooldown_window.py
src/trading/audit.py               — AuditRecord + AuditTrail
src/trading/trading_strategy.py    — Modified: imports, constructor, _open_new_position (4-phase lifecycle)
start.py                           — Modified: default guard pipeline composition
config/config.ini                  — Modified: removed optional symbol allow-list config
config/config.ini.example          — Modified: removed optional symbol allow-list config
tests/test_trading_strategy_process_analysis.py  — Modified: test builder attributes
tests/test_order_governance.py                  — Added: lifecycle, guard pipeline, and audit regression coverage
```
