# 🛡️ Order Governance Pipeline

> **Module path:** `src/trading/guards/`
> **Type:** Pre-execution guard chain for trading signal validation

---

## Agent Persona & Role

The Governance Pipeline is the **last line of defense before any order signal reaches the simulated market.** It enforces declarative, configurable rules that every trading signal must pass before execution: symbol whitelist checks, cooldown windows, and position size limits.

The pipeline follows the **Chain of Responsibility pattern** — guards run sequentially and fail fast. If any guard fails, the order is blocked and an audit rejection is recorded.

---

## Pipeline Architecture

```
TradingStrategy → GuardPipeline.evaluate(intent, capital, config)
    ├── ConfiguredSymbolGuard     (whitelist check)
    ├── MaxPositionSizeGuard       (explicit requested size cap)
    └── CooldownWindowGuard        (time since last SQLite-recorded BUY/SELL)
         ↓
    Result: pass → TradingStrategy proceeds
            fail → audit rejection recorded
```

### GuardPipeline (`pipeline.py`)
- Runs guards in order and stops at the first failure
- Returns `list[GuardResult]` for the guards that were evaluated
- All guards must pass for order execution
- First failure short-circuits remaining guards (fail-fast)

---

## Guard: ConfiguredSymbolGuard (`configured_symbol.py`)

**Purpose:** Ensures the trading signal targets a configured symbol.

**Logic:**
- Signal must reference `config.CRYPTO_PAIR`
- Prevents phantom pairs or misconfigured symbols

**Edge Cases:**
- Unknown symbol → blocked with reason "does not match configured trading pair"

---

## Guard: CooldownWindowGuard (`cooldown_window.py`)

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

## Guard: MaxPositionSizeGuard (`max_position_size.py`)

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

## Friction Recording

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

## Configuration

All guard parameters are set in `config/config.ini`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_position_size` | 0.10 (10%) | Maximum explicit/requested position size; RiskManager also clamps fallback sizing to this cap |
| `timeframe` | 4h | Cooldown duration source |
| `crypto_pair` | BTC/USDC | Single configured trading pair |

Guards are **declarative** — they can be reviewed and modified without reading any code paths.
