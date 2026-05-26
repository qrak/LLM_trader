# 🛡️ Order Governance Pipeline

> **Module path:** `src/trading/guards/`
> **Type:** Pre-execution guard chain for trading signal validation

---

## Agent Persona & Role

The Governance Pipeline is the **last line of defense before any order signal reaches the simulated market.** It enforces declarative, configurable rules that every trading signal must pass before execution: symbol whitelist checks, cooldown windows, and position size limits.

The pipeline follows the **Chain of Responsibility pattern** — each guard runs independently, accumulating pass/fail decisions. If any guard fails, the order is blocked and the friction is recorded for Brain Agent learning.

---

## Pipeline Architecture

```
TradingStrategy → GuardPipeline.run(signal, position)
    ├── ConfiguredSymbolGuard     (whitelist check)
     ├── MaxPositionSizeGuard       (portfolio exposure limit)
     └── CooldownWindowGuard        (time since last trade)
         ↓
    Result: pass → TradingStrategy proceeds
            fail → Friction recorded → Brain Agent learns
```

### GuardPipeline (`pipeline.py`)
- Runs all guards in order, collecting results
- Returns `GuardResult(passed: bool, reason: str | None)`
- All guards must pass for order execution
- First failure short-circuits remaining guards (fail-fast)

---

## Guard: ConfiguredSymbolGuard (`configured_symbol.py`)

**Purpose:** Ensures the trading signal targets a configured symbol.

**Logic:**
- Signal must reference a symbol in the active configuration
- Prevents phantom pairs or misconfigured symbols

**Edge Cases:**
- Unknown symbol → blocked with friction "Symbol not in configured trading pairs"
- Multi-symbol signals → each symbol checked individually (future capability)

---

## Guard: CooldownWindowGuard (`cooldown_window.py`)

**Purpose:** Prevents rapid-fire trading by enforcing a minimum time window between consecutive executed BUY/SELL trades.

**Logic:**
- Reads the most recent executed BUY/SELL timestamp from trade history
- Cooldown = minimum `cooldown_minutes` (configurable)
- If elapsed < cooldown → block

**Edge Cases:**
- No prior trade → immediately passes
- Cooldown applies uniformly after the most recent BUY/SELL; direction is not treated specially
- Cooldown = 0 → disabled effectively

---

## Guard: MaxPositionSizeGuard (`max_position_size.py`)

**Purpose:** Caps total portfolio exposure across all open positions.

**Logic:**
- Tracks current open position exposure as fraction of portfolio
- New position size + existing exposure must ≤ `MAX_POSITION_SIZE` (default 10%)
- Falls back to tiered sizing: HIGH → 3%, MEDIUM → 2%, LOW → 1%

**Edge Cases:**
- No open positions → check is purely against max size
- Multi-symbol portfolio → checks aggregate exposure
- Position close / TP hit → recalculates available exposure
- Brain-learned thresholds may override default max position size

---

## Friction Recording

When a guard blocks an order, the friction is recorded:

```
VectorMemoryService.block_trade_feedback(guard_type, reason, context)
```

This feeds the Brain Agent's `get_context()` which shows the LLM:
- Recent blocked trades (last 5, max 168h old)
- Guard type + reason for each block
- Enables the LLM to understand why previous similar signals were rejected

---

## Configuration

All guard parameters are set in `config/config.ini`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_position_size` | 0.10 (10%) | Maximum portfolio exposure |
| `cooldown_minutes` | 60 | Minimum minutes between same-direction trades |
| `configured_symbols` | BTC/USDC | Active trading pairs |

Guards are **declarative** — they can be reviewed and modified without reading any code paths.
