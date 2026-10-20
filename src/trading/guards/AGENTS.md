# AGENTS.md — Trading Guards

Scope: `src/trading/guards/`. Root [`AGENTS.md`](../../../AGENTS.md) is canonical.

Two guards, chain of responsibility, fail-fast: `GuardPipeline` (`pipeline.py`) runs `evaluate(intent, capital, config)` in order, returns `list[GuardResult]` for the guards evaluated, short-circuits on the first failure. All must pass or the order is blocked and an audit rejection is recorded.

- `ConfiguredSymbolGuard` (`configured_symbol.py`): signal must reference `config.CRYPTO_PAIR`; anything else is blocked with "does not match configured trading pair".
- `MaxPositionSizeGuard` (`max_position_size.py`): reads `MAX_POSITION_SIZE` (default 0.10) and requires it positive and finite. A positive finite requested `position_size` must be ≤ that cap. Missing, non-finite or non-positive requested sizes pass through so `RiskManager` applies fallback sizing; an invalid `MAX_POSITION_SIZE` fails closed.

## Friction boundary

Guard-pipeline failures are audit-recorded before risk calculation and never call vector memory directly. `VectorMemoryService.store_blocked_trade(...)` is written only by `RiskManager` frictions, R:R minimum blocks and premature SL-tightening blocks; `BrainContextProvider.get_context()` then feeds the last 5 blocks (≤168h) with guard type and reason into the prompt.

Config (`config/config.ini`): `max_position_size` 0.10 — the cap for explicit **and** RiskManager fallback sizing — `timeframe` 4h, `crypto_pair` BTC/USDC. Guards stay declarative: reviewable without reading trading code.
