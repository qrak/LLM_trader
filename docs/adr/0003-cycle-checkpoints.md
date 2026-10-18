# ADR 0003 — One persisted cycle checkpoint, not a workflow engine

- Status: accepted
- Date: 2026-10-18
- Deciders: repo owner + Hermes

## Context

`CryptoTradingBot._execute_trading_check` is a linear sequence: fetch ticker → check
position → analyse → decide → guard → book locally → forward → resolve intent. The
executor already owns durable state; the bot did not persist where *its own cycle* was. A
process death between "forwarded" and "confirmed" therefore relied on the next cycle's
reconciliation to notice, and nothing prevented the same decision from being forwarded
twice after a restart.

## Decision

Persist exactly one checkpoint per cycle in `data/cycle_state.json`
(`src/trading/cycle_state.py`), written atomically (temp file + `os.replace`), with a
state from `ANALYZING → DECIDED → INTENT_SENT → CONFIRMED/FAILED` and a stable
`decision_hash` of the intent (action, symbol, prices, quantity).

Only one transition changes behaviour: a checkpoint in `INTENT_SENT` whose
`decision_hash` matches the decision about to be forwarded suppresses the send and logs a
warning; reconciliation with the executor decides what actually happened. Every other
state is diagnostics.

## Alternatives considered

- **LangGraph / Temporal / a real workflow engine** — rejected: one linear cycle with six
  steps does not need a scheduler, a server or a DSL; the operational cost dwarfs the win.
- **Full DAG with per-step resumption** — rejected: steps are not independently
  re-runnable (re-running analysis produces a different decision), so "resume mid-DAG"
  would be a lie. Only the send step is idempotent-by-identity.
- **No checkpoint, rely on reconcile** — rejected: reconcile tells you the state *after*
  the fact; it does not stop a duplicate send in the same startup window.
- **Persist in SQLite** — rejected: one record, one writer, no queries; a JSON file is
  easier to inspect during an incident.

## Consequences

- Duplicate suppression is provable by hash, and the file survives `Shift+R` and reboots.
- A stuck `INTENT_SENT` is visible in `data/cycle_state.json`; operators get a warning line
  instead of a silent second order.
- Cost: the checkpoint must be updated at the right moment. It is written before booking a
  local position, so a suppressed duplicate also skips local booking.

## Verification

- `tests/domain_trading/test_cycle_state.py` (hash stability, atomic write, corrupt file,
  duplicate rule).
- Manual (dry-run/testnet): kill the bot between forward and confirm, restart, assert one
  executor intent and one exchange order.
