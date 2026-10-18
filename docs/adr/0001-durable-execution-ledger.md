# ADR 0001 — Durable execution ledger in the executor, intent journal in the bot

- Status: accepted
- Date: 2026-10-18
- Deciders: repo owner + Hermes (implementation review)

## Context

Two processes act on the same money: the bot decides, the executor places orders. Both
restart (crash, `Shift+R` reload, Windows reboot) while an order may be in flight. The
question "did this intent reach the exchange?" must have an answer that survives a
process exit, otherwise a restart can duplicate an order or the bot can book a fill it
never got.

The repo already had: `latest_decision.json` (file fallback), HTTP `POST /decision`,
a verdict journal and an append-only position-intent JSONL on the bot side.

## Decision

Keep two records with different jobs:

1. **Executor** owns the durable ledger (`llm_trader_executor/src/execution_ledger.py`,
   SQLite): one explicit file per `account|environment`, `schema_version`, command
   states `pending…error`, and it refuses to open a file stamped with another namespace.
2. **Bot** owns the intent journal (`src/trading/executor_reconciliation.py`): append-only
   JSONL of what it *believes* it sent, with a `decision_hash`/`event_id`, and a rule that
   local position state may only change on an executor-confirmed source.

The bot never treats its own optimism as evidence. `INTENT_SENT` without confirmation is
reconciled against the executor on the next cycle or at startup.

## Alternatives considered

- **In-memory state only** — rejected: a restart loses "order in flight", and the first
  thing a restart does is re-send.
- **Single JSONL file for everything** — rejected: JSONL is a log, not a queryable state
  machine; the executor needs to answer "what is the current command state for symbol X"
  without replaying and guessing.
- **Redis / external queue** — rejected: adds a service to a single-box deployment;
  SQLite gives the same durability with zero operations.
- **Trusting the exchange's open-orders list as the only truth** — rejected: it cannot
  distinguish "my order" from a manually placed one, and it is unavailable during
  exchange degradation.

## Consequences

- Restart paths are testable without an exchange: replay the journal, assert no second
  forward for the same `decision_hash`.
- Two sources of truth must agree; divergence is reported (`_report_state_divergence`)
  instead of silently fixed.
- Cost: every new intent shape must carry a stable identity, or it cannot be reconciled.

## Verification

- `tests/domain_brain/test_executor_reconciliation*`, `tests/test_recovery_identity.py`
  (executor), ledger namespace/schema tests.
- Manual: kill the executor mid-order, restart, assert exactly one exchange order.
