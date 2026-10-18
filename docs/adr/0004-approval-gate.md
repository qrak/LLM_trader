# ADR 0004 — Human approval gate is opt-in and expires into a refusal

- Status: accepted
- Date: 2026-10-18
- Deciders: repo owner + Hermes

## Context

High-stakes automation needs a human in the loop for the cases that matter, but the bot
runs unattended for weeks. A gate that is always on would stop trading; a gate that waits
forever would freeze the cycle; a gate whose timeout executes the trade would defeat its
purpose.

## Decision

`risk_management.require_approval_above_usdc` (default `0` = disabled). When set, a
BUY/SELL whose quote value (`entry_price × quantity`) reaches the threshold is written to
`data/approvals.json` as `pending`, the bot waits in-cycle up to `approval_ttl_minutes`,
polling `approval_poll_seconds`, and:

- `approved` → the decision proceeds through the normal path (guards, forward, checkpoint);
- `rejected` → dropped, logged, never executed;
- `expired` → dropped, logged, never executed — expiry is a refusal;
- unknown/missing entry → dropped.

The admin console exposes `GET /api/admin/approvals`, `POST .../{id}/approve`,
`POST .../{id}/reject`. There is no "approve all" and no auto-approve.

## Alternatives considered

- **Push the approval into the executor** — rejected: the executor's job is to execute
  what it is told, with its own safety guards; putting a queue there duplicates state.
- **Slack/Teams webhook approval** — deferred: the console already has auth and a session;
  a second channel means a second identity model.
- **Auto-execute on timeout** — rejected: the price context that justified the decision is
  gone after the TTL; executing late is not the same trade.
- **Always-on gate for every order** — rejected: the operator would rubber-stamp, which is
  worse than no gate.

## Consequences

- A cycle can block for up to the TTL (default 15 min). Exit monitoring runs in its own
  task, so protections are unaffected.
- Approvals are auditable: `approvals.json` keeps the decision payload and the final
  status.
- Default behaviour is unchanged; enabling it is a config edit, not a code change.

## Verification

- `tests/domain_trading/test_pending_approval.py` (approve, reject, expiry-as-refusal,
  persistence across instances, corrupt file).
- Manual: set the threshold below the current position value, force analysis, approve in
  the console, confirm exactly one forward.
