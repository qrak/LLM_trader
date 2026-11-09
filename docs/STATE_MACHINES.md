# State machines and lifecycle vocabulary

This page is the single map of every implicit state machine in the bot. It tells a human
operator, a coding agent and the trading model what a state means, which evidence may
change it, and where it is visible. Code that disagrees with this page is a defect.

The bot is a mirror, not the exchange. The evidence hierarchy is:

1. Exchange fills and orders (the venue is the truth).
2. The executor's verdict journal and exit journal.
3. The executor `/position` payload only when `confirmation.state == "confirmed"`,
   `reconcile_status == "verified"`, `protection_status == "active"`, no unresolved or
   duplicated protection, both order ids present, and the quantity matches at venue
   precision.
4. The local `current_position` mirror.
5. The local position-intent ledger.

Never infer upward. `unverified` never means `flat`; a missing exit record never means the
position was closed; a local SL/TP value is a mirror, not proof of exchange protection.

## 1. Stored machines

### 1.1 Analysis cycle — `CycleState`

Owner: `src/trading/lifecycle.py`, persisted by `src/trading/cycle_state.py` in
`data/trading/cycle_state.json`. The file holds the **last finished cycle**. The currently
running cycle is an in-memory marker (`_analyzing_since`), so a crash never overwrites the
last terminal state and never weakens duplicate suppression.

| State | Meaning |
|---|---|
| `analyzing` | The loop is building context and calling the model; in memory only |
| `decided` | A decision exists but nothing was forwarded (HOLD, or executor disabled) |
| `skipped` | A decision was deliberately discarded; `detail` carries the reason code |
| `intent_sent` | A command was forwarded and confirmation is still missing (or a duplicate was suppressed) |
| `confirmed` | The executor proved the command |
| `failed` | The analysis or the forward failed, or the executor refused the command |

Transition table (one cycle; a new cycle replaces the file, it is not a transition):

```text
analyzing   -> decided | skipped | failed
decided     -> intent_sent | failed
intent_sent -> confirmed | failed
skipped, confirmed, failed -> {}
```

Reason codes in `detail` (`SKIP_REASON_LABELS`): `analysis_error`, `stale_position`,
`protection_changed`, `duplicate_intent`, `approval_not_granted`, `forward_failed`,
`no_decision`, `no_analysis`, `executor_disabled`, `hold`, `no_verdict`.

Duplicate suppression: a BUY/SELL whose decision hash matches either the last
finished checkpoint in `intent_sent` **or a pending/unknown ENTRY in the
position-intent ledger** is not forwarded again; the cycle records `intent_sent`
with `duplicate_intent`. The ledger stores `decision_hash` with the ENTRY, so a
later HOLD cycle that overwrites the checkpoint no longer releases suppression;
it is released when the executor confirms or refuses that entry. A restart
reloads both the checkpoint and the ledger.

### 1.2 Order review — `OrderLifecycle`

Owner: `src/trading/order_lifecycle.py`. This is a **pre-trade review** machine, not an
execution machine. Execution proof lives in the position-intent ledger; there is no
`EXECUTED` order state.

```text
INTENT -> READY_FOR_REVIEW | REJECTED
READY_FOR_REVIEW -> REJECTED
REJECTED -> {}
```

### 1.3 Position intent — `PositionIntent`

Owner: `src/trading/executor_reconciliation.py`; persisted append-only in
`data/trading/bot_position_intents.jsonl` (last line per key wins). States are the module
constants moved to `src/trading/lifecycle.py`.

| State | Meaning |
|---|---|
| `pending` | The command was registered and forwarded; no terminal verdict yet |
| `unknown` | The executor could not answer, or the evidence is incomplete; the local state is unchanged |
| `confirmed` | The executor's journal/receipt proves the command (and the fill, when applicable) |
| `refused` | The executor or the bot's own policy refused the command |
| `local_only` | No command was sent (executor integration disabled); there is no confirmation channel |

Evidence-driven transitions (`INTENT_TRANSITIONS`):

```text
pending    -> unknown | confirmed | refused
unknown    -> confirmed | refused
confirmed  -> unknown          (a later receipt may retract a confirmation, e.g. a CLOSE without a bookable fill)
refused    -> {}
local_only -> {}
```

Registering a new command for the same key (a repeated CLOSE for the same position) is a
**supersede**, not an evidence transition: it may start again as `pending`, `unknown` or
`local_only` from any state. Only `executor_verdict_journal`, `executor_exit_journal` and
`executor_position_tracker` may confirm or refuse; any other source is refused and logged
at CRITICAL. An illegal evidence transition is logged at CRITICAL and the ledger is **not**
rewritten.

### 1.4 Approval — `PendingApprovalStore`

Owner: `src/trading/pending_approval.py`; persisted in `data/trading/approvals.json`.
An approval means "not executed yet", never "rejected".

```text
pending -> approved | rejected | expired
approved, rejected, expired -> {}
```

`expired` is derived on read from `expires_at`. Approve/reject on a non-pending record is a
no-op that returns the current status. The bot polls the store while a decision waits, so
an approval only matters while the analysis cycle is blocked (TTL and poll interval are
`approval_ttl_minutes` / `approval_poll_seconds`, default 15 min / 5 s).

### 1.5 Brain rebuild — `BrainLifecycle`

Owner: `src/dashboard/dashboard_state.py`; in memory and broadcast over the websocket.
`idle -> updating -> rebuilt | error`; `rebuilt`/`error` may go back to `updating`. This is
the learning state, not the analysis cycle; the cycle never "learns".

## 2. Derived classifications

These have no transition table because they are recomputed from evidence on every read.
They are values, not stored state.

| Value | Source | Meaning |
|---|---|---|
| `ExecutorQueryState` | last reconciliation + `EXECUTOR_API_ENABLED` | `disabled`, `open`, `flat`, `unknown`. `disabled` is not `flat`. |
| `ReconcileState` | `reconcile_local_position()` | `no_position`, `open_executor_reported`, `exit_booked`, `divergence_unproven`, `unverified` |
| `VerificationState` | last reconciliation + freshness | `exchange_verified`, `executor_reported`, `unverified`; stale evidence downgrades to `unverified` |
| `PositionPhase` | position + latest intent + reconciliation + verification | `flat`, `entering`, `closing`, `divergence`, `open_verified`, `open_reported`, `open_unverified` |

Position phase precedence:

1. no local position -> `flat`
2. latest CLOSE intent `pending`/`unknown` -> `closing`
3. latest ENTRY intent `pending`/`unknown` -> `entering`
4. latest ENTRY intent `refused` -> `divergence` (local position without a proven entry)
5. latest CLOSE intent `confirmed` but the position is still present -> `divergence`
6. reconciliation `divergence_unproven` -> `divergence`
7. verification `exchange_verified` -> `open_verified`
8. verification `executor_reported` -> `open_reported`
9. otherwise -> `open_unverified`

Supporting classifications that are **not** position states:

- Fill/market verification (`FILL_VERIFICATION_VERIFIED`, `FILL_VERIFICATION_DIVERGENT`,
  `FILL_VERIFICATION_UNAVAILABLE`) and learning eligibility
  (`execution_quality_is_eligible`) decide whether a closed trade may become a lesson or
  a statistic. Different machine; see `src/trading/data_models.py`.
- Semantic-rule freshness (`fresh`, `maturing`, `stale`, `legacy`) is derived from age and
  evidence in `src/trading/vector_memory_rules.py`; rules are never deleted by age.
- Provider circuit (`closed`, `open`, `half_open`) is derived from consecutive failures and
  the cooldown in `src/managers/provider_health.py`; `half_open` means the cooldown elapsed
  and a probe is allowed.
- Dashboard websocket (`connected`, `reconnecting`, `disconnected`) is UI-only.
- Shutdown/reload (`running`, `shutting_down`, `reload_requested`) is process lifecycle.

## 3. Machine-readable state

`GET /api/status/state` is public, `no-store`, and carries no approval amounts or payloads:

```json
{
  "available": true,
  "generated_at": "2026-11-09T01:55:00+00:00",
  "cycle": {"state": "intent_sent", "label": "Intent sent — awaiting proof", "severity": "warn",
            "since": "...", "age_seconds": 42, "symbol": "BTC/USDC", "detail": "unknown"},
  "position": {"has_position": true, "symbol": "BTC/USDC", "direction": "LONG",
               "phase": "open_unverified", "label": "Open — unverified", "severity": "warn",
               "verification": {"state": "unverified", "label": "Unverified",
                                "severity": "warn", "checked_at": null, "age_seconds": null,
                                "detail": "..."},
               "latest_intent": {"action": "ENTRY", "state": "pending", "label": "Pending",
                                 "at": "...", "detail": "..."}},
  "approval": {"pending": 0, "label": "None", "severity": "neutral", "oldest_expires_at": null},
  "executor": {"enabled": true, "state": "unknown", "label": "Unknown"},
  "brain": {"status": "idle", "label": "Brain idle", "severity": "neutral"}
}
```

- `available: false` means the dashboard has no live provider or the view failed; it is not
  a trading state.
- The same object is embedded as `lifecycle` in `GET /api/brain/position`.
- Severity is one of `neutral`, `info`, `warn`, `critical`, `ok` and drives the dashboard
  chip colour. Labels are the only human text; the UI does not invent its own.
- The position card renders `verification.age_seconds` as `checked 12s ago`; when the age is
  missing it falls back to the raw `checked_at` timestamp.

The trading model sees one line in its position context:

```text
- Execution proof: exchange-verified (checked 12s ago) — protection active
- Execution proof: executor-reported (checked 3m ago) — NOT exchange-verified
- Execution proof: unverified — do not assume exchange protection
```

## 4. Operator and agent rules

- Read `/api/status/state` before acting on a position; do not reconstruct state from logs.
- A `refused` command did not run; do not treat its requested SL/TP as active.
- An `unknown` intent is not a failure and not a success; local state stays unchanged.
- A close is booked only from fill evidence (price, positive quantity, fee provenance).
- A local exit condition (ticker hit) is an intent, not an exit price; it books nothing.
- `divergence` means the local book and the venue disagree; never sell assets to make a
  local CLOSE true.
- The exchange's protective leg is a catastrophe backstop; the stop that runs is the bot's
  real-market check (`reference_exit_check_seconds`).
- Approval pending means the analysis loop is waiting; it is not a rejection.
- A config file newer than the process means "restart required"; the admin form edits the
  file only.

## 5. Files

Vocabulary identifiers as exported by `src/trading/lifecycle.py`:
`CycleState` (`ANALYZING`, `DECIDED`, `SKIPPED`, `INTENT_SENT`, `CONFIRMED`, `FAILED`),
`INTENT_PENDING`, `INTENT_UNKNOWN`, `INTENT_CONFIRMED`, `INTENT_REFUSED`,
`INTENT_LOCAL_ONLY`, `INTENT_ACTION_ENTRY`, `INTENT_ACTION_UPDATE`, `INTENT_ACTION_CLOSE`,
`RECONCILE_NO_POSITION`, `RECONCILE_OPEN_EXECUTOR_REPORTED`, `RECONCILE_EXIT_BOOKED`,
`RECONCILE_DIVERGENCE`, `RECONCILE_UNVERIFIED`, `VERIFICATION_EXCHANGE_VERIFIED`,
`VERIFICATION_EXECUTOR_REPORTED`, `VERIFICATION_UNVERIFIED`, `PositionPhase`,
`ExecutorQueryState`, `BrainLifecycle`.

| Concern | File |
|---|---|
| Vocabulary, transitions, projection | `src/trading/lifecycle.py` |
| Cycle checkpoint persistence | `src/trading/cycle_state.py` |
| Order review | `src/trading/order_lifecycle.py` |
| Intent ledger, reconciliation, verification | `src/trading/executor_reconciliation.py` |
| Approval store | `src/trading/pending_approval.py` |
| Verification/exit monitor | `src/trading/position_status_monitor.py` |
| Snapshot + cycle orchestration | `src/app.py` |
| API | `src/dashboard/routers/ws_router.py`, `src/dashboard/routers/brain.py` |
| Dashboard UI | `src/dashboard/static/modules/lifecycle_panel.js`, `position_panel.js` |
| Admin approvals / restart | `src/dashboard/static/admin/index.html`, `src/dashboard/routers/admin.py` |
