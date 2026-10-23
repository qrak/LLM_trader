# ADR 0005 — The real price owns the exit; the exchange leg is a catastrophe backstop

- Status: accepted
- Date: 2026-10-22
- Deciders: repo owner + Hermes

## Context

The bot reads the REAL market (`[exchanges] sandbox = false`) while the executor
trades a TESTNET whose protective orders rest on the testnet book. With the
executor live the bot deliberately stopped evaluating its own bracket
(`manages_exits() = False`), so the exchange leg was the only stop.

On 2026-10-22 the testnet printed a venue-wide flash crash: BTC/USDC traded down
to 66,975 while the real market held near 84,037. The stop-loss leg at 83,100
filled at 83,016.90 — a price the real market never made. The booking machinery
did the right thing (the trade was recorded, excluded from statistics and from
learning), but the position was gone for a reason the market never produced. The
same failure mode recurs: measured over three days, the testnet closed minutes
below the real market by more than 1% twenty times on BTC/USDT and twice on
BTC/USDC.

## Decision

Two layers, each owning what it can prove:

1. **Real price owns the exit.** `PositionStatusMonitor.check_reference_exits()`
   evaluates the position's SL/TP against the real price the bot already reads,
   every `reference_exit_check_seconds` (default 10 s, 0 disables), and hands a
   hit to the executor immediately through `forward_local_exit_request()`. This is
   an INTENT: `TradingStrategy.check_reference_exit()` books nothing, updates no
   metrics and persists nothing; only fill evidence closes the trade.
2. **The exchange leg is a catastrophe backstop.** The executor places the
   stop-loss leg at `max(2.5 x |entry - requested stop|, 1.5% x entry)` from the
   entry (`PROTECTION_BACKSTOP_MULTIPLIER`, `PROTECTION_BACKSTOP_MIN_PCT`), so
   venue-only noise cannot pre-empt the monitor. The take-profit leg is unchanged.

The two settings move together: disabling the real-price check while the executor
still places a widened leg leaves the position with a backstop and no stop, which
is why `1.0`/`0` in the executor and `0` in the bot are documented as one rollback
pair, and why the bot warns when `sandbox = true` makes the "real" check measure
the testnet book.

## Alternatives considered

- **Keep the exchange stop at the requested level and only add the real-price
  check** — rejected: with the monitor live but the leg still tight, the leg keeps
  firing first on venue noise, so nothing about the incident changes.
- **Remove the exchange stop entirely, keep only the target leg** — rejected:
  fully immune to venue noise (including the -20% crash), but a dead bot or
  executor would leave a leveraged-percentage position with no protection at all.
- **Widen the target leg too** — rejected: a venue spike to the target sells at
  the target price, which is a fill the strategy wanted, and keeping the leg means
  a dead bot still gets its exit. The asymmetry is deliberate.
- **Re-price or trail the backstop as the market moves** — deferred: it would keep
  the noise margin constant, but every re-placement is an unprotected window
  (`naked_window_seconds`) and a chance to fail, for a margin that is already
  wider than every observed excursion except a venue-wide crash.
- **Have the executor tighten the leg when the bot's heartbeat goes stale** —
  deferred: it inverts the dependency (the executor would watch the bot) and needs
  a heartbeat contract neither service has today.
- **Let the executor manage the real-price exit itself** — rejected: it would move
  a trading decision into the venue adapter and put two closers on one position,
  while the bot already owns the position levels, the model's tightening and the
  booking path.

## Consequences

- A real-market stop is now acted on within seconds instead of hours; the
  intra-cycle window that fabrication class relied on is closed.
- When the bot or the executor is down, protection is the widened leg: the
  worst case on that position is roughly `2.5x` the intended stop distance, not
  the intended one. That is the price of not being stopped out by a venue artifact.
- A venue-wide crash (the 2026-10-22 event) still triggers the widened leg. No
  sane backstop survives a -20% venue move; the difference is that it now takes a
  move of that scale to do it.
- MAE/MFE (`max_drawdown_pct`) remain unpopulated while the executor is live: the
  fast check deliberately does not write. Tracked as separate work, not silently
  changed here.

## Verification

- Bot: `tests/domain_trading/test_guards_and_exits.py::TestRealMarketExitCheck`
  (incident scenario, stop and target hits, short symmetry, cadence bounds,
  forwarding retries, alert pacing, no-price warning, failing forward),
  `test_bookkeeping_authority.py::test_reference_exit_writes_nothing_until_fill_evidence`,
  `test_config_and_pricing.py::TestReferenceExitCheckCadence`,
  `test_app_runtime.py::test_the_monitor_can_forward_a_close_through_the_public_hook`.
- Executor: `tests/test_protection_backstop.py` (rule, bounds, placement, tracked
  entry price, market-check fallback, legacy pair, report and exit-journal fields).
- Manual: restarted bot/executor print the backstop mode at startup; a real price
  at the stop logs `LOCAL EXIT CONDITION … forwarded=yes`.
