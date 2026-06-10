# Ticker Fetch Retry Fix & Cascading Error Protection

## Problem: Retry System Didn't Work

### Evidence from Bot Logs (2026-06-10 07:52:00)

```
app.py._fetch_current_ticker - Error fetching current ticker: binance GET https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDC
position_status_monitor.py.run_hard_exit_checks - Skipping hard exit checks because current ticker price is unavailable
notifier.py.send_position_status - Error sending position status: float division by zero
```

### Root Cause Chain

**① `_fetch_current_ticker()` has no retry mechanism**

`src/app.py:473-486` wraps the ccxt `fetch_ticker()` call in `try/except Exception` that **swallows ALL errors** and returns `None`. The `@retry_async` decorator exists in `src/utils/decorators.py` and is used on methods like `_load_exchange`, coingecko fetches, etc., but was **never applied** to `_fetch_current_ticker`. Even if it were applied, the internal `except Exception` would prevent the decorator from ever seeing the exception (the method returns `None` normally instead of raising).

There is also **no inline retry loop** inside the method — it's a single-shot call with no backoff.

**② When ticker fetch fails → hard exit checks are silently skipped**

`position_status_monitor.py:185-186`:
```python
ticker = await self.fetch_current_ticker()  # Returns None on failure
current_price = float(ticker.get('last', ticker.get('close', 0))) if ticker else None
# current_price is now None
```

At line 159, `run_hard_exit_checks` logs a warning and aborts — no failover to the last known price, no retry, no fallback candle close price.

**③ Cascading ZeroDivisionError in notification**

When `current_price` is `None` and the notifier is called (line 196), it substitutes `0.0`. This flows into `base_notifier.py:185-190`:

```python
stop_distance_pct = ((position.stop_loss - current_price) / current_price) * 100
target_distance_pct = ((position.take_profit - current_price) / current_price) * 100
```

`0.0` as divisor → `ZeroDivisionError` → entire `send_position_status` fails.

### Why The Existing Retry Infrastructure Couldn't Help

| Layer | Has retry? | Why it failed |
|-------|-----------|---------------|
| `@retry_async` decorator | ✅ Exists in `decorators.py` | Not applied to `_fetch_current_ticker()` |
| `_NETWORK_EXCEPTIONS` handling | ✅ Catches timeouts, connection errors | Not reachable — internal try/except swallows everything first |
| Exchange-level retry (`handle_exchange_error`) | ✅ Retries rate limits, logs others | Same swallow problem |
| `_fetch_ticker_data()` wrapper | ❌ No retry | Just catches and returns `(None, None)` |
| `position_status_monitor._loop()` | ❌ No retry | One shot per 15-min cycle, no last-price caching |

## Fix Plan

### Fix 1: Apply `@retry_async` to `_fetch_current_ticker` (app.py)

**Change:** Add `@retry_async(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)` decorator and **remove the internal `try/except Exception`** so retryable errors (network errors, timeouts, rate limits) propagate to the decorator.

- Network errors (timeouts, DNS, connection refused) → `_NETWORK_EXCEPTIONS` → retried with exponential backoff
- Exchange errors → `ccxt.ExchangeError` → rate limits retried, non-retryable (BadSymbol, etc.) re-raised
- Non-retryable/unexpected → `handle_unexpected_error()` → re-raised immediately
- After all retries exhausted → error propagates to callers who already handle `None` returns

**Caller impact:**
- `_fetch_ticker_data()` (line 327) — has `try/except Exception` that catches whatever the decorator re-raises, returns `(None, None)`. Safe.
- Direct call at line 261 (startup ticker fetch) — **not** inside any try/except. Wrap in a try/except that logs and continues.

**Why this is safe:** The two call sites already handle the None/Nothing case gracefully:
- `_fetch_ticker_data()` already returns `(None, None)` on failure
- Line 261's result is only used for dashboard price update (which handles None)
- Position status monitor already checks for None price

### Fix 2: Guard against zero price in `calculate_stop_target_distances` (base_notifier.py)

**Change:** Add early return `(0.0, 0.0)` when `current_price` is None or `<= 0`.

This prevents the cascading `ZeroDivisionError` when ticker fetch fails and the notifier still receives `current_price=0.0`.

### Fix 3: Skip notification when price is unavailable (position_status_monitor.py)

**Change:** In `_loop()` at line 193-198, guard the `send_position_status` call with a check that `current_price is not None` AND `current_price > 0`, instead of substituting `0.0`.

Sending a status message with `$0.00` as the current price is misleading anyway — better to skip the update and keep the last valid status.

### Fix 4: Tests

**New test file:** `tests/test_ticker_retry.py`

Test scenarios:
1. `_fetch_current_ticker` retries on `ccxt.RequestTimeout` → succeeds on 3rd attempt → returns ticker
2. `_fetch_current_ticker` retries on `aiohttp.ClientConnectorError` → exhausts → `_fetch_ticker_data` returns `(None, None)`
3. `_fetch_current_ticker` does NOT retry on `ccxt.BadSymbol` → immediately propagates (non-retryable)
4. `_fetch_current_ticker` retries on `ccxt.RateLimitExceeded` → succeeds on retry
5. `calculate_stop_target_distances` with `current_price=0.0` → returns `(0.0, 0.0)` instead of crashing
6. `calculate_stop_target_distances` with `current_price=None` → returns `(0.0, 0.0)` instead of crashing
7. Position status monitor skips notification when price is unavailable

**Existing test impact:** `test_app_discord_message_flow.py` mocks `_fetch_ticker_data` directly, not affected. `test_exit_monitoring.py` uses separate PositionStatusMonitor with injected mocks, not affected.

## Files Modified

| File | Change |
|------|--------|
| `src/app.py` | Add `@retry_async` to `_fetch_current_ticker`, remove internal try/except, wrap line 261 in try/except |
| `src/notifiers/base_notifier.py` | Guard `calculate_stop_target_distances` against zero/None price |
| `src/trading/position_status_monitor.py` | Skip notification when price unavailable instead of sending $0.00 |
| `tests/test_ticker_retry.py` | NEW: Retry behavior tests |

## Verification

1. Run full test suite: `.venv/bin/python -m pytest tests/ -x --tb=short -q` (baseline ~900 tests)
2. New test file passes independently
3. No regressions in existing ticker-related tests (`test_app_discord_message_flow`, `test_exit_monitoring`, `test_api_rate_limiting_backoff`)
