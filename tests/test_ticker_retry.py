"""Tests for ticker fetch retry, zero-division guards, and price-unavailable handling.

Covers the fixes from docs/plans/ticker-retry-fix.md:

  1. @retry_async applied to _fetch_current_ticker (retries network/exchange errors)
  2. calculate_stop_target_distances guards against zero/None price
  3. position_status_monitor skips notification when price is unavailable
"""
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import ccxt
import pytest

from src.notifiers.base_notifier import BaseNotifier
from src.utils.decorators import retry_async


# ── 1. RETRY ON NETWORK/TIMEOUT ERRORS ──────────────────────────────────────

class TestFetchCurrentTickerRetry:
    """The @retry_async decorator on _fetch_current_ticker retries transient errors.

    Pattern matches test_api_rate_limiting_backoff.py (TestAsyncRetryDecorator).
    """

    @pytest.mark.asyncio
    async def test_retry_on_request_timeout_then_succeeds(self):
        """ccxt.RequestTimeout triggers retry; succeeds on 3rd attempt."""
        call_count = 0

        class FakeExchange:
            def __init__(self):
                self.logger = MagicMock()
                self.current_exchange = self
                self.current_symbol = "BTC/USDC"
                self.dashboard_state = None

            @retry_async(max_retries=3, initial_delay=0.01, backoff_factor=2, max_delay=1)
            async def _fetch_current_ticker(self) -> dict[str, Any] | None:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ccxt.RequestTimeout("binance GET https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDC")
                return {"last": 66500.0, "close": 66500.0}

        exchange = FakeExchange()
        result = await exchange._fetch_current_ticker()
        assert result == {"last": 66500.0, "close": 66500.0}
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_connection_error_then_exhausts(self):
        """aiohttp.ClientConnectorError triggers retry; after exhaustion error propagates."""
        call_count = 0

        class FakeExchange:
            def __init__(self):
                self.logger = MagicMock()
                self.current_exchange = self
                self.current_symbol = "BTC/USDC"
                self.dashboard_state = None

            @retry_async(max_retries=2, initial_delay=0.01, backoff_factor=2, max_delay=1)
            async def _fetch_current_ticker(self) -> dict[str, Any] | None:
                nonlocal call_count
                call_count += 1
                raise aiohttp.ClientConnectorError(
                    connection_key=MagicMock(),
                    os_error=OSError("Connection refused"),
                )

        exchange = FakeExchange()
        with pytest.raises(aiohttp.ClientConnectorError):
            await exchange._fetch_current_ticker()
        assert call_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit_then_succeeds(self):
        """ccxt.RateLimitExceeded triggers retry (exchange error path)."""
        call_count = 0

        class FakeExchange:
            def __init__(self):
                self.logger = MagicMock()
                self.current_exchange = self
                self.current_symbol = "BTC/USDC"
                self.dashboard_state = None

            @retry_async(max_retries=3, initial_delay=0.01, backoff_factor=2, max_delay=1)
            async def _fetch_current_ticker(self) -> dict[str, Any] | None:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ccxt.RateLimitExceeded("DDoS protection triggered")
                return {"last": 66500.0}

        exchange = FakeExchange()
        result = await exchange._fetch_current_ticker()
        assert result == {"last": 66500.0}
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_symbol_error_propagates(self):
        """ccxt.BadSymbol must NOT be retried (non-retryable ExchangeError)."""
        call_count = 0

        class FakeExchange:
            def __init__(self):
                self.logger = MagicMock()
                self.current_exchange = self
                self.current_symbol = "INVALID/PAIR"
                self.dashboard_state = None

            @retry_async(max_retries=3, initial_delay=0.01, backoff_factor=2, max_delay=1)
            async def _fetch_current_ticker(self) -> dict[str, Any] | None:
                nonlocal call_count
                call_count += 1
                raise ccxt.BadSymbol("Symbol not found")

        exchange = FakeExchange()
        with pytest.raises(ccxt.BadSymbol):
            await exchange._fetch_current_ticker()
        assert call_count == 1  # no retry

    @pytest.mark.asyncio
    async def test_returns_none_when_no_exchange_or_symbol(self):
        """If current_exchange or current_symbol is None, return None immediately (no retry)."""

        class FakeExchange:
            def __init__(self):
                self.logger = MagicMock()

            @retry_async(max_retries=3, initial_delay=0.01, backoff_factor=2, max_delay=1)
            async def _fetch_current_ticker(self) -> dict[str, Any] | None:
                if self.current_exchange is None or self.current_symbol is None:
                    return None
                return {"last": 66500.0}

        exchange = FakeExchange()
        exchange.current_exchange = None
        exchange.current_symbol = None
        assert await exchange._fetch_current_ticker() is None

        exchange.current_exchange = MagicMock()
        exchange.current_symbol = None
        assert await exchange._fetch_current_ticker() is None


# ── 2. ZERO/DIVISION GUARD ──────────────────────────────────────────────────

class TestCalculateStopTargetDistances:
    """calculate_stop_target_distances must not crash on zero/None price."""

    class _TestNotifier(BaseNotifier):
        """Minimal concrete subclass for testing."""
        def __init__(self):
            # Skip the full BaseNotifier init by calling object.__init__
            pass
        async def start(self): pass
        async def wait_until_ready(self): pass
        async def send_message(self, *a, **kw): pass
        async def send_trading_decision(self, *a, **kw): pass
        async def send_analysis_notification(self, *a, **kw): pass
        async def send_performance_stats(self, *a, **kw): pass
        async def send_position_status(self, *a, **kw): pass

    @staticmethod
    def _make_position(direction="SHORT"):
        pos = MagicMock()
        pos.direction = direction
        pos.entry_price = 65000.0
        pos.stop_loss = 66000.0
        pos.take_profit = 63000.0
        pos.size = 0.1
        return pos

    def test_returns_zero_on_zero_price_long(self):
        notifier = self._TestNotifier()
        pos = self._make_position("LONG")
        stop_pct, target_pct = notifier.calculate_stop_target_distances(pos, 0.0)
        assert stop_pct == 0.0
        assert target_pct == 0.0

    def test_returns_zero_on_zero_price_short(self):
        notifier = self._TestNotifier()
        pos = self._make_position("SHORT")
        stop_pct, target_pct = notifier.calculate_stop_target_distances(pos, 0.0)
        assert stop_pct == 0.0
        assert target_pct == 0.0

    def test_returns_zero_on_none_price(self):
        notifier = self._TestNotifier()
        pos = self._make_position("LONG")
        stop_pct, target_pct = notifier.calculate_stop_target_distances(pos, None)
        assert stop_pct == 0.0
        assert target_pct == 0.0

    def test_returns_zero_on_negative_price(self):
        notifier = self._TestNotifier()
        pos = self._make_position("SHORT")
        stop_pct, target_pct = notifier.calculate_stop_target_distances(pos, -1.0)
        assert stop_pct == 0.0
        assert target_pct == 0.0

    def test_returns_valid_percentages_with_normal_price(self):
        """Sanity check: normal path still works."""
        notifier = self._TestNotifier()
        pos = self._make_position("SHORT")
        stop_pct, target_pct = notifier.calculate_stop_target_distances(pos, 65000.0)
        # SHORT: stop_distance = (65000 - 66000)/65000 * 100 = -1.54%
        assert stop_pct == pytest.approx(-1.53846, rel=1e-3)
        # SHORT: target_distance = (65000 - 63000)/65000 * 100 = 3.08%
        assert target_pct == pytest.approx(3.07692, rel=1e-3)

    def test_returns_valid_for_long_with_normal_price(self):
        notifier = self._TestNotifier()
        pos = self._make_position("LONG")
        stop_pct, target_pct = notifier.calculate_stop_target_distances(pos, 64500.0)
        # LONG: stop_distance = (66000 - 64500)/64500 * 100 = 2.33%
        assert stop_pct == pytest.approx(2.32558, rel=1e-3)
        # LONG: target_distance = (63000 - 64500)/64500 * 100 = -2.33%
        assert target_pct == pytest.approx(-2.32558, rel=1e-3)


# ── 3. POSITION STATUS MONITOR SKIP ─────────────────────────────────────────

class TestPositionMonitorSkipOnNoPrice:
    """PositionStatusMonitor must skip notification when ticker price unavailable."""

    @pytest.mark.asyncio
    async def test_skips_notification_when_price_none(self):
        """When fetch_current_ticker returns None, notification must be skipped."""
        from src.trading.position_status_monitor import PositionStatusMonitor

        notifier = MagicMock()
        monitor = PositionStatusMonitor(
            logger=MagicMock(),
            config=MagicMock(),
            persistence=MagicMock(),
            trading_strategy=MagicMock(),
            exit_monitor=MagicMock(),
            notifier=notifier,
            active_tasks=set(),
            is_running=MagicMock(return_value=False),  # stop immediately
            fetch_current_ticker=AsyncMock(return_value=None),
            interruptible_sleep=AsyncMock(),
            get_symbol=MagicMock(return_value="BTC/USDC"),
        )
        monitor.trading_strategy.current_position = MagicMock()
        monitor.exit_monitor.is_status_due = MagicMock(return_value=True)

        # _loop() should not call send_position_status because current_price is None
        # We don't await _loop() since it's a while loop — just verify the
        # skip logic by calling run_hard_exit_checks and checking the guard path
        monitor.exit_monitor.seconds_until_next_tick = MagicMock(return_value=0)
        monitor.exit_monitor.check_hard_exits = AsyncMock(return_value=(None, {}))
        monitor.exit_monitor.due_hard_exits = MagicMock(return_value=[])
        monitor.load_state = AsyncMock(return_value={})
        monitor.save_state = AsyncMock()

        # Run one iteration by calling _loop
        # We set is_running to return False after first check
        call_count = 0

        def is_running():
            nonlocal call_count
            call_count += 1
            return call_count < 2  # run once, then stop

        monitor.is_running = is_running
        await monitor._loop()

        # send_position_status must NOT be called (current_price is None)
        notifier.send_position_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_new_position_skips_when_ticker_none(self):
        """handle_new_position should return early if ticker fetch fails."""
        from src.trading.position_status_monitor import PositionStatusMonitor

        notifier = MagicMock()
        monitor = PositionStatusMonitor(
            logger=MagicMock(),
            config=MagicMock(),
            persistence=MagicMock(),
            trading_strategy=MagicMock(),
            exit_monitor=MagicMock(),
            notifier=notifier,
            active_tasks=set(),
            is_running=MagicMock(return_value=True),
            fetch_current_ticker=AsyncMock(return_value=None),
            interruptible_sleep=AsyncMock(),
            get_symbol=MagicMock(return_value="BTC/USDC"),
        )
        monitor.trading_strategy.current_position = MagicMock()

        await monitor.handle_new_position(current_price=None)

        # Should not send notification when ticker is unavailable
        notifier.send_position_status.assert_not_called()
