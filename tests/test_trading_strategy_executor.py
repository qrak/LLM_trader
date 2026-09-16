"""Characterization tests for existing-position handling and the executor handshake.

Split out of test_trading_strategy_branches.py; the shared builders live there.
"""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.trading.data_models import MarketConditions
from tests.test_trading_strategy_branches import (
    _make_position,
    _make_strategy,
)


class TestHandleExistingPosition:
    """Verify signal handling for existing positions."""

    def test_close_signal_exits(self):
        """CLOSE signal triggers close_position."""
        pos = _make_position(direction="LONG")
        strategy, _, _, _, _, _ = _make_strategy(current_position=pos)
        result = asyncio.run(strategy._handle_existing_position(
            signal="CLOSE", confidence="MEDIUM",
            stop_loss=None, take_profit=None,
            current_price=105.0, symbol="BTC/USDC", reasoning="Market reversing",
            market_conditions=MarketConditions(),
        ))
        assert result is not None
        assert result.action == "CLOSE"
        assert strategy.current_position is None

    def test_close_long_signal_exits(self):
        """CLOSE_LONG signal also triggers close."""
        pos = _make_position(direction="LONG")
        strategy, _, _, _, _, _ = _make_strategy(current_position=pos)
        result = asyncio.run(strategy._handle_existing_position(
            signal="CLOSE_LONG", confidence="HIGH",
            stop_loss=None, take_profit=None,
            current_price=105.0, symbol="BTC/USDC", reasoning="Target reached",
            market_conditions=MarketConditions(),
        ))
        assert result is not None
        assert result.action == "CLOSE"
        assert strategy.current_position is None

    def test_update_rejected_too_soon(self):
        """UPDATE rejected when min interval hasn't elapsed."""
        pos = _make_position(direction="LONG")
        strategy, _, _, _, _, _ = _make_strategy(current_position=pos)
        strategy._last_position_update_time = datetime.now(timezone.utc)
        result = asyncio.run(strategy._handle_existing_position(
            signal="UPDATE", confidence="MEDIUM",
            stop_loss=96.0, take_profit=None,
            current_price=105.0, symbol="BTC/USDC", reasoning="Tighten SL",
            market_conditions=MarketConditions(),
        ))
        assert result is None

    def test_update_sl_succeeds(self):
        """UPDATE with SL change works after interval (within 150% widening cap)."""
        pos = _make_position(stop_loss=95.0, take_profit=115.0, direction="LONG")
        strategy, _, _, _, _, _ = _make_strategy(current_position=pos)
        strategy._last_position_update_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        result = asyncio.run(strategy._handle_existing_position(
            signal="UPDATE", confidence="MEDIUM",
            stop_loss=93.0, take_profit=115.0,
            current_price=105.0, symbol="BTC/USDC", reasoning="Widen SL",
            market_conditions=MarketConditions(),
        ))
        assert result is not None
        assert result.action == "UPDATE"

    def test_update_tp_succeeds(self):
        """UPDATE with TP change works."""
        pos = _make_position(stop_loss=95.0, take_profit=110.0, direction="LONG")
        strategy, _, _, _, _, _ = _make_strategy(current_position=pos)
        strategy._last_position_update_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        result = asyncio.run(strategy._handle_existing_position(
            signal="UPDATE", confidence="HIGH",
            stop_loss=95.0, take_profit=120.0,
            current_price=105.0, symbol="BTC/USDC", reasoning="Extended TP",
            market_conditions=MarketConditions(),
        ))
        assert result is not None
        assert result.action == "UPDATE"

    def test_close_signal_skipped_when_executor_has_no_position(self):
        """CLOSE signal skipped when executor reports no open position."""
        pos = _make_position(direction="LONG")
        strategy, _, _, _, _, _ = _make_strategy(current_position=pos)
        strategy._executor_has_position = AsyncMock(return_value=False)

        result = asyncio.run(strategy._handle_existing_position(
            signal="CLOSE", confidence="HIGH",
            stop_loss=None, take_profit=None,
            current_price=105.0, symbol="BTC/USDC", reasoning="Reversing",
            market_conditions=MarketConditions(),
        ))
        assert result is None

    def test_update_signal_skipped_when_executor_has_no_position(self):
        """UPDATE skipped when executor reports no open position."""
        pos = _make_position(stop_loss=95.0, take_profit=115.0, direction="LONG")
        strategy, _, _, _, _, _ = _make_strategy(current_position=pos)
        strategy._last_position_update_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        strategy._executor_has_position = AsyncMock(return_value=False)

        result = asyncio.run(strategy._handle_existing_position(
            signal="UPDATE", confidence="MEDIUM",
            stop_loss=93.0, take_profit=115.0,
            current_price=105.0, symbol="BTC/USDC", reasoning="Widen SL",
            market_conditions=MarketConditions(),
        ))
        assert result is None

    def test_executor_has_position_returns_true_when_disabled(self):
        """When EXECUTOR_API_ENABLED is False, assume position exists."""
        strategy, _, _, _, _, _ = _make_strategy(
            current_position=_make_position(),
            EXECUTOR_API_ENABLED=False,
        )
        result = asyncio.run(strategy._executor_has_position("BTC/USDC"))
        assert result is True

    def test_executor_has_position_returns_true_when_no_url(self):
        """When EXECUTOR_API_URL is empty, assume position exists."""
        strategy, _, _, _, _, _ = _make_strategy(
            current_position=_make_position(),
            EXECUTOR_API_ENABLED=True,
            EXECUTOR_API_URL="",
        )
        result = asyncio.run(strategy._executor_has_position("BTC/USDC"))
        assert result is True


class TestExecutorTriState:
    """Regression guard for ghost-position fix: _executor_has_position
    returns True (open) / False (confirmed closed) / None (unverifiable).

    None must never trigger a close/update; False must clear ghost state.
    """

    @staticmethod
    def _make_executor_strategy(**overrides):
        return _make_strategy(
            current_position=_make_position(direction="LONG"),
            EXECUTOR_API_ENABLED=True,
            EXECUTOR_API_URL="http://executor:8000",
            **overrides,
        )

    def _stub_http_client(self, strategy, status_code=200, payload=None, exc=None):
        """Replace _get_http_client with a mock returning a stubbed response."""
        client = MagicMock()
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = "stub body"
        resp.json.return_value = payload if payload is not None else {}
        if exc is not None:
            client.get = AsyncMock(side_effect=exc)
        else:
            client.get = AsyncMock(return_value=resp)
        strategy._get_http_client = MagicMock(return_value=client)
        return client

    def test_http_200_open_true(self):
        """Executor 200 + open:true → True (position exists)."""
        strategy, _, _, _, _, _ = self._make_executor_strategy()
        self._stub_http_client(strategy, 200, {"open": True})
        result = asyncio.run(strategy._executor_has_position("BTC/USDC"))
        assert result is True

    def test_position_query_url_strips_decision_suffix(self):
        """EXECUTOR_API_URL ends with /decision — the position query must hit
        the base /position path, not /decision/position (which 404s).

        Regression guard: config default is http://127.0.0.1:9199/decision,
        and naive url + \"/position\" produced /decision/position → the
        executor answered 404 and every UPDATE/CLOSE was skipped.
        """
        strategy, _, _, _, _, _ = _make_strategy(
            current_position=_make_position(direction="LONG"),
            EXECUTOR_API_ENABLED=True,
            EXECUTOR_API_URL="http://127.0.0.1:9199/decision",
        )
        client = self._stub_http_client(strategy, 200, {"open": False})

        result = asyncio.run(strategy._executor_has_position("BTC/USDC"))

        assert result is False
        called_url = client.get.call_args[0][0]
        assert called_url == "http://127.0.0.1:9199/position"
        assert "/decision/position" not in called_url

    def test_position_query_url_base_url_without_suffix(self):
        """EXECUTOR_API_URL without /decision suffix still works (backward compat)."""
        strategy, _, _, _, _, _ = self._make_executor_strategy()
        client = self._stub_http_client(strategy, 200, {"open": False})

        result = asyncio.run(strategy._executor_has_position("BTC/USDC"))

        assert result is False
        called_url = client.get.call_args[0][0]
        assert called_url == "http://executor:8000/position"

    def test_http_200_open_false(self):
        """Executor 200 + open:false → False (confirmed no position)."""
        strategy, _, _, _, _, _ = self._make_executor_strategy()
        self._stub_http_client(strategy, 200, {"open": False})
        result = asyncio.run(strategy._executor_has_position("BTC/USDC"))
        assert result is False

    def test_http_error_returns_none(self):
        """HTTP 404/500 → None (cannot verify, do not act)."""
        strategy, _, _, _, _, _ = self._make_executor_strategy()
        self._stub_http_client(strategy, status_code=404)
        result = asyncio.run(strategy._executor_has_position("BTC/USDC"))
        assert result is None

    def test_network_error_returns_none(self):
        """Connection failure → None (cannot verify, do not act)."""
        strategy, _, _, _, _, _ = self._make_executor_strategy()
        self._stub_http_client(strategy, exc=RuntimeError("connection refused"))
        result = asyncio.run(strategy._executor_has_position("BTC/USDC"))
        assert result is None

    def test_close_false_resets_ghost_state(self):
        """CLOSE + executor confirmed no position → reset local state, no close."""
        pos = _make_position(direction="LONG")
        strategy, _, persistence, _, _, _ = self._make_executor_strategy()
        strategy.current_position = pos
        strategy._executor_has_position = AsyncMock(return_value=False)

        result = asyncio.run(strategy._handle_existing_position(
            signal="CLOSE", confidence="HIGH",
            stop_loss=None, take_profit=None,
            current_price=105.0, symbol="BTC/USDC", reasoning="Reversing",
            market_conditions=MarketConditions(),
        ))
        assert result is None
        assert strategy.current_position is None
        persistence.async_save_position.assert_called_with(None)

    def test_close_none_keeps_position(self):
        """CLOSE + unverifiable executor → skip close, keep local state."""
        pos = _make_position(direction="LONG")
        strategy, _, persistence, _, _, _ = self._make_executor_strategy()
        strategy.current_position = pos
        strategy._executor_has_position = AsyncMock(return_value=None)

        result = asyncio.run(strategy._handle_existing_position(
            signal="CLOSE", confidence="HIGH",
            stop_loss=None, take_profit=None,
            current_price=105.0, symbol="BTC/USDC", reasoning="Reversing",
            market_conditions=MarketConditions(),
        ))
        assert result is None
        assert strategy.current_position is pos
        persistence.async_save_position.assert_not_called()

    def test_update_false_resets_ghost_state(self):
        """UPDATE + executor confirmed no position → reset local state, no update."""
        pos = _make_position(stop_loss=95.0, take_profit=115.0, direction="LONG")
        strategy, _, persistence, _, _, _ = self._make_executor_strategy()
        strategy.current_position = pos
        strategy._executor_has_position = AsyncMock(return_value=False)

        result = asyncio.run(strategy._handle_existing_position(
            signal="UPDATE", confidence="MEDIUM",
            stop_loss=93.0, take_profit=115.0,
            current_price=105.0, symbol="BTC/USDC", reasoning="Widen SL",
            market_conditions=MarketConditions(),
        ))
        assert result is None
        assert strategy.current_position is None
        persistence.async_save_position.assert_called_with(None)

    def test_update_none_keeps_position(self):
        """UPDATE + unverifiable executor → skip update, keep local state."""
        pos = _make_position(stop_loss=95.0, take_profit=115.0, direction="LONG")
        strategy, _, persistence, _, _, _ = self._make_executor_strategy()
        strategy.current_position = pos
        strategy._executor_has_position = AsyncMock(return_value=None)

        result = asyncio.run(strategy._handle_existing_position(
            signal="UPDATE", confidence="MEDIUM",
            stop_loss=93.0, take_profit=115.0,
            current_price=105.0, symbol="BTC/USDC", reasoning="Widen SL",
            market_conditions=MarketConditions(),
        ))
        assert result is None
        assert strategy.current_position is pos
        persistence.async_save_position.assert_not_called()


class TestPositionContextSlTightening:
    """Regression guard for bd4e43b: sentinel must nudge SL toward entry."""

    def test_short_progress_not_zero(self):
        """SHORT: price_progress MUST NOT be 0% after the fix."""
        pos = _make_position(entry_price=100.0, stop_loss=105.0, take_profit=85.0, direction="SHORT")
        strategy, _, _, brain, _, _ = _make_strategy(current_position=pos)
        brain.get_dynamic_thresholds = MagicMock(return_value={})
        strategy.brain_service = brain
        ctx = strategy.get_position_context(current_price=95.0)
        assert "price progress" in ctx.lower()
        lines = ctx.split("\n")
        progress_line = [line_item for line_item in lines if "progress" in line_item.lower()]
        assert progress_line, "No progress line in context"
        assert "0.0%" not in progress_line[0], (
            f"BUG REGRESSION: SHORT price_progress is 0.0%. Context:\n{ctx}"
        )

    def test_no_position_context(self):
        """get_position_context without position shows capital status."""
        strategy, _, _, _, _, _ = _make_strategy(current_position=None)
        ctx = strategy.get_position_context()
        assert "Status: None" in ctx


class TestSlWideningDirectionalLogs:
    """Verify correct log branches for LONG vs SHORT SL widening."""

    def test_short_sl_widening_log(self):
        """SHORT: SL moved higher (wider) logs SHORT-specific message within 150% cap."""
        pos = _make_position(stop_loss=105.0, take_profit=90.0, direction="SHORT")
        strategy, logger, _, _, _, _ = _make_strategy(current_position=pos)
        updated = asyncio.run(strategy._update_position_parameters(
            stop_loss=107.0, take_profit=None, current_price=100.0,
        ))
        assert updated is True
        widening_logs = [c for c in logger.info.call_args_list if "Widening" in str(c) and "SHORT" in str(c)]
        assert len(widening_logs) > 0, f"No SHORT widening log in: {[c[0][0] for c in logger.info.call_args_list]}"

    def test_long_sl_widening_log(self):
        """LONG: SL moved lower (wider) logs LONG-specific message within 150% cap."""
        pos = _make_position(stop_loss=95.0, take_profit=110.0, direction="LONG")
        strategy, logger, _, _, _, _ = _make_strategy(current_position=pos)
        updated = asyncio.run(strategy._update_position_parameters(
            stop_loss=93.0, take_profit=None, current_price=105.0,
        ))
        assert updated is True
        widening_logs = [c for c in logger.info.call_args_list if "Widening" in str(c) and "LONG" in str(c)]
        assert len(widening_logs) > 0


class TestEntryRollback:
    """Entry confirmation: the executor processes orders asynchronously, so a
    queued entry may still be blocked. rollback_blocked_entry() must roll back
    the local phantom (and record a compensating CLOSE) only when the executor
    explicitly and repeatedly reports no position after an HTTP-delivered
    forward — never on file-fallback delivery or transient query errors.
    """

    @pytest.mark.asyncio
    async def test_confirm_true_when_executor_confirms(self):
        strategy, *_ = _make_strategy()
        strategy._executor_has_position = AsyncMock(return_value=True)
        assert await strategy.confirm_entry_with_executor("BTC/USDC") is True

    @pytest.mark.asyncio
    async def test_confirm_via_verdict_journal_executed(self, tmp_path):
        """order_id present + journal says executed → keep position."""
        strategy, *_ = _make_strategy()
        journal = tmp_path / "executor_verdicts.jsonl"
        journal.write_text(
            '{"order_id": "order-abc", "verdict": "executed", "reason": ""}\n',
            encoding="utf-8",
        )
        strategy.config = SimpleNamespace(EXECUTOR_VERDICT_PATH=str(journal))
        with patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_DELAY", 0.001):
            assert await strategy.confirm_entry_with_executor("BTC/USDC", order_id="order-abc") is True

    @pytest.mark.asyncio
    async def test_confirm_via_verdict_journal_blocked(self, tmp_path):
        """order_id present + journal says blocked → roll back (False)."""
        strategy, *_ = _make_strategy()
        journal = tmp_path / "executor_verdicts.jsonl"
        journal.write_text(
            '{"order_id": "order-abc", "verdict": "blocked", "reason": "Notional exceeds max"}\n',
            encoding="utf-8",
        )
        strategy.config = SimpleNamespace(EXECUTOR_VERDICT_PATH=str(journal))
        with patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_DELAY", 0.001):
            assert await strategy.confirm_entry_with_executor("BTC/USDC", order_id="order-abc") is False

    @pytest.mark.asyncio
    async def test_confirm_via_verdict_journal_error(self, tmp_path):
        """order_id present + journal says error → roll back (False)."""
        strategy, *_ = _make_strategy()
        journal = tmp_path / "executor_verdicts.jsonl"
        journal.write_text(
            '{"order_id": "order-abc", "verdict": "error", "reason": "insufficient funds"}\n',
            encoding="utf-8",
        )
        strategy.config = SimpleNamespace(EXECUTOR_VERDICT_PATH=str(journal))
        with patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_DELAY", 0.001):
            assert await strategy.confirm_entry_with_executor("BTC/USDC", order_id="order-abc") is False

    @pytest.mark.asyncio
    async def test_confirm_via_verdict_journal_no_entry_fail_open(self, tmp_path):
        """order_id present but no journal entry (executor not yet processed /
        journal missing) → fail-open True, never roll back a possibly-live order."""
        strategy, *_ = _make_strategy()
        missing = tmp_path / "does_not_exist.jsonl"
        strategy.config = SimpleNamespace(EXECUTOR_VERDICT_PATH=str(missing))
        with patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_ATTEMPTS", 2), \
             patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_DELAY", 0.001):
            assert await strategy.confirm_entry_with_executor("BTC/USDC", order_id="order-abc") is True

    @pytest.mark.asyncio
    async def test_confirm_via_verdict_journal_late_verdict(self, tmp_path):
        """2026-08-11 race regression via journal: executor writes the verdict
        after several polls (its 10s queue tick) — must keep polling, not roll
        back on absence. A verdict appearing on poll 3 → True."""
        strategy, *_ = _make_strategy()
        journal = tmp_path / "executor_verdicts.jsonl"
        strategy.config = SimpleNamespace(EXECUTOR_VERDICT_PATH=str(journal))

        async def delayed_write():
            await asyncio.sleep(0.005)
            journal.write_text(
                '{"order_id": "order-abc", "verdict": "executed", "reason": ""}\n',
                encoding="utf-8",
            )

        task = asyncio.create_task(delayed_write())
        try:
            with patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_ATTEMPTS", 10), \
                 patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_DELAY", 0.002):
                assert await strategy.confirm_entry_with_executor("BTC/USDC", order_id="order-abc") is True
        finally:
            task.cancel()

    @pytest.mark.asyncio
    async def test_confirm_false_when_stably_no_position(self):
        strategy, *_ = _make_strategy()
        strategy._executor_has_position = AsyncMock(return_value=False)
        with patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_ATTEMPTS", 2), \
             patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_MIN_FALSE_REPORTS", 2), \
             patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_DELAY", 0.001):
            assert await strategy.confirm_entry_with_executor("BTC/USDC") is False

    @pytest.mark.asyncio
    async def test_confirm_true_when_executor_slow_to_process(self):
        """2026-08-11 race regression: the executor polls its queue every 10s,
        so a freshly forwarded entry legitimately reports 'no position' for
        several polls before the next tick processes it. With the old
        MIN_FALSE_REPORTS=2 the bot gave up after ~5s, rolled back the phantom,
        and recorded a compensating CLOSE — while the executor then executed
        the SHORT 6s later, leaving a real position the bot no longer tracked.
        MIN_FALSE_REPORTS=6 (15s) must outlast the 10s tick: a few early False
        reports followed by a confirmed position → True, no rollback."""
        strategy, *_ = _make_strategy()
        strategy._executor_has_position = AsyncMock(
            side_effect=[False, False, False, False, True]
        )
        with patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_ATTEMPTS", 10), \
             patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_MIN_FALSE_REPORTS", 6), \
             patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_DELAY", 0.001):
            assert await strategy.confirm_entry_with_executor("BTC/USDC") is True

    @pytest.mark.asyncio
    async def test_confirm_false_when_executor_really_blocked(self):
        """True block: position never appears within the 15s window → rollback."""
        strategy, *_ = _make_strategy()
        strategy._executor_has_position = AsyncMock(return_value=False)
        with patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_ATTEMPTS", 10), \
             patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_MIN_FALSE_REPORTS", 6), \
             patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_DELAY", 0.001):
            assert await strategy.confirm_entry_with_executor("BTC/USDC") is False
        assert strategy._executor_has_position.await_count == 6

    @pytest.mark.asyncio
    async def test_confirm_fail_open_on_query_errors(self):
        strategy, *_ = _make_strategy()
        strategy._executor_has_position = AsyncMock(return_value=None)
        with patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_ATTEMPTS", 2), \
             patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_DELAY", 0.001):
            assert await strategy.confirm_entry_with_executor("BTC/USDC") is True

    @pytest.mark.asyncio
    async def test_confirm_true_after_transient_errors_then_confirmed(self):
        strategy, *_ = _make_strategy()
        strategy._executor_has_position = AsyncMock(side_effect=[None, None, True])
        with patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_ATTEMPTS", 5), \
             patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_DELAY", 0.001):
            assert await strategy.confirm_entry_with_executor("BTC/USDC") is True

    @pytest.mark.asyncio
    async def test_rollback_noop_without_position(self):
        strategy, _, persistence, *_ = _make_strategy(current_position=None)
        await strategy.rollback_blocked_entry("BTC/USDC", forward_delivered=True)
        persistence.async_save_position.assert_not_called()
        persistence.async_save_trade_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_noop_when_forward_undelivered(self):
        """File-fallback delivery may execute later — never roll back."""
        strategy, _, persistence, *_ = _make_strategy(current_position=_make_position())
        strategy._executor_has_position = AsyncMock(return_value=False)
        with patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_ATTEMPTS", 2), \
             patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_DELAY", 0.001):
            await strategy.rollback_blocked_entry("BTC/USDC", forward_delivered=False)
        strategy._executor_has_position.assert_not_called()
        persistence.async_save_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_keeps_position_when_executor_confirms(self):
        pos = _make_position()
        strategy, _, persistence, *_ = _make_strategy(current_position=pos)
        strategy._executor_has_position = AsyncMock(return_value=True)
        with patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_ATTEMPTS", 2), \
             patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_DELAY", 0.001):
            await strategy.rollback_blocked_entry("BTC/USDC", forward_delivered=True)
        assert strategy.current_position is pos
        persistence.async_save_position.assert_not_called()
        persistence.async_save_trade_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_rollback_blocked_clears_position_and_records_close(self):
        pos = _make_position()
        strategy, logger, persistence, *_ = _make_strategy(current_position=pos)
        strategy._executor_has_position = AsyncMock(return_value=False)
        with patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_ATTEMPTS", 2), \
             patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_MIN_FALSE_REPORTS", 2), \
             patch("src.trading.executor_reconciliation.ENTRY_CONFIRM_DELAY", 0.001):
            await strategy.rollback_blocked_entry("BTC/USDC", forward_delivered=True)
        assert strategy.current_position is None
        persistence.async_save_position.assert_awaited_once_with(None)
        decision = persistence.async_save_trade_decision.call_args.args[0]
        assert decision.action == "CLOSE"
        assert decision.symbol == "BTC/USDC"
        assert decision.quantity == pos.size
        assert decision.price == pos.entry_price
        assert decision.fee == 0.0
        assert "blocked" in (decision.reasoning or "").lower()
        rollback_logs = [c for c in logger.warning.call_args_list if "rolled back local phantom" in str(c)]
        assert len(rollback_logs) > 0

