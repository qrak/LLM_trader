"""Comprehensive tests for TradingStrategy friction capture from RiskManager.

Validates the closed-loop feedback pipeline:
  1. RiskManager frictions are captured after calculate_entry_parameters()
  2. Frictions are persisted via vector_memory.store_blocked_trade()
  3. store_blocked_trade receives correct parameters (guard_type, deltas, context)
  4. Blocked entry (R/R minimum) also stores via store_blocked_trade
  5. Failures in friction storage are handled gracefully (non-fatal)
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from src.managers.risk_manager import RiskManager
from src.trading.trading_strategy import TradingStrategy


# ── Fixture builders ─────────────────────────────────────────────


def _make_config(**overrides) -> SimpleNamespace:
    """Minimal config stub with only the attrs TradingStrategy accesses."""
    defaults = {
        "MAX_POSITION_SIZE": 0.10,
        "POSITION_SIZE_FALLBACK_LOW": 0.01,
        "POSITION_SIZE_FALLBACK_MEDIUM": 0.02,
        "POSITION_SIZE_FALLBACK_HIGH": 0.03,
        "TRANSACTION_FEE_PERCENT": 0.001,
        "DEMO_QUOTE_CAPITAL": 10000.0,
        "TIMEFRAME": "4h",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_mock_brain():
    """Create a mock brain_service with minimal needed methods."""
    brain = MagicMock()
    brain.vector_memory = MagicMock()
    brain.vector_memory.store_blocked_trade.return_value = True
    brain.vector_memory.trade_count = 0
    brain.get_dynamic_thresholds.return_value = {"rr_borderline_min": 1.5}
    return brain


def _make_mock_statistics():
    """Create a mock statistics_service."""
    stats = MagicMock()
    stats.get_current_capital.return_value = 10000.0
    return stats


def _make_mock_extractor():
    """Create a mock position_extractor."""
    ext = MagicMock()
    ext.extract_trading_info.return_value = ("BUY", "HIGH", 95.0, 110.0, 0.05, "Strong setup")
    ext.validate_signal.return_value = True
    return ext


def _make_mock_position_factory():
    """Create a mock position_factory that returns a Position."""
    from datetime import datetime, timezone
    from src.trading.data_models import Position

    factory = MagicMock()
    factory.create_position.return_value = Position(
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        size=5.0,
        entry_time=datetime.now(timezone.utc),
        confidence="HIGH",
        direction="LONG",
        symbol="BTC/USDC",
        size_pct=0.05,
    )
    return factory


def _make_strategy(
    *,
    risk_manager=None,
    brain_service=None,
    position_size: float | None = 0.05,
    stop_loss: float | None = 95.0,
    take_profit: float | None = 110.0,
    rr_borderline_min: float = 1.5,
) -> TradingStrategy:
    """Build a TradingStrategy with all dependencies mocked."""
    logger = MagicMock()
    persistence = MagicMock()
    persistence.load_position.return_value = None
    persistence.async_save_trade_decision = AsyncMock()
    persistence.async_save_position = AsyncMock()

    brain = brain_service or _make_mock_brain()
    brain.get_dynamic_thresholds.return_value = {"rr_borderline_min": rr_borderline_min}

    statistics = _make_mock_statistics()
    memory_service = MagicMock()
    risk_mgr = risk_manager or RiskManager(logger=MagicMock(), config=_make_config())
    config = _make_config()

    extractor = _make_mock_extractor()
    extractor.extract_trading_info.return_value = (
        "BUY", "HIGH", stop_loss, take_profit, position_size, "AI reasoning"
    )

    factory = _make_mock_position_factory()

    return TradingStrategy(
        logger=logger,
        persistence=persistence,
        brain_service=brain,
        statistics_service=statistics,
        memory_service=memory_service,
        risk_manager=risk_mgr,
        config=config,
        position_extractor=extractor,
        position_factory=factory,
    )


# ── Friction Capture from RiskManager ────────────────────────────


class TestFrictionCaptureFromRiskManager:
    """Verify frictions from RiskManager are captured and persisted."""

    @pytest.mark.asyncio
    async def test_position_size_friction_persisted_as_blocked_trade(self):
        """When RiskManager clamps position size, it's stored as blocked trade."""
        rm = RiskManager(logger=MagicMock(), config=_make_config(max_position_size=0.02))
        strategy = _make_strategy(risk_manager=rm, position_size=0.30)

        await strategy._open_new_position(
            signal="BUY", confidence="HIGH",
            stop_loss=95.0, take_profit=110.0,
            position_size=0.30, current_price=100.0,
            symbol="BTC/USDC", reasoning="Test",
        )

        brain_vm = strategy.brain_service.vector_memory
        brain_vm.store_blocked_trade.assert_called()
        # First call should contain position_size_clamp
        calls = brain_vm.store_blocked_trade.call_args_list
        size_clamp_calls = [
            c for c in calls
            if c.kwargs.get("guard_type") == "position_size_clamp"
        ]
        assert len(size_clamp_calls) >= 1

    @pytest.mark.asyncio
    async def test_sl_distance_friction_persisted(self):
        """SL distance clamping triggers store_blocked_trade."""
        rm = RiskManager(logger=MagicMock(), config=_make_config())
        strategy = _make_strategy(risk_manager=rm, stop_loss=80.0)

        await strategy._open_new_position(
            signal="BUY", confidence="HIGH",
            stop_loss=80.0, take_profit=110.0,
            position_size=0.05, current_price=100.0,
            symbol="BTC/USDC", reasoning="Test",
        )

        brain_vm = strategy.brain_service.vector_memory
        guard_types = [
            c.kwargs.get("guard_type")
            for c in brain_vm.store_blocked_trade.call_args_list
        ]
        assert "sl_distance_max" in guard_types

    @pytest.mark.asyncio
    async def test_friction_carries_correct_deltas(self):
        """store_blocked_trade receives the correct suggested vs clamped deltas."""
        rm = RiskManager(logger=MagicMock(), config=_make_config(max_position_size=0.05))
        strategy = _make_strategy(risk_manager=rm, position_size=0.30)

        await strategy._open_new_position(
            signal="BUY", confidence="HIGH",
            stop_loss=95.0, take_profit=110.0,
            position_size=0.30, current_price=100.0,
            symbol="BTC/USDC", reasoning="Test",
        )

        call = strategy.brain_service.vector_memory.store_blocked_trade.call_args_list[0]
        kwargs = call.kwargs
        assert kwargs["guard_type"] == "position_size_clamp"
        assert kwargs["direction"] == "N/A"
        assert kwargs["current_price"] == pytest.approx(100.0)
        assert "reasoning_snippet" in kwargs

    @pytest.mark.asyncio
    async def test_multiple_frictions_all_persisted(self):
        """Multiple guards triggered — all are stored."""
        rm = RiskManager(logger=MagicMock(), config=_make_config(max_position_size=0.02))
        strategy = _make_strategy(risk_manager=rm, stop_loss=80.0, position_size=0.30)

        await strategy._open_new_position(
            signal="BUY", confidence="HIGH",
            stop_loss=80.0, take_profit=110.0,
            position_size=0.30, current_price=100.0,
            symbol="BTC/USDC", reasoning="Test",
        )

        call_count = strategy.brain_service.vector_memory.store_blocked_trade.call_count
        assert call_count >= 2  # size_clamp + sl_distance_max


# ── R/R Minimum Guard (blocked entry) ────────────────────────────


class TestRRMinimumGuard:
    """Verify blocked entries due to poor R/R also persist as blocked trades."""

    @pytest.mark.asyncio
    async def test_poor_rr_stores_blocked_trade(self):
        """R/R below brain threshold stores blocked trade before returning HOLD."""
        strategy = _make_strategy(stop_loss=95.0, take_profit=100.0, rr_borderline_min=3.0)
        # SL=5%, TP=0% → R/R near 0, blocked

        decision = await strategy._open_new_position(
            signal="BUY", confidence="HIGH",
            stop_loss=95.0, take_profit=100.0,
            position_size=0.05, current_price=100.0,
            symbol="BTC/USDC", reasoning="Weak setup",
        )

        assert decision.action == "HOLD"
        assert "blocked" in decision.reasoning.lower()

        # Verify store_blocked_trade was called with rr_minimum
        brain_vm = strategy.brain_service.vector_memory
        rr_calls = [
            c for c in brain_vm.store_blocked_trade.call_args_list
            if c.kwargs.get("guard_type") == "rr_minimum"
        ]
        assert len(rr_calls) >= 1
        call = rr_calls[0]
        assert call.kwargs["direction"] == "LONG"
        assert call.kwargs["required_rr"] == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_good_rr_bypasses_blocked_trade(self):
        """R/R above threshold → no rr_minimum block."""
        strategy = _make_strategy(stop_loss=95.0, take_profit=115.0, rr_borderline_min=1.5)
        # SL=5%, TP=15% → R/R = 3.0, well above minimum

        decision = await strategy._open_new_position(
            signal="BUY", confidence="HIGH",
            stop_loss=95.0, take_profit=115.0,
            position_size=0.05, current_price=100.0,
            symbol="BTC/USDC", reasoning="Good setup",
        )

        assert decision.action != "HOLD"
        brain_vm = strategy.brain_service.vector_memory
        rr_blocked = [
            c for c in brain_vm.store_blocked_trade.call_args_list
            if c.kwargs.get("guard_type") == "rr_minimum"
        ]
        assert len(rr_blocked) == 0

    @pytest.mark.asyncio
    async def test_rr_blocked_trade_includes_reasoning_snippet(self):
        """Blocked trade stores the AI reasoning snippet for LLM feedback."""
        strategy = _make_strategy(stop_loss=95.0, take_profit=100.0, rr_borderline_min=10.0)

        await strategy._open_new_position(
            signal="BUY", confidence="HIGH",
            stop_loss=95.0, take_profit=100.0,
            position_size=0.05, current_price=100.0,
            symbol="BTC/USDC", reasoning="I think this will rebound strongly from support",
        )

        rr_calls = [
            c for c in strategy.brain_service.vector_memory.store_blocked_trade.call_args_list
            if c.kwargs.get("guard_type") == "rr_minimum"
        ]
        assert len(rr_calls) >= 1
        snippet = rr_calls[0].kwargs["reasoning_snippet"]
        assert "rebound" in snippet


# ── Graceful Degradation ─────────────────────────────────────────


class TestFrictionStorageGracefulDegradation:
    """Friction storage failures must not crash the trading loop."""

    @pytest.mark.asyncio
    async def test_store_blocked_trade_exception_does_not_block_entry(self):
        """store_blocked_trade raising doesn't prevent position from opening."""
        rm = RiskManager(logger=MagicMock(), config=_make_config(max_position_size=0.02))
        strategy = _make_strategy(risk_manager=rm, position_size=0.30)
        strategy.brain_service.vector_memory.store_blocked_trade.side_effect = RuntimeError("DB down")

        # Should NOT raise — should catch and log
        decision = await strategy._open_new_position(
            signal="BUY", confidence="HIGH",
            stop_loss=95.0, take_profit=110.0,
            position_size=0.30, current_price=100.0,
            symbol="BTC/USDC", reasoning="Test",
        )

        # Position should still be opened
        assert decision is not None
        strategy.logger.warning.assert_any_call(
            "Failed to store friction event from RiskManager", exc_info=True
        )

    @pytest.mark.asyncio
    async def test_rr_blocked_trade_exception_does_not_block_hold(self):
        """store_blocked_trade raising during R/R block doesn't crash."""
        strategy = _make_strategy(stop_loss=95.0, take_profit=100.0, rr_borderline_min=10.0)
        strategy.brain_service.vector_memory.store_blocked_trade.side_effect = RuntimeError("DB down")

        decision = await strategy._open_new_position(
            signal="BUY", confidence="HIGH",
            stop_loss=95.0, take_profit=100.0,
            position_size=0.05, current_price=100.0,
            symbol="BTC/USDC", reasoning="Test",
        )

        assert decision.action == "HOLD"  # Still returns HOLD despite storage failure
        strategy.logger.warning.assert_any_call(
            "Failed to store blocked trade event", exc_info=True
        )


# ── Parameter Propagation ────────────────────────────────────────


class TestBlockedTradeParameterPropagation:
    """Verify all params flow correctly from friction to store_blocked_trade."""

    @pytest.mark.asyncio
    async def test_volatility_level_propagated(self):
        """The volatility_level from RiskAssessment reaches store_blocked_trade."""
        rm = RiskManager(logger=MagicMock(), config=_make_config())
        strategy = _make_strategy(risk_manager=rm, stop_loss=80.0,
                                  position_size=0.05)

        await strategy._open_new_position(
            signal="BUY", confidence="HIGH",
            stop_loss=80.0, take_profit=110.0,
            position_size=0.05, current_price=100.0,
            symbol="BTC/USDC", reasoning="Test",
        )

        # Find a friction call
        friction_calls = [
            c for c in strategy.brain_service.vector_memory.store_blocked_trade.call_args_list
            if c.kwargs.get("guard_type") != "rr_minimum"
        ]
        assert len(friction_calls) >= 1
        vol = friction_calls[0].kwargs.get("volatility_level")
        assert vol in ("HIGH", "MEDIUM", "LOW")

    @pytest.mark.asyncio
    async def test_confidence_propagated(self):
        """Confidence level reaches store_blocked_trade."""
        rm = RiskManager(logger=MagicMock(), config=_make_config(max_position_size=0.02))
        strategy = _make_strategy(risk_manager=rm, position_size=0.30)

        await strategy._open_new_position(
            signal="SELL", confidence="LOW",
            stop_loss=105.0, take_profit=90.0,
            position_size=0.30, current_price=100.0,
            symbol="BTC/USDC", reasoning="Bearish thesis",
        )

        friction_calls = [
            c for c in strategy.brain_service.vector_memory.store_blocked_trade.call_args_list
            if c.kwargs.get("guard_type") != "rr_minimum"
        ]
        assert len(friction_calls) >= 1
        assert friction_calls[0].kwargs["confidence"] == "LOW"

    @pytest.mark.asyncio
    async def test_suggested_sl_tp_included(self):
        """Suggested SL/TP values reach store_blocked_trade."""
        rm = RiskManager(logger=MagicMock(), config=_make_config())
        strategy = _make_strategy(risk_manager=rm, stop_loss=80.0)

        await strategy._open_new_position(
            signal="BUY", confidence="HIGH",
            stop_loss=80.0, take_profit=130.0,
            position_size=0.05, current_price=100.0,
            symbol="BTC/USDC", reasoning="Test",
        )

        friction_calls = [
            c for c in strategy.brain_service.vector_memory.store_blocked_trade.call_args_list
            if c.kwargs.get("guard_type") != "rr_minimum"
        ]
        assert len(friction_calls) >= 1
        call = friction_calls[0]
        # suggested_rr should be from the RiskAssessment
        assert "suggested_rr" in call.kwargs
        assert "suggested_sl_pct" in call.kwargs
        assert "suggested_tp_pct" in call.kwargs

    @pytest.mark.asyncio
    async def test_metadata_includes_full_friction_dict(self):
        """The raw friction dict is included as metadata for audit trail."""
        rm = RiskManager(logger=MagicMock(), config=_make_config(max_position_size=0.02))
        strategy = _make_strategy(risk_manager=rm, position_size=0.30)

        await strategy._open_new_position(
            signal="BUY", confidence="HIGH",
            stop_loss=95.0, take_profit=110.0,
            position_size=0.30, current_price=100.0,
            symbol="BTC/USDC", reasoning="Test",
        )

        friction_calls = [
            c for c in strategy.brain_service.vector_memory.store_blocked_trade.call_args_list
            if c.kwargs.get("guard_type") != "rr_minimum"
        ]
        assert len(friction_calls) >= 1
        metadata = friction_calls[0].kwargs.get("metadata", {})
        assert "friction" in metadata
        assert metadata["friction"]["guard_type"] == "position_size_clamp"
