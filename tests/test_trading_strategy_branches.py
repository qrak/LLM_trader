"""Coverage gap filler: trading_strategy.py uncovered branches.

Covers:
  1. _update_position_parameters — SL tightening guard, widening, TP changes
  2. close_position — normal close, brain update, error paths
  3. _extract_price_from_result — all extraction paths
  4. _extract_confluence_factors — valid/invalid/edge scores
  5. _extract_market_conditions — data extraction from analysis result
  6. _build_conditions_from_position — reconstruction from stored fields
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest

from src.dashboard.dashboard_state import DashboardState
from src.managers.risk_manager import RiskManager
from src.trading.data_models import MarketConditions, Position, TradeDecision
from src.trading.stop_loss_tightening_policy import StopLossTighteningPolicy
from src.trading.trading_strategy import TradingStrategy


# ═════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════


def _make_config(**overrides):
    defaults = {
        "MAX_POSITION_SIZE": 0.10,
        "POSITION_SIZE_FALLBACK_LOW": 0.01,
        "POSITION_SIZE_FALLBACK_MEDIUM": 0.02,
        "POSITION_SIZE_FALLBACK_HIGH": 0.03,
        "TRANSACTION_FEE_PERCENT": 0.001,
        "DEMO_QUOTE_CAPITAL": 10000.0,
        "TIMEFRAME": "4h",
        "QUOTE_CURRENCY": "USDT",
        "STOP_LOSS_TYPE": "hard",
        "STOP_LOSS_CHECK_INTERVAL": "15m",
        "TAKE_PROFIT_TYPE": "hard",
        "TAKE_PROFIT_CHECK_INTERVAL": "15m",
        "SL_TIGHTENING_SCALPING": 0.25,
        "SL_TIGHTENING_INTRADAY": 0.20,
        "SL_TIGHTENING_SWING": 0.15,
        "SL_TIGHTENING_POSITION": 0.10,
        "SL_TIGHTENING_FLOOR": 0.05,
        "SL_TIGHTENING_CEILING": 0.40,
        "SL_TIGHTENING_MIN_SAMPLES": 10,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_position(**overrides):
    defaults = {
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "take_profit": 110.0,
        "size": 5.0,
        "entry_time": datetime.now(timezone.utc),
        "confidence": "HIGH",
        "direction": "LONG",
        "symbol": "BTC/USDC",
        "size_pct": 0.05,
        "quote_amount": 500.0,
        "entry_fee": 0.50,
        "adx_at_entry": 35.0,
        "rsi_at_entry": 55.0,
    }
    defaults.update(overrides)
    return Position(**defaults)


def _make_strategy(*, current_position=None, config=None, tightening_policy=None, **overrides):
    """Build a minimal TradingStrategy with all deps mocked."""
    logger = MagicMock()
    persistence = MagicMock()
    persistence.load_position.return_value = current_position
    persistence.async_save_trade_decision = AsyncMock()
    persistence.async_save_position = AsyncMock()
    persistence.get_entry_decision_for_position = MagicMock(return_value=None)

    brain = MagicMock()
    brain.vector_memory = MagicMock()
    brain.vector_memory.store_blocked_trade.return_value = True
    brain.vector_memory.trade_count = 0
    brain.get_dynamic_thresholds.return_value = {"rr_borderline_min": 1.5}

    stats = MagicMock()
    stats.get_current_capital.return_value = 10000.0

    memory_svc = MagicMock()
    risk_mgr = MagicMock()
    risk_mgr.get_and_clear_frictions = MagicMock(return_value=[])
    risk_mgr.calculate_entry_parameters = MagicMock()

    extractor = MagicMock()
    factory = MagicMock()
    factory.create_updated_position = MagicMock(return_value=_make_position())
    factory.create_position = MagicMock(return_value=_make_position())

    cfg = config or _make_config()
    cfg = SimpleNamespace(**{**cfg.__dict__, **overrides}) if overrides else cfg

    # Patch _tf_minutes to be set from TIMEFRAME
    with patch('src.utils.timeframe_validator.TimeframeValidator') as mock_tfv:
        mock_tfv.to_minutes.return_value = 240
        strategy = TradingStrategy(
            logger=logger,
            persistence=persistence,
            brain_service=brain,
            statistics_service=stats,
            memory_service=memory_svc,
            risk_manager=risk_mgr,
            config=cfg,
            position_extractor=extractor,
            position_factory=factory,
            tightening_policy=tightening_policy,
        )

    # Override current_position if needed
    if current_position is not None:
        strategy.current_position = current_position

    return strategy, logger, persistence, brain, stats, factory


class TestUpdatePositionParameters:
    """Cover _update_position_parameters (lines 533-629)."""

    def test_no_position_returns_false(self):
        """When no position exists, no update is performed."""
        strategy, _, _, _, _, factory = _make_strategy(current_position=None)
        strategy.current_position = None

        updated = asyncio.run(strategy._update_position_parameters(
            stop_loss=90.0, take_profit=120.0, current_price=105.0,
        ))
        assert updated is False
        factory.create_updated_position.assert_not_called()

    def test_same_sl_returns_no_update(self):
        """When SL is unchanged, only TP change triggers update."""
        pos = _make_position(stop_loss=95.0, take_profit=110.0)
        strategy, _, _, _, _, factory = _make_strategy(current_position=pos)

        updated = asyncio.run(strategy._update_position_parameters(
            stop_loss=95.0, take_profit=110.0, current_price=105.0,
        ))
        assert updated is False

    def test_sl_tightening_rejected_when_progress_too_low(self):
        """Premature SL tightening is blocked by progress guard."""
        pos = _make_position(stop_loss=95.0, take_profit=110.0, direction="LONG")
        policy = StopLossTighteningPolicy(swing_threshold=0.20)
        strategy, _, _, _, _, factory = _make_strategy(current_position=pos, tightening_policy=policy)

        updated = asyncio.run(strategy._update_position_parameters(
            stop_loss=99.0, take_profit=None, current_price=101.5,
        ))
        # progress = (101.5 - 100) / (110 - 100) = 1.5/10 = 0.15 < 0.20
        assert updated is False
        factory.create_updated_position.assert_not_called()

    def test_sl_tightening_allowed_when_progress_sufficient(self):
        """SL tightening passes when price progress exceeds threshold."""
        pos = _make_position(stop_loss=95.0, take_profit=110.0, direction="LONG")
        policy = StopLossTighteningPolicy(swing_threshold=0.15)
        strategy, _, persistence, _, _, factory = _make_strategy(current_position=pos, tightening_policy=policy)
        factory.create_updated_position.return_value = _make_position(stop_loss=99.0)

        updated = asyncio.run(strategy._update_position_parameters(
            stop_loss=99.0, take_profit=None, current_price=102.0,
        ))
        # progress = (102 - 100) / (110 - 100) = 2/10 = 0.20 >= 0.15
        assert updated is True
        factory.create_updated_position.assert_called_once()
        persistence.async_save_position.assert_called_once()

    def test_sl_widening_long(self):
        """LONG SL moved lower (wider) is allowed."""
        pos = _make_position(stop_loss=95.0, take_profit=110.0, direction="LONG")
        strategy, _, persistence, _, _, factory = _make_strategy(current_position=pos)
        factory.create_updated_position.return_value = _make_position(stop_loss=92.0)

        updated = asyncio.run(strategy._update_position_parameters(
            stop_loss=92.0, take_profit=None, current_price=105.0,
        ))
        assert updated is True
        factory.create_updated_position.assert_called_once()

    def test_sl_widening_short(self):
        """SHORT SL moved higher (wider) is allowed."""
        pos = _make_position(stop_loss=105.0, take_profit=90.0, direction="SHORT")
        strategy, _, persistence, _, _, factory = _make_strategy(current_position=pos)
        factory.create_updated_position.return_value = _make_position(stop_loss=108.0)

        updated = asyncio.run(strategy._update_position_parameters(
            stop_loss=108.0, take_profit=None, current_price=100.0,
        ))
        assert updated is True

    def test_sl_tightening_short_rejected(self):
        """SHORT SL tightening (moving SL lower) blocked by progress guard."""
        pos = _make_position(stop_loss=105.0, take_profit=90.0, direction="SHORT")
        policy = StopLossTighteningPolicy(swing_threshold=0.20)
        strategy, _, _, _, _, factory = _make_strategy(current_position=pos, tightening_policy=policy)
        # For SHORT: tightening means stop_loss < old_sl (moving closer to entry)
        # price_progress = (entry_price - current_price) / tp_distance_total
        # tp_distance_total = |90 - 100| = 10
        # progress = (100 - 99) / 10 = 1/10 = 0.10 < 0.20

        updated = asyncio.run(strategy._update_position_parameters(
            stop_loss=103.0, take_profit=None, current_price=99.0,
        ))
        assert updated is False

    def test_tp_change_only(self):
        """Changing only take_profit works."""
        pos = _make_position(stop_loss=95.0, take_profit=110.0)
        strategy, _, persistence, _, _, factory = _make_strategy(current_position=pos)
        factory.create_updated_position.return_value = _make_position(take_profit=115.0)

        updated = asyncio.run(strategy._update_position_parameters(
            stop_loss=None, take_profit=115.0, current_price=105.0,
        ))
        assert updated is True
        factory.create_updated_position.assert_called_once()

    def test_both_sl_and_tp_changed(self):
        """Both SL and TP updated simultaneously."""
        pos = _make_position(stop_loss=95.0, take_profit=110.0)
        strategy, _, persistence, _, _, factory = _make_strategy(current_position=pos)
        factory.create_updated_position.return_value = _make_position(
            stop_loss=93.0, take_profit=115.0,
        )

        updated = asyncio.run(strategy._update_position_parameters(
            stop_loss=93.0, take_profit=115.0, current_price=105.0,
        ))
        assert updated is True

    def test_sl_tightening_zero_tp_distance(self):
        """Edge case: tp_distance_total is 0 (TP=entry) — progress=0."""
        pos = _make_position(stop_loss=95.0, take_profit=100.0, entry_price=100.0, direction="LONG")
        policy = StopLossTighteningPolicy(swing_threshold=0.20)
        strategy, _, _, _, _, factory = _make_strategy(current_position=pos, tightening_policy=policy)

        updated = asyncio.run(strategy._update_position_parameters(
            stop_loss=99.0, take_profit=None, current_price=102.0,
        ))
        # tp_distance_total = 0 → policy returns allowed=False
        assert updated is False

    def test_sl_tightening_no_current_price_rejected(self):
        """Tightening without a valid current price is rejected for safety."""
        pos = _make_position(stop_loss=95.0, take_profit=110.0, direction="LONG")
        strategy, _, _, _, _, factory = _make_strategy(current_position=pos)

        # LONG: stop_loss=96.0 > old_sl=95.0 is tightening.
        # Policy rejects when current_price is None/0 as a safety measure.
        updated = asyncio.run(strategy._update_position_parameters(
            stop_loss=96.0, take_profit=None, current_price=None,
        ))
        assert updated is False
        factory.create_updated_position.assert_not_called()


# ═════════════════════════════════════════════════════════════════
# SECTION 2: close_position
# ═════════════════════════════════════════════════════════════════


class TestClosePosition:
    """Cover close_position (lines 152-227)."""

    def test_no_position_returns_early(self):
        """When no position exists, nothing happens."""
        strategy, _, persistence, brain, stats, _ = _make_strategy(current_position=None)
        strategy.current_position = None

        asyncio.run(strategy.close_position("stop_loss", 90.0))
        brain.update_from_closed_trade.assert_not_called()
        stats.recalculate.assert_not_called()

    def test_close_position_stop_loss(self):
        """Normal stop-loss close saves decision, updates brain, recalculates stats."""
        pos = _make_position(direction="LONG")
        strategy, _, persistence, brain, stats, _ = _make_strategy(current_position=pos)

        asyncio.run(strategy.close_position("stop_loss", 90.0))

        # Verify decision saved
        persistence.async_save_trade_decision.assert_called_once()
        decision = persistence.async_save_trade_decision.call_args[0][0]
        assert decision.action == "CLOSE_LONG"
        assert "stop_loss" in decision.reasoning
        cast(MagicMock, strategy.memory_service.add_decision).assert_called_once_with(decision)

        # Verify brain updated
        brain.update_from_closed_trade.assert_called_once()

        # Verify stats recalculated
        stats.recalculate.assert_called_once_with(10000.0)

        # Verify position cleared
        persistence.async_save_position.assert_called_with(None)
        assert strategy.current_position is None

    def test_close_position_take_profit(self):
        """Take-profit close flow."""
        pos = _make_position(direction="SHORT")
        strategy, _, persistence, brain, stats, _ = _make_strategy(current_position=pos)

        asyncio.run(strategy.close_position("take_profit", 85.0))

        decision = persistence.async_save_trade_decision.call_args[0][0]
        assert decision.action == "CLOSE_SHORT"
        assert "take_profit" in decision.reasoning

    def test_close_position_with_market_conditions(self):
        """Market conditions are passed to brain on close."""
        pos = _make_position()
        strategy, _, _, brain, _, _ = _make_strategy(current_position=pos)

        conditions = MarketConditions(trend_direction="BULLISH", adx=40)
        asyncio.run(strategy.close_position("analysis_signal", 105.0, market_conditions=conditions))

        call_kwargs = brain.update_from_closed_trade.call_args[1]
        assert call_kwargs["market_conditions"] == conditions

    def test_close_position_notifies_dashboard_lifecycle(self):
        """Dashboard lifecycle is marked around close-time brain learning."""
        pos = _make_position(direction="LONG")
        strategy, _, _, _, _, _ = _make_strategy(current_position=pos)
        dashboard_state = DashboardState()
        dashboard_state.mark_brain_rebuild_started = AsyncMock()
        dashboard_state.mark_brain_rebuild_completed = AsyncMock()
        dashboard_state.mark_brain_rebuild_failed = AsyncMock()
        strategy.set_dashboard_state(dashboard_state)

        asyncio.run(strategy.close_position("take_profit", 110.0))

        dashboard_state.mark_brain_rebuild_started.assert_awaited_once_with(
            "Learning from closed LONG trade"
        )
        dashboard_state.mark_brain_rebuild_completed.assert_awaited_once_with(
            "Brain state rebuilt from closed trade"
        )
        dashboard_state.mark_brain_rebuild_failed.assert_not_awaited()

    def test_close_position_brain_update_failure_non_fatal(self):
        """Brain update error doesn't prevent position close."""
        pos = _make_position()
        strategy, _, persistence, brain, stats, _ = _make_strategy(current_position=pos)
        dashboard_state = DashboardState()
        dashboard_state.mark_brain_rebuild_started = AsyncMock()
        dashboard_state.mark_brain_rebuild_completed = AsyncMock()
        dashboard_state.mark_brain_rebuild_failed = AsyncMock()
        strategy.set_dashboard_state(dashboard_state)
        brain.update_from_closed_trade.side_effect = RuntimeError("Brain crash")

        # Should not raise
        asyncio.run(strategy.close_position("stop_loss", 90.0))

        # Position still cleared
        assert strategy.current_position is None
        persistence.async_save_position.assert_called_with(None)
        dashboard_state.mark_brain_rebuild_started.assert_awaited_once()
        dashboard_state.mark_brain_rebuild_completed.assert_not_awaited()
        dashboard_state.mark_brain_rebuild_failed.assert_awaited_once_with(
            "Brain rebuild failed after trade close"
        )

    def test_close_position_stats_recalculate_failure_non_fatal(self):
        """Stats recalculate error doesn't prevent position close."""
        pos = _make_position()
        strategy, _, persistence, brain, stats, _ = _make_strategy(current_position=pos)
        stats.recalculate.side_effect = RuntimeError("Stats crash")

        asyncio.run(strategy.close_position("stop_loss", 90.0))
        assert strategy.current_position is None

    def test_close_position_calculates_correct_pnl_long(self):
        """Long position closing with profit calculates P&L correctly."""
        pos = _make_position(entry_price=100.0, direction="LONG")
        strategy, _, persistence, _, _, _ = _make_strategy(current_position=pos)

        asyncio.run(strategy.close_position("take_profit", 115.0))
        decision = persistence.async_save_trade_decision.call_args[0][0]
        assert "P&L: +15.00%" in decision.reasoning

    def test_open_new_position_records_decision_in_memory(self):
        """Opening a position persists the entry and refreshes short-term memory."""
        strategy, _, persistence, _, _, _ = _make_strategy(current_position=None)
        risk_assessment = SimpleNamespace(
            stop_loss=95.0,
            take_profit=112.0,
            size_pct=0.05,
            quantity=5.0,
            entry_fee=0.5,
            sl_distance_pct=0.05,
            tp_distance_pct=0.12,
            rr_ratio=2.4,
            quote_amount=500.0,
            volatility_level="MEDIUM",
        )
        cast(MagicMock, strategy.risk_manager.calculate_entry_parameters).return_value = risk_assessment

        decision = asyncio.run(strategy._open_new_position(
            signal="BUY",
            confidence="HIGH",
            stop_loss=95.0,
            take_profit=112.0,
            position_size=0.05,
            current_price=100.0,
            symbol="BTC/USDC",
            reasoning="Breakout continuation",
            market_conditions=MarketConditions(adx=30.0),
        ))

        persistence.async_save_trade_decision.assert_called_once_with(decision)
        cast(MagicMock, strategy.memory_service.add_decision).assert_called_once_with(decision)

    def test_close_position_calculates_correct_pnl_short(self):
        """Short position closing with profit calculates P&L correctly."""
        pos = _make_position(entry_price=100.0, direction="SHORT")
        strategy, _, persistence, _, _, _ = _make_strategy(current_position=pos)

        asyncio.run(strategy.close_position("take_profit", 85.0))
        decision = persistence.async_save_trade_decision.call_args[0][0]
        assert "P&L: +15.00%" in decision.reasoning


# ═════════════════════════════════════════════════════════════════
# SECTION 3: _extract_price_from_result
# ═════════════════════════════════════════════════════════════════


class TestExtractPriceFromResult:
    """Cover _extract_price_from_result (lines 631-650)."""

    def test_price_from_current_price_key(self):
        """Extract from result['current_price']."""
        strategy, _, _, _, _, _ = _make_strategy(current_position=_make_position())
        result = {"current_price": 50000.0}
        price = strategy._extract_price_from_result(result)
        assert price == 50000.0

    def test_price_from_context_object(self):
        """Extract from result['context'].current_price when no 'current_price' key."""
        strategy, _, _, _, _, _ = _make_strategy(current_position=_make_position())
        ctx = MagicMock()
        ctx.current_price = 42500.0
        result = {"context": ctx}
        price = strategy._extract_price_from_result(result)
        assert price == 42500.0

    def test_price_fallback_returns_zero(self):
        """When neither key exists, returns 0.0 with warning."""
        strategy, _, _, _, _, _ = _make_strategy(current_position=_make_position())
        result = {"some_other_key": "value"}
        price = strategy._extract_price_from_result(result)
        assert price == 0.0

    def test_price_context_is_none(self):
        """context key present but value is None — falls to default."""
        strategy, _, _, _, _, _ = _make_strategy(current_position=_make_position())
        result = {"context": None}
        price = strategy._extract_price_from_result(result)
        assert price == 0.0


# ═════════════════════════════════════════════════════════════════
# SECTION 4: _extract_confluence_factors
# ═════════════════════════════════════════════════════════════════


class TestExtractConfluenceFactors:
    """Cover _extract_confluence_factors (lines 770-795)."""

    def test_extract_valid_factors(self):
        """Valid numeric factors extracted as (name, score) tuples."""
        strategy, _, _, _, _, _ = _make_strategy(current_position=_make_position())
        result = {
            "analysis": {
                "confluence_factors": {
                    "trend_alignment": 85,
                    "volume_support": 70,
                    "pattern_quality": 60,
                }
            }
        }
        factors = strategy._extract_confluence_factors(result)
        assert len(factors) == 3
        assert ("trend_alignment", 85.0) in factors

    def test_extract_empty_factors(self):
        """Empty confluence_factors returns empty tuple."""
        strategy, _, _, _, _, _ = _make_strategy(current_position=_make_position())
        result = {"analysis": {"confluence_factors": {}}}
        factors = strategy._extract_confluence_factors(result)
        assert factors == ()

    def test_no_analysis_key(self):
        """Missing 'analysis' key returns empty tuple."""
        strategy, _, _, _, _, _ = _make_strategy(current_position=_make_position())
        result = {"other": "data"}
        factors = strategy._extract_confluence_factors(result)
        assert factors == ()

    def test_factors_out_of_range_excluded(self):
        """Scores outside 0-100 range are silently excluded."""
        strategy, _, _, _, _, _ = _make_strategy(current_position=_make_position())
        result = {
            "analysis": {
                "confluence_factors": {
                    "valid": 50,
                    "too_high": 150,
                    "negative": -10,
                }
            }
        }
        factors = strategy._extract_confluence_factors(result)
        assert len(factors) == 1
        assert factors[0] == ("valid", 50.0)

    def test_non_numeric_scores_skipped(self):
        """Non-numeric scores (strings, etc.) are silently skipped."""
        strategy, _, _, _, _, _ = _make_strategy(current_position=_make_position())
        result = {
            "analysis": {
                "confluence_factors": {
                    "valid": 42,
                    "bad_type": "high",
                    "also_bad": None,
                }
            }
        }
        factors = strategy._extract_confluence_factors(result)
        assert factors == (("valid", 42.0),)


# ═════════════════════════════════════════════════════════════════
# SECTION 5: _build_conditions_from_position
# ═════════════════════════════════════════════════════════════════


class TestBuildConditionsFromPosition:
    """Cover _build_conditions_from_position (lines 652-672)."""

    def test_build_conditions_from_position_defaults(self):
        """Default position values map to classified conditions."""
        pos = _make_position()
        conditions = TradingStrategy._build_conditions_from_position(pos)

        assert conditions.trend_direction == "NEUTRAL"
        assert conditions.adx == 35.0
        assert conditions.rsi == 55.0
        assert conditions.rsi_level is not None
        assert conditions.volatility == "MEDIUM"

    def test_build_conditions_bullish_position(self):
        """Position with BULLISH trend maps correctly."""
        pos = _make_position(trend_direction_at_entry="BULLISH", adx_at_entry=45.0)
        conditions = TradingStrategy._build_conditions_from_position(pos)
        assert conditions.trend_direction == "BULLISH"
        assert conditions.adx == 45.0


# ═════════════════════════════════════════════════════════════════
# SECTION 6: _extract_market_conditions
# ═════════════════════════════════════════════════════════════════


class TestExtractMarketConditions:
    """Cover _extract_market_conditions (lines 674-768)."""

    def test_full_market_conditions_extraction(self):
        """All fields populated from analysis result."""
        strategy, _, _, _, _, _ = _make_strategy(current_position=_make_position())
        result = {
            "analysis": {
                "trend": {
                    "direction": "BULLISH",
                    "strength_4h": 45,
                    "timeframe_alignment": "ALIGNED",
                },
            },
            "technical_data": {
                "adx": 42.0,
                "rsi": 65.0,
                "atr": 150.0,
                "atr_percent": 2.5,
                "bollinger_bands": {"percent_b": 0.75},
            },
            "raw_response": "BUY signal, bullish pattern detected",
        }

        conditions = strategy._extract_market_conditions(result)
        assert conditions.trend_direction == "BULLISH"
        assert conditions.trend_strength == 45
        assert conditions.timeframe_alignment == "ALIGNED"
        assert conditions.adx == 42.0
        assert conditions.rsi == 65.0

    def test_market_conditions_with_rsi_as_string(self):
        """RSI value provided as string (e.g., from JSON) is converted."""
        strategy, _, _, _, _, _ = _make_strategy(current_position=_make_position())
        result = {
            "analysis": {"trend": {}},
            "technical_data": {"adx": 30, "rsi": "72.5"},
        }
        conditions = strategy._extract_market_conditions(result)
        assert conditions.rsi == 72.5

    def test_market_conditions_empty_result(self):
        """Empty result returns default conditions (sentiment + fallback trend)."""
        strategy, _, _, _, _, _ = _make_strategy(current_position=_make_position())
        conditions = strategy._extract_market_conditions({})
        # Defaults: NEUTRAL trend from raw_response fallback + sentiment defaults
        assert conditions.trend_direction == "NEUTRAL"
        assert conditions.fear_greed_index == 50
        assert conditions.market_sentiment == "NEUTRAL"

    def test_market_conditions_fallback_from_raw_response(self):
        """When trend_direction is empty, extract from raw_response."""
        strategy, _, _, _, _, _ = _make_strategy(current_position=_make_position())
        result = {
            "analysis": {"trend": {}},
            "technical_data": {},
            "raw_response": "Strong BULLISH momentum expected",
        }
        conditions = strategy._extract_market_conditions(result)
        assert conditions.trend_direction == "BULLISH"

    def test_market_conditions_bearish_fallback(self):
        """Bearish keywords in raw_response determine direction."""
        strategy, _, _, _, _, _ = _make_strategy(current_position=_make_position())
        result = {
            "analysis": {"trend": {}},
            "technical_data": {},
            "raw_response": "Downtrend likely to continue, BEARISH outlook",
        }
        conditions = strategy._extract_market_conditions(result)
        assert conditions.trend_direction == "BEARISH"

    def test_market_conditions_exception_handling(self):
        """Exception during extraction returns empty dict."""
        strategy, _, _, _, _, _ = _make_strategy(current_position=_make_position())
        result = {"analysis": None}  # .get() on None will fail
        conditions = strategy._extract_market_conditions(result)
        assert conditions == MarketConditions()
