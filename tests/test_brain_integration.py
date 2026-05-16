"""Tests for brain.py changes: classify_adx_label integration in _build_rich_context_string."""
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch
import pytest

from src.trading.data_models import ExitExecutionContext, MarketConditions, Position, TradeDecision
from src.trading.brain import TradingBrainService


def _make_brain(exit_execution_context=None, timeframe_minutes=240):
    """Create a TradingBrainService with mocked dependencies."""
    logger = MagicMock()
    persistence = MagicMock()
    vector_memory = MagicMock()
    vector_memory.trade_count = 0
    vector_memory.get_relevant_rules.return_value = []
    return TradingBrainService(
        logger=logger,
        persistence=persistence,
        vector_memory=vector_memory,
        exit_execution_context=exit_execution_context,
        timeframe_minutes=timeframe_minutes,
    )


def _make_position(**overrides):
    defaults = dict(
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        size=1.0,
        entry_time=datetime(2026, 4, 30, tzinfo=timezone.utc),
        confidence="HIGH",
        direction="LONG",
        symbol="BTC/USDC",
    )
    defaults.update(overrides)
    return Position(**defaults)


# ── _build_rich_context_string ──────────────────────────────────


class TestBuildRichContextString:
    """Verify _build_rich_context_string uses classify_adx_label."""

    def setup_method(self):
        self.brain = _make_brain()

    def test_high_adx_label(self):
        ctx = self.brain._build_rich_context_string(adx=30)
        assert "High ADX" in ctx

    def test_low_adx_label(self):
        ctx = self.brain._build_rich_context_string(adx=10)
        assert "Low ADX" in ctx

    def test_medium_adx_label(self):
        ctx = self.brain._build_rich_context_string(adx=22)
        assert "Medium ADX" in ctx

    def test_contains_trend_direction(self):
        ctx = self.brain._build_rich_context_string(trend_direction="BULLISH", adx=25)
        assert "BULLISH" in ctx

    def test_contains_volatility(self):
        ctx = self.brain._build_rich_context_string(adx=25, volatility_level="HIGH")
        assert "HIGH Volatility" in ctx

    def test_rsi_included_when_not_neutral(self):
        ctx = self.brain._build_rich_context_string(adx=25, rsi_level="OVERBOUGHT")
        assert "RSI OVERBOUGHT" in ctx

    def test_rsi_excluded_when_neutral(self):
        ctx = self.brain._build_rich_context_string(adx=25, rsi_level="NEUTRAL")
        assert "RSI" not in ctx

    def test_weekend_flag(self):
        ctx = self.brain._build_rich_context_string(adx=25, is_weekend=True)
        assert "Weekend Low Volume" in ctx

    def test_separator_is_plus(self):
        ctx = self.brain._build_rich_context_string(adx=25)
        assert " + " in ctx

    def test_macd_included_when_not_neutral(self):
        ctx = self.brain._build_rich_context_string(adx=25, macd_signal="BULLISH")
        assert "MACD BULLISH" in ctx

    def test_volume_included_when_not_normal(self):
        ctx = self.brain._build_rich_context_string(adx=25, volume_state="ACCUMULATION")
        assert "Volume ACCUMULATION" in ctx

    def test_bb_included_when_not_middle(self):
        ctx = self.brain._build_rich_context_string(adx=25, bb_position="UPPER")
        assert "Price at BB UPPER" in ctx

    def test_sentiment_included(self):
        ctx = self.brain._build_rich_context_string(adx=25, market_sentiment="EXTREME_FEAR")
        assert "Sentiment EXTREME_FEAR" in ctx

    def test_order_book_included(self):
        ctx = self.brain._build_rich_context_string(adx=25, order_book_bias="BUY_PRESSURE")
        assert "OrderBook BUY_PRESSURE" in ctx

    def test_exit_execution_included(self):
        ctx = self.brain._build_rich_context_string(
            adx=25,
            exit_execution_context=ExitExecutionContext(
                stop_loss_type="hard",
                stop_loss_check_interval="15m",
                take_profit_type="hard",
                take_profit_check_interval="15m",
            ),
        )
        assert "Exit Execution: SL hard/15m | TP hard/15m" in ctx


class TestUpdateFromClosedTrade:
    """Verify closed trades persist exit execution settings into vector memory."""

    def setup_method(self):
        self.brain = _make_brain()

    def test_closed_trade_stores_exit_execution_metadata_and_context(self):
        position = Position(
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            size=1.0,
            entry_time=datetime(2026, 4, 30, tzinfo=timezone.utc),
            confidence="HIGH",
            direction="LONG",
            symbol="BTC/USDC",
            stop_loss_type_at_entry="hard",
            stop_loss_check_interval_at_entry="15m",
            take_profit_type_at_entry="soft",
            take_profit_check_interval_at_entry="4h",
        )

        self.brain.update_from_closed_trade(
            position=position,
            close_price=110.0,
            close_reason="take_profit",
            market_conditions=MarketConditions(adx=30.0, trend_direction="BULLISH"),
        )

        call_kwargs = self.brain.vector_memory.store_experience.call_args.kwargs
        assert "Exit Execution: SL hard/15m | TP soft/4h" in call_kwargs["market_context"]
        assert call_kwargs["metadata"]["stop_loss_type"] == "hard"
        assert call_kwargs["metadata"]["stop_loss_check_interval"] == "15m"
        assert call_kwargs["metadata"]["take_profit_type"] == "soft"
        assert call_kwargs["metadata"]["take_profit_check_interval"] == "4h"

    def test_closed_trade_fills_unknown_exit_execution_from_configured_default(self):
        brain = _make_brain(ExitExecutionContext(
            stop_loss_type="hard",
            stop_loss_check_interval="15m",
            take_profit_type="hard",
            take_profit_check_interval="15m",
        ))
        position = _make_position()

        brain.update_from_closed_trade(
            position=position,
            close_price=110.0,
            close_reason="take_profit",
            market_conditions=MarketConditions(adx=30.0, trend_direction="BULLISH"),
        )

        call_kwargs = brain.vector_memory.store_experience.call_args.kwargs
        assert "Exit Execution: SL hard/15m | TP hard/15m" in call_kwargs["market_context"]
        assert call_kwargs["metadata"]["stop_loss_type"] == "hard"
        assert call_kwargs["metadata"]["stop_loss_check_interval"] == "15m"
        assert call_kwargs["metadata"]["take_profit_type"] == "hard"
        assert call_kwargs["metadata"]["take_profit_check_interval"] == "15m"

    def test_closed_trade_stores_original_ai_decision_snapshot(self):
        position = Position(
            entry_price=100.0,
            stop_loss=96.0,
            take_profit=108.0,
            size=1.0,
            entry_time=datetime(2026, 4, 30, tzinfo=timezone.utc),
            confidence="MEDIUM",
            direction="LONG",
            symbol="BTC/USDC",
        )
        entry_decision = TradeDecision(
            timestamp=datetime(2026, 4, 30, tzinfo=timezone.utc),
            symbol="BTC/USDC",
            action="BUY",
            confidence="HIGH",
            price=100.0,
            reasoning="Strong breakout momentum is likely to continue.",
        )

        self.brain.update_from_closed_trade(
            position=position,
            close_price=98.0,
            close_reason="sideways",
            entry_decision=entry_decision,
            market_conditions=MarketConditions(adx=16.0, trend_direction="NEUTRAL"),
        )

        call_kwargs = self.brain.vector_memory.store_experience.call_args.kwargs
        assert call_kwargs["confidence"] == "HIGH"
        assert call_kwargs["reasoning"] == "Strong breakout momentum is likely to continue."
        assert call_kwargs["metadata"]["entry_action"] == "BUY"
        assert call_kwargs["metadata"]["entry_confidence"] == "HIGH"
        assert call_kwargs["metadata"]["ai_reasoning"] == "Strong breakout momentum is likely to continue."

    @pytest.mark.parametrize(("timeframe_minutes", "expected_interval"), [
        (5, 10),
        (15, 10),
        (60, 7),
        (120, 7),
        (240, 5),
        (720, 5),
        (1440, 3),
        (10080, 3),
        (0, 5),
        (-1, 5),
        ("invalid", 5),
    ])
    def test_reflection_interval_derives_from_timeframe(self, timeframe_minutes, expected_interval):
        assert TradingBrainService._derive_reflection_interval(timeframe_minutes) == expected_interval
        assert _make_brain(timeframe_minutes=timeframe_minutes)._reflection_interval == expected_interval

    def test_reflection_runs_on_default_four_hour_interval(self):
        self.brain._trade_count = 4
        self.brain._trigger_reflection = MagicMock()
        self.brain._trigger_loss_reflection = MagicMock()
        self.brain._trigger_ai_mistake_reflection = MagicMock()

        self.brain.update_from_closed_trade(
            position=_make_position(),
            close_price=105.0,
            close_reason="take_profit",
            market_conditions=MarketConditions(adx=30.0, trend_direction="BULLISH"),
        )

        self.brain._trigger_reflection.assert_called_once()
        self.brain._trigger_loss_reflection.assert_called_once()
        self.brain._trigger_ai_mistake_reflection.assert_called_once()

    def test_reflection_uses_timeframe_derived_closed_trade_interval(self):
        brain = _make_brain(timeframe_minutes=60)
        brain._trade_count = 4
        brain._trigger_reflection = MagicMock()
        brain._trigger_loss_reflection = MagicMock()
        brain._trigger_ai_mistake_reflection = MagicMock()

        brain.update_from_closed_trade(
            position=_make_position(),
            close_price=105.0,
            close_reason="take_profit",
            market_conditions=MarketConditions(adx=30.0, trend_direction="BULLISH"),
        )

        brain._trigger_reflection.assert_not_called()
        brain._trigger_loss_reflection.assert_not_called()
        brain._trigger_ai_mistake_reflection.assert_not_called()

        brain._trade_count = 6
        brain.update_from_closed_trade(
            position=_make_position(entry_time=datetime(2026, 5, 1, tzinfo=timezone.utc)),
            close_price=105.0,
            close_reason="take_profit",
            market_conditions=MarketConditions(adx=30.0, trend_direction="BULLISH"),
        )

        brain._trigger_reflection.assert_called_once()
        brain._trigger_loss_reflection.assert_called_once()
        brain._trigger_ai_mistake_reflection.assert_called_once()


# ── get_vector_context ──────────────────────────────────────────


class TestGetVectorContext:
    """Verify get_vector_context calls _build_rich_context_string and forwards to vector_memory."""

    def setup_method(self):
        self.brain = _make_brain()
        # Stub the two methods get_vector_context calls after building the query
        self.brain.vector_memory.get_context_for_prompt.return_value = "some context"
        self.brain.vector_memory.get_stats_for_context.return_value = {
            "total_trades": 0, "win_rate": 0, "avg_pnl": 0
        }

    def test_calls_vector_memory_get_context(self):
        self.brain.get_vector_context(adx=30, trend_direction="BULLISH")
        self.brain.vector_memory.get_context_for_prompt.assert_called_once()

    def test_query_contains_adx_label(self):
        self.brain.get_vector_context(adx=30, trend_direction="BULLISH")
        call_args = self.brain.vector_memory.get_context_for_prompt.call_args
        query_str = call_args[0][0]  # first positional arg is context_query
        assert "High ADX" in query_str

    def test_query_contains_exit_execution_context(self):
        self.brain.get_vector_context(
            adx=30,
            trend_direction="BULLISH",
            exit_execution_context=ExitExecutionContext(
                stop_loss_type="hard",
                stop_loss_check_interval="15m",
                take_profit_type="soft",
                take_profit_check_interval="4h",
            ),
        )
        call_args = self.brain.vector_memory.get_context_for_prompt.call_args
        query_str = call_args[0][0]
        display_context = call_args.kwargs.get("display_context") or call_args[0][2]
        assert "Exit Execution: SL hard/15m | TP soft/4h" in query_str
        assert "Exit Execution: SL hard/15m | TP soft/4h" in display_context


# ── get_dynamic_thresholds ──────────────────────────────────────


class TestGetDynamicThresholds:
    """Verify thresholds structure returned by get_dynamic_thresholds."""

    def setup_method(self):
        self.brain = _make_brain()
        self.brain.vector_memory.compute_optimal_thresholds.return_value = {}

    def test_returns_all_expected_keys(self):
        t = self.brain.get_dynamic_thresholds()
        expected = {
            "adx_strong_threshold", "avg_sl_pct", "min_rr_recommended",
            "confidence_threshold", "safe_mae_pct",
            "adx_weak_threshold", "min_confluences_weak", "min_confluences_standard",
            "position_reduce_mixed", "position_reduce_divergent", "min_position_size",
            "rr_borderline_min", "rr_strong_setup",
            "trade_count", "learned_keys",
        }
        assert expected.issubset(set(t.keys()))

    def test_defaults_when_empty(self):
        t = self.brain.get_dynamic_thresholds()
        assert t["adx_strong_threshold"] == 25
        assert t["min_rr_recommended"] == 2.0
        assert t["confidence_threshold"] == 70
        assert t["min_position_size"] == 0.02


class TestReflectionRuleFormatting:
    """Verify reflection-generated rule text preserves full ADX bucket labels."""

    def test_trigger_reflection_preserves_high_adx_label(self):
        brain = _make_brain()

        win_metas = [
            {
                "outcome": "WIN",
                "market_regime": "BULLISH",
                "adx_at_entry": 30,
                "direction": "LONG",
            }
            for _ in range(10)
        ]

        brain.vector_memory._get_trade_metadatas.return_value = win_metas

        brain._trigger_reflection()

        assert brain.vector_memory.store_semantic_rule.called
        rule_text = brain.vector_memory.store_semantic_rule.call_args.kwargs["rule_text"]
        assert "with High ADX." in rule_text

    def test_reflection_blocked_when_losses_drop_win_rate_below_threshold(self):
        """Matching losses that push win rate below 60% should prevent a best-practice rule."""
        brain = _make_brain()

        # 5 wins + 4 losses on the same pattern → 55.5% win rate
        all_metas = [
            {"outcome": "WIN", "market_regime": "BULLISH", "adx_at_entry": 30, "direction": "LONG"}
            for _ in range(5)
        ] + [
            {"outcome": "LOSS", "market_regime": "BULLISH", "adx_at_entry": 30, "direction": "LONG"}
            for _ in range(4)
        ]

        brain.vector_memory._get_trade_metadatas.return_value = all_metas

        brain._trigger_reflection()

        assert not brain.vector_memory.store_semantic_rule.called

    def test_trigger_reflection_stores_wins_and_losses_in_metadata(self):
        """Stored best-practice metadata includes win/loss counts and profitability metrics."""
        brain = _make_brain()

        all_metas = [
            {
                "outcome": "WIN", "market_regime": "BULLISH", "adx_at_entry": 30,
                "direction": "LONG", "pnl_pct": 2.0, "close_reason": "take_profit",
            }
            for _ in range(8)
        ] + [
            {
                "outcome": "LOSS", "market_regime": "BULLISH", "adx_at_entry": 30,
                "direction": "LONG", "pnl_pct": -1.0, "close_reason": "stop_loss",
            }
            for _ in range(2)
        ]

        brain.vector_memory._get_trade_metadatas.return_value = all_metas

        brain._trigger_reflection()

        assert brain.vector_memory.store_semantic_rule.called
        kwargs = brain.vector_memory.store_semantic_rule.call_args.kwargs
        meta = kwargs["metadata"]
        assert meta["rule_type"] == "best_practice"
        assert meta["wins"] == 8
        assert meta["losses"] == 2
        assert meta["win_rate"] == pytest.approx(80.0, abs=0.1)
        assert meta["avg_pnl_pct"] == pytest.approx(1.4, abs=0.1)
        assert meta["profit_factor"] > 1.0

    def test_reflection_fills_missing_exit_profile_and_retires_legacy_unknown_rule(self):
        brain = _make_brain(ExitExecutionContext(
            stop_loss_type="hard",
            stop_loss_check_interval="15m",
            take_profit_type="hard",
            take_profit_check_interval="15m",
        ))

        win_metas = [
            {
                "outcome": "WIN",
                "market_regime": "BULLISH",
                "adx_at_entry": 30,
                "direction": "LONG",
                "pnl_pct": 1.5,
            }
            for _ in range(5)
        ]

        brain.vector_memory._get_trade_metadatas.return_value = win_metas

        brain._trigger_reflection()

        kwargs = brain.vector_memory.store_semantic_rule.call_args.kwargs
        assert kwargs["rule_id"] == "rule_best_long_bullish_high_adx_sl_hard_15m_tp_hard_15m"
        assert "Exit profile: SL hard/15m | TP hard/15m" in kwargs["rule_text"]
        assert kwargs["metadata"]["dominant_exit_profile"] == "SL hard/15m | TP hard/15m"
        assert kwargs["metadata"]["dominant_stop_loss_interval"] == "15m"
        assert kwargs["metadata"]["dominant_take_profit_interval"] == "15m"
        brain.vector_memory.deactivate_semantic_rules.assert_called_once_with([
            "rule_best_long_bullish_high_adx_sl_unknown_unknown_tp_unknown_unknown"
        ])

    def test_refresh_semantic_rules_checks_all_stored_rules_for_stale_profiles(self):
        brain = _make_brain(ExitExecutionContext(
            stop_loss_type="hard",
            stop_loss_check_interval="15m",
            take_profit_type="hard",
            take_profit_check_interval="15m",
        ))
        brain.vector_memory.semantic_rule_count = 75
        brain.vector_memory.get_active_rules.return_value = []

        brain.refresh_semantic_rules_if_stale()

        brain.vector_memory.get_active_rules.assert_called_once_with(n_results=75)

    def test_trigger_loss_reflection_stores_failure_reason_and_recommended_adjustment(self):
        """Loss reflection should diagnose why losses happened and suggest improvements."""
        brain = _make_brain()

        loss_metas = [
            {
                "outcome": "LOSS", "market_regime": "BULLISH", "adx_at_entry": 16,
                "direction": "LONG", "close_reason": "stop_loss", "pnl_pct": -1.5,
            }
            for _ in range(4)
        ]

        brain.vector_memory._get_trade_metadatas.return_value = loss_metas

        brain._trigger_loss_reflection()

        assert brain.vector_memory.store_semantic_rule.called
        kwargs = brain.vector_memory.store_semantic_rule.call_args.kwargs
        meta = kwargs["metadata"]
        assert meta.get("failure_reason")
        assert meta.get("recommended_adjustment")
        assert "ADX" in meta["failure_reason"] or "stop" in meta["failure_reason"]
        assert meta["rule_type"] in ("anti_pattern", "corrective")
        assert meta["wins"] == 0
        assert meta["losses"] == 4

    def test_trigger_loss_reflection_differentiates_hard_stop_profile(self):
        """Hard stop-loss exits should be normalized as stop losses but keep exit profile metadata."""
        brain = _make_brain()

        loss_metas = [
            {
                "outcome": "LOSS", "market_regime": "NEUTRAL", "adx_at_entry": 16,
                "direction": "LONG", "close_reason": "hard_stop", "pnl_pct": -1.2,
                "stop_loss_type": "hard", "stop_loss_check_interval": "1m",
                "take_profit_type": "soft", "take_profit_check_interval": "15m",
            }
            for _ in range(3)
        ]

        brain.vector_memory._get_trade_metadatas.return_value = loss_metas

        brain._trigger_loss_reflection()

        assert brain.vector_memory.store_semantic_rule.called
        kwargs = brain.vector_memory.store_semantic_rule.call_args.kwargs
        meta = kwargs["metadata"]
        assert meta["dominant_close_reason"] == "stop_loss"
        assert meta["dominant_stop_loss_type"] == "hard"
        assert meta["dominant_take_profit_type"] == "soft"
        assert meta["dominant_exit_profile"] == "SL hard/1m | TP soft/15m"
        assert "hard" in meta["failure_reason"]
        assert "hard" in meta["recommended_adjustment"]

    def test_loss_reflection_retires_stale_unknown_rule_when_loss_type_changes(self):
        brain = _make_brain(ExitExecutionContext(
            stop_loss_type="hard",
            stop_loss_check_interval="15m",
            take_profit_type="hard",
            take_profit_check_interval="15m",
        ))

        loss_metas = [
            {
                "outcome": "LOSS", "market_regime": "NEUTRAL", "adx_at_entry": 22,
                "direction": "LONG", "close_reason": "stop_loss", "pnl_pct": -1.0,
            }
            for _ in range(3)
        ]
        win_metas = [
            {
                "outcome": "WIN", "market_regime": "NEUTRAL", "adx_at_entry": 22,
                "direction": "LONG", "close_reason": "stop_loss", "pnl_pct": 1.0,
            }
            for _ in range(3)
        ]
        brain.vector_memory._get_trade_metadatas.return_value = loss_metas + win_metas

        brain._trigger_loss_reflection()

        assert brain.vector_memory.store_semantic_rule.call_args.kwargs["rule_id"] == (
            "rule_corrective_long_neutral_stop_loss_sl_hard_15m_tp_hard_15m"
        )
        brain.vector_memory.deactivate_semantic_rules.assert_has_calls([
            call(["rule_anti_pattern_long_neutral_stop_loss_sl_unknown_unknown_tp_unknown_unknown"]),
            call(["rule_corrective_long_neutral_stop_loss_sl_unknown_unknown_tp_unknown_unknown"]),
        ])

    def test_trigger_ai_mistake_reflection_stores_sideways_overconfidence_rule(self):
        """Repeated HIGH-confidence sideways failures should become AI-mistake rules."""
        brain = _make_brain()

        mistake_metas = [
            {
                "outcome": "LOSS", "market_regime": "NEUTRAL", "adx_at_entry": 15,
                "direction": "LONG", "close_reason": "sideways", "pnl_pct": -0.4,
                "confidence": "HIGH", "entry_confidence": "HIGH",
                "reasoning": "Strong breakout momentum will continue.",
                "max_profit_pct": 0.2, "stop_loss_type": "hard",
                "stop_loss_check_interval": "1m", "take_profit_type": "soft",
                "take_profit_check_interval": "15m",
            }
            for _ in range(3)
        ]

        brain.vector_memory._get_trade_metadatas.return_value = mistake_metas

        brain._trigger_ai_mistake_reflection()

        assert brain.vector_memory.store_semantic_rule.called
        kwargs = brain.vector_memory.store_semantic_rule.call_args.kwargs
        meta = kwargs["metadata"]
        assert meta["rule_type"] == "ai_mistake"
        assert meta["mistake_type"] == "sideways_overconfidence"
        assert meta["entry_confidence"] == "HIGH"
        assert meta["failed_assumption"] == "expected breakout continuation"
        assert meta["dominant_exit_profile"] == "SL hard/1m | TP soft/15m"
        assert "downgrade" in meta["recommended_adjustment"]
        assert "AI MISTAKE" in kwargs["rule_text"]

    def test_loss_reflection_rule_id_is_deterministic(self):
        """Repeated loss reflections for the same pattern should produce the same rule_id."""
        brain = _make_brain()

        loss_metas = [
            {
                "outcome": "LOSS", "market_regime": "BEARISH", "adx_at_entry": 22,
                "direction": "SHORT", "close_reason": "stop_loss", "pnl_pct": -2.0,
            }
            for _ in range(3)
        ]

        brain.vector_memory._get_trade_metadatas.return_value = loss_metas

        brain._trigger_loss_reflection()
        first_id = brain.vector_memory.store_semantic_rule.call_args.kwargs["rule_id"]

        brain.vector_memory.store_semantic_rule.reset_mock()
        brain._trigger_loss_reflection()
        second_id = brain.vector_memory.store_semantic_rule.call_args.kwargs["rule_id"]

        assert first_id == second_id

    def test_best_practice_rule_id_is_deterministic(self):
        """Repeated positive reflections for the same pattern should produce the same rule_id."""
        brain = _make_brain()

        win_metas = [
            {"outcome": "WIN", "market_regime": "BULLISH", "adx_at_entry": 30, "direction": "LONG"}
            for _ in range(6)
        ]

        brain.vector_memory._get_trade_metadatas.return_value = win_metas

        brain._trigger_reflection()
        first_id = brain.vector_memory.store_semantic_rule.call_args.kwargs["rule_id"]

        brain.vector_memory.store_semantic_rule.reset_mock()
        brain._trigger_reflection()
        second_id = brain.vector_memory.store_semantic_rule.call_args.kwargs["rule_id"]

        assert first_id == second_id

