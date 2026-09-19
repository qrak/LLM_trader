"""Dense domain tests for the trading brain's learning loop and prompt context.

Merges the brain-integration, brain-context-journal and prompt-contract-feedback
suites into one module: rich context and query formatting, closed-trade
experience recording, reflection cadence and semantic-rule synthesis, the
post-mortem trade journal, the CRITICAL FEEDBACK contract, and the
config-bounded gates the learned thresholds feed.
"""

import re
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.trading.brain import TradingBrainService
from src.trading.brain_context import BrainContextProvider
from src.trading.data_models import ExitExecutionContext, MarketSnapshot, TradeDecision
from src.trading.stop_loss_tightening_policy import (
    StopLossTighteningPolicy,
    TighteningEvaluation,
)
from src.trading.vector_memory import VectorMemoryService
from tests.conftest import make_market_conditions, make_position

HARD_15M = ExitExecutionContext(
    stop_loss_type="hard",
    stop_loss_check_interval="15m",
    take_profit_type="hard",
    take_profit_check_interval="15m",
)

EXIT_HARD_SOFT = {
    "stop_loss_type_at_entry": "hard",
    "stop_loss_check_interval_at_entry": "15m",
    "take_profit_type_at_entry": "soft",
    "take_profit_check_interval_at_entry": "4h",
}

ENTRY_DECISION = TradeDecision(
    timestamp=datetime(2026, 9, 17, tzinfo=timezone.utc),
    symbol="BTC/USDT",
    action="BUY",
    confidence="HIGH",
    price=50000.0,
    reasoning="Strong breakout momentum is likely to continue.",
)

TIGHTENING = TighteningEvaluation(
    is_tightening=True,
    price_progress=0.25,
    base_min_progress=0.20,
    effective_min_progress=0.20,
    allowed=True,
    source="config",
    reason="sufficient progress",
)

WIN_META = {
    "outcome": "WIN",
    "market_regime": "BULLISH",
    "adx_at_entry": 30,
    "direction": "LONG",
    "pnl_pct": 2.0,
    "close_reason": "take_profit",
}

LOSS_META = {
    "outcome": "LOSS",
    "market_regime": "NEUTRAL",
    "adx_at_entry": 16,
    "direction": "LONG",
    "pnl_pct": -1.5,
    "close_reason": "stop_loss",
}

MISTAKE_META = {
    "outcome": "LOSS",
    "market_regime": "NEUTRAL",
    "adx_at_entry": 15,
    "direction": "LONG",
    "close_reason": "sideways",
    "pnl_pct": -0.4,
    "confidence": "HIGH",
    "entry_confidence": "HIGH",
    "reasoning": "Strong breakout momentum will continue.",
    "max_profit_pct": 0.2,
    "stop_loss_type": "hard",
    "stop_loss_check_interval": "1m",
    "take_profit_type": "soft",
    "take_profit_check_interval": "15m",
}

JOURNAL_ROWS = [
    {
        "verdict": "overestimated_breakout",
        "created_at": "2026-06-18 12:00:00",
        "symbol": "BTC/USDC",
        "lesson_learned": "Lesson A",
        "pnl_pct": -3.2,
    },
    {
        "verdict": "good_exit",
        "created_at": "2026-06-17 12:00:00",
        "symbol": "ETH/USDC",
        "lesson_learned": "Lesson B",
        "pnl_pct": 2.1,
    },
]

FEEDBACK = """## CRITICAL FEEDBACK: System Rejections

The following trade suggestions were BLOCKED by risk guards. ADJUST your parameters before proposing the next trade.

### R/R Minimum Guard (2 recent):
  1. LONG (HIGH confidence, MEDIUM volatility):
     - Your R/R: 1.20 | Required: 2.00 (gap: -0.80)
     - Your SL: 3.00% from entry
     - Your TP: 4.00% from entry
  2. SHORT (MEDIUM confidence, HIGH volatility):
     - Your R/R: 1.50 | Required: 2.50 (gap: -1.00)

### SL Too Far (max 10%) (1 recent):
  1. SHORT (LOW confidence, MEDIUM volatility):
     - Your R/R: 0.67 | Required: 0.00 (gap: +0.67)
     - Your SL: 15.00% from entry
     - Your TP: 10.00% from entry
     - Your thesis: "Expecting quick bounce off support"

### PRE-FLIGHT CHECKLIST (MANDATORY):
- Before outputting BUY/SELL, verify: R/R >= required minimum from Decision Rules; historical values above are diagnostic only.
- If volatility is HIGH, widen SL to >1x ATR to achieve required R/R.
- If volatility is LOW, do not use 2x+ ATR SL — tighten to keep R/R viable.
- Compare your proposed SL/TP against the last rejection patterns above.
- If Decision Rules set a positive R/R floor and reasonable SL/TP cannot meet it, output HOLD.
"""

TRUNCATION_MARKER = (
    "[Brain context truncated: token budget reached. Rely on standard analysis for remaining decisions.]"
)


def vector_memory_double(trade_count: int = 0) -> MagicMock:
    """VectorMemoryService double answering every brain query with an empty result."""
    memory = MagicMock()
    memory.trade_count = trade_count
    memory.experience_count = trade_count
    memory.get_context_for_prompt.return_value = ""
    memory.get_stats_for_context.return_value = {"total_trades": 0, "win_rate": 0.0, "avg_pnl": 0.0}
    memory.get_blocked_trade_feedback.return_value = ""
    memory.get_relevant_rules.return_value = []
    memory.compute_confidence_stats.return_value = {}
    memory.get_direction_bias.return_value = None
    memory.get_confidence_recommendation.return_value = None
    memory.compute_optimal_thresholds.return_value = {}
    memory.compute_per_profile_stats.return_value = {}
    memory.compute_rsi_performance.return_value = {}
    memory.compute_volume_performance.return_value = {}
    memory.compute_weekend_performance.return_value = {}
    return memory


def learning_brain(
    trade_count: int = 0,
    exit_execution_context: ExitExecutionContext | None = None,
    post_mortem_repo: MagicMock | None = None,
    **kwargs,
) -> TradingBrainService:
    """TradingBrainService wired to doubles, with an optional configured exit profile."""
    return TradingBrainService(
        logger=MagicMock(),
        persistence=MagicMock(),
        vector_memory=vector_memory_double(trade_count),
        exit_execution_context=exit_execution_context,
        post_mortem_repo=post_mortem_repo,
        **kwargs,
    )


def count_checklist_items(context: str) -> int:
    """Count mandatory PRE-FLIGHT CHECKLIST bullets in a built prompt."""
    in_checklist = False
    items = 0
    for line in context.split("\n"):
        if "PRE-FLIGHT CHECKLIST" in line:
            in_checklist = True
            continue
        if in_checklist and line.startswith("- "):
            items += 1
        elif in_checklist and not line.startswith("- ") and line.strip():
            in_checklist = False
    return items


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        pytest.param({"adx": 30}, "High ADX", id="high-adx"),
        pytest.param({"adx": 10}, "Low ADX", id="low-adx"),
        pytest.param({"adx": 22}, "Medium ADX", id="medium-adx"),
        pytest.param({"adx": 25, "trend_direction": "BEARISH"}, "BEARISH", id="trend"),
        pytest.param({"adx": 25, "volatility_level": "HIGH"}, "HIGH Volatility", id="volatility"),
        pytest.param({"adx": 25, "rsi_level": "OVERBOUGHT"}, "RSI OVERBOUGHT", id="rsi"),
        pytest.param({"adx": 25, "macd_signal": "BEARISH"}, "MACD BEARISH", id="macd"),
        pytest.param({"adx": 25, "volume_state": "DISTRIBUTION"}, "Volume DISTRIBUTION", id="volume"),
        pytest.param({"adx": 25, "bb_position": "LOWER"}, "Price at BB LOWER", id="bollinger"),
        pytest.param({"adx": 25, "is_weekend": True}, "Weekend Low Volume", id="weekend"),
        pytest.param({"adx": 25, "market_sentiment": "EXTREME_FEAR"}, "Sentiment EXTREME_FEAR", id="sentiment"),
        pytest.param({"adx": 25, "order_book_bias": "SELL_PRESSURE"}, "OrderBook SELL_PRESSURE", id="orderbook"),
        pytest.param(
            {"adx": 25, "exit_execution_context": HARD_15M},
            "Exit Execution: SL hard/15m | TP hard/15m",
            id="exit-execution",
        ),
    ],
)
def test_rich_context_string_renders_every_classified_component(overrides, fragment):
    context = learning_brain().context_provider.build_rich_context_string(MarketSnapshot(**overrides))

    assert fragment in context
    assert " + " in context


def test_rich_context_string_omits_neutral_components_and_rsi_when_flat():
    context = learning_brain().context_provider.build_rich_context_string(MarketSnapshot(adx=22))

    assert context == "NEUTRAL + Medium ADX + MEDIUM Volatility"
    assert "RSI" not in context


def test_snapshot_from_conditions_maps_stored_state_and_drops_unmodelled_fields():
    conditions = make_market_conditions(
        trend_direction="BULLISH",
        adx=30.0,
        volatility="HIGH",
        rsi_level="OVERBOUGHT",
        macd_signal="BULLISH",
        volume_state="ACCUMULATION",
        bb_position="UPPER",
        is_weekend=True,
        market_sentiment="EXTREME_FEAR",
        order_book_bias="BUY_PRESSURE",
        choppiness=70.0,
        atr_percentage=3.5,
        vwap=99.0,
        mfi=61.0,
        cmf=0.2,
    )
    snapshot = MarketSnapshot.from_conditions(conditions, HARD_15M)

    context = learning_brain().context_provider.build_rich_context_string(snapshot)

    assert context == (
        "BULLISH + High ADX + HIGH Volatility + RSI OVERBOUGHT + MACD BULLISH + Volume ACCUMULATION "
        "+ Price at BB UPPER + Weekend Low Volume + Sentiment EXTREME_FEAR + OrderBook BUY_PRESSURE "
        "+ Exit Execution: SL hard/15m | TP hard/15m"
    )
    assert (snapshot.choppiness, snapshot.atr_percentage, snapshot.vwap, snapshot.mfi, snapshot.cmf) == (
        None,
        0.0,
        0.0,
        None,
        None,
    )


def test_get_vector_context_sends_query_display_context_and_appends_stats():
    brain = learning_brain(trade_count=8)
    brain.vector_memory.get_context_for_prompt.return_value = "## Similar Past Trades\n- Trade 1: WIN\n"
    brain.vector_memory.get_stats_for_context.return_value = {
        "total_trades": 12,
        "win_rate": 66.6,
        "avg_pnl": 1.25,
    }
    snapshot = MarketSnapshot(adx=30, trend_direction="BULLISH", atr_percentage=2.5, exit_execution_context=HARD_15M)

    context = brain.context_provider.get_vector_context(snapshot)

    assert context == (
        "## Similar Past Trades\n- Trade 1: WIN\n"
        "### Learned Stats for This Context:\n"
        "- Win Rate in similar conditions: 67% (12 trades)\n"
        "- Avg P&L: +1.25%\n"
    )
    query, k = brain.vector_memory.get_context_for_prompt.call_args.args
    assert k == 5
    assert "High ADX" in query
    assert "Exit Execution: SL hard/15m | TP hard/15m" in query
    assert brain.vector_memory.get_context_for_prompt.call_args.kwargs == {
        "display_context": "BULLISH + High ADX + MEDIUM Volatility + Exit Execution: SL hard/15m | TP hard/15m",
        "current_atr_percentage": 2.5,
    }
    stats_query = brain.vector_memory.get_stats_for_context.call_args
    assert stats_query.args == (query,)
    assert stats_query.kwargs == {"k": 20}


def test_get_vector_context_omits_stats_block_and_empty_atr():
    brain = learning_brain()
    brain.vector_memory.get_context_for_prompt.return_value = "## LIMITED"

    context = brain.context_provider.get_vector_context(MarketSnapshot(adx=30))

    assert context == "## LIMITED"
    assert "Learned Stats" not in context
    assert brain.vector_memory.get_context_for_prompt.call_args.kwargs["current_atr_percentage"] is None


def test_get_dynamic_thresholds_returns_defaults_and_sl_tightening_payload(config):
    brain = learning_brain(tightening_policy=StopLossTighteningPolicy.from_config(config))

    thresholds = brain.get_dynamic_thresholds()

    assert {
        "adx_strong_threshold",
        "avg_sl_pct",
        "min_rr_recommended",
        "confidence_threshold",
        "safe_mae_pct",
        "adx_weak_threshold",
        "min_confluences_weak",
        "min_confluences_standard",
        "position_reduce_mixed",
        "position_reduce_divergent",
        "min_position_size",
        "rr_borderline_min",
        "rr_strong_setup",
        "trade_count",
        "learned_keys",
    }.issubset(thresholds)
    assert thresholds["adx_strong_threshold"] == 25
    assert thresholds["min_rr_recommended"] == 2.0
    assert thresholds["confidence_threshold"] == 60
    assert thresholds["min_position_size"] == 0.02
    assert thresholds["rr_borderline_min"] == 0.0
    assert thresholds["trade_count"] == 0
    assert thresholds["learned_keys"] == []
    tightening = thresholds["sl_tightening"]
    assert {"base_threshold", "effective_threshold", "effective_threshold_pct", "source"}.issubset(tightening)
    assert tightening["base_threshold"] == config.SL_TIGHTENING_SWING == 0.15
    assert tightening["source"] == "config"
    assert tightening["effective_threshold_pct"] == thresholds["sl_tightening_pct"]


@pytest.mark.parametrize(
    ("sample_count", "learned", "expected_source", "expected_pct"),
    [
        pytest.param(20, 0.25, "brain", 25, id="above-min-samples"),
        pytest.param(10, 0.25, "brain", 25, id="exactly-min-samples"),
        pytest.param(9, 0.25, "config", 15, id="below-min-samples"),
        pytest.param(20, 0.50, "brain", 40, id="clamped-to-ceiling"),
        pytest.param(20, 0.01, "brain", 5, id="clamped-to-floor"),
        pytest.param(20, None, "config", 15, id="learned-missing"),
        pytest.param(20, "0.25", "brain", 25, id="learned-as-numeric-string"),
        pytest.param(20, "not-a-number", "config", 15, id="learned-corrupted"),
    ],
)
def test_sl_tightening_override_respects_config_samples_floor_and_ceiling(
    config, sample_count, learned, expected_source, expected_pct
):
    brain = learning_brain(tightening_policy=StopLossTighteningPolicy.from_config(config))
    brain.vector_memory.compute_optimal_thresholds.return_value = {
        "sl_tightening": {"learned_threshold": learned, "sample_count": sample_count, "source": "brain"}
    }

    tightening = brain.get_dynamic_thresholds()["sl_tightening"]

    assert tightening["source"] == expected_source
    assert tightening["effective_threshold_pct"] == expected_pct
    assert tightening["learned_threshold"] == learned


@pytest.mark.parametrize("learned_rr", [1.5, 2.5, 1.2, 0.8])
def test_dynamic_rr_threshold_does_not_depend_on_choppiness(config, learned_rr):
    """Prompt and executor must receive the same brain-derived R/R threshold."""
    brain = learning_brain()
    brain.vector_memory.compute_optimal_thresholds.return_value = {"rr_borderline_min": learned_rr}

    thresholds = brain.get_dynamic_thresholds()

    assert config.MIN_RR_ENTRY == 1.0
    assert thresholds["rr_borderline_min"] == learned_rr


def test_update_from_closed_trade_stores_exit_profile_and_entry_snapshot():
    brain = learning_brain(exit_execution_context=HARD_15M)

    brain.update_from_closed_trade(
        position=make_position(**EXIT_HARD_SOFT),
        close_price=52000.0,
        close_reason="take_profit",
        entry_decision=ENTRY_DECISION,
        market_conditions=make_market_conditions(adx=30.0, trend_direction="BULLISH"),
    )

    stored = brain.vector_memory.store_experience.call_args.kwargs
    assert stored["outcome"] == "WIN"
    assert stored["direction"] == "LONG"
    assert stored["close_reason"] == "take_profit"
    assert stored["confidence"] == "HIGH"
    assert stored["reasoning"] == ENTRY_DECISION.reasoning
    assert stored["trade_id"] == "trade_2026-09-17T00:00:00+00:00"
    assert stored["market_context"] == (
        "BULLISH + High ADX + MEDIUM Volatility + Exit Execution: SL hard/15m | TP soft/4h"
    )
    assert stored["metadata"]["stop_loss_type"] == "hard"
    assert stored["metadata"]["stop_loss_check_interval"] == "15m"
    assert stored["metadata"]["take_profit_type"] == "soft"
    assert stored["metadata"]["take_profit_check_interval"] == "4h"
    assert stored["metadata"]["entry_action"] == "BUY"
    assert stored["metadata"]["entry_confidence"] == "HIGH"
    assert stored["metadata"]["ai_reasoning"] == ENTRY_DECISION.reasoning
    assert brain.vector_memory.update_rule_validation_feedback.call_args.kwargs == {
        "current_context": stored["market_context"],
        "outcome": "WIN",
    }
    assert brain.vector_memory.store_semantic_rule.called is False


def test_update_from_closed_trade_without_entry_decision_falls_back_to_position():
    brain = learning_brain(exit_execution_context=HARD_15M)

    brain.update_from_closed_trade(
        position=make_position(confidence="MEDIUM", **EXIT_HARD_SOFT),
        close_price=52000.0,
        close_reason="take_profit",
        market_conditions=make_market_conditions(adx=30.0),
    )

    stored = brain.vector_memory.store_experience.call_args.kwargs
    assert stored["confidence"] == "MEDIUM"
    assert stored["reasoning"] == "N/A"
    assert stored["metadata"]["entry_action"] == "LONG"
    assert stored["metadata"]["entry_confidence"] == "MEDIUM"
    assert stored["metadata"]["ai_reasoning"] == ""


@pytest.mark.parametrize(
    ("position_overrides", "default_context", "expected_profile", "expected_metadata"),
    [
        pytest.param({}, HARD_15M, "SL hard/15m | TP hard/15m", ("hard", "15m", "hard", "15m"), id="all-unknown"),
        pytest.param(
            {"stop_loss_type_at_entry": "hard", "stop_loss_check_interval_at_entry": "15m"},
            HARD_15M,
            "SL hard/15m | TP hard/15m",
            ("hard", "15m", "hard", "15m"),
            id="only-sl-known",
        ),
        pytest.param(EXIT_HARD_SOFT, HARD_15M, "SL hard/15m | TP soft/4h", ("hard", "15m", "soft", "4h"), id="position-wins"),
        pytest.param({}, None, "", ("unknown", "unknown", "unknown", "unknown"), id="unknown-everywhere"),
    ],
)
def test_update_from_closed_trade_resolves_exit_profile_from_position_then_default(
    position_overrides, default_context, expected_profile, expected_metadata
):
    brain = learning_brain(exit_execution_context=default_context)

    brain.update_from_closed_trade(
        position=make_position(**position_overrides),
        close_price=52000.0,
        close_reason="take_profit",
        market_conditions=make_market_conditions(adx=30.0),
    )

    metadata = brain.vector_memory.store_experience.call_args.kwargs["metadata"]
    market_context = brain.vector_memory.store_experience.call_args.kwargs["market_context"]
    assert ("Exit Execution:" in market_context) is bool(expected_profile)
    if expected_profile:
        assert f"Exit Execution: {expected_profile}" in market_context
    assert (
        metadata["stop_loss_type"],
        metadata["stop_loss_check_interval"],
        metadata["take_profit_type"],
        metadata["take_profit_check_interval"],
    ) == expected_metadata


@pytest.mark.parametrize(
    ("conditions", "expected_vwap", "expected_chandelier"),
    [
        pytest.param({"vwap": 49500.0, "chandelier_long": 49250.0}, 0.01, 0.015, id="levels-below-entry"),
        pytest.param({"vwap": 50000.0, "chandelier_long": 50000.0}, 0.0, 0.0, id="levels-at-entry"),
        pytest.param({}, None, None, id="levels-never-computed"),
    ],
)
def test_update_from_closed_trade_records_normalised_level_distances(conditions, expected_vwap, expected_chandelier):
    brain = learning_brain(exit_execution_context=HARD_15M)

    brain.update_from_closed_trade(
        position=make_position(),
        close_price=52000.0,
        close_reason="take_profit",
        market_conditions=make_market_conditions(**conditions),
    )

    metadata = brain.vector_memory.store_experience.call_args.kwargs["metadata"]
    assert metadata["vwap_at_entry"] == conditions.get("vwap", 0.0)
    assert metadata["chandelier_long"] == conditions.get("chandelier_long", 0.0)
    if expected_vwap is None:
        assert metadata["vwap_distance_pct"] is None
        assert metadata["chandelier_distance_pct"] is None
    else:
        assert metadata["vwap_distance_pct"] == pytest.approx(expected_vwap)
        assert metadata["chandelier_distance_pct"] == pytest.approx(expected_chandelier)


@pytest.mark.parametrize(
    ("timeframe_minutes", "expected_interval"),
    [
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
    ],
)
def test_reflection_interval_derives_from_timeframe(timeframe_minutes, expected_interval):
    assert TradingBrainService._derive_reflection_interval(timeframe_minutes) == expected_interval
    assert learning_brain(timeframe_minutes=timeframe_minutes)._reflection_interval == expected_interval


@pytest.mark.parametrize(
    ("timeframe_minutes", "start_count", "expected_calls"),
    [
        pytest.param(240, 4, 1, id="four-hour-first-multiple"),
        pytest.param(240, 9, 1, id="four-hour-second-multiple"),
        pytest.param(60, 4, 0, id="intraday-below-interval"),
        pytest.param(60, 6, 1, id="intraday-first-multiple"),
        pytest.param(60, 13, 1, id="intraday-second-multiple"),
        pytest.param(1440, 5, 1, id="position-first-multiple"),
        pytest.param(5, 9, 1, id="scalping-first-multiple"),
    ],
)
def test_reflection_fires_on_exact_closed_trade_multiple(timeframe_minutes, start_count, expected_calls):
    brain = learning_brain(timeframe_minutes=timeframe_minutes)
    brain._trade_count = start_count
    brain.trigger_reflection = MagicMock()
    brain.trigger_loss_reflection = MagicMock()
    brain.trigger_ai_mistake_reflection = MagicMock()

    brain.update_from_closed_trade(
        position=make_position(),
        close_price=52000.0,
        close_reason="take_profit",
        market_conditions=make_market_conditions(adx=30.0, trend_direction="BULLISH"),
    )

    triggers = (
        brain.trigger_reflection,
        brain.trigger_loss_reflection,
        brain.trigger_ai_mistake_reflection,
    )
    assert [trigger.call_count for trigger in triggers] == [expected_calls] * 3


def test_update_from_closed_trade_propagates_reflection_failure_after_storing():
    brain = learning_brain(trade_count=4)
    brain.reflection_engine = MagicMock()
    brain.reflection_engine.trigger_reflection.side_effect = RuntimeError("reflection blew up")

    with pytest.raises(RuntimeError, match="reflection blew up"):
        brain.update_from_closed_trade(
            position=make_position(),
            close_price=52000.0,
            close_reason="take_profit",
            market_conditions=make_market_conditions(adx=30.0),
        )

    assert brain.vector_memory.store_experience.called is True
    assert brain.vector_memory.update_rule_validation_feedback.called is True
    brain.reflection_engine.trigger_loss_reflection.assert_not_called()


def test_reflection_engine_swallows_storage_errors_and_corrupted_metadata():
    brain = learning_brain()
    failure = RuntimeError("vector store down")
    brain.vector_memory.get_trade_metadatas.side_effect = failure

    brain.trigger_reflection()
    brain.trigger_loss_reflection()
    brain.trigger_ai_mistake_reflection()

    assert brain.vector_memory.store_semantic_rule.called is False
    assert [entry.args[0] for entry in brain.logger.warning.call_args_list] == [
        "Reflection failed: %s",
        "Loss reflection failed: %s",
        "AI mistake reflection failed: %s",
    ]
    assert [entry.args[1] for entry in brain.logger.warning.call_args_list] == [failure] * 3

    brain.logger.warning.reset_mock()
    brain.vector_memory.get_trade_metadatas.side_effect = None
    brain.vector_memory.get_trade_metadatas.return_value = [{"outcome": "WIN"}, None, "garbage"]

    brain.trigger_reflection()

    assert brain.vector_memory.store_semantic_rule.called is False
    assert brain.logger.warning.call_args.args[0] == "Reflection failed: %s"

    brain.vector_memory.get_trade_metadatas.return_value = [{"outcome": "WIN", "pnl_pct": "not-a-number"}] * 10

    brain.trigger_reflection()

    corrupted = brain.vector_memory.store_semantic_rule.call_args.kwargs
    assert corrupted["rule_id"] == "rule_best_unknown_neutral_low_adx_sl_unknown_unknown_tp_unknown_unknown"
    assert "with Low ADX." in corrupted["rule_text"]
    assert corrupted["metadata"]["win_rate"] == 100.0
    assert corrupted["metadata"]["avg_pnl_pct"] == 0.0

    brain.vector_memory.store_semantic_rule.reset_mock()
    brain.vector_memory.get_trade_metadatas.return_value = [{"outcome": "WIN"}] * 10

    brain.trigger_reflection()

    blank = brain.vector_memory.store_semantic_rule.call_args.kwargs
    assert blank["rule_id"] == "rule_best_unknown_neutral_low_adx_sl_unknown_unknown_tp_unknown_unknown"
    assert blank["metadata"]["source_trades"] == 10


@pytest.mark.parametrize("evaluation", [None, TIGHTENING], ids=["no-evaluation", "tightening-evaluation"])
def test_track_position_update_forwards_evaluation_and_timeframe(evaluation):
    brain = learning_brain(timeframe_minutes=240)
    brain.experience_recorder = MagicMock()
    position = make_position()

    brain.track_position_update(
        position=position,
        old_sl=49000.0,
        old_tp=52000.0,
        new_sl=49500.0,
        new_tp=52000.0,
        current_price=51000.0,
        current_pnl_pct=2.0,
        tightening_evaluation=evaluation,
        market_conditions=make_market_conditions(),
    )

    forwarded = brain.experience_recorder.track_position_update.call_args.kwargs
    assert forwarded["position"] is position
    assert forwarded["old_sl"] == 49000.0
    assert forwarded["new_sl"] == 49500.0
    assert forwarded["new_tp"] == 52000.0
    assert forwarded["current_price"] == 51000.0
    assert forwarded["current_pnl_pct"] == 2.0
    assert forwarded["tightening_evaluation"] is evaluation
    assert forwarded["timeframe_minutes"] == 240


def test_best_practice_reflection_stores_rule_with_metrics_and_exit_profile():
    brain = learning_brain(exit_execution_context=HARD_15M)
    losses = [dict(LOSS_META, market_regime="BULLISH", adx_at_entry=30, close_reason="stop_loss", pnl_pct=-1.0) for _ in range(4)]
    brain.vector_memory.get_trade_metadatas.return_value = [dict(WIN_META)] * 6 + losses

    brain.trigger_reflection()

    rule = brain.vector_memory.store_semantic_rule.call_args.kwargs
    assert rule["rule_id"] == "rule_best_long_bullish_high_adx_sl_hard_15m_tp_hard_15m"
    assert "with High ADX." in rule["rule_text"]
    assert rule["rule_text"] == (
        "LONG trades perform well in BULLISH market with High ADX. "
        "Exit profile: SL hard/15m | TP hard/15m. "
        "(6 wins, 4 losses — 60% win rate)"
    )
    metadata = rule["metadata"]
    assert metadata["rule_type"] == "best_practice"
    assert metadata["wins"] == 6
    assert metadata["losses"] == 4
    assert metadata["win_rate"] == pytest.approx(60.0, abs=0.1)
    assert metadata["avg_pnl_pct"] == pytest.approx(0.8, abs=0.1)
    assert metadata["profit_factor"] > 1.0
    assert metadata["source_trades"] == 10
    assert metadata["dominant_exit_profile"] == "SL hard/15m | TP hard/15m"
    assert metadata["dominant_stop_loss_interval"] == "15m"
    assert metadata["dominant_take_profit_interval"] == "15m"


@pytest.mark.parametrize(
    ("trigger", "metas", "expected_stored", "expected_prefix"),
    [
        pytest.param("trigger_reflection", [dict(WIN_META)] * 4, False, "", id="four-wins"),
        pytest.param("trigger_reflection", [dict(WIN_META)] * 5, True, "rule_best_", id="five-wins"),
        pytest.param(
            "trigger_reflection",
            [dict(WIN_META)] * 5 + [dict(LOSS_META, market_regime="BULLISH", adx_at_entry=30, pnl_pct=-1.0)] * 4,
            False,
            "",
            id="win-rate-below-sixty",
        ),
        pytest.param(
            "trigger_reflection",
            [dict(WIN_META)] * 6 + [dict(LOSS_META, market_regime="BULLISH", adx_at_entry=30, pnl_pct=-1.0)] * 4,
            True,
            "rule_best_",
            id="win-rate-exactly-sixty",
        ),
        pytest.param("trigger_loss_reflection", [dict(LOSS_META)] * 2, False, "", id="two-losses"),
        pytest.param("trigger_loss_reflection", [dict(LOSS_META)] * 3, True, "rule_anti_pattern_", id="three-losses"),
        pytest.param("trigger_ai_mistake_reflection", [dict(MISTAKE_META)], False, "", id="one-mistake"),
        pytest.param("trigger_ai_mistake_reflection", [dict(MISTAKE_META)] * 2, True, "rule_ai_mistake_", id="two-mistakes"),
    ],
)
def test_reflection_requires_minimum_samples_and_sixty_percent_win_rate(trigger, metas, expected_stored, expected_prefix):
    brain = learning_brain(exit_execution_context=HARD_15M)
    brain.vector_memory.get_trade_metadatas.return_value = metas
    triggers = {
        "trigger_reflection": brain.trigger_reflection,
        "trigger_loss_reflection": brain.trigger_loss_reflection,
        "trigger_ai_mistake_reflection": brain.trigger_ai_mistake_reflection,
    }

    triggers[trigger]()

    stored = brain.vector_memory.store_semantic_rule.call_args
    assert bool(stored) is expected_stored
    if expected_stored:
        assert stored.kwargs["rule_id"].startswith(expected_prefix)


def test_loss_reflection_stores_corrective_rule_with_diagnostics():
    brain = learning_brain()
    brain.vector_memory.get_trade_metadatas.return_value = [dict(LOSS_META)] * 4

    brain.trigger_loss_reflection()

    rule = brain.vector_memory.store_semantic_rule.call_args.kwargs
    metadata = rule["metadata"]
    assert metadata["failure_reason"]
    assert metadata["recommended_adjustment"]
    assert "ADX" in metadata["failure_reason"] or "stop" in metadata["failure_reason"]
    assert metadata["rule_type"] in ("anti_pattern", "corrective")
    assert metadata["wins"] == 0
    assert metadata["losses"] == 4
    assert metadata["dominant_close_reason"] == "stop_loss"
    assert "ADX >= 20" in metadata["recommended_adjustment"]


@pytest.mark.parametrize(
    ("metas", "default_context", "expected"),
    [
        pytest.param(
            [
                dict(
                    LOSS_META,
                    close_reason="hard_stop",
                    pnl_pct=-1.2,
                    stop_loss_type="hard",
                    stop_loss_check_interval="1m",
                    take_profit_type="soft",
                    take_profit_check_interval="15m",
                )
            ]
            * 3,
            None,
            {
                "rule_id": "rule_anti_pattern_long_neutral_stop_loss_sl_hard_1m_tp_soft_15m",
                "profile": "SL hard/1m | TP soft/15m",
                "stop_type": "hard",
                "take_profit_type": "soft",
            },
            id="explicit-hard-stop-profile",
        ),
        pytest.param(
            [dict(LOSS_META, adx_at_entry=22)] * 3
            + [dict(WIN_META, market_regime="NEUTRAL", adx_at_entry=22, close_reason="stop_loss", pnl_pct=1.0)] * 3,
            HARD_15M,
            {
                "rule_id": "rule_corrective_long_neutral_stop_loss_sl_hard_15m_tp_hard_15m",
                "profile": "SL hard/15m | TP hard/15m",
                "stop_type": "hard",
                "take_profit_type": "hard",
            },
            id="unknown-profile-filled-from-brain-default",
        ),
    ],
)
def test_loss_reflection_normalises_close_reason_and_fills_exit_profile(metas, default_context, expected):
    brain = learning_brain(exit_execution_context=default_context)
    brain.vector_memory.get_trade_metadatas.return_value = metas

    brain.trigger_loss_reflection()

    rule = brain.vector_memory.store_semantic_rule.call_args.kwargs
    metadata = rule["metadata"]
    assert rule["rule_id"] == expected["rule_id"]
    assert metadata["dominant_close_reason"] == "stop_loss"
    assert metadata["dominant_stop_loss_type"] == expected["stop_type"]
    assert metadata["dominant_take_profit_type"] == expected["take_profit_type"]
    assert metadata["dominant_exit_profile"] == expected["profile"]
    assert "hard" in metadata["failure_reason"]
    assert "hard" in metadata["recommended_adjustment"]


def test_ai_mistake_reflection_stores_sideways_overconfidence_rule():
    brain = learning_brain()
    brain.vector_memory.get_trade_metadatas.return_value = [dict(MISTAKE_META)] * 2

    brain.trigger_ai_mistake_reflection()

    rule = brain.vector_memory.store_semantic_rule.call_args.kwargs
    metadata = rule["metadata"]
    assert metadata["rule_type"] == "ai_mistake"
    assert metadata["mistake_type"] == "sideways_overconfidence"
    assert metadata["entry_confidence"] == "HIGH"
    assert metadata["failed_assumption"] == "expected breakout continuation"
    assert metadata["dominant_exit_profile"] == "SL hard/1m | TP soft/15m"
    assert "downgrade" in metadata["recommended_adjustment"]
    assert "AI MISTAKE" in rule["rule_text"]
    assert "hard" in metadata["recommended_adjustment"]


@pytest.mark.parametrize(
    ("trigger", "metas"),
    [
        pytest.param("trigger_reflection", [dict(WIN_META)] * 6, id="best-practice"),
        pytest.param("trigger_loss_reflection", [dict(LOSS_META)] * 3, id="anti-pattern"),
        pytest.param("trigger_ai_mistake_reflection", [dict(MISTAKE_META)] * 3, id="ai-mistake"),
    ],
)
def test_reflection_rule_ids_are_deterministic_across_runs(trigger, metas):
    brain = learning_brain(exit_execution_context=HARD_15M)
    brain.vector_memory.get_trade_metadatas.return_value = metas
    triggers = {
        "trigger_reflection": brain.trigger_reflection,
        "trigger_loss_reflection": brain.trigger_loss_reflection,
        "trigger_ai_mistake_reflection": brain.trigger_ai_mistake_reflection,
    }

    triggers[trigger]()
    first = brain.vector_memory.store_semantic_rule.call_args.kwargs["rule_id"]
    brain.vector_memory.store_semantic_rule.reset_mock()
    triggers[trigger]()

    assert brain.vector_memory.store_semantic_rule.call_args.kwargs["rule_id"] == first


@pytest.mark.parametrize(
    ("rows", "expected_present", "expected_absent"),
    [
        pytest.param(
            JOURNAL_ROWS,
            [
                "### Trade Journal (Recent Post-Mortem Lessons):",
                "— overestimated_breakout (2026-06-18, BTC/USDC): Lesson A, P&L: -3.2%",
                "— good_exit (2026-06-17, ETH/USDC): Lesson B, P&L: +2.1%",
            ],
            [],
            id="two-lessons",
        ),
        pytest.param(
            [
                {
                    "verdict": "test",
                    "created_at": "2026-06-18 12:00:00",
                    "symbol": "BTC/USDC",
                    "lesson_learned": "Lesson",
                    "pnl_pct": None,
                }
            ],
            [
                "### Trade Journal (Recent Post-Mortem Lessons):",
                "— test (2026-06-18, BTC/USDC): Lesson",
            ],
            ["P&L:"],
            id="missing-pnl",
        ),
    ],
)
def test_trade_journal_renders_recent_lessons_with_formatted_pnl(rows, expected_present, expected_absent):
    repo = MagicMock()
    repo.get_recent_post_mortems.return_value = rows
    brain = learning_brain(post_mortem_repo=repo)

    context = brain.get_context(MarketSnapshot(adx=25))

    for fragment in expected_present:
        assert fragment in context
    for fragment in expected_absent:
        assert fragment not in context
    assert repo.get_recent_post_mortems.call_args.kwargs == {"limit": 5}


@pytest.mark.parametrize(
    ("rows", "failure", "expected_warning"),
    [
        pytest.param([], None, False, id="no-post-mortems"),
        pytest.param(None, None, False, id="repo-not-configured"),
        pytest.param([], RuntimeError("DB error"), True, id="repo-failure"),
        pytest.param(
            [{"created_at": "2026-06-18 12:00:00", "symbol": "BTC/USDC", "lesson_learned": "Lesson A"}],
            None,
            True,
            id="row-missing-verdict",
        ),
    ],
)
def test_trade_journal_section_is_absent_without_usable_lessons(rows, failure, expected_warning):
    repo = None if rows is None else MagicMock()
    if repo is not None:
        repo.get_recent_post_mortems.return_value = rows
        repo.get_recent_post_mortems.side_effect = failure
    brain = learning_brain(post_mortem_repo=repo)

    context = brain.get_context(MarketSnapshot(adx=25))

    assert "### Trade Journal (Recent Post-Mortem Lessons):" not in context
    assert "P&L:" not in context
    assert "Lesson A" not in context
    assert brain.logger.warning.called is expected_warning
    if expected_warning:
        assert brain.logger.warning.call_args.args[0] == (
            "Failed to retrieve post-mortem lessons for brain context: %s"
        )


def test_critical_feedback_block_reaches_prompt_intact():
    brain = learning_brain(trade_count=5)
    brain.vector_memory.get_blocked_trade_feedback.return_value = FEEDBACK
    brain.vector_memory.get_context_for_prompt.return_value = "## Vector Context: similar trades"

    context = brain.get_context(MarketSnapshot(adx=25, trend_direction="BULLISH"))

    assert "## CRITICAL FEEDBACK: System Rejections" in context
    assert "### R/R Minimum Guard (2 recent):" in context
    assert "### SL Too Far (max 10%) (1 recent):" in context
    assert "### PRE-FLIGHT CHECKLIST (MANDATORY):" in context
    assert '- Your thesis: "Expecting quick bounce off support"' in context
    assert "bounce off support" in context
    assert brain.vector_memory.get_blocked_trade_feedback.call_args.kwargs == {"n": 5, "max_age_hours": 168}

    rr_lines = re.findall(
        r"Your R/R: \d+\.\d+ \| Required: \d+\.\d+ \(gap: [+-]\d+\.\d+\)",
        context,
    )
    pct_lines = re.findall(r"Your (SL|TP): \d+\.\d+% from entry", context)
    assert rr_lines == [
        "Your R/R: 1.20 | Required: 2.00 (gap: -0.80)",
        "Your R/R: 1.50 | Required: 2.50 (gap: -1.00)",
        "Your R/R: 0.67 | Required: 0.00 (gap: +0.67)",
    ]
    assert len(pct_lines) == 4
    assert count_checklist_items(context) == 5

    headers = [line for line in context.split("\n") if line.startswith("#")]
    assert headers
    for line in headers:
        assert line.startswith(("## ", "### "))
    for line in context.split("\n"):
        if line.startswith("- "):
            assert not line.endswith(" ")
            assert not line.endswith("\t")
    assert context.count("(") == context.count(")")

    positions = [
        context.index(fragment)
        for fragment in (
            "### Confidence Calibration:",
            "CRITICAL FEEDBACK",
            "## Vector Context",
        )
    ]
    assert positions == sorted(positions)


@pytest.mark.parametrize(
    ("feedback", "failure", "expected_blank_run"),
    [
        pytest.param("", None, False, id="empty"),
        pytest.param("\n\n\n", None, True, id="newlines-only"),
        pytest.param("SAMPLE", TypeError("must be real number, not str"), False, id="corrupted-friction-record"),
    ],
)
def test_critical_feedback_section_is_absent_for_empty_or_failing_sources(feedback, failure, expected_blank_run):
    brain = learning_brain(trade_count=5)
    brain.vector_memory.get_blocked_trade_feedback.return_value = feedback
    brain.vector_memory.get_blocked_trade_feedback.side_effect = failure
    brain.vector_memory.get_context_for_prompt.return_value = "## Vector Context: similar trades"

    context = brain.get_context(MarketSnapshot(adx=25))

    assert "CRITICAL FEEDBACK" not in context
    assert "System Rejections" not in context
    assert "## Trading Brain (5 closed trades)" in context
    assert "### Confidence Calibration:" in context
    assert "## Vector Context: similar trades" in context
    assert bool(re.findall(r"\n{4,}", context)) is expected_blank_run


@pytest.mark.parametrize(
    ("trade_count", "bias_rendered"),
    [pytest.param(0, False, id="zero-trades"), pytest.param(5, True, id="five-trades")],
)
def test_critical_feedback_coexists_with_bias_vector_context_and_rules(trade_count, bias_rendered):
    brain = learning_brain(trade_count=trade_count)
    brain.vector_memory.get_blocked_trade_feedback.return_value = FEEDBACK
    brain.vector_memory.get_context_for_prompt.return_value = "## Vector Context: Similar Past Trades"
    brain.vector_memory.get_relevant_rules.return_value = [
        {"similarity": 85, "metadata": {"rule_type": "anti_pattern"}, "text": "Avoid longs into resistance"},
    ]
    brain.vector_memory.get_direction_bias.return_value = {"long_count": 4, "short_count": 0}
    brain.vector_memory.compute_confidence_stats.return_value = {
        "HIGH": {"total_trades": 3, "winning_trades": 2, "win_rate": 66.7, "avg_pnl_pct": 2.5},
    }

    context = brain.get_context(MarketSnapshot(adx=25, trend_direction="BULLISH"))

    assert "CRITICAL FEEDBACK" in context
    assert ("### Direction Bias Check:" in context) is bias_rendered
    assert ("- Historical trades: 4 LONG, 0 SHORT" in context) is bias_rendered
    assert ("NO SHORT TRADES IN HISTORY" in context) is bias_rendered
    assert "## Vector Context: Similar Past Trades" in context
    assert "### Learned Trading Rules (relevant to current conditions):" in context
    assert "[85% match] [⚠️ AVOID] Avoid longs into resistance" in context
    assert "### Trade Journal (Recent Post-Mortem Lessons):" not in context
    assert brain.vector_memory.get_blocked_trade_feedback.call_args.kwargs == {"n": 5, "max_age_hours": 168}

    ordered = ["CRITICAL FEEDBACK", "## Vector Context", "### Learned Trading Rules"]
    if bias_rendered:
        ordered.insert(0, "### Confidence Calibration:")
    else:
        assert "## Trading Brain" not in context
    positions = [context.index(fragment) for fragment in ordered]
    assert positions == sorted(positions)


def test_critical_feedback_survives_unicode_and_token_budget_truncation():
    brain = learning_brain(trade_count=5)
    brain.vector_memory.get_blocked_trade_feedback.return_value = (
        "## CRITICAL FEEDBACK: System Rejections\n\nUnicode: émoji ✓ • ★\n"
    )

    context = brain.get_context(MarketSnapshot(adx=25))

    assert "émoji ✓ • ★" in context
    assert "CRITICAL FEEDBACK" in context
    assert TRUNCATION_MARKER not in context

    brain.vector_memory.get_blocked_trade_feedback.return_value = (
        "## CRITICAL FEEDBACK: System Rejections\n\n"
        + "\n\n".join(f"entry {index}: " + "x" * 120 for index in range(150))
    )

    truncated = brain.get_context(MarketSnapshot(adx=25))

    assert TRUNCATION_MARKER in truncated
    assert truncated.endswith("]")
    assert BrainContextProvider.BRAIN_CONTEXT_MAX_CHARS <= len(truncated) <= (
        BrainContextProvider.BRAIN_CONTEXT_MAX_CHARS + 120
    )

    brain.vector_memory.get_blocked_trade_feedback.return_value = (
        "## CRITICAL FEEDBACK: System Rejections\n" + "x" * 20000
    )

    single_block = brain.get_context(MarketSnapshot(adx=25))

    assert TRUNCATION_MARKER in single_block
    assert len(single_block) < 400


def test_empty_brain_context_only_contains_the_regime_risk_profile():
    brain = learning_brain()

    context = brain.get_context(MarketSnapshot(adx=25, trend_direction="BULLISH"))

    assert context.startswith("\n## ACTIVE RISK PROFILE: NEUTRAL")
    assert "Profile reason: ATR 0.0%" in context
    assert "###" not in context
    assert "Trading Brain" not in context
    assert "CRITICAL FEEDBACK" not in context
    assert brain.vector_memory.get_relevant_rules.call_args.kwargs == {
        "current_context": "BULLISH + High ADX + MEDIUM Volatility",
        "n_results": 3,
    }


def test_empty_brain_uses_limited_data_note_instead_of_apply_insights():
    brain = learning_brain()
    limited = (
        "RELEVANT PAST EXPERIENCES (Context: NEUTRAL + High ADX, active window: last 5 days):\n\n"
        "⚠️ LIMITED DATA: 1 retrieved trade(s), only 0 trade(s) closed in total. Treat these as ANECDOTES, "
        "not as an established pattern — do NOT call an anti-pattern match on this basis. "
        "Standard analysis recommended.\n"
    )
    brain.vector_memory.get_context_for_prompt.return_value = limited

    context = brain.get_context(MarketSnapshot(adx=25))

    assert limited in context
    assert "NOTE: Limited historical data available. Rely on standard technical analysis for this decision." in context
    assert "### Apply Insights" not in context
    assert "ANTI-PATTERN / AI MISTAKE" not in context


@pytest.mark.parametrize(
    ("trade_count", "expected_lines"),
    [
        pytest.param(2, [], id="below-evidence-floor"),
        pytest.param(
            3,
            [
                "- RSI 30-40 (OVERSOLD): 2/2 wins (65% WR) — STRONG SIGNAL",
                "- RSI 60-70 (OVERBOUGHT): 1/4 wins (35% WR) — WEAK SIGNAL",
                "- ACCUMULATION: 1/3 wins (35% WR) — UNFAVORABLE",
                "- Weekend: 2/2 wins (65% WR) — FAVORABLE",
                "- Weekday: 1/3 wins (66% WR) — FAVORABLE",
            ],
            id="at-evidence-floor",
        ),
    ],
)
def test_condition_performance_block_is_gated_by_trade_count_and_win_rate(trade_count, expected_lines):
    brain = learning_brain(trade_count=trade_count)
    brain.vector_memory.compute_rsi_performance.return_value = {
        "OVERSOLD": {"range": "30-40", "total_trades": 2, "winning_trades": 2, "win_rate": 65.0},
        "OVERBOUGHT": {"range": "60-70", "total_trades": 4, "winning_trades": 1, "win_rate": 35.0},
        "NEUTRAL": {"range": "40-60", "total_trades": 1, "winning_trades": 1, "win_rate": 100.0},
    }
    brain.vector_memory.compute_volume_performance.return_value = {
        "ACCUMULATION": {"total_trades": 3, "winning_trades": 1, "win_rate": 34.9},
    }
    brain.vector_memory.compute_weekend_performance.return_value = {
        "weekend": {"total_trades": 2, "winning_trades": 2, "win_rate": 65.0},
        "weekday": {"total_trades": 3, "winning_trades": 1, "win_rate": 66.0},
    }

    context = brain.get_context(MarketSnapshot(adx=25))

    assert ("### Market Condition Performance (from trade history):" in context) is bool(expected_lines)
    for line in expected_lines:
        assert line in context
    assert "RSI 40-60 (NEUTRAL)" not in context
    assert brain.vector_memory.compute_rsi_performance.called is (trade_count >= 3)
    assert brain.vector_memory.compute_per_profile_stats.called is (trade_count >= 3)


def test_blocked_trade_feedback_formatter_drops_non_finite_rr_and_caps_events():
    memory = MagicMock()
    memory.get_recent_blocked_trades.return_value = [
        {
            "guard_type": "rr_minimum",
            "direction": "LONG",
            "confidence": "HIGH",
            "suggested_rr": float("nan"),
            "required_rr": float("nan"),
            "suggested_sl_pct": 0.03,
            "suggested_tp_pct": 0.04,
            "volatility_level": "MEDIUM",
        },
        {
            "guard_type": "rr_minimum",
            "direction": "LONG",
            "confidence": "HIGH",
            "suggested_rr": 1.2,
            "required_rr": 2.0,
            "rr_delta": -0.8,
            "volatility_level": "MEDIUM",
        },
        {
            "guard_type": "rr_minimum",
            "direction": "SHORT",
            "confidence": "LOW",
            "suggested_rr": 1.1,
            "required_rr": 2.0,
            "rr_delta": -0.9,
            "volatility_level": "HIGH",
        },
        {
            "guard_type": "rr_minimum",
            "direction": "SHORT",
            "confidence": "LOW",
            "suggested_rr": 1.0,
            "required_rr": 2.0,
            "rr_delta": -1.0,
            "volatility_level": "HIGH",
        },
        {
            "guard_type": "sl_clamp",
            "direction": "LONG",
            "confidence": "MEDIUM",
            "suggested_rr": 1.8,
            "required_rr": 1.8,
            "rr_delta": 0.0,
            "volatility_level": "LOW",
        },
    ]

    feedback = VectorMemoryService.get_blocked_trade_feedback(memory, n=5, max_age_hours=168)

    assert memory.get_recent_blocked_trades.call_args.kwargs == {"n": 5, "max_age_hours": 168}
    assert "### R/R Minimum Guard (4 recent):" in feedback
    assert "### Stop-Loss Clamping (1 recent):" in feedback
    assert "  1. LONG (HIGH confidence, MEDIUM volatility):" in feedback
    assert "  3. SHORT (LOW confidence, HIGH volatility):" in feedback
    assert "  4. " not in feedback
    assert "- Your SL: 3.00% from entry" in feedback
    assert "- Your TP: 4.00% from entry" in feedback
    assert feedback.count("Your R/R: 1.20 | Required: 2.00 (gap: -0.80)") == 1
    assert "nan" not in feedback.lower()
    assert count_checklist_items(feedback) == 5


def test_blocked_trade_feedback_formatter_raises_on_non_numeric_rr():
    memory = MagicMock()
    memory.get_recent_blocked_trades.return_value = [
        {"guard_type": "rr_minimum", "direction": "LONG", "suggested_rr": "1.20", "required_rr": 2.0},
    ]

    with pytest.raises(TypeError, match="must be real number, not str"):
        VectorMemoryService.get_blocked_trade_feedback(memory, n=5, max_age_hours=168)
