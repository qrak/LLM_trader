"""Dense domain tests for the trading brain's learning loop and prompt context.

Merges the brain-integration, brain-context-journal and prompt-contract-feedback
suites into one module: rich context and query formatting, closed-trade
experience recording, reflection cadence and semantic-rule synthesis, the
post-mortem trade journal, the CRITICAL FEEDBACK contract, and the
config-bounded gates the learned thresholds feed.
"""

import re
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import chromadb
import numpy as np
import pytest
from chromadb.config import Settings
from starlette.testclient import TestClient

from src.analyzer.analysis_context import AnalysisContext
from src.analyzer.formatters import (
    LongTermFormatter,
    MarketFormatter,
    MarketOverviewFormatter,
    TechnicalFormatter,
)
from src.analyzer.prompts.prompt_builder import PromptBuilder
from src.analyzer.prompts.template_manager import TemplateManager
from src.analyzer.technical_calculator import TechnicalCalculator
from src.dashboard.server import DashboardServer
from src.trading.brain import TradingBrainService
from src.trading.data_models import (
    ExitExecutionContext,
    MarketSnapshot,
    Position,
    TradeDecision,
)
from src.trading.executor_reconciliation import ExitEvidence
from src.trading.market_conditions_extractor import MarketConditionsExtractor
from src.trading.stop_loss_tightening_policy import (
    StopLossTighteningPolicy,
    TighteningEvaluation,
)
from src.trading.vector_memory import VectorMemoryService
from src.utils.format_utils import FormatUtils
from src.utils.prompt_budget import PromptBudgetExceeded, PromptBudgetPolicy
from src.utils.timeframe_validator import TimeframeValidator
from src.utils.token_counter import TokenCounter
from tests.conftest import (
    make_config,
    make_market_conditions,
    verified_market_evidence,
)
from tests.conftest import (
    make_position as build_position,
)


def make_position(**overrides):
    """Closed-trade learning fixtures represent independently verified entries."""
    return build_position(**{"entry_execution_quality": verified_market_evidence(), **overrides})


def proven_exit(**overrides) -> ExitEvidence:
    """Evidence of a PROVEN exit: real fill price + amount + stable event id.

    Wave 5: the brain may only learn from a close with evidence (see the guard tests at
    the end of the closed-trade section). Every test below that asserts LEARNING has to
    pass this — a close without it is refused.
    """
    values = {
        "source": "executor_exit_journal",
        "price": 52000.0,
        "quantity": 0.01,
        "event_id": "probe-event-1",
        "fill_verification": "verified",
        "market_verification": verified_market_evidence(52000.0),
    }
    values.update(overrides)
    return ExitEvidence(**values)


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
    PromptBudgetPolicy.BRAIN_NOTICE
)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        pytest.param({"_portfolio_pnl_pct": 0.1332466574281699}, 0.1332466574281699, id="small-profit"),
        pytest.param({"_portfolio_pnl_pct": 12.34}, 12.34, id="profit"),
        pytest.param({"_portfolio_pnl_pct": -5.0}, -5.0, id="loss"),
        pytest.param({"_portfolio_pnl_pct": 0.0}, 0.0, id="breakeven"),
        pytest.param({}, 0.0, id="absent-default"),
        pytest.param({"_portfolio_pnl_pct": "1.25"}, 1.25, id="numeric-string"),
    ],
)
def test_extracted_portfolio_pnl_survives_position_storage_and_brain_context(tmp_path, result, expected):
    conditions = MarketConditionsExtractor(MagicMock()).extract_market_conditions(result)
    position = make_position(conditions_at_entry=conditions)
    restored = Position.from_dict(position.to_dict())
    logger = MagicMock()
    embedding = MagicMock()
    embedding.encode.return_value = np.array([1.0, 0.0, 0.0])
    client = chromadb.PersistentClient(path=str(tmp_path / "brain"), settings=Settings(anonymized_telemetry=False))
    memory = VectorMemoryService(logger, client, embedding)
    brain = TradingBrainService(logger, MagicMock(), memory, prompt_budget=PromptBudgetPolicy(TokenCounter(), logger))

    brain.experience_recorder.record_closed_trade(
        position=restored,
        close_price=52000.0,
        close_reason="take_profit",
        market_conditions=restored.conditions_at_entry,
        entry_decision=ENTRY_DECISION,
    )

    stored = memory._collection.get(ids=[f"trade_{restored.entry_time.isoformat()}"])
    tag = f"PortfolioPnL={expected:+.1f}%"
    assert conditions.portfolio_pnl_pct == expected
    assert restored.conditions_at_entry.portfolio_pnl_pct == expected
    assert stored["metadatas"][0]["portfolio_pnl_pct"] == expected
    assert tag in stored["documents"][0]
    assert tag in memory.get_context_for_prompt(stored["documents"][0], k=1)
    assert stored["metadatas"][0]["pnl_pct"] == restored.calculate_pnl(52000.0)


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
    logger = MagicMock()
    kwargs.setdefault("prompt_budget", PromptBudgetPolicy(TokenCounter(), logger))
    return TradingBrainService(
        logger=logger,
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


def test_vector_stats_heading_is_separated_when_context_has_no_trailing_newline():
    brain = learning_brain(trade_count=8)
    brain.vector_memory.get_context_for_prompt.return_value = "experience ends here"
    brain.vector_memory.get_stats_for_context.return_value = {
        "total_trades": 7, "win_rate": 28.6, "avg_pnl": 0.07,
    }
    context = brain.context_provider.get_vector_context(MarketSnapshot(adx=30))
    assert "experience ends here\n\n### Learned Stats for This Context:" in context
    assert brain.vector_memory.get_context_for_prompt.call_args.kwargs["include_rules"] is False


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
        "## Similar Past Trades\n- Trade 1: WIN\n\n"
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
        "include_rules": False,
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
        "sl_tightening": {"learned_threshold": learned, "sample_count": sample_count,
                          "effective_sample_count": sample_count, "source": "brain"}
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
        evidence=proven_exit(),
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
        evidence=proven_exit(),
    )

    stored = brain.vector_memory.store_experience.call_args.kwargs
    assert stored["confidence"] == "MEDIUM"
    assert stored["reasoning"] == "N/A"
    assert stored["metadata"]["entry_action"] == "LONG"
    assert stored["metadata"]["entry_confidence"] == "MEDIUM"
    assert stored["metadata"]["ai_reasoning"] == ""


@pytest.mark.parametrize("entry_trend", ["BULLISH", "BEARISH", "NEUTRAL"])
def test_closed_experience_preserves_entry_trend_separately_from_exit(entry_trend):
    brain = learning_brain()
    assert brain.update_from_closed_trade(
        position=make_position(trend_direction_at_entry=entry_trend),
        close_price=52000.0, close_reason="take_profit",
        market_conditions=make_market_conditions(trend_direction="BEARISH"), evidence=proven_exit(),
    )
    metadata = brain.vector_memory.store_experience.call_args.kwargs["metadata"]
    assert metadata["trend_direction_at_entry"] == entry_trend
    assert metadata["market_regime"] == "BEARISH"


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
@pytest.mark.parametrize(("drawdowns", "expected"), [
    ([-1.0, -2.0, -3.0, -4.0], None),
    ([-1.0, -2.0, -3.0, -4.0, -5.0], 0.05),
    ([0.0, 0.0, 0.0, 0.0, 0.0], 0.0),
])
def test_closed_trades_round_trip_through_chroma_to_dashboard_and_prompt(tmp_path, direction, drawdowns, expected):
    logger = MagicMock()
    embedding = MagicMock()
    embedding.encode.return_value = np.array([1.0, 0.0, 0.0])
    client = chromadb.PersistentClient(path=str(tmp_path / "brain"), settings=Settings(anonymized_telemetry=False))
    memory = VectorMemoryService(logger, client, embedding)
    brain = TradingBrainService(logger, MagicMock(), memory, timeframe_minutes=15, prompt_budget=PromptBudgetPolicy(TokenCounter(), logger))
    config = make_config(DATA_DIR=str(tmp_path), TIMEFRAME="15m")
    formatter = FormatUtils()
    overview = MarketOverviewFormatter(logger, formatter)
    long_term = LongTermFormatter(logger, formatter)
    builder = PromptBuilder(
        timeframe="15m", logger=logger, config=config, format_utils=formatter,
        overview_formatter=overview, long_term_formatter=long_term,
        technical_formatter=TechnicalFormatter(TechnicalCalculator(logger, formatter), logger, formatter),
        market_formatter=MarketFormatter(logger, formatter, config, MagicMock(), overview, long_term),
        timeframe_validator=TimeframeValidator,
        template_manager=TemplateManager(config, logger, TimeframeValidator),
    )
    assert brain.get_dynamic_thresholds()["safe_mae_pct"] == 0
    assert brain.get_dynamic_thresholds()["safe_mae_sample_count"] == 0
    entry = datetime(2026, 9, 17, tzinfo=timezone.utc)
    for index, drawdown in enumerate(drawdowns):
        position = make_position(
            direction=direction, entry_time=entry + timedelta(minutes=index),
            stop_loss=49000.0 if direction == "LONG" else 51000.0,
            take_profit=53000.0 if direction == "LONG" else 47000.0,
            trend_direction_at_entry="BULLISH", max_drawdown_pct=drawdown,
            confluence_factors=(("support_resistance_strength", 78),),
        )
        close_price = 52000.0 if direction == "LONG" else 48000.0
        assert brain.update_from_closed_trade(
            position, close_price, "take_profit", make_market_conditions(trend_direction="BEARISH"),
            evidence=proven_exit(price=close_price, event_id=f"exit-{index}"),
        )
    stored = memory.get_trade_metadatas()
    assert len(stored) == len(drawdowns)
    assert all(row["trend_direction_at_entry"] == "BULLISH" for row in stored)
    assert all(row["market_regime"] == "BEARISH" for row in stored)
    thresholds = brain.get_dynamic_thresholds()
    prompt = builder.build_system_prompt(
        "BTC/USDC", AnalysisContext(symbol="BTC/USDC", current_price=50000.0),
        dynamic_thresholds=thresholds,
    )
    if expected is None:
        assert "safe_mae_pct" not in thresholds["learned_keys"]
        assert "Insufficient trade data for MAE baseline" in prompt
    else:
        assert thresholds["safe_mae_pct"] == pytest.approx(expected)
        assert thresholds["safe_mae_sample_count"] == len(drawdowns)
        assert "safe_mae_pct" in thresholds["learned_keys"]
        assert f"MAE evidence: {len(drawdowns)} winning trades" in prompt
        assert "not a guaranteed safe limit or an SL floor" in prompt
        assert "Insufficient trade data for MAE baseline" not in prompt
        if expected:
            assert "survived up to 5.00% drawdown" in prompt
        else:
            assert "had no observed adverse move" in prompt
    server = DashboardServer(
        brain_service=brain, vector_memory=memory, analysis_engine=None, config=config,
        logger=logger, unified_parser=None, persistence=None, exchange_manager=None,
    )
    with TestClient(server.app) as http:
        response = http.get("/api/brain/vectors")
        assert response.status_code == 200
        stats = response.json()["factor_stats"]
        assert stats["support_resistance_HIGH"]["total_trades"] == len(drawdowns)
        assert stats["support_resistance_HIGH"]["avg_score"] == 78
        assert stats["cat_Trend: BULLISH"]["total_trades"] == len(drawdowns)
        assert "cat_Trend: BEARISH" not in stats


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
        evidence=proven_exit(),
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
        evidence=proven_exit(),
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
        evidence=proven_exit(),
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
            evidence=proven_exit(),
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
    brain.context_provider.prompt_budget.brain_max_tokens = 1500
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

    feedback = brain.vector_memory.get_blocked_trade_feedback.return_value
    budget = brain.context_provider.prompt_budget
    assert budget.token_counter.count_tokens(feedback) > budget.brain_max_tokens
    with pytest.raises(PromptBudgetExceeded, match="Protected brain context") as error:
        brain.get_context(MarketSnapshot(adx=25))
    assert str(error.value) == "Protected brain context exceeds 1500 estimated tokens; no lesson was cut"
    assert brain.vector_memory.get_blocked_trade_feedback.return_value == feedback

    brain.vector_memory.get_blocked_trade_feedback.return_value = (
        "## CRITICAL FEEDBACK: System Rejections\n" + "x" * 20000
    )

    feedback = brain.vector_memory.get_blocked_trade_feedback.return_value
    assert budget.token_counter.count_tokens(feedback) > budget.brain_max_tokens
    with pytest.raises(PromptBudgetExceeded, match="Protected brain context") as error:
        brain.get_context(MarketSnapshot(adx=25))
    assert str(error.value) == "Protected brain context exceeds 1500 estimated tokens; no lesson was cut"


def test_large_brain_context_keeps_recent_lessons_without_growing_between_calls():
    repository = MagicMock()
    repository.get_recent_post_mortems.return_value = [
        {
            "verdict": "good_exit", "created_at": "2026-10-13 00:00:00",
            "symbol": "BTC/USDC", "pnl_pct": 2.0,
            "lesson_learned": f"Lesson {index}: " + "Historical entry and exit evidence. " * 30,
        }
        for index in range(5)
    ]
    brain = learning_brain(trade_count=10000, post_mortem_repo=repository)
    brain.vector_memory.get_context_for_prompt.return_value = (
        "Relevant past experiences:\n" + "Detailed historical reasoning. " * 1000
    )
    budget = brain.context_provider.prompt_budget
    assert budget.brain_max_tokens == 50000
    first = brain.get_context(MarketSnapshot(adx=25))
    assert len(first) > 12000
    assert budget.token_counter.count_tokens(first) < budget.brain_max_tokens
    assert TRUNCATION_MARKER not in first
    for index in range(5):
        assert first.count(f"Lesson {index}:") == 1
    for _ in range(200):
        assert brain.get_context(MarketSnapshot(adx=25)) == first
    repository.get_recent_post_mortems.assert_called_with(limit=5)
    brain.vector_memory.get_context_for_prompt.assert_called_with(
        brain.context_provider.build_query_document(MarketSnapshot(adx=25)), 3,
        display_context=brain.context_provider.build_rich_context_string(MarketSnapshot(adx=25)),
        current_atr_percentage=None, include_rules=False,
    )
    brain.vector_memory.get_relevant_rules.assert_called_with(
        current_context="NEUTRAL + High ADX + MEDIUM Volatility", n_results=3,
    )
    brain.logger.warning.assert_not_called()


def test_brain_emergency_budget_still_bounds_oversized_records_and_warns():
    brain = learning_brain(trade_count=5)
    budget = brain.context_provider.prompt_budget
    budget.brain_max_tokens = 2000
    brain.vector_memory.get_context_for_prompt.return_value = (
        "Relevant past experiences:\n\n" + "\n\n".join("x" * 80 for _ in range(1000))
    )
    context = brain.get_context(MarketSnapshot(adx=25))
    assert TRUNCATION_MARKER in context
    assert budget.token_counter.count_tokens(context) <= budget.brain_max_tokens
    warning = brain.logger.warning.call_args
    assert warning.args[0] == "Brain context reduced: estimated tokens %d -> %d (limit=%d)"
    assert warning.args[1] > budget.brain_max_tokens
    assert warning.args[2] == budget.token_counter.count_tokens(context)
    assert warning.args[3] == budget.brain_max_tokens


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




class TestBrainRefusesUnprovenLessons:
    """A close without evidence never becomes an experience (the 2026-09-21 class).

    The guard is enforced in ``update_from_closed_trade`` as defence in depth: the close
    path applies the same rule, but the brain must not be teachable by ANY caller that
    has no proven exit fact behind it.
    """

    def test_verified_exit_cannot_teach_an_unknown_entry(self):
        brain = learning_brain()
        assert brain.update_from_closed_trade(
            position=make_position(entry_execution_quality=None), close_price=52000.0,
            close_reason="take_profit", market_conditions=make_market_conditions(), evidence=proven_exit(),
        ) is False
        brain.vector_memory.store_experience.assert_not_called()

    def test_without_evidence_nothing_is_stored_and_the_skip_is_logged(self):
        brain = learning_brain()
        logger = brain.logger

        stored = brain.update_from_closed_trade(
            position=make_position(),
            close_price=52000.0,
            close_reason="take_profit",
            market_conditions=make_market_conditions(adx=30.0),
        )

        assert stored is False
        brain.vector_memory.store_experience.assert_not_called()
        brain.vector_memory.update_rule_validation_feedback.assert_not_called()
        messages = [str(call.args) for call in logger.warning.call_args_list]
        assert any("Lesson skipped: no exit evidence" in message for message in messages)
        assert any("no exit evidence was recorded" in message for message in messages)

    @pytest.mark.parametrize(
        "evidence",
        [
            pytest.param(None, id="no-evidence-object"),
            pytest.param(proven_exit(source="local_monitor"), id="non-confirming-source"),
            pytest.param(proven_exit(source="analysis_signal"), id="analysis-signal-source"),
            pytest.param(proven_exit(price=None), id="unknown-price"),
            pytest.param(proven_exit(price=0.0), id="zero-price"),
            pytest.param(proven_exit(quantity=None), id="unknown-quantity"),
            pytest.param(proven_exit(quantity=float("nan")), id="nan-quantity"),
            pytest.param(proven_exit(event_id=None), id="no-stable-event-id"),
            pytest.param("not-evidence", id="wrong-type"),
        ],
    )
    def test_unusable_evidence_is_refused(self, evidence):
        brain = learning_brain()

        stored = brain.update_from_closed_trade(
            position=make_position(),
            close_price=52000.0,
            close_reason="stop_loss",
            market_conditions=make_market_conditions(adx=30.0),
            evidence=evidence,
        )

        assert stored is False
        brain.vector_memory.store_experience.assert_not_called()

    def test_proven_evidence_learns_exactly_once_per_call(self):
        brain = learning_brain()

        first = brain.update_from_closed_trade(
            position=make_position(),
            close_price=52000.0,
            close_reason="take_profit",
            market_conditions=make_market_conditions(adx=30.0),
            evidence=proven_exit(),
        )
        second = brain.update_from_closed_trade(
            position=make_position(),
            close_price=52000.0,
            close_reason="take_profit",
            market_conditions=make_market_conditions(adx=30.0),
            evidence=proven_exit(),
        )

        assert (first, second) == (True, True)
        assert brain.vector_memory.store_experience.call_count == 2

    def test_mapping_evidence_with_the_same_fields_is_accepted(self):
        """A caller that only produces a dict is validated with the same rules."""
        brain = learning_brain()

        stored = brain.update_from_closed_trade(
            position=make_position(),
            close_price=52000.0,
            close_reason="take_profit",
            market_conditions=make_market_conditions(adx=30.0),
            evidence={
                "source": "executor_exit_journal",
                "price": 52000.0,
                "quantity": 0.01,
                "event_id": "dict-event",
                "fill_verification": "verified",
                "market_verification": verified_market_evidence(52000.0),
            },
        )
        refused = brain.update_from_closed_trade(
            position=make_position(),
            close_price=52000.0,
            close_reason="take_profit",
            market_conditions=make_market_conditions(adx=30.0),
            evidence={
                "source": "executor_exit_journal",
                "price": 52000.0,
                "quantity": None,
                "event_id": "dict-event-2",
            },
        )

        assert (stored, refused) == (True, False)
        assert brain.vector_memory.store_experience.call_count == 1


@pytest.mark.parametrize("effective_count", [0, 9.99, float("nan"), float("inf"), True])
def test_sl_tightening_policy_requires_effective_samples(config, effective_count):
    policy = StopLossTighteningPolicy.from_config(config)
    payload = {"sl_tightening": {
        "learned_threshold": 0.25, "sample_count": 20,
        "effective_sample_count": effective_count,
    }}
    assert policy.resolve_effective_threshold(0.15, payload) == (0.15, "config")


def test_sl_tightening_policy_accepts_sufficient_effective_samples(config):
    policy = StopLossTighteningPolicy.from_config(config)
    payload = {"sl_tightening": {
        "learned_threshold": 0.25, "sample_count": 20, "effective_sample_count": 10.0,
    }}
    assert policy.resolve_effective_threshold(0.15, payload) == (0.25, "brain")


def test_brain_threshold_cache_expires_without_new_experiences(monkeypatch):
    brain = learning_brain()
    clock = [1000.0]
    monkeypatch.setattr("src.trading.brain_context.monotonic", lambda: clock[0])
    brain.vector_memory.compute_optimal_thresholds.side_effect = [
        {"sl_tightening": {"learned_threshold": 0.25, "sample_count": 20,
                           "effective_sample_count": 15}}, {},
    ]
    first = brain.get_dynamic_thresholds()
    assert first["sl_tightening_source"] == "brain"
    clock[0] += 299
    assert brain.get_dynamic_thresholds()["sl_tightening_source"] == "brain"
    brain.vector_memory.compute_optimal_thresholds.assert_called_once()
    clock[0] += 1
    assert brain.get_dynamic_thresholds()["sl_tightening_source"] == "config"
    assert brain.vector_memory.compute_optimal_thresholds.call_count == 2


def test_brain_does_not_mutate_cached_sl_evidence():
    brain = learning_brain()
    payload = {"learned_threshold": 0.25, "sample_count": 20, "effective_sample_count": 15}
    brain.vector_memory.compute_optimal_thresholds.return_value = {"sl_tightening": payload}
    brain.get_dynamic_thresholds()
    assert payload == {"learned_threshold": 0.25, "sample_count": 20, "effective_sample_count": 15}


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_sl_decay_chroma_round_trip_expires_to_config_and_updates_prompt(tmp_path, monkeypatch, direction):
    logger = MagicMock()
    embedding = MagicMock()
    embedding.encode.return_value = np.array([1.0, 0.0, 0.0])
    client = chromadb.PersistentClient(path=str(tmp_path / "brain"), settings=Settings(anonymized_telemetry=False))
    memory = VectorMemoryService(logger, client, embedding, timeframe_minutes=240)
    config = make_config(TIMEFRAME="4h")
    policy = StopLossTighteningPolicy.from_config(config)
    brain = TradingBrainService(logger, MagicMock(), memory, timeframe_minutes=240, tightening_policy=policy)
    now = datetime.now(timezone.utc)
    for index in range(12):
        position = make_position(
            direction=direction, entry_time=now - timedelta(days=2, minutes=index),
            stop_loss=49000.0 if direction == "LONG" else 51000.0,
            take_profit=52000.0 if direction == "LONG" else 48000.0,
        )
        current = 51000.0 if direction == "LONG" else 49000.0
        new_sl = 49500.0 if direction == "LONG" else 50500.0
        brain.experience_recorder.track_position_update(
            position, position.stop_loss, position.take_profit, new_sl, position.take_profit,
            current, 2.0, make_market_conditions(),
            tightening_evaluation=policy.evaluate_update(position, new_sl, current, 240),
            timeframe_minutes=240,
        )
        brain.experience_recorder.record_closed_trade(
            position, current, "analysis_signal", make_market_conditions(),
        )
    clock = [1000.0]
    wall = [datetime.now(timezone.utc)]
    monkeypatch.setattr("src.trading.brain_context.monotonic", lambda: clock[0])
    monkeypatch.setattr("src.trading.vector_memory_analytics.datetime", MagicMock(now=lambda *_: wall[0]))
    first = brain.get_dynamic_thresholds()
    assert first["sl_tightening_source"] == "brain"
    assert first["sl_tightening"]["sample_count"] == 12
    assert first["sl_tightening"]["effective_sample_count"] == pytest.approx(12)
    template = TemplateManager(config, logger, TimeframeValidator)
    assert "source: brain" in template.build_system_prompt("BTC/USDT", "4h", dynamic_thresholds=first)
    wall[0] += timedelta(days=57)
    clock[0] += 300
    expired = brain.get_dynamic_thresholds()
    assert memory.experience_count == 24
    assert expired["sl_tightening_source"] == "config"
    assert expired["sl_tightening_pct"] == 15
    assert "learned_threshold" not in expired["sl_tightening"]
    assert "15%+" in template.build_system_prompt("BTC/USDT", "4h", dynamic_thresholds=expired)
    assert "source: config" in template.build_system_prompt("BTC/USDT", "4h", dynamic_thresholds=expired)
