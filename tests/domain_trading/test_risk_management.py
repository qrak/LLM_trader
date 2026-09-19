"""Risk management domain: RiskManager guards, position sizing, friction feedback.

Merged suite for the friction contracts of ``RiskManager`` (SL/TP distance
floors, wrong-side SL/TP correction, position-size clamping, confidence
fallbacks), the ``TradingStrategy`` translation of those frictions into
``vector_memory.store_blocked_trade`` events, the config-driven R/R entry floor,
and the closed-loop feedback edge cases (empty state, saturation, storage
outage, latency, boundary inputs).
"""

from __future__ import annotations

import asyncio
import inspect
import math
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, MagicMock

import chromadb
import pytest
from sentence_transformers import SentenceTransformer

from src.managers.risk_manager import RiskManager
from src.trading.brain import TradingBrainService
from src.trading.data_models import MarketConditions, MarketSnapshot
from src.trading.trading_strategy import TradingStrategy
from src.trading.vector_memory import VectorMemoryService
from tests.conftest import (
    make_config,
    mock_brain,
    mock_extractor,
    mock_persistence,
    mock_statistics,
)

FRICTION_KEYS = frozenset({"guard_type", "direction", "detail"})
PRICE = 100.0
CAPITAL = 10000.0
NEUTRAL_CAP = 0.08
FEE_PERCENT = 0.00075


def risk_service(**overrides: Any) -> RiskManager:
    """RiskManager on a silent logger with the shared config defaults."""
    return RiskManager(logger=MagicMock(), config=make_config(**overrides))


def entry_parameters(manager: RiskManager, **overrides: Any):
    """RiskManager.calculate_entry_parameters with the shared entry defaults."""
    values: dict[str, Any] = {
        "signal": "BUY",
        "current_price": PRICE,
        "capital": CAPITAL,
        "confidence": "HIGH",
    }
    values.update(overrides)
    return manager.calculate_entry_parameters(**values)


def drain_frictions(manager: RiskManager) -> list[dict[str, Any]]:
    """Guard reports accumulated since the previous drain."""
    return manager.get_and_clear_frictions()


def friction_for(manager: RiskManager, guard_type: str) -> dict[str, Any]:
    """The single drained friction of one guard type, whatever else fired alongside."""
    matches = [
        item for item in drain_frictions(manager) if item["guard_type"] == guard_type
    ]
    assert len(matches) == 1
    return matches[0]


def make_strategy(
    *,
    config: SimpleNamespace | None = None,
    risk_manager: RiskManager | None = None,
    brain: MagicMock | None = None,
    stop_loss: float | None = 95.0,
    take_profit: float | None = 110.0,
    position_size: float | None = 0.05,
    rr_borderline_min: float = 1.5,
    brain_thresholds: dict[str, Any] | None = None,
    min_rr_entry: float = 1.0,
) -> TradingStrategy:
    """TradingStrategy on mocked deps, a recording vector memory and a real RiskManager."""
    service_brain = brain or mock_brain()
    service_brain.get_dynamic_thresholds.return_value = (
        brain_thresholds
        if brain_thresholds is not None
        else {"rr_borderline_min": rr_borderline_min}
    )
    service_brain.vector_memory = MagicMock()
    service_brain.vector_memory.store_blocked_trade.return_value = True
    service_brain.vector_memory.trade_count = 0

    persistence = mock_persistence()

    statistics = mock_statistics()
    statistics.get_current_capital.return_value = CAPITAL

    extractor = mock_extractor()
    extractor.extract_trading_info.return_value = (
        "BUY",
        "HIGH",
        stop_loss,
        take_profit,
        position_size,
        "AI reasoning",
    )
    extractor.validate_signal.return_value = True

    return TradingStrategy(
        logger=MagicMock(),
        persistence=persistence,
        brain_service=service_brain,
        statistics_service=statistics,
        memory_service=MagicMock(),
        risk_manager=risk_manager or RiskManager(logger=MagicMock(), config=make_config()),
        config=config or make_config(MIN_RR_ENTRY=min_rr_entry),
        position_extractor=extractor,
    )


async def run_entry(
    strategy: TradingStrategy,
    *,
    signal: str = "BUY",
    confidence: str = "HIGH",
    stop_loss: float | None = 95.0,
    take_profit: float | None = 110.0,
    position_size: float | None = 0.05,
    current_price: float = PRICE,
    reasoning: str = "Test",
):
    """Drive one full entry through the friction and R/R pipeline."""
    return await strategy._open_new_position(
        signal=signal,
        confidence=confidence,
        stop_loss=stop_loss,
        take_profit=take_profit,
        position_size=position_size,
        current_price=current_price,
        symbol="BTC/USDC",
        reasoning=reasoning,
        market_conditions=MarketConditions(),
    )


def blocked_calls(strategy: TradingStrategy, guard_type: str | None = None) -> list[dict[str, Any]]:
    """Keyword payloads handed to vector_memory.store_blocked_trade."""
    payloads = [
        call.kwargs for call in strategy.brain_service.vector_memory.store_blocked_trade.call_args_list
    ]
    if guard_type is None:
        return payloads
    return [payload for payload in payloads if payload["guard_type"] == guard_type]


def no_feedback() -> str:
    """Vector store holding no rejections."""
    return ""


def failing_feedback() -> str:
    """Vector store outage while reading rejections."""
    raise RuntimeError("DB crash")


def empty_state_brain(respond: Callable[[], str]) -> tuple[TradingBrainService, list[Any]]:
    """Brain with zero closed trades on a vector-memory double recording feedback lookups."""
    seen: list[Any] = []

    def feedback(**kwargs: Any) -> str:
        seen.append(kwargs)
        return respond()

    vector_memory = MagicMock()
    vector_memory.trade_count = 0
    vector_memory.get_blocked_trade_feedback.side_effect = feedback
    vector_memory.get_context_for_prompt.return_value = ""
    vector_memory.get_relevant_rules.return_value = []
    vector_memory.compute_confidence_stats.return_value = {}
    vector_memory.get_confidence_recommendation.return_value = ""
    vector_memory.get_direction_bias.return_value = None

    brain = TradingBrainService(
        logger=MagicMock(),
        persistence=MagicMock(),
        vector_memory=vector_memory,
    )
    return brain, seen


def chroma_client() -> chromadb.ClientAPI:
    """Fresh in-memory Chroma client."""
    client = chromadb.Client(
        chromadb.config.Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=False,
        )
    )
    client.reset()
    return client


def opened_vector_memory(embedding_model: SentenceTransformer) -> VectorMemoryService:
    """VectorMemoryService with its collections initialized."""
    service = VectorMemoryService(
        logger=MagicMock(),
        chroma_client=chroma_client(),
        embedding_model=embedding_model,
    )
    service._ensure_initialized()
    return service


def store_block(
    service: VectorMemoryService,
    *,
    guard_type: str = "rr_minimum",
    direction: str = "LONG",
    confidence: str = "HIGH",
    suggested_rr: float = 1.2,
    required_rr: float = 2.0,
    suggested_sl_pct: float = 0.03,
    suggested_tp_pct: float = 0.04,
    suggested_sl: float = 97.0,
    suggested_tp: float = 104.0,
    current_price: float = PRICE,
    volatility_level: str = "MEDIUM",
    reasoning_snippet: str = "Test",
) -> bool:
    """Store one blocked-trade event with the full guard payload."""
    return service.store_blocked_trade(
        guard_type=guard_type,
        direction=direction,
        confidence=confidence,
        suggested_rr=suggested_rr,
        required_rr=required_rr,
        suggested_sl_pct=suggested_sl_pct,
        suggested_tp_pct=suggested_tp_pct,
        suggested_sl=suggested_sl,
        suggested_tp=suggested_tp,
        current_price=current_price,
        volatility_level=volatility_level,
        reasoning_snippet=reasoning_snippet,
    )


@pytest.fixture(scope="module")
def embedding_model() -> SentenceTransformer:
    """Shared MiniLM encoder for the real VectorMemoryService fixtures."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture
def vector_memory(embedding_model: SentenceTransformer) -> VectorMemoryService:
    """Empty VectorMemoryService."""
    return opened_vector_memory(embedding_model)


@pytest.fixture
def saturated_vector_memory(embedding_model: SentenceTransformer) -> VectorMemoryService:
    """VectorMemoryService pre-populated with 55 blocked trades over six guard types."""
    service = opened_vector_memory(embedding_model)
    guards = [
        "rr_minimum",
        "sl_distance_max",
        "sl_distance_min",
        "sl_below_entry",
        "tp_below_entry",
    ]
    for guard in guards:
        for index in range(10):
            store_block(
                service,
                guard_type=guard,
                suggested_rr=1.0 + index * 0.1,
                required_rr=2.0,
                reasoning_snippet=f"Test #{index} for {guard}",
            )
    for index in range(5):
        store_block(
            service,
            guard_type="position_size_clamp",
            suggested_rr=0.5 + index * 0.1,
            required_rr=1.5,
            reasoning_snippet=f"Size clamp #{index}",
        )
    return service


def test_friction_buffer_lifecycle() -> None:
    """Valid entries stay silent, guards accumulate until drained, instances stay isolated."""
    manager = risk_service()
    entry_parameters(manager, stop_loss=95.0, take_profit=110.0, position_size=0.05)
    assert drain_frictions(manager) == []
    assert drain_frictions(manager) == []

    clamped = risk_service()
    entry_parameters(clamped, position_size=0.50)
    assert len(drain_frictions(clamped)) == 1
    assert drain_frictions(clamped) == []

    accumulating = risk_service()
    entry_parameters(accumulating, position_size=0.50)
    entry_parameters(accumulating, stop_loss=99.5)
    assert len(drain_frictions(accumulating)) >= 2

    first, second = risk_service(), risk_service()
    entry_parameters(first, position_size=0.50)
    assert len(drain_frictions(first)) == 1
    assert drain_frictions(second) == []


def test_position_size_clamp_friction_contract() -> None:
    """The clamp friction reports the requested size, the regime cap and their delta."""
    manager = risk_service()
    assessment = entry_parameters(manager, position_size=0.30)

    drained = drain_frictions(manager)
    assert [item["guard_type"] for item in drained] == ["position_size_clamp"]
    friction = drained[0]
    assert FRICTION_KEYS.issubset(friction)
    assert type(friction["guard_type"]) is str
    assert type(friction["direction"]) is str
    assert type(friction["detail"]) is str
    assert friction["direction"] == "N/A"
    assert friction["suggested_size"] == pytest.approx(0.30)
    assert friction["max_size"] == pytest.approx(NEUTRAL_CAP)
    assert friction["suggested_size"] - friction["max_size"] == pytest.approx(0.22)
    assert friction["detail"] == "Position size 30.0% clamped to max 8.0%"

    assert assessment.size_pct == pytest.approx(NEUTRAL_CAP)
    assert assessment.quote_amount == pytest.approx(800.0)
    assert assessment.quantity == pytest.approx(8.0)
    assert assessment.entry_fee == pytest.approx(0.6)


def test_position_size_clamp_logs_operator_warning() -> None:
    """The clamp is announced with the requested and capped percentages."""
    logger = MagicMock()
    manager = RiskManager(logger=logger, config=make_config())
    entry_parameters(manager, position_size=0.50)

    logger.warning.assert_any_call(
        "%s position size %.2f%% exceeds cap %.2f%%, clamping", "AI", 50.0, 8.0
    )
    friction = drain_frictions(manager)[0]
    assert friction["max_size"] == pytest.approx(NEUTRAL_CAP)


def test_config_max_position_size_caps_every_regime_profile() -> None:
    """MAX_POSITION_SIZE is the hard ceiling: a 5% request under a 2% config cap is clamped
    to 2%, and the AGGRESSIVE profile's 10% cap is cut to the same 2% (F5)."""
    manager = risk_service(MAX_POSITION_SIZE=0.02)
    assessment = entry_parameters(manager, position_size=0.05)

    assert assessment.size_pct == pytest.approx(0.02)
    assert friction_for(manager, "position_size_clamp")["max_size"] == pytest.approx(0.02)

    aggressive = risk_service(MAX_POSITION_SIZE=0.02)
    raised = entry_parameters(
        aggressive,
        signal="SELL",
        position_size=0.20,
        market_conditions=MarketConditions(atr=0.5, atr_percentage=0.5),
        choppiness=70.0,
    )

    assert raised.regime_profile == "aggressive"
    assert raised.size_pct == pytest.approx(0.02)
    assert friction_for(aggressive, "position_size_clamp")["max_size"] == pytest.approx(0.02)


SIZING_CASES = [
    pytest.param(0.50, "HIGH", {}, 0.08, 800.0, 8.0, 1, id="ai_size_above_cap_clamped"),
    pytest.param(1.0, "HIGH", {}, 0.08, 800.0, 8.0, 1, id="ai_size_far_above_cap_clamped"),
    pytest.param(0.08, "HIGH", {}, 0.08, 800.0, 8.0, 0, id="ai_size_exactly_at_cap"),
    pytest.param(0.05, "HIGH", {}, 0.05, 500.0, 5.0, 0, id="ai_size_within_cap"),
    pytest.param(None, "HIGH", {"POSITION_SIZE_FALLBACK_HIGH": 0.06}, 0.06, 600.0, 6.0, 0, id="missing_size_high_fallback"),
    pytest.param(None, "MEDIUM", {}, 0.02, 200.0, 2.0, 0, id="missing_size_medium_fallback"),
    pytest.param(None, "LOW", {}, 0.01, 100.0, 1.0, 0, id="missing_size_low_fallback"),
    pytest.param(None, "UNKNOWN", {}, 0.02, 200.0, 2.0, 0, id="unknown_confidence_medium_fallback"),
    pytest.param(math.nan, "UNKNOWN", {"POSITION_SIZE_FALLBACK_MEDIUM": 0.025}, 0.025, 250.0, 2.5, 0, id="nan_size_medium_fallback"),
    pytest.param(0.0, "HIGH", {}, 0.03, 300.0, 3.0, 0, id="zero_size_falls_back_silently"),
    pytest.param(-0.5, "HIGH", {}, 0.03, 300.0, 3.0, 0, id="negative_size_falls_back_silently"),
    pytest.param(math.inf, "HIGH", {}, 0.03, 300.0, 3.0, 0, id="infinite_size_falls_back_silently"),
]


@pytest.mark.parametrize(
    ("position_size", "confidence", "overrides", "expected_pct", "expected_quote", "expected_quantity", "expected_clamps"),
    SIZING_CASES,
)
def test_position_size_resolution_matrix(
    position_size: float | None,
    confidence: str,
    overrides: dict[str, Any],
    expected_pct: float,
    expected_quote: float,
    expected_quantity: float,
    expected_clamps: int,
) -> None:
    """Requested sizes are clamped to the regime cap; invalid ones use a confidence fallback."""
    manager = risk_service(**overrides)
    assessment = entry_parameters(manager, position_size=position_size, confidence=confidence)

    assert assessment.size_pct == pytest.approx(expected_pct)
    assert assessment.quote_amount == pytest.approx(expected_quote)
    assert assessment.quantity == pytest.approx(expected_quantity)
    assert assessment.entry_fee == pytest.approx(expected_quote * FEE_PERCENT)
    assert len(drain_frictions(manager)) == expected_clamps


def test_invalid_confidence_fallback_config_warns_and_uses_medium() -> None:
    """A non-finite configured fallback is replaced by the MEDIUM default."""
    logger = MagicMock()
    manager = RiskManager(logger=logger, config=make_config(POSITION_SIZE_FALLBACK_LOW=math.inf))
    assessment = entry_parameters(manager, position_size=None, confidence="LOW")

    assert assessment.size_pct == pytest.approx(0.02)
    assert drain_frictions(manager) == []
    logger.warning.assert_any_call(
        "Configured fallback position size for %s confidence is invalid, using MEDIUM fallback",
        "LOW",
    )


DISTANCE_GUARDS = [
    pytest.param("BUY", 80.0, "sl_distance_max", "LONG", 0.20, 0.10, 90.0, "clamped to max 10%", id="long_sl_far"),
    pytest.param("SELL", 120.0, "sl_distance_max", "SHORT", 0.20, 0.10, 110.0, "clamped to max 10%", id="short_sl_far"),
    pytest.param("BUY", 99.8, "sl_distance_min", "LONG", 0.002, 0.01, 99.0, "expanded to min 1%", id="long_sl_tight"),
    pytest.param("SELL", 100.2, "sl_distance_min", "SHORT", 0.002, 0.01, 101.0, "expanded to min 1%", id="short_sl_tight"),
]


@pytest.mark.parametrize(
    ("signal", "stop_loss", "guard_type", "direction", "suggested_pct", "corrected_pct", "final_sl", "detail_tail"),
    DISTANCE_GUARDS,
)
def test_sl_distance_guard_matrix(
    signal: str,
    stop_loss: float,
    guard_type: str,
    direction: str,
    suggested_pct: float,
    corrected_pct: float,
    final_sl: float,
    detail_tail: str,
) -> None:
    """SL distances outside the [1%, 10%] band are corrected and reported."""
    manager = risk_service()
    assessment = entry_parameters(manager, signal=signal, stop_loss=stop_loss)

    drained = drain_frictions(manager)
    assert [item["guard_type"] for item in drained] == [guard_type]
    friction = drained[0]
    assert FRICTION_KEYS.issubset(friction)
    assert type(friction["guard_type"]) is str
    assert type(friction["direction"]) is str
    assert type(friction["detail"]) is str
    assert friction["direction"] == direction
    assert friction["suggested_sl_pct"] == pytest.approx(suggested_pct)
    assert friction["corrected_sl_pct"] == pytest.approx(corrected_pct)
    assert friction["current_price"] == pytest.approx(PRICE)
    assert friction["volatility_level"] in ("HIGH", "MEDIUM", "LOW")
    assert friction["detail"].endswith(detail_tail)

    assert assessment.direction == direction
    assert assessment.stop_loss == pytest.approx(final_sl)
    assert assessment.sl_distance_pct == pytest.approx(corrected_pct)


@pytest.mark.parametrize("stop_loss", [90.0, 99.0], ids=["at_ten_percent", "at_one_percent"])
def test_sl_distance_exact_boundaries_are_not_guarded(stop_loss: float) -> None:
    """Distances exactly at the 10% cap and exactly at the 1% floor pass untouched."""
    manager = risk_service()
    assessment = entry_parameters(manager, stop_loss=stop_loss)

    assert drain_frictions(manager) == []
    assert assessment.stop_loss == pytest.approx(stop_loss)
    assert assessment.sl_distance_pct == pytest.approx(abs(PRICE - stop_loss) / PRICE)


SIDE_GUARDS = [
    pytest.param("BUY", 102.0, None, 1.0, "sl_below_entry", "LONG", "suggested_sl", "dynamic_sl", 102.0, 98.0, 98.0, 104.0, id="long_sl_above_entry"),
    pytest.param("SELL", 95.0, None, 1.0, "sl_below_entry", "SHORT", "suggested_sl", "dynamic_sl", 95.0, 102.0, 102.0, 96.0, id="short_sl_below_entry"),
    pytest.param("BUY", None, 95.0, 2.0, "tp_below_entry", "LONG", "suggested_tp", "dynamic_tp", 95.0, 108.0, 96.0, 108.0, id="long_tp_below_entry"),
    pytest.param("SELL", None, 105.0, 2.0, "tp_below_entry", "SHORT", "suggested_tp", "dynamic_tp", 105.0, 92.0, 104.0, 92.0, id="short_tp_above_entry"),
]


@pytest.mark.parametrize(
    ("signal", "stop_loss", "take_profit", "atr", "guard_type", "direction", "suggested_key", "dynamic_key", "suggested", "dynamic", "expected_sl", "expected_tp"),
    SIDE_GUARDS,
)
def test_invalid_side_guard_matrix(
    signal: str,
    stop_loss: float | None,
    take_profit: float | None,
    atr: float,
    guard_type: str,
    direction: str,
    suggested_key: str,
    dynamic_key: str,
    suggested: float,
    dynamic: float,
    expected_sl: float,
    expected_tp: float,
) -> None:
    """SL/TP on the wrong side of entry is replaced by the dynamic level and reported."""
    manager = risk_service()
    assessment = entry_parameters(
        manager,
        signal=signal,
        stop_loss=stop_loss,
        take_profit=take_profit,
        market_conditions=MarketConditions(atr=atr, atr_percentage=atr),
    )

    drained = drain_frictions(manager)
    assert [item["guard_type"] for item in drained] == [guard_type]
    friction = drained[0]
    assert FRICTION_KEYS.issubset(friction)
    assert type(friction["guard_type"]) is str
    assert type(friction["direction"]) is str
    assert type(friction["detail"]) is str
    assert friction["direction"] == direction
    assert friction[suggested_key] == pytest.approx(suggested)
    assert friction[dynamic_key] == pytest.approx(dynamic)
    assert friction["current_price"] == pytest.approx(PRICE)
    assert friction["volatility_level"] in ("HIGH", "MEDIUM", "LOW")
    assert friction["detail"].endswith("using dynamic")
    assert "$100.00" in friction["detail"]

    assert assessment.direction == direction
    assert assessment.stop_loss == pytest.approx(expected_sl)
    assert assessment.take_profit == pytest.approx(expected_tp)


VOLATILITY_CASES = [
    pytest.param(5.0, 80.0, "sl_distance_max", "HIGH", "conservative", id="atr_pct_five"),
    pytest.param(4.0, 80.0, "sl_distance_max", "HIGH", "conservative", id="atr_pct_at_high_boundary"),
    pytest.param(3.0, 80.0, "sl_distance_max", "MEDIUM", "neutral", id="atr_pct_at_medium_boundary"),
    pytest.param(2.0, 80.0, "sl_distance_max", "MEDIUM", "neutral", id="atr_pct_two"),
    pytest.param(1.5, 99.8, "sl_distance_min", "MEDIUM", "neutral", id="atr_pct_at_low_boundary"),
    pytest.param(1.0, 99.8, "sl_distance_min", "LOW", "neutral", id="atr_pct_one"),
]


@pytest.mark.parametrize(
    ("atr_percentage", "stop_loss", "guard_type", "volatility", "profile"),
    VOLATILITY_CASES,
)
def test_volatility_label_matrix(
    atr_percentage: float,
    stop_loss: float,
    guard_type: str,
    volatility: str,
    profile: str,
) -> None:
    """Frictions and the assessment carry the volatility label and regime of the entry."""
    manager = risk_service()
    assessment = entry_parameters(
        manager,
        stop_loss=stop_loss,
        market_conditions=MarketConditions(atr=1.0, atr_percentage=atr_percentage),
    )

    drained = drain_frictions(manager)
    assert [item["guard_type"] for item in drained] == [guard_type]
    assert drained[0]["volatility_level"] == volatility
    assert assessment.volatility_level == volatility
    assert assessment.regime_profile == profile


@pytest.mark.parametrize(
    ("market_conditions", "expected_sl", "expected_tp"),
    [
        pytest.param(None, 96.0, 108.0, id="no_conditions"),
        pytest.param(MarketConditions(), 96.0, 108.0, id="zero_atr"),
        pytest.param(MarketConditions(atr=-5.0, atr_percentage=0.0), 96.0, 108.0, id="negative_atr"),
        pytest.param(MarketConditions(atr=0.0, atr_percentage=-2.0), 96.0, 108.0, id="negative_atr_percentage"),
    ],
)
def test_dynamic_sl_tp_fall_back_to_two_percent_of_price(
    market_conditions: MarketConditions | None, expected_sl: float, expected_tp: float
) -> None:
    """Missing, zero or negative ATR falls back to 2% of price for both dynamic levels."""
    manager = risk_service()
    assessment = entry_parameters(manager, market_conditions=market_conditions)

    assert assessment.stop_loss == pytest.approx(expected_sl)
    assert assessment.take_profit == pytest.approx(expected_tp)
    assert assessment.sl_distance_pct == pytest.approx(0.04)
    assert assessment.tp_distance_pct == pytest.approx(0.08)
    assert assessment.rr_ratio == pytest.approx(2.0)
    assert assessment.volatility_level == "MEDIUM"
    assert assessment.regime_profile == "neutral"
    assert drain_frictions(manager) == []


def test_ai_levels_pass_through_unchanged_when_valid() -> None:
    """AI-provided SL/TP inside the distance floors are used verbatim."""
    manager = risk_service()
    assessment = entry_parameters(
        manager, stop_loss=95.0, take_profit=110.0, market_conditions=MarketConditions()
    )

    assert assessment.stop_loss == pytest.approx(95.0)
    assert assessment.take_profit == pytest.approx(110.0)
    assert drain_frictions(manager) == []


def test_rr_ratio_is_tp_distance_over_sl_distance() -> None:
    """R/R is derived from the final SL/TP distances, not from the AI levels."""
    manager = risk_service()
    assessment = entry_parameters(manager, stop_loss=95.0, take_profit=110.0)

    assert assessment.direction == "LONG"
    assert assessment.entry_price == pytest.approx(PRICE)
    assert assessment.sl_distance_pct == pytest.approx(0.05)
    assert assessment.tp_distance_pct == pytest.approx(0.10)
    assert assessment.rr_ratio == pytest.approx(2.0)
    assert drain_frictions(manager) == []


def test_sl_at_entry_expands_to_the_min_floor() -> None:
    """An SL exactly at entry is expanded to 1%, so R/R uses the corrected level."""
    manager = risk_service()
    assessment = entry_parameters(
        manager, stop_loss=PRICE, market_conditions=MarketConditions(atr=1.0, atr_percentage=1.0)
    )

    friction = drain_frictions(manager)[0]
    assert friction["guard_type"] == "sl_distance_min"
    assert friction["suggested_sl_pct"] == pytest.approx(0.0)
    assert friction["corrected_sl_pct"] == pytest.approx(0.01)
    assert assessment.stop_loss == pytest.approx(99.0)
    assert assessment.sl_distance_pct == pytest.approx(0.01)
    assert assessment.tp_distance_pct == pytest.approx(0.04)
    assert assessment.rr_ratio == pytest.approx(4.0)


def test_short_take_profit_above_entry_leaves_sub_one_rr() -> None:
    """A SHORT whose TP sits above the SL distance ends up with R/R below 1."""
    manager = risk_service()
    assessment = entry_parameters(
        manager,
        signal="SELL",
        stop_loss=110.0,
        take_profit=105.0,
        market_conditions=MarketConditions(atr=2.0, atr_percentage=2.0),
    )

    drained = drain_frictions(manager)
    assert [item["guard_type"] for item in drained] == ["tp_below_entry"]
    assert drained[0]["suggested_tp"] == pytest.approx(105.0)
    assert drained[0]["dynamic_tp"] == pytest.approx(92.0)

    assert assessment.direction == "SHORT"
    assert assessment.stop_loss == pytest.approx(110.0)
    assert assessment.take_profit == pytest.approx(92.0)
    assert assessment.sl_distance_pct == pytest.approx(0.10)
    assert assessment.tp_distance_pct == pytest.approx(0.08)
    assert assessment.rr_ratio == pytest.approx(0.8)


INVALID_SL_CASES = [
    pytest.param(math.nan, 96.0, [], id="nan_sl_uses_dynamic"),
    pytest.param(float("inf"), 90.0, ["sl_distance_max"], id="infinite_sl_is_clamped"),
    pytest.param(-10.0, 96.0, [], id="negative_sl_uses_dynamic"),
]


@pytest.mark.parametrize(("stop_loss", "expected_sl", "guards"), INVALID_SL_CASES)
def test_invalid_ai_stop_loss_matrix(
    stop_loss: float, expected_sl: float, guards: list[str]
) -> None:
    """NaN and negative SLs count as absent and take the dynamic 2% ATR level, while an
    infinite SL survives the positivity check and is clamped by the distance guard."""
    manager = risk_service()
    assessment = entry_parameters(manager, stop_loss=stop_loss)

    assert [item["guard_type"] for item in drain_frictions(manager)] == guards
    assert math.isfinite(assessment.stop_loss)
    assert math.isfinite(assessment.take_profit)
    assert assessment.stop_loss == pytest.approx(expected_sl)
    assert assessment.stop_loss > 0


def test_zero_and_negative_capital_exposure() -> None:
    """Zero capital scales the position to nothing instead of dividing by zero; a corrupt
    negative balance is rejected by contract instead of becoming a negative order (F14)."""
    manager = risk_service()
    flat = entry_parameters(manager, capital=0.0)

    assert flat.size_pct == pytest.approx(0.03)
    assert flat.quantity == 0.0
    assert flat.quote_amount == 0.0
    assert drain_frictions(manager) == []

    for capital in (-1000.0, -0.01, math.nan):
        with pytest.raises(ValueError, match="capital must be a non-negative finite number"):
            entry_parameters(manager, capital=capital)


def test_extreme_distances_are_guarded_into_a_finite_rr() -> None:
    """A 0.01% SL and a 100% TP are re-anchored to the 1% floor and the 50% cap, leaving a
    finite R/R far above the entry floor."""
    manager = risk_service()
    assessment = entry_parameters(
        manager,
        stop_loss=99.99,
        take_profit=200.0,
        market_conditions=MarketConditions(atr=0.5, atr_percentage=0.5),
    )

    assert [item["guard_type"] for item in drain_frictions(manager)] == [
        "sl_distance_min",
        "tp_distance_max",
    ]
    assert assessment.stop_loss == pytest.approx(99.0)
    assert assessment.take_profit == pytest.approx(150.0)
    assert math.isfinite(assessment.rr_ratio)
    assert assessment.rr_ratio == pytest.approx(50.0)
    assert assessment.rr_ratio >= 0


def test_take_profit_and_size_exact_boundaries_are_not_guarded() -> None:
    """A TP exactly 50% away and a size exactly at the profile cap pass untouched."""
    manager = risk_service()
    assessment = entry_parameters(
        manager, stop_loss=90.0, take_profit=150.0, position_size=NEUTRAL_CAP
    )

    assert drain_frictions(manager) == []
    assert assessment.take_profit == pytest.approx(150.0)
    assert assessment.size_pct == pytest.approx(NEUTRAL_CAP)


def test_non_positive_entry_price_is_rejected() -> None:
    """A zero, negative or non-finite price is refused by contract instead of dividing by
    it (F13 in ``docs/SRC_AUDIT_FINDINGS.md``)."""
    manager = risk_service()

    for price in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError, match="current_price must be a positive finite number"):
            entry_parameters(manager, current_price=price)


def test_entry_direction_maps_both_entry_aliases_and_refuses_close_signals() -> None:
    """BUY/LONG open long and SELL/SHORT open short; a CLOSE/UPDATE signal is refused as an
    entry instead of silently receiving SHORT risk parameters (F8)."""
    manager = risk_service()

    assert [
        manager.validate_signal(signal)
        for signal in ("BUY", "SELL", "CLOSE", "CLOSE_LONG", "CLOSE_SHORT", "HOLD")
    ] == [True, True, True, True, True, False]

    assert entry_parameters(manager, signal="BUY").direction == "LONG"
    assert entry_parameters(manager, signal="LONG").direction == "LONG"

    for signal in ("SELL", "SHORT"):
        assert entry_parameters(manager, signal=signal).direction == "SHORT"

    for signal in ("CLOSE", "CLOSE_LONG", "CLOSE_SHORT", "UPDATE", "HOLD"):
        with pytest.raises(ValueError, match="is not an entry signal"):
            entry_parameters(manager, signal=signal)


@pytest.mark.parametrize(
    ("signal", "confidence", "stop_loss", "take_profit"),
    [
        ("BUY", "HIGH", 95.0, 110.0),
        ("SELL", "LOW", 105.0, 90.0),
    ],
    ids=["long_high_confidence", "short_low_confidence"],
)
async def test_size_clamp_friction_reaches_vector_memory(
    signal: str, confidence: str, stop_loss: float, take_profit: float
) -> None:
    """A clamped size is stored as a blocked trade carrying the full friction payload, and
    the payload's R/R pair is the trade's own R/R against the effective entry floor (F10)."""
    strategy = make_strategy(stop_loss=stop_loss, take_profit=take_profit, position_size=0.30)
    decision = await run_entry(
        strategy,
        signal=signal,
        confidence=confidence,
        stop_loss=stop_loss,
        take_profit=take_profit,
        position_size=0.30,
    )

    assert decision.action == signal
    assert decision.position_size == pytest.approx(NEUTRAL_CAP)
    assert strategy.current_position is not None
    assert strategy.current_position.size == pytest.approx(8.0)
    assert strategy.current_position.quote_amount == pytest.approx(800.0)

    calls = blocked_calls(strategy)
    assert len(calls) == 1
    payload = calls[0]
    assert payload["guard_type"] == "position_size_clamp"
    assert payload["direction"] == "N/A"
    assert payload["confidence"] == confidence
    assert payload["current_price"] == pytest.approx(PRICE)
    assert payload["suggested_rr"] == pytest.approx(2.0)
    assert payload["required_rr"] == pytest.approx(1.5)
    assert payload["suggested_sl_pct"] == pytest.approx(0.05)
    assert payload["suggested_tp_pct"] == pytest.approx(0.10)
    assert payload["suggested_sl"] == pytest.approx(stop_loss)
    assert payload["suggested_tp"] == pytest.approx(take_profit)
    assert payload["volatility_level"] in ("HIGH", "MEDIUM", "LOW")
    assert payload["reasoning_snippet"] == "Position size 30.0% clamped to max 8.0%"
    assert payload["metadata"]["friction"]["guard_type"] == "position_size_clamp"
    assert payload["metadata"]["friction"]["suggested_size"] == pytest.approx(0.30)


async def test_sl_distance_friction_reaches_vector_memory() -> None:
    """SL clamping is stored with the suggested pct, the corrected level and the volatility;
    the absolute ``suggested_sl`` is the level the AI requested, not the corrected one (F10)."""
    strategy = make_strategy(stop_loss=80.0, take_profit=130.0)
    decision = await run_entry(strategy, stop_loss=80.0, take_profit=130.0)

    assert decision.action == "BUY"
    assert decision.stop_loss == pytest.approx(90.0)
    assert decision.take_profit == pytest.approx(130.0)

    calls = blocked_calls(strategy)
    assert len(calls) == 1
    payload = calls[0]
    assert payload["guard_type"] == "sl_distance_max"
    assert payload["direction"] == "LONG"
    assert {
        "suggested_rr",
        "suggested_sl_pct",
        "suggested_tp_pct",
        "reasoning_snippet",
    }.issubset(payload)
    assert payload["suggested_rr"] == pytest.approx(3.0)
    assert payload["required_rr"] == pytest.approx(1.5)
    assert payload["suggested_sl_pct"] == pytest.approx(0.20)
    assert payload["suggested_tp_pct"] == pytest.approx(0.30)
    assert payload["suggested_sl"] == pytest.approx(80.0)
    assert payload["suggested_tp"] == pytest.approx(130.0)
    assert payload["volatility_level"] in ("HIGH", "MEDIUM", "LOW")
    assert payload["reasoning_snippet"] == "SL distance 20.0% clamped to max 10%"
    assert payload["metadata"]["friction"]["corrected_sl_pct"] == pytest.approx(0.10)


async def test_multiple_frictions_are_all_stored() -> None:
    """Every drained guard report becomes its own blocked-trade event, in guard order."""
    strategy = make_strategy(stop_loss=80.0, take_profit=120.0, position_size=0.30)
    decision = await run_entry(
        strategy, stop_loss=80.0, take_profit=120.0, position_size=0.30
    )

    assert decision.action == "BUY"
    calls = blocked_calls(strategy)
    assert [payload["guard_type"] for payload in calls] == ["sl_distance_max", "position_size_clamp"]
    assert calls[0]["direction"] == "LONG"
    assert calls[1]["direction"] == "N/A"
    assert calls[0]["metadata"]["friction"]["guard_type"] == "sl_distance_max"
    assert calls[1]["metadata"]["friction"]["guard_type"] == "position_size_clamp"


RR_GATE_BLOCKS = [
    pytest.param("BUY", 95.0, 100.0, "LONG", ["tp_below_entry", "rr_minimum"], 1.6, "I think this will rebound strongly from support", id="long_tp_at_entry"),
    pytest.param("SELL", 105.0, 96.0, "SHORT", ["rr_minimum"], 0.8, "Weak setup", id="short_rr_below_floor"),
]


@pytest.mark.parametrize(
    ("signal", "stop_loss", "take_profit", "direction", "guards", "expected_rr", "reasoning"),
    RR_GATE_BLOCKS,
)
async def test_rr_gate_blocks_entry_below_floor(
    signal: str,
    stop_loss: float,
    take_profit: float,
    direction: str,
    guards: list[str],
    expected_rr: float,
    reasoning: str,
) -> None:
    """An R/R under min(brain borderline, config floor) returns HOLD and stores rr_minimum."""
    strategy = make_strategy(
        stop_loss=stop_loss,
        take_profit=take_profit,
        rr_borderline_min=3.0,
        min_rr_entry=3.0,
    )
    decision = await run_entry(
        strategy,
        signal=signal,
        stop_loss=stop_loss,
        take_profit=take_profit,
        reasoning=reasoning,
    )

    assert decision.action == "HOLD"
    assert "Entry blocked" in decision.reasoning
    assert "below minimum 3.0" in decision.reasoning
    assert reasoning in decision.reasoning
    assert strategy.current_position is None

    calls = blocked_calls(strategy)
    assert [payload["guard_type"] for payload in calls] == guards
    blocked = blocked_calls(strategy, "rr_minimum")
    assert len(blocked) == 1
    assert blocked[0]["direction"] == direction
    assert blocked[0]["required_rr"] == pytest.approx(3.0)
    assert blocked[0]["suggested_rr"] == pytest.approx(expected_rr)
    assert blocked[0]["reasoning_snippet"] == reasoning[:200]
    assert "metadata" not in blocked[0]


RR_GATE_PASSES = [
    pytest.param(95.0, 115.0, 1.5, 1.0, 3.0, id="above_floor"),
    pytest.param(95.0, 105.0, 1.0, 1.0, 1.0, id="exactly_at_floor"),
    pytest.param(95.0, 108.0, 1.5, 1.0, 1.6, id="brain_raises_the_bar_above_the_config_floor"),
]


@pytest.mark.parametrize(
    ("stop_loss", "take_profit", "brain_rr", "config_rr", "expected_rr"),
    RR_GATE_PASSES,
)
async def test_rr_gate_allows_entry_at_or_above_floor(
    stop_loss: float,
    take_profit: float,
    brain_rr: float,
    config_rr: float,
    expected_rr: float,
) -> None:
    """R/R at or above max(brain borderline, config floor) opens without an rr_minimum
    event — the config value is the floor and the brain may only raise it (F12 in
    ``docs/SRC_AUDIT_FINDINGS.md``)."""
    strategy = make_strategy(
        stop_loss=stop_loss,
        take_profit=take_profit,
        rr_borderline_min=brain_rr,
        min_rr_entry=config_rr,
    )
    decision = await run_entry(strategy, stop_loss=stop_loss, take_profit=take_profit)

    assert decision.action == "BUY"
    assert blocked_calls(strategy, "rr_minimum") == []
    assert decision.indicators_json["rr_ratio_at_entry"] == pytest.approx(expected_rr)
    assert strategy.current_position is not None
    assert strategy.current_position.rr_ratio_at_entry == pytest.approx(expected_rr)


async def test_brain_threshold_can_only_raise_the_config_floor() -> None:
    """The brain's rr_borderline_min raises the bar (1.6 R/R opens under a 1.0 floor and is
    blocked under 3.0) and can never lower MIN_RR_ENTRY (0.5 vs 3.0 still blocks) (F12)."""
    permissive = make_strategy(stop_loss=95.0, take_profit=108.0, rr_borderline_min=1.5, min_rr_entry=1.0)
    assert (await run_entry(permissive, stop_loss=95.0, take_profit=108.0)).action == "BUY"

    raised = make_strategy(stop_loss=95.0, take_profit=108.0, rr_borderline_min=3.0, min_rr_entry=1.0)
    raised_decision = await run_entry(raised, stop_loss=95.0, take_profit=108.0)

    assert raised_decision.action == "HOLD"
    assert blocked_calls(raised, "rr_minimum")[0]["required_rr"] == pytest.approx(3.0)

    lowered = make_strategy(stop_loss=95.0, take_profit=108.0, rr_borderline_min=0.5, min_rr_entry=3.0)
    lowered_decision = await run_entry(lowered, stop_loss=95.0, take_profit=108.0)

    assert lowered_decision.action == "HOLD"
    assert blocked_calls(lowered, "rr_minimum")[0]["required_rr"] == pytest.approx(3.0)


async def test_zero_configured_min_rr_disables_the_config_floor() -> None:
    """MIN_RR_ENTRY=0 is an explicit "no config floor" instead of silently becoming 1.0: the
    0.25 R/R trade opens, and only the brain's bar can still block it (F11)."""
    unguarded = make_strategy(
        stop_loss=98.0, take_profit=100.5, brain_thresholds={}, min_rr_entry=0.0
    )
    decision = await run_entry(unguarded, stop_loss=98.0, take_profit=100.5)

    assert decision.action == "BUY"
    assert blocked_calls(unguarded, "rr_minimum") == []

    brain_gated = make_strategy(
        stop_loss=98.0, take_profit=100.5, rr_borderline_min=2.0, min_rr_entry=0.0
    )
    gated_decision = await run_entry(brain_gated, stop_loss=98.0, take_profit=100.5)

    assert gated_decision.action == "HOLD"
    blocked = blocked_calls(brain_gated, "rr_minimum")
    assert blocked[0]["required_rr"] == pytest.approx(2.0)
    assert blocked[0]["suggested_rr"] == pytest.approx(0.25)


BRAIN_THRESHOLD_CASES = [
    pytest.param({}, id="missing_key"),
    pytest.param({"rr_borderline_min": "not-a-number"}, id="garbage_value"),
    pytest.param({"rr_borderline_min": None}, id="none_value"),
]


@pytest.mark.parametrize("thresholds", BRAIN_THRESHOLD_CASES)
async def test_unusable_brain_thresholds_fall_back_to_the_config_floor(
    thresholds: dict[str, Any],
) -> None:
    """A missing or non-numeric rr_borderline_min cannot disable the R/R gate: the config
    floor applies."""
    strategy = make_strategy(
        stop_loss=98.0,
        take_profit=100.5,
        brain_thresholds=thresholds,
        min_rr_entry=2.0,
    )
    decision = await run_entry(strategy, stop_loss=98.0, take_profit=100.5)

    assert decision.action == "HOLD"
    assert blocked_calls(strategy, "rr_minimum")[0]["required_rr"] == pytest.approx(2.0)


async def test_friction_buffer_does_not_leak_into_the_next_entry() -> None:
    """The strategy drains the buffer per entry, so a clamped entry does not re-store its
    friction on the following clean one."""
    strategy = make_strategy(position_size=0.30)
    clamped = await run_entry(strategy, position_size=0.30)

    assert clamped.action == "BUY"
    assert [payload["guard_type"] for payload in blocked_calls(strategy)] == [
        "position_size_clamp"
    ]

    clean = await run_entry(strategy, position_size=0.05)
    assert clean.action == "BUY"
    assert [payload["guard_type"] for payload in blocked_calls(strategy)] == [
        "position_size_clamp"
    ]


STORAGE_OUTAGES = [
    pytest.param(1.0, 95.0, 110.0, 0.30, "BUY", True, "Failed to store friction event from RiskManager", id="friction_storage_outage"),
    pytest.param(3.0, 95.0, 104.0, 0.05, "HOLD", False, "Failed to store blocked trade event", id="blocked_trade_storage_outage"),
]


@pytest.mark.parametrize(
    ("min_rr_entry", "stop_loss", "take_profit", "position_size", "expected_action", "position_opened", "expected_warning"),
    STORAGE_OUTAGES,
)
async def test_storage_outage_never_breaks_the_trading_loop(
    min_rr_entry: float,
    stop_loss: float,
    take_profit: float,
    position_size: float,
    expected_action: str,
    position_opened: bool,
    expected_warning: str,
) -> None:
    """A raising store_blocked_trade is logged and the decision flow continues."""
    strategy = make_strategy(
        stop_loss=stop_loss,
        take_profit=take_profit,
        position_size=position_size,
        min_rr_entry=min_rr_entry,
    )
    strategy.brain_service.vector_memory.store_blocked_trade.side_effect = RuntimeError("DB down")

    decision = await run_entry(
        strategy,
        stop_loss=stop_loss,
        take_profit=take_profit,
        position_size=position_size,
    )

    assert decision is not None
    assert decision.action == expected_action
    assert (strategy.current_position is not None) is position_opened
    assert strategy.brain_service.vector_memory.store_blocked_trade.call_count == 1
    strategy.logger.warning.assert_any_call(expected_warning, exc_info=True)


async def test_entry_offloads_slow_blocked_trade_storage() -> None:
    """A slow synchronous store hook runs on a worker thread, so the event loop keeps
    ticking while the Chroma upsert completes (F9 in ``docs/SRC_AUDIT_FINDINGS.md``)."""
    strategy = make_strategy(position_size=0.30)
    delay = 0.1
    stored: list[tuple[str, int]] = []
    ticks = 0

    def slow_store(**payload: Any) -> bool:
        time.sleep(delay)
        stored.append((payload["guard_type"], threading.get_ident()))
        return True

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.005)
            ticks += 1

    assert not inspect.iscoroutinefunction(VectorMemoryService.store_blocked_trade)
    strategy.brain_service.vector_memory.store_blocked_trade = slow_store
    ticker_task = asyncio.ensure_future(ticker())
    try:
        decision = await run_entry(strategy, position_size=0.30)
    finally:
        ticker_task.cancel()
        with suppress(asyncio.CancelledError):
            await ticker_task

    assert [guard for guard, _thread in stored] == ["position_size_clamp"]
    assert stored[0][1] != threading.get_ident()
    assert ticks >= 2
    assert decision.action == "BUY"
    assert strategy.current_position is not None


@pytest.mark.parametrize(
    ("respond", "expected_warning"),
    [
        (no_feedback, None),
        (failing_feedback, "Failed to fetch blocked trade feedback: %s"),
    ],
    ids=["no_blocked_trades", "vector_store_outage"],
)
def test_brain_context_empty_state(respond: Callable[[], str], expected_warning: str | None) -> None:
    """With zero closed trades and no rejections the brain context stays free of feedback."""
    brain, feedback_calls = empty_state_brain(respond)
    context = brain.get_context(MarketSnapshot(adx=25))

    assert "CRITICAL FEEDBACK" not in context
    assert "System Rejections" not in context
    assert "Trading Brain" not in context
    assert len(feedback_calls) == 1
    assert feedback_calls[0] == {"n": 5, "max_age_hours": 168}
    if expected_warning is None:
        assert brain.logger.warning.call_args_list == []
    else:
        brain.logger.warning.assert_any_call(expected_warning, ANY)


def test_saturated_blocked_trades_still_render_grouped_feedback(
    saturated_vector_memory: VectorMemoryService,
) -> None:
    """55 stored rejections stay addressable, grouped and non-empty in the feedback prompt."""
    service = saturated_vector_memory

    recent = service.get_recent_blocked_trades(n=5)
    assert len(recent) == 5
    assert service.get_blocked_trade_count() >= 50

    feedback = service.get_blocked_trade_feedback(n=5)
    assert len(feedback) > 0
    assert "CRITICAL FEEDBACK" in feedback
    assert "###" in feedback

    full_feedback = service.get_blocked_trade_feedback(n=50)
    assert "### SL Too Far (max 10%)" in full_feedback
    assert "### R/R Minimum Guard" in full_feedback
    assert "### Position Size Clamp" in full_feedback
    assert len(full_feedback) > len(feedback)


def test_brain_context_survives_saturated_feedback(
    saturated_vector_memory: VectorMemoryService,
) -> None:
    """50+ rejections render into the brain context inside the token budget."""
    service = saturated_vector_memory
    vector_memory = MagicMock()
    vector_memory.trade_count = 0
    vector_memory.get_blocked_trade_feedback.side_effect = (
        lambda **kwargs: service.get_blocked_trade_feedback(**kwargs)
    )
    vector_memory.get_context_for_prompt.return_value = ""
    vector_memory.get_relevant_rules.return_value = []
    vector_memory.compute_confidence_stats.return_value = {}
    vector_memory.get_confidence_recommendation.return_value = ""
    vector_memory.get_direction_bias.return_value = None

    brain = TradingBrainService(
        logger=MagicMock(),
        persistence=MagicMock(),
        vector_memory=vector_memory,
    )
    context = brain.get_context(MarketSnapshot(adx=25))

    assert "CRITICAL FEEDBACK" in context
    assert len(context) < 10000


def test_blocked_trade_write_burst_stays_fast_and_complete(vector_memory: VectorMemoryService) -> None:
    """A burst of sequential blocked-trade writes stays fast and keeps every record."""
    start = time.monotonic()
    for index in range(20):
        assert store_block(vector_memory, guard_type=f"concurrent_{index}") is True
    elapsed = time.monotonic() - start

    assert elapsed < 5.0
    assert vector_memory.get_blocked_trade_count() == 20


def test_snippet_is_capped_and_unknown_guards_render(vector_memory: VectorMemoryService) -> None:
    """The store itself truncates a snippet to 200 chars (F15) and unknown guards render
    title-cased."""
    long_reasoning = "Analysis: " + "X" * 2000
    capped = long_reasoning[:200]
    assert (
        store_block(vector_memory, guard_type="custom_new_guard", reasoning_snippet=long_reasoning)
        is True
    )

    stored = vector_memory.get_recent_blocked_trades(n=1)
    assert len(stored) == 1
    assert stored[0]["reasoning_snippet"] == capped
    assert len(stored[0]["reasoning_snippet"]) == 200

    assert store_block(vector_memory, guard_type="custom_new_guard", reasoning_snippet="") is True
    feedback = vector_memory.get_blocked_trade_feedback(n=5)
    assert "### Custom New Guard (2 recent):" in feedback
    assert capped in feedback
    assert long_reasoning not in feedback


def test_uninitialized_vector_memory_degrades_to_no_ops() -> None:
    """A service whose collections never came up stores nothing and reports empty instead
    of raising into the trading loop."""
    service = VectorMemoryService(logger=MagicMock(), chroma_client=None, embedding_model=None)

    assert store_block(service) is False
    assert service.get_recent_blocked_trades(n=5) == []
    assert service.get_blocked_trade_feedback(n=5) == ""
    assert service.get_blocked_trade_count() == 0
    service.logger.warning.assert_called_once_with(
        "VectorMemoryService not initialized, cannot store blocked trade."
    )
