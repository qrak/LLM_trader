"""Focused tests for RiskManager position sizing guardrails."""

from math import nan
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.managers.risk_manager import RiskManager
from src.trading.data_models import MarketConditions


def _make_config(
    max_position_size: float = 0.10,
    fallback_low: float = 0.01,
    fallback_medium: float = 0.02,
    fallback_high: float = 0.03,
) -> SimpleNamespace:
    return SimpleNamespace(
        MAX_POSITION_SIZE=max_position_size,
        POSITION_SIZE_FALLBACK_LOW=fallback_low,
        POSITION_SIZE_FALLBACK_MEDIUM=fallback_medium,
        POSITION_SIZE_FALLBACK_HIGH=fallback_high,
        TRANSACTION_FEE_PERCENT=0.001,
    )


def _calculate_entry(manager: RiskManager, position_size: float | None, confidence: str = "HIGH") -> Any:
    return manager.calculate_entry_parameters(
        signal="BUY",
        current_price=100.0,
        capital=10000.0,
        confidence=confidence,
        stop_loss=95.0,
        take_profit=110.0,
        position_size=position_size,
        market_conditions=MarketConditions(atr=2.0, atr_percentage=2.0),
    )


def test_ai_position_size_is_clamped_to_configured_cap() -> None:
    logger = MagicMock()
    manager = RiskManager(logger=logger, config=_make_config(max_position_size=0.10))

    assessment = _calculate_entry(manager, position_size=0.50)

    assert assessment.size_pct == pytest.approx(0.10)
    assert assessment.quote_amount == pytest.approx(1000.0)
    assert assessment.quantity == pytest.approx(10.0)
    logger.warning.assert_any_call(
        "%s position size %.2f%% exceeds cap %.2f%%, clamping",
        "AI",
        50.0,
        10.0,
    )


def test_missing_ai_position_size_uses_configured_confidence_fallback() -> None:
    manager = RiskManager(logger=MagicMock(), config=_make_config(fallback_high=0.06))

    assessment = _calculate_entry(manager, position_size=None, confidence="HIGH")

    assert assessment.size_pct == pytest.approx(0.06)
    assert assessment.quote_amount == pytest.approx(600.0)
    assert assessment.quantity == pytest.approx(6.0)


def test_invalid_ai_position_size_uses_medium_fallback_for_unknown_confidence() -> None:
    manager = RiskManager(logger=MagicMock(), config=_make_config(fallback_medium=0.025))

    assessment = _calculate_entry(manager, position_size=nan, confidence="UNKNOWN")

    assert assessment.size_pct == pytest.approx(0.025)
    assert assessment.quote_amount == pytest.approx(250.0)
