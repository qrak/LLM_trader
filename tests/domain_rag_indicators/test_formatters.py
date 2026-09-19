"""Dense domain tests for the formatters that build the analysis prompt.

Covers the EV framework section, the market period-metrics rendering and the
technical price-action section, including the numeric boundaries of each.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.analyzer.formatters.ev_formatter import EVFrameworkFormatter
from src.analyzer.formatters.market_formatter import MarketFormatter
from src.analyzer.formatters.technical_formatter import TechnicalFormatter
from src.trading.regime_risk_profile import RegimeRiskProfile, RegimeRiskProfileSelector

PARTIAL_30D = {
    "30D": {
        "metrics": {
            "period": "30D (Partial)",
            "avg_price": 77123.66,
            "lowest_price": 74135.99,
            "highest_price": 79515.00,
            "price_change_percent": 2.08416868,
            "total_volume": 60209.40,
        },
        "indicator_changes": {},
    }
}

FULL_7D = {
    "7D": {
        "metrics": {
            "avg_price": 77311.09,
            "lowest_price": 74922.58,
            "highest_price": 79505.79,
            "price_change_percent": -2.19448997,
            "total_volume": 33472.38,
        },
        "indicator_changes": {},
    }
}

NO_CHANGE_1H = {
    "1H": {
        "metrics": {
            "lowest_price": 100.0,
            "highest_price": 200.0,
            "price_change_percent": None,
            "total_volume": 0,
        },
        "indicator_changes": {},
    }
}


def candles(
    opens: list[float],
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[float] | None = None,
) -> np.ndarray:
    """OHLCV matrix in the column order the analyzer stores."""
    open_array = np.asarray(opens, dtype=float)
    close_array = np.asarray(closes, dtype=float)
    high_array = np.asarray(highs, dtype=float) if highs is not None else open_array + 1.0
    low_array = np.asarray(lows, dtype=float) if lows is not None else open_array - 0.5
    volume_array = (
        np.asarray(volumes, dtype=float) if volumes is not None else np.full(len(open_array), 1000.0)
    )
    timestamps = np.arange(len(open_array)) * 60000
    return np.column_stack((timestamps, open_array, high_array, low_array, close_array, volume_array))


def price_action_context(ohlcv) -> MagicMock:
    """Minimal analysis context exposing only what the section reads."""
    context = MagicMock()
    context.current_price = 100.0
    context.ohlcv_candles = ohlcv
    return context


@pytest.fixture
def ev_formatter(config):
    return EVFrameworkFormatter(config)


@pytest.fixture
def market_formatter(format_utils):
    return MarketFormatter(format_utils=format_utils)


@pytest.fixture
def technical_formatter(format_utils):
    return TechnicalFormatter(technical_calculator=MagicMock(), format_utils=format_utils)


def test_ev_formatter_exposes_capital_fee_and_standard_fee_model(ev_formatter):
    assert ev_formatter.starting_capital == 10000.0
    assert ev_formatter.fee_percent == 0.00075
    assert EVFrameworkFormatter.STANDARD_POSITION_PCT == pytest.approx(
        RegimeRiskProfileSelector.get_position_size_cap(RegimeRiskProfile.NEUTRAL)
    )
    for notional, expected_fee in ((800.0, 1.20), (480.0, 0.72), (0.0, 0.0)):
        assert ev_formatter.round_trip_fee(notional) == pytest.approx(expected_fee)


@pytest.mark.parametrize(
    ("current_capital", "realized_pnl", "realized_pct"),
    [
        (10500.0, "$+500.00", "+5.00%"),
        (9500.0, "$-500.00", "-5.00%"),
        (10000.0, "$+0.00", "+0.00%"),
    ],
    ids=["profit", "loss", "flat"],
)
def test_ev_section_reports_portfolio_status(ev_formatter, current_capital, realized_pnl, realized_pct):
    section = ev_formatter.build_ev_framework_section(current_capital=current_capital)

    assert "## EXPECTED VALUE FRAMEWORK" in section
    assert "- Starting Capital: $10,000.00" in section
    assert f"- Current Capital: ${current_capital:,.2f}" in section
    assert f"- Realized P&L: {realized_pnl} ({realized_pct})" in section


def test_ev_section_fee_and_threshold_rules_are_position_based(ev_formatter):
    section = ev_formatter.build_ev_framework_section(current_capital=10000.0)

    assert "Round-trip trading fee: 0.150% of the position size (0.075% per side)" in section
    assert "e.g. $1.20 on a standard 8% position ($800.00)" in section
    assert "Take the trade if EV > 1.5" in section
    assert "EV must exceed $1.80" in section
    assert "HOLD if EV is negative or below the 1.5" in section
    assert "Never reject a positive EV trade purely due to fear of loss" in section
    assert "$7.50" not in section


@pytest.mark.parametrize(
    ("market_metrics", "expected_tokens", "forbidden_tokens"),
    [
        (
            PARTIAL_30D,
            ["30D (PARTIAL):", "Avg$77123.66", "Range$74135.99-79515.00", "Δ↑2.08416868%", "Vol:60209.40"],
            ["\n30D:"],
        ),
        (
            FULL_7D,
            ["7D:", "Avg$77311.09", "Δ↓2.19448997%"],
            ["7D (PARTIAL):"],
        ),
        (
            NO_CHANGE_1H,
            ["1H:", "Range$100.00-200.00"],
            ["Δ", "Vol:"],
        ),
    ],
    ids=["partial-period-label", "full-period-label", "no-change-no-volume"],
)
def test_market_period_metrics_renders_label_and_values(
    market_formatter, market_metrics, expected_tokens, forbidden_tokens
):
    result = market_formatter.format_market_period_metrics(market_metrics)

    for token in expected_tokens:
        assert token in result
    for token in forbidden_tokens:
        assert token not in result


def test_market_period_metrics_blank_input_contract(market_formatter):
    """Empty input and rowless periods both render nothing — no bare header (F2)."""
    assert market_formatter.format_market_period_metrics({}) == ""
    assert market_formatter.format_market_period_metrics({"30D": {}}) == ""
    assert market_formatter.format_market_period_metrics({"30D": {"metrics": {}}}) == ""


@pytest.mark.parametrize(
    ("indicator_changes", "expected_tokens"),
    [
        (
            {"rsi_change": 12.5, "macd_line_change": 11.0, "adx_change": 15.0, "stoch_k_change": 13.0},
            ["RSI ↑ 12.50", "MACD ↑ 11.00", "ADX ↑ 15.00", "Stoch ↑ 13.00"],
        ),
        (
            {"rsi_change": -12.5, "macd_line_change": -11.0, "adx_change": -15.0, "stoch_k_change": -13.0},
            ["RSI ↓ 12.50", "MACD ↓ 11.00", "ADX ↓ 15.00", "Stoch ↓ 13.00"],
        ),
        (
            {"rsi_change": 0.05, "macd_line_change": 0.5, "adx_change": 0.2, "stoch_k_change": 0.4},
            [],
        ),
    ],
    ids=["rising-above-thresholds", "falling-above-thresholds", "below-thresholds-omitted"],
)
def test_market_period_metrics_compresses_indicator_changes(market_formatter, indicator_changes, expected_tokens):
    market_metrics = {"7D": {"metrics": {"avg_price": 100.0}, "indicator_changes": indicator_changes}}

    result = market_formatter.format_market_period_metrics(market_metrics)

    for token in expected_tokens:
        assert token in result
    if not expected_tokens:
        assert "RSI" not in result
        assert "MACD" not in result
        assert "ADX" not in result
        assert "Stoch" not in result


@pytest.mark.parametrize(
    ("opens", "closes", "volumes", "expected_close_trend", "expected_volume_trend"),
    [
        (
            [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            [101.0, 102.0, 103.0, 102.0, 105.0, 106.0, 107.0, 106.0, 109.0, 110.0],
            None,
            "↑RISING (8G/2R, +8.9%)",
            "→STABLE (+0%)",
        ),
        ([100.0] * 10, [100.0] * 10, None, "→FLAT (10G/0R, +0.0%)", "→STABLE (+0%)"),
        ([110.0] * 10, [100.0 - 2 * i for i in range(10)], None, "↓FALLING (0G/10R, -18.0%)", "→STABLE (+0%)"),
        (
            [100.0] * 10,
            list(np.linspace(100.0, 200.0, 10)),
            list(np.linspace(1000.0, 2000.0, 10)),
            "↑RISING (10G/0R, +100.0%)",
            "↑INCREASING (+100%)",
        ),
    ],
    ids=["legacy-rising", "flat", "falling", "rising-volume-surge"],
)
def test_price_action_classifies_close_and_volume_trends(
    technical_formatter, opens, closes, volumes, expected_close_trend, expected_volume_trend
):
    context = price_action_context(candles(opens, closes, volumes=volumes))

    result = technical_formatter.format_price_action_section(context, {})

    assert "## Price Action:" in result
    assert "- Price:100.00" in result
    assert f"Close Trend: {expected_close_trend}" in result
    assert f"- Volume: {expected_volume_trend}" in result
    assert "(NORMAL)" in result


def test_price_action_falls_back_on_short_history_and_guards_zero_baselines(technical_formatter):
    fallback = "## Price Action:\n- Price:100.00 | VWAP:N/A TWAP:N/A"

    assert technical_formatter.format_price_action_section(price_action_context(None), {}) == fallback
    assert (
        technical_formatter.format_price_action_section(price_action_context(candles([100.0], [100.0])), {})
        == fallback
    )

    zeroed = candles([0.0] * 5, [0.0] * 5, volumes=[0.0] * 5)
    zero_result = technical_formatter.format_price_action_section(price_action_context(zeroed), {})
    assert "→FLAT (5G/0R, +0.0%)" in zero_result
    assert "→STABLE (+0%)" in zero_result


def test_price_action_drops_non_finite_candles_before_scoring(technical_formatter):
    """A NaN close or high removes its candle instead of leaking into the prompt (F1)."""
    nan_close = candles([100.0] * 5, [100.0, 101.0, 102.0, 103.0, float("nan")])
    nan_high = candles(
        [100.0] * 5, [100.0, 101.0, 102.0, 103.0, 104.0], highs=[101.0] * 4 + [float("nan")]
    )

    for ohlcv in (nan_close, nan_high):
        result = technical_formatter.format_price_action_section(price_action_context(ohlcv), {})

        assert "nan" not in result.lower()
        assert "Close Trend: ↑RISING (4G/0R, +3.0%)" in result
        assert "(NORMAL)" in result
