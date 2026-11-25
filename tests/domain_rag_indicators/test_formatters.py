

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.analyzer.formatters.ev_formatter import EVFrameworkFormatter
from src.analyzer.formatters.long_term_formatter import LongTermFormatter
from src.analyzer.formatters.market_formatter import MarketFormatter
from src.analyzer.formatters.technical_formatter import TechnicalFormatter
from src.analyzer.technical_calculator import TechnicalCalculator
from src.trading.regime_risk_profile import RegimeRiskProfile, RegimeRiskProfileSelector
from tests.conftest import make_config

PARTIAL_30d = {
    "30d": {
        "metrics": {
            "period": "30d (Partial)",
            "avg_price": 77123.66,
            "lowest_price": 74135.99,
            "highest_price": 79515.00,
            "price_change_percent": 2.08416868,
            "total_volume": 60209.40,
        },
        "indicator_changes": {},
    }
}

FULL_7d = {
    "7d": {
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

NO_CHANGE_24h = {
    "24h": {
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
    assert ev_formatter.standard_position_pct == pytest.approx(
        RegimeRiskProfileSelector.get_position_size_cap(RegimeRiskProfile.NEUTRAL)
    )
    for notional, expected_fee in ((800.0, 1.20), (480.0, 0.72), (0.0, 0.0)):
        assert ev_formatter.round_trip_fee(notional) == pytest.approx(expected_fee)


def test_ev_worked_example_follows_the_cap_the_runtime_enforces():
    tight = EVFrameworkFormatter(make_config(MAX_POSITION_SIZE=0.05))

    section = tight.build_ev_framework_section(current_capital=10000.0)

    assert "e.g. $0.75 on a standard 5% position ($500.00)" in section
    assert "e.g. $1.20 on a standard 8% position ($800.00)" not in section
    assert "EV must exceed $1.12" in section


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


def test_ev_section_asks_for_p_win_without_promising_a_score_nobody_computes(ev_formatter):
    section = ev_formatter.build_ev_framework_section(current_capital=10000.0)

    assert "State P(win) AND its basis" in section
    assert "label it as a setup-specific estimate, not a measured probability" in section
    assert "historical win rate and sample size separately" in section
    assert "scored against the realized outcome" not in section


def test_ev_section_with_zero_inference_cost_is_identical(ev_formatter):
    assert ev_formatter.build_ev_framework_section(
        current_capital=10000.0, inference_cost_usd=0.0
    ) == ev_formatter.build_ev_framework_section(current_capital=10000.0)


def test_ev_section_charges_inference_cost_in_the_threshold(ev_formatter):
    section = ev_formatter.build_ev_framework_section(
        current_capital=10000.0, inference_cost_usd=0.50
    )

    assert "- Inference cost (rolling average per analysis): $0.5000" in section
    assert (
        "- Net break-even: $1.70 (round-trip fee $1.20 + inference $0.5000); "
        "EV threshold 1.5× = $2.55"
    ) in section
    assert "1.5× (round-trip fee of your position + inference cost)" in section
    assert "EV must exceed $2.55" in section
    assert "− inference cost ($0.5000)" in section
    assert "1.5× cost threshold" in section
    assert "EV must exceed $1.80" not in section


@pytest.mark.parametrize(
    "bad_cost",
    [-1.0, float("nan"), float("inf"), float("-inf"), None, "x"],
    ids=["negative", "nan", "inf", "negative-inf", "none", "string"],
)
def test_ev_section_treats_invalid_inference_cost_as_zero(ev_formatter, bad_cost):
    assert ev_formatter.build_ev_framework_section(
        current_capital=10000.0, inference_cost_usd=bad_cost
    ) == ev_formatter.build_ev_framework_section(current_capital=10000.0)


@pytest.mark.parametrize(
    ("market_metrics", "expected_tokens", "forbidden_tokens"),
    [
        (
            PARTIAL_30d,
            ["30d (Partial):", "Avg$77123.66", "Range$74135.99-79515.00", "Δ↑2.08416868%", "Vol:60209.40"],
            ["\n30d:"],
        ),
        (
            FULL_7d,
            ["7d:", "Avg$77311.09", "Δ↓2.19448997%"],
            ["7d (Partial):"],
        ),
        (
            NO_CHANGE_24h,
            ["24h:", "Range$100.00-200.00"],
            ["Δ", "Vol:"],
        ),
    ],
    ids=["partial-period-label", "full-period-label", "no-change-no-volume"],
)
def test_market_period_metrics_renders_label_and_values(
    market_formatter, market_metrics, expected_tokens, forbidden_tokens
):
    result = market_formatter.format_market_period_metrics(market_metrics, "4h")

    for token in expected_tokens:
        assert token in result
    for token in forbidden_tokens:
        assert token not in result


def test_market_period_metrics_names_the_frame_and_the_daily_row(market_formatter):

    frame_only = market_formatter.format_market_period_metrics(FULL_7d, "4h")
    with_daily = market_formatter.format_market_period_metrics(
        {**FULL_7d, "180d": PARTIAL_30d["30d"]}, "4h"
    )

    assert frame_only.startswith("## Period Metrics (windows of 4h candles)\n")
    assert with_daily.startswith(
        "## Period Metrics (windows of 4h candles; the 180d row uses daily candles)\n"
    )


def test_market_period_metrics_blank_input_contract(market_formatter):

    assert market_formatter.format_market_period_metrics({}, "4h") == ""
    assert market_formatter.format_market_period_metrics({"30d": {}}, "4h") == ""
    assert market_formatter.format_market_period_metrics({"30d": {"metrics": {}}}, "4h") == ""


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
    market_metrics = {"7d": {"metrics": {"avg_price": 100.0}, "indicator_changes": indicator_changes}}

    result = market_formatter.format_market_period_metrics(market_metrics, "4h")

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
            "↑INCREASING (+38%)",
        ),
    ],
    ids=["legacy-rising", "flat", "falling", "rising-volume-surge"],
)
def test_price_action_classifies_close_and_volume_trends(
    technical_formatter, opens, closes, volumes, expected_close_trend, expected_volume_trend
):
    context = price_action_context(candles(opens, closes, volumes=volumes))

    result = technical_formatter.format_price_action_section(context, {})

    assert "### Price Action:" in result
    assert "- Price:100.00" in result
    assert f"Close Trend (10 bars): {expected_close_trend}" in result
    assert f"- Volume: {expected_volume_trend}" in result
    assert "(NORMAL)" in result


def test_price_action_volume_trend_uses_the_window_mean_not_two_candles(technical_formatter):

    volumes = [100.0] * 8 + [5000.0, 100.0]
    context = price_action_context(candles([100.0] * 10, [100.0] * 10, volumes=volumes))

    result = technical_formatter.format_price_action_section(context, {})

    assert "- Volume: ↓DECLINING (-84%)" in result


def test_price_action_falls_back_on_short_history_and_guards_zero_baselines(technical_formatter):
    fallback = "### Price Action:\n- Price:100.00 | VWAP:N/A (20 bars) TWAP:N/A (20 bars)"

    assert technical_formatter.format_price_action_section(price_action_context(None), {}) == fallback
    assert (
        technical_formatter.format_price_action_section(price_action_context(candles([100.0], [100.0])), {})
        == fallback
    )

    zeroed = candles([0.0] * 5, [0.0] * 5, volumes=[0.0] * 5)
    zero_result = technical_formatter.format_price_action_section(price_action_context(zeroed), {})
    assert "→FLAT (5G/0R, +0.0%)" in zero_result
    assert "→STABLE (+0%)" in zero_result


@pytest.mark.parametrize(
    ("vwap", "expected"),
    [(98.0, "VWAP:98.00 (20 bars; price +2.04% above)"), (102.0, "VWAP:102.00 (20 bars; price -1.96% below)")],
    ids=["above", "below"],
)
def test_price_action_states_which_side_of_vwap_the_price_is_on(
    technical_formatter, vwap, expected
):
    context = price_action_context(candles([100.0] * 10, [100.0] * 10))

    result = technical_formatter.format_price_action_section(context, {"vwap": vwap})

    assert expected in result


def test_price_action_leaves_vwap_bare_without_a_usable_value(technical_formatter):
    context = price_action_context(candles([100.0] * 10, [100.0] * 10))

    for td in ({}, {"vwap": float("nan")}, {"vwap": 0.0}):
        result = technical_formatter.format_price_action_section(context, td)
        assert "(price " not in result


def test_price_action_reads_the_production_vwap_array_once(technical_formatter):
    context = price_action_context(candles([100.0] * 10, [100.0] * 10))

    result = technical_formatter.format_price_action_section(
        context, {"vwap": np.array([97.0, 98.0])}
    )

    assert "VWAP:98.00 (20 bars; price +2.04% above)" in result
    assert "VWAP:N/A" not in result


def test_price_action_does_not_invent_a_vwap_side(technical_formatter):
    context = price_action_context(candles([100.0] * 10, [100.0] * 10))

    at_vwap = technical_formatter.format_price_action_section(context, {"vwap": 100.0})
    assert "VWAP:100.00 (20 bars; price at VWAP)" in at_vwap

    context.current_price = float("nan")
    without_price = technical_formatter.format_price_action_section(context, {"vwap": 98.0})
    assert "VWAP:98.00" in without_price
    assert "above" not in without_price and "below" not in without_price


def test_price_action_drops_non_finite_candles_before_scoring(technical_formatter):

    nan_close = candles([100.0] * 5, [100.0, 101.0, 102.0, 103.0, float("nan")])
    nan_high = candles(
        [100.0] * 5, [100.0, 101.0, 102.0, 103.0, 104.0], highs=[101.0] * 4 + [float("nan")]
    )

    for ohlcv in (nan_close, nan_high):
        result = technical_formatter.format_price_action_section(price_action_context(ohlcv), {})

        assert "nan" not in result.lower()
        assert "Close Trend (4 bars): ↑RISING (4G/0R, +3.0%)" in result
        assert "(NORMAL)" in result


def test_technical_prompt_uses_history_only_for_temporal_context(technical_formatter):
    technical_formatter.technical_calculator = TechnicalCalculator()
    ohlcv = candles([100.0] * 30, [120.0] * 30)
    context = price_action_context(ohlcv)
    context.current_price = 105.0
    context.technical_patterns = {}
    context.technical_data = {
        "rsi": 60.0,
        "adx": 25.0,
        "ichimoku_span_a": 110.0,
        "ichimoku_span_b": 108.0,
        "basic_support": 100.0,
        "basic_resistance": 105.0,
        "retested_support": 95.0,
        "retested_resistance": 112.0,
        "donchian_upper": 104.0,
        "donchian_lower": 96.0,
        "pivot_r2": 117.5,
        "fib_pivot_s2": 97.5,
        "atr": 5.0,
    }
    context.technical_history = {
        "rsi": np.linspace(40.0, 60.0, 30),
        "adx": np.linspace(15.0, 25.0, 30),
    }
    result = technical_formatter.format_technical_analysis(context, "4h")

    assert "RSI:60.00 (↑ +7.6)" in result
    assert "ADX:25.00 (→ +3.8)" in result
    assert "Ichi:☁️↑" in result
    assert "[40." not in result
    assert "Distances from live price" in result
    assert "1× ATR stop floor from live price: LONG SL at or below 100.00, SHORT SL at or above 110.00 (ATR 5.00000000)" in result
    assert "Support:100.00 (-5.00000000; -1.00 ATR)" in result
    assert "Resistance:105.00 (0.00; 0.00 ATR)" in result
    assert "Retested S/R (120 bars): Support:95.00 (-10.00; -2.00 ATR), Resistance:112.00 (+7.00000000; +1.40 ATR)" in result
    assert "Donchian (20 bars): U:104.00 (-1.00000000; -0.20 ATR), L:96.00 (-9.00000000; -1.80 ATR)" in result
    assert "R2:117.50 (+12.50; +2.50 ATR)" in result
    assert "S2:97.50 (-7.50000000; -1.50 ATR)" in result
    assert "R1:N/A" in result

    context.technical_data["atr"] = float("inf")
    levels = technical_formatter.format_key_levels_section(context.technical_data, context.current_price)
    assert "Support:100.00 (-5.00000000)" in levels
    assert " ATR)" not in levels
    assert "1× ATR stop floor" not in levels

    levels = technical_formatter.format_key_levels_section(context.technical_data, None)
    assert "Support:100.00" in levels
    assert "Distances from live price" not in levels
    assert "(-5.00" not in levels


def test_long_term_ichimoku_uses_calculator_daily_keys(format_utils):
    formatter = LongTermFormatter(format_utils=format_utils)
    daily = {
        "daily_ichimoku_conversion": 101.0,
        "daily_ichimoku_base": 102.0,
        "daily_ichimoku_span_a": 103.0,
        "daily_ichimoku_span_b": 104.0,
    }

    section = formatter.format_long_term_analysis(daily, current_price=105.0)

    assert "## Ichimoku Analysis (Daily):" in section
    assert "Tenkan: 101.00" in section
    assert "Kijun: 102.00" in section
    assert "Cloud Position: Above Cloud (Bullish)" in section


@pytest.mark.parametrize("missing", [None, float("nan")], ids=["none", "nan"])
@pytest.mark.parametrize(
    ("key", "partner", "label"),
    [
        ("daily_rsi", None, "Daily RSI"),
        ("daily_macd_line", "daily_macd_signal", "Daily MACD"),
        ("daily_macd_signal", "daily_macd_line", "Daily MACD"),
        ("daily_stoch_k", None, "Daily Stoch"),
        ("daily_adx", None, "Trend Strength"),
        ("daily_ichimoku_conversion", None, "Tenkan"),
        ("daily_ichimoku_base", None, "Kijun"),
        ("daily_ichimoku_span_a", "daily_ichimoku_span_b", "Cloud Position"),
        ("daily_ichimoku_span_b", "daily_ichimoku_span_a", "Cloud Position"),
    ],
    ids=[
        "rsi",
        "macd-line",
        "macd-signal",
        "stoch",
        "adx",
        "tenkan",
        "kijun",
        "cloud-span-a",
        "cloud-span-b",
    ],
)
def test_long_term_formatter_omits_missing_daily_measurements(format_utils, key, partner, label, missing):
    payload = {key: missing}
    if partner is not None:
        payload[partner] = 100.0

    section = LongTermFormatter(format_utils=format_utils).format_long_term_analysis(
        payload, current_price=100.0
    )

    assert label not in section


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"daily_rsi": 0.0}, "Daily RSI: 0.00000000 (Oversold)"),
        ({"daily_macd_line": 0.0, "daily_macd_signal": -1.0}, "Daily MACD: Bullish"),
        ({"daily_stoch_k": 0.0}, "Daily Stoch: 0.00000000 (Oversold)"),
        ({"daily_adx": 0.0}, "Trend Strength (Daily ADX): 0.00000000 (Weak/No Trend)"),
        ({"daily_ichimoku_conversion": 0.0}, "Tenkan: 0.00000000"),
        ({"daily_ichimoku_base": 0.0}, "Kijun: 0.00000000"),
        (
            {"daily_ichimoku_span_a": 0.0, "daily_ichimoku_span_b": 100.0},
            "Cloud Position: Above Cloud (Bullish)",
        ),
    ],
    ids=["rsi", "macd", "stoch", "adx", "tenkan", "kijun", "cloud"],
)
def test_long_term_formatter_keeps_zero_measurements(format_utils, payload, expected):
    section = LongTermFormatter(format_utils=format_utils).format_long_term_analysis(
        payload, current_price=150.0
    )

    assert expected in section


def test_long_term_formatter_renders_short_daily_history_from_the_calculator(format_utils):
    closes = list(np.linspace(100.0, 120.0, 20))
    ohlcv = candles(
        [close - 0.5 for close in closes],
        closes,
        highs=[close + 1.0 for close in closes],
        lows=[close - 1.0 for close in closes],
    )
    long_term_data = TechnicalCalculator().get_long_term_indicators(ohlcv)

    section = LongTermFormatter(format_utils=format_utils).format_long_term_analysis(
        long_term_data, current_price=120.0
    )

    assert "Daily RSI: 100.00 (Overbought)" in section
    assert "Daily MACD" not in section
    assert "Trend Strength" not in section
    assert "Ichimoku" not in section


def test_long_term_sma_sections_read_the_calculator_sma_dicts(format_utils):
    daily = {"sma_values": {20: 70.0, 200: 80.0}, "volume_sma_values": {20: 12.5}}

    section = LongTermFormatter(format_utils=format_utils).format_long_term_analysis(
        daily, current_price=105.0
    )

    assert "## Simple Moving Averages (Daily):\nSMA20: 70.00 | SMA200: 80.00" in section
    assert "## Volume Moving Averages (Daily):\nVol SMA20: 12.50" in section
    assert "## Price Position vs SMAs (Daily):\nSMA20: 50.00% above | SMA200: 31.25% above" in section


def test_long_term_sma_sections_are_absent_without_the_sma_dicts(format_utils):
    section = LongTermFormatter(format_utils=format_utils).format_long_term_analysis(
        {"daily_rsi": 55.0}, current_price=100.0
    )

    assert "Simple Moving Averages" not in section
    assert "Price Position vs SMAs" not in section


def test_long_term_formatter_renders_the_sma_sections_from_the_calculator(format_utils):
    closes = list(np.linspace(100.0, 200.0, 200))
    ohlcv = candles(
        [close - 0.5 for close in closes],
        closes,
        highs=[close + 1.0 for close in closes],
        lows=[close - 1.0 for close in closes],
    )
    long_term_data = TechnicalCalculator().get_long_term_indicators(ohlcv)

    assert set(long_term_data["sma_values"]) == {20, 50, 100, 200}
    assert {20, 50} <= set(long_term_data["volume_sma_values"])

    section = LongTermFormatter(format_utils=format_utils).format_long_term_analysis(
        long_term_data, current_price=200.0
    )

    assert "## Simple Moving Averages (Daily):" in section
    assert "## Volume Moving Averages (Daily):" in section
    assert "## Price Position vs SMAs (Daily):" in section
    assert "SMA200: 150.00" in section


def test_td_setup_display_names_the_setup_phase(technical_formatter):
    assert technical_formatter._format_td_setup({"td_setup": np.array([1.0, 2.0, np.nan])}) == " | TD Setup:2↑"
    assert technical_formatter._format_td_setup({"td_setup": -9.0}) == " | TD Setup:9↓⚠️"
    assert technical_formatter._format_td_setup({"td_setup": 0.0}) == ""
    assert technical_formatter._format_td_setup({"td_setup": None}) == ""
    assert technical_formatter._format_td_setup({}) == ""
