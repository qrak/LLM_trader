

import numpy as np
import pytest

from src.analyzer.pattern_engine.indicator_patterns import IndicatorPatternEngine
from src.analyzer.pattern_engine.indicator_patterns.structure_patterns import (
    PIVOT_WINDOW,
    break_of_structure_numba,
    double_top_bottom_numba,
    find_pivots_numba,
    swing_structure_numba,
)
from src.utils.pattern_recency import filter_recent_patterns


def wave(count: int, slope: float = 0.0, amplitude: float = 5.0, period: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    index = np.arange(count, dtype=float)
    closes = 100.0 + slope * index + amplitude * np.sin(2 * np.pi * index / period)
    highs = closes + 1.0
    lows = closes - 1.0
    return highs, lows, closes


def test_pivots_are_found_on_a_wave():

    highs, lows, _ = wave(80)

    high_idx, low_idx = find_pivots_numba(highs, lows, PIVOT_WINDOW)

    assert len(high_idx) >= 2
    assert len(low_idx) >= 2
    assert np.all(highs[high_idx] > highs[high_idx - 1])
    assert np.all(lows[low_idx] < lows[low_idx - 1])


def test_a_rising_wave_is_an_uptrend():

    highs, lows, _ = wave(80, slope=0.5)

    state, last_high, last_low, previous_high, previous_low = swing_structure_numba(
        highs, lows, PIVOT_WINDOW
    )

    assert state == 1
    assert last_high > previous_high
    assert last_low > previous_low


def test_a_falling_wave_is_a_downtrend():

    highs, lows, _ = wave(80, slope=-0.5)

    state, last_high, last_low, previous_high, previous_low = swing_structure_numba(
        highs, lows, PIVOT_WINDOW
    )

    assert state == -1
    assert last_high < previous_high
    assert last_low < previous_low


def test_a_higher_high_with_a_lower_low_is_mixed_structure():

    highs, lows, _ = wave(80)
    high_idx, low_idx = find_pivots_numba(highs, lows, PIVOT_WINDOW)
    highs = highs.copy()
    lows = lows.copy()
    highs[high_idx[-1]] = highs[high_idx[-2]] + 2.0
    lows[low_idx[-1]] = lows[low_idx[-2]] - 2.0

    state, *_ = swing_structure_numba(highs, lows, PIVOT_WINDOW)

    assert state == 0


def test_a_close_above_the_last_pivot_high_breaks_structure_up():

    highs, lows, closes = wave(80, slope=0.5)
    closes = closes.copy()
    closes[-1] = float(np.max(highs)) + 1.0

    direction, level, bars_ago = break_of_structure_numba(closes, highs, lows, PIVOT_WINDOW)

    assert direction == 1
    assert level > 0.0
    assert bars_ago >= 0


def test_a_close_below_the_last_pivot_low_breaks_structure_down():

    highs, lows, closes = wave(80, slope=-0.5)
    closes = closes.copy()
    closes[-1] = float(np.min(lows)) - 1.0

    direction, level, _ = break_of_structure_numba(closes, highs, lows, PIVOT_WINDOW)

    assert direction == -1
    assert level > 0.0


def test_two_equal_highs_are_a_double_top():

    highs, lows, closes = wave(80)
    high_idx, low_idx = find_pivots_numba(highs, lows, PIVOT_WINDOW)
    highs = highs.copy()
    lows = lows.copy()
    highs[high_idx[-1]] = highs[high_idx[-2]]
    lows[low_idx[-1]] = lows[low_idx[-2]] - 5.0

    kind, level, _ = double_top_bottom_numba(highs, lows, closes, PIVOT_WINDOW, 0.005)

    assert kind == 1
    assert level == pytest.approx(float(highs[high_idx[-1]]))


def test_two_equal_lows_are_a_double_bottom():

    highs, lows, closes = wave(80, slope=0.5)
    _, low_idx = find_pivots_numba(highs, lows, PIVOT_WINDOW)
    lows = lows.copy()
    lows[low_idx[-1]] = lows[low_idx[-2]]

    kind, level, _ = double_top_bottom_numba(highs, lows, closes, PIVOT_WINDOW, 0.005)

    assert kind == -1
    assert level == pytest.approx(float(lows[low_idx[-1]]))


def test_a_double_top_stops_counting_once_price_closes_above_it():

    highs, lows, closes = wave(80)
    high_idx, low_idx = find_pivots_numba(highs, lows, PIVOT_WINDOW)
    highs = highs.copy()
    lows = lows.copy()
    highs[high_idx[-1]] = highs[high_idx[-2]]
    lows[low_idx[-1]] = lows[low_idx[-2]] - 5.0
    closes = closes.copy()
    closes[-1] = float(highs[high_idx[-1]]) + 10.0

    kind, _, _ = double_top_bottom_numba(highs, lows, closes, PIVOT_WINDOW, 0.005)

    assert kind == 0


def test_a_double_bottom_stops_counting_once_price_closes_below_it():

    highs, lows, closes = wave(80, slope=0.5)
    _, low_idx = find_pivots_numba(highs, lows, PIVOT_WINDOW)
    lows = lows.copy()
    lows[low_idx[-1]] = lows[low_idx[-2]]
    closes = closes.copy()
    closes[-1] = float(lows[low_idx[-1]]) - 10.0

    kind, _, _ = double_top_bottom_numba(highs, lows, closes, PIVOT_WINDOW, 0.005)

    assert kind == 0


def test_the_more_recent_of_a_double_top_and_a_double_bottom_wins():

    highs, lows, closes = wave(80)
    high_idx, low_idx = find_pivots_numba(highs, lows, PIVOT_WINDOW)
    highs = highs.copy()
    lows = lows.copy()
    highs[high_idx[-1]] = highs[high_idx[-2]]
    lows[low_idx[-1]] = lows[low_idx[-2]]
    closes = closes.copy()
    closes[-1] = (float(highs[high_idx[-1]]) + float(lows[low_idx[-1]])) / 2.0

    kind, _, bars_ago = double_top_bottom_numba(highs, lows, closes, PIVOT_WINDOW, 0.005)

    expected = -1 if low_idx[-1] > high_idx[-1] else 1
    assert kind == expected
    assert bars_ago == len(closes) - 1 - int(max(high_idx[-1], low_idx[-1]))


def test_structure_patterns_carry_the_candle_they_were_confirmed_on():

    highs, lows, closes = wave(200, slope=0.5)
    opens = closes - 0.2
    volume = np.full(len(closes), 1000.0)
    timestamps = np.arange(len(closes), dtype=float) * 4 * 3600 * 1000
    ohlcv = np.column_stack([timestamps, opens, highs, lows, closes, volume])

    patterns = IndicatorPatternEngine().detect_patterns({}, ohlcv, None, None)

    last_index = len(closes) - 1
    for pattern in patterns["structure"]:
        assert pattern["index"] < last_index
        assert pattern["index"] == last_index - pattern["details"].get("bars_ago", last_index - pattern["index"])


def test_a_reported_pivot_age_is_never_below_the_pivot_window():

    highs, lows, closes = wave(200, slope=0.5)
    opens = closes - 0.2
    volume = np.full(len(closes), 1000.0)
    timestamps = np.arange(len(closes), dtype=float) * 4 * 3600 * 1000
    ohlcv = np.column_stack([timestamps, opens, highs, lows, closes, volume])

    patterns = IndicatorPatternEngine().detect_patterns({}, ohlcv, None, None)

    for pattern in patterns["structure"]:
        bars_ago = pattern["details"].get("bars_ago")
        if bars_ago is not None:
            assert bars_ago >= PIVOT_WINDOW


def test_a_structure_whose_last_pivot_is_old_is_filtered_out():

    highs, lows, closes = wave(200)
    tail = np.arange(1, 41, dtype=float)
    closes = np.concatenate([closes, closes[-1] + tail * 3.0])
    highs = np.concatenate([highs, closes[-40:] + 1.0])
    lows = np.concatenate([lows, closes[-40:] - 1.0])
    opens = closes - 0.2
    volume = np.full(len(closes), 1000.0)
    timestamps = np.arange(len(closes), dtype=float) * 4 * 3600 * 1000
    ohlcv = np.column_stack([timestamps, opens, highs, lows, closes, volume])

    patterns = IndicatorPatternEngine().detect_patterns({}, ohlcv, None, None)
    kept = filter_recent_patterns(patterns, len(closes), "4h")

    assert patterns["structure"]
    assert kept["structure"] == []


def test_the_engine_reports_structure_as_its_own_category():

    highs, lows, closes = wave(200, slope=0.5)
    opens = closes - 0.2
    volume = np.full(len(closes), 1000.0)
    timestamps = np.arange(len(closes), dtype=float) * 4 * 3600 * 1000
    ohlcv = np.column_stack([timestamps, opens, highs, lows, closes, volume])

    patterns = IndicatorPatternEngine().detect_patterns({}, ohlcv, None, None)

    types = [pattern["type"] for pattern in patterns["structure"]]
    assert "structure_uptrend" in types
    assert any(pattern["description"].startswith("Swing structure") for pattern in patterns["structure"])


def test_a_short_series_reports_no_structure():

    highs, lows, closes = wave(10)
    opens = closes - 0.2
    volume = np.full(len(closes), 1000.0)
    timestamps = np.arange(len(closes), dtype=float) * 4 * 3600 * 1000
    ohlcv = np.column_stack([timestamps, opens, highs, lows, closes, volume])

    patterns = IndicatorPatternEngine().detect_patterns({}, ohlcv, None, None)

    assert patterns["structure"] == []
