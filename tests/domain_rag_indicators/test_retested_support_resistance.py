

import numpy as np
import pytest

from src.indicators.base.technical_indicators import TechnicalIndicators
from src.indicators.support_resistance import retested_support_resistance_numba


def test_technical_indicators_facade_uses_retested_levels():
    close = np.tile(np.array([100.0, 101.0]), 70)
    ti = TechnicalIndicators()
    ti.get_data(np.column_stack((np.arange(len(close)), close, close + 1, close - 1, close, np.ones(len(close)))))

    support, resistance = ti.retested_support_resistance()

    assert support[-1] == pytest.approx(99.0)
    assert resistance[-1] == pytest.approx(102.0)
    assert np.isnan(support[:120]).all()


def test_three_confirmed_touches_on_each_side():
    low = np.full(25, 99.0)
    high = np.full(25, 101.0)
    close = np.full(25, 100.0)
    for index, level in ((3, 98.0), (8, 98.2), (13, 97.9)):
        low[index] = level
    for index, level in ((5, 102.0), (10, 101.8), (15, 102.1)):
        high[index] = level

    support, resistance = retested_support_resistance_numba(high, low, close, 22, 3, 0.005)

    assert np.isnan(support[13])
    assert np.isnan(resistance[15])
    assert support[-1] == pytest.approx((98.0 + 98.2 + 97.9) / 3)
    assert resistance[-1] == pytest.approx((102.0 + 101.8 + 102.1) / 3)
    assert np.isnan(support[:22]).all()

    low[13] = 99.0
    support_without_third_touch, _ = retested_support_resistance_numba(high, low, close, 22, 3, 0.005)
    assert np.isnan(support_without_third_touch[-1])


def test_one_side_is_valid_without_the_other_and_price_must_be_on_correct_side():
    low = np.full(30, 99.0)
    high = np.full(30, 101.0)
    close = np.full(30, 100.0)
    low[[3, 8, 13]] = 98.0

    support, resistance = retested_support_resistance_numba(high, low, close, 28, 3, 0.005)
    assert support[-1] == pytest.approx(98.0)
    assert np.isnan(resistance[-1])

    close[-1] = 97.0
    support, resistance = retested_support_resistance_numba(high, low, close, 28, 3, 0.005)
    assert np.isnan(support[-1])
    assert np.isnan(resistance[-1])


def test_window_expiry_and_tolerance_do_not_invent_touches():
    low = np.full(40, 99.0)
    high = np.full(40, 101.0)
    close = np.full(40, 100.0)
    low[[3, 8, 13]] = [98.0, 98.2, 97.9]

    support, _ = retested_support_resistance_numba(high, low, close, 20, 3, 0.005)
    assert np.isfinite(support[20])
    assert np.isnan(support[-1])

    support, _ = retested_support_resistance_numba(high, low, close, 20, 3, 0.001)
    assert np.isnan(support[20])


def test_historical_values_do_not_change_after_new_candles():
    close = np.full(35, 100.0)
    low = np.full(35, 99.0)
    high = np.full(35, 101.0)
    low[[3, 8, 13]] = 98.0
    high[[5, 10, 15]] = 102.0

    earlier = retested_support_resistance_numba(high[:25], low[:25], close[:25], 20, 3, 0.005)
    full = retested_support_resistance_numba(high, low, close, 20, 3, 0.005)
    for before, after in zip(earlier, full, strict=True):
        np.testing.assert_array_equal(before, after[:25])
