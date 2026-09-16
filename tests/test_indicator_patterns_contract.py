"""Characterization tests for the numba indicator-pattern functions.

These pin the CURRENT behaviour of the twin pattern detectors (`rsi`, `stochastic`,
`volume`, `divergence`, `sar`) so the duplicated implementations can be collapsed
onto one shared core without changing any result. The expected values were recorded
from the pre-refactor implementation — they are a safety net, not a specification.

If a value here changes, the refactor changed behaviour: stop and compare, do not
"fix" the expectation.
"""
import json

import numpy as np

from src.analyzer.pattern_engine.indicator_patterns.divergence_patterns import (
    detect_bearish_divergence_numba,
    detect_bullish_divergence_numba,
)
from src.analyzer.pattern_engine.indicator_patterns.rsi_patterns import (
    detect_rsi_overbought_numba,
    detect_rsi_oversold_numba,
)
from src.analyzer.pattern_engine.indicator_patterns.stochastic_patterns import (
    detect_stoch_overbought_numba,
    detect_stoch_oversold_numba,
)
from src.analyzer.pattern_engine.indicator_patterns.volume_patterns import (
    detect_climax_volume_numba,
    detect_volume_dryup_numba,
    detect_volume_spike_numba,
)
from src.indicators.trend.sar_utils import (
    get_initial_sar_state,
    initialize_sar_arrays,
    update_bearish_sar,
    update_bullish_sar,
)

BASELINE = json.loads(r"""{
 "rsi_oversold::empty": [
  false,
  -1,
  0.0
 ],
 "rsi_overbought::empty": [
  false,
  -1,
  0.0
 ],
 "rsi_oversold::mid": [
  false,
  -1,
  50.0
 ],
 "rsi_overbought::mid": [
  false,
  -1,
  50.0
 ],
 "rsi_oversold::oversold_run": [
  true,
  0,
  22.0
 ],
 "rsi_overbought::oversold_run": [
  false,
  -1,
  22.0
 ],
 "rsi_oversold::at_threshold": [
  false,
  -1,
  30.0
 ],
 "rsi_overbought::at_threshold": [
  false,
  -1,
  30.0
 ],
 "rsi_oversold::touch_below_once": [
  false,
  -1,
  45.0
 ],
 "rsi_overbought::touch_below_once": [
  false,
  -1,
  45.0
 ],
 "rsi_oversold::overbought_run": [
  false,
  -1,
  80.0
 ],
 "rsi_overbought::overbought_run": [
  true,
  0,
  80.0
 ],
 "rsi_oversold::near_flat": [
  true,
  0,
  29.6
 ],
 "rsi_overbought::near_flat": [
  false,
  -1,
  29.6
 ],
 "rsi_oversold::min_periods_3": [
  true,
  0,
  23.0
 ],
 "rsi_overbought::min_periods_3": [
  true,
  0,
  77.0
 ],
 "stoch_oversold::empty": [
  false,
  0,
  0.0
 ],
 "stoch_overbought::empty": [
  false,
  0,
  0.0
 ],
 "stoch_oversold::nan_last": [
  false,
  0,
  0.0
 ],
 "stoch_overbought::nan_last": [
  false,
  0,
  0.0
 ],
 "stoch_oversold::low": [
  true,
  0,
  15.0
 ],
 "stoch_overbought::low": [
  false,
  0,
  0.0
 ],
 "stoch_oversold::boundary_low": [
  false,
  0,
  0.0
 ],
 "stoch_overbought::boundary_low": [
  false,
  0,
  0.0
 ],
 "stoch_oversold::high": [
  false,
  0,
  0.0
 ],
 "stoch_overbought::high": [
  true,
  0,
  85.0
 ],
 "stoch_oversold::boundary_high": [
  false,
  0,
  0.0
 ],
 "stoch_overbought::boundary_high": [
  false,
  0,
  0.0
 ],
 "volume_spike::short": [
  false,
  0.0,
  0.0,
  0.0
 ],
 "volume_dryup::short": [
  false,
  0.0,
  0.0,
  0.0
 ],
 "volume_climax::short": [
  false,
  0.0,
  0.0,
  0.0
 ],
 "volume_spike::nan_last": [
  false,
  0.0,
  0.0,
  0.0
 ],
 "volume_dryup::nan_last": [
  false,
  0.0,
  0.0,
  0.0
 ],
 "volume_climax::nan_last": [
  false,
  0.0,
  0.0,
  0.0
 ],
 "volume_spike::flat_spike": [
  true,
  300.0,
  100.0,
  3.0
 ],
 "volume_dryup::flat_spike": [
  false,
  300.0,
  100.0,
  3.0
 ],
 "volume_climax::flat_spike": [
  false,
  0.0,
  0.0,
  0.0
 ],
 "volume_spike::flat_no_spike": [
  false,
  150.0,
  100.0,
  1.5
 ],
 "volume_dryup::flat_no_spike": [
  false,
  150.0,
  100.0,
  1.5
 ],
 "volume_climax::flat_no_spike": [
  false,
  0.0,
  0.0,
  0.0
 ],
 "volume_spike::flat_dryup": [
  false,
  30.0,
  100.0,
  0.3
 ],
 "volume_dryup::flat_dryup": [
  true,
  30.0,
  100.0,
  0.3
 ],
 "volume_climax::flat_dryup": [
  false,
  0.0,
  0.0,
  0.0
 ],
 "volume_spike::flat_no_dryup": [
  false,
  80.0,
  100.0,
  0.8
 ],
 "volume_dryup::flat_no_dryup": [
  false,
  80.0,
  100.0,
  0.8
 ],
 "volume_climax::flat_no_dryup": [
  false,
  0.0,
  0.0,
  0.0
 ],
 "volume_spike::climax": [
  true,
  400.0,
  100.0,
  4.0
 ],
 "volume_dryup::climax": [
  false,
  400.0,
  100.0,
  4.0
 ],
 "volume_climax::climax": [
  true,
  400.0,
  100.0,
  4.0
 ],
 "volume_spike::negative_avg": [
  false,
  0.0,
  0.0,
  0.0
 ],
 "volume_dryup::negative_avg": [
  false,
  0.0,
  0.0,
  0.0
 ],
 "volume_climax::negative_avg": [
  false,
  0.0,
  0.0,
  0.0
 ],
 "divergence_bullish::short": [
  false,
  -1,
  -1,
  0.0,
  0.0,
  0.0,
  0.0
 ],
 "divergence_bearish::short": [
  false,
  -1,
  -1,
  0.0,
  0.0,
  0.0,
  0.0
 ],
 "divergence_bullish::flat": [
  false,
  -1,
  -1,
  0.0,
  0.0,
  0.0,
  0.0
 ],
 "divergence_bearish::flat": [
  false,
  -1,
  -1,
  0.0,
  0.0,
  0.0,
  0.0
 ],
 "divergence_bullish::trend_down": [
  false,
  -1,
  -1,
  0.0,
  0.0,
  0.0,
  0.0
 ],
 "divergence_bearish::trend_down": [
  false,
  -1,
  -1,
  0.0,
  0.0,
  0.0,
  0.0
 ],
 "divergence_bullish::seeded_7": [
  false,
  -1,
  -1,
  0.0,
  0.0,
  0.0,
  0.0
 ],
 "divergence_bearish::seeded_7": [
  false,
  -1,
  -1,
  0.0,
  0.0,
  0.0,
  0.0
 ],
 "divergence_bullish::seeded_11": [
  false,
  -1,
  -1,
  0.0,
  0.0,
  0.0,
  0.0
 ],
 "divergence_bearish::seeded_11": [
  false,
  -1,
  -1,
  0.0,
  0.0,
  0.0,
  0.0
 ],
 "sar_bullish::i1": [
  1,
  99.02,
  102.0,
  0.04
 ],
 "sar_bearish::i1": [
  -1,
  199.98,
  101.0,
  0.04
 ],
 "sar_bullish::i2": [
  -1,
  NaN,
  102.0,
  0.02
 ],
 "sar_bearish::i2": [
  1,
  NaN,
  103.0,
  0.02
 ],
 "sar_bullish::i3": [
  -1,
  NaN,
  101.0,
  0.02
 ],
 "sar_bearish::i3": [
  1,
  NaN,
  102.0,
  0.02
 ],
 "sar_bullish::i5": [
  -1,
  NaN,
  104.0,
  0.02
 ],
 "sar_bearish::i5": [
  1,
  NaN,
  105.0,
  0.02
 ],
 "sar_initial_state": [
  1.0,
  100.0,
  101.0,
  0.02
 ],
 "divergence_bullish::bull_true": [
  true,
  15,
  30,
  95.0,
  90.0,
  30.0,
  40.0
 ],
 "divergence_bearish::bull_true": [
  false,
  -1,
  -1,
  0.0,
  0.0,
  0.0,
  0.0
 ],
 "divergence_bullish::bull_false": [
  false,
  -1,
  -1,
  0.0,
  0.0,
  0.0,
  0.0
 ],
 "divergence_bearish::bull_false": [
  false,
  -1,
  -1,
  0.0,
  0.0,
  0.0,
  0.0
 ],
 "divergence_bullish::bear_true": [
  false,
  -1,
  -1,
  0.0,
  0.0,
  0.0,
  0.0
 ],
 "divergence_bearish::bear_true": [
  true,
  15,
  30,
  105.0,
  110.0,
  70.0,
  60.0
 ],
 "divergence_bullish::bear_false": [
  false,
  -1,
  -1,
  0.0,
  0.0,
  0.0,
  0.0
 ],
 "divergence_bearish::bear_false": [
  false,
  -1,
  -1,
  0.0,
  0.0,
  0.0,
  0.0
 ]
}""")


def A(*vals):
    return np.array([float(v) for v in vals])


def _same(actual, expected) -> bool:
    """Compare tuples/lists elementwise, treating NaN as equal to NaN."""
    if isinstance(actual, (tuple, list)) or isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            return False
        return all(_same(a, b) for a, b in zip(actual, expected))
    a, b = float(actual), float(expected)
    if np.isnan(a) and np.isnan(b):
        return True
    return a == b


def current_results() -> dict:
    """Recompute every case with the live implementation (same inputs as the baseline)."""
    cases = {}

    rsi_inputs = {
        "empty": (A(),),
        "mid": (A(*([50.0] * 10)),),
        "oversold_run": (A(60, 55, 40, 28, 26, 24, 22),),
        "at_threshold": (A(60, 40, 30.0),),
        "touch_below_once": (A(60, 29.0, 45.0),),
        "overbought_run": (A(40, 50, 65, 72, 75, 78, 80),),
        "near_flat": (A(29.9, 29.8, 29.7, 29.6),),
    }
    for name, (arr,) in rsi_inputs.items():
        cases[f"rsi_oversold::{name}"] = detect_rsi_oversold_numba(arr)
        cases[f"rsi_overbought::{name}"] = detect_rsi_overbought_numba(arr)
    cases["rsi_oversold::min_periods_3"] = detect_rsi_oversold_numba(A(40, 25, 24, 23), 30.0, 3)
    cases["rsi_overbought::min_periods_3"] = detect_rsi_overbought_numba(A(60, 75, 76, 77), 70.0, 3)

    stoch_inputs = {
        "empty": A(),
        "nan_last": A(50, 40, np.nan),
        "low": A(60, 40, 15),
        "boundary_low": A(60, 40, 20.0),
        "high": A(40, 60, 85),
        "boundary_high": A(40, 60, 80.0),
    }
    for name, arr in stoch_inputs.items():
        cases[f"stoch_oversold::{name}"] = detect_stoch_oversold_numba(arr)
        cases[f"stoch_overbought::{name}"] = detect_stoch_overbought_numba(arr)

    vol_inputs = {
        "short": A(100, 110),
        "nan_last": A(*([100.0] * 25), np.nan),
        "flat_spike": A(*([100.0] * 25), 300.0),
        "flat_no_spike": A(*([100.0] * 25), 150.0),
        "flat_dryup": A(*([100.0] * 25), 30.0),
        "flat_no_dryup": A(*([100.0] * 25), 80.0),
        "climax": A(*([100.0] * 55), 400.0),
        "negative_avg": A(*([-5.0] * 25), 100.0),
    }
    for name, arr in vol_inputs.items():
        cases[f"volume_spike::{name}"] = detect_volume_spike_numba(arr)
        cases[f"volume_dryup::{name}"] = detect_volume_dryup_numba(arr)
        cases[f"volume_climax::{name}"] = detect_climax_volume_numba(arr)

    series = {
        "short": (A(*range(8)), A(*range(8))),
        "flat": (A(*([100.0] * 40)), A(*([50.0] * 40))),
        "trend_down": (A(*[100 - i for i in range(40)]), A(*[60 - i for i in range(40)])),
    }
    rng = np.random.RandomState(7)
    series["seeded_7"] = (np.round(rng.uniform(90, 110, 60), 3), np.round(rng.uniform(20, 80, 60), 3))
    rng = np.random.RandomState(11)
    series["seeded_11"] = (np.round(rng.uniform(90, 110, 60), 3), np.round(rng.uniform(20, 80, 60), 3))
    for name, (p, ind) in series.items():
        cases[f"divergence_bullish::{name}"] = detect_bullish_divergence_numba(p, ind)
        cases[f"divergence_bearish::{name}"] = detect_bearish_divergence_numba(p, ind)

    high = A(101, 102, 103, 102, 104, 105)
    low = A(100, 101, 102, 101, 103, 104)
    for i in (1, 2, 3, 5):
        sar, ep, af = initialize_sar_arrays(len(high))
        sar[0] = 99.0
        ep[0] = 100.0
        af[0] = 0.02
        cases[f"sar_bullish::i{i}"] = (update_bullish_sar(i, high, low, sar, ep, af, 0.02, 0.2),
                                       round(float(sar[i]), 6), round(float(ep[i]), 6), round(float(af[i]), 6))
        sar, ep, af = initialize_sar_arrays(len(high))
        sar[0] = 200.0
        ep[0] = 199.0
        af[0] = 0.02
        cases[f"sar_bearish::i{i}"] = (update_bearish_sar(i, high, low, sar, ep, af, 0.02, 0.2),
                                       round(float(sar[i]), 6), round(float(ep[i]), 6), round(float(af[i]), 6))
    cases["sar_initial_state"] = tuple(round(float(x), 6) for x in get_initial_sar_state(high, low, 0.02))

    # Crafted series that MUST hit the True paths (bullish/bearish divergence + negatives).
    def _anchored(anchors, n=45):
        xs = [a[0] for a in anchors]
        ys = [a[1] for a in anchors]
        return np.interp(np.arange(n), xs, ys)

    crafted = {
        "bull_true": (_anchored([(0, 110), (15, 95), (22, 100), (30, 90), (44, 104)]),
                      _anchored([(0, 60), (15, 30), (22, 55), (30, 40), (44, 70)])),
        "bull_false": (_anchored([(0, 110), (15, 95), (22, 100), (30, 90), (44, 104)]),
                       _anchored([(0, 60), (15, 45), (22, 55), (30, 35), (44, 70)])),
        "bear_true": (_anchored([(0, 90), (15, 105), (22, 100), (30, 110), (44, 96)]),
                      _anchored([(0, 40), (15, 70), (22, 45), (30, 60), (44, 30)])),
        "bear_false": (_anchored([(0, 90), (15, 105), (22, 100), (30, 110), (44, 96)]),
                       _anchored([(0, 40), (15, 60), (22, 45), (30, 75), (44, 30)])),
    }
    for name, (p_arr, ind_arr) in crafted.items():
        cases[f"divergence_bullish::{name}"] = detect_bullish_divergence_numba(p_arr, ind_arr)
        cases[f"divergence_bearish::{name}"] = detect_bearish_divergence_numba(p_arr, ind_arr)

    return cases


def test_pattern_behaviour_matches_recorded_baseline():
    actual = current_results()
    assert set(actual) == set(BASELINE), (
        f"zestaw przypadkow sie zmienil: brakuje {sorted(set(BASELINE) - set(actual))}, "
        f"doszlo {sorted(set(actual) - set(BASELINE))}"
    )
    mismatches = {k: (actual[k], BASELINE[k]) for k in actual if not _same(actual[k], BASELINE[k])}
    assert not mismatches, f"zmiana zachowania w {len(mismatches)} przypadkach: {mismatches}"
