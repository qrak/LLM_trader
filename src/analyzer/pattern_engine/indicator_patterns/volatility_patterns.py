"""Volatility Pattern Detection - Pure NumPy/Numba Implementation

Detects volatility-based patterns:
1. ATR spikes (sudden volatility increase - risk warning)
2. Bollinger Band squeezes (band width at the low end of its own range)

All functions use @njit for performance.

``BB_SQUEEZE_PERCENTILE`` / ``BB_SQUEEZE_LOOKBACK`` are the Bollinger squeeze
threshold: the band width has to sit inside the lowest ``BB_SQUEEZE_PERCENTILE``
percent of the trailing ``BB_SQUEEZE_LOOKBACK`` widths. The pattern engine quotes
both numbers in the prompt, so they live here once.
"""


import numpy as np
from numba import njit

BB_SQUEEZE_PERCENTILE = 20.0
BB_SQUEEZE_LOOKBACK = 120


@njit(cache=True)
def detect_atr_spike_numba(
    atr: np.ndarray,
    spike_threshold: float = 1.5,
    lookback: int = 14
) -> tuple[bool, int, float, float]:
    """
    Detect sudden ATR spikes (volatility explosion).

    ATR spike = Current ATR is significantly higher than recent average.
    Indicates sudden volatility increase, often at trend reversals or breakouts.
    Returns:
        (spike_detected, periods_ago, current_atr, average_atr)
    """
    if len(atr) < lookback + 1:
        return (False, -1, 0.0, 0.0)

    current_atr = atr[-1]

    recent_atr = atr[-(lookback+1):-1]
    avg_atr = np.mean(recent_atr)

    if avg_atr < 0.0001:
        return (False, -1, current_atr, avg_atr)  # type: ignore

    spike_ratio = current_atr / avg_atr

    if spike_ratio >= spike_threshold:
        return (True, 0, current_atr, avg_atr)  # type: ignore

    return (False, -1, current_atr, avg_atr)  # type: ignore


@njit(cache=True)
def detect_bb_squeeze_numba(
    bb_upper: np.ndarray,
    bb_lower: np.ndarray,
    squeeze_percentile: float = 20.0,
    lookback: int = 120
) -> tuple[bool, float, float, float]:
    """
    Detect Bollinger Band squeeze (low volatility).

    BB squeeze = the current band width sits in the lowest `squeeze_percentile`
    percent of the last `lookback` widths. The window is 120 candles (~20 days on
    a 4h chart) so the reading is relative to a real price history instead of a
    handful of candles.
    Returns:
        (squeeze_detected, current_width, percentile_width, width_percentile)
        - squeeze_detected: True if the current width is at or below the threshold
        - current_width: current band width (price units)
        - percentile_width: threshold width at `squeeze_percentile` (price units)
        - width_percentile: share of the window narrower than now, 0..100
          (0 = tightest band in the window, 100 = widest) — scale-free, this is
          what a confidence score must be derived from
    """
    if len(bb_upper) < lookback or len(bb_lower) < lookback:
        return (False, 0.0, 0.0, 0.0)

    current_width, percentile_width, width_percentile = _bb_width_position_at(
        bb_upper, bb_lower, len(bb_upper) - 1, squeeze_percentile, lookback
    )

    if current_width <= percentile_width:
        return (True, current_width, percentile_width, width_percentile)

    return (False, current_width, percentile_width, width_percentile)


@njit(cache=True)
def _bb_width_position_at(
    bb_upper: np.ndarray,
    bb_lower: np.ndarray,
    end: int,
    squeeze_percentile: float,
    lookback: int
) -> tuple[float, float, float]:
    """Band width of bar `end` and its position inside the trailing width window."""
    start = end - lookback + 1
    widths = bb_upper[start:end + 1] - bb_lower[start:end + 1]
    current_width = widths[lookback - 1]

    sorted_widths = np.sort(widths)
    percentile_index = int(lookback * squeeze_percentile / 100.0)
    percentile_index = max(0, min(percentile_index, lookback - 1))
    percentile_width = sorted_widths[percentile_index]

    narrower = 0
    for i in range(lookback):
        if widths[i] < current_width:
            narrower += 1

    return current_width, percentile_width, 100.0 * narrower / lookback


@njit(cache=True)
def bb_squeeze_streak_numba(
    bb_upper: np.ndarray,
    bb_lower: np.ndarray,
    squeeze_percentile: float = 20.0,
    lookback: int = 120
) -> int:
    """
    Consecutive bars the band width has stayed inside the squeeze threshold.

    A squeeze is a state, so this is how long it has already lasted: a long run
    is an old compression, not a fresh setup. 0 when the last bar is not squeezed.
    """
    n = len(bb_upper)
    if n < lookback or len(bb_lower) < lookback:
        return 0

    bars = 0
    for end in range(n - 1, lookback - 2, -1):
        current_width, percentile_width, _ = _bb_width_position_at(
            bb_upper, bb_lower, end, squeeze_percentile, lookback
        )
        if current_width > percentile_width:
            break
        bars += 1
    return bars


@njit(cache=True)
def _volatility_trend_at(atr: np.ndarray, end: int, lookback: int) -> int:
    """The volatility-trend rule evaluated as of bar `end`."""
    if end + 1 < lookback:
        return 0

    increasing = 0
    decreasing = 0

    for i in range(end - lookback + 2, end + 1):
        diff = atr[i] - atr[i - 1]
        if diff > 0.0:
            increasing += 1
        elif diff < 0.0:
            decreasing += 1

    threshold = (lookback - 1) * 0.6

    if float(increasing) >= threshold:
        return 1
    if float(decreasing) >= threshold:
        return -1
    return 0


@njit(cache=True)
def detect_volatility_trend_numba(
    atr: np.ndarray,
    lookback: int = 10
) -> int:
    """
    Detect trend in volatility (increasing, decreasing, or stable).
    Returns:
        1 = increasing volatility, -1 = decreasing volatility, 0 = stable
    """
    if len(atr) < lookback:
        return 0

    return _volatility_trend_at(atr, len(atr) - 1, lookback)


@njit(cache=True)
def volatility_trend_streak_numba(
    atr: np.ndarray,
    lookback: int = 10
) -> int:
    """
    Consecutive bars the volatility-trend rule has returned its current value.

    The trend is a state, so this is how long it has already held: a long run
    describes an established regime rather than a fresh change. 0 for a stable
    reading, which carries no state to measure.
    """
    n = len(atr)
    if n < lookback:
        return 0

    current = _volatility_trend_at(atr, n - 1, lookback)
    if current == 0:
        return 0

    bars = 0
    for end in range(n - 1, lookback - 2, -1):
        if _volatility_trend_at(atr, end, lookback) != current:
            break
        bars += 1
    return bars


@njit(cache=True)
def detect_keltner_squeeze_numba(
    kc_upper: np.ndarray,
    kc_lower: np.ndarray,
    bb_upper: np.ndarray,
    bb_lower: np.ndarray
) -> tuple[bool, int, int]:
    """
    Detect TTM Squeeze (Bollinger Bands inside Keltner Channels).

    This is John Carter's TTM Squeeze indicator: the published pairing is
    BB(20, 2 sigma) against KC(20, 1.5 x ATR), so the two band sets handed in
    must come from those settings for the reading to mean compression in the
    canonical sense.
    Returns:
        (in_squeeze, bars_in_squeeze, bars_since_release)
        - in_squeeze: True when the BB sit inside the KC on the last bar
        - bars_in_squeeze: consecutive bars the bands have been inside, 0 when
          off. The squeeze is a state, not an event, so a long run describes an
          old compression rather than a fresh setup.
        - bars_since_release: bars since the bands last left the channel, 0 when
          they left on the last bar, -1 when the series never carried a squeeze
    """
    n = len(kc_upper)
    if n < 1 or len(bb_upper) < n or len(kc_lower) < n or len(bb_lower) < n:
        return (False, 0, -1)

    inside = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        inside[i] = bb_upper[i] <= kc_upper[i] and bb_lower[i] >= kc_lower[i]

    bars_in_squeeze = 0
    while bars_in_squeeze < n and inside[n - 1 - bars_in_squeeze]:
        bars_in_squeeze += 1

    if bars_in_squeeze > 0:
        return (True, bars_in_squeeze, -1)

    for i in range(n - 1, 0, -1):
        if inside[i - 1] and not inside[i]:
            return (False, 0, (n - 1) - i)

    return (False, 0, -1)
