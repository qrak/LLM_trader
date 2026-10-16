"""
Volume Pattern Detection - Pure NumPy/Numba implementation

Detects volume spikes, dry-ups, and abnormal volume conditions.
Critical for confirming breakouts and identifying accumulation/distribution.
"""

import math

import numpy as np
from numba import njit


@njit
def _calculate_average_volume_numba(volume: np.ndarray, lookback: int) -> tuple:
    """
    Calculate average volume with NaN handling (helper function).
    Uses last closed candle (data now excludes incomplete candle at fetch time).
    Returns:
        (is_valid: bool, avg_volume: float, valid_count: int)
    """
    if len(volume) < lookback + 1:
        return False, 0.0, 0

    current_vol = volume[-1]

    if math.isnan(current_vol) or current_vol < 0:
        return False, 0.0, 0

    recent_volume = volume[-(lookback + 1):-1]

    valid_count = 0
    vol_sum = 0.0
    for i in range(len(recent_volume)):
        if not math.isnan(recent_volume[i]) and recent_volume[i] > 0:
            vol_sum += recent_volume[i]
            valid_count += 1

    if valid_count < lookback // 2:
        return False, 0.0, valid_count

    avg_vol = vol_sum / valid_count

    if avg_vol <= 0:
        return False, 0.0, valid_count

    return True, avg_vol, valid_count


@njit
def _detect_volume_ratio_numba(volume: np.ndarray, threshold: float, lookback: int, above: bool) -> tuple:
    """
    Compare the last closed candle's volume with the lookback average (shared core).
    Returns:
        (triggered: bool, current_volume: float, avg_volume: float, ratio: float)
    """
    is_valid, avg_vol, _ = _calculate_average_volume_numba(volume, lookback)

    if not is_valid:
        return False, 0.0, 0.0, 0.0

    current_vol = volume[-1]
    ratio = current_vol / avg_vol

    if above:
        triggered = ratio >= threshold
    else:
        triggered = ratio <= threshold

    if triggered:
        return True, float(current_vol), float(avg_vol), float(ratio)

    return False, float(current_vol), float(avg_vol), float(ratio)


@njit
def detect_volume_spike_numba(volume: np.ndarray, multiplier: float = 2.5, lookback: int = 20) -> tuple:
    """
    Detect volume spike: last closed candle volume significantly above average.

    Strong confirmation signal for breakouts when volume > multiplier * avg_volume.

    Returns:
        (is_spike, current_volume, avg_volume, spike_ratio)
    """
    return _detect_volume_ratio_numba(volume, multiplier, lookback, True)


@njit
def detect_volume_dryup_numba(volume: np.ndarray, threshold: float = 0.5, lookback: int = 20) -> tuple:
    """
    Detect volume dry-up: last closed candle volume significantly below average.

    Often precedes major moves - low volume indicates consolidation.

    Returns:
        (is_dryup, current_volume, avg_volume, dryup_ratio)
    """
    return _detect_volume_ratio_numba(volume, threshold, lookback, False)


@njit
def detect_climax_volume_numba(volume: np.ndarray, multiplier: float = 3.0, lookback: int = 50) -> tuple:
    """
    Detect climax volume: extreme volume spike indicating potential exhaustion.

    Very high volume often marks trend reversals (buying/selling climax).

    Returns:
        (is_climax, current_volume, avg_volume, climax_ratio)
    """
    return _detect_volume_ratio_numba(volume, multiplier, lookback, True)


@njit
def detect_volume_price_divergence_numba(volume: np.ndarray, prices: np.ndarray, lookback: int = 20) -> tuple:
    """
    Detect divergence between volume and price movement.
    Data already excludes incomplete candle at fetch time.

    Bearish: Price rising but volume declining (weak rally)
    Bullish: Price falling but volume declining (weak selloff)
    Returns:
        (found: bool, is_bearish: bool, price_change_pct: float, volume_change_pct: float)
    """
    if len(volume) < lookback or len(prices) < lookback:
        return False, False, 0.0, 0.0

    recent_volume = volume[-lookback:]
    recent_prices = prices[-lookback:]

    for i in range(len(recent_volume)):
        if math.isnan(recent_volume[i]) or math.isnan(recent_prices[i]):
            return False, False, 0.0, 0.0

    first_half_prices = recent_prices[:lookback // 2]
    second_half_prices = recent_prices[lookback // 2:]

    avg_first_price = np.mean(first_half_prices)
    avg_second_price = np.mean(second_half_prices)

    price_change_pct = ((avg_second_price / avg_first_price) - 1.0) * 100.0 if avg_first_price > 0 else 0.0

    first_half_volume = recent_volume[:lookback // 2]
    second_half_volume = recent_volume[lookback // 2:]

    avg_first_volume = np.mean(first_half_volume)
    avg_second_volume = np.mean(second_half_volume)

    volume_change_pct = ((avg_second_volume / avg_first_volume) - 1.0) * 100.0 if avg_first_volume > 0 else 0.0

    if price_change_pct > 2.0 and volume_change_pct < -15.0:
        return True, True, float(price_change_pct), float(volume_change_pct)

    if price_change_pct < -2.0 and volume_change_pct < -15.0:
        return True, False, float(price_change_pct), float(volume_change_pct)

    return False, False, 0.0, 0.0


@njit
def _up_volume_ratio_at(volume: np.ndarray, prices: np.ndarray, end: int, lookback: int) -> float:
    """Share of directional volume that traded on up bars as of bar `end`.

    Returns -1.0 for a window that is unusable (too short, misaligned, holding a
    NaN or carrying no directional volume).
    """
    if end < lookback or end >= len(volume) or end >= len(prices):
        return -1.0

    for i in range(end - lookback, end + 1):
        if math.isnan(volume[i]) or math.isnan(prices[i]):
            return -1.0

    up_volume = 0.0
    down_volume = 0.0

    for i in range(end - lookback + 1, end + 1):
        price_change = prices[i] - prices[i - 1]
        if price_change > 0:
            up_volume += volume[i]
        elif price_change < 0:
            down_volume += volume[i]

    total_volume = up_volume + down_volume

    if total_volume <= 0:
        return -1.0

    return up_volume / total_volume


@njit
def _accumulation_phase(up_volume_ratio: float) -> int:
    """1 = accumulation, -1 = distribution, 0 = neither (or an unusable window)."""
    if up_volume_ratio < 0.0:
        return 0
    if up_volume_ratio > 0.60:
        return 1
    if up_volume_ratio < 0.40:
        return -1
    return 0


@njit
def detect_accumulation_distribution_numba(volume: np.ndarray, prices: np.ndarray, lookback: int = 10) -> tuple:
    """
    Detect accumulation (buying pressure) or distribution (selling pressure).
    Data already excludes incomplete candle at fetch time.

    Accumulation: Volume increases on up days, decreases on down days
    Distribution: Volume increases on down days, decreases on up days
    Returns:
        (found: bool, is_accumulation: bool, strength: float, up_volume_ratio: float)
    """
    if len(volume) < lookback + 1 or len(prices) < lookback + 1:
        return False, False, 0.0, 0.0

    up_volume_ratio = _up_volume_ratio_at(volume, prices, len(volume) - 1, lookback)
    phase = _accumulation_phase(up_volume_ratio)

    if phase == 0:
        return False, False, 0.0, float(max(up_volume_ratio, 0.0))

    strength = abs(float(up_volume_ratio - 0.5)) * 2.0

    return True, phase > 0, strength, float(up_volume_ratio)


@njit
def accumulation_distribution_streak_numba(
    volume: np.ndarray,
    prices: np.ndarray,
    is_accumulation: bool,
    lookback: int = 10
) -> int:
    """
    Consecutive bars the same volume phase has been detected.

    The phase is a state, so this is how long it has already lasted: a long run is
    an established flow regime rather than a fresh shift. 0 for a phase that only
    started on the last bar's window boundary.
    """
    if len(volume) < lookback + 1 or len(prices) < lookback + 1:
        return 0

    phase = 1 if is_accumulation else -1

    bars = 0
    for end in range(len(volume) - 1, lookback - 1, -1):
        if _accumulation_phase(_up_volume_ratio_at(volume, prices, end, lookback)) != phase:
            break
        bars += 1
    return bars
