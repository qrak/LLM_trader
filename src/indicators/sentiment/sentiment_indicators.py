import numpy as np
from numba import njit
from dataclasses import dataclass


@dataclass
class FearGreedConfig:
    """Configuration for Fear and Greed Index."""
    rsi_length: int = 14
    macd_fast_length: int = 12
    macd_slow_length: int = 26
    macd_signal_length: int = 9
    mfi_length: int = 14
    window_size: int = 50

@njit(cache=True)
def _calculate_rsi_window(close, rsi_length):
    """Calculate RSI for a window of data."""
    window_size = len(close)
    rsi_list = np.full(window_size, np.nan)
    gains = np.maximum(0, close[1:] - close[:-1])
    losses = np.maximum(0, close[:-1] - close[1:])

    avg_gain = np.sum(gains[:rsi_length]) / rsi_length
    avg_loss = np.sum(losses[:rsi_length]) / rsi_length

    if avg_loss == 0:
        rsi_list[rsi_length - 1] = 100
    else:
        rs = avg_gain / avg_loss
        rsi_list[rsi_length - 1] = 100 - (100 / (1 + rs))

    for i in range(rsi_length, window_size):
        avg_gain = ((avg_gain * (rsi_length - 1)) + gains[i - 1]) / rsi_length
        avg_loss = ((avg_loss * (rsi_length - 1)) + losses[i - 1]) / rsi_length
        if avg_loss == 0:
            rsi_list[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi_list[i] = 100 - (100 / (1 + rs))
    
    return rsi_list


@njit(cache=True)
def _calculate_macd_window(close, macd_fast_length, macd_slow_length, macd_signal_length):
    """Calculate MACD for a window of data."""
    window_size = len(close)
    macd_list = np.full(window_size, np.nan, dtype=np.float64)
    signal_list = np.full(window_size, np.nan, dtype=np.float64)
    histogram_list = np.full(window_size, np.nan, dtype=np.float64)

    fast_ema = np.mean(close[:macd_fast_length])
    slow_ema = np.mean(close[:macd_slow_length])
    signal = np.nan

    multiplier_fast = 2 / (macd_fast_length + 1)
    multiplier_slow = 2 / (macd_slow_length + 1)
    multiplier_signal = 2 / (macd_signal_length + 1)

    for i in range(1, window_size):
        fast_ema = (close[i] - fast_ema) * multiplier_fast + fast_ema
        slow_ema = (close[i] - slow_ema) * multiplier_slow + slow_ema

        if i >= macd_slow_length - 1:
            macd = fast_ema - slow_ema
            macd_list[i] = macd

            if not np.isnan(macd) and np.isnan(signal):
                signal = macd

            if i >= macd_slow_length + macd_signal_length - 2:
                if not np.isnan(signal):
                    signal = (macd - signal) * multiplier_signal + signal
                    signal_list[i] = signal
                    histogram_list[i] = macd - signal
    
    return macd_list, signal_list, histogram_list


@njit(cache=True)
def _calculate_mfi_window(high, low, close, volume, mfi_length):
    """Calculate MFI for a window of data."""
    window_size = len(close)
    mfi_list = np.full(window_size, np.nan)
    tp = (high + low + close) / 3
    rmf = tp * volume

    for i in range(mfi_length, window_size):
        pmf = np.sum(rmf[i - mfi_length + 1:i + 1][tp[i - mfi_length + 1:i + 1] > tp[i - mfi_length:i]])
        nmf = np.sum(rmf[i - mfi_length + 1:i + 1][tp[i - mfi_length + 1:i + 1] < tp[i - mfi_length:i]])

        if nmf == 0:
            mfi_list[i] = 100
        else:
            mfr = pmf / nmf
            mfi_list[i] = 100 * mfr / (1 + mfr)
    
    return mfi_list


@njit(cache=True)
def _normalize_value(value, min_val, max_val):
    """Normalize a value to 0-100 range."""
    if min_val == max_val:
        return 0
    normalized = (value - min_val) / (max_val - min_val) * 100
    return max(0, min(100, normalized))


@njit(cache=True)
def _calculate_fear_greed_for_window(rsi_list, histogram_list, mfi_list, window_size, 
                                   rsi_length, macd_slow_length, macd_signal_length, mfi_length):
    """Calculate fear and greed index for a specific window."""
    start_idx = max(rsi_length, macd_slow_length + macd_signal_length - 1, mfi_length)
    result = np.full(window_size, np.nan)
    
    for i in range(start_idx, window_size):
        # Normalize RSI
        normalized_rsi = (rsi_list[i] - 30) / (70 - 30) * 100
        normalized_rsi = max(0, min(100, normalized_rsi))

        # Normalize MACD histogram
        max_histogram = np.nanmax(histogram_list[max(0, i - window_size):i])
        min_histogram = np.nanmin(histogram_list[max(0, i - window_size):i])
        normalized_macd_histogram = _normalize_value(histogram_list[i], min_histogram, max_histogram)

        # Normalize MFI
        normalized_mfi = max(0, min(100, mfi_list[i]))

        result[i] = (normalized_rsi + normalized_macd_histogram + normalized_mfi) / 3
    
    return result


@njit(cache=True)
def _fear_and_greed_index_numba(close, high, low, volume, rsi_length, macd_fast_length, macd_slow_length,
                               macd_signal_length, mfi_length, window_size):
    n = len(close)
    fear_and_greed_index = np.full(n, np.nan)

    for start in range(n - window_size + 1):
        end = start + window_size
        window_close = close[start:end]
        window_high = high[start:end]
        window_low = low[start:end]
        window_volume = volume[start:end]

        # Calculate indicators for this window
        rsi_list = _calculate_rsi_window(window_close, rsi_length)
        macd_list, signal_list, histogram_list = _calculate_macd_window(
            window_close, macd_fast_length, macd_slow_length, macd_signal_length)
        mfi_list = _calculate_mfi_window(window_high, window_low, window_close, window_volume, mfi_length)

        # Calculate fear and greed for this window
        window_fg = _calculate_fear_greed_for_window(
            rsi_list, histogram_list, mfi_list, window_size,
            rsi_length, macd_slow_length, macd_signal_length, mfi_length)
        
        # Copy results to main array
        for i in range(window_size):
            if not np.isnan(window_fg[i]):
                fear_and_greed_index[start + i] = window_fg[i]

    fear_and_greed_index = np.nan_to_num(fear_and_greed_index, nan=50)
    return fear_and_greed_index


def fear_and_greed_index_numba(
    close: np.ndarray, 
    high: np.ndarray, 
    low: np.ndarray, 
    volume: np.ndarray, 
    config: FearGreedConfig
) -> np.ndarray:
    """
    Fear and Greed Index - Simple interface using config object.
    
    Calculates a composite sentiment indicator based on RSI, MACD, and MFI.
    
    Args:
        close: Close prices
        high: High prices  
        low: Low prices
        volume: Volume data
        config: Configuration object containing all parameters
        
    Returns:
        Fear and greed index array (0-100, where <30 is fear, >70 is greed)
    """
    return _fear_and_greed_index_numba(
        close, high, low, volume,
        config.rsi_length, config.macd_fast_length, config.macd_slow_length,
        config.macd_signal_length, config.mfi_length, config.window_size
    )
