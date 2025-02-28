import numpy as np
from numba import njit

@njit(cache=True)
def fear_and_greed_index_numba(close, high, low, volume, rsi_length, macd_fast_length, macd_slow_length,
                               macd_signal_length, mfi_length, window_size):
    n = len(close)
    fear_and_greed_index = np.full(n, np.nan)

    for start in range(n - window_size + 1):
        end = start + window_size
        window_close = close[start:end]
        window_high = high[start:end]
        window_low = low[start:end]
        window_volume = volume[start:end]

        rsi_list = np.full(window_size, np.nan)
        gains = np.maximum(0, window_close[1:] - window_close[:-1])
        losses = np.maximum(0, window_close[:-1] - window_close[1:])

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

        macd_list = np.full(window_size, np.nan, dtype=np.float64)
        signal_list = np.full(window_size, np.nan, dtype=np.float64)
        histogram_list = np.full(window_size, np.nan, dtype=np.float64)

        fast_ema = np.mean(window_close[:macd_fast_length])
        slow_ema = np.mean(window_close[:macd_slow_length])
        signal = np.nan

        multiplier_fast = 2 / (macd_fast_length + 1)
        multiplier_slow = 2 / (macd_slow_length + 1)
        multiplier_signal = 2 / (macd_signal_length + 1)

        for i in range(1, window_size):
            fast_ema = (window_close[i] - fast_ema) * multiplier_fast + fast_ema
            slow_ema = (window_close[i] - slow_ema) * multiplier_slow + slow_ema

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

        mfi_list = np.full(window_size, np.nan)
        tp = (window_high + window_low + window_close) / 3
        rmf = tp * window_volume

        for i in range(mfi_length, window_size):
            pmf = np.sum(rmf[i - mfi_length + 1:i + 1][tp[i - mfi_length + 1:i + 1] > tp[i - mfi_length:i]])
            nmf = np.sum(rmf[i - mfi_length + 1:i + 1][tp[i - mfi_length + 1:i + 1] < tp[i - mfi_length:i]])

            if nmf == 0:
                mfi_list[i] = 100
            else:
                mfr = pmf / nmf
                mfi_list[i] = 100 * mfr / (1 + mfr)

        for i in range(max(rsi_length, macd_slow_length + macd_signal_length - 1, mfi_length), window_size):
            normalized_rsi = (rsi_list[i] - 30) / (70 - 30) * 100
            if normalized_rsi < 0:
                normalized_rsi = 0
            elif normalized_rsi > 100:
                normalized_rsi = 100

            max_histogram = np.nanmax(histogram_list[max(0, i - window_size):i])
            min_histogram = np.nanmin(histogram_list[max(0, i - window_size):i])

            if max_histogram == min_histogram:
                normalized_macd_histogram = 0
            else:
                normalized_macd_histogram = (histogram_list[i] - min_histogram) / (max_histogram - min_histogram) * 100
            if normalized_macd_histogram < 0:
                normalized_macd_histogram = 0
            elif normalized_macd_histogram > 100:
                normalized_macd_histogram = 100

            normalized_mfi = mfi_list[i]
            if normalized_mfi < 0:
                normalized_mfi = 0
            elif normalized_mfi > 100:
                normalized_mfi = 100

            fear_and_greed_index[start + i] = (normalized_rsi + normalized_macd_histogram + normalized_mfi) / 3

    fear_and_greed_index = np.nan_to_num(fear_and_greed_index, nan=50)

    return fear_and_greed_index
