import numpy as np
from numba import njit


@njit(cache=True)
def log_return_numba(close, length=1, cumulative=False):
    n = len(close)
    log_return = np.full(n, np.nan)

    if cumulative:
        for i in range(length, n):
            log_return[i] = np.log(close[i] / close[0])
    else:
        for i in range(length, n):
            log_return[i] = np.log(close[i] / close[i - length])

    return log_return

@njit(cache=True)
def percent_return_numba(close, length=1, cumulative=False):
    n = len(close)
    pct_return = np.full(n, np.nan)

    if cumulative:
        for i in range(length, n):
            pct_return[i] = (close[i] / close[0]) - 1
    else:
        for i in range(length, n):
            pct_return[i] = (close[i] / close[i - length]) - 1

    return pct_return

@njit(cache=True)
def pdist_numba(open_, high, low, close, drift=1):
    n = len(open_)
    pdist = np.full(n, np.nan)

    for i in range(n):
        if i - drift >= 0:
            pdist[i] = 2 * (high[i] - low[i]) - abs(close[i] - open_[i]) + abs(open_[i] - close[i - drift])
        else:
            pdist[i] = np.nan

    return pdist
