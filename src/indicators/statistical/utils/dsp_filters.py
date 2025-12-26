"""
Digital Signal Processing Utilities for Statistical Indicators
Contains filtering functions used in statistical analysis.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def f_ess(source, length_):
    """Enhanced super smoother filter function"""
    s = 1.414
    a = np.exp(-s * np.pi / length_)
    b = 2 * a * np.cos(s * np.pi / length_)
    c2 = b
    c3 = -a * a
    c1 = 1 - c2 - c3
    out = np.zeros_like(source)
    for i in range(2, len(source)):
        out[i] = c1 * (source[i] + source[i - 1]) / 2 + c2 * out[i - 1] + c3 * out[i - 2]
    return out


@njit(cache=True)
def f_hp(source, maxlen):
    """High-pass filter function"""
    c = 360 * np.pi / 180
    alpha = (1 - np.sin(c / maxlen)) / np.cos(c / maxlen)
    hp = np.zeros_like(source)
    for i in range(1, len(source)):
        hp[i] = 0.5 * (1 + alpha) * (source[i] - source[i - 1]) + alpha * hp[i - 1]
    return hp
