"""
Correlation Analysis Utilities for Statistical Indicators
Contains functions for calculating correlation and spectral analysis.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def calculate_correlation_matrix(filt, maxlen, avelen):
    """Calculate correlation matrix for the filtered data"""
    corr = np.zeros(maxlen * 2)
    
    for lag in range(maxlen):
        m = avelen if avelen != 0 else lag
        sx, sy, sxx, syy, sxy = 0.0, 0.0, 0.0, 0.0, 0.0
        
        for i in range(m):
            x = filt[i]
            y = filt[lag + i]
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y
            syy += y * y
            
        denominator = (m * sxx - sx * sx) * (m * syy - sy * sy)
        if denominator > 0:
            corr[lag] = (m * sxy - sx * sy) / np.sqrt(denominator)
    
    return corr


@njit(cache=True)
def calculate_spectral_components(corr, minlen, maxlen, avelen):
    """Calculate spectral components using correlation data"""
    c = 2 * np.pi
    cospart = np.zeros(maxlen * 2)
    sinpart = np.zeros(maxlen * 2)
    sqsum = np.zeros(maxlen * 2)
    
    for period in range(minlen, maxlen):
        cospart[period] = 0
        sinpart[period] = 0
        for n in range(avelen, maxlen):
            cospart[period] += corr[n] * np.cos(c * n / period)
            sinpart[period] += corr[n] * np.sin(c * n / period)
        sqsum[period] = cospart[period] ** 2 + sinpart[period] ** 2
    
    return sqsum


@njit(cache=True)
def smooth_power_spectrum(sqsum, minlen, maxlen):
    """Apply smoothing to the power spectrum"""
    r1 = np.zeros(maxlen * 2)
    r2 = np.zeros(maxlen * 2)
    
    for period in range(minlen, maxlen):
        r2[period] = r1[period]
        r1[period] = 0.2 * sqsum[period] ** 2 + 0.8 * r2[period]
    
    return r1


@njit(cache=True)
def calculate_dominant_cycle(r1, minlen, maxlen, avelen):
    """Calculate the dominant cycle from power spectrum"""
    maxpwr = np.max(r1[minlen:maxlen])
    
    if maxpwr == 0:
        return 1
    
    pwr = np.zeros(maxlen * 2)
    for period in range(avelen, maxlen):
        pwr[period] = r1[period] / maxpwr
    
    peakpwr = np.max(pwr[minlen:maxlen])
    spx, sp = 0.0, 0.0
    
    # First pass: high power periods
    for period in range(minlen, maxlen):
        if pwr[period] >= 0.5:
            spx += period * pwr[period]
            sp += pwr[period]
    
    # Second pass: medium power periods if peak is significant
    for period in range(minlen, maxlen):
        if peakpwr >= 0.25 and pwr[period] >= 0.25:
            spx += period * pwr[period]
            sp += pwr[period]
    
    dominantcycle = spx / sp if sp != 0 else 0
    dominantcycle = max(dominantcycle, 1)
    return dominantcycle
