# 📊 Numba JIT Indicator Library — 50+ Home-Grown Technical Indicators

> **Module path:** `src/indicators/` (9 category directories + `base/`)
> **Type:** Self-contained technical analysis library, Numba JIT-compiled
> **Size:** ~96 `@njit(cache=True)` functions across 13 Python files
> **Design Decision:** Built from scratch instead of using TA-Lib / pandas-ta — 0 external TA dependencies

---

## Why A Custom Indicator Library?

LLM Trader does **not** use TA-Lib or pandas-ta. Every indicator is implemented from scratch in pure Python + Numba `@njit` for JIT compilation. This was a deliberate architectural choice:

| Concern | TA-Lib | This Library |
|---------|--------|-------------|
| **Dependencies** | C library, platform-specific builds, `.dll`/`.so` hell | Pure Python + Numba (already a dependency) |
| **Multi-timeframe isolation** | Single global state | `TechnicalCalculator` creates fresh `TechnicalIndicators` instances per timeframe |
| **NaN handling** | Inconsistent per indicator | Uniform: pre-fill with `np.nan`, first valid at index `length` |
| **Rolling window bugs** | None | Several found and fixed: CCI O(N×L)→O(N), stochastic NaN bleed-through, MACD 0.0 sentinel |
| **Customization** | Fork or wrapper layer | Direct: `@njit` your own variant alongside originals |
| **Modularity** | One giant DLL | 9 category modules, tree-shakeable imports |

---

## Architecture

### Layer Diagram

```
TechnicalIndicators (facade — 890 lines, methods directly on class)
  └── IndicatorBase (data holder — 160 lines)
        ├── get_data(OHLCV) → numpy arrays
        └── calculate_indicator(func, *args) → timing + CSV logging wrapper

Category Modules (each a single .py file)
  ├── momentum_indicators.py     — 14 @njit functions (RSI, MACD, Stochastic, ...)
  ├── trend_indicators.py        — 8 @njit + 8 trend_calculation_utils + 4 sar_utils
  ├── volume_indicators.py       — 13 @njit (MFI, OBV, CMF, CCI, VWAP, TWAP, ...)
  ├── volatility_indicators.py   — 8 @njit (ATR, Bollinger, Keltner, Choppiness, ...)
  ├── statistical_indicators.py  — 14 @njit + 4 correlation + 2 DSP filters
  ├── support_resistance_indicators.py — 9 @njit (pivot points, Fibonacci, floating levels)
  ├── overlap_indicators.py      — 3 @njit (SMA, EMA, EWMA)
  ├── price_transform_indicators.py — 3 @njit (log return, % return, price distribution)
  └── sentiment_indicators.py    — 6 @njit (Fear & Greed Index variants)
```

### Import Structure

All category functions are re-exported at the module level via `__init__.py` imports. The `TechnicalIndicators` class in `base/technical_indicators.py` imports them all and exposes each as a direct method:

```python
ti = TechnicalIndicators()
ti.get_data(ohlcv_array)
rsi_values = ti.rsi(length=14)         # Direct — no delegation
macd_line, signal, hist = ti.macd()
```

There is **no delegation layer** — the category sub-object pattern was eliminated in a refactor. Every indicator is a one-line direct method call.

---

## Numba JIT Compilation Pattern

### Standard Template

Every indicator function follows this pattern:

```python
@njit(cache=True)
def rsi_numba(close: np.ndarray, length: int) -> np.ndarray:
    n = len(close)
    result = np.full(n, np.nan)       # Pre-fill NaN

    # Sliding window computation — no Python list allocations
    for i in range(length, n):
        ...

    return result
```

Key rules:
- **`@njit(cache=True)`** — JIT compiled, cached to disk after first call (avoids recompilation on restart)
- **Input is always `np.ndarray`** — float64 preferred, int64 auto-converted
- **Output is always `np.ndarray`** — pre-allocated with `np.full(n, np.nan)`
- **No Python objects in hot loops** — lists, dicts, and function calls inside loops are forbidden by Numba
- **`math` module** — `math.nan`, `math.isnan()`, `math.inf` are allowed (compiled to C)
- **Return type** — scalar (single value), 1D array, or tuple of arrays

### Performance Baseline

| Indicator | 1000 candles | 100k candles | Speedup vs vanilla Python |
|-----------|-------------|--------------|--------------------------|
| RSI(14) | ~0.0001s | ~0.002s | ~50× |
| MACD(12,26,9) | ~0.0002s | ~0.005s | ~80× |
| CCI(14) | ~0.0003s | ~0.004s | ~6× (fixed from bug) |
| Bollinger (20,2) | ~0.0002s | ~0.003s | ~60× |
| ADX(14) | ~0.0005s | ~0.008s | ~40× |

**Real-world workload:** Computing all 40+ indicators across 999 candles takes ~15-25ms total.

---

## Indicator Inventory (Complete)

### Momentum (14 functions)
RSI, MACD (line/signal/histogram), Stochastic (%K/%D), ROC, Momentum, Williams %R, TSI, RMI, PPO, Coppock Curve, Ultimate Oscillator, KST, Relative Strength calculation, RSI divergence detection

### Trend (8 functions + 12 utility functions)
ADX (+DI/-DI), Supertrend, Ichimoku Cloud (tenkan/kijun/senkou/chikou), Parabolic SAR, Vortex (+VI/-VI), TRIX, PFE, TD Sequential (setup/countdown)

**Utility files:**
- `trend_calculation_utils.py` — true range, ATR helper, directional movement, rolling true range sum
- `sar_utils.py` — acceleration factor logic, SAR point stepping

### Volume (13 functions)
MFI, OBV, OBV Slope, PVT, Chaikin Money Flow, Accumulation/Distribution Line, Force Index, Ease of Movement, Volume Profile, Rolling VWAP, TWAP, Average Quote Volume, CCI

### Volatility (8 functions)
ATR, Bollinger Bands (upper/lower/%B/width), Chandelier Exit (long/short), VHF, EBSW, Keltner Channels (upper/lower), Donchian Channels (upper/lower), Choppiness Index

### Statistical (14 functions + 4 correlation + 2 DSP)
Kurtosis, Skewness, Standard Deviation, Variance, Z-Score, MAD, Quantile, Entropy, Hurst Exponent, Linear Regression (slope/r²/intercept), APA Adaptive EOT, EOT Calculation

**Utility sub-package** `statistical/utils/`:
- `correlation_analysis.py` — 4 functions: autocorrelation, rolling correlation, cross-correlation, Spearman rank
- `dsp_filters.py` — 2 functions: low-pass filter, high-pass filter (basic IIR-style)

### Support/Resistance (9 functions)
Support & Resistance (basic), Find S/R (swing-point based), Advanced S/R (cluster detection), Pivot Points (classic), Fibonacci Pivot Points, Fibonacci Retracement, Floating Levels, Fibonacci Bollinger Bands

### Overlap (3 functions)
SMA, EMA, EWMA (all support array inputs for vectorized calculation)

### Price Transforms (3 functions)
Log Return, Percent Return, Price Distribution

### Sentiment (6 functions)
Fear & Greed Index (5 market-based variants), Fear & Greed with configurable thresholds

---

## Multi-Timeframe Isolation

Each analysis cycle creates **three isolated `TechnicalIndicators` instances** via `TechnicalCalculator`:

| Instance | Candle Data | Purpose |
|----------|-------------|---------|
| Current timeframe indicators | 999 candles (4h) | Primary cycle analysis |
| Long-term indicators | 365 daily candles | Long-term trend context |
| Weekly macro indicators | 300 weekly candles | Macro cycle phase |

Each instance holds its own `open/high/low/close/volume` numpy arrays. State interference between timeframes is impossible by construction.

---

## NaN Propagation Convention

Every indicator uses a **uniform NaN strategy**:

1. **Output array**: Pre-filled with `np.nan`
2. **First valid index**: Indicator-specific, typically based on each function's `required_length` (for example RSI(14) starts at index 14, while some multi-parameter indicators can start earlier or later)
3. **Insufficient data**: If `len(data) < length`, entire output is NaN
4. **Division by zero**: Checked with `if avg_loss == 0` guards; result set to 100 (RSI), 0 (CCI), etc.
5. **NaN in input**: Not handled explicitly — caller guarantees clean data (DataFetcher excludes incomplete candles upstream)

This convention means downstream consumers (TechnicalCalculator, prompt formatters) must handle NaN at array boundaries — which they do by slicing to the visible candle count.

---

## Correctness History

Several bugs were found and fixed during development that would have been invisible with TA-Lib:

| Bug | Symptom | Root Cause | Fix |
|-----|---------|-----------|-----|
| **CCI drifting** | Values diverged from reference as data grew | `np.roll()` allocating new arrays each iteration → O(N×L) with accumulation error | Single-pass sliding sum → O(N) |
| **Stochastic NaN bleed** | First %K value was 0.0 instead of NaN | Missing NaN assignment before first valid period | Pre-fill with `np.full(n, np.nan)` |
| **MACD 0.0 sentinel** | First histogram values were literal 0.0 | MACD line started at idx 25 but histogram used 0.0 as unset sentinel | Explicit NaN pre-fill |
| **Bollinger %B** | Out-of-bounds values (>1.0 or <0.0) on very first valid candle | Division using incomplete rolling window | Skip %B calculation until `length` candles into window |

---

## Edge Cases & Guardrails

| Scenario | Handling |
|----------|----------|
| **Zero volume array** | Division by zero guards in VWAP, MFI, CMF — result `np.nan` with downstream fallback |
| **Flat price (all identical)** | RSI = 100 (static at oversold boundary), ADX = 0, volatility = 0 |
| **Single candle** | All indicators return all-NaN — `required_length` validation catches this upstream |
| **Non-float64 input** | `get_data()` normalizes to `np.float64` — int/timestamps auto-converted |
| **Extreme values (>1e10)** | Floating point saturation possible — no explicit clamp (BTC/USDC at <10⁶ is safe) |
| **Memory fragmentation** | Each cycle creates 3× indicator instances (~15MB of arrays) — GC collects between cycles |
