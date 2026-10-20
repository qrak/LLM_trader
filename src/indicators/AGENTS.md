# AGENTS.md — Indicator Library

Scope: `src/indicators/`. Root [`AGENTS.md`](../../AGENTS.md) is canonical.

No TA-Lib, no pandas-ta: 79 `@njit(cache=True)` functions across 12 modules, pure Python + Numba. Never add an external TA dependency, never wrap these in a delegation layer — `TechnicalIndicators` exposes each indicator as a direct method, so a new indicator is one `@njit` function plus one method and one `__init__.py` export.

| Module | Dir | Content |
|---|---|---|
| `momentum_indicators.py` | `momentum/` | 14 — RSI, MACD, Stochastic, ROC, Momentum, Williams %R, TSI, RMI, PPO, Coppock, UO, KST, divergence |
| `trend_indicators.py` | `trend/` | 8 — ADX(+DI/-DI), Supertrend, Ichimoku, PSAR, Vortex, TRIX, PFE, TD Setup (setup phase only, no countdown) |
| `trend_calculation_utils.py` | `trend/` | 8 — true range, ATR helper, directional movement, rolling TR sum |
| `sar_utils.py` | `trend/` | 5 — acceleration factor, SAR stepping |
| `volume_indicators.py` | `volume/` | 12 — MFI, OBV, Net Flow Ratio, PVT, CMF, A/D, Force Index, EOM, Volume Profile, VWAP, TWAP, CCI |
| `volatility_indicators.py` | `volatility/` | 8 — ATR, Bollinger, Chandelier, VHF, EBSW, Keltner, Donchian, Choppiness |
| `statistical_indicators.py` | `statistical/` | 9 — moments, quantile, entropy, Hurst, regression |
| `support_resistance_indicators.py` | `support_resistance/` | 4 — rolling range, retested levels (3+ touches, ±0.5%, 120 bars), classic and Fibonacci pivots |
| `overlap_indicators.py` | `overlap/` | 2 — SMA, EMA (array inputs, vectorized) |
| `price_transform_indicators.py` | `price/` | 2 — log return, price distribution |
| `risk_reward.py` | `price/` | 1 — Numba R/R arithmetic shared with `RiskManager` |
| `sentiment_indicators.py` | `sentiment/` | 6 — Fear & Greed variants |

Facade in `base/technical_indicators.py`; data holder and timing/CSV wrapper in `base/indicator_base.py` (`calculate_indicator(func, *args, required_length=…)`, optional `save_to_csv`, `measure_time`).

## Function contract

- `@njit(cache=True)` on every indicator; input `np.ndarray` (float64 preferred, int auto-converted), output `np.ndarray` pre-filled with `np.full(n, np.nan)`.
- No Python lists, dicts or object calls in hot loops; `math.nan` / `math.isnan` / `math.inf` are allowed. Return scalar, 1D array or tuple of arrays.
- Single pass over the window — no `np.roll` accumulation, O(N) not O(N×L).

## NaN and first-valid index

Uniform: output pre-filled NaN; `len(data) < length` → all NaN; division guarded (`avg_loss == 0` → RSI 100, CCI 0); input NaN is not handled — `DataFetcher` excludes the incomplete candle upstream; consumers slice to the visible candle count.

First valid bar: rolling windows `length - 1`; Wilder-seeded RSI `length`; RMI `length + momentum - 1`; MACD line `slow - 1`, signal/histogram `slow`; TRIX `length + drift`; Hurst `max_lag + 2`. MACD/TRIX warm-up differs from TA-Lib, mature values match.

## Pitfalls (already fixed — do not regress)

- CCI: single-pass sliding sum. `np.roll` per iteration gave O(N×L) plus drift.
- Stochastic: NaN assignment before the first valid period, or the first %K is 0.0.
- MACD histogram: explicit NaN pre-fill; 0.0 as "unset" leaked into output.
- Bollinger %B: skip until a full window is available, or the first bar exceeds [0, 1].

## Isolation

`TechnicalCalculator` builds three independent `TechnicalIndicators` per cycle: 4h/999 candles, 365 daily, 300 weekly. Each holds its own OHLCV arrays — never share an instance across timeframes. Instance arrays exist for both `get_data(ohlcv)` and, later, `calculate_indicator`; `get_data` normalizes to float64.

Edge behaviour: zero volume → `np.nan` with downstream fallback; flat price → RSI 100, ADX 0, volatility 0; single candle → all NaN; >1e10 values saturate with no clamp (irrelevant below 10⁶).
