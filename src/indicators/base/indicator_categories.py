from typing import TypeVar, Tuple

import numpy as np

from src.indicators.base import IndicatorCategory
from src.indicators.indicators.momentum import *
from src.indicators.indicators.overlap import *
from src.indicators.indicators.price import *
from src.indicators.indicators.sentiment import *
from src.indicators.indicators.statistical import *
from src.indicators.indicators.support_resistance import *
from src.indicators.indicators.trend import *
from src.indicators.indicators.volatility import *
from src.indicators.indicators.volume import *

T = TypeVar('T')


class MomentumIndicators(IndicatorCategory['MomentumIndicators']):
    def __init__(self, base, overlap_indicators):
        super().__init__(base)
        self.overlap = overlap_indicators

    def rsi(self, length: int = 14) -> np.ndarray:
        return self._base.calculate_indicator(
            rsi_numba,
            self.close,
            length,
            required_length=length
        )

    def macd(
            self,
            fast_length: int = 12,
            slow_length: int = 26,
            signal_length: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            macd_numba,
            self.close,
            fast_length,
            slow_length,
            signal_length,
            required_length=slow_length
        )

    def stochastic(
            self,
            period_k: int = 5,
            smooth_k: int = 3,
            period_d: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            stochastic_numba,
            self.high,
            self.low,
            self.close,
            period_k,
            smooth_k,
            period_d,
            required_length=3
        )

    def roc(self, length: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            roc_numba,
            self.close,
            length,
            required_length=length + 1
        )

    def momentum(self, length: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            momentum_numba,
            self.close,
            length,
            required_length=length
        )

    def williams_r(self, length: int = 14) -> np.ndarray:
        return self._base.calculate_indicator(
            williams_r_numba,
            self.high,
            self.low,
            self.close,
            length,
            required_length=length
        )

    def tsi(self, long_length: int = 25, short_length: int = 13) -> np.ndarray:
        return self._base.calculate_indicator(
            tsi_numba,
            self.close,
            long_length,
            short_length,
            required_length=long_length + short_length
        )

    def rmi(self, length: int = 14, momentum_length: int = 5) -> np.ndarray:
        return self._base.calculate_indicator(
            rmi_numba,
            self.close,
            length,
            momentum_length,
            required_length=length + momentum_length
        )

    def ppo(self, fast_length: int = 12, slow_length: int = 26) -> np.ndarray:
        return self._base.calculate_indicator(
            ppo_numba,
            self.close,
            fast_length,
            slow_length,
            required_length=max(fast_length, slow_length)
        )

    def coppock_curve(self, wl1: int = 14, wl2: int = 11, wma_length: int = 10) -> np.ndarray:
        return self._base.calculate_indicator(
            coppock_curve_numba,
            self.close,
            wl1,
            wl2,
            wma_length
        )

    def detect_rsi_divergence(self, rsi_values: np.ndarray, length: int=14) -> np.ndarray:
        return self._base.calculate_indicator(
            detect_rsi_divergence,
            self.close,
            rsi_values,
            length,
            required_length=length)

    def relative_strength_index(self, benchmark_close: np.ndarray, window: int = 14) -> np.ndarray:
        return self._base.calculate_indicator(
            calculate_relative_strength_numba,
            self.close,
            benchmark_close,
            window,
            required_length=window
        )

    def kst(self,
            roc1_length: int = 5,
            roc2_length: int = 10,
            roc3_length: int = 15,
            roc4_length: int = 20,
            sma1_length: int = 3,
            sma2_length: int = 5,
            sma3_length: int = 7,
            sma4_length: int = 9
        ) -> np.ndarray:

        roc1 = self.roc(length=roc1_length)
        roc2 = self.roc(length=roc2_length)
        roc3 = self.roc(length=roc3_length)
        roc4 = self.roc(length=roc4_length)

        rcma1 = self.overlap.sma(data_series=roc1, length=sma1_length)
        rcma2 = self.overlap.sma(data_series=roc2, length=sma2_length)
        rcma3 = self.overlap.sma(data_series=roc3, length=sma3_length)
        rcma4 = self.overlap.sma(data_series=roc4, length=sma4_length)

        return rcma1 * 1 + rcma2 * 2 + rcma3 * 3 + rcma4 * 4

    def uo(self, fast=7, medium=14, slow=28, fast_w=4.0, medium_w=2.0, slow_w=1.0, drift=1) -> np.ndarray:
        from ..indicators.momentum.momentum_indicators import UltimateOscillatorConfig
        config = UltimateOscillatorConfig(
            fast=fast,
            medium=medium,
            slow=slow,
            fast_w=fast_w,
            medium_w=medium_w,
            slow_w=slow_w,
            drift=drift
        )
        return self._base.calculate_indicator(
            uo_numba,
            self.high,
            self.low,
            self.close,
            config,
            required_length=slow)

class OverlapIndicators(IndicatorCategory['OverlapIndicators']):
    def ema(self, data_series: np.ndarray, length: int = 10) -> np.ndarray:
        return self._base.calculate_indicator(
            ema_numba,
            data_series,
            length,
            required_length=length
        )

    def sma(self, data_series: np.ndarray, length: int = 10) -> np.ndarray:
        return self._base.calculate_indicator(
            sma_numba,
            data_series,
            length,
            required_length=length
        )

    def ewma(self, span: int = 10) -> np.ndarray:
        return self._base.calculate_indicator(
            ewma_numba,
            self.close,
            span
        )


class PriceTransformIndicators(IndicatorCategory['PriceTransformIndicators']):
    def log_return(self, length: int = 1, cumulative: bool = False) -> np.ndarray:
        return self._base.calculate_indicator(
            log_return_numba,
            self.close,
            length,
            cumulative,
            required_length=length
        )

    def percent_return(self, length: int = 1, cumulative: bool = False) -> np.ndarray:
        return self._base.calculate_indicator(
            percent_return_numba,
            self.close,
            length,
            cumulative,
            required_length=length
        )

    def pdist(self, drift: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            pdist_numba,
            self.open,
            self.high,
            self.low,
            self.close,
            drift,
            required_length=1
        )


class SentimentIndicators(IndicatorCategory['SentimentIndicators']):
    def fear_and_greed_index(
            self,
            rsi_length: int = 14,
            macd_fast_length: int = 12,
            macd_slow_length: int = 26,
            macd_signal_length: int = 9,
            mfi_length: int = 14,
            window_size: int = 60
    ) -> np.ndarray:
        from ..indicators.sentiment.sentiment_indicators import FearGreedConfig
        config = FearGreedConfig(
            rsi_length=rsi_length,
            macd_fast_length=macd_fast_length,
            macd_slow_length=macd_slow_length,
            macd_signal_length=macd_signal_length,
            mfi_length=mfi_length,
            window_size=window_size
        )
        return self._base.calculate_indicator(
            fear_and_greed_index_numba,
            self.close,
            self.high,
            self.low,
            self.volume,
            config,
            required_length=max(rsi_length, macd_slow_length + macd_signal_length - 1, mfi_length)
        )


class StatisticalIndicators(IndicatorCategory['StatisticalIndicators']):
    def kurtosis(self, length: int = 30) -> np.ndarray:
        return self._base.calculate_indicator(
            kurtosis_numba,
            self.close,
            length,
            required_length=length
        )

    def skew(self, length: int = 30) -> np.ndarray:
        return self._base.calculate_indicator(
            skew_numba,
            self.close,
            length,
            required_length=length
        )

    def stdev(self, length: int = 30, ddof: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            stdev_numba,
            self.close,
            length,
            ddof,
            required_length=length
        )

    def variance(self, length: int = 30, ddof: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            variance_numba,
            self.close,
            length,
            ddof,
            required_length=length
        )

    def zscore(self, length: int = 30, std: float = 1.0) -> np.ndarray:
        return self._base.calculate_indicator(
            zscore_numba,
            self.close,
            length,
            std,
            required_length=length
        )

    def mad(self, length: int = 30) -> np.ndarray:
        return self._base.calculate_indicator(
            mad_numba,
            self.close,
            length,
            required_length=length
        )

    def quantile(self, length: int = 30, q: float = 0.5) -> np.ndarray:
        return self._base.calculate_indicator(
            quantile_numba,
            self.close,
            length,
            q,
            required_length=length
        )

    def entropy(self, length: int = 10, base: float = 2.0) -> np.ndarray:
        return self._base.calculate_indicator(
            entropy_numba,
            self.close,
            length,
            base,
            required_length=length
        )

    def hurst(self, max_lag: int = 20) -> np.ndarray:
        return self._base.calculate_indicator(
            hurst_numba,
            self.close,
            max_lag,
            required_length=max_lag + 2
        )

    def linreg(self, length: int = 14, r: bool = False) -> np.ndarray:
        return self._base.calculate_indicator(
            linreg_numba,
            self.close,
            length,
            r,
            required_length=length
        )

    def apa_adaptive_eot(self, q1: float = 0.8, q2: float = 0.4, min_len: int = 10, max_len: int = 48,
                         ave_len: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            apa_adaptive_eot_numba,
            self.close,
            q1,
            q2,
            min_len,
            max_len,
            ave_len,
            required_length=max_len
        )

    def calculate_eot(self, length: int = 21, q1: float = 0.8, q2: float = 0.4) -> np.ndarray:
        return self._base.calculate_indicator(
            calculate_eot_numba,
            self.close,
            length,
            q1,
            q2,
            required_length=length
        )


class SupportResistanceIndicators(IndicatorCategory['SupportResistanceIndicators']):
    def support_resistance(self, length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            support_resistance_numba,
            self.high,
            self.low,
            length,
            required_length=length
        )

    def find_support_resistance(self, window: int = 30) -> Tuple[float, float]:
        support, resistance = self.support_resistance_advanced(length=window)
        return self._base.calculate_indicator(
            find_support_resistance_numba,
            self.close,
            support,
            resistance,
            window,
            required_length=window
        )

    def support_resistance_advanced(self, length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            support_resistance_numba_advanced,
            self.high,
            self.low,
            self.close,
            self.volume,
            length,
            required_length=length
        )

    def advanced_support_resistance(
            self,
            length: int = 25,
            strength_threshold: int = 1,
            persistence: int = 1,
            volume_factor: float = 1.3,
            price_factor: float = 0.004
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            advanced_support_resistance_numba,
            self.high,
            self.low,
            self.close,
            self.volume,
            length,
            strength_threshold,
            persistence,
            volume_factor,
            price_factor,
            required_length=length
        )

    def fibonacci_retracement(self, length: int = 20) -> np.ndarray:
        return self._base.calculate_indicator(
            fibonacci_retracement_numba,
            length,
            self.high,
            self.low,
            required_length=length
        )

    def fibonacci_bollinger_bands(self, length: int = 20, mult: float = 3.0) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        hlc3 = (self.high + self.low + self.close) / 3
        return self._base.calculate_indicator(
            fibonacci_bollinger_bands_numba,
            hlc3,
            self.volume,
            length,
            mult,
            required_length=length
        )

    def floating_levels(
            self,
            lookback: int = 20,
            level_up: float = 50.0,
            level_down: float = 50.0,
            length: int = 7,
            multiplier: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            floating_levels_numba,
            self.high,
            self.low,
            self.close,
            length,
            multiplier,
            lookback,
            level_up,
            level_down,
            required_length=lookback
        )

    def pivot_points(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate standard pivot points and support/resistance levels
        
        Returns:
            Tuple of (pivot_point, r1, r2, r3, r4, s1, s2, s3, s4) arrays
        """
        return self._base.calculate_indicator(
            pivot_points_numba,
            self.high,
            self.low,
            self.close,
            required_length=1
        )


class TrendIndicators(IndicatorCategory['TrendIndicators']):
    def adx(self, length: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            adx_numba,
            self.high,
            self.low,
            self.close,
            length,
            required_length=length
        )

    def supertrend(self, length: int = 7, multiplier: float = 3.0) -> Tuple[
        np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            supertrend_numba,
            self.high,
            self.low,
            self.close,
            length,
            multiplier
        )

    def ichimoku_cloud(
            self,
            conversion_length: int = 9,
            base_length: int = 26,
            lagging_span2_length: int = 52,
            displacement: int = 26
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            ichimoku_cloud_numba,
            self.high,
            self.low,
            conversion_length,
            base_length,
            lagging_span2_length,
            displacement,
            required_length=max(base_length, lagging_span2_length)
        )

    def parabolic_sar(self, step: float = 0.02, max_step: float = 0.2) -> np.ndarray:
        return self._base.calculate_indicator(
            parabolic_sar_numba,
            self.high,
            self.low,
            step,
            max_step
        )

    def vortex_indicator(self, length: int = 14) -> Tuple[np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            vortex_indicator_numba,
            self.high,
            self.low,
            self.close,
            length,
            required_length=length + 1
        )

    def trix(self, length: int = 18, scalar: float = 100, drift: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            trix_numba,
            self.close,
            length,
            scalar,
            drift,
            required_length=length
        )

    def pfe(self, n: int = 10, m: int = 10) -> np.ndarray:
        return self._base.calculate_indicator(
            pfe_numba,
            self.close,
            n,
            m
        )

    def td_sequential(self, length: int = 9) -> np.ndarray:
        return self._base.calculate_indicator(
            td_sequential_numba,
            self.close,
            length,
            required_length=5
        )


class VolatilityIndicators(IndicatorCategory['VolatilityIndicators']):
    def atr(self, length: int = 14, mamode: str = 'rma', percent: bool = False) -> np.ndarray:
        return self._base.calculate_indicator(
            atr_numba,
            self.high,
            self.low,
            self.close,
            length,
            mamode,
            percent,
            required_length=length
        )

    def bollinger_bands(self, length: int = 20, num_std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            bollinger_bands_numba,
            self.close,
            length,
            num_std_dev,
            required_length=length
        )

    def chandelier_exit(self, length: int = 22, multiplier: float = 3.0, mamode: str = 'rma') -> Tuple[
        np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            chandelier_exit_numba,
            self.high,
            self.low,
            self.close,
            length,
            multiplier,
            mamode
        )

    def vhf(self, length: int = 28) -> np.ndarray:
        return self._base.calculate_indicator(
            vhf_numba,
            self.close,
            length,
            required_length=length
        )

    def ebsw(self, length: int = 40, bars: int = 10) -> np.ndarray:
        return self._base.calculate_indicator(
            ebsw_numba,
            self.close,
            length,
            bars,
            required_length=length
        )

    def keltner_channels(self, length: int = 20, multiplier: float = 2.0, mamode: str = 'ema') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            keltner_channels_numba,
            self.high,
            self.low,
            self.close,
            length,
            multiplier,
            mamode,
            required_length=length
        )

    def donchian_channels(self, length: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._base.calculate_indicator(
            donchian_channels_numba,
            self.high,
            self.low,
            length,
            required_length=length
        )


class VolumeIndicators(IndicatorCategory['VolumeIndicators']):
    def cci(self, length: int = 14, c: float = 0.015) -> np.ndarray:
        return self._base.calculate_indicator(
            cci_numba,
            self.high,
            self.low,
            self.close,
            length,
            c,
            required_length=length)

    def mfi(self, length: int = 14, drift: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            mfi_numba,
            self.high,
            self.low,
            self.close,
            self.volume,
            length,
            drift,
            required_length=length
        )

    def obv(self, length: int = 14, initial: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            obv_numba,
            self.close,
            self.volume,
            length,
            initial,
            required_length=length
        )

    def pvt(self, length: int = 14, drift: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            pvt_numba,
            self.close,
            self.volume,
            length,
            drift,
            required_length=length
        )

    def chaikin_money_flow(self, length: int = 20) -> np.ndarray:
        return self._base.calculate_indicator(
            chaikin_money_flow_numba,
            self.high,
            self.low,
            self.close,
            self.volume,
            length,
            required_length=length
        )

    def accumulation_distribution_line(self) -> np.ndarray:
        return self._base.calculate_indicator(
            ad_line_numba,
            self.high,
            self.low,
            self.close,
            self.volume
        )

    def force_index(self, length: int = 13) -> np.ndarray:
        return self._base.calculate_indicator(
            force_index_numba,
            self.close,
            self.volume,
            length,
            required_length=length + 1
        )

    def eom(self, length: int = 14, divisor: int = 100000000, drift: int = 1) -> np.ndarray:
        return self._base.calculate_indicator(
            eom_numba,
            self.high,
            self.low,
            self.volume,
            length,
            divisor,
            drift,
            required_length=length
        )

    def volume_profile(self, length: int = 48, num_bins: int = 10) -> np.ndarray:
        return self._base.calculate_indicator(
            volume_profile_numba,
            self.close,
            self.volume,
            length,
            num_bins,
            required_length=length
        )

    def rolling_vwap(self, length: int = 14) -> np.ndarray:
        return self._base.calculate_indicator(
            rolling_vwap_numba,
            self.high,
            self.low,
            self.close,
            self.volume,
            length,
            required_length=length
        )

    def twap(self, length: int = 14) -> np.ndarray:
        return self._base.calculate_indicator(
            twap_numba,
            self.high,
            self.low,
            self.close,
            length,
            required_length=length
        )

    def average_quote_volume(self, window_size=14):
        return self._base.calculate_indicator(
            average_quote_volume_numba,
            self.close,
            self.volume, window_size,
            required_length=window_size
        )