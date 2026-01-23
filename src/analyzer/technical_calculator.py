from typing import Dict, Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.factories import TechnicalIndicatorsFactory
from src.indicators.base.technical_indicators import TechnicalIndicators
from src.logger.logger import Logger
from src.utils.profiler import profile_performance
from src.utils.data_utils import get_last_valid_value
from src.analyzer.pattern_engine.indicator_patterns.ma_crossover_patterns import (
    detect_golden_cross_numba, detect_death_cross_numba
)


class TechnicalCalculator:
    """Core calculator for technical indicators"""
    
    def __init__(self, logger: Optional[Logger] = None, format_utils=None, ti_factory: "TechnicalIndicatorsFactory" = None):
        """Initialize the technical indicator calculator"""
        if ti_factory is None:
            raise ValueError("ti_factory is required - must be injected from app.py")
        self.logger = logger
        self.format_utils = format_utils
        self.ti_factory = ti_factory
        
    @profile_performance
    def get_indicators(self, ohlcv_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate all technical indicators - no caching, always fresh"""
        self.ti = self.ti_factory.create_for_current_timeframe(ohlcv_data)
        
        indicators = {}
        
        # Calculate indicators by category
        indicators.update(self._calculate_volume_indicators())
        indicators.update(self._calculate_momentum_indicators())
        indicators.update(self._calculate_volatility_indicators(ohlcv_data))
        indicators.update(self._calculate_trend_indicators())
        indicators.update(self._calculate_support_resistance_indicators())
        
        if self.logger:
            pass
        
        return indicators

    def _calculate_volume_indicators(self) -> Dict[str, np.ndarray]:
        """Calculate volume-based indicators"""
        return {
            "vwap": self.ti.rolling_vwap(length=20),
            "twap": self.ti.twap(length=20),
            "mfi": self.ti.mfi(length=14),
            "obv": self.ti.obv(length=20),
            "cmf": self.ti.chaikin_money_flow(length=20),
            "force_index": self.ti.force_index(length=20),
            "cci": self.ti.cci(length=14),
            "pvt": self.ti.pvt(length=14),
            "ad_line": self.ti.accumulation_distribution_line(),
        }

    def _calculate_momentum_indicators(self) -> Dict[str, np.ndarray]:
        """Calculate momentum indicators"""
        stoch_k, stoch_d = self.ti.stochastic(period_k=14, smooth_k=3, period_d=3)

        indicators = {
            "rsi": self.ti.rsi(length=14),
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "williams_r": self.ti.williams_r(length=14),
            "uo": self.ti.uo(),
            "tsi": self.ti.tsi(long_length=20, short_length=10),
            "rmi": self.ti.rmi(length=20, momentum_length=5),
            "ppo": self.ti.ppo(fast_length=12, slow_length=26),
            "coppock": self.ti.coppock_curve(wl1=11, wl2=14, wma_length=10),
            "kst": self.ti.kst(),
            "roc_14": self.ti.roc(length=14),
        }
        
        macd_line, macd_signal, macd_hist = self.ti.macd(fast_length=12, slow_length=26, signal_length=9)
        indicators["macd_line"] = macd_line
        indicators["macd_signal"] = macd_signal
        indicators["macd_hist"] = macd_hist
        
        return indicators

    def _calculate_volatility_indicators(self, ohlcv_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate volatility indicators"""
        indicators = {
            "atr": self.ti.atr(length=20),
            "atr_percent": self.ti.atr(length=20, percent=True),
        }
        
        bb_upper, bb_middle, bb_lower = self.ti.bollinger_bands(length=20, num_std_dev=2)
        indicators["bb_upper"] = bb_upper
        indicators["bb_middle"] = bb_middle
        indicators["bb_lower"] = bb_lower
        
        close_prices = ohlcv_data[:, 4]
        band_width = bb_upper - bb_lower
        indicators["bb_percent_b"] = np.where(
            band_width != 0,
            (close_prices - bb_lower) / band_width,
            np.nan
        )
        
        # Keltner Channels
        kc_upper, kc_middle, kc_lower = self.ti.keltner_channels(length=20, multiplier=2)
        indicators["kc_upper"] = kc_upper
        indicators["kc_middle"] = kc_middle
        indicators["kc_lower"] = kc_lower
        
        # Donchian Channels (only retain extremes for formatting/patterns)
        donchian_upper, _, donchian_lower = self.ti.donchian_channels(length=20)
        indicators["donchian_upper"] = donchian_upper
        indicators["donchian_lower"] = donchian_lower
        
        # Chandelier Exit
        long_exit, short_exit = self.ti.chandelier_exit(length=20, multiplier=3.0)
        indicators["chandelier_long"] = long_exit
        indicators["chandelier_short"] = short_exit
        
        # Choppiness Index
        indicators["choppiness"] = self.ti.choppiness_index(length=14)
        
        return indicators

    def _calculate_trend_indicators(self) -> Dict[str, np.ndarray]:
        """Calculate trend indicators"""
        # Calculate ADX once
        adx, plus_di, minus_di = self.ti.adx(length=14)

        indicators = {
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "trix": self.ti.trix(length=20),
            "pfe": self.ti.pfe(n=20, m=5),
            "td_sequential": self.ti.td_sequential(length=9),
            "sar": self.ti.parabolic_sar(step=0.02, max_step=0.2),
        }
        
        supertrend, supertrend_direction = self.ti.supertrend(length=20, multiplier=3.0)
        indicators["supertrend"] = supertrend
        indicators["supertrend_direction"] = supertrend_direction

        # Ichimoku Cloud
        conversion, base, span_a, span_b = self.ti.ichimoku_cloud(
            conversion_length=9,
            base_length=26,
            lagging_span2_length=52,
            displacement=26
        )
        indicators["ichimoku_span_a"] = span_a
        indicators["ichimoku_span_b"] = span_b
        
        # Vortex
        vortex_plus, vortex_minus = self.ti.vortex_indicator(length=20)
        indicators["vortex_plus"] = vortex_plus
        indicators["vortex_minus"] = vortex_minus
        
        # SMAs
        indicators["sma_20"] = self.ti.sma(self.ti.close, 20)
        indicators["sma_50"] = self.ti.sma(self.ti.close, 50)
        indicators["sma_200"] = self.ti.sma(self.ti.close, 200)
        
        return indicators

    def _calculate_support_resistance_indicators(self) -> Dict[str, np.ndarray]:
        """Calculate support and resistance indicators"""
        indicators = {
            "kurtosis": self.ti.kurtosis(length=20),
            "zscore": self.ti.zscore(length=20),
            "hurst": self.ti.hurst(max_lag=20),
            "entropy": self.ti.entropy(length=20),
            "skewness": self.ti.skew(length=20),
            "variance": self.ti.variance(length=20),
            "linreg_slope": self.ti.linreg(length=14, r=False),
            "linreg_r2": self.ti.linreg(length=14, r=True) ** 2,
        }
        
        # Basic Support/Resistance
        support, resistance = self.ti.support_resistance(length=20)
        indicators["basic_support"] = support
        indicators["basic_resistance"] = resistance
        
        # Advanced Support/Resistance
        adv_support, adv_resistance = self.ti.advanced_support_resistance(
            length=20,
            strength_threshold=1,
            persistence=1,
            volume_factor=1.5,
            price_factor=0.004
        )
        indicators["advanced_support"] = adv_support
        indicators["advanced_resistance"] = adv_resistance
        
        # Pivot Points
        pivot_point, r1, r2, r3, r4, s1, s2, s3, s4 = self.ti.pivot_points()
        indicators["pivot_point"] = pivot_point
        indicators["pivot_r1"] = r1
        indicators["pivot_r2"] = r2
        indicators["pivot_r3"] = r3
        indicators["pivot_r4"] = r4
        indicators["pivot_s1"] = s1
        indicators["pivot_s2"] = s2
        indicators["pivot_s3"] = s3
        indicators["pivot_s4"] = s4
        
        # Fibonacci Pivot Points
        fib_pivot, fib_r1, fib_r2, fib_r3, fib_s1, fib_s2, fib_s3 = self.ti.fibonacci_pivot_points()
        indicators["fib_pivot_point"] = fib_pivot
        indicators["fib_pivot_r1"] = fib_r1
        indicators["fib_pivot_r2"] = fib_r2
        indicators["fib_pivot_r3"] = fib_r3
        indicators["fib_pivot_s1"] = fib_s1
        indicators["fib_pivot_s2"] = fib_s2
        indicators["fib_pivot_s3"] = fib_s3
        
        return indicators
        
    def get_long_term_indicators(self, ohlcv_data: np.ndarray) -> Dict[str, Any]:
        """Calculate long-term indicators for historical data - no caching, always fresh"""
        # Create new TI instance for long-term calculations (avoid interference with regular timeframe indicators)
        ti_lt = self.ti_factory.create_for_long_term(ohlcv_data)
        available_days = len(ohlcv_data)

        sma_values, volume_sma_values = self._compute_sma_sets(ti_lt, available_days)
        price_change_pct, volume_change_pct = self._compute_change_metrics(ti_lt, available_days)
        volatility = self._compute_volatility(ti_lt, available_days)

        daily_indicators = self._compute_daily_indicators(ti_lt, available_days)
        
        # Add macro trend analysis based on SMA relationships (pass already-calculated price_change_pct)
        macro_trend_analysis = self._compute_macro_trend_analysis(ti_lt, available_days, sma_values, price_change_pct)

        result = {
            'sma_values': sma_values,
            'volume_sma_values': volume_sma_values,
            'price_change': price_change_pct,
            'volume_change': volume_change_pct,
            'volatility': volatility,
            'available_days': available_days,
            'macro_trend': macro_trend_analysis,
            **daily_indicators
        }

        # Ensure we're not returning numpy types that might not be recognized properly
        result = {k: float(v) if isinstance(v, (np.floating, float)) and not np.isnan(v) else v
                  for k, v in result.items() if k not in ('sma_values', 'volume_sma_values')}

        return result

    def get_weekly_macro_indicators(self, weekly_ohlcv_data: np.ndarray) -> Dict[str, Any]:
        """Calculate macro indicators using weekly data (200W SMA methodology) - no caching, always fresh"""
        ti_weekly = self.ti_factory.create_for_weekly(weekly_ohlcv_data)
        available_weeks = len(weekly_ohlcv_data)

        # REUSE existing helper methods (already timeframe-agnostic)
        weekly_sma_values, weekly_volume_sma_values = self._compute_sma_sets(ti_weekly, available_weeks)
        weekly_price_change, weekly_volume_change = self._compute_change_metrics(ti_weekly, available_weeks)
        weekly_volatility = self._compute_volatility(ti_weekly, available_weeks)

        # Pre-calculate SMA arrays for crossover detection (avoid redundant calculation in macro analysis)
        sma_arrays = {}
        if available_weeks >= 50:
            sma_arrays['sma_50'] = ti_weekly.sma(ti_weekly.close, 50)
        if available_weeks >= 200:
            sma_arrays['sma_200'] = ti_weekly.sma(ti_weekly.close, 200)

        # NEW: Weekly-specific macro analysis (pass SMA arrays to avoid recalculation)
        weekly_macro_analysis = self._compute_weekly_macro_trend_analysis(
            ti_weekly, available_weeks, weekly_sma_values, weekly_ohlcv_data, weekly_price_change, sma_arrays
        )

        result = {
            'weekly_sma_values': weekly_sma_values,
            'weekly_volume_sma_values': weekly_volume_sma_values,
            'weekly_price_change': weekly_price_change,
            'weekly_volume_change': weekly_volume_change,
            'weekly_volatility': weekly_volatility,
            'available_weeks': available_weeks,
            'weekly_macro_trend': weekly_macro_analysis
        }

        result = {k: float(v) if isinstance(v, (np.floating, float)) and not np.isnan(v) else v
                  for k, v in result.items() if k not in ('weekly_sma_values', 'weekly_volume_sma_values')}

        return result

    def _compute_weekly_macro_trend_analysis(
        self, ti: TechnicalIndicators, available_weeks: int, 
        weekly_sma_values: Dict[int, float], ohlcv_data: np.ndarray, price_change_pct: float,
        sma_arrays: Dict[str, np.ndarray] = None
    ) -> Dict[str, Any]:
        """Weekly macro trend using 200W SMA methodology with timestamps.
        
        Args:
            ti: TechnicalIndicators instance with weekly data
            available_weeks: Number of weeks available
            weekly_sma_values: Dict of current SMA values
            ohlcv_data: Weekly OHLCV array for timestamp extraction
            price_change_pct: Already-calculated price change percentage from _compute_change_metrics
            sma_arrays: Pre-calculated SMA arrays to avoid redundant computation
        """
        formatter = self.format_utils
        analysis = {
            'trend_direction': 'Neutral',
            'weekly_sma_alignment': 'Mixed',
            'golden_cross': False,
            'death_cross': False,
            'price_above_200w_sma': False,
            'sma_50w_vs_200w': 'Neutral',
            'distance_from_200w_sma_pct': None,
            'cycle_phase': None,
            'confidence_score': 0,
            'multi_year_trend': None
        }
        
        # Skip if insufficient data
        if available_weeks < 50:
            if self.logger:
                self.logger.debug(f"Insufficient weekly data: {available_weeks} weeks")
            return analysis
        
        current_price = float(ti.close[-1])
        
        # Multi-year trend with timestamps (use already-calculated price_change_pct)
        if available_weeks >= 2:
            years = available_weeks / 52.0
            start_ts = ohlcv_data[0, 0] / 1000
            end_ts = ohlcv_data[-1, 0] / 1000
            
            analysis['multi_year_trend'] = {
                'weeks_analyzed': available_weeks,
                'years_analyzed': round(years, 1),
                'price_change_pct': price_change_pct,
                'start_date': formatter.format_date_from_timestamp(start_ts),
                'end_date': formatter.format_date_from_timestamp(end_ts)
            }
        
        # 200W SMA analysis (the gold standard)
        if 200 in weekly_sma_values:
            sma_200w = weekly_sma_values[200]
            analysis['price_above_200w_sma'] = current_price > sma_200w
            distance = ((current_price - sma_200w) / sma_200w) * 100
            analysis['distance_from_200w_sma_pct'] = float(distance)
        
        # Golden/Death Cross with timestamps
        if 50 in weekly_sma_values and 200 in weekly_sma_values:
            sma_50w_array = sma_arrays['sma_50']
            sma_200w_array = sma_arrays['sma_200']
            

            
            golden_found, golden_weeks_ago, _, _ = detect_golden_cross_numba(sma_50w_array, sma_200w_array)
            if golden_found:
                analysis['golden_cross'] = True
                analysis['golden_cross_weeks_ago'] = golden_weeks_ago
                cross_ts = ohlcv_data[-(golden_weeks_ago + 1), 0] / 1000
                analysis['golden_cross_date'] = formatter.format_date_from_timestamp(cross_ts)
                if self.logger:
                    self.logger.info(f"🌟 Weekly Golden Cross: {golden_weeks_ago}w ago ({analysis['golden_cross_date']})")
            
            death_found, death_weeks_ago, _, _ = detect_death_cross_numba(sma_50w_array, sma_200w_array)
            if death_found:
                analysis['death_cross'] = True
                analysis['death_cross_weeks_ago'] = death_weeks_ago
                cross_ts = ohlcv_data[-(death_weeks_ago + 1), 0] / 1000
                analysis['death_cross_date'] = formatter.format_date_from_timestamp(cross_ts)
                if self.logger:
                    self.logger.warning(f"⚠️ Weekly Death Cross: {death_weeks_ago}w ago ({analysis['death_cross_date']})")
            
            # SMA relationship
            if weekly_sma_values[50] > weekly_sma_values[200]:
                analysis['sma_50w_vs_200w'] = 'Bullish'
            elif weekly_sma_values[50] < weekly_sma_values[200]:
                analysis['sma_50w_vs_200w'] = 'Bearish'
        
        # SMA alignment check
        if all(p in weekly_sma_values for p in [20, 50, 100, 200]):
            smas = [weekly_sma_values[p] for p in [20, 50, 100, 200]]
            if all(smas[i] >= smas[i+1] for i in range(len(smas)-1)):
                analysis['weekly_sma_alignment'] = 'Bullish (Ascending)'
            elif all(smas[i] <= smas[i+1] for i in range(len(smas)-1)):
                analysis['weekly_sma_alignment'] = 'Bearish (Descending)'
        
        # Trend direction with confidence
        bullish = sum([
            analysis['price_above_200w_sma'], 
            analysis['sma_50w_vs_200w'] == 'Bullish',
            analysis['golden_cross'], 
            analysis['weekly_sma_alignment'] == 'Bullish (Ascending)',
            analysis.get('distance_from_200w_sma_pct', 0) > 20
        ])
        bearish = sum([
            not analysis['price_above_200w_sma'], 
            analysis['sma_50w_vs_200w'] == 'Bearish',
            analysis['death_cross'], 
            analysis['weekly_sma_alignment'] == 'Bearish (Descending)',
            analysis.get('distance_from_200w_sma_pct', 0) < -20
        ])
        
        total = bullish + bearish
        if bullish >= 3 and bullish > bearish:
            analysis['trend_direction'] = 'Bullish'
            analysis['confidence_score'] = int((bullish / max(total, 1)) * 100)
        elif bearish >= 3 and bearish > bullish:
            analysis['trend_direction'] = 'Bearish'
            analysis['confidence_score'] = int((bearish / max(total, 1)) * 100)
        else:
            analysis['confidence_score'] = 50
        
        return analysis

    

    def _compute_sma_sets(self, ti: TechnicalIndicators, available_days: int):
        sma_periods = [20, 50, 100, 200]
        sma_values: Dict[int, float] = {}
        volume_sma_values: Dict[int, float] = {}
        for period in sma_periods:
            if available_days >= period:
                # Use technical indicators directly instead of extracted arrays
                sma = ti.sma(ti.close, period)
                vol_sma = ti.sma(ti.volume, period)
                if not np.isnan(sma[-1]):
                    sma_values[period] = float(sma[-1])
                if not np.isnan(vol_sma[-1]):
                    volume_sma_values[period] = float(vol_sma[-1])
        return sma_values, volume_sma_values

    def _compute_change_metrics(self, ti: TechnicalIndicators, available_days: int):
        price_change_pct = volume_change_pct = None
        if available_days >= 2:
            # Use technical indicators data directly
            price_change_pct = float((ti.close[-1] / ti.close[0] - 1) * 100)
            volume_change_pct = float((ti.volume[-1] / max(ti.volume[0], 1) - 1) * 100)
            
            if self.logger:
                pass
                # self.logger.debug(f"Long-term price change: {price_change_pct:.2f}% over {available_days} days")
                # self.logger.debug(f"Long-term volume change: {volume_change_pct:.2f}%")
        else:
            if self.logger:
                self.logger.warning(f"Not enough data to calculate change metrics (only {available_days} days)")
        
        return price_change_pct, volume_change_pct

    def _compute_volatility(self, ti: TechnicalIndicators, available_days: int):
        if available_days >= 7:
            # Use technical indicators data directly
            daily_returns = np.diff(ti.close) / ti.close[:-1]
            return float(np.std(daily_returns) * 100)
        return None
    
    def _compute_macro_trend_analysis(self, ti: TechnicalIndicators, available_days: int, sma_values: Dict[int, float], price_change_pct: float) -> Dict[str, Any]:
        """Analyze macro trend using SMA relationships and 365-day context.
        
        Args:
            price_change_pct: Already-calculated price change percentage from _compute_change_metrics
        """
        analysis = {
            'trend_direction': 'Neutral',
            'sma_alignment': 'Mixed',
            'golden_cross': False,
            'death_cross': False,
            'price_above_200sma': False,
            'sma_50_vs_200': 'Neutral',
            'long_term_price_change_pct': None
        }
        
        if available_days < 200:
            if self.logger:
                self.logger.debug(f"Not enough data for macro trend analysis (need 200 days, have {available_days})")
            return analysis
        
        current_price = float(ti.close[-1])
        
        # Use already-calculated price change percentage (no redundant calculation)
        analysis['long_term_price_change_pct'] = price_change_pct
        if self.logger:
            pass # self.logger.debug(f"Macro trend: {available_days}-day price change = {price_change_pct:.2f}%")
        
        # Check price position relative to key SMAs
        if 200 in sma_values:
            analysis['price_above_200sma'] = current_price > sma_values[200]
            
        # Analyze SMA 50 vs SMA 200 relationship (Golden/Death Cross context)
        if 50 in sma_values and 200 in sma_values:
            sma_50 = sma_values[50]
            sma_200 = sma_values[200]
            
            if sma_50 > sma_200:
                analysis['sma_50_vs_200'] = 'Bullish'
                # Check if this could be a golden cross scenario
                if available_days >= 250:  # Need enough data to check trend
                    sma_50_prev = ti.sma(ti.close, 50)[-10]  # 10 periods ago
                    sma_200_prev = ti.sma(ti.close, 200)[-10]
                    if sma_50_prev <= sma_200_prev and sma_50 > sma_200:
                        analysis['golden_cross'] = True
            elif sma_50 < sma_200:
                analysis['sma_50_vs_200'] = 'Bearish'
                # Check if this could be a death cross scenario
                if available_days >= 250:
                    sma_50_prev = ti.sma(ti.close, 50)[-10]
                    sma_200_prev = ti.sma(ti.close, 200)[-10]
                    if sma_50_prev >= sma_200_prev and sma_50 < sma_200:
                        analysis['death_cross'] = True
        
        # Determine overall SMA alignment
        if 20 in sma_values and 50 in sma_values and 100 in sma_values and 200 in sma_values:
            smas = [sma_values[20], sma_values[50], sma_values[100], sma_values[200]]
            if all(smas[i] >= smas[i+1] for i in range(len(smas)-1)):
                analysis['sma_alignment'] = 'Bullish (Ascending)'
            elif all(smas[i] <= smas[i+1] for i in range(len(smas)-1)):
                analysis['sma_alignment'] = 'Bearish (Descending)'
            else:
                analysis['sma_alignment'] = 'Mixed'
        
        # Determine overall trend direction with improved logic
        # Count bullish signals
        bullish_signals = []
        if analysis['price_above_200sma']:
            bullish_signals.append('price_above_200sma')
        if analysis['sma_50_vs_200'] == 'Bullish':
            bullish_signals.append('sma_50_vs_200')
        if analysis['golden_cross']:
            bullish_signals.append('golden_cross')
        if analysis['sma_alignment'] == 'Bullish (Ascending)':
            bullish_signals.append('sma_alignment')
        # Add price change as a signal if significant
        if analysis['long_term_price_change_pct'] is not None:
            if analysis['long_term_price_change_pct'] > 20:  # More than 20% gain
                bullish_signals.append('strong_price_appreciation')
            elif analysis['long_term_price_change_pct'] > 10:  # More than 10% gain
                bullish_signals.append('moderate_price_appreciation')
        
        # Count bearish signals
        bearish_signals = []
        if not analysis['price_above_200sma']:
            bearish_signals.append('price_below_200sma')
        if analysis['sma_50_vs_200'] == 'Bearish':
            bearish_signals.append('sma_50_vs_200')
        if analysis['death_cross']:
            bearish_signals.append('death_cross')
        if analysis['sma_alignment'] == 'Bearish (Descending)':
            bearish_signals.append('sma_alignment')
        # Add price change as a signal if significantly negative
        if analysis['long_term_price_change_pct'] is not None:
            if analysis['long_term_price_change_pct'] < -20:  # More than 20% loss
                bearish_signals.append('strong_price_decline')
            elif analysis['long_term_price_change_pct'] < -10:  # More than 10% loss
                bearish_signals.append('moderate_price_decline')
        
        bullish_count = len(bullish_signals)
        bearish_count = len(bearish_signals)
        
        if self.logger:
            pass
            # self.logger.debug(f"Macro trend bullish signals ({bullish_count}): {bullish_signals}")
            # self.logger.debug(f"Macro trend bearish signals ({bearish_count}): {bearish_signals}")
        
        # Determine trend direction (need at least 2 signals for clear direction)
        if bullish_count >= 2 and bullish_count > bearish_count:
            analysis['trend_direction'] = 'Bullish'
        elif bearish_count >= 2 and bearish_count > bullish_count:
            analysis['trend_direction'] = 'Bearish'
        else:
            analysis['trend_direction'] = 'Neutral'
            
        return analysis

    def _compute_daily_indicators(self, ti: TechnicalIndicators, available_days: int) -> Dict[str, Any]:
        """Compute daily indicators based on available data."""
        out: Dict[str, Any] = self._initialize_daily_indicators()
        
        if available_days >= 14:
            self._compute_14_day_indicators(ti, out)
        
        if available_days >= 26:
            self._compute_26_day_indicators(ti, out)
            
        if available_days >= 52:
            self._compute_52_day_indicators(ti, out)
            
        return out
    
    def _initialize_daily_indicators(self) -> Dict[str, Any]:
        """Initialize dictionary with daily indicator keys."""
        return {
            'daily_rsi': None,
            'daily_macd_line': None,
            'daily_macd_signal': None,
            'daily_macd_hist': None,
            'daily_atr': None,
            'daily_adx': None,
            'daily_plus_di': None,
            'daily_minus_di': None,
            'daily_obv': None,
            'daily_ichimoku_conversion': None,
            'daily_ichimoku_base': None,
            'daily_ichimoku_span_a': None,
            'daily_ichimoku_span_b': None
        }
    
    def _compute_14_day_indicators(self, ti: TechnicalIndicators, out: Dict[str, Any]) -> None:
        """Compute indicators that require 14 days of data."""
        # RSI
        rsi_vals = ti.rsi(length=14)
        if rsi_vals is not None and not np.isnan(rsi_vals[-1]):
            out['daily_rsi'] = float(rsi_vals[-1])
        
        # ATR
        atr_vals = ti.atr(length=14)
        if atr_vals is not None and not np.isnan(atr_vals[-1]):
            out['daily_atr'] = float(atr_vals[-1])
        
        # ADX and DI
        adx_vals, plus_di_vals, minus_di_vals = ti.adx(length=14)
        if adx_vals is not None and not np.isnan(adx_vals[-1]):
            out['daily_adx'] = float(adx_vals[-1])
        if plus_di_vals is not None and not np.isnan(plus_di_vals[-1]):
            out['daily_plus_di'] = float(plus_di_vals[-1])
        if minus_di_vals is not None and not np.isnan(minus_di_vals[-1]):
            out['daily_minus_di'] = float(minus_di_vals[-1])
        
        # OBV
        obv_vals = ti.obv()
        if obv_vals is not None and not np.isnan(obv_vals[-1]):
            out['daily_obv'] = float(obv_vals[-1])
    
    def _compute_26_day_indicators(self, ti: TechnicalIndicators, out: Dict[str, Any]) -> None:
        """Compute indicators that require 26 days of data."""
        macd_line, macd_signal, macd_hist = ti.macd()
        
        if macd_line is not None and not np.isnan(macd_line[-1]):
            out['daily_macd_line'] = float(macd_line[-1])
        if macd_signal is not None and not np.isnan(macd_signal[-1]):
            out['daily_macd_signal'] = float(macd_signal[-1])
        if macd_hist is not None and not np.isnan(macd_hist[-1]):
            out['daily_macd_hist'] = float(macd_hist[-1])
    
    def _compute_52_day_indicators(self, ti: TechnicalIndicators, out: Dict[str, Any]) -> None:
        """Compute indicators that require 52 days of data."""
        conversion, base, span_a, span_b = ti.ichimoku_cloud()
        
        if conversion is not None and not np.isnan(conversion[-1]):
            out['daily_ichimoku_conversion'] = float(conversion[-1])
        if base is not None and not np.isnan(base[-1]):
            out['daily_ichimoku_base'] = float(base[-1])
        
        # Handle span A
        self._process_ichimoku_span(span_a, out, 'daily_ichimoku_span_a')
        
        # Handle span B
        self._process_ichimoku_span(span_b, out, 'daily_ichimoku_span_b')
    
    def _process_ichimoku_span(self, span_data, out: Dict[str, Any], key: str) -> None:
        """Process Ichimoku span data safely."""
        value = get_last_valid_value(span_data)
        if value is not None:
            out[key] = value
