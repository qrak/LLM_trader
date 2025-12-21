from typing import Dict, Any, Optional
import numpy as np

from src.indicators.base.technical_indicators import TechnicalIndicators
from src.indicators.constants import INDICATOR_THRESHOLDS
from src.logger.logger import Logger


class TechnicalCalculator:
    """Core calculator for technical indicators"""
    
    def __init__(self, logger: Optional[Logger] = None, format_utils=None):
        """Initialize the technical indicator calculator"""
        self.logger = logger
        self.format_utils = format_utils
        self.ti = TechnicalIndicators()
        
        # Reference the global indicator thresholds constant
        self.INDICATOR_THRESHOLDS = INDICATOR_THRESHOLDS
        
    def get_indicators(self, ohlcv_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate all technical indicators - no caching, always fresh"""
        self.ti.get_data(ohlcv_data)
        
        indicators = {}
        
        # Calculate indicators by category
        indicators.update(self._calculate_volume_indicators())
        indicators.update(self._calculate_momentum_indicators())
        indicators.update(self._calculate_volatility_indicators(ohlcv_data))
        indicators.update(self._calculate_trend_indicators())
        indicators.update(self._calculate_support_resistance_indicators())
        
        # Add signal interpretations
        self._add_signal_interpretations(indicators, ohlcv_data)
        
        if self.logger:
            self.logger.debug("Calculated technical indicators")
        
        return indicators

    def _calculate_volume_indicators(self) -> Dict[str, np.ndarray]:
        """Calculate volume-based indicators"""
        return {
            "vwap": self.ti.vol.rolling_vwap(length=20),
            "twap": self.ti.vol.twap(length=20),
            "mfi": self.ti.vol.mfi(length=14),
            "obv": self.ti.vol.obv(length=20),
            "cmf": self.ti.vol.chaikin_money_flow(length=20),
            "force_index": self.ti.vol.force_index(length=20),
            "cci": self.ti.vol.cci(length=14),
        }

    def _calculate_momentum_indicators(self) -> Dict[str, np.ndarray]:
        """Calculate momentum indicators"""
        indicators = {
            "rsi": self.ti.momentum.rsi(length=14),
            "stoch_k": self.ti.momentum.stochastic(period_k=14, smooth_k=3, period_d=3)[0],
            "stoch_d": self.ti.momentum.stochastic(period_k=14, smooth_k=3, period_d=3)[1],
            "williams_r": self.ti.momentum.williams_r(length=14),
            "uo": self.ti.momentum.uo(),
            "tsi": self.ti.momentum.tsi(long_length=20, short_length=10),
            "rmi": self.ti.momentum.rmi(length=20, momentum_length=5),
            "ppo": self.ti.momentum.ppo(fast_length=12, slow_length=26),
            "coppock": self.ti.momentum.coppock_curve(wl1=11, wl2=14, wma_length=10),
            "kst": self.ti.momentum.kst(),
        }
        
        # Calculate MACD separately
        macd_line, macd_signal, macd_hist = self.ti.momentum.macd(fast_length=12, slow_length=26, signal_length=9)
        indicators["macd_line"] = macd_line
        indicators["macd_signal"] = macd_signal
        indicators["macd_hist"] = macd_hist
        
        return indicators

    def _calculate_volatility_indicators(self, ohlcv_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate volatility indicators"""
        indicators = {
            "atr": self.ti.volatility.atr(length=20),
        }
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.ti.volatility.bollinger_bands(length=20, num_std_dev=2)
        indicators["bb_upper"] = bb_upper
        indicators["bb_middle"] = bb_middle
        indicators["bb_lower"] = bb_lower
        
        # Calculate BB %B
        close_prices = ohlcv_data[:, 3]
        band_width = bb_upper - bb_lower
        indicators["bb_percent_b"] = np.where(
            band_width != 0,
            (close_prices - bb_lower) / band_width,
            np.nan
        )
        
        # Keltner Channels
        kc_upper, kc_middle, kc_lower = self.ti.volatility.keltner_channels(length=20, multiplier=2)
        indicators["kc_upper"] = kc_upper
        indicators["kc_middle"] = kc_middle
        indicators["kc_lower"] = kc_lower
        
        # Donchian Channels (only retain extremes for formatting/patterns)
        donchian_upper, _, donchian_lower = self.ti.volatility.donchian_channels(length=20)
        indicators["donchian_upper"] = donchian_upper
        indicators["donchian_lower"] = donchian_lower
        
        # Chandelier Exit
        long_exit, short_exit = self.ti.volatility.chandelier_exit(length=20, multiplier=3.0)
        indicators["chandelier_long"] = long_exit
        indicators["chandelier_short"] = short_exit
        
        # ATR Percentage
        current_price = ohlcv_data[-1, 4] if len(ohlcv_data) > 0 else 1
        atr_values = indicators["atr"]
        indicators["atr_percent"] = (atr_values / current_price) * 100 if current_price > 0 else np.full_like(atr_values, np.nan)
        
        return indicators

    def _calculate_trend_indicators(self) -> Dict[str, np.ndarray]:
        """Calculate trend indicators"""
        indicators = {
            "adx": self.ti.trend.adx(length=14)[0],
            "plus_di": self.ti.trend.adx(length=14)[1],
            "minus_di": self.ti.trend.adx(length=14)[2],
            "trix": self.ti.trend.trix(length=20),
            "pfe": self.ti.trend.pfe(n=20, m=5),
            "td_sequential": self.ti.trend.td_sequential(length=9),
            "sar": self.ti.trend.parabolic_sar(step=0.02, max_step=0.2),
        }
        
        # Supertrend
        supertrend, supertrend_direction = self.ti.trend.supertrend(length=20, multiplier=3.0)
        indicators["supertrend"] = supertrend
        indicators["supertrend_direction"] = supertrend_direction

        # Ichimoku Cloud
        conversion, base, span_a, span_b = self.ti.trend.ichimoku_cloud(
            conversion_length=9,
            base_length=26,
            lagging_span2_length=52,
            displacement=26
        )
        indicators["ichimoku_span_a"] = span_a
        indicators["ichimoku_span_b"] = span_b
        
        # Vortex
        vortex_plus, vortex_minus = self.ti.trend.vortex_indicator(length=20)
        indicators["vortex_plus"] = vortex_plus
        indicators["vortex_minus"] = vortex_minus
        
        # SMAs
        indicators["sma_20"] = self.ti.overlap.sma(self.ti.close, 20)
        indicators["sma_50"] = self.ti.overlap.sma(self.ti.close, 50)
        indicators["sma_200"] = self.ti.overlap.sma(self.ti.close, 200)
        
        return indicators

    def _calculate_support_resistance_indicators(self) -> Dict[str, np.ndarray]:
        """Calculate support and resistance indicators"""
        indicators = {
            "kurtosis": self.ti.statistical.kurtosis(length=20),
            "zscore": self.ti.statistical.zscore(length=20),
            "hurst": self.ti.statistical.hurst(max_lag=20),
        }
        
        # Basic Support/Resistance
        support, resistance = self.ti.support_resistance.support_resistance(length=20)
        indicators["basic_support"] = support
        indicators["basic_resistance"] = resistance
        
        # Advanced Support/Resistance
        adv_support, adv_resistance = self.ti.support_resistance.advanced_support_resistance(
            length=20,
            strength_threshold=1,
            persistence=1,
            volume_factor=1.5,
            price_factor=0.004
        )
        indicators["advanced_support"] = adv_support
        indicators["advanced_resistance"] = adv_resistance
        
        # Pivot Points
        pivot_point, r1, r2, r3, r4, s1, s2, s3, s4 = self.ti.support_resistance.pivot_points()
        indicators["pivot_point"] = pivot_point
        indicators["pivot_r1"] = r1
        indicators["pivot_r2"] = r2
        indicators["pivot_r3"] = r3
        indicators["pivot_r4"] = r4
        indicators["pivot_s1"] = s1
        indicators["pivot_s2"] = s2
        indicators["pivot_s3"] = s3
        indicators["pivot_s4"] = s4
        
        return indicators
        
    def get_long_term_indicators(self, ohlcv_data: np.ndarray) -> Dict[str, Any]:
        """Calculate long-term indicators for historical data - no caching, always fresh"""
        # Create new TI instance for long-term calculations (avoid interference with regular timeframe indicators)
        ti_lt = TechnicalIndicators()
        ti_lt.get_data(ohlcv_data)
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
        ti_weekly = TechnicalIndicators()
        ti_weekly.get_data(weekly_ohlcv_data)
        available_weeks = len(weekly_ohlcv_data)

        # REUSE existing helper methods (already timeframe-agnostic)
        weekly_sma_values, weekly_volume_sma_values = self._compute_sma_sets(ti_weekly, available_weeks)
        weekly_price_change, weekly_volume_change = self._compute_change_metrics(ti_weekly, available_weeks)
        weekly_volatility = self._compute_volatility(ti_weekly, available_weeks)

        # Pre-calculate SMA arrays for crossover detection (avoid redundant calculation in macro analysis)
        sma_arrays = {}
        if available_weeks >= 50:
            sma_arrays['sma_50'] = ti_weekly.overlap.sma(ti_weekly.close, 50)
        if available_weeks >= 200:
            sma_arrays['sma_200'] = ti_weekly.overlap.sma(ti_weekly.close, 200)

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
            
            # Cycle phase based on 200W distance
            if distance < -10:
                analysis['cycle_phase'] = 'Deep Bear Market'
            elif distance < 0:
                analysis['cycle_phase'] = 'Bear Market Bottom Zone'
            elif distance < 50:
                analysis['cycle_phase'] = 'Early Bull Market'
            elif distance < 100:
                analysis['cycle_phase'] = 'Mid Bull Market'
            elif distance < 200:
                analysis['cycle_phase'] = 'Late Bull Market'
            else:
                analysis['cycle_phase'] = 'Extreme Bull Market'
        
        # Golden/Death Cross with timestamps (use pre-calculated arrays if available)
        if 50 in weekly_sma_values and 200 in weekly_sma_values:
            # Use passed arrays or calculate if not provided (backward compatibility)
            if sma_arrays and 'sma_50' in sma_arrays and 'sma_200' in sma_arrays:
                sma_50w_array = sma_arrays['sma_50']
                sma_200w_array = sma_arrays['sma_200']
            else:
                sma_50w_array = ti.overlap.sma(ti.close, 50)
                sma_200w_array = ti.overlap.sma(ti.close, 200)
            
            from src.analyzer.pattern_engine.indicator_patterns.ma_crossover_patterns import (
                detect_golden_cross_numba, detect_death_cross_numba
            )
            
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

    # ---------------- Helper Methods (extracted for clarity) ----------------
    def _compute_sma_sets(self, ti: TechnicalIndicators, available_days: int):
        sma_periods = [20, 50, 100, 200]
        sma_values: Dict[int, float] = {}
        volume_sma_values: Dict[int, float] = {}
        for period in sma_periods:
            if available_days >= period:
                # Use technical indicators directly instead of extracted arrays
                sma = ti.overlap.sma(ti.close, period)
                vol_sma = ti.overlap.sma(ti.volume, period)
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
                self.logger.debug(f"Long-term price change: {price_change_pct:.2f}% over {available_days} days")
                self.logger.debug(f"Long-term volume change: {volume_change_pct:.2f}%")
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
            self.logger.debug(f"Macro trend: {available_days}-day price change = {price_change_pct:.2f}%")
        
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
                    sma_50_prev = ti.overlap.sma(ti.close, 50)[-10]  # 10 periods ago
                    sma_200_prev = ti.overlap.sma(ti.close, 200)[-10]
                    if sma_50_prev <= sma_200_prev and sma_50 > sma_200:
                        analysis['golden_cross'] = True
            elif sma_50 < sma_200:
                analysis['sma_50_vs_200'] = 'Bearish'
                # Check if this could be a death cross scenario
                if available_days >= 250:
                    sma_50_prev = ti.overlap.sma(ti.close, 50)[-10]
                    sma_200_prev = ti.overlap.sma(ti.close, 200)[-10]
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
            self.logger.debug(f"Macro trend bullish signals ({bullish_count}): {bullish_signals}")
            self.logger.debug(f"Macro trend bearish signals ({bearish_count}): {bearish_signals}")
        
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
        rsi_vals = ti.momentum.rsi(length=14)
        if rsi_vals is not None and not np.isnan(rsi_vals[-1]):
            out['daily_rsi'] = float(rsi_vals[-1])
        
        # ATR
        atr_vals = ti.volatility.atr(length=14)
        if atr_vals is not None and not np.isnan(atr_vals[-1]):
            out['daily_atr'] = float(atr_vals[-1])
        
        # ADX and DI
        adx_vals, plus_di_vals, minus_di_vals = ti.trend.adx(length=14)
        if adx_vals is not None and not np.isnan(adx_vals[-1]):
            out['daily_adx'] = float(adx_vals[-1])
        if plus_di_vals is not None and not np.isnan(plus_di_vals[-1]):
            out['daily_plus_di'] = float(plus_di_vals[-1])
        if minus_di_vals is not None and not np.isnan(minus_di_vals[-1]):
            out['daily_minus_di'] = float(minus_di_vals[-1])
        
        # OBV
        obv_vals = ti.vol.obv()
        if obv_vals is not None and not np.isnan(obv_vals[-1]):
            out['daily_obv'] = float(obv_vals[-1])
    
    def _compute_26_day_indicators(self, ti: TechnicalIndicators, out: Dict[str, Any]) -> None:
        """Compute indicators that require 26 days of data."""
        macd_line, macd_signal, macd_hist = ti.momentum.macd()
        
        if macd_line is not None and not np.isnan(macd_line[-1]):
            out['daily_macd_line'] = float(macd_line[-1])
        if macd_signal is not None and not np.isnan(macd_signal[-1]):
            out['daily_macd_signal'] = float(macd_signal[-1])
        if macd_hist is not None and not np.isnan(macd_hist[-1]):
            out['daily_macd_hist'] = float(macd_hist[-1])
    
    def _compute_52_day_indicators(self, ti: TechnicalIndicators, out: Dict[str, Any]) -> None:
        """Compute indicators that require 52 days of data."""
        conversion, base, span_a, span_b = ti.trend.ichimoku_cloud()
        
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
        if span_data is not None and len(span_data) > 0:
            span_valid = np.where(~np.isnan(span_data))[0]
            if len(span_valid) > 0:
                out[key] = float(span_data[span_valid[-1]])

    def _add_signal_interpretations(self, indicators: dict, ohlcv_data: np.ndarray) -> None:
        """Add signal interpretations for various indicators.
        
        Args:
            indicators: Dictionary to add signal interpretations to
            ohlcv_data: OHLCV data array for current price
        """
        if ohlcv_data is None or len(ohlcv_data) == 0:
            return
            
        current_price = float(ohlcv_data[-1, 4])  # Close price
        
        # Ichimoku Signal
        if all(key in indicators for key in ['ichimoku_span_a', 'ichimoku_span_b']):
            span_a = indicators.get('ichimoku_span_a')
            span_b = indicators.get('ichimoku_span_b')
            
            # Handle numpy arrays by taking the last value
            if hasattr(span_a, '__iter__') and not isinstance(span_a, str):
                span_a = span_a[-1] if len(span_a) > 0 else None
            if hasattr(span_b, '__iter__') and not isinstance(span_b, str):
                span_b = span_b[-1] if len(span_b) > 0 else None
            
            if isinstance(span_a, (int, float)) and isinstance(span_b, (int, float)) and not (np.isnan(span_a) or np.isnan(span_b)):
                cloud_top = max(span_a, span_b)
                cloud_bottom = min(span_a, span_b)
                
                if current_price > cloud_top:
                    indicators["ichimoku_signal"] = 1  # Bullish
                elif current_price < cloud_bottom:
                    indicators["ichimoku_signal"] = -1  # Bearish
                else:
                    indicators["ichimoku_signal"] = 0  # In cloud
            else:
                indicators["ichimoku_signal"] = 0
        else:
            indicators["ichimoku_signal"] = 0
            
        # Bollinger Bands Signal
        if all(key in indicators for key in ['bb_upper', 'bb_middle', 'bb_lower']):
            bb_upper = indicators.get('bb_upper')
            bb_middle = indicators.get('bb_middle')
            bb_lower = indicators.get('bb_lower')
            
            # Handle numpy arrays by taking the last value
            if hasattr(bb_upper, '__iter__') and not isinstance(bb_upper, str):
                bb_upper = bb_upper[-1] if len(bb_upper) > 0 else None
            if hasattr(bb_middle, '__iter__') and not isinstance(bb_middle, str):
                bb_middle = bb_middle[-1] if len(bb_middle) > 0 else None
            if hasattr(bb_lower, '__iter__') and not isinstance(bb_lower, str):
                bb_lower = bb_lower[-1] if len(bb_lower) > 0 else None
            
            if all(isinstance(val, (int, float)) and not np.isnan(val) for val in [bb_upper, bb_middle, bb_lower]):
                # Calculate distance to each band as percentage
                upper_dist = abs(current_price - bb_upper) / bb_upper
                _middle_dist = abs(current_price - bb_middle) / bb_middle
                lower_dist = abs(current_price - bb_lower) / bb_lower
                
                # Find closest band (threshold of 2% to determine "near")
                threshold = 0.02
                if upper_dist < threshold:
                    indicators["bb_signal"] = 1  # Near upper band
                elif lower_dist < threshold:
                    indicators["bb_signal"] = -1  # Near lower band
                else:
                    indicators["bb_signal"] = 0  # Near middle or between bands
            else:
                indicators["bb_signal"] = 0
        else:
            indicators["bb_signal"] = 0
