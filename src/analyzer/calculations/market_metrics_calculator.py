from typing import Dict, List
import numpy as np

from src.logger.logger import Logger


class MarketMetricsCalculator:
    """Handles calculation of market metrics and technical pattern detection"""
    INDICATOR_CHANGE_KEYS = (
        "rsi",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "adx",
        "stoch_k",
        "stoch_d",
        "mfi",
        "obv",
        "bb_percent_b",
        "atr",
        "cmf",
        "force_index",
    )
    
    def __init__(self, logger: Logger):
        """Initialize the calculator
        
        Args:
            logger: Logger instance
        """
        self.logger = logger
    
    def update_period_metrics(self, data: List, context) -> None:
        """Calculate and update market metrics for different time periods"""
        period_metrics = {}
        
        hourly_distribution = {}
        for item in data:
            hour = item['timestamp'].hour
            hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
        
        # Get timeframe from context to calculate correct candle counts
        from src.utils.timeframe_validator import TimeframeValidator
        timeframe = context.timeframe if hasattr(context, 'timeframe') and context.timeframe else '1h'
        
        # Calculate period candle requirements based on actual timeframe
        try:
            periods = {
                "1D": TimeframeValidator.calculate_period_candles(timeframe, "24h"),
                "2D": TimeframeValidator.calculate_period_candles(timeframe, "48h"),
                "3D": TimeframeValidator.calculate_period_candles(timeframe, "72h"),
                "7D": TimeframeValidator.calculate_period_candles(timeframe, "7d"),
                "30D": TimeframeValidator.calculate_period_candles(timeframe, "30d")
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate dynamic periods for timeframe {timeframe}: {e}")
            # Fallback to 1h assumptions
            periods = {
                "1D": 24,
                "2D": 48,
                "3D": 72,
                "7D": 168,
                "30D": 720
            }
        
        self.logger.debug(f"Total candles available for period metrics: {len(data)}, Timeframe: {timeframe}")
        self.logger.debug(f"Period candle requirements: {periods}")
        
        try:
            for period_name, required_candles in periods.items():
                if len(data) >= required_candles:
                    self.logger.debug(f"Calculating full {period_name} metrics with {required_candles} candles")
                    period_metrics[period_name] = self._calculate_period_metrics(data[-required_candles:], period_name, context)
                else:
                    if period_name in ["1D", "2D", "3D"]:
                        self.logger.warning(f"Insufficient data for {period_name} analysis. Need {required_candles}, have {len(data)} candles")
                        period_metrics[period_name] = self._calculate_period_metrics(data, f"{period_name} (Partial)", context)
                    elif period_name == "7D" and len(data) >= periods["1D"]:  # Use dynamic 1D requirement
                        self.logger.warning(f"Insufficient data for 7D metrics. Only {len(data)} candles available, need {required_candles}")
                        period_metrics["7D"] = self._calculate_period_metrics(data, "7D (Partial)", context)
                    elif period_name == "30D" and len(data) >= periods["7D"]:  # Use dynamic 7D requirement
                        self.logger.warning(f"Insufficient data for 30D metrics. Only {len(data)} candles available, need {required_candles}")
                        period_metrics["30D"] = self._calculate_period_metrics(data, "30D (Partial)", context)
                    else:
                        self.logger.warning(f"Cannot calculate {period_name} metrics - not enough data (need {required_candles}, have {len(data)})")
            
            # Log what was calculated
            for period_name, metrics_data in period_metrics.items():
                if metrics_data and 'metrics' in metrics_data:
                    basic_metrics = metrics_data['metrics']
                    self.logger.debug(f"{period_name}: price_change={basic_metrics.get('price_change')}, "
                                     f"price_change_percent={basic_metrics.get('price_change_percent')}%, "
                                     f"data_points={basic_metrics.get('data_points')}")
                    if 'indicator_changes' in metrics_data:
                        ind_changes = metrics_data['indicator_changes']
                        rsi_change = ind_changes.get('rsi_change', 'N/A')
                        macd_change = ind_changes.get('macd_line_change', 'N/A')
                        self.logger.debug(f"{period_name} indicator changes: RSI={rsi_change}, MACD={macd_change}")
            
            context.market_metrics = period_metrics
        
        except Exception as e:
            self.logger.error(f"Error updating period metrics: {e}")
            if not period_metrics and len(data) > 0:
                self.logger.warning("Setting fallback period metrics due to error")
                period_metrics["1D"] = self._calculate_period_metrics(data[-min(24, len(data)):], "1D (Fallback)", context)
                context.market_metrics = period_metrics
                
    def _calculate_period_metrics(self, data: List, period_name: str, context) -> Dict:
        """Calculate metrics for a specific time period"""
        # Calculate core metrics directly from data (do this FIRST to avoid redundant calculations)
        basic_metrics = self._calculate_basic_metrics(data, period_name)
        
        # Calculate indicator changes
        start_idx = -len(data)
        end_idx = -1
        indicator_changes = self._calculate_indicator_changes_for_period(context, start_idx, end_idx)
        
        # Use support/resistance from technical_calculator instead of duplicating
        current_price = data[-1]["close"]
        td = context.technical_data
        
        # Get support/resistance from existing technical indicators
        support_level = current_price
        resistance_level = current_price
        
        if 'advanced_support' in td and 'advanced_resistance' in td:
            adv_support = td.get('advanced_support', np.nan)
            adv_resistance = td.get('advanced_resistance', np.nan)
            
            # Handle array indicators - take the last value, following promptt_builder.py pattern
            try:
                if len(adv_support) > 0:
                    adv_support = adv_support[-1]
            except TypeError:
                # adv_support is already a scalar value
                pass
                
            try:
                if len(adv_resistance) > 0:
                    adv_resistance = adv_resistance[-1]
            except TypeError:
                # adv_resistance is already a scalar value
                pass
            
            # Use valid values or fallback to already-calculated values from basic_metrics
            if not np.isnan(adv_support):
                support_level = adv_support
            else:
                support_level = basic_metrics["lowest_price"]
                
            if not np.isnan(adv_resistance):
                resistance_level = adv_resistance
            else:
                resistance_level = basic_metrics["highest_price"]
        else:
            # Fallback to already-calculated values from basic_metrics
            support_level = basic_metrics["lowest_price"]
            resistance_level = basic_metrics["highest_price"]
        
        levels = {
            "support": support_level,
            "resistance": resistance_level
        }
        
        return {
            "metrics": basic_metrics,
            "indicator_changes": indicator_changes,
            "key_levels": levels
        }
    
    def _calculate_basic_metrics(self, data: List[Dict], period_name: str) -> Dict:
        """Calculate basic price and volume metrics"""
        prices = [candle["close"] for candle in data]
        highs = [candle["high"] for candle in data]
        lows = [candle["low"] for candle in data]
        volumes = [candle["volume"] for candle in data]
        
        return {
            "highest_price": max(highs),
            "lowest_price": min(lows),
            "avg_price": sum(prices) / len(prices),
            "total_volume": sum(volumes),
            "avg_volume": sum(volumes) / len(volumes),
            "price_change": prices[-1] - prices[0],
            "price_change_percent": ((prices[-1] / prices[0]) - 1) * 100,
            "volatility": (max(highs) - min(lows)) / min(lows) * 100,
            "period": period_name,
            "data_points": len(prices)
        }
    
    def _calculate_indicator_changes_for_period(self, context, start_idx: int, end_idx: int) -> Dict:
        """Calculate changes in technical indicators over the period"""
        indicator_changes = {}
        
        if not hasattr(context, 'technical_history'):
            self.logger.debug("No technical_history available in context")
            return indicator_changes
            
        history = context.technical_history
        self.logger.debug(f"Calculating indicator changes from index {start_idx} to {end_idx}")
        
        relevant_keys = [key for key in self.INDICATOR_CHANGE_KEYS if key in history]
        if not relevant_keys:
            self.logger.debug("No whitelisted indicators available for change calculation")
            return indicator_changes

        for ind_name in relevant_keys:
            values = history.get(ind_name)
            try:
                if len(values) >= abs(start_idx):
                    try:
                        start_value = float(values[start_idx])
                        end_value = float(values[end_idx])
                        change = end_value - start_value
                        change_pct = (change / abs(start_value)) * 100 if start_value != 0 else 0
                        
                        indicator_changes[f"{ind_name}_start"] = start_value
                        indicator_changes[f"{ind_name}_end"] = end_value
                        indicator_changes[f"{ind_name}_change"] = change
                        indicator_changes[f"{ind_name}_change_pct"] = change_pct
                        
                        # Log key indicators
                        if ind_name in ['rsi', 'macd_line', 'adx']:
                            self.logger.debug(f"{ind_name}: start={start_value:.2f}, end={end_value:.2f}, change={change:.2f}")
                    except (IndexError, ValueError, TypeError) as e:
                        self.logger.debug(f"Could not calculate change for {ind_name}: {e}")
                else:
                    self.logger.debug(f"{ind_name} has only {len(values)} values, need {abs(start_idx)}")
            except TypeError:
                # values is a scalar numpy value, not an array
                self.logger.debug(f"{ind_name} is scalar, skipping")
                pass
        
        self.logger.debug(f"Calculated {len(indicator_changes)} indicator change metrics across {len(relevant_keys)} indicators")
        return indicator_changes
    
