"""Market Metrics Calculator module.

Provides functionality for analyzer.market_metrics_calculator.py.
"""
import math

import numpy as np

from src.logger.logger import Logger
from src.utils.data_utils import last_or_scalar
from src.utils.timeframe_validator import TimeframeValidator


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
        """Initialize the calculator"""
        self.logger = logger

    def update_period_metrics(self, context) -> None:
        """Calculate and update market metrics for different time periods."""
        period_metrics = {}
        ohlcv = context.ohlcv_candles

        if ohlcv is None or len(ohlcv) == 0:
            self.logger.warning("No OHLCV data for period metrics")
            return

        if not context.timeframe:
            raise ValueError("context.timeframe is required to compute period metrics")
        timeframe = context.timeframe

        try:
            periods = {
                "1D": TimeframeValidator.calculate_period_candles(timeframe, "24h"),
                "2D": TimeframeValidator.calculate_period_candles(timeframe, "48h"),
                "3D": TimeframeValidator.calculate_period_candles(timeframe, "72h"),
                "7D": TimeframeValidator.calculate_period_candles(timeframe, "7d"),
                "30D": TimeframeValidator.calculate_period_candles(timeframe, "30d")
            }
        except (ValueError, TypeError) as exc:
            # The old fallback mapped 1D to 24 candles, i.e. it silently assumed an hourly
            # frame and reported wrong period metrics instead of naming the bad frame.
            raise ValueError(
                f"Cannot derive period candles for timeframe {timeframe!r}"
            ) from exc
        n = len(ohlcv)
        try:
            for period_name, required_candles in periods.items():
                if n >= required_candles:
                    period_metrics[period_name] = self._calculate_period_metrics(ohlcv[-required_candles:], period_name, context)
                elif period_name in ["1D", "2D", "3D"]:
                    self.logger.warning("Insufficient data for %s analysis. Need %s, have %s candles", period_name, required_candles, n)
                    period_metrics[period_name] = self._calculate_period_metrics(ohlcv, f"{period_name} (Partial)", context)
                elif period_name == "7D" and n >= periods["1D"]:
                    self.logger.warning("Insufficient data for 7D metrics. Only %s candles available, need %s", n, required_candles)
                    period_metrics["7D"] = self._calculate_period_metrics(ohlcv, "7D (Partial)", context)
                elif period_name == "30D" and n >= periods["7D"]:
                    self.logger.warning("Insufficient data for 30D metrics. Only %s candles available, need %s", n, required_candles)
                    period_metrics["30D"] = self._calculate_period_metrics(ohlcv, "30D (Partial)", context)
                else:
                    self.logger.warning("Cannot calculate %s metrics - not enough data (need %s, have %s)", period_name, required_candles, n)

            context.market_metrics = period_metrics

        except Exception as e:  # noqa: BLE001
            self.logger.error("Error updating period metrics: %s", e)
            if not period_metrics and n > 0:
                self.logger.warning("Setting fallback period metrics due to error")
                period_metrics["1D"] = self._calculate_period_metrics(ohlcv[-min(24, n):], "1D (Fallback)", context)
                context.market_metrics = period_metrics

    def _calculate_period_metrics(self, ohlcv_slice: np.ndarray, period_name: str, context) -> dict:
        """Calculate metrics for a specific time period."""
        basic_metrics = self._calculate_basic_metrics(ohlcv_slice, period_name)

        start_idx = -len(ohlcv_slice)
        end_idx = -1
        indicator_changes = self._calculate_indicator_changes_for_period(
            context, start_idx, end_idx, basic_metrics["total_volume"]
        )

        td = context.technical_data

        support = last_or_scalar(td.get("retested_support", np.nan))
        resistance = last_or_scalar(td.get("retested_resistance", np.nan))
        support_level = support if math.isfinite(support) else basic_metrics["lowest_price"]
        resistance_level = resistance if math.isfinite(resistance) else basic_metrics["highest_price"]

        levels = {
            "support": support_level,
            "resistance": resistance_level
        }

        return {
            "metrics": basic_metrics,
            "indicator_changes": indicator_changes,
            "key_levels": levels
        }

    def _calculate_basic_metrics(self, ohlcv_slice: np.ndarray, period_name: str) -> dict:
        """Calculate basic price and volume metrics using numpy vectorization."""
        highs = ohlcv_slice[:, 2]
        lows = ohlcv_slice[:, 3]
        prices = ohlcv_slice[:, 4]
        volumes = ohlcv_slice[:, 5]

        high_max = float(np.max(highs))
        low_min = float(np.min(lows))

        return {
            "highest_price": high_max,
            "lowest_price": low_min,
            "avg_price": float(np.mean(prices)),
            "total_volume": float(np.sum(volumes)),
            "avg_volume": float(np.mean(volumes)),
            "price_change": float(prices[-1] - prices[0]),
            "price_change_percent": float((prices[-1] / prices[0] - 1) * 100) if prices[0] != 0 else 0.0,
            "volatility": float((high_max - low_min) / low_min * 100) if low_min != 0 else 0.0,
            "period": period_name,
            "data_points": len(prices)
        }

    def _calculate_indicator_changes_for_period(
        self, context, start_idx: int, end_idx: int, total_volume: float
    ) -> dict:
        """Calculate changes in technical indicators over the period"""
        indicator_changes = {}

        try:
            history = context.technical_history
        except AttributeError:
            self.logger.debug("No technical_history available in context")
            return indicator_changes

        relevant_keys = [key for key in self.INDICATOR_CHANGE_KEYS if key in history]
        if not relevant_keys:
            self.logger.debug("No selected indicators available for change calculation")
            return indicator_changes

        for ind_name in relevant_keys:
            values = history.get(ind_name)
            try:
                if len(values) >= abs(start_idx):
                    try:
                        start_value = float(values[start_idx])
                        end_value = float(values[end_idx])
                        change = end_value - start_value

                        indicator_changes[f"{ind_name}_start"] = start_value
                        indicator_changes[f"{ind_name}_end"] = end_value
                        indicator_changes[f"{ind_name}_change"] = change
                        if ind_name == "obv":
                            indicator_changes["obv_flow_ratio"] = change / total_volume if total_volume > 0 else 0.0
                        else:
                            indicator_changes[f"{ind_name}_change_pct"] = (change / abs(start_value)) * 100 if start_value != 0 else 0
                    except (IndexError, ValueError, TypeError) as e:
                        self.logger.debug("Could not calculate change for %s: %s", ind_name, e)
                else:
                    self.logger.debug("%s has only %s values, need %s", ind_name, len(values), abs(start_idx))
            except TypeError:
                self.logger.debug("%s is scalar, skipping", ind_name)

        return indicator_changes

