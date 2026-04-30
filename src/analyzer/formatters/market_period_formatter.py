"""
Market Period Formatter - Formats period-based market metrics.
Handles 1h, 4h, 24h, 7d, 30d period metrics and indicator changes.
"""
from typing import Optional, TYPE_CHECKING

from src.logger.logger import Logger

if TYPE_CHECKING:
    from src.utils.format_utils import FormatUtils


class MarketPeriodFormatter:
    """Formatter for period-based market metrics and indicator changes."""

    def __init__(self, logger: Optional[Logger] = None, format_utils: "FormatUtils" = None):
        """Initialize the market period formatter.

        Args:
            logger: Optional logger instance
            format_utils: Format utilities for value formatting (required)
        """
        self.logger = logger
        if format_utils is None:
            raise ValueError("format_utils is required for MarketPeriodFormatter")
        self.format_utils = format_utils

    def format_market_period_metrics(self, market_metrics: dict) -> str:
        """Format market metrics for different periods (compressed format)."""
        if not market_metrics:
            return ""

        sections = ["## Period Metrics"]

        for period, period_data in market_metrics.items():
            if not period_data:
                continue

            # Extract metrics from nested structure
            metrics = period_data.get('metrics', {})
            if not metrics:
                continue

            # Build compressed single-line format
            parts = []

            # Price metrics (compressed)
            avg_price = metrics.get('avg_price')
            lowest_price = metrics.get('lowest_price')
            highest_price = metrics.get('highest_price')
            price_change_percent = metrics.get('price_change_percent')

            if avg_price:
                parts.append(f"Avg${self.format_utils.fmt(avg_price)}")
            if lowest_price and highest_price:
                parts.append(f"Range${self.format_utils.fmt(lowest_price)}-{self.format_utils.fmt(highest_price)}")
            if price_change_percent is not None:
                direction = "↑" if price_change_percent >= 0 else "↓"
                parts.append(f"Δ{direction}{self.format_utils.fmt(abs(price_change_percent))}%")

            # Volume metrics (compressed)
            total_volume = metrics.get('total_volume')
            if total_volume:
                parts.append(f"Vol:{self.format_utils.fmt(total_volume)}")

            # Indicator changes (compressed)
            if 'indicator_changes' in period_data:
                ind_parts = self._format_indicator_changes_compressed(period_data['indicator_changes'])
                if ind_parts:
                    parts.append(ind_parts)

            if parts:
                period_label = str(metrics.get('period') or period).upper()
                sections.append(f"\n{period_label}: {' | '.join(parts)}")

        return "".join(sections)

    def _format_indicator_changes_compressed(self, indicator_changes: dict) -> str:
        """Format indicator changes in compressed format (e.g., 'RSI↓9.1 ADX↑5.1')."""
        if not indicator_changes:
            return ""

        parts = []

        # RSI changes
        rsi_change = indicator_changes.get('rsi_change')
        if rsi_change is not None and abs(rsi_change) > 0.1:  # Only show significant changes
            direction = "↑ " if rsi_change >= 0 else "↓ "
            parts.append(f"RSI {direction}{self.format_utils.fmt(abs(rsi_change))}")

        # MACD changes
        macd_change = indicator_changes.get('macd_line_change')
        if macd_change is not None and abs(macd_change) > 1:  # Only show significant changes
            direction = "↑ " if macd_change >= 0 else "↓ "
            parts.append(f"MACD {direction}{self.format_utils.fmt(abs(macd_change))}")

        # ADX changes
        adx_change = indicator_changes.get('adx_change')
        if adx_change is not None and abs(adx_change) > 0.5:  # Only show significant changes
            direction = "↑ " if adx_change >= 0 else "↓ "
            parts.append(f"ADX {direction}{self.format_utils.fmt(abs(adx_change))}")

        # Stochastic %K changes
        stoch_change = indicator_changes.get('stoch_k_change')
        if stoch_change is not None and abs(stoch_change) > 1:  # Only show significant changes
            direction = "↑ " if stoch_change >= 0 else "↓ "
            parts.append(f"Stoch {direction}{self.format_utils.fmt(abs(stoch_change))}")

        return " ".join(parts) if parts else ""
