
import math
from typing import TYPE_CHECKING

from src.logger.logger import Logger

if TYPE_CHECKING:
    from src.utils.format_utils import FormatUtils


class LongTermFormatter:
    def __init__(self, logger: Logger | None = None, format_utils: "FormatUtils | None" = None):

        self.logger = logger
        if format_utils is None:
            raise ValueError("format_utils is required for LongTermFormatter")
        self.format_utils = format_utils

    def format_long_term_analysis(self, long_term_data: dict, current_price: float | None = None) -> str:

        if not long_term_data:
            return ""

        sections = []

        sma_section = self._format_sma_section(long_term_data)
        if sma_section:
            sections.append(sma_section)

        volume_sma_section = self._format_volume_sma_section(long_term_data)
        if volume_sma_section:
            sections.append(volume_sma_section)

        if current_price:
            price_position_section = self._format_price_position_section(long_term_data, current_price)
            if price_position_section:
                sections.append(price_position_section)

        daily_indicators_section = self._format_daily_indicators_section(long_term_data)
        if daily_indicators_section:
            sections.append(daily_indicators_section)

        adx_section = self._format_adx_section(long_term_data)
        if adx_section:
            sections.append(adx_section)

        if current_price:
            ichimoku_section = self._format_ichimoku_section(long_term_data, current_price)
            if ichimoku_section:
                sections.append(ichimoku_section)

        if "macro_trend" in long_term_data:
            macro_trend_section = self._format_macro_trend_section(long_term_data["macro_trend"])
            if macro_trend_section:
                sections.append(macro_trend_section)

        if sections:
            return "\n\n".join(sections)

        return ""

    SMA_PERIODS = (20, 50, 100, 200)
    VOLUME_SMA_PERIODS = (20, 50)

    @staticmethod
    def _measurement(value: float | None) -> float | None:

        if isinstance(value, (int, float)) and not math.isnan(value):
            return float(value)
        return None

    def _format_items_for_periods(self, values: dict, periods: tuple,
                                  format_fn, item_template: str) -> list[str]:

        items = []
        for period in periods:
            value = self._measurement(values.get(period))
            if value is not None:
                items.append(item_template.format(period=period, value=format_fn(value)))
        return items

    def _format_sma_section(self, long_term_data: dict) -> str:

        items = self._format_items_for_periods(
            long_term_data.get("sma_values") or {},
            self.SMA_PERIODS,
            self.format_utils.fmt,
            "SMA{period}: {value}"
        )
        return "## Simple Moving Averages (Daily):\n" + " | ".join(items) if items else ""

    def _format_volume_sma_section(self, long_term_data: dict) -> str:

        items = self._format_items_for_periods(
            long_term_data.get("volume_sma_values") or {},
            self.VOLUME_SMA_PERIODS,
            self.format_utils.fmt,
            "Vol SMA{period}: {value}"
        )
        return "## Volume Moving Averages (Daily):\n" + " | ".join(items) if items else ""

    def _format_price_position_section(self, long_term_data: dict, current_price: float) -> str:

        sma_values = long_term_data.get("sma_values") or {}
        items = []
        for period in self.SMA_PERIODS:
            sma_value = self._measurement(sma_values.get(period))
            if sma_value:
                percentage = ((current_price - sma_value) / sma_value) * 100
                direction = "above" if percentage > 0 else "below"
                items.append(f"SMA{period}: {self.format_utils.fmt(abs(percentage))}% {direction}")

        return "## Price Position vs SMAs (Daily):\n" + " | ".join(items) if items else ""

    def _format_daily_indicators_section(self, long_term_data: dict) -> str:

        indicator_items = []

        rsi_val = self._measurement(long_term_data.get("daily_rsi"))
        if rsi_val is not None:
            rsi_status = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
            indicator_items.append(f"Daily RSI: {self.format_utils.fmt(rsi_val)} ({rsi_status})")

        macd_line = self._measurement(long_term_data.get("daily_macd_line"))
        macd_signal = self._measurement(long_term_data.get("daily_macd_signal"))
        if macd_line is not None and macd_signal is not None:
            macd_status = "Bullish" if macd_line > macd_signal else "Bearish"
            indicator_items.append(f"Daily MACD: {macd_status}")

        stoch_val = self._measurement(long_term_data.get("daily_stoch_k"))
        if stoch_val is not None:
            stoch_status = "Overbought" if stoch_val > 80 else "Oversold" if stoch_val < 20 else "Neutral"
            indicator_items.append(f"Daily Stoch: {self.format_utils.fmt(stoch_val)} ({stoch_status})")

        if indicator_items:
            return "## Daily Indicators:\n" + " | ".join(indicator_items)
        return ""

    def _format_adx_section(self, long_term_data: dict) -> str:

        adx_val = self._measurement(long_term_data.get("daily_adx"))
        if adx_val is None:
            return ""

        if adx_val < 25:
            strength = "Weak/No Trend"
        elif adx_val < 50:
            strength = "Strong Trend"
        elif adx_val < 75:
            strength = "Very Strong Trend"
        else:
            strength = "Extremely Strong Trend"

        return f"## Trend Strength (Daily ADX): {self.format_utils.fmt(adx_val)} ({strength})"

    def _format_ichimoku_section(self, long_term_data: dict, current_price: float) -> str:

        ichimoku_items = []

        tenkan = self._measurement(long_term_data.get("daily_ichimoku_conversion"))
        if tenkan is not None:
            ichimoku_items.append(f"Tenkan: {self.format_utils.fmt(tenkan)}")

        kijun = self._measurement(long_term_data.get("daily_ichimoku_base"))
        if kijun is not None:
            ichimoku_items.append(f"Kijun: {self.format_utils.fmt(kijun)}")

        span_a = self._measurement(long_term_data.get("daily_ichimoku_span_a"))
        span_b = self._measurement(long_term_data.get("daily_ichimoku_span_b"))
        if span_a is not None and span_b is not None:
            cloud_top = max(span_a, span_b)
            cloud_bottom = min(span_a, span_b)

            if current_price > cloud_top:
                cloud_position = "Above Cloud (Bullish)"
            elif current_price < cloud_bottom:
                cloud_position = "Below Cloud (Bearish)"
            else:
                cloud_position = "Inside Cloud (Neutral)"

            ichimoku_items.append(f"Cloud Position: {cloud_position}")

        if ichimoku_items:
            return "## Ichimoku Analysis (Daily):\n" + " | ".join(ichimoku_items)
        return ""

    def _format_macro_trend_section(self, macro_trend: dict) -> str:

        if not macro_trend:
            return ""

        trend_direction = macro_trend.get("trend_direction", "Neutral")
        sma_alignment = macro_trend.get("sma_alignment", "Mixed")
        sma_50_vs_200 = macro_trend.get("sma_50_vs_200", "Neutral")
        price_above_200sma = macro_trend.get("price_above_200sma", False)
        golden_cross = macro_trend.get("golden_cross", False)
        death_cross = macro_trend.get("death_cross", False)
        long_term_price_change_pct = macro_trend.get("long_term_price_change_pct")

        status_parts = []
        status_parts.append(f"Trend: {trend_direction}")

        if long_term_price_change_pct is not None:
            change_sign = "+" if long_term_price_change_pct >= 0 else ""
            status_parts.append(f"365D Change: {change_sign}{self.format_utils.fmt(long_term_price_change_pct)}%")

        status_parts.append(f"SMA Alignment: {sma_alignment}")
        status_parts.append(f"50vs200 SMA: {sma_50_vs_200}")
        status_parts.append(f"Price>200SMA: {'✓' if price_above_200sma else '✗'}")

        if golden_cross:
            status_parts.append("Golden Cross Detected")
        elif death_cross:
            status_parts.append("Death Cross Detected")

        return f"## Macro Trend Analysis (365D):\n{' | '.join(status_parts)}"

    def _format_weekly_macro_section(self, weekly_macro: dict) -> str:

        if not weekly_macro:
            return ""

        trend = weekly_macro.get("trend_direction", "Neutral")
        confidence = weekly_macro.get("confidence_score", 0)
        cycle_phase = weekly_macro.get("cycle_phase")
        distance = weekly_macro.get("distance_from_200w_sma_pct")

        lines = ["## Weekly Macro Trend (200W SMA Analysis):"]
        lines.append(f"  • Overall Trend: **{trend}** (Confidence: {confidence}%)")

        if cycle_phase:
            lines.append(f"  • Market Cycle Phase: {cycle_phase}")
        if distance is not None:
            lines.append(f"  • Distance from 200W SMA: {distance:+.1f}%")
        if weekly_macro.get("price_above_200w_sma") is not None:
            status = "Above" if weekly_macro["price_above_200w_sma"] else "Below"
            lines.append(f"  • Price vs 200W SMA: {status}")

        if weekly_macro.get("golden_cross"):
            weeks = weekly_macro.get("golden_cross_weeks_ago", 0)
            date = weekly_macro.get("golden_cross_date", "N/A")
            lines.append(f"  • Golden Cross: {weeks} weeks ago ({date})")
        if weekly_macro.get("death_cross"):
            weeks = weekly_macro.get("death_cross_weeks_ago", 0)
            date = weekly_macro.get("death_cross_date", "N/A")
            lines.append(f"  • Death Cross: {weeks} weeks ago ({date})")

        if weekly_macro.get("multi_year_trend"):
            mt = weekly_macro["multi_year_trend"]
            years = mt.get("years_analyzed", 0)
            pct = mt.get("price_change_pct", 0)
            start = mt.get("start_date", "N/A")
            end = mt.get("end_date", "N/A")
            lines.append(f"  • {years:.1f}-Year Trend: {pct:+.1f}% ({start} to {end})")

        return "\n".join(lines)
