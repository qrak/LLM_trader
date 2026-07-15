"""MarketConditionsExtractor — extracts market conditions from analysis results.

Extracted from TradingStrategy to reduce SRP violation (was 1151 lines / 19 methods).
"""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from src.utils.indicator_classifier import (
    classify_bb_position,
    classify_macd_signal,
    classify_market_sentiment,
    classify_order_book_bias,
    classify_rsi_label,
    classify_volume_state,
    classify_volatility_level,
)
from .data_models import MarketConditions, Position

if TYPE_CHECKING:
    from src.logger.logger import Logger


class MarketConditionsExtractor:
    """Extracts market conditions, confluence factors, and prices from analysis results."""

    def __init__(self, logger: Logger) -> None:
        self.logger = logger

    def extract_price(self, result: dict) -> float:
        """Extract current price from analysis result."""
        if "current_price" in result:
            return float(result["current_price"])

        if "context" in result and result["context"] is not None:
            return float(result["context"].current_price)

        self.logger.warning("Could not extract price from result keys: %s", list(result.keys()))
        return 0.0

    def extract_market_conditions(self, result: dict) -> MarketConditions:
        """Extract market conditions from analysis result for brain learning."""
        conditions: dict[str, Any] = {}

        try:
            analysis = result.get("analysis", {})

            # Trend info
            trend = analysis.get("trend", {})
            if trend:
                conditions["trend_direction"] = trend.get("direction", "NEUTRAL")
                conditions["trend_strength"] = trend.get("strength_4h", trend.get("strength", 50))
                conditions["timeframe_alignment"] = trend.get("timeframe_alignment")

            # Technical data
            tech_data = result.get("technical_data", {})
            if tech_data:
                conditions["adx"] = tech_data.get("adx", 0)
                rsi_raw = tech_data.get("rsi", 50)
                try:
                    rsi_value = float(rsi_raw)
                except (TypeError, ValueError):
                    rsi_value = 50.0
                conditions["rsi"] = rsi_value
                conditions["rsi_level"] = classify_rsi_label(rsi_value)
                conditions["choppiness"] = tech_data.get("choppiness", None)

                macd = tech_data.get("macd", {})
                conditions["macd_signal"] = classify_macd_signal(tech_data)
                if conditions["macd_signal"] == "NEUTRAL" and macd:
                    conditions["macd_signal"] = macd.get("signal", "NEUTRAL")

                current_price = result.get("current_price")
                context_obj = result.get("context")
                if current_price is None and context_obj is not None:
                    current_price = context_obj.current_price
                conditions["bb_position"] = classify_bb_position(tech_data, current_price)
                bb = tech_data.get("bollinger_bands", {})
                if conditions["bb_position"] == "MIDDLE" and bb:
                    pct_b = bb.get("percent_b", 0.5)
                    if pct_b > 0.95:
                        conditions["bb_position"] = "UPPER"
                    elif pct_b < 0.05:
                        conditions["bb_position"] = "LOWER"

                conditions["volume_state"] = classify_volume_state(tech_data)
                vol_data = tech_data.get("volume", {})
                if conditions["volume_state"] == "NORMAL" and vol_data:
                    conditions["volume_state"] = vol_data.get("state", "NORMAL")

                conditions["atr"] = tech_data.get("atr", 0)
                atr_pct_raw = tech_data.get("atr_percent")
                if atr_pct_raw is None:
                    atr_pct_raw = tech_data.get("atr_percentage")
                try:
                    atr_pct = float(atr_pct_raw) if atr_pct_raw is not None else 2.0
                except (TypeError, ValueError):
                    atr_pct = 2.0
                conditions["atr_percentage"] = atr_pct
                conditions["volatility"] = classify_volatility_level({"atr_percent": atr_pct})

            context_obj = result.get("context")
            sentiment_data = result.get("sentiment")
            microstructure_data = result.get("market_microstructure")
            if context_obj is not None:
                if sentiment_data is None:
                    sentiment_data = context_obj.sentiment
                if microstructure_data is None:
                    microstructure_data = context_obj.market_microstructure

            conditions["market_sentiment"] = classify_market_sentiment(sentiment_data)
            conditions["fear_greed_index"] = sentiment_data.get("fear_greed_index", 50) if sentiment_data else 50
            conditions["order_book_bias"] = classify_order_book_bias(microstructure_data)

            # Fallback: extract trend direction from raw response signal
            raw_response = result.get("raw_response", "").lower()
            if not conditions.get("trend_direction"):
                signal_match = re.search(
                    r'signal["\s:]*\[?(BUY|SELL|HOLD|CLOSE)\b', raw_response, re.IGNORECASE
                )
                if signal_match:
                    signal_word = signal_match.group(1).upper()
                    if signal_word == "BUY":
                        conditions["trend_direction"] = "BULLISH"
                    elif signal_word == "SELL":
                        conditions["trend_direction"] = "BEARISH"
                    else:
                        conditions["trend_direction"] = "NEUTRAL"
                else:
                    bullish_hits = len(re.findall(r'\b(bullish|uptrend)\b', raw_response))
                    bearish_hits = len(re.findall(r'\b(bearish|downtrend)\b', raw_response))
                    if bullish_hits > bearish_hits:
                        conditions["trend_direction"] = "BULLISH"
                    elif bearish_hits > bullish_hits:
                        conditions["trend_direction"] = "BEARISH"
                    else:
                        conditions["trend_direction"] = "NEUTRAL"
        except Exception as e:
            self.logger.warning("Could not extract market conditions: %s", e)

        return MarketConditions(
            trend_direction=conditions.get("trend_direction", "NEUTRAL"),
            adx=float(conditions.get("adx", 0.0)),
            rsi=float(conditions.get("rsi", 50.0)),
            rsi_level=conditions.get("rsi_level", "NEUTRAL"),
            volatility=conditions.get("volatility", "MEDIUM"),
            atr=float(conditions.get("atr", 0.0)),
            atr_percentage=float(conditions.get("atr_percentage", 0.0)),
            macd_signal=conditions.get("macd_signal", "NEUTRAL"),
            bb_position=conditions.get("bb_position", "MIDDLE"),
            volume_state=conditions.get("volume_state", "NORMAL"),
            is_weekend=bool(conditions.get("is_weekend", False)),
            market_sentiment=conditions.get("market_sentiment", "NEUTRAL"),
            order_book_bias=conditions.get("order_book_bias", "BALANCED"),
            fear_greed_index=int(conditions.get("fear_greed_index", 50)),
            trend_strength=float(conditions.get("trend_strength", 0.0)),
            timeframe_alignment=conditions.get("timeframe_alignment"),
            choppiness=conditions.get("choppiness"),
        )

    def extract_confluence_factors(self, result: dict) -> tuple:
        """Extract confluence factors from analysis result for brain learning.

        Returns: tuple of (factor_name, score) pairs
        """
        factors = []
        try:
            analysis = result.get("analysis", {})
            confluence_factors = analysis.get("confluence_factors", {})
            if confluence_factors:
                for factor_name, score in confluence_factors.items():
                    try:
                        score_value = float(score)
                        if 0 <= score_value <= 100:
                            factors.append((factor_name, score_value))
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            self.logger.warning("Could not extract confluence factors: %s", e)
        return tuple(factors)

    @staticmethod
    def build_conditions_from_position(position: Position) -> MarketConditions:
        """Reconstruct market conditions from Position's stored entry fields.

        Used when closing via SL/TP hit where no fresh analysis is available.
        """
        rsi = position.rsi_at_entry
        rsi_level = classify_rsi_label(rsi)

        return MarketConditions(
            trend_direction=position.trend_direction_at_entry,
            adx=position.adx_at_entry,
            rsi=rsi,
            rsi_level=rsi_level,
            volatility=position.volatility_level,
            macd_signal=position.macd_signal_at_entry,
            bb_position=position.bb_position_at_entry,
            volume_state=position.volume_state_at_entry,
            market_sentiment=position.market_sentiment_at_entry,
            order_book_bias=position.order_book_bias_at_entry,
        )
