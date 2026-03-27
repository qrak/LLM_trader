"""Pure-function indicator classification utilities.

Converts raw technical indicator values into categorical labels used
by TradingBrainService to build context query strings. These functions
are the single source of truth for classification thresholds, shared
between the AnalysisEngine (live trading) and the dashboard router.
"""
from typing import Any, Dict, Optional


def classify_trend_direction(technical_data: Dict[str, Any]) -> str:
    """Classify trend direction from +DI/-DI crossover."""
    di_plus = technical_data.get("plus_di", 0.0)
    di_minus = technical_data.get("minus_di", 0.0)
    if di_plus > di_minus + 5:
        return "BULLISH"
    if di_minus > di_plus + 5:
        return "BEARISH"
    return "NEUTRAL"


def classify_volatility_level(technical_data: Dict[str, Any]) -> str:
    """Classify volatility from ATR as percentage of price."""
    atr_pct = technical_data.get("atr_percent", 2.0)
    if atr_pct > 3.0:
        return "HIGH"
    if atr_pct < 1.5:
        return "LOW"
    return "MEDIUM"


def classify_adx_label(adx: float) -> str:
    """Classify ADX value into trend strength label.

    Single source of truth for ADX labeling used across brain, dashboard, and vector memory.
    """
    if adx >= 25:
        return "High ADX"
    if adx < 20:
        return "Low ADX"
    return "Medium ADX"


def classify_rsi_label(rsi: float) -> str:
    """Classify RSI value into momentum zone label.

    Single source of truth for RSI labeling used across brain, dashboard, and vector memory.
    """
    if rsi >= 70:
        return "OVERBOUGHT"
    if rsi >= 60:
        return "STRONG"
    if rsi <= 30:
        return "OVERSOLD"
    if rsi <= 40:
        return "WEAK"
    return "NEUTRAL"


def classify_rsi_level(technical_data: Dict[str, Any]) -> str:
    """Classify RSI into momentum zones from technical data dict."""
    rsi = technical_data.get("rsi", 50.0)
    return classify_rsi_label(rsi)


def classify_macd_signal(technical_data: Dict[str, Any]) -> str:
    """Classify MACD as BULLISH/BEARISH based on line vs signal line."""
    macd_line = technical_data.get("macd_line")
    macd_signal_line = technical_data.get("macd_signal")
    if macd_line is not None and macd_signal_line is not None:
        if macd_line > macd_signal_line:
            return "BULLISH"
        if macd_line < macd_signal_line:
            return "BEARISH"
    return "NEUTRAL"


def classify_volume_state(technical_data: Dict[str, Any]) -> str:
    """Classify volume trend from On-Balance Volume slope."""
    obv_slope = technical_data.get("obv_slope", 0.0)
    if obv_slope > 0.5:
        return "ACCUMULATION"
    if obv_slope < -0.5:
        return "DISTRIBUTION"
    return "NORMAL"


def classify_bb_position(
    technical_data: Dict[str, Any],
    current_price: Optional[float],
) -> str:
    """Classify price position relative to Bollinger Bands."""
    bb_upper = technical_data.get("bb_upper")
    bb_lower = technical_data.get("bb_lower")
    if bb_upper is not None and bb_lower is not None and current_price:
        if current_price >= bb_upper * 0.99:
            return "UPPER"
        if current_price <= bb_lower * 1.01:
            return "LOWER"
    return "MIDDLE"


def classify_market_sentiment(sentiment_data: Optional[Dict[str, Any]]) -> str:
    """Classify Fear & Greed index into sentiment zones."""
    if sentiment_data:
        fear_greed = sentiment_data.get("fear_greed_index", 50)
        if isinstance(fear_greed, (int, float)):
            if fear_greed <= 25:
                return "EXTREME_FEAR"
            if fear_greed <= 45:
                return "FEAR"
            if fear_greed >= 75:
                return "EXTREME_GREED"
            if fear_greed >= 55:
                return "GREED"
    return "NEUTRAL"


def classify_order_book_bias(microstructure_data: Optional[Dict[str, Any]]) -> str:
    """Classify order book pressure from bid/ask imbalance."""
    if microstructure_data:
        order_book = microstructure_data.get("order_book", {})
        imbalance = order_book.get("imbalance", 0) if isinstance(order_book, dict) else 0
        if imbalance > 0.1:
            return "BUY_PRESSURE"
        if imbalance < -0.1:
            return "SELL_PRESSURE"
    return "BALANCED"


def build_context_string_from_technical_data(
    technical_data: Dict[str, Any],
    current_price: Optional[float] = None,
    sentiment_data: Optional[Dict[str, Any]] = None,
    microstructure_data: Optional[Dict[str, Any]] = None,
    is_weekend: bool = False,
) -> str:
    """Build the rich context string from raw technical indicators.

    Produces the exact same format as TradingBrainService._build_rich_context_string
    so that dashboard similarity queries are semantically identical to those
    issued during live trading.

    Args:
        technical_data: Dict of raw indicator values (rsi, adx, macd_line …).
        current_price: Optional current price for Bollinger Band positioning.
        sentiment_data: Optional sentiment dict with ``fear_greed_index`` key.
        microstructure_data: Optional order book microstructure dict.
        is_weekend: Whether the current day is Saturday or Sunday.

    Returns:
        Space-and-plus-separated categorical context string.
    """
    trend_direction = classify_trend_direction(technical_data)
    adx = technical_data.get("adx", 0.0)
    adx_label = classify_adx_label(adx)
    volatility_level = classify_volatility_level(technical_data)
    rsi_level = classify_rsi_level(technical_data)
    macd_signal = classify_macd_signal(technical_data)
    volume_state = classify_volume_state(technical_data)
    bb_position = classify_bb_position(technical_data, current_price)
    market_sentiment = classify_market_sentiment(sentiment_data)
    order_book_bias = classify_order_book_bias(microstructure_data)

    context_parts = [trend_direction, adx_label, f"{volatility_level} Volatility"]

    if rsi_level != "NEUTRAL":
        context_parts.append(f"RSI {rsi_level}")
    if macd_signal != "NEUTRAL":
        context_parts.append(f"MACD {macd_signal}")
    if volume_state != "NORMAL":
        context_parts.append(f"Volume {volume_state}")
    if bb_position != "MIDDLE":
        context_parts.append(f"Price at BB {bb_position}")
    if is_weekend:
        context_parts.append("Weekend Low Volume")
    if market_sentiment not in ("NEUTRAL", ""):
        context_parts.append(f"Sentiment {market_sentiment}")
    if order_book_bias not in ("BALANCED", ""):
        context_parts.append(f"OrderBook {order_book_bias}")

    return " + ".join(context_parts)


def build_query_document_from_technical_data(
    technical_data: Dict[str, Any],
    current_price: Optional[float] = None,
    sentiment_data: Optional[Dict[str, Any]] = None,
    microstructure_data: Optional[Dict[str, Any]] = None,
    is_weekend: bool = False,
) -> str:
    """Build an enriched query document for vector similarity search.

    Produces the same format as TradingBrainService._build_query_document so
    that dashboard similarity queries use the richer embedding format that
    mirrors stored experience documents.

    Args:
        technical_data: Dict of raw indicator values (rsi, adx, macd_line …).
        current_price: Optional current price for Bollinger Band positioning.
        sentiment_data: Optional sentiment dict with ``fear_greed_index`` key.
        microstructure_data: Optional order book microstructure dict.
        is_weekend: Whether the current day is Saturday or Sunday.

    Returns:
        Enriched query string with Indicators and Structure lines.
    """
    context_str = build_context_string_from_technical_data(
        technical_data=technical_data,
        current_price=current_price,
        sentiment_data=sentiment_data,
        microstructure_data=microstructure_data,
        is_weekend=is_weekend,
    )

    adx = technical_data.get("adx", 0.0)
    adx_label = classify_adx_label(adx)
    rsi = technical_data.get("rsi", 50.0)
    rsi_level = classify_rsi_level(technical_data)
    volatility_level = classify_volatility_level(technical_data)
    macd_signal = classify_macd_signal(technical_data)
    bb_position = classify_bb_position(technical_data, current_price)
    market_sentiment = classify_market_sentiment(sentiment_data)
    order_book_bias = classify_order_book_bias(microstructure_data)

    indicator_parts = [
        f"ADX={adx:.1f} ({adx_label})",
        f"RSI={rsi:.1f} ({rsi_level})",
        f"Vol={volatility_level}",
        f"MACD={macd_signal}",
        f"BB={bb_position}",
    ]
    structure_parts: list[str] = []
    if market_sentiment not in ("NEUTRAL", ""):
        structure_parts.append(f"Sentiment={market_sentiment}")
    if order_book_bias not in ("BALANCED", ""):
        structure_parts.append(f"OB={order_book_bias}")

    lines = [context_str, f"Indicators: {' | '.join(indicator_parts)}"]
    if structure_parts:
        lines.append(f"Structure: {' | '.join(structure_parts)}")

    return " ".join(lines)
