"""Pure-function indicator classification utilities.

Converts raw technical indicator values into categorical labels used
by TradingBrainService to build context query strings. These functions
are the single source of truth for classification thresholds, shared
between the AnalysisEngine (live trading) and the dashboard router.
"""
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from src.config.protocol import ConfigProtocol


EXIT_EXECUTION_UNKNOWN = "unknown"
EXIT_EXECUTION_KEYS = (
    "stop_loss_type",
    "stop_loss_check_interval",
    "take_profit_type",
    "take_profit_check_interval",
)
EXIT_EXECUTION_TYPES = {"soft", "hard", EXIT_EXECUTION_UNKNOWN}


def _normalize_exit_execution_value(value: Any, default: str = EXIT_EXECUTION_UNKNOWN) -> str:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    return normalized or default


def build_exit_execution_context(
    stop_loss_type: Any = EXIT_EXECUTION_UNKNOWN,
    stop_loss_check_interval: Any = EXIT_EXECUTION_UNKNOWN,
    take_profit_type: Any = EXIT_EXECUTION_UNKNOWN,
    take_profit_check_interval: Any = EXIT_EXECUTION_UNKNOWN,
) -> Dict[str, str]:
    """Return normalized SL/TP execution settings for brain memory/query context."""
    stop_type = _normalize_exit_execution_value(stop_loss_type)
    take_profit_exit_type = _normalize_exit_execution_value(take_profit_type)
    if stop_type not in EXIT_EXECUTION_TYPES:
        stop_type = EXIT_EXECUTION_UNKNOWN
    if take_profit_exit_type not in EXIT_EXECUTION_TYPES:
        take_profit_exit_type = EXIT_EXECUTION_UNKNOWN

    return {
        "stop_loss_type": stop_type,
        "stop_loss_check_interval": _normalize_exit_execution_value(stop_loss_check_interval),
        "take_profit_type": take_profit_exit_type,
        "take_profit_check_interval": _normalize_exit_execution_value(take_profit_check_interval),
    }


def build_exit_execution_context_from_config(
    config: "ConfigProtocol",
    timeframe: str = EXIT_EXECUTION_UNKNOWN,
) -> Dict[str, str]:
    """Build risk-execution context from config attributes."""
    interval_default = timeframe or EXIT_EXECUTION_UNKNOWN
    return build_exit_execution_context(
        stop_loss_type=config.STOP_LOSS_TYPE,
        stop_loss_check_interval=config.STOP_LOSS_CHECK_INTERVAL or interval_default,
        take_profit_type=config.TAKE_PROFIT_TYPE,
        take_profit_check_interval=config.TAKE_PROFIT_CHECK_INTERVAL or interval_default,
    )


def build_exit_execution_context_from_position(position: Any) -> Dict[str, str]:
    """Build risk-execution context from a position entry snapshot."""
    return build_exit_execution_context(
        stop_loss_type=position.stop_loss_type_at_entry,
        stop_loss_check_interval=position.stop_loss_check_interval_at_entry,
        take_profit_type=position.take_profit_type_at_entry,
        take_profit_check_interval=position.take_profit_check_interval_at_entry,
    )


def format_exit_execution_context(
    exit_execution_context: Optional[Dict[str, Any]] = None,
    *,
    include_unknown: bool = False,
) -> str:
    """Format SL/TP execution settings for vector documents and query strings."""
    raw_context = exit_execution_context or {}
    context = build_exit_execution_context(
        stop_loss_type=raw_context.get("stop_loss_type", EXIT_EXECUTION_UNKNOWN),
        stop_loss_check_interval=raw_context.get("stop_loss_check_interval", EXIT_EXECUTION_UNKNOWN),
        take_profit_type=raw_context.get("take_profit_type", EXIT_EXECUTION_UNKNOWN),
        take_profit_check_interval=raw_context.get("take_profit_check_interval", EXIT_EXECUTION_UNKNOWN),
    )
    if not include_unknown and all(value == EXIT_EXECUTION_UNKNOWN for value in context.values()):
        return ""
    return (
        f"Exit Execution: SL {context['stop_loss_type']}/{context['stop_loss_check_interval']} | "
        f"TP {context['take_profit_type']}/{context['take_profit_check_interval']}"
    )


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
        imbalance = order_book.get("imbalance", 0)
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
    exit_execution_context: Optional[Dict[str, Any]] = None,
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
        exit_execution_context: Optional SL/TP execution settings snapshot.

    Returns:
        Space-and-plus-separated categorical context string.
    """
    trend_direction = classify_trend_direction(technical_data)
    adx = technical_data.get("adx", 0.0)
    volatility_level = classify_volatility_level(technical_data)
    rsi_level = classify_rsi_level(technical_data)
    macd_signal = classify_macd_signal(technical_data)
    volume_state = classify_volume_state(technical_data)
    bb_position = classify_bb_position(technical_data, current_price)
    market_sentiment = classify_market_sentiment(sentiment_data)
    order_book_bias = classify_order_book_bias(microstructure_data)

    return build_context_string_from_classified_values(
        trend_direction=trend_direction,
        adx=adx,
        volatility_level=volatility_level,
        rsi_level=rsi_level,
        macd_signal=macd_signal,
        volume_state=volume_state,
        bb_position=bb_position,
        is_weekend=is_weekend,
        market_sentiment=market_sentiment,
        order_book_bias=order_book_bias,
        exit_execution_context=exit_execution_context,
    )


def build_context_string_from_classified_values(
    trend_direction: str,
    adx: float,
    volatility_level: str,
    rsi_level: str,
    macd_signal: str,
    volume_state: str,
    bb_position: str,
    is_weekend: bool = False,
    market_sentiment: str = "NEUTRAL",
    order_book_bias: str = "BALANCED",
    exit_execution_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the rich context string from already classified market values."""
    adx_label = classify_adx_label(adx)

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
    exit_execution_text = format_exit_execution_context(exit_execution_context)
    if exit_execution_text:
        context_parts.append(exit_execution_text)

    return " + ".join(context_parts)


def build_query_document_from_technical_data(
    technical_data: Dict[str, Any],
    current_price: Optional[float] = None,
    sentiment_data: Optional[Dict[str, Any]] = None,
    microstructure_data: Optional[Dict[str, Any]] = None,
    is_weekend: bool = False,
    exit_execution_context: Optional[Dict[str, Any]] = None,
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
        exit_execution_context: Optional SL/TP execution settings snapshot.

    Returns:
        Enriched query string with Indicators and Structure lines.
    """
    adx = technical_data.get("adx", 0.0)
    rsi = technical_data.get("rsi", 50.0)
    rsi_level = classify_rsi_level(technical_data)
    volatility_level = classify_volatility_level(technical_data)
    macd_signal = classify_macd_signal(technical_data)
    bb_position = classify_bb_position(technical_data, current_price)
    market_sentiment = classify_market_sentiment(sentiment_data)
    order_book_bias = classify_order_book_bias(microstructure_data)

    return build_query_document_from_classified_values(
        trend_direction=classify_trend_direction(technical_data),
        adx=adx,
        rsi=rsi,
        volatility_level=volatility_level,
        rsi_level=rsi_level,
        macd_signal=macd_signal,
        volume_state=classify_volume_state(technical_data),
        bb_position=bb_position,
        is_weekend=is_weekend,
        market_sentiment=market_sentiment,
        order_book_bias=order_book_bias,
        exit_execution_context=exit_execution_context,
    )


def build_query_document_from_classified_values(
    trend_direction: str,
    adx: float,
    rsi: float,
    volatility_level: str,
    rsi_level: str,
    macd_signal: str,
    volume_state: str,
    bb_position: str,
    is_weekend: bool = False,
    market_sentiment: str = "NEUTRAL",
    order_book_bias: str = "BALANCED",
    exit_execution_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build an enriched vector query document from already classified values."""
    context_str = build_context_string_from_classified_values(
        trend_direction=trend_direction,
        adx=adx,
        volatility_level=volatility_level,
        rsi_level=rsi_level,
        macd_signal=macd_signal,
        volume_state=volume_state,
        bb_position=bb_position,
        is_weekend=is_weekend,
        market_sentiment=market_sentiment,
        order_book_bias=order_book_bias,
        exit_execution_context=exit_execution_context,
    )
    adx_label = classify_adx_label(adx)

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
    exit_execution_text = format_exit_execution_context(exit_execution_context)
    if exit_execution_text:
        structure_parts.append(exit_execution_text)

    lines = [context_str, f"Indicators: {' | '.join(indicator_parts)}"]
    if structure_parts:
        lines.append(f"Structure: {' | '.join(structure_parts)}")

    return " ".join(lines)
