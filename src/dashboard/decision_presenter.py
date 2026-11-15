

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.indicator_classifier import (
    build_context_string_from_technical_data,
    build_exit_execution_context_from_config,
    build_query_document_from_technical_data,
    classify_adx_label,
    classify_trend_direction,
    format_exit_execution_context,
)

RULE_METADATA_FIELDS = frozenset((
    "rule_type",
    "direction",
    "regime",
    "adx_bucket",
    "win_rate",
    "loss_rate",
    "wins",
    "losses",
    "avg_pnl_pct",
    "profit_factor",
    "expectancy_pct",
    "failure_reason",
    "recommended_adjustment",
    "mistake_type",
    "entry_confidence",
    "failed_assumption",
    "dominant_close_reason",
    "dominant_exit_profile",
    "dominant_stop_loss_type",
    "dominant_stop_loss_interval",
    "dominant_take_profit_type",
    "dominant_take_profit_interval",
    "created_at",
    "last_validated_at",
    "last_contradicted_at",
    "age_days",
    "freshness_score",
    "freshness_label",
    "evidence_score",
    "final_score",
    "support_count",
    "validation_hit_count",
    "contradiction_count",
    "source_timeframe_minutes",
    "source_timeframe_bucket",
))


def read_json_file(file_path: Path) -> Any:

    if not file_path.exists():
        return None
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def extract_persisted_technical_data(data: dict[str, Any]) -> dict[str, Any]:

    technical_data = data.get("technical_data")
    if technical_data:
        return technical_data

    response = data.get("response", {})
    return {
        key: value
        for key, value in response.items()
        if key != "text_analysis"
    }


def extract_current_atr_percentage(config) -> float | None:

    prev_response_file = Path(config.DATA_DIR) / "trading" / "previous_response.json"
    try:
        data = read_json_file(prev_response_file)
        if data is None:
            return None
        technical_data = extract_persisted_technical_data(data)
        raw = technical_data.get("atr_percent", technical_data.get("atr_percentage"))
        return float(raw) if raw is not None else None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def distance_pct_or_fallback(stored_pct: float | None, entry_price: float, target_price: float) -> float:

    if stored_pct and stored_pct > 0:
        return stored_pct
    if entry_price <= 0:
        return 0.0
    return abs(target_price - entry_price) / entry_price


def format_execution_label(data: dict[str, Any], prefix: str) -> str:
    execution_type = data.get(f"{prefix}_type") or "unknown"
    check_interval = data.get(f"{prefix}_check_interval") or "unknown"
    return f"{execution_type} / {check_interval}"


def _clip(value: str, limit: int) -> str:

    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def excerpt_text(text: str | None, limit: int = 400) -> str:

    if not text:
        return ""
    cleaned = re.sub(r"```json[\s\S]*?```", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return _clip(cleaned, limit)


def build_decision_synopsis_data(
    *,
    now: dict[str, Any],
    last_decision: dict[str, Any],
    position: dict[str, Any],
    memory: dict[str, Any],
    journal: dict[str, Any],
) -> dict[str, Any]:

    action = now.get("action") or last_decision.get("signal") or "--"
    confidence = now.get("confidence")
    if confidence in (None, "--"):
        confidence = last_decision.get("confidence")
    trend = now.get("trend") or "--"

    top_rule = None
    top_rules = memory.get("top_rules") or []
    if top_rules:
        r0 = top_rules[0]
        top_rule = {
            "type": r0.get("rule_type") or "rule",
            "text": (r0.get("rule_text") or "")[:160],
        }

    latest_journal = None
    items = journal.get("items") or []
    if items:
        lesson = (items[0].get("lesson_learned") or "").strip()
        if lesson:
            latest_journal = {
                "verdict": items[0].get("verdict") or "lesson",
                "lesson": lesson[:180],
            }

    return {
        "has_position": bool(position.get("has_position")),
        "direction": position.get("direction"),
        "symbol": position.get("symbol"),
        "entry_price": position.get("entry_price"),
        "confidence": confidence,
        "action": action,
        "trend": trend,
        "current_context": memory.get("current_context"),
        "top_rule": top_rule,
        "latest_journal": latest_journal,
        "blocked_count": (memory.get("blocked") or {}).get("blocked_count") or 0,
    }


def build_risk_management(current: dict[str, Any], at_entry: dict[str, Any] | None = None) -> dict[str, Any]:
    entry = at_entry or current
    current_labels = {
        "stop_loss": format_execution_label(current, "stop_loss"),
        "take_profit": format_execution_label(current, "take_profit"),
    }
    entry_labels = {
        "stop_loss": format_execution_label(entry, "stop_loss"),
        "take_profit": format_execution_label(entry, "take_profit"),
    }
    return {
        "current": current,
        "at_entry": entry,
        "current_labels": current_labels,
        "at_entry_labels": entry_labels,
        "policy_changed": current != entry,
    }


def extract_market_status(data: dict[str, Any], unified_parser=None) -> dict[str, Any]:

    response = data.get("response", {})
    text = response.get("text_analysis", "")
    technical_data = extract_persisted_technical_data(data)
    status = {
        "trend": "NEUTRAL",
        "action": "--",
        "confidence": "--",
        "adx": response.get("adx"),
        "rsi": response.get("rsi")
    }

    if technical_data:
        status["adx"] = technical_data.get("adx", status["adx"])
        status["rsi"] = technical_data.get("rsi", status["rsi"])
        status["trend"] = classify_trend_direction(technical_data)

    parsed_analysis = unified_parser.extract_json_block(text, unwrap_key="analysis") if unified_parser else None

    if parsed_analysis:
        signal = parsed_analysis.get("signal")
        if signal:
            status["action"] = str(signal).upper()
        confidence_raw = parsed_analysis.get("confidence")
        if confidence_raw is not None:
            try:
                confidence_value = float(confidence_raw)
                status["confidence"] = int(confidence_value) if confidence_value == int(confidence_value) else confidence_value
            except (TypeError, ValueError):
                pass
    else:
        signal_match = re.search(r"\bSIGNAL\s*:\s*([A-Z_]+)\b", text, re.IGNORECASE)
        if signal_match:
            status["action"] = signal_match.group(1).upper()

        confidence_match = re.search(r"\bConfidence\s*:\s*(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
        if confidence_match:
            confidence_value = float(confidence_match.group(1))
            status["confidence"] = int(confidence_value) if confidence_value.is_integer() else confidence_value

    if status["trend"] == "NEUTRAL":
        if "BEARISH" in text.upper():
            status["trend"] = "BEARISH"
        elif "BULLISH" in text.upper():
            status["trend"] = "BULLISH"
    return status

def build_current_market_context(config, logger, unified_parser=None) -> tuple[str, str]:

    data_dir = config.DATA_DIR
    prev_response_file = Path(data_dir) / "trading" / "previous_response.json"
    if not prev_response_file.exists():
        return "", ""
    try:
        data = read_json_file(prev_response_file)
        technical_data = extract_persisted_technical_data(data)
        exit_execution_context = build_exit_execution_context_from_config(config, config.TIMEFRAME)
        if not technical_data:
            status = extract_market_status(data, unified_parser)
            adx = status["adx"] or 0
            adx_label = classify_adx_label(adx)
            fallback = f"{status['trend']} + {adx_label} + MEDIUM Volatility"
            exit_execution_text = format_exit_execution_context(exit_execution_context)
            if exit_execution_text:
                fallback = f"{fallback} + {exit_execution_text}"
            return fallback, fallback
        current_price: float | None = None
        response = data.get("response", {})
        current_price = response.get("current_price")
        sentiment_data: dict[str, Any] | None = data.get("sentiment")
        is_weekend = datetime.now(timezone.utc).weekday() >= 5
        shared_kwargs: dict[str, Any] = {
            "technical_data": technical_data,
            "current_price": current_price,
            "sentiment_data": sentiment_data,
            "is_weekend": is_weekend,
            "exit_execution_context": exit_execution_context,
        }
        display_context = build_context_string_from_technical_data(**shared_kwargs)
        query_document = build_query_document_from_technical_data(**shared_kwargs)
        return display_context, query_document
    except Exception:  # pylint: disable=broad-exception-caught
        logger.error("Failed to build market context", exc_info=True)  # noqa: G201
        return "", ""
