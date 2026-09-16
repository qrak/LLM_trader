"""View-model builders for the brain dashboard.

Every function here is pure presentation/derivation: it reads persisted analysis
files or in-memory summaries and returns dashboard-shaped dicts. The BrainRouter
owns the HTTP surface and delegates to these builders.
"""

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
    """Helper to read JSON file synchronously for offloading to a thread."""
    if not file_path.exists():
        return None
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def extract_persisted_technical_data(data: dict[str, Any]) -> dict[str, Any]:
    """Return persisted indicator values from previous response data."""
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
    """Current ATR% from the latest persisted analysis (drift flags in panel match factors).

    Mirrors market_conditions_extractor: prefers ``atr_percent``, accepts ``atr_percentage``.
    ``read_json_file`` yields None for a missing file — that means "no data yet", not an error.
    """
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
    """Return stored distance percent or derive it from entry and target prices."""
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
    """Truncate to `limit` characters, marking the cut with an ellipsis."""
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def excerpt_text(text: str | None, limit: int = 400) -> str:
    """Return a compact single-line excerpt for dashboard display."""
    if not text:
        return ""
    cleaned = re.sub(r"```json[\s\S]*?```", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return _clip(cleaned, limit)


def short_label(text: str | None, n: int = 28) -> str:
    """Truncate a node label for readable graph display."""
    return _clip((text or "").strip(), n)


def build_decision_synopsis(
    *,
    now: dict[str, Any],
    position: dict[str, Any],
    memory: dict[str, Any],
    journal: dict[str, Any],
    last_decision: dict[str, Any],
) -> str:
    """Compose a deterministic multi-sentence situational summary."""
    parts: list[str] = []
    if position.get("has_position"):
        direction = position.get("direction", "?")
        symbol = position.get("symbol", "?")
        entry = position.get("entry_price")
        conf = position.get("confidence")
        entry_txt = f" from {entry}" if entry is not None else ""
        conf_txt = f" (entry confidence {conf})" if conf is not None else ""
        parts.append(f"Open {direction} on {symbol}{entry_txt}{conf_txt}.")
    else:
        parts.append("No open position (flat).")

    action = now.get("action") or last_decision.get("signal") or "--"
    confidence = now.get("confidence")
    if confidence in (None, "--"):
        confidence = last_decision.get("confidence")
    trend = now.get("trend") or "--"
    conf_txt = f" at {confidence}% confidence" if confidence not in (None, "--") else ""
    parts.append(f"Latest model signal is {action}{conf_txt} with market trend {trend}.")

    ctx = memory.get("current_context")
    if ctx:
        parts.append(f"Memory is searching similar history under: {ctx}.")

    top_rules = memory.get("top_rules") or []
    if top_rules:
        rule0 = top_rules[0]
        rtype = rule0.get("rule_type") or "rule"
        rtext = (rule0.get("rule_text") or "")[:160]
        if rtext:
            parts.append(f"Top active {rtype}: {rtext}")

    items = journal.get("items") or []
    if items:
        lesson = (items[0].get("lesson_learned") or "").strip()
        verdict = items[0].get("verdict") or "lesson"
        if lesson:
            parts.append(f"Latest journal ({verdict}): {lesson[:180]}")

    blocked_count = (memory.get("blocked") or {}).get("blocked_count") or 0
    if blocked_count:
        parts.append(f"System blocked {blocked_count} trade attempt(s) recently (friction).")

    return " ".join(parts)


def build_decision_synopsis_data(
    *,
    now: dict[str, Any],
    last_decision: dict[str, Any],
    position: dict[str, Any],
    memory: dict[str, Any],
    journal: dict[str, Any],
) -> dict[str, Any]:
    """Extract structured synopsis fields for rich dashboard rendering."""
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


def build_decision_graph(
    *,
    now: dict[str, Any],
    last_decision: dict[str, Any],
    position: dict[str, Any],
    memory: dict[str, Any],
    journal: dict[str, Any],
) -> dict[str, Any]:
    """Build hierarchical multi-source decision graph nodes and edges."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    action = str(now.get("action") or last_decision.get("signal") or "--")
    conf = now.get("confidence")
    if conf in (None, "--"):
        conf = last_decision.get("confidence")
    conf_s = f"{conf}%" if conf not in (None, "--") else ""
    nodes.append(
        {
            "id": "hub_now",
            "type": "decision",
            "label": f"{action} {conf_s}".strip(),
            "level": 0,
            "group": "decision",
            "title": f"Signal {action} | conf {conf_s} | trend {now.get('trend')}",
            "data": {
                "action": action,
                "confidence": conf,
                "trend": now.get("trend"),
                "adx": now.get("adx"),
                "rsi": now.get("rsi"),
                "reasoning_excerpt": last_decision.get("reasoning_excerpt"),
                "timestamp": last_decision.get("timestamp") or now.get("timestamp"),
            },
        }
    )

    ctx = memory.get("current_context") or "No context"
    nodes.append(
        {
            "id": "hub_context",
            "type": "context",
            "label": short_label(str(ctx), 32),
            "level": 1,
            "group": "context",
            "title": str(ctx),
            "data": {"current_context": ctx},
        }
    )
    edges.append({"id": "e_now_ctx", "from": "hub_now", "to": "hub_context"})

    if position.get("has_position"):
        pos_label = f"{position.get('direction')} {short_label(str(position.get('symbol') or ''), 10)}"
        title = (
            f"Open {position.get('direction')} {position.get('symbol')} "
            f"entry={position.get('entry_price')}"
        )
    else:
        pos_label = "FLAT"
        title = "No open position"
    nodes.append(
        {
            "id": "hub_position",
            "type": "position",
            "label": pos_label,
            "level": 1,
            "group": "position",
            "title": title,
            "data": position,
        }
    )
    edges.append({"id": "e_now_pos", "from": "hub_now", "to": "hub_position"})

    experiences = memory.get("top_experiences") or []
    experience_count = memory.get("experience_count")
    if experience_count is None:
        experience_count = len(experiences)
    nodes.append(
        {
            "id": "hub_memory",
            "type": "hub",
            "label": f"Memory ({experience_count})",
            "level": 1,
            "group": "memory_hub",
            "title": "Similar vector experiences",
            "data": {"experience_count": experience_count},
        }
    )
    edges.append({"id": "e_ctx_mem", "from": "hub_context", "to": "hub_memory"})
    for i, exp in enumerate(experiences):
        nid = f"exp_{i}"
        outcome = str(exp.get("outcome") or "?")
        pnl = exp.get("pnl_pct")
        pnl_s = f"{pnl:+.1f}%" if isinstance(pnl, (int, float)) else ""
        direction = exp.get("direction") or ""
        group = "experience"
        if outcome.upper() == "WIN":
            group = "experience_win"
        elif outcome.upper() == "LOSS":
            group = "experience_loss"
        nodes.append(
            {
                "id": nid,
                "type": "experience",
                "label": short_label(f"{outcome} {pnl_s} {direction}", 24),
                "level": 2,
                "group": group,
                "title": f"sim={exp.get('similarity')} | {exp.get('document_excerpt') or ''}",
                "data": exp,
            }
        )
        edges.append({"id": f"e_mem_{i}", "from": "hub_memory", "to": nid})

    rules = memory.get("top_rules") or []
    rule_count = memory.get("rule_count")
    if rule_count is None:
        rule_count = len(rules)
    nodes.append(
        {
            "id": "hub_rules",
            "type": "hub",
            "label": f"Rules ({rule_count})",
            "level": 1,
            "group": "rules_hub",
            "title": "Active semantic rules",
            "data": {},
        }
    )
    edges.append({"id": "e_now_rules", "from": "hub_now", "to": "hub_rules"})
    for i, rule in enumerate(rules):
        nid = f"rule_{i}"
        rtype = rule.get("rule_type") or "rule"
        nodes.append(
            {
                "id": nid,
                "type": "rule",
                "label": short_label(f"{rtype}: {rule.get('rule_text') or ''}", 28),
                "level": 2,
                "group": f"rule_{rtype}",
                "title": rule.get("rule_text") or "",
                "data": rule,
            }
        )
        edges.append({"id": f"e_rule_{i}", "from": "hub_rules", "to": nid})

    items = journal.get("items") or []
    journal_count = journal.get("count")
    if journal_count is None:
        journal_count = len(items)
    nodes.append(
        {
            "id": "hub_journal",
            "type": "hub",
            "label": f"Journal ({journal_count})",
            "level": 1,
            "group": "journal_hub",
            "title": "Recent post-mortem lessons",
            "data": {},
        }
    )
    edges.append({"id": "e_now_journal", "from": "hub_now", "to": "hub_journal"})
    for i, pm in enumerate(items):
        nid = f"pm_{i}"
        nodes.append(
            {
                "id": nid,
                "type": "journal",
                "label": short_label(f"{pm.get('verdict') or 'lesson'} {pm.get('symbol') or ''}", 28),
                "level": 2,
                "group": "journal",
                "title": pm.get("lesson_learned") or "",
                "data": pm,
            }
        )
        edges.append({"id": f"e_pm_{i}", "from": "hub_journal", "to": nid})

    blocked = memory.get("blocked") or {}
    b_items = blocked.get("items") or []
    b_count = blocked.get("blocked_count")
    if b_count is None:
        b_count = len(b_items)
    if b_count:
        nodes.append(
            {
                "id": "hub_blocked",
                "type": "hub",
                "label": f"Blocked ({b_count})",
                "level": 1,
                "group": "blocked_hub",
                "title": "System-rejected / friction events",
                "data": {"blocked_count": b_count},
            }
        )
        edges.append({"id": "e_now_blocked", "from": "hub_now", "to": "hub_blocked"})
        for i, ev in enumerate(b_items):
            nid = f"blk_{i}"
            label_src = ev.get("guard_type") or ev.get("reason") or "blocked"
            nodes.append(
                {
                    "id": nid,
                    "type": "blocked",
                    "label": short_label(str(label_src), 24),
                    "level": 2,
                    "group": "blocked",
                    "title": str(ev),
                    "data": ev if isinstance(ev, dict) else {"value": ev},
                }
            )
            edges.append({"id": f"e_blk_{i}", "from": "hub_blocked", "to": nid})

    return {"nodes": nodes, "edges": edges}


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
    """Helper to extract market status from previous_response data."""
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
    """Build rich context query string from current market conditions.

    Reads ``technical_data`` from ``previous_response.json`` (persisted by the
    analysis engine after each run) and applies the same indicator classification
    logic used during live trading so that similarity queries are semantically
    identical to the documents stored in vector memory.

    Returns: tuple of (display_context, query_document). display_context is the
        categorical string for display; query_document is the enriched string
        for embedding search. Both are empty strings on failure.
    """
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
