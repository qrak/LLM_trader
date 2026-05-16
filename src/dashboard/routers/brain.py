"""Router for brain state, memory vectors, and positions."""
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request

from src.utils.indicator_classifier import (
    build_exit_execution_context,
    build_exit_execution_context_from_config,
    build_exit_execution_context_from_position,
    build_context_string_from_technical_data,
    build_query_document_from_technical_data,
    classify_adx_label,
    classify_trend_direction,
    format_exit_execution_context,
)


RULE_METADATA_FIELDS = (
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
)


def _read_json_file(file_path: Path) -> Any:
    """Helper to read JSON file synchronously for offloading to a thread."""
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_persisted_technical_data(data: dict[str, Any]) -> dict[str, Any]:
    """Return persisted indicator values from new or legacy previous_response shapes."""
    technical_data = data.get("technical_data")
    if technical_data:
        return technical_data

    response = data.get("response", {})
    return {
        key: value
        for key, value in response.items()
        if key != "text_analysis"
    }


def _distance_pct_or_fallback(stored_pct: float | None, entry_price: float, target_price: float) -> float:
    """Return stored distance percent or derive it from entry and target prices."""
    if stored_pct and stored_pct > 0:
        return stored_pct
    if entry_price <= 0:
        return 0.0
    return abs(target_price - entry_price) / entry_price


def _format_execution_label(data: dict[str, Any], prefix: str) -> str:
    execution_type = data.get(f"{prefix}_type") or "unknown"
    check_interval = data.get(f"{prefix}_check_interval") or "unknown"
    return f"{execution_type} / {check_interval}"


def _build_risk_management(current: dict[str, Any], at_entry: dict[str, Any] | None = None) -> dict[str, Any]:
    entry = at_entry or current
    current_labels = {
        "stop_loss": _format_execution_label(current, "stop_loss"),
        "take_profit": _format_execution_label(current, "take_profit"),
    }
    entry_labels = {
        "stop_loss": _format_execution_label(entry, "stop_loss"),
        "take_profit": _format_execution_label(entry, "take_profit"),
    }
    return {
        "current": current,
        "at_entry": entry,
        "current_labels": current_labels,
        "at_entry_labels": entry_labels,
        "policy_changed": current != entry,
    }


def _extract_market_status(data: dict[str, Any], unified_parser=None) -> dict[str, Any]:
    """Helper to extract market status from previous_response data."""
    response = data.get("response", {})
    text = response.get("text_analysis", "")
    technical_data = _extract_persisted_technical_data(data)
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

    parsed_analysis = None
    if unified_parser:
        try:
            parsed_analysis = unified_parser.extract_json_block(text, unwrap_key="analysis")
        except Exception:
            parsed_analysis = None

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
        signal_match = re.search(r'\bSIGNAL\s*:\s*([A-Z_]+)\b', text, re.IGNORECASE)
        if signal_match:
            status["action"] = signal_match.group(1).upper()

        confidence_match = re.search(r'\bConfidence\s*:\s*(\d+(?:\.\d+)?)\s*%', text, re.IGNORECASE)
        if confidence_match:
            confidence_value = float(confidence_match.group(1))
            status["confidence"] = int(confidence_value) if confidence_value.is_integer() else confidence_value

    if status["trend"] == "NEUTRAL":
        if "BEARISH" in text.upper():
            status["trend"] = "BEARISH"
        elif "BULLISH" in text.upper():
            status["trend"] = "BULLISH"
    return status

def _build_current_market_context(config, logger, unified_parser=None) -> tuple[str, str]:
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
        data = _read_json_file(prev_response_file)
        technical_data = _extract_persisted_technical_data(data)
        exit_execution_context = build_exit_execution_context_from_config(config, config.TIMEFRAME)
        if not technical_data:
            status = _extract_market_status(data, unified_parser)
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
        is_weekend = datetime.now().weekday() >= 5
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
        logger.error("Failed to build market context", exc_info=True)
        return "", ""


class BrainRouter:
    """Handles endpoints for the trading brain status, rules, and memory."""
    UNKNOWN_EXIT_PROFILE = "SL unknown/unknown | TP unknown/unknown"

    # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-instance-attributes
    def __init__(self, config, logger, dashboard_state, vector_memory, unified_parser, persistence, exchange_manager):
        self.router = APIRouter(prefix="/api/brain", tags=["brain"])
        self.config = config
        self.logger = logger
        self.dashboard_state = dashboard_state
        self.vector_memory = vector_memory
        self.unified_parser = unified_parser
        self.persistence = persistence
        self.exchange_manager = exchange_manager

        self.router.add_api_route("/status", self.get_brain_status, methods=["GET"])
        self.router.add_api_route("/memory", self.get_vector_memory, methods=["GET"])
        self.router.add_api_route("/rules", self.get_active_rules, methods=["GET"])
        self.router.add_api_route("/vectors", self.get_vector_details, methods=["GET"])
        self.router.add_api_route("/position", self.get_current_position, methods=["GET"])
        self.router.add_api_route("/refresh-price", self.refresh_current_price, methods=["GET"])
        self.router.add_api_route("/blocked-trades", self.get_blocked_trades, methods=["GET"])
        self.router.add_api_route("/lifecycle", self.get_brain_lifecycle, methods=["GET"])
        self.router.add_api_route("/refresh", self.refresh_brain_state, methods=["POST"])

    def _resolve_rule_exit_profile(self, metadata: dict[str, Any]) -> str:
        """Resolve semantic-rule exit profile using stored metadata plus config fallback."""
        timeframe = self.config.TIMEFRAME
        default_context = build_exit_execution_context_from_config(self.config, timeframe)
        context = build_exit_execution_context(
            stop_loss_type=metadata.get("dominant_stop_loss_type") or metadata.get("stop_loss_type"),
            stop_loss_check_interval=(
                metadata.get("dominant_stop_loss_interval") or metadata.get("stop_loss_check_interval")
            ),
            take_profit_type=metadata.get("dominant_take_profit_type") or metadata.get("take_profit_type"),
            take_profit_check_interval=(
                metadata.get("dominant_take_profit_interval") or metadata.get("take_profit_check_interval")
            ),
        )
        resolved = context.with_defaults(default_context)
        return format_exit_execution_context(resolved, include_unknown=True).removeprefix("Exit Execution: ")

    def _render_rule_text(self, rule_text: str, metadata: dict[str, Any]) -> str:
        """Return dashboard rule text with legacy unknown exit profile corrected."""
        if self.UNKNOWN_EXIT_PROFILE not in rule_text:
            return rule_text
        replacement_profile = self._resolve_rule_exit_profile(metadata)
        if replacement_profile == self.UNKNOWN_EXIT_PROFILE:
            return rule_text
        return rule_text.replace(self.UNKNOWN_EXIT_PROFILE, replacement_profile)

    async def get_brain_status(self) -> dict[str, Any]:
        """Get the current thought process/status of the brain."""
        cached = self.dashboard_state.get_cached("brain_status", ttl_seconds=30.0)
        if cached:
            result = dict(cached)
            result["brain_lifecycle"] = self.dashboard_state.get_brain_lifecycle()
            return result
        try:
            data_dir = self.config.DATA_DIR
        except AttributeError:
            data_dir = "data"
        try:
            timeframe = self.config.TIMEFRAME
        except AttributeError:
            timeframe = "unknown"
        prev_response_file = Path(data_dir) / "trading" / "previous_response.json"
        stats_file = Path(data_dir) / "trading" / "statistics.json"
        status = {
            "status": "active",
            "trend": "--",
            "confidence": "--",
            "action": "--",
            "adx": None,
            "rsi": None,
            "exit_management": build_exit_execution_context_from_config(self.config, timeframe).to_dict(),
            "brain_lifecycle": self.dashboard_state.get_brain_lifecycle(),
        }
        try:
            if prev_response_file.exists():
                prev_data = await asyncio.to_thread(_read_json_file, prev_response_file)
                extracted = _extract_market_status(prev_data, self.unified_parser)
                status.update(extracted)
        except Exception:
            self.logger.error("Failed to load brain status", exc_info=True)

        try:
            if stats_file.exists():
                stats = await asyncio.to_thread(_read_json_file, stats_file)
                status.update({
                    "total_trades": stats.get("total_trades", 0),
                    "win_rate": stats.get("win_rate", 0),
                    "current_capital": stats.get("current_capital", 0)
                })
        except Exception:
            self.logger.error("Failed to load statistics", exc_info=True)
        self.dashboard_state.set_cached("brain_status", status)
        return status

    async def get_brain_lifecycle(self) -> dict[str, Any]:
        """Get current brain rebuild lifecycle state."""
        return self.dashboard_state.get_brain_lifecycle()

    async def refresh_brain_state(self) -> dict[str, Any]:
        """Invalidate brain-bound dashboard caches and notify clients to refetch."""
        self.dashboard_state.invalidate_brain_caches()
        await self.dashboard_state.broadcast_brain_state_updated("Brain dashboard caches refreshed")
        return {
            "success": True,
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "lifecycle": self.dashboard_state.get_brain_lifecycle(),
        }

    async def get_vector_memory(self, limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
        """Get recent vector memories (synapses)."""
        cached = self.dashboard_state.get_cached(f"memory_{limit}", ttl_seconds=30.0)
        if cached:
            return cached
        data_dir = self.config.DATA_DIR
        result = {
            "experience_count": 0,
            "trades": [],
            "stats": {}
        }
        if self.vector_memory:
            result["experience_count"] = self.vector_memory.experience_count
            result["stats"] = self.vector_memory.compute_confidence_stats()
        trade_history_file = Path(data_dir) / "trading" / "trade_history.json"
        try:
            if trade_history_file.exists():
                trades = await asyncio.to_thread(_read_json_file, trade_history_file)
                result["trades"] = [
                    {
                        "id": f"trade_{i}",
                        "timestamp": t.get("timestamp"),
                        "action": t.get("action"),
                        "price": t.get("price"),
                        "confidence": t.get("confidence"),
                        "reasoning": t.get("reasoning", "")[:100]
                    }
                    for i, t in enumerate(trades[-limit:])
                ]
        except Exception:
            self.logger.error("Failed to load trade history for memory", exc_info=True)

        self.dashboard_state.set_cached(f"memory_{limit}", result)
        return result

    async def get_active_rules(self) -> list[dict[str, Any]]:
        """Get currently active semantic rules."""
        cached = self.dashboard_state.get_cached("rules", ttl_seconds=30.0)
        if cached:
            return cached
        if not self.vector_memory:
            return []
        try:
            raw_rules = self.vector_memory.get_active_rules(n_results=20)
            rules = []
            for r in raw_rules:
                mapped_rule = dict(r)
                meta = r.get("metadata", {})
                mapped_rule["rule_text"] = self._render_rule_text(r.get("text", ""), meta)
                mapped_rule["source_trades"] = meta.get("source_trades") or meta.get("source_loss_count")
                for field in RULE_METADATA_FIELDS:
                    mapped_rule[field] = meta.get(field)
                if mapped_rule.get("dominant_exit_profile") in (None, "", self.UNKNOWN_EXIT_PROFILE):
                    resolved_exit_profile = self._resolve_rule_exit_profile(meta)
                    mapped_rule["dominant_exit_profile"] = resolved_exit_profile
                mapped_rule["rule_type"] = mapped_rule["rule_type"] or "best_practice"
                rules.append(mapped_rule)
            self.dashboard_state.set_cached("rules", rules)
            return rules
        except Exception:
            self.logger.error("Failed to retrieve active rules", exc_info=True)
            return []

    @staticmethod
    def _empty_vector_details() -> dict[str, Any]:
        return {
            "experience_count": 0,
            "experiences": [],
            "confidence_stats": {},
            "adx_stats": {},
            "factor_stats": {},
            "rule_count": 0,
            "current_context": None,
        }

    @staticmethod
    def _sort_number(value: Any) -> float:
        try:
            return float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _build_vector_details_result(
        self,
        query: str | None,
        limit: int,
        sort_by: str,
        order: str,
    ) -> dict[str, Any]:
        result = self._empty_vector_details()
        result["experience_count"] = self.vector_memory.trade_count
        result["rule_count"] = self.vector_memory.semantic_rule_count
        result["confidence_stats"] = self.vector_memory.compute_confidence_stats()
        result["adx_stats"] = self.vector_memory.compute_adx_performance()
        result["factor_stats"] = self.vector_memory.compute_factor_performance()

        where_filter = {"outcome": {"$ne": "UPDATE"}}
        embed_query = query
        display_context = query
        if not embed_query:
            display_context, embed_query = _build_current_market_context(
                self.config, self.logger, self.unified_parser
            )
            if display_context:
                result["current_context"] = display_context

        if embed_query:
            experiences = self.vector_memory.retrieve_similar_experiences(
                embed_query, k=limit, where=where_filter
            )
        else:
            experiences = self.vector_memory.get_all_experiences(limit=limit, where=where_filter)

        def get_sort_key(item):
            meta = item.metadata
            if sort_by == "date":
                return str(meta.get("timestamp", ""))
            if sort_by == "similarity":
                return self._sort_number(item.similarity)
            if sort_by == "pnl":
                return self._sort_number(meta.get("pnl_pct"))
            if sort_by == "outcome":
                return str(meta.get("outcome", ""))
            if sort_by == "confidence":
                conf_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
                return conf_map.get(str(meta.get("confidence", "LOW")).upper(), 0)
            if sort_by == "direction":
                return str(meta.get("direction", ""))
            return 0

        experiences.sort(key=get_sort_key, reverse=order == "desc")
        result["experiences"] = [
            {
                "id": exp.id,
                "document": exp.document,
                "similarity": exp.similarity,
                "recency": exp.recency,
                "hybrid_score": exp.hybrid_score,
                "metadata": exp.metadata,
            }
            for exp in experiences[:limit]
        ]
        return result

    async def get_vector_details(
        self, request: Request, query: str = Query(default=None, max_length=500), limit: int = Query(default=50, ge=1, le=500)
    ) -> dict[str, Any]:
        """Get detailed vector memory contents from ChromaDB."""
        # Validate inputs to prevent DoS via unbounded cache keys
        sort_by = request.query_params.get("sort_by", "date")
        if sort_by not in ["date", "similarity", "pnl", "outcome", "confidence", "direction"]:
            sort_by = "date"

        order = request.query_params.get("order", "desc")
        if order not in ["asc", "desc"]:
            order = "desc"

        cache_key = f"vectors_{limit}_{sort_by}_{order}"
        if not query:
            cached = self.dashboard_state.get_cached(cache_key, ttl_seconds=30.0)
            if cached:
                return cached
        result = self._empty_vector_details()
        if not self.vector_memory:
            return result
        try:
            result = await asyncio.to_thread(
                self._build_vector_details_result, query, limit, sort_by, order
            )
            if not query:
                self.dashboard_state.set_cached(cache_key, result)
        except Exception:
            self.logger.error("Failed to retrieve vector details", exc_info=True)
            result["error"] = "Internal error retrieving vector details"
        return result

    async def get_current_position(self) -> dict[str, Any]:
        """Get current open position details."""
        cached = self.dashboard_state.get_cached("position", ttl_seconds=10.0)
        if cached:
            return cached
        current_price = self.dashboard_state.current_price
        if current_price is None:
            data_dir = self.config.DATA_DIR
            prev_response_file = Path(data_dir) / "trading" / "previous_response.json"
            try:
                if prev_response_file.exists():
                    data = await asyncio.to_thread(_read_json_file, prev_response_file)
                    prompt = data.get("prompt", "")
                    match = re.search(r"Current Price:\s*\$?([\d,]+\.?\d*)", prompt)
                    if match:
                        current_price = float(match.group(1).replace(",", ""))
            except Exception:
                self.logger.error("Failed to parse current price from prompt", exc_info=True)

        if not self.persistence:
            return {"has_position": False, "error": "Persistence not available"}
        position = self.persistence.load_position()
        if not position:
            current_exit_management = build_exit_execution_context_from_config(
                self.config, self.config.TIMEFRAME
            ).to_dict()
            res = {
                "has_position": False,
                "current_price": current_price,
                "exit_management": current_exit_management,
                "risk_management": _build_risk_management(current_exit_management),
            }
            self.dashboard_state.set_cached("position", res)
            return res
        sl_distance_pct = _distance_pct_or_fallback(
            position.sl_distance_pct,
            position.entry_price,
            position.stop_loss,
        )
        tp_distance_pct = _distance_pct_or_fallback(
            position.tp_distance_pct,
            position.entry_price,
            position.take_profit,
        )
        current_exit_management = build_exit_execution_context_from_config(
            self.config, self.config.TIMEFRAME
        ).to_dict()
        entry_exit_management = build_exit_execution_context_from_position(position).to_dict()
        res = {
            "has_position": True,
            "current_price": current_price,
            "direction": position.direction,
            "symbol": position.symbol,
            "entry_price": position.entry_price,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "entry_time": position.entry_time.isoformat(),
            "sl_distance_pct": sl_distance_pct,
            "tp_distance_pct": tp_distance_pct,
            "rr_ratio": position.rr_ratio_at_entry,
            "confidence": position.confidence,
            "size": position.size,
            "size_pct": position.size_pct,
            "quote_amount": position.quote_amount,
            "adx_at_entry": position.adx_at_entry,
            "rsi_at_entry": position.rsi_at_entry,
            "max_drawdown_pct": position.max_drawdown_pct,
            "max_profit_pct": position.max_profit_pct,
            "confluence_factors": position.confluence_factors,
            "exit_management": current_exit_management,
            "exit_management_at_entry": entry_exit_management,
            "risk_management": _build_risk_management(current_exit_management, entry_exit_management),
        }
        self.dashboard_state.set_cached("position", res)
        return res

    async def refresh_current_price(self) -> dict[str, Any]:
        """Fetch fresh price from exchange and update dashboard state."""
        if not self.exchange_manager:
            return {"success": False, "error": "Exchange manager not available"}
        try:
            symbol = self.config.CRYPTO_PAIR
            exchange, _ = await self.exchange_manager.find_symbol_exchange(symbol)
            if not exchange:
                return {"success": False, "error": f"No exchange found for {symbol}"}
            ticker = await exchange.fetch_ticker(symbol)
            if not ticker:
                return {"success": False, "error": "Failed to fetch ticker"}
            price = float(ticker.get('last', ticker.get('close', 0)))
            if price > 0 and self.dashboard_state:
                await self.dashboard_state.update_price(price)
            return {"success": True, "current_price": price, "symbol": symbol}
        except Exception:
            self.logger.error("Internal error during price refresh", exc_info=True)
            return {"success": False, "error": "Internal error during price refresh"}

    async def get_blocked_trades(
        self,
        limit: int = Query(default=50, ge=1, le=500),
        guard_type: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Get recent system-rejected (blocked) trade events from ChromaDB.

        Exposes friction reports for dashboard visualization and debugging.
        """
        if not self.vector_memory:
            return {"blocked_count": 0, "blocked_trades": []}

        try:
            blocked = self.vector_memory.get_recent_blocked_trades(
                n=limit,
                guard_type=guard_type,
                max_age_hours=168,
            )
            return {
                "blocked_count": self.vector_memory.get_blocked_trade_count(),
                "blocked_trades": blocked,
            }
        except Exception:
            self.logger.error("Failed to retrieve blocked trades for dashboard", exc_info=True)
            return {"blocked_count": 0, "blocked_trades": [], "error": "Internal error"}
