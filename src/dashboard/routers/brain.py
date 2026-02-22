"""Router for brain state, memory vectors, and positions."""
import json
import re
from pathlib import Path
from typing import Dict, Any, List

from fastapi import APIRouter, Request

def _extract_market_status(data: Dict[str, Any], unified_parser=None) -> Dict[str, Any]:
    """Helper to extract market status from previous_response data."""
    response = data.get("response", {})
    text = response.get("text_analysis", "")
    status = {
        "trend": "NEUTRAL",
        "action": "--",
        "confidence": "--",
        "adx": response.get("adx"),
        "rsi": response.get("rsi")
    }
    if unified_parser:
        analysis = unified_parser.extract_json_block(text, unwrap_key='analysis')
        if analysis:
            trend_data = analysis.get("trend")
            if isinstance(trend_data, dict):
                status["trend"] = trend_data.get("direction", "NEUTRAL")
            status["action"] = analysis.get("signal", "--")
            status["confidence"] = analysis.get("confidence", "--")
    if status["trend"] == "NEUTRAL":
        if "BEARISH" in text.upper():
            status["trend"] = "BEARISH"
        elif "BULLISH" in text.upper():
            status["trend"] = "BULLISH"
    return status

def _build_current_market_context(config, logger, unified_parser=None) -> str:
    """Build context query string from current market conditions."""
    data_dir = config.DATA_DIR
    prev_response_file = Path(data_dir) / "trading" / "previous_response.json"
    if not prev_response_file.exists():
        return ""
    try:
        with open(prev_response_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            status = _extract_market_status(data, unified_parser)
            adx = status["adx"] or 0
            adx_label = "High ADX" if adx >= 25 else ("Low ADX" if adx < 20 else "Medium ADX")
            prompt = data.get("prompt", "")
            vol = "MEDIUM Volatility"
            if "HIGH" in prompt.upper() and "VOLATILITY" in prompt.upper():
                vol = "HIGH Volatility"
            elif "LOW" in prompt.upper() and "VOLATILITY" in prompt.upper():
                vol = "LOW Volatility"
            return f"{status['trend']} + {adx_label} + {vol}"
    except Exception:
        logger.error("Failed to build market context", exc_info=True)
        return ""

class BrainRouter:
    """Handles endpoints for the trading brain status, rules, and memory."""
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

    async def get_brain_status(self) -> Dict[str, Any]:
        """Get the current thought process/status of the brain."""
        cached = self.dashboard_state.get_cached("brain_status", ttl_seconds=30.0)
        if cached:
            return cached
        data_dir = getattr(self.config, "DATA_DIR", "data")
        prev_response_file = Path(data_dir) / "trading" / "previous_response.json"
        stats_file = Path(data_dir) / "trading" / "statistics.json"
        status = {"status": "active", "trend": "--", "confidence": "--", "action": "--", "adx": None, "rsi": None}
        if prev_response_file.exists():
            try:
                with open(prev_response_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    extracted = _extract_market_status(data, self.unified_parser)
                    status.update(extracted)
            except Exception:
                self.logger.error("Failed to load brain status", exc_info=True)
        if stats_file.exists():
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                    status.update({
                        "total_trades": stats.get("total_trades", 0),
                        "win_rate": stats.get("win_rate", 0),
                        "current_capital": stats.get("current_capital", 0)
                    })
            except Exception:
                self.logger.error("Failed to load statistics", exc_info=True)
        self.dashboard_state.set_cached("brain_status", status)
        return status

    async def get_vector_memory(self, limit: int = 100) -> Dict[str, Any]:
        """Get recent vector memories (synapses)."""
        cached = self.dashboard_state.get_cached("memory", ttl_seconds=30.0)
        if cached:
            return cached
        data_dir = getattr(self.config, "DATA_DIR", "data")
        result = {
            "experience_count": 0,
            "trades": [],
            "stats": {}
        }
        if self.vector_memory:
            result["experience_count"] = self.vector_memory.experience_count
            result["stats"] = self.vector_memory.compute_confidence_stats()
        trade_history_file = Path(data_dir) / "trading" / "trade_history.json"
        if trade_history_file.exists():
            try:
                with open(trade_history_file, "r", encoding="utf-8") as f:
                    trades = json.load(f)
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
        self.dashboard_state.set_cached("memory", result)
        return result

    async def get_active_rules(self) -> List[Dict[str, Any]]:
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
                mapped_rule["rule_text"] = r.get("text", "")
                meta = r.get("metadata", {})
                mapped_rule["win_rate"] = meta.get("win_rate")
                mapped_rule["source_trades"] = meta.get("source_trades")
                rules.append(mapped_rule)
            self.dashboard_state.set_cached("rules", rules)
            return rules
        except Exception:
            self.logger.error("Failed to retrieve active rules", exc_info=True)
            return []

    async def get_vector_details(self, request: Request, query: str = None, limit: int = 50) -> Dict[str, Any]:
        """Get detailed vector memory contents from ChromaDB."""
        sort_by = request.query_params.get("sort_by", "date")
        order = request.query_params.get("order", "desc")
        cache_key = f"vectors_{limit}_{sort_by}_{order}"
        if not query:
            cached = self.dashboard_state.get_cached(cache_key, ttl_seconds=30.0)
            if cached:
                return cached
        result = {
            "experience_count": 0,
            "experiences": [],
            "confidence_stats": {},
            "adx_stats": {},
            "factor_stats": {},
            "rule_count": 0,
            "current_context": None
        }
        if not self.vector_memory:
            return result
        try:
            result["experience_count"] = self.vector_memory.trade_count
            result["rule_count"] = self.vector_memory.semantic_rule_count
            result["confidence_stats"] = self.vector_memory.compute_confidence_stats()
            result["adx_stats"] = self.vector_memory.compute_adx_performance()
            result["factor_stats"] = self.vector_memory.compute_factor_performance()
            where_filter = {"outcome": {"$ne": "UPDATE"}}
            context_query = query
            if not context_query:
                context_query = _build_current_market_context(self.config, self.logger, self.unified_parser)
                if context_query:
                    result["current_context"] = context_query
            if context_query:
                experiences = self.vector_memory.retrieve_similar_experiences(
                    context_query, k=limit, where=where_filter
                )
            else:
                experiences = self.vector_memory.get_all_experiences(limit=limit, where=where_filter)
            reverse = order == "desc"
            def get_sort_key(item):
                # pylint: disable=too-many-return-statements
                meta = item.metadata
                if sort_by == "date":
                    return meta.get("timestamp", "")
                if sort_by == "similarity":
                    return item.similarity
                if sort_by == "pnl":
                    return meta.get("pnl_pct", 0)
                if sort_by == "outcome":
                    return meta.get("outcome", "")
                if sort_by == "confidence":
                    conf_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
                    return conf_map.get(meta.get("confidence", "LOW"), 0)
                if sort_by == "direction":
                    return meta.get("direction", "")
                return 0
            experiences.sort(key=get_sort_key, reverse=reverse)
            experiences_list = []
            for exp in experiences[:limit]:
                experiences_list.append({
                    "id": exp.id,
                    "document": exp.document,
                    "similarity": exp.similarity,
                    "recency": exp.recency,
                    "hybrid_score": exp.hybrid_score,
                    "metadata": exp.metadata
                })
            result["experiences"] = experiences_list
            if not query:
                self.dashboard_state.set_cached(cache_key, result)
        except Exception:
            self.logger.error("Failed to retrieve vector details", exc_info=True)
            result["error"] = "Internal error retrieving vector details"
        return result

    async def get_current_position(self) -> Dict[str, Any]:
        """Get current open position details."""
        cached = self.dashboard_state.get_cached("position", ttl_seconds=10.0)
        if cached:
            return cached
        current_price = self.dashboard_state.current_price
        if current_price is None:
            data_dir = self.config.DATA_DIR
            prev_response_file = Path(data_dir) / "trading" / "previous_response.json"
            if prev_response_file.exists():
                try:
                    with open(prev_response_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
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
            res = {"has_position": False, "current_price": current_price}
            self.dashboard_state.set_cached("position", res)
            return res
        res = {
            "has_position": True,
            "current_price": current_price,
            "direction": position.direction,
            "symbol": position.symbol,
            "entry_price": position.entry_price,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "entry_time": position.entry_time.isoformat(),
            "sl_distance_pct": position.sl_distance_pct,
            "tp_distance_pct": position.tp_distance_pct,
            "rr_ratio": position.rr_ratio_at_entry,
            "confidence": position.confidence,
            "size": position.size,
            "size_pct": position.size_pct,
            "quote_amount": position.quote_amount,
            "adx_at_entry": position.adx_at_entry,
            "rsi_at_entry": position.rsi_at_entry,
            "max_drawdown_pct": position.max_drawdown_pct,
            "max_profit_pct": position.max_profit_pct,
            "confluence_factors": position.confluence_factors
        }
        self.dashboard_state.set_cached("position", res)
        return res

    async def refresh_current_price(self) -> Dict[str, Any]:
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
