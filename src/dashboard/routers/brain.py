from fastapi import APIRouter, Request
from typing import Dict, Any, List
import json
from pathlib import Path


router = APIRouter(prefix="/api/brain", tags=["brain"])


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
    data_dir = getattr(config, "DATA_DIR", "data")
    prev_response_file = Path(data_dir) / "trading" / "previous_response.json"
    if not prev_response_file.exists():
        return ""
    try:
        with open(prev_response_file, "r") as f:
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

@router.get("/status")
async def get_brain_status(request: Request) -> Dict[str, Any]:
    """Get the current thought process/status of the brain."""
    dashboard_state = request.app.state.dashboard_state
    cached = dashboard_state.get_cached("brain_status", ttl_seconds=30.0)
    if cached:
        return cached
    config = request.app.state.config
    logger = request.app.state.logger
    unified_parser = getattr(request.app.state, "unified_parser", None)
    data_dir = getattr(config, "DATA_DIR", "data")
    prev_response_file = Path(data_dir) / "trading" / "previous_response.json"
    stats_file = Path(data_dir) / "trading" / "statistics.json"
    status = {"status": "active", "trend": "--", "confidence": "--", "action": "--", "adx": None, "rsi": None}
    if prev_response_file.exists():
        try:
            with open(prev_response_file, "r") as f:
                data = json.load(f)
                extracted = _extract_market_status(data, unified_parser)
                status.update(extracted)
        except Exception:
            logger.error("Failed to load brain status", exc_info=True)
    if stats_file.exists():
        try:
            with open(stats_file, "r") as f:
                stats = json.load(f)
                status.update({
                    "total_trades": stats.get("total_trades", 0),
                    "win_rate": stats.get("win_rate", 0),
                    "current_capital": stats.get("current_capital", 0)
                })
        except Exception:
            logger.error("Failed to load statistics", exc_info=True)
    dashboard_state.set_cached("brain_status", status)
    return status


@router.get("/memory")
async def get_vector_memory(request: Request, limit: int = 100) -> Dict[str, Any]:
    """Get recent vector memories (synapses)."""
    dashboard_state = request.app.state.dashboard_state
    cached = dashboard_state.get_cached("memory", ttl_seconds=30.0)
    if cached:
        return cached
    vector_memory = request.app.state.vector_memory
    config = request.app.state.config
    logger = request.app.state.logger
    data_dir = getattr(config, "DATA_DIR", "data")
    result = {
        "experience_count": 0,
        "trades": [],
        "stats": {}
    }
    if vector_memory:
        result["experience_count"] = vector_memory.experience_count
        result["stats"] = vector_memory.compute_confidence_stats()
    trade_history_file = Path(data_dir) / "trading" / "trade_history.json"
    if trade_history_file.exists():
        try:
            with open(trade_history_file, "r") as f:
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
            logger.error("Failed to load trade history for memory", exc_info=True)
    dashboard_state.set_cached("memory", result)
    return result

@router.get("/rules")
async def get_active_rules(request: Request) -> List[Dict[str, Any]]:
    """Get currently active semantic rules."""
    dashboard_state = request.app.state.dashboard_state
    cached = dashboard_state.get_cached("rules", ttl_seconds=30.0)
    if cached:
        return cached
    vector_memory = request.app.state.vector_memory
    logger = request.app.state.logger
    if not vector_memory:
        return []
    try:
        rules = vector_memory.get_active_rules(n_results=20)
        dashboard_state.set_cached("rules", rules)
        return rules
    except Exception:
        logger.error("Failed to retrieve active rules", exc_info=True)
        return []

@router.get("/vectors")
async def get_vector_details(request: Request, query: str = None, limit: int = 50) -> Dict[str, Any]:
    """Get detailed vector memory contents from ChromaDB."""
    # We only cache if query is None (standard dashboard view)
    dashboard_state = request.app.state.dashboard_state
    if not query:
        cached = dashboard_state.get_cached("vectors", ttl_seconds=30.0)
        if cached:
            return cached
    vector_memory = request.app.state.vector_memory
    config = request.app.state.config
    logger = request.app.state.logger
    unified_parser = getattr(request.app.state, "unified_parser", None)
    result = {
        "experience_count": 0,
        "experiences": [],
        "confidence_stats": {},
        "adx_stats": {},
        "factor_stats": {},
        "rule_count": 0,
        "current_context": None
    }
    if not vector_memory:
        return result
    try:
        result["experience_count"] = vector_memory.trade_count
        result["rule_count"] = vector_memory.semantic_rule_count
        result["confidence_stats"] = vector_memory.compute_confidence_stats()
        result["adx_stats"] = vector_memory.compute_adx_performance()
        result["factor_stats"] = vector_memory.compute_factor_performance()
        where_filter = {"outcome": {"$ne": "UPDATE"}}
        context_query = query
        if not context_query:
            context_query = _build_current_market_context(config, logger, unified_parser)
            if context_query:
                result["current_context"] = context_query
        if context_query:
            experiences = vector_memory.retrieve_similar_experiences(context_query, k=limit, where=where_filter)
        else:
            experiences = vector_memory.get_all_experiences(limit=limit, where=where_filter)
        sort_by = request.query_params.get("sort_by", "date")
        order = request.query_params.get("order", "desc")
        reverse = (order == "desc")
        def get_sort_key(item):
            meta = item.metadata
            if sort_by == "date":
                return meta.get("timestamp", "")
            elif sort_by == "similarity":
                return item.similarity
            elif sort_by == "pnl":
                return meta.get("pnl_pct", 0)
            elif sort_by == "outcome":
                return meta.get("outcome", "")
            elif sort_by == "confidence":
                conf_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
                return conf_map.get(meta.get("confidence", "LOW"), 0)
            elif sort_by == "direction":
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
            dashboard_state.set_cached("vectors", result)
    except Exception:
        logger.error("Failed to retrieve vector details", exc_info=True)
        result["error"] = "Internal error retrieving vector details"
    return result

@router.get("/position")
async def get_current_position(request: Request) -> Dict[str, Any]:
    """Get current open position details."""
    dashboard_state = request.app.state.dashboard_state
    cached = dashboard_state.get_cached("position", ttl_seconds=10.0)
    if cached:
        return cached
    import re
    persistence = request.app.state.persistence
    config = request.app.state.config
    logger = request.app.state.logger
    current_price = dashboard_state.current_price
    if current_price is None:
        data_dir = getattr(config, "DATA_DIR", "data")
        prev_response_file = Path(data_dir) / "trading" / "previous_response.json"
        if prev_response_file.exists():
            try:
                with open(prev_response_file, "r") as f:
                    data = json.load(f)
                    prompt = data.get("prompt", "")
                    match = re.search(r"Current Price:\s*\$?([\d,]+\.?\d*)", prompt)
                    if match:
                        current_price = float(match.group(1).replace(",", ""))
            except Exception:
                logger.error("Failed to parse current price from prompt", exc_info=True)
    if not persistence:
        return {"has_position": False, "error": "Persistence not available"}
    position = persistence.load_position()
    if not position:
        res = {"has_position": False, "current_price": current_price}
        dashboard_state.set_cached("position", res)
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
    dashboard_state.set_cached("position", res)
    return res


@router.get("/refresh-price")
async def refresh_current_price(request: Request) -> Dict[str, Any]:
    """Fetch fresh price from exchange and update dashboard state."""
    exchange_manager = getattr(request.app.state, 'exchange_manager', None)
    dashboard_state = getattr(request.app.state, 'dashboard_state', None)
    config = request.app.state.config
    logger = request.app.state.logger
    if not exchange_manager:
        return {"success": False, "error": "Exchange manager not available"}
    try:
        symbol = getattr(config, 'CRYPTO_PAIR', 'BTC/USDC')
        exchange, _ = await exchange_manager.find_symbol_exchange(symbol)
        if not exchange:
            return {"success": False, "error": f"No exchange found for {symbol}"}
        ticker = await exchange.fetch_ticker(symbol)
        if not ticker:
            return {"success": False, "error": "Failed to fetch ticker"}
        price = float(ticker.get('last', ticker.get('close', 0)))
        if price > 0 and dashboard_state:
            await dashboard_state.update_price(price)
        return {"success": True, "current_price": price, "symbol": symbol}
    except Exception:
        logger.error("Internal error during price refresh", exc_info=True)
        return {"success": False, "error": "Internal error during price refresh"}
