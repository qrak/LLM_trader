from fastapi import APIRouter, Request
from typing import Dict, Any, List
import json
from pathlib import Path

router = APIRouter(prefix="/api/brain", tags=["brain"])


def _build_current_market_context(config) -> str:
    """Build context query string from current market conditions.

    Reads previous_response.json to extract trend, ADX, and other indicators.

    Returns:
        Context string like "BULLISH + High ADX + HIGH Volatility" or empty string if unavailable.
    """
    data_dir = getattr(config, "DATA_DIR", "data")
    prev_response_file = Path(data_dir) / "trading" / "previous_response.json"

    if not prev_response_file.exists():
        return ""

    try:
        with open(prev_response_file, "r") as f:
            data = json.load(f)
            response = data.get("response", {})

            # Extract trend from text analysis
            text = response.get("text_analysis", "")
            if "BULLISH" in text.upper():
                trend = "BULLISH"
            elif "BEARISH" in text.upper():
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"

            # Extract ADX
            adx = response.get("adx", 0) or 0
            if adx >= 25:
                adx_label = "High ADX"
            elif adx < 20:
                adx_label = "Low ADX"
            else:
                adx_label = "Medium ADX"

            # Extract volatility from prompt if available
            prompt = data.get("prompt", "")
            if "HIGH" in prompt.upper() and "VOLATILITY" in prompt.upper():
                vol = "HIGH Volatility"
            elif "LOW" in prompt.upper() and "VOLATILITY" in prompt.upper():
                vol = "LOW Volatility"
            else:
                vol = "MEDIUM Volatility"

            return f"{trend} + {adx_label} + {vol}"
    except Exception:
        return ""

@router.get("/status")
async def get_brain_status(request: Request) -> Dict[str, Any]:
    """Get the current thought process/status of the brain."""
    brain_service = request.app.state.brain_service
    config = request.app.state.config
    unified_parser = getattr(request.app.state, "unified_parser", None)
    data_dir = getattr(config, "DATA_DIR", "data")
    prev_response_file = Path(data_dir) / "trading" / "previous_response.json"
    stats_file = Path(data_dir) / "trading" / "statistics.json"
    status = {
        "status": "active",
        "trend": "--",
        "confidence": "--",
        "action": "--",
        "adx": None,
        "rsi": None
    }
    if prev_response_file.exists():
        try:
            with open(prev_response_file, "r") as f:
                data = json.load(f)
                response = data.get("response", {})
                text = response.get("text_analysis", "")
                if "BULLISH" in text.upper():
                    status["trend"] = "BULLISH"
                elif "BEARISH" in text.upper():
                    status["trend"] = "BEARISH"
                else:
                    status["trend"] = "NEUTRAL"
                status["adx"] = response.get("adx")
                status["rsi"] = response.get("rsi")
                if unified_parser:
                    analysis = unified_parser.extract_json_block(text, unwrap_key='analysis')
                    if analysis:
                        status["action"] = analysis.get("signal", "--")
                        status["confidence"] = analysis.get("confidence", "--")
        except Exception:
            pass
    if stats_file.exists():
        try:
            with open(stats_file, "r") as f:
                stats = json.load(f)
                status["total_trades"] = stats.get("total_trades", 0)
                status["win_rate"] = stats.get("win_rate", 0)
                status["current_capital"] = stats.get("current_capital", 0)
        except Exception:
            pass
    return status


@router.get("/memory")
async def get_vector_memory(request: Request, limit: int = 100) -> Dict[str, Any]:
    """Get recent vector memories (synapses)."""
    vector_memory = request.app.state.vector_memory
    config = request.app.state.config
    data_dir = getattr(config, "DATA_DIR", "data")
    
    result = {
        "experience_count": 0,
        "trades": [],
        "stats": {}
    }
    
    # Get from vector memory if available
    if vector_memory:
        result["experience_count"] = vector_memory.experience_count
        result["stats"] = vector_memory.compute_confidence_stats()
    
    # Also load trade history as "memories"
    trade_history_file = Path(data_dir) / "trading" / "trade_history.json"
    if trade_history_file.exists():
        try:
            with open(trade_history_file, "r") as f:
                trades = json.load(f)
                # Format for visualization
                result["trades"] = [
                    {
                        "id": f"trade_{i}",
                        "timestamp": t.get("timestamp"),
                        "action": t.get("action"),
                        "price": t.get("price"),
                        "confidence": t.get("confidence"),
                        "reasoning": t.get("reasoning", "")[:100]  # Truncate
                    }
                    for i, t in enumerate(trades[-limit:])
                ]
        except Exception:
            pass
    
    return result

@router.get("/rules")
async def get_active_rules(request: Request) -> List[Dict[str, Any]]:
    """Get currently active semantic rules."""
    vector_memory = request.app.state.vector_memory
    if not vector_memory:
        return []
    
    try:
        rules = vector_memory.get_active_rules(n_results=20)
        return rules
    except Exception:
        return []

@router.get("/vectors")
async def get_vector_details(request: Request, query: str = None, limit: int = 50) -> Dict[str, Any]:
    """Get detailed vector memory contents from ChromaDB."""
    vector_memory = request.app.state.vector_memory
    config = request.app.state.config

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
        # Get counts (trade_count excludes UPDATE entries)
        result["experience_count"] = vector_memory.trade_count
        result["rule_count"] = vector_memory.semantic_rule_count

        # Get stats breakdowns
        result["confidence_stats"] = vector_memory.compute_confidence_stats()
        result["adx_stats"] = vector_memory.compute_adx_performance()
        result["factor_stats"] = vector_memory.compute_factor_performance()

        # Exclude UPDATE entries from results using WHERE filter
        where_filter = {"outcome": {"$ne": "UPDATE"}}

        # Build context query - use provided query or auto-generate from current market
        context_query = query
        if not context_query:
            context_query = _build_current_market_context(config)
            if context_query:
                result["current_context"] = context_query

        # Retrieve experiences with similarity if we have a context
        if context_query:
            experiences = vector_memory.retrieve_similar_experiences(
                context_query, k=limit, where=where_filter
            )
        else:
            # Fallback to get all if no context available
            experiences = vector_memory.get_all_experiences(limit=limit, where=where_filter)

        # Apply sorting
        sort_by = request.query_params.get("sort_by", "date")
        order = request.query_params.get("order", "desc")
        reverse = (order == "desc")

        def get_sort_key(item):
            meta = item.get("metadata", {})
            if sort_by == "date":
                return meta.get("timestamp", "")
            elif sort_by == "similarity":
                return item.get("similarity", 0)
            elif sort_by == "pnl":
                return meta.get("pnl_pct", 0)
            elif sort_by == "outcome":
                return meta.get("outcome", "")
            elif sort_by == "confidence":
                # Map confidence strings to numeric values for sorting
                conf_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
                return conf_map.get(meta.get("confidence", "LOW"), 0)
            elif sort_by == "direction":
                return meta.get("direction", "")
            return 0

        experiences.sort(key=get_sort_key, reverse=reverse)
        result["experiences"] = experiences[:limit]  # Re-slice after sort if needed

            
    except Exception as e:
        result["error"] = str(e)
    return result


@router.get("/position")
async def get_current_position(request: Request) -> Dict[str, Any]:
    """Get current open position details for the position panel."""
    import re
    persistence = request.app.state.persistence
    config = request.app.state.config
    dashboard_state = getattr(request.app.state, 'dashboard_state', None)
    current_price = dashboard_state.current_price if dashboard_state else None
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
                pass
    if not persistence:
        return {"has_position": False, "error": "Persistence not available"}
    position = persistence.load_position()
    if not position:
        return {"has_position": False, "current_price": current_price}
    return {
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


@router.get("/refresh-price")
async def refresh_current_price(request: Request) -> Dict[str, Any]:
    """Fetch fresh price from exchange and update dashboard state."""
    exchange_manager = getattr(request.app.state, 'exchange_manager', None)
    dashboard_state = getattr(request.app.state, 'dashboard_state', None)
    config = request.app.state.config
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
    except Exception as e:
        return {"success": False, "error": str(e)}
