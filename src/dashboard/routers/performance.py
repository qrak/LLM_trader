from fastapi import APIRouter, Request
from typing import Dict, Any
import json
from pathlib import Path


router = APIRouter(prefix="/api/performance", tags=["performance"])

@router.get("/history")
async def get_performance_history(request: Request) -> Dict[str, Any]:
    """Get historical performance data for the chart."""
    config = request.app.state.config
    logger = request.app.state.logger
    data_dir = getattr(config, "DATA_DIR", "data")
    trade_history_file = Path(data_dir) / "trading" / "trade_history.json"
    stats_file = Path(data_dir) / "trading" / "statistics.json"
    equity_curve = []
    stats = {}
    if stats_file.exists():
        try:
            with open(stats_file, "r") as f:
                stats = json.load(f)
        except Exception:
            logger.error("Failed to load stats file", exc_info=True)
    if trade_history_file.exists():
        try:
            with open(trade_history_file, "r") as f:
                trades = json.load(f)
            initial_capital = stats.get("initial_capital", 10000.0)
            running_capital = initial_capital
            for trade in trades:
                ts = trade.get("timestamp")
                action = trade.get("action", "")
                reasoning = trade.get("reasoning", "")
                if "CLOSE" in action and "P&L:" in reasoning:
                    try:
                        pnl_str = reasoning.split("P&L:")[1].strip().split("%")[0].replace("+", "")
                        pnl_pct = float(pnl_str)
                        pnl_quote = running_capital * (pnl_pct / 100)
                        running_capital += pnl_quote
                        equity_curve.append({
                            "time": ts,
                            "value": round(running_capital, 2),
                            "action": action
                        })
                    except (ValueError, IndexError):
                        pass
                elif action == "BUY":
                    equity_curve.append({
                        "time": ts,
                        "value": round(running_capital, 2),
                        "action": action
                    })
        except Exception:
            logger.error("Failed to process trade history", exc_info=True)
            return {"error": "Failed to load trade history"}
    return {
        "history": equity_curve,
        "stats": stats
    }

@router.get("/stats")
async def get_statistics(request: Request) -> Dict[str, Any]:
    """Get trading statistics summary."""
    config = request.app.state.config
    logger = request.app.state.logger
    data_dir = getattr(config, "DATA_DIR", "data")
    stats_file = Path(data_dir) / "trading" / "statistics.json"
    if stats_file.exists():
        try:
            with open(stats_file, "r") as f:
                return json.load(f)
        except Exception:
            logger.error("Failed to load statistics", exc_info=True)
            return {"error": "Failed to load stats"}
    return {}

