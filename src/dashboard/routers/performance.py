"""Router for performance tracking and statistics."""
import json
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter

from src.trading.statistics_calculator import StatisticsCalculator


class PerformanceRouter:
    """Handles performance history and statistics endpoints."""
    def __init__(self, config, logger, dashboard_state):
        self.router = APIRouter(prefix="/api/performance", tags=["performance"])
        self.config = config
        self.logger = logger
        self.dashboard_state = dashboard_state

        self.router.add_api_route("/history", self.get_performance_history, methods=["GET"])
        self.router.add_api_route("/stats", self.get_statistics, methods=["GET"])

    async def get_performance_history(self) -> Dict[str, Any]:
        """Get historical performance data for the chart."""
        cached = self.dashboard_state.get_cached("performance_history", ttl_seconds=60.0)
        if cached:
            return cached
        try:
            data_dir = self.config.DATA_DIR
        except AttributeError:
            data_dir = "data"
        trade_history_file = Path(data_dir) / "trading" / "trade_history.json"
        stats_file = Path(data_dir) / "trading" / "statistics.json"
        equity_curve = []
        stats = {}
        if stats_file.exists():
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    stats = json.load(f)
            except Exception:
                self.logger.error("Failed to load stats file", exc_info=True)
        if trade_history_file.exists():
            try:
                with open(trade_history_file, "r", encoding="utf-8") as f:
                    trades = json.load(f)
                initial_capital = stats.get("initial_capital", 10000.0)
                running_capital = initial_capital
                # pylint: disable=protected-access
                closed_trades = StatisticsCalculator._extract_closed_trades(trades)
                closed_trade_idx = 0
                open_position = None
                for trade in trades:
                    ts = trade.get("timestamp")
                    action = trade.get("action", "").upper()
                    if action in ("BUY", "SELL"):
                        open_position = trade
                        equity_curve.append({
                            "time": ts,
                            "value": round(running_capital, 2),
                            "action": action,
                            "price": trade.get("price")
                        })
                    elif action in ("CLOSE", "CLOSE_LONG", "CLOSE_SHORT") and open_position:
                        if closed_trade_idx < len(closed_trades):
                            running_capital += closed_trades[closed_trade_idx].pnl_quote
                            closed_trade_idx += 1
                        equity_curve.append({
                            "time": ts,
                            "value": round(running_capital, 2),
                            "action": action,
                            "price": trade.get("price")
                        })
                        open_position = None
            except Exception:
                self.logger.error("Failed to process trade history", exc_info=True)
                return {"error": "Failed to load trade history"}
        result = {
            "history": equity_curve,
            "stats": stats
        }
        self.dashboard_state.set_cached("performance_history", result)
        return result

    async def get_statistics(self) -> Dict[str, Any]:
        """Get trading statistics summary."""
        cached = self.dashboard_state.get_cached("statistics", ttl_seconds=60.0)
        if cached:
            return cached
        data_dir = self.config.DATA_DIR
        stats_file = Path(data_dir) / "trading" / "statistics.json"
        if stats_file.exists():
            try:
                with open(stats_file, "r", encoding="utf-8") as f:
                    result = json.load(f)
                    self.dashboard_state.set_cached("statistics", result)
                    return result
            except Exception:
                self.logger.error("Failed to load statistics", exc_info=True)
                return {"error": "Failed to load stats"}
        return {}
