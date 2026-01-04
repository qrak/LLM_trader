"""Statistics calculator for trading performance metrics.

Calculates all-time cumulative statistics including Sharpe/Sortino ratios,
drawdowns, win rate, and other performance metrics from trade history.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import math


@dataclass(slots=True)
class TradingStatistics:
    """All-time cumulative trading statistics.
    
    Updated after every closed trade to provide the AI with
    performance context for position sizing decisions.
    """
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl_pct: float = 0.0
    total_pnl_quote: float = 0.0
    avg_trade_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_pnl_pct": self.total_pnl_pct,
            "total_pnl_quote": self.total_pnl_quote,
            "avg_trade_pct": self.avg_trade_pct,
            "best_trade_pct": self.best_trade_pct,
            "worst_trade_pct": self.worst_trade_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "avg_drawdown_pct": self.avg_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "profit_factor": self.profit_factor,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingStatistics':
        """Create TradingStatistics from dictionary."""
        return cls(
            total_trades=data.get("total_trades", 0),
            winning_trades=data.get("winning_trades", 0),
            losing_trades=data.get("losing_trades", 0),
            win_rate=data.get("win_rate", 0.0),
            total_pnl_pct=data.get("total_pnl_pct", 0.0),
            total_pnl_quote=data.get("total_pnl_quote", 0.0),
            avg_trade_pct=data.get("avg_trade_pct", 0.0),
            best_trade_pct=data.get("best_trade_pct", 0.0),
            worst_trade_pct=data.get("worst_trade_pct", 0.0),
            max_drawdown_pct=data.get("max_drawdown_pct", 0.0),
            avg_drawdown_pct=data.get("avg_drawdown_pct", 0.0),
            sharpe_ratio=data.get("sharpe_ratio", 0.0),
            sortino_ratio=data.get("sortino_ratio", 0.0),
            profit_factor=data.get("profit_factor", 0.0),
            last_updated=datetime.fromisoformat(data["last_updated"]) if "last_updated" in data else datetime.now(),
        )


class StatisticsCalculator:
    """Calculates trading performance statistics from trade history."""

    @staticmethod
    def calculate_from_history(
        trade_history: List[Dict[str, Any]],
        initial_capital: float = 10000.0
    ) -> TradingStatistics:
        """Calculate all statistics from full trade history.
        
        Args:
            trade_history: List of trade decision dictionaries
            initial_capital: Starting capital for equity curve calculation
            
        Returns:
            TradingStatistics with all metrics calculated
        """
        if not trade_history:
            return TradingStatistics()
        trades = StatisticsCalculator._extract_closed_trades(trade_history)
        if not trades:
            return TradingStatistics()
        pnl_percentages = [t["pnl_pct"] for t in trades]
        pnl_amounts = [t["pnl_quote"] for t in trades]
        total_trades = len(trades)
        winning_trades = sum(1 for pnl in pnl_percentages if pnl > 0)
        losing_trades = sum(1 for pnl in pnl_percentages if pnl < 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        total_pnl_pct = sum(pnl_percentages)
        total_pnl_quote = sum(pnl_amounts)
        avg_trade_pct = total_pnl_pct / total_trades if total_trades > 0 else 0.0
        best_trade_pct = max(pnl_percentages) if pnl_percentages else 0.0
        worst_trade_pct = min(pnl_percentages) if pnl_percentages else 0.0
        equity_curve = StatisticsCalculator._build_equity_curve(trades, initial_capital)
        max_dd, avg_dd = StatisticsCalculator._calculate_drawdowns(equity_curve)
        sharpe = StatisticsCalculator._calculate_sharpe_ratio(pnl_percentages)
        sortino = StatisticsCalculator._calculate_sortino_ratio(pnl_percentages)
        profit_factor = StatisticsCalculator._calculate_profit_factor(pnl_amounts)
        return TradingStatistics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl_pct=total_pnl_pct,
            total_pnl_quote=total_pnl_quote,
            avg_trade_pct=avg_trade_pct,
            best_trade_pct=best_trade_pct,
            worst_trade_pct=worst_trade_pct,
            max_drawdown_pct=max_dd,
            avg_drawdown_pct=avg_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            profit_factor=profit_factor,
            last_updated=datetime.now(),
        )

    @staticmethod
    def _extract_closed_trades(trade_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract closed trades with P&L from history.
        
        Pairs BUY/SELL entries with their corresponding CLOSE actions.
        """
        closed_trades = []
        open_position: Optional[Dict[str, Any]] = None
        for trade in trade_history:
            action = trade.get("action", "").upper()
            if action in ("BUY", "SELL"):
                open_position = trade
            elif action in ("CLOSE", "CLOSE_LONG", "CLOSE_SHORT") and open_position:
                entry_price = open_position.get("price", 0)
                exit_price = trade.get("price", 0)
                quantity = open_position.get("quantity", 0)
                if open_position["action"].upper() == "BUY":
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                    pnl_quote = (exit_price - entry_price) * quantity
                else:
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100 if entry_price > 0 else 0
                    pnl_quote = (entry_price - exit_price) * quantity
                closed_trades.append({
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "pnl_quote": pnl_quote,
                    "quantity": quantity,
                    "direction": "LONG" if open_position["action"].upper() == "BUY" else "SHORT",
                })
                open_position = None
        return closed_trades

    @staticmethod
    def _build_equity_curve(
        trades: List[Dict[str, Any]], 
        initial_capital: float
    ) -> List[float]:
        """Build equity curve from closed trades."""
        equity = [initial_capital]
        current = initial_capital
        for trade in trades:
            current += trade["pnl_quote"]
            equity.append(current)
        return equity

    @staticmethod
    def _calculate_drawdowns(equity_curve: List[float]) -> tuple:
        """Calculate max and average drawdown from equity curve.
        
        Returns:
            Tuple of (max_drawdown_pct, avg_drawdown_pct)
        """
        if len(equity_curve) < 2:
            return 0.0, 0.0
        drawdowns = []
        peak = equity_curve[0]
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            if peak > 0:
                dd = ((equity - peak) / peak) * 100
                drawdowns.append(dd)
        max_dd = min(drawdowns) if drawdowns else 0.0
        negative_dds = [d for d in drawdowns if d < 0]
        avg_dd = sum(negative_dds) / len(negative_dds) if negative_dds else 0.0
        return max_dd, avg_dd

    @staticmethod
    def _calculate_sharpe_ratio(
        returns: List[float], 
        risk_free_rate: float = 0.0,
        annualization_factor: float = 365
    ) -> float:
        """Calculate Sharpe ratio from trade returns.
        
        Uses trade-based returns rather than time-series for simplicity.
        Assumes one trade per period for annualization.
        """
        if len(returns) < 2:
            return 0.0
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0
        if std_dev == 0:
            return 0.0
        sharpe = (mean_return - risk_free_rate) / std_dev
        return round(sharpe, 2)

    @staticmethod
    def _calculate_sortino_ratio(
        returns: List[float],
        risk_free_rate: float = 0.0
    ) -> float:
        """Calculate Sortino ratio (downside-risk adjusted).
        
        Uses only negative returns for volatility calculation.
        """
        if len(returns) < 2:
            return 0.0
        mean_return = sum(returns) / len(returns)
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return float('inf') if mean_return > 0 else 0.0
        downside_variance = sum(r ** 2 for r in negative_returns) / len(negative_returns)
        downside_std = math.sqrt(downside_variance)
        if downside_std == 0:
            return 0.0
        sortino = (mean_return - risk_free_rate) / downside_std
        return round(sortino, 2)

    @staticmethod
    def _calculate_profit_factor(pnl_amounts: List[float]) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = sum(p for p in pnl_amounts if p > 0)
        gross_loss = abs(sum(p for p in pnl_amounts if p < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return round(gross_profit / gross_loss, 2)
