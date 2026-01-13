"""Statistics calculator for trading performance metrics.

Calculates all-time cumulative statistics including Sharpe/Sortino ratios,
drawdowns, win rate, and other performance metrics from trade history.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import math
import numpy as np


from src.utils.dataclass_utils import SerializableMixin


@dataclass(slots=True)
class TradingStatistics(SerializableMixin):
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
    initial_capital: float = 10000.0
    current_capital: float = 10000.0
    avg_trade_pct: float = 0.0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class StatisticsCalculator:
    """Calculates trading performance statistics from trade history."""

    @staticmethod
    def calculate_from_history(
        trade_history: List[Dict[str, Any]],
        initial_capital: float = 10000.0
    ) -> TradingStatistics:
        """Calculate all statistics from full trade history using numpy optimization."""
        if not trade_history:
            return TradingStatistics()
            
        trades = StatisticsCalculator._extract_closed_trades(trade_history)
        if not trades:
            return TradingStatistics()
            
        # Convert to numpy arrays for vectorized operations
        pnl_percentages = np.array([t["pnl_pct"] for t in trades])
        pnl_amounts = np.array([t["pnl_quote"] for t in trades])
        
        total_trades = len(trades)
        winning_trades = int(np.sum(pnl_percentages > 0))
        losing_trades = int(np.sum(pnl_percentages < 0))
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        total_pnl_pct = float(np.sum(pnl_percentages))
        total_pnl_quote = float(np.sum(pnl_amounts))
        
        avg_trade_pct = total_pnl_pct / total_trades if total_trades > 0 else 0.0
        best_trade_pct = float(np.max(pnl_percentages)) if total_trades > 0 else 0.0
        worst_trade_pct = float(np.min(pnl_percentages)) if total_trades > 0 else 0.0
        
        # Calculate equity curve and drawdowns
        equity_curve = np.zeros(len(pnl_amounts) + 1)
        equity_curve[0] = initial_capital
        equity_curve[1:] = np.cumsum(pnl_amounts) + initial_capital
        
        max_dd, avg_dd = StatisticsCalculator._calculate_drawdowns(equity_curve)
        
        sharpe = StatisticsCalculator._calculate_sharpe_ratio(pnl_percentages)
        sortino = StatisticsCalculator._calculate_sortino_ratio(pnl_percentages)
        profit_factor = StatisticsCalculator._calculate_profit_factor(pnl_amounts)
        
        current_capital = initial_capital + total_pnl_quote
        
        return TradingStatistics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl_pct=total_pnl_pct,
            total_pnl_quote=total_pnl_quote,
            initial_capital=initial_capital,
            current_capital=current_capital,
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
        """Extract closed trades with P&L from history."""
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
    def _calculate_drawdowns(equity_curve: np.ndarray) -> tuple:
        """Calculate max and average drawdown from equity curve using numpy."""
        if len(equity_curve) < 2:
            return 0.0, 0.0
            
        # Efficient peak calculation using accumulative maximum
        peaks = np.maximum.accumulate(equity_curve)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdowns = np.where(peaks > 0, (equity_curve - peaks) / peaks * 100, 0.0)
            
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        
        negative_dds = drawdowns[drawdowns < 0]
        avg_dd = float(np.mean(negative_dds)) if len(negative_dds) > 0 else 0.0
        
        return max_dd, avg_dd

    @staticmethod
    def _calculate_sharpe_ratio(
        returns: np.ndarray, 
        risk_free_rate: float = 0.0
    ) -> float:
        """Calculate Sharpe ratio using numpy."""
        if len(returns) < 2:
            return 0.0
            
        mean_return = np.mean(returns)
        std_dev = np.std(returns)
        
        if std_dev == 0:
            return 0.0
            
        sharpe = (mean_return - risk_free_rate) / std_dev
        return round(float(sharpe), 2)

    @staticmethod
    def _calculate_sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> float:
        """Calculate Sortino ratio using numpy."""
        if len(returns) < 2:
            return 0.0
            
        mean_return = np.mean(returns)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf') if mean_return > 0 else 0.0
            
        downside_std = np.std(negative_returns)
        # Note: Standard Sortino uses sqrt(sum(r^2)/N), np.std uses sqrt(sum((r-mean)^2)/N)
        # For true downside deviation relative to 0 target:
        downside_deviation = np.sqrt(np.mean(negative_returns ** 2))
        
        if downside_deviation == 0:
            return 0.0
            
        sortino = (mean_return - risk_free_rate) / downside_deviation
        return round(float(sortino), 2)

    @staticmethod
    def _calculate_profit_factor(pnl_amounts: np.ndarray) -> float:
        """Calculate profit factor using numpy."""
        gross_profit = np.sum(pnl_amounts[pnl_amounts > 0])
        gross_loss = abs(np.sum(pnl_amounts[pnl_amounts < 0]))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
            
        return round(float(gross_profit / gross_loss), 2)
