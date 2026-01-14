"""Trading statistics service for performance metrics.

Manages statistics state, recalculation, and context formatting.
"""

from src.logger.logger import Logger
from src.managers.persistence_manager import PersistenceManager
from .statistics_calculator import TradingStatistics, StatisticsCalculator


class TradingStatisticsService:
    """Service for managing trading statistics.
    
    Responsibilities:
    - Hold TradingStatistics state
    - Recalculate stats using StatisticsCalculator
    - Format statistics context for AI
    - Provide current capital
    """
    
    def __init__(self, logger: Logger, persistence: PersistenceManager):
        """Initialize trading statistics service.
        
        Args:
            logger: Logger instance
            persistence: Persistence service for loading/saving statistics and trade history
        """
        self.logger = logger
        self.persistence = persistence
        self.statistics = persistence.load_statistics()
    
    def recalculate(self, initial_capital: float = 10000.0) -> None:
        """Recalculate all statistics from trade history.
        
        Should be called after every closed trade.
        
        Args:
            initial_capital: Starting capital for equity curve calculation
        """
        history = self.persistence.load_trade_history()
        self.statistics = StatisticsCalculator.calculate_from_history(history, initial_capital)
        self.persistence.save_statistics(self.statistics)
        self.logger.info(
            f"Recalculated statistics: {self.statistics.total_trades} trades, "
            f"Win Rate: {self.statistics.win_rate:.1f}%, "
            f"Sharpe: {self.statistics.sharpe_ratio:.2f}"
        )
    
    def get_current_capital(self, initial_capital: float) -> float:
        """Get current capital (initial + realized P&L).
        
        Falls back to initial_capital if no statistics available.
        
        Args:
            initial_capital: The starting capital from config (DEMO_QUOTE_CAPITAL)
            
        Returns:
            Current capital accounting for all closed trade P&L
        """
        if self.statistics.total_trades == 0:
            return initial_capital
        return self.statistics.current_capital
    
    def get_context(self) -> str:
        """Get formatted statistics context for AI prompt injection.
        
        Returns:
            Formatted string with performance statistics
        """
        stats = self.statistics
        if stats.total_trades == 0:
            return ""
        
        lines = [
            "PERFORMANCE STATISTICS:",
            f"- Total Trades: {stats.total_trades} (Win Rate: {stats.win_rate:.1f}%)",
            f"- Avg Trade: {stats.avg_trade_pct:+.2f}% | Best: {stats.best_trade_pct:+.2f}% | Worst: {stats.worst_trade_pct:+.2f}%",
            f"- Total P&L: ${stats.total_pnl_quote:+,.2f} ({stats.total_pnl_pct:+.2f}%)",
            f"- Max Drawdown: {stats.max_drawdown_pct:.2f}%",
            f"- Sharpe Ratio: {stats.sharpe_ratio:.2f} | Sortino: {stats.sortino_ratio:.2f}",
        ]
        
        if stats.profit_factor > 0 and stats.profit_factor != float('inf'):
            lines.append(f"- Profit Factor: {stats.profit_factor:.2f}")
        
        return "\n".join(lines)
