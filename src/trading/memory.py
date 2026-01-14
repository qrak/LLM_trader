"""Trading memory service for managing recent decision context.

Handles short-term memory of trading decisions for AI context injection.
"""

from typing import Optional, List

from src.logger.logger import Logger
from src.managers.persistence_manager import PersistenceManager
from .dataclasses import TradingMemory, TradeDecision


class TradingMemoryService:
    """Service for managing short-term trading memory.
    
    Responsibilities:
    - Build memory from recent trade history
    - Add new decisions to memory
    - Provide formatted memory context for AI
    """
    
    def __init__(self, logger: Logger, persistence: PersistenceManager, max_memory: int = 10):
        """Initialize trading memory service.
        
        Args:
            logger: Logger instance
            persistence: Persistence service for loading trade history
            max_memory: Maximum number of decisions to keep in memory
        """
        self.logger = logger
        self.persistence = persistence
        self.max_memory = max_memory
        self.memory = self._build_memory_from_history()
    
    def add_decision(self, decision: TradeDecision) -> None:
        """Add a trading decision to memory.
        
        Args:
            decision: Trade decision to add
        """
        self.memory.add_decision(decision)
    
    def get_context_summary(self, current_price: Optional[float] = None) -> str:
        """Get formatted memory context for prompt injection.
        
        Args:
            current_price: Current market price for P&L calculation
            
        Returns:
            Formatted memory context with overall P&L data from all trades
        """
        full_history_dicts = self.persistence.load_trade_history()
        full_history = [TradeDecision.from_dict(d) for d in full_history_dicts]
        
        return self.memory.get_context_summary(current_price, full_history)
    
    def get_recent_decisions(self, n: int = 5) -> List[TradeDecision]:
        """Get recent decisions from memory.
        
        Args:
            n: Number of recent decisions to retrieve
            
        Returns:
            List of recent trade decisions
        """
        return self.memory.get_recent_decisions(n)
    
    def _build_memory_from_history(self) -> TradingMemory:
        """Build TradingMemory from recent trade history.
        
        Returns:
            TradingMemory instance with recent decisions loaded
        """
        memory = TradingMemory(max_decisions=self.max_memory)
        
        history = self.persistence.load_trade_history()
        if not history:
            return memory
        
        recent_history = history[-self.max_memory:]
        
        for trade_data in recent_history:
            try:
                decision = TradeDecision.from_dict(trade_data)
                memory.add_decision(decision)
            except Exception as e:
                self.logger.warning(f"Could not load decision from history: {e}")
        
        self.logger.info(f"Built memory with {len(memory.decisions)} decisions from history")
        return memory
