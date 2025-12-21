"""Data persistence for trading decisions and positions."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.logger.logger import Logger
from .dataclasses import Position, TradeDecision, TradingMemory


class DataPersistence:
    """Manages persistence of trading positions, decisions, and memory."""
    
    def __init__(self, logger: Logger, data_dir: str = "trading_data", max_memory: int = 10):
        """Initialize data persistence.
        
        Args:
            logger: Logger instance
            data_dir: Directory for trading data files
            max_memory: Maximum number of decisions to keep in memory
        """
        self.logger = logger
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.positions_file = self.data_dir / "positions.json"
        self.history_file = self.data_dir / "trade_history.json"
        self.previous_response_file = self.data_dir / "previous_response.json"
        
        self.max_memory = max_memory
        # Memory is kept in RAM only, built from recent history
        self.memory = self._build_memory_from_history()
    
    # ==================== Position Management ====================
    
    def save_position(self, position: Optional[Position]) -> None:
        """Save current position to disk."""
        try:
            if position is None:
                if self.positions_file.exists():
                    self.positions_file.unlink()
                return
            
            data = {
                "entry_price": position.entry_price,
                "stop_loss": position.stop_loss,
                "take_profit": position.take_profit,
                "size": position.size,
                "entry_time": position.entry_time.isoformat(),
                "confidence": position.confidence,
                "direction": position.direction,
                "symbol": position.symbol,
            }
            
            with open(self.positions_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.debug(f"Saved position: {position.direction} {position.symbol}")
        except Exception as e:
            self.logger.error(f"Error saving position: {e}")
    
    def load_position(self) -> Optional[Position]:
        """Load current position from disk."""
        if not self.positions_file.exists():
            return None
        
        try:
            with open(self.positions_file, 'r') as f:
                data = json.load(f)
                return Position(
                    entry_price=data["entry_price"],
                    stop_loss=data["stop_loss"],
                    take_profit=data["take_profit"],
                    size=data["size"],
                    entry_time=datetime.fromisoformat(data["entry_time"]),
                    confidence=data.get("confidence", "MEDIUM"),
                    direction=data.get("direction", "LONG"),
                    symbol=data.get("symbol", "BTC/USDT"),
                )
        except Exception as e:
            self.logger.error(f"Error loading position: {e}")
            return None
    
    # ==================== Trade History ====================
    
    def save_trade_decision(self, decision: TradeDecision) -> None:
        """Save a trade decision to history."""
        try:
            # Add to in-memory context
            self.memory.add_decision(decision)
            
            # Add to persistent history
            history = self.load_trade_history()
            history.append(decision.to_dict())
            
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
                
            self.logger.info(f"Saved trade decision: {decision.action} @ ${decision.price:,.2f}")
        except Exception as e:
            self.logger.error(f"Error saving trade decision: {e}")
    
    def load_trade_history(self) -> List[Dict[str, Any]]:
        """Load full trade history."""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading trade history: {e}")
            return []
    
    def load_last_n_decisions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Load the last n trade decisions from history."""
        history = self.load_trade_history()
        valid_actions = {"BUY", "SELL", "CLOSE", "CLOSE_LONG", "CLOSE_SHORT"}
        
        filtered = [
            d for d in history
            if d.get("action", "").upper() in valid_actions
        ]
        
        # Sort by timestamp descending
        filtered.sort(
            key=lambda x: datetime.fromisoformat(x["timestamp"]),
            reverse=True
        )
        
        return filtered[:n]
    
    # ==================== Trading Memory ====================
    
    def _load_memory(self) -> TradingMemory:
        """Load trading memory from disk."""
        if not self.memory_file.exists():
            return TradingMemory(max_decisions=self.max_memory)
        
        try:
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
                return TradingMemory.from_list(data, self.max_memory)
        except Exception as e:
            self.logger.error(f"Error loading trading memory: {e}")
            return TradingMemory(max_decisions=self.max_memory)
    
    def _save_memory(self) -> None:
        """Save trading memory to disk."""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory.to_list(), f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving trading memory: {e}")
    
    def get_memory_context(self, current_price: Optional[float] = None) -> str:
        """Get formatted memory context for prompt injection.
        
        Args:
            current_price: Current market price for P&L calculation
            
        Returns:
            Formatted memory context with overall P&L data from all trades
        """
        # Load full trade history for accurate performance calculation
        full_history_dicts = self.load_trade_history()
        full_history = [TradeDecision.from_dict(d) for d in full_history_dicts]
        
        return self.memory.get_context_summary(current_price, full_history)
    
    def get_recent_decisions(self, n: int = 5) -> List[TradeDecision]:
        """Get recent decisions from memory."""
        return self.memory.get_recent_decisions(n)
    
    def get_last_analysis_time(self) -> Optional[datetime]:
        """Get timestamp of last analysis (from most recent decision)."""
        history = self.load_trade_history()
        if not history:
            return None
        
        try:
            last_decision = history[-1]
            return datetime.fromisoformat(last_decision["timestamp"])
        except Exception as e:
            self.logger.warning(f"Could not get last analysis time: {e}")
            return None
    
    # ==================== Previous Response ====================
    
    def save_previous_response(self, response: str) -> None:
        """Save the previous AI response."""
        try:
            with open(self.previous_response_file, 'w') as f:
                json.dump({
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving previous response: {e}")
    
    def load_previous_response(self) -> Optional[str]:
        """Load the previous AI response."""
        if not self.previous_response_file.exists():
            return None
        
        try:
            with open(self.previous_response_file, 'r') as f:
                data = json.load(f)
                return data.get("response")
        except Exception as e:
            self.logger.error(f"Error loading previous response: {e}")
            return None
    
    # ==================== P&L Calculation ====================
    
    def calculate_historical_pnl(self) -> Dict[str, float]:
        """Calculate historical P&L from trade history."""
        history = self.load_trade_history()
        
        total_pnl = 0.0
        winning_trades = 0
        losing_trades = 0
        
        for i, trade in enumerate(history):
            action = trade.get("action", "").upper()
            
            # Look for matching close
            if action in ("BUY", "SELL"):
                entry_price = trade.get("price", 0)
                direction = "LONG" if action == "BUY" else "SHORT"
                
                # Find the next close action
                for j in range(i + 1, len(history)):
                    close_trade = history[j]
                    close_action = close_trade.get("action", "").upper()
                    
                    if close_action in ("CLOSE", "CLOSE_LONG", "CLOSE_SHORT"):
                        exit_price = close_trade.get("price", 0)
                        
                        if direction == "LONG":
                            pnl = ((exit_price - entry_price) / entry_price) * 100
                        else:
                            pnl = ((entry_price - exit_price) / entry_price) * 100
                        
                        total_pnl += pnl
                        if pnl > 0:
                            winning_trades += 1
                        else:
                            losing_trades += 1
                        break
        
        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        return {
            "total_pnl": total_pnl,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "total_trades": total_trades,
            "win_rate": win_rate,
        }
    
    # ==================== Memory Management ====================
    
    def _build_memory_from_history(self) -> TradingMemory:
        """Build TradingMemory from recent trade history.
        
        Returns:
            TradingMemory instance with recent decisions loaded
        """
        memory = TradingMemory(max_decisions=self.max_memory)
        
        history = self.load_trade_history()
        if not history:
            return memory
        
        # Load the most recent decisions (up to max_memory)
        recent_history = history[-self.max_memory:]
        
        for trade_data in recent_history:
            try:
                decision = TradeDecision.from_dict(trade_data)
                memory.add_decision(decision)
            except Exception as e:
                self.logger.warning(f"Could not load decision from history: {e}")
        
        self.logger.debug(f"Built memory with {len(memory.decisions)} decisions from history")
        return memory
