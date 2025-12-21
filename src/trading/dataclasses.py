"""Dataclasses for trading system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass(frozen=True, slots=True)
class Position:
    """Represents an active trading position."""
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    entry_time: datetime
    confidence: str  # HIGH, MEDIUM, LOW
    direction: str   # LONG, SHORT
    symbol: str
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        if self.direction == 'LONG':
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            return ((self.entry_price - current_price) / self.entry_price) * 100
    
    def is_stop_hit(self, current_price: float) -> bool:
        """Check if stop loss is hit."""
        if self.direction == 'LONG':
            return current_price <= self.stop_loss
        else:
            return current_price >= self.stop_loss
    
    def is_target_hit(self, current_price: float) -> bool:
        """Check if take profit is hit."""
        if self.direction == 'LONG':
            return current_price >= self.take_profit
        else:
            return current_price <= self.take_profit


@dataclass(slots=True)
class TradeDecision:
    """Represents a trading decision from the AI."""
    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL, HOLD, CLOSE
    confidence: str  # HIGH, MEDIUM, LOW
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: float = 0.0  # Percentage of portfolio
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "action": self.action,
            "confidence": self.confidence,
            "price": self.price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size": self.position_size,
            "reasoning": self.reasoning,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeDecision':
        """Create TradeDecision from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            symbol=data["symbol"],
            action=data["action"],
            confidence=data["confidence"],
            price=data["price"],
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            position_size=data.get("position_size", 0.0),
            reasoning=data.get("reasoning", ""),
        )


@dataclass(slots=True)
class TradingMemory:
    """Rolling memory of recent trading decisions for context."""
    decisions: List[TradeDecision] = field(default_factory=list)
    max_decisions: int = 10
    
    def add_decision(self, decision: TradeDecision) -> None:
        """Add a decision to memory, maintaining max size."""
        self.decisions.append(decision)
        if len(self.decisions) > self.max_decisions:
            self.decisions.pop(0)
    
    def get_recent_decisions(self, n: int = 5) -> List[TradeDecision]:
        """Get the n most recent decisions."""
        return self.decisions[-n:]
    
    def get_context_summary(self, current_price: Optional[float] = None) -> str:
        """Generate a concise summary for prompt injection.
        
        Args:
            current_price: Current market price for P&L calculation on open positions
            
        Returns:
            Formatted summary of last 5 decisions with P&L data
        """
        if not self.decisions:
            return "No previous trading decisions."
        
        lines = ["RECENT TRADING HISTORY (Last 5 Decisions):"]
        recent = self.decisions[-5:]  # Last 5 decisions
        
        # Calculate P&L for BUY/SELL pairs
        total_pnl_usdt = 0.0
        total_pnl_pct = 0.0
        closed_trades = 0
        
        # Track open positions to calculate P&L
        open_position = None
        for i, decision in enumerate(recent):
            if decision.action in ['BUY', 'SELL']:
                open_position = decision
            elif decision.action == 'CLOSE' and open_position:
                # Calculate P&L for closed trade
                if open_position.action == 'BUY':
                    pnl_pct = ((decision.price - open_position.price) / open_position.price) * 100
                    pnl_usdt = (decision.price - open_position.price) * open_position.position_size
                else:  # SELL
                    pnl_pct = ((open_position.price - decision.price) / open_position.price) * 100
                    pnl_usdt = (open_position.price - decision.price) * open_position.position_size
                
                total_pnl_usdt += pnl_usdt
                total_pnl_pct += pnl_pct
                closed_trades += 1
                open_position = None
        
        # Format each decision
        for decision in recent:
            time_str = decision.timestamp.strftime("%Y-%m-%d %H:%M")
            reasoning = decision.reasoning[:80] + "..." if len(decision.reasoning) > 80 else decision.reasoning
            lines.append(
                f"- [{time_str}] {decision.action} @ ${decision.price:,.2f} "
                f"(Conf: {decision.confidence}) - {reasoning}"
            )
        
        # Add P&L summary if we have closed trades
        if closed_trades > 0:
            avg_pnl_pct = total_pnl_pct / closed_trades
            lines.append("")
            lines.append(f"Performance Summary (Last {closed_trades} Closed Trades):")
            lines.append(f"- Total P&L: ${total_pnl_usdt:+,.2f} USDT ({total_pnl_pct:+.2f}%)")
            lines.append(f"- Average P&L per Trade: {avg_pnl_pct:+.2f}%")
            lines.append(f"- Win Rate: {sum(1 for i in range(closed_trades) if total_pnl_pct > 0) / closed_trades * 100:.1f}%" if closed_trades > 0 else "- Win Rate: N/A")
        
        return "\n".join(lines)
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries for JSON serialization."""
        return [d.to_dict() for d in self.decisions]
    
    @classmethod
    def from_list(cls, data: List[Dict[str, Any]], max_decisions: int = 10) -> 'TradingMemory':
        """Create TradingMemory from list of dictionaries."""
        memory = cls(max_decisions=max_decisions)
        for item in data:
            memory.decisions.append(TradeDecision.from_dict(item))
        return memory
