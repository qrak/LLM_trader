"""Dataclasses for trading system."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any


from src.utils.data_utils import SerializableMixin


@dataclass(slots=True)
class Position(SerializableMixin):

    """Represents an active trading position.

    Includes confluence_factors from entry for brain learning on close.
    """
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float  # Quantity in base currency (e.g., BTC)
    entry_time: datetime
    confidence: str  # HIGH, MEDIUM, LOW
    direction: str   # LONG, SHORT
    symbol: str
    # Confluence factors at entry time for factor performance learning
    # Stored as tuple of (name, score) pairs for frozen dataclass compatibility
    confluence_factors: tuple = field(default_factory=tuple)
    # Transaction fee paid at entry (in USDT)
    entry_fee: float = 0.0
    quote_amount: float = 0.0   # Invested annual quote currency (e.g. USDT)
    # AI's suggested position size as percentage of capital (0.0-1.0)
    size_pct: float = 0.0
    # Market conditions at entry for Brain learning
    atr_at_entry: float = 0.0           # ATR value when position opened
    volatility_level: str = "MEDIUM"    # HIGH, MEDIUM, LOW (derived from ATR%)
    sl_distance_pct: float = 0.0        # abs(entry - SL) / entry as decimal
    tp_distance_pct: float = 0.0        # abs(TP - entry) / entry as decimal
    rr_ratio_at_entry: float = 0.0      # tp_distance / sl_distance
    adx_at_entry: float = 0.0           # ADX value at entry time
    rsi_at_entry: float = 50.0          # RSI value at entry time for threshold learning
    # Performance metrics (MAE/MFE)
    max_drawdown_pct: float = 0.0       # Max adverse excursion (MAE)
    max_profit_pct: float = 0.0         # Max favorable excursion (MFE)

    def calculate_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        if self.direction == 'LONG':
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            return ((self.entry_price - current_price) / self.entry_price) * 100
            
    def update_metrics(self, current_price: float) -> None:
        """Update live performance metrics (MAE/MFE)."""
        pnl = self.calculate_pnl(current_price)
        
        # Update Maximum Adverse Excursion (lowest negative P&L)
        if pnl < 0 and pnl < self.max_drawdown_pct:
            self.max_drawdown_pct = pnl
            
        # Update Maximum Favorable Excursion (highest positive P&L)
        if pnl > 0 and pnl > self.max_profit_pct:
            self.max_profit_pct = pnl

    def calculate_closing_fee(self, close_price: float, fee_percent: float) -> float:
        """Calculate the transaction fee for closing this position.
        
        Args:
            close_price: Price at which position is closed
            fee_percent: Fee percentage (default 0.075% for limit orders)
            
        Returns:
            Fee amount in USDT
        """
        return close_price * self.size * fee_percent

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
class TradeDecision(SerializableMixin):
    """Represents a trading decision from the AI."""
    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL, HOLD, CLOSE
    confidence: str  # HIGH, MEDIUM, LOW
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: float = 0.0  # AI's suggested percentage of capital (0.0-1.0)
    quote_amount: float = 0.0   # Invested quote currency amount (e.g. USDT)
    quantity: float = 0.0  # Actual quantity in base currency (e.g., BTC)
    fee: float = 0.0  # Transaction fee in quote currency (e.g. USDT)
    reasoning: str = ""


@dataclass(slots=True)
class TradingMemory(SerializableMixin):
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
    
    def get_context_summary(self, current_price: Optional[float] = None, full_history: Optional[List['TradeDecision']] = None) -> str:
        """Generate a concise summary for prompt injection.
        
        Args:
            current_price: Current market price for P&L calculation on open positions
            full_history: Complete trade history for calculating overall performance
            
        Returns:
            Formatted summary of last 5 decisions with overall P&L data from all trades
        """
        if not self.decisions:
            return "No previous trading decisions."
        
        lines = ["## Recent Trading History (Last 5 Decisions):"]
        recent = self.decisions[-5:]  # Last 5 decisions for context
        
        # Calculate P&L from FULL trade history, not just recent decisions
        history_to_analyze = full_history if full_history else self.decisions
        # Helper to ensure timezone-aware timestamps for sorting
        def _ensure_utc(dt: datetime) -> datetime:
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt
        # Ensure chronological order for P&L calculation (handle mixed tz-aware/naive)
        history_to_analyze = sorted(history_to_analyze, key=lambda x: _ensure_utc(x.timestamp))
        total_pnl_quote = 0.0
        total_pnl_pct = 0.0
        closed_trades = 0
        winning_trades = 0
        
        # Track open positions to calculate P&L across entire history
        open_position = None
        for decision in history_to_analyze:
            if decision.action in ['BUY', 'SELL']:
                open_position = decision
            elif decision.action in ['CLOSE', 'CLOSE_LONG', 'CLOSE_SHORT'] and open_position:
                # Calculate P&L for closed trade
                if open_position.action == 'BUY':
                    pnl_pct = ((decision.price - open_position.price) / open_position.price) * 100
                    pnl_quote = (decision.price - open_position.price) * open_position.quantity
                else:  # SELL
                    pnl_pct = ((open_position.price - decision.price) / open_position.price) * 100
                    pnl_quote = (open_position.price - decision.price) * open_position.quantity
                
                total_pnl_quote += pnl_quote
                total_pnl_pct += pnl_pct
                closed_trades += 1
                if pnl_pct > 0:
                    winning_trades += 1
                open_position = None
        
        # Format each recent decision for context
        for decision in recent:
            time_str = decision.timestamp.strftime("%Y-%m-%d %H:%M")
            # Keep full reasoning for better AI context (no truncation)
            lines.append(
                f"- [{time_str}] {decision.action} @ ${decision.price:,.2f} "
                f"(Conf: {decision.confidence}) - {decision.reasoning}"
            )
        
        # Add overall performance summary from ALL closed trades
        if closed_trades > 0:
            avg_pnl_pct = total_pnl_pct / closed_trades
            win_rate = (winning_trades / closed_trades) * 100
            lines.append("")
            lines.append(f"## Overall Performance ({closed_trades} Total Closed Trades):")
            lines.append(f"- Total P&L: ${total_pnl_quote:+,.2f} ({total_pnl_pct:+.2f}%)")
            lines.append(f"- Average P&L per Trade: {avg_pnl_pct:+.2f}%")
            lines.append(f"- Win Rate: {win_rate:.1f}% ({winning_trades}/{closed_trades} trades)")
        
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


@dataclass(slots=True)
class VectorSearchResult(SerializableMixin):
    """Represents a search result from VectorMemory."""
    id: str
    document: str
    similarity: float
    recency: float
    hybrid_score: float
    metadata: Dict[str, Any]


@dataclass(slots=True)
class RiskAssessment(SerializableMixin):
    """Represents the calculated risk parameters for a trade."""
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    size_pct: float
    quote_amount: float
    entry_fee: float
    sl_distance_pct: float
    tp_distance_pct: float
    rr_ratio: float
    volatility_level: str


@dataclass(slots=True)
class ConfidenceLevelStats(SerializableMixin):
    """Statistics for a single confidence level (HIGH/MEDIUM/LOW)."""
    win_rate: float
    avg_pnl: float
    total_trades: int


@dataclass(slots=True)
class ADXBucketStats(SerializableMixin):
    """Performance statistics for an ADX range bucket."""
    bucket: str  # e.g., "0-20", "20-40"
    win_rate: float
    avg_pnl: float
    total_trades: int


@dataclass(slots=True)
class FactorPerformance(SerializableMixin):
    """Performance metrics for a confluence factor."""
    factor_name: str
    win_rate: float
    avg_score: float
    sample_size: int


@dataclass(slots=True)
class SemanticRule(SerializableMixin):
    """A semantic trading rule learned from trade clusters."""
    rule_id: str
    rule_text: str
    win_rate: Optional[float] = None
    source_trades: Optional[int] = None
    created_at: Optional[datetime] = None
    similarity: float = 0.0


@dataclass(slots=True)
class ClosedTradeResult(SerializableMixin):
    """Result of a closed trade for statistics calculation."""
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_quote: float
    quantity: float
    direction: str  # LONG, SHORT
