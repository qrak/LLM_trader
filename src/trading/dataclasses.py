"""Dataclasses for trading system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


# ==================== Trading Brain Dataclasses ====================

@dataclass(slots=True)
class TradingInsight:
    """Single distilled trading lesson.
    
    Represents a learned insight from closed trades, categorized by type
    and associated with specific market conditions.
    """
    lesson: str                      # Concise insight (max 400 chars)
    category: str                    # STOP_LOSS, ENTRY_TIMING, RISK_MANAGEMENT, MARKET_REGIME
    condition: str                   # Market condition when learned (e.g., "Downtrend + High Vol")
    trade_count: int                 # Number of trades validating this insight
    last_validated: datetime         # Last time this lesson proved relevant
    confidence_impact: str           # HIGH, MEDIUM, LOW - which confidence level this affects
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "lesson": self.lesson,
            "category": self.category,
            "condition": self.condition,
            "trade_count": self.trade_count,
            "last_validated": self.last_validated.isoformat(),
            "confidence_impact": self.confidence_impact,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingInsight':
        """Create TradingInsight from dictionary."""
        return cls(
            lesson=data["lesson"],
            category=data["category"],
            condition=data["condition"],
            trade_count=data["trade_count"],
            last_validated=datetime.fromisoformat(data["last_validated"]),
            confidence_impact=data["confidence_impact"],
        )


@dataclass(slots=True)
class ConfidenceStats:
    """Performance statistics per confidence level.
    
    Tracks win/loss ratios and P&L to help calibrate entry standards
    for different confidence levels.
    """
    level: str                       # HIGH, MEDIUM, LOW
    total_trades: int = 0
    winning_trades: int = 0
    avg_pnl_pct: float = 0.0
    win_rate: float = 0.0
    
    def update(self, is_win: bool, pnl_pct: float) -> None:
        """Update stats with new trade result."""
        self.total_trades += 1
        if is_win:
            self.winning_trades += 1
        # Rolling average P&L
        self.avg_pnl_pct = ((self.avg_pnl_pct * (self.total_trades - 1)) + pnl_pct) / self.total_trades
        self.win_rate = (self.winning_trades / self.total_trades) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "level": self.level,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "avg_pnl_pct": self.avg_pnl_pct,
            "win_rate": self.win_rate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfidenceStats':
        """Create ConfidenceStats from dictionary."""
        stats = cls(level=data["level"])
        stats.total_trades = data.get("total_trades", 0)
        stats.winning_trades = data.get("winning_trades", 0)
        stats.avg_pnl_pct = data.get("avg_pnl_pct", 0.0)
        stats.win_rate = data.get("win_rate", 0.0)
        return stats


@dataclass(slots=True)
class FactorStats:
    """Performance statistics per confluence factor bucket.
    
    Tracks win/loss ratios for specific factor score ranges
    to learn which confluence factors correlate with winning trades.
    """
    factor_name: str                 # e.g., "volume_support"
    bucket: str                      # LOW (0-30), MEDIUM (31-69), HIGH (70-100)
    total_trades: int = 0
    winning_trades: int = 0
    avg_score: float = 0.0           # Average score when this bucket triggered
    avg_pnl_pct: float = 0.0
    win_rate: float = 0.0
    
    def update(self, is_win: bool, pnl_pct: float, score: float) -> None:
        """Update stats with new trade result."""
        self.total_trades += 1
        if is_win:
            self.winning_trades += 1
        # Rolling averages
        self.avg_score = ((self.avg_score * (self.total_trades - 1)) + score) / self.total_trades
        self.avg_pnl_pct = ((self.avg_pnl_pct * (self.total_trades - 1)) + pnl_pct) / self.total_trades
        self.win_rate = (self.winning_trades / self.total_trades) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "factor_name": self.factor_name,
            "bucket": self.bucket,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "avg_score": self.avg_score,
            "avg_pnl_pct": self.avg_pnl_pct,
            "win_rate": self.win_rate,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FactorStats':
        """Create FactorStats from dictionary."""
        stats = cls(
            factor_name=data["factor_name"],
            bucket=data["bucket"]
        )
        stats.total_trades = data.get("total_trades", 0)
        stats.winning_trades = data.get("winning_trades", 0)
        stats.avg_score = data.get("avg_score", 0.0)
        stats.avg_pnl_pct = data.get("avg_pnl_pct", 0.0)
        stats.win_rate = data.get("win_rate", 0.0)
        return stats


@dataclass(slots=True)
class TradingBrain:
    """Bounded memory system for distilled trading insights.
    
    Stores learned trading wisdom from closed trades in a fixed-size structure.
    Uses FIFO eviction with category balancing to maintain diversity.
    Now includes confluence factor performance tracking for adaptive learning.
    """
    insights: List[TradingInsight] = field(default_factory=list)
    confidence_stats: Dict[str, ConfidenceStats] = field(default_factory=lambda: {
        'HIGH': ConfidenceStats('HIGH'),
        'MEDIUM': ConfidenceStats('MEDIUM'),
        'LOW': ConfidenceStats('LOW')
    })
    factor_performance: Dict[str, FactorStats] = field(default_factory=dict)  # key: "factor_bucket" e.g. "volume_support_HIGH"
    last_updated: datetime = field(default_factory=datetime.now)
    total_closed_trades: int = 0
    max_insights: int = 10           # Fixed size - FIFO eviction
    min_sample_size: int = 5         # Minimum trades before insight is "Validated"
    
    # Category limits for diversity (class-level constant)
    MAX_PER_CATEGORY: Dict[str, int] = field(default_factory=lambda: {
        'STOP_LOSS': 3,
        'ENTRY_TIMING': 3,
        'RISK_MANAGEMENT': 2,
        'MARKET_REGIME': 2
    })
    
    def add_insight(self, insight: TradingInsight) -> None:
        """Add insight with FIFO eviction and category balancing."""
        # Count insights per category
        category_counts = {}
        for ins in self.insights:
            category_counts[ins.category] = category_counts.get(ins.category, 0) + 1
        
        # Check if category limit reached
        category_limit = self.MAX_PER_CATEGORY.get(insight.category, 2)
        if category_counts.get(insight.category, 0) >= category_limit:
            # Evict oldest insight in this category
            for i, ins in enumerate(self.insights):
                if ins.category == insight.category:
                    self.insights.pop(i)
                    break
        elif len(self.insights) >= self.max_insights:
            # Evict oldest insight from most-represented category
            max_category = max(category_counts, key=category_counts.get) if category_counts else None
            if max_category:
                for i, ins in enumerate(self.insights):
                    if ins.category == max_category:
                        self.insights.pop(i)
                        break
            else:
                self.insights.pop(0)
        
        self.insights.append(insight)
        self.last_updated = datetime.now()
    
    def update_confidence_stats(self, confidence: str, is_win: bool, pnl_pct: float) -> None:
        """Update confidence level statistics with new trade result."""
        # Normalize confidence level
        level = confidence.upper() if confidence else "MEDIUM"
        if level not in self.confidence_stats:
            level = "MEDIUM"  # Default fallback
        
        self.confidence_stats[level].update(is_win, pnl_pct)
        self.total_closed_trades += 1
        self.last_updated = datetime.now()
    
    def get_insights_by_category(self, category: str) -> List[TradingInsight]:
        """Get all insights for a specific category."""
        return [ins for ins in self.insights if ins.category == category]
    
    def get_confidence_recommendation(self) -> Optional[str]:
        """Generate recommendation based on confidence calibration.
        
        Returns insight if calibration suggests adjustments needed.
        """
        high_stats = self.confidence_stats.get('HIGH')
        medium_stats = self.confidence_stats.get('MEDIUM')
        
        if high_stats and high_stats.total_trades >= 5:
            if high_stats.win_rate < 60:
                return f"HIGH confidence win rate is only {high_stats.win_rate:.0f}% - increase entry criteria"
            if medium_stats and medium_stats.total_trades >= 5:
                if medium_stats.win_rate > high_stats.win_rate:
                    return "MEDIUM confidence outperforming HIGH - current HIGH standards may be too loose"
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "insights": [ins.to_dict() for ins in self.insights],
            "confidence_stats": {k: v.to_dict() for k, v in self.confidence_stats.items()},
            "factor_performance": {k: v.to_dict() for k, v in self.factor_performance.items()},
            "last_updated": self.last_updated.isoformat(),
            "total_closed_trades": self.total_closed_trades,
            "max_insights": self.max_insights,
            "min_sample_size": self.min_sample_size,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingBrain':
        """Create TradingBrain from dictionary."""
        brain = cls()
        brain.insights = [TradingInsight.from_dict(ins) for ins in data.get("insights", [])]
        brain.confidence_stats = {
            k: ConfidenceStats.from_dict(v) 
            for k, v in data.get("confidence_stats", {}).items()
        }
        # Ensure all confidence levels exist
        for level in ['HIGH', 'MEDIUM', 'LOW']:
            if level not in brain.confidence_stats:
                brain.confidence_stats[level] = ConfidenceStats(level)
        
        # Load factor performance stats
        brain.factor_performance = {
            k: FactorStats.from_dict(v)
            for k, v in data.get("factor_performance", {}).items()
        }
        
        brain.last_updated = datetime.fromisoformat(data["last_updated"]) if "last_updated" in data else datetime.now()
        brain.total_closed_trades = data.get("total_closed_trades", 0)
        brain.max_insights = data.get("max_insights", 10)
        brain.min_sample_size = data.get("min_sample_size", 5)
        return brain


# ==================== Position Dataclasses ====================

@dataclass(frozen=True, slots=True)
class Position:
    """Represents an active trading position.
    
    Includes confluence_factors from entry for brain learning on close.
    """
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    entry_time: datetime
    confidence: str  # HIGH, MEDIUM, LOW
    direction: str   # LONG, SHORT
    symbol: str
    # Confluence factors at entry time for factor performance learning
    # Stored as tuple of (name, score) pairs for frozen dataclass compatibility
    confluence_factors: tuple = field(default_factory=tuple)
    
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
        
        lines = ["RECENT TRADING HISTORY (Last 5 Decisions):"]
        recent = self.decisions[-5:]  # Last 5 decisions for context
        
        # Calculate P&L from FULL trade history, not just recent decisions
        history_to_analyze = full_history if full_history else self.decisions
        # Ensure chronological order for P&L calculation
        history_to_analyze = sorted(history_to_analyze, key=lambda x: x.timestamp)
        total_pnl_usdt = 0.0
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
                    pnl_usdt = (decision.price - open_position.price) * open_position.position_size
                else:  # SELL
                    pnl_pct = ((open_position.price - decision.price) / open_position.price) * 100
                    pnl_usdt = (open_position.price - decision.price) * open_position.position_size
                
                total_pnl_usdt += pnl_usdt
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
            lines.append(f"OVERALL PERFORMANCE ({closed_trades} Total Closed Trades):")
            lines.append(f"- Total P&L: ${total_pnl_usdt:+,.2f} USDT ({total_pnl_pct:+.2f}%)")
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
