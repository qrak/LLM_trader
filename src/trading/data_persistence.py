"""Data persistence for trading decisions and positions."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.logger.logger import Logger
from .dataclasses import Position, TradeDecision, TradingMemory, TradingBrain, TradingInsight, ConfidenceStats


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
        self.brain_file = self.data_dir / "trading_brain.json"
        self.last_analysis_file = self.data_dir / "last_analysis.json"
        
        self.max_memory = max_memory
        # Memory is kept in RAM only, built from recent history
        self.memory = self._build_memory_from_history()
        
        # Load trading brain (bounded memory system)
        self.brain = self._load_brain()
    
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
    
    def save_last_analysis_time(self, timestamp: Optional[datetime] = None) -> None:
        """Save the timestamp of the last successful analysis.
        
        Args:
            timestamp: Timestamp to save (defaults to now)
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            with open(self.last_analysis_file, 'w') as f:
                json.dump({
                    "timestamp": timestamp.isoformat()
                }, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving last analysis time: {e}")
    
    def get_last_analysis_time(self) -> Optional[datetime]:
        """Get timestamp of last successful analysis."""
        if not self.last_analysis_file.exists():
            return None
        
        try:
            with open(self.last_analysis_file, 'r') as f:
                data = json.load(f)
                return datetime.fromisoformat(data["timestamp"])
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

    # ==================== Trading Brain ====================
    
    def _load_brain(self) -> TradingBrain:
        """Load trading brain from disk."""
        if not self.brain_file.exists():
            return TradingBrain()
        
        try:
            with open(self.brain_file, 'r') as f:
                data = json.load(f)
                brain = TradingBrain.from_dict(data)
                self.logger.debug(f"Loaded trading brain with {len(brain.insights)} insights, {brain.total_closed_trades} closed trades")
                return brain
        except Exception as e:
            self.logger.error(f"Error loading trading brain: {e}")
            return TradingBrain()
    
    def save_brain(self) -> None:
        """Save trading brain to disk."""
        try:
            with open(self.brain_file, 'w') as f:
                json.dump(self.brain.to_dict(), f, indent=2)
            self.logger.debug(f"Saved trading brain with {len(self.brain.insights)} insights")
        except Exception as e:
            self.logger.error(f"Error saving trading brain: {e}")
    
    def get_brain_context(self) -> str:
        """Generate formatted brain context for prompt injection.
        
        Returns:
            Formatted string with confidence calibration and distilled insights.
        """
        if self.brain.total_closed_trades == 0:
            return ""  # No context if no closed trades
        
        lines = [
            "=",
            f"LEARNED TRADING WISDOM (Distilled from {self.brain.total_closed_trades} closed trades):",
            "=",
            "",
            "CONFIDENCE CALIBRATION:",
        ]
        
        # Format confidence stats
        for level in ['HIGH', 'MEDIUM', 'LOW']:
            stats = self.brain.confidence_stats.get(level)
            if stats and stats.total_trades > 0:
                lines.append(
                    f"- {level} Confidence: Win Rate {stats.win_rate:.0f}% "
                    f"({stats.winning_trades}/{stats.total_trades} trades) | Avg P&L: {stats.avg_pnl_pct:+.2f}%"
                )
        
        # Add confidence recommendation if available
        recommendation = self.brain.get_confidence_recommendation()
        if recommendation:
            lines.append(f"  → INSIGHT: {recommendation}")
        
        # Add distilled insights by category
        if self.brain.insights:
            lines.extend(["", "DISTILLED INSIGHTS (Most Recent):"])
            
            # Group insights by category for better organization
            categories_order = ['STOP_LOSS', 'ENTRY_TIMING', 'RISK_MANAGEMENT', 'MARKET_REGIME']
            insight_count = 0
            
            for category in categories_order:
                category_insights = self.brain.get_insights_by_category(category)
                for insight in category_insights[-2:]:  # Max 2 per category in context
                    insight_count += 1
                    days_ago = (datetime.now() - insight.last_validated).days
                    lines.append(
                        f"{insight_count}. [{insight.category}] {insight.lesson} | "
                        f"Validated: {insight.trade_count} trades | "
                        f"Condition: {insight.condition}"
                    )
        
        lines.extend([
            "",
            "APPLY THESE INSIGHTS: Avoid repeating documented mistakes. Adjust confidence based on calibration data.",
            "=" * 60,
        ])
        
        return "\n".join(lines)
    
    def update_brain_from_closed_trade(
        self,
        position: Position,
        close_price: float,
        close_reason: str,
        entry_decision: Optional[TradeDecision] = None,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> None:
        """Extract insights from a closed trade and update brain.
        
        Uses rule-based extraction to generate insights from trade outcomes.
        
        Args:
            position: The closed position
            close_price: Price at which position was closed
            close_reason: Why position was closed (stop_loss, take_profit, analysis_signal)
            entry_decision: Original entry decision (for reasoning context)
            market_conditions: Market state at close (trend, volatility, etc.)
        """
        # Calculate P&L
        pnl_pct = position.calculate_pnl(close_price)
        is_win = pnl_pct > 0
        
        # Update confidence statistics
        self.brain.update_confidence_stats(position.confidence, is_win, pnl_pct)
        
        # Build market condition string
        conditions = market_conditions or {}
        condition_str = self._build_condition_string(conditions)
        
        # Extract insights based on trade outcome
        insights = self._extract_insights_from_trade(
            position=position,
            close_price=close_price,
            close_reason=close_reason,
            pnl_pct=pnl_pct,
            is_win=is_win,
            condition_str=condition_str,
            entry_decision=entry_decision
        )
        
        # Add new insights to brain
        for insight in insights:
            self.brain.add_insight(insight)
        
        # Save updated brain
        self.save_brain()
        
        self.logger.info(
            f"Updated brain: {len(insights)} insight(s) added from "
            f"{position.direction} trade ({close_reason}, P&L: {pnl_pct:+.2f}%)"
        )
    
    def _build_condition_string(self, conditions: Dict[str, Any]) -> str:
        """Build human-readable market condition string.
        
        Args:
            conditions: Dictionary with market indicators
            
        Returns:
            Formatted condition string (e.g., "Downtrend + High Vol")
        """
        parts = []
        
        # Trend direction
        trend = conditions.get('trend_direction', '').upper()
        if trend in ('BULLISH', 'UP', 'UPTREND'):
            parts.append("Uptrend")
        elif trend in ('BEARISH', 'DOWN', 'DOWNTREND'):
            parts.append("Downtrend")
        else:
            parts.append("Sideways")
        
        # Trend strength (ADX)
        adx = conditions.get('adx', 0)
        if adx > 30:
            parts.append("Strong Trend")
        elif adx < 20:
            parts.append("Weak Trend")
        
        # Volatility
        vol = conditions.get('volatility', '').upper()
        if vol == 'HIGH' or conditions.get('atr_percentile', 0) > 70:
            parts.append("High Vol")
        elif vol == 'LOW' or conditions.get('atr_percentile', 0) < 30:
            parts.append("Low Vol")
        
        return " + ".join(parts) if parts else "Unknown"
    
    def _extract_insights_from_trade(
        self,
        position: Position,
        close_price: float,
        close_reason: str,
        pnl_pct: float,
        is_win: bool,
        condition_str: str,
        entry_decision: Optional[TradeDecision]
    ) -> List[TradingInsight]:
        """Rule-based insight extraction from closed trade.
        
        Args:
            position: Closed position
            close_price: Exit price
            close_reason: Why closed (stop_loss, take_profit, analysis_signal)
            pnl_pct: P&L percentage
            is_win: Whether trade was profitable
            condition_str: Market conditions at entry
            entry_decision: Original entry decision
            
        Returns:
            List of extracted insights
        """
        insights = []
        now = datetime.now()
        
        # Calculate risk metrics
        if position.direction == 'LONG':
            sl_distance_pct = ((position.entry_price - position.stop_loss) / position.entry_price) * 100
            tp_distance_pct = ((position.take_profit - position.entry_price) / position.entry_price) * 100
        else:  # SHORT
            sl_distance_pct = ((position.stop_loss - position.entry_price) / position.entry_price) * 100
            tp_distance_pct = ((position.entry_price - position.take_profit) / position.entry_price) * 100
        
        rr_ratio = tp_distance_pct / sl_distance_pct if sl_distance_pct > 0 else 0
        
        # ========== STOP LOSS ANALYSIS ==========
        if close_reason == 'stop_loss':
            # SL was hit - analyze if it was appropriate
            if sl_distance_pct < 1.5:
                insights.append(TradingInsight(
                    lesson=f"SL too tight ({sl_distance_pct:.1f}%) caused early exit. Consider 1.5-2% minimum in {condition_str}.",
                    category="STOP_LOSS",
                    condition=condition_str,
                    trade_count=1,
                    last_validated=now,
                    confidence_impact=position.confidence
                ))
            elif sl_distance_pct > 3.0:
                insights.append(TradingInsight(
                    lesson=f"Wide SL ({sl_distance_pct:.1f}%) led to large loss. Tighten to 2-3% or skip in {condition_str}.",
                    category="STOP_LOSS",
                    condition=condition_str,
                    trade_count=1,
                    last_validated=now,
                    confidence_impact=position.confidence
                ))
        
        # ========== TAKE PROFIT ANALYSIS ==========
        elif close_reason == 'take_profit':
            # TP was hit - record successful setup
            if rr_ratio >= 2:
                insights.append(TradingInsight(
                    lesson=f"Successful {rr_ratio:.1f}:1 R/R setup. Entry criteria and SL/TP placement optimal in {condition_str}.",
                    category="RISK_MANAGEMENT",
                    condition=condition_str,
                    trade_count=1,
                    last_validated=now,
                    confidence_impact=position.confidence
                ))
        
        # ========== SIGNAL CLOSE ANALYSIS ==========
        elif close_reason == 'analysis_signal':
            if is_win:
                insights.append(TradingInsight(
                    lesson=f"Proactive exit at {pnl_pct:+.1f}% preserved gains. Good read of reversal signals in {condition_str}.",
                    category="ENTRY_TIMING",
                    condition=condition_str,
                    trade_count=1,
                    last_validated=now,
                    confidence_impact=position.confidence
                ))
            elif pnl_pct < -1.0:
                insights.append(TradingInsight(
                    lesson=f"Signal close at {pnl_pct:.1f}% loss. Consider earlier exit or waiting for better setup in {condition_str}.",
                    category="ENTRY_TIMING",
                    condition=condition_str,
                    trade_count=1,
                    last_validated=now,
                    confidence_impact=position.confidence
                ))
        
        # ========== CONFIDENCE CALIBRATION INSIGHTS ==========
        # Check if confidence was miscalibrated
        confidence_stats = self.brain.confidence_stats.get(position.confidence)
        if confidence_stats and confidence_stats.total_trades >= 5:
            if position.confidence == 'HIGH' and confidence_stats.win_rate < 55:
                insights.append(TradingInsight(
                    lesson=f"HIGH confidence trades underperforming ({confidence_stats.win_rate:.0f}% win rate). Raise entry bar.",
                    category="RISK_MANAGEMENT",
                    condition="All Conditions",
                    trade_count=confidence_stats.total_trades,
                    last_validated=now,
                    confidence_impact="HIGH"
                ))
        
        # ========== MARKET REGIME INSIGHTS ==========
        # Detect regime-specific patterns
        if "Strong Trend" in condition_str:
            if position.direction == 'SHORT' and close_reason == 'stop_loss' and "Uptrend" in condition_str:
                insights.append(TradingInsight(
                    lesson=f"Shorting strong uptrends fails. Wait for trend exhaustion or momentum divergence.",
                    category="MARKET_REGIME",
                    condition=condition_str,
                    trade_count=1,
                    last_validated=now,
                    confidence_impact=position.confidence
                ))
            elif position.direction == 'LONG' and close_reason == 'stop_loss' and "Downtrend" in condition_str:
                insights.append(TradingInsight(
                    lesson=f"Buying strong downtrends fails. Wait for reversal confirmation or key support.",
                    category="MARKET_REGIME",
                    condition=condition_str,
                    trade_count=1,
                    last_validated=now,
                    confidence_impact=position.confidence
                ))
        
        return insights
