"""Trading brain service for learning and adaptive parameters.

Handles brain state management, learning from closed trades, and providing AI context.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List

from src.logger.logger import Logger
from .persistence import TradingPersistence
from .vector_memory import VectorMemoryService
from .dataclasses import Position, TradeDecision, TradingBrain, FactorStats


class TradingBrainService:
    """Service for managing trading brain and learning from trades.
    
    Responsibilities:
    - Update brain from closed trades
    - Provide brain context for AI prompts
    - Suggest parameters based on learned data
    - Get dynamic thresholds
    """
    
    def __init__(self, logger: Logger, persistence: TradingPersistence, vector_memory: Optional[VectorMemoryService] = None):
        """Initialize trading brain service.
        
        Args:
            logger: Logger instance
            persistence: Persistence service for loading/saving brain state
            vector_memory: Optional injected vector memory (for testing)
        """
        self.logger = logger
        self.persistence = persistence
        self.brain = persistence.load_brain()
        
        if vector_memory:
            self.vector_memory = vector_memory
        else:
            self.vector_memory = VectorMemoryService(
                logger=logger,
                data_dir=str(persistence.data_dir / "brain_vector_db")
            )
    
    def update_from_closed_trade(
        self,
        position: Position,
        close_price: float,
        close_reason: str,
        entry_decision: Optional[TradeDecision] = None,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> None:
        """Extract insights from a closed trade and update brain.
        
        Args:
            position: The closed position
            close_price: Price at which position was closed
            close_reason: Why position was closed (stop_loss, take_profit, analysis_signal)
            entry_decision: Original entry decision (for reasoning context)
            market_conditions: Market state at close (trend, volatility, etc.)
        """
        pnl_pct = position.calculate_pnl(close_price)
        is_win = pnl_pct > 0
        
        self.brain.update_confidence_stats(position.confidence, is_win, pnl_pct)
        
        self._update_factor_performance(position.confluence_factors, is_win, pnl_pct)
        
        adx = position.adx_at_entry if position.adx_at_entry > 0 else 0
        if adx == 0 and market_conditions:
            adx = market_conditions.get("adx", 0)
            
        if adx > 0:
            if adx < 20:
                self.brain.adx_performance["LOW"].update(is_win, pnl_pct)
            elif adx < 25:
                self.brain.adx_performance["MEDIUM"].update(is_win, pnl_pct)
            else:
                self.brain.adx_performance["HIGH"].update(is_win, pnl_pct)
        
        if position.rr_ratio_at_entry > 0:
            if position.rr_ratio_at_entry < 1.5:
                self.brain.rr_performance["LOW"].update(is_win, pnl_pct)
            elif position.rr_ratio_at_entry < 2.0:
                self.brain.rr_performance["MEDIUM"].update(is_win, pnl_pct)
            else:
                self.brain.rr_performance["HIGH"].update(is_win, pnl_pct)
        conditions = market_conditions or {}
        condition_str = self._build_condition_string(conditions)
        
        self.persistence.save_brain(self.brain)
        
        trade_id = f"trade_{position.entry_time.isoformat()}"
        reasoning = entry_decision.reasoning if entry_decision else "N/A"
        self.vector_memory.store_experience(
            trade_id=trade_id,
            market_context=condition_str,
            outcome="WIN" if is_win else "LOSS",
            pnl_pct=pnl_pct,
            direction=position.direction,
            confidence=position.confidence,
            reasoning=reasoning,
            metadata={
                "close_reason": close_reason,
                "adx": adx,
                "rr_ratio": position.rr_ratio_at_entry,
            }
        )
        
        self.logger.info(
            f"Updated brain from {position.direction} trade ({close_reason}, P&L: {pnl_pct:+.2f}%)"
        )
    
    def get_context(
        self,
        trend_direction: str = "NEUTRAL",
        adx: float = 0,
        volatility_level: str = "MEDIUM",
        rsi_level: str = "NEUTRAL",
        macd_signal: str = "NEUTRAL",
        volume_state: str = "NORMAL",
        bb_position: str = "MIDDLE"
    ) -> str:
        """Generate formatted brain context for prompt injection using vector retrieval.
        
        Args:
            trend_direction: Current trend (BULLISH/BEARISH/NEUTRAL)
            adx: Current ADX value
            volatility_level: Current volatility (HIGH/MEDIUM/LOW)
            rsi_level: RSI state (OVERBOUGHT/STRONG/NEUTRAL/WEAK/OVERSOLD)
            macd_signal: MACD signal (BULLISH/BEARISH/NEUTRAL)
            volume_state: Volume state (ACCUMULATION/NORMAL/DISTRIBUTION)
            bb_position: Bollinger Band position (UPPER/MIDDLE/LOWER)
        
        Returns:
            Formatted string with vector-retrieved experiences and confidence calibration.
        """
        lines = []
        
        # Section 1: Confidence Calibration (lightweight, always available)
        if self.brain.total_closed_trades > 0:
            lines.extend([
                "",
                f"## TRADING BRAIN ({self.brain.total_closed_trades} closed trades)",
                "",
                "CONFIDENCE CALIBRATION:",
            ])
            
            for level in ['HIGH', 'MEDIUM', 'LOW']:
                stats = self.brain.confidence_stats.get(level)
                if stats and stats.total_trades > 0:
                    lines.append(
                        f"- {level} Confidence: Win Rate {stats.win_rate:.0f}% "
                        f"({stats.winning_trades}/{stats.total_trades} trades) | Avg P&L: {stats.avg_pnl_pct:+.2f}%"
                    )
            
            recommendation = self.brain.get_confidence_recommendation()
            if recommendation:
                lines.append(f"  → INSIGHT: {recommendation}")
        
        # Section 2: Vector-Retrieved Past Experiences (context-aware)
        vector_context = self.get_vector_context(
            trend_direction=trend_direction,
            adx=adx,
            volatility_level=volatility_level,
            rsi_level=rsi_level,
            macd_signal=macd_signal,
            volume_state=volume_state,
            bb_position=bb_position,
            k=5
        )
        
        if vector_context:
            lines.extend(["", vector_context])
        
        if lines:
            lines.extend([
                "",
                "APPLY THESE INSIGHTS: Learn from similar past trades. Weight recent wins higher.",
                "",
            ])
        
        return "\n".join(lines)
    
    def get_parameter_suggestions(
        self,
        volatility_level: str = "MEDIUM",
        confidence: str = "MEDIUM",
        current_atr_pct: float = 2.0
    ) -> Dict[str, float]:
        """Get SL/TP/size suggestions from trading brain.
        
        Args:
            volatility_level: Current market volatility (HIGH/MEDIUM/LOW)
            confidence: Current signal confidence level
            current_atr_pct: Current ATR as percentage of price
        
        Returns:
            Dict with sl_pct, tp_pct, size_pct, min_rr, and source
        """
        return self.brain.suggest_parameters(
            volatility_level=volatility_level,
            confidence=confidence,
            current_atr_pct=current_atr_pct
        )
    
    def get_dynamic_thresholds(self) -> Dict[str, Any]:
        """Get Brain-learned thresholds for response template injection.
        
        Returns:
            Dict with ADX threshold, avg SL%, min R/R, confidence threshold, safe MAE%
        """
        thresholds = self.brain.get_optimal_thresholds()
        return {
            "adx_strong_threshold": thresholds.get("adx_strong", 25),
            "avg_sl_pct": thresholds.get("avg_sl_pct", 2.5),
            "min_rr_recommended": thresholds.get("min_rr", 2.0),
            "confidence_threshold": thresholds.get("confidence_threshold", 70),
            "safe_mae_pct": thresholds.get("safe_mae_pct", 0),
        }
    
    def get_vector_context(
        self,
        trend_direction: str = "NEUTRAL",
        adx: float = 0,
        volatility_level: str = "MEDIUM",
        rsi_level: str = "NEUTRAL",
        macd_signal: str = "NEUTRAL",
        volume_state: str = "NORMAL",
        bb_position: str = "MIDDLE",
        k: int = 5
    ) -> str:
        """Get context from similar past experiences via vector retrieval.
        
        Uses semantic search to find trades in similar market conditions.
        
        Args:
            trend_direction: Current trend (BULLISH/BEARISH/NEUTRAL)
            adx: Current ADX value
            volatility_level: Current volatility (HIGH/MEDIUM/LOW)
            rsi_level: RSI state (OVERBOUGHT/STRONG/NEUTRAL/WEAK/OVERSOLD)
            macd_signal: MACD signal (BULLISH/BEARISH/NEUTRAL)
            volume_state: Volume state (ACCUMULATION/NORMAL/DISTRIBUTION)
            bb_position: Bollinger Band position (UPPER/MIDDLE/LOWER)
            k: Number of experiences to retrieve
            
        Returns:
            Formatted string with similar past trades for prompt injection.
        """
        # Build rich semantic context string
        adx_label = "High ADX" if adx >= 25 else "Low ADX" if adx < 20 else "Medium ADX"
        
        # Build comprehensive context description
        context_parts = [
            trend_direction,
            adx_label,
            f"{volatility_level} Volatility"
        ]
        
        # Add momentum indicators
        if rsi_level != "NEUTRAL":
            context_parts.append(f"RSI {rsi_level}")
        if macd_signal != "NEUTRAL":
            context_parts.append(f"MACD {macd_signal}")
        
        # Add volume and price position
        if volume_state != "NORMAL":
            context_parts.append(f"Volume {volume_state}")
        if bb_position != "MIDDLE":
            context_parts.append(f"Price at BB {bb_position}")
        
        context_query = " + ".join(context_parts)
        
        vector_context = self.vector_memory.get_context_for_prompt(context_query, k)
        
        if not vector_context:
            return ""
        
        stats = self.vector_memory.get_stats_for_context(context_query, k=20)
        if stats["total_trades"] > 0:
            vector_context += (
                f"\nLEARNED STATS FOR THIS CONTEXT:\n"
                f"- Win Rate in similar conditions: {stats['win_rate']:.0f}% "
                f"({stats['total_trades']} trades)\n"
                f"- Avg P&L: {stats['avg_pnl']:+.2f}%\n"
            )
        
        return vector_context
    
    def _build_condition_string(self, conditions: Dict[str, Any]) -> str:
        """Build human-readable market condition string."""
        parts = []
        
        trend = conditions.get('trend_direction', '').upper()
        if trend in ('BULLISH', 'UP', 'UPTREND'):
            parts.append("Uptrend")
        elif trend in ('BEARISH', 'DOWN', 'DOWNTREND'):
            parts.append("Downtrend")
        else:
            parts.append("Sideways")
        
        adx = conditions.get('adx', 0)
        if adx > 30:
            parts.append("Strong Trend")
        elif adx < 20:
            parts.append("Weak Trend")
        
        vol = conditions.get('volatility', '').upper()
        if vol == 'HIGH' or conditions.get('atr_percentile', 0) > 70:
            parts.append("High Vol")
        elif vol == 'LOW' or conditions.get('atr_percentile', 0) < 30:
            parts.append("Low Vol")
        
        return " + ".join(parts) if parts else "Unknown"
    
    def _update_factor_performance(
        self, 
        confluence_factors: tuple, 
        is_win: bool, 
        pnl_pct: float
    ) -> None:
        """Update factor performance statistics from closed trade."""
        if not confluence_factors:
            return
        
        for factor_name, score in confluence_factors:
            if score <= 30:
                bucket = "LOW"
            elif score <= 69:
                bucket = "MEDIUM"
            else:
                bucket = "HIGH"
            
            key = f"{factor_name}_{bucket}"
            
            if key not in self.brain.factor_performance:
                self.brain.factor_performance[key] = FactorStats(
                    factor_name=factor_name,
                    bucket=bucket
                )
            
            self.brain.factor_performance[key].update(is_win, pnl_pct, score)
        
        self.logger.debug(
            f"Updated factor performance for {len(confluence_factors)} factors"
        )
    

