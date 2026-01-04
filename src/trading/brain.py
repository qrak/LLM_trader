"""Trading brain service for learning and adaptive parameters.

Handles brain state management, learning from closed trades, and providing AI context.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List

from src.logger.logger import Logger
from .persistence import TradingPersistence
from .dataclasses import Position, TradeDecision, TradingBrain, TradingInsight, FactorStats


class TradingBrainService:
    """Service for managing trading brain and learning from trades.
    
    Responsibilities:
    - Update brain from closed trades
    - Provide brain context for AI prompts
    - Suggest parameters based on learned data
    - Get dynamic thresholds
    """
    
    def __init__(self, logger: Logger, persistence: TradingPersistence):
        """Initialize trading brain service.
        
        Args:
            logger: Logger instance
            persistence: Persistence service for loading/saving brain state
        """
        self.logger = logger
        self.persistence = persistence
        self.brain = persistence.load_brain()
    
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
        
        if is_win and position.max_drawdown_pct < 0:
            self.brain.winning_mae.append(abs(position.max_drawdown_pct))
        
        if is_win and position.max_profit_pct > 0:
            self.brain.winning_mfe.append(position.max_profit_pct)
            
        if is_win and position.sl_distance_pct > 0:
            self.brain.winning_sl_distances.append(position.sl_distance_pct)
            self.brain.winning_sl_distances = self.brain.winning_sl_distances[-20:]
        
        conditions = market_conditions or {}
        condition_str = self._build_condition_string(conditions)
        
        insights = self._extract_insights_from_trade(
            position=position,
            close_price=close_price,
            close_reason=close_reason,
            pnl_pct=pnl_pct,
            is_win=is_win,
            condition_str=condition_str,
            entry_decision=entry_decision
        )
        
        for insight in insights:
            self.brain.add_insight(insight)
        
        self.persistence.save_brain(self.brain)
        
        self.logger.info(
            f"Updated brain: {len(insights)} insight(s) added from "
            f"{position.direction} trade ({close_reason}, P&L: {pnl_pct:+.2f}%)"
        )
    
    def get_context(
        self,
        volatility_level: str = "MEDIUM",
        confidence: str = "MEDIUM",
        current_atr_pct: float = 2.0
    ) -> str:
        """Generate formatted brain context for prompt injection.
        
        Returns:
            Formatted string with confidence calibration and distilled insights.
        """
        if self.brain.total_closed_trades == 0:
            return ""
        
        lines = [
            "=" * 60,
            f"LEARNED TRADING WISDOM (Distilled from {self.brain.total_closed_trades} closed trades):",
            "=" * 60,
            "",
            "CONFIDENCE CALIBRATION:",
        ]
        
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
        
        if self.brain.factor_performance:
            lines.extend(["", "FACTOR PERFORMANCE (Confluence Learning):"])
            
            factor_groups = {}
            for key, stats in self.brain.factor_performance.items():
                if stats.total_trades >= self.brain.min_sample_size:
                    if stats.factor_name not in factor_groups:
                        factor_groups[stats.factor_name] = []
                    factor_groups[stats.factor_name].append(stats)
            
            for factor_name, stats_list in factor_groups.items():
                best_stat = max(stats_list, key=lambda s: s.win_rate)
                if best_stat.win_rate >= 60:
                    lines.append(
                        f"- {factor_name}: {best_stat.bucket} scores → "
                        f"{best_stat.win_rate:.0f}% win rate ({best_stat.total_trades} trades)"
                    )
        
        if self.brain.insights:
            now = datetime.now()
            validated_insights = []
            provisional_insights = []
            
            for insight in self.brain.insights:
                weeks_old = (now - insight.last_validated).days / 7
                decay_weight = 0.95 ** weeks_old
                
                if insight.trade_count >= self.brain.min_sample_size:
                    validated_insights.append((insight, decay_weight))
                else:
                    provisional_insights.append((insight, decay_weight))
            
            validated_insights.sort(key=lambda x: x[1], reverse=True)
            
            if validated_insights:
                lines.extend(["", "VALIDATED INSIGHTS (Statistically Significant):"])
                for i, (insight, weight) in enumerate(validated_insights[:6]):
                    status = "★" if weight > 0.9 else "○"
                    lines.append(
                        f"{status} [{insight.category}] {insight.lesson} "
                        f"({insight.trade_count} trades)"
                    )
            
            if provisional_insights and len(validated_insights) < 3:
                lines.extend(["", "PROVISIONAL (Needs more data):"])
                for insight, _ in provisional_insights[:2]:
                    lines.append(f"  - {insight.lesson} ({insight.trade_count} trades)")
        
        if self.brain.total_closed_trades >= self.brain.min_sample_size:
            recs = self.brain.suggest_parameters(volatility_level, confidence, current_atr_pct)
            if recs["source"] != "atr_fallback":
                lines.extend([
                    "",
                    "DYNAMIC PARAMETER RECOMMENDATIONS:",
                    f"- Suggested SL: {recs['sl_pct']*100:.2f}% | TP: {recs['tp_pct']*100:.2f}%",
                    f"- Suggested position size: {recs['size_pct']*100:.1f}%",
                    f"- Minimum R/R ratio: {recs['min_rr']:.1f}:1",
                ])
                if len(self.brain.winning_mae) >= 3:
                    safe_mae = sorted(self.brain.winning_mae)[int(len(self.brain.winning_mae) * 0.75)]
                    lines.append(f"- Safe MAE: {safe_mae:.2f}% (based on {len(self.brain.winning_mae)} winning trades)")
                if len(self.brain.winning_mfe) >= 3:
                    safe_mfe = sorted(self.brain.winning_mfe)[int(len(self.brain.winning_mfe) * 0.75)]
                    lines.append(f"- Safe MFE: {safe_mfe:.2f}% (based on {len(self.brain.winning_mfe)} winning trades)")
        
        lines.extend([
            "",
            "APPLY THESE INSIGHTS: Prioritize factors with proven high win rates. Weight recent insights higher.",
            "=" * 60,
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
        """Rule-based insight extraction from closed trade."""
        insights = []
        now = datetime.now()
        
        if position.direction == 'LONG':
            sl_distance_pct = ((position.entry_price - position.stop_loss) / position.entry_price) * 100
            tp_distance_pct = ((position.take_profit - position.entry_price) / position.entry_price) * 100
        else:
            sl_distance_pct = ((position.stop_loss - position.entry_price) / position.entry_price) * 100
            tp_distance_pct = ((position.entry_price - position.take_profit) / position.entry_price) * 100
        
        rr_ratio = tp_distance_pct / sl_distance_pct if sl_distance_pct > 0 else 0
        
        if close_reason == 'stop_loss':
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
        
        elif close_reason == 'take_profit':
            if rr_ratio >= 2:
                insights.append(TradingInsight(
                    lesson=f"Successful {rr_ratio:.1f}:1 R/R setup. Entry criteria and SL/TP placement optimal in {condition_str}.",
                    category="RISK_MANAGEMENT",
                    condition=condition_str,
                    trade_count=1,
                    last_validated=now,
                    confidence_impact=position.confidence
                ))
        
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
