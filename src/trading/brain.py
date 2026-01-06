"""Trading brain service for learning and adaptive parameters.

Handles brain state management, learning from closed trades, and providing AI context.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List

from src.logger.logger import Logger
from .persistence import TradingPersistence
from .vector_memory import VectorMemoryService
from .dataclasses import Position, TradeDecision


class TradingBrainService:
    """Service for managing trading brain and learning from trades.
    
    Responsibilities:
    - Update brain from closed trades
    - Provide brain context for AI prompts
    - Suggest parameters based on learned data
    - Get dynamic thresholds
    """
    
    def __init__(self, logger: Logger, persistence: TradingPersistence, vector_memory: Optional[VectorMemoryService] = None):
        """Initialize trading brain service (vector-only mode).

        Args:
            logger: Logger instance
            persistence: Persistence service (used for data_dir path)
            vector_memory: Optional injected vector memory (for testing)
        """
        self.logger = logger
        self.persistence = persistence

        if vector_memory:
            self.vector_memory = vector_memory
        else:
            self.vector_memory = VectorMemoryService(
                logger=logger,
                data_dir=str(persistence.data_dir / "brain_vector_db")
            )

        # Cache for computed stats (invalidated when new trades arrive)
        self._stats_cache: Dict[str, Any] = {}
        self._cache_trade_count: int = 0
        self._trade_count: int = 0
        self._reflection_interval: int = 10
    
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

        conditions = market_conditions or {}
        condition_str = self._build_condition_string(conditions)

        # Invalidate stats cache (new trade added)
        self._stats_cache = {}
        
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
                "adx_at_entry": position.adx_at_entry,
                "rsi_at_entry": position.rsi_at_entry,
                "atr_at_entry": position.atr_at_entry,
                "volatility_level": position.volatility_level,
                "sl_distance_pct": position.sl_distance_pct,
                "tp_distance_pct": position.tp_distance_pct,
                "rr_ratio": position.rr_ratio_at_entry,
                "max_drawdown_pct": position.max_drawdown_pct,
                "max_profit_pct": position.max_profit_pct,
                **self._extract_factor_scores(position.confluence_factors),
            }
        )
        
        self.logger.info(
            f"Updated brain from {position.direction} trade ({close_reason}, P&L: {pnl_pct:+.2f}%)"
        )

        self._trade_count += 1
        if self._trade_count % self._reflection_interval == 0:
            self._trigger_reflection()
    
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

        exp_count = self.vector_memory.experience_count
        if exp_count > 0:
            lines.extend([
                "",
                f"## TRADING BRAIN ({exp_count} closed trades)",
                "",
                "CONFIDENCE CALIBRATION:",
            ])

            conf_stats = self._get_cached_stats(
                "confidence", self.vector_memory.compute_confidence_stats
            )
            for level in ['HIGH', 'MEDIUM', 'LOW']:
                stats = conf_stats.get(level, {})
                if stats.get("total_trades", 0) > 0:
                    lines.append(
                        f"- {level} Confidence: Win Rate {stats['win_rate']:.0f}% "
                        f"({stats['winning_trades']}/{stats['total_trades']} trades) | Avg P&L: {stats['avg_pnl_pct']:+.2f}%"
                    )

            recommendation = self.vector_memory.get_confidence_recommendation()
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

        semantic_rules = self.vector_memory.get_active_rules(n_results=3)
        if semantic_rules:
            lines.extend([
                "LEARNED TRADING RULES (from reflection on past trades):",
            ])
            for rule in semantic_rules:
                lines.append(f"- {rule['text']}")
            lines.append("")

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
        # Use ATR-based defaults with confidence adjustments
        recommendations = {
            "sl_pct": current_atr_pct * 2 / 100,
            "tp_pct": current_atr_pct * 4 / 100,
            "size_pct": 0.02,
            "min_rr": 2.0,
            "source": "atr_fallback"
        }

        # Adjust based on confidence
        confidence_map = {"HIGH": 0.03, "MEDIUM": 0.02, "LOW": 0.01}
        recommendations["size_pct"] = confidence_map.get(confidence.upper(), 0.02)

        return recommendations

    def get_dynamic_thresholds(self) -> Dict[str, Any]:
        """Get Brain-learned thresholds from vector store.

        Returns:
            Dict with ADX threshold, avg SL%, min R/R, confidence threshold, safe MAE%.
        """
        thresholds = self._get_cached_stats(
            "thresholds", self.vector_memory.compute_optimal_thresholds
        )
        return {
            "adx_strong_threshold": thresholds.get("adx_strong_threshold", 25),
            "avg_sl_pct": thresholds.get("avg_sl_pct", 2.5),
            "min_rr_recommended": thresholds.get("min_rr_recommended", 2.0),
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

    def _get_cached_stats(self, key: str, compute_fn) -> Dict[str, Any]:
        """Get stats from cache or compute and cache them.

        Args:
            key: Cache key for the stats type.
            compute_fn: Function to call if cache miss.

        Returns:
            Computed or cached statistics.
        """
        current_count = self.vector_memory.experience_count
        if current_count != self._cache_trade_count:
            self._stats_cache = {}
            self._cache_trade_count = current_count

        if key not in self._stats_cache:
            self._stats_cache[key] = compute_fn()

        return self._stats_cache[key]

    def _extract_factor_scores(self, confluence_factors: tuple) -> Dict[str, float]:
        """Extract factor scores into flat dict for vector metadata."""
        scores: Dict[str, float] = {}
        if not confluence_factors:
            return scores
        for factor_name, score in confluence_factors:
            clean_name = factor_name.replace(" ", "_").lower()
            scores[f"{clean_name}_score"] = float(score)
        return scores

    def _trigger_reflection(self) -> None:
        """Reflect on recent trades and synthesize semantic rules.

        Called automatically every N trades. Analyzes winning trade patterns
        and stores insights as reusable rules.
        """
        try:
            experiences = self.vector_memory.retrieve_similar_experiences(
                "WIN trade analysis", k=20, use_decay=True
            )

            wins = [e for e in experiences if e["metadata"].get("outcome") == "WIN"]
            if len(wins) < 5:
                self.logger.debug("Not enough winning trades for reflection")
                return

            pattern_counts: Dict[str, int] = {}
            for win in wins:
                meta = win["metadata"]
                regime = meta.get("market_regime", "NEUTRAL")
                adx = meta.get("adx_at_entry", 0)
                direction = meta.get("direction", "UNKNOWN")

                adx_label = "HIGH_ADX" if adx >= 25 else "LOW_ADX" if adx < 20 else "MED_ADX"
                pattern_key = f"{direction}_{regime}_{adx_label}"
                pattern_counts[pattern_key] = pattern_counts.get(pattern_key, 0) + 1

            if not pattern_counts:
                return

            best_pattern = max(pattern_counts.items(), key=lambda x: x[1])
            pattern_key, count = best_pattern
            if count < 3:
                return

            parts = pattern_key.split("_")
            direction = parts[0]
            regime = parts[1]
            adx_level = parts[2].replace("_", " ").title()

            rule_text = (
                f"{direction} trades perform well in {regime} market with {adx_level}. "
                f"({count} recent wins follow this pattern)"
            )

            rule_id = f"rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.vector_memory.store_semantic_rule(
                rule_id=rule_id,
                rule_text=rule_text,
                metadata={
                    "source_pattern": pattern_key,
                    "source_win_count": count,
                    "total_analyzed": len(wins),
                }
            )

            self.logger.info(f"Reflection complete: stored rule '{rule_text}'")

        except Exception as e:
            self.logger.warning(f"Reflection failed: {e}")
