"""Trading brain service for learning and adaptive parameters.

Handles brain state management, learning from closed trades, and providing AI context.
"""

from datetime import datetime
from collections import Counter
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from src.logger.logger import Logger
from .vector_memory import VectorMemoryService
from .data_models import Position, TradeDecision

if TYPE_CHECKING:
    from src.managers.persistence_manager import PersistenceManager


class TradingBrainService:
    """Service for managing trading brain and learning from trades.

    Responsibilities:
    - Update brain from closed trades
    - Provide brain context for AI prompts
    - Suggest parameters based on learned data
    - Get dynamic thresholds
    """

    def __init__(
        self,
        logger: Logger,
        persistence: "PersistenceManager",
        vector_memory: VectorMemoryService,
    ):
        """Initialize trading brain service.

        Args:
            logger: Logger instance
            persistence: Persistence service
            vector_memory: Injected vector memory service (required)
        """
        self.logger = logger
        self.persistence = persistence
        self.vector_memory = vector_memory

        # Cache for computed stats (invalidated when new trades arrive)
        self._stats_cache: Dict[str, Any] = {}
        self._cache_trade_count: int = 0
        self._reflection_interval: int = 10

        # Initialize trade count from persistent storage
        # This ensures reflection triggers consistently across restarts
        self._trade_count: int = self.vector_memory.trade_count

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
            position: Closed position
            close_price: Exit price
            close_reason: Reason for closing
            entry_decision: Original entry decision (for reasoning)
            market_conditions: Market state at close (or from entry if preferred)
        """
        pnl_pct = position.calculate_pnl(close_price)
        is_win = pnl_pct > 0

        conditions = market_conditions or {}

        # Use rich context builder for consistent vector keys
        condition_str = self._build_rich_context_string(
            trend_direction=conditions.get("trend_direction", "NEUTRAL"),
            adx=float(conditions.get("adx", 0.0)),
            volatility_level=conditions.get("volatility", "MEDIUM"),
            rsi_level=conditions.get("rsi_level", "NEUTRAL"),
            macd_signal=conditions.get("macd_signal", "NEUTRAL"),
            volume_state=conditions.get("volume_state", "NORMAL"),
            bb_position=conditions.get("bb_position", "MIDDLE"),
            is_weekend=conditions.get("is_weekend", False),
            market_sentiment=conditions.get("market_sentiment", "NEUTRAL"),
            order_book_bias=conditions.get("order_book_bias", "BALANCED")
        )

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
                "fear_greed_index": conditions.get("fear_greed_index", 50),
                "market_regime": conditions.get("trend_direction", "NEUTRAL"),
                "is_weekend": conditions.get("is_weekend", False),
                "position_size_pct": position.size_pct,
                "confluence_count": self._count_strong_confluences(position.confluence_factors),
                "timeframe_alignment": conditions.get("timeframe_alignment"),
                **self._extract_factor_scores(position.confluence_factors),
            }
        )

        self.logger.info("Updated brain from %s trade (%s, P&L: %s%%)", position.direction, close_reason, f"{pnl_pct:+.2f}")

        self._trade_count += 1
        if self._trade_count % self._reflection_interval == 0:
            self._trigger_reflection()
            self._trigger_loss_reflection()

    def get_context(
        self,
        trend_direction: str = "NEUTRAL",
        adx: float = 0,
        volatility_level: str = "MEDIUM",
        rsi_level: str = "NEUTRAL",
        macd_signal: str = "NEUTRAL",
        volume_state: str = "NORMAL",
        bb_position: str = "MIDDLE",
        is_weekend: bool = False,
        market_sentiment: str = "NEUTRAL",
        order_book_bias: str = "BALANCED"
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
            is_weekend: Whether current day is Saturday or Sunday
            market_sentiment: Fear & Greed state (EXTREME_FEAR/FEAR/NEUTRAL/GREED/EXTREME_GREED)
            order_book_bias: Order book pressure (BUY_PRESSURE/SELL_PRESSURE/BALANCED)

        Returns:
            Formatted string with vector-retrieved experiences and confidence calibration.
        """
        lines = []

        exp_count = self.vector_memory.trade_count  # Excludes UPDATE entries
        if exp_count > 0:
            lines.extend([
                "",
                f"## Trading Brain ({exp_count} closed trades)",
                "",
                "### Confidence Calibration:",
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

            # Direction bias warning
            direction_bias = self.vector_memory.get_direction_bias()
            if direction_bias:
                lines.extend([
                    "",
                    "### Direction Bias Check:",
                    f"- Historical trades: {direction_bias['long_count']} LONG, {direction_bias['short_count']} SHORT",
                ])
                if direction_bias['short_count'] == 0:
                    lines.append("- ⚠️ NO SHORT TRADES IN HISTORY: Consider SHORT opportunities more carefully; lack of data means you may be missing valid setups.")

        # Section 2: Vector-Retrieved Past Experiences (context-aware)
        vector_context = self.get_vector_context(
            trend_direction=trend_direction,
            adx=adx,
            volatility_level=volatility_level,
            rsi_level=rsi_level,
            macd_signal=macd_signal,
            volume_state=volume_state,
            bb_position=bb_position,
            is_weekend=is_weekend,
            market_sentiment=market_sentiment,
            order_book_bias=order_book_bias,
            k=5
        )

        if vector_context:
            lines.extend(["", vector_context])

        # Check for limited data warning in the proper way
        has_limited_data = "⚠️ LIMITED DATA" in vector_context if vector_context else False

        if lines and not has_limited_data:
            lines.extend([
                "",
                "### Apply Insights (CoT Step 6 - Historical Evidence):",
                "- MANDATORY: If win rate in similar conditions <50%, reduce your confidence by 10 points and state this adjustment.",
                "- MANDATORY: If AVOID PATTERNS match current conditions (>50% similarity), state \"⚠️ ANTI-PATTERN MATCH\" and justify any override.",
                "- Weight recent wins higher. Check for pattern repetition that led to losses.",
                "",
            ])
        elif lines and has_limited_data:
             lines.extend([
                "",
                "NOTE: Limited historical data available. Rely on standard technical analysis for this decision.",
                "",
            ])

        # Build context for semantic rule matching
        adx_label = "High ADX" if adx >= 25 else "Low ADX" if adx < 20 else "Medium ADX"
        rule_context = f"{trend_direction} + {adx_label} + {volatility_level} Volatility"

        semantic_rules = self.vector_memory.get_relevant_rules(
            current_context=rule_context,
            n_results=3
        )
        if semantic_rules:
            lines.extend([
                "### Learned Trading Rules (relevant to current conditions):",
            ])
            for rule in semantic_rules:
                similarity = rule.get("similarity", 0)
                lines.append(f"- [{similarity:.0f}% match] {rule['text']}")
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
        # Volatility multipliers for SL/TP distances
        # High volatility = wider stops to avoid premature exits
        # Low volatility = tighter stops for better risk management
        volatility_multipliers = {
            "HIGH": {"sl": 2.5, "tp": 4.5},
            "MEDIUM": {"sl": 2.0, "tp": 4.0},
            "LOW": {"sl": 1.5, "tp": 3.0}
        }
        
        multipliers = volatility_multipliers.get(volatility_level.upper(), volatility_multipliers["MEDIUM"])
        
        recommendations = {
            "sl_pct": current_atr_pct * multipliers["sl"] / 100,
            "tp_pct": current_atr_pct * multipliers["tp"] / 100,
            "size_pct": 0.02,
            "min_rr": 2.0,
            "source": f"atr_fallback_vol_{volatility_level.lower()}"
        }

        # Adjust position size based on confidence AND volatility
        # High volatility = reduce size even further for risk management
        base_size = {"HIGH": 0.03, "MEDIUM": 0.02, "LOW": 0.01}
        volatility_size_adj = {"HIGH": 0.8, "MEDIUM": 1.0, "LOW": 1.0}  # Reduce size in high vol
        
        base = base_size.get(confidence.upper(), 0.02)
        vol_adj = volatility_size_adj.get(volatility_level.upper(), 1.0)
        recommendations["size_pct"] = base * vol_adj

        return recommendations

    def get_dynamic_thresholds(self) -> Dict[str, Any]:
        """Get Brain-learned thresholds from vector store.

        Returns:
            Dict with learned thresholds. Defaults used when insufficient data.
        """
        thresholds = self._get_cached_stats(
            "thresholds", self.vector_memory.compute_optimal_thresholds
        )
        return {
            # Core thresholds (existing)
            "adx_strong_threshold": thresholds.get("adx_strong_threshold", 25),
            "avg_sl_pct": thresholds.get("avg_sl_pct", 2.5),
            "min_rr_recommended": thresholds.get("min_rr_recommended", 2.0),
            "confidence_threshold": thresholds.get("confidence_threshold", 70),
            "safe_mae_pct": thresholds.get("safe_mae_pct", 0),
            # Extended thresholds (new - defaults are industry standard)
            "adx_weak_threshold": thresholds.get("adx_weak_threshold", 20),
            "min_confluences_weak": thresholds.get("min_confluences_weak", 4),
            "min_confluences_standard": thresholds.get("min_confluences_standard", 3),
            "position_reduce_mixed": thresholds.get("position_reduce_mixed", 0.20),
            "position_reduce_divergent": thresholds.get("position_reduce_divergent", 0.35),
            "min_position_size": thresholds.get("min_position_size", 0.10),
            "rr_borderline_min": thresholds.get("rr_borderline_min", 1.5),
            "rr_strong_setup": thresholds.get("rr_strong_setup", 2.5),
        }

    def _build_rich_context_string(
        self,
        trend_direction: str = "NEUTRAL",
        adx: float = 0,
        volatility_level: str = "MEDIUM",
        rsi_level: str = "NEUTRAL",
        macd_signal: str = "NEUTRAL",
        volume_state: str = "NORMAL",
        bb_position: str = "MIDDLE",
        is_weekend: bool = False,
        market_sentiment: str = "NEUTRAL",
        order_book_bias: str = "BALANCED",
    ) -> str:
        """Build rich semantic context string for vector storage and retrieval.

        This unified method ensures that the context stored in memory matches
        the format of the context used for querying, maximizing vector similarity.
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

        # Add new enriched context fields
        if is_weekend:
            context_parts.append("Weekend Low Volume")
        if market_sentiment not in ("NEUTRAL", ""):
            context_parts.append(f"Sentiment {market_sentiment}")
        if order_book_bias not in ("BALANCED", ""):
            context_parts.append(f"OrderBook {order_book_bias}")

        return " + ".join(context_parts)

    def get_vector_context(
        self,
        trend_direction: str = "NEUTRAL",
        adx: float = 0,
        volatility_level: str = "MEDIUM",
        rsi_level: str = "NEUTRAL",
        macd_signal: str = "NEUTRAL",
        volume_state: str = "NORMAL",
        bb_position: str = "MIDDLE",
        is_weekend: bool = False,
        market_sentiment: str = "NEUTRAL",
        order_book_bias: str = "BALANCED",
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
            is_weekend: Whether current day is Saturday or Sunday
            market_sentiment: Fear & Greed state
            order_book_bias: Order book pressure
            k: Number of experiences to retrieve

        Returns:
            Formatted string with similar past trades for prompt injection.
        """
        context_query = self._build_rich_context_string(
            trend_direction=trend_direction,
            adx=adx,
            volatility_level=volatility_level,
            rsi_level=rsi_level,
            macd_signal=macd_signal,
            volume_state=volume_state,
            bb_position=bb_position,
            is_weekend=is_weekend,
            market_sentiment=market_sentiment,
            order_book_bias=order_book_bias
        )

        vector_context = self.vector_memory.get_context_for_prompt(context_query, k)

        if not vector_context:
            return ""

        stats = self.vector_memory.get_stats_for_context(context_query, k=20)
        if stats["total_trades"] > 0:
            vector_context += (
                f"### Learned Stats for This Context:\n"
                f"- Win Rate in similar conditions: {stats['win_rate']:.0f}% "
                f"({stats['total_trades']} trades)\n"
                f"- Avg P&L: {stats['avg_pnl']:+.2f}%\n"
            )

        return vector_context



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

    def _count_strong_confluences(self, confluence_factors: tuple) -> int:
        """Count factors with score > 50 (supporting the trade)."""
        if not confluence_factors:
            return 0
        return sum(1 for _, score in confluence_factors if score > 50)

    def _count_patterns(self, experiences: List[Any], key_builder) -> Dict[str, int]:
        """Helper to count patterns in experiences."""
        return Counter(key_builder(exp.metadata) for exp in experiences)

    def _trigger_reflection(self) -> None:
        """Reflect on recent trades and synthesize semantic rules.

        Called automatically every N trades. Analyzes winning trade patterns
        and stores insights as reusable rules.
        """
        try:
            experiences = self.vector_memory.retrieve_similar_experiences(
                "recent trading experiences", k=20, use_decay=True, where={"outcome": "WIN"}
            )

            wins = experiences  # Already filtered by DB
            if len(wins) < 10:
                self.logger.debug("Not enough winning trades for reflection (need 10+)")
                return

            def build_win_key(meta):
                regime = meta.get("market_regime", "NEUTRAL")
                adx = meta.get("adx_at_entry", 0)
                direction = meta.get("direction", "UNKNOWN")
                adx_label = "HIGH_ADX" if adx >= 25 else "LOW_ADX" if adx < 20 else "MED_ADX"
                return f"{direction}_{regime}_{adx_label}"

            pattern_counts = self._count_patterns(wins, build_win_key)

            if not pattern_counts:
                return

            best_pattern = pattern_counts.most_common(1)[0]
            pattern_key, count = best_pattern
            if count < 5:
                self.logger.debug("Pattern %s rejected: only %s occurrences (need 5+)", pattern_key, count)
                return

            # Validate win rate before storing rule
            # Get all experiences (wins + losses) for this pattern to calculate win rate
            all_pattern_experiences = self.vector_memory.retrieve_similar_experiences(
                "recent trading experiences", k=50, use_decay=True
            )
            pattern_wins = sum(1 for exp in all_pattern_experiences
                             if exp.metadata.get("outcome") == "WIN"
                             and build_win_key(exp.metadata) == pattern_key)
            pattern_total = sum(1 for exp in all_pattern_experiences
                              if build_win_key(exp.metadata) == pattern_key)

            if pattern_total > 0:
                win_rate = pattern_wins / pattern_total
                if win_rate < 0.6:
                    self.logger.debug("Pattern %s rejected: win rate %s < 60%% (%s/%s trades)", pattern_key, f"{win_rate:.0%}", pattern_wins, pattern_total)
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

            self.logger.info("Reflection complete: stored rule '%s'", rule_text)

        except Exception as e:
            self.logger.warning("Reflection failed: %s", e)

    def _trigger_loss_reflection(self) -> None:
        """Reflect on losing trades and synthesize anti-patterns.

        Called automatically every N trades. Analyzes LOSS trades to identify
        conditions that consistently lead to losses so they can be avoided.
        """
        try:
            experiences = self.vector_memory.retrieve_similar_experiences(
                "recent trading experiences", k=20, use_decay=True, where={"outcome": "LOSS"}
            )

            losses = experiences
            if len(losses) < 5:
                self.logger.debug("Not enough losing trades for anti-pattern reflection (need 5+)")
                return

            def build_loss_key(meta):
                regime = meta.get("market_regime", "NEUTRAL")
                close_reason = meta.get("close_reason", "unknown")
                direction = meta.get("direction", "UNKNOWN")
                return f"{direction}_{regime}_{close_reason}"

            pattern_counts: Dict[str, int] = self._count_patterns(losses, build_loss_key)

            if not pattern_counts:
                return

            worst_pattern = pattern_counts.most_common(1)[0]
            pattern_key, count = worst_pattern
            if count < 3:
                self.logger.debug("Anti-pattern %s rejected: only %s occurrences (need 3+)", pattern_key, count)
                return

            parts = pattern_key.split("_")
            direction = parts[0]
            regime = parts[1]
            close_reason = "_".join(parts[2:])

            rule_text = (
                f"⚠️ AVOID: {direction} trades in {regime} market often hit {close_reason}. "
                f"({count} recent losses follow this pattern)"
            )

            rule_id = f"anti_rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.vector_memory.store_semantic_rule(
                rule_id=rule_id,
                rule_text=rule_text,
                metadata={
                    "rule_type": "anti_pattern",
                    "source_pattern": pattern_key,
                    "source_loss_count": count,
                }
            )

            self.logger.info("Anti-pattern reflection complete: stored rule '%s'", rule_text)

        except Exception as e:
            self.logger.warning("Loss reflection failed: %s", e)

    def track_position_update(
        self,
        position: Position,
        old_sl: float,
        old_tp: float,
        new_sl: float,
        new_tp: float,
        current_price: float,
        current_pnl_pct: float,
        market_conditions: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track position update decisions for learning.

        Args:
            position: Active position
            old_sl: Previous stop loss
            old_tp: Previous take profit
            new_sl: New stop loss
            new_tp: New take profit
            current_price: Current market price
            current_pnl_pct: Current unrealized P&L
            market_conditions: Market state at time of update
        """
        conditions = market_conditions or {}

        sl_moved = new_sl != old_sl
        tp_moved = new_tp != old_tp

        if sl_moved and not tp_moved:
            action_type = "SL_TRAIL"
        elif tp_moved and not sl_moved:
            action_type = "TP_EXTEND"
        elif sl_moved and tp_moved:
            action_type = "BOTH"
        else:
            return

        # Use rich context builder for consistent vector keys
        market_context = self._build_rich_context_string(
            trend_direction=conditions.get("trend_direction", "NEUTRAL"),
            adx=float(conditions.get("adx", 0.0)),
            volatility_level=conditions.get("volatility", "MEDIUM"),
            rsi_level=conditions.get("rsi_level", "NEUTRAL"),
            macd_signal=conditions.get("macd_signal", "NEUTRAL"),
            volume_state=conditions.get("volume_state", "NORMAL"),
            bb_position=conditions.get("bb_position", "MIDDLE"),
            is_weekend=conditions.get("is_weekend", False),
            market_sentiment=conditions.get("market_sentiment", "NEUTRAL"),
            order_book_bias=conditions.get("order_book_bias", "BALANCED")
        )
        
        update_id = f"update_{int(datetime.now().timestamp())}"
        reasoning_str = f"Moved {action_type}: SL {old_sl:.2f}→{new_sl:.2f}, TP {old_tp:.2f}→{new_tp:.2f}"

        self.vector_memory.store_experience(
            trade_id=update_id,
            market_context=market_context,
            outcome="UPDATE",
            pnl_pct=current_pnl_pct,
            direction=position.direction,
            confidence=position.confidence,
            reasoning=reasoning_str,
            metadata={
                "action_type": action_type,
                "current_price": current_price,
                "sl_change": new_sl - old_sl,
                "tp_change": new_tp - old_tp,
                "pnl_at_update": current_pnl_pct,
                "adx_at_update": conditions.get("adx", 0),
                "volatility": conditions.get("volatility", "MEDIUM"),
            }
        )

        self.logger.debug("Tracked position update: %s at %s% PnL", action_type, f"{current_pnl_pct:+.1f}")
