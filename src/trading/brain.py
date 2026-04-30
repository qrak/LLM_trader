"""Trading brain service for learning and adaptive parameters.

Handles brain state management, learning from closed trades, and providing AI context.
"""

from datetime import datetime
from collections import Counter
from typing import Optional, Dict, Any, TYPE_CHECKING
from uuid import uuid4

from src.logger.logger import Logger
<<<<<<< HEAD
from src.utils.indicator_classifier import classify_adx_label, classify_rsi_label
=======
from src.utils.indicator_classifier import (
    build_context_string_from_classified_values,
    build_exit_execution_context_from_position,
    build_query_document_from_classified_values,
    classify_adx_label,
    classify_rsi_label,
)
>>>>>>> main
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
<<<<<<< HEAD
=======
        exit_execution_context = build_exit_execution_context_from_position(position)
>>>>>>> main

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
<<<<<<< HEAD
            order_book_bias=conditions.get("order_book_bias", "BALANCED")
=======
            order_book_bias=conditions.get("order_book_bias", "BALANCED"),
            exit_execution_context=exit_execution_context,
>>>>>>> main
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
<<<<<<< HEAD
            symbol=getattr(position, "symbol", ""),
=======
            symbol=position.symbol,
>>>>>>> main
            close_reason=close_reason,
            metadata={
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
                "market_sentiment": conditions.get("market_sentiment", ""),
                "order_book_bias": conditions.get("order_book_bias", ""),
                "macd_signal": conditions.get("macd_signal", ""),
                "bb_pos": conditions.get("bb_position", ""),
<<<<<<< HEAD
=======
                **exit_execution_context,
>>>>>>> main
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
        rsi: float = 50.0,
        volatility_level: str = "MEDIUM",
        rsi_level: str = "NEUTRAL",
        macd_signal: str = "NEUTRAL",
        volume_state: str = "NORMAL",
        bb_position: str = "MIDDLE",
        is_weekend: bool = False,
        market_sentiment: str = "NEUTRAL",
        order_book_bias: str = "BALANCED",
        exit_execution_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate formatted brain context for prompt injection using vector retrieval.

        Args:
            trend_direction: Current trend (BULLISH/BEARISH/NEUTRAL)
            adx: Current ADX value
            rsi: Current RSI numeric value
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
            rsi=rsi,
            volatility_level=volatility_level,
            rsi_level=rsi_level,
            macd_signal=macd_signal,
            volume_state=volume_state,
            bb_position=bb_position,
            is_weekend=is_weekend,
            market_sentiment=market_sentiment,
            order_book_bias=order_book_bias,
            exit_execution_context=exit_execution_context,
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
                "- REGIME MISMATCH: If a retrieved experience's Context or Match Factors show a fundamentally different regime (e.g., High ADX vs current Low ADX, different volatility level marked ⚠️), treat it as informational only — not as a statistical prior for confidence adjustment.",
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
        adx_label = classify_adx_label(adx)
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
            # Metadata for origin labeling
            "trade_count": self.vector_memory.trade_count,
            "learned_keys": list(thresholds.keys()),
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
<<<<<<< HEAD
=======
        exit_execution_context: Optional[Dict[str, Any]] = None,
>>>>>>> main
    ) -> str:
        """Build rich semantic context string for vector storage and retrieval.

        This unified method ensures that the context stored in memory matches
        the format of the context used for querying, maximizing vector similarity.
        """
<<<<<<< HEAD
        # Build rich semantic context string
        adx_label = classify_adx_label(adx)

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
=======
        return build_context_string_from_classified_values(
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
            exit_execution_context=exit_execution_context,
        )
>>>>>>> main

    def _build_query_document(
        self,
        trend_direction: str,
        adx: float,
        rsi: float,
        volatility_level: str,
        rsi_level: str,
        macd_signal: str,
        volume_state: str,
        bb_position: str,
<<<<<<< HEAD
        market_sentiment: str,
        order_book_bias: str,
=======
        is_weekend: bool = False,
        market_sentiment: str = "NEUTRAL",
        order_book_bias: str = "BALANCED",
        exit_execution_context: Optional[Dict[str, Any]] = None,
>>>>>>> main
    ) -> str:
        """Build a query document that mirrors _build_experience_document format.

        Reduces embedding asymmetry between stored documents and query by using
        the same structural format (Indicators line with numeric values).

        Args:
            trend_direction: Current trend direction
            adx: Current ADX numeric value
            rsi: Current RSI numeric value
            volatility_level: Current volatility label
            rsi_level: RSI label (OVERBOUGHT/STRONG/NEUTRAL/WEAK/OVERSOLD)
            macd_signal: MACD signal label
            volume_state: Volume state label
            bb_position: BB position label
<<<<<<< HEAD
=======
            is_weekend: Whether current day is Saturday or Sunday
>>>>>>> main
            market_sentiment: Fear & Greed label
            order_book_bias: Order book bias label

        Returns:
            Query string formatted like stored experience documents.
        """
<<<<<<< HEAD
        adx_label = classify_adx_label(adx)
        context_str = self._build_rich_context_string(
            trend_direction=trend_direction,
            adx=adx,
=======
        return build_query_document_from_classified_values(
            trend_direction=trend_direction,
            adx=adx,
            rsi=rsi,
>>>>>>> main
            volatility_level=volatility_level,
            rsi_level=rsi_level,
            macd_signal=macd_signal,
            volume_state=volume_state,
            bb_position=bb_position,
<<<<<<< HEAD
            market_sentiment=market_sentiment,
            order_book_bias=order_book_bias,
        )

        indicator_parts = [
            f"ADX={adx:.1f} ({adx_label})",
            f"RSI={rsi:.1f} ({rsi_level})",
            f"Vol={volatility_level}",
            f"MACD={macd_signal}",
            f"BB={bb_position}",
        ]
        structure_parts = []
        if market_sentiment not in ("NEUTRAL", ""):
            structure_parts.append(f"Sentiment={market_sentiment}")
        if order_book_bias not in ("BALANCED", ""):
            structure_parts.append(f"OB={order_book_bias}")

        lines = [context_str]
        lines.append(f"Indicators: {' | '.join(indicator_parts)}")
        if structure_parts:
            lines.append(f"Structure: {' | '.join(structure_parts)}")

        return " ".join(lines)

=======
            is_weekend=is_weekend,
            market_sentiment=market_sentiment,
            order_book_bias=order_book_bias,
            exit_execution_context=exit_execution_context,
        )

>>>>>>> main
    def get_vector_context(
        self,
        trend_direction: str = "NEUTRAL",
        adx: float = 0,
        rsi: float = 50.0,
        volatility_level: str = "MEDIUM",
        rsi_level: str = "NEUTRAL",
        macd_signal: str = "NEUTRAL",
        volume_state: str = "NORMAL",
        bb_position: str = "MIDDLE",
        is_weekend: bool = False,
        market_sentiment: str = "NEUTRAL",
        order_book_bias: str = "BALANCED",
<<<<<<< HEAD
=======
        exit_execution_context: Optional[Dict[str, Any]] = None,
>>>>>>> main
        k: int = 5
    ) -> str:
        """Get context from similar past experiences via vector retrieval.

        Uses semantic search to find trades in similar market conditions.

        Args:
            trend_direction: Current trend (BULLISH/BEARISH/NEUTRAL)
            adx: Current ADX value
            rsi: Current RSI numeric value
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
<<<<<<< HEAD
            order_book_bias=order_book_bias
=======
            order_book_bias=order_book_bias,
            exit_execution_context=exit_execution_context,
>>>>>>> main
        )

        # Use richer query document for embedding search (mirrors storage format)
        query_document = self._build_query_document(
            trend_direction=trend_direction,
            adx=adx,
            rsi=rsi,
            volatility_level=volatility_level,
            rsi_level=rsi_level,
            macd_signal=macd_signal,
            volume_state=volume_state,
            bb_position=bb_position,
<<<<<<< HEAD
            market_sentiment=market_sentiment,
            order_book_bias=order_book_bias,
=======
            is_weekend=is_weekend,
            market_sentiment=market_sentiment,
            order_book_bias=order_book_bias,
            exit_execution_context=exit_execution_context,
>>>>>>> main
        )

        vector_context = self.vector_memory.get_context_for_prompt(
            query_document, k, display_context=context_query
        )

        if not vector_context:
            return ""

        stats = self.vector_memory.get_stats_for_context(query_document, k=20)
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

    def _trigger_reflection(self) -> None:
        """Reflect on recent trades and synthesize semantic rules.

        Called automatically every N trades. Analyzes winning trade patterns
        and stores insights as reusable rules.
        """
        try:
            all_metas = self.vector_memory._get_trade_metadatas(exclude_updates=True)
            win_metas = [m for m in all_metas if m.get("outcome") == "WIN"]

            if len(win_metas) < 10:
                self.logger.debug("Not enough winning trades for reflection (need 10+)")
                return

            def build_win_key(meta: Dict[str, Any]) -> str:
                regime = meta.get("market_regime", "NEUTRAL")
                adx = meta.get("adx_at_entry", 0)
                direction = meta.get("direction", "UNKNOWN")
                adx_label = "HIGH_ADX" if adx >= 25 else "LOW_ADX" if adx < 20 else "MED_ADX"
                return f"{direction}_{regime}_{adx_label}"

            pattern_counts = Counter(build_win_key(m) for m in win_metas)

            if not pattern_counts:
                return

            best_pattern = pattern_counts.most_common(1)[0]
            pattern_key, count = best_pattern
            if count < 5:
                self.logger.debug("Pattern %s rejected: only %s occurrences (need 5+)", pattern_key, count)
                return

            # Validate win rate: count all trades (wins + losses) matching this pattern
            pattern_total = sum(1 for m in all_metas if build_win_key(m) == pattern_key)
            pattern_wins = sum(1 for m in win_metas if build_win_key(m) == pattern_key)

            win_rate = pattern_wins / pattern_total if pattern_total > 0 else 0.0
            if win_rate < 0.6:
                self.logger.debug("Pattern %s rejected: win rate %s < 60%% (%s/%s trades)", pattern_key, f"{win_rate:.0%}", pattern_wins, pattern_total)
                return

            parts = pattern_key.split("_", 2)
            if len(parts) != 3:
                self.logger.debug("Pattern %s rejected: malformed key", pattern_key)
                return

            direction, regime, adx_bucket = parts
            adx_level_map = {
                "HIGH_ADX": "High ADX",
                "MED_ADX": "Med ADX",
                "LOW_ADX": "Low ADX",
            }
            adx_level = adx_level_map.get(adx_bucket, adx_bucket.replace("_", " ").title())

            rule_text = (
                f"{direction} trades perform well in {regime} market with {adx_level}. "
                f"({count} recent wins follow this pattern)"
            )

            rule_id = f"rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            stored_win_rate = round(win_rate * 100, 1) if pattern_total > 0 else 0.0
            self.vector_memory.store_semantic_rule(
                rule_id=rule_id,
                rule_text=rule_text,
                metadata={
                    "source_pattern": pattern_key,
                    "source_trades": pattern_total,
                    "win_rate": stored_win_rate,
                    "total_analyzed": len(win_metas),
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
            all_metas = self.vector_memory._get_trade_metadatas(exclude_updates=True)
            loss_metas = [m for m in all_metas if m.get("outcome") == "LOSS"]

            if len(loss_metas) < 5:
                self.logger.debug("Not enough losing trades for anti-pattern reflection (need 5+)")
                return

            def build_loss_key(meta: Dict[str, Any]) -> str:
                regime = meta.get("market_regime", "NEUTRAL")
                close_reason = meta.get("close_reason", "unknown")
                direction = meta.get("direction", "UNKNOWN")
                return f"{direction}_{regime}_{close_reason}"

            pattern_counts = Counter(build_loss_key(m) for m in loss_metas)

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
        exit_execution_context = build_exit_execution_context_from_position(position)

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
<<<<<<< HEAD
            order_book_bias=conditions.get("order_book_bias", "BALANCED")
=======
            order_book_bias=conditions.get("order_book_bias", "BALANCED"),
            exit_execution_context=exit_execution_context,
>>>>>>> main
        )
        
        update_id = f"update_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:8]}"
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
                **exit_execution_context,
            }
        )

        self.logger.debug("Tracked position update: %s at %s%% PnL", action_type, f"{current_pnl_pct:+.1f}")
