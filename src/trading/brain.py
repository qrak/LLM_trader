"""Trading brain service for learning and adaptive parameters.

Handles brain state management, learning from closed trades, and providing AI context.
"""

from datetime import datetime
from collections import Counter
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from uuid import uuid4

from src.logger.logger import Logger
from src.utils.indicator_classifier import (
    build_context_string_from_classified_values,
    build_exit_execution_context,
    build_exit_execution_context_from_position,
    build_query_document_from_classified_values,
    classify_adx_label,
    EXIT_EXECUTION_KEYS,
    format_exit_execution_context,
)
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

    UNKNOWN_EXIT_PROFILE = "SL unknown/unknown | TP unknown/unknown"
    UNKNOWN_EXIT_PROFILE_KEY = "sl_unknown_unknown|tp_unknown_unknown"
    DEFAULT_REFLECTION_TIMEFRAME_MINUTES = 240
    SCALPING_REFLECTION_INTERVAL = 10
    INTRADAY_REFLECTION_INTERVAL = 7
    SWING_REFLECTION_INTERVAL = 5
    POSITION_REFLECTION_INTERVAL = 3

    @classmethod
    def _derive_reflection_interval(cls, timeframe_minutes: int) -> int:
        """Derive closed-trade reflection cadence from active timeframe."""
        try:
            minutes = int(timeframe_minutes)
        except (TypeError, ValueError):
            minutes = cls.DEFAULT_REFLECTION_TIMEFRAME_MINUTES
        if minutes <= 0:
            minutes = cls.DEFAULT_REFLECTION_TIMEFRAME_MINUTES
        if minutes < 60:
            return cls.SCALPING_REFLECTION_INTERVAL
        if minutes < 240:
            return cls.INTRADAY_REFLECTION_INTERVAL
        if minutes < 1440:
            return cls.SWING_REFLECTION_INTERVAL
        return cls.POSITION_REFLECTION_INTERVAL

    def __init__(
        self,
        logger: Logger,
        persistence: "PersistenceManager",
        vector_memory: VectorMemoryService,
        exit_execution_context: Optional[Dict[str, Any]] = None,
        timeframe_minutes: int = DEFAULT_REFLECTION_TIMEFRAME_MINUTES,
    ):
        """Initialize trading brain service.

        Args:
            logger: Logger instance
            persistence: Persistence service
            vector_memory: Injected vector memory service (required)
            exit_execution_context: Configured fallback SL/TP execution context.
            timeframe_minutes: Active analysis timeframe in minutes.
        """
        self.logger = logger
        self.persistence = persistence
        self.vector_memory = vector_memory
        self._default_exit_execution_context: Dict[str, str] = build_exit_execution_context(
            **(exit_execution_context or {})
        )

        # Cache for computed stats (invalidated when new trades arrive)
        self._stats_cache: Dict[str, Any] = {}
        self._cache_trade_count: int = 0
        self._reflection_interval: int = self._derive_reflection_interval(timeframe_minutes)
        self.logger.debug(
            "Trading brain reflection interval: every %s closed trades",
            self._reflection_interval,
        )

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
        exit_execution_context = self._resolve_exit_execution_context(
            build_exit_execution_context_from_position(position)
        )
        entry_confidence = entry_decision.confidence if entry_decision else position.confidence
        entry_action = entry_decision.action if entry_decision else position.direction
        reasoning = entry_decision.reasoning if entry_decision else "N/A"

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
            order_book_bias=conditions.get("order_book_bias", "BALANCED"),
            exit_execution_context=exit_execution_context,
        )

        # Invalidate stats cache (new trade added)
        self._stats_cache = {}

        trade_id = f"trade_{position.entry_time.isoformat()}"
        self.vector_memory.store_experience(
            trade_id=trade_id,
            market_context=condition_str,
            outcome="WIN" if is_win else "LOSS",
            pnl_pct=pnl_pct,
            direction=position.direction,
            confidence=entry_confidence,
            reasoning=reasoning,
            symbol=position.symbol,
            close_reason=close_reason,
            metadata={
                "entry_action": entry_action,
                "entry_confidence": entry_confidence,
                "ai_reasoning": reasoning if reasoning != "N/A" else "",
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
                **exit_execution_context,
                **self._extract_factor_scores(position.confluence_factors),
            }
        )

        self.logger.info("Updated brain from %s trade (%s, P&L: %s%%)", position.direction, close_reason, f"{pnl_pct:+.2f}")

        self._trade_count += 1
        if self._trade_count % self._reflection_interval == 0:
            self._trigger_reflection()
            self._trigger_loss_reflection()
            self._trigger_ai_mistake_reflection()

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
            vector_context = self._replace_unknown_exit_profile_text(vector_context)
            lines.extend(["", vector_context])

        # Check for limited data warning in the proper way
        has_limited_data = "⚠️ LIMITED DATA" in vector_context if vector_context else False

        if lines and not has_limited_data:
            lines.extend([
                "",
                "### Apply Insights (CoT Step 6 - Historical Evidence):",
                "- MANDATORY: If win rate in similar conditions <50%, reduce your confidence by 10 points and state this adjustment.",
                "- MANDATORY: If AVOID PATTERNS match current conditions (>50% similarity), state \"⚠️ ANTI-PATTERN MATCH\" and justify any override.",
                "- AI MISTAKE MEMORY: If an AI-mistake rule matches, compare the current setup to the failed assumption and downgrade confidence unless the missing confirmation is now present.",
                "- EXIT EXECUTION MEMORY: Treat hard/soft SL/TP settings as part of the setup. Do not reuse a rule learned under a different exit profile without explaining the mismatch.",
                "- REGIME MISMATCH: If a retrieved experience's Context or Match Factors show a fundamentally different regime (e.g., High ADX vs current Low ADX, different volatility level marked ⚠️), treat it as informational only — not as a statistical prior for confidence adjustment.",
                "- OUTCOME BALANCE: Weight both wins AND losses. If a corrective or anti-pattern rule matches, explicitly state the adjustment you are applying (e.g. stricter confluences, higher R/R, reduced position size) before finalising your decision.",
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
                meta = rule.get("metadata", {})
                rule_type = meta.get("rule_type", "best_practice")
                type_tags = {
                    "anti_pattern": " [⚠️ AVOID]",
                    "corrective": " [⚡ IMPROVE]",
                    "ai_mistake": " [🧠 AI MISTAKE]",
                }
                type_tag = type_tags.get(rule_type, "")
                rule_text = self._render_rule_text(rule)
                lines.append(f"- [{similarity:.0f}% match]{type_tag} {rule_text}")
                failure = meta.get("failure_reason")
                if failure:
                    lines.append(f"  → Why it failed: {failure}")
                recommended = meta.get("recommended_adjustment")
                if recommended:
                    lines.append(f"  → Apply: {recommended}")
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
            "min_position_size": thresholds.get("min_position_size", 0.02),
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
        exit_execution_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build rich semantic context string for vector storage and retrieval.

        This unified method ensures that the context stored in memory matches
        the format of the context used for querying, maximizing vector similarity.
        """
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
        is_weekend: bool = False,
        market_sentiment: str = "NEUTRAL",
        order_book_bias: str = "BALANCED",
        exit_execution_context: Optional[Dict[str, Any]] = None,
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
            is_weekend: Whether current day is Saturday or Sunday
            market_sentiment: Fear & Greed label
            order_book_bias: Order book bias label

        Returns:
            Query string formatted like stored experience documents.
        """
        return build_query_document_from_classified_values(
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
        )

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
        exit_execution_context: Optional[Dict[str, Any]] = None,
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
            order_book_bias=order_book_bias,
            exit_execution_context=exit_execution_context,
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
            is_weekend=is_weekend,
            market_sentiment=market_sentiment,
            order_book_bias=order_book_bias,
            exit_execution_context=exit_execution_context,
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

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        """Return a float for optional numeric metadata values."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_key_part(value: Any, default: str = "unknown") -> str:
        """Normalize metadata values for deterministic semantic rule IDs."""
        if value is None:
            return default
        normalized = str(value).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        return normalized or default

    def _resolve_exit_execution_context(
        self,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Return SL/TP execution metadata with configured defaults filled in."""
        raw_context = metadata or {}
        context = build_exit_execution_context(
            stop_loss_type=raw_context.get("stop_loss_type"),
            stop_loss_check_interval=raw_context.get("stop_loss_check_interval"),
            take_profit_type=raw_context.get("take_profit_type"),
            take_profit_check_interval=raw_context.get("take_profit_check_interval"),
        )
        resolved: Dict[str, str] = {}
        for key in EXIT_EXECUTION_KEYS:
            value = context[key]
            if value == "unknown":
                value = self._default_exit_execution_context.get(key, "unknown")
            resolved[key] = value
        return resolved

    def _resolve_rule_exit_execution_context(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Resolve exit execution context from semantic-rule metadata."""
        return self._resolve_exit_execution_context({
            "stop_loss_type": metadata.get("dominant_stop_loss_type") or metadata.get("stop_loss_type"),
            "stop_loss_check_interval": (
                metadata.get("dominant_stop_loss_interval") or metadata.get("stop_loss_check_interval")
            ),
            "take_profit_type": metadata.get("dominant_take_profit_type") or metadata.get("take_profit_type"),
            "take_profit_check_interval": (
                metadata.get("dominant_take_profit_interval") or metadata.get("take_profit_check_interval")
            ),
        })

    def _format_exit_profile_from_context(self, context: Dict[str, str]) -> str:
        """Format a normalized SL/TP execution context."""
        return self._format_exit_profile(
            context["stop_loss_type"],
            context["stop_loss_check_interval"],
            context["take_profit_type"],
            context["take_profit_check_interval"],
        )

    def _replace_unknown_exit_profile_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Replace legacy unknown exit-profile text with resolved rule/default metadata."""
        if self.UNKNOWN_EXIT_PROFILE not in text:
            return text
        if metadata:
            replacement_profile = self._format_exit_profile_from_context(
                self._resolve_rule_exit_execution_context(metadata)
            )
        else:
            replacement_profile = self._format_exit_profile_from_context(self._default_exit_execution_context)
        if replacement_profile == self.UNKNOWN_EXIT_PROFILE:
            return text
        return text.replace(self.UNKNOWN_EXIT_PROFILE, replacement_profile)

    def _render_rule_text(self, rule: Dict[str, Any]) -> str:
        """Return rule text with legacy unknown exit profile corrected for display."""
        metadata = rule.get("metadata", {})
        return self._replace_unknown_exit_profile_text(rule.get("text", ""), metadata)

    def _dominant_exit_execution_context(self, metas: List[Dict[str, Any]]) -> Dict[str, str]:
        """Return the most common resolved SL/TP execution context for a trade group."""
        contexts = [self._resolve_exit_execution_context(meta) for meta in metas]
        if not contexts:
            return self._resolve_exit_execution_context({})
        dominant_context: Dict[str, str] = {}
        for key in EXIT_EXECUTION_KEYS:
            values = [context[key] for context in contexts]
            dominant_context[key] = Counter(values).most_common(1)[0][0]
        return dominant_context

    def _legacy_unknown_exit_pattern_key(self, pattern_key: str) -> str:
        """Return the equivalent rule key that used the old unknown exit profile."""
        parts = pattern_key.split("|")
        if len(parts) < 2:
            return pattern_key
        parts[-2:] = self.UNKNOWN_EXIT_PROFILE_KEY.split("|")
        return "|".join(parts)

    def _deactivate_legacy_unknown_exit_rule(self, rule_id_prefix: str, pattern_key: str) -> None:
        """Deactivate the stale unknown-profile rule once a resolved replacement exists."""
        legacy_pattern_key = self._legacy_unknown_exit_pattern_key(pattern_key)
        if legacy_pattern_key == pattern_key:
            return
        legacy_rule_id = f"{rule_id_prefix}_{self._sanitize_rule_key(legacy_pattern_key)}"
        self.vector_memory.deactivate_semantic_rules([legacy_rule_id])

    def refresh_semantic_rules_if_stale(self) -> None:
        """Refresh semantic rules once when active rules still use unknown exit profiles."""
        default_profile = self._format_exit_profile_from_context(self._default_exit_execution_context)
        if default_profile == self.UNKNOWN_EXIT_PROFILE:
            return
        try:
            n_results = max(50, self.vector_memory.semantic_rule_count)
            active_rules = self.vector_memory.get_active_rules(n_results=n_results)
            has_stale_rule = any(
                self.UNKNOWN_EXIT_PROFILE in str(rule.get("text", ""))
                or "sl_unknown_unknown_tp_unknown_unknown" in str(rule.get("rule_id", ""))
                for rule in active_rules
            )
            if not has_stale_rule:
                return
            self.logger.info("Refreshing semantic rules with missing exit execution profiles")
            self._trigger_reflection()
            self._trigger_loss_reflection()
            self._trigger_ai_mistake_reflection()
        except Exception as e:
            self.logger.warning("Semantic rule refresh failed: %s", e)

    def _sanitize_rule_key(self, value: str) -> str:
        """Create a stable Chroma ID suffix from a composite rule key."""
        return self._safe_key_part(value.replace("|", "_"))

    def _normalize_close_reason(self, reason: Any) -> str:
        """Normalize exit reasons while preserving stop-loss semantics."""
        normalized = self._safe_key_part(reason)
        stop_aliases = {
            "stop_loss",
            "stop_loss_hit",
            "hard_stop",
            "hard_stop_loss",
            "soft_stop",
            "soft_stop_loss",
            "emergency_stop",
        }
        if normalized in stop_aliases or ("stop" in normalized and "loss" in normalized):
            return "stop_loss"
        return normalized

    def _is_stop_loss_reason(self, reason: Any) -> bool:
        """Return whether a close reason represents any stop-loss style exit."""
        return self._normalize_close_reason(reason) == "stop_loss"

    def _adx_bucket_for_meta(self, meta: Dict[str, Any]) -> str:
        """Classify ADX into the reflection buckets used for rule keys."""
        adx = self._as_float(meta.get("adx_at_entry"), 0.0)
        if adx >= 25:
            return "HIGH_ADX"
        if adx < 20:
            return "LOW_ADX"
        return "MED_ADX"

    @staticmethod
    def _adx_level_from_bucket(adx_bucket: str) -> str:
        """Format an ADX bucket for rule text."""
        adx_level_map = {
            "HIGH_ADX": "High ADX",
            "MED_ADX": "Med ADX",
            "LOW_ADX": "Low ADX",
        }
        return adx_level_map.get(adx_bucket, adx_bucket.replace("_", " ").title())

    def _build_exit_profile_key(self, meta: Dict[str, Any]) -> str:
        """Build a deterministic key for hard/soft SL/TP execution settings."""
        context = self._resolve_exit_execution_context(meta)
        stop_type = self._safe_key_part(context["stop_loss_type"])
        stop_interval = self._safe_key_part(context["stop_loss_check_interval"])
        take_profit_type = self._safe_key_part(context["take_profit_type"])
        take_profit_interval = self._safe_key_part(context["take_profit_check_interval"])
        return f"sl_{stop_type}_{stop_interval}|tp_{take_profit_type}_{take_profit_interval}"

    @staticmethod
    def _format_exit_profile(
        stop_type: str,
        stop_interval: str,
        take_profit_type: str,
        take_profit_interval: str,
    ) -> str:
        """Format a hard/soft SL/TP profile for rules and dashboard metadata."""
        profile = format_exit_execution_context(
            {
                "stop_loss_type": stop_type,
                "stop_loss_check_interval": stop_interval,
                "take_profit_type": take_profit_type,
                "take_profit_check_interval": take_profit_interval,
            },
            include_unknown=True,
        )
        return profile.removeprefix("Exit Execution: ")

    def _meta_values(
        self,
        metas: List[Dict[str, Any]],
        key: str,
        *,
        absolute: bool = False,
        positive_only: bool = False,
        non_zero: bool = False,
    ) -> List[float]:
        """Return normalized numeric metadata values with optional filtering."""
        values: List[float] = []
        for meta in metas:
            if meta.get(key) is None:
                continue
            value = self._as_float(meta.get(key), 0.0)
            if absolute:
                value = abs(value)
            if positive_only and value <= 0.0:
                continue
            if non_zero and value == 0.0:
                continue
            values.append(value)
        return values

    def _average_meta(self, metas: List[Dict[str, Any]], key: str, **filters: Any) -> float:
        """Average a numeric metadata field after applying optional filters."""
        values = self._meta_values(metas, key, **filters)
        return sum(values) / len(values) if values else 0.0

    def _compute_loss_diagnostics(self, losses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute reused diagnostic averages for losing trades."""
        alignment_values = [m.get("timeframe_alignment") for m in losses if m.get("timeframe_alignment")]
        mixed_count = sum(1 for value in alignment_values if value in ("MIXED", "DIVERGENT"))
        return {
            "avg_adx": self._average_meta(losses, "adx_at_entry"),
            "avg_rr": self._average_meta(losses, "rr_ratio"),
            "avg_confluence": self._average_meta(losses, "confluence_count"),
            "avg_loss_mfe": self._average_meta(losses, "max_profit_pct", positive_only=True),
            "mixed_alignment_ratio": mixed_count / len(alignment_values) if alignment_values else 0.0,
        }

    def _build_rule_metadata(
        self,
        rule_type: str,
        source_pattern: str,
        metrics: Dict[str, Any],
        **extra: Any,
    ) -> Dict[str, Any]:
        """Build common semantic-rule metadata for best, loss, and AI-mistake rules."""
        metadata = {
            "rule_type": rule_type,
            "source_pattern": source_pattern,
            "source_trades": metrics["total"],
            "wins": metrics["wins"],
            "losses": metrics["losses"],
            "win_rate": round(metrics["win_rate"] * 100, 1),
            "loss_rate": round(metrics["loss_rate"] * 100, 1),
            "avg_pnl_pct": round(metrics["avg_pnl"], 2),
            "profit_factor": round(min(metrics["profit_factor"], 99.0), 2),
            "expectancy_pct": round(metrics["expectancy_pct"], 2),
            "avg_mae_pct": round(metrics["avg_mae"], 2),
            "avg_mfe_pct": round(metrics["avg_mfe"], 2),
            "dominant_close_reason": metrics["dominant_close_reason"],
            "dominant_exit_profile": metrics["dominant_exit_profile"],
            "dominant_stop_loss_type": metrics["dominant_stop_loss_type"],
            "dominant_stop_loss_interval": metrics["dominant_stop_loss_interval"],
            "dominant_take_profit_type": metrics["dominant_take_profit_type"],
            "dominant_take_profit_interval": metrics["dominant_take_profit_interval"],
        }
        metadata.update(extra)
        return metadata

    def _is_sideways_failure(self, meta: Dict[str, Any]) -> bool:
        """Identify losses or flat outcomes where the market failed to trend."""
        pnl = self._as_float(meta.get("pnl_pct"), 0.0)
        if pnl > 0.2:
            return False

        regime = str(meta.get("market_regime", "")).upper()
        volatility = str(meta.get("volatility_level", "")).upper()
        close_reason = self._normalize_close_reason(meta.get("close_reason", ""))
        reasoning = str(meta.get("reasoning") or meta.get("ai_reasoning") or "").lower()
        adx = self._as_float(meta.get("adx_at_entry"), 0.0)

        return (
            regime in ("NEUTRAL", "SIDEWAYS", "RANGING")
            or volatility == "LOW"
            or 0 < adx < 20
            or close_reason in ("sideways", "range_exit", "timeout", "time_exit", "flat_exit")
            or "sideways" in reasoning
            or "range" in reasoning
            or "chop" in reasoning
        )

    def _classify_ai_mistake(self, meta: Dict[str, Any]) -> str:
        """Classify whether an outcome contradicts the AI's entry confidence."""
        confidence = str(meta.get("entry_confidence") or meta.get("confidence") or "").upper()
        if confidence not in ("HIGH", "MEDIUM"):
            return ""

        pnl = self._as_float(meta.get("pnl_pct"), 0.0)
        outcome = meta.get("outcome")
        sideways_failure = self._is_sideways_failure(meta)

        if confidence == "HIGH" and sideways_failure:
            return "sideways_overconfidence"
        if confidence == "HIGH" and outcome == "LOSS":
            return "overconfident_loss"
        if sideways_failure and outcome == "LOSS":
            return "sideways_failure"
        if confidence == "HIGH" and pnl <= 0.2:
            return "low_follow_through_overconfidence"
        return ""

    def _derive_ai_assumption(self, metas: List[Dict[str, Any]]) -> str:
        """Summarize the failed assumption from stored AI reasoning text."""
        joined_reasoning = " ".join(
            str(meta.get("reasoning") or meta.get("ai_reasoning") or "") for meta in metas
        ).lower()
        if any(token in joined_reasoning for token in ("breakout", "break out", "breakdown", "break down")):
            return "expected breakout continuation"
        if any(token in joined_reasoning for token in ("trend", "momentum", "continuation")):
            return "expected trend or momentum follow-through"
        if any(token in joined_reasoning for token in ("reversal", "mean reversion", "bounce")):
            return "expected reversal follow-through"
        if any(token in joined_reasoning for token in ("support", "resistance")):
            return "expected support/resistance reaction"
        return "AI confidence exceeded realised market follow-through"

    def _derive_ai_mistake_reason(
        self,
        mistake_metas: List[Dict[str, Any]],
        mistake_type: str,
        failed_assumption: str,
    ) -> str:
        """Explain what the AI got wrong for a repeated mistake cluster."""
        reasons: List[str] = []
        high_confidence_count = sum(
            1 for meta in mistake_metas
            if str(meta.get("entry_confidence") or meta.get("confidence") or "").upper() == "HIGH"
        )
        sideways_count = sum(1 for meta in mistake_metas if self._is_sideways_failure(meta))

        if high_confidence_count:
            reasons.append(f"AI used HIGH confidence on {high_confidence_count} failed or flat trade(s)")
        if sideways_count:
            reasons.append(f"market stayed sideways/choppy on {sideways_count} trade(s)")
        if mistake_type == "low_follow_through_overconfidence":
            reasons.append("trade produced too little follow-through for HIGH confidence")
        reasons.append(f"failed assumption: {failed_assumption}")
        return "; ".join(reasons)

    def _derive_ai_mistake_adjustment(
        self,
        mistake_metas: List[Dict[str, Any]],
        mistake_type: str,
    ) -> str:
        """Generate prompt-ready corrections for repeated AI judgment mistakes."""
        suggestions: List[str] = []
        if "sideways" in mistake_type or any(self._is_sideways_failure(meta) for meta in mistake_metas):
            suggestions.append(
                "downgrade confidence in neutral/low-ADX markets and HOLD unless expansion volume or ADX >= 20 confirms follow-through"
            )
        if "overconfidence" in mistake_type:
            suggestions.append("cap confidence one level lower until the missing confirmation is present")

        stop_type = self._dominant_exit_execution_context(mistake_metas)["stop_loss_type"]
        if stop_type == "hard":
            suggestions.append("with hard SL, reduce size or place invalidation beyond structure because chop can force exits")
        elif stop_type == "soft":
            suggestions.append("with soft SL, require a clear invalidation rule and shorter monitoring interval in chop")

        if not suggestions:
            suggestions.append("require stronger confirmation before repeating this AI reasoning pattern")
        return "; ".join(suggestions)

    def _trigger_reflection(self) -> None:
        """Reflect on recent trades and synthesize best-practice semantic rules.

        Called automatically every N trades. Identifies profitable patterns from
        win/loss history and stores outcome-aware rules with richer diagnostics.
        """
        try:
            all_metas = self.vector_memory._get_trade_metadatas(exclude_updates=True)
            win_metas = [m for m in all_metas if m.get("outcome") == "WIN"]

            if len(win_metas) < 5:
                self.logger.debug("Not enough winning trades for reflection (need 5+)")
                return

            def build_win_key(meta: Dict[str, Any]) -> str:
                regime = meta.get("market_regime", "NEUTRAL")
                direction = meta.get("direction", "UNKNOWN")
                adx_label = self._adx_bucket_for_meta(meta)
                return "|".join([
                    self._safe_key_part(direction).upper(),
                    self._safe_key_part(regime).upper(),
                    adx_label,
                    self._build_exit_profile_key(meta),
                ])

            pattern_counts = Counter(build_win_key(m) for m in win_metas)

            if not pattern_counts:
                return

            best_pattern = pattern_counts.most_common(1)[0]
            pattern_key, win_count = best_pattern
            if win_count < 3:
                self.logger.debug(
                    "Pattern %s rejected: only %s win occurrences (need 3+)", pattern_key, win_count
                )
                return

            group_metas = [m for m in all_metas if build_win_key(m) == pattern_key]
            metrics = self._compute_group_metrics(group_metas)

            if metrics["win_rate"] < 0.6:
                self.logger.debug(
                    "Pattern %s rejected: win rate %s < 60%% (%s/%s trades)",
                    pattern_key, f"{metrics['win_rate']:.0%}", metrics["wins"], metrics["total"],
                )
                return

            sample_meta = group_metas[0] if group_metas else {}
            direction = sample_meta.get("direction", "UNKNOWN")
            regime = sample_meta.get("market_regime", "NEUTRAL")
            adx_level = self._adx_level_from_bucket(self._adx_bucket_for_meta(sample_meta))
            exit_profile = metrics["dominant_exit_profile"]

            losses = metrics["losses"]
            rule_text = (
                f"{direction} trades perform well in {regime} market with {adx_level}. "
                f"Exit profile: {exit_profile}. "
                f"({win_count} wins, {losses} losses — {metrics['win_rate']:.0%} win rate)"
            )

            rule_id = f"rule_best_{self._sanitize_rule_key(pattern_key)}"
            stored = self.vector_memory.store_semantic_rule(
                rule_id=rule_id,
                rule_text=rule_text,
                metadata=self._build_rule_metadata(
                    "best_practice",
                    pattern_key,
                    metrics,
                    total_analyzed=len(win_metas),
                ),
            )
            if stored:
                self._deactivate_legacy_unknown_exit_rule("rule_best", pattern_key)

            self.logger.info("Reflection complete: stored best-practice rule '%s'", rule_text)

        except Exception as e:
            self.logger.warning("Reflection failed: %s", e)

    def _trigger_loss_reflection(self) -> None:
        """Reflect on losing trades and synthesize anti-pattern and corrective rules.

        Called automatically every N trades. Analyzes LOSS and mixed-outcome patterns
        to identify conditions that hurt profitability, storing actionable guidance.
        """
        try:
            all_metas = self.vector_memory._get_trade_metadatas(exclude_updates=True)
            loss_metas = [m for m in all_metas if m.get("outcome") == "LOSS"]

            if len(loss_metas) < 3:
                self.logger.debug("Not enough losing trades for anti-pattern reflection (need 3+)")
                return

            def build_loss_key(meta: Dict[str, Any]) -> str:
                regime = meta.get("market_regime", "NEUTRAL")
                close_reason = self._normalize_close_reason(meta.get("close_reason", "unknown"))
                direction = meta.get("direction", "UNKNOWN")
                return "|".join([
                    self._safe_key_part(direction).upper(),
                    self._safe_key_part(regime).upper(),
                    close_reason,
                    self._build_exit_profile_key(meta),
                ])

            pattern_counts = Counter(build_loss_key(m) for m in loss_metas)

            if not pattern_counts:
                return

            worst_pattern = pattern_counts.most_common(1)[0]
            pattern_key, loss_count = worst_pattern
            if loss_count < 2:
                self.logger.debug(
                    "Anti-pattern %s rejected: only %s occurrences (need 2+)", pattern_key, loss_count
                )
                return

            group_metas = [m for m in all_metas if build_loss_key(m) == pattern_key]
            metrics = self._compute_group_metrics(group_metas)

            failure_reason = self._derive_failure_reason(metrics)
            recommended_adjustment = self._derive_recommended_adjustment(metrics)

            sample_meta = group_metas[0] if group_metas else {}
            direction = sample_meta.get("direction", "UNKNOWN")
            regime = sample_meta.get("market_regime", "NEUTRAL")
            close_reason = metrics["dominant_close_reason"]
            exit_profile = metrics["dominant_exit_profile"]

            rule_type = "anti_pattern" if metrics["loss_rate"] >= 0.6 else "corrective"
            type_label = "⚠️ AVOID" if rule_type == "anti_pattern" else "⚡ IMPROVE"
            rule_text = (
                f"{type_label}: {direction} trades in {regime} market often exit via {close_reason}. "
                f"Exit profile: {exit_profile}. "
                f"({loss_count} losses, {metrics['wins']} wins — {metrics['win_rate']:.0%} win rate)"
            )

            rule_id = f"rule_{rule_type}_{self._sanitize_rule_key(pattern_key)}"
            stored = self.vector_memory.store_semantic_rule(
                rule_id=rule_id,
                rule_text=rule_text,
                metadata=self._build_rule_metadata(
                    rule_type,
                    pattern_key,
                    metrics,
                    source_loss_count=loss_count,
                    failure_reason=failure_reason,
                    recommended_adjustment=recommended_adjustment,
                ),
            )
            if stored:
                self._deactivate_legacy_unknown_exit_rule("rule_anti_pattern", pattern_key)
                self._deactivate_legacy_unknown_exit_rule("rule_corrective", pattern_key)

            self.logger.info(
                "Loss reflection complete: stored %s rule '%s'", rule_type, rule_text
            )

        except Exception as e:
            self.logger.warning("Loss reflection failed: %s", e)

    def _trigger_ai_mistake_reflection(self) -> None:
        """Reflect on cases where the AI's confidence or premise was wrong."""
        try:
            all_metas = self.vector_memory._get_trade_metadatas(exclude_updates=True)
            mistake_metas = [meta for meta in all_metas if self._classify_ai_mistake(meta)]

            if len(mistake_metas) < 2:
                self.logger.debug("Not enough AI mistake samples for reflection (need 2+)")
                return

            def build_mistake_key(meta: Dict[str, Any]) -> str:
                mistake_type = self._classify_ai_mistake(meta)
                confidence = meta.get("entry_confidence") or meta.get("confidence") or "UNKNOWN"
                direction = meta.get("direction", "UNKNOWN")
                regime = meta.get("market_regime", "NEUTRAL")
                return "|".join([
                    mistake_type,
                    self._safe_key_part(confidence).upper(),
                    self._safe_key_part(direction).upper(),
                    self._safe_key_part(regime).upper(),
                    self._build_exit_profile_key(meta),
                ])

            pattern_counts = Counter(build_mistake_key(meta) for meta in mistake_metas)
            pattern_key, mistake_count = pattern_counts.most_common(1)[0]
            if mistake_count < 2:
                self.logger.debug(
                    "AI mistake pattern %s rejected: only %s occurrence(s) (need 2+)",
                    pattern_key, mistake_count,
                )
                return

            matched_mistakes = [meta for meta in mistake_metas if build_mistake_key(meta) == pattern_key]
            sample_meta = matched_mistakes[0]
            base_parts = pattern_key.split("|")
            mistake_type = base_parts[0]
            confidence = sample_meta.get("entry_confidence") or sample_meta.get("confidence") or "UNKNOWN"
            direction = sample_meta.get("direction", "UNKNOWN")
            regime = sample_meta.get("market_regime", "NEUTRAL")

            def build_base_key(meta: Dict[str, Any]) -> str:
                meta_confidence = meta.get("entry_confidence") or meta.get("confidence") or "UNKNOWN"
                return "|".join([
                    self._safe_key_part(meta_confidence).upper(),
                    self._safe_key_part(meta.get("direction", "UNKNOWN")).upper(),
                    self._safe_key_part(meta.get("market_regime", "NEUTRAL")).upper(),
                    self._build_exit_profile_key(meta),
                ])

            base_key = "|".join(base_parts[1:])
            comparison_group = [meta for meta in all_metas if build_base_key(meta) == base_key]
            metrics = self._compute_group_metrics(comparison_group or matched_mistakes)
            failed_assumption = self._derive_ai_assumption(matched_mistakes)
            failure_reason = self._derive_ai_mistake_reason(
                matched_mistakes, mistake_type, failed_assumption
            )
            recommended_adjustment = self._derive_ai_mistake_adjustment(matched_mistakes, mistake_type)

            rule_text = (
                f"🧠 AI MISTAKE: {confidence} confidence {direction} calls in {regime} market repeated "
                f"{mistake_type.replace('_', ' ')}. Exit profile: {metrics['dominant_exit_profile']}. "
                f"Failed assumption: {failed_assumption}. "
                f"({mistake_count} mistake(s), {metrics['win_rate']:.0%} win rate in comparable trades)"
            )

            rule_id = f"rule_ai_mistake_{self._sanitize_rule_key(pattern_key)}"
            stored = self.vector_memory.store_semantic_rule(
                rule_id=rule_id,
                rule_text=rule_text,
                metadata=self._build_rule_metadata(
                    "ai_mistake",
                    pattern_key,
                    metrics,
                    source_mistake_count=mistake_count,
                    mistake_type=mistake_type,
                    entry_confidence=str(confidence).upper(),
                    failed_assumption=failed_assumption,
                    failure_reason=failure_reason,
                    recommended_adjustment=recommended_adjustment,
                ),
            )
            if stored:
                self._deactivate_legacy_unknown_exit_rule("rule_ai_mistake", pattern_key)

            self.logger.info("AI mistake reflection complete: stored rule '%s'", rule_text)

        except Exception as e:
            self.logger.warning("AI mistake reflection failed: %s", e)

    def _compute_group_metrics(self, group_metas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute outcome statistics for a group of trade metadata records."""
        wins = [m for m in group_metas if m.get("outcome") == "WIN"]
        losses = [m for m in group_metas if m.get("outcome") == "LOSS"]
        total = len(group_metas)
        win_count = len(wins)
        loss_count = len(losses)

        win_rate = win_count / total if total > 0 else 0.0
        loss_rate = loss_count / total if total > 0 else 0.0

        all_pnls = [self._as_float(m.get("pnl_pct"), 0.0) for m in group_metas]
        win_pnls = [self._as_float(m.get("pnl_pct"), 0.0) for m in wins]
        loss_pnls = [self._as_float(m.get("pnl_pct"), 0.0) for m in losses]

        avg_pnl = sum(all_pnls) / total if total > 0 else 0.0
        avg_win_pct = sum(win_pnls) / win_count if win_count > 0 else 0.0
        avg_loss_pct = sum(loss_pnls) / loss_count if loss_count > 0 else 0.0

        gross_profit = sum(p for p in all_pnls if p > 0)
        gross_loss = abs(sum(p for p in all_pnls if p < 0))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = 999.0
        else:
            profit_factor = 1.0

        expectancy_pct = win_rate * avg_win_pct + loss_rate * avg_loss_pct

        avg_mae = self._average_meta(losses, "max_drawdown_pct", absolute=True, non_zero=True)
        avg_mfe = self._average_meta(wins, "max_profit_pct", positive_only=True)

        loss_reasons = Counter(self._normalize_close_reason(m.get("close_reason", "unknown")) for m in losses)
        dominant_close_reason = loss_reasons.most_common(1)[0][0] if loss_reasons else "unknown"
        profile_metas = losses if losses else group_metas
        dominant_exit_context = self._dominant_exit_execution_context(profile_metas)
        dominant_stop_loss_type = dominant_exit_context["stop_loss_type"]
        dominant_stop_loss_interval = dominant_exit_context["stop_loss_check_interval"]
        dominant_take_profit_type = dominant_exit_context["take_profit_type"]
        dominant_take_profit_interval = dominant_exit_context["take_profit_check_interval"]
        dominant_exit_profile = self._format_exit_profile_from_context(dominant_exit_context)

        return {
            "total": total,
            "wins": win_count,
            "losses": loss_count,
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "avg_pnl": avg_pnl,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            "profit_factor": profit_factor,
            "expectancy_pct": expectancy_pct,
            "avg_mae": avg_mae,
            "avg_mfe": avg_mfe,
            "dominant_close_reason": dominant_close_reason,
            "dominant_exit_profile": dominant_exit_profile,
            "dominant_stop_loss_type": dominant_stop_loss_type,
            "dominant_stop_loss_interval": dominant_stop_loss_interval,
            "dominant_take_profit_type": dominant_take_profit_type,
            "dominant_take_profit_interval": dominant_take_profit_interval,
            "loss_metas": losses,
        }

    def _derive_failure_reason(self, metrics: Dict[str, Any]) -> str:
        """Derive the primary cause of losses from trade group metrics."""
        losses = metrics["loss_metas"]
        if not losses:
            return ""

        reasons: List[str] = []
        dominant_reason = metrics["dominant_close_reason"]

        diagnostics = self._compute_loss_diagnostics(losses)
        avg_adx = diagnostics["avg_adx"]
        avg_rr = diagnostics["avg_rr"]
        avg_confluence = diagnostics["avg_confluence"]
        avg_loss_mfe = diagnostics["avg_loss_mfe"]

        if self._is_stop_loss_reason(dominant_reason) and avg_adx > 0 and avg_adx < 20:
            reasons.append(
                f"low-ADX ({avg_adx:.0f}) choppy conditions caused {metrics['dominant_stop_loss_type']} stop-loss exits"
            )
        elif self._is_stop_loss_reason(dominant_reason):
            reasons.append(f"stop-loss hit on {metrics['losses']} of {metrics['total']} trades")
        if self._is_stop_loss_reason(dominant_reason) and metrics["dominant_stop_loss_type"] != "unknown":
            reasons.append(f"{metrics['dominant_stop_loss_type']} stop-loss execution was active")
        if avg_rr > 0 and avg_rr < 1.5:
            reasons.append(f"low average R/R ({avg_rr:.1f}) insufficient to offset losses")
        if avg_confluence > 0 and avg_confluence < 3:
            reasons.append(f"weak entry confluence (avg {avg_confluence:.0f} factors)")
        if avg_loss_mfe > 0.5:
            reasons.append(f"trades moved favorably (avg MFE +{avg_loss_mfe:.1f}%) before reversing")
        if not reasons:
            reasons.append(f"exits via {dominant_reason}")

        return "; ".join(reasons)

    def _derive_recommended_adjustment(self, metrics: Dict[str, Any]) -> str:
        """Generate actionable guidance to improve profitability for this pattern."""
        losses = metrics["loss_metas"]
        if not losses:
            return ""

        suggestions: List[str] = []

        diagnostics = self._compute_loss_diagnostics(losses)
        avg_adx = diagnostics["avg_adx"]
        avg_rr = diagnostics["avg_rr"]
        avg_confluence = diagnostics["avg_confluence"]

        if avg_adx > 0 and avg_adx < 20:
            suggestions.append("require ADX >= 20 before entry to avoid choppy markets")
        if avg_rr > 0 and avg_rr < 1.5:
            suggestions.append("require R/R >= 1.5 for this setup type")
        if avg_confluence > 0 and avg_confluence < 3:
            suggestions.append("demand at least 3 aligned confluences before entry")

        dominant_reason = metrics["dominant_close_reason"]
        stop_type = metrics["dominant_stop_loss_type"]
        if self._is_stop_loss_reason(dominant_reason) and stop_type == "hard" and avg_adx > 0 and avg_adx < 20:
            suggestions.append("avoid hard-stop entries in low-ADX chop unless breakout confirmation is present")
        elif self._is_stop_loss_reason(dominant_reason) and stop_type == "hard":
            suggestions.append("for hard SL setups, reduce position size or place invalidation beyond structure")
        elif self._is_stop_loss_reason(dominant_reason) and stop_type == "soft":
            suggestions.append("for soft SL setups, define the invalidation trigger and monitor it on the configured interval")

        if diagnostics["avg_loss_mfe"] > 0.0:
            avg_loss_mfe = diagnostics["avg_loss_mfe"]
            if avg_loss_mfe > 0.5:
                suggestions.append(
                    f"move SL to breakeven after +{avg_loss_mfe * 0.5:.1f}% gain to protect against reversals"
                )
        if diagnostics["mixed_alignment_ratio"] > 0.5:
            suggestions.append("reduce position size or HOLD when timeframes are MIXED or DIVERGENT")

        if not suggestions:
            win_rate_pct = round(metrics["win_rate"] * 100)
            suggestions.append(
                f"require stronger signal confirmation (current win rate {win_rate_pct}%)"
            )

        return "; ".join(suggestions)

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
            order_book_bias=conditions.get("order_book_bias", "BALANCED"),
            exit_execution_context=exit_execution_context,
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
