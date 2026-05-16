"""Prompt context and threshold retrieval for the trading brain."""

from typing import Any, Callable

from src.utils.indicator_classifier import (
    build_context_string_from_classified_values,
    build_query_document_from_classified_values,
    classify_adx_label,
)

from .brain_exit_profiles import ExitProfileResolver
from .data_models import ExitExecutionContext
from .vector_memory import VectorMemoryService


class BrainContextProvider:
    """Build LLM prompt context from vector memory and learned rules."""

    def __init__(self, vector_memory: VectorMemoryService, exit_profiles: ExitProfileResolver):
        """Initialize context provider dependencies and cache state."""
        self.vector_memory = vector_memory
        self.exit_profiles = exit_profiles
        self._stats_cache: dict[str, Any] = {}
        self._cache_trade_count: int = 0

    def clear_stats_cache(self) -> None:
        """Invalidate cached vector-memory statistics."""
        self._stats_cache = {}

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
        exit_execution_context: ExitExecutionContext | None = None,
    ) -> str:
        """Generate formatted brain context for prompt injection using vector retrieval."""
        lines = []
        exp_count = self.vector_memory.trade_count
        if exp_count > 0:
            lines.extend([
                "",
                f"## Trading Brain ({exp_count} closed trades)",
                "",
                "### Confidence Calibration:",
            ])
            conf_stats = self.get_cached_stats(
                "confidence", self.vector_memory.compute_confidence_stats
            )
            for level in ["HIGH", "MEDIUM", "LOW"]:
                stats = conf_stats.get(level, {})
                if stats.get("total_trades", 0) > 0:
                    lines.append(
                        f"- {level} Confidence: Win Rate {stats['win_rate']:.0f}% "
                        f"({stats['winning_trades']}/{stats['total_trades']} trades) | Avg P&L: {stats['avg_pnl_pct']:+.2f}%"
                    )
            recommendation = self.vector_memory.get_confidence_recommendation()
            if recommendation:
                lines.append(f"  → INSIGHT: {recommendation}")
            direction_bias = self.vector_memory.get_direction_bias()
            if direction_bias:
                lines.extend([
                    "",
                    "### Direction Bias Check:",
                    f"- Historical trades: {direction_bias['long_count']} LONG, {direction_bias['short_count']} SHORT",
                ])
                if direction_bias["short_count"] == 0:
                    lines.append("- ⚠️ NO SHORT TRADES IN HISTORY: Consider SHORT opportunities more carefully; lack of data means you may be missing valid setups.")
        try:
            blocked_feedback = self.vector_memory.get_blocked_trade_feedback(
                n=5, max_age_hours=168
            )
            if blocked_feedback:
                lines.extend(["", blocked_feedback])
        except Exception:
            pass
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
            k=5,
        )
        if vector_context:
            vector_context = self.exit_profiles.replace_unknown_exit_profile_text(vector_context)
            lines.extend(["", vector_context])
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
        adx_label = classify_adx_label(adx)
        rule_context = f"{trend_direction} + {adx_label} + {volatility_level} Volatility"
        semantic_rules = self.vector_memory.get_relevant_rules(
            current_context=rule_context,
            n_results=3,
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
                rule_text = self.exit_profiles.render_rule_text(rule)
                lines.append(f"- [{similarity:.0f}% match]{type_tag} {rule_text}")
                failure = meta.get("failure_reason")
                if failure:
                    lines.append(f"  → Why it failed: {failure}")
                recommended = meta.get("recommended_adjustment")
                if recommended:
                    lines.append(f"  → Apply: {recommended}")
            lines.append("")
        return "\n".join(lines)

    def get_dynamic_thresholds(self) -> dict[str, Any]:
        """Get brain-learned thresholds from vector store."""
        thresholds = self.get_cached_stats(
            "thresholds", self.vector_memory.compute_optimal_thresholds
        )
        return {
            "adx_strong_threshold": thresholds.get("adx_strong_threshold", 25),
            "avg_sl_pct": thresholds.get("avg_sl_pct", 2.5),
            "min_rr_recommended": thresholds.get("min_rr_recommended", 2.0),
            "confidence_threshold": thresholds.get("confidence_threshold", 70),
            "safe_mae_pct": thresholds.get("safe_mae_pct", 0),
            "adx_weak_threshold": thresholds.get("adx_weak_threshold", 20),
            "min_confluences_weak": thresholds.get("min_confluences_weak", 4),
            "min_confluences_standard": thresholds.get("min_confluences_standard", 3),
            "position_reduce_mixed": thresholds.get("position_reduce_mixed", 0.20),
            "position_reduce_divergent": thresholds.get("position_reduce_divergent", 0.35),
            "min_position_size": thresholds.get("min_position_size", 0.02),
            "rr_borderline_min": thresholds.get("rr_borderline_min", 1.5),
            "rr_strong_setup": thresholds.get("rr_strong_setup", 2.5),
            "trade_count": self.vector_memory.trade_count,
            "learned_keys": list(thresholds.keys()),
        }

    @staticmethod
    def build_rich_context_string(
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
        exit_execution_context: ExitExecutionContext | None = None,
    ) -> str:
        """Build rich semantic context string for vector storage and retrieval."""
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

    @staticmethod
    def build_query_document(
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
        exit_execution_context: ExitExecutionContext | None = None,
    ) -> str:
        """Build a query document that mirrors stored experience document format."""
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
        exit_execution_context: ExitExecutionContext | None = None,
        k: int = 5,
    ) -> str:
        """Get context from similar past experiences via vector retrieval."""
        context_query = self.build_rich_context_string(
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
        query_document = self.build_query_document(
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

    def get_cached_stats(self, key: str, compute_fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Get stats from cache or compute and cache them."""
        current_count = self.vector_memory.experience_count
        if current_count != self._cache_trade_count:
            self._stats_cache = {}
            self._cache_trade_count = current_count
        if key not in self._stats_cache:
            self._stats_cache[key] = compute_fn()
        return self._stats_cache[key]