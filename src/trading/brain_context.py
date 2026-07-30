"""Prompt context and threshold retrieval for the trading brain."""

from collections.abc import Callable
from typing import Any

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

    # Maximum characters for the full brain context block injected into the system prompt.
    # Prevents token bloat as trade count grows; sections are truncated tail-first
    # (journal → rules → experiences) while preserving headers and calibration stats.
    BRAIN_CONTEXT_MAX_CHARS = 12000

    def __init__(
        self,
        vector_memory: VectorMemoryService,
        exit_profiles: ExitProfileResolver,
        post_mortem_repo: Any | None = None,
        logger: Any | None = None,
    ):
        """Initialize context provider dependencies and cache state."""
        self.vector_memory = vector_memory
        self.exit_profiles = exit_profiles
        self.post_mortem_repo = post_mortem_repo
        self.logger = logger
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
        # --- NEW: enriched context fields (July 2026) ---
        choppiness: float | None = None,
        trend_strength: float = 0.0,
        atr_percentage: float = 0.0,
        mfi: float | None = None,
        cmf: float | None = None,
        vwap: float = 0.0,
        supertrend_direction: str = "NEUTRAL",
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
                        f"({stats['winning_trades']}/{stats['total_trades']} trades) | "
                        f"Avg P&L: {stats['avg_pnl_pct']:+.2f}%"
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
                    lines.append(
                        "- ⚠️ NO SHORT TRADES IN HISTORY: Consider SHORT opportunities more carefully; "
                        "lack of data means you may be missing valid setups."
                    )
        try:
            blocked_feedback = self.vector_memory.get_blocked_trade_feedback(
                n=5, max_age_hours=168
            )
            if blocked_feedback:
                lines.extend(["", blocked_feedback])
        except Exception as exc:  # noqa: BLE001
            if self.logger:
                self.logger.warning("Failed to fetch blocked trade feedback: %s", exc)
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
            choppiness=choppiness,
            trend_strength=trend_strength,
            atr_percentage=atr_percentage,
            mfi=mfi,
            cmf=cmf,
            vwap=vwap,
            supertrend_direction=supertrend_direction,
            k=3,
        )
        if vector_context:
            vector_context = self.exit_profiles.replace_unknown_exit_profile_text(vector_context)
            lines.extend(["", vector_context])
        has_limited_data = "⚠️ LIMITED DATA" in vector_context if vector_context else False
        if lines and not has_limited_data:
            lines.extend([
                "",
                "### Apply Insights (CoT Step 6 - Historical Evidence):",
                (
                    "- CONFIDENCE: If win rate in similar conditions <50%, reduce confidence by 10 points and "
                    "state it. Weight both wins AND losses, not just the favorable cases."
                ),
                (
                    '- ANTI-PATTERN / AI MISTAKE: If an AVOID or AI-mistake rule matches (>50% similarity), state '
                    '"⚠️ ANTI-PATTERN MATCH", compare the current setup to the failed assumption, and downgrade '
                    "confidence unless the missing confirmation is now present. State the adjustment you apply "
                    "(stricter confluences, higher R/R, or reduced size)."
                ),
                (
                    "- REGIME / EXIT MISMATCH: Treat a retrieved experience as informational only when its regime "
                    "(ADX/volatility marked ⚠️) or its hard/soft SL/TP exit profile differs from current conditions; "
                    "do not use it as a confidence prior without explaining the mismatch."
                ),
                (
                    "- CHOPPINESS MISMATCH: If a stored trade had low choppiness (Trending) but the current market "
                    "shows high choppiness (Choppy), treat this as a ⚠️ regime mismatch — a trade that worked in "
                    "a clean trend may fail in noise. Reduce confidence accordingly."
                ),
                (
                    "- VOLUME DIVERGENCE: If a stored trade entered during ACCUMULATION but current conditions "
                    "show DISTRIBUTION, the trade thesis may not hold. Flag this explicitly in your reasoning."
                ),
                (
                    "- MONEY FLOW CONTRADICTION: If CMF (Chaikin Money Flow) or MFI contradicts price direction "
                    "(e.g., BULLISH setup with CMF < 0 or MFI < 50), state the divergence and downgrade confidence "
                    "unless there is clear evidence of reversal building."
                ),
                (
                    "- ATR SCALE AWARENESS: Compare ATR% not just raw ATR — ATR=$1,500 means 2.5% on a $60K BTC vs "
                    "15% on a $10K BTC. Same raw ATR, very different risk. Factor this into your position sizing."
                ),
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
                freshness = meta.get("freshness_label")
                support = meta.get("support_count") or meta.get("source_trades")
                final_score = meta.get("final_score") or rule.get("final_score")
                evidence_bits = []
                if freshness:
                    evidence_bits.append(f"freshness: {freshness}")
                if support:
                    evidence_bits.append(f"evidence: {support} trades")
                if final_score is not None:
                    evidence_bits.append(f"score: {float(final_score):.0f}%")
                evidence_text = f" ({', '.join(evidence_bits)})" if evidence_bits else ""
                lines.append(f"- [{similarity:.0f}% match]{type_tag} {rule_text}{evidence_text}")
                failure = meta.get("failure_reason")
                if failure:
                    lines.append(f"  → Why it failed: {failure}")
                recommended = meta.get("recommended_adjustment")
                if recommended:
                    lines.append(f"  → Apply: {recommended}")
            lines.append("")

        # --- Trade Journal (Post-Mortem Lessons) ---
        journal_context = self._build_trade_journal_context()
        if journal_context:
            lines.extend([
                "",
                "### Trade Journal (Recent Post-Mortem Lessons):",
                journal_context,
                "",
            ])

        # --- Risk Profile (adaptive, brain-driven) ---
        risk_profile_text = self._build_risk_profile_context(atr_percentage)
        if risk_profile_text:
            lines.extend(["", risk_profile_text])

        result = "\n".join(lines)
        if len(result) > self.BRAIN_CONTEXT_MAX_CHARS:
            # Truncate at entry boundary (\n\n) to keep clean sections
            truncated = result[: self.BRAIN_CONTEXT_MAX_CHARS].rsplit("\n\n", 1)[0]
            if not truncated:
                truncated = result[: self.BRAIN_CONTEXT_MAX_CHARS].rsplit("\n", 1)[0]
            result = (
                truncated + "\n\n[Brain context truncated: token budget reached. "
                "Rely on standard analysis for remaining decisions.]"
            )
        return result


    def _build_trade_journal_context(self) -> str:
        """Build a formatted string of recent post-mortem lessons for prompt injection.

        Returns:
            Formatted multi-line string, or empty string if no post-mortems exist.
        """
        if not self.post_mortem_repo:
            return ""
        try:
            recent = self.post_mortem_repo.get_recent_post_mortems(limit=5)
            if not recent:
                return ""
            lines = []
            for pm in recent:
                pnl_str = f", P&L: {pm['pnl_pct']:+.1f}%" if pm.get("pnl_pct") is not None else ""
                lines.append(
                    f"— {pm['verdict']} ({pm['created_at'][:10]}, {pm['symbol']}): {pm['lesson_learned']}{pnl_str}"
                )
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            if self.logger:
                self.logger.warning("Failed to retrieve post-mortem lessons for brain context: %s", e)
            return ""

    def _build_risk_profile_context(self, atr_percentage: float) -> str:
        """Build risk profile context using all available vector memory analytics.

        Uses confidence stats (HIGH/MEDIUM/LOW win rates), ADX regime performance,
        trade count, and volatility to select profile and provide actionable guidance.

        Returns a short string for the system prompt, or empty if not enough data.
        """
        trade_count = self.vector_memory.trade_count
        if trade_count < 5:
            return ""

        conf_stats = self.vector_memory.compute_confidence_stats()
        adx_perf = self.vector_memory.compute_adx_performance()

        # ── Weighted composite score ──
        # HIGH confidence = 50% weight, MEDIUM = 35%, LOW = 15%
        high = conf_stats.get("HIGH", {})
        med = conf_stats.get("MEDIUM", {})
        low = conf_stats.get("LOW", {})

        h_wr = high.get("win_rate", 50.0)
        h_n = high.get("total_trades", 0)
        m_wr = med.get("win_rate", 50.0)
        m_n = med.get("total_trades", 0)
        l_wr = low.get("win_rate", 50.0)
        l_n = low.get("total_trades", 0)

        # Weighted composite win rate: HIGH×0.50, MED×0.35, LOW×0.15
        weighted_denom = (h_n * 0.50) + (m_n * 0.35) + (l_n * 0.15) or 1
        composite_wr = (
            (h_wr * h_n * 0.50) + (m_wr * m_n * 0.35) + (l_wr * l_n * 0.15)
        ) / weighted_denom

        # ── ADX regime bias ──
        adx_bias = 0.0
        adx_detail = ""
        for bucket_key, bucket_data in adx_perf.items():
            b_wr = bucket_data.get("win_rate", 50.0)
            b_n = bucket_data.get("total_trades", 0)
            if b_n >= 2:
                if b_wr >= 65:
                    adx_bias += 0.15
                    adx_detail = f"{bucket_data.get('level', bucket_key)} ({b_wr:.0f}% WR, {b_n} trades)"
                elif b_wr <= 40:
                    adx_bias -= 0.15

        # ── Decision matrix ──
        profile: str
        reason: str
        guidance: str

        # Safety overrides first
        if atr_percentage > 4.0:
            profile = "CONSERVATIVE"
            reason = f"High ATR ({atr_percentage:.1f}%) — protect capital"
            guidance = "SL: 2.5× ATR, TP: 5× ATR, Max position: 5%. Only high-conviction entries."
        elif trade_count < 10 or composite_wr < 35:
            profile = "CONSERVATIVE"
            reason = f"Low data confidence ({trade_count} trades, composite WR {composite_wr:.0f}%)"
            guidance = "SL: 2.5× ATR, TP: 5× ATR, Max position: 5%."
        elif composite_wr >= 60 and adx_bias >= 0.05 and atr_percentage < 3.0:
            profile = "AGGRESSIVE"
            reason_parts = [f"Strong composite WR ({composite_wr:.0f}%)"]
            if adx_detail:
                reason_parts.append(f"ADX edge: {adx_detail}")
            reason = "; ".join(reason_parts)
            guidance = "SL: 1.5× ATR, TP: 3× ATR, Max position: 10%. Take calculated risks."
        elif composite_wr >= 55 and h_wr >= 65 and atr_percentage < 2.5:
            profile = "AGGRESSIVE"
            reason = f"HIGH-confidence WR {h_wr:.0f}% ({h_n} trades) + moderate ATR ({atr_percentage:.1f}%)"
            guidance = "SL: 1.5× ATR, TP: 3× ATR, Max position: 10%."
        elif h_wr <= 42 and h_n >= 3:
            profile = "CONSERVATIVE"
            reason = f"Weak HIGH-confidence ({h_wr:.0f}% over {h_n} trades)"
            guidance = "SL: 2.5× ATR, TP: 5× ATR, Max position: 5%. Wait for clearer edge."
        elif adx_bias <= -0.10:
            profile = "CONSERVATIVE"
            reason = "Negative ADX regime edge"
            guidance = "SL: 2.5× ATR, TP: 5× ATR, Max position: 5%."
        else:
            profile = "NEUTRAL"
            reason = (
                f"Composite WR {composite_wr:.0f}% (HIGH: {h_wr:.0f}%/{h_n}t, "
                f"MED: {m_wr:.0f}%/{m_n}t, LOW: {l_wr:.0f}%/{l_n}t), "
                f"ATR {atr_percentage:.1f}%"
            )
            guidance = "SL: 2× ATR, TP: 4× ATR, Max position: 8%."

        return (
            f"### Active Risk Profile: {profile}\n"
            f"Reason: {reason}\n"
            f"→ {guidance}\n"
            f"(Composite WR: {composite_wr:.0f}% across {trade_count} trades "
            f"weighted: HIGH×0.50 MED×0.35 LOW×0.15)"
        )

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
            "sl_tightening": thresholds.get("sl_tightening"),
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
        # --- NEW: enriched fields (July 2026) ---
        choppiness: float | None = None,
        trend_strength: float = 0.0,
        atr_percentage: float = 0.0,
        mfi: float | None = None,
        cmf: float | None = None,
        vwap: float = 0.0,
        supertrend_direction: str = "NEUTRAL",
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
            choppiness=choppiness,
            trend_strength=trend_strength,
            atr_percentage=atr_percentage,
            mfi=mfi,
            cmf=cmf,
            vwap=vwap,
            supertrend_direction=supertrend_direction,
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
        # --- NEW: enriched query fields (July 2026) ---
        choppiness: float | None = None,
        trend_strength: float = 0.0,
        atr_percentage: float = 0.0,
        mfi: float | None = None,
        cmf: float | None = None,
        vwap: float = 0.0,
        supertrend_direction: str = "NEUTRAL",
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
            choppiness=choppiness,
            trend_strength=trend_strength,
            atr_percentage=atr_percentage,
            mfi=mfi,
            cmf=cmf,
            vwap=vwap,
            supertrend_direction=supertrend_direction,
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

