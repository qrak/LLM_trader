"""Semantic rule reflection engine for the trading brain."""

from collections import Counter
from typing import Any

from src.logger.logger import Logger

from .brain_exit_profiles import ExitProfileResolver
from .brain_patterns import TradePatternAnalyzer
from .vector_memory import VectorMemoryService


class BrainReflectionEngine:
    """Rebuild learned semantic rules from stored closed-trade metadata."""

    def __init__(
        self,
        logger: Logger,
        vector_memory: VectorMemoryService,
        analyzer: TradePatternAnalyzer,
        exit_profiles: ExitProfileResolver,
    ):
        """Initialize the reflection engine with its storage and analysis dependencies."""
        self.logger = logger
        self.vector_memory = vector_memory
        self.analyzer = analyzer
        self.exit_profiles = exit_profiles

    def deactivate_legacy_unknown_exit_rule(self, rule_id_prefix: str, pattern_key: str) -> None:
        """Deactivate the stale unknown-profile rule once a resolved replacement exists."""
        legacy_pattern_key = self.exit_profiles.legacy_unknown_exit_pattern_key(pattern_key)
        if legacy_pattern_key == pattern_key:
            return
        legacy_rule_id = f"{rule_id_prefix}_{self.exit_profiles.sanitize_rule_key(legacy_pattern_key)}"
        self.vector_memory.deactivate_semantic_rules([legacy_rule_id])

    def refresh_semantic_rules_if_stale(self) -> None:
        """Refresh semantic rules once when active rules still use unknown exit profiles."""
        default_profile = self.exit_profiles.format_exit_profile_from_context(
            self.exit_profiles.default_exit_execution_context
        )
        if default_profile == self.exit_profiles.UNKNOWN_EXIT_PROFILE:
            return
        try:
            n_results = max(50, self.vector_memory.semantic_rule_count)
            active_rules = self.vector_memory.get_active_rules(n_results=n_results)
            has_stale_rule = any(
                self.exit_profiles.UNKNOWN_EXIT_PROFILE in str(rule.get("text", ""))
                or "sl_unknown_unknown_tp_unknown_unknown" in str(rule.get("rule_id", ""))
                for rule in active_rules
            )
            if not has_stale_rule:
                return
            self.logger.info("Refreshing semantic rules with missing exit execution profiles")
            self.trigger_reflection()
            self.trigger_loss_reflection()
            self.trigger_ai_mistake_reflection()
        except Exception as e:
            self.logger.warning("Semantic rule refresh failed: %s", e)

    def trigger_reflection(self) -> None:
        """Reflect on recent trades and synthesize best-practice semantic rules."""
        try:
            all_metas = self.vector_memory._get_trade_metadatas(exclude_updates=True)
            win_metas = [meta for meta in all_metas if meta.get("outcome") == "WIN"]
            if len(win_metas) < 5:
                self.logger.debug("Not enough winning trades for reflection (need 5+)")
                return

            def build_win_key(meta: dict[str, Any]) -> str:
                regime = meta.get("market_regime", "NEUTRAL")
                direction = meta.get("direction", "UNKNOWN")
                adx_label = self.analyzer.adx_bucket_for_meta(meta)
                return "|".join([
                    self.exit_profiles.safe_key_part(direction).upper(),
                    self.exit_profiles.safe_key_part(regime).upper(),
                    adx_label,
                    self.exit_profiles.build_exit_profile_key(meta),
                ])

            pattern_counts = Counter(build_win_key(meta) for meta in win_metas)
            if not pattern_counts:
                return
            pattern_key, win_count = pattern_counts.most_common(1)[0]
            if win_count < 3:
                self.logger.debug(
                    "Pattern %s rejected: only %s win occurrences (need 3+)", pattern_key, win_count
                )
                return
            group_metas = [meta for meta in all_metas if build_win_key(meta) == pattern_key]
            metrics = self.analyzer.compute_group_metrics(group_metas)
            if metrics["win_rate"] < 0.6:
                self.logger.debug(
                    "Pattern %s rejected: win rate %s < 60%% (%s/%s trades)",
                    pattern_key, f"{metrics['win_rate']:.0%}", metrics["wins"], metrics["total"],
                )
                return
            sample_meta = group_metas[0] if group_metas else {}
            direction = sample_meta.get("direction", "UNKNOWN")
            regime = sample_meta.get("market_regime", "NEUTRAL")
            adx_level = self.analyzer.adx_level_from_bucket(self.analyzer.adx_bucket_for_meta(sample_meta))
            exit_profile = metrics["dominant_exit_profile"]
            losses = metrics["losses"]
            rule_text = (
                f"{direction} trades perform well in {regime} market with {adx_level}. "
                f"Exit profile: {exit_profile}. "
                f"({win_count} wins, {losses} losses — {metrics['win_rate']:.0%} win rate)"
            )
            rule_id = f"rule_best_{self.exit_profiles.sanitize_rule_key(pattern_key)}"
            stored = self.vector_memory.store_semantic_rule(
                rule_id=rule_id,
                rule_text=rule_text,
                metadata=self.analyzer.build_rule_metadata(
                    "best_practice",
                    pattern_key,
                    metrics,
                    total_analyzed=len(win_metas),
                ),
            )
            if stored:
                self.deactivate_legacy_unknown_exit_rule("rule_best", pattern_key)
            self.logger.info("Reflection complete: stored best-practice rule '%s'", rule_text)
        except Exception as e:
            self.logger.warning("Reflection failed: %s", e)

    def trigger_loss_reflection(self) -> None:
        """Reflect on losing trades and synthesize anti-pattern and corrective rules."""
        try:
            all_metas = self.vector_memory._get_trade_metadatas(exclude_updates=True)
            loss_metas = [meta for meta in all_metas if meta.get("outcome") == "LOSS"]
            if len(loss_metas) < 3:
                self.logger.debug("Not enough losing trades for anti-pattern reflection (need 3+)")
                return

            def build_loss_key(meta: dict[str, Any]) -> str:
                regime = meta.get("market_regime", "NEUTRAL")
                close_reason = self.analyzer.normalize_close_reason(meta.get("close_reason", "unknown"))
                direction = meta.get("direction", "UNKNOWN")
                return "|".join([
                    self.exit_profiles.safe_key_part(direction).upper(),
                    self.exit_profiles.safe_key_part(regime).upper(),
                    close_reason,
                    self.exit_profiles.build_exit_profile_key(meta),
                ])

            pattern_counts = Counter(build_loss_key(meta) for meta in loss_metas)
            if not pattern_counts:
                return
            pattern_key, loss_count = pattern_counts.most_common(1)[0]
            if loss_count < 2:
                self.logger.debug(
                    "Anti-pattern %s rejected: only %s occurrences (need 2+)", pattern_key, loss_count
                )
                return
            group_metas = [meta for meta in all_metas if build_loss_key(meta) == pattern_key]
            metrics = self.analyzer.compute_group_metrics(group_metas)
            failure_reason = self.analyzer.derive_failure_reason(metrics)
            recommended_adjustment = self.analyzer.derive_recommended_adjustment(metrics)
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
            rule_id = f"rule_{rule_type}_{self.exit_profiles.sanitize_rule_key(pattern_key)}"
            stored = self.vector_memory.store_semantic_rule(
                rule_id=rule_id,
                rule_text=rule_text,
                metadata=self.analyzer.build_rule_metadata(
                    rule_type,
                    pattern_key,
                    metrics,
                    source_loss_count=loss_count,
                    failure_reason=failure_reason,
                    recommended_adjustment=recommended_adjustment,
                ),
            )
            if stored:
                self.deactivate_legacy_unknown_exit_rule("rule_anti_pattern", pattern_key)
                self.deactivate_legacy_unknown_exit_rule("rule_corrective", pattern_key)
            self.logger.info(
                "Loss reflection complete: stored %s rule '%s'", rule_type, rule_text
            )
        except Exception as e:
            self.logger.warning("Loss reflection failed: %s", e)

    def trigger_ai_mistake_reflection(self) -> None:
        """Reflect on cases where the AI's confidence or premise was wrong."""
        try:
            all_metas = self.vector_memory._get_trade_metadatas(exclude_updates=True)
            mistake_metas = [meta for meta in all_metas if self.analyzer.classify_ai_mistake(meta)]
            if len(mistake_metas) < 2:
                self.logger.debug("Not enough AI mistake samples for reflection (need 2+)")
                return

            def build_mistake_key(meta: dict[str, Any]) -> str:
                mistake_type = self.analyzer.classify_ai_mistake(meta)
                confidence = meta.get("entry_confidence") or meta.get("confidence") or "UNKNOWN"
                direction = meta.get("direction", "UNKNOWN")
                regime = meta.get("market_regime", "NEUTRAL")
                return "|".join([
                    mistake_type,
                    self.exit_profiles.safe_key_part(confidence).upper(),
                    self.exit_profiles.safe_key_part(direction).upper(),
                    self.exit_profiles.safe_key_part(regime).upper(),
                    self.exit_profiles.build_exit_profile_key(meta),
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

            def build_base_key(meta: dict[str, Any]) -> str:
                meta_confidence = meta.get("entry_confidence") or meta.get("confidence") or "UNKNOWN"
                return "|".join([
                    self.exit_profiles.safe_key_part(meta_confidence).upper(),
                    self.exit_profiles.safe_key_part(meta.get("direction", "UNKNOWN")).upper(),
                    self.exit_profiles.safe_key_part(meta.get("market_regime", "NEUTRAL")).upper(),
                    self.exit_profiles.build_exit_profile_key(meta),
                ])

            base_key = "|".join(base_parts[1:])
            comparison_group = [meta for meta in all_metas if build_base_key(meta) == base_key]
            metrics = self.analyzer.compute_group_metrics(comparison_group or matched_mistakes)
            failed_assumption = self.analyzer.derive_ai_assumption(matched_mistakes)
            failure_reason = self.analyzer.derive_ai_mistake_reason(
                matched_mistakes, mistake_type, failed_assumption
            )
            recommended_adjustment = self.analyzer.derive_ai_mistake_adjustment(matched_mistakes, mistake_type)
            rule_text = (
                f"🧠 AI MISTAKE: {confidence} confidence {direction} calls in {regime} market repeated "
                f"{mistake_type.replace('_', ' ')}. Exit profile: {metrics['dominant_exit_profile']}. "
                f"Failed assumption: {failed_assumption}. "
                f"({mistake_count} mistake(s), {metrics['win_rate']:.0%} win rate in comparable trades)"
            )
            rule_id = f"rule_ai_mistake_{self.exit_profiles.sanitize_rule_key(pattern_key)}"
            stored = self.vector_memory.store_semantic_rule(
                rule_id=rule_id,
                rule_text=rule_text,
                metadata=self.analyzer.build_rule_metadata(
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
                self.deactivate_legacy_unknown_exit_rule("rule_ai_mistake", pattern_key)
            self.logger.info("AI mistake reflection complete: stored rule '%s'", rule_text)
        except Exception as e:
            self.logger.warning("AI mistake reflection failed: %s", e)