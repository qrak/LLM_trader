

from collections import Counter
from typing import Any

from src.logger.logger import Logger

from .brain_exit_profiles import ExitProfileResolver
from .brain_patterns import TradePatternAnalyzer
from .vector_memory import VectorMemoryService


class BrainReflectionEngine:
    """Rebuild learned semantic rules from stored closed-trade metadata."""

    LOSS_PATTERN_MIN_LOSSES = 5
    ANTI_PATTERN_MIN_LOSS_RATE = 0.6
    BASELINE_MIN_CLOSED_TRADES = 10

    def __init__(
        self,
        logger: Logger,
        vector_memory: VectorMemoryService,
        analyzer: TradePatternAnalyzer,
        exit_profiles: ExitProfileResolver,
    ):

        self.logger = logger
        self.vector_memory = vector_memory
        self.analyzer = analyzer
        self.exit_profiles = exit_profiles

    @staticmethod
    def _closed_win_rate(metas: list[dict[str, Any]]) -> tuple[float, int]:

        closed = [meta for meta in metas if meta.get("outcome") in ("WIN", "LOSS")]
        if not closed:
            return 0.0, 0
        wins = sum(1 for meta in closed if meta.get("outcome") == "WIN")
        return wins / len(closed), len(closed)

    def trigger_reflection(self) -> None:

        try:
            all_metas = self.vector_memory.get_trade_metadatas(exclude_updates=True)
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
            surprise_values = [
                float(m.get("surprise_ratio", 1.0))
                for m in group_metas
                if m.get("surprise_ratio") is not None
            ]
            avg_surprise = sum(surprise_values) / len(surprise_values) if surprise_values else 0.0
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
                f"({win_count} wins, {losses} losses — {metrics['win_rate']:.0%} win rate"
            )
            if avg_surprise > 0:
                flag = " ⚠️ high surprise" if avg_surprise > 1.5 else ""
                rule_text += f", avg surprise: {avg_surprise:.2f}{flag}"
            rule_text += ")"
            rule_id = f"rule_best_{self.exit_profiles.sanitize_rule_key(pattern_key)}"
            self.vector_memory.store_semantic_rule(
                rule_id=rule_id,
                rule_text=rule_text,
                metadata=self.analyzer.build_rule_metadata(
                    "best_practice",
                    pattern_key,
                    metrics,
                    total_analyzed=len(win_metas),
                ),
            )
            self.logger.info("Reflection complete: stored best-practice rule '%s'", rule_text)
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Reflection failed: %s", e)

    def trigger_loss_reflection(self) -> None:

        try:
            all_metas = self.vector_memory.get_trade_metadatas(exclude_updates=True)
            loss_metas = [meta for meta in all_metas if meta.get("outcome") == "LOSS"]
            if len(loss_metas) < self.LOSS_PATTERN_MIN_LOSSES:
                self.logger.debug(
                    "Not enough losing trades for anti-pattern reflection (need %s+)",
                    self.LOSS_PATTERN_MIN_LOSSES,
                )
                return

            def build_loss_key(meta: dict[str, Any]) -> str:
                direction = meta.get("direction", "UNKNOWN")
                regime = meta.get("market_regime", "NEUTRAL")
                return "|".join([
                    self.exit_profiles.safe_key_part(direction).upper(),
                    self.exit_profiles.safe_key_part(regime).upper(),
                ])

            pattern_counts = Counter(build_loss_key(meta) for meta in loss_metas)
            if not pattern_counts:
                return
            pattern_key, loss_count = pattern_counts.most_common(1)[0]
            if loss_count < self.LOSS_PATTERN_MIN_LOSSES:
                self.logger.debug(
                    "Anti-pattern %s rejected: only %s occurrences (need %s+)",
                    pattern_key,
                    loss_count,
                    self.LOSS_PATTERN_MIN_LOSSES,
                )
                return
            group_metas = [meta for meta in all_metas if build_loss_key(meta) == pattern_key]
            metrics = self.analyzer.compute_group_metrics(group_metas)
            baseline_win_rate, closed_count = self._closed_win_rate(all_metas)
            if (
                metrics["loss_rate"] < self.ANTI_PATTERN_MIN_LOSS_RATE
                and metrics["win_rate"] >= baseline_win_rate
            ):
                self.logger.debug(
                    "Anti-pattern %s rejected: %s losses in %s trades is not below the %.0f%% baseline",
                    pattern_key,
                    metrics["losses"],
                    metrics["total"],
                    baseline_win_rate * 100,
                )
                return
            significant = closed_count >= self.BASELINE_MIN_CLOSED_TRADES and (
                self.analyzer.is_significantly_worse(
                    metrics["wins"], metrics["losses"], baseline_win_rate
                )
            )
            rule_type = (
                "anti_pattern"
                if significant and metrics["loss_rate"] >= self.ANTI_PATTERN_MIN_LOSS_RATE
                else "corrective"
            )
            surprise_values = [
                float(m.get("surprise_ratio", 1.0))
                for m in group_metas
                if m.get("surprise_ratio") is not None
            ]
            avg_surprise = sum(surprise_values) / len(surprise_values) if surprise_values else 0.0
            failure_reason = self.analyzer.derive_failure_reason(metrics)
            recommended_adjustment = self.analyzer.derive_recommended_adjustment(metrics)
            sample_meta = group_metas[0] if group_metas else {}
            direction = sample_meta.get("direction", "UNKNOWN")
            regime = sample_meta.get("market_regime", "NEUTRAL")
            close_reason = metrics["dominant_close_reason"]
            exit_profile = metrics["dominant_exit_profile"]
            type_label = "⚠️ AVOID" if rule_type == "anti_pattern" else "⚡ IMPROVE"
            rule_text = (
                f"{type_label}: {direction} trades in {regime} market exit most often via {close_reason} "
                f"({metrics['losses']} losses, {metrics['wins']} wins — {metrics['win_rate']:.0%} win rate "
                f"vs {baseline_win_rate:.0%} baseline, n={metrics['total']}"
            )
            if avg_surprise > 0:
                rule_text += f", avg surprise: {avg_surprise:.2f}"
            rule_text += f"). Exit profile: {exit_profile}."
            rule_id = f"rule_{rule_type}_{self.exit_profiles.sanitize_rule_key(pattern_key)}"
            self.vector_memory.store_semantic_rule(
                rule_id=rule_id,
                rule_text=rule_text,
                metadata=self.analyzer.build_rule_metadata(
                    rule_type,
                    pattern_key,
                    metrics,
                    source_loss_count=loss_count,
                    failure_reason=failure_reason,
                    recommended_adjustment=recommended_adjustment,
                    direction=direction,
                    regime=regime,
                    baseline_win_rate=round(baseline_win_rate * 100, 1),
                ),
            )
            self.logger.info(
                "Loss reflection complete: stored %s rule '%s'", rule_type, rule_id
            )
        except Exception as e:  # noqa: BLE001
            self.logger.warning("Loss reflection failed: %s", e)

    def trigger_ai_mistake_reflection(self) -> None:

        try:
            all_metas = self.vector_memory.get_trade_metadatas(exclude_updates=True)
            mistake_metas = [meta for meta in all_metas if self.analyzer.classify_ai_mistake(meta)]
            if len(mistake_metas) < 2:
                self.logger.debug("Not enough AI mistake samples for reflection (need 2+)")
                return

            def build_mistake_key(meta: dict[str, Any]) -> str:
                mistake_type = self.analyzer.classify_ai_mistake(meta)
                confidence = meta.get("entry_confidence") or meta.get("confidence") or "UNKNOWN"
                return "|".join([
                    self.exit_profiles.safe_key_part(mistake_type).upper(),
                    self.exit_profiles.safe_key_part(confidence).upper(),
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
            mistake_type = self.analyzer.classify_ai_mistake(sample_meta)
            confidence = sample_meta.get("entry_confidence") or sample_meta.get("confidence") or "UNKNOWN"

            def build_comparable_key(meta: dict[str, Any]) -> str:
                meta_confidence = meta.get("entry_confidence") or meta.get("confidence") or "UNKNOWN"
                return self.exit_profiles.safe_key_part(meta_confidence).upper()

            comparable_key = self.exit_profiles.safe_key_part(confidence).upper()
            comparison_group = [meta for meta in all_metas if build_comparable_key(meta) == comparable_key]
            metrics = self.analyzer.compute_group_metrics(comparison_group or matched_mistakes)
            group_metas_eval = comparison_group or matched_mistakes
            surprise_values = [
                float(m.get("surprise_ratio", 1.0))
                for m in group_metas_eval
                if m.get("surprise_ratio") is not None
            ]
            avg_surprise = sum(surprise_values) / len(surprise_values) if surprise_values else 0.0
            failed_assumption = self.analyzer.derive_ai_assumption(matched_mistakes)
            failure_reason = self.analyzer.derive_ai_mistake_reason(
                matched_mistakes, mistake_type, failed_assumption
            )
            recommended_adjustment = self.analyzer.derive_ai_mistake_adjustment(matched_mistakes, mistake_type)
            rule_text = (
                f"🧠 AI MISTAKE: {confidence} confidence calls repeated "
                f"{mistake_type.replace('_', ' ')}. Exit profile: {metrics['dominant_exit_profile']}. "
                f"Failed assumption: {failed_assumption}. "
                f"({mistake_count} mistake(s), {metrics['win_rate']:.0%} win rate in comparable trades"
            )
            if avg_surprise > 0:
                rule_text += f", avg surprise: {avg_surprise:.2f}"
            rule_text += ")"
            rule_id = f"rule_ai_mistake_{self.exit_profiles.sanitize_rule_key(pattern_key)}"
            self.vector_memory.store_semantic_rule(
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
            self.logger.info("AI mistake reflection complete: stored rule '%s'", rule_id)
        except Exception as e:  # noqa: BLE001
            self.logger.warning("AI mistake reflection failed: %s", e)

