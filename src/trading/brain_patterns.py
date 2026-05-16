"""Trade metadata analysis helpers for trading brain reflection."""

from collections import Counter
from typing import Any

from .brain_exit_profiles import ExitProfileResolver


class TradePatternAnalyzer:
    """Analyze stored trade metadata for learned rules and diagnostics."""

    def __init__(self, exit_profiles: ExitProfileResolver):
        """Initialize the analyzer with exit-profile helpers."""
        self.exit_profiles = exit_profiles

    @staticmethod
    def extract_factor_scores(confluence_factors: tuple) -> dict[str, float]:
        """Extract factor scores into flat dict for vector metadata."""
        scores: dict[str, float] = {}
        if not confluence_factors:
            return scores
        for factor_name, score in confluence_factors:
            clean_name = factor_name.replace(" ", "_").lower()
            scores[f"{clean_name}_score"] = float(score)
        return scores

    @staticmethod
    def count_strong_confluences(confluence_factors: tuple) -> int:
        """Count factors with score > 50 supporting the trade."""
        if not confluence_factors:
            return 0
        return sum(1 for _, score in confluence_factors if score > 50)

    @staticmethod
    def as_float(value: Any, default: float = 0.0) -> float:
        """Return a float for optional numeric metadata values."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def normalize_close_reason(self, reason: Any) -> str:
        """Normalize exit reasons while preserving stop-loss semantics."""
        normalized = self.exit_profiles.safe_key_part(reason)
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

    def is_stop_loss_reason(self, reason: Any) -> bool:
        """Return whether a close reason represents any stop-loss style exit."""
        return self.normalize_close_reason(reason) == "stop_loss"

    def adx_bucket_for_meta(self, meta: dict[str, Any]) -> str:
        """Classify ADX into the reflection buckets used for rule keys."""
        adx = self.as_float(meta.get("adx_at_entry"), 0.0)
        if adx >= 25:
            return "HIGH_ADX"
        if adx < 20:
            return "LOW_ADX"
        return "MED_ADX"

    @staticmethod
    def adx_level_from_bucket(adx_bucket: str) -> str:
        """Format an ADX bucket for rule text."""
        adx_level_map = {
            "HIGH_ADX": "High ADX",
            "MED_ADX": "Med ADX",
            "LOW_ADX": "Low ADX",
        }
        return adx_level_map.get(adx_bucket, adx_bucket.replace("_", " ").title())

    def meta_values(
        self,
        metas: list[dict[str, Any]],
        key: str,
        *,
        absolute: bool = False,
        positive_only: bool = False,
        non_zero: bool = False,
    ) -> list[float]:
        """Return normalized numeric metadata values with optional filtering."""
        values: list[float] = []
        for meta in metas:
            if meta.get(key) is None:
                continue
            value = self.as_float(meta.get(key), 0.0)
            if absolute:
                value = abs(value)
            if positive_only and value <= 0.0:
                continue
            if non_zero and value == 0.0:
                continue
            values.append(value)
        return values

    def average_meta(self, metas: list[dict[str, Any]], key: str, **filters: Any) -> float:
        """Average a numeric metadata field after applying optional filters."""
        values = self.meta_values(metas, key, **filters)
        return sum(values) / len(values) if values else 0.0

    def compute_loss_diagnostics(self, losses: list[dict[str, Any]]) -> dict[str, float]:
        """Compute reused diagnostic averages for losing trades."""
        alignment_values = [meta.get("timeframe_alignment") for meta in losses if meta.get("timeframe_alignment")]
        mixed_count = sum(1 for value in alignment_values if value in ("MIXED", "DIVERGENT"))
        return {
            "avg_adx": self.average_meta(losses, "adx_at_entry"),
            "avg_rr": self.average_meta(losses, "rr_ratio"),
            "avg_confluence": self.average_meta(losses, "confluence_count"),
            "avg_loss_mfe": self.average_meta(losses, "max_profit_pct", positive_only=True),
            "mixed_alignment_ratio": mixed_count / len(alignment_values) if alignment_values else 0.0,
        }

    @staticmethod
    def build_rule_metadata(
        rule_type: str,
        source_pattern: str,
        metrics: dict[str, Any],
        **extra: Any,
    ) -> dict[str, Any]:
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

    def is_sideways_failure(self, meta: dict[str, Any]) -> bool:
        """Identify losses or flat outcomes where the market failed to trend."""
        pnl = self.as_float(meta.get("pnl_pct"), 0.0)
        if pnl > 0.2:
            return False
        regime = str(meta.get("market_regime", "")).upper()
        volatility = str(meta.get("volatility_level", "")).upper()
        close_reason = self.normalize_close_reason(meta.get("close_reason", ""))
        reasoning = str(meta.get("reasoning") or meta.get("ai_reasoning") or "").lower()
        adx = self.as_float(meta.get("adx_at_entry"), 0.0)
        return (
            regime in ("NEUTRAL", "SIDEWAYS", "RANGING")
            or volatility == "LOW"
            or 0 < adx < 20
            or close_reason in ("sideways", "range_exit", "timeout", "time_exit", "flat_exit")
            or "sideways" in reasoning
            or "range" in reasoning
            or "chop" in reasoning
        )

    def classify_ai_mistake(self, meta: dict[str, Any]) -> str:
        """Classify whether an outcome contradicts the AI's entry confidence."""
        confidence = str(meta.get("entry_confidence") or meta.get("confidence") or "").upper()
        if confidence not in ("HIGH", "MEDIUM"):
            return ""
        pnl = self.as_float(meta.get("pnl_pct"), 0.0)
        outcome = meta.get("outcome")
        sideways_failure = self.is_sideways_failure(meta)
        if confidence == "HIGH" and sideways_failure:
            return "sideways_overconfidence"
        if confidence == "HIGH" and outcome == "LOSS":
            return "overconfident_loss"
        if sideways_failure and outcome == "LOSS":
            return "sideways_failure"
        if confidence == "HIGH" and pnl <= 0.2:
            return "low_follow_through_overconfidence"
        return ""

    @staticmethod
    def derive_ai_assumption(metas: list[dict[str, Any]]) -> str:
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

    def derive_ai_mistake_reason(
        self,
        mistake_metas: list[dict[str, Any]],
        mistake_type: str,
        failed_assumption: str,
    ) -> str:
        """Explain what the AI got wrong for a repeated mistake cluster."""
        reasons: list[str] = []
        high_confidence_count = sum(
            1 for meta in mistake_metas
            if str(meta.get("entry_confidence") or meta.get("confidence") or "").upper() == "HIGH"
        )
        sideways_count = sum(1 for meta in mistake_metas if self.is_sideways_failure(meta))
        if high_confidence_count:
            reasons.append(f"AI used HIGH confidence on {high_confidence_count} failed or flat trade(s)")
        if sideways_count:
            reasons.append(f"market stayed sideways/choppy on {sideways_count} trade(s)")
        if mistake_type == "low_follow_through_overconfidence":
            reasons.append("trade produced too little follow-through for HIGH confidence")
        reasons.append(f"failed assumption: {failed_assumption}")
        return "; ".join(reasons)

    def derive_ai_mistake_adjustment(
        self,
        mistake_metas: list[dict[str, Any]],
        mistake_type: str,
    ) -> str:
        """Generate prompt-ready corrections for repeated AI judgment mistakes."""
        suggestions: list[str] = []
        if "sideways" in mistake_type or any(self.is_sideways_failure(meta) for meta in mistake_metas):
            suggestions.append(
                "downgrade confidence in neutral/low-ADX markets and HOLD unless expansion volume or ADX >= 20 confirms follow-through"
            )
        if "overconfidence" in mistake_type:
            suggestions.append("cap confidence one level lower until the missing confirmation is present")
        stop_type = self.exit_profiles.dominant_exit_execution_context(mistake_metas).stop_loss_type
        if stop_type == "hard":
            suggestions.append("with hard SL, reduce size or place invalidation beyond structure because chop can force exits")
        elif stop_type == "soft":
            suggestions.append("with soft SL, require a clear invalidation rule and shorter monitoring interval in chop")
        if not suggestions:
            suggestions.append("require stronger confirmation before repeating this AI reasoning pattern")
        return "; ".join(suggestions)

    def compute_group_metrics(self, group_metas: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute outcome statistics for a group of trade metadata records."""
        wins = [meta for meta in group_metas if meta.get("outcome") == "WIN"]
        losses = [meta for meta in group_metas if meta.get("outcome") == "LOSS"]
        total = len(group_metas)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total if total > 0 else 0.0
        loss_rate = loss_count / total if total > 0 else 0.0
        all_pnls = [self.as_float(meta.get("pnl_pct"), 0.0) for meta in group_metas]
        win_pnls = [self.as_float(meta.get("pnl_pct"), 0.0) for meta in wins]
        loss_pnls = [self.as_float(meta.get("pnl_pct"), 0.0) for meta in losses]
        avg_pnl = sum(all_pnls) / total if total > 0 else 0.0
        avg_win_pct = sum(win_pnls) / win_count if win_count > 0 else 0.0
        avg_loss_pct = sum(loss_pnls) / loss_count if loss_count > 0 else 0.0
        gross_profit = sum(pnl for pnl in all_pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in all_pnls if pnl < 0))
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = 999.0
        else:
            profit_factor = 1.0
        expectancy_pct = win_rate * avg_win_pct + loss_rate * avg_loss_pct
        avg_mae = self.average_meta(losses, "max_drawdown_pct", absolute=True, non_zero=True)
        avg_mfe = self.average_meta(wins, "max_profit_pct", positive_only=True)
        loss_reasons = Counter(self.normalize_close_reason(meta.get("close_reason", "unknown")) for meta in losses)
        dominant_close_reason = loss_reasons.most_common(1)[0][0] if loss_reasons else "unknown"
        profile_metas = losses if losses else group_metas
        dominant_exit_context = self.exit_profiles.dominant_exit_execution_context(profile_metas)
        dominant_stop_loss_type = dominant_exit_context.stop_loss_type
        dominant_stop_loss_interval = dominant_exit_context.stop_loss_check_interval
        dominant_take_profit_type = dominant_exit_context.take_profit_type
        dominant_take_profit_interval = dominant_exit_context.take_profit_check_interval
        dominant_exit_profile = self.exit_profiles.format_exit_profile_from_context(dominant_exit_context)
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

    def derive_failure_reason(self, metrics: dict[str, Any]) -> str:
        """Derive the primary cause of losses from trade group metrics."""
        losses = metrics["loss_metas"]
        if not losses:
            return ""
        reasons: list[str] = []
        dominant_reason = metrics["dominant_close_reason"]
        diagnostics = self.compute_loss_diagnostics(losses)
        avg_adx = diagnostics["avg_adx"]
        avg_rr = diagnostics["avg_rr"]
        avg_confluence = diagnostics["avg_confluence"]
        avg_loss_mfe = diagnostics["avg_loss_mfe"]
        if self.is_stop_loss_reason(dominant_reason) and avg_adx > 0 and avg_adx < 20:
            reasons.append(
                f"low-ADX ({avg_adx:.0f}) choppy conditions caused {metrics['dominant_stop_loss_type']} stop-loss exits"
            )
        elif self.is_stop_loss_reason(dominant_reason):
            reasons.append(f"stop-loss hit on {metrics['losses']} of {metrics['total']} trades")
        if self.is_stop_loss_reason(dominant_reason) and metrics["dominant_stop_loss_type"] != "unknown":
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

    def derive_recommended_adjustment(self, metrics: dict[str, Any]) -> str:
        """Generate actionable guidance to improve profitability for this pattern."""
        losses = metrics["loss_metas"]
        if not losses:
            return ""
        suggestions: list[str] = []
        diagnostics = self.compute_loss_diagnostics(losses)
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
        if self.is_stop_loss_reason(dominant_reason) and stop_type == "hard" and avg_adx > 0 and avg_adx < 20:
            suggestions.append("avoid hard-stop entries in low-ADX chop unless breakout confirmation is present")
        elif self.is_stop_loss_reason(dominant_reason) and stop_type == "hard":
            suggestions.append("for hard SL setups, reduce position size or place invalidation beyond structure")
        elif self.is_stop_loss_reason(dominant_reason) and stop_type == "soft":
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