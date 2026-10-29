

from __future__ import annotations

import math
from enum import Enum
from typing import Any, ClassVar


class RegimeRiskProfile(Enum):
    """Risk profiles derived from market regime, not win-rate history."""

    AGGRESSIVE = "aggressive"
    NEUTRAL = "neutral"
    CONSERVATIVE = "conservative"


_PROFILE_PARAMS: dict[RegimeRiskProfile, tuple[float, float, float]] = {
    RegimeRiskProfile.AGGRESSIVE:   (1.5, 3.0, 0.10),
    RegimeRiskProfile.NEUTRAL:      (2.0, 4.0, 0.08),
    RegimeRiskProfile.CONSERVATIVE: (2.5, 5.0, 0.05),
}


class RegimeRiskProfileSelector:
    """Selects the active risk profile based on choppiness and ATR percentage.

    Pure regime-driven — no brain/trade-history dependency. Works from day one.
    """

    CHOPPINESS_RANGING: ClassVar[float] = 61.8
    CHOPPINESS_TRENDING: ClassVar[float] = 38.2
    HIGH_VOLATILITY_ATR_PCT: ClassVar[float] = 4.0
    LOW_VOLATILITY_ATR_PCT: ClassVar[float] = 1.5

    @staticmethod
    def _validated_position_cap(max_position_size: float | None) -> float | None:

        if max_position_size is None:
            return None
        try:
            value = float(max_position_size)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_position_size must be a positive finite decimal") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError("max_position_size must be a positive finite decimal")
        return value

    def select_profile(
        self,
        choppiness: float | None = None,
        atr_percentage: float = 2.0,
    ) -> RegimeRiskProfile:

        if atr_percentage >= self.HIGH_VOLATILITY_ATR_PCT:
            return RegimeRiskProfile.CONSERVATIVE

        if choppiness is None:
            return RegimeRiskProfile.NEUTRAL

        if choppiness > self.CHOPPINESS_RANGING:
            if atr_percentage < self.LOW_VOLATILITY_ATR_PCT:
                return RegimeRiskProfile.AGGRESSIVE
            return RegimeRiskProfile.NEUTRAL

        if choppiness < self.CHOPPINESS_TRENDING:
            return RegimeRiskProfile.NEUTRAL

        return RegimeRiskProfile.NEUTRAL

    @staticmethod
    def get_sl_multiplier(profile: RegimeRiskProfile) -> float:

        return _PROFILE_PARAMS[profile][0]

    @staticmethod
    def get_tp_multiplier(profile: RegimeRiskProfile) -> float:

        return _PROFILE_PARAMS[profile][1]

    @staticmethod
    def get_position_size_cap(profile: RegimeRiskProfile) -> float:

        return _PROFILE_PARAMS[profile][2]

    @staticmethod
    def effective_caps(max_position_size: float | None = None) -> dict[RegimeRiskProfile, float]:

        limit = RegimeRiskProfileSelector._validated_position_cap(max_position_size)
        return {
            profile: cap if limit is None else min(cap, limit)
            for profile, (_sl, _tp, cap) in _PROFILE_PARAMS.items()
        }

    @staticmethod
    def build_profile_context(
        profile: RegimeRiskProfile,
        reason: str,
        brain_stats: dict[str, dict[str, Any]] | None = None,
        max_position_size: float | None = None,
    ) -> str:

        limit = RegimeRiskProfileSelector._validated_position_cap(max_position_size)
        sl_multiplier = RegimeRiskProfileSelector.get_sl_multiplier(profile)
        position_cap = RegimeRiskProfileSelector.get_position_size_cap(profile)
        effective_cap = position_cap if limit is None else min(position_cap, limit)
        cap_text = f"{effective_cap * 100:g}%"
        descriptions = {
            RegimeRiskProfile.AGGRESSIVE: (
                "AGGRESSIVE — ranging market with low/medium volatility. "
                f"Tighter SL ({sl_multiplier:g}x ATR), TP at structure (range boundary), max position {cap_text}. "
                "Mean-reversion setups at range boundaries."
            ),
            RegimeRiskProfile.NEUTRAL: (
                "NEUTRAL — trending/transitional market. "
                f"Standard SL ({sl_multiplier:g}x ATR), TP at structure (resistance/support), max position {cap_text}."
            ),
            RegimeRiskProfile.CONSERVATIVE: (
                "CONSERVATIVE — high volatility (>4% ATR). "
                f"Wider SL ({sl_multiplier:g}x ATR) to survive noise, TP at structure, max position {cap_text}. "
                "Prioritize capital preservation."
            ),
        }
        desc = descriptions.get(profile, descriptions[RegimeRiskProfile.NEUTRAL])

        lines = [
            f"## ACTIVE RISK PROFILE: {desc}",
            f"Profile reason: {reason}",
        ]

        if brain_stats:
            profile_key = profile.value
            my_stats = brain_stats.get(profile_key, {})
            if isinstance(my_stats, dict) and my_stats.get("total_trades", 0) >= 3:
                wr = my_stats["win_rate"]
                total = my_stats["total_trades"]
                wins = my_stats["winning_trades"]

                if wr is not None and wr < 40:
                    lines.extend([
                        (f"  ⚠️ BRAIN WARNING: {profile.value.upper()} profile underperforming "
                         f"({wins}/{total} wins, {wr:.0f}% WR). "
                         f"Reduce confidence, tighten entry criteria, or wait for a better profile match.")
                    ])
                elif wr is not None and wr >= 65:
                    lines.extend([
                        (f"  ✅ BRAIN CONFIRMED: {profile.value.upper()} profile performing well "
                         f"({wins}/{total} wins, {wr:.0f}% WR). "
                         f"Confidence in this profile's parameters is justified.")
                    ])
                else:
                    lines.extend([
                        (f"  ℹ️ BRAIN: {profile.value.upper()} profile at {wins}/{total} wins "
                         f"({wr:.0f}% WR). Standard risk parameters apply.")
                    ])

            other_lines = []
            for key, stats in brain_stats.items():
                if key != profile_key and stats.get("total_trades", 0) >= 3:
                    o_wr = stats["win_rate"]
                    if o_wr is not None:
                        other_lines.append(
                            f"- {key}: {stats['winning_trades']}/{stats['total_trades']} "
                            f"({o_wr:.0f}% WR)"
                        )
            if other_lines:
                lines.append("  Other profiles (for reference):")
                lines.extend(other_lines)

        lines.append("")
        return "\n".join(lines)
