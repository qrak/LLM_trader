"""Regime-based risk profile selector using choppiness and ATR.

Replaces the old dead-code RiskProfileSelector (deleted — was based on
win-rate stats and never wired into active code). This one uses currently
computed indicators (choppiness, ATR%) available every trading cycle.

No trade history required — purely regime-driven.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar


class RegimeRiskProfile(Enum):
    """Risk profiles derived from market regime, not win-rate history."""

    AGGRESSIVE = "aggressive"       # ranging + low vol → tighter SL, wider TP, bigger size
    NEUTRAL = "neutral"             # transitional / trending / default
    CONSERVATIVE = "conservative"   # high volatility or unclear regime


# Multiplier tables: SL (ATR multiple), TP (ATR multiple), position cap (fraction)
_PROFILE_PARAMS: dict[RegimeRiskProfile, tuple[float, float, float]] = {
    RegimeRiskProfile.AGGRESSIVE:   (1.5, 3.0, 0.10),   # tighter SL + wider TP + 10% max
    RegimeRiskProfile.NEUTRAL:      (2.0, 4.0, 0.08),   # standard
    RegimeRiskProfile.CONSERVATIVE: (2.5, 5.0, 0.05),   # wider SL (survives noise), wider TP for R:R
}


class RegimeRiskProfileSelector:
    """Selects the active risk profile based on choppiness and ATR percentage.

    Pure regime-driven — no brain/trade-history dependency. Works from day one.
    """

    # Thresholds
    CHOPPINESS_RANGING: ClassVar[float] = 61.8
    CHOPPINESS_TRENDING: ClassVar[float] = 38.2
    HIGH_VOLATILITY_ATR_PCT: ClassVar[float] = 4.0
    LOW_VOLATILITY_ATR_PCT: ClassVar[float] = 1.5

    def select_profile(
        self,
        choppiness: float | None = None,
        atr_percentage: float = 2.0,
    ) -> RegimeRiskProfile:
        """Select profile based on regime and volatility.

        Args:
            choppiness: Choppiness index (0-100). None = assume transitional.
            atr_percentage: ATR as % of current price.

        Returns:
            RegimeRiskProfile enum.
        """
        # 1. Safety: high volatility → conservative regardless of regime
        if atr_percentage >= self.HIGH_VOLATILITY_ATR_PCT:
            return RegimeRiskProfile.CONSERVATIVE

        # 2. Safety: no choppiness data → neutral
        if choppiness is None:
            return RegimeRiskProfile.NEUTRAL

        # 3. Ranging (high choppiness) + low/medium vol → aggressive mean-reversion
        if choppiness > self.CHOPPINESS_RANGING:
            if atr_percentage < self.LOW_VOLATILITY_ATR_PCT:
                # Low vol range → highest confidence mean-reversion
                return RegimeRiskProfile.AGGRESSIVE
            # Medium vol range → still tradeable with standard params
            return RegimeRiskProfile.NEUTRAL

        # 4. Trending (low choppiness) → standard trend-following
        if choppiness < self.CHOPPINESS_TRENDING:
            return RegimeRiskProfile.NEUTRAL

        # 5. Transitional (38-62) → neutral
        return RegimeRiskProfile.NEUTRAL

    @staticmethod
    def get_sl_multiplier(profile: RegimeRiskProfile) -> float:
        """ATR multiplier for stop-loss distance."""
        return _PROFILE_PARAMS[profile][0]

    @staticmethod
    def get_tp_multiplier(profile: RegimeRiskProfile) -> float:
        """ATR multiplier for take-profit distance."""
        return _PROFILE_PARAMS[profile][1]

    @staticmethod
    def get_position_size_cap(profile: RegimeRiskProfile) -> float:
        """Maximum position size as fraction of capital."""
        return _PROFILE_PARAMS[profile][2]

    @staticmethod
    def build_profile_context(
        profile: RegimeRiskProfile,
        reason: str,
        brain_stats: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        """Build a profile notification for the system prompt.

        When brain_stats (from compute_per_profile_stats) is provided, includes
        learned adjustments — the LLM reads this to adapt its sizing/confidence.
        """
        descriptions = {
            RegimeRiskProfile.AGGRESSIVE: (
                "AGGRESSIVE — ranging market with low/medium volatility. "
                "Tighter SL (1.5x ATR), closer TP (3x ATR), max position 10%. "
                "Mean-reversion setups at range boundaries."
            ),
            RegimeRiskProfile.NEUTRAL: (
                "NEUTRAL — trending/transitional market. "
                "Standard SL (2x ATR), standard TP (4x ATR), max position 8%."
            ),
            RegimeRiskProfile.CONSERVATIVE: (
                "CONSERVATIVE — high volatility (>4% ATR). "
                "Wider SL (2.5x ATR) to survive noise, wider TP (5x ATR), max position 5%. "
                "Prioritize capital preservation."
            ),
        }
        desc = descriptions.get(profile, descriptions[RegimeRiskProfile.NEUTRAL])

        lines = [
            f"## ACTIVE RISK PROFILE: {desc}",
            f"Profile reason: {reason}",
        ]

        # --- Brain-learned per-profile performance ---
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

            # Show other profiles for context
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
