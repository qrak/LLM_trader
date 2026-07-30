"""Risk profile selector based on brain performance and market conditions.

Adaptively switches between Aggressive, Conservative, and Neutral risk profiles
based on win rate trends, market volatility, and portfolio P&L state.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vector_memory import VectorMemoryService


class RiskProfile(Enum):
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    NEUTRAL = "neutral"


class RiskProfileSelector:
    """Selects the active risk profile based on brain analytics and market state."""

    # Thresholds for switching profiles
    HIGH_CONFIDENCE_WIN_RATE_AGGRESSIVE = 0.65   # Win rate above this → Aggressive
    HIGH_CONFIDENCE_WIN_RATE_CONSERVATIVE = 0.45  # Win rate below this → Conservative
    RECENT_LOSS_STREAK_CONSERVATIVE = 2            # Consecutive losses → Conservative
    RECENT_WIN_STREAK_AGGRESSIVE = 3               # Consecutive wins → Aggressive
    HIGH_VOLATILITY_ATR_PCT = 4.0                  # ATR% above this → force Conservative
    LOW_VOLATILITY_ATR_PCT = 1.5                   # ATR% below this → allow Aggressive
    PROFITABLE_BIAS_PNL_PCT = 5.0                  # P&L above +5% → relax toward Aggressive
    LOSING_BIAS_PNL_PCT = -10.0                     # P&L below -10% → tighten to Conservative
    MIN_TRADES_FOR_PROFILE = 5                      # Need at least 5 trades before profiling

    def __init__(self, vector_memory: VectorMemoryService) -> None:
        self._vm = vector_memory

    def select_profile(
        self,
        atr_percentage: float = 2.0,
        current_pnl_pct: float = 0.0,
        recent_trades: list[dict] | None = None,
    ) -> RiskProfile:
        """Select the active risk profile based on current conditions.

        Args:
            atr_percentage: Current ATR as percentage of price.
            current_pnl_pct: Current portfolio P&L percentage.
            recent_trades: Optional list of recent trade dicts with 'pnl' keys.
        """
        # 1. Safety override: high volatility → always Conservative
        if atr_percentage > self.HIGH_VOLATILITY_ATR_PCT:
            return RiskProfile.CONSERVATIVE

        # 2. Deep drawdown → always Conservative
        if current_pnl_pct < self.LOSING_BIAS_PNL_PCT:
            return RiskProfile.CONSERVATIVE

        # 3. Not enough data → Neutral
        trade_count = self._vm.trade_count
        if trade_count < self.MIN_TRADES_FOR_PROFILE:
            return RiskProfile.NEUTRAL

        # 4. Analyze confidence stats
        conf_stats = self._vm.compute_confidence_stats()
        high_stats = conf_stats.get("HIGH", {"win_rate": 0.5, "trade_count": 0})
        high_win_rate = high_stats.get("win_rate", 0.5)
        high_count = high_stats.get("trade_count", 0)

        # 5. Recent streak analysis
        if recent_trades:
            streak = self._compute_streak(recent_trades)
            if streak <= -self.RECENT_LOSS_STREAK_CONSERVATIVE:
                return RiskProfile.CONSERVATIVE
            if streak >= self.RECENT_WIN_STREAK_AGGRESSIVE:
                return RiskProfile.AGGRESSIVE

        # 6. Confidence-based profile
        if high_count >= 3:
            if high_win_rate >= self.HIGH_CONFIDENCE_WIN_RATE_AGGRESSIVE:
                return RiskProfile.AGGRESSIVE
            if high_win_rate <= self.HIGH_CONFIDENCE_WIN_RATE_CONSERVATIVE:
                return RiskProfile.CONSERVATIVE

        # 7. P&L bias
        if current_pnl_pct >= self.PROFITABLE_BIAS_PNL_PCT:
            return RiskProfile.AGGRESSIVE

        # 8. Low volatility with neutral stats → slightly aggressive
        if atr_percentage < self.LOW_VOLATILITY_ATR_PCT:
            return RiskProfile.AGGRESSIVE

        return RiskProfile.NEUTRAL

    @staticmethod
    def _compute_streak(recent_trades: list[dict]) -> int:
        """Count consecutive wins (positive) or losses (negative) from the most recent trade."""
        if not recent_trades:
            return 0
        streak = 0
        for trade in reversed(recent_trades):
            pnl = trade.get("pnl", 0)
            if pnl > 0:
                if streak >= 0:
                    streak += 1
                else:
                    break
            elif pnl < 0:
                if streak <= 0:
                    streak -= 1
                else:
                    break
            # pnl == 0: break even, doesn't extend streak
        return streak

    def get_profile_sl_multiplier(self, profile: RiskProfile) -> float:
        """SL ATR multiplier per profile."""
        return {
            RiskProfile.AGGRESSIVE: 1.5,
            RiskProfile.NEUTRAL: 2.0,
            RiskProfile.CONSERVATIVE: 2.5,
        }[profile]

    def get_profile_tp_multiplier(self, profile: RiskProfile) -> float:
        """TP ATR multiplier per profile."""
        return {
            RiskProfile.AGGRESSIVE: 3.0,
            RiskProfile.NEUTRAL: 4.0,
            RiskProfile.CONSERVATIVE: 5.0,
        }[profile]

    def get_profile_position_size_cap(self, profile: RiskProfile) -> float:
        """Max position size fraction per profile."""
        return {
            RiskProfile.AGGRESSIVE: 0.10,
            RiskProfile.NEUTRAL: 0.08,
            RiskProfile.CONSERVATIVE: 0.05,
        }[profile]

    def build_profile_context(self, profile: RiskProfile, reason: str) -> str:
        """Build the risk profile context block for the system prompt."""
        profile_descriptions = {
            RiskProfile.AGGRESSIVE: (
                "AGGRESSIVE — winning streak or high-confidence edge detected. "
                "Tighter SL (1.5× ATR), wider TP (3× ATR), max position 10%. "
                "Take calculated risks on positive EV setups."
            ),
            RiskProfile.NEUTRAL: (
                "NEUTRAL — standard market conditions. "
                "Default SL (2× ATR), default TP (4× ATR), max position 8%. "
                "Balanced risk-reward approach."
            ),
            RiskProfile.CONSERVATIVE: (
                "CONSERVATIVE — high volatility, losing streak, or drawdown detected. "
                "Wider SL (2.5× ATR) to avoid noise exits, wider TP (5× ATR) for better R:R, max position 5%. "
                "Prioritize capital preservation. Only take high-conviction setups."
            ),
        }
        desc = profile_descriptions.get(profile, profile_descriptions[RiskProfile.NEUTRAL])
        return (
            f"\n## ACTIVE RISK PROFILE: {desc}\n"
            f"Profile reason: {reason}\n"
        )
