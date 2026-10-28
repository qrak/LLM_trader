"""Expected Value framework formatter for trading prompts.

Builds a dynamic EV-thinking section that addresses Optiver's finding:
LLMs understand EV conceptually but default to conservative/heuristic decisions.
This module injects a structured EV decision framework with live capital tracking.

Fee model: round trip = 2 × fee_percent × POSITION notional (0.075% per side on
Binance spot). The worked example anchors on the effective NEUTRAL position cap
(`RegimeRiskProfileSelector.effective_caps`, i.e. the profile cap clamped by
`max_position_size`) — fees must never be computed off the whole portfolio, or every EV
estimate looks ~6× costlier than the trade it describes, and never off a cap the runtime
would refuse, or the hurdle sits above what a legal position can earn.

Inference cost is a rolling per-analysis average supplied by the app. When it is
positive the section adds the cost, nets it out of step 5 and raises the 1.5×
threshold to fees + inference. With no measured cost the section is byte-identical
to the fee-only version, so the prompt does not change before the first measurement.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.trading.regime_risk_profile import RegimeRiskProfile, RegimeRiskProfileSelector

if TYPE_CHECKING:
    from src.config.loader import Config


class EVFrameworkFormatter:
    """Builds the Expected Value framework section for the system prompt."""

    def __init__(self, config: Config) -> None:
        self._config = config

    @property
    def standard_position_pct(self) -> float:
        """NEUTRAL position cap after the configured ceiling, as a fraction of capital."""
        return RegimeRiskProfileSelector.effective_caps(self._config.MAX_POSITION_SIZE)[
            RegimeRiskProfile.NEUTRAL
        ]

    @property
    def starting_capital(self) -> float:
        return float(self._config.DEMO_QUOTE_CAPITAL)

    @property
    def fee_percent(self) -> float:
        """Per-side fee rate as a fraction of the position notional."""
        return float(self._config.TRANSACTION_FEE_PERCENT)

    def round_trip_fee(self, position_notional: float) -> float:
        """Round-trip fee in dollars for a position of the given notional."""
        return position_notional * self.fee_percent * 2

    def build_ev_framework_section(
        self, current_capital: float, inference_cost_usd: float = 0.0
    ) -> str:
        """Build the EV framework block injected into the system prompt."""
        pnl = current_capital - self.starting_capital
        pnl_pct = (pnl / self.starting_capital) * 100 if self.starting_capital > 0 else 0.0
        side_fee_pct = self.fee_percent * 100
        round_trip_pct = side_fee_pct * 2
        standard_position_pct = self.standard_position_pct
        standard_pct = standard_position_pct * 100
        example_notional = current_capital * standard_position_pct
        example_fee = self.round_trip_fee(example_notional)

        inference_cost = 0.0
        if not isinstance(inference_cost_usd, bool):
            try:
                candidate = float(inference_cost_usd)
            except (TypeError, ValueError):
                candidate = 0.0
            if math.isfinite(candidate):
                inference_cost = max(0.0, candidate)

        cost_base = example_fee + inference_cost
        breakeven_ev = cost_base * 1.5

        step_five = (
            f"  5. EV = P(win) × avg_win + (1−P(win)) × (−avg_loss) − round-trip fee "
            f"({round_trip_pct:.3f}% of the position size)"
        )
        hold_rule = "- **HOLD if EV is negative or below the 1.5× fee threshold**"
        cost_lines: list[str] = []
        threshold_lines: list[str] = []
        threshold_rule = (
            f"- **Take the trade if EV > 1.5× the round-trip fee of your position** "
            f"(fee ${example_fee:,.2f} on a standard {standard_pct:.0f}% position → "
            f"EV must exceed ${breakeven_ev:,.2f})"
        )
        if inference_cost > 0:
            step_five += f" − inference cost (${inference_cost:,.4f})"
            hold_rule = "- **HOLD if EV is negative or below the 1.5× cost threshold**"
            cost_lines.append(
                f"- Inference cost (rolling average per analysis): ${inference_cost:,.4f}"
            )
            threshold_lines.append(
                f"- Net break-even: ${cost_base:,.2f} (round-trip fee ${example_fee:,.2f} + "
                f"inference ${inference_cost:,.4f}); EV threshold 1.5× = ${breakeven_ev:,.2f}"
            )
            threshold_rule = (
                f"- **Take the trade if EV > 1.5× (round-trip fee of your position + inference cost)** "
                f"(fee ${example_fee:,.2f} on a standard {standard_pct:.0f}% position + "
                f"inference ${inference_cost:,.4f} → EV must exceed ${breakeven_ev:,.2f})"
            )

        lines = [
            "",
            "## EXPECTED VALUE FRAMEWORK",
            "",
            "You are managing a paper-trading portfolio. Your objective is to maximize ",
            "expected value (EV) over a series of trades — NOT to avoid losses at all costs.",
            "",
            "### Portfolio Status",
            f"- Starting Capital: ${self.starting_capital:,.2f}",
            f"- Current Capital: ${current_capital:,.2f}",
            f"- Realized P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)",
            (
                f"- Round-trip trading fee: {round_trip_pct:.3f}% of the position size "
                f"({side_fee_pct:.3f}% per side) — e.g. ${example_fee:,.2f} on a standard "
                f"{standard_pct:.0f}% position (${example_notional:,.2f})"
            ),
            *cost_lines,
            "",
            "### EV Decision Rule",
            "For every BUY or SELL decision, explicitly estimate:",
            "  1. P(win) — your assessed probability the trade reaches TP before SL",
            "  2. avg_win — expected profit in dollars if TP is hit",
            "  3. P(lose) — probability the trade hits SL first (≈ 1 − P(win))",
            "  4. avg_loss — expected loss in dollars if SL is hit",
            step_five,
            "",
            "### EV Thresholds",
            *threshold_lines,
            threshold_rule,
            hold_rule,
            "- **After the system R/R floor is met, EV decides entry quality.** An R/R near the active floor with a well-supported P(win) can be +EV; a high R/R with a low P(win) can be negative EV and must be rejected. Do not HOLD solely because the ratio is close to the floor.",
            (
                "- **State P(win) AND its basis** (which confluences, regime and structure support it). An "
                "unsupported or inflated P(win) does not justify the entry."
            ),
            "- **Never reject a positive EV trade purely due to fear of loss**",
            "- A trade that loses money with good EV reasoning is a GOOD DECISION — luck is not strategy",
            "",
            "### Anti-Conservative Bias",
            "Optiver's research shows LLMs default to overly conservative strategies:",
            "- Trading smaller than optimal on +EV opportunities",
            "- Prioritizing hedging/avoidance over EV maximization",
            "- Failing to commit to +EV trades when uncertainty exists",
            "",
            "**Counteract these biases.** A missed +EV opportunity is mathematically identical to a realized loss.",
            "The goal is NOT a high win rate — it is a positive cumulative EV across all trades.",
            "",
            "**Priority rule:** if the Decision Gate passes (evidence + risk) AND EV is positive, the trade is the correct action.",
            "Defaulting to HOLD under uncertainty is the exact bias this section exists to counter — only HOLD when the gate fails or EV is negative.",
            "",
        ]
        return "\n".join(lines)
