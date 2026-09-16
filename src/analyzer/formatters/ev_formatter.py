"""Expected Value framework formatter for trading prompts.

Builds a dynamic EV-thinking section that addresses Optiver's finding:
LLMs understand EV conceptually but default to conservative/heuristic decisions.
This module injects a structured EV decision framework with live capital tracking.

Fee model: round trip = 2 × fee_percent × POSITION notional (0.075% per side on
Binance spot). The worked example anchors on the standard (NEUTRAL profile)
position cap — fees must never be computed off the whole portfolio, or every EV
estimate looks ~6× costlier than the trade it describes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.loader import Config


class EVFrameworkFormatter:
    """Builds the Expected Value framework section for the system prompt."""

    STANDARD_POSITION_PCT = 0.08

    def __init__(self, config: Config) -> None:
        self._config = config

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

    def build_ev_framework_section(self, current_capital: float) -> str:
        """Build the EV framework block injected into the system prompt."""
        pnl = current_capital - self.starting_capital
        pnl_pct = (pnl / self.starting_capital) * 100 if self.starting_capital > 0 else 0.0
        side_fee_pct = self.fee_percent * 100
        round_trip_pct = side_fee_pct * 2
        standard_pct = self.STANDARD_POSITION_PCT * 100
        example_notional = current_capital * self.STANDARD_POSITION_PCT
        example_fee = self.round_trip_fee(example_notional)
        breakeven_ev = example_fee * 1.5

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
            "",
            "### EV Decision Rule",
            "For every BUY or SELL decision, explicitly estimate:",
            "  1. P(win) — your assessed probability the trade reaches TP before SL",
            "  2. avg_win — expected profit in dollars if TP is hit",
            "  3. P(lose) — probability the trade hits SL first (≈ 1 − P(win))",
            "  4. avg_loss — expected loss in dollars if SL is hit",
            (
                f"  5. EV = P(win) × avg_win + (1−P(win)) × (−avg_loss) − round-trip fee "
                f"({round_trip_pct:.3f}% of the position size)"
            ),
            "",
            "### EV Thresholds",
            (
                f"- **Take the trade if EV > 1.5× the round-trip fee of your position** "
                f"(fee ${example_fee:,.2f} on a standard {standard_pct:.0f}% position → "
                f"EV must exceed ${breakeven_ev:,.2f})"
            ),
            "- **HOLD if EV is negative or below the 1.5× fee threshold**",
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
