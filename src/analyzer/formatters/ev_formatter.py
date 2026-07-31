"""Expected Value framework formatter for trading prompts.

Builds a dynamic EV-thinking section that addresses Optiver's finding:
LLMs understand EV conceptually but default to conservative/heuristic decisions.
This module injects a structured EV decision framework with live capital tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.loader import Config


class EVFrameworkFormatter:
    """Builds the Expected Value framework section for the system prompt."""

    def __init__(self, config: Config) -> None:
        self._config = config

    @property
    def starting_capital(self) -> float:
        return float(self._config.DEMO_QUOTE_CAPITAL)

    @property
    def fee_percent(self) -> float:
        return float(self._config.TRANSACTION_FEE_PERCENT)

    def build_ev_framework_section(self, current_capital: float) -> str:
        """Build the EV framework block injected into the system prompt.

        Args:
            current_capital: Current portfolio value from statistics tracker.
        """
        pnl = current_capital - self.starting_capital
        pnl_pct = (pnl / self.starting_capital) * 100 if self.starting_capital > 0 else 0.0
        fee_per_trade = self.starting_capital * self.fee_percent
        breakeven_ev = fee_per_trade * 1.5  # must exceed 1.5× fee to be +EV

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
            f"- Fee per round-trip trade: ${fee_per_trade:,.2f} ({self.fee_percent*100:.3f}%)",
            "",
            "### EV Decision Rule",
            "For every BUY or SELL decision, explicitly estimate:",
            "  1. P(win) — your assessed probability the trade reaches TP before SL",
            "  2. avg_win — expected profit in dollars if TP is hit",
            "  3. P(lose) — probability the trade hits SL first (≈ 1 − P(win))",
            "  4. avg_loss — expected loss in dollars if SL is hit",
            f"  5. EV = P(win) × avg_win + (1−P(win)) × (−avg_loss) − ${fee_per_trade:.2f}",
            "",
            "### EV Thresholds",
            f"- **Take the trade if EV > ${breakeven_ev:.2f}** (1.5× round-trip fee)",
            f"- **HOLD if EV is negative or below ${breakeven_ev:.2f}**",
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

    def build_ev_quick_section(self, current_capital: float) -> str:
        """Lightweight EV snapshot for the user prompt (changes every cycle)."""
        pnl = current_capital - self.starting_capital
        pnl_pct = (pnl / self.starting_capital) * 100 if self.starting_capital > 0 else 0.0

        return (
            f"## PORTFOLIO STATUS\n"
            f"Starting: ${self.starting_capital:,.0f} | "
            f"Current: ${current_capital:,.0f} | "
            f"P&L: ${pnl:+,.0f} ({pnl_pct:+.1f}%)\n"
        )
