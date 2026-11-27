

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.analyzer.ev_diagnostics import EVCalculator, EVInputs

if TYPE_CHECKING:
    from src.config.loader import Config


class EVFrameworkFormatter:
    def __init__(self, config: Config) -> None:
        self._config = config

    def ev_inputs(self, current_capital: float, inference_cost_usd: float = 0.0) -> EVInputs:

        capital = float(current_capital) if isinstance(current_capital, (int, float)) else 0.0
        if not math.isfinite(capital) or capital < 0:
            capital = 0.0
        inference_cost = 0.0
        try:
            candidate = float(inference_cost_usd)
        except (TypeError, ValueError):
            candidate = 0.0
        if math.isfinite(candidate):
            inference_cost = max(0.0, candidate)
        return EVInputs(
            capital=capital,
            fee_percent=float(self._config.TRANSACTION_FEE_PERCENT),
            inference_cost_usd=inference_cost,
        )

    def build_ev_framework_section(self, inputs: EVInputs) -> str:

        starting_capital = float(self._config.DEMO_QUOTE_CAPITAL)
        pnl = inputs.capital - starting_capital
        pnl_pct = (pnl / starting_capital) * 100 if starting_capital > 0 else 0.0
        side_fee_pct = inputs.fee_percent * 100
        round_trip_pct = side_fee_pct * 2
        lines = [
            "",
            "## EXPECTED VALUE FRAMEWORK",
            "",
            "Maximize expected value (EV) over a series of trades, not win rate.",
            "",
            "### Portfolio Status",
            f"- Starting Capital: ${starting_capital:,.2f}",
            f"- Current Capital: ${inputs.capital:,.2f}",
            f"- Realized P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)",
            f"- Round-trip trading fee: {round_trip_pct:.3f}% of position notional ({side_fee_pct:.3f}% per side)",
            f"- Inference cost (rolling average per analysis): ${inputs.inference_cost_usd:,.4f}",
            "",
            "### EV Decision Rule",
            (
                "- Evaluate each setup at its own proposed size, using the same entry, SL and TP for R/R and EV. "
                "Use the same quantity for both outcomes: capital × size_fraction ÷ entry. "
                "Gross win = quantity × |TP − entry|; gross loss = quantity × |entry − SL|."
            ),
            (
                "- Cost = round-trip fee on that setup's notional (capital × size_fraction) + rolling inference cost. "
                "EV = p_win × gross win − (1 − p_win) × gross loss − cost. "
                f"The EV condition is strictly EV > {EVCalculator.THRESHOLD_MULTIPLE:g}× that setup's cost; "
                "HOLD if it fails. Never use a fee or threshold from another position size."
            ),
            (
                "- Use EV internally alongside the evidence and risk rules. Do not report numeric EV, "
                "its threshold or a required win probability. Python computes diagnostics from ev_scenarios "
                "after the response; a scenario is not an order."
            ),
            (
                "- Put your estimate of reaching TP before SL only in p_win. Explain its basis qualitatively "
                "in reasoning (confluences, regime and structure). Brain's historical win rate and sample size "
                "are not p_win. An unsupported or inflated estimate does not justify entry."
            ),
            "- Do not reject a setup that clears the evidence, risk and EV conditions purely out of fear of loss.",
            "",
        ]
        return "\n".join(lines)
