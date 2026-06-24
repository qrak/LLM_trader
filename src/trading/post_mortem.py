"""LLM-driven post-mortem analysis for closed trades."""

import asyncio
import json
from typing import Any

from pydantic import BaseModel, Field

from src.managers.post_mortem_repository import PostMortemRepository


class PostMortemResult(BaseModel):
    """Validated post-mortem analysis from the LLM."""

    verdict: str = Field(..., min_length=1, description="Short snake_case tag, e.g. overestimated_breakout")
    llm_analysis: str = Field(..., min_length=1, description="Full analysis of what happened")
    expected_vs_actual: str = Field(..., min_length=1, description="What was expected vs what actually happened")
    lesson_learned: str = Field(..., min_length=1, description="Concise actionable lesson for future trades")


POST_MORTEM_SYSTEM_PROMPT = """\
You are a trading post-mortem analyst. You analyze closed trades to extract actionable lessons.

You will receive:
- The original entry reasoning (why the trade was opened)
- Entry indicators (ADX, RSI, trend, volatility, SL/TP, confidence, direction)
- Exit data (exit reason, exit price, P&L %, hold duration)
- Market conditions at exit

Produce a JSON object with EXACTLY these fields:
{
  "verdict": "snake_case_tag",           // e.g. overestimated_breakout, good_exit, plan_followed, premature_entry, held_too_long
  "llm_analysis": "2-4 sentence analysis of what happened",
  "expected_vs_actual": "what was expected vs what actually happened",
  "lesson_learned": "one concise actionable sentence for future trades"
}

Rules:
- Output ONLY the JSON object, no markdown, no explanation before or after.
- verdict must be snake_case, no spaces.
- lesson_learned must be phrased as guidance ("When X, do Y").
- Be honest: if the trade was good, say so. If it was bad, identify the specific mistake."""


class PostMortemService:
    """Orchestrates LLM post-mortem analysis after position close."""

    def __init__(
        self,
        logger: Any,
        model_manager: Any,
        unified_parser: Any,
        repository: PostMortemRepository,
    ) -> None:
        """Initialize post-mortem service dependencies.

        Args:
            logger: Logger instance.
            model_manager: ModelManager for LLM calls (reuses config provider chain).
            unified_parser: UnifiedParser for JSON extraction.
            repository: PostMortemRepository for storage.
        """
        self.logger = logger
        self.model_manager = model_manager
        self.unified_parser = unified_parser
        self.repository = repository

    async def analyze_closed_trade(
        self,
        closed_position: Any,
        entry_decision: Any,
        exit_decision: Any,
        pnl: float,
        reason: str,
        market_conditions: Any | None = None,
    ) -> PostMortemResult | None:
        """Analyze a closed trade and store the post-mortem.

        Args:
            closed_position: The Position object that was just closed.
            entry_decision: The original entry TradeDecision (must not be None).
            exit_decision: The exit TradeDecision just written to SQLite.
            pnl: P&L percentage of the closed trade.
            reason: Close reason (stop_loss / take_profit / analysis_signal).
            market_conditions: Optional MarketConditions at exit time.

        Returns:
            PostMortemResult if successful, None on any failure.
        """
        try:
            prompt = self._build_prompt(closed_position, entry_decision, exit_decision, pnl, reason, market_conditions)
            response_text = await self.model_manager.send_prompt(
                prompt=prompt,
                system_message=POST_MORTEM_SYSTEM_PROMPT,
                provider=None,
                model=None,
            )
            if not response_text:
                self.logger.warning("Post-mortem: empty LLM response")
                return None

            result = self._parse_response(response_text)
            if result is None:
                self.logger.warning("Post-mortem: failed to parse LLM response")
                return None

            # Store in SQLite + FTS5 (offload to thread for async safety)
            await asyncio.to_thread(
                self.repository.insert_post_mortem,
                trade_id=None,
                symbol=closed_position.symbol,
                direction=closed_position.direction,
                verdict=result.verdict,
                llm_analysis=result.llm_analysis,
                expected_vs_actual=result.expected_vs_actual,
                lesson_learned=result.lesson_learned,
                pnl_pct=pnl,
                close_reason=reason,
            )
            self.logger.info(
                "Post-mortem stored: %s %s verdict=%s pnl=%.2f%%",
                closed_position.symbol, closed_position.direction, result.verdict, pnl,
            )
            return result
        except Exception:
            self.logger.warning("Post-mortem analysis failed", exc_info=True)
            return None

    def _build_prompt(
        self,
        closed_position: Any,
        entry_decision: Any,
        exit_decision: Any,
        pnl: float,
        reason: str,
        market_conditions: Any | None,
    ) -> str:
        """Build the user prompt with trade data for the LLM."""
        lines = [
            "Analyze the following closed trade and produce a post-mortem.",
            "",
            f"## Trade: {getattr(closed_position, 'symbol', '?')} {getattr(closed_position, 'direction', '?')}",
            f"## Close Reason: {reason}",
            f"## P&L: {pnl:+.2f}%",
            "",
            "## Entry Data:",
            f"- Entry Price: {getattr(closed_position, 'entry_price', '?')}",
            f"- Stop Loss: {getattr(closed_position, 'stop_loss', '?')}",
            f"- Take Profit: {getattr(closed_position, 'take_profit', '?')}",
            f"- Position Size: {getattr(closed_position, 'size_pct', 0):.1%} of capital",
            f"- Confidence at Entry: {getattr(closed_position, 'confidence', '?')}",
            f"- ADX at Entry: {getattr(closed_position, 'adx_at_entry', '?')}",
            f"- RSI at Entry: {getattr(closed_position, 'rsi_at_entry', '?')}",
            f"- Trend at Entry: {getattr(closed_position, 'trend_direction_at_entry', '?')}",
            f"- Volatility at Entry: {getattr(closed_position, 'volatility_level', '?')}",
            f"- R/R Ratio at Entry: {getattr(closed_position, 'rr_ratio_at_entry', '?')}",
            f"- Max Drawdown During Trade: {getattr(closed_position, 'max_drawdown_pct', 0):.2f}%",
            f"- Max Profit During Trade: {getattr(closed_position, 'max_profit_pct', 0):.2f}%",
        ]

        # Entry reasoning (the original AI justification for opening)
        entry_reasoning = getattr(entry_decision, 'reasoning', '') or '(no reasoning recorded)'
        lines.extend(["", "## Original Entry Reasoning:", entry_reasoning])

        # Exit data
        lines.extend([
            "",
            "## Exit Data:",
            f"- Exit Price: {getattr(exit_decision, 'price', '?')}",
            f"- Exit Reasoning: {getattr(exit_decision, 'reasoning', '?')}",
        ])

        # Hold duration
        entry_time = getattr(closed_position, 'entry_time', None)
        exit_timestamp = getattr(exit_decision, 'timestamp', None)
        if entry_time is not None and exit_timestamp is not None:
            try:
                hold_duration = exit_timestamp - entry_time
                lines.append(f"- Hold Duration: {hold_duration}")
            except Exception:
                pass

        # Market conditions at exit (if available)
        if market_conditions is not None:
            lines.extend(["", "## Market Conditions at Exit:", str(market_conditions)])

        lines.extend(["", "Produce the JSON post-mortem now."])
        return "\n".join(lines)

    def _parse_response(self, response_text: str) -> PostMortemResult | None:
        """Parse and validate the LLM response into PostMortemResult.

        Tries markdown code block extraction first, then falls back to raw JSON
        parsing. Validates with Pydantic.
        """
        try:
            data = self.unified_parser.extract_json_block(response_text)
            if data:
                return PostMortemResult(**data)
        except Exception as e:
            self.logger.debug("Post-mortem markdown parse error: %s", e)

        # Fallback: try raw JSON (LLM may return plain JSON without ``` fences)
        try:
            data = json.loads(response_text.strip())
            return PostMortemResult(**data)
        except Exception as e:
            self.logger.debug("Post-mortem raw JSON parse error: %s", e)
            return None
