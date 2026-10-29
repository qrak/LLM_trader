

import asyncio
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.managers.post_mortem_repository import PostMortemRepository
from src.parsing.unified_parser import UnifiedParser

VERDICT_PATTERN = r"^[a-z0-9]+(?:_[a-z0-9]+)*$"
MAX_VERDICT_LENGTH = 64
MAX_TEXT_FIELD_LENGTH = 4000
RESPONSE_PREVIEW_LENGTH = 200


class PostMortemResult(BaseModel):
    """Validated post-mortem analysis from the LLM.

    Hard contract: every field must be present, non-blank and of a plausible
    length, and the verdict must be a snake_case tag. Missing fields are
    rejected — they are never filled in with invented content.
    """

    verdict: str = Field(
        ...,
        min_length=1,
        max_length=MAX_VERDICT_LENGTH,
        pattern=VERDICT_PATTERN,
        description="Short snake_case tag, e.g. overestimated_breakout",
    )
    llm_analysis: str = Field(
        ..., min_length=1, max_length=MAX_TEXT_FIELD_LENGTH, description="Full analysis of what happened"
    )
    expected_vs_actual: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_FIELD_LENGTH,
        description="What was expected vs what actually happened",
    )
    lesson_learned: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_FIELD_LENGTH,
        description="Concise actionable lesson for future trades",
    )

    @field_validator("verdict", "llm_analysis", "expected_vs_actual", "lesson_learned")
    @classmethod
    def _reject_blank(cls, value: str) -> str:

        if not isinstance(value, str) or not value.strip():
            raise ValueError("must not be blank")
        return value.strip()



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
        unified_parser: UnifiedParser,
        repository: PostMortemRepository,
    ) -> None:

        self.logger = logger
        self.model_manager = model_manager
        self.unified_parser = unified_parser
        self.repository = repository
        self._last_parse_failure_reason = ""

    async def analyze_closed_trade(
        self,
        closed_position: Any,
        entry_decision: Any,
        exit_decision: Any,
        pnl: float,
        reason: str,
        trade_id: int | None = None,
        market_conditions: Any | None = None,
    ) -> PostMortemResult | None:

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
                self.logger.warning(
                    "Post-mortem: failed to parse LLM response (reason=%s) | preview: %s",
                    self._last_parse_failure_reason,
                    self._response_preview(response_text),
                )
                return None

            await asyncio.to_thread(
                self.repository.insert_post_mortem,
                trade_id=trade_id,
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

        lines = [
            "Analyze the following closed trade and produce a post-mortem.",
            "",
            f"## Trade: {closed_position.symbol} {closed_position.direction}",
            f"## Close Reason: {reason}",
            f"## P&L: {pnl:+.2f}%",
            "",
            "## Entry Data:",
            f"- Entry Price: {closed_position.entry_price}",
            f"- Stop Loss: {closed_position.stop_loss}",
            f"- Take Profit: {closed_position.take_profit}",
            f"- Position Size: {closed_position.size_pct:.1%} of capital",
            f"- Confidence at Entry: {closed_position.confidence}",
            f"- ADX at Entry: {closed_position.adx_at_entry}",
            f"- RSI at Entry: {closed_position.rsi_at_entry}",
            f"- Trend at Entry: {closed_position.trend_direction_at_entry}",
            f"- Volatility at Entry: {closed_position.volatility_level}",
            f"- R/R Ratio at Entry: {closed_position.rr_ratio_at_entry}",
            f"- Max Drawdown During Trade: {closed_position.max_drawdown_pct:.2f}%",
            f"- Max Profit During Trade: {closed_position.max_profit_pct:.2f}%",
        ]

        entry_reasoning = entry_decision.reasoning or "(no reasoning recorded)"
        lines.extend(["", "## Original Entry Reasoning:", entry_reasoning])

        lines.extend([
            "",
            "## Exit Data:",
            f"- Exit Price: {exit_decision.price}",
            f"- Exit Reasoning: {exit_decision.reasoning}",
        ])

        entry_time = closed_position.entry_time
        exit_timestamp = exit_decision.timestamp
        if entry_time is not None and exit_timestamp is not None:
            try:
                hold_duration = exit_timestamp - entry_time
                lines.append(f"- Hold Duration: {hold_duration}")
            except TypeError:
                pass

        if market_conditions is not None:
            lines.extend(["", "## Market Conditions at Exit:", str(market_conditions)])

        lines.extend(["", "Produce the JSON post-mortem now."])
        return "\n".join(lines)

    def _parse_response(self, response_text: str) -> PostMortemResult | None:

        self._last_parse_failure_reason = "unknown"
        data = self._extract_payload(response_text)
        if data is None:
            self._last_parse_failure_reason = "no decodable JSON object in the response"
            return None
        try:
            return PostMortemResult(**data)
        except ValidationError as error:
            self._last_parse_failure_reason = f"schema validation failed: {error.errors()[0].get('msg', error)}"
            return None
        except TypeError as error:
            self._last_parse_failure_reason = f"payload is not a mapping: {error}"
            return None

    def _extract_payload(self, response_text: str) -> dict[str, Any] | None:

        if not response_text or not response_text.strip():
            return None

        try:
            data = self.unified_parser.extract_json_block(response_text)
        except Exception as error:  # noqa: BLE001
            self.logger.debug("Post-mortem markdown parse error: %s", error)
            data = None
        if data:
            return data

        return UnifiedParser.parse_json_object(response_text)

    @staticmethod
    def _response_preview(response_text: str, limit: int = RESPONSE_PREVIEW_LENGTH) -> str:

        if not response_text:
            return "<empty>"
        preview = " ".join(str(response_text).split())
        if len(preview) > limit:
            preview = preview[:limit] + "..."
        return preview

