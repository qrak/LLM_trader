"""One-shot self-correction: a decision that breaks the contract is re-asked once with
the exact violations attached; the second failure is logged and left to the guards."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.analyzer.analysis_result_processor import AnalysisResultProcessor
from src.analyzer.pattern_quality_scorer import PatternQualityScorer
from src.analyzer.risk_reward_validator import RiskRewardValidator
from src.analyzer.trend_validator import TrendValidator
from src.parsing.unified_parser import UnifiedParser
from src.utils.format_utils import FormatUtils


def _analysis(signal: str) -> dict:
    payload = {
        "signal": signal,
        "confidence": 70,
        "entry_price": None,
        "stop_loss": None,
        "take_profit": None,
        "risk_reward_ratio": None,
        "position_size": 0.0,
        "quantity": 0.0,
        "order_type": None,
        "reasoning": "test",
        "symbol": "BTC/USDC",
        "reduce_only": False,
        "leverage": 1,
    }
    if signal == "SELL":
        payload.update(entry_price=100.0, stop_loss=105.0, take_profit=None,
                       risk_reward_ratio=2.0, position_size=0.05)
    return payload


def _reply(signal: str) -> str:
    return json.dumps({
        "narrative": f"11) RISK/REWARD: N/A\n12) DECISION: {signal}",
        "analysis": _analysis(signal),
    })


def _processor(replies: list[str]):
    manager = MagicMock()
    manager.supports_image_analysis.return_value = False
    manager.send_prompt_streaming = AsyncMock(side_effect=replies)
    logger = MagicMock()
    parser = UnifiedParser(logger=logger, format_utils=FormatUtils())
    processor = AnalysisResultProcessor(
        manager, logger, parser, TrendValidator(), PatternQualityScorer(), RiskRewardValidator()
    )
    return processor, manager, logger


@pytest.mark.asyncio
async def test_violation_triggers_exactly_one_corrective_retry():
    processor, manager, _ = _processor([_reply("SELL"), _reply("HOLD")])

    result = await processor.process_analysis("system", "user prompt")

    assert manager.send_prompt_streaming.await_count == 2
    second_prompt = manager.send_prompt_streaming.await_args_list[1].kwargs["prompt"]
    assert "CONTRACT CORRECTION" in second_prompt
    assert "take_profit" in second_prompt
    assert result["analysis"]["signal"] == "HOLD"


@pytest.mark.asyncio
async def test_second_violation_is_logged_and_not_retried_again():
    processor, manager, logger = _processor([_reply("SELL"), _reply("SELL")])

    result = await processor.process_analysis("system", "user prompt")

    assert manager.send_prompt_streaming.await_count == 2
    logger.error.assert_called()
    # The R/R guard still owns the last word: a second violation ends as HOLD, not an order.
    assert result["analysis"]["signal"] == "HOLD"


@pytest.mark.asyncio
async def test_clean_response_is_not_retried():
    processor, manager, _ = _processor([_reply("HOLD")])

    await processor.process_analysis("system", "user prompt")

    assert manager.send_prompt_streaming.await_count == 1
