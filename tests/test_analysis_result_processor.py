"""Tests for AnalysisResultProcessor execution paths."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.analyzer.analysis_result_processor import AnalysisResultProcessor
from src.analyzer.pattern_quality_scorer import PatternQualityScorer
from src.analyzer.trend_validator import TrendValidator
from src.parsing.unified_parser import UnifiedParser
from src.utils.format_utils import FormatUtils


def _make_processor(*, supports_image=True, chart_error: Exception | None = None):
    model_manager = MagicMock()
    model_manager.supports_image_analysis.return_value = supports_image
    model_manager.describe_provider_and_model.return_value = ("google", "gemini-3.5-flash")

    if chart_error is None:
        model_manager.send_prompt_with_chart_analysis = AsyncMock(return_value='{"analysis": {"signal": "HOLD"}}')
    else:
        model_manager.send_prompt_with_chart_analysis = AsyncMock(side_effect=chart_error)

    model_manager.send_prompt_streaming = AsyncMock(return_value='{"analysis": {"signal": "HOLD"}}')

    unified_parser = MagicMock()
    unified_parser.parse_ai_response.return_value = {"analysis": {"signal": "HOLD", "trend": {}}}
    unified_parser.validate_ai_response.return_value = True

    processor = AnalysisResultProcessor(
        model_manager=model_manager,
        logger=MagicMock(),
        unified_parser=unified_parser,
        trend_validator=TrendValidator(),
        quality_scorer=PatternQualityScorer(),
    )
    return processor


@pytest.mark.asyncio
async def test_process_analysis_uses_chart_path_when_supported() -> None:
    processor = _make_processor(supports_image=True)

    await processor.process_analysis(system_prompt="system", prompt="prompt", chart_image=b"img")

    processor.model_manager.send_prompt_with_chart_analysis.assert_awaited_once()
    processor.model_manager.send_prompt_streaming.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_analysis_falls_back_to_streaming_when_chart_call_fails() -> None:
    processor = _make_processor(supports_image=True, chart_error=Exception("chart failure"))

    await processor.process_analysis(system_prompt="system", prompt="prompt", chart_image=b"img")

    processor.model_manager.send_prompt_with_chart_analysis.assert_awaited_once()
    processor.model_manager.send_prompt_streaming.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_analysis_uses_streaming_when_chart_not_supported() -> None:
    processor = _make_processor(supports_image=False)

    await processor.process_analysis(system_prompt="system", prompt="prompt", chart_image=b"img")

    processor.model_manager.send_prompt_with_chart_analysis.assert_not_awaited()
    processor.model_manager.send_prompt_streaming.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_analysis_returns_error_when_response_validation_fails() -> None:
    processor = _make_processor(supports_image=False)
    processor.unified_parser.validate_ai_response.return_value = False

    result = await processor.process_analysis(system_prompt="system", prompt="prompt")

    assert result["error"] == "Invalid response format"
    assert "raw_response" in result


def test_invocation_result_error_formatting() -> None:
    from src.managers.provider_types import InvocationResult

    res_custom = InvocationResult(
        success=False,
        response=None,
        provider="google",
        model="gemini-3.5-flash",
        error_message="Empty or invalid response content from Google AI"
    )
    assert res_custom.error == "Empty or invalid response content from Google AI"


@pytest.mark.asyncio
async def test_process_analysis_logs_clean_chart_warning() -> None:
    processor = _make_processor(supports_image=True, chart_error=ValueError("Empty response content from Google AI"))

    await processor.process_analysis(system_prompt="system", prompt="prompt", chart_image=b"img")

    processor.logger.warning.assert_called_once()
    warning_args = processor.logger.warning.call_args[0]
    assert warning_args[0] == "Chart analysis failed: %s. Falling back to text-only analysis."
    assert "Empty response content from Google AI" in str(warning_args[1])
    assert not str(warning_args[1]).startswith("Chart analysis failed:")


def _json_block(signal: str = "HOLD") -> str:
    return (
        "```json\n"
        f'{{"analysis": {{"signal": "{signal}", "confidence": 82, "entry_price": 77880, '
        '"stop_loss": 76500, "take_profit": 80640, "position_size": 0.08, '
        '"risk_reward_ratio": 2.0, "reasoning": "Valid setup."}}\n'
        "```"
    )


def _make_repair_processor(first_response: str, repair_response: str):
    model_manager = MagicMock()
    model_manager.supports_image_analysis.return_value = False
    model_manager.send_prompt_streaming = AsyncMock(return_value=first_response)
    model_manager.send_contract_repair = AsyncMock(return_value=repair_response)

    processor = AnalysisResultProcessor(
        model_manager=model_manager,
        logger=MagicMock(),
        unified_parser=UnifiedParser(logger=MagicMock(), format_utils=FormatUtils()),
        trend_validator=TrendValidator(),
        quality_scorer=PatternQualityScorer(),
    )
    return processor, model_manager


@pytest.mark.asyncio
async def test_process_analysis_repairs_missing_json_block() -> None:
    narrative = "1) MARKET STRUCTURE: price below both SMAs; 2) DECISION: HOLD — no edge."
    processor, model_manager = _make_repair_processor(first_response=narrative, repair_response=_json_block("HOLD"))

    result = await processor.process_analysis(system_prompt="system", prompt="prompt")

    model_manager.send_contract_repair.assert_awaited_once()
    repair_kwargs = model_manager.send_contract_repair.await_args.kwargs
    assert repair_kwargs["system_message"] == "system"
    assert repair_kwargs["prompt"] == "prompt"
    assert repair_kwargs["previous_response"] == narrative
    assert result["analysis"]["signal"] == "HOLD"
    assert result["response_validation"]["status"] == "valid"
    assert "parse_error" not in result
    assert result["raw_response"].startswith(narrative)
    assert "```json" in result["raw_response"]


@pytest.mark.asyncio
async def test_process_analysis_skips_repair_when_json_block_present() -> None:
    processor, model_manager = _make_repair_processor(first_response=_json_block("BUY"), repair_response="unused")

    result = await processor.process_analysis(system_prompt="system", prompt="prompt")

    model_manager.send_contract_repair.assert_not_awaited()
    assert result["analysis"]["signal"] == "BUY"
    assert result["response_validation"]["status"] == "valid"


@pytest.mark.asyncio
async def test_process_analysis_keeps_fallback_when_repair_fails() -> None:
    processor, model_manager = _make_repair_processor(first_response="narrative only", repair_response="still no json here")

    result = await processor.process_analysis(system_prompt="system", prompt="prompt")

    model_manager.send_contract_repair.assert_awaited_once()
    assert result["parse_error"] == "Failed to parse response"
    assert result["response_validation"]["status"] == "invalid"
    assert result["raw_response"] == "narrative only"

