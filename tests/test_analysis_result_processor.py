"""Tests for AnalysisResultProcessor execution paths."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.analyzer.analysis_result_processor import AnalysisResultProcessor
from src.analyzer.pattern_quality_scorer import PatternQualityScorer
from src.analyzer.trend_validator import TrendValidator


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
