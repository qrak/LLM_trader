"""Unit tests for the Google GenAI provider request wiring."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai import errors

from src.platforms.ai_providers.google import GoogleAIClient


def _make_client() -> GoogleAIClient:
    return GoogleAIClient(api_key="test-key", model="gemini-test", logger=MagicMock())


def _fake_response(text: str = "ok") -> SimpleNamespace:
    part = SimpleNamespace(text=text)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    usage = SimpleNamespace(prompt_token_count=1, candidates_token_count=2, total_token_count=3)
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


class TestGenerationConfig:
    def test_config_omits_sampling_fields(self) -> None:
        client = _make_client()
        config = client._create_generation_config(
            {
                "max_tokens": 123,
                "thinking_level": "medium",
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 40,
            },
            effective_model="gemini-3.5-flash",
        )
        dumped = config.model_dump(exclude_none=False)
        assert dumped["max_output_tokens"] == 123
        assert "temperature" not in dumped or dumped["temperature"] is None
        assert "top_p" not in dumped or dumped["top_p"] is None
        assert "top_k" not in dumped or dumped["top_k"] is None

    def test_config_includes_legacy_sampling_for_gemini_2_models(self) -> None:
        client = _make_client()
        config = client._create_generation_config(
            {"max_tokens": 123, "temperature": 0.8, "top_p": 0.9, "top_k": 32},
            effective_model="gemini-2.5-flash",
        )

        assert config.temperature == 0.8
        assert config.top_p == 0.9
        assert config.top_k == 32

    def test_config_omits_sampling_for_unknown_models(self) -> None:
        client = _make_client()
        config = client._create_generation_config(
            {"max_tokens": 123, "temperature": 0.8, "top_p": 0.9, "top_k": 32},
            effective_model="vendor-model",
        )
        dumped = config.model_dump(exclude_none=False)

        assert "temperature" not in dumped or dumped["temperature"] is None
        assert "top_p" not in dumped or dumped["top_p"] is None
        assert "top_k" not in dumped or dumped["top_k"] is None

    def test_config_includes_code_execution_only_when_enabled(self) -> None:
        client = _make_client()
        config = client._create_generation_config(
            {"max_tokens": 123, "thinking_level": "high"},
            effective_model="gemini-3-flash-preview",
            include_code_execution=True,
        )
        assert config.tools is not None
        assert len(config.tools) == 1

    def test_config_omits_code_execution_for_legacy_models(self) -> None:
        client = _make_client()
        config = client._create_generation_config(
            {"max_tokens": 123, "thinking_level": "high"},
            effective_model="gemini-2.5-flash",
            include_code_execution=True,
        )

        assert config.tools is None


class TestThinkingFallback:
    def test_helper_uses_sdk_api_error_details(self) -> None:
        client = _make_client()
        exception = errors.APIError(
            400,
            {"error": {"message": "Invalid thinking_config field", "status": "INVALID_ARGUMENT"}},
        )

        assert client._should_retry_without_thinking(exception)

    @pytest.mark.asyncio
    async def test_chat_completion_retries_once_without_thinking(self) -> None:
        client = _make_client()
        generate_content = AsyncMock(side_effect=[ValueError("400 invalid thinking_config field"), _fake_response()])
        client.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))

        response = await client.chat_completion(
            model="gemini-test",
            messages=[{"role": "user", "content": "hello"}],
            model_config={"max_tokens": 123, "thinking_level": "high"},
        )

        assert response is not None
        assert response.choices[0].message.content == "ok"
        assert generate_content.await_count == 2
        first_config = generate_content.await_args_list[0].kwargs["config"]
        second_config = generate_content.await_args_list[1].kwargs["config"]
        assert first_config.thinking_config is not None
        assert second_config.thinking_config is None

    @pytest.mark.asyncio
    async def test_chat_completion_does_not_retry_non_thinking_errors(self) -> None:
        client = _make_client()
        generate_content = AsyncMock(side_effect=ValueError("400 invalid model"))
        client.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
        client._handle_exception = MagicMock(return_value=None)

        response = await client.chat_completion(
            model="gemini-test",
            messages=[{"role": "user", "content": "hello"}],
            model_config={"max_tokens": 123, "thinking_level": "high"},
        )

        assert response is None
        assert generate_content.await_count == 1
        client._handle_exception.assert_called_once()


class TestChartAnalysis:
    @pytest.mark.asyncio
    async def test_chart_analysis_sends_prompt_and_image_part(self) -> None:
        client = _make_client()
        generate_content = AsyncMock(return_value=_fake_response("chart ok"))
        client.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))

        response = await client.chat_completion_with_chart_analysis(
            model="gemini-3-flash-preview",
            messages=[{"role": "user", "content": "analyze chart"}],
            chart_image=b"fake-png-bytes",
            model_config={"max_tokens": 123, "thinking_level": "high", "google_code_execution": True},
        )

        assert response is not None
        assert response.choices[0].message.content == "chart ok"
        contents = generate_content.await_args.kwargs["contents"]
        assert contents[0] == "analyze chart"
        assert len(contents) == 2
        config = generate_content.await_args.kwargs["config"]
        assert config.tools is not None


class TestClientLifecycle:
    @pytest.mark.asyncio
    async def test_close_awaits_async_sdk_client_when_available(self) -> None:
        client = _make_client()
        aclose = AsyncMock()
        client.client = SimpleNamespace(aio=SimpleNamespace(aclose=aclose))

        await client.close()

        aclose.assert_awaited_once()
        assert client.client is None