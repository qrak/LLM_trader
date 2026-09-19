"""Provider-client wiring for the AI provider chain.

Covers the five clients that wrap an SDK (Google GenAI, OpenRouter, DeepSeek,
LM Studio, BlockRun.AI), the contracts they inherit from ``BaseAIClient``, and
the ``ProviderOrchestrator`` routes that dispatch text and chart requests
through them (single-provider, fallback chain, availability and guidance).
"""
import base64
import io
import struct
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from blockrun_llm.types import ChatChoice, ChatMessage, ChatResponse
from google.genai import errors

import src.platforms.ai_providers.deepseek as deepseek_module
import src.platforms.ai_providers.lmstudio as lmstudio_module
import src.platforms.ai_providers.openrouter as openrouter_module
from src.config.loader import VALID_PROVIDERS, Config
from src.managers.provider_orchestrator import ProviderOrchestrator
from src.managers.provider_types import InvocationResult, ProviderClients
from src.platforms.ai_providers.base import BaseAIClient
from src.platforms.ai_providers.blockrun import BlockRunClient
from src.platforms.ai_providers.deepseek import DeepSeekClient
from src.platforms.ai_providers.google import GoogleAIClient
from src.platforms.ai_providers.lmstudio import LMStudioClient
from src.platforms.ai_providers.openrouter import OpenRouterClient
from src.platforms.ai_providers.response_models import ChatResponseModel
from tests.conftest import make_config

_WALLET_KEY = "0x0000000000000000000000000000000000000000000000000000000000000001"
_OPENROUTER_MODELS = {
    "OPENROUTER_BASE_MODEL": "primary/model",
    "OPENROUTER_FALLBACK_MODEL": "fallback/model",
}
_DEEPSEEK_USAGE_EXTRA = {
    "completion_tokens_details": SimpleNamespace(reasoning_tokens=5),
    "prompt_cache_hit_tokens": 7,
    "prompt_cache_miss_tokens": 4,
}


def _google_client() -> GoogleAIClient:
    return GoogleAIClient(api_key="test-key", model="gemini-test", logger=MagicMock())


def _openrouter_client() -> OpenRouterClient:
    return OpenRouterClient(
        api_key="test-key", base_url="https://example.test/api/v1", logger=MagicMock()
    )


def _deepseek_client() -> DeepSeekClient:
    return DeepSeekClient(
        api_key="test-key", base_url="https://api.deepseek.test", logger=MagicMock()
    )


def _lmstudio_client() -> LMStudioClient:
    return LMStudioClient(base_url="http://localhost:1234/v1", logger=MagicMock())


def _blockrun_client(wallet_key: str = _WALLET_KEY) -> BlockRunClient:
    return BlockRunClient(
        wallet_key=wallet_key, base_url="https://blockrun.ai/api", logger=MagicMock()
    )


def _sdk_response(
    text: str | None = "ok",
    tokens: tuple[int, int, int] | None = (1, 2, 3),
    *,
    usage_extra: dict[str, Any] | None = None,
    response_id: str = "gen-123",
    model: str = "test-model",
    role: str = "assistant",
) -> SimpleNamespace:
    """Completion double for the three OpenAI-shaped clients, shaped like the SDK."""
    message = SimpleNamespace(role=role, content=text)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = None
    if tokens is not None:
        usage = SimpleNamespace(
            prompt_tokens=tokens[0],
            completion_tokens=tokens[1],
            total_tokens=tokens[2],
            **(usage_extra or {}),
        )
    return SimpleNamespace(choices=[choice], usage=usage, id=response_id, model=model)


def _google_sdk_response(text: str = "ok") -> SimpleNamespace:
    """Google SDK response double with candidates, parts and usage metadata."""
    content = SimpleNamespace(parts=[SimpleNamespace(text=text)])
    candidate = SimpleNamespace(content=content)
    usage = SimpleNamespace(
        prompt_token_count=1, candidates_token_count=2, total_token_count=3
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


def _openai_sdk(create: Any, models_list: Any | None = None) -> SimpleNamespace:
    """OpenAI SDK client double exposing chat.completions.create and models.list."""
    models = SimpleNamespace(list=models_list) if models_list else SimpleNamespace()
    return SimpleNamespace(
        models=models, chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def _png_bytes(width: int, height: int) -> bytes:
    """Minimal PNG signature plus IHDR header carrying the given dimensions."""
    header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR"
    return header + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00\xaa\xbb\xcc\xdd"


def _orchestrator_config(**overrides: Any) -> SimpleNamespace:
    """Orchestrator config double on conftest production defaults plus get_model_config."""
    config = make_config(**overrides)
    config.get_model_config = lambda _model: {"max_tokens": 16}
    return config


def _orchestrator(
    config: SimpleNamespace | None = None, **clients: Any
) -> ProviderOrchestrator:
    """Orchestrator wired with only the provider doubles passed as keywords."""
    return ProviderOrchestrator(
        logger=MagicMock(),
        config=config if config else _orchestrator_config(),
        clients=ProviderClients(**clients),
    )


class _RecordingProvider:
    """Provider double recording the (route, model) of every call, replaying queued replies."""

    def __init__(self, responses: list[ChatResponseModel | None]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    async def chat_completion(
        self, model: str, _messages: list[dict[str, Any]], _model_config: dict[str, Any]
    ) -> ChatResponseModel | None:
        self.calls.append(("text", model))
        return self.responses.pop(0)

    async def chat_completion_with_chart_analysis(
        self,
        model: str,
        _messages: list[dict[str, Any]],
        _chart_image: Any,
        _model_config: dict[str, Any],
    ) -> ChatResponseModel | None:
        self.calls.append(("chart", model))
        return self.responses.pop(0)


class _FakeHttpxResponse:
    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.test/api/v1/generation")
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=request,
                response=httpx.Response(self.status_code, request=request),
            )


class _FakeHttpxClient:
    def __init__(self, response: _FakeHttpxResponse) -> None:
        self._response = response
        self.requests: list[dict[str, Any]] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb) -> bool:
        return False

    async def get(
        self, url: str, params: Any = None, headers: Any = None
    ) -> _FakeHttpxResponse:
        self.requests.append({"url": url, "params": params, "headers": headers})
        return self._response


def _patch_httpx(monkeypatch: pytest.MonkeyPatch, response: _FakeHttpxResponse) -> _FakeHttpxClient:
    """Point the OpenRouter REST lookup at a scripted httpx client."""
    fake = _FakeHttpxClient(response)
    monkeypatch.setattr(openrouter_module.httpx, "AsyncClient", lambda **_kwargs: fake)
    return fake


class _FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> Any:
        async def gen() -> Any:
            for chunk in self._chunks:
                yield chunk

        return gen()


class _BrokenStream:
    """Yields the given chunks, then raises mid-iteration."""

    def __init__(self, chunks: list[Any], error: Exception) -> None:
        self._chunks = chunks
        self._error = error

    def __aiter__(self) -> Any:
        async def gen() -> Any:
            for chunk in self._chunks:
                yield chunk
            raise self._error

        return gen()


def _stream_chunk(
    text: str | None = None, usage: SimpleNamespace | None = None
) -> SimpleNamespace:
    """Streaming chunk double; a None text yields the usage-only keep-alive shape."""
    choices = [SimpleNamespace(delta=SimpleNamespace(content=text))] if text is not None else []
    return SimpleNamespace(choices=choices, usage=usage)


def _blockrun_sdk_response(content: str) -> ChatResponse:
    return ChatResponse(
        id="test-id",
        object="chat.completion",
        created=1234567890,
        model="openai/gpt-4o",
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        cost_usd=0.001,
    )


class TestClientConstructionAndShutdown:
    @pytest.mark.parametrize(
        ("module", "factory", "expected_kwargs"),
        [
            (
                openrouter_module,
                _openrouter_client,
                {"api_key": "test-key", "base_url": "https://example.test/api/v1"},
            ),
            (
                deepseek_module,
                _deepseek_client,
                {"api_key": "test-key", "base_url": "https://api.deepseek.test"},
            ),
            (
                lmstudio_module,
                _lmstudio_client,
                {"base_url": "http://localhost:1234/v1", "api_key": "lm-studio"},
            ),
        ],
        ids=["openrouter", "deepseek", "lmstudio"],
    )
    async def test_initialize_client_builds_sdk_with_provider_credentials(
        self, module: Any, factory: Any, expected_kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_async_openai(**kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace()

        monkeypatch.setattr(module, "AsyncOpenAI", fake_async_openai)
        client = factory()

        await client._initialize_client()

        assert calls == [expected_kwargs]
        assert client._client is not None

    async def test_close_awaits_sdk_teardown_and_clears_the_handle(self) -> None:
        """All four SDK-backed clients await their transport close and drop the handle."""
        for factory in (_openrouter_client, _deepseek_client, _lmstudio_client):
            client = factory()
            close_mock = AsyncMock()
            client._client = SimpleNamespace(close=close_mock)

            await client.close()

            close_mock.assert_awaited_once()
            assert client._client is None, factory.__name__

        google = _google_client()
        aclose = AsyncMock()
        google.client = SimpleNamespace(aio=SimpleNamespace(aclose=aclose))

        await google.close()

        aclose.assert_awaited_once()
        assert google.client is None


class TestSharedBaseContract:
    @pytest.mark.parametrize(
        "factory",
        [
            _google_client,
            _openrouter_client,
            _deepseek_client,
            _lmstudio_client,
            _blockrun_client,
        ],
        ids=["google", "openrouter", "deepseek", "lmstudio", "blockrun"],
    )
    def test_shared_base_helpers_are_inherited_by_every_client(self, factory: Any) -> None:
        """Every client inherits BaseAIClient's text, redaction and param-detection helpers."""
        client = factory()
        assert type(client).__mro__[1] is BaseAIClient

        messages = [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "last"},
        ]
        assert client._extract_user_text_from_messages(messages) == "last"
        client.api_key = "secret-api-key"
        assert client._sanitize_error_message("boom secret-api-key") == "boom [REDACTED_API_KEY]"
        assert client.process_chart_image(io.BytesIO(b"png-bytes")) == b"png-bytes"

        assert client._detect_unsupported_param(
            "got an unexpected keyword argument 'reasoning_effort'"
        ) == "reasoning_effort"
        assert client._detect_unsupported_param("unknown parameter: 'top_k'") == "top_k"
        assert client._detect_unsupported_param(
            "Additional properties are not allowed ('thinking_budget' was unexpected)"
        ) == "thinking_budget"
        assert client._detect_unsupported_param("plain provider failure") is None


class TestGoogleGenerationConfig:
    @pytest.mark.parametrize(
        ("model_config", "include_thinking", "include_code_execution", "expect_thinking", "expect_tools"),
        [
            ({"max_tokens": 123, "thinking_level": "medium"}, True, False, True, False),
            ({"max_tokens": 123}, False, False, False, False),
            ({"max_tokens": 123, "thinking_level": "high"}, True, True, True, True),
            ({"max_tokens": 123, "thinking_level": "high"}, False, False, False, False),
            ({"max_tokens": 123, "thinking_level": "ultra"}, True, False, False, False),
            (
                {"max_tokens": 123, "temperature": 0.7, "top_p": 0.9, "top_k": 40},
                False,
                False,
                False,
                False,
            ),
        ],
        ids=[
            "thinking",
            "no-thinking",
            "thinking+tools",
            "no-tools",
            "unknown-thinking-level",
            "sampling-ignored",
        ],
    )
    def test_generation_config_sends_tokens_thinking_and_optional_tools(
        self,
        model_config: dict[str, Any],
        include_thinking: bool,
        include_code_execution: bool,
        expect_thinking: bool,
        expect_tools: bool,
    ) -> None:
        """Google sends max_output_tokens plus optional thinking/tools and never sampling knobs."""
        client = _google_client()

        config = client._create_generation_config(
            model_config,
            include_thinking=include_thinking,
            include_code_execution=include_code_execution,
        )
        dumped = config.model_dump(exclude_none=False)

        assert dumped["max_output_tokens"] == 123
        assert dumped["temperature"] is None
        assert dumped["top_p"] is None
        assert dumped["top_k"] is None
        assert (config.thinking_config is not None) is expect_thinking
        assert (config.tools is not None) is expect_tools
        if expect_tools:
            assert len(config.tools) == 1


class TestGoogleThinkingFallback:
    def test_unsupported_feature_detection_uses_sdk_error_details(self) -> None:
        """Only a 400 status naming the feature triggers the thinking/code-execution retry."""
        client = _google_client()
        thinking_error = errors.APIError(
            400, {"error": {"message": "Invalid thinking_config field", "status": "INVALID_ARGUMENT"}}
        )
        code_error = errors.APIError(
            400, {"error": {"message": "unsupported code_execution tool", "status": "INVALID_ARGUMENT"}}
        )

        assert client._should_retry_without_thinking(thinking_error)
        assert client._should_retry_without_code_execution(code_error)
        assert not client._should_retry_without_thinking(code_error)
        assert not client._should_retry_without_thinking(
            errors.APIError(403, {"error": {"message": "Invalid thinking_config field"}})
        )
        assert not client._should_retry_without_thinking(ValueError("network hiccup"))

    async def test_chat_completion_retries_without_thinking_only_for_thinking_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A thinking_config rejection costs one extra call; any other error is returned as-is."""
        client = _google_client()
        generate_content = AsyncMock(
            side_effect=[
                ValueError("400 invalid thinking_config field"),
                _google_sdk_response("ok"),
            ]
        )
        client.client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        )

        response = await client.chat_completion(
            model="gemini-test",
            messages=[{"role": "user", "content": "hello"}],
            model_config={"max_tokens": 123, "thinking_level": "high"},
        )

        assert response.choices[0].message.content == "ok"
        assert response.usage.prompt_tokens == 1
        assert response.usage.completion_tokens == 2
        assert response.usage.total_tokens == 3
        assert generate_content.await_count == 2
        first_config = generate_content.await_args_list[0].kwargs["config"]
        second_config = generate_content.await_args_list[1].kwargs["config"]
        assert first_config.thinking_config is not None
        assert second_config.thinking_config is None

        failing = _google_client()
        rejections = AsyncMock(side_effect=ValueError("400 invalid model"))
        failing.client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=rejections))
        )
        handler = MagicMock(return_value=None)
        monkeypatch.setattr(failing, "_handle_exception", handler)

        assert (
            await failing.chat_completion(
                model="gemini-test",
                messages=[{"role": "user", "content": "hello"}],
                model_config={"max_tokens": 123, "thinking_level": "high"},
            )
            is None
        )
        assert rejections.await_count == 1
        handler.assert_called_once()


class TestGoogleChartAndImageCost:
    async def test_chart_analysis_sends_prompt_and_image_with_code_execution(self) -> None:
        """The chart request carries the flattened prompt plus one image part and the tool."""
        client = _google_client()
        generate_content = AsyncMock(return_value=_google_sdk_response("chart ok"))
        client.client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        )

        response = await client.chat_completion_with_chart_analysis(
            model="gemini-3-flash-preview",
            messages=[{"role": "user", "content": "analyze chart"}],
            chart_image=b"fake-png-bytes",
            model_config={"max_tokens": 123, "thinking_level": "high", "google_code_execution": True},
        )

        assert response.choices[0].message.content == "chart ok"
        contents = generate_content.await_args.kwargs["contents"]
        assert contents[0] == "analyze chart"
        assert len(contents) == 2
        assert generate_content.await_args.kwargs["config"].tools is not None

    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            (_png_bytes(16, 8), (16, 8)),
            (
                (
                    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\xff\xc0\x00\x0b\x08"
                    b"\x00\x20\x00\x30\x03\x01\x22\x00\x02\x11"
                ),
                (48, 32),
            ),
            (b"GIF89a......", None),
            (b"\x89PNG\r\n\x1a\nshort", None),
        ],
        ids=["png", "jpeg", "unknown-format", "truncated-png"],
    )
    def test_image_dimensions_parsing_matrix(self, image: bytes, expected: Any) -> None:
        """PNG and JPEG headers yield (width, height); unknown or truncated bytes yield None."""
        assert _google_client()._get_image_dimensions(image) == expected

    def test_image_token_estimation_tiles_and_null(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Token cost is ceil(w/75)*ceil(h/75)*258, zero for unusable bytes."""
        client = _google_client()

        assert client._estimate_image_tokens(None) == 0
        assert client._estimate_image_tokens(b"GIF89a") == 0
        assert client._estimate_image_tokens(_png_bytes(75, 75)) == 258
        assert client._estimate_image_tokens(_png_bytes(76, 75)) == 516

        monkeypatch.setattr(client, "_get_image_dimensions", MagicMock(return_value=(1920, 1080)))
        assert client._estimate_image_tokens(b"fake") == 100620


class TestMultimodalChartRequests:
    @pytest.mark.parametrize(
        (
            "factory",
            "model_config",
            "expected_model",
            "expected_prefix",
            "expected_extra_body",
            "usage_extra",
        ),
        [
            (
                _openrouter_client,
                {"max_tokens": 16, "openrouter_reasoning_effort": "low"},
                "openrouter/vision-model",
                {"role": "user", "content": "System instructions: follow rules"},
                {"reasoning": {"effort": "low"}},
                None,
            ),
            (
                _deepseek_client,
                {"max_tokens": 128},
                "deepseek-flash",
                {"role": "system", "content": "follow rules"},
                None,
                _DEEPSEEK_USAGE_EXTRA,
            ),
            (
                _lmstudio_client,
                {"max_tokens": 16, "openrouter_reasoning_effort": "max"},
                "local-vision",
                {"role": "user", "content": "System: follow rules"},
                None,
                None,
            ),
        ],
        ids=["openrouter", "deepseek", "lmstudio"],
    )
    async def test_chart_analysis_attaches_image_to_the_last_user_message(
        self,
        factory: Any,
        model_config: dict[str, Any],
        expected_model: str,
        expected_prefix: dict[str, str],
        expected_extra_body: dict[str, Any] | None,
        usage_extra: dict[str, Any] | None,
    ) -> None:
        """Each OpenAI-shaped client reshapes the system prompt its own way and inlines the chart."""
        client = factory()
        create = AsyncMock(return_value=_sdk_response("chart ok", usage_extra=usage_extra))
        client._client = _openai_sdk(create)

        response = await client.chat_completion_with_chart_analysis(
            model=expected_model,
            messages=[
                {"role": "system", "content": "follow rules"},
                {"role": "user", "content": "analyze this"},
            ],
            chart_image=b"fake-png",
            model_config=model_config,
        )

        assert response.choices[0].message.content == "chart ok"
        sent = create.await_args.kwargs
        assert sent["model"] == expected_model
        assert sent["messages"][0] == expected_prefix
        multimodal = sent["messages"][1]["content"]
        assert multimodal[0] == {"type": "text", "text": "analyze this"}
        assert multimodal[1]["type"] == "image_url"
        assert multimodal[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert sent.get("extra_body") == expected_extra_body
        assert "openrouter_reasoning_effort" not in sent


class TestOpenRouterRequestWiring:
    async def test_chat_completion_forwards_config_without_consuming_reasoning(self) -> None:
        """Reasoning travels in extra_body on every call and the shared config keeps its key."""
        client = _openrouter_client()
        create = AsyncMock(return_value=_sdk_response())
        client._client = _openai_sdk(create)
        shared_config = {"max_tokens": 128, "openrouter_reasoning_effort": "max"}

        first = await client.chat_completion(
            model="openrouter/model",
            messages=[{"role": "user", "content": "hello"}],
            model_config=shared_config,
        )
        second = await client.chat_completion(
            model="openrouter/model",
            messages=[{"role": "user", "content": "again"}],
            model_config=shared_config,
        )

        assert first.choices[0].message.content == "ok"
        assert second.usage.total_tokens == 3
        assert create.await_count == 2
        for call in create.await_args_list:
            assert call.kwargs["model"] == "openrouter/model"
            assert call.kwargs["max_tokens"] == 128
            assert call.kwargs["extra_body"] == {"reasoning": {"effort": "max"}}
            assert "reasoning" not in call.kwargs
            assert "openrouter_reasoning_effort" not in call.kwargs
        assert create.await_args_list[0].kwargs["messages"] == [
            {"role": "user", "content": "hello"}
        ]
        assert shared_config["openrouter_reasoning_effort"] == "max"

    async def test_chat_completion_filters_unsupported_params_and_keeps_penalties(self) -> None:
        """Canonical penalties pass through; the aliases and presence_penalty are filtered."""
        client = _openrouter_client()
        create = AsyncMock(return_value=_sdk_response())
        client._client = _openai_sdk(create)

        await client.chat_completion(
            model="openrouter/model",
            messages=[{"role": "user", "content": "hello"}],
            model_config={
                "temperature": 0.7,
                "top_k": 40,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.2,
                "freq_penalty": 0.3,
                "pres_penalty": 0.4,
                "thinking_budget": 100,
            },
        )

        sent = create.await_args.kwargs
        assert sent["temperature"] == 0.7
        assert sent["frequency_penalty"] == 0.1
        assert "presence_penalty" not in sent
        assert "top_k" not in sent
        assert "freq_penalty" not in sent
        assert "pres_penalty" not in sent
        assert "thinking_budget" not in sent
        assert "extra_body" not in sent

    async def test_generation_cost_maps_the_rest_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 200 from /generation is remapped to the internal token/cost keys."""
        client = _openrouter_client()
        fake = _patch_httpx(
            monkeypatch,
            _FakeHttpxResponse(
                200,
                {
                    "data": {
                        "model": "openrouter/model",
                        "total_cost": 0.001,
                        "tokens_prompt": 11,
                        "tokens_completion": 22,
                        "native_tokens_prompt": 33,
                        "native_tokens_completion": 44,
                    }
                },
            ),
        )

        cost = await client.get_generation_cost("gen-123", retry_delay=0)

        assert cost == {
            "model": "openrouter/model",
            "total_cost": 0.001,
            "prompt_tokens": 11,
            "completion_tokens": 22,
            "native_prompt_tokens": 33,
            "native_completion_tokens": 44,
        }
        request = fake.requests[0]
        assert request["url"] == "https://example.test/api/v1/generation"
        assert request["params"] == {"id": "gen-123"}
        assert request["headers"]["Authorization"] == "Bearer test-key"

        _patch_httpx(monkeypatch, _FakeHttpxResponse(200, {"data": None}))

        assert await client.get_generation_cost("gen-123", retry_delay=0) is None
        assert client.logger.warning.call_count == 0
        assert client.logger.debug.call_count == 0

    @pytest.mark.parametrize(
        ("status", "payload", "expect_debug"),
        [
            (404, {"error": {"message": "not found"}}, True),
            (500, {}, False),
            (429, {"error": {"message": "rate limited"}}, False),
            (401, {"error": {"message": "unauthorized"}}, False),
        ],
        ids=["not-indexed", "server-error", "rate-limited", "unauthorized"],
    )
    async def test_generation_cost_degrades_to_none_on_indexing_and_http_failures(
        self, monkeypatch: pytest.MonkeyPatch, status: int, payload: Any, expect_debug: bool
    ) -> None:
        """Only a 404 is a debug-level 'not indexed yet'; every other HTTP failure warns."""
        client = _openrouter_client()
        _patch_httpx(monkeypatch, _FakeHttpxResponse(status, payload))

        assert await client.get_generation_cost("gen-123", retry_delay=0) is None
        assert client.logger.warning.call_count == (0 if expect_debug else 1)
        assert client.logger.debug.call_count == (1 if expect_debug else 0)


class TestDeepSeekRequestWiring:
    async def test_chat_completion_forwards_reasoning_effort_and_logs_cache_counters(self) -> None:
        """DeepSeek forwards reasoning_effort and reports reasoning plus cache-hit counters."""
        client = _deepseek_client()
        create = AsyncMock(
            return_value=_sdk_response(
                "deepseek ok",
                (11, 22, 33),
                usage_extra=_DEEPSEEK_USAGE_EXTRA,
                response_id="ds-123",
                model="deepseek-flash",
            )
        )
        client._client = _openai_sdk(create)

        response = await client.chat_completion(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "hello"}],
            model_config={"max_tokens": 128, "reasoning_effort": "max"},
        )

        assert response.choices[0].message.content == "deepseek ok"
        assert response.usage.prompt_tokens == 11
        assert response.usage.completion_tokens == 22
        assert response.usage.total_tokens == 33
        sent = create.await_args.kwargs
        assert sent["model"] == "deepseek-flash"
        assert sent["reasoning_effort"] == "max"
        assert sent["max_tokens"] == 128
        assert sent["messages"] == [{"role": "user", "content": "hello"}]
        log_call = client.logger.info.call_args
        assert log_call[0][0] == (
            "DeepSeek token breakdown: prompt=%s, completion=%s (incl. reasoning=%s), "
            "total=%s, cache_hit=%s, cache_miss=%s"
        )
        assert log_call[0][1:] == (11, 22, 5, 33, 7, 4)

    async def test_chat_completion_drops_rejected_parameter_and_retries(self) -> None:
        """A rejected parameter is removed from the call and the retry carries the rest."""
        client = _deepseek_client()
        calls: list[dict[str, Any]] = []

        async def create(**kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            if len(calls) == 1:
                raise TypeError("got an unexpected keyword argument 'reasoning_effort'")
            return _sdk_response("retried ok", (11, 22, 33), usage_extra=_DEEPSEEK_USAGE_EXTRA)

        client._client = _openai_sdk(create)

        response = await client.chat_completion(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "hello"}],
            model_config={"max_tokens": 8, "reasoning_effort": "max"},
        )

        assert response.choices[0].message.content == "retried ok"
        assert calls[0] == {
            "model": "deepseek-flash",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
            "reasoning_effort": "max",
        }
        assert calls[1] == {
            "model": "deepseek-flash",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
        }

    async def test_chat_completion_gives_up_when_parameter_rejection_persists(self) -> None:
        """A provider that keeps rejecting the same parameter is not retried past one drop."""
        client = _deepseek_client()
        attempts: list[dict[str, Any]] = []

        async def create(**kwargs: Any) -> SimpleNamespace:
            attempts.append(kwargs)
            raise TypeError("got an unexpected keyword argument 'reasoning_effort'")

        client._client = _openai_sdk(create)

        response = await client.chat_completion(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "hello"}],
            model_config={"max_tokens": 8, "reasoning_effort": "max"},
        )

        assert response is None
        assert len(attempts) == 2
        assert "reasoning_effort" in attempts[0]
        assert "reasoning_effort" not in attempts[1]


class TestMalformedPayloadDegradation:
    async def test_malformed_sdk_payloads_have_pinned_degradation(self) -> None:
        """Missing choices, empty choices, None usage, None content and truncated JSON."""
        client = _deepseek_client()
        current: dict[str, Any] = {}

        async def create(**_kwargs: Any) -> Any:
            return current["payload"]

        client._client = _openai_sdk(create)

        async def invoke(payload: Any) -> ChatResponseModel | None:
            current["payload"] = payload
            return await client.chat_completion(
                model="deepseek-flash",
                messages=[{"role": "user", "content": "hi"}],
                model_config={},
            )

        no_choices = await invoke(SimpleNamespace(usage=None, id="x", model="m"))
        assert no_choices.error.startswith("Response creation error:")
        assert no_choices.choices == []

        empty_choices = await invoke(SimpleNamespace(choices=None, usage=None, id="x", model="m"))
        assert empty_choices.choices == []
        assert empty_choices.error is None
        assert empty_choices.usage is None

        none_usage = await invoke(_sdk_response("ok", None))
        assert none_usage.choices[0].message.content == "ok"
        assert none_usage.usage is None

        truncated = await invoke(
            _sdk_response(
                '{"signal": "BUY", "entry_price": 5', usage_extra=_DEEPSEEK_USAGE_EXTRA
            )
        )
        assert truncated.choices[0].message.content == '{"signal": "BUY", "entry_price": 5'
        assert truncated.error is None

        none_content = await invoke(_sdk_response(None, usage_extra=_DEEPSEEK_USAGE_EXTRA))
        assert none_content is None


class TestLMStudioRequestWiring:
    async def test_chat_completion_rewrites_system_messages_and_filters_foreign_params(self) -> None:
        """LM Studio turns system prompts into 'System: ...' user turns and drops foreign keys."""
        client = _lmstudio_client()
        create = AsyncMock(
            return_value=_sdk_response("lms ok", (3, 4, 7), response_id="lms-123", model="local-model")
        )
        client._client = _openai_sdk(create)

        response = await client.chat_completion(
            model="local-model",
            messages=[
                {"role": "system", "content": "follow rules"},
                {"role": "user", "content": "hello"},
            ],
            model_config={"max_tokens": 64, "openrouter_reasoning_effort": "max"},
        )

        assert response.choices[0].message.content == "lms ok"
        assert response.usage.prompt_tokens == 3
        assert response.usage.total_tokens == 7
        sent = create.await_args.kwargs
        assert sent["model"] == "local-model"
        assert sent["messages"] == [
            {"role": "user", "content": "System: follow rules"},
            {"role": "user", "content": "hello"},
        ]
        assert sent["max_tokens"] == 64
        assert "openrouter_reasoning_effort" not in sent

    async def test_model_auto_selection_caches_the_loaded_model_and_fails_loud_when_none(self) -> None:
        """An empty model name resolves once to the first loaded model and is cached."""
        client = _lmstudio_client()
        create = AsyncMock(return_value=_sdk_response("auto ok"))
        models_list = AsyncMock(return_value=SimpleNamespace(data=[SimpleNamespace(id="loaded/model")]))
        client._client = _openai_sdk(create, models_list)

        first = await client.chat_completion(
            model="", messages=[{"role": "user", "content": "a"}], model_config={}
        )
        second = await client.chat_completion(
            model="", messages=[{"role": "user", "content": "b"}], model_config={}
        )

        assert first.choices[0].message.content == "auto ok"
        assert second is not None
        assert create.await_args.kwargs["model"] == "loaded/model"
        client.logger.info.assert_any_call("Auto-selected loaded model: %s", "loaded/model")
        models_list.assert_awaited_once()

        idle = _lmstudio_client()
        idle._client = _openai_sdk(AsyncMock(), AsyncMock(return_value=SimpleNamespace(data=[])))
        assert (
            await idle.chat_completion(
                model="", messages=[{"role": "user", "content": "hi"}], model_config={}
            )
            is None
        )
        idle.logger.error.assert_any_call(
            "LM Studio Error: %s", "No model specified and no models loaded in LM Studio"
        )


class TestLMStudioStreaming:
    async def test_stream_accumulates_content_skips_empty_chunks_and_takes_usage(self) -> None:
        """Only text-bearing deltas reach the callback; the final usage chunk wins."""
        client = _lmstudio_client()
        chunks = [
            _stream_chunk("Hel"),
            _stream_chunk(),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None))], usage=None
            ),
            _stream_chunk("lo!"),
            _stream_chunk(
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=6, total_tokens=11)
            ),
        ]
        create = AsyncMock(return_value=_FakeStream(chunks))
        client._client = _openai_sdk(create)
        received: list[str] = []

        async def callback(text: str) -> None:
            received.append(text)

        response = await client.stream_chat_completion(
            model="local-model",
            messages=[{"role": "user", "content": "hi"}],
            model_config={"max_tokens": 64, "openrouter_reasoning_effort": "max"},
            callback=callback,
        )

        assert response.choices[0].message.content == "Hello!"
        assert received == ["Hel", "lo!"]
        sent = create.await_args.kwargs
        assert sent["stream"] is True
        assert sent["max_tokens"] == 64
        assert "openrouter_reasoning_effort" not in sent
        assert response.usage.prompt_tokens == 5
        assert response.usage.total_tokens == 11

    @pytest.mark.parametrize(
        ("chunks", "error", "expected_content"),
        [
            ([_stream_chunk("partial")], RuntimeError("stream broke"), "partial"),
            ([], RuntimeError("immediate broke"), None),
        ],
        ids=["failure-after-first-chunk", "failure-before-any-content"],
    )
    async def test_stream_failure_returns_partial_output_or_none(
        self, chunks: list[Any], error: Exception, expected_content: str | None
    ) -> None:
        """A broken stream keeps already emitted text but yields None when nothing arrived."""
        client = _lmstudio_client()
        client._client = _openai_sdk(AsyncMock(return_value=_BrokenStream(chunks, error)))

        response = await client.stream_chat_completion(
            model="local-model",
            messages=[{"role": "user", "content": "hi"}],
            model_config={},
        )

        if expected_content is None:
            assert response is None
        else:
            assert response.choices[0].message.content == expected_content

    @pytest.mark.parametrize(
        "marker",
        ["CUDA error: ErrorDeviceLost while running the model", "vk::Queue::submit failed"],
        ids=["device-lost", "vulkan-queue"],
    )
    async def test_gpu_crash_maps_to_friendly_gpu_error(self, marker: str) -> None:
        """Both GPU crash markers become a gpu_crash error instead of a bare None."""
        client = _lmstudio_client()

        async def create(**_kwargs: Any) -> Any:
            raise RuntimeError(marker)

        client._client = _openai_sdk(create)

        response = await client.chat_completion(
            model="local-model",
            messages=[{"role": "user", "content": "hi"}],
            model_config={},
        )

        assert response.choices == []
        assert response.error.startswith("gpu_crash:")
        assert "n_gpu_layers" in response.error


class TestBlockRunClient:
    def test_private_key_redaction_follows_the_wallet_key_length_rule(self) -> None:
        """A key longer than ten characters is truncated; shorter strings pass through."""
        client = _blockrun_client()

        assert client._redact_private_key("short") == "short"
        redacted = client._redact_private_key(f"Error with key {_WALLET_KEY} in request")
        assert redacted == "Error with key 0x0000...0001 in request"
        assert _WALLET_KEY not in redacted
        assert _blockrun_client("")._redact_private_key("Some error message") == "Some error message"

    def test_model_prefix_defaults_to_openai_and_keeps_custom_prefixes(self) -> None:
        """A bare model name gets the openai/ prefix; an existing provider is untouched."""
        client = _blockrun_client()

        assert client._ensure_provider_prefix("gpt-4o") == "openai/gpt-4o"
        assert client._ensure_provider_prefix("anthropic/claude-sonnet-4") == "anthropic/claude-sonnet-4"
        assert client._ensure_provider_prefix("deepseek/deepseek-reasoner") == "deepseek/deepseek-reasoner"

    def test_user_text_extraction_concatenates_every_user_message(self) -> None:
        """All user messages are joined; a system-only payload extracts nothing."""
        client = _blockrun_client()

        messages = [
            {"role": "system", "content": "You are a trading bot"},
            {"role": "user", "content": "Analyze BTC chart"},
            {"role": "user", "content": "What about ETH?"},
        ]

        assert client._extract_all_user_text_from_messages(messages) == (
            "Analyze BTC chart\n\nWhat about ETH?"
        )
        assert client._extract_all_user_text_from_messages(
            [{"role": "system", "content": "System prompt"}]
        ) == ""
        assert client._extract_all_user_text_from_messages([]) == ""

    def test_multimodal_preparation_prefixes_system_and_targets_last_user_message(self) -> None:
        """System turns become user prefixes; only the final user turn carries the image."""
        client = _blockrun_client()
        multimodal = [
            {"type": "text", "text": "Analyze"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
        ]

        two = client._prepare_multimodal_messages(
            [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User prompt"},
            ],
            multimodal,
        )
        assert two[0] == {"role": "user", "content": "System instructions: System prompt"}
        assert two[1] == {"role": "user", "content": multimodal}

        single = client._prepare_multimodal_messages(
            [{"role": "user", "content": "User prompt"}], multimodal
        )
        assert single[0]["content"] == multimodal

        pair = client._prepare_multimodal_messages(
            [
                {"role": "user", "content": "First prompt"},
                {"role": "user", "content": "Second prompt"},
            ],
            multimodal,
        )
        assert pair[0]["content"] == "First prompt"
        assert pair[1]["content"] == multimodal

    async def test_chat_completion_and_chart_analysis_convert_sdk_responses(self) -> None:
        """The SDK is called with prefixed models and filtered kwargs, and replies are converted."""
        client = _blockrun_client()
        sdk = AsyncMock()
        client._client = sdk

        sdk.chat_completion = AsyncMock(return_value=_blockrun_sdk_response("BUY signal detected"))

        result = await client.chat_completion(
            "gpt-4o",
            [{"role": "user", "content": "Analyze BTC"}],
            {"temperature": 0.5, "max_tokens": 1000},
        )

        assert result.choices[0].message.content == "BUY signal detected"
        assert result.model == "openai/gpt-4o"
        assert sdk.chat_completion.await_args.kwargs == {
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Analyze BTC"}],
            "temperature": 0.5,
            "max_tokens": 1000,
        }

        sdk.chat_completion = AsyncMock(return_value=_blockrun_sdk_response("again"))

        await client.chat_completion(
            "gpt-4o",
            [{"role": "user", "content": "Analyze BTC"}],
            {"temperature": None, "max_tokens": 1000, "reasoning_effort": "max"},
        )

        assert sdk.chat_completion.await_args.kwargs == {
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Analyze BTC"}],
            "max_tokens": 1000,
        }

        fake_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
            "/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        sdk.chat_completion = AsyncMock(
            return_value=_blockrun_sdk_response("Bullish engulfing pattern detected")
        )

        chart = await client.chat_completion_with_chart_analysis(
            "gpt-4o",
            [{"role": "user", "content": "Analyze this chart"}],
            io.BytesIO(fake_png),
            {"temperature": 0.3},
        )

        assert chart.choices[0].message.content == "Bullish engulfing pattern detected"
        chart_kwargs = sdk.chat_completion.await_args.kwargs
        assert chart_kwargs["model"] == "openai/gpt-4o"
        assert chart_kwargs["temperature"] == 0.3
        assert chart_kwargs["messages"][0]["content"] == [
            {"type": "text", "text": "Analyze this chart"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(fake_png).decode()}"
                },
            },
        ]

        sdk.chat_completion = AsyncMock(return_value=_blockrun_sdk_response(""))

        empty = await client.chat_completion(
            "gpt-4o", [{"role": "user", "content": "Analyze"}], {}
        )

        assert empty.choices == []
        assert empty.error == "BlockRun returned empty content"

        sdk.chat_completion = AsyncMock(return_value=None)

        assert (
            await client.chat_completion("gpt-4o", [{"role": "user", "content": "Analyze"}], {})
            is None
        )


class TestBlockRunOrchestration:
    async def test_blockrun_invoke_routes_text_chart_and_dispatch(self) -> None:
        """_invoke_blockrun picks the text or chart client method and invoke() routes to it."""
        text_provider = _RecordingProvider([ChatResponseModel.from_content("BUY BTC")])
        orchestrator = _orchestrator(blockrun=text_provider)

        text = await orchestrator._invoke_blockrun(
            metadata=orchestrator.get_metadata("blockrun"),
            messages=[{"role": "user", "content": "Analyze BTC"}],
            effective_model="deepseek/deepseek-reasoner",
            chart=False,
            chart_image=None,
        )

        assert text.success is True
        assert text.provider == "blockrun"
        assert text.model == "deepseek/deepseek-reasoner"
        assert text.response.choices[0].message.content == "BUY BTC"
        assert text_provider.calls == [("text", "deepseek/deepseek-reasoner")]

        chart_provider = _RecordingProvider([ChatResponseModel.from_content("Bullish engulfing")])
        chart_orchestrator = _orchestrator(blockrun=chart_provider)

        chart = await chart_orchestrator._invoke_blockrun(
            metadata=chart_orchestrator.get_metadata("blockrun"),
            messages=[{"role": "user", "content": "Analyze chart"}],
            effective_model="openai/gpt-4o",
            chart=True,
            chart_image=b"fake-png-bytes",
        )

        assert chart.success is True
        assert chart.provider == "blockrun"
        assert "Bullish engulfing" in chart.response.choices[0].message.content
        assert chart_provider.calls == [("chart", "openai/gpt-4o")]

        dispatch_provider = _RecordingProvider([ChatResponseModel.from_content("SELL BTC")])
        dispatched = await _orchestrator(blockrun=dispatch_provider).invoke(
            "blockrun", [{"role": "user", "content": "Test"}]
        )

        assert dispatched.success is True
        assert dispatched.provider == "blockrun"
        assert dispatched.model == "deepseek/deepseek-reasoner"
        assert dispatched.response.choices[0].message.content == "SELL BTC"

    async def test_blockrun_invoke_reports_failure_for_none_error_and_missing_client(self) -> None:
        """A None reply, an error reply and an absent client all fail with a distinct message."""
        absent = await _orchestrator().invoke("blockrun", [{"role": "user", "content": "Test"}])

        assert absent.success is False
        assert absent.provider == "blockrun"
        assert absent.model == "deepseek/deepseek-reasoner"
        assert absent.error == "Provider 'blockrun' is not available"

        none_provider = _RecordingProvider([None])
        none_orchestrator = _orchestrator(blockrun=none_provider)

        none_result = await none_orchestrator._invoke_blockrun(
            metadata=none_orchestrator.get_metadata("blockrun"),
            messages=[{"role": "user", "content": "Test"}],
            effective_model="deepseek/deepseek-reasoner",
            chart=False,
            chart_image=None,
        )

        assert none_result.success is False
        assert none_result.provider == "blockrun"
        assert none_result.error_message == "Empty or invalid response content from BlockRun"

        error_provider = _RecordingProvider([ChatResponseModel.from_error("Rate limit exceeded")])
        error_orchestrator = _orchestrator(blockrun=error_provider)

        error_result = await error_orchestrator._invoke_blockrun(
            metadata=error_orchestrator.get_metadata("blockrun"),
            messages=[{"role": "user", "content": "Test"}],
            effective_model="deepseek/deepseek-reasoner",
            chart=False,
            chart_image=None,
        )

        assert error_result.success is False
        assert error_result.response.error == "Rate limit exceeded"
        assert error_result.error == "Rate limit exceeded"

    def test_blockrun_availability_metadata_and_model_resolution(self) -> None:
        """Metadata, availability, chart support and model resolution for a wired provider."""
        client = _blockrun_client()
        orchestrator = _orchestrator(blockrun=client)
        metadata = orchestrator.get_metadata("blockrun")

        assert orchestrator.is_available("blockrun") is True
        assert orchestrator.supports_chart("blockrun") is True
        assert orchestrator.supports_chart("all") is True
        assert metadata.name == "BlockRun.AI"
        assert metadata.default_model == "deepseek/deepseek-reasoner"
        assert metadata.supports_chart is True
        assert orchestrator.resolve_model("blockrun") == "deepseek/deepseek-reasoner"
        assert orchestrator.resolve_model("blockrun", "openai/gpt-4o") == "openai/gpt-4o"

        bare = _orchestrator()

        assert bare.is_available("blockrun") is False
        assert bare.supports_chart("blockrun") is False
        assert bare.supports_chart("all") is False
        assert bare.resolve_model("blockrun") == "deepseek/deepseek-reasoner"
        assert bare.resolve_model("blockrun", "explicit/model") == "explicit/model"
        assert bare.resolve_model("nonexistent-provider") == "unknown-model"
        assert bare.get_metadata("nonexistent-provider") is None

    async def test_blockrun_participates_in_text_and_chart_fallback_chains(self) -> None:
        """'all' chains reach blockrun for text and charts, and skip it when unavailable."""
        client = _blockrun_client()
        client.chat_completion = AsyncMock(return_value=ChatResponseModel.from_content("BUY"))
        client.chat_completion_with_chart_analysis = AsyncMock(
            return_value=ChatResponseModel.from_content("Pattern detected")
        )
        orchestrator = _orchestrator(blockrun=client)

        text = await orchestrator.get_text_response("all", [{"role": "user", "content": "Test"}])

        assert text.success is True
        assert text.provider == "blockrun"

        chart = await orchestrator.get_chart_response(
            "all", [{"role": "user", "content": "Test"}], chart_image=b"fake-png"
        )

        assert chart.success is True
        assert chart.provider == "blockrun"
        assert client.chat_completion_with_chart_analysis.await_count == 1

        skipped = await _orchestrator().invoke_with_fallback(
            ["blockrun", "local", "openrouter"], [{"role": "user", "content": "Test"}]
        )

        assert skipped.success is False
        assert skipped.provider == "none"
        assert skipped.error == "No providers available"

        client.chat_completion = AsyncMock(return_value=ChatResponseModel.from_content("HOLD"))
        used = await orchestrator.invoke_with_fallback(
            ["blockrun"], [{"role": "user", "content": "Test"}]
        )

        assert used.success is True
        assert used.response.choices[0].message.content == "HOLD"

    def test_blockrun_and_deepseek_unavailable_guidance_and_failure_messages(self) -> None:
        """Missing-client guidance names the env key per provider and failures name the provider."""
        orchestrator = _orchestrator()

        orchestrator._log_unavailable_guidance("blockrun")
        assert orchestrator.logger.error.call_args[0][0] == (
            "BlockRun client not initialized. Check BLOCKRUN_WALLET_KEY in keys.env"
        )

        orchestrator._log_unavailable_guidance("deepseek")
        assert "DEEPSEEK_API_KEY" in orchestrator.logger.error.call_args[0][0]
        assert orchestrator.logger.error.call_count == 2

        orchestrator._log_failure("blockrun")
        assert orchestrator.logger.warning.call_args[0][0] == "BlockRun.AI failed or rate limited."


class TestProviderChainRouting:
    async def test_deepseek_orchestrator_routing_and_chart_support(self) -> None:
        """DeepSeek resolves to its default model for text and charts, and supports charts."""
        provider = _RecordingProvider(
            [ChatResponseModel.from_content("HOLD"), ChatResponseModel.from_content("HOLD")]
        )
        orchestrator = _orchestrator(deepseek=provider)

        text = await orchestrator.invoke("deepseek", [{"role": "user", "content": "hi"}])

        assert text.success
        assert text.provider == "deepseek"
        assert text.model == "deepseek-flash"
        assert provider.calls == [("text", "deepseek-flash")]

        chart = await orchestrator.invoke(
            "deepseek", [{"role": "user", "content": "hi"}], chart=True, chart_image=b"png"
        )

        assert chart.success
        assert chart.provider == "deepseek"
        assert chart.model == "deepseek-flash"
        assert provider.calls == [("text", "deepseek-flash"), ("chart", "deepseek-flash")]

        assert orchestrator.supports_chart("deepseek") is True
        assert _orchestrator().supports_chart("deepseek") is False

    async def test_all_chart_chain_includes_deepseek(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The 'all' chart chain lists googleai, deepseek, openrouter and blockrun in order."""
        orchestrator = _orchestrator(deepseek=_RecordingProvider([]))
        captured: dict[str, list[str]] = {}

        async def fake_fallback(providers: list[str], _messages: list[dict[str, str]], **_kwargs: Any) -> InvocationResult:
            captured["providers"] = providers
            return InvocationResult(
                success=False,
                response=ChatResponseModel.from_error("none"),
                provider="none",
                model="none",
            )

        monkeypatch.setattr(orchestrator, "invoke_with_fallback", fake_fallback)

        await orchestrator.get_chart_response("all", [{"role": "user", "content": "hi"}], b"png")

        assert captured["providers"] == ["googleai", "deepseek", "openrouter", "blockrun"]

    async def test_openrouter_retries_the_fallback_model_unless_model_is_overridden(self) -> None:
        """An invalid primary reply retries the configured fallback once, unless pinned."""
        config = _orchestrator_config(**_OPENROUTER_MODELS)
        fallback_provider = _RecordingProvider(
            [ChatResponseModel.from_error("rate_limit: busy"), ChatResponseModel.from_content("fallback ok")]
        )
        orchestrator = _orchestrator(config, openrouter=fallback_provider)

        result = await orchestrator.invoke("openrouter", [{"role": "user", "content": "hello"}])

        assert result.success
        assert result.model == "fallback/model"
        assert [model for _, model in fallback_provider.calls] == ["primary/model", "fallback/model"]

        override_provider = _RecordingProvider([ChatResponseModel.from_error("rate_limit: busy")])
        override_orchestrator = _orchestrator(config, openrouter=override_provider)

        overridden = await override_orchestrator.invoke(
            "openrouter", [{"role": "user", "content": "hello"}], model="override/model"
        )

        assert not overridden.success
        assert overridden.model == "override/model"
        assert override_provider.calls == [("text", "override/model")]


class TestLoaderWiring:
    def test_provider_registry_and_deepseek_model_config_wiring(self) -> None:
        """DeepSeek is a valid provider and its model maps to the reasoning-effort config."""
        assert "deepseek" in VALID_PROVIDERS

        config = Config()
        model_config = config.get_model_config(config.DEEPSEEK_MODEL)

        assert "reasoning_effort" in model_config
        assert "openrouter_reasoning_effort" not in model_config

    def test_provider_registry_omits_blockrun_despite_documentation(self) -> None:
        """Measured: config.ini documents blockrun, but the loader registry rejects it."""
        assert "blockrun" not in VALID_PROVIDERS

        unvalidated = Config.__new__(Config)
        unvalidated._config_data = {"ai_providers": {"provider": "blockrun"}}

        with pytest.raises(ValueError, match="Invalid AI provider 'blockrun'"):
            unvalidated._validate_provider()

    def test_blockrun_config_defaults_and_overrides_reach_the_orchestrator(self) -> None:
        """BlockRun's endpoint and model come from the loaded config, overrides included."""
        config = Config()

        assert config.BLOCKRUN_BASE_URL == "https://blockrun.ai/api"
        assert config.BLOCKRUN_MODEL == "deepseek/deepseek-reasoner"
        assert _orchestrator(config, blockrun=_blockrun_client()).get_metadata(
            "blockrun"
        ).default_model == config.BLOCKRUN_MODEL

        overridden = _orchestrator(
            _orchestrator_config(
                BLOCKRUN_MODEL="anthropic/claude-sonnet-4",
                BLOCKRUN_BASE_URL="https://custom.blockrun.ai/api",
            ),
            blockrun=_blockrun_client(),
        )

        assert overridden.config.BLOCKRUN_BASE_URL == "https://custom.blockrun.ai/api"
        assert overridden.get_metadata("blockrun").default_model == "anthropic/claude-sonnet-4"
        assert overridden.resolve_model("blockrun") == "anthropic/claude-sonnet-4"


class TestProviderClientsContainer:
    def test_every_provider_slot_defaults_to_none_and_accepts_an_injected_client(self) -> None:
        """The container holds one slot per provider and coexists with an injected blockrun."""
        client = _blockrun_client()
        clients = ProviderClients(
            google=None,
            google_paid=None,
            openrouter=None,
            lmstudio=None,
            blockrun=client,
        )

        assert clients.blockrun is client
        assert clients.google is None
        assert clients.google_paid is None
        assert clients.openrouter is None
        assert clients.lmstudio is None
        assert clients.deepseek is None
        assert ProviderClients().blockrun is None
