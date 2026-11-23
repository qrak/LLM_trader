
import asyncio
import json
import struct
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from google.genai import errors

import src.platforms.ai_providers.anthropic_compatible as anthropic_module
import src.platforms.ai_providers.openai_compatible as openai_compatible_module
import src.platforms.ai_providers.openrouter as openrouter_module
from src.config.loader import VALID_PROVIDERS, Config
from src.managers.provider_orchestrator import ProviderOrchestrator
from src.managers.provider_types import ProviderClients
from src.platforms.ai_providers.anthropic_compatible import AnthropicCompatibleClient
from src.platforms.ai_providers.base import BaseAIClient
from src.platforms.ai_providers.deepseek import DeepSeekClient
from src.platforms.ai_providers.google import GoogleAIClient
from src.platforms.ai_providers.openai_compatible import OpenAICompatibleClient
from src.platforms.ai_providers.openrouter import OpenRouterClient
from src.platforms.ai_providers.response_models import ChatResponseModel
from src.utils import decorators as decorators_module
from tests.conftest import make_config

_OPENROUTER_MODELS = {"OPENROUTER_BASE_MODEL": "primary/model"}
_DEEPSEEK_USAGE_EXTRA = {
    "completion_tokens_details": SimpleNamespace(reasoning_tokens=5),
    "prompt_cache_hit_tokens": 7,
    "prompt_cache_miss_tokens": 4,
}
_GO_USAGE_EXTRA = {
    "cost": 0.0,
    "opencode": SimpleNamespace(billing_source="go", allowance=0.0041, list_cost=0.0101),
}


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(decorators_module, "_add_jitter", lambda _delay: 0.0)


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


def _opencode_client(session_id: str = "llm-trader-test") -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://opencode.ai/inference/go/openai/v1",
        logger=MagicMock(),
        default_headers={"User-Agent": "llm-trader/1.0", "x-opencode-session": session_id},
    )


def _commandcode_client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        api_key="test-key",
        base_url="https://api.commandcode.ai/provider/v1",
        logger=MagicMock(),
        default_headers={"User-Agent": "llm-trader/1.0"},
    )


def _custom_client(protocol: str = "openai") -> Any:
    if protocol == "anthropic":
        return AnthropicCompatibleClient(
            api_key="test-key", base_url="https://api.anthropic.test/v1", logger=MagicMock()
        )
    return OpenAICompatibleClient(
        api_key="test-key", base_url="http://localhost:1234/v1", logger=MagicMock()
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

    content = SimpleNamespace(parts=[SimpleNamespace(text=text)])
    candidate = SimpleNamespace(content=content)
    usage = SimpleNamespace(
        prompt_token_count=1, candidates_token_count=2, total_token_count=3
    )
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


def _openai_sdk(create: Any, models_list: Any | None = None) -> SimpleNamespace:

    models = SimpleNamespace(list=models_list) if models_list else SimpleNamespace()
    return SimpleNamespace(
        models=models, chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def _png_bytes(width: int, height: int) -> bytes:

    header = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR"
    return header + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00\xaa\xbb\xcc\xdd"


def _orchestrator_config(**overrides: Any) -> SimpleNamespace:

    config = make_config(**overrides)
    config.get_model_config = lambda _model: {"max_tokens": 16}
    return config


def _orchestrator(
    config: SimpleNamespace | None = None, **clients: Any
) -> ProviderOrchestrator:

    return ProviderOrchestrator(
        logger=MagicMock(),
        config=config if config else _orchestrator_config(),
        clients=ProviderClients(**clients),
    )


class _RecordingProvider:
    def __init__(self, responses: list[ChatResponseModel | None]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    async def chat_completion(
        self, model: str, _messages: list[dict[str, Any]], _model_config: dict[str, Any]
    ) -> ChatResponseModel | None:
        self.calls.append(("text", model))
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

    fake = _FakeHttpxClient(response)
    monkeypatch.setattr(openrouter_module.httpx, "AsyncClient", lambda **_kwargs: fake)
    return fake


class TestClientConstructionAndShutdown:
    @pytest.mark.parametrize(
        ("factory", "base_url"),
        [
            (_openrouter_client, "https://example.test/api/v1"),
            (_deepseek_client, "https://api.deepseek.test"),
            (_custom_client, "http://localhost:1234/v1"),
        ],
        ids=["openrouter", "deepseek", "custom"],
    )
    async def test_initialize_client_builds_sdk_with_provider_credentials_and_no_sdk_retries(
        self, factory: Any, base_url: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_async_openai(**kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace()

        monkeypatch.setattr(openai_compatible_module, "AsyncOpenAI", fake_async_openai)
        client = factory()

        await client._initialize_client()

        assert calls == [
            {"api_key": "test-key", "base_url": base_url, "default_headers": {}, "max_retries": 0}
        ]
        assert client._client is not None

    async def test_close_awaits_sdk_teardown_and_clears_the_handle(self) -> None:

        for factory in (_openrouter_client, _deepseek_client, _custom_client):
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
            _custom_client,
            _opencode_client,
            _commandcode_client,
        ],
        ids=["google", "openrouter", "deepseek", "custom", "opencode", "commandcode"],
    )
    def test_shared_base_helpers_are_inherited_by_every_client(self, factory: Any) -> None:

        client = factory()
        assert isinstance(client, BaseAIClient)

        client.api_key = "secret-api-key"
        assert client._sanitize_error_message("boom secret-api-key") == "boom [REDACTED_API_KEY]"

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
        ("model_config", "include_thinking", "expect_thinking"),
        [
            ({"max_tokens": 123, "thinking_level": "medium"}, True, True),
            ({"max_tokens": 123}, False, False),
            ({"max_tokens": 123, "thinking_level": "high"}, False, False),
            ({"max_tokens": 123, "thinking_level": "ultra"}, True, False),
            (
                {"max_tokens": 123, "temperature": 0.7, "top_p": 0.9, "top_k": 40},
                False,
                False,
            ),
        ],
        ids=[
            "thinking",
            "no-thinking",
            "thinking-off",
            "unknown-thinking-level",
            "sampling-ignored",
        ],
    )
    def test_generation_config_sends_tokens_and_thinking(
        self,
        model_config: dict[str, Any],
        include_thinking: bool,
        expect_thinking: bool,
    ) -> None:

        client = _google_client()

        config = client._create_generation_config(
            model_config,
            include_thinking=include_thinking,
        )
        dumped = config.model_dump(exclude_none=False)

        assert dumped["max_output_tokens"] == 123
        assert dumped["temperature"] is None
        assert dumped["top_p"] is None
        assert dumped["top_k"] is None
        assert (config.thinking_config is not None) is expect_thinking


class TestGoogleThinkingFallback:
    def test_unsupported_feature_detection_uses_sdk_error_details(self) -> None:

        client = _google_client()
        thinking_error = errors.APIError(
            400, {"error": {"message": "Invalid thinking_config field", "status": "INVALID_ARGUMENT"}}
        )
        code_error = errors.APIError(
            400, {"error": {"message": "unsupported code_execution tool", "status": "INVALID_ARGUMENT"}}
        )

        assert client._should_retry_without_thinking(thinking_error)
        assert not client._should_retry_without_thinking(code_error)
        assert not client._should_retry_without_thinking(
            errors.APIError(403, {"error": {"message": "Invalid thinking_config field"}})
        )
        assert not client._should_retry_without_thinking(ValueError("network hiccup"))

    async def test_chat_completion_retries_without_thinking_only_for_thinking_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:

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


class TestOpenRouterRequestWiring:
    async def test_chat_completion_forwards_config_without_consuming_reasoning(self) -> None:

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

        client = _openrouter_client()
        _patch_httpx(monkeypatch, _FakeHttpxResponse(status, payload))

        assert await client.get_generation_cost("gen-123", retry_delay=0) is None
        assert client.logger.warning.call_count == (0 if expect_debug else 1)
        assert client.logger.debug.call_count == (1 if expect_debug else 0)


class TestDeepSeekRequestWiring:
    @pytest.mark.parametrize("empty_payload", [{}, {"choices": []}, {"choices": None}])
    @pytest.mark.parametrize(
        "content",
        ['{"analysis": {"signal": "HOLD"}}', '{"verdict": "plan_followed", "lesson_learned": "Keep the plan"}'],
        ids=["analysis", "post-mortem"],
    )
    async def test_empty_http_completion_is_retried_unchanged(
        self, empty_payload: dict[str, Any], content: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleep = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep)
        requests: list[bytes] = []

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(request.content)
            payload = empty_payload if len(requests) == 1 else {
                "id": "recovered",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-flash",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            }
            return httpx.Response(200, json=payload)

        client = _deepseek_client()
        client._client = openai_compatible_module.AsyncOpenAI(
            api_key="test-key", base_url="https://api.deepseek.test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)), max_retries=0,
        )
        kwargs = {
            "model": "deepseek-flash", "messages": [{"role": "user", "content": "Return JSON"}],
            "model_config": {"response_format": {"type": "json_object"}},
        }
        try:
            response = await client.chat_completion(**kwargs)
        finally:
            await client.close()

        assert response.choices[0].message.content == content
        assert len(requests) == 2
        assert requests[0] == requests[1]
        sleep.assert_awaited_once()

    @pytest.mark.parametrize("code", [401, 503])
    async def test_http_error_envelope_keeps_its_cause_and_retry_classification(
        self, code: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleep = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep)
        requests: list[bytes] = []

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(request.content)
            return httpx.Response(200, json={"error": {"code": code, "message": "provider failure test-key"}})

        client = _deepseek_client()
        client._client = openai_compatible_module.AsyncOpenAI(
            api_key="test-key", base_url="https://api.deepseek.test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)), max_retries=0,
        )
        try:
            response = await client.chat_completion(
                model="deepseek-flash", messages=[{"role": "user", "content": "Return JSON"}], model_config={},
            )
        finally:
            await client.close()

        assert response.error is not None
        assert str(code) in response.error
        assert "[REDACTED_API_KEY]" in response.error
        assert "test-key" not in response.error
        assert len(requests) == (6 if code == 503 else 1)
        assert sleep.await_count == len(requests) - 1
        assert all(body == requests[0] for body in requests)

    async def test_chat_completion_forwards_reasoning_effort_and_logs_cache_counters(self) -> None:

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
            "total=%s, cache_hit=%s, cache_miss=%s, finish_reason=%s"
        )
        assert log_call[0][1:] == (11, 22, 5, 33, 7, 4, "stop")

    async def test_chat_completion_drops_rejected_parameter_and_retries(self) -> None:

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

    async def test_json_output_mode_travels_with_the_request(self) -> None:

        json_mode = {"max_tokens": 128, "reasoning_effort": "max", "response_format": {"type": "json_object"}}
        text_client = _deepseek_client()
        text_create = AsyncMock(return_value=_sdk_response("{}", (11, 22, 33)))
        text_client._client = _openai_sdk(text_create)

        await text_client.chat_completion(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "hello"}],
            model_config=json_mode,
        )

        assert text_create.await_args.kwargs["response_format"] == {"type": "json_object"}

    async def test_empty_reply_is_resent_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:

        async def no_sleep(_delay: float) -> None:
            return None

        monkeypatch.setattr(asyncio, "sleep", no_sleep)
        client = _deepseek_client()
        calls: list[dict[str, Any]] = []
        replies = [
            _sdk_response("", (11, 22, 33), usage_extra=_DEEPSEEK_USAGE_EXTRA),
            _sdk_response(
                '{"analysis": {"signal": "HOLD"}}', (11, 22, 33), usage_extra=_DEEPSEEK_USAGE_EXTRA
            ),
        ]

        async def create(**kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            return replies[min(len(calls), len(replies)) - 1]

        client._client = _openai_sdk(create)

        response = await client.chat_completion(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "hello"}],
            model_config={"max_tokens": 128, "response_format": {"type": "json_object"}},
        )

        assert len(calls) == 2
        assert calls[0] == calls[1]
        assert response.choices[0].message.content == '{"analysis": {"signal": "HOLD"}}'


class TestOpenAICompatibleWiring:
    @pytest.mark.parametrize(
        ("factory", "url", "headers"),
        [
            (
                _opencode_client,
                "https://opencode.ai/inference/go/openai/v1/chat/completions",
                {"user-agent": "llm-trader/1.0", "x-opencode-session": "llm-trader-test"},
            ),
            (
                _commandcode_client,
                "https://api.commandcode.ai/provider/v1/chat/completions",
                {"user-agent": "llm-trader/1.0"},
            ),
            (_custom_client, "http://localhost:1234/v1/chat/completions", {}),
        ],
        ids=["opencode", "commandcode", "custom"],
    )
    async def test_endpoint_headers_and_model_reach_the_wire(
        self, factory: Any, url: str, headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
    ) -> None:

        seen: list[httpx.Request] = []

        def handle(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={
                "id": "gen-1", "object": "chat.completion", "created": 1, "model": "vendor/model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            })

        real_async_openai = openai_compatible_module.AsyncOpenAI

        def build_client(**kwargs: Any) -> Any:
            return real_async_openai(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))

        monkeypatch.setattr(openai_compatible_module, "AsyncOpenAI", build_client)
        client = factory()
        try:
            response = await client.chat_completion("vendor/model", [{"role": "user", "content": "hi"}], {"max_tokens": 8})
        finally:
            await client.close()

        assert response.choices[0].message.content == "ok"
        assert len(seen) == 1
        assert str(seen[0].url) == url
        for name, value in headers.items():
            assert seen[0].headers[name] == value
        if "x-opencode-session" not in headers:
            assert "x-opencode-session" not in seen[0].headers
        assert json.loads(seen[0].content)["model"] == "vendor/model"

    async def test_reasoning_effort_is_forwarded_only_when_the_config_carries_it(self) -> None:

        client = _commandcode_client()
        create = AsyncMock(return_value=_sdk_response("ok"))
        client._client = _openai_sdk(create)

        await client.chat_completion("m", [{"role": "user", "content": "hello"}], {"max_tokens": 64, "reasoning_effort": "high"})
        await client.chat_completion("m", [{"role": "user", "content": "hello"}], {"max_tokens": 64})

        assert create.await_args_list[0].kwargs["reasoning_effort"] == "high"
        assert "reasoning_effort" not in create.await_args_list[1].kwargs

    async def test_custom_endpoint_never_receives_the_openrouter_knob(self) -> None:

        client = _custom_client()
        create = AsyncMock(return_value=_sdk_response("ok"))
        client._client = _openai_sdk(create)

        await client.chat_completion(
            "local-model",
            [{"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}],
            {"max_tokens": 64, "openrouter_reasoning_effort": "max"},
        )

        sent = create.await_args.kwargs
        assert "openrouter_reasoning_effort" not in sent
        assert sent["messages"][0] == {"role": "system", "content": "rules"}


class TestAnthropicCompatibleClient:
    @staticmethod
    def _wire(monkeypatch: pytest.MonkeyPatch, handle: Any) -> None:
        real = anthropic_module.httpx.AsyncClient
        monkeypatch.setattr(
            anthropic_module.httpx, "AsyncClient",
            lambda **kwargs: real(**kwargs, transport=httpx.MockTransport(handle)),
        )

    async def test_request_shape_and_response_mapping(self, monkeypatch: pytest.MonkeyPatch) -> None:

        seen: list[httpx.Request] = []

        def handle(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json={
                "id": "msg_1", "model": "claude-x", "type": "message", "role": "assistant",
                "content": [{"type": "thinking", "thinking": "hidden"}, {"type": "text", "text": "Hel"}, {"type": "text", "text": "lo"}],
                "stop_reason": "max_tokens",
                "usage": {"input_tokens": 11, "output_tokens": 22},
            })

        self._wire(monkeypatch, handle)
        client = _custom_client("anthropic")
        try:
            response = await client.chat_completion(
                "claude-x",
                [{"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}],
                {"max_tokens": 64, "temperature": 1.0, "reasoning_effort": "high", "response_format": {"type": "json_object"}},
            )
        finally:
            await client.close()

        body = json.loads(seen[0].content)
        assert str(seen[0].url) == "https://api.anthropic.test/v1/messages"
        assert seen[0].headers["x-api-key"] == "test-key"
        assert seen[0].headers["anthropic-version"] == "2023-06-01"
        assert body == {"model": "claude-x", "max_tokens": 64, "system": "rules",
                        "messages": [{"role": "user", "content": "hi"}]}
        assert response.choices[0].message.content == "Hello"
        assert response.choices[0].finish_reason == "length"
        assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens) == (11, 22, 33)
        assert response.id == "msg_1"

    @pytest.mark.parametrize(
        ("status", "prefix", "attempts"),
        [(429, "rate_limit:", 6), (529, "overloaded:", 6), (503, "overloaded:", 6), (401, "authentication:", 1)],
        ids=["rate-limited", "anthropic-overloaded", "unavailable", "bad-key"],
    )
    async def test_http_errors_map_to_retry_classes(
        self, status: int, prefix: str, attempts: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        seen: list[httpx.Request] = []

        def handle(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(status, json={"type": "error", "error": {"type": "x", "message": "overloaded" if status == 529 else "nope"}})

        self._wire(monkeypatch, handle)
        client = _custom_client("anthropic")
        try:
            response = await client.chat_completion("claude-x", [{"role": "user", "content": "hi"}], {"max_tokens": 8})
        finally:
            await client.close()

        assert response.error.startswith(prefix)
        assert len(seen) == attempts


class TestMalformedPayloadDegradation:
    async def test_malformed_sdk_payloads_have_pinned_degradation(self) -> None:

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


class TestProviderRouting:
    async def test_the_configured_provider_serves_the_request_once(self) -> None:

        provider = _RecordingProvider([ChatResponseModel.from_content("HOLD")])
        orchestrator = _orchestrator(deepseek=provider)

        text = await orchestrator.get_text_response("deepseek", [{"role": "user", "content": "hi"}])

        assert text.success
        assert text.provider == "deepseek"
        assert text.model == "deepseek-flash"
        assert provider.calls == [("text", "deepseek-flash")]

    async def test_a_failed_model_is_not_followed_by_any_other_model(self) -> None:

        provider = _RecordingProvider([ChatResponseModel.from_error("rate_limit: busy")])
        orchestrator = _orchestrator(_orchestrator_config(**_OPENROUTER_MODELS), openrouter=provider)

        result = await orchestrator.get_text_response("openrouter", [{"role": "user", "content": "hello"}])

        assert not result.success
        assert result.model == "primary/model"
        assert provider.calls == [("text", "primary/model")]

    async def test_a_model_override_is_sent_as_given(self) -> None:

        provider = _RecordingProvider([ChatResponseModel.from_content("ok")])
        orchestrator = _orchestrator(_orchestrator_config(**_OPENROUTER_MODELS), openrouter=provider)

        result = await orchestrator.get_text_response(
            "openrouter", [{"role": "user", "content": "hello"}], model="override/model"
        )

        assert result.success
        assert provider.calls == [("text", "override/model")]

    async def test_an_unconfigured_provider_names_its_missing_setting(self) -> None:

        orchestrator = _orchestrator()

        result = await orchestrator.get_text_response("custom", [{"role": "user", "content": "hi"}])

        assert not result.success
        assert result.error == "Provider 'custom' is not available"
        orchestrator.logger.error.assert_called_once_with(
            "%s client not initialized. Check %s", "Custom endpoint", "custom_base_url in config.ini"
        )


class TestLoaderWiring:
    def test_provider_registry_and_deepseek_model_config_wiring(self) -> None:

        assert VALID_PROVIDERS == {"googleai", "openrouter", "deepseek", "opencode", "commandcode", "custom"}

        config = Config()
        model_config = config.get_model_config(config.DEEPSEEK_MODEL)

        assert "reasoning_effort" in model_config
        assert "openrouter_reasoning_effort" not in model_config
        assert model_config["response_format"] == {"type": "json_object"}

    @pytest.mark.parametrize("provider", ["all", "local", "blockrun"])
    def test_retired_provider_names_are_rejected(self, provider: str) -> None:

        unvalidated = Config.__new__(Config)
        unvalidated._config_data = {"ai_providers": {"provider": provider}}

        with pytest.raises(ValueError, match=f"Invalid AI provider '{provider}'"):
            unvalidated._validate_provider()

    def test_custom_provider_needs_a_base_url_and_a_known_protocol(self) -> None:

        missing_url = Config.__new__(Config)
        missing_url._config_data = {"ai_providers": {"provider": "custom"}}
        with pytest.raises(RuntimeError, match="custom_base_url"):
            missing_url._validate_provider()

        bad_protocol = Config.__new__(Config)
        bad_protocol._config_data = {"ai_providers": {"provider": "custom", "custom_base_url": "http://x/v1", "custom_protocol": "grpc"}}
        with pytest.raises(ValueError, match="custom_protocol"):
            bad_protocol._validate_provider()


class TestProviderClientsContainer:
    def test_all_lists_only_the_injected_clients(self) -> None:

        client = _custom_client()

        assert ProviderClients(custom=client).all() == [client]
        assert ProviderClients().all() == []


class TestReportedCost:
    async def test_go_allowance_is_read_from_the_usage_payload(self) -> None:

        client = _opencode_client()
        create = AsyncMock(return_value=_sdk_response(usage_extra=_GO_USAGE_EXTRA))
        client._client = _openai_sdk(create)

        response = await client.chat_completion(
            model="deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "hi"}],
            model_config={"max_tokens": 32},
        )

        assert response is not None
        assert response.usage is not None
        assert response.usage.reported_cost == pytest.approx(0.0041)

    async def test_zero_balance_cost_is_not_taken_for_the_reported_amount(self) -> None:

        client = _opencode_client()
        create = AsyncMock(return_value=_sdk_response(usage_extra={"cost": 0.0}))
        client._client = _openai_sdk(create)

        response = await client.chat_completion(
            model="deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "hi"}],
            model_config={"max_tokens": 32},
        )

        assert response is not None
        assert response.usage is not None
        assert response.usage.reported_cost is None

    async def test_usage_without_a_reported_amount_leaves_the_cost_to_the_estimate(self) -> None:

        client = _opencode_client()
        create = AsyncMock(return_value=_sdk_response())
        client._client = _openai_sdk(create)

        response = await client.chat_completion(
            model="deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "hi"}],
            model_config={"max_tokens": 32},
        )

        assert response is not None
        assert response.usage is not None
        assert response.usage.reported_cost is None
