"""Unit tests for the OpenRouter provider (raw openai SDK / AsyncOpenAI) and fallback wiring."""
from types import SimpleNamespace
from typing import Self
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import src.platforms.ai_providers.openrouter as openrouter_module
from src.managers.provider_orchestrator import ProviderOrchestrator
from src.managers.provider_types import ProviderClients
from src.platforms.ai_providers.openrouter import OpenRouterClient
from src.platforms.ai_providers.response_models import ChatResponseModel


def _make_client() -> OpenRouterClient:
    return OpenRouterClient(api_key="test-key", base_url="https://example.test/api/v1", logger=MagicMock())


def _fake_sdk_response(text: str = "ok") -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content=text)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    return SimpleNamespace(choices=[choice], usage=usage, id="gen-123", model="test-model")


class TestClientConstruction:
    @pytest.mark.asyncio
    async def test_initialize_client_uses_api_key_and_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        def fake_async_openai(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace()

        monkeypatch.setattr(openrouter_module, "AsyncOpenAI", fake_async_openai)
        client = _make_client()

        await client._initialize_client()

        assert calls == [{"api_key": "test-key", "base_url": "https://example.test/api/v1"}]
        assert client._client is not None

    @pytest.mark.asyncio
    async def test_close_closes_sdk_client(self) -> None:
        client = _make_client()
        close_mock = AsyncMock()
        client._client = SimpleNamespace(close=close_mock)

        await client.close()

        close_mock.assert_awaited_once()
        assert client._client is None


class TestRequestWiring:
    @pytest.mark.asyncio
    async def test_chat_completion_forwards_config_and_wraps_reasoning_in_extra_body(self) -> None:
        client = _make_client()
        create = AsyncMock(return_value=_fake_sdk_response())
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        response = await client.chat_completion(
            model="openrouter/model",
            messages=[{"role": "user", "content": "hello"}],
            model_config={"max_tokens": 128, "openrouter_reasoning_effort": "max"},
        )

        assert response is not None
        assert response.choices[0].message.content == "ok"
        sent_kwargs = create.await_args.kwargs
        assert sent_kwargs["model"] == "openrouter/model"
        assert sent_kwargs["messages"] == [{"role": "user", "content": "hello"}]
        assert sent_kwargs["max_tokens"] == 128
        # reasoning is not a standard openai-SDK field — it must travel via extra_body
        assert sent_kwargs["extra_body"] == {"reasoning": {"effort": "max"}}
        assert "reasoning" not in sent_kwargs
        assert "openrouter_reasoning_effort" not in sent_kwargs

    @pytest.mark.asyncio
    async def test_chat_completion_forwards_reasoning_on_every_call_without_mutating_config(self) -> None:
        """A shared config dict must keep its reasoning effort across calls (regression: pop used to consume it)."""
        client = _make_client()
        create = AsyncMock(return_value=_fake_sdk_response())
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        shared_config = {"max_tokens": 64, "openrouter_reasoning_effort": "max"}

        await client.chat_completion(model="openrouter/model", messages=[{"role": "user", "content": "a"}], model_config=shared_config)
        await client.chat_completion(model="openrouter/model", messages=[{"role": "user", "content": "b"}], model_config=shared_config)

        assert create.await_count == 2
        for call in create.await_args_list:
            assert call.kwargs["extra_body"] == {"reasoning": {"effort": "max"}}
        # the caller's shared config is not consumed by the call
        assert shared_config["openrouter_reasoning_effort"] == "max"

    @pytest.mark.asyncio
    async def test_chat_completion_forwards_canonical_penalties_and_prefilters_top_k(self) -> None:
        client = _make_client()
        create = AsyncMock(return_value=_fake_sdk_response())
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        response = await client.chat_completion(
            model="openrouter/model",
            messages=[{"role": "user", "content": "hello"}],
            model_config={
                "temperature": 0.7,
                "top_k": 40,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.2,
            },
        )

        assert response is not None
        sent_kwargs = create.await_args.kwargs
        assert sent_kwargs["temperature"] == 0.7
        assert sent_kwargs["frequency_penalty"] == 0.1
        # presence_penalty is never forwarded to OpenRouter (kept filtered for parity
        # with the pre-consolidation dedicated-SDK client)
        assert "presence_penalty" not in sent_kwargs
        assert "top_k" not in sent_kwargs
        assert "extra_body" not in sent_kwargs

    @pytest.mark.asyncio
    async def test_chart_analysis_uses_multimodal_shape_and_extra_body(self) -> None:
        client = _make_client()
        create = AsyncMock(return_value=_fake_sdk_response("chart ok"))
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        response = await client.chat_completion_with_chart_analysis(
            model="openrouter/vision-model",
            messages=[
                {"role": "system", "content": "follow rules"},
                {"role": "user", "content": "analyze this"},
            ],
            chart_image=b"fake-png",
            model_config={"max_tokens": 16, "openrouter_reasoning_effort": "low"},
        )

        assert response is not None
        assert response.choices[0].message.content == "chart ok"
        sent_kwargs = create.await_args.kwargs
        sent_messages = sent_kwargs["messages"]
        assert sent_messages[0] == {"role": "user", "content": "System instructions: follow rules"}
        multimodal_content = sent_messages[1]["content"]
        assert multimodal_content[0] == {"type": "text", "text": "analyze this"}
        assert multimodal_content[1]["type"] == "image_url"
        assert multimodal_content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        assert sent_kwargs["extra_body"] == {"reasoning": {"effort": "low"}}


class _FakeHttpxResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        assert self._payload is not None
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.test/api/v1/generation")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=request, response=response)


class _FakeHttpxClient:
    def __init__(self, response: _FakeHttpxResponse) -> None:
        self._response = response
        self.requests: list[dict] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb) -> bool:
        return False

    async def get(self, url: str, params: dict | None = None, headers: dict | None = None) -> _FakeHttpxResponse:
        self.requests.append({"url": url, "params": params, "headers": headers})
        return self._response


def _patch_httpx(monkeypatch: pytest.MonkeyPatch, response: _FakeHttpxResponse) -> _FakeHttpxClient:
    fake_client = _FakeHttpxClient(response)
    monkeypatch.setattr(openrouter_module.httpx, "AsyncClient", lambda **kwargs: fake_client)
    return fake_client


class TestGenerationCost:
    @pytest.mark.asyncio
    async def test_generation_cost_maps_rest_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _make_client()
        fake = _patch_httpx(
            monkeypatch,
            _FakeHttpxResponse(200, {"data": {
                "model": "openrouter/model",
                "total_cost": 0.001,
                "tokens_prompt": 11,
                "tokens_completion": 22,
                "native_tokens_prompt": 33,
                "native_tokens_completion": 44,
            }}),
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

    @pytest.mark.asyncio
    async def test_generation_cost_returns_none_when_not_yet_indexed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _make_client()
        _patch_httpx(monkeypatch, _FakeHttpxResponse(404, {"error": {"message": "not found"}}))

        cost = await client.get_generation_cost("gen-123", retry_delay=0)

        assert cost is None
        client.logger.debug.assert_called_once()

    @pytest.mark.asyncio
    async def test_generation_cost_returns_none_on_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _make_client()
        _patch_httpx(monkeypatch, _FakeHttpxResponse(500, {}))

        cost = await client.get_generation_cost("gen-123", retry_delay=0)

        assert cost is None
        client.logger.warning.assert_called_once()


class TestOpenRouterOrchestratorFallback:
    @pytest.mark.asyncio
    async def test_openrouter_retries_configured_fallback_when_primary_response_is_invalid(self) -> None:
        config = _ConfigStub()
        openrouter_client = _FallbackClient([
            ChatResponseModel.from_error("rate_limit: busy"),
            ChatResponseModel.from_content("fallback ok"),
        ])
        orchestrator = ProviderOrchestrator(
            logger=MagicMock(),
            config=config,
            clients=ProviderClients(openrouter=openrouter_client),
        )

        result = await orchestrator.invoke("openrouter", [{"role": "user", "content": "hello"}])

        assert result.success
        assert result.model == "fallback/model"
        assert openrouter_client.calls == ["primary/model", "fallback/model"]

    @pytest.mark.asyncio
    async def test_openrouter_does_not_retry_fallback_when_model_override_is_explicit(self) -> None:
        config = _ConfigStub()
        openrouter_client = _FallbackClient([ChatResponseModel.from_error("rate_limit: busy")])
        orchestrator = ProviderOrchestrator(
            logger=MagicMock(),
            config=config,
            clients=ProviderClients(openrouter=openrouter_client),
        )

        result = await orchestrator.invoke(
            "openrouter",
            [{"role": "user", "content": "hello"}],
            model="override/model",
        )

        assert not result.success
        assert result.model == "override/model"
        assert openrouter_client.calls == ["override/model"]


class _ConfigStub:
    GOOGLE_STUDIO_MODEL = "gemini-3.5-flash"
    OPENROUTER_BASE_MODEL = "primary/model"
    OPENROUTER_FALLBACK_MODEL = "fallback/model"
    LM_STUDIO_MODEL = "local/model"
    BLOCKRUN_MODEL = "blockrun/model"
    BLOCKRUN_BASE_URL = "https://blockrun.ai/api"
    DEEPSEEK_MODEL = "deepseek-flash"

    def get_model_config(self, _model: str) -> dict[str, int]:
        return {"max_tokens": 16}


class _FallbackClient:
    def __init__(self, responses: list[ChatResponseModel]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    async def chat_completion(
        self,
        model: str,
        _messages: list[dict[str, str]],
        _model_config: dict[str, int],
    ) -> ChatResponseModel:
        self.calls.append(model)
        return self.responses.pop(0)

    async def chat_completion_with_chart_analysis(
        self,
        model: str,
        _messages: list[dict[str, str]],
        _chart_image: object,
        _model_config: dict[str, int],
    ) -> ChatResponseModel:
        self.calls.append(model)
        return self.responses.pop(0)
