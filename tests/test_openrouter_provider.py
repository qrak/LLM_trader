"""Unit tests for OpenRouter provider compatibility and fallback wiring."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
    async def test_initialize_client_passes_server_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        def fake_openrouter(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace()

        monkeypatch.setattr(openrouter_module, "OpenRouter", fake_openrouter)
        client = _make_client()

        await client._initialize_client()

        assert calls == [{"api_key": "test-key", "server_url": "https://example.test/api/v1"}]

    def test_create_client_falls_back_when_server_url_keyword_is_unsupported(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls = []
        fallback_sdk = SimpleNamespace()

        def fake_openrouter(**kwargs):
            calls.append(kwargs)
            if "server_url" in kwargs:
                raise TypeError("__init__() got an unexpected keyword argument 'server_url'")
            return fallback_sdk

        monkeypatch.setattr(openrouter_module, "OpenRouter", fake_openrouter)
        client = _make_client()

        sdk = client._create_client()

        assert sdk is fallback_sdk
        assert calls == [
            {"api_key": "test-key", "server_url": "https://example.test/api/v1"},
            {"api_key": "test-key"},
        ]
        client.logger.warning.assert_called_once()


class TestRequestWiring:
    @pytest.mark.asyncio
    async def test_chat_completion_forwards_canonical_penalties_and_prefilters_top_k(self) -> None:
        client = _make_client()
        send_async = AsyncMock(return_value=_fake_sdk_response())
        client._client = SimpleNamespace(chat=SimpleNamespace(send_async=send_async))

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
        sent_kwargs = send_async.await_args.kwargs
        assert sent_kwargs["frequency_penalty"] == 0.1
        # presence_penalty was removed in OpenRouter SDK 0.11+; expect it filtered out
        assert "presence_penalty" not in sent_kwargs
        assert "top_k" not in sent_kwargs

    @pytest.mark.asyncio
    async def test_chart_analysis_uses_openrouter_multimodal_shape(self) -> None:
        client = _make_client()
        send_async = AsyncMock(return_value=_fake_sdk_response("chart ok"))
        client._client = SimpleNamespace(chat=SimpleNamespace(send_async=send_async))

        response = await client.chat_completion_with_chart_analysis(
            model="openrouter/vision-model",
            messages=[
                {"role": "system", "content": "follow rules"},
                {"role": "user", "content": "analyze this"},
            ],
            chart_image=b"fake-png",
            model_config={"max_tokens": 16},
        )

        assert response is not None
        assert response.choices[0].message.content == "chart ok"
        sent_messages = send_async.await_args.kwargs["messages"]
        assert sent_messages[0] == {"role": "user", "content": "System instructions: follow rules"}
        multimodal_content = sent_messages[1]["content"]
        assert multimodal_content[0] == {"type": "text", "text": "analyze this"}
        assert multimodal_content[1]["type"] == "image_url"
        assert multimodal_content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_generation_cost_extracts_generation_data_fields(self) -> None:
        client = _make_client()
        data = SimpleNamespace(
            model="openrouter/model",
            total_cost=0.001,
            tokens_prompt=11,
            tokens_completion=22,
            native_tokens_prompt=33,
            native_tokens_completion=44,
        )
        get_generation = MagicMock(return_value=SimpleNamespace(data=data))
        client._client = SimpleNamespace(generations=SimpleNamespace(get_generation=get_generation))

        cost = await client.get_generation_cost("gen-123", retry_delay=0)

        assert cost == {
            "model": "openrouter/model",
            "total_cost": 0.001,
            "prompt_tokens": 11,
            "completion_tokens": 22,
            "native_prompt_tokens": 33,
            "native_completion_tokens": 44,
        }


class TestClientCleanup:
    @pytest.mark.asyncio
    async def test_close_calls_async_and_sync_context_exits(self) -> None:
        client = _make_client()
        async_exit = AsyncMock()
        sync_exit = MagicMock()
        client._client = SimpleNamespace(__aexit__=async_exit, __exit__=sync_exit)

        await client.close()

        async_exit.assert_awaited_once_with(None, None, None)
        sync_exit.assert_called_once_with(None, None, None)
        assert client._client is None


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