"""Unit tests for the DeepSeek provider (official api.deepseek.com) wiring."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.platforms.ai_providers.deepseek as deepseek_module
from src.managers.provider_orchestrator import ProviderOrchestrator
from src.managers.provider_types import InvocationResult, ProviderClients
from src.platforms.ai_providers.deepseek import DeepSeekClient
from src.platforms.ai_providers.response_models import ChatResponseModel


def _make_client() -> DeepSeekClient:
    return DeepSeekClient(api_key="test-key", base_url="https://api.deepseek.test", logger=MagicMock())


def _fake_sdk_response(text: str = "ok") -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content=text)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    details = SimpleNamespace(reasoning_tokens=5)
    usage = SimpleNamespace(
        prompt_tokens=11,
        completion_tokens=22,
        total_tokens=33,
        completion_tokens_details=details,
        prompt_cache_hit_tokens=7,
        prompt_cache_miss_tokens=4,
    )
    return SimpleNamespace(choices=[choice], usage=usage, id="ds-123", model="deepseek-flash")


class TestClientConstruction:
    @pytest.mark.asyncio
    async def test_initialize_client_uses_api_key_and_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        def fake_async_openai(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace()

        monkeypatch.setattr(deepseek_module, "AsyncOpenAI", fake_async_openai)
        client = _make_client()

        await client._initialize_client()

        assert calls == [{"api_key": "test-key", "base_url": "https://api.deepseek.test"}]
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
    async def test_chat_completion_forwards_model_messages_and_reasoning_effort(self) -> None:
        client = _make_client()
        create = AsyncMock(return_value=_fake_sdk_response("deepseek ok"))
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        response = await client.chat_completion(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "hello"}],
            model_config={"max_tokens": 128, "reasoning_effort": "max"},
        )

        assert response is not None
        assert response.choices[0].message.content == "deepseek ok"
        assert response.usage.prompt_tokens == 11
        sent_kwargs = create.await_args.kwargs
        assert sent_kwargs["model"] == "deepseek-flash"
        assert sent_kwargs["reasoning_effort"] == "max"
        assert sent_kwargs["max_tokens"] == 128
        assert sent_kwargs["messages"] == [{"role": "user", "content": "hello"}]

    @pytest.mark.asyncio
    async def test_chart_analysis_builds_multimodal_block_on_last_user_message(self) -> None:
        client = _make_client()
        create = AsyncMock(return_value=_fake_sdk_response("chart ok"))
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        response = await client.chat_completion_with_chart_analysis(
            model="deepseek-flash",
            messages=[
                {"role": "system", "content": "follow rules"},
                {"role": "user", "content": "analyze this"},
            ],
            chart_image=b"fake-png",
            model_config={"max_tokens": 128},
        )

        assert response is not None
        assert response.choices[0].message.content == "chart ok"
        sent_messages = create.await_args.kwargs["messages"]
        assert sent_messages[0] == {"role": "system", "content": "follow rules"}
        multimodal_content = sent_messages[1]["content"]
        assert multimodal_content[0] == {"type": "text", "text": "analyze this"}
        assert multimodal_content[1]["type"] == "image_url"
        assert multimodal_content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_chat_completion_drops_rejected_parameter_and_retries(self) -> None:
        """A param-rejection error drops only that param and retries the call."""
        client = _make_client()
        calls: list[dict] = []

        async def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise TypeError("got an unexpected keyword argument 'reasoning_effort'")
            return _fake_sdk_response("retried ok")

        client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        response = await client.chat_completion(
            model="deepseek-flash",
            messages=[{"role": "user", "content": "hello"}],
            model_config={"max_tokens": 8, "reasoning_effort": "max"},
        )

        assert response is not None
        assert response.choices[0].message.content == "retried ok"
        assert calls[1] == {"model": "deepseek-flash", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 8}


class TestOrchestratorDeepSeekRouting:
    @pytest.mark.asyncio
    async def test_text_invoke_uses_default_model(self) -> None:
        client = _RecordingDeepSeekClient([ChatResponseModel.from_content("HOLD")])
        orchestrator = ProviderOrchestrator(
            logger=MagicMock(), config=_ConfigStub(), clients=ProviderClients(deepseek=client)
        )

        result = await orchestrator.invoke("deepseek", [{"role": "user", "content": "hi"}])

        assert result.success
        assert result.provider == "deepseek"
        assert result.model == "deepseek-flash"
        assert client.calls == [("text", "deepseek-flash")]

    @pytest.mark.asyncio
    async def test_chart_invoke_uses_default_model(self) -> None:
        client = _RecordingDeepSeekClient([ChatResponseModel.from_content("HOLD")])
        orchestrator = ProviderOrchestrator(
            logger=MagicMock(), config=_ConfigStub(), clients=ProviderClients(deepseek=client)
        )

        result = await orchestrator.invoke(
            "deepseek", [{"role": "user", "content": "hi"}], chart=True, chart_image=b"png"
        )

        assert result.success
        assert result.model == "deepseek-flash"
        assert client.calls == [("chart", "deepseek-flash")]

    def test_supports_chart_true_when_client_available(self) -> None:
        orchestrator = ProviderOrchestrator(
            logger=MagicMock(),
            config=_ConfigStub(),
            clients=ProviderClients(deepseek=_RecordingDeepSeekClient([])),
        )
        assert orchestrator.supports_chart("deepseek") is True

    def test_supports_chart_false_without_client(self) -> None:
        orchestrator = ProviderOrchestrator(
            logger=MagicMock(), config=_ConfigStub(), clients=ProviderClients()
        )
        assert orchestrator.supports_chart("deepseek") is False

    @pytest.mark.asyncio
    async def test_all_chart_chain_includes_deepseek(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orchestrator = ProviderOrchestrator(
            logger=MagicMock(),
            config=_ConfigStub(),
            clients=ProviderClients(deepseek=_RecordingDeepSeekClient([])),
        )
        captured: dict[str, list[str]] = {}

        async def fake_fallback(providers, messages, **kwargs):
            captured["providers"] = providers
            return InvocationResult(
                success=False,
                response=ChatResponseModel.from_error("none"),
                provider="none",
                model="none",
            )

        monkeypatch.setattr(orchestrator, "invoke_with_fallback", fake_fallback)

        await orchestrator.get_chart_response("all", [{"role": "user", "content": "hi"}], b"png")

        assert "deepseek" in captured["providers"]

    def test_unavailable_guidance_mentions_env_key(self) -> None:
        orchestrator = ProviderOrchestrator(
            logger=MagicMock(), config=_ConfigStub(), clients=ProviderClients()
        )

        orchestrator._log_unavailable_guidance("deepseek")

        call_args = orchestrator.logger.error.call_args[0]
        assert "DEEPSEEK_API_KEY" in call_args[0]


class TestLoaderWiring:
    def test_deepseek_is_a_valid_provider(self) -> None:
        from src.config.loader import VALID_PROVIDERS

        assert "deepseek" in VALID_PROVIDERS

    def test_model_config_uses_deepseek_reasoning_effort(self) -> None:
        from src.config.loader import Config

        cfg = Config()
        model_config = cfg.get_model_config(cfg.DEEPSEEK_MODEL)

        assert "reasoning_effort" in model_config
        assert "openrouter_reasoning_effort" not in model_config


class _ConfigStub:
    GOOGLE_STUDIO_MODEL = "gemini-3.5-flash"
    OPENROUTER_BASE_MODEL = "primary/model"
    OPENROUTER_FALLBACK_MODEL = "fallback/model"
    LM_STUDIO_MODEL = "local/model"
    BLOCKRUN_MODEL = "blockrun/model"
    BLOCKRUN_BASE_URL = "https://blockrun.ai/api"
    DEEPSEEK_MODEL = "deepseek-flash"

    def get_model_config(self, _model: str) -> dict[str, object]:
        return {"max_tokens": 128}


class _RecordingDeepSeekClient:
    def __init__(self, responses: list[ChatResponseModel]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    async def chat_completion(
        self, model: str, _messages: list[dict[str, str]], _model_config: dict[str, object]
    ) -> ChatResponseModel:
        self.calls.append(("text", model))
        return self.responses.pop(0)

    async def chat_completion_with_chart_analysis(
        self, model: str, _messages: list[dict[str, str]], _chart_image: object, _model_config: dict[str, object]
    ) -> ChatResponseModel:
        self.calls.append(("chart", model))
        return self.responses.pop(0)
