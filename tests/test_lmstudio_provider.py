"""Unit tests for the LM Studio provider (OpenAI-compatible endpoint) wiring."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import src.platforms.ai_providers.lmstudio as lmstudio_module
from src.platforms.ai_providers.lmstudio import LMStudioClient


def _make_client() -> LMStudioClient:
    return LMStudioClient(base_url="http://localhost:1234/v1", logger=MagicMock())


def _fake_sdk_response(text: str = "ok") -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content=text)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7)
    return SimpleNamespace(choices=[choice], usage=usage, id="lms-123", model="local-model")


def _mock_sdk(create: AsyncMock, models_list: AsyncMock | None = None) -> SimpleNamespace:
    models = SimpleNamespace(list=models_list) if models_list else SimpleNamespace()
    return SimpleNamespace(
        models=models,
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )


class TestClientConstruction:
    @pytest.mark.asyncio
    async def test_initialize_client_uses_base_url_and_static_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        def fake_async_openai(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace()

        monkeypatch.setattr(lmstudio_module, "AsyncOpenAI", fake_async_openai)
        client = _make_client()

        await client._initialize_client()

        assert calls == [{"base_url": "http://localhost:1234/v1", "api_key": "lm-studio"}]
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
    async def test_chat_completion_rewrites_system_message_and_forwards_config(self) -> None:
        client = _make_client()
        create = AsyncMock(return_value=_fake_sdk_response("lms ok"))
        client._client = _mock_sdk(create)

        response = await client.chat_completion(
            model="local-model",
            messages=[
                {"role": "system", "content": "follow rules"},
                {"role": "user", "content": "hello"},
            ],
            model_config={"max_tokens": 64, "openrouter_reasoning_effort": "max"},
        )

        assert response is not None
        assert response.choices[0].message.content == "lms ok"
        assert response.usage.prompt_tokens == 3
        sent_kwargs = create.await_args.kwargs
        assert sent_kwargs["model"] == "local-model"
        assert sent_kwargs["messages"] == [
            {"role": "user", "content": "System: follow rules"},
            {"role": "user", "content": "hello"},
        ]
        assert sent_kwargs["max_tokens"] == 64
        # OpenRouter-specific key is never forwarded to LM Studio
        assert "openrouter_reasoning_effort" not in sent_kwargs

    @pytest.mark.asyncio
    async def test_chat_completion_auto_selects_first_loaded_model_when_empty(self) -> None:
        client = _make_client()
        create = AsyncMock(return_value=_fake_sdk_response("auto ok"))
        models_list = AsyncMock(return_value=SimpleNamespace(data=[SimpleNamespace(id="loaded/model")]))
        client._client = _mock_sdk(create, models_list)

        response = await client.chat_completion(
            model="",
            messages=[{"role": "user", "content": "hi"}],
            model_config={"max_tokens": 8},
        )

        assert response is not None
        assert response.choices[0].message.content == "auto ok"
        assert create.await_args.kwargs["model"] == "loaded/model"
        client.logger.info.assert_any_call("Auto-selected loaded model: %s", "loaded/model")

    @pytest.mark.asyncio
    async def test_auto_selected_model_is_cached_across_calls(self) -> None:
        client = _make_client()
        create = AsyncMock(return_value=_fake_sdk_response())
        models_list = AsyncMock(return_value=SimpleNamespace(data=[SimpleNamespace(id="loaded/model")]))
        client._client = _mock_sdk(create, models_list)

        await client.chat_completion(model="", messages=[{"role": "user", "content": "a"}], model_config={})
        await client.chat_completion(model="", messages=[{"role": "user", "content": "b"}], model_config={})

        models_list.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_chat_completion_returns_none_when_no_model_and_none_loaded(self) -> None:
        client = _make_client()
        models_list = AsyncMock(return_value=SimpleNamespace(data=[]))
        client._client = _mock_sdk(AsyncMock(), models_list)

        response = await client.chat_completion(
            model="",
            messages=[{"role": "user", "content": "hi"}],
            model_config={},
        )

        assert response is None
        client.logger.error.assert_any_call(
            "LM Studio Error: %s", "No model specified and no models loaded in LM Studio"
        )

    @pytest.mark.asyncio
    async def test_chart_analysis_attaches_image_to_last_user_message(self) -> None:
        client = _make_client()
        create = AsyncMock(return_value=_fake_sdk_response("chart ok"))
        client._client = _mock_sdk(create)

        response = await client.chat_completion_with_chart_analysis(
            model="local-vision",
            messages=[
                {"role": "system", "content": "follow rules"},
                {"role": "user", "content": "analyze this"},
            ],
            chart_image=b"fake-png",
            model_config={"max_tokens": 16},
        )

        assert response is not None
        assert response.choices[0].message.content == "chart ok"
        sent_messages = create.await_args.kwargs["messages"]
        assert sent_messages[0] == {"role": "user", "content": "System: follow rules"}
        multimodal_content = sent_messages[1]["content"]
        assert multimodal_content[0] == {"type": "text", "text": "analyze this"}
        assert multimodal_content[1]["type"] == "image_url"
        assert multimodal_content[1]["image_url"]["url"].startswith("data:image/png;base64,")


class _FakeStream:
    def __init__(self, chunks: list) -> None:
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk

        return gen()


class _BrokenStream:
    """Yields the given chunks, then raises mid-iteration."""

    def __init__(self, chunks: list, error: Exception) -> None:
        self._chunks = chunks
        self._error = error

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk
            raise self._error

        return gen()


def _chunk(text: str | None = None, usage: SimpleNamespace | None = None) -> SimpleNamespace:
    choices = [SimpleNamespace(delta=SimpleNamespace(content=text))] if text is not None else []
    return SimpleNamespace(choices=choices, usage=usage)


class TestStreaming:
    @pytest.mark.asyncio
    async def test_stream_accumulates_content_calls_callback_and_takes_usage(self) -> None:
        client = _make_client()
        chunks = [
            _chunk("Hel"),
            _chunk("lo!"),
            _chunk(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=6, total_tokens=11)),
        ]
        create = AsyncMock(return_value=_FakeStream(chunks))
        client._client = _mock_sdk(create)

        received: list[str] = []

        async def callback(text: str) -> None:
            received.append(text)

        response = await client.stream_chat_completion(
            model="local-model",
            messages=[{"role": "user", "content": "hi"}],
            model_config={"max_tokens": 64, "openrouter_reasoning_effort": "max"},
            callback=callback,
        )

        assert response is not None
        assert response.choices[0].message.content == "Hello!"
        assert received == ["Hel", "lo!"]
        sent_kwargs = create.await_args.kwargs
        assert sent_kwargs["stream"] is True
        assert sent_kwargs["max_tokens"] == 64
        assert "openrouter_reasoning_effort" not in sent_kwargs
        assert response.usage is not None
        assert response.usage.total_tokens == 11

    @pytest.mark.asyncio
    async def test_stream_returns_partial_content_when_failure_after_first_chunk(self) -> None:
        client = _make_client()
        create = AsyncMock(return_value=_BrokenStream([_chunk("partial")], RuntimeError("stream broke")))
        client._client = _mock_sdk(create)

        response = await client.stream_chat_completion(
            model="local-model",
            messages=[{"role": "user", "content": "hi"}],
            model_config={},
        )

        assert response is not None
        assert response.choices[0].message.content == "partial"

    @pytest.mark.asyncio
    async def test_stream_failure_before_any_content_returns_none(self) -> None:
        client = _make_client()
        create = AsyncMock(return_value=_BrokenStream([], RuntimeError("immediate broke")))
        client._client = _mock_sdk(create)

        response = await client.stream_chat_completion(
            model="local-model",
            messages=[{"role": "user", "content": "hi"}],
            model_config={},
        )

        assert response is None


class TestErrorMapping:
    @pytest.mark.asyncio
    async def test_gpu_crash_maps_to_friendly_error(self) -> None:
        client = _make_client()

        async def create(**_kwargs):
            raise RuntimeError("CUDA error: ErrorDeviceLost while running the model")

        client._client = _mock_sdk(create)

        response = await client.chat_completion(
            model="local-model",
            messages=[{"role": "user", "content": "hi"}],
            model_config={},
        )

        assert response is not None
        assert response.error is not None and response.error.startswith("gpu_crash:")
