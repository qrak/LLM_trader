"""Provider fallback chains, API retry backoff, corrupt LLM output and model-manager messages.

Covers the retry/jitter mechanics in src.utils.decorators, the ProviderOrchestrator routing
of rate-limited or unparseable provider replies, the TradingAnalysisModel / UnifiedParser
rejection of corrupt payloads, and ModelManager message assembly for contract repair.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import aiohttp
import ccxt
import pytest
from pydantic import BaseModel, ValidationError

from src.managers.model_manager import ModelManager
from src.managers.provider_orchestrator import ProviderOrchestrator
from src.managers.provider_types import InvocationResult, ProviderClients
from src.parsing.unified_parser import UnifiedParser
from src.platforms.ai_providers.response_models import (
    ChatResponseModel,
    ChoiceModel,
    MessageModel,
    TradingAnalysisModel,
    TradingAnalysisResponseModel,
    TradingSignal,
    TrendDirection,
)
from src.utils.decorators import (
    _ApiRetryContext,
    _RetryContext,
    _should_retry_api_error,
    retry_api_call,
    retry_async,
)
from tests.conftest import make_config, null_logger

_BASE_MODEL = "google/gemini-3-flash-preview"
_FALLBACK_MODEL = "deepseek/deepseek-v4.1-flash"
_GOOGLE_MODEL = "gemini-3.8-flash"
_LOCAL_MODEL = "local-model"
_TREND_MESSAGE = "Input should be 'BULLISH', 'BEARISH' or 'NEUTRAL'"

_FULL_BUY: dict[str, Any] = {
    "signal": "BUY",
    "confidence": 80,
    "entry_price": 50000,
    "stop_loss": 49000,
    "take_profit": 52000,
    "position_size": 0.1,
    "risk_reward_ratio": 2.0,
    "reasoning": "test",
}

_TRUNCATED_JSON = '{"analysis": {"signal": "BUY", "confidence": 85, "entry_price": 50000'
_BINARY_NOISE = (
    "\x00\x01\x02\x00\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03\ufffd\ufffd\ufffdRAG\x00\x00"
)
_CONCATENATED_JSON = (
    '{"analysis": {"signal": "BUY", "confidence": 50}}\n'
    '{"analysis": {"signal": "SELL", "confidence": 80}}'
)
_INJECTED_HTML = (
    '{"analysis": {"signal": "HOLD", "confidence": 50, "reasoning": "'
    '<script>alert(\\"xss\\")</script> market looks normal"}}'
)
_OVERSIZED_REASONING = (
    '{"analysis": {"signal": "HOLD", "confidence": 50, "reasoning": "' + "A" * 100_000 + '"}}'
)
_STRING_ENTRY_PRICE = json.dumps(
    {
        "analysis": {
            "signal": "BUY",
            "confidence": 80,
            "entry_price": "high",
            "stop_loss": 48000,
            "take_profit": 52000,
            "position_size": 0.1,
            "risk_reward_ratio": 2.0,
            "reasoning": "price is going high",
        }
    }
)
_NON_FINITE = json.dumps(
    {
        "analysis": {
            "signal": "BUY",
            "confidence": float("inf"),
            "entry_price": float("nan"),
            "stop_loss": 48000,
            "take_profit": 52000,
            "position_size": float("-inf"),
            "risk_reward_ratio": 2.0,
            "reasoning": "moon or bust",
        }
    }
)


class _SdkError(BaseModel):
    """Pydantic error double shaped like an SDK error object."""

    code: int
    message: str = ""
    metadata: dict[str, Any] | None = None


class _RecordingProvider:
    """Provider double replaying queued replies and recording every (route, model) call."""

    def __init__(self, responses: list[ChatResponseModel]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    async def chat_completion(
        self, model: str, messages: Any, model_config: Any
    ) -> ChatResponseModel:
        self.calls.append(("text", model))
        return self.responses.pop(0)


def _rate_limited_response() -> ChatResponseModel:
    return ChatResponseModel(
        choices=[
            ChoiceModel(
                message=MessageModel(content=""),
                error={
                    "code": 429,
                    "message": "Rate limit exceeded. Try again later.",
                    "metadata": {
                        "provider_name": "openrouter",
                        "raw": {"retryable": True},
                    },
                },
            )
        ],
        error="rate_limit",
    )


def _orchestrator(**clients: Any) -> ProviderOrchestrator:
    """Orchestrator on the production config defaults with only the given clients wired."""
    config = make_config()
    config.get_model_config = lambda _model: {"max_tokens": 128}
    return ProviderOrchestrator(
        logger=null_logger(),
        config=config,
        clients=ProviderClients(
            google=clients.get("googleai"),
            openrouter=clients.get("openrouter"),
            lmstudio=clients.get("local"),
        ),
    )


def _api_retry_context(**kwargs: Any) -> _ApiRetryContext:
    return _ApiRetryContext(
        SimpleNamespace(logger=null_logger()),
        lambda *args, **rest: None,
        (),
        {},
        **kwargs,
    )


def _scripted_api_client(
    script: list[dict[str, Any]], **retry_kwargs: Any
) -> tuple[Any, list[str]]:
    """retry_api_call client replaying a scripted list, holding its last reply afterwards."""
    calls: list[str] = []

    class _Client:
        def __init__(self) -> None:
            self.logger = null_logger()
            self.model = "test-model"

        @retry_api_call(**retry_kwargs)
        async def fetch(self, model: str, messages: Any, config: Any) -> dict[str, Any]:
            calls.append(model)
            return script[min(len(calls), len(script)) - 1]

    return _Client(), calls


def _capture_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []

    async def tracking_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", tracking_sleep)
    return sleeps


def _model_manager() -> tuple[ModelManager, MagicMock]:
    manager = ModelManager.__new__(ModelManager)
    manager.logger = null_logger()
    counter = MagicMock()
    counter.count_tokens.return_value = 10
    manager.token_counter = counter
    return manager, counter


def _repair_manager(result: InvocationResult) -> tuple[ModelManager, MagicMock]:
    manager, _ = _model_manager()
    manager.provider = "deepseek"
    manager.cost_storage = MagicMock()
    manager.model_pricing = MagicMock()
    manager.unified_parser = MagicMock()
    manager.unified_parser.format_error_response = MagicMock(return_value="formatted error")
    orchestrator = MagicMock()
    orchestrator.get_text_response = AsyncMock(return_value=result)
    manager._orchestrator = orchestrator
    return manager, orchestrator


class TestRetryJitterAndBackoff:
    def test_jitter_keeps_every_delay_inside_a_quarter_of_its_base(self) -> None:
        """Each base delay keeps 75% to 125% of its value and the result is randomised."""
        ratios = [
            _RetryContext._add_jitter(base) / base
            for base in (0.01, 0.5, 1.0, 2.0, 5.0, 10.0)
            for _ in range(50)
        ]

        assert len(ratios) == 300
        assert min(ratios) >= 0.75
        assert max(ratios) <= 1.25
        assert len(set(ratios)) > 1

    def test_zero_delay_survives_jitter_unchanged(self) -> None:
        assert _RetryContext._add_jitter(0.0) == 0.0
        assert len({_RetryContext._add_jitter(1.0) for _ in range(100)}) > 1

    @pytest.mark.parametrize(
        ("initial_delay", "backoff_factor", "max_delay", "expected_bases"),
        [
            (0.01, 2, 60, [0.01, 0.02, 0.04]),
            (10, 2, 15, [10, 15, 15]),
        ],
        ids=["uncapped", "capped-at-max-delay"],
    )
    async def test_wait_grows_by_the_backoff_factor_and_stops_at_max_delay(
        self,
        monkeypatch: pytest.MonkeyPatch,
        initial_delay: float,
        backoff_factor: float,
        max_delay: float,
        expected_bases: list[float],
    ) -> None:
        """Three allowed retries produce three jittered waits over the capped bases."""
        sleeps = _capture_sleeps(monkeypatch)
        context = _api_retry_context(
            max_retries=3,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor,
            max_delay=max_delay,
        )

        for attempt in range(4):
            if not context._should_retry(attempt):
                break
            await context._wait_and_increment(attempt)

        bases = [
            min(initial_delay * backoff_factor**index, max_delay) for index in range(3)
        ]
        assert bases == expected_bases
        assert len(sleeps) == 3
        assert all(0.75 * base <= sleep <= 1.25 * base for sleep, base in zip(sleeps, bases))
        assert sleeps[0] < sleeps[-1] * 1.5
        assert (
            _api_retry_context(
                max_retries=3, initial_delay=1, backoff_factor=2, max_delay=60
            )._should_retry(3)
            is False
        )


class TestApiRetryBackoff:
    @pytest.mark.parametrize(
        ("script", "max_retries", "expected_calls", "expected_result"),
        [
            (
                [{"error": {"code": 502, "message": "Bad Gateway"}, "choices": []}],
                2,
                3,
                {"error": {"code": 502, "message": "Bad Gateway"}, "choices": []},
            ),
            (
                [
                    {"error": {"code": 502, "message": "Bad Gateway"}, "choices": []},
                    {"choices": [{"message": {"role": "assistant", "content": "recovered"}}]},
                ],
                3,
                2,
                {"choices": [{"message": {"role": "assistant", "content": "recovered"}}]},
            ),
            (
                [
                    {"error": {"code": 503, "message": "Service Unavailable"}, "choices": []},
                    {"error": {"code": 503, "message": "Service Unavailable"}, "choices": []},
                    {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                ],
                2,
                3,
                {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            ),
            (
                [
                    {
                        "choices": [
                            {
                                "error": {
                                    "code": 429,
                                    "message": "slow down",
                                    "metadata": {"raw": {"retryable": True}},
                                }
                            }
                        ]
                    },
                    {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
                ],
                2,
                2,
                {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            ),
        ],
        ids=[
            "exhausted",
            "recovers-after-one-retry",
            "recovers-after-two-retries",
            "choice-level-429",
        ],
    )
    async def test_retryable_payloads_retry_until_success_or_exhaustion(
        self,
        monkeypatch: pytest.MonkeyPatch,
        script: list[dict[str, Any]],
        max_retries: int,
        expected_calls: int,
        expected_result: dict[str, Any],
    ) -> None:
        """A retryable payload is re-sent up to max_retries and the last reply is returned."""
        sleeps = _capture_sleeps(monkeypatch)
        client, calls = _scripted_api_client(
            script, max_retries=max_retries, initial_delay=0.01, backoff_factor=2, max_delay=1
        )

        result = await client.fetch("test-model", [{"role": "user", "content": "hi"}], {})

        assert result == expected_result
        assert calls == ["test-model"] * expected_calls
        assert len(sleeps) == expected_calls - 1
        bases = [0.01 * 2**index for index in range(len(sleeps))]
        assert all(0.75 * base <= sleep <= 1.25 * base for sleep, base in zip(sleeps, bases))

    async def test_non_retryable_payload_is_returned_without_waiting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 400 payload is not retryable, so the first reply is returned immediately."""
        sleeps = _capture_sleeps(monkeypatch)
        payload = {"error": {"code": 400, "message": "bad request"}, "choices": []}
        client, calls = _scripted_api_client(
            [payload], max_retries=3, initial_delay=0.01, backoff_factor=2, max_delay=1
        )

        result = await client.fetch("test-model", [{"role": "user", "content": "hi"}], {})

        assert result == payload
        assert calls == ["test-model"]
        assert sleeps == []

    def test_error_classification_table(self) -> None:
        """Retryable codes, the retryable flag and the timeout string are the only retries."""
        table: list[tuple[Any, bool]] = [
            ({"code": 500}, True),
            ({"code": 502}, True),
            ({"code": 503}, True),
            ({"code": 504}, True),
            ({"code": 429, "message": "rate limit"}, False),
            ({"code": 429, "metadata": {"raw": {"retryable": True}}}, True),
            ({"code": 400, "metadata": {"raw": {"retryable": False}}}, False),
            ({"code": 400}, False),
            ("timeout", True),
            ("rate_limit", False),
            ("Service Unavailable", False),
            (None, False),
            ({}, False),
        ]

        measured = {repr(value): _should_retry_api_error(value) for value, _expected in table}

        assert measured == {repr(value): expected for value, expected in table}

    def test_response_shapes_that_count_as_retryable(self) -> None:
        """Dict and SDK replies are both inspected, at top level and inside choices."""
        context = _api_retry_context(
            max_retries=2, initial_delay=0.01, backoff_factor=2, max_delay=1
        )
        table: list[tuple[Any, bool]] = [
            ({"error": {"code": 503}}, True),
            ({"error": "timeout"}, True),
            ({"choices": [{"error": {"code": 503}}]}, True),
            (
                {"choices": [{"error": {"code": 429, "metadata": {"raw": {"retryable": True}}}}]},
                True,
            ),
            ({"choices": [{"message": {"content": "hi"}}]}, False),
            (None, False),
            (SimpleNamespace(error=_SdkError(code=503)), True),
            (SimpleNamespace(error=SimpleNamespace(code=503, message="x")), False),
            (SimpleNamespace(choices=[SimpleNamespace(error=_SdkError(code=503))]), True),
            (SimpleNamespace(choices=[SimpleNamespace(error=None)]), False),
        ]

        measured = [context._is_retryable_response(value) for value, _expected in table]

        assert measured == [expected for _value, expected in table]

    def test_sdk_error_with_explicit_none_metadata_is_not_retryable(self) -> None:
        """A dumped SDK error carrying metadata=None is classified instead of crashing the
        classifier, and an explicit retryable flag still wins (F16)."""
        context = _api_retry_context(
            max_retries=2, initial_delay=0.01, backoff_factor=2, max_delay=1
        )

        assert context._is_retryable_response(SimpleNamespace(error=_SdkError(code=400))) is False
        retryable = _SdkError(code=400, metadata={"raw": {"retryable": True}})
        assert context._is_retryable_response(SimpleNamespace(error=retryable)) is True


class TestAsyncRetryDecorator:
    @pytest.mark.parametrize(
        ("error_factory", "max_retries"),
        [
            (
                lambda: aiohttp.ClientConnectorError(
                    connection_key=MagicMock(), os_error=OSError("Connection refused")
                ),
                3,
            ),
            (lambda: ccxt.RateLimitExceeded("DDoS protection triggered"), 2),
            (lambda: asyncio.TimeoutError("slow"), 2),
        ],
        ids=["connector-error", "exchange-rate-limit", "timeout"],
    )
    async def test_retryable_errors_are_retried_until_success(
        self, monkeypatch: pytest.MonkeyPatch, error_factory: Any, max_retries: int
    ) -> None:
        """Network, rate-limit and timeout errors are retried; the third attempt succeeds."""
        sleeps = _capture_sleeps(monkeypatch)
        calls: list[str] = []

        class _Service:
            def __init__(self) -> None:
                self.logger = null_logger()

            @retry_async(
                max_retries=max_retries, initial_delay=0.01, backoff_factor=2, max_delay=1
            )
            async def fetch_pair(self, pair: str = "BTC/USDC") -> dict[str, Any]:
                calls.append(pair)
                if len(calls) < 3:
                    raise error_factory()
                return {"price": 50000}

        result = await _Service().fetch_pair(pair="BTC/USDC")

        assert result == {"price": 50000}
        assert calls == ["BTC/USDC"] * 3
        assert len(sleeps) == 2

    @pytest.mark.parametrize(
        ("error_type", "message"),
        [
            (ccxt.BadSymbol, "Symbol not found"),
            (ccxt.AuthenticationError, "bad key"),
            (ValueError, "plain failure"),
        ],
        ids=["bad-symbol", "authentication-error", "plain-value-error"],
    )
    async def test_non_retryable_errors_propagate_after_one_attempt(
        self, error_type: type[Exception], message: str
    ) -> None:
        """Exchange errors without a rate-limit phrase and plain exceptions are never retried."""
        calls: list[str] = []
        error = error_type(message)

        class _Service:
            def __init__(self) -> None:
                self.logger = null_logger()

            @retry_async(max_retries=3, initial_delay=0.01, backoff_factor=2, max_delay=1)
            async def fetch_ticker(self, pair: str = "INVALID/PAIR") -> dict[str, Any]:
                calls.append(pair)
                raise error

        with pytest.raises(error_type) as raised:
            await _Service().fetch_ticker(pair="INVALID/PAIR")

        assert raised.value is error
        assert calls == ["INVALID/PAIR"]


class TestOrchestratorCorruptPayloads:
    @pytest.mark.parametrize(
        "payload",
        [
            _TRUNCATED_JSON,
            _BINARY_NOISE,
            _CONCATENATED_JSON,
            _INJECTED_HTML,
            _OVERSIZED_REASONING,
        ],
        ids=[
            "truncated-json",
            "binary-noise",
            "concatenated-objects",
            "injected-html",
            "100k-reasoning",
        ],
    )
    async def test_non_empty_corrupt_content_is_a_successful_reply(self, payload: str) -> None:
        """The orchestrator never parses provider text: any non-empty content is accepted."""
        client = _RecordingProvider([ChatResponseModel.from_content(payload)])
        orchestrator = _orchestrator(openrouter=client)

        result = await orchestrator.invoke("openrouter", [{"role": "user", "content": "analyze"}])

        assert result.success is True
        assert result.provider == "openrouter"
        assert result.model == _BASE_MODEL
        assert result.response.choices[0].message.content == payload
        assert result.error is None
        assert client.calls == [("text", _BASE_MODEL)]

    async def test_empty_choices_spend_the_model_fallback_then_fail(self) -> None:
        """An empty choices list is invalid, costs one retry on the fallback model and fails."""
        empty = ChatResponseModel(choices=[])
        client = _RecordingProvider([empty, empty])
        orchestrator = _orchestrator(openrouter=client)

        result = await orchestrator.invoke("openrouter", [{"role": "user", "content": "analyze"}])

        assert result.success is False
        assert result.model == _FALLBACK_MODEL
        assert result.error_message == "Empty or invalid response content from OpenRouter"
        assert result.error == "Empty or invalid response content from OpenRouter"
        assert result.response.error is None
        assert client.calls == [("text", _BASE_MODEL), ("text", _FALLBACK_MODEL)]

    async def test_rate_limited_provider_hands_over_to_the_next_provider(self) -> None:
        """A 429 from OpenRouter costs one model fallback and then reaches Google."""
        limited = _RecordingProvider([_rate_limited_response()] * 2)
        google = _RecordingProvider([ChatResponseModel.from_content("google response ok")])
        orchestrator = _orchestrator(openrouter=limited, googleai=google)

        result = await orchestrator.invoke_with_fallback(
            ["openrouter", "googleai"], [{"role": "user", "content": "hello"}]
        )

        assert result.success is True
        assert result.provider == "google"
        assert result.model == _GOOGLE_MODEL
        assert result.response.choices[0].message.content == "google response ok"
        assert limited.calls == [("text", _BASE_MODEL), ("text", _FALLBACK_MODEL)]
        assert google.calls == [("text", _GOOGLE_MODEL)]

    async def test_all_providers_rate_limited_returns_the_last_failure(self) -> None:
        """When every provider is rate limited the chain ends on the last provider's failure."""
        openrouter = _RecordingProvider([_rate_limited_response()] * 2)
        google = _RecordingProvider([_rate_limited_response()])
        local = _RecordingProvider([_rate_limited_response()])
        orchestrator = _orchestrator(openrouter=openrouter, googleai=google, local=local)

        result = await orchestrator.invoke_with_fallback(
            ["openrouter", "googleai", "local"], [{"role": "user", "content": "hello"}]
        )

        assert result.success is False
        assert result.provider == "lmstudio"
        assert result.model == _LOCAL_MODEL
        assert result.error == "rate_limit"
        assert result.used_paid_tier is False
        assert openrouter.calls == [("text", _BASE_MODEL), ("text", _FALLBACK_MODEL)]
        assert google.calls == [("text", _GOOGLE_MODEL)]
        assert local.calls == [("text", _LOCAL_MODEL)]

    @pytest.mark.parametrize(
        ("replies", "expected_provider", "expected_content", "expected_counts"),
        [
            (
                {
                    "googleai": [_BINARY_NOISE],
                    "local": [_BINARY_NOISE],
                    "openrouter": [_BINARY_NOISE],
                },
                "google",
                _BINARY_NOISE,
                {"googleai": 1, "local": 0, "openrouter": 0},
            ),
            (
                {
                    "googleai": ["{broken json\x00"],
                    "openrouter": ['{"analysis": {"signal": "HOLD", "confidence": 50}}'],
                    "local": ["should not reach local"],
                },
                "google",
                "{broken json\x00",
                {"googleai": 1, "local": 0, "openrouter": 0},
            ),
        ],
        ids=["all-providers-corrupt", "first-corrupt-then-valid"],
    )
    async def test_corrupt_first_provider_stops_the_chain(
        self,
        replies: dict[str, list[str]],
        expected_provider: str,
        expected_content: str,
        expected_counts: dict[str, int],
    ) -> None:
        """Corrupt text looks like a good reply, so the first provider ends the chain."""
        clients = {
            name: _RecordingProvider(
                [ChatResponseModel.from_content(content) for content in contents]
            )
            for name, contents in replies.items()
        }
        orchestrator = _orchestrator(**clients)

        result = await orchestrator.invoke_with_fallback(
            ["googleai", "local", "openrouter"], [{"role": "user", "content": "analyze"}]
        )

        assert result.success is True
        assert result.provider == expected_provider
        assert result.response.choices[0].message.content == expected_content
        assert {name: len(client.calls) for name, client in clients.items()} == expected_counts


class TestTradingAnalysisContract:
    @pytest.mark.parametrize(
        ("payload", "expected_message"),
        [
            (
                {key: value for key, value in _FULL_BUY.items() if key != "stop_loss"},
                "Value error, BUY response is missing required execution fields: stop_loss",
            ),
            (
                {
                    "signal": "SELL",
                    "confidence": 70,
                    "entry_price": 50000,
                    "stop_loss": 51000,
                    "risk_reward_ratio": 1.5,
                    "reasoning": "going down",
                },
                (
                    "Value error, SELL response is missing required execution fields: "
                    "take_profit, position_size"
                ),
            ),
            (
                {
                    "signal": "UPDATE",
                    "confidence": 60,
                    "stop_loss": 49000,
                    "reasoning": "adjust SL",
                },
                "Value error, UPDATE response requires entry_price to represent the current price",
            ),
            (
                {
                    "signal": "UPDATE",
                    "confidence": 60,
                    "entry_price": 50000,
                    "reasoning": "no change",
                },
                "Value error, UPDATE response must include a new stop_loss or take_profit",
            ),
        ],
        ids=[
            "buy-missing-stop-loss",
            "sell-missing-take-profit-and-size",
            "update-missing-entry-price",
            "update-missing-levels",
        ],
    )
    def test_execution_signals_require_the_full_execution_payload(
        self, payload: dict[str, Any], expected_message: str
    ) -> None:
        with pytest.raises(ValidationError, match=expected_message):
            TradingAnalysisModel(**payload)

    @pytest.mark.parametrize(
        ("payload", "expected_message"),
        [
            (
                {"signal": "HOLD", "confidence": 101, "reasoning": "too confident"},
                "Input should be less than or equal to 100",
            ),
            (
                {"signal": "HOLD", "confidence": -1, "reasoning": "negative"},
                "Input should be greater than or equal to 0",
            ),
            (
                {**_FULL_BUY, "position_size": 1.5},
                "Input should be less than or equal to 1",
            ),
            (
                {**_FULL_BUY, "position_size": -0.5},
                "Input should be greater than or equal to 0",
            ),
            (
                {**_FULL_BUY, "entry_price": 0},
                "Input should be greater than 0",
            ),
            (
                {"signal": "MOON", "confidence": 80, "reasoning": "not a real signal"},
                "Input should be 'BUY', 'SELL', 'HOLD', 'CLOSE' or 'UPDATE'",
            ),
        ],
        ids=[
            "confidence-above-100",
            "confidence-negative",
            "size-above-one",
            "size-negative",
            "price-zero",
            "unknown-signal",
        ],
    )
    def test_out_of_range_or_unknown_values_are_rejected(
        self, payload: dict[str, Any], expected_message: str
    ) -> None:
        with pytest.raises(ValidationError, match=expected_message):
            TradingAnalysisModel(**payload)

    @pytest.mark.parametrize(
        ("signal", "reasoning"),
        [("HOLD", "wait and see"), ("CLOSE", "exit now")],
        ids=["hold", "close"],
    )
    def test_non_execution_signals_accept_a_minimal_payload(
        self, signal: str, reasoning: str
    ) -> None:
        """HOLD and CLOSE need no execution fields, so only signal and confidence apply."""
        model = TradingAnalysisModel(signal=signal, confidence=40, reasoning=reasoning)

        assert model.signal is TradingSignal(signal)
        assert model.confidence == 40
        assert model.reasoning == reasoning
        assert (
            model.entry_price,
            model.stop_loss,
            model.take_profit,
            model.position_size,
            model.risk_reward_ratio,
        ) == (None, None, None, None, None)
        assert model.reduce_only is False
        assert model.leverage == 1.0

    def test_boundary_values_are_accepted_and_coerced(self) -> None:
        """Confidence 0/100, size 0/1 and a whole-number price are inside the contract."""
        floor = TradingAnalysisModel(signal="HOLD", confidence=0)
        ceiling = TradingAnalysisModel(signal="HOLD", confidence=100, reasoning="top")
        full = TradingAnalysisModel(**{**_FULL_BUY, "position_size": 1})
        empty = TradingAnalysisModel(**{**_FULL_BUY, "position_size": 0})

        assert (floor.confidence, floor.reasoning) == (0, "")
        assert ceiling.confidence == 100
        assert full.position_size == 1.0
        assert empty.position_size == 0.0
        assert type(full.entry_price) is float
        assert full.entry_price == 50000.0
        assert full.signal is TradingSignal.BUY

    def test_nested_and_extra_fields_follow_the_allow_policy(self) -> None:
        """Nested trend and key_levels are validated, unknown keys kept, a bad trend rejected."""
        model = TradingAnalysisModel(
            signal="HOLD",
            confidence=50,
            reasoning="x",
            trend={"direction": "BULLISH", "strength_4h": 60},
            key_levels={"support": [100.0], "resistance": [200]},
            unknown_key=1,
        )

        assert model.trend.direction is TrendDirection.BULLISH
        assert model.trend.strength_4h == 60
        assert model.trend.strength_daily is None
        assert model.key_levels.support == [100.0]
        assert model.key_levels.resistance == [200.0]
        assert model.model_extra == {"unknown_key": 1}

        with pytest.raises(ValidationError, match=_TREND_MESSAGE):
            TradingAnalysisModel(
                signal="HOLD", confidence=50, reasoning="x", trend={"direction": "SIDEWAYS"}
            )

    @pytest.mark.parametrize(
        ("raw", "expected_message"),
        [
            (
                _STRING_ENTRY_PRICE,
                "Input should be a valid number, unable to parse string as a number",
            ),
            (_NON_FINITE, "Input should be a finite number"),
            (_TRUNCATED_JSON, "Invalid JSON: EOF while parsing an object at line 1 column 69"),
        ],
        ids=["string-price", "non-finite-number", "truncated-json"],
    )
    def test_corrupt_json_payloads_fail_response_validation(
        self, raw: str, expected_message: str
    ) -> None:
        with pytest.raises(ValidationError, match=expected_message):
            TradingAnalysisResponseModel.model_validate_json(raw)

    def test_truncated_payload_is_not_even_valid_json(self) -> None:
        """The truncated reply fails at the token level, before any schema check."""
        with pytest.raises(json.JSONDecodeError) as raised:
            json.loads(_TRUNCATED_JSON)

        assert (raised.value.lineno, raised.value.colno, raised.value.msg) == (
            1,
            70,
            "Expecting ',' delimiter",
        )


class TestUnifiedParserCorruptBlocks:
    @pytest.mark.parametrize(
        "text",
        [
            '```json\n{"analysis": {"signal": "HO',
            '```json\n{"analysis": {"signal": "HOLD"}}',
            _BINARY_NOISE,
            '{"analysis": {"signal": "HOLD", "confidence": 50}}',
            "",
        ],
        ids=["truncated-body", "unterminated-fence", "binary-noise", "unfenced-json", "empty"],
    )
    def test_payloads_without_a_parseable_fenced_block_yield_nothing(self, text: str) -> None:
        assert UnifiedParser.extract_json_block(text) is None
        assert UnifiedParser.extract_json_block(text, unwrap_key="analysis") is None

    def test_last_parseable_block_wins_and_unwrap_selects_the_analysis(self) -> None:
        """A repaired reply appends a good block after a broken one; unwrap drills into it."""
        repaired = '```json\n{broken\n```\n```json\n{"analysis": {"signal": "SELL"}}\n```'
        single = '```json\n{"analysis": {"signal": "HOLD", "confidence": 60}}\n```'

        assert UnifiedParser.extract_json_block(repaired) == {"analysis": {"signal": "SELL"}}
        assert UnifiedParser.extract_json_block(repaired, unwrap_key="analysis") == {
            "signal": "SELL"
        }
        assert UnifiedParser.extract_json_block(single) == {
            "analysis": {"signal": "HOLD", "confidence": 60}
        }
        assert UnifiedParser.extract_json_block(single, unwrap_key="analysis") == {
            "signal": "HOLD",
            "confidence": 60,
        }
        assert UnifiedParser._iter_json_blocks(repaired) == [
            "{broken",
            '{"analysis": {"signal": "SELL"}}',
        ]


class TestModelManagerMessages:
    @pytest.mark.parametrize(
        ("prompt", "system_message", "expected_messages", "expected_count_args"),
        [
            (
                "user query",
                "system instructions",
                [
                    {"role": "system", "content": "system instructions"},
                    {"role": "user", "content": "user query"},
                ],
                [call("system instructions"), call("user query")],
            ),
            (
                "user query only",
                None,
                [{"role": "user", "content": "user query only"}],
                [call("user query only")],
            ),
        ],
        ids=["with-system", "without-system"],
    )
    def test_prepare_messages_shapes_roles_and_counts_tokens(
        self,
        prompt: str,
        system_message: str | None,
        expected_messages: list[dict[str, str]],
        expected_count_args: list[Any],
    ) -> None:
        """Roles and contents are exact, each string is counted and session stats reset once."""
        manager, counter = _model_manager()

        messages = manager._prepare_messages(prompt, system_message=system_message)

        assert messages == expected_messages
        assert counter.count_tokens.call_args_list == expected_count_args
        counter.reset_session_stats.assert_called_once_with()

    def test_empty_system_message_is_treated_as_absent(self) -> None:
        """A falsy system message is dropped, so only the user turn is sent and counted."""
        manager, counter = _model_manager()

        messages = manager._prepare_messages("prompt only", system_message="")

        assert messages == [{"role": "user", "content": "prompt only"}]
        assert counter.count_tokens.call_args_list == [call("prompt only")]

    @pytest.mark.parametrize(
        ("provider", "model", "expected_provider", "expected_model"),
        [
            (None, None, "deepseek", None),
            ("googleai", "gemini-3.8-flash", "googleai", "gemini-3.8-flash"),
        ],
        ids=["defaults-to-manager-provider", "explicit-override"],
    )
    async def test_contract_repair_replays_the_turn_and_asks_for_the_block(
        self,
        provider: str | None,
        model: str | None,
        expected_provider: str,
        expected_model: str | None,
    ) -> None:
        """The repair call replays system/user/assistant plus one instruction for the block."""
        json_block = '```json\n{"analysis": {"signal": "HOLD", "confidence": 60}}\n```'
        manager, orchestrator = _repair_manager(
            InvocationResult(
                success=True,
                response=ChatResponseModel.from_content(json_block),
                provider="deepseek",
                model="deepseek-flash",
            )
        )

        text = await manager.send_contract_repair(
            system_message="system instructions",
            prompt="original user prompt",
            previous_response="narrative without the block",
            provider=provider,
            model=model,
        )

        assert text == json_block
        provider_arg, messages, model_arg = orchestrator.get_text_response.await_args.args
        assert (provider_arg, model_arg) == (expected_provider, expected_model)
        assert [message["role"] for message in messages] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert [message["content"] for message in messages[:3]] == [
            "system instructions",
            "original user prompt",
            "narrative without the block",
        ]
        assert messages[3]["content"] == (
            "Your previous reply omitted the required ```json block. Output ONLY the "
            "```json block for the decision described above — valid JSON, no other text."
        )

    async def test_contract_repair_formats_a_provider_error(self) -> None:
        """A failed repair request returns the parser's formatted error, not the raw reply."""
        manager, _orchestrator_double = _repair_manager(
            InvocationResult(
                success=False,
                response=ChatResponseModel.from_error("rate_limit"),
                provider="deepseek",
                model="deepseek-flash",
            )
        )

        text = await manager.send_contract_repair(
            system_message="s",
            prompt="p",
            previous_response="r",
            provider=None,
            model=None,
        )

        assert text == "formatted error"
        assert manager.unified_parser.format_error_response.call_args.args == ("rate_limit",)
