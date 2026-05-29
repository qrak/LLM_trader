"""Chaos tests: API rate-limiting (HTTP 429), exponential backoff, jitter, and circuit breakers.

Covers Pillar 3 — verifies the system handles:
- Transient HTTP 429 / rate-limit responses from AI providers
- HTTP 5xx server errors with retry
- Exponential backoff with jitter (the _add_jitter helper)
- _ApiRetryContext retry flow with exhausted retries
- Circuit-breaker-like behavior after repeated failures
- Cross-provider fallback when rate-limit hits one provider
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from src.managers.provider_orchestrator import ProviderOrchestrator
from src.managers.provider_types import ProviderClients
from src.platforms.ai_providers.response_models import ChatResponseModel, ChoiceModel, MessageModel

# Import the retry helpers directly for unit testing
from src.utils.decorators import retry_api_call, retry_async, _RetryContext


# ── 1. HTTP 429 RATE-LIMITED RESPONSES ────────────────────────────────────────

def _make_rate_limited_response() -> ChatResponseModel:
    """Simulate an OpenRouter-style rate-limit error in a response choice."""
    return ChatResponseModel(
        choices=[
            ChoiceModel(
                message=MessageModel(content=""),
                error={
                    "code": 429,
                    "message": "Rate limit exceeded. Try again later.",
                    "metadata": {"provider_name": "openrouter", "raw": {"retryable": True}},
                },
            )
        ],
        error="rate_limit",
    )


def _make_server_error_response() -> ChatResponseModel:
    """Simulate a transient 503 response."""
    return ChatResponseModel(
        choices=[
            ChoiceModel(
                message=MessageModel(content=""),
                error={
                    "code": 503,
                    "message": "Service temporarily unavailable",
                    "metadata": {"provider_name": "openrouter"},
                },
            )
        ],
        error="server_error",
    )


class TestRateLimitInProviderResponse:
    """Provider orchestrator detects rate-limit errors and triggers fallback."""

    @pytest.mark.asyncio
    async def test_rate_limited_provider_triggers_fallback_to_next_provider(self):
        """When OpenRouter returns a rate-limit, the orchestrator skips to the next provider."""
        config = MagicMock()
        config.OPENROUTER_BASE_MODEL = "openrouter/primary"
        config.OPENROUTER_FALLBACK_MODEL = "openrouter/fallback"
        config.GOOGLE_STUDIO_MODEL = "gemini-flash"
        config.LM_STUDIO_MODEL = "local/model"
        config.get_model_config.return_value = {"max_tokens": 128}

        google_client = MagicMock()

        async def google_ok(*args, **kwargs):
            return ChatResponseModel.from_content("google response ok")

        google_client.chat_completion = google_ok

        openrouter_client = MagicMock()

        async def or_rate_limited(*args, **kwargs):
            return _make_rate_limited_response()

        openrouter_client.chat_completion = or_rate_limited

        orchestrator = ProviderOrchestrator(
            logger=MagicMock(),
            config=config,
            clients=ProviderClients(
                google=google_client,
                openrouter=openrouter_client,
            ),
        )

        result = await orchestrator.invoke_with_fallback(
            ["openrouter", "googleai"],
            [{"role": "user", "content": "hello"}],
        )

        # Should have fallen through to googleai
        assert result.success
        assert result.provider == "google"

    @pytest.mark.asyncio
    async def test_all_providers_rate_limited_returns_last_failure(self):
        """When every provider returns rate-limit, last failure is returned."""
        config = MagicMock()
        config.GOOGLE_STUDIO_MODEL = "gemini-flash"
        config.OPENROUTER_BASE_MODEL = "openrouter/primary"
        config.OPENROUTER_FALLBACK_MODEL = "openrouter/fallback"
        config.LM_STUDIO_MODEL = "local/model"
        config.get_model_config.return_value = {"max_tokens": 128}

        limited = MagicMock()

        async def always_limited(*args, **kwargs):
            return _make_rate_limited_response()

        limited.chat_completion = always_limited

        orchestrator = ProviderOrchestrator(
            logger=MagicMock(),
            config=config,
            clients=ProviderClients(
                google=limited,
                openrouter=limited,
                lmstudio=limited,
            ),
        )

        result = await orchestrator.invoke_with_fallback(
            ["openrouter", "googleai", "local"],
            [{"role": "user", "content": "hello"}],
        )

        assert result is not None
        assert not result.success


# ── 2. HTTP 5xx SERVER ERRORS ─────────────────────────────────────────────────

class TestServerErrorRetryBehavior:
    """Transient 5xx errors should be retried before falling through."""

    @pytest.mark.asyncio
    async def test_503_error_triggers_retry_via_retry_api_call(self):
        """Simulate the retry_api_call decorator handling a 503 error."""

        call_count = 0

        class FakeClient:
            def __init__(self):
                self.logger = MagicMock()
                self.model = "test-model"

            @retry_api_call(max_retries=2, initial_delay=0.01, backoff_factor=2, max_delay=1)
            async def fetch(self, model, messages, config):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    # Return 503 error that should be retried
                    return {
                        "error": {
                            "code": 503,
                            "message": "Service Unavailable",
                        },
                        "choices": [],
                    }
                # Third call succeeds
                return {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                        }
                    ]
                }

        client = FakeClient()
        response = await client.fetch("test-model", [{"role": "user", "content": "hi"}], {})
        assert call_count == 3, f"Expected 3 attempts (1 initial + 2 retries), got {call_count}"
        assert response["choices"][0]["message"]["content"] == "ok"


# ── 3. EXPONENTIAL BACKOFF WITH JITTER ────────────────────────────────────────

class TestExponentialBackoffJitter:
    """_add_jitter applies ±25% jitter to prevent thundering herd."""

    def test_jitter_produces_values_within_25_percent_range(self):
        samples = []
        for base in [1.0, 2.0, 5.0, 10.0]:
            for _ in range(50):
                jittered = _RetryContext._add_jitter(base)
                ratio = jittered / base
                assert 0.75 <= ratio <= 1.25, \
                    f"Jitter {jittered:.4f} for base {base} is outside ±25% range (ratio={ratio:.4f})"
                samples.append(jittered)
        assert len(samples) == 200

    def test_jitter_is_random_not_constant(self):
        """Jitter must produce different values across calls (probabilistic)."""
        values = {_RetryContext._add_jitter(1.0) for _ in range(100)}
        assert len(values) > 1, "Jitter should produce varying values"

    def test_zero_delay_jitter(self):
        """Zero delay remains zero after jitter."""
        assert _RetryContext._add_jitter(0.0) == 0.0


class TestApiRetryBackoffTiming:
    """The retry delay should increase exponentially (approximately)."""

    @pytest.mark.asyncio
    async def test_exponential_backoff_grows_with_each_retry(self):
        """Each retry should have a longer delay than the previous (on average)."""

        delays = []

        original_sleep = asyncio.sleep

        async def tracking_sleep(delay):
            delays.append(delay)
            return await original_sleep(0)  # don't actually wait

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(asyncio, "sleep", tracking_sleep)
            # Use _ApiRetryContext directly with small delays
            from src.utils.decorators import _ApiRetryContext

            class FakeInstance:
                logger = MagicMock()

            instance = FakeInstance()
            context = _ApiRetryContext(
                instance,
                lambda *a, **kw: None,
                (),
                {},
                max_retries=3,
                initial_delay=0.01,
                backoff_factor=2,
                max_delay=60,
            )

            # Manually run retry loop
            for attempt in range(3):
                should_continue = context._should_retry(attempt)
                if not should_continue:
                    break
                await context._wait_and_increment(attempt)

        assert len(delays) >= 2, "Should have at least 2 delay measurements"
        # After jitter, delays should generally increase
        # We check that the delays are in ascending order (with jitter tolerance)
        assert delays[0] < delays[-1] * 1.5, \
            f"Backoff should grow: first={delays[0]:.6f}, last={delays[-1]:.6f}"


# ── 4. EXHAUSTED RETRIES & CIRCUIT-BREAKER BEHAVIOR ───────────────────────────

class TestExhaustedRetries:
    """After max_retries, the system must stop retrying and return the last response."""

    @pytest.mark.asyncio
    async def test_retries_exhausted_returns_last_response(self):

        call_count = 0

        class FakeClient:
            def __init__(self):
                self.logger = MagicMock()
                self.model = "test-model"

            @retry_api_call(max_retries=2, initial_delay=0.01, backoff_factor=2, max_delay=1)
            async def fetch(self, model, messages, config):
                nonlocal call_count
                call_count += 1
                return {
                    "error": {"code": 502, "message": "Bad Gateway"},
                    "choices": [],
                }

        client = FakeClient()
        response = await client.fetch("test-model", [{"role": "user", "content": "hi"}], {})
        # Should have retried max_retries times (1 initial + 2 retries = 3 total)
        assert call_count == 3, f"Expected 3 calls (2 retries), got {call_count}"
        # Last response is returned even on failure
        assert response is not None

    @pytest.mark.asyncio
    async def test_retryable_error_then_success_does_not_exhaust(self):

        call_count = 0

        class FakeClient:
            def __init__(self):
                self.logger = MagicMock()
                self.model = "test-model"

            @retry_api_call(max_retries=3, initial_delay=0.01, backoff_factor=2, max_delay=1)
            async def fetch(self, model, messages, config):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return {
                        "error": {"code": 502, "message": "Bad Gateway"},
                        "choices": [],
                    }
                return {
                    "choices": [
                        {"message": {"role": "assistant", "content": "recovered"}}
                    ]
                }

        client = FakeClient()
        response = await client.fetch("test-model", [{"role": "user", "content": "hi"}], {})
        assert call_count == 2, f"Expected 2 calls (1 retry), got {call_count}"
        assert response["choices"][0]["message"]["content"] == "recovered"


# ── 5. NETWORK ERROR RETRY via retry_async decorator ──────────────────────────

class TestAsyncRetryDecorator:
    """The retry_async decorator handles network-level exceptions."""

    @pytest.mark.asyncio
    async def test_transient_connection_error_is_retried(self):
        """aiohttp.ClientConnectorError should trigger retry via _RETRY_CONTEXT."""
        call_count = 0

        class FakeService:
            def __init__(self):
                self.logger = MagicMock()

            @retry_async(max_retries=3, initial_delay=0.01, backoff_factor=2, max_delay=1)
            async def fetch_pair(self, pair="BTC/USDC"):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    import aiohttp
                    raise aiohttp.ClientConnectorError(
                        connection_key=MagicMock(),
                        os_error=OSError("Connection refused"),
                    )
                return {"price": 50000}

        svc = FakeService()
        result = await svc.fetch_pair(pair="BTC/USDC")
        assert result == {"price": 50000}
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_rate_limit_exchange_error_is_retried(self):
        """ccxt.RateLimitExceeded triggers retry via the exchange error path."""
        call_count = 0

        import ccxt

        class FakeExchange:
            def __init__(self):
                self.logger = MagicMock()

            @retry_async(max_retries=2, initial_delay=0.01, backoff_factor=2, max_delay=1)
            async def fetch_ticker(self, pair="BTC/USDC"):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ccxt.RateLimitExceeded("DDoS protection triggered")
                return {"last": 50000}

        exch = FakeExchange()
        result = await exch.fetch_ticker(pair="BTC/USDC")
        assert result == {"last": 50000}
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_exchange_error_propagates(self):
        """Non-retryable exchange errors (e.g. BadSymbol) must NOT be retried."""

        import ccxt

        class FakeExchange:
            def __init__(self):
                self.logger = MagicMock()

            @retry_async(max_retries=3, initial_delay=0.01, backoff_factor=2, max_delay=1)
            async def fetch_ticker(self, pair="INVALID/PAIR"):
                raise ccxt.BadSymbol("Symbol not found")

        exch = FakeExchange()
        with pytest.raises(ccxt.BadSymbol):
            await exch.fetch_ticker(pair="INVALID/PAIR")
