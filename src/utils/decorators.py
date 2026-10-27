"""Decorators module.

Provides functionality for utils.decorators.py.
"""
import asyncio
import functools
import json
import random
import socket
import traceback
from typing import Any

import aiodns
import aiohttp
import ccxt

_RATE_LIMIT_PHRASES = {
    "too many requests", "rate limit", "429", "ratelimit",
    "ddos protection", "system-level rate limit exceeded"
}

_NETWORK_EXCEPTIONS = (
    ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection, ccxt.RateLimitExceeded,
    TimeoutError, ConnectionResetError, aiohttp.ClientConnectorError, aiohttp.ClientOSError,
    asyncio.TimeoutError, socket.gaierror, OSError, aiodns.error.DNSError
)

_TRUNCATED_FINISH_REASONS = {"length", "max_tokens"}
_FILTERED_FINISH_REASONS = {"content_filter", "safety"}
_META_PAYLOAD_KEYS = {"type", "response_format", "format"}


def _add_jitter(delay: float) -> float:
    """Add ±25% random jitter to prevent thundering herd on concurrent retries."""
    return delay * (0.75 + random.random() * 0.5)


class _RetryContext:
    """Helper class to manage retry logic and reduce complexity."""

    def __init__(self, instance, func, args, kwargs, max_retries, initial_delay, backoff_factor, max_delay):
        self.logger = instance.logger
        self.pair = kwargs.get("pair") or (args[0] if args else None)
        self.class_name = instance.__class__.__name__
        self.func_name = func.__name__
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.attempt = 0
        self.delay = initial_delay

    def _format_prefix(self) -> str:
        """Generate consistent log message prefix."""
        return f"{self.pair + ' - ' if self.pair else ''}"

    def _should_continue_retrying(self) -> bool:
        """Check if we should continue retrying."""
        self.attempt += 1
        return self.max_retries == -1 or self.attempt <= self.max_retries

    def _log_failure(self, error_type: str, error: Exception):
        """Log final failure after exhausting retries."""
        prefix = self._format_prefix()
        self.logger.error(
            f"{prefix}Function {self.class_name}.{self.func_name} failed after {self.max_retries} retries. "
            f"Last error: {error_type} - {error}")

    async def _handle_retryable_error(self, template: str, error: Exception, error_type: str | None = None) -> bool:
        """Common logic for handling retryable errors."""
        if not self._should_continue_retrying():
            self._log_failure(error_type or type(error).__name__, error)
            return False

        prefix = self._format_prefix()
        self.logger.warning(
            f"{prefix}{template.format(self.attempt)} for {self.class_name}.{self.func_name} "
            f"in {self.delay:.2f} seconds. Type: {type(error).__name__}, Error: {error}")

        await asyncio.sleep(_add_jitter(self.delay))
        self.delay = min(self.delay * self.backoff_factor, self.max_delay)
        return True

    @staticmethod
    def _classify_retryable_error(e: Exception) -> str:
        msg = str(e).lower()
        if isinstance(e, (ccxt.RateLimitExceeded, ccxt.DDoSProtection)) or any(p in msg for p in _RATE_LIMIT_PHRASES):
            return "Rate limit/DDoS. Retry {}"
        if isinstance(e, (ccxt.RequestTimeout, TimeoutError, asyncio.TimeoutError)) or "timeout" in msg:
            return "Timeout. Retry {}"
        if isinstance(e, (ccxt.NetworkError, aiohttp.ClientConnectorError, aiohttp.ClientOSError, socket.gaierror, ConnectionResetError)):
            return "Network issue. Retry {}"
        if isinstance(e, OSError) and "network is unreachable" in msg:
            return "Network unreachable. Retry {}"
        return "Retry {}"

    @staticmethod
    def _is_exchange_rate_limit_error(e: ccxt.ExchangeError) -> bool:
        msg = str(e).lower()
        return any(p in msg for p in _RATE_LIMIT_PHRASES)

    async def handle_network_error(self, error: Exception) -> bool:
        """Handle network-related errors."""
        template = self._classify_retryable_error(error)
        return await self._handle_retryable_error(template, error)

    async def handle_exchange_error(self, error: ccxt.ExchangeError) -> bool:
        """Handle exchange-specific errors."""
        if not self._is_exchange_rate_limit_error(error):
            prefix = self._format_prefix()
            self.logger.error(
                f"{prefix}Non-retryable ExchangeError in {self.class_name}.{self.func_name}: "
                f"{type(error).__name__} - {error}")
            return False

        template = "Rate limit (ExchangeError). Retry {}"
        return await self._handle_retryable_error(template, error, "ExchangeError")

    def handle_unexpected_error(self, error: Exception):
        """Handle unexpected errors that shouldn't be retried."""
        prefix = self._format_prefix()
        self.logger.error(
            f"{prefix}Unexpected error in {self.class_name}.{self.func_name}: "
            f"{type(error).__name__} - {error}\n{traceback.format_exc()}")


class _ApiRetryContext:
    """Helper class to manage API retry logic."""

    def __init__(self, instance, func, args, kwargs, max_retries, initial_delay, backoff_factor, max_delay,
                 retry_on_empty: bool = False):
        self.logger = instance.logger
        self.model = kwargs.get("model", args[0] if args else "unknown")
        self.func = func
        self.instance = instance
        self.args = args
        self.kwargs = kwargs
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.retry_on_empty = retry_on_empty
        self.retry_reason = ""
        self.last_rejection_reason = ""

    async def execute_with_retry(self) -> dict[str, Any] | None:
        """Execute the function with retry logic."""
        attempt = 0
        last_response: dict[str, Any] | None = None

        while attempt <= self.max_retries:
            try:
                response = await self.func(self.instance, *self.args, **self.kwargs)
                last_response = response

                if self._is_retryable_response(response):
                    if not self._should_retry(attempt):
                        break
                    await self._wait_and_increment(attempt)
                    attempt += 1
                    continue

                self._log_recovery(attempt)
                return response

            except Exception as e:
                self._log_exception(e)
                raise

        return last_response

    def _is_retryable_response(self, response: Any) -> bool:
        """Check if the response indicates a retryable error. Supports both dict and SDK Pydantic objects."""
        self.retry_reason = ""
        if response is None:
            return False
        if isinstance(response, dict):
            retryable = self._check_dict_response(response) or self._check_decisionless_reply(response)
        else:
            retryable = self._check_sdk_response(response) or self._check_decisionless_reply(response)
        if retryable:
            self.last_rejection_reason = self.retry_reason
        return retryable

    def _check_decisionless_reply(self, response: Any) -> bool:
        """Retry a reply that serves no usable JSON object, for opted-in providers.

        DeepSeek's JSON-output mode occasionally answers without an object — empty content,
        prose, or a degenerate object that only repeats the requested ``response_format``
        (``{"type": "json_object"}``, seen live 2026-09-27 after the reasoning phase used the
        whole completion budget). A reply cut off before its JSON closes looks identical to
        a parser, so ``finish_reason`` names which one happened and the whole rejected text
        is logged for replay (live 2026-10-27: 14.4k completion tokens, 11.2k of them
        reasoning, retried successfully 57 s later). The parser rejects such a cycle
        outright, so the identical request is resent.

        A well-formed object from a DIFFERENT contract of the same provider is a served
        answer, never a broken one: the post-mortem prompt asks for
        ``verdict``/``lesson_learned`` and has no ``analysis`` payload to hand back, so
        requiring one resent that answer three times per closed trade (2026-09-28, four
        paid calls for one post-mortem). The reply does not say which prompt it answers —
        the ``retry_on_empty`` opt-in means \"this provider must serve a JSON object\", not
        \"every prompt it serves is the trading decision contract\".
        """
        if not self.retry_on_empty:
            return False
        error = response.get("error") if isinstance(response, dict) else getattr(response, "error", None)
        if error:
            return False
        content = self._extract_content(response)
        if content is None:
            return False
        reason = self._rejection_reason(response, content)
        if reason is None:
            return False
        self.retry_reason = f"reply without a usable JSON payload — {reason}"
        self.logger.warning(
            "Provider reply rejected for model %s — %s (completion_tokens=%s)",
            self.model, reason, self._completion_tokens(response),
        )
        self.logger.warning(
            "Rejected reply content for model %s (%s chars): %s", self.model, len(content), content
        )
        return True

    def _rejection_reason(self, response: Any, content: str) -> str | None:
        """Named reason the reply serves no JSON payload, None when it does serve one.

        ``finish_reason`` separates a reply cut by the output cap from one the model ended
        with broken JSON: both decode to nothing, but only the first is a budget question,
        and the decoder message names where the JSON stopped in either case.
        """
        if self._first_choice(response) is None:
            return "reply carried no choices"
        if not content.strip():
            return "empty reply"
        finish_reason = self._finish_reason(response)
        payload, parse_error = self._first_json_object(content)
        if payload is None:
            if parse_error is None:
                return "prose without a JSON object"
            if finish_reason in _TRUNCATED_FINISH_REASONS:
                return f"truncated at the output cap (finish_reason={finish_reason})"
            if finish_reason in _FILTERED_FINISH_REASONS:
                return f"reply stopped by the content filter (finish_reason={finish_reason})"
            return f"malformed JSON ({parse_error}, finish_reason={finish_reason or 'none'})"
        if not self._carries_no_payload(payload):
            return None
        return f"object without a payload key (finish_reason={finish_reason or 'none'})"

    @staticmethod
    def _should_retry_api_error(error_value: Any) -> bool:
        """Check if an API error should trigger a retry."""
        if isinstance(error_value, str):
            try:
                error_value = json.loads(error_value)
            except ValueError:
                return error_value == "timeout"
        if isinstance(error_value, dict):
            if error_value.get("code") in (500, 502, 503, 504):
                return True
            metadata = error_value.get("metadata")
            raw = metadata.get("raw") if isinstance(metadata, dict) else None
            if isinstance(raw, dict) and raw.get("retryable"):
                return True
        return error_value == "timeout"

    @staticmethod
    def _first_choice(response: Any) -> Any:
        """First choice of a dict payload or an SDK reply, None when the reply carries none."""
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        return choices[0] if choices else None

    @staticmethod
    def _finish_reason(response: Any) -> str | None:
        """Normalised stop reason of the first choice, None when the provider reports none."""
        first = _ApiRetryContext._first_choice(response)
        if first is None:
            return None
        reason = first.get("finish_reason") if isinstance(first, dict) else getattr(first, "finish_reason", None)
        return reason.strip().lower() if isinstance(reason, str) and reason.strip() else None

    @staticmethod
    def _carries_no_payload(payload: dict[str, Any]) -> bool:
        """True when a reply object holds no payload key of its own.

        ``{"type": "json_object"}`` is the requested ``response_format`` coming back (live
        2026-09-27) and ``{}`` is an empty answer — neither holds anything a parser could
        read, so both are resent. A key naming a payload means the provider served a real
        answer, whichever contract's prompt produced it.
        """
        return set(payload) <= _META_PAYLOAD_KEYS

    @staticmethod
    def _error_label(error_value: Any) -> str:
        """Short label for an explicit provider error: its code, or the raw value."""
        if isinstance(error_value, str):
            try:
                error_value = json.loads(error_value)
            except ValueError:
                return f"({error_value})" if error_value else "(unspecified)"
        if isinstance(error_value, dict):
            code = error_value.get("code")
            return f"({code})" if code is not None else "(unspecified)"
        return "(unspecified)"

    @staticmethod
    def _completion_tokens(response: Any) -> Any:
        """Completion token count of a dict payload or an SDK reply, ``?`` when absent."""
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        tokens = getattr(usage, "completion_tokens", None)
        if tokens is None and isinstance(usage, dict):
            tokens = usage.get("completion_tokens")
        return tokens if isinstance(tokens, int) else "?"

    @staticmethod
    def _first_json_object(content: str) -> tuple[dict[str, Any] | None, str | None]:
        """First JSON object in the reply text, with the decoder message when there is none.

        The decoder message is kept verbatim: ``Unterminated string starting at ... (char N)``
        is what separates a reply cut off mid-string from one carrying broken JSON.
        """
        first_brace = content.find("{")
        if first_brace == -1:
            return None, None
        try:
            payload, _ = json.JSONDecoder().raw_decode(content, first_brace)
        except ValueError as error:
            return None, str(error)
        return (payload, None) if isinstance(payload, dict) else (None, None)

    @classmethod
    def _extract_content(cls, response: Any) -> str | None:
        """Visible reply text from an SDK object or a dict payload, None when absent."""
        first = cls._first_choice(response)
        if first is None:
            return ""
        error = first.get("error") if isinstance(first, dict) else getattr(first, "error", None)
        if error:
            return None
        message = getattr(first, "message", None)
        if message is None and isinstance(first, dict):
            message = first.get("message")
        if message is None:
            return None
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        return content if isinstance(content, str) else None

    def _check_dict_response(self, response: dict[str, Any]) -> bool:
        """Check dict-based response for retryable errors."""
        if response.get("error") and self._should_retry_api_error(response["error"]):
            self.retry_reason = f"provider error {self._error_label(response['error'])}"
            self.logger.warning("Retryable top-level error for model %s: %s", self.model, response["error"])
            return True
        choices = response.get("choices", [])
        if choices:
            first_choice = choices[0]
            if "error" in first_choice:
                choice_error = first_choice["error"]
                if self._should_retry_api_error(choice_error):
                    error_code = choice_error.get("code", "unknown")
                    error_msg = choice_error.get("message", "unknown")
                    provider = choice_error.get("metadata", {}).get("provider_name", "unknown")
                    self.retry_reason = f"provider error {self._error_label(choice_error)}"
                    self.logger.warning("Retryable error from %s in response choices for model %s: [%s] %s", provider, self.model, error_code, error_msg)
                    return True
        return False

    @staticmethod
    def _sdk_error_dict(error: Any) -> dict[str, Any] | None:
        """Normalise an SDK error object into a plain dict, or None when absent."""
        if not error:
            return None
        if isinstance(error, dict):
            return error
        try:
            return error.model_dump()
        except AttributeError:
            return {"message": str(error)}

    def _check_sdk_response(self, response: Any) -> bool:
        """Check an SDK response object for retryable errors using duck-typing."""
        error = getattr(response, "error", None)
        error_dict = self._sdk_error_dict(error)
        if self._should_retry_api_error(error) or (error_dict and self._should_retry_api_error(error_dict)):
            self.retry_reason = f"provider error {self._error_label(error_dict or error)}"
            self.logger.warning("Retryable SDK error for model %s: %s", self.model, error)
            return True

        choices = getattr(response, "choices", None) or []
        choice_error = getattr(choices[0], "error", None) if choices else None
        error_dict = self._sdk_error_dict(choice_error)
        if not (error_dict and self._should_retry_api_error(error_dict)):
            return False

        self.retry_reason = f"provider error {self._error_label(error_dict)}"
        metadata = getattr(choice_error, "metadata", None)
        self.logger.warning(
            "Retryable SDK error from %s in choices for model %s: [%s] %s",
            getattr(metadata, "provider_name", "unknown"),
            self.model,
            getattr(choice_error, "code", "unknown"),
            getattr(choice_error, "message", "unknown"),
        )
        return True

    def _should_retry(self, attempt: int) -> bool:
        """Determine if we should continue retrying."""
        if attempt >= self.max_retries:
            reason = f" (last reason: {self.retry_reason})" if self.retry_reason else ""
            self.logger.error(
                "API call to model %s failed after %s retries%s", self.model, self.max_retries, reason
            )
            return False
        return True

    def _log_recovery(self, attempt: int) -> None:
        """Announce a reply that only arrived after earlier attempts were rejected."""
        if attempt == 0:
            return
        retries = "retry" if attempt == 1 else "retries"
        self.logger.info(
            "Reply from model %s accepted on attempt %s of %s after %s %s (last reason: %s)",
            self.model, attempt + 1, self.max_retries + 1, attempt, retries,
            self.last_rejection_reason or "unknown",
        )

    async def _wait_and_increment(self, attempt: int) -> None:
        """Wait before the next attempt, naming the reason for every retry."""
        wait_time = min(self.initial_delay * (self.backoff_factor ** attempt), self.max_delay)
        reason = f" — {self.retry_reason}" if self.retry_reason else ""
        self.logger.warning(
            "Retrying model %s in %.2fs (retry %s of %s)%s",
            self.model, wait_time, attempt + 1, self.max_retries, reason,
        )
        await asyncio.sleep(_add_jitter(wait_time))

    def _log_exception(self, e: Exception):
        """Log exception details."""
        self.logger.error("Error in API call to model %s: %s - %s", self.model, type(e).__name__, e)
        self.logger.error("Traceback:\n%s", traceback.format_exc())


def retry_async(max_retries: int = -1, initial_delay: float = 1, backoff_factor: float = 2, max_delay: float = 3600):
    """Generic retry decorator for async instance methods."""
    def decorator(func: Any):
        @functools.wraps(func)
        async def wrapper(self, *args: Any, **kwargs: Any):
            context = _RetryContext(self, func, args, kwargs, max_retries, initial_delay, backoff_factor, max_delay)

            while True:
                try:
                    return await func(self, *args, **kwargs)
                except _NETWORK_EXCEPTIONS as e:
                    if not await context.handle_network_error(e):
                        raise
                except ccxt.ExchangeError as e:
                    if not await context.handle_exchange_error(e):
                        raise
                except Exception as e:
                    context.handle_unexpected_error(e)
                    raise
        return wrapper
    return decorator


def retry_async_default(func: Any) -> Any:
    """Apply the project-wide async retry policy: 3 attempts, 1s base delay, x2 backoff, 30s cap."""
    return retry_async(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)(func)


def retry_api_call(
    max_retries: int = 3,
    initial_delay: float = 1,
    backoff_factor: float = 2,
    max_delay: float = 60,
    retry_on_empty: bool = False,
):
    """Retry decorator for API call methods that return a dict possibly containing an 'error' key.

    ``retry_on_empty`` additionally retries missing choices, blank content, prose,
    an object that only echoes the requested format, or a reply whose JSON was cut
    off. Any usable JSON object, including a post-mortem, is a served answer.
    Explicit errors retain their own retry classification. Off by default for
    non-JSON providers.

    Every retry names its reason: a provider error carries its code, and a rejected
    reply carries one of ``empty reply``, ``prose without a JSON object``,
    ``object without a payload key``, ``truncated at the output cap``,
    ``reply stopped by the content filter`` or ``malformed JSON (...)``. ``finish_reason``
    separates a reply cut by the output cap from one the model ended with broken JSON;
    the rejected content is logged in full so the event can be replayed.
    """
    def decorator(func: Any):
        @functools.wraps(func)
        async def wrapper(self, *args: Any, **kwargs: Any):
            context = _ApiRetryContext(
                self, func, args, kwargs, max_retries, initial_delay, backoff_factor, max_delay,
                retry_on_empty,
            )
            return await context.execute_with_retry()
        return wrapper
    return decorator


def retry_api_call_default(func: Any) -> Any:
    """Apply the project-wide API retry policy: 3 attempts, 1s base delay, x2 backoff, 30s cap."""
    return retry_api_call(max_retries=3, initial_delay=1, backoff_factor=2, max_delay=30)(func)
