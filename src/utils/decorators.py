import asyncio
import functools
import logging
import traceback
import socket
from typing import Any, Dict

import ccxt
import aiohttp
import aiodns

# -------------------------------------------------------------
# Internal helper utilities (extracted to reduce cyclomatic
# complexity of the decorator wrapper functions). Keeping
# signatures small and pure simplifies testing and reuse.
# -------------------------------------------------------------

_RATE_LIMIT_PHRASES = {
    'too many requests', 'rate limit', '429', 'ratelimit',
    'ddos protection', 'system-level rate limit exceeded'
}

_NETWORK_EXCEPTIONS = (
    ccxt.NetworkError, ccxt.RequestTimeout, ccxt.DDoSProtection, ccxt.RateLimitExceeded,
    TimeoutError, ConnectionResetError, aiohttp.ClientConnectorError, aiohttp.ClientOSError,
    asyncio.TimeoutError, socket.gaierror, OSError, aiodns.error.DNSError
)


def _log(logger, level: str, message: str):
    log_func = getattr(logger, level) if logger else getattr(logging, level)
    if logger and hasattr(logger, 'findCaller'):
        log_func(message, stacklevel=3)
    else:
        log_func(message)


def _classify_retryable_error(e: Exception) -> str:
    msg = str(e).lower()
    if isinstance(e, (ccxt.RateLimitExceeded, ccxt.DDoSProtection)) or any(p in msg for p in _RATE_LIMIT_PHRASES):
        return "Rate limit/DDoS. Retry {}"
    if isinstance(e, (ccxt.RequestTimeout, TimeoutError, asyncio.TimeoutError)) or 'timeout' in msg:
        return "Timeout. Retry {}"
    if isinstance(e, (ccxt.NetworkError, aiohttp.ClientConnectorError, aiohttp.ClientOSError, socket.gaierror, ConnectionResetError)):
        return "Network issue. Retry {}"
    if isinstance(e, OSError) and 'network is unreachable' in msg:
        return "Network unreachable. Retry {}"
    return "Retry {}"


def _is_exchange_rate_limit_error(e: ccxt.ExchangeError) -> bool:
    msg = str(e).lower()
    return any(p in msg for p in _RATE_LIMIT_PHRASES)


def retry_async(max_retries: int = -1, initial_delay: float = 1, backoff_factor: float = 2, max_delay: float = 3600):
    """Generic retry decorator for async instance methods.

    Args:
        max_retries: -1 for infinite retries, otherwise max attempts before raising.
        initial_delay: Initial backoff delay seconds.
        backoff_factor: Multiplier applied each retry.
        max_delay: Upper bound for backoff delay.
    """
    def decorator(func: Any):
        @functools.wraps(func)
        async def wrapper(self, *args: Any, **kwargs: Any):
            context = _RetryContext(self, func, args, kwargs, max_retries, initial_delay, backoff_factor, max_delay)
            
            while True:  # Controlled exit via return or raise
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


class _RetryContext:
    """Helper class to manage retry logic and reduce complexity."""
    
    def __init__(self, instance, func, args, kwargs, max_retries, initial_delay, backoff_factor, max_delay):
        self.logger = getattr(instance, 'logger', None)
        self.pair = kwargs.get('pair') or (args[0] if args and isinstance(args[0], str) else None)
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
        _log(self.logger, 'error', 
             f"{prefix}Function {self.class_name}.{self.func_name} failed after {self.max_retries} retries. "
             f"Last error: {error_type} - {error}")
    
    async def _handle_retryable_error(self, template: str, error: Exception, error_type: str = None) -> bool:
        """Common logic for handling retryable errors."""
        if not self._should_continue_retrying():
            self._log_failure(error_type or type(error).__name__, error)
            return False
        
        prefix = self._format_prefix()
        _log(self.logger, 'warning',
             f"{prefix}{template.format(self.attempt)} for {self.class_name}.{self.func_name} "
             f"in {self.delay:.2f} seconds. Type: {type(error).__name__}, Error: {error}")
        
        await asyncio.sleep(self.delay)
        self.delay = min(self.delay * self.backoff_factor, self.max_delay)
        return True
    
    async def handle_network_error(self, error: Exception) -> bool:
        """Handle network-related errors."""
        template = _classify_retryable_error(error)
        return await self._handle_retryable_error(template, error)
    
    async def handle_exchange_error(self, error: ccxt.ExchangeError) -> bool:
        """Handle exchange-specific errors."""
        if not _is_exchange_rate_limit_error(error):
            prefix = self._format_prefix()
            _log(self.logger, 'error',
                 f"{prefix}Non-retryable ExchangeError in {self.class_name}.{self.func_name}: "
                 f"{type(error).__name__} - {error}")
            return False
        
        template = "Rate limit (ExchangeError). Retry {}"
        return await self._handle_retryable_error(template, error, "ExchangeError")
    
    def handle_unexpected_error(self, error: Exception):
        """Handle unexpected errors that shouldn't be retried."""
        prefix = self._format_prefix()
        _log(self.logger, 'error',
             f"{prefix}Unexpected error in {self.class_name}.{self.func_name}: "
             f"{type(error).__name__} - {error}\n{traceback.format_exc()}")


def _should_retry_api_error(error_value: Any) -> bool:
    """Check if an API error should trigger a retry."""
    # Top-level error codes
    if isinstance(error_value, dict):
        error_code = error_value.get('code')
        if error_code in (500, 502, 503, 504):  # Server errors
            return True
        # Check for retryable flag from OpenRouter
        if error_value.get('metadata', {}).get('raw', {}).get('retryable'):
            return True
    return error_value == 'timeout'


def retry_api_call(max_retries: int = 3, initial_delay: float = 1, backoff_factor: float = 2, max_delay: float = 60):
    """Retry decorator for API call methods that return a dict possibly containing an 'error' key."""
    def decorator(func: Any):
        @functools.wraps(func)
        async def wrapper(self, *args: Any, **kwargs: Any):
            context = _ApiRetryContext(self, func, args, kwargs, max_retries, initial_delay, backoff_factor, max_delay)
            return await context.execute_with_retry()
        return wrapper
    return decorator


class _ApiRetryContext:
    """Helper class to manage API retry logic."""
    
    def __init__(self, instance, func, args, kwargs, max_retries, initial_delay, backoff_factor, max_delay):
        self.logger = getattr(instance, 'logger', None) or logging.getLogger(__name__)
        self.model = kwargs.get('model', args[0] if args else 'unknown')
        self.func = func
        self.instance = instance
        self.args = args
        self.kwargs = kwargs
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
    
    async def execute_with_retry(self) -> Dict[str, Any] | None:
        """Execute the function with retry logic."""
        attempt = 0
        last_response: Dict[str, Any] | None = None

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
                    
                return response  # success or non-retryable error structure
                
            except Exception as e:
                self._log_exception(e)
                raise

        # Exhausted retries (only reached on retryable error path)
        return last_response
    
    def _is_retryable_response(self, response: Dict[str, Any]) -> bool:
        """Check if the response indicates a retryable error."""
        if not isinstance(response, dict):
            return False
        
        # Check top-level error
        if response.get('error') and _should_retry_api_error(response['error']):
            self.logger.warning(f"Retryable top-level error for model {self.model}: {response['error']}")
            return True
        
        # Check for errors embedded in choices array (OpenRouter format)
        choices = response.get('choices', [])
        if choices and isinstance(choices, list):
            first_choice = choices[0]
            if isinstance(first_choice, dict) and 'error' in first_choice:
                choice_error = first_choice['error']
                if _should_retry_api_error(choice_error):
                    error_code = choice_error.get('code', 'unknown')
                    error_msg = choice_error.get('message', 'unknown')
                    provider = choice_error.get('metadata', {}).get('provider_name', 'unknown')
                    self.logger.warning(
                        f"Retryable error from {provider} in response choices for model {self.model}: "
                        f"[{error_code}] {error_msg}"
                    )
                    return True
        
        return False
    
    def _should_retry(self, attempt: int) -> bool:
        """Determine if we should continue retrying."""
        if attempt >= self.max_retries:
            self.logger.error(f"API call to model {self.model} failed after {self.max_retries} retries")
            return False
        return True
    
    async def _wait_and_increment(self, attempt: int):
        """Wait before next retry attempt."""
        wait_time = min(self.initial_delay * (self.backoff_factor ** attempt), self.max_delay)
        self.logger.warning(
            f"API returned error for model {self.model}. "
            f"Retrying in {wait_time:.2f}s ({attempt + 1}/{self.max_retries})"
        )
        await asyncio.sleep(wait_time)
    
    def _log_exception(self, e: Exception):
        """Log exception details."""
        self.logger.error(f"Error in API call to model {self.model}: {type(e).__name__} - {e}")
        self.logger.error(f"Traceback:\n{traceback.format_exc()}")