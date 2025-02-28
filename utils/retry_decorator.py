import asyncio
import functools
import logging
import time
import traceback
from collections import defaultdict
from typing import Any
from typing import Callable, Dict

import ccxt


def retry_async(max_retries: int = -1, initial_delay: float = 1, backoff_factor: float = 2, max_delay: float = 3600):
    def decorator(func: Any):
        @functools.wraps(func)
        async def wrapper(self, *args: Any, **kwargs: Any):
            logger = getattr(self, 'logger', None)

            pair = kwargs.get('pair') or (args[0] if args and isinstance(args[0], str) else None)

            def log_message(level: str, message: str) -> None:
                full_message = f"{pair} - {message}" if pair else message
                log_func = getattr(logger, level) if logger else getattr(logging, level)
                log_func(full_message, stacklevel=3)

            class_name = self.__class__.__name__

            retries = 0
            delay = initial_delay
            while max_retries == -1 or retries <= max_retries:
                try:
                    return await func(self, *args, **kwargs)
                except (ccxt.NetworkError, ccxt.RequestTimeout,
                        ccxt.DDoSProtection, ccxt.RateLimitExceeded, TimeoutError, ConnectionResetError) as e:
                    retries += 1
                    wait_time = min(delay * (backoff_factor ** (retries - 1)), max_delay)
                    log_message('warning',
                                f"Retry {retries} for {class_name}.{func.__name__} in {wait_time:.2f} seconds. "
                                f"Type: {type(e).__name__}, Error: {str(e)}")
                    await asyncio.sleep(wait_time)
                except ccxt.ExchangeError as e:
                    if "Too many requests" in str(e) or "rate limit" in str(e).lower():
                        retries += 1
                        wait_time = min(delay * (backoff_factor ** (retries - 1)), max_delay)
                        log_message('warning',
                                    f"Rate limit hit. Retry {retries} for {class_name}.{func.__name__} in {wait_time:.2f} seconds. "
                                    f"Error: {str(e)}")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
                except (ccxt.BadSymbol, ccxt.BadRequest):
                    raise
                except Exception as e:
                    log_message('error', f"Error in {class_name}.{func.__name__}: {type(e).__name__} - {str(e)}")
                    log_message('error', f"Traceback:\n{traceback.format_exc()}")
                    raise

            log_message('error', f"Function {class_name}.{func.__name__} failed after {max_retries} retries.")
            return None

        return wrapper
    return decorator

class TimingStats:
    def __init__(self):
        self.call_count: int = 0
        self.total_time: float = 0.0
        self.min_time: float = float('inf')
        self.max_time: float = 0.0

    def update(self, execution_time: float) -> None:
        self.call_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)

    @property
    def average_time(self) -> float:
        return self.total_time / self.call_count if self.call_count > 0 else 0.0


class TimingManager:
    def __init__(self, logger):
        self.stats: Dict[str, TimingStats] = defaultdict(TimingStats)
        self.logger = logger

    def update_stats(self, func_name: str, execution_time: float) -> None:
        self.stats[func_name].update(execution_time)

    def log_stats(self, func_name: str) -> None:
        stats = self.stats[func_name]
        self.logger.debug(
            f"Function: {func_name} - "
            f"Calls: {stats.call_count}, "
            f"Avg: {stats.average_time:.4f}s, "
            f"Min: {stats.min_time:.4f}s, "
            f"Max: {stats.max_time:.4f}s, "
            f"Total: {stats.total_time:.4f}s"
        )


def timing_decorator(func: Callable):
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(self, *args, **kwargs)
                return result
            finally:
                execution_time = time.perf_counter() - start_time
                self.timing_manager.update_stats(func.__name__, execution_time)
                self.timing_manager.log_stats(func.__name__)
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(self, *args, **kwargs)
                return result
            finally:
                execution_time = time.perf_counter() - start_time
                self.timing_manager.update_stats(func.__name__, execution_time)
                self.timing_manager.log_stats(func.__name__)
        return sync_wrapper