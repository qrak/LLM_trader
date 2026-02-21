import time
import functools
import asyncio
from typing import Callable, Any
from src.config.loader import config
from src.utils.protocols import HasLogger

def profile_performance(func: Callable) -> Callable:
    """
    Decorator to measure and log the execution time of a method.
    Only active when logger_debug is True in config.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        # Check config at runtime to allow hot reloading
        if not config.LOGGER_DEBUG:
            return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) \
                else func(*args, **kwargs)

        instance = args[0] if args else None
        logger = instance.logger if isinstance(instance, HasLogger) else None

        start_time = time.perf_counter()
        class_name = args[0].__class__.__name__ if args else ''
        method_name = func.__name__

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000  # Convert to ms

            # Identify if it's a "slow" operation (>1s) for highlight
            slow_marker = " [SLOW]" if duration > 1000 else ""
            msg = f"Performance: {class_name}.{method_name} took {duration:.2f}ms{slow_marker}"

            if logger:
                logger.debug(msg)
            else:
                # Fallback print if logger not found on instance
                print(f"[DEBUG] {msg}")

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        if not config.LOGGER_DEBUG:
            return func(*args, **kwargs)

        instance = args[0] if args else None
        logger = instance.logger if isinstance(instance, HasLogger) else None

        start_time = time.perf_counter()
        class_name = args[0].__class__.__name__ if args else ''
        method_name = func.__name__

        try:
            return func(*args, **kwargs)
        finally:
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000

            slow_marker = " [SLOW]" if duration > 1000 else ""
            msg = f"Performance: {class_name}.{method_name} took {duration:.2f}ms{slow_marker}"

            if logger:
                logger.debug(msg)
            else:
                print(f"[DEBUG] {msg}")

    # Return appropriate wrapper based on whether the original function is async
    if asyncio.iscoroutinefunction(func):
        return wrapper
    else:
        return sync_wrapper
