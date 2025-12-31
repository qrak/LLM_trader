"""
Protocols for strict typing across the codebase.
Replaces dynamic getattr/hasattr checks with compile-time guarantees.
"""
from typing import Protocol, runtime_checkable
from src.logger.logger import Logger


@runtime_checkable
class HasLogger(Protocol):
    """Protocol for classes that have a logger attribute."""
    logger: Logger
