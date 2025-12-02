# src/ai_agent/utils/timing.py
"""Timing and performance utilities."""

import time
from functools import wraps
from typing import Callable, Any
from loguru import logger


def timed(func: Callable) -> Callable:
    """Decorator to time function execution.
    Args:
        func: Function to time
    Returns:
        Wrapped function
    """

    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} took {elapsed:.4f} seconds")
        return result

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} took {elapsed:.4f} seconds")
        return result

    # Return appropriate wrapper based on function type
    import asyncio

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "operation"):
        """Initialize timer.

        Args:
            name: Name of operation being timed
        """
        self.name = name
        self.start_time: float = 0
        self.elapsed: float = 0

    def __enter__(self) -> "Timer":
        """Start timer."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop timer and log."""
        self.elapsed = time.perf_counter() - self.start_time
        logger.debug(f"{self.name} took {self.elapsed:.4f} seconds")
