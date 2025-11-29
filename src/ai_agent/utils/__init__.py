"""Utility modules."""

# src/ai_agent/utils/__init__.py
"""Utility functions and helpers."""

from ai_agent.utils.logger import setup_logger
from ai_agent.utils.helpers import (
    sanitize_filename,
    compute_file_hash,
    format_file_size,
)

__all__ = [
    "setup_logger",
    "sanitize_filename",
    "compute_file_hash",
    "format_file_size",
]


# src/ai_agent/utils/logger.py
"""Logging configuration."""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: Path | None = None,
    rotation: str = "10 MB",
) -> None:
    """Setup application logger.
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        rotation: Log rotation size
    """
    # Remove default handler
    logger.remove()

    # Add console handler with colors
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        colorize=True,
    )

    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation=rotation,
            retention="1 week",
            compression="zip",
        )

        logger.info(f"Logging to file: {log_file}")


# src/ai_agent/utils/helpers.py
"""Helper utility functions."""

import hashlib
import re
from pathlib import Path
from typing import Optional


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters.
    Args:
        filename: Original filename
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "", filename)

    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(". ")

    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit(".", 1) if "." in sanitized else (sanitized, "")
        sanitized = name[: 255 - len(ext) - 1] + "." + ext if ext else name[:255]

    return sanitized or "untitled"


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file.
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)
    Returns:
        Hex digest of file hash
    """
    hash_obj = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    Args:
        size_bytes: Size in bytes
    Returns:
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.2f} PB"


def extract_code_blocks(text: str, language: Optional[str] = None) -> list[dict[str, str]]:
    """Extract code blocks from markdown text.
    Args:
        text: Markdown text
        language: Filter by specific language
    Returns:
        List of code block dicts with 'language' and 'code' keys
    """
    pattern = r"```(\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    blocks = []
    for lang, code in matches:
        if language is None or lang == language:
            blocks.append(
                {
                    "language": lang or "text",
                    "code": code.strip(),
                }
            )

    return blocks


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length.
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """Split a list into chunks.
    Args:
        lst: List to split
        chunk_size: Size of each chunk
    Returns:
        List of chunks
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


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
