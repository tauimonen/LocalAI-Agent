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
