"""Text splitting utilities for RAG.

Provides various strategies for splitting documents into chunks
suitable for embedding and retrieval.
"""

from typing import List, Optional
import re
from loguru import logger


class TextSplitter:
    """Split text into chunks for embedding.
    Attributes:
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        """Initialize text splitter.
        Args:
            chunk_size: Target size of each chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        """Split text into overlapping chunks.
        Args:
            text: Text to split
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < text_length:
                last_period = chunk.rfind(". ")
                if last_period > self.chunk_size // 2:
                    end = start + last_period + 1
                    chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk.strip())

            start = end - self.chunk_overlap

        return chunks

    def split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentence-based chunks.
        Args:
            text: Text to split
        Returns:
            List of chunks, each containing multiple sentences
        """
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = overlap_length

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


class RecursiveCharacterTextSplitter(TextSplitter):
    """Recursively split text using multiple separators.

    Tries to split on paragraphs first, then sentences, then words.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ) -> None:
        """Initialize recursive text splitter.
        Args:
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            separators: List of separators to try in order
        """
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split(self, text: str) -> List[str]:
        """Split text recursively using separators.
        Args:
            text: Text to split
        Returns:
            List of text chunks
        """
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text.
        Args:
            text: Text to split
            separators: Remaining separators to try
        Returns:
            List of text chunks
        """
        final_chunks = []

        # Get the first separator
        separator = separators[0] if separators else ""

        # Split by separator
        if separator:
            splits = text.split(separator)
        else:
            splits = [text]

        # Process each split
        good_splits = []
        for split in splits:
            if len(split) < self.chunk_size:
                good_splits.append(split)
            else:
                # Split is too large, try next separator
                if len(separators) > 1:
                    sub_chunks = self._split_text(split, separators[1:])
                    good_splits.extend(sub_chunks)
                else:
                    # No more separators, force split
                    good_splits.append(split)

        # Merge small chunks and create overlaps
        merged_chunks = self._merge_splits(good_splits, separator)

        return merged_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge small splits and add overlaps.
        Args:
            splits: List of text splits
            separator: Separator used for splitting
        Returns:
            List of merged chunks
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = len(split)

            if current_length + split_length + len(separator) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = separator.join(current_chunk)
                chunks.append(chunk_text)

                # Create overlap
                overlap = []
                overlap_length = 0
                for item in reversed(current_chunk):
                    item_length = len(item) + len(separator)
                    if overlap_length + item_length <= self.chunk_overlap:
                        overlap.insert(0, item)
                        overlap_length += item_length
                    else:
                        break

                current_chunk = overlap
                current_length = overlap_length

            current_chunk.append(split)
            current_length += split_length + len(separator)

        # Add remaining chunk
        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks
