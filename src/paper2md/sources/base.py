"""Abstract base class for paper source extractors.

To add a new source:
1. Create a new .py file in this directory
2. Subclass ``BaseExtractor`` and implement ``can_handle()`` and ``extract()``
3. The ``SourceRegistry`` discovers it automatically
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import ConvertConfig, ParseResult


class BaseExtractor(ABC):
    """Abstract base for all paper source extractors."""

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Unique source type identifier (e.g. 'arxiv', 'nature', 'pdf_url')."""

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name (e.g. 'arXiv', 'Nature', 'PDF (URL)')."""

    @abstractmethod
    async def can_handle(self, config: ConvertConfig) -> bool:
        """Return True if this extractor can process the given source."""

    @abstractmethod
    async def extract(self, config: ConvertConfig) -> ParseResult:
        """Extract and convert the paper to structured Markdown.

        Returns a ``ParseResult`` with metadata, markdown body, and images.
        """

    async def validate(self, config: ConvertConfig) -> None:
        """Optional validation hook. Raises ValueError if config is invalid."""
        if not config.source or not config.source.strip():
            raise ValueError("Source URL/path cannot be empty.")
