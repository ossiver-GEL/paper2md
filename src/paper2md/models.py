"""Data models for the paper2md conversion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


class PaperMetadata:
    """Structured metadata for an academic paper.

    All fields are optional; extractors populate what they can.
    """

    def __init__(
        self,
        title: str = "",
        authors: list[str] | None = None,
        abstract: str = "",
        doi: str = "",
        arxiv_id: str = "",
        journal: str = "",
        year: str = "",
        url: str = "",
    ) -> None:
        self.title = title
        self.authors = authors or []
        self.abstract = abstract
        self.doi = doi
        self.arxiv_id = arxiv_id
        self.journal = journal
        self.year = year
        self.url = url

    def to_frontmatter(self) -> str:
        """Generate YAML frontmatter for the markdown output."""
        lines = ["---"]
        if self.title:
            lines.append(f'title: "{self.title}"')
        if self.authors:
            authors_str = ", ".join(self.authors)
            lines.append(f"authors: [{authors_str}]")
        if self.abstract:
            lines.append(f"abstract: >\n  {self.abstract[:300]}")
        if self.doi:
            lines.append(f"doi: {self.doi}")
        if self.arxiv_id:
            # Quote to prevent YAML interpreting dotted IDs (1706.03762) as floats
            lines.append(f"arxiv_id: \"{self.arxiv_id}\"")
        if self.journal:
            lines.append(f"journal: {self.journal}")
        if self.year:
            lines.append(f"year: {self.year}")
        if self.url:
            lines.append(f"url: {self.url}")
        lines.append("---\n")
        return "\n".join(lines)


@dataclass
class ImageAsset:
    """Represents an image extracted from a paper."""

    filename: str
    data: bytes
    caption: str = ""
    alt_text: str = ""


@dataclass
class ParseResult:
    """Result of parsing a paper."""

    metadata: PaperMetadata = field(default_factory=PaperMetadata)
    markdown_body: str = ""
    images: list[ImageAsset] = field(default_factory=list)
    source_type: str = "pdf_url"
    raw_source_url: str = ""


@dataclass
class ConvertConfig:
    """Configuration for a single paper conversion.

    Per-document parameters come from the MCP tool call.
    Server-wide settings (API keys, backend) are set via
    environment variables and injected by server.py.
    """

    source: str
    output_dir: Path
    mineru_api_key: str | None = None
    mineru_use_local: bool = False
    mineru_backend: str = "auto"
    model_version: str = "vlm"
    language: str = "en"
    is_ocr: bool = False
    enable_table: bool = True
    enable_formula: bool = True
    timeout: int = 600
