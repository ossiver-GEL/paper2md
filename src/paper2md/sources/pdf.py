"""PDF source extractors (URL and local file).

Uses MinerU for PDF parsing. Supports both cloud API and local deployment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

from ..mineru.client import MinerUClient
from ..mineru.local import MinerULocal
from ..models import ConvertConfig, ImageAsset, PaperMetadata, ParseResult
from .base import BaseExtractor

logger = logging.getLogger(__name__)


# Common PDF MIME types and extensions
PDF_EXTENSIONS = {".pdf"}
PDF_MIME_TYPES = {"application/pdf", "application/x-pdf"}


class PdfUrlExtractor(BaseExtractor):
    """Extract papers from a remote PDF URL using MinerU cloud API."""

    @property
    def source_type(self) -> str:
        return "pdf_url"

    @property
    def display_name(self) -> str:
        return "PDF (URL)"

    async def can_handle(self, config: ConvertConfig) -> bool:
        """Check if source is a remote PDF URL."""
        source = config.source.strip()

        # Must be a URL
        parsed = urlparse(source)
        if not parsed.scheme or parsed.scheme not in ("http", "https"):
            return False

        # Check if it ends with .pdf or has PDF in URL
        path_lower = parsed.path.lower()
        if path_lower.endswith(".pdf") or "/pdf/" in path_lower:
            return True

        # Try HEAD request to check Content-Type
        try:
            import httpx

            # Synchronous check for simplicity in can_handle
            # (could use async but it's a quick check)
            return False  # Defer to HEAD check in extract()
        except Exception:
            pass

        return path_lower.endswith(".pdf")

    async def extract(self, config: ConvertConfig) -> ParseResult:
        """Extract PDF from URL using MinerU."""
        source = config.source.strip()
        logger.info("Extracting PDF from URL: %s", source)

        use_precision = bool(config.mineru_api_key)

        client = MinerUClient(
            api_key=config.mineru_api_key,
            use_precision=use_precision,
            model_version=config.model_version,
            language=config.language,
            enable_table=config.enable_table,
            enable_formula=config.enable_formula,
            is_ocr=config.is_ocr,
            timeout=config.timeout,
        )

        result = await client.parse_url(source)

        if result.state == "failed":
            raise RuntimeError(
                f"MinerU extraction failed: {result.error_message}"
            )

        # Parse paper title from markdown (first H1)
        title = self._extract_title(result.markdown_text)

        # Convert MinerU images to ImageAsset list
        images = [
            ImageAsset(filename=name, data=data)
            for name, data in result.images.items()
        ]

        return ParseResult(
            metadata=PaperMetadata(title=title, url=source),
            markdown_body=result.markdown_text,
            images=images,
            source_type="pdf_url",
            raw_source_url=source,
        )

    def _extract_title(self, markdown: str) -> str:
        """Extract paper title from the first H1 in markdown."""
        for line in markdown.split("\n"):
            line = line.strip()
            if line.startswith("# ") and not line.startswith("## "):
                return line[2:].strip()
        return ""


class PdfLocalExtractor(BaseExtractor):
    """Extract papers from a local PDF file using MinerU.

    Supports both:
    - Local MinerU deployment (GPU-accelerated, via mineru CLI v3.4+)
    - MinerU cloud API (file upload)
    """

    @property
    def source_type(self) -> str:
        return "pdf_local"

    @property
    def display_name(self) -> str:
        return "PDF (local file)"

    async def can_handle(self, config: ConvertConfig) -> bool:
        """Check if source is a local PDF file path."""
        source = config.source.strip()

        # Check if it's a file path (not a URL)
        parsed = urlparse(source)
        # On Windows, urlparse interprets "C:" as a scheme; treat single-letter
        # schemes as drive letters, not URLs.
        is_windows_drive = (
            len(parsed.scheme) == 1 and parsed.scheme.isalpha()
        )
        if parsed.scheme and not is_windows_drive and parsed.scheme not in ("file", ""):
            return False

        # Normalize path
        path = Path(source)
        if path.exists() and path.is_file():
            ext = path.suffix.lower()
            if ext in PDF_EXTENSIONS:
                return True

        return False

    async def extract(self, config: ConvertConfig) -> ParseResult:
        """Extract local PDF using MinerU (local or cloud)."""
        source = config.source.strip()
        file_path = Path(source)

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {source}")

        logger.info("Extracting local PDF: %s", file_path)

        # Determine extraction method
        if config.mineru_use_local:
            return await self._extract_local(file_path, config)
        elif config.mineru_api_key:
            return await self._extract_cloud(file_path, config)
        else:
            # Default: try local first, fall back to agent API
            try:
                return await self._extract_local(file_path, config)
            except Exception as e:
                logger.warning(
                    "Local MinerU failed: %s. Falling back to agent API.", e
                )
                return await self._extract_cloud(file_path, config)

    async def _extract_local(
        self, file_path: Path, config: ConvertConfig
    ) -> ParseResult:
        """Use local MinerU (mineru CLI v3.4+)."""
        local = MinerULocal(
            method="auto",
            backend=config.mineru_backend,
            language=config.language,
            enable_table=config.enable_table,
            enable_formula=config.enable_formula,
        )

        result = await local.parse(file_path, output_dir=config.output_dir)

        if result.state == "failed":
            raise RuntimeError(
                f"Local MinerU extraction failed: {result.error_message}"
            )

        title = self._extract_title(result.markdown_text)

        images = [
            ImageAsset(filename=name, data=data)
            for name, data in result.images.items()
        ]

        return ParseResult(
            metadata=PaperMetadata(title=title, url=str(file_path)),
            markdown_body=result.markdown_text,
            images=images,
            source_type="pdf_local",
            raw_source_url=str(file_path),
        )

    async def _extract_cloud(
        self, file_path: Path, config: ConvertConfig
    ) -> ParseResult:
        """Upload to MinerU cloud API."""
        use_precision = bool(config.mineru_api_key)

        client = MinerUClient(
            api_key=config.mineru_api_key,
            use_precision=use_precision,
            model_version=config.model_version,
            language=config.language,
            enable_table=config.enable_table,
            enable_formula=config.enable_formula,
            is_ocr=config.is_ocr,
            timeout=config.timeout,
        )

        result = await client.parse_file(file_path)

        if result.state == "failed":
            raise RuntimeError(
                f"MinerU extraction failed: {result.error_message}"
            )

        title = self._extract_title(result.markdown_text)

        images = [
            ImageAsset(filename=name, data=data)
            for name, data in result.images.items()
        ]

        return ParseResult(
            metadata=PaperMetadata(title=title, url=str(file_path)),
            markdown_body=result.markdown_text,
            images=images,
            source_type="pdf_local",
            raw_source_url=str(file_path),
        )

    def _extract_title(self, markdown: str) -> str:
        """Extract paper title from the first H1 in markdown."""
        for line in markdown.split("\n"):
            line = line.strip()
            if line.startswith("# ") and not line.startswith("## "):
                return line[2:].strip()
        return ""
