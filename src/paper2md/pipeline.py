"""Conversion pipeline — coordinates source extraction and output writing.

The main entry point for paper conversion. It:
1. Auto-detects the appropriate source extractor via SourceRegistry
2. Runs extraction (source-specific parser or MinerU for PDFs)
3. Writes unified Markdown output with images and YAML frontmatter
"""

from __future__ import annotations

import logging
from pathlib import Path

from .models import ConvertConfig, ParseResult
from .sources.registry import SourceRegistry
from .storage import OutputWriter

logger = logging.getLogger(__name__)


class ConversionPipeline:
    """Orchestrates the detect → extract → write conversion workflow.

    Usage:
        config = ConvertConfig(
            source="https://arxiv.org/abs/2301.12345",
            output_dir=Path("./output"),
        )
        pipeline = ConversionPipeline()
        result_path = await pipeline.run(config)
    """

    async def run(self, config: ConvertConfig) -> Path:
        """Execute the full conversion pipeline.

        Returns:
            Path to the output Markdown file.

        Raises:
            ValueError: If no suitable extractor is found for the source.
            RuntimeError: If extraction or writing fails.
        """
        logger.info("Starting conversion: %s", config.source)

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Find matching extractor
        extractor = await SourceRegistry.find_extractor(config)
        if extractor is None:
            raise ValueError(
                f"No extractor found for source: {config.source}\n"
                f"Supported sources: {self._format_sources()}\n"
                f"Provide a URL, arXiv ID, DOI, or local PDF path."
            )

        logger.info(
            "Using %s extractor (type=%s)",
            extractor.display_name,
            extractor.source_type,
        )

        # Step 2: Extract
        try:
            result: ParseResult = await extractor.extract(config)
        except Exception as exc:
            logger.exception("Extraction failed for %s", config.source)
            raise RuntimeError(
                f"Extraction failed with {extractor.display_name}: {exc}"
            ) from exc

        if not result.markdown_body.strip():
            logger.warning("Extraction produced empty markdown for %s", config.source)

        # Step 3: Write output
        writer = OutputWriter(output_dir)
        md_path = writer.write(result)

        logger.info(
            "Conversion complete: %s (type=%s, images=%d)",
            md_path, result.source_type, len(result.images),
        )

        return md_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def list_sources() -> list[dict]:
        """Return metadata for all registered source extractors."""
        return SourceRegistry.list_sources()

    @staticmethod
    def _format_sources() -> str:
        """Format registered sources for error messages."""
        sources = SourceRegistry.list_sources()
        return ", ".join(
            f"{s['type']} ({s['name']})" for s in sources
        )
