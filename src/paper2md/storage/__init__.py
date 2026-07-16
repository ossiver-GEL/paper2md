"""Output writer — writes ParseResult to disk as Markdown with images.

Produces:
    output_dir/
    ├── paper_title.md       # YAML frontmatter + Markdown body
    └── images/              # Extracted figures (content-hash named)
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from ..models import ImageAsset, ParseResult

logger = logging.getLogger(__name__)

# Pattern to match Markdown image references: ![alt](path)
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


class OutputWriter:
    """Writes parsed paper to disk as Markdown with local image assets.

    Handles:
    - Writing .md file with YAML frontmatter (metadata)
    - Saving embedded image bytes to images/ subdirectory
    - Rewriting image references in markdown to local relative paths
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self, result: ParseResult, filename: str | None = None
    ) -> Path:
        """Write parsed result to disk.

        Args:
            result: The parsed paper with markdown, metadata, and images.
            filename: Output filename stem (without .md extension).
                      Defaults to a sanitized version of the paper title.

        Returns:
            Path to the written Markdown file.
        """
        if filename is None:
            filename = self._sanitize_filename(result.metadata.title or "paper")

        # Save embedded image bytes to images/
        saved_images: dict[str, str] = {}
        for img in result.images:
            local_path = self._save_image(img)
            saved_images[img.filename] = str(
                local_path.relative_to(self.output_dir)
            )

        # Rewrite markdown image references to local paths
        body = result.markdown_body
        body = self._rewrite_image_refs(body, saved_images)

        # Assemble: frontmatter + body
        frontmatter = result.metadata.to_frontmatter()
        full_content = frontmatter + "\n" + body

        md_path = self.output_dir / f"{filename}.md"
        md_path.write_text(full_content, encoding="utf-8")

        logger.info("Written output: %s", md_path)
        return md_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_image(self, img: ImageAsset) -> Path:
        """Save an image to images/ with content-hash naming."""
        content_hash = hashlib.sha256(img.data).hexdigest()[:12]
        ext = self._guess_extension(img.filename, img.data)
        safe_name = f"{content_hash}.{ext}"
        dest = self.images_dir / safe_name

        if not dest.exists():
            dest.write_bytes(img.data)

        return dest

    def _rewrite_image_refs(
        self, body: str, image_map: dict[str, str]
    ) -> str:
        """Rewrite Markdown image references to use local paths.

        Matches both filename-only references and URL-based references.
        """
        def _replace(match: re.Match) -> str:
            alt = match.group(1)
            ref = match.group(2)

            # Check if there's a mapped local image
            for orig_name, local_path in image_map.items():
                # Match by filename ending
                if orig_name in ref or ref.endswith(orig_name):
                    return f"![{alt}]({local_path})"

            # If the reference is a filename that matches a saved image
            ref_basename = ref.rsplit("/", 1)[-1]
            if ref_basename in image_map:
                return f"![{alt}]({image_map[ref_basename]})"

            return match.group(0)  # Keep original

        return _MD_IMAGE_RE.sub(_replace, body)

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Create a safe filename from a paper title."""
        # Remove non-alphanumeric characters
        safe = re.sub(r"[^\w\s-]", "", name)
        # Replace whitespace with underscores
        safe = re.sub(r"\s+", "_", safe)
        # Truncate to reasonable length
        return safe[:100].strip("_") or "paper"

    @staticmethod
    def _guess_extension(filename: str, data: bytes) -> str:
        """Guess file extension from filename or magic bytes."""
        import imghdr

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext in ("png", "jpg", "jpeg", "gif", "webp", "bmp", "svg"):
            return ext

        # Try to detect from magic bytes
        detected = imghdr.what(None, data)
        if detected:
            return detected

        return "png"  # Default fallback
