"""arXiv paper source extractor.

Uses the arXiv API for metadata and ar5iv HTML for structured content.
ar5iv (ar5iv.labs.arxiv.org) converts LaTeX to HTML5 using LaTeXML —
the same engine that powers arXiv's official HTML offering (since 2023).

Extraction strategy (3-tier fallback):
1. Fetch metadata from arXiv API (Atom XML) — always available
2. Parse ar5iv HTML via structural DOM walker
3. If ar5iv quality is low → try MinerU on arXiv PDF (arxiv.org/pdf/{id}.pdf)
4. If MinerU unavailable/fails → fall back to arXiv abstract page

The structural walker processes the DOM top-down by ar5iv class names
(ltx_section, ltx_para, ltx_figure, etc.), producing clean Markdown
with preserved math (via alttext), figures, and tables.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag

from ..models import ConvertConfig, ImageAsset, PaperMetadata, ParseResult
from .base import BaseExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# arXiv ID patterns (ordered: specific → general)
ARXIV_ID_PATTERNS = [
    re.compile(r"arxiv\.org/abs/([a-zA-Z0-9.\-]+?)(?:v\d+)?(?:\.pdf)?/?$"),
    re.compile(r"arxiv\.org/pdf/([a-zA-Z0-9.\-]+?)(?:v\d+)?(?:\.pdf)?/?$"),
    re.compile(r"ar5iv\.labs\.arxiv\.org/html/([a-zA-Z0-9.\-]+)"),
    re.compile(r"^([a-zA-Z\-]+/\d{4,}|\d{4}\.\d{4,})(v\d+)?$"),
]

ARXIV_API = "https://export.arxiv.org/api/query"
AR5IV_BASE = "https://ar5iv.labs.arxiv.org/html"

# Quality threshold for accepting ar5iv content (0–1 scale)
AR5IV_QUALITY_THRESHOLD = 0.4


class ArxivExtractor(BaseExtractor):
    """Extract papers from arXiv using ar5iv HTML or arXiv API."""

    @property
    def source_type(self) -> str:
        return "arxiv"

    @property
    def display_name(self) -> str:
        return "arXiv"

    async def can_handle(self, config: ConvertConfig) -> bool:
        """Check if the source looks like an arXiv paper."""
        source = config.source.strip()

        # Try parsing as arXiv ID or URL
        arxiv_id = self._extract_arxiv_id(source)
        return arxiv_id is not None

    async def extract(self, config: ConvertConfig) -> ParseResult:
        """Extract arXiv paper via ar5iv HTML or arXiv API."""
        source = config.source.strip()
        arxiv_id = self._extract_arxiv_id(source)
        if not arxiv_id:
            raise ValueError(f"Could not extract arXiv ID from: {source}")

        logger.info("Extracting arXiv paper: %s", arxiv_id)

        # Get metadata from arXiv API (always available)
        metadata = await self._fetch_metadata(arxiv_id)

        # Try ar5iv HTML first
        result = await self._extract_ar5iv(arxiv_id)
        if result and result.markdown_body.strip():
            quality = self._assess_content_quality(result.markdown_body)
            logger.info(
                "ar5iv quality: headings=%d, paras=%d, density=%.2f, "
                "has_errors=%s, score=%.1f",
                quality["heading_count"],
                quality["para_count"],
                quality["content_density"],
                quality["has_error_indicators"],
                quality["score"],
            )

            if quality["score"] >= 0.4:  # Acceptable quality threshold
                result.metadata = metadata
                result.source_type = "arxiv"
                result.raw_source_url = f"https://arxiv.org/abs/{arxiv_id}"
                return result
            else:
                logger.info(
                    "ar5iv content quality too low (score=%.1f), "
                    "trying MinerU PDF fallback",
                    quality["score"],
                )

        # Tier 2 fallback: MinerU on arXiv PDF
        mineru_result = await self._extract_pdf_via_mineru(arxiv_id, config)
        if mineru_result:
            mineru_result.metadata = metadata
            mineru_result.source_type = "arxiv"
            mineru_result.raw_source_url = f"https://arxiv.org/abs/{arxiv_id}"
            return mineru_result

        # Tier 3 fallback: arXiv abstract page
        logger.info("Falling back to arXiv abstract page for %s", arxiv_id)
        result = await self._extract_arxiv_abstract(arxiv_id)
        result.metadata = metadata
        result.source_type = "arxiv"
        result.raw_source_url = f"https://arxiv.org/abs/{arxiv_id}"
        return result

    # ------------------------------------------------------------------
    # Content quality assessment
    # ------------------------------------------------------------------

    @staticmethod
    def _assess_content_quality(markdown: str) -> dict:
        """Assess ar5iv extraction quality with multiple structural metrics.

        Returns a dict with:
            heading_count: number of distinct section headings
            para_count: number of substantial paragraphs (>100 chars)
            content_density: ratio of content chars to total chars
            has_error_indicators: ar5iv error page detected
            score: composite quality score (0.0 = bad, 1.0 = excellent)
        """
        lines = markdown.split("\n")

        # Count distinct headings (any level)
        headings = [l.strip() for l in lines if re.match(r"^#{1,6}\s+\S", l)]
        heading_count = len(headings)

        # Count substantial paragraphs (non-heading lines with real content)
        paras = [
            l.strip()
            for l in lines
            if len(l.strip()) > 80
            and not l.strip().startswith("#")
            and not l.strip().startswith("!")
            and not l.strip().startswith("|")
            and not l.strip().startswith("*")
        ]
        para_count = len(paras)

        # Content density: ratio of meaningful chars to total
        total_chars = max(len(markdown), 1)
        content_chars = sum(len(p) for p in paras) + sum(len(h) for h in headings)
        density = min(content_chars / total_chars, 1.0)

        # Error indicators: ar5iv error pages have specific patterns
        error_phrases = [
            "Conversion report",
            "Report an issue",
            "View original on arXiv",
            "Feeling lucky?",
            "See pages",
            "ar5iv",
        ]
        error_lines = sum(
            1
            for l in lines
            if any(phrase.lower() in l.lower() for phrase in error_phrases)
        )
        has_errors = error_lines > 2  # A few matches could be coincidental

        # Composite score (weighted)
        # - A good paper typically has 5+ headings and 20+ paragraphs
        heading_score = min(heading_count / 5.0, 1.0)
        para_score = min(para_count / 20.0, 1.0)
        density_score = density  # Already 0-1
        error_penalty = 0.0 if not has_errors else -0.5

        score = max(
            0.0,
            (heading_score * 0.35)
            + (para_score * 0.35)
            + (density_score * 0.30)
            + error_penalty,
        )

        return {
            "heading_count": heading_count,
            "para_count": para_count,
            "content_density": round(density, 2),
            "has_error_indicators": has_errors,
            "score": round(score, 2),
        }

    # ------------------------------------------------------------------
    # arXiv ID extraction
    # ------------------------------------------------------------------

    def _extract_arxiv_id(self, source: str) -> str | None:
        """Extract arXiv paper ID from various input formats."""
        source = source.strip()

        # Strip common prefixes: arxiv: / arxiv:// / arXiv:
        source = re.sub(r"^arxiv:(?://)?\s*", "", source, flags=re.IGNORECASE)

        for pattern in ARXIV_ID_PATTERNS:
            match = pattern.search(source)
            if match:
                # Remove version suffix (v1, v2, etc.)
                raw_id = match.group(1)
                # Strip version suffix
                return re.sub(r"v\d+$", "", raw_id)

        return None

    # ------------------------------------------------------------------
    # Metadata from arXiv API
    # ------------------------------------------------------------------

    async def _fetch_metadata(self, arxiv_id: str) -> PaperMetadata:
        """Fetch paper metadata from the arXiv API."""
        params = {
            "id_list": arxiv_id,
            "max_results": 1,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(ARXIV_API, params=params)
            resp.raise_for_status()

            # Parse Atom XML response
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }
            root = ET.fromstring(resp.text)
            entry = root.find(".//atom:entry", ns)
            if entry is None:
                return PaperMetadata(arxiv_id=arxiv_id)

            title = self._elem_text(entry, "atom:title", ns)
            abstract = self._elem_text(entry, "atom:summary", ns)
            doi = self._elem_text(entry, "arxiv:doi", ns) or ""
            journal = self._elem_text(entry, "arxiv:journal_ref", ns) or ""

            # Extract authors
            authors = []
            for author_elem in entry.findall("atom:author", ns):
                name = self._elem_text(author_elem, "atom:name", ns)
                if name:
                    authors.append(name)

            # Extract year from published date
            published = self._elem_text(entry, "atom:published", ns) or ""
            year = published[:4] if published else ""

            return PaperMetadata(
                title=title.strip().replace("\n", " ") if title else "",
                authors=authors,
                abstract=abstract.strip() if abstract else "",
                doi=doi,
                arxiv_id=arxiv_id,
                journal=journal,
                year=year,
                url=f"https://arxiv.org/abs/{arxiv_id}",
            )

    @staticmethod
    def _elem_text(
        parent: ET.Element, tag: str, ns: dict
    ) -> str | None:
        """Get text content of an XML element."""
        elem = parent.find(tag, ns)
        return elem.text if elem is not None else None

    # ------------------------------------------------------------------
    # ar5iv HTML extraction — structural walker
    # ------------------------------------------------------------------

    async def _extract_ar5iv(self, arxiv_id: str) -> ParseResult | None:
        """Extract paper content from ar5iv HTML rendering.

        Walks the ar5iv article DOM structurally rather than recursively,
        producing clean Markdown with preserved sections, math, figures,
        and tables.
        """
        url = f"{AR5IV_BASE}/{arxiv_id}"
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "paper2md/1.0"})
                if resp.status_code != 200:
                    logger.debug("ar5iv returned %d for %s", resp.status_code, arxiv_id)
                    return None

                soup = BeautifulSoup(resp.text, "lxml")
                article = soup.find("article")
                if not article:
                    return None

                images: list[ImageAsset] = []
                md_parts: list[str] = []
                seen_section_headings: set[str] = set()

                # Walk article's direct children sequentially
                for child in article.children:
                    if not isinstance(child, Tag):
                        continue

                    tag_name = child.name
                    classes = self._classes(child)

                    # --- Document title (use metadata instead) ---
                    if "ltx_title_document" in classes:
                        continue

                    # --- Authors block (use metadata instead) ---
                    elif "ltx_authors" in classes:
                        continue

                    # --- Abstract ---
                    elif "ltx_abstract" in classes:
                        md_parts.append("## Abstract\n")
                        md_parts.append(self._process_content_blocks(child, images))
                        md_parts.append("")

                    # --- Main sections ---
                    elif tag_name == "section" and "ltx_section" in classes:
                        heading = self._find_section_heading(child)
                        norm = self._normalize_heading(heading)
                        if norm and norm in seen_section_headings:
                            continue
                        if norm:
                            seen_section_headings.add(norm)
                        result = self._process_section(child, images, level=2)
                        if result:
                            md_parts.append(result)

                    # --- Subsections at top level ---
                    elif tag_name == "section" and "ltx_subsection" in classes:
                        heading = self._find_section_heading(child)
                        norm = self._normalize_heading(heading)
                        if norm and norm in seen_section_headings:
                            continue
                        if norm:
                            seen_section_headings.add(norm)
                        result = self._process_section(child, images, level=3)
                        if result:
                            md_parts.append(result)

                    # --- Bibliography / references (skip) ---
                    elif "ltx_bibliography" in classes:
                        continue

                    # --- Pagination breaks (skip) ---
                    elif "ltx_pagination" in classes:
                        continue

                    # --- Top-level license paragraph (skip) ---
                    elif "ltx_para" in classes:
                        # Only skip if it's the license text at the top
                        text = child.get_text(strip=True)[:100].lower()
                        if "permission" in text or "granted" in text:
                            continue
                        md_parts.append(self._process_paragraph(child))

                return ParseResult(
                    markdown_body="\n".join(p for p in md_parts if p),
                    images=images,
                )

        except (httpx.HTTPError, Exception) as e:
            logger.warning("ar5iv extraction failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Structural content processors
    # ------------------------------------------------------------------

    def _process_section(
        self, elem: Tag, images: list[ImageAsset], level: int
    ) -> str:
        """Process a <section class=\"ltx_section\"> or ltx_subsection.

        Extracts the heading, then walks children for content blocks.
        """
        heading = self._find_section_heading(elem)
        parts: list[str] = []

        if heading:
            prefix = "#" * level
            parts.append(f"\n{prefix} {heading}\n")

        for child in elem.children:
            if not isinstance(child, Tag):
                continue

            tag = child.name
            classes = self._classes(child)

            # Skip the heading element itself (already extracted)
            if tag in ("h2", "h3", "h4") and "ltx_title" in classes:
                continue

            # Nested subsection
            if tag == "section" and "ltx_subsection" in classes:
                result = self._process_section(child, images, level + 1)
                if result:
                    parts.append(result)

            # Sub-subsection
            elif tag == "section" and "ltx_subsubsection" in classes:
                result = self._process_section(child, images, level + 2)
                if result:
                    parts.append(result)

            # Paragraph
            elif "ltx_para" in classes or tag == "p":
                text = self._process_paragraph(child)
                if text:
                    parts.append(text)

            # Figure (image or table type)
            elif "ltx_figure" in classes or tag == "figure":
                if "ltx_table" in classes:
                    result = self._process_table(child)
                elif child.find("img"):
                    result = self._process_figure(child, images)
                else:
                    result = ""
                if result:
                    parts.append(result)

            # Table
            elif tag == "table" or "ltx_tabular" in classes:
                result = self._process_table(child)
                if result:
                    parts.append(result)

            # Math block
            elif tag == "math":
                result = self._process_math(child)
                if result:
                    parts.append(result)

            # Lists
            elif tag == "ul" or "ltx_itemize" in classes:
                result = self._process_list(child, ordered=False)
                if result:
                    parts.append(result)
            elif tag == "ol" or "ltx_enumerate" in classes:
                result = self._process_list(child, ordered=True)
                if result:
                    parts.append(result)

            # Theorem/proof/definition environments
            elif "ltx_theorem" in classes or "ltx_proof" in classes:
                thm_heading = self._find_section_heading(child)
                if thm_heading:
                    parts.append(f"\n**{thm_heading}**\n")
                parts.append(self._process_content_blocks(child, images))

            # Generic div: recurse into its children
            elif tag == "div":
                result = self._process_content_blocks(child, images)
                if result:
                    parts.append(result)

            # Anything else with text content
            else:
                text = child.get_text(strip=True)
                if text and len(text) > 3:
                    parts.append(self._inline_to_md(child))

        return "\n".join(p for p in parts if p)

    def _process_content_blocks(
        self, elem: Tag, images: list[ImageAsset]
    ) -> str:
        """Process children of a container element as content blocks."""
        parts: list[str] = []
        for child in elem.children:
            if not isinstance(child, Tag):
                continue
            tag = child.name
            classes = self._classes(child)

            if "ltx_para" in classes or tag == "p":
                text = self._process_paragraph(child)
                if text:
                    parts.append(f"\n{text}\n")
            elif "ltx_title" in classes:
                # Skip section titles inside content (already handled by headings)
                continue
            elif "ltx_figure" in classes or tag == "figure":
                if "ltx_table" in classes:
                    result = self._process_table(child)
                elif child.find("img"):
                    result = self._process_figure(child, images)
                else:
                    result = ""
                if result:
                    parts.append(result)
            elif tag == "math":
                result = self._process_math(child)
                if result:
                    parts.append(result)
            elif tag in ("table",) or "ltx_tabular" in classes:
                result = self._process_table(child)
                if result:
                    parts.append(result)
            elif tag in ("ul", "ol"):
                result = self._process_list(child, ordered=(tag == "ol"))
                if result:
                    parts.append(result)
            elif tag == "div":
                inner = self._process_content_blocks(child, images)
                if inner:
                    parts.append(inner)
            else:
                text = self._inline_to_md(child).strip()
                if text:
                    parts.append(text)

        return "\n".join(p for p in parts if p)

    def _process_paragraph(self, elem: Tag) -> str:
        """Convert a paragraph element to clean markdown text."""
        text = self._inline_to_md(elem).strip()
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        return text if text else ""

    def _process_figure(
        self, elem: Tag, images: list[ImageAsset]
    ) -> str:
        """Convert a figure element to markdown, downloading the image."""
        img = elem.find("img")
        if not img:
            return ""

        src = img.get("src", "")
        alt = img.get("alt", "Figure")

        # Resolve relative URL
        if src.startswith("/"):
            src = f"https://ar5iv.labs.arxiv.org{src}"

        # Extract caption first
        caption = ""
        figcaption = elem.find("figcaption")
        if figcaption:
            caption = figcaption.get_text(strip=True)

        # Use caption as alt text if img alt is uninformative
        if alt in ("Refer to caption", "Figure", "image", ""):
            if caption:
                alt = re.sub(r"^Figure\s+\d+\s*[:.]?\s*", "", caption)

        # Download image
        local_src = src
        if src.startswith("http"):
            try:
                import httpx
                import hashlib

                img_resp = httpx.get(src, timeout=15, follow_redirects=True)
                if img_resp.status_code == 200:
                    content_hash = hashlib.sha256(img_resp.content).hexdigest()[:12]
                    ext = src.rsplit(".", 1)[-1].split("?")[0] or "png"
                    filename = f"{content_hash}.{ext}"
                    local_src = f"images/{filename}"
                    images.append(
                        ImageAsset(
                            filename=filename,
                            data=img_resp.content,
                            caption=caption,
                            alt_text=alt,
                        )
                    )
            except Exception:
                pass

        md = f"![{alt}]({local_src})"
        if caption:
            caption_clean = re.sub(r"^Figure\s+\d+\s*[:.]?\s*", "", caption)
            md += f"\n*{caption_clean}*"

        return f"\n{md}\n"

    def _process_table(self, elem: Tag) -> str:
        """Convert HTML table to Markdown table.

        Handles tables both as direct <table> elements and inside
        <figure class=\"ltx_table\"> wrappers.
        """
        # Find the actual <table> element
        table = elem if elem.name == "table" else elem.find("table")
        if table is None:
            return ""

        rows = table.find_all("tr")
        if not rows:
            return ""

        md_rows: list[str] = []
        col_count = max(
            len(row.find_all(["th", "td"])) for row in rows
        )

        for i, row in enumerate(rows):
            cells = row.find_all(["th", "td"])
            cell_texts: list[str] = []
            for cell in cells:
                ct = cell.get_text(strip=True).replace("|", "\\|").replace("\n", " ")
                cell_texts.append(ct)
            # Pad to column count
            while len(cell_texts) < col_count:
                cell_texts.append("")

            md_rows.append("| " + " | ".join(cell_texts) + " |")
            if i == 0:
                md_rows.append("|" + "|".join([" --- " for _ in range(col_count)]) + "|")

        return "\n" + "\n".join(md_rows) + "\n"

    def _process_math(self, elem: Tag) -> str:
        """Convert MathML to LaTeX."""
        # Prefer alttext (ar5iv provides this)
        alttext = elem.get("alttext", "")
        if alttext and alttext.strip():
            latex = alttext.strip()
            return f"${latex}$"

        # Try LaTeX annotation
        annotation = elem.find(
            'semantics/annotation[@encoding="application/x-tex"]'
        )
        if annotation and annotation.string:
            latex = annotation.string.strip()
            return f"${latex}$"

        # Fallback
        text = elem.get_text(strip=True)
        return f"${text}$" if text else ""

    def _process_list(self, elem: Tag, ordered: bool) -> str:
        """Convert HTML list to Markdown list."""
        items = elem.find_all("li", recursive=False)
        if not items:
            items = elem.find_all(class_="ltx_item")

        md_items: list[str] = []
        for i, item in enumerate(items):
            marker = f"{i + 1}." if ordered else "-"
            text = self._inline_to_md(item).strip()
            md_items.append(f"{marker} {text}")

        return "\n" + "\n".join(md_items) + "\n"

    def _find_section_heading(self, elem: Tag) -> str:
        """Extract heading text from a section element.

        Finds h2/h3/h4 with ltx_title class, skips ltx_tag (section number)
        and returns clean heading text.
        """
        for tag_name in ("h2", "h3", "h4", "h1"):
            heading = elem.find(tag_name, class_=re.compile("ltx_title"))
            if heading:
                # Collect text excluding ltx_tag spans
                parts = []
                for child in heading.children:
                    if isinstance(child, Tag) and "ltx_tag" in self._classes(child):
                        continue
                    if isinstance(child, str):
                        parts.append(child.strip())
                    elif isinstance(child, Tag):
                        parts.append(child.get_text(strip=True))
                text = " ".join(p for p in parts if p).strip()
                # Remove leading numbers if present (section numbering)
                return text

        return ""

    def _inline_to_md(self, elem: Tag) -> str:
        """Convert inline HTML elements to markdown text."""
        result = []
        for child in elem.children:
            if isinstance(child, str):
                result.append(child)
            elif isinstance(child, Tag):
                tag = child.name
                classes = self._classes(child)

                if tag == "math":
                    result.append(self._process_math(child))
                elif "ltx_cite" in classes or tag == "cite":
                    cite_text = child.get_text(strip=True).strip("[]")
                    result.append(f"[{cite_text}]")
                elif tag in ("em", "i"):
                    result.append(f"*{self._inline_to_md(child)}*")
                elif tag in ("strong", "b"):
                    result.append(f"**{self._inline_to_md(child)}**")
                elif tag == "a":
                    href = child.get("href", "")
                    text = self._inline_to_md(child)
                    result.append(f"[{text}]({href})" if href else text)
                elif tag == "sub":
                    result.append(f"_{self._inline_to_md(child)}_")
                elif tag == "sup":
                    result.append(f"^{self._inline_to_md(child)}^")
                elif tag in ("span",):
                    result.append(self._inline_to_md(child))
                elif tag == "br":
                    result.append("\n")
                elif tag == "img":
                    alt = child.get("alt", "")
                    src = child.get("src", "")
                    if src.startswith("/"):
                        src = f"https://ar5iv.labs.arxiv.org{src}"
                    result.append(f"![{alt}]({src})")
                elif tag in ("p", "div", "section"):
                    # Recurse into block containers for inline content
                    result.append(self._inline_to_md(child))
                else:
                    result.append(child.get_text())
        return "".join(result)

    # ------------------------------------------------------------------
    # MinerU PDF fallback (ar5iv quality too low → download + parse PDF)
    # ------------------------------------------------------------------

    async def _extract_pdf_via_mineru(
        self, arxiv_id: str, config: ConvertConfig
    ) -> ParseResult | None:
        """Download arXiv PDF → delegate to PdfLocalExtractor for MinerU parsing."""
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        logger.info("Downloading arXiv PDF for MinerU fallback: %s", pdf_url)

        import tempfile
        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as http:
                resp = await http.get(pdf_url)
                if resp.status_code != 200:
                    logger.warning("arXiv PDF download failed: HTTP %d", resp.status_code)
                    return None

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(resp.content)
                    tmp_path = Path(tmp.name)

            logger.info("Downloaded PDF: %d bytes, delegating to PdfLocalExtractor",
                        len(resp.content))

            # Reuse existing PdfLocalExtractor logic (local-first, cloud-fallback)
            from .pdf import PdfLocalExtractor
            extractor = PdfLocalExtractor()
            # Override source to point to our temp file
            pdf_config = ConvertConfig(
                source=str(tmp_path),
                output_dir=config.output_dir,
                mineru_api_key=config.mineru_api_key,
                mineru_use_local=config.mineru_use_local,
                mineru_backend=config.mineru_backend,
                model_version=config.model_version,
                language=config.language,
                is_ocr=config.is_ocr,
                enable_table=config.enable_table,
                enable_formula=config.enable_formula,
                timeout=config.timeout,
            )
            result = await extractor.extract(pdf_config)
            tmp_path.unlink(missing_ok=True)

            logger.info("MinerU fallback succeeded: %d chars, %d images",
                        len(result.markdown_body), len(result.images))
            return result

        except Exception as e:
            logger.warning("MinerU fallback error: %s", e)
            return None

    # ------------------------------------------------------------------
    # arXiv abstract page fallback (last resort)
    # ------------------------------------------------------------------

    async def _extract_arxiv_abstract(self, arxiv_id: str) -> ParseResult:
        """Fallback: extract from arXiv abstract page HTML."""
        url = f"https://arxiv.org/abs/{arxiv_id}"
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "paper2md/1.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            md_parts = []
            abstract_block = soup.find("blockquote", class_="abstract")
            if abstract_block:
                abstract_text = abstract_block.get_text(" ", strip=True)
                abstract_text = re.sub(r"^Abstract:\s*", "", abstract_text)
                md_parts.append(f"## Abstract\n\n{abstract_text}\n")

            return ParseResult(markdown_body="\n".join(md_parts))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _classes(elem: Tag) -> list[str]:
        """Get class list as strings."""
        cls = elem.get("class", [])
        if isinstance(cls, str):
            return [cls]
        return list(cls)

    @staticmethod
    def _normalize_heading(heading: str) -> str:
        """Normalize heading for deduplication.

        Strips section numbers (e.g. '1', '2.1', 'III'), extra whitespace,
        and lowercases.  '1 Introduction' and 'Introduction' become the same key.
        """
        h = heading.strip().lower()
        # Strip leading section numbers: "1 ", "2.1 ", "iii ", etc.
        h = re.sub(r"^[0-9ivxlc\.]+\s+", "", h)
        return re.sub(r"\s+", " ", h)
