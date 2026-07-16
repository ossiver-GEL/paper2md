"""Nature.com paper source extractor.

Parses Nature article HTML pages (nature.com/articles/...).
Nature provides structured HTML with semantic markup:
- <article> wrapper
- <h1> for title
- <section data-title="..."> for sections (abstract, main, methods, etc.)
- Figures in <figure> with <img> + <figcaption>
- Tables as <table> elements
- Math in MathML or images with alt text
- CC license info in "Rights and permissions" section
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup, Tag

from ..models import ConvertConfig, ImageAsset, PaperMetadata, ParseResult
from .base import BaseExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Nature URL / DOI patterns
NATURE_DOI_RE = re.compile(
    r"(?:nature\.com/articles/|doi\.org/|^)(10\.1038/[a-zA-Z0-9\-./]+)"
)
NATURE_URL_RE = re.compile(
    r"https?://(?:www\.)?nature\.com/articles/[a-zA-Z0-9\-]+"
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36 "
    "paper2md/1.0"
)

# Sections considered "main content" (rendered first)
MAIN_SECTIONS = {
    "abstract", "main", "results", "discussion", "methods",
    "background", "introduction", "conclusion",
}

# Sections placed at end (supplementary material)
SUPPLEMENTARY_SECTIONS = {
    "data availability", "code availability", "references",
    "acknowledgements", "author information", "author contributions",
    "ethics declarations", "competing interests", "additional information",
    "peer review", "rights and permissions", "about this article",
    "change history", "funding", "extended data",
    "supplementary information", "source data", "footnotes",
    "corresponding author", "editorial summary",
}

# Transform thumbnail URL → full-size image
FULL_IMAGE_RE = re.compile(
    r"(media\.springernature\.com)/(?:lw\d+|full)/(.*\.(?:png|jpg|jpeg|gif|webp))",
    re.IGNORECASE,
)


class NatureExtractor(BaseExtractor):
    """Extract papers from Nature.com article pages."""

    @property
    def source_type(self) -> str:
        return "nature"

    @property
    def display_name(self) -> str:
        return "Nature"

    async def can_handle(self, config: ConvertConfig) -> bool:
        """Check if the source is a Nature article URL."""
        source = config.source.strip()
        return bool(NATURE_URL_RE.search(source)) or bool(
            NATURE_DOI_RE.search(source)
        )

    async def extract(self, config: ConvertConfig) -> ParseResult:
        """Extract Nature article from HTML page.

        Detects CC license, downloads full-size figures, processes
        all sections (main + supplementary), and produces clean Markdown.
        """
        source = config.source.strip()

        # Resolve to full Nature URL if given a DOI
        url = source
        doi_match = NATURE_DOI_RE.search(source)
        if doi_match and "nature.com" not in source:
            doi = doi_match.group(1)
            # Nature URLs use only the article suffix (e.g. s43018-025-00991-6),
            # not the full DOI (e.g. 10.1038/s43018-025-00991-6).
            article_id = doi.split("/", 1)[-1] if "/" in doi else doi
            url = f"https://www.nature.com/articles/{article_id}"
        else:
            url = source

        # Strip .pdf suffix — Nature articles have an HTML version at the same URL
        url = re.sub(r"\.pdf$", "", url)

        logger.info("Extracting Nature article: %s", url)

        async with httpx.AsyncClient(
            timeout=60,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "lxml")
            article = soup.find("article")
            body = soup.select_one(".c-article-body")

            if article is None or body is None:
                raise RuntimeError(
                    "Nature article body not found. The page layout may "
                    "have changed or access may be blocked."
                )

            # Check CC license
            license_name, license_url = self._check_cc_license(soup)

            # Extract metadata
            metadata = self._extract_metadata(soup, url, license_name)

            # Collect all article sections
            images: list[ImageAsset] = []
            body_parts: list[str] = []
            supplementary_parts: list[str] = []

            # Title
            title_elem = soup.find("h1", class_=re.compile("c-article-title"))
            if title_elem:
                body_parts.append(f"# {title_elem.get_text(strip=True)}\n")

            # Authors
            authors_list = soup.find("ul", class_=re.compile("c-article-author"))
            if authors_list:
                author_names = [
                    a.get_text(strip=True)
                    for a in authors_list.find_all(
                        "a", attrs={"data-test": "author-name"}
                    )
                ]
                if author_names:
                    body_parts.append(f"*{', '.join(author_names)}*\n")

            # License line
            if license_name:
                cc_line = (
                    f"> **License:** [{license_name}]({license_url}) — "
                    f"this is an open-access article.\n"
                )
                body_parts.append(cc_line)

            # Process sections: first from main body, then from article
            all_sections = self._find_all_sections(soup, body)
            processed_sections: set[str] = set()

            for section in all_sections:
                data_title = self._get_section_title(section).strip().lower()
                if not data_title:
                    continue

                # Detect duplicates (case-insensitive)
                normalized = data_title.lower()
                if normalized in processed_sections:
                    continue
                processed_sections.add(normalized)

                heading_text = self._get_section_heading_text(section)
                section_md = self._process_nature_section(
                    section, images, client
                )

                if not section_md.strip():
                    continue

                # Sort into main or supplementary
                target = (
                    supplementary_parts
                    if normalized in SUPPLEMENTARY_SECTIONS
                    else body_parts
                )

                # Suppress headings that are figure/table captions
                is_fig_heading = heading_text and self._is_figure_heading(
                    heading_text
                )

                if heading_text and not is_fig_heading and normalized != "abstract":
                    target.append(f"\n## {heading_text}\n")
                elif heading_text and normalized == "abstract":
                    target.append("## Abstract\n")

                target.append(section_md)

            # Combine: body first, then supplementary
            body_parts.extend(supplementary_parts)

            # Deduplicate images by filename
            seen_names: set[str] = set()
            unique_images = []
            for img in images:
                if img.filename not in seen_names:
                    seen_names.add(img.filename)
                    unique_images.append(img)

            return ParseResult(
                metadata=metadata,
                markdown_body="\n".join(body_parts),
                images=unique_images,
                source_type="nature",
                raw_source_url=url,
            )

    # ------------------------------------------------------------------
    # CC License detection
    # ------------------------------------------------------------------

    def _check_cc_license(
        self, soup: BeautifulSoup
    ) -> tuple[str | None, str | None]:
        """Check if the article has a Creative Commons license.

        Looks in the 'Rights and permissions' section for CC license links.
        Returns (license_name, license_url) or (None, None).
        """
        rights = soup.select_one(
            'section[data-title="Rights and permissions"]'
        )
        if rights is None:
            return None, None

        for link in rights.find_all("a", href=True):
            href = link["href"]
            if "creativecommons.org/licenses/" in href:
                normalized = href.replace("http://", "https://")
                match = re.search(
                    r"/licenses/([^/]+)/([^/]+)/?", normalized
                )
                if match:
                    name = f"CC {match.group(1).upper()} {match.group(2)}"
                else:
                    name = (
                        link.get_text(strip=True) or "Creative Commons license"
                    )
                return name, normalized

        return None, None

    # ------------------------------------------------------------------
    # Section discovery and processing
    # ------------------------------------------------------------------

    def _find_all_sections(
        self, soup: BeautifulSoup, body: Tag
    ) -> list[Tag]:
        """Find all unique article sections.

        Nature's HTML duplicates content across two DOM structures:
        - <div class=\"c-article-section\"> inside .c-article-body
        - <section data-title=\"...\"> inside <article>

        The <section data-title=\"...\"> elements are the canonical source.
        We use them exclusively to avoid double extraction.
        """
        article = soup.find("article")
        if article is None:
            return []

        sections = article.find_all("section", attrs={"data-title": True})
        if sections:
            return sections

        # Fallback: use c-article-section divs if no data-title sections
        return body.find_all("div", class_=re.compile("c-article-section"))

    def _is_figure_heading(self, text: str) -> bool:
        """Check if a heading looks like a figure/table/supplementary caption.

        Nature wraps figures/tables in sections with data-title like
        'Fig. 1: MIRA workflow.' or 'Supplementary Information (download PDF)'.
        These should not appear as ## section headings.
        """
        text_lower = text.strip().lower()
        # Figure/table patterns
        if re.match(
            r"^(fig\.?|figure|table|extended data)\s*\d+",
            text_lower,
        ):
            return True
        # Supplementary / source data patterns
        if re.match(
            r"^(supplementary|source data|cite this|similar content|"
            r"authors and affiliations|peer review information)",
            text_lower,
        ):
            return True
        # Short headings ending with period (likely figure captions)
        if len(text) < 100 and text.rstrip().endswith("."):
            # Check if it matches known figure caption patterns
            if any(
                kw in text_lower
                for kw in ["workflow", "evaluation of", "diagnostic accuracy",
                           "disease management", "safety and robustness"]
            ):
                return True
        return False

    def _get_section_title(self, section: Tag) -> str:
        """Get the data-title or heading text of a section."""
        # Nature sections have data-title attribute
        data_title = section.get("data-title", "")
        if data_title:
            return str(data_title)

        # Fallback: find heading element
        heading = section.find(["h2", "h3", "h4"])
        if heading:
            return heading.get_text(strip=True)

        # Fallback: use id
        section_id = section.get("id", "")
        return section_id.replace("-", " ").title()

    def _get_section_heading_text(self, section: Tag) -> str:
        """Get clean heading text for markdown output."""
        heading = section.find(["h2", "h3", "h4"])
        if heading:
            return heading.get_text(strip=True)

        data_title = section.get("data-title", "")
        if data_title:
            return str(data_title)

        return ""

    def _process_nature_section(
        self,
        section: Tag,
        images: list[ImageAsset],
        client: httpx.AsyncClient | None = None,
    ) -> str:
        """Process a single Nature article section into Markdown."""
        # Find the content container
        content_div = section.find(
            class_=re.compile("c-article-section__content")
        )
        if content_div is None:
            content_div = section

        return self._section_to_md(content_div, images, client)

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    def _extract_metadata(
        self, soup: BeautifulSoup, url: str, license_name: str | None = None
    ) -> PaperMetadata:
        """Extract structured metadata from Nature article HTML."""
        # Title
        title = ""
        title_elem = soup.find("h1", class_=re.compile("c-article-title"))
        if title_elem:
            title = title_elem.get_text(strip=True)

        # Authors
        authors = []
        author_list = soup.find("ul", class_=re.compile("c-article-author"))
        if author_list:
            author_links = author_list.find_all(
                "a", attrs={"data-test": "author-name"}
            )
            authors = [a.get_text(strip=True) for a in author_links]

        # Abstract
        abstract = ""
        abstract_div = soup.find("div", id=re.compile("Abs"))
        if abstract_div:
            abstract_content = abstract_div.find(
                class_=re.compile("c-article-section__content")
            )
            if abstract_content:
                abstract = abstract_content.get_text(" ", strip=True)

        # DOI
        doi = ""
        doi_meta = soup.find("meta", attrs={"name": "citation_doi"})
        if doi_meta:
            doi = doi_meta.get("content", "")
        else:
            doi_span = soup.find(
                "span", class_=re.compile("c-bibliographic-information__doi")
            )
            if doi_span:
                doi_match = re.search(
                    r"10\.\d{4,}/[^\s]+", doi_span.get_text()
                )
                if doi_match:
                    doi = doi_match.group(0)

        # Journal
        journal = "Nature"
        journal_meta = soup.find("meta", attrs={"name": "citation_journal_title"})
        if journal_meta:
            journal = journal_meta.get("content", "Nature")

        # Year
        year = ""
        date_meta = soup.find("meta", attrs={"name": "citation_publication_date"})
        if date_meta:
            year = date_meta.get("content", "")[:4]
        else:
            date_elem = soup.find("time")
            if date_elem:
                datetime_val = date_elem.get("datetime", "")
                year = datetime_val[:4]

        # Append license to abstract if available
        if license_name:
            abstract = (
                f"[{license_name}] {abstract}"
                if abstract
                else f"[{license_name}]"
            )

        return PaperMetadata(
            title=title,
            authors=authors,
            abstract=abstract,
            doi=doi,
            journal=journal,
            year=year,
            url=url,
        )

    # ------------------------------------------------------------------
    # Content extraction helpers
    # ------------------------------------------------------------------

    def _section_to_md(
        self,
        elem: Tag,
        images: list[ImageAsset],
        client: httpx.AsyncClient | None = None,
        depth: int = 0,
    ) -> str:
        """Convert a Nature article section/div to Markdown."""
        parts = []

        for child in elem.children:
            if isinstance(child, str):
                text = child.strip()
                if text:
                    parts.append(text)
                continue

            if not isinstance(child, Tag):
                continue

            tag_name = child.name
            classes = self._classes(child)

            # Skip hidden / navigation / non-content elements
            if "c-article__supplementary" in classes:
                continue
            if "c-article-references" in classes:
                continue

            # --- Headings ---
            if tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(tag_name[1])
                text = child.get_text(strip=True)
                parts.append(f"\n{'#' * level} {text}\n")

            # --- Paragraphs ---
            elif tag_name == "p":
                text = self._inline_to_md(child)
                if text.strip():
                    parts.append(f"\n{text}\n")

            # --- Figures ---
            elif tag_name == "figure":
                result = self._figure_to_md(child, images, client)
                if result:
                    parts.append(result)

            # --- Tables ---
            elif tag_name == "table":
                parts.append(self._table_to_md(child))

            # --- Table wrappers (Nature-specific) ---
            elif "c-article-table" in " ".join(classes):
                table = child.find("table")
                if table:
                    parts.append(self._table_to_md(table))

            # --- Lists ---
            elif tag_name == "ul":
                parts.append(self._list_to_md(child, ordered=False))
            elif tag_name == "ol":
                parts.append(self._list_to_md(child, ordered=True))

            # --- Math (MathML) ---
            elif tag_name == "math":
                parts.append(self._mathml_to_latex(child))

            # --- Equations (Nature uses div.c-article-equation) ---
            elif "c-article-equation" in " ".join(classes):
                tex = child.select_one(".mathjax-tex")
                if tex:
                    latex = tex.get_text(strip=True)
                    number_el = child.select_one(
                        ".c-article-equation__number"
                    )
                    number = number_el.get_text(strip=True) if number_el else ""
                    if number:
                        parts.append(
                            f"\n$$\n{latex}\n$$\n\n*Equation {number}*\n"
                        )
                    else:
                        parts.append(f"\n$$\n{latex}\n$$\n")
                else:
                    img = child.find("img")
                    if img:
                        alt = img.get("alt", "")
                        if alt:
                            alt_clean = alt.replace("$$", "").strip()
                            parts.append(f"\n$$\n{alt_clean}\n$$\n")

            # --- Blockquotes ---
            elif tag_name == "blockquote":
                text = self._inline_to_md(child)
                lines = text.strip().split("\n")
                quoted = "\n".join(f"> {line}" for line in lines)
                parts.append(f"\n{quoted}\n")

            # --- Divs: recurse ---
            elif tag_name == "div":
                skip_classes = [
                    "c-article-header",
                    "c-article-tools",
                    "c-article__share",
                    "c-article__metrics",
                ]
                if any(s in " ".join(classes) for s in skip_classes):
                    continue
                parts.append(
                    self._section_to_md(child, images, client, depth)
                )

            # --- Spans ---
            elif tag_name == "span":
                parts.append(self._inline_to_md(child))

            # --- Links ---
            elif tag_name == "a":
                text = self._inline_to_md(child)
                href = child.get("href", "")
                if href and text.strip():
                    full_href = href
                    if href.startswith("/"):
                        full_href = f"https://www.nature.com{href}"
                    parts.append(f"[{text}]({full_href})")
                else:
                    parts.append(text)

            # --- Code blocks ---
            elif tag_name in ("pre", "code"):
                lang = ""
                if tag_name == "code":
                    classes_str = " ".join(child.get("class", []))
                    lang_match = re.search(r"language-(\w+)", classes_str)
                    if lang_match:
                        lang = lang_match.group(1)
                code_text = child.get_text()
                parts.append(f"\n```{lang}\n{code_text}\n```\n")

            # --- Other block elements ---
            elif tag_name in ("section", "article", "main"):
                parts.append(
                    self._section_to_md(child, images, client, depth + 1)
                )

            else:
                text = child.get_text(strip=True)
                if text:
                    parts.append(text)

        return "\n".join(p for p in parts if p)

    def _inline_to_md(self, elem: Tag) -> str:
        """Convert inline HTML elements to Markdown text."""
        result = []
        for child in elem.children:
            if isinstance(child, str):
                result.append(child)
            elif isinstance(child, Tag):
                tag = child.name
                if tag == "math":
                    result.append(self._mathml_to_latex(child))
                elif tag in ("em", "i", "italic"):
                    result.append(f"*{self._inline_to_md(child)}*")
                elif tag in ("strong", "b", "bold"):
                    result.append(f"**{self._inline_to_md(child)}**")
                elif tag == "a":
                    href = child.get("href", "")
                    text = self._inline_to_md(child)
                    if href:
                        result.append(f"[{text}]({href})")
                    else:
                        result.append(text)
                elif tag == "sub":
                    result.append(f"_{self._inline_to_md(child)}_")
                elif tag == "sup":
                    result.append(f"^{self._inline_to_md(child)}^")
                elif tag == "code":
                    result.append(f"`{child.get_text()}`")
                elif tag == "img":
                    alt = child.get("alt", "image")
                    src = child.get("src", "")
                    result.append(f"![{alt}]({src})")
                elif tag == "span":
                    result.append(self._inline_to_md(child))
                elif tag == "br":
                    result.append("\n")
                else:
                    result.append(child.get_text())
        return "".join(result)

    # ------------------------------------------------------------------
    # Image downloading
    # ------------------------------------------------------------------

    def _full_size_image_url(self, url: str) -> str:
        """Transform a Nature thumbnail URL to full-size.

        E.g. media.springernature.com/lw685/... → .../full/...
        """
        absolute = urljoin("https://www.nature.com", url)
        absolute = absolute.split("?", 1)[0]
        return FULL_IMAGE_RE.sub(r"\1/full/\2", absolute)

    async def _download_nature_image(
        self,
        img_url: str,
        client: httpx.AsyncClient | None,
    ) -> tuple[bytes, str] | None:
        """Download an image from Nature, trying full-size first.

        Returns (image_data, resolved_url) or None.
        """
        if client is None:
            return None

        full_url = self._full_size_image_url(img_url)
        try:
            resp = await client.get(full_url)
            if resp.status_code == 200:
                return resp.content, full_url
        except Exception:
            pass

        # Fall back to original URL
        try:
            original = urljoin("https://www.nature.com", img_url)
            resp = await client.get(original)
            if resp.status_code == 200:
                return resp.content, original
        except Exception:
            pass

        return None

    def _figure_to_md(
        self,
        elem: Tag,
        images: list[ImageAsset],
        client: httpx.AsyncClient | None = None,
    ) -> str:
        """Convert a Nature figure to Markdown, downloading full-size image."""
        img = elem.find("img")
        if not img:
            return ""

        src = img.get("src", "")
        alt = img.get("alt", "Figure")

        # Resolve URL
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            src = f"https://www.nature.com{src}"

        # Extract caption parts
        caption_title = ""
        caption_body = ""

        # Nature figure captions: title in data-test="figure-caption-text"
        title_node = elem.select_one('[data-test="figure-caption-text"]')
        if title_node:
            caption_title = title_node.get_text(strip=True)
            # Remove "Fig. X: " or "Fig. X " prefix
            caption_title = re.sub(
                r"^Fig\.?\s*\d+\s*:?\s*", "", caption_title
            )

        # Bottom caption
        bottom_caption = elem.select_one('[data-test="bottom-caption"]')
        if bottom_caption:
            caption_body = bottom_caption.get_text(" ", strip=True)

        # Fallback to figcaption
        if not caption_title and not caption_body:
            figcaption = elem.find("figcaption")
            if figcaption:
                caption_title = figcaption.get_text(strip=True)
            else:
                p = elem.find("p")
                if p:
                    caption_title = p.get_text(strip=True)

        # Download image
        local_src = src  # Default: keep remote URL
        if client:
            result = None
            try:
                full_url = self._full_size_image_url(src)

                sync_client = httpx.Client(timeout=30, follow_redirects=True)
                try:
                    resp = sync_client.get(full_url)
                    if resp.status_code == 200:
                        result = (resp.content, full_url)
                except Exception:
                    try:
                        resp = sync_client.get(src)
                        if resp.status_code == 200:
                            result = (resp.content, src)
                    except Exception:
                        pass
                finally:
                    sync_client.close()
            except Exception:
                pass

            if result:
                img_data, _ = result
                import hashlib

                content_hash = hashlib.sha256(img_data).hexdigest()[:12]
                ext = src.rsplit(".", 1)[-1].split("?")[0] or "png"
                if ext.lower() not in ("png", "jpg", "jpeg", "gif", "webp"):
                    ext = "png"
                filename = f"{content_hash}.{ext}"
                local_src = f"images/{filename}"
                images.append(
                    ImageAsset(
                        filename=filename,
                        data=img_data,
                        caption=caption_body or caption_title,
                        alt_text=alt,
                    )
                )

        # Build Markdown
        md = f"![{alt}]({local_src})"
        if caption_title:
            md += f"\n\n**{caption_title}**"
        if caption_body:
            md += f"\n\n*{caption_body}*"

        return f"\n{md}\n"

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

    def _table_to_md(self, elem: Tag) -> str:
        """Convert HTML table to Markdown."""
        rows = elem.find_all("tr")
        if not rows:
            return ""

        md_rows = []
        for i, row in enumerate(rows):
            cells = row.find_all(["th", "td"])
            cell_texts = [
                cell.get_text(strip=True).replace("|", "\\|")
                .replace("\n", " ")
                for cell in cells
            ]
            if not cell_texts:
                continue
            # Pad rows to same column count
            while len(cell_texts) < max(
                len(r.find_all(["th", "td"])) for r in rows
            ):
                cell_texts.append("")

            md_rows.append("| " + " | ".join(cell_texts) + " |")
            if i == 0:
                md_rows.append(
                    "|" + "|".join([" --- " for _ in cell_texts]) + "|"
                )

        return "\n" + "\n".join(md_rows) + "\n"

    def _list_to_md(self, elem: Tag, ordered: bool) -> str:
        """Convert HTML list to Markdown."""
        items = elem.find_all("li", recursive=False)
        md_items = []
        for i, item in enumerate(items):
            marker = f"{i + 1}." if ordered else "-"
            text = self._inline_to_md(item).strip()
            md_items.append(f"{marker} {text}")
        return "\n" + "\n".join(md_items) + "\n"

    def _mathml_to_latex(self, elem: Tag) -> str:
        """Convert MathML to LaTeX."""
        # Check for LaTeX annotation
        annotation = elem.find(
            'semantics/annotation[@encoding="application/x-tex"]'
        )
        if annotation and annotation.string:
            latex = annotation.string.strip()
            display = elem.get("display", "inline")
            if display == "block":
                return f"\n$$\n{latex}\n$$\n"
            return f"${latex}$"

        # Alttext fallback
        alttext = elem.get("alttext", "")
        if alttext:
            return f"${alttext}$"

        return elem.get_text(strip=True)
