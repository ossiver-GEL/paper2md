"""JATS XML → Markdown converter.

Converts NLM Journal Archiving and Interchange Tag Suite (JATS) XML — the
standard format used by PubMed Central, Europe PMC, and many journal
platforms — into clean Markdown with YAML metadata.

Supports:
- Full JATS article metadata (title, authors, affiliations, DOI, abstract)
- Hierarchical sections with heading levels
- Paragraphs, lists, inline formatting (italic, bold, sup/sub)
- Figures (<fig>) with <graphic> → downloaded images
- Tables (<table-wrap>) → Markdown tables
- Mathematics (<tex-math>, <mml:math>) → LaTeX math delimiters
- Reference lists (<ref-list>)
- Supplementary materials

Spec: https://jats.nlm.nih.gov/archiving/tag-library/1.4/
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import ImageAsset, PaperMetadata, ParseResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XML namespace handling
# ---------------------------------------------------------------------------

# JATS commonly uses these namespaces; strip them for simpler tag matching
_JATS_NS = {
    "mml": "http://www.w3.org/1998/Math/MathML",
    "xlink": "http://www.w3.org/1999/xlink",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}


def _strip_ns(tag: str) -> str:
    """Remove XML namespace from a tag string.

    '{http://...}article-title' → 'article-title'
    """
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _find(element: ET.Element, tag: str) -> ET.Element | None:
    """Find the first child element matching *tag* (namespace-insensitive).

    Searches direct children only. For recursive search, use _find_deep.
    """
    for child in element:
        if _strip_ns(child.tag) == tag:
            return child
    return None


def _find_deep(element: ET.Element, tag: str) -> ET.Element | None:
    """Recursively find the first descendant element matching *tag*."""
    for child in element:
        if _strip_ns(child.tag) == tag:
            return child
        result = _find_deep(child, tag)
        if result is not None:
            return result
    return None


def _find_all(element: ET.Element, tag: str) -> list[ET.Element]:
    """Find all child elements matching *tag* (namespace-insensitive)."""
    return [c for c in element if _strip_ns(c.tag) == tag]


def _get_text(element: ET.Element) -> str:
    """Get the full inner text of an element, preserving some whitespace.

    For inline elements, text is joined without extra spaces.
    For block elements, text blocks are separated by newlines.
    """
    if element is None:
        return ""
    # itertext() yields all text nodes in document order
    parts = []
    for text in element.itertext():
        cleaned = text.strip()
        if cleaned:
            parts.append(cleaned)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# JATS → Markdown converter
# ---------------------------------------------------------------------------


class JATSParser:
    """Parse JATS XML into structured Markdown with metadata.

    Usage::

        parser = JATSParser()
        result = parser.parse(xml_bytes)

        print(result.metadata.title)
        print(result.markdown_body)
        for img in result.images:
            save(img.data, img.filename)
    """

    # Maps JATS section types to markdown heading level
    _SECTION_HEADING_LEVELS = {
        "abstract": 2,   # ##
        "intro": 2,      # ##
        "methods": 2,    # ##
        "results": 2,    # ##
        "discussion": 2, # ##
        "conclusions": 2,# ##
        "supplementary-material": 2,
        "acknowledgments": 2,
    }

    def __init__(self) -> None:
        self._images: list[ImageAsset] = []
        self._image_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, xml_data: bytes | str) -> ParseResult:
        """Parse JATS XML into a ParseResult.

        Args:
            xml_data: Raw JATS XML as bytes or string.

        Returns:
            ParseResult with metadata, markdown_body, and images.
        """
        from ..models import ImageAsset, PaperMetadata, ParseResult

        if isinstance(xml_data, bytes):
            xml_data = xml_data.decode("utf-8", errors="replace")

        # Clean XML declaration / DOCTYPE — ET.fromstring chokes on some DOCTYPEs
        xml_data = self._clean_xml_header(xml_data)

        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError as exc:
            # Try stripping the DOCTYPE entirely
            doctype_stripped = re.sub(
                r"<!DOCTYPE[^>]*>", "", xml_data, flags=re.IGNORECASE
            )
            root = ET.fromstring(doctype_stripped)

        # Reset state
        self._images = []
        self._image_counter = 0

        # Locate <article> root
        article = root if _strip_ns(root.tag) == "article" else _find(root, "article")
        if article is None:
            raise RuntimeError("JATS XML missing <article> root element")

        # Find <front> and <body>
        front = _find(article, "front")
        body = _find(article, "body")
        back = _find(article, "back")

        # Parse metadata
        metadata = self._parse_metadata(front)

        # Parse body sections → markdown
        md_parts: list[str] = []

        # Title and authors (as H1 + italic)
        if metadata.title:
            md_parts.append(f"# {metadata.title}\n")
        if metadata.authors:
            md_parts.append(f"*{', '.join(metadata.authors)}*\n")

        # Abstract
        if front is not None:
            abstract_md = self._parse_abstract(front)
            if abstract_md:
                md_parts.append(abstract_md)

        # Body sections
        if body is not None:
            body_md = self._parse_body(body)
            if body_md:
                md_parts.append(body_md)

        # Back matter (references, acknowledgments, etc.)
        if back is not None:
            back_md = self._parse_back(back)
            if back_md:
                md_parts.append(back_md)

        markdown_body = "\n".join(md_parts)

        return ParseResult(
            metadata=metadata,
            markdown_body=markdown_body,
            images=list(self._images),
            source_type="europe_pmc",
            raw_source_url=metadata.url,
        )

    # ------------------------------------------------------------------
    # Metadata parsing
    # ------------------------------------------------------------------

    def _parse_metadata(self, front: ET.Element | None) -> PaperMetadata:
        """Extract PaperMetadata from <front> element."""
        from ..models import PaperMetadata

        meta = PaperMetadata()

        if front is None:
            return meta

        article_meta = _find(front, "article-meta")
        if article_meta is None:
            return meta

        # Title
        title_group = _find(article_meta, "title-group")
        if title_group is not None:
            article_title = _find(title_group, "article-title")
            if article_title is not None:
                meta.title = _get_text(article_title)

        # Authors
        contrib_group = _find(article_meta, "contrib-group")
        if contrib_group is not None:
            for contrib in _find_all(contrib_group, "contrib"):
                if contrib.get("contrib-type", "") == "author":
                    name = self._parse_contributor_name(contrib)
                    if name:
                        meta.authors.append(name)

        # Abstract (just text for metadata)
        abstract_el = _find(article_meta, "abstract")
        if abstract_el is not None:
            meta.abstract = _get_text(abstract_el)[:500]

        # DOI
        for aid in _find_all(article_meta, "article-id"):
            if aid.get("pub-id-type", "") == "doi":
                meta.doi = aid.text or ""
                break

        # Journal
        journal_meta = _find(front, "journal-meta")
        if journal_meta is not None:
            # <journal-title> is nested inside <journal-title-group>
            journal_title = _find_deep(journal_meta, "journal-title")
            if journal_title is not None:
                meta.journal = _get_text(journal_title)
            # Try abbreviated title as fallback
            if not meta.journal:
                abbr = _find_deep(journal_meta, "abbrev-journal-title")
                if abbr is not None:
                    meta.journal = _get_text(abbr)

        # Year
        pub_date = _find(article_meta, "pub-date")
        if pub_date is not None:
            year_el = _find(pub_date, "year")
            if year_el is not None:
                meta.year = year_el.text or ""

        # URL (from self-uri or DOI)
        for aid in _find_all(article_meta, "article-id"):
            if aid.get("pub-id-type", "") == "doi" and aid.text:
                meta.url = f"https://doi.org/{aid.text}"
                break

        # CC License (from <permissions>)
        permissions = _find(article_meta, "permissions")
        if permissions is not None:
            license_el = _find(permissions, "license")
            if license_el is not None:
                license_text = _get_text(license_el)
                # Store license in abstract for frontmatter (metadata has no license field)
                # We'll append it to the abstract
                if license_text:
                    existing_abstract = meta.abstract
                    license_short = license_text[:200]
                    if existing_abstract:
                        meta.abstract = f"[{license_short}] {existing_abstract}"
                    else:
                        meta.abstract = license_short

        return meta

    @staticmethod
    def _parse_contributor_name(contrib: ET.Element) -> str:
        """Parse an author name from a <contrib> element.

        JATS uses <name> with <surname> and <given-names> (Western style).
        Falls back to <string-name> if <name> is absent.
        """
        name_el = _find(contrib, "name")
        if name_el is not None:
            surname = _find(name_el, "surname")
            given = _find(name_el, "given-names")
            sn = surname.text.strip() if surname is not None and surname.text else ""
            gn = given.text.strip() if given is not None and given.text else ""
            if sn and gn:
                return f"{gn} {sn}"
            return sn or gn or ""

        # Fallback: <string-name>
        string_name = _find(contrib, "string-name")
        if string_name is not None:
            return _get_text(string_name)

        return ""

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    def _parse_abstract(self, front: ET.Element) -> str:
        """Parse <abstract> into Markdown."""
        article_meta = _find(front, "article-meta")
        if article_meta is None:
            return ""

        abstract_el = _find(article_meta, "abstract")
        if abstract_el is None:
            return ""

        parts = ["## Abstract\n"]

        # 1. Check for a preamble <title> (e.g. "Summary") before the <sec> list
        preamble_title = _find(abstract_el, "title")
        if preamble_title is not None:
            preamble_text = _get_text(preamble_title)
            if preamble_text.strip() and preamble_text.strip().lower() not in ("abstract",):
                parts.append(f"**{preamble_text}**  ")

        # 2. Check for text directly in <abstract> before the first <sec>
        #    (some JATS put unstructured summary text here)
        if abstract_el.text and abstract_el.text.strip():
            preamble_text = abstract_el.text.strip()
            if preamble_text:
                parts.append(preamble_text + "\n")

        # 3. Process abstract sections (structured abstracts) or plain paragraphs
        secs = _find_all(abstract_el, "sec")
        if secs:
            for sec in secs:
                title_el = _find(sec, "title")
                if title_el is not None:
                    section_title = _get_text(title_el)
                    # Skip non-content sections like "Keywords"
                    if section_title.strip().lower() in ("keywords", "keyword"):
                        continue
                    parts.append(f"**{section_title}**  ")
                # Collect paragraphs
                for para in _find_all(sec, "p"):
                    parts.append(self._para_to_md(para))
        else:
            # Plain (unstructured) abstract
            for para in _find_all(abstract_el, "p"):
                parts.append(self._para_to_md(para))

        # If no content found, use raw text
        if len(parts) == 1:  # only "## Abstract"
            text = _get_text(abstract_el)
            if text:
                parts.append(text + "\n")

        # Extract keywords from <kwd-group> (may be in article-meta, not inside abstract)
        kwd_group = _find(article_meta, "kwd-group")
        if kwd_group is not None:
            keywords = []
            for kwd in _find_all(kwd_group, "kwd"):
                kw = _get_text(kwd)
                if kw:
                    keywords.append(kw)
            if keywords:
                parts.append("\n**Keywords:** " + "; ".join(keywords) + "\n")

        return "\n".join(parts) + "\n"

    # ------------------------------------------------------------------
    # Body sections
    # ------------------------------------------------------------------

    def _parse_body(self, body: ET.Element) -> str:
        """Parse <body> into Markdown."""
        parts: list[str] = []

        for child in body:
            tag = _strip_ns(child.tag)
            if tag == "sec":
                parts.append(self._section_to_md(child, level=2))
            elif tag == "p":
                parts.append(self._para_to_md(child))
            elif tag == "fig":
                parts.append(self._fig_to_md(child))
            elif tag == "table-wrap":
                parts.append(self._table_to_md(child))
            elif tag == "disp-formula":
                parts.append(self._formula_to_md(child))
            elif tag == "boxed-text":
                parts.append(self._boxed_text_to_md(child))

        return "\n".join(parts)

    def _section_to_md(self, section: ET.Element, level: int = 2) -> str:
        """Recursively convert a <sec> element to Markdown."""
        parts: list[str] = []

        # Section title
        title_el = _find(section, "title")
        heading_text = _get_text(title_el) if title_el is not None else ""
        if heading_text:
            hashes = "#" * min(level, 6)
            parts.append(f"\n{hashes} {heading_text}\n")

        # Process child elements
        for child in section:
            tag = _strip_ns(child.tag)
            if tag == "title":
                continue  # Already handled
            elif tag == "sec":
                parts.append(self._section_to_md(child, level=level + 1))
            elif tag == "p":
                parts.append(self._para_to_md(child))
            elif tag == "fig":
                parts.append(self._fig_to_md(child))
            elif tag == "table-wrap":
                parts.append(self._table_to_md(child))
            elif tag == "disp-formula":
                parts.append(self._formula_to_md(child))
            elif tag == "list":
                parts.append(self._list_to_md(child))
            elif tag == "boxed-text":
                # Callout / info box
                box_md = self._section_to_md(child, level=level + 1)
                parts.append(f"> {box_md.replace(chr(10), chr(10) + '> ')}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Paragraphs & inline formatting
    # ------------------------------------------------------------------

    def _para_to_md(self, para: ET.Element) -> str:
        """Convert a <p> element to Markdown.

        Handles mixed content: JATS paragraphs can interleave inline elements
        (italic, bold, xref, etc.) with block elements (fig, table-wrap,
        boxed-text). Text and blocks are serialized in document order.
        """
        _BLOCK_TAGS = frozenset({"fig", "table-wrap", "boxed-text", "disp-formula"})

        has_blocks = any(
            _strip_ns(c.tag) in _BLOCK_TAGS for c in para
        )

        if not has_blocks:
            # Simple paragraph — inline formatting only
            text = self._inline_to_md(para)
            return text.strip() + "\n" if text.strip() else ""

        # ------------------------------------------------------------------
        # Mixed content: walk in document order, flushing inline text
        # whenever we encounter a block-level child.
        # ------------------------------------------------------------------
        output: list[str] = []
        inline_buf: list[str] = []

        def _flush_inline() -> None:
            if inline_buf:
                text = " ".join(inline_buf).strip()
                if text:
                    output.append(text + "\n")
                inline_buf.clear()

        # Text that precedes the first child element
        if para.text and para.text.strip():
            inline_buf.append(para.text.strip())

        for child in para:
            tag = _strip_ns(child.tag)

            if tag in _BLOCK_TAGS:
                _flush_inline()
                if tag == "fig":
                    output.append(self._fig_to_md(child))
                elif tag == "table-wrap":
                    output.append(self._table_to_md(child))
                elif tag == "boxed-text":
                    output.append(self._boxed_text_to_md(child))
                elif tag == "disp-formula":
                    output.append(self._formula_to_md(child))
            else:
                # Inline element — render and accumulate
                rendered = self._inline_to_md(child)
                if rendered.strip():
                    inline_buf.append(rendered.strip())

            # Tail text that follows this child (before the next sibling)
            if child.tail and child.tail.strip():
                inline_buf.append(child.tail.strip())

        _flush_inline()

        return "".join(output)

    def _boxed_text_to_md(self, boxed: ET.Element) -> str:
        """Convert a <boxed-text> element (callout / info box) to Markdown.

        Boxed text typically contains a <caption> with a <title> and
        nested <sec> elements. Rendered as a blockquote with heading.
        """
        parts: list[str] = []

        # Caption / title
        caption = _find(boxed, "caption")
        if caption is not None:
            cap_title = _find(caption, "title")
            if cap_title is not None:
                parts.append(f"\n> **{_get_text(cap_title)}**\n")

        # Process child sections and paragraphs
        for child in boxed:
            tag = _strip_ns(child.tag)
            if tag == "caption":
                continue  # Handled above
            elif tag == "sec":
                sec_md = self._section_to_md(child, level=3)
                # Indent each line with blockquote marker
                for line in sec_md.split("\n"):
                    stripped = line.strip()
                    if stripped:
                        parts.append(f"> {stripped}\n")
            elif tag == "p":
                para_md = self._para_to_md(child)
                for line in para_md.split("\n"):
                    stripped = line.strip()
                    if stripped:
                        parts.append(f"> {stripped}\n")
            elif tag == "fig":
                fig_md = self._fig_to_md(child)
                for line in fig_md.split("\n"):
                    stripped = line.strip()
                    if stripped:
                        parts.append(f"> {stripped}\n")

        return "".join(parts)

    def _inline_to_md(self, element: ET.Element) -> str:
        """Recursively convert inline JATS elements to Markdown.

        Handles: italic, bold, sup, sub, monospace, underline, ext-link,
        xref, inline-formula, named-content, and plain text.
        """
        parts: list[str] = []

        # Text node at start
        if element.text:
            parts.append(element.text)

        for child in element:
            tag = _strip_ns(child.tag)

            if tag == "italic":
                parts.append(f"*{self._inline_to_md(child)}*")
            elif tag == "bold":
                parts.append(f"**{self._inline_to_md(child)}**")
            elif tag == "sup":
                parts.append(f"^{self._inline_to_md(child)}^")
            elif tag == "sub":
                parts.append(f"~{self._inline_to_md(child)}~")
            elif tag == "monospace":
                parts.append(f"`{self._inline_to_md(child)}`")
            elif tag == "underline":
                parts.append(f"<u>{self._inline_to_md(child)}</u>")
            elif tag == "ext-link":
                href = child.get("{http://www.w3.org/1999/xlink}href", child.get("xlink:href", ""))
                inner = self._inline_to_md(child)
                parts.append(f"[{inner}]({href})")
            elif tag == "xref":
                # Cross-reference — just keep the text
                parts.append(self._inline_to_md(child))
            elif tag in ("inline-formula", "inline-graphic"):
                parts.append(self._inline_formula_to_md(child))
            elif tag == "named-content":
                parts.append(self._inline_to_md(child))
            elif tag == "break":
                parts.append("\n")
            else:
                parts.append(self._inline_to_md(child))

            # Tail text after child
            if child.tail:
                parts.append(child.tail)

        return "".join(parts)

    def _inline_formula_to_md(self, formula: ET.Element) -> str:
        """Convert an inline formula (<inline-formula>) to LaTeX math."""
        # Try <tex-math> first
        tex = _find(formula, "tex-math")
        if tex is not None:
            tex_text = _get_text(tex) or (tex.text or "")
            return f"${tex_text}$"

        # Try <mml:math> (MathML) — just wrap it, or note it
        mml = None
        for child in formula:
            if _strip_ns(child.tag) == "math":
                mml = child
                break
        if mml is not None:
            # Best effort: generate alt text note
            alttext = mml.get("alttext", "")
            if alttext:
                return f"${alttext}$"
            return "[MathML formula]"

        return ""

    # ------------------------------------------------------------------
    # Lists
    # ------------------------------------------------------------------

    def _list_to_md(self, list_el: ET.Element) -> str:
        """Convert a JATS <list> to a Markdown list."""
        list_type = list_el.get("list-type", "bullet")
        parts: list[str] = []

        items = _find_all(list_el, "list-item")
        for i, item in enumerate(items):
            # Get the main content of the list item
            label = ""
            label_el = _find(item, "label")
            if label_el is not None:
                label = _get_text(label_el)

            para = _find(item, "p")
            if para is not None:
                text = self._inline_to_md(para)
            else:
                text = _get_text(item)

            if list_type == "order" or list_type == "roman-lower":
                prefix = f"{label}. " if label else f"{i + 1}. "
                parts.append(f"{prefix}{text}")
            else:
                prefix = f"{label} " if label else "- "
                parts.append(f"{prefix}{text}")

        return "\n".join(parts) + "\n"

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def _fig_to_md(self, fig: ET.Element) -> str:
        """Convert a <fig> element to Markdown image reference."""
        caption = ""
        caption_el = _find(fig, "caption")
        if caption_el is not None:
            title_el = _find(caption_el, "title")
            caption = _get_text(title_el) if title_el is not None else _get_text(caption_el)

        # Find <graphic> with image URL
        graphic = _find(fig, "graphic")
        if graphic is None:
            # Maybe nested in <alternatives>
            alternatives = _find(fig, "alternatives")
            if alternatives is not None:
                graphic = _find(alternatives, "graphic")

        if graphic is not None:
            href = graphic.get("{http://www.w3.org/1999/xlink}href", graphic.get("xlink:href", ""))
            if href:
                # Generate a descriptive alt text from caption
                alt_text = caption[:100] if caption else f"Figure {self._image_counter + 1}"
                self._image_counter += 1
                return f"\n![{alt_text}]({href})\n"
                # Note: actual image download happens in the extractor,
                # since we need httpx for async fetching. The URL is preserved
                # in the markdown for the extractor to post-process.

            caption_el = _find(fig, "caption")
            if caption_el is not None:
                caption = _get_text(caption_el)

        # Fallback: just show the caption
        if caption:
            return f"\n> **Figure:** {caption}\n"
        return ""

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def _table_to_md(self, table_wrap: ET.Element) -> str:
        """Convert a <table-wrap> to a Markdown table.

        Handles <label>, <caption> (with nested <p>), the XHTML <table>,
        and <table-wrap-foot> footnotes.
        """
        parts: list[str] = []

        # Table title / label
        label_el = _find(table_wrap, "label")
        title_el = _find(table_wrap, "title")

        # Caption may be in <caption> (JATS) rather than <title>
        caption_el = _find(table_wrap, "caption")
        caption_text = ""
        if caption_el is not None:
            # <caption> often contains a <p> with the actual text
            cap_p = _find(caption_el, "p")
            if cap_p is not None:
                caption_text = _get_text(cap_p)
            else:
                caption_text = _get_text(caption_el)

        lbl = f"{_get_text(label_el)}. " if label_el is not None else ""
        ttl = _get_text(title_el) if title_el is not None else ""
        description = caption_text or ttl
        if lbl.strip() or description.strip():
            parts.append(f"\n**{lbl}{description}**\n")

        # Find the actual HTML <table> inside <table-wrap>
        # JATS may embed XHTML tables directly
        table_el = None
        for child in table_wrap:
            tag = _strip_ns(child.tag)
            if tag == "table":
                table_el = child
                break
            # Sometimes wrapped in <alternatives>
            if tag == "alternatives":
                table_el = _find(child, "table")
                if table_el is not None:
                    break

        if table_el is not None:
            # Parse thead + tbody
            rows: list[list[str]] = []

            thead = _find(table_el, "thead") or _find(table_el, "tgroup")
            # Find ALL tbody sections (multi-part tables have multiple tbodies)
            all_tbodies = _find_all(table_el, "tbody")
            if not all_tbodies:
                all_tbodies = [table_el]  # fallback: parse the table element directly

            # Header row
            if thead is not None:
                trs = _find_all(thead, "tr") or _find_all(thead, "row")
                for row in trs:
                    cells = []
                    for cell in _find_all(row, "th"):
                        cells.append(_get_text(cell).replace("|", r"\|"))
                    if cells:
                        rows.append(cells)

            # Body rows — iterate all tbodies (multi-part table support)
            for tbody in all_tbodies:
                trs = _find_all(tbody, "tr") or _find_all(tbody, "row")
                for row in trs:
                    cells = []
                    for cell in _find_all(row, "td") or _find_all(row, "entry"):
                        cells.append(_get_text(cell).replace("|", r"\|"))
                    if cells:
                        rows.append(cells)

            if rows:
                # Build Markdown table
                max_cols = max(len(r) for r in rows)
                normalized = [r + [""] * (max_cols - len(r)) for r in rows]

                for i, row in enumerate(normalized):
                    parts.append("| " + " | ".join(row) + " |")
                    if i == 0:
                        parts.append("| " + " | ".join(["---"] * max_cols) + " |")

                parts.append("")

        # ------------------------------------------------------------------
        # Table footnotes (<table-wrap-foot> — may be multiple)
        # ------------------------------------------------------------------
        for table_foot in _find_all(table_wrap, "table-wrap-foot"):
            for fn in _find_all(table_foot, "fn"):
                fn_label = _find(fn, "label")
                fn_para = _find(fn, "p")

                label_text = _get_text(fn_label) if fn_label is not None else ""
                para_text = self._para_to_md(fn_para) if fn_para is not None else ""

                if label_text:
                    parts.append(f"^{label_text}^ {para_text.strip()}")
                else:
                    parts.append(f"> {para_text.strip()}\n")

            if parts and not parts[-1].endswith("\n"):
                parts.append("")

        # ------------------------------------------------------------------
        # Nested <table-wrap> (multi-part tables)
        # Some JATS documents nest additional <table-wrap> elements within
        # the parent to represent multi-part tables. Recurse into them.
        # ------------------------------------------------------------------
        for child in table_wrap:
            tag = _strip_ns(child.tag)
            if tag == "table-wrap":
                parts.append(self._table_to_md(child))

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Formulas
    # ------------------------------------------------------------------

    def _formula_to_md(self, formula: ET.Element) -> str:
        """Convert a <disp-formula> to block LaTeX math."""
        tex = _find(formula, "tex-math")
        if tex is not None:
            tex_text = _get_text(tex) or (tex.text or "")
            return f"\n$$\n{tex_text}\n$$\n"

        mml = None
        for child in formula:
            if _strip_ns(child.tag) == "math":
                mml = child
                break
        if mml is not None:
            alttext = mml.get("alttext", "")
            if alttext:
                return f"\n$$\n{alttext}\n$$\n"
            return "\n[MathML formula]\n"

        return ""

    # ------------------------------------------------------------------
    # Back matter
    # ------------------------------------------------------------------

    def _parse_back(self, back: ET.Element) -> str:
        """Parse <back> element (references, acknowledgments, etc.).

        Handles <ack> at both top-level and nested inside <sec>.
        """
        parts: list[str] = []

        # Acknowledgments — may be direct <ack> or inside a <sec>
        ack = _find(back, "ack")
        if ack is None:
            # Look inside <sec> elements
            for sec in _find_all(back, "sec"):
                title_el = _find(sec, "title")
                sec_title = _get_text(title_el) if title_el is not None else ""
                if sec_title.lower() in ("acknowledgements", "acknowledgments", "acknowledgement"):
                    ack = sec
                    break

        if ack is not None:
            tag = _strip_ns(ack.tag)
            if tag == "sec":
                parts.append("\n## Acknowledgements\n")
                # Process all children except title
                for child in ack:
                    ctag = _strip_ns(child.tag)
                    if ctag == "title":
                        continue
                    elif ctag == "p":
                        parts.append(self._para_to_md(child))
                    elif ctag == "fig":
                        parts.append(self._fig_to_md(child))
            else:
                parts.append("\n## Acknowledgements\n")
                for para in _find_all(ack, "p"):
                    parts.append(self._para_to_md(para))

        # Other top-level sections in back (appendices, etc.)
        for sec in _find_all(back, "sec"):
            title_el = _find(sec, "title")
            sec_title = _get_text(title_el) if title_el is not None else ""
            if sec_title.lower() in ("acknowledgements", "acknowledgments", "acknowledgement"):
                continue  # Already handled above
            parts.append(self._section_to_md(sec, level=2))

        # Reference list
        ref_list = _find(back, "ref-list")
        if ref_list is not None:
            parts.append("\n## References\n")
            refs = _find_all(ref_list, "ref")
            for i, ref in enumerate(refs, 1):
                ref_text = self._ref_to_md(ref)
                if ref_text:
                    parts.append(f"{i}. {ref_text}")

        # Supplementary material
        for supp in _find_all(back, "supplementary-material"):
            parts.append("\n## Supplementary Material\n")
            parts.append(self._section_to_md(supp, level=3))

        # Footnotes group (appendix, supplementary data links, etc.)
        fn_group = _find(back, "fn-group")
        if fn_group is not None:
            for fn in _find_all(fn_group, "fn"):
                label_el = _find(fn, "label")
                label = _get_text(label_el) if label_el is not None else ""
                if label:
                    parts.append(f"\n## {label}\n")
                for para in _find_all(fn, "p"):
                    parts.append(self._para_to_md(para))

        return "\n".join(parts) + "\n" if parts else ""

    def _ref_to_md(self, ref: ET.Element) -> str:
        """Convert a single <ref> to a formatted reference string."""
        # Try <element-citation> or <mixed-citation>
        citation = _find(ref, "element-citation") or _find(ref, "mixed-citation")
        if citation is None:
            citation = ref

        # Build reference string from common JATS elements
        pieces: list[str] = []

        # Authors
        person_group = _find(citation, "person-group")
        if person_group is not None:
            names = _find_all(person_group, "name")
            if names:
                author_strs = [
                    self._parse_contributor_name(n) for n in names[:6]
                ]
                authors = ", ".join(a for a in author_strs if a)
                if len(names) > 6:
                    authors += ", et al."
                pieces.append(authors)

        # Title
        article_title = _find(citation, "article-title")
        if article_title is not None:
            pieces.append(f'"{_get_text(article_title)}"')

        # Source (journal)
        source = _find(citation, "source")
        if source is not None:
            pieces.append(_get_text(source))

        # Year
        year_el = _find(citation, "year")
        if year_el is not None:
            pieces.append(f"({year_el.text})")

        # Volume / issue / pages
        volume = _find(citation, "volume")
        issue = _find(citation, "issue")
        fpage = _find(citation, "fpage")
        lpage = _find(citation, "lpage")
        vol_issue = ""
        if volume is not None:
            vol_issue = volume.text or ""
        if issue is not None:
            vol_issue += f"({issue.text})" if vol_issue else f"({issue.text})"
        if fpage is not None:
            vol_issue += f":{fpage.text}" if vol_issue else f"{fpage.text}"
            if lpage is not None:
                vol_issue += f"-{lpage.text}"
        if vol_issue:
            pieces.append(vol_issue)

        # DOI
        pub_id = _find(citation, "pub-id")
        if pub_id is not None and pub_id.get("pub-id-type", "") == "doi":
            pieces.append(f"DOI: {pub_id.text}")

        return ". ".join(pieces)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_xml_header(xml_str: str) -> str:
        """Prepare XML string for ET.fromstring by stripping problematic headers."""
        # Remove XML declaration (ET handles it)
        xml_str = re.sub(r"<\?xml[^?]*\?>", "", xml_str, count=1)

        # Keep DOCTYPE but note: ET.fromstring may need it for entities.
        # If parsing fails, caller retries without DOCTYPE.

        return xml_str.strip()
