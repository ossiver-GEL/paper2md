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

        # Process abstract sections (structured abstracts) or plain paragraphs
        secs = _find_all(abstract_el, "sec")
        if secs:
            for sec in secs:
                title_el = _find(sec, "title")
                if title_el is not None:
                    section_title = _get_text(title_el)
                    parts.append(f"**{section_title}**  ")
                # Collect paragraphs
                for para in _find_all(sec, "p"):
                    parts.append(self._para_to_md(para))
        else:
            # Plain (unstructured) abstract
            for para in _find_all(abstract_el, "p"):
                parts.append(self._para_to_md(para))

        # If no <p> elements, use raw text
        if len(parts) == 1:  # only "## Abstract"
            text = _get_text(abstract_el)
            if text:
                parts.append(text + "\n")

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
        """Convert a <p> element to Markdown, preserving inline formatting.

        Also handles <fig> and <table-wrap> that may appear inline
        within paragraphs (common in JATS).
        """
        # Check for block-level children inside the paragraph
        parts: list[str] = []
        has_block_children = False

        for child in para:
            tag = _strip_ns(child.tag)
            if tag == "fig":
                has_block_children = True
                parts.append(self._fig_to_md(child))
            elif tag == "table-wrap":
                has_block_children = True
                parts.append(self._table_to_md(child))

        # If there were block children, extract the text separately
        if has_block_children:
            # Get text that's not inside fig/table
            text_parts = []
            for child in para:
                tag = _strip_ns(child.tag)
                if tag not in ("fig", "table-wrap"):
                    text_parts.append(self._inline_to_md(child))
            text = " ".join(t for t in text_parts if t.strip())
            if text.strip():
                parts.insert(0, text.strip() + "\n")
            return "\n".join(parts) + "\n"

        # Simple paragraph — just inline formatting
        text = self._inline_to_md(para)
        return text.strip() + "\n" if text.strip() else ""

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
        """Convert a <table-wrap> to a Markdown table."""
        parts: list[str] = []

        # Table title / label
        label_el = _find(table_wrap, "label")
        title_el = _find(table_wrap, "title")
        if label_el is not None or title_el is not None:
            lbl = f"{_get_text(label_el)}. " if label_el is not None else ""
            ttl = _get_text(title_el) if title_el is not None else ""
            parts.append(f"\n**{lbl}{ttl}**\n")

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

        if table_el is None:
            return ""

        # Parse thead + tbody
        rows: list[list[str]] = []

        thead = _find(table_el, "thead") or _find(table_el, "tgroup")
        tbody = _find(table_el, "tbody") or table_el

        # Header row
        if thead is not None:
            for row in _find_all(thead, "tr") or _find_all(thead, "row"):
                cells = []
                for cell in _find_all(row, "th"):
                    cells.append(_get_text(cell).replace("|", r"\|"))
                if cells:
                    rows.append(cells)

        # Body rows
        for row in _find_all(tbody, "tr") or _find_all(tbody, "row"):
            cells = []
            for cell in _find_all(row, "td") or _find_all(row, "entry"):
                cells.append(_get_text(cell).replace("|", r"\|"))
            if cells:
                rows.append(cells)

        if not rows:
            return ""

        # Build Markdown table
        max_cols = max(len(r) for r in rows)
        # Normalize rows
        normalized = [r + [""] * (max_cols - len(r)) for r in rows]

        for i, row in enumerate(normalized):
            parts.append("| " + " | ".join(row) + " |")
            if i == 0:
                parts.append("| " + " | ".join(["---"] * max_cols) + " |")

        parts.append("")
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
        """Parse <back> element (references, acknowledgments, etc.)."""
        parts: list[str] = []

        # Acknowledgments
        ack = _find(back, "ack")
        if ack is not None:
            parts.append("\n## Acknowledgements\n")
            for para in _find_all(ack, "p"):
                parts.append(self._para_to_md(para))

        # Reference list
        ref_list = _find(back, "ref-list")
        if ref_list is not None:
            parts.append("\n## References\n")
            refs = _find_all(ref_list, "ref")
            for i, ref in enumerate(refs, 1):
                ref_text = self._ref_to_md(ref)
                if ref_text:
                    parts.append(f"{i}. {ref_text}")

        # Supplementary material (sometimes in back)
        supp = _find(back, "supplementary-material")
        if supp is not None:
            parts.append("\n## Supplementary Material\n")
            parts.append(self._section_to_md(supp, level=3))

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
