"""Europe PMC paper source extractor.

Uses the free Europe PMC REST API to fetch full-text JATS XML for
open-access papers deposited in PubMed Central / Europe PMC.

Europe PMC (https://europepmc.org/) provides:
- Search by DOI, PMID, PMCID, or keyword
- Full-text JATS XML for OA papers (no API key required)
- Structured metadata with abstracts and references

Extraction strategy:
1. Resolve input (DOI / PMID / PMCID / URL) → PMCID
2. Fetch JATS XML full text via the fullTextXML endpoint
3. Parse JATS → Markdown with jats_parser
4. Download referenced images from PMC
5. Fall back to Europe PMC metadata-only if full text unavailable
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, Tag

from ..models import ConvertConfig, ImageAsset, PaperMetadata, ParseResult
from .base import BaseExtractor
from .jats_parser import JATSParser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36 "
    "paper2md/1.0"
)

# Patterns for extracting identifiers from URLs and raw strings
DOI_PATTERN = re.compile(
    r"(?:doi\.org/|DOI:?\s*|doi:?\s*)?(10\.\d{4,}/[^\s\"'<>]+)",
    re.IGNORECASE,
)
_BARE_DOI_RE = re.compile(r"^10\.\d{4,}/[^\s\"'<>]+$")
PMID_PATTERN = re.compile(
    r"(?:pubmed\.ncbi\.nlm\.nih\.gov/|PMID:?\s*)(\d+)",
    re.IGNORECASE,
)
PMCID_PATTERN = re.compile(
    r"(?:ncbi\.nlm\.nih\.gov/pmc/articles/|pmc\.ncbi\.nlm\.nih\.gov/articles/|PMC)(\d+)",
    re.IGNORECASE,
)
EPMC_URL_PATTERN = re.compile(
    r"europepmc\.org/article/(?:PMC|MED)/(\d+)",
    re.IGNORECASE,
)


class EuropePMCExtractor(BaseExtractor):
    """Extract papers from Europe PMC using JATS XML full text."""

    @property
    def source_type(self) -> str:
        return "europe_pmc"

    @property
    def display_name(self) -> str:
        return "Europe PMC"

    # ------------------------------------------------------------------
    # Source detection
    # ------------------------------------------------------------------

    async def can_handle(self, config: ConvertConfig) -> bool:
        """Check if the source can be resolved to a PMC paper.

        Accepts: DOI, PMID, PMCID, Europe PMC URL, PubMed URL, or PMC URL.
        """
        source = config.source.strip()

        # Direct Europe PMC / PubMed / PMC URLs
        if any(domain in source.lower() for domain in [
            "europepmc.org", "pubmed.ncbi.nlm.nih.gov",
            "ncbi.nlm.nih.gov/pmc", "pmc.ncbi.nlm.nih.gov",
        ]):
            return True

        # DOI (any source — we'll cross-check with Europe PMC)
        if DOI_PATTERN.search(source):
            return True

        # Bare DOI (starts with 10.)
        if _BARE_DOI_RE.match(source):
            return True

        # Raw PMID or PMCID
        if PMID_PATTERN.search(source) or PMCID_PATTERN.search(source):
            return True

        return False

    # ------------------------------------------------------------------
    # Main extraction
    # ------------------------------------------------------------------

    async def extract(self, config: ConvertConfig) -> ParseResult:
        """Run Europe PMC extraction: resolve → fetch JATS → convert → download images."""
        source = config.source.strip()
        logger.info("Europe PMC extraction: %s", source)

        async with httpx.AsyncClient(
            timeout=60,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            # Step 1: Resolve to PMCID and get metadata
            pmc_id, search_meta = await self._resolve_to_pmcid(client, source)
            if pmc_id is None:
                # No PMC match — return metadata-only result
                if search_meta:
                    return self._metadata_only_result(search_meta, source)
                raise RuntimeError(
                    f"Could not find paper in Europe PMC: {source}"
                )

            logger.info("Resolved to PMCID: %s", pmc_id)

            # Step 2: Fetch JATS XML full text
            jats_xml = await self._fetch_jats_xml(client, pmc_id)

            if jats_xml is None:
                # No full text XML — return metadata-only
                meta = search_meta or PaperMetadata()
                return ParseResult(
                    metadata=meta,
                    markdown_body=self._format_metadata_md(meta, source),
                    images=[],
                    source_type="europe_pmc",
                    raw_source_url=source,
                )

            # Step 3: Parse JATS XML → Markdown
            parser = JATSParser()
            result = parser.parse(jats_xml)
            result.source_type = "europe_pmc"
            result.raw_source_url = source

            # Step 4: Download images referenced in the markdown
            result = await self._download_images(client, result, pmc_id)

            return result

    # ------------------------------------------------------------------
    # Resolve → PMCID
    # ------------------------------------------------------------------

    async def _resolve_to_pmcid(
        self, client: httpx.AsyncClient, source: str
    ) -> tuple[str | None, PaperMetadata | None]:
        """Resolve a source string to a Europe PMC PMCID.

        Returns:
            (pmcid, metadata) — pmcid is None if paper not in PMC.
            Metadata is populated from the search result when available.
        """
        # Try extracting a PMCID directly first
        pmc_match = PMCID_PATTERN.search(source)
        if pmc_match:
            pmcid = f"PMC{pmc_match.group(1)}"
            return pmcid, None

        epmc_match = EPMC_URL_PATTERN.search(source)
        if epmc_match:
            # Determine if this is a PMC or PMID-based URL
            if "/PMC" in source.upper() or "/PPR" in source.upper():
                # Direct PMC ID
                pmcid = f"PMC{epmc_match.group(1)}"
                return pmcid, None
            else:
                # MED = PubMed ID — need to resolve to PMCID
                pmid = epmc_match.group(1)
                return await self._search_by_pmid(client, pmid)

        # Search by DOI
        doi_match = DOI_PATTERN.search(source)
        if doi_match:
            doi = doi_match.group(1).rstrip(".")
            return await self._search_by_doi(client, doi)

        # Search by PMID
        pmid_match = PMID_PATTERN.search(source)
        if pmid_match:
            pmid = pmid_match.group(1)
            return await self._search_by_pmid(client, pmid)

        # Treat as a free-text search (title or keyword)
        return await self._search_free_text(client, source)

    async def _search_by_doi(
        self, client: httpx.AsyncClient, doi: str
    ) -> tuple[str | None, PaperMetadata | None]:
        """Search Europe PMC by DOI."""
        params = {
            "query": f'DOI:"{doi}"',
            "resultType": "core",
            "format": "json",
        }
        return await self._do_search(client, params, doi)

    async def _search_by_pmid(
        self, client: httpx.AsyncClient, pmid: str
    ) -> tuple[str | None, PaperMetadata | None]:
        """Search Europe PMC by PMID."""
        params = {
            "query": f"EXT_ID:{pmid}",
            "resultType": "core",
            "format": "json",
        }
        return await self._do_search(client, params, pmid)

    async def _search_free_text(
        self, client: httpx.AsyncClient, text: str
    ) -> tuple[str | None, PaperMetadata | None]:
        """Free-text search on Europe PMC."""
        # Sanitize: quote the text for exact phrase matching
        clean = text.strip().replace('"', '')
        if len(clean) > 200:
            clean = clean[:200]
        params = {
            "query": f'"{clean}"',
            "resultType": "core",
            "format": "json",
        }
        return await self._do_search(client, params, text)

    async def _do_search(
        self,
        client: httpx.AsyncClient,
        params: dict,
        label: str,
    ) -> tuple[str | None, PaperMetadata | None]:
        """Execute a Europe PMC search and extract PMCID + metadata."""
        try:
            resp = await client.get(
                f"{EUROPE_PMC_API}/search",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Europe PMC search failed for '%s': %s", label, exc)
            return None, None

        results = (
            data.get("resultList", {}).get("result", [])
        )
        if not results:
            logger.info("No Europe PMC results for '%s'", label)
            return None, None

        first = results[0]

        # Build metadata from search result
        meta = PaperMetadata(
            title=first.get("title", ""),
            authors=self._parse_author_string(first.get("authorString", "")),
            abstract=first.get("abstractText", "")[:500] if first.get("abstractText") else "",
            doi=first.get("doi", ""),
            journal=first.get("journalTitle", "") or (
                first.get("journalInfo", {}).get("journal", {}).get("title", "")
                if isinstance(first.get("journalInfo"), dict) else ""
            ),
            year=str(first.get("pubYear", "")),
            url=f"https://doi.org/{first.get('doi', '')}" if first.get("doi") else "",
        )

        # Check for PMCID
        pmcid = first.get("pmcid", "")
        if pmcid:
            return pmcid, meta

        # Check if in PMC at all
        if first.get("inPMC", "") != "Y":
            logger.info("Paper not in PMC: '%s'", label)
            return None, meta

        # Try fullTextIdList for the PMC ID
        ft_list = first.get("fullTextIdList", {}).get("fullTextId", [])
        if isinstance(ft_list, list):
            for ft in ft_list:
                if isinstance(ft, dict) and ft.get("site") == "PMC":
                    return ft.get("id", ""), meta
        elif isinstance(ft_list, dict) and ft_list.get("site") == "PMC":
            return ft_list.get("id", ""), meta

        return None, meta

    # ------------------------------------------------------------------
    # Fetch JATS XML
    # ------------------------------------------------------------------

    async def _fetch_jats_xml(
        self, client: httpx.AsyncClient, pmc_id: str
    ) -> bytes | None:
        """Download JATS XML full text from Europe PMC.

        Endpoint: /rest/{pmcid}/fullTextXML
        """
        try:
            resp = await client.get(
                f"{EUROPE_PMC_API}/{pmc_id}/fullTextXML",
            )
            if resp.status_code == 404:
                logger.info("No full-text XML available for %s", pmc_id)
                return None
            resp.raise_for_status()

            content = resp.content
            if not content or len(content) < 100:
                logger.warning("Empty JATS XML response for %s", pmc_id)
                return None

            # Validate it's actually XML
            text = content.decode("utf-8", errors="replace")
            if not text.strip().startswith("<?xml") and "<article" not in text[:500]:
                logger.warning("Response doesn't look like JATS XML for %s", pmc_id)
                return None

            logger.info(
                "Downloaded JATS XML for %s: %d bytes", pmc_id, len(content)
            )
            return content

        except httpx.HTTPStatusError as exc:
            logger.warning(
                "HTTP %d fetching JATS XML for %s", exc.response.status_code, pmc_id
            )
            return None
        except Exception as exc:
            logger.warning("Error fetching JATS XML for %s: %s", pmc_id, exc)
            return None

    # ------------------------------------------------------------------
    # Image downloading
    # ------------------------------------------------------------------

    async def _download_images(
        self,
        client: httpx.AsyncClient,
        result: ParseResult,
        pmc_id: str,
    ) -> ParseResult:
        """Download images referenced in the markdown and replace URLs.

        Strategy:
        1. Parse the PMC article HTML page to get CDN image URLs
        2. Map JATS figure filenames (gr1.jpg) → CDN URLs
        3. Download images and replace references in markdown
        """
        import hashlib

        # Find all markdown image references: ![alt](url)
        md_img_re = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
        matches = list(md_img_re.finditer(result.markdown_body))

        if not matches:
            return result

        # Build a map of JATS filename → CDN URL by scraping the PMC HTML
        jats_to_cdn = await self._build_image_url_map(client, pmc_id)

        images: list[ImageAsset] = []
        replacements: list[tuple[str, str]] = []  # (old_ref, new_filename)

        for match in matches:
            alt_text = match.group(1)
            img_ref = match.group(2)  # e.g. "gr1.jpg"

            # Resolve to an actual downloadable URL
            img_url = jats_to_cdn.get(img_ref)
            if img_url is None:
                # Try constructing common URL patterns
                img_url = self._resolve_image_url(img_ref, pmc_id)

            if img_url is None:
                logger.debug("Could not resolve image URL for: %s", img_ref)
                continue

            try:
                img_resp = await client.get(
                    img_url, timeout=30,
                    headers={"Referer": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"},
                )
                img_resp.raise_for_status()
                img_data = img_resp.content
            except Exception as exc:
                logger.debug("Failed to download image %s: %s", img_url, exc)
                continue

            if not img_data or len(img_data) < 100:
                continue

            # Determine extension from Content-Type or URL
            ct = img_resp.headers.get("Content-Type", "")
            ext = self._guess_image_ext(img_ref, ct)

            # Generate a stable filename from content hash
            content_hash = hashlib.sha256(img_data).hexdigest()[:12]
            filename = f"{content_hash}{ext}"

            images.append(ImageAsset(
                filename=filename,
                data=img_data,
                caption="",
                alt_text=alt_text,
            ))
            replacements.append((img_ref, filename))

            logger.debug(
                "Downloaded image: %s → %s (%d bytes)",
                img_url, filename, len(img_data),
            )

        # Apply replacements in markdown
        body = result.markdown_body
        for old_ref, new_filename in replacements:
            # Replace both the image reference and any plain-text mentions
            body = body.replace(f"]({old_ref})", f"](images/{new_filename})")

        result.markdown_body = body
        result.images = images

        return result

    # ------------------------------------------------------------------
    # Image URL resolution
    # ------------------------------------------------------------------

    async def _build_image_url_map(
        self,
        client: httpx.AsyncClient,
        pmc_id: str,
    ) -> dict[str, str]:
        """Scrape the PMC article HTML to map JATS filenames → CDN image URLs.

        PMC HTML contains <figure> elements with <img> tags pointing to
        CDN URLs like:
            cdn.ncbi.nlm.nih.gov/pmc/blobs/{bucket}/{pmcid}/{hash}/{filename}.jpg

        Returns:
            Dict mapping filename (e.g. 'gr1.jpg') → full CDN URL.
        """
        try:
            # Strip 'PMC' prefix for the URL
            pmc_num = pmc_id.replace("PMC", "").replace("pmc", "")
            resp = await client.get(
                f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/",
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
            resp.raise_for_status()
            html = resp.text
        except Exception as exc:
            logger.debug("Failed to fetch PMC page for images: %s", exc)
            return {}

        # Parse with regex to find figure img URLs
        # Pattern: <img ... src="https://cdn.ncbi.nlm.nih.gov/pmc/blobs/.../filename.jpg" ...>
        img_pattern = re.compile(
            r'<img[^>]+src="(https://cdn\.ncbi\.nlm\.nih\.gov/pmc/blobs/[^"]+/([^"/]+\.(?:jpg|jpeg|png|gif|webp)))"',
            re.IGNORECASE,
        )
        url_map: dict[str, str] = {}
        for m in img_pattern.finditer(html):
            full_url = m.group(1)
            filename = m.group(2)
            # Normalize filename
            filename_lower = filename.lower()
            if filename_lower not in url_map:
                url_map[filename_lower] = full_url

        logger.debug(
            "Built image URL map for %s: %d entries", pmc_id, len(url_map)
        )
        return url_map

    @staticmethod
    def _resolve_image_url(filename: str, pmc_id: str) -> str | None:
        """Try to construct an image URL from known patterns.

        This is a fallback when the PMC HTML scraping doesn't find the image.
        """
        # Try Europe PMC figure endpoint (returns HTML page with embedded image)
        # We don't use this directly since it returns HTML, not raw image.

        # Try to use the PMC id-based URL pattern (rarely works without blob hash)
        pmc_num = pmc_id.replace("PMC", "").replace("pmc", "")
        candidate_urls = [
            f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/bin/{filename}",
            f"https://europepmc.org/articles/{pmc_id}/figure/{filename}",
        ]
        # We can't verify without downloading, so return the first candidate
        # The caller will handle download failures
        return candidate_urls[0]

    @staticmethod
    def _guess_image_ext(url: str, content_type: str) -> str:
        """Guess image file extension from URL path or Content-Type."""
        # From Content-Type
        ct_map = {
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/svg+xml": ".svg",
            "image/tiff": ".tiff",
        }
        for mime, ext in ct_map.items():
            if mime in content_type:
                return ext

        # From URL path
        path = urlparse(url).path.lower()
        for ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".tiff"):
            if path.endswith(ext):
                return ext

        return ".jpg"  # Default

    # ------------------------------------------------------------------
    # Author parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_author_string(author_str: str) -> list[str]:
        """Parse a Europe PMC author string into a list of names.

        Europe PMC author strings are semicolon-separated:
            "Li J, Hu S, Shi C, Dong Z, ..."
        """
        if not author_str:
            return []
        return [name.strip() for name in author_str.split(",") if name.strip()]

    # ------------------------------------------------------------------
    # Metadata-only fallback
    # ------------------------------------------------------------------

    def _metadata_only_result(
        self, meta: PaperMetadata, source: str
    ) -> ParseResult:
        """Produce a result with only metadata, no full text."""
        md = self._format_metadata_md(meta, source)
        return ParseResult(
            metadata=meta,
            markdown_body=md,
            images=[],
            source_type="europe_pmc",
            raw_source_url=source,
        )

    @staticmethod
    def _format_metadata_md(meta: PaperMetadata, source: str) -> str:
        """Format a metadata-only markdown output."""
        lines = [
            f"# {meta.title}\n" if meta.title else "",
            f"*{', '.join(meta.authors)}*\n" if meta.authors else "",
            "## Abstract\n" if meta.abstract else "",
            meta.abstract + "\n" if meta.abstract else "",
            (
                "> **Note:** Full text is not available in Europe PMC. "
                "Metadata retrieved from Europe PMC search API.\n"
            ),
            f"> **Source:** {source}\n",
        ]
        return "\n".join(l for l in lines if l)
