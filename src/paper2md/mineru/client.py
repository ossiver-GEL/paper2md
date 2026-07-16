"""MinerU Cloud API client.

Supports both precision extract API (with token) and agent lightweight API.
Handles task submission, polling, result download, and zip extraction.
"""

from __future__ import annotations

import asyncio
import io
import logging
import zipfile
from pathlib import Path
import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Precision Extract API (requires token, supports up to 200MB / 200 pages)
PRECISION_BASE = "https://mineru.net/api/v4"
PRECISION_SUBMIT = f"{PRECISION_BASE}/extract/task"
PRECISION_BATCH_SUBMIT = f"{PRECISION_BASE}/extract/task/batch"
PRECISION_BATCH_UPLOAD = f"{PRECISION_BASE}/file-urls/batch"
PRECISION_RESULT_TMPL = f"{PRECISION_BASE}/extract/task/{{task_id}}"
PRECISION_BATCH_RESULT_TMPL = f"{PRECISION_BASE}/extract-results/batch/{{batch_id}}"

# Agent Lightweight API (no token, max 10MB / 20 pages)
AGENT_BASE = "https://mineru.net/api/v1/agent"
AGENT_URL_SUBMIT = f"{AGENT_BASE}/parse/url"
AGENT_FILE_SUBMIT = f"{AGENT_BASE}/parse/file"
AGENT_RESULT_TMPL = f"{AGENT_BASE}/parse/{{task_id}}"

# Polling configuration
POLL_INTERVAL_SEC = 3
POLL_TIMEOUT_SEC = 600

# ---------------------------------------------------------------------------
# Data models for API responses
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class MinerUTaskResult:
    """Result from a MinerU extraction task."""

    task_id: str
    state: str  # done, pending, running, failed, converting, waiting-file
    markdown_text: str = ""
    full_zip_bytes: bytes | None = None
    images: dict[str, bytes] = field(default_factory=dict)  # filename -> data
    error_message: str = ""
    error_code: int = 0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MinerUClient:
    """Async HTTP client for the MinerU document extraction API.

    Usage:
        client = MinerUClient(api_key="sk-...")
        result = await client.parse_url("https://example.com/paper.pdf")
        # result.markdown_text contains the full markdown
    """

    def __init__(
        self,
        api_key: str | None = None,
        use_precision: bool = True,
        model_version: str = "vlm",
        language: str = "en",
        enable_table: bool = True,
        enable_formula: bool = True,
        is_ocr: bool = False,
        timeout: int = POLL_TIMEOUT_SEC,
    ):
        """
        Args:
            api_key: MinerU API token. If None, uses agent lightweight API.
            use_precision: Whether to use precision API (requires api_key).
            model_version: 'pipeline', 'vlm' (recommended), or 'MinerU-HTML'.
            language: Document language code (e.g., 'en', 'ch').
            enable_table: Enable table recognition.
            enable_formula: Enable formula recognition.
            is_ocr: Enable OCR for scanned PDFs.
            timeout: Polling timeout in seconds.
        """
        self.api_key = api_key
        self.use_precision = use_precision and api_key is not None
        self.model_version = model_version
        self.language = language
        self.enable_table = enable_table
        self.enable_formula = enable_formula
        self.is_ocr = is_ocr
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def parse_url(self, url: str) -> MinerUTaskResult:
        """Parse a document from a remote URL.

        Args:
            url: Publicly accessible URL to the document.

        Returns:
            MinerUTaskResult with markdown_text populated on success.
        """
        if self.use_precision:
            return await self._precision_parse_url(url)
        else:
            return await self._agent_parse_url(url)

    async def parse_file(self, file_path: str | Path) -> MinerUTaskResult:
        """Parse a local document file by uploading it.

        Args:
            file_path: Path to a local PDF/image/document file.

        Returns:
            MinerUTaskResult with markdown_text populated on success.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return MinerUTaskResult(
                task_id="",
                state="failed",
                error_message=f"File not found: {file_path}",
            )

        if self.use_precision:
            return await self._precision_parse_file(file_path)
        else:
            return await self._agent_parse_file(file_path)

    # ------------------------------------------------------------------
    # Precision API (token required)
    # ------------------------------------------------------------------

    async def _precision_parse_url(self, url: str) -> MinerUTaskResult:
        """Submit URL to precision API and poll for result."""
        headers = self._auth_headers()
        payload = {
            "url": url,
            "model_version": self.model_version,
            "language": self.language,
            "enable_table": self.enable_table,
            "enable_formula": self.enable_formula,
            "is_ocr": self.is_ocr,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            # Step 1: Submit task
            resp = await client.post(PRECISION_SUBMIT, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                return MinerUTaskResult(
                    task_id="",
                    state="failed",
                    error_message=data.get("msg", "Unknown error"),
                )
            task_id = data["data"]["task_id"]
            logger.info("Precision task submitted: %s", task_id)

            # Step 2: Poll until done
            result = await self._poll_precision(client, headers, task_id)
            return result

    async def _precision_parse_file(self, file_path: Path) -> MinerUTaskResult:
        """Upload local file to precision API and parse."""
        headers = self._auth_headers()
        file_size = file_path.stat().st_size
        file_name = file_path.name

        async with httpx.AsyncClient(timeout=60) as client:
            # Step 1: Get upload URL
            batch_payload = {
                "files": [{"name": file_name}],
                "model_version": self.model_version,
                "language": self.language,
                "enable_table": self.enable_table,
                "enable_formula": self.enable_formula,
                "is_ocr": self.is_ocr,
            }
            resp = await client.post(
                PRECISION_BATCH_UPLOAD, headers=headers, json=batch_payload
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                return MinerUTaskResult(
                    task_id="",
                    state="failed",
                    error_message=data.get("msg", "Unknown error"),
                )

            batch_id = data["data"]["batch_id"]
            upload_url = data["data"]["file_urls"][0]

            # Step 2: Upload file to signed URL
            with open(file_path, "rb") as f:
                put_resp = await client.put(upload_url, content=f.read())
                if put_resp.status_code not in (200, 201):
                    return MinerUTaskResult(
                        task_id="",
                        state="failed",
                        error_message=f"Upload failed: HTTP {put_resp.status_code}",
                    )
            logger.info("File uploaded, batch_id=%s", batch_id)

            # Step 3: Poll batch result
            result = await self._poll_precision_batch(client, headers, batch_id)
            return result

    async def _poll_precision(
        self, client: httpx.AsyncClient, headers: dict, task_id: str
    ) -> MinerUTaskResult:
        """Poll precision single-task endpoint."""
        url = PRECISION_RESULT_TMPL.format(task_id=task_id)
        start = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > self.timeout:
                return MinerUTaskResult(
                    task_id=task_id,
                    state="failed",
                    error_message=f"Polling timed out after {self.timeout}s",
                )

            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                return MinerUTaskResult(
                    task_id=task_id,
                    state="failed",
                    error_message=data.get("msg", "Unknown error"),
                )

            state = data["data"].get("state", "unknown")
            logger.debug("Task %s state: %s (elapsed: %.0fs)", task_id, state, elapsed)

            if state == "done":
                return await self._download_and_extract(
                    client, data["data"].get("full_zip_url", ""), task_id
                )
            elif state == "failed":
                return MinerUTaskResult(
                    task_id=task_id,
                    state="failed",
                    error_message=data["data"].get("err_msg", "Extraction failed"),
                )

            await asyncio.sleep(POLL_INTERVAL_SEC)

    async def _poll_precision_batch(
        self, client: httpx.AsyncClient, headers: dict, batch_id: str
    ) -> MinerUTaskResult:
        """Poll precision batch endpoint."""
        url = PRECISION_BATCH_RESULT_TMPL.format(batch_id=batch_id)
        start = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > self.timeout:
                return MinerUTaskResult(
                    task_id=batch_id,
                    state="failed",
                    error_message=f"Batch polling timed out after {self.timeout}s",
                )

            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                return MinerUTaskResult(
                    task_id=batch_id,
                    state="failed",
                    error_message=data.get("msg", "Unknown error"),
                )

            results = data["data"].get("extract_result", [])
            if not results:
                return MinerUTaskResult(
                    task_id=batch_id,
                    state="failed",
                    error_message="No extract results in batch response",
                )

            # Find first completed or failed result
            for r in results:
                state = r.get("state", "unknown")
                logger.debug(
                    "Batch %s / %s state: %s", batch_id, r.get("file_name"), state
                )
                if state == "done":
                    return await self._download_and_extract(
                        client, r.get("full_zip_url", ""), batch_id
                    )
                elif state == "failed":
                    return MinerUTaskResult(
                        task_id=batch_id,
                        state="failed",
                        error_message=r.get("err_msg", "Extraction failed"),
                    )
                elif state in ("pending", "running", "converting"):
                    break  # Still processing, wait
                elif state == "waiting-file":
                    break  # Still uploading, wait

            await asyncio.sleep(POLL_INTERVAL_SEC)

    async def _download_and_extract(
        self, client: httpx.AsyncClient, zip_url: str, task_id: str
    ) -> MinerUTaskResult:
        """Download result zip and extract full.md + images."""
        if not zip_url:
            return MinerUTaskResult(
                task_id=task_id,
                state="failed",
                error_message="No zip URL in completed task response",
            )

        logger.info("Downloading result zip: %s", zip_url)
        resp = await client.get(zip_url)
        resp.raise_for_status()
        zip_bytes = resp.content

        markdown_text = ""
        images: dict[str, bytes] = {}

        # Extract markdown and images from the zip
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                lower = name.lower()
                # Find the main markdown file
                if name.endswith("full.md") or name == "full.md":
                    markdown_text = zf.read(name).decode("utf-8", errors="replace")
                elif lower.endswith(".md") and not markdown_text:
                    markdown_text = zf.read(name).decode("utf-8", errors="replace")
                # Extract images (common formats)
                elif lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")):
                    # Use just the filename (strip directory paths)
                    fname = name.rsplit("/", 1)[-1]
                    images[fname] = zf.read(name)

        return MinerUTaskResult(
            task_id=task_id,
            state="done",
            markdown_text=markdown_text,
            full_zip_bytes=zip_bytes,
            images=images,
        )

    # ------------------------------------------------------------------
    # Agent Lightweight API (no token)
    # ------------------------------------------------------------------

    async def _agent_parse_url(self, url: str) -> MinerUTaskResult:
        """Submit URL to agent API and poll for result."""
        payload = {
            "url": url,
            "language": self.language,
            "enable_table": self.enable_table,
            "enable_formula": self.enable_formula,
            "is_ocr": self.is_ocr,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(AGENT_URL_SUBMIT, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                return MinerUTaskResult(
                    task_id="",
                    state="failed",
                    error_message=data.get("msg", "Unknown error"),
                )
            task_id = data["data"]["task_id"]
            logger.info("Agent task submitted: %s", task_id)
            return await self._poll_agent(client, task_id)

    async def _agent_parse_file(self, file_path: Path) -> MinerUTaskResult:
        """Upload local file via agent signed-URL flow."""
        file_name = file_path.name

        async with httpx.AsyncClient(timeout=60) as client:
            # Step 1: Get signed upload URL
            payload = {
                "file_name": file_name,
                "language": self.language,
                "enable_table": self.enable_table,
                "enable_formula": self.enable_formula,
                "is_ocr": self.is_ocr,
            }
            resp = await client.post(AGENT_FILE_SUBMIT, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                return MinerUTaskResult(
                    task_id="",
                    state="failed",
                    error_message=data.get("msg", "Unknown error"),
                )

            task_id = data["data"]["task_id"]
            upload_url = data["data"]["file_url"]

            # Step 2: Upload file
            with open(file_path, "rb") as f:
                put_resp = await client.put(upload_url, content=f.read())
                if put_resp.status_code not in (200, 201):
                    return MinerUTaskResult(
                        task_id="",
                        state="failed",
                        error_message=f"Upload failed: HTTP {put_resp.status_code}",
                    )
            logger.info("Agent file uploaded, task_id=%s", task_id)

            # Step 3: Poll
            return await self._poll_agent(client, task_id)

    async def _poll_agent(
        self, client: httpx.AsyncClient, task_id: str
    ) -> MinerUTaskResult:
        """Poll agent task endpoint."""
        url = AGENT_RESULT_TMPL.format(task_id=task_id)
        start = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start
            if elapsed > self.timeout:
                return MinerUTaskResult(
                    task_id=task_id,
                    state="failed",
                    error_message=f"Polling timed out after {self.timeout}s",
                )

            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                return MinerUTaskResult(
                    task_id=task_id,
                    state="failed",
                    error_message=data.get("msg", "Unknown error"),
                )

            state = data["data"].get("state", "unknown")
            logger.debug("Agent task %s state: %s", task_id, state)

            if state == "done":
                md_url = data["data"].get("markdown_url", "")
                if md_url:
                    md_resp = await client.get(md_url)
                    md_resp.raise_for_status()
                    return MinerUTaskResult(
                        task_id=task_id,
                        state="done",
                        markdown_text=md_resp.text,
                    )
                return MinerUTaskResult(
                    task_id=task_id,
                    state="done",
                    markdown_text="",
                    error_message="No markdown_url in response",
                )
            elif state == "failed":
                return MinerUTaskResult(
                    task_id=task_id,
                    state="failed",
                    error_message=data["data"].get(
                        "err_msg", "Extraction failed"
                    ),
                    error_code=data["data"].get("err_code", 0),
                )

            await asyncio.sleep(POLL_INTERVAL_SEC)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict:
        """Build authorization headers for precision API."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
