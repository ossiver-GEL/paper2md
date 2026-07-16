"""MinerU local deployment integration.

Uses the official `mineru` CLI (v3.4+) for local GPU-accelerated document
parsing. Requires `pip install mineru[all]`.

The mineru CLI auto-detects GPU, downloads models on first use, and starts
a temporary local API service for processing.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .client import MinerUTaskResult

logger = logging.getLogger(__name__)

# Minimum mineru version that supports auto model download
MIN_MINERU_VERSION = (3, 0, 0)


class MinerULocal:
    """Wrapper for local MinerU (mineru CLI v3.4+) with GPU support.

    Auto-detects GPU and uses the official mineru CLI which:
    - Downloads ML models automatically on first run
    - Starts a local FastAPI service for processing
    - Supports pipeline (CPU), vlm-engine (GPU), and hybrid-engine (GPU)

    Usage:
        local = MinerULocal()
        result = await local.parse("paper.pdf", output_dir="/tmp/out")
    """

    def __init__(
        self,
        backend: str = "auto",
        method: str = "auto",
        device: str = "auto",
        language: str = "en",
        enable_table: bool = True,
        enable_formula: bool = True,
    ):
        """
        Args:
            backend: 'pipeline' (CPU), 'vlm-engine' (GPU), 'hybrid-engine'
                     (GPU, recommended), or 'auto' (detect GPU → hybrid).
            method: 'auto', 'txt', or 'ocr'. Auto-detects OCR need.
            device: 'auto', 'cuda', 'cpu'. Used for backend selection only.
            language: Document language for OCR (e.g. 'en', 'ch').
            enable_table: Enable table recognition.
            enable_formula: Enable formula recognition.
        """
        self._resolved_device = self._resolve_device(device)
        self.backend = self._resolve_backend(backend)
        self.method = method
        self.language = language
        self.enable_table = enable_table
        self.enable_formula = enable_formula

    @property
    def device(self) -> str:
        """Detected device ('cuda' or 'cpu')."""
        return self._resolved_device

    def _resolve_device(self, device: str) -> str:
        """Detect available GPU and return 'cuda' or 'cpu'."""
        if device not in ("auto", "cuda", "cpu"):
            return device

        if device == "cpu":
            return "cpu"

        # Check NVIDIA GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip().split("\n")[0]
                logger.info("GPU detected: %s, using CUDA", gpu_name)
                return "cuda"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info("PyTorch CUDA available: %s", gpu_name)
                return "cuda"
        except ImportError:
            pass

        logger.warning("No GPU detected, falling back to CPU")
        return "cpu"

    def _resolve_backend(self, backend: str) -> str:
        """Resolve backend based on device and preference."""
        if backend not in ("auto", "pipeline", "vlm-engine", "hybrid-engine",
                           "vlm-http-client", "hybrid-http-client"):
            return backend
        if backend != "auto":
            return backend
        return "hybrid-engine" if self._resolved_device == "cuda" else "pipeline"

    async def parse(
        self, file_path: str | Path, output_dir: str | Path | None = None
    ) -> MinerUTaskResult:
        """Parse a local document using mineru CLI.

        Args:
            file_path: Path to the document file.
            output_dir: Output directory for results. If None, uses a temp dir.

        Returns:
            MinerUTaskResult with markdown_text populated.
        """
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            return MinerUTaskResult(
                task_id="", state="failed",
                error_message=f"File not found: {file_path}",
            )

        # Verify mineru CLI is available
        if not shutil.which("mineru"):
            return MinerUTaskResult(
                task_id="", state="failed",
                error_message=(
                    "mineru CLI not found. Install with: "
                    "pip install 'mineru[all]'"
                ),
            )

        if output_dir is None:
            import tempfile
            tmp = tempfile.mkdtemp(prefix="paper2md_mineru_")
            output_dir = Path(tmp)
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Normalize paths for cross-platform
        file_str = str(file_path).replace("\\", "/")
        out_str = str(output_dir).replace("\\", "/")

        # Build mineru CLI command (v3.4+)
        cmd = [
            "mineru",
            "-p", file_str,
            "-o", out_str,
            "-b", self.backend,
            "-m", self.method,
        ]

        # Optional flags
        if self.language and self.language != "en":
            cmd.extend(["-l", self.language])
        if not self.enable_formula:
            cmd.extend(["-f", "false"])
        if not self.enable_table:
            cmd.extend(["-t", "false"])

        logger.info("Running mineru: %s", " ".join(cmd))

        try:
            import asyncio
            loop = asyncio.get_running_loop()

            # mineru starts a temporary local API service — give it more time
            timeout = 1800  # 30 min for large docs

            proc = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True, text=True, timeout=timeout,
                    env=os.environ,
                ),
            )

            if proc.returncode != 0:
                stderr_tail = (proc.stderr or "")[-500:]
                logger.error("mineru failed (code %d): %s", proc.returncode, stderr_tail)
                return MinerUTaskResult(
                    task_id="", state="failed",
                    error_message=(
                        f"mineru CLI failed (code {proc.returncode}). "
                        f"{stderr_tail[:200]}"
                    ),
                )

            logger.info("mineru completed successfully")

            # mineru v3.4 output structure:
            #   <output_dir>/<stem>/<method>/<stem>.md
            #   <output_dir>/<stem>/<method>/images/*.jpg
            md_text, images = self._find_output(output_dir, file_path)

            return MinerUTaskResult(
                task_id=f"local_{file_path.stem}",
                state="done",
                markdown_text=md_text,
                images=images,
            )

        except subprocess.TimeoutExpired:
            return MinerUTaskResult(
                task_id="", state="failed",
                error_message="mineru CLI timed out (30 min limit)",
            )
        except Exception as e:
            logger.exception("mineru execution error")
            return MinerUTaskResult(
                task_id="", state="failed",
                error_message=f"mineru error: {e}",
            )

    def _find_output(
        self, output_dir: Path, source_file: Path
    ) -> tuple[str, dict[str, bytes]]:
        """Locate markdown and images from mineru's output.

        mineru v3.4 creates: <output_dir>/<stem>/<method>/<stem>.md
        with images in: <output_dir>/<stem>/<method>/images/
        """
        best_md = ""
        images: dict[str, bytes] = {}

        for root, _dirs, files in os.walk(str(output_dir)):
            for f in files:
                path = Path(root) / f
                if f.endswith(".md"):
                    try:
                        content = path.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        continue
                    if len(content.strip()) > len(best_md):
                        best_md = content.strip()
                elif f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")):
                    # Deduplicate by filename
                    if f not in images:
                        images[f] = path.read_bytes()

        return best_md, images
