"""MinerU document parsing integration.

Two modes:
- Cloud API: Precision extract (with API token) or agent lightweight (no token)
- Local GPU: mineru CLI v3.4+ with hybrid-engine / pipeline backends
"""

from .client import MinerUClient

__all__ = ["MinerUClient"]
