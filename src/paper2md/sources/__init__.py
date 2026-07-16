"""Paper source extractors — auto-discovered by SourceRegistry.

To add a new paper source (e.g. PubMed, IEEE):
1. Create a new .py file in this directory
2. Subclass ``BaseExtractor`` and implement all abstract methods
3. The registry discovers it automatically — no other files need changes
"""

from .base import BaseExtractor
from .registry import SourceRegistry

__all__ = ["BaseExtractor", "SourceRegistry"]
