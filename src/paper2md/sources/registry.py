"""Source registry — auto-discovers paper source extractors at runtime.

To add a new paper source:
1. Create a new file in this directory (e.g. ``pubmed.py``)
2. Define a class that subclasses ``BaseExtractor``
3. That's it — the registry finds it automatically

Uses Python introspection to walk the ``paper2md.sources`` package
and register every concrete ``BaseExtractor`` subclass.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseExtractor
    from ..models import ConvertConfig

logger = logging.getLogger(__name__)


class SourceRegistry:
    """Auto-discovering registry for paper source extractors.

    Walks the ``paper2md.sources`` package and registers every
    concrete subclass of ``BaseExtractor``.
    """

    _extractors: dict[str, type[BaseExtractor]] = {}
    _instances: dict[str, BaseExtractor] = {}
    _initialized: bool = False

    @classmethod
    def discover(cls) -> None:
        """Scan the sources package for BaseExtractor subclasses.

        Called automatically on first registry access.
        """
        if cls._initialized:
            return

        import paper2md.sources as sources_pkg

        pkg_path = Path(sources_pkg.__path__[0])  # type: ignore[attr-defined]

        for _, module_name, _is_pkg in pkgutil.iter_modules(
            [str(pkg_path)], prefix="paper2md.sources.",
        ):
            if module_name.endswith(".__") or module_name.split(".")[-1].startswith("_"):
                continue

            try:
                module = importlib.import_module(module_name)

                for _name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseExtractor)
                        and obj is not BaseExtractor
                        and not inspect.isabstract(obj)
                    ):
                        instance = obj()
                        key = instance.source_type
                        cls._extractors[key] = obj
                        cls._instances[key] = instance
            except (ImportError, ModuleNotFoundError) as exc:
                logger.debug("Skipping source module %s: %s", module_name, exc)

        cls._initialized = True

    @classmethod
    def get_extractor(cls, source_type: str) -> BaseExtractor | None:
        """Get an extractor instance by source type string (e.g. 'arxiv')."""
        cls.discover()
        return cls._instances.get(source_type)

    @classmethod
    async def find_extractor(
        cls, config: ConvertConfig
    ) -> BaseExtractor | None:
        """Find the first extractor that can handle the given source.

        Returns None if no registered extractor matches.
        """
        cls.discover()
        for instance in cls._instances.values():
            try:
                if await instance.can_handle(config):
                    return instance
            except Exception:
                continue
        return None

    @classmethod
    def list_sources(cls) -> list[dict]:
        """Return metadata for all registered extractors.

        Each dict has keys: type, name, description.
        """
        cls.discover()
        return [
            {
                "type": key,
                "name": instance.display_name,
                "description": instance.__class__.__doc__ or "",
            }
            for key, instance in cls._instances.items()
        ]


# Late import for type checking inside discover()
from .base import BaseExtractor  # noqa: E402
