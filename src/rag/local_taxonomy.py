"""Local taxonomy provider.

Loads category data from the repo-managed ``data/categories.json``.

Expected JSON shape::

    [
      {
        "categoryName": "BTC",
        "wordsAssociatedWithCategory": ["BTC", "Bitcoin", "bitcoin"],
        "includedPhrases": ["BITCOIN NETWORK"]
      },
      ...
    ]

This shape is what :meth:`CategoryProcessor.process_api_categories` expects.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

from src.logger.logger import Logger


class LocalTaxonomyProvider:
    """Serves category taxonomy from a local JSON file.

    Parameters
    ----------
    logger:
        Logger instance.
    categories_file:
        Absolute path to the categories JSON file.  If *None*, the default
        ``data/categories.json`` relative to the project root is used.
    """

    def __init__(
        self,
        logger: Logger,
        categories_file: str | None = None,
    ) -> None:
        self.logger = logger
        self._categories_file = categories_file or self._default_categories_path()
        self._cache: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_categories(
        self,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Return the list of category dicts from local taxonomy JSON.

        Results are cached in memory; pass ``force_refresh=True`` to reload
        from disk.
        """
        if self._cache is not None and not force_refresh:
            return self._cache

        categories = self._load_from_file()
        if categories:
            self._cache = categories
            self.logger.debug(
                "LocalTaxonomyProvider: loaded %d categories from %s",
                len(categories),
                self._categories_file,
            )
        else:
            self.logger.warning(
                "LocalTaxonomyProvider: no categories found in %s",
                self._categories_file,
            )

        return self._cache or []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_from_file(self) -> list[dict[str, Any]]:
        """Load and parse the categories JSON file."""
        if not os.path.exists(self._categories_file):
            self.logger.error(
                "LocalTaxonomyProvider: categories file not found: %s",
                self._categories_file,
            )
            return []
        try:
            with open(self._categories_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # data may be {"timestamp": ..., "categories": [...]} or a plain list
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "categories" in data:
                return data["categories"]
            self.logger.error(
                "LocalTaxonomyProvider: unexpected JSON shape in %s",
                self._categories_file,
            )
            return []
        except Exception as exc:
            self.logger.error(
                "LocalTaxonomyProvider: error reading %s: %s",
                self._categories_file,
                exc,
            )
            return []

    @staticmethod
    def _default_categories_path() -> str:
        """Resolve the default ``data/categories.json`` path."""
        if getattr(sys, "frozen", False):
            base = os.path.dirname(sys.executable)
        else:
            # __file__ is src/rag/local_taxonomy.py → up 3 levels
            base = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        return os.path.join(base, "data", "categories.json")
