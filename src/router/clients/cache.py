"""On-disk response cache for resumable evaluation.

Stores ``ModelResponse`` objects as JSON files keyed by a hash of the
``CacheKey`` (model + prompt + parameters). This allows evaluation runs
to be interrupted and resumed without re-querying models for prompts
that have already been answered.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from router.models import CacheKey, ModelResponse

logger = logging.getLogger(__name__)


class ResponseCache:
    """File-backed response cache keyed by ``CacheKey`` hashes.

    Each cached response is stored as ``{hash}.json`` in the cache directory.

    Args:
        cache_dir: Directory to store cached response files. Created
            automatically if it doesn't exist.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: CacheKey) -> ModelResponse | None:
        """Retrieve a cached response, or ``None`` if not present.

        Args:
            key: The cache lookup key.

        Returns:
            The cached ``ModelResponse``, or ``None`` on cache miss.
        """
        path = self._key_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ModelResponse(**data)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            logger.warning("Corrupt cache entry %s: %s — removing", path.name, exc)
            path.unlink(missing_ok=True)
            return None

    def put(self, key: CacheKey, response: ModelResponse) -> None:
        """Store a response in the cache.

        Overwrites any existing entry for the same key.

        Args:
            key: The cache key identifying this request.
            response: The model response to cache.
        """
        path = self._key_path(key)
        path.write_text(response.model_dump_json(indent=2), encoding="utf-8")

    def has(self, key: CacheKey) -> bool:
        """Check whether a cache entry exists for the given key.

        Args:
            key: The cache lookup key.

        Returns:
            ``True`` if a cached response file exists.
        """
        return self._key_path(key).exists()

    @property
    def size(self) -> int:
        """Return the number of cached entries."""
        return len(list(self._cache_dir.glob("*.json")))

    def clear(self) -> int:
        """Remove all cached entries.

        Returns:
            The number of entries that were deleted.
        """
        entries = list(self._cache_dir.glob("*.json"))
        for entry in entries:
            entry.unlink(missing_ok=True)
        logger.info("Cleared %d cache entries from %s", len(entries), self._cache_dir)
        return len(entries)

    def _key_path(self, key: CacheKey) -> Path:
        """Compute the filesystem path for a cache key."""
        return self._cache_dir / f"{key.to_hash()}.json"

    def __repr__(self) -> str:
        return f"ResponseCache(dir={self._cache_dir!r}, size={self.size})"
