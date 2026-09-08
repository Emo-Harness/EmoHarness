"""Content-addressed Target-call cache (ADR-0006).

Every Target call is keyed by model id, decoding parameters, prompt text, and
media content hashes rather than media paths, so identical media reached via
different paths (or a Builder round that only edits a prompt) hit the same
cache entry. ``CachedTarget`` wraps any ``TargetModel`` with get-or-call-and-put
semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from emoharness.common.hashing import sha256_file_cached, stable_hash
from emoharness.common.io import read_json, write_json_atomic
from emoharness.target.base import TargetModel, TargetRequest, TargetResponse


def request_key(model_id: str, request: TargetRequest) -> str:
    """Stable cache key for one call: media paths are replaced by content hashes."""
    video = request.video_path
    payload: dict[str, Any] = {
        'model_id': model_id,
        'prompt': request.prompt,
        'tag': request.tag,
        'system': request.system,
        'video_hash': sha256_file_cached(Path(video)) if video else None,
        'image_hashes': [sha256_file_cached(Path(p)) for p in request.image_paths],
        'json_schema': request.json_schema,
        'max_tokens': request.max_tokens,
        'temperature': request.temperature,
        'top_p': request.top_p,
        'seed': request.seed,
    }
    return stable_hash(payload)


class CallCache:
    """On-disk ``TargetResponse`` cache, sharded by the first two key characters."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _path_for(self, key: str) -> Path:
        return self._root / key[:2] / f'{key}.json'

    def get(self, key: str) -> TargetResponse | None:
        """Return the cached response for ``key``, marked ``cached=True``, or None."""
        path = self._path_for(key)
        if not path.exists():
            return None
        payload = read_json(path)
        return TargetResponse.model_validate({**payload, 'cached': True})

    def put(self, key: str, response: TargetResponse) -> None:
        """Persist ``response`` under ``key``."""
        write_json_atomic(self._path_for(key), response.model_dump(mode='json'))


class CachedTarget:
    """Wraps a ``TargetModel`` with a get-or-call-and-put cache."""

    def __init__(self, inner: TargetModel, cache: CallCache) -> None:
        self._inner = inner
        self._cache = cache
        self._hits = 0
        self._misses = 0

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def generate(self, request: TargetRequest) -> TargetResponse:
        """Return the cached response for ``request`` if present, else call and cache."""
        key = request_key(self._inner.model_id, request)
        cached = self._cache.get(key)
        if cached is not None:
            self._hits += 1
            return cached
        self._misses += 1
        response = self._inner.generate(request)
        self._cache.put(key, response)
        return response
