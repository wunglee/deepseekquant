from typing import Any, Optional

from cachetools import LRUCache


class LRUCacheWrapper:
    """LRU缓存封装（职责单一：读写），使用 cachetools.LRUCache"""

    def __init__(self, maxsize: int = 128) -> None:
        self._cache = LRUCache(maxsize=maxsize)

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value
