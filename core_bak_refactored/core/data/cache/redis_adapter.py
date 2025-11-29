from typing import Any, Optional


class RedisCacheAdapter:
    """Redis缓存适配器（职责单一：接口占位）
    - 为避免引入外部连接，此处提供最小占位实现，便于测试
    - 后续可替换为真实Redis客户端
    """

    def __init__(self) -> None:
        self._store = {}

    def get(self, key: str) -> Optional[bytes]:
        return self._store.get(key)

    def setex(self, key: str, ttl: int, value: bytes) -> None:
        # 仅模拟写入，不实现TTL
        self._store[key] = value
