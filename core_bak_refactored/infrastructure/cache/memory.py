from typing import Any, Dict, Optional


class MemoryTTLCache:
    """内存TTL缓存（职责单一：读写），最小实现用于测试"""

    def __init__(self) -> None:
        self._mem: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        return self._mem.get(key)

    def set(self, key: str, value: Any) -> None:
        self._mem[key] = value
