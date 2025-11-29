from typing import Any, Dict, Optional

class CoreCacheService:
    """领域层缓存子系统（轻量占位实现）
    - 职责：为领域层提供可替换的缓存接口
    - 约束：不设默认策略与时效；由上层配置驱动
    """
    def __init__(self) -> None:
        self._mem: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        return self._mem.get(key)

    def set(self, key: str, value: Any) -> None:
        self._mem[key] = value

    def clear(self) -> None:
        self._mem.clear()
