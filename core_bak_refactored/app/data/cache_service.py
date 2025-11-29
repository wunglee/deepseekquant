from typing import Any, Dict, Optional

class CacheService:
    """应用层缓存子系统（轻量门面）
    - 职责：为应用层数据服务提供可替换的缓存接口（保持轻量）
    - 约束：不引入业务默认值；策略与时效由上层配置/专家规则决定
    """

    def __init__(self) -> None:
        self._mem: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        return self._mem.get(key)

    def set(self, key: str, value: Any) -> None:
        self._mem[key] = value

    def clear(self) -> None:
        self._mem.clear()
