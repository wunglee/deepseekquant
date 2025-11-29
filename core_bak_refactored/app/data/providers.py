from typing import Any, Callable, Dict, List, Optional
from datetime import datetime

from core_bak_refactored.core.data.data_fetcher import MarketData

class CustomProviderAdapter:
    """应用层数据源适配器（自定义）
    - 用途：将任意自定义函数适配为统一的获取接口
    - 约束：不设默认策略；行为由注入的函数决定
    """
    def __init__(self, fetch_fn: Callable[[str, str, str, str, bool], Any]) -> None:
        self._fetch_fn = fetch_fn

    async def fetch(self, symbol: str, period: str, interval: str, data_type: str, adjustments: bool) -> Optional[List[MarketData]]:
        res = await self._fetch_fn(symbol, period, interval, data_type, adjustments)
        return res
