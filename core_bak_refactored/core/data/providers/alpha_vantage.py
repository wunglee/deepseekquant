from typing import Any, Callable, Dict, List, Optional

from core_bak_refactored.core.data.data_fetcher import DataFetcher, MarketData


class AlphaVantageProvider:
    """Alpha Vantage 数据源适配器（职责单一：从AV获取）
    - 默认委派到原始 DataFetcher 的 `_fetch_alpha_vantage_data`
    - 支持注入 `fetch_fn` 以便测试覆盖
    """

    def __init__(self, fetcher: Optional[DataFetcher] = None, fetch_fn: Optional[Callable[[str, str, str, str, bool], Any]] = None) -> None:
        self._fetcher = fetcher
        self._fetch_fn = fetch_fn

    async def fetch(self, symbol: str, period: str, interval: str, data_type: str, adjustments: bool) -> Optional[List[MarketData]]:
        if self._fetch_fn is not None:
            res = await self._fetch_fn(symbol, period, interval, data_type, adjustments)
            return res
        if self._fetcher is None:
            return None
        method = getattr(self._fetcher, '_fetch_alpha_vantage_data', None)
        if callable(method):
            return await method(symbol, period, interval, data_type, adjustments)
        return None
