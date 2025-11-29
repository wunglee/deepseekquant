from typing import Any, Callable, Dict, List, Optional

from core_bak_refactored.core.data.data_fetcher import DataFetcher, MarketData


class DataFetcherOrchestrator:
    """数据获取入口编排器（职责单一：委派与编排）
    - 仅负责装配与委派，不承载具体数据源逻辑
    - 不修改原始 DataFetcher，实现与行为以其为准
    """

    def __init__(self, config: Dict[str, Any], custom_sources: Optional[Dict[str, Callable]] = None) -> None:
        self._fetcher = DataFetcher(config=config, custom_sources=custom_sources or {})

    async def get_historical_data(self, symbols: List[str], period: str = '1y', interval: str = '1d',
                                  data_type: str = 'ohlcv', adjustments: bool = True) -> Dict[str, List[MarketData]]:
        return await self._fetcher.get_historical_data(symbols, period, interval, data_type, adjustments)

    async def get_real_time_data(self, symbols: List[str], data_types: Optional[List[str]] = None) -> Dict[str, MarketData]:
        return await self._fetcher.get_real_time_data(symbols, data_types or ['quote', 'trade', 'summary'])

    async def stream_real_time_data(self, symbols: List[str], callback: Callable[[Dict], None],
                                    data_types: Optional[List[str]] = None) -> None:
        await self._fetcher.stream_real_time_data(symbols, callback, data_types or ['quote', 'trade', 'summary'])

    async def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        return await self._fetcher.get_fundamental_data(symbol)

    async def get_market_status(self) -> Dict[str, Any]:
        return await self._fetcher.get_market_status()

    def get_data_quality_metrics(self) -> Dict[str, Any]:
        return self._fetcher.get_data_quality_metrics()
