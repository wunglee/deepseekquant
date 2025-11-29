# 应用层数据服务（DataService）

from typing import Any, Dict, List, Optional, Callable

# 领域层依赖
from core_bak_refactored.core.data.data_fetcher import DataFetcher, MarketData


class DataService:
    """应用层数据门面服务
    
    职责：
    - 为应用层提供统一的数据接口（历史/实时/基本面）
    - 通过依赖注入委派到领域层 `DataFetcher`
    - 不引入业务默认值；遵循配置与专家规则（见 .qoder/rules/PECIFICATIONS.md）
    """

    def __init__(self, config: Dict[str, Any], custom_sources: Optional[Dict[str, Callable]] = None) -> None:
        # 保持轻量：直接委派至领域层 DataFetcher
        self.config = config
        self.custom_sources = custom_sources or {}
        self._fetcher = DataFetcher(config=self.config, custom_sources=self.custom_sources)

    async def get_historical_data(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
        data_type: str = "ohlcv",
        adjustments: bool = True,
    ) -> Dict[str, List[MarketData]]:
        """获取历史数据（委派至领域层）"""
        return await self._fetcher.get_historical_data(
            symbols=symbols,
            period=period,
            interval=interval,
            data_type=data_type,
            adjustments=adjustments,
        )

    async def get_real_time_data(self, symbols: List[str], data_types: Optional[List[str]] = None) -> Dict[str, MarketData]:
        """获取实时数据（委派至领域层）"""
        return await self._fetcher.get_real_time_data(symbols=symbols, data_types=data_types)

    async def get_fundamental_data(self, symbol: str) -> Dict[str, Any]:
        """获取基本面数据（委派至领域层）"""
        return await self._fetcher.get_fundamental_data(symbol=symbol)

    async def cleanup(self) -> None:
        """释放领域层资源"""
        # 领域层 DataFetcher 暂未提供显式关闭接口；保留占位以兼容未来扩展
        # 如需要，可在 DataFetcher 中实现关闭 aiohttp_session/requests_session 的清理方法
        try:
            if hasattr(self._fetcher, "aiohttp_session"):
                await self._fetcher.aiohttp_session.close()
        except Exception:
            pass
