from typing import Any, Dict

from core_bak_refactored.core.data.data_fetcher import DataFetcher


async def get_market_status(fetcher: DataFetcher) -> Dict[str, Any]:
    """读取市场状态（职责单一：聚合读取）"""
    return await fetcher.get_market_status()
