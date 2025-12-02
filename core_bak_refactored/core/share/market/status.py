from typing import Any, Dict

from core_bak_refactored.core.data.fetcher_orchestrator import DataFetcherOrchestrator


async def get_market_status(orchestrator: DataFetcherOrchestrator) -> Dict[str, Any]:
    """读取市场状态（职责单一：聚合读取）
    
    Args:
        orchestrator: 数据获取编排器（包含MarketStatusService）
    
    Returns:
        市场状态字典（包含开盘状态、VIX、板块表现等）
    """
    return await orchestrator.get_market_status()
