from typing import Any, Callable, Dict, List, Optional

from core_bak_refactored.core.data.fetcher.fetcher_orchestrator import DataFetcherOrchestrator


async def stream(orchestrator: DataFetcherOrchestrator, symbols: List[str], callback: Callable[[Dict], None],
                 data_types: Optional[List[str]] = None) -> None:
    """实时流封装（职责单一：委派流式）
    
    Args:
        orchestrator: 数据获取编排器
        symbols: 股票代码列表
        callback: 回调函数
        data_types: 数据类型列表
    """
    await orchestrator.stream_real_time_data(symbols, callback, data_types or ['quote', 'trade', 'summary'])
