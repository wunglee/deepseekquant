from typing import Any, Callable, Dict, List, Optional

from core_bak_refactored.core.data.data_fetcher import DataFetcher


async def stream(fetcher: DataFetcher, symbols: List[str], callback: Callable[[Dict], None],
                 data_types: Optional[List[str]] = None) -> None:
    """实时流封装（职责单一：委派流式）"""
    await fetcher.stream_real_time_data(symbols, callback, data_types or ['quote', 'trade', 'summary'])
