import pytest

from core_bak_refactored.core.data.realtime.stream import stream
from core_bak_refactored.core.data.fetcher_orchestrator import DataFetcherOrchestrator


@pytest.mark.asyncio
async def test_realtime_stream_delegate():
    def cb(_):
        pass
    orchestrator = DataFetcherOrchestrator({'primary': 'alpha_vantage', 'cache_enabled': False})
    # 仅验证调用路径存在，不验证实际网络流
    await stream(orchestrator, ['AAPL'], cb, ['quote'])
