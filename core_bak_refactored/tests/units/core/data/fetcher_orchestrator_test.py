import pytest
from datetime import datetime

from core_bak_refactored.core.data.fetcher_orchestrator import DataFetcherOrchestrator
from core_bak_refactored.core.data.data_fetcher import MarketData
from core_bak_refactored.core.share.market_enums import MarketCode


@pytest.mark.asyncio
async def test_fetcher_orchestrator_delegates_with_custom_source():
    async def mock_fetch(symbol, period, interval, data_type, adjustments):
        return [MarketData(symbol=symbol, timestamp=datetime(2024,1,1), open=1, high=2, low=0.5, close=1.5, volume=10,
                           metadata={'market_type': MarketCode.US.value})]

    config = {'primary': 'yahoo', 'fallback_sources': [], 'cache_enabled': False}
    orch = DataFetcherOrchestrator(config, custom_sources={'yahoo': mock_fetch})
    res = await orch.get_historical_data(['AAPL'], '1y', '1d', 'ohlcv', True)
    assert 'AAPL' in res and len(res['AAPL']) == 1
