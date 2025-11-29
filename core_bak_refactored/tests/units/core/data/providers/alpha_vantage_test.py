import pytest
from datetime import datetime

from core_bak_refactored.core.data.providers.alpha_vantage import AlphaVantageProvider
from core_bak_refactored.core.data.data_fetcher import MarketData


@pytest.mark.asyncio
async def test_alpha_vantage_provider_with_injected_fetch():
    async def mock_fetch(symbol, period, interval, data_type, adjustments):
        return [MarketData(symbol=symbol, timestamp=datetime(2024,1,1), open=1, high=2, low=0.5, close=1.5, volume=10)]

    provider = AlphaVantageProvider(fetch_fn=mock_fetch)
    res = await provider.fetch('MSFT', '1y', '1d', 'ohlcv', True)
    assert isinstance(res, list) and res[0].symbol == 'MSFT'
