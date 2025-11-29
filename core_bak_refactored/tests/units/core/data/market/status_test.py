import pytest
from core_bak_refactored.core.data.market.status import get_market_status
from core_bak_refactored.core.data.data_fetcher import DataFetcher


@pytest.mark.asyncio
async def test_market_status_reading():
    fetcher = DataFetcher({'primary': 'yahoo', 'cache_enabled': False})
    status = await get_market_status(fetcher)
    assert isinstance(status, dict) and 'timestamp' in status
