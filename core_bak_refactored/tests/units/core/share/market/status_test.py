import pytest
from core_bak_refactored.core.share.market.status import get_market_status
from core_bak_refactored.core.data.fetcher_orchestrator import DataFetcherOrchestrator


@pytest.mark.asyncio
async def test_market_status_reading():
    orchestrator = DataFetcherOrchestrator({'primary': 'alpha_vantage', 'cache_enabled': False})
    status = await get_market_status(orchestrator)
    assert isinstance(status, dict) and 'timestamp' in status
