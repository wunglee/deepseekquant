import pytest
from datetime import datetime
from typing import List

from core_bak_refactored.app.data.cache_service import CacheService
from core_bak_refactored.app.data.providers import CustomProviderAdapter
from core_bak_refactored.app.data.quality_monitor import BasicQualityMonitor
from core_bak_refactored.app.data.models import MarketData
from core_bak_refactored.core.share.market_enums import MarketCode


@pytest.mark.asyncio
async def test_app_modules_structure_and_basic_behaviors():
    cache = CacheService()
    cache.set('k', 123)
    assert cache.get('k') == 123
    cache.clear()
    assert cache.get('k') is None

    async def fetch_fn(symbol: str, period: str, interval: str, data_type: str, adjustments: bool) -> List[MarketData]:
        return [
            MarketData(
                symbol=symbol,
                timestamp=datetime(2024, 2, 1),
                open=10.0, high=10.5, low=9.5, close=10.3, volume=1000,
                metadata={'market_type': MarketCode.SG}
            )
        ]

    adapter = CustomProviderAdapter(fetch_fn)
    data = await adapter.fetch('D05.SI', '1mo', '1d', 'ohlcv', True)
    assert data and data[0].metadata['market_type'] == MarketCode.SG

    qm = BasicQualityMonitor()
    report = qm.assess(data)
    assert 'overall_score' in report and 'dimension_scores' in report
