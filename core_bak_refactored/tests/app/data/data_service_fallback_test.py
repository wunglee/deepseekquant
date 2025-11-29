import pytest
from datetime import datetime
from typing import List

from core_bak_refactored.app.data.data_service import DataService
from core_bak_refactored.core.data.data_fetcher import MarketData
from core_bak_refactored.core.share.market_enums import MarketCode


@pytest.mark.asyncio
async def test_data_service_fallback_to_custom_source():
    """主数据源失败时，回退到备用自定义数据源"""

    async def fail_primary(symbol: str, period: str, interval: str, data_type: str, adjustments: bool):
        return None

    async def ok_fallback(symbol: str, period: str, interval: str, data_type: str, adjustments: bool) -> List[MarketData]:
        return [
            MarketData(
                symbol=symbol,
                timestamp=datetime(2024, 1, 2),
                open=200.0,
                high=205.0,
                low=198.0,
                close=203.0,
                volume=2_000_000,
                metadata={'market_type': MarketCode.US, 'data_source': 'custom_ok'}
            )
        ]

    custom_sources = {
        'fail_primary': fail_primary,
        'custom_ok': ok_fallback
    }

    config = {
        'cache_enabled': False,
        'primary': 'fail_primary',
        'fallback_sources': ['custom_ok']
    }

    svc = DataService(config=config, custom_sources=custom_sources)
    result = await svc.get_historical_data(symbols=['MSFT'], period='1mo', interval='1d', data_type='ohlcv')

    assert 'MSFT' in result
    assert result['MSFT'][0].metadata.get('data_source') == 'custom_ok'
    assert result['MSFT'][0].metadata.get('market_type') == MarketCode.US

    await svc.cleanup()


@pytest.mark.asyncio
async def test_data_service_all_sources_fail_returns_empty():
    """主数据源与备用数据源均失败时应返回空结果集"""

    async def fail_one(symbol: str, period: str, interval: str, data_type: str, adjustments: bool):
        return None

    async def fail_two(symbol: str, period: str, interval: str, data_type: str, adjustments: bool):
        raise RuntimeError("fallback failed")

    custom_sources = {
        'fail_src': fail_one,
        'fail2': fail_two
    }

    config = {
        'cache_enabled': False,
        'primary': 'fail_src',
        'fallback_sources': ['fail2']
    }

    svc = DataService(config=config, custom_sources=custom_sources)
    result = await svc.get_historical_data(symbols=['TSLA'], period='1mo', interval='1d', data_type='ohlcv')

    assert isinstance(result, dict)
    assert 'TSLA' not in result or len(result.get('TSLA', [])) == 0

    await svc.cleanup()
