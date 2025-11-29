import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime

from core_bak_refactored.core.data.movable_fetcher import MovableDataFetcher
from core_bak_refactored.core.data.data_fetcher import MarketData
from core_bak_refactored.core.share.market_enums import MarketCode


@pytest.mark.asyncio
async def test_movable_fetcher_delegates_properly():
    """验证可动的拆卸版本正确委派到原始 DataFetcher"""
    config = {
        'cache_enabled': False,
        'primary_source': 'yahoo',
        'fallback_sources': [],
        'request_timeout': 10,
    }
    mf = MovableDataFetcher(config)
    assert mf._orig is not None
    assert hasattr(mf._orig, 'get_historical_data')


@pytest.mark.asyncio
async def test_movable_fetcher_private_methods_accessible():
    """验证私有方法可委派"""
    config = {'cache_enabled': False}
    mf = MovableDataFetcher(config)
    assert callable(mf._fetch_yahoo_data)
    assert callable(mf._fetch_alpha_vantage_data)
    assert callable(mf._fetch_symbol_data)


@pytest.mark.asyncio
async def test_movable_fetcher_get_data_quality_metrics():
    """验证质量指标方法委派"""
    config = {'cache_enabled': False}
    mf = MovableDataFetcher(config)
    result = mf.get_data_quality_metrics()
    assert isinstance(result, dict)
