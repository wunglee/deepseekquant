"""MarketStatusService 单元测试"""
import pytest
from unittest.mock import AsyncMock
from core_bak_refactored.core.share.market.market_status_service import MarketStatusService


@pytest.mark.asyncio
async def test_market_status_service():
    """测试市场状态服务"""
    # Mock历史数据获取器
    mock_fetcher = AsyncMock()
    mock_fetcher.get_historical_data = AsyncMock(return_value={})
    
    service = MarketStatusService(historical_data_fetcher=mock_fetcher)
    
    # 获取市场状态
    status = await service.get_market_status()
    
    assert isinstance(status, dict)
    assert 'market_open' in status
    assert 'timestamp' in status
    assert 'liquidity_conditions' in status
    assert 'volatility_regime' in status
    assert 'market_sentiment' in status
