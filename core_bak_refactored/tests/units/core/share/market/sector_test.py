"""
板块表现模块单元测试

测试路径同构：
- 被测试代码：core_bak_refactored/core/share/market/sector.py
- 测试代码：core_bak_refactored/tests/units/core/share/market/sector_test.py

测试范围：
1. get_sector_performance() - 获取板块表现数据
2. calculate_daily_volatility() - 计算日波动率
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from core_bak_refactored.core.share.market.sector import get_sector_performance, calculate_daily_volatility


@pytest.mark.asyncio
async def test_get_sector_performance_success():
    """测试成功获取板块表现数据"""
    # Mock fetcher
    mock_fetcher = MagicMock()
    mock_data = {
        'XLK': [
            {'close': 100, 'volume': 1000000},
            {'close': 102, 'volume': 1100000}
        ],
        'XLV': [
            {'close': 50, 'volume': 500000},
            {'close': 49, 'volume': 510000}
        ]
    }
    mock_fetcher.get_historical_data = AsyncMock(return_value=mock_data)
    
    result = await get_sector_performance(mock_fetcher)
    
    # 验证调用
    mock_fetcher.get_historical_data.assert_called_once()
    
    # 验证结果
    assert 'Technology' in result
    assert 'Healthcare' in result
    assert result['Technology']['daily_return'] == pytest.approx(2.0, rel=1e-2)
    assert result['Technology']['current_price'] == 102
    assert result['Healthcare']['daily_return'] == pytest.approx(-2.0, rel=1e-2)


@pytest.mark.asyncio
async def test_get_sector_performance_empty_data():
    """测试空数据情况"""
    mock_fetcher = MagicMock()
    mock_fetcher.get_historical_data = AsyncMock(return_value={})
    
    result = await get_sector_performance(mock_fetcher)
    
    assert result == {}


@pytest.mark.asyncio
async def test_get_sector_performance_exception():
    """测试异常处理"""
    mock_fetcher = MagicMock()
    mock_fetcher.get_historical_data = AsyncMock(side_effect=Exception("API Error"))
    
    result = await get_sector_performance(mock_fetcher)
    
    assert result == {}


def test_calculate_daily_volatility_success():
    """测试成功计算波动率"""
    # 模拟5天收盘价数据
    data = [
        {'close': 100},
        {'close': 102},
        {'close': 101},
        {'close': 103},
        {'close': 104}
    ]
    
    volatility = calculate_daily_volatility(data)
    
    # 验证返回值大于0且小于1（年化波动率合理范围）
    assert 0 < volatility < 1


def test_calculate_daily_volatility_with_objects():
    """测试使用对象属性的波动率计算"""
    # 模拟MarketData对象
    class MockMarketData:
        def __init__(self, close):
            self.close = close
    
    data = [
        MockMarketData(100),
        MockMarketData(102),
        MockMarketData(101),
        MockMarketData(103),
        MockMarketData(104)
    ]
    
    volatility = calculate_daily_volatility(data)
    
    assert 0 < volatility < 1


def test_calculate_daily_volatility_insufficient_data():
    """测试数据不足的情况"""
    data = [{'close': 100}]
    
    volatility = calculate_daily_volatility(data)
    
    assert volatility == 0.0


def test_calculate_daily_volatility_empty_data():
    """测试空数据"""
    data = []
    
    volatility = calculate_daily_volatility(data)
    
    assert volatility == 0.0


def test_calculate_daily_volatility_zero_prices():
    """测试零价格数据"""
    data = [
        {'close': 0},
        {'close': 0}
    ]
    
    volatility = calculate_daily_volatility(data)
    
    assert volatility == 0.0


def test_calculate_daily_volatility_invalid_data():
    """测试无效数据格式"""
    data = [
        "invalid",
        123
    ]
    
    volatility = calculate_daily_volatility(data)
    
    assert volatility == 0.0
