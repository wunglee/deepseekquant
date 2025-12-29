"""
市场广度模块单元测试

测试路径同构：
- 被测试代码：core_bak_refactored/core/share/market/breadth.py
- 测试代码：core_bak_refactored/tests/units/core/share/market/breadth_test.py

测试范围：
1. get_advance_decline() - 获取涨跌家数
2. _get_default_symbols() - 获取默认股票列表
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from core_bak_refactored.core.share.market.breadth import get_advance_decline, _get_default_symbols


@pytest.mark.asyncio
async def test_get_advance_decline_success():
    """测试成功获取涨跌家数"""
    # Mock fetcher
    mock_fetcher = MagicMock()
    mock_data = {
        'AAPL': {'change': 2.5},
        'MSFT': {'change': -1.2},
        'GOOGL': {'change': 0.8},
        'AMZN': {'change': 0},
        'META': {'change': -0.5}
    }
    mock_fetcher.get_real_time_data = AsyncMock(return_value=mock_data)
    
    result = await get_advance_decline(mock_fetcher, ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'])
    
    assert result['advances'] == 2  # AAPL, GOOGL
    assert result['declines'] == 2  # MSFT, META
    assert result['unchanged'] == 1  # AMZN
    assert result['total_issues'] == 5
    assert result['advance_decline_ratio'] == 1.0  # 2/2
    assert 'timestamp' in result


@pytest.mark.asyncio
async def test_get_advance_decline_default_symbols():
    """测试使用默认符号"""
    mock_fetcher = MagicMock()
    mock_data = {
        'AAPL': {'change': 1.0},
        'MSFT': {'change': -1.0}
    }
    mock_fetcher.get_real_time_data = AsyncMock(return_value=mock_data)
    
    result = await get_advance_decline(mock_fetcher, symbols=None)
    
    # 应该返回合理的统计数据
    assert 'advances' in result
    assert 'declines' in result
    assert 'total_issues' in result


@pytest.mark.asyncio
async def test_get_advance_decline_exception():
    """测试异常处理"""
    mock_fetcher = MagicMock()
    mock_fetcher.get_real_time_data = AsyncMock(side_effect=Exception("API error"))
    
    result = await get_advance_decline(mock_fetcher, ['AAPL'])
    
    assert result['advances'] == 0
    assert result['declines'] == 0
    assert result['unchanged'] == 0
    assert 'error' in result


def test_get_default_symbols():
    """测试获取默认符号列表"""
    mock_fetcher = MagicMock()
    symbols = _get_default_symbols(mock_fetcher)
    
    assert isinstance(symbols, list)
    assert len(symbols) > 0
    assert 'AAPL' in symbols
