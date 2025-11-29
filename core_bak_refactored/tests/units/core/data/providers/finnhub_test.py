import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime
from core_bak_refactored.core.data.providers.finnhub import (
    fetch_finnhub_data,
    fetch_finnhub_quote,
    _map_interval_to_finnhub,
    _calculate_start_time
)
from .async_mock_helper import AsyncContextManager


class TestFetchFinnhubData:
    """测试Finnhub数据提供者。"""

    @pytest.mark.asyncio
    async def test_fetch_candle_data_success(self):
        """测试成功获取K线数据。"""
        mock_response_data = {
            's': 'ok',
            't': [1609459200, 1609545600],  # Unix时间戳
            'o': [150.0, 151.0],
            'h': [152.0, 153.0],
            'l': [149.0, 150.0],
            'c': [151.0, 152.0],
            'v': [1000000, 1100000]
        }
        
        # 创建 mock response 对象
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        # 使用AsyncContextManager包装
        mock_session = Mock()
        mock_session.get = Mock(return_value=AsyncContextManager(mock_response))
        
        credentials = {'api_key': 'test_key'}
        
        result = await fetch_finnhub_data(
            'AAPL', '1y', '1d', 'ohlcv', True, credentials, mock_session
        )
        
        assert result is not None, "Result should not be None"
        assert len(result) == 2, f"Expected 2 records, got {len(result)}"
        assert result[0]['symbol'] == 'AAPL'
        assert result[0]['open'] == 150.0
        assert result[0]['metadata']['data_source'] == 'finnhub'

    @pytest.mark.asyncio
    async def test_fetch_no_api_key(self):
        """测试缺少API密钥。"""
        credentials = {}
        mock_session = Mock()
        
        result = await fetch_finnhub_data(
            'AAPL', '1y', '1d', 'ohlcv', True, credentials, mock_session
        )
        
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_no_data_status(self):
        """测试无数据状态响应。"""
        mock_response = {'s': 'no_data'}
        
        mock_session = Mock()
        mock_session.get = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value.status = 200
        mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
        
        credentials = {'api_key': 'test_key'}
        
        result = await fetch_finnhub_data(
            'INVALID', '1y', '1d', 'ohlcv', True, credentials, mock_session
        )
        
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_empty_timestamps(self):
        """测试空时间戳响应。"""
        mock_response = {
            's': 'ok',
            't': [],
            'o': [],
            'h': [],
            'l': [],
            'c': [],
            'v': []
        }
        
        mock_session = Mock()
        mock_session.get = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value.status = 200
        mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
        
        credentials = {'api_key': 'test_key'}
        
        result = await fetch_finnhub_data(
            'AAPL', '1y', '1d', 'ohlcv', True, credentials, mock_session
        )
        
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_quote_success(self):
        """测试获取实时报价。"""
        mock_quote_data = {
            'c': 150.0,  # current price
            'd': 1.0,    # change
            'dp': 0.67,  # percent change
            'h': 152.0,  # high
            'l': 149.0,  # low
            'o': 149.5,  # open
            'pc': 149.0, # previous close
            't': 1609459200  # timestamp
        }
        
        # 创建 mock response 对象
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_quote_data)
        
        # 使用AsyncContextManager包装
        mock_session = Mock()
        mock_session.get = Mock(return_value=AsyncContextManager(mock_response))
        
        credentials = {'api_key': 'test_key'}
        
        result = await fetch_finnhub_quote('AAPL', credentials, mock_session)
        
        assert result is not None, "Result should not be None"
        assert result['symbol'] == 'AAPL'
        assert result['current_price'] == 150.0
        assert result['change'] == 1.0

    def test_map_interval_to_finnhub(self):
        """测试时间间隔映射。"""
        assert _map_interval_to_finnhub('1m') == '1'
        assert _map_interval_to_finnhub('5m') == '5'
        assert _map_interval_to_finnhub('15m') == '15'
        assert _map_interval_to_finnhub('1h') == '60'
        assert _map_interval_to_finnhub('1d') == 'D'
        assert _map_interval_to_finnhub('1wk') == 'W'
        assert _map_interval_to_finnhub('1mo') == 'M'
        assert _map_interval_to_finnhub('unknown') == 'D'  # 默认值

    def test_calculate_start_time(self):
        """测试开始时间计算。"""
        end_time = datetime(2024, 1, 1)
        
        # 1天
        start_time = _calculate_start_time(end_time, '1d')
        assert (end_time - start_time).days == 1
        
        # 1个月
        start_time = _calculate_start_time(end_time, '1mo')
        assert (end_time - start_time).days == 30
        
        # 1年
        start_time = _calculate_start_time(end_time, '1y')
        assert (end_time - start_time).days == 365

    @pytest.mark.asyncio
    async def test_fetch_http_error(self):
        """测试HTTP错误处理。"""
        mock_session = Mock()
        mock_session.get = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value.status = 401
        mock_session.get.return_value.__aenter__.return_value.text = AsyncMock(return_value='Unauthorized')
        
        credentials = {'api_key': 'invalid_key'}
        
        result = await fetch_finnhub_data(
            'AAPL', '1y', '1d', 'ohlcv', True, credentials, mock_session
        )
        
        assert result is None
