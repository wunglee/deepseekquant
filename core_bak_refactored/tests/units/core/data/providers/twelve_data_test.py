import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime
from core_bak_refactored.core.data.providers.twelve_data import (
    fetch_twelve_data,
    fetch_twelve_data_quote,
    _map_interval_to_twelve_data,
    _calculate_outputsize
)
from .async_mock_helper import AsyncContextManager


class TestFetchTwelveData:
    """测试Twelve Data提供者。"""

    @pytest.mark.asyncio
    async def test_fetch_time_series_success(self):
        """测试成功获取时间序列数据。"""
        mock_response = {
            'meta': {'symbol': 'AAPL'},
            'values': [
                {
                    'datetime': '2024-01-01 09:30:00',
                    'open': '150.0',
                    'high': '152.0',
                    'low': '149.0',
                    'close': '151.0',
                    'volume': '1000000'
                }
            ]
        }
        
        # 创建 mock response 对象
        mock_response_obj = Mock()
        mock_response_obj.status = 200
        mock_response_obj.json = AsyncMock(return_value=mock_response)
        
        # 使用AsyncContextManager包装
        mock_session = Mock()
        mock_session.get = Mock(return_value=AsyncContextManager(mock_response_obj))
        
        credentials = {'api_key': 'test_key'}
        
        result = await fetch_twelve_data(
            'AAPL', '1y', '1d', 'ohlcv', True, credentials, mock_session
        )
        
        assert result is not None, "Result should not be None"
        assert len(result) == 1, f"Expected 1 record, got {len(result)}"
        assert result[0]['symbol'] == 'AAPL'
        assert result[0]['open'] == 150.0
        assert result[0]['metadata']['data_source'] == 'twelve_data'

    @pytest.mark.asyncio
    async def test_fetch_no_api_key(self):
        """测试缺少API密钥。"""
        credentials = {}
        mock_session = Mock()
        
        result = await fetch_twelve_data(
            'AAPL', '1y', '1d', 'ohlcv', True, credentials, mock_session
        )
        
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_error_response(self):
        """测试错误响应。"""
        mock_response = {
            'status': 'error',
            'message': 'Invalid API key'
        }
        
        mock_session = Mock()
        mock_session.get = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value.status = 200
        mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
        
        credentials = {'api_key': 'invalid_key'}
        
        result = await fetch_twelve_data(
            'AAPL', '1y', '1d', 'ohlcv', True, credentials, mock_session
        )
        
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_empty_values(self):
        """测试空数据响应。"""
        mock_response = {
            'meta': {'symbol': 'AAPL'},
            'values': []
        }
        
        mock_session = Mock()
        mock_session.get = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value.status = 200
        mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response)
        
        credentials = {'api_key': 'test_key'}
        
        result = await fetch_twelve_data(
            'INVALID', '1y', '1d', 'ohlcv', True, credentials, mock_session
        )
        
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_quote_success(self):
        """测试获取实时报价。"""
        mock_quote = {
            'symbol': 'AAPL',
            'name': 'Apple Inc',
            'exchange': 'NASDAQ',
            'currency': 'USD',
            'datetime': '2024-01-01 16:00:00',
            'open': '149.0',
            'high': '152.0',
            'low': '148.5',
            'close': '151.0',
            'volume': '50000000',
            'previous_close': '149.0',
            'change': '2.0',
            'percent_change': '1.34',
            'average_volume': '48000000',
            'fifty_two_week': {
                'low': '120.0',
                'high': '180.0',
                'change': '31.0',
                'change_percent': '25.83'
            }
        }
        
        # 创建 mock response 对象
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_quote)
        
        # 使用AsyncContextManager包装
        mock_session = Mock()
        mock_session.get = Mock(return_value=AsyncContextManager(mock_response))
        
        credentials = {'api_key': 'test_key'}
        
        result = await fetch_twelve_data_quote('AAPL', credentials, mock_session)
        
        assert result is not None, "Result should not be None"
        assert result['symbol'] == 'AAPL'
        assert result['name'] == 'Apple Inc'
        assert result['close'] == 151.0
        assert result['fifty_two_week']['high'] == 180.0

    def test_map_interval_to_twelve_data(self):
        """测试时间间隔映射。"""
        assert _map_interval_to_twelve_data('1m') == '1min'
        assert _map_interval_to_twelve_data('5m') == '5min'
        assert _map_interval_to_twelve_data('1h') == '1h'
        assert _map_interval_to_twelve_data('1d') == '1day'
        assert _map_interval_to_twelve_data('1wk') == '1week'
        assert _map_interval_to_twelve_data('1mo') == '1month'
        assert _map_interval_to_twelve_data('unknown') == '1day'

    def test_calculate_outputsize(self):
        """测试输出大小计算。"""
        # 1天，1分钟间隔
        size = _calculate_outputsize('1d', '1m')
        assert size == 390
        
        # 1年，日线
        size = _calculate_outputsize('1y', '1d')
        assert size == 365
        
        # 应不超过5000
        size = _calculate_outputsize('5y', '1m')
        assert size <= 5000

    @pytest.mark.asyncio
    async def test_fetch_http_error(self):
        """测试HTTP错误处理。"""
        mock_session = Mock()
        mock_session.get = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value.status = 429
        mock_session.get.return_value.__aenter__.return_value.text = AsyncMock(return_value='Rate Limit Exceeded')
        
        credentials = {'api_key': 'test_key'}
        
        result = await fetch_twelve_data(
            'AAPL', '1y', '1d', 'ohlcv', True, credentials, mock_session
        )
        
        assert result is None
