import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime
from core_bak_refactored.core.data.providers.iex_cloud import (
    fetch_iex_cloud_data,
    fetch_iex_quote,
    _determine_endpoint,
    _map_period_to_iex_range
)


class TestFetchIEXCloudData:
    """测试IEX Cloud数据提供者。"""

    @pytest.mark.asyncio
    async def test_fetch_daily_data_success(self):
        """测试成功获取日线数据。"""
        mock_response_data = [
            {
                'date': '2024-01-01',
                'open': 150.0,
                'high': 152.0,
                'low': 149.0,
                'close': 151.0,
                'volume': 1000000,
                'uVolume': 1000000,
                'change': 1.0,
                'changePercent': 0.67,
                'changeOverTime': 0.01
            }
        ]
        
        mock_session = Mock()
        mock_session.get = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value.status = 200
        mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response_data)
        
        credentials = {'api_key': 'test_key'}
        
        result = await fetch_iex_cloud_data(
            'AAPL', '1y', '1d', 'ohlcv', True, credentials, mock_session
        )
        
        assert result is not None
        assert len(result) == 1
        assert result[0]['symbol'] == 'AAPL'
        assert result[0]['open'] == 150.0
        assert result[0]['metadata']['data_source'] == 'iex_cloud'

    @pytest.mark.asyncio
    async def test_fetch_no_api_key(self):
        """测试缺少API密钥。"""
        credentials = {}
        mock_session = Mock()
        
        result = await fetch_iex_cloud_data(
            'AAPL', '1y', '1d', 'ohlcv', True, credentials, mock_session
        )
        
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_empty_response(self):
        """测试空响应。"""
        mock_session = Mock()
        mock_session.get = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value.status = 200
        mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(return_value=[])
        
        credentials = {'api_key': 'test_key'}
        
        result = await fetch_iex_cloud_data(
            'INVALID', '1y', '1d', 'ohlcv', True, credentials, mock_session
        )
        
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_intraday_data(self):
        """测试获取分钟级数据。"""
        mock_response_data = [
            {
                'date': '2024-01-01',
                'minute': '09:30',
                'open': 150.0,
                'high': 150.5,
                'low': 149.8,
                'close': 150.2,
                'volume': 10000
            }
        ]
        
        mock_session = Mock()
        mock_session.get = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value.status = 200
        mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_response_data)
        
        credentials = {'api_key': 'test_key'}
        
        result = await fetch_iex_cloud_data(
            'AAPL', '1d', '1m', 'ohlcv', True, credentials, mock_session
        )
        
        assert result is not None
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_fetch_quote_success(self):
        """测试获取实时报价。"""
        mock_quote = {
            'symbol': 'AAPL',
            'latestPrice': 150.0,
            'latestTime': '4:00 PM',
            'open': 149.0,
            'high': 151.0,
            'low': 148.5,
            'close': 150.0,
            'volume': 50000000,
            'marketCap': 2500000000000,
            'peRatio': 25.5,
            'week52High': 180.0,
            'week52Low': 120.0,
            'ytdChange': 0.15
        }
        
        mock_session = Mock()
        mock_session.get = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value.status = 200
        mock_session.get.return_value.__aenter__.return_value.json = AsyncMock(return_value=mock_quote)
        
        credentials = {'api_key': 'test_key'}
        
        result = await fetch_iex_quote('AAPL', credentials, mock_session)
        
        assert result is not None
        assert result['symbol'] == 'AAPL'
        assert result['latest_price'] == 150.0
        assert result['market_cap'] == 2500000000000

    def test_determine_endpoint_intraday(self):
        """测试分钟级数据端点选择。"""
        endpoint = _determine_endpoint('1m', '1d')
        assert endpoint == 'intraday-prices'
        
        endpoint = _determine_endpoint('5m', '1d')
        assert endpoint == 'intraday-prices'

    def test_determine_endpoint_daily(self):
        """测试日线数据端点选择。"""
        endpoint = _determine_endpoint('1d', '1y')
        assert endpoint == 'chart'

    def test_map_period_to_iex_range(self):
        """测试期间映射。"""
        assert _map_period_to_iex_range('1d') == '1d'
        assert _map_period_to_iex_range('1mo') == '1m'
        assert _map_period_to_iex_range('1y') == '1y'
        assert _map_period_to_iex_range('max') == 'max'
        assert _map_period_to_iex_range('unknown') == '1y'  # 默认值

    @pytest.mark.asyncio
    async def test_fetch_http_error(self):
        """测试HTTP错误处理。"""
        mock_session = Mock()
        mock_session.get = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value.status = 404
        mock_session.get.return_value.__aenter__.return_value.text = AsyncMock(return_value='Not Found')
        
        credentials = {'api_key': 'test_key'}
        
        result = await fetch_iex_cloud_data(
            'INVALID', '1y', '1d', 'ohlcv', True, credentials, mock_session
        )
        
        assert result is None
