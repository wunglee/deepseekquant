import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from core_bak_refactored.core.data.providers.alpha_vantage import AlphaVantageProvider


class AsyncContextManager:
    """Helper class to properly mock async context managers."""
    def __init__(self, return_value):
        self.return_value = return_value
    
    async def __aenter__(self):
        return self.return_value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class TestAlphaVantageProvider:
    """测试Alpha Vantage数据提供者。"""

    @pytest.mark.asyncio
    async def test_fetch_daily_data_success(self):
        """测试成功获取日线数据。"""
        mock_response_data = {
            'Time Series (Daily)': {
                '2024-01-01': {
                    '1. open': '150.00',
                    '2. high': '152.00',
                    '3. low': '149.00',
                    '4. close': '151.00',
                    '5. volume': '1000000'
                }
            }
        }
        
        # 创建 mock response 对象
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        # 使用AsyncContextManager包装
        mock_session = Mock()
        mock_session.get = Mock(return_value=AsyncContextManager(mock_response))
        
        credentials = {'api_key': 'test_key', 'base_url': 'https://test.com'}
        
        provider = AlphaVantageProvider(api_credentials=credentials, aiohttp_session=mock_session)
        result = await provider.fetch('AAPL', '1y', 'daily', 'ohlcv', True)
        
        assert result is not None, "Result should not be None"
        assert len(result) == 1, f"Expected 1 record, got {len(result)}"
        assert result[0].symbol == 'AAPL'
        assert result[0].open == 150.0

    @pytest.mark.asyncio
    async def test_fetch_no_api_key(self):
        """测试缺少API密钥。"""
        credentials = {}
        mock_session = Mock()
        
        provider = AlphaVantageProvider(api_credentials=credentials, aiohttp_session=mock_session)
        result = await provider.fetch('AAPL', '1y', 'daily', 'ohlcv', True)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_api_error_response(self):
        """测试API错误响应。"""
        mock_response = {'Error Message': 'Invalid API call'}
        
        # 创建 mock response 对象
        mock_resp = Mock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        
        # 使用AsyncContextManager包装
        mock_session = Mock()
        mock_session.get = Mock(return_value=AsyncContextManager(mock_resp))
        
        credentials = {'api_key': 'test_key'}
        
        provider = AlphaVantageProvider(api_credentials=credentials, aiohttp_session=mock_session)
        result = await provider.fetch('INVALID', '1y', 'daily', 'ohlcv', True)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_rate_limit(self):
        """测试API限流。"""
        mock_response = {'Note': 'API rate limit exceeded'}
        
        # 创建 mock response 对象
        mock_resp = Mock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=mock_response)
        
        # 使用AsyncContextManager包装
        mock_session = Mock()
        mock_session.get = Mock(return_value=AsyncContextManager(mock_resp))
        
        credentials = {'api_key': 'test_key'}
        
        provider = AlphaVantageProvider(api_credentials=credentials, aiohttp_session=mock_session)
        result = await provider.fetch('AAPL', '1y', 'daily', 'ohlcv', True)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_intraday_data(self):
        """测试获取分钟级数据。"""
        mock_response_data = {
            'Time Series (5min)': {
                '2024-01-01 10:00:00': {
                    '1. open': '150.00',
                    '2. high': '151.00',
                    '3. low': '149.50',
                    '4. close': '150.50',
                    '5. volume': '50000'
                }
            }
        }
        
        # 创建 mock response 对象
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        # 使用AsyncContextManager包装
        mock_session = Mock()
        mock_session.get = Mock(return_value=AsyncContextManager(mock_response))
        
        credentials = {'api_key': 'test_key'}
        
        provider = AlphaVantageProvider(api_credentials=credentials, aiohttp_session=mock_session)
        result = await provider.fetch('AAPL', '1d', '5m', 'ohlcv', True)
        
        assert result is not None, "Result should not be None"
        assert len(result) == 1, f"Expected 1 record, got {len(result) if result else 0}"

    @pytest.mark.asyncio
    async def test_fetch_unsupported_interval(self):
        """测试不支持的时间间隔。"""
        credentials = {'api_key': 'test_key'}
        mock_session = Mock()
        
        provider = AlphaVantageProvider(api_credentials=credentials, aiohttp_session=mock_session)
        result = await provider.fetch('AAPL', '1y', 'unsupported', 'ohlcv', True)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_data_sorted_by_time(self):
        """测试返回数据按时间排序。"""
        mock_response_data = {
            'Time Series (Daily)': {
                '2024-01-03': {'1. open': '152', '2. high': '153', '3. low': '151', '4. close': '152', '5. volume': '1000000'},
                '2024-01-01': {'1. open': '150', '2. high': '151', '3. low': '149', '4. close': '150', '5. volume': '1000000'},
                '2024-01-02': {'1. open': '151', '2. high': '152', '3. low': '150', '4. close': '151', '5. volume': '1000000'}
            }
        }
        
        # 创建 mock response 对象
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_response_data)
        
        # 使用AsyncContextManager包装
        mock_session = Mock()
        mock_session.get = Mock(return_value=AsyncContextManager(mock_response))
        
        credentials = {'api_key': 'test_key'}
        
        provider = AlphaVantageProvider(api_credentials=credentials, aiohttp_session=mock_session)
        result = await provider.fetch('AAPL', '1y', 'daily', 'ohlcv', True)
        
        assert result is not None, "Result should not be None"
        assert len(result) == 3, f"Expected 3 records, got {len(result)}"
        # 验证排序（从旧到新）
        assert result[0].timestamp < result[1].timestamp < result[2].timestamp, "Data should be sorted chronologically"
