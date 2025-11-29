import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import pandas as pd
from core_bak_refactored.core.data.providers.yahoo import fetch_yahoo_data


class TestFetchYahooData:
    """测试Yahoo Finance数据提供者。"""

    @pytest.mark.asyncio
    async def test_fetch_yahoo_data_ohlcv_success(self):
        """测试成功获取OHLCV数据。"""
        # 模拟yfinance返回数据
        mock_hist = pd.DataFrame({
            'Open': [150.0, 151.0],
            'High': [152.0, 153.0],
            'Low': [149.0, 150.0],
            'Close': [151.0, 152.0],
            'Volume': [1000000, 1100000],
            'Adj Close': [151.0, 152.0]
        }, index=pd.DatetimeIndex(['2024-01-01', '2024-01-02']))
        
        with patch('core_bak_refactored.core.data.providers.yahoo.yf.Ticker') as MockTicker:
            mock_ticker = Mock()
            mock_ticker.history.return_value = mock_hist
            MockTicker.return_value = mock_ticker
            
            result = await fetch_yahoo_data('AAPL', '1y', '1d', 'ohlcv', True)
            
            assert result is not None
            assert len(result) == 2
            assert result[0]['symbol'] == 'AAPL'
            assert result[0]['open'] == 150.0
            assert result[0]['close'] == 151.0

    @pytest.mark.asyncio
    async def test_fetch_yahoo_data_empty_response(self):
        """测试空响应情况。"""
        with patch('core_bak_refactored.core.data.providers.yahoo.yf.Ticker') as MockTicker:
            mock_ticker = Mock()
            mock_ticker.history.return_value = pd.DataFrame()
            MockTicker.return_value = mock_ticker
            
            result = await fetch_yahoo_data('INVALID', '1y', '1d', 'ohlcv', True)
            
            assert result is None

    @pytest.mark.asyncio
    async def test_fetch_yahoo_data_dividends(self):
        """测试获取分红数据。"""
        mock_dividends = pd.Series([0.5, 0.6], index=pd.DatetimeIndex(['2024-01-01', '2024-01-02']))
        
        with patch('core_bak_refactored.core.data.providers.yahoo.yf.Ticker') as MockTicker:
            mock_ticker = Mock()
            mock_ticker.dividends = mock_dividends
            MockTicker.return_value = mock_ticker
            
            result = await fetch_yahoo_data('AAPL', '1y', '1d', 'dividends', False)
            
            assert result is not None
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_fetch_yahoo_data_invalid_type(self):
        """测试无效数据类型。"""
        with patch('core_bak_refactored.core.data.providers.yahoo.yf.Ticker'):
            with pytest.raises(ValueError, match="不支持的数据类型"):
                await fetch_yahoo_data('AAPL', '1y', '1d', 'invalid_type', True)

    @pytest.mark.asyncio
    async def test_fetch_yahoo_data_exception_handling(self):
        """测试异常处理。"""
        with patch('core_bak_refactored.core.data.providers.yahoo.yf.Ticker') as MockTicker:
            MockTicker.side_effect = Exception("Network error")
            
            result = await fetch_yahoo_data('AAPL', '1y', '1d', 'ohlcv', True)
            
            assert result is None

    @pytest.mark.asyncio
    async def test_fetch_yahoo_data_with_adjustments(self):
        """测试价格调整参数。"""
        mock_hist = pd.DataFrame({
            'Open': [150.0],
            'High': [152.0],
            'Low': [149.0],
            'Close': [151.0],
            'Volume': [1000000]
        }, index=pd.DatetimeIndex(['2024-01-01']))
        
        with patch('core_bak_refactored.core.data.providers.yahoo.yf.Ticker') as MockTicker:
            mock_ticker = Mock()
            mock_ticker.history.return_value = mock_hist
            MockTicker.return_value = mock_ticker
            
            result = await fetch_yahoo_data('AAPL', '1y', '1d', 'ohlcv', True)
            
            # 验证history被调用时传入auto_adjust=True
            mock_ticker.history.assert_called_once()
            call_kwargs = mock_ticker.history.call_args[1]
            assert call_kwargs['auto_adjust'] is True

    @pytest.mark.asyncio
    async def test_fetch_yahoo_data_metadata(self):
        """测试返回数据包含正确的元数据。"""
        mock_hist = pd.DataFrame({
            'Open': [150.0],
            'High': [152.0],
            'Low': [149.0],
            'Close': [151.0],
            'Volume': [1000000]
        }, index=pd.DatetimeIndex(['2024-01-01']))
        
        with patch('core_bak_refactored.core.data.providers.yahoo.yf.Ticker') as MockTicker:
            mock_ticker = Mock()
            mock_ticker.history.return_value = mock_hist
            MockTicker.return_value = mock_ticker
            
            result = await fetch_yahoo_data('AAPL', '1y', '1d', 'ohlcv', True)
            
            assert result[0]['metadata']['data_source'] == 'yahoo'
            assert result[0]['metadata']['period'] == '1y'
            assert result[0]['metadata']['interval'] == '1d'
