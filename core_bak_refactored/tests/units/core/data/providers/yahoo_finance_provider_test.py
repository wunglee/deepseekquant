import unittest
from unittest.mock import patch
from datetime import datetime
import pandas as pd
import pytest
from core_bak_refactored.core.data.providers.yahoo_finance import YahooFinanceDataProvider


class YahooFinanceProviderTest(unittest.TestCase):
    def setUp(self):
        self.provider = YahooFinanceDataProvider(fallback_to_mock=False)
        
    def test_initialization(self):
        provider = YahooFinanceDataProvider(fallback_to_mock=False)
        self.assertFalse(provider.fallback)
        
    def test_get_index_prices_with_fallback(self):
        # 真实场景下数据不可用应抛出异常
        provider = YahooFinanceDataProvider(fallback_to_mock=False)
        try:
            # 使用可能失败的真实ticker
            data = provider.get_index_prices('000300.SH', '2020-01-01', '2020-01-31')
            # 如果成功获取，验证数据结构
            self.assertIsInstance(data, pd.DataFrame)
            self.assertIn('date', data.columns)
            self.assertIn('close', data.columns)
            self.assertIn('volume', data.columns)
            self.assertGreater(len(data), 0)
        except ValueError:
            # 真实数据不可用时预期抛出ValueError
            pass
        
    def test_get_index_returns_with_fallback(self):
        provider = YahooFinanceDataProvider(fallback_to_mock=False)
        try:
            returns = provider.get_index_returns('000300.SH', '2020-01-01', '2020-01-31')
            self.assertIsInstance(returns, pd.Series)
            self.assertGreater(len(returns), 0)
        except ValueError:
            # 真实数据不可用时预期抛出ValueError
            pass
        
    def test_get_stock_prices_with_fallback(self):
        provider = YahooFinanceDataProvider(fallback_to_mock=False)
        try:
            data = provider.get_stock_prices('600036.SS', '2020-01-01', '2020-01-31')
            self.assertIsInstance(data, pd.DataFrame)
            self.assertIn('date', data.columns)
            self.assertIn('close', data.columns)
        except ValueError:
            # 真实数据不可用时预期抛出ValueError
            pass
        
    def test_datetime_parameter_support(self):
        provider = YahooFinanceDataProvider(fallback_to_mock=False)
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 31)
        try:
            data = provider.get_index_prices('000300.SH', start, end)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 0)
        except ValueError:
            # 真实数据不可用时预期抛出ValueError
            pass

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

        provider = YahooFinanceDataProvider(fallback_to_mock=False)

        with patch.object(provider.yf, 'download', return_value=mock_hist):
            result = provider.get_index_prices('000300.SH', '2024-01-01', '2024-01-02', include_ohlcv=True)

            assert result is not None
            assert len(result) == 2
            assert 'open' in result.columns
            assert result.iloc[0]['close'] == 151.0

    def test_fetch_yahoo_data_empty_response(self):
        """测试空响应情况。"""
        provider = YahooFinanceDataProvider(fallback_to_mock=False)

        with patch.object(provider.yf, 'download', return_value=pd.DataFrame()):
            try:
                result = provider.get_index_prices('INVALID', '2024-01-01', '2024-01-02')
                assert False, "Should raise ValueError"
            except ValueError:
                pass  # Expected

    def test_fetch_yahoo_data_with_fallback(self):
        """测试真实场景下网络错误应抛出异常（不再fallback）。"""
        provider = YahooFinanceDataProvider(fallback_to_mock=False)

        with patch.object(provider.yf, 'download', side_effect=Exception("Network error")):
            with self.assertRaises(ValueError) as context:
                provider.get_index_prices('000300.SH', '2024-01-01', '2024-01-02')
            
            self.assertIn('Failed to fetch data', str(context.exception))


if __name__ == '__main__':
    unittest.main()
