import unittest
from datetime import datetime
import pandas as pd

from core_bak_refactored.core.data._fragments.yahoo_finance_provider import YahooFinanceDataProvider


class YahooFinanceProviderTest(unittest.TestCase):
    def setUp(self):
        self.provider = YahooFinanceDataProvider(fallback_to_mock=True)
        
    def test_initialization(self):
        provider = YahooFinanceDataProvider(fallback_to_mock=True)
        self.assertTrue(provider.fallback)
        
    def test_get_index_prices_with_fallback(self):
        # 即使yfinance不可用，也应该能通过fallback获取Mock数据
        data = self.provider.get_index_prices('000300.SH', '2020-01-01', '2020-01-31')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('date', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)
        self.assertGreater(len(data), 0)
        
    def test_get_index_returns_with_fallback(self):
        returns = self.provider.get_index_returns('000300.SH', '2020-01-01', '2020-01-31')
        self.assertIsInstance(returns, pd.Series)
        self.assertGreater(len(returns), 0)
        
    def test_get_stock_prices_with_fallback(self):
        data = self.provider.get_stock_prices('600036.SS', '2020-01-01', '2020-01-31')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('date', data.columns)
        self.assertIn('close', data.columns)
        
    def test_datetime_parameter_support(self):
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 31)
        data = self.provider.get_index_prices('000300.SH', start, end)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)


if __name__ == '__main__':
    unittest.main()
