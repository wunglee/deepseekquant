import unittest
import pandas as pd
from datetime import datetime

from core_bak_refactored.core.data.tushare_provider import TushareDataProvider


class TushareProviderTest(unittest.TestCase):
    def test_get_index_prices_fallback_to_mock(self):
        provider = TushareDataProvider(token=None, fallback_to_mock=True)
        data = provider.get_index_prices('000300.SH', '2015-06-01', '2015-06-30')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('date', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)
        self.assertGreater(len(data), 0)

    def test_get_stock_prices_fallback_to_mock(self):
        provider = TushareDataProvider(token=None, fallback_to_mock=True)
        data = provider.get_stock_prices('600036.SH', datetime(2015, 6, 1), datetime(2015, 6, 30))
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('date', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)
        self.assertGreater(len(data), 0)


if __name__ == '__main__':
    unittest.main()
