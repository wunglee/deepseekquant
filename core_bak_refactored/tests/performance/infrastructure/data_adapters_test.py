import unittest
import pandas as pd

from core_bak_refactored.infrastructure.data_adapters import (
    DataSourceAdapter, TushareAdapter, AKShareAdapter, YFinanceAdapter
)


class DataAdaptersTest(unittest.TestCase):
    def test_base_adapter_not_implemented(self):
        adapter = DataSourceAdapter({})
        with self.assertRaises(NotImplementedError):
            adapter.fetch_stock_data('symbol', '2020-01-01', '2020-01-31')

    def test_tushare_adapter_instantiation(self):
        adapter = TushareAdapter({'tushare_token': None})
        self.assertIsNotNone(adapter)
        # Without token/lib, fetch will return empty DataFrame or raise
        # Just ensure it doesn't crash on instantiation

    def test_akshare_adapter_instantiation(self):
        adapter = AKShareAdapter({})
        self.assertIsNotNone(adapter)

    def test_yfinance_adapter_instantiation(self):
        adapter = YFinanceAdapter({})
        self.assertIsNotNone(adapter)


if __name__ == '__main__':
    unittest.main()
