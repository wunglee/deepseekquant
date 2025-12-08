import unittest
import pandas as pd
from datetime import datetime

from core_bak_refactored.core.data.providers.tushare import TushareDataProvider


class TushareProviderTest(unittest.TestCase):
    def test_get_index_prices_fallback_to_mock(self):
        """测试无token时的行为（已废弃 - 2025-12-03重构）"""
        self.skipTest("fallback_to_mock 功能已移除")
        provider = TushareDataProvider(token=None)
        data = provider.get_index_prices('000300.SH', '2015-06-01', '2015-06-30')
        self.assertIsInstance(data, object)  # 应该返回PriceData对象
        self.assertIsInstance(data.records, list)
        self.assertIsInstance(data.symbol, str)
        self.assertIsInstance(data.start_date, pd.Timestamp)
        self.assertIsInstance(data.end_date, pd.Timestamp)
        self.assertIsInstance(data.count, int)
        self.assertGreater(len(data.records), 0)
        
        # 验证第一条记录的字段
        first_record = data.records[0]
        self.assertTrue(hasattr(first_record, 'date'))
        self.assertTrue(hasattr(first_record, 'open'))
        self.assertTrue(hasattr(first_record, 'high'))
        self.assertTrue(hasattr(first_record, 'low'))
        self.assertTrue(hasattr(first_record, 'close'))
        self.assertTrue(hasattr(first_record, 'volume'))

    def test_get_stock_prices_fallback_to_mock(self):
        """测试无token时的行为（已废弃 - 2025-12-03重构）"""
        self.skipTest("fallback_to_mock 功能已移除")
        provider = TushareDataProvider(token=None)
        data = provider.get_stock_prices('600036.SH', datetime(2015, 6, 1), datetime(2015, 6, 30))
        self.assertIsInstance(data, object)  # 应该返回PriceData对象
        self.assertIsInstance(data.records, list)
        self.assertIsInstance(data.symbol, str)
        self.assertEqual(data.symbol, '600036.SH')
        self.assertGreater(len(data.records), 0)
        
        # 验证记录字段
        first_record = data.records[0]
        self.assertTrue(hasattr(first_record, 'date'))
        self.assertTrue(hasattr(first_record, 'open'))
        self.assertTrue(hasattr(first_record, 'high'))
        self.assertTrue(hasattr(first_record, 'low'))
        self.assertTrue(hasattr(first_record, 'close'))
        self.assertTrue(hasattr(first_record, 'volume'))


if __name__ == '__main__':
    unittest.main()