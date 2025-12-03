import unittest

from core_bak_refactored.core.share.market.market_enums import MarketCode, DataSource, REGIONAL_DATA_SOURCE_PRIORITY


class MarketEnumsTest(unittest.TestCase):
    def test_market_code_values_and_validation(self):
        codes = MarketCode.get_all_codes()
        self.assertIn('CN', codes)
        self.assertIn('US', codes)
        self.assertIn('HK', codes)
        self.assertTrue(MarketCode.is_valid('CN'))
        self.assertFalse(MarketCode.is_valid('XX'))
        # __str__ should return value
        self.assertEqual(str(MarketCode.CN), 'CN')

    def test_data_source_values_and_validation(self):
        sources = DataSource.get_all_sources()
        self.assertIn('yahoo', sources)
        self.assertIn('mock', sources)
        self.assertTrue(DataSource.is_valid('yahoo'))
        self.assertFalse(DataSource.is_valid('unknown_source'))
        self.assertEqual(str(DataSource.MOCK), 'mock')

    def test_regional_priority_mapping_basic(self):
        # CN market should prioritize JOINQUANT/TUSHARE/WIND/YAHOO/MOCK in some order
        # 注意：由于生产环境中不再包含MOCK，此处仅验证基本结构
        cn_priority = REGIONAL_DATA_SOURCE_PRIORITY.get(MarketCode.CN)
        self.assertIsNotNone(cn_priority)
        self.assertGreater(len(cn_priority), 0)
        # US market should include YAHOO
        us_priority = REGIONAL_DATA_SOURCE_PRIORITY.get(MarketCode.US)
        self.assertIsNotNone(us_priority)
        self.assertIn(DataSource.YAHOO, us_priority)
        # Default key should exist
        self.assertIn(MarketCode.UNKNOWN, REGIONAL_DATA_SOURCE_PRIORITY)


if __name__ == '__main__':
    unittest.main()
