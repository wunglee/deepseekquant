"""
数据模型测试
"""

import unittest


from core_bak_refactored.core.share.models import MarketData


class TestMarketData(unittest.TestCase):
    """测试MarketData数据类"""
    
    def test_market_data_creation(self):
        """测试创建MarketData实例"""
        data = MarketData(
            symbol="AAPL",
            timestamp=datetime(2025, 1, 1),
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1000000
        )
        
        self.assertEqual(data.symbol, "AAPL")
        self.assertEqual(data.open, 150.0)
        self.assertEqual(data.close, 153.0)
        self.assertEqual(data.volume, 1000000)
    
    def test_market_data_with_optional_fields(self):
        """测试可选字段"""
        data = MarketData(
            symbol="AAPL",
            timestamp=datetime(2025, 1, 1),
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=1000000,
            adj_close=152.5,
            dividends=0.5,
            splits=1.0
        )
        
        self.assertEqual(data.adj_close, 152.5)
        self.assertEqual(data.dividends, 0.5)
        self.assertEqual(data.splits, 1.0)


if __name__ == '__main__':
    unittest.main()
