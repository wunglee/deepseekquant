"""
MarketData业务验证测试
"""

import unittest


from core_bak_refactored.core.data.validation import (
    validate_market_data,
    validate_data_list,
    clean_market_data
)


class TestValidateMarketData(unittest.TestCase):
    """测试MarketData验证功能"""
    
    def test_valid_market_data_dict(self):
        """测试有效的MarketData字典"""
        data = {
            'symbol': 'AAPL',
            'timestamp': datetime(2025, 1, 1),
            'open': 150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 153.0,
            'volume': 1000000
        }
        
        result = validate_market_data(data)
        
        self.assertTrue(result['valid'])
        self.assertEqual(result['error_count'], 0)
    
    def test_invalid_price_negative(self):
        """测试负价格"""
        data = {
            'symbol': 'AAPL',
            'timestamp': datetime(2025, 1, 1),
            'open': -150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 153.0,
            'volume': 1000000
        }
        
        result = validate_market_data(data)
        
        self.assertFalse(result['valid'])
        self.assertGreater(result['error_count'], 0)
    
    def test_invalid_ohlc_logic(self):
        """测试OHLC逻辑错误（High < Low）"""
        data = {
            'symbol': 'AAPL',
            'timestamp': datetime(2025, 1, 1),
            'open': 150.0,
            'high': 148.0,  # High < Low
            'low': 149.0,
            'close': 153.0,
            'volume': 1000000
        }
        
        result = validate_market_data(data)
        
        self.assertFalse(result['valid'])
        self.assertGreater(result['error_count'], 0)
    
    def test_missing_required_field(self):
        """测试缺失必需字段"""
        data = {
            'symbol': 'AAPL',
            # 缺少 timestamp
            'open': 150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 153.0,
            'volume': 1000000
        }
        
        result = validate_market_data(data)
        
        self.assertFalse(result['valid'])
        self.assertGreater(result['error_count'], 0)


class TestValidateDataList(unittest.TestCase):
    """测试批量验证功能"""
    
    def test_valid_data_list(self):
        """测试有效数据列表"""
        data_list = [
            {
                'symbol': 'AAPL',
                'timestamp': datetime(2025, 1, 1),
                'open': 150.0,
                'high': 155.0,
                'low': 149.0,
                'close': 153.0,
                'volume': 1000000
            },
            {
                'symbol': 'MSFT',
                'timestamp': datetime(2025, 1, 1),
                'open': 350.0,
                'high': 355.0,
                'low': 349.0,
                'close': 353.0,
                'volume': 2000000
            }
        ]
        
        result = validate_data_list(data_list)
        
        self.assertEqual(result['total'], 2)
        self.assertEqual(result['valid_count'], 2)
        self.assertEqual(result['invalid_count'], 0)
        self.assertEqual(result['valid_ratio'], 1.0)
    
    def test_mixed_data_list(self):
        """测试混合数据列表（部分有效，部分无效）"""
        data_list = [
            {
                'symbol': 'AAPL',
                'timestamp': datetime(2025, 1, 1),
                'open': 150.0,
                'high': 155.0,
                'low': 149.0,
                'close': 153.0,
                'volume': 1000000
            },
            {
                'symbol': 'INVALID',
                'timestamp': datetime(2025, 1, 1),
                'open': -150.0,  # 无效价格
                'high': 155.0,
                'low': 149.0,
                'close': 153.0,
                'volume': 1000000
            }
        ]
        
        result = validate_data_list(data_list)
        
        self.assertEqual(result['total'], 2)
        self.assertEqual(result['valid_count'], 1)
        self.assertEqual(result['invalid_count'], 1)
        self.assertEqual(result['valid_ratio'], 0.5)


class TestCleanMarketData(unittest.TestCase):
    """测试数据清洗功能"""
    
    def test_clean_valid_data(self):
        """测试清洗有效数据"""
        data = {
            'symbol': 'AAPL',
            'timestamp': datetime(2025, 1, 1),
            'open': 150.0,
            'high': 155.0,
            'low': 149.0,
            'close': 153.0,
            'volume': 1000000
        }
        
        cleaned = clean_market_data(data)
        
        self.assertIsNotNone(cleaned)
        self.assertEqual(cleaned['symbol'], 'AAPL')
    
    def test_clean_data_with_too_many_errors(self):
        """测试错误过多的数据（无法清洗）"""
        data = {
            'symbol': 'INVALID',
            # 缺少大量必需字段
            'timestamp': datetime(2025, 1, 1),
        }
        
        cleaned = clean_market_data(data)
        
        self.assertIsNone(cleaned)


if __name__ == '__main__':
    unittest.main()
