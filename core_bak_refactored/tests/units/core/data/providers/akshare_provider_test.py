"""
AKShare数据提供者单元测试
"""

import unittest
import pandas as pd
from datetime import datetime

from core_bak_refactored.core.data.providers.akshare_provider import AKShareDataProvider


class AKShareProviderTest(unittest.TestCase):
    """AKShare数据提供者测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.provider = AKShareDataProvider(fallback_to_mock=True)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.provider)
        # 即使akshare未安装，也应该能初始化（会fallback到mock）
        self.assertTrue(hasattr(self.provider, 'fallback'))
    
    def test_get_index_prices_with_string_dates(self):
        """测试获取指数价格（字符串日期）"""
        data = self.provider.get_index_prices('000300.SH', '2024-01-01', '2024-01-31')
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('date', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)
        self.assertGreater(len(data), 0)
        
        # 验证数据类型
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(data['date']))
        self.assertTrue(pd.api.types.is_numeric_dtype(data['close']))
        self.assertTrue(pd.api.types.is_numeric_dtype(data['volume']))
    
    def test_get_index_prices_with_datetime_objects(self):
        """测试获取指数价格（datetime对象）"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        data = self.provider.get_index_prices('000300.SH', start_date, end_date)
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
    
    def test_get_index_returns(self):
        """测试获取指数收益率"""
        returns = self.provider.get_index_returns('000300.SH', '2024-01-01', '2024-01-31')
        
        self.assertIsInstance(returns, pd.Series)
        self.assertGreater(len(returns), 0)
        
        # 验证收益率在合理范围内（-100% 到 +100%）
        self.assertTrue((returns >= -1.0).all())
        self.assertTrue((returns <= 1.0).all())
    
    def test_get_stock_prices(self):
        """测试获取个股价格"""
        # 使用平安银行作为测试股票
        data = self.provider.get_stock_prices('000001', '2024-01-01', '2024-01-31')
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('date', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)
        self.assertGreater(len(data), 0)
    
    def test_index_mapping(self):
        """测试指数代码映射"""
        # 测试A股指数映射
        self.assertEqual(
            self.provider._map_index_to_akshare('000300.SH'),
            'sh000300'
        )
        self.assertEqual(
            self.provider._map_index_to_akshare('399001.SZ'),
            'sz399001'
        )
        
        # 测试自动转换
        self.assertEqual(
            self.provider._map_index_to_akshare('000016.SH'),
            'sh000016'
        )
    
    def test_data_sorting(self):
        """测试数据按日期排序"""
        data = self.provider.get_index_prices('000300.SH', '2024-01-01', '2024-01-31')
        
        # 验证日期是升序排列的
        dates = data['date'].tolist()
        self.assertEqual(dates, sorted(dates))
    
    def test_fallback_to_mock(self):
        """测试回退到Mock数据"""
        # 强制使用不存在的指数代码
        provider_with_fallback = AKShareDataProvider(fallback_to_mock=True)
        
        try:
            # 即使请求失败，也应该能获取到mock数据
            data = provider_with_fallback._fallback_to_mock(
                'INVALID_INDEX',
                '2024-01-01',
                '2024-01-31'
            )
            
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 0)
        except Exception as e:
            # 如果mock也失败，至少验证异常被正确抛出
            self.assertIn('failed', str(e).lower())
    
    def test_data_standardization(self):
        """测试数据标准化"""
        data = self.provider.get_index_prices('000300.SH', '2024-01-01', '2024-01-10')
        
        # 验证列名标准化
        self.assertEqual(set(data.columns), {'date', 'close', 'volume'})
        
        # 验证无缺失值
        self.assertFalse(data['date'].isnull().any())
        self.assertFalse(data['close'].isnull().any())
        
        # 验证数值类型
        self.assertTrue(data['close'].dtype in [float, 'float64'])
        self.assertTrue(data['volume'].dtype in [float, 'float64'])


if __name__ == '__main__':
    unittest.main()
