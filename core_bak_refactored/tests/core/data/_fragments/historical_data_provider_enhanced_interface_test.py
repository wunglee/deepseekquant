import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from core_bak_refactored.core.data._fragments.yahoo_finance_provider import YahooFinanceDataProvider
from core_bak_refactored.core.data._fragments.historical_data_provider import MockHistoricalDataProvider


class TestHistoricalDataProviderEnhancedInterface(unittest.TestCase):
    """增强型历史数据提供者接口测试"""

    def setUp(self):
        """设置测试环境"""
        # 使用Mock数据提供者进行测试（避免网络依赖）
        self.mock_provider = MockHistoricalDataProvider()
        
        # 如果需要测试Yahoo Finance提供者，可以取消注释下面的代码
        # self.yahoo_provider = YahooFinanceDataProvider(fallback_to_mock=True)

    def test_get_stock_prices_interface(self):
        """测试个股价格获取接口"""
        # 测试Mock提供者
        data = self.mock_provider.get_stock_prices('600036.SS', '2020-01-01', '2020-01-31')
        
        # 验证返回数据结构
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIn('date', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)
        
        # 验证日期格式
        self.assertIsInstance(data['date'].iloc[0], pd.Timestamp)

    def test_get_volatility_index_interface(self):
        """测试波动率指数获取接口"""
        # 测试Mock提供者
        volatility_data = self.mock_provider.get_volatility_index('VIX', '2020-01-01', '2020-01-31')
        
        # 验证返回数据结构
        self.assertIsInstance(volatility_data, pd.Series)
        self.assertGreater(len(volatility_data), 0)
        
        # 验证数据值在合理范围内
        self.assertTrue((volatility_data >= 0.05).all())  # 不应低于5%
        self.assertTrue((volatility_data <= 0.5).all())   # 不应高于50%

    def test_validate_data_quality_interface(self):
        """测试数据质量验证接口"""
        # 创建测试数据
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000000, 2000000, 100)
        })
        
        # 测试Mock提供者
        quality_report = self.mock_provider.validate_data_quality(data)
        
        # 验证返回数据结构
        self.assertIsInstance(quality_report, dict)
        self.assertIn('completeness_score', quality_report)
        self.assertIn('consistency_score', quality_report)
        self.assertIn('accuracy_score', quality_report)
        self.assertIn('outliers_detected', quality_report)
        self.assertIn('total_rows', quality_report)
        self.assertIn('missing_values', quality_report)
        
        # 验证评分在合理范围内
        self.assertGreaterEqual(quality_report['completeness_score'], 0.0)
        self.assertLessEqual(quality_report['completeness_score'], 1.0)
        self.assertGreaterEqual(quality_report['consistency_score'], 0.0)
        self.assertLessEqual(quality_report['consistency_score'], 1.0)
        self.assertGreaterEqual(quality_report['accuracy_score'], 0.0)
        self.assertLessEqual(quality_report['accuracy_score'], 1.0)

    def test_datetime_parameter_support(self):
        """测试datetime参数支持"""
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 1, 31)
        
        # 测试Mock提供者支持datetime参数
        data = self.mock_provider.get_index_prices('000300.SH', start_date, end_date)
        
        # 验证返回数据结构
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIn('date', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)

    def test_yahoo_finance_datetime_parameter_support(self):
        """测试Yahoo Finance提供者的datetime参数支持"""
        provider = YahooFinanceDataProvider(fallback_to_mock=True)
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2020, 1, 31)
        
        # 测试Yahoo Finance提供者支持datetime参数
        data = provider.get_index_prices('000300.SS', start_date, end_date)
        
        # 验证返回数据结构（可能是模拟数据，但接口应该工作）
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('date', data.columns)
        self.assertIn('close', data.columns)
        self.assertIn('volume', data.columns)


if __name__ == '__main__':
    unittest.main()