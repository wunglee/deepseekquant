import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from core_bak_refactored.core.data.quality.data_quality_enhancer import DataQualityEnhancer, DataQualityReport
from core_bak_refactored.tests.fixtures.core.data.mock_historical_data_provider import MockHistoricalDataProvider


class TestDataQualityEnhancer(unittest.TestCase):
    """数据质量增强器测试"""

    def setUp(self):
        """设置测试环境"""
        self.mock_provider = MockHistoricalDataProvider()
        self.enhancer = DataQualityEnhancer()  # 不再需要primary_source参数

    def test_validate_data_quality_with_good_data(self):
        """测试高质量数据的质量验证"""
        # 生成高质量数据
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'close': np.random.uniform(100, 200, 100),  # 价格在合理范围内
            'volume': np.random.uniform(1000000, 2000000, 100)  # 成交量在合理范围内
        })
        
        report = self.enhancer.validate_data_quality(data)
        
        # 验证评分应该较高
        self.assertGreater(report.completeness_score, 0.9)
        self.assertGreater(report.consistency_score, 0.9)
        self.assertGreater(report.accuracy_score, 0.9)
        self.assertGreater(report.overall_score, 0.9)

    def test_validate_data_quality_with_missing_data(self):
        """测试包含缺失值的数据质量验证"""
        # 生成包含缺失值的数据
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        close_prices = np.random.uniform(100, 200, 100)
        volumes = np.random.uniform(1000000, 2000000, 100)
        
        # 插入一些缺失值
        close_prices[5] = np.nan
        close_prices[15] = np.nan
        volumes[25] = np.nan
        
        data = pd.DataFrame({
            'date': dates,
            'close': close_prices,
            'volume': volumes
        })
        
        report = self.enhancer.validate_data_quality(data)
        
        # 验证缺失值被正确检测
        self.assertEqual(report.missing_values, 3)
        # 验证完整性评分应该较低
        self.assertLess(report.completeness_score, 1.0)

    def test_validate_data_quality_with_negative_prices(self):
        """测试包含负价格的数据质量验证"""
        # 生成包含负价格的数据
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        close_prices = np.random.uniform(100, 200, 100)
        
        # 插入一些负价格
        close_prices[5] = -10
        close_prices[15] = -20
        
        data = pd.DataFrame({
            'date': dates,
            'close': close_prices,
            'volume': np.random.uniform(1000000, 2000000, 100)
        })
        
        report = self.enhancer.validate_data_quality(data)
        
        # 验证准确性评分应该较低
        self.assertLess(report.accuracy_score, 1.0)

    def test_get_enhanced_prices_deprecated(self):
        """测试get_enhanced_prices已被删除
        
        架构变更（2025-12-06）：
        - get_enhanced_prices()方法已完全删除
        - DataQualityEnhancer不再有该方法
        """
        # 验证方法不存在
        self.assertFalse(hasattr(self.enhancer, 'get_enhanced_prices'),
                        "get_enhanced_prices方法应该已被完全删除")

    def test_data_quality_report_structure(self):
        """测试数据质量报告结构"""
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10, freq='D'),
            'close': np.random.uniform(100, 200, 10),
            'volume': np.random.uniform(1000000, 2000000, 10)
        })
        
        report = self.enhancer.validate_data_quality(data)
        
        # 验证报告包含所有必要的字段
        self.assertIsInstance(report, DataQualityReport)
        self.assertIsInstance(report.completeness_score, float)
        self.assertIsInstance(report.consistency_score, float)
        self.assertIsInstance(report.accuracy_score, float)
        # Handle numpy integers
        self.assertIsInstance(int(report.outliers_detected), int)
        self.assertIsInstance(int(report.total_rows), int)
        self.assertIsInstance(int(report.missing_values), int)
        self.assertIsInstance(report.overall_score, float)


if __name__ == '__main__':
    unittest.main()