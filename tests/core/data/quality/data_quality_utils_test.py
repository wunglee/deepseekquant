"""
数据质量工具函数测试套件
测试数据质量评估工具函数的正确性和一致性

强类型改造（2025-12-11）:
- 新增 validate_data_quality(PriceData) 测试
- 验证强类型约束和向后兼容性
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, project_root)


class TestDataQualityUtils(unittest.TestCase):
    """数据质量工具函数测试类"""
    
    def setUp(self):
        """测试前准备"""
        from core_bak_refactored.core.data.quality.data_quality_utils import (
            calculate_consistency_score, 
            calculate_accuracy_score, 
            detect_outliers,
            validate_data_quality
        )
        from core_bak_refactored.core.data.providers.protocols import PriceData
        
        self.calculate_consistency_score = calculate_consistency_score
        self.calculate_accuracy_score = calculate_accuracy_score
        self.detect_outliers = detect_outliers
        self.validate_data_quality = validate_data_quality
        self.PriceData = PriceData
        self.OHLCVRecord = OHLCVRecord
    
    def test_calculate_consistency_score_empty_data(self):
        """测试空数据的一致性评分"""
        data = pd.DataFrame()
        score = self.calculate_consistency_score(data)
        self.assertEqual(score, 0.0)
    
    def test_calculate_consistency_score_no_numeric_columns(self):
        """测试无数值列的一致性评分"""
        data = pd.DataFrame({'name': ['A', 'B', 'C'], 'category': ['X', 'Y', 'Z']})
        score = self.calculate_consistency_score(data)
        self.assertEqual(score, 0.5)
    
    def test_calculate_consistency_score_perfect_consistency(self):
        """测试完美一致性评分"""
        data = pd.DataFrame({
            'price': [100.0, 101.0, 102.0],
            'volume': [1000, 2000, 3000]
        })
        score = self.calculate_consistency_score(data)
        self.assertEqual(score, 1.0)
    
    def test_calculate_consistency_score_mixed_types(self):
        """测试混合类型的一致性评分"""
        data = pd.DataFrame({
            'price': [100.0, 101.0, 102.0],
            'volume': ['1000', '2000', '3000']  # 字符串类型但可转换为数值
        })
        score = self.calculate_consistency_score(data)
        self.assertEqual(score, 1.0)
    
    def test_calculate_accuracy_score_empty_data(self):
        """测试空数据的准确性评分"""
        data = pd.DataFrame()
        score = self.calculate_accuracy_score(data)
        self.assertEqual(score, 0.0)
    
    def test_calculate_accuracy_score_perfect_accuracy(self):
        """测试完美准确性评分"""
        data = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0, 104.0]
        })
        score = self.calculate_accuracy_score(data)
        self.assertEqual(score, 1.0)
    
    def test_calculate_accuracy_score_negative_prices(self):
        """测试负价格的准确性评分"""
        data = pd.DataFrame({
            'close': [100.0, -50.0, 102.0, 103.0, 104.0]  # 包含负价格
        })
        score = self.calculate_accuracy_score(data)
        # 应该扣分：0.2 * (1/5) = 0.04
        self.assertAlmostEqual(score, 0.96, places=2)
    
    def test_calculate_accuracy_score_extreme_prices(self):
        """测试极端价格的准确性评分"""
        data = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0, 3500.0]  # 极端价格，均值约780，3500 < 780*10
        })
        score = self.calculate_accuracy_score(data)
        # 3500 < 均值(~780) * 10 = 7800，所以不会被认为是极端价格
        # 因此应该是完美分数
        self.assertEqual(score, 1.0)
    
    def test_detect_outliers_empty_data(self):
        """测试空数据的异常值检测"""
        data = pd.DataFrame()
        outliers = self.detect_outliers(data)
        self.assertEqual(outliers, 0)
    
    def test_detect_outliers_no_outliers(self):
        """测试无异常值的数据"""
        data = pd.DataFrame({
            'price': [100.0, 101.0, 102.0, 103.0, 104.0]
        })
        outliers = self.detect_outliers(data)
        self.assertEqual(outliers, 0)
    
    def test_detect_outliers_with_outliers(self):
        """测试包含异常值的数据"""
        data = pd.DataFrame({
            'price': [100.0, 101.0, 102.0, 103.0, 1000.0]  # 1000是异常值
        })
        outliers = self.detect_outliers(data)
        self.assertGreater(outliers, 0)


class TestValidateDataQuality(unittest.TestCase):
    """测试 validate_data_quality 强类型函数"""
    
    def setUp(self):
        """测试前准备"""
        from core_bak_refactored.core.data.quality.data_quality_utils import validate_data_quality
        from core_bak_refactored.core.data.providers.protocols import PriceData, OHLCVRecord
        from core_bak_refactored.core.data.quality.quality_types import DataQualityReport
        
        self.validate_data_quality = validate_data_quality
        self.PriceData = PriceData
        self.OHLCVRecord = OHLCVRecord
        self.DataQualityReport = DataQualityReport
    
    def test_validate_empty_price_data(self):
        """测试空 PriceData 的验证"""
        price_data = self.PriceData(
            symbol='TEST',
            records=[],
            start_date=pd.Timestamp.now(),
            end_date=pd.Timestamp.now(),
            count=0
        )
        
        report = self.validate_data_quality(price_data)
        
        self.assertIsInstance(report, self.DataQualityReport)
        self.assertEqual(report.completeness_score, 0.0)
        self.assertEqual(report.total_rows, 0)
    
    def test_validate_perfect_quality_price_data(self):
        """测试完美质量的 PriceData"""
        records = [
            self.OHLCVRecord(
                date=pd.Timestamp('2023-01-01'),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000
            ),
            self.OHLCVRecord(
                date=pd.Timestamp('2023-01-02'),
                open=103.0,
                high=108.0,
                low=102.0,
                close=106.0,
                volume=1500
            ),
            self.OHLCVRecord(
                date=pd.Timestamp('2023-01-03'),
                open=106.0,
                high=110.0,
                low=105.0,
                close=109.0,
                volume=1200
            )
        ]
        
        price_data = self.PriceData(
            symbol='PERFECT',
            records=records,
            start_date=pd.Timestamp('2023-01-01'),
            end_date=pd.Timestamp('2023-01-03'),
            count=3
        )
        
        report = self.validate_data_quality(price_data)
        
        self.assertIsInstance(report, self.DataQualityReport)
        self.assertEqual(report.total_rows, 3)
        self.assertEqual(report.completeness_score, 1.0)  # 无缺失值
        self.assertEqual(report.consistency_score, 1.0)  # 类型一致
        self.assertEqual(report.accuracy_score, 1.0)  # 价格合理
    
    def test_validate_price_data_with_issues(self):
        """测试有质量问题的 PriceData"""
        records = [
            self.OHLCVRecord(
                date=pd.Timestamp('2023-01-01'),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000
            ),
            self.OHLCVRecord(
                date=pd.Timestamp('2023-01-02'),
                open=103.0,
                high=108.0,
                low=102.0,
                close=-50.0,  # 负价格问题
                volume=1500
            ),
        ]
        
        price_data = self.PriceData(
            symbol='ISSUE',
            records=records,
            start_date=pd.Timestamp('2023-01-01'),
            end_date=pd.Timestamp('2023-01-02'),
            count=2
        )
        
        report = self.validate_data_quality(price_data)
        
        self.assertIsInstance(report, self.DataQualityReport)
        self.assertEqual(report.total_rows, 2)
        # 准确性评分应该降低（因为负价格）
        self.assertLess(report.accuracy_score, 1.0)
    
    def test_validate_price_data_integration(self):
        """测试 PriceData 完整集成"""
        # 模拟真实数据
        records = []
        base_price = 100.0
        for i in range(10):
            records.append(
                self.OHLCVRecord(
                    date=pd.Timestamp('2023-01-01') + pd.Timedelta(days=i),
                    open=base_price + i,
                    high=base_price + i + 2,
                    low=base_price + i - 1,
                    close=base_price + i + 1,
                    volume=1000 + i * 100
                )
            )
        
        price_data = self.PriceData(
            symbol='INTEGRATION',
            records=records,
            start_date=pd.Timestamp('2023-01-01'),
            end_date=pd.Timestamp('2023-01-10'),
            count=10
        )
        
        report = self.validate_data_quality(price_data)
        
        # 验证报告结构
        self.assertIsInstance(report, self.DataQualityReport)
        self.assertEqual(report.total_rows, 10)
        self.assertGreaterEqual(report.completeness_score, 0.0)
        self.assertLessEqual(report.completeness_score, 1.0)
        self.assertGreaterEqual(report.consistency_score, 0.0)
        self.assertLessEqual(report.consistency_score, 1.0)
        self.assertGreaterEqual(report.accuracy_score, 0.0)
        self.assertLessEqual(report.accuracy_score, 1.0)


if __name__ == '__main__':
    unittest.main()