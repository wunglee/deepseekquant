"""
测试断言工具类的单元测试

职责：
- 验证 TestAssertions 工具类的断言方法正确性
- 确保断言在预期场景下通过/失败
- 验证错误消息格式正确
"""

import unittest
import pandas as pd
from dataclasses import dataclass
from core_bak_refactored.tests.common.assertions_test import TestAssertions


@dataclass
class MockQualityReport:
    """Mock 数据质量报告"""
    overall_score: float


@dataclass
class MockUATResult:
    """Mock UAT 验收结果"""
    passed: bool
    details: str


class TestAssertionsTest(unittest.TestCase):
    """测试断言工具类的单元测试"""
    
    # ==================== assert_quality_score ====================
    
    def test_assert_quality_score_pass_with_default_threshold(self):
        """测试：质量评分达标（使用默认阈值60%）"""
        report = MockQualityReport(overall_score=0.75)
        
        # 应该通过，不抛出异常
        try:
            TestAssertions.assert_quality_score(self, report)
        except AssertionError:
            self.fail("质量评分75%应该通过默认阈值60%")
    
    def test_assert_quality_score_fail_below_threshold(self):
        """测试：质量评分未达标（低于阈值）"""
        report = MockQualityReport(overall_score=0.50)
        
        # 应该失败
        with self.assertRaises(AssertionError) as context:
            TestAssertions.assert_quality_score(self, report, threshold=0.60)
        
        # 验证错误消息
        self.assertIn("数据质量评分过低", str(context.exception))
        # 格式化后是 50.00%
        self.assertIn("50", str(context.exception))
    
    def test_assert_quality_score_real_data_higher_threshold(self):
        """测试：真实数据使用更高阈值90%"""
        report_pass = MockQualityReport(overall_score=0.92)
        report_fail = MockQualityReport(overall_score=0.75)
        
        # 92% 应该通过
        TestAssertions.assert_quality_score(self, report_pass, use_real_data=True)
        
        # 75% 应该失败（真实数据要求90%）
        with self.assertRaises(AssertionError):
            TestAssertions.assert_quality_score(self, report_fail, use_real_data=True)
    
    def test_assert_quality_score_with_context(self):
        """测试：错误消息包含上下文信息"""
        report = MockQualityReport(overall_score=0.50)
        
        with self.assertRaises(AssertionError) as context:
            TestAssertions.assert_quality_score(
                self, report, threshold=0.60, context="AKShare数据"
            )
        
        self.assertIn("AKShare数据", str(context.exception))
    
    # ==================== assert_error_within_threshold ====================
    
    def test_assert_error_within_threshold_pass(self):
        """测试：误差在阈值内"""
        TestAssertions.assert_error_within_threshold(
            self, error=0.15, threshold=0.25, context="预测误差"
        )
    
    def test_assert_error_within_threshold_fail(self):
        """测试：误差超过阈值"""
        with self.assertRaises(AssertionError) as context:
            TestAssertions.assert_error_within_threshold(
                self, error=0.30, threshold=0.25, context="预测误差"
            )
        
        self.assertIn("误差超限", str(context.exception))
        # 格式化后是 30.00%
        self.assertIn("30", str(context.exception))
        self.assertIn("25", str(context.exception))
    
    def test_assert_error_within_threshold_boundary(self):
        """测试：误差等于阈值（边界情况）"""
        # 等于阈值应该通过（使用 assertLessEqual）
        TestAssertions.assert_error_within_threshold(
            self, error=0.25, threshold=0.25
        )
    
    # ==================== assert_percentage_range ====================
    
    def test_assert_percentage_range_pass(self):
        """测试：百分比在范围内"""
        TestAssertions.assert_percentage_range(
            self, value=0.90, min_pct=0.85, max_pct=1.0, context="相关性"
        )
    
    def test_assert_percentage_range_fail_below_min(self):
        """测试：百分比低于下限"""
        with self.assertRaises(AssertionError) as context:
            TestAssertions.assert_percentage_range(
                self, value=0.80, min_pct=0.85, max_pct=1.0, context="相关性"
            )
        
        self.assertIn("低于下限", str(context.exception))
    
    def test_assert_percentage_range_fail_above_max(self):
        """测试：百分比高于上限"""
        with self.assertRaises(AssertionError) as context:
            TestAssertions.assert_percentage_range(
                self, value=1.10, min_pct=0.85, max_pct=1.0, context="相关性"
            )
        
        self.assertIn("高于上限", str(context.exception))
    
    def test_assert_percentage_range_boundaries(self):
        """测试：百分比等于边界值"""
        # 等于下限
        TestAssertions.assert_percentage_range(
            self, value=0.85, min_pct=0.85, max_pct=1.0
        )
        
        # 等于上限
        TestAssertions.assert_percentage_range(
            self, value=1.0, min_pct=0.85, max_pct=1.0
        )
    
    # ==================== assert_dataframe_valid ====================
    
    def test_assert_dataframe_valid_pass(self):
        """测试：DataFrame 有效"""
        df = pd.DataFrame({
            'date': ['2020-01-01', '2020-01-02'],
            'close': [100, 105],
            'volume': [1000, 1200]
        })
        
        TestAssertions.assert_dataframe_valid(
            self, df, 
            required_columns=['date', 'close', 'volume'],
            min_rows=2,
            context="价格数据"
        )
    
    def test_assert_dataframe_valid_fail_none(self):
        """测试：DataFrame 为 None"""
        with self.assertRaises(AssertionError) as context:
            TestAssertions.assert_dataframe_valid(
                self, None, context="价格数据"
            )
        
        self.assertIn("数据为None", str(context.exception))
    
    def test_assert_dataframe_valid_fail_wrong_type(self):
        """测试：数据类型错误"""
        with self.assertRaises(AssertionError) as context:
            TestAssertions.assert_dataframe_valid(
                self, {"data": "dict"}, context="价格数据"
            )
        
        self.assertIn("数据类型错误", str(context.exception))
    
    def test_assert_dataframe_valid_fail_insufficient_rows(self):
        """测试：数据行数不足"""
        df = pd.DataFrame({'close': [100]})
        
        with self.assertRaises(AssertionError) as context:
            TestAssertions.assert_dataframe_valid(
                self, df, min_rows=10, context="事件窗口"
            )
        
        self.assertIn("数据行数不足", str(context.exception))
    
    def test_assert_dataframe_valid_fail_missing_columns(self):
        """测试：缺少必需列"""
        df = pd.DataFrame({'close': [100, 105]})
        
        with self.assertRaises(AssertionError) as context:
            TestAssertions.assert_dataframe_valid(
                self, df, 
                required_columns=['date', 'close', 'volume'],
                context="价格数据"
            )
        
        self.assertIn("缺少必需列", str(context.exception))
        self.assertIn("date", str(context.exception))
        self.assertIn("volume", str(context.exception))
    
    # ==================== assert_uat_result_passed ====================
    
    def test_assert_uat_result_passed_pass(self):
        """测试：UAT 验收通过"""
        result = MockUATResult(passed=True, details="所有检查通过")
        
        TestAssertions.assert_uat_result_passed(
            self, result, context="加权平均误差"
        )
    
    def test_assert_uat_result_passed_fail(self):
        """测试：UAT 验收失败"""
        result = MockUATResult(passed=False, details="误差超过25%")
        
        with self.assertRaises(AssertionError) as context:
            TestAssertions.assert_uat_result_passed(
                self, result, context="加权平均误差"
            )
        
        self.assertIn("UAT验收失败", str(context.exception))
        self.assertIn("误差超过25%", str(context.exception))
    
    # ==================== assert_performance_metric ====================
    
    def test_assert_performance_metric_pass(self):
        """测试：性能达标"""
        TestAssertions.assert_performance_metric(
            self, elapsed_time=3.5, max_time=5.0, context="API响应"
        )
    
    def test_assert_performance_metric_fail(self):
        """测试：性能超限"""
        with self.assertRaises(AssertionError) as context:
            TestAssertions.assert_performance_metric(
                self, elapsed_time=6.5, max_time=5.0, context="API响应"
            )
        
        self.assertIn("性能超限", str(context.exception))
        self.assertIn("6.50s", str(context.exception))
        self.assertIn("5.0s", str(context.exception))
    
    def test_assert_performance_metric_boundary(self):
        """测试：性能等于上限"""
        TestAssertions.assert_performance_metric(
            self, elapsed_time=5.0, max_time=5.0
        )
    
    # ==================== assert_statistical_significance ====================
    
    def test_assert_statistical_significance_pass(self):
        """测试：统计显著"""
        TestAssertions.assert_statistical_significance(
            self, p_value=0.01, alpha=0.05, context="t检验"
        )
    
    def test_assert_statistical_significance_fail(self):
        """测试：统计不显著"""
        with self.assertRaises(AssertionError) as context:
            TestAssertions.assert_statistical_significance(
                self, p_value=0.10, alpha=0.05, context="t检验"
            )
        
        self.assertIn("统计不显著", str(context.exception))
        self.assertIn("p=0.1000", str(context.exception))
    
    def test_assert_statistical_significance_boundary(self):
        """测试：p值等于alpha（边界情况）"""
        # p=0.05 不应该通过（使用 assertLess，不是 assertLessEqual）
        with self.assertRaises(AssertionError):
            TestAssertions.assert_statistical_significance(
                self, p_value=0.05, alpha=0.05
            )


if __name__ == '__main__':
    unittest.main()
