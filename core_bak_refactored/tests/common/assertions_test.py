"""
通用测试断言工具（helper模块，不要求一对一映射，保留 test_ 前缀）

职责：
- 提供标准化的测试断言方法
- 统一错误信息格式
- 增强测试可读性和可维护性

说明：
- 本文件作为公共测试工具模块存在，不对应单一生产代码文件
- 因此命名上保留 `test_assertions.py`，在“一对一”规则中视为工具例外
"""

import unittest
from typing import Any, Optional


class TestAssertions:
    """
    通用测试断言工具类
    
    职责：提供标准化的断言方法，供所有测试用例复用
    """
    
    @staticmethod
    def assert_quality_score(
        test_case: unittest.TestCase,
        quality_report,
        threshold: float = 0.60,
        use_real_data: bool = False,
        context: str = ""
    ):
        """
        断言数据质量评分达标
        
        Args:
            test_case: unittest.TestCase实例
            quality_report: 数据质量报告对象（需有overall_score属性）
            threshold: 质量阈值（默认60%）
            use_real_data: 是否使用真实数据（影响阈值）
            context: 上下文信息（用于错误提示）
        
        Examples:
            >>> class MyTest(unittest.TestCase):
            ...     def test_quality(self):
            ...         report = quality_checker.check_quality(data)
            ...         TestAssertions.assert_quality_score(self, report)
        """
        actual_threshold = 0.90 if use_real_data else threshold
        
        test_case.assertGreaterEqual(
            quality_report.overall_score,
            actual_threshold,
            f"{context}数据质量评分过低: {quality_report.overall_score:.2%} < {actual_threshold:.0%}"
        )
    
    @staticmethod
    def assert_error_within_threshold(
        test_case: unittest.TestCase,
        error: float,
        threshold: float,
        context: str = ""
    ):
        """
        断言误差在阈值内
        
        Args:
            test_case: unittest.TestCase实例
            error: 误差值
            threshold: 阈值
            context: 上下文信息
        
        Examples:
            >>> TestAssertions.assert_error_within_threshold(
            ...     self, prediction_error, 0.25, "单事件"
            ... )
        """
        test_case.assertLessEqual(
            error,
            threshold,
            f"{context}误差超限: {error:.2%} > {threshold:.0%}"
        )
    
    @staticmethod
    def assert_percentage_range(
        test_case: unittest.TestCase,
        value: float,
        min_pct: float,
        max_pct: float,
        context: str = ""
    ):
        """
        断言百分比值在指定范围内
        
        Args:
            test_case: unittest.TestCase实例
            value: 待验证的值
            min_pct: 最小百分比（如0.10表示10%）
            max_pct: 最大百分比
            context: 上下文信息
        
        Examples:
            >>> TestAssertions.assert_percentage_range(
            ...     self, correlation, 0.85, 1.0, "跨市场相关性"
            ... )
        """
        test_case.assertGreaterEqual(
            value, min_pct,
            f"{context}低于下限: {value:.2%} < {min_pct:.0%}"
        )
        test_case.assertLessEqual(
            value, max_pct,
            f"{context}高于上限: {value:.2%} > {max_pct:.0%}"
        )
    
    @staticmethod
    def assert_dataframe_valid(
        test_case: unittest.TestCase,
        data,
        required_columns: list = None,
        min_rows: int = 1,
        context: str = ""
    ):
        """
        断言DataFrame有效性
        
        Args:
            test_case: unittest.TestCase实例
            data: DataFrame对象
            required_columns: 必需的列名列表
            min_rows: 最小行数
            context: 上下文信息
        
        Examples:
            >>> TestAssertions.assert_dataframe_valid(
            ...     self, event_data, ['close', 'volume'], min_rows=10, "事件窗口数据"
            ... )
        """
        import pandas as pd
        
        test_case.assertIsNotNone(data, f"{context}数据为None")
        test_case.assertIsInstance(data, pd.DataFrame, f"{context}数据类型错误")
        test_case.assertGreaterEqual(
            len(data), min_rows,
            f"{context}数据行数不足: {len(data)} < {min_rows}"
        )
        
        if required_columns:
            missing_cols = set(required_columns) - set(data.columns)
            test_case.assertEqual(
                len(missing_cols), 0,
                f"{context}缺少必需列: {missing_cols}"
            )
    
    @staticmethod
    def assert_uat_result_passed(
        test_case: unittest.TestCase,
        uat_result,
        context: str = ""
    ):
        """
        断言UAT验收结果通过
        
        Args:
            test_case: unittest.TestCase实例
            uat_result: UAT验收结果对象（需有passed和details属性）
            context: 上下文信息
        
        Examples:
            >>> result = validator.validate_weighted_average_error(...)
            >>> TestAssertions.assert_uat_result_passed(self, result, "加权平均误差")
        """
        test_case.assertTrue(
            uat_result.passed,
            f"{context}UAT验收失败: {uat_result.details}"
        )
    
    @staticmethod
    def assert_performance_metric(
        test_case: unittest.TestCase,
        elapsed_time: float,
        max_time: float,
        context: str = ""
    ):
        """
        断言性能指标达标
        
        Args:
            test_case: unittest.TestCase实例
            elapsed_time: 实际耗时（秒）
            max_time: 最大耗时限制（秒）
            context: 上下文信息
        
        Examples:
            >>> TestAssertions.assert_performance_metric(
            ...     self, elapsed_time, 5.0, "系统响应时间"
            ... )
        """
        test_case.assertLessEqual(
            elapsed_time, max_time,
            f"{context}性能超限: {elapsed_time:.2f}s > {max_time:.1f}s"
        )
    
    @staticmethod
    def assert_statistical_significance(
        test_case: unittest.TestCase,
        p_value: float,
        alpha: float = 0.05,
        context: str = ""
    ):
        """
        断言统计显著性
        
        Args:
            test_case: unittest.TestCase实例
            p_value: p值
            alpha: 显著性水平（默认0.05）
            context: 上下文信息
        
        Examples:
            >>> TestAssertions.assert_statistical_significance(
            ...     self, p_value, 0.05, "行业差异t检验"
            ... )
        """
        test_case.assertLess(
            p_value, alpha,
            f"{context}统计不显著: p={p_value:.4f} >= {alpha}"
        )
