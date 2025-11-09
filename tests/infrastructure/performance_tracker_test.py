#!/usr/bin/env python3
"""
DeepSeekQuant PerformanceTracker 测试模块 - 独立文件
"""

import unittest
import os
import sys

# 修复导入路径
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.performance_tracker import PerformanceTracker, PerformanceConfig
from test_common import DeepSeekQuantTestBase


class TestPerformanceTracker(DeepSeekQuantTestBase):
    """性能跟踪组件测试"""

    def setUp(self):
        self.config = PerformanceConfig()
        self.tracker = PerformanceTracker(self.config, "TestProcessor")

    def test_01_initial_state(self):
        """测试初始状态"""
        report = self.tracker.get_report()
        self.assertEqual(report['total_operations'], 0)
        self.assertEqual(report['successful_operations'], 0)
        self.assertEqual(report['failed_operations'], 0)
        self.assertEqual(report['error_rate'], 0.0)

    # 在测试文件中修复浮点数比较
    def test_02_record_success(self):
        """测试记录成功操作"""
        self.tracker.record_success(0.1)
        self.tracker.record_success(0.2)

        report = self.tracker.get_report()
        self.assertEqual(report['total_operations'], 2)
        self.assertEqual(report['successful_operations'], 2)
        self.assertEqual(report['failed_operations'], 0)
        self.assertEqual(report['error_rate'], 0.0)
        # 使用近似比较
        self.assertAlmostEqual(report['avg_processing_time'], 0.15, places=2)

    def test_03_record_failure(self):
        """测试记录失败操作"""
        self.tracker.record_success(0.1)
        self.tracker.record_failure(0.2)
        self.tracker.record_failure(0.3)

        report = self.tracker.get_report()
        self.assertEqual(report['total_operations'], 3)
        self.assertEqual(report['successful_operations'], 1)
        self.assertEqual(report['failed_operations'], 2)
        self.assertAlmostEqual(report['error_rate'], 2 / 3, places=2)

    def test_04_history_limit(self):
        """测试历史记录限制"""
        self.config.max_history_size = 5

        # 记录超过限制的操作
        for i in range(10):
            self.tracker.record_success(0.1)

        self.assertEqual(len(self.tracker.history), 5)

    def test_05_get_error_rate(self):
        """测试获取错误率"""
        for i in range(4):
            self.tracker.record_success(0.1)
        for i in range(1):
            self.tracker.record_failure(0.2)

        error_rate = self.tracker.get_error_rate()
        self.assertEqual(error_rate, 0.2)  # 1/5 = 0.2


if __name__ == '__main__':
    unittest.main(verbosity=2)
