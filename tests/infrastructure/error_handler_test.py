#!/usr/bin/env python3
"""
DeepSeekQuant ErrorHandler 测试模块 - 独立文件
"""

import unittest
import os
import sys

# 修复导入路径
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from infrastructure.error_handler import ErrorHandler, ErrorHandlerConfig


class TestErrorHandler(unittest.TestCase):
    """错误处理组件测试"""

    def setUp(self):
        self.config = ErrorHandlerConfig()
        self.handler = ErrorHandler(self.config, "TestProcessor")

    def test_01_initial_state(self):
        """测试初始状态"""
        summary = self.handler.get_error_summary()
        self.assertEqual(summary['total_errors'], 0)
        self.assertIsNone(summary['last_error'])
        self.assertEqual(summary['recent_errors'], 0)

    def test_02_record_error(self):
        """测试记录错误"""
        try:
            raise ValueError("测试错误")
        except ValueError as e:
            self.handler.record_error(e, "test_context")

        summary = self.handler.get_error_summary()
        self.assertEqual(summary['total_errors'], 1)
        self.assertIsNotNone(summary['last_error'])
        self.assertEqual(summary['last_error']['error_type'], "ValueError")
        self.assertEqual(summary['last_error']['context'], "test_context")

    def test_03_history_limit(self):
        """测试错误历史限制"""
        self.config.max_error_history = 3

        # 记录超过限制的错误
        for i in range(5):
            try:
                raise ValueError(f"错误 {i}")
            except ValueError as e:
                self.handler.record_error(e, f"context_{i}")

        self.assertEqual(len(self.handler.error_history), 3)
        self.assertEqual(self.handler.error_count, 5)  # 计数应该不受限制

    def test_04_disable_logging(self):
        """测试禁用错误日志记录"""
        self.config.enable_error_logging = False

        try:
            raise ValueError("测试错误")
        except ValueError as e:
            self.handler.record_error(e, "test_context")

        # 错误计数应该增加，但历史记录不应该
        self.assertEqual(self.handler.error_count, 1)
        self.assertEqual(len(self.handler.error_history), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
