#!/usr/bin/env python3
"""
DeepSeekQuant CircuitBreaker 测试模块 - 独立文件
"""

import unittest
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch

# 修复导入路径
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.components.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from test_common import DeepSeekQuantTestBase


class TestCircuitBreaker(DeepSeekQuantTestBase):
    """熔断器组件测试"""

    def setUp(self):
        self.config = CircuitBreakerConfig()
        self.circuit_breaker = CircuitBreaker(self.config, "TestProcessor")

    def test_01_initial_state(self):
        """测试初始状态"""
        status = self.circuit_breaker.get_status()
        self.assertEqual(status['state'], "CLOSED")
        self.assertEqual(status['failure_count'], 0)

    def test_02_allow_request_closed(self):
        """测试关闭状态允许请求"""
        self.assertTrue(self.circuit_breaker.allow_request())

    def test_03_record_failure_threshold(self):
        """测试失败记录达到阈值"""
        for i in range(self.config.failure_threshold):
            self.circuit_breaker.record_failure()

        status = self.circuit_breaker.get_status()
        self.assertEqual(status['state'], "OPEN")
        self.assertEqual(status['failure_count'], self.config.failure_threshold)

    def test_04_allow_request_open(self):
        """测试打开状态拒绝请求"""
        # 触发熔断
        for i in range(self.config.failure_threshold):
            self.circuit_breaker.record_failure()

        self.assertFalse(self.circuit_breaker.allow_request())

    def test_05_record_success_reset(self):
        """测试成功记录重置"""
        # 记录一些失败
        self.circuit_breaker.record_failure()
        self.circuit_breaker.record_failure()

        # 记录成功
        self.circuit_breaker.record_success()

        status = self.circuit_breaker.get_status()
        self.assertEqual(self.circuit_breaker.get_status()['failure_count'], 0)

    def test_06_half_open_state(self):
        """测试半开状态"""
        # 触发熔断器打开
        for i in range(self.config.failure_threshold):
            self.circuit_breaker.record_failure()

        # 验证熔断器已打开
        self.assertEqual(self.circuit_breaker.state.state, "OPEN")
        self.assertFalse(self.circuit_breaker.allow_request())

        # 模拟时间过去，进入半开状态
        recovery_timeout = self.config.recovery_timeout
        future_time = datetime.now() + timedelta(seconds=recovery_timeout + 1)

        import core.components.circuit_breaker as cb_module
        with patch.object(cb_module, 'datetime') as mock_datetime:
            mock_datetime.now.return_value = future_time
            mock_datetime.fromisoformat = datetime.fromisoformat
            mock_datetime.fromtimestamp = datetime.fromtimestamp

            # 现在应该允许请求（半开状态）
            self.assertTrue(self.circuit_breaker.allow_request())
            self.assertEqual(self.circuit_breaker.state.state, "HALF_OPEN")

    def test_07_half_open_to_closed(self):
        """测试半开状态转为关闭"""
        # 进入半开状态
        self.circuit_breaker.state.state = "HALF_OPEN"

        # 记录足够的成功次数
        for i in range(self.config.half_open_max_requests):
            self.circuit_breaker.record_success()

        status = self.circuit_breaker.get_status()
        self.assertEqual(status['state'], "CLOSED")


if __name__ == '__main__':
    unittest.main(verbosity=2)
