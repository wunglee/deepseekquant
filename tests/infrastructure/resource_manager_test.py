#!/usr/bin/env python3
"""
DeepSeekQuant ResourceManager 测试模块 - 独立文件
"""

import unittest
import os
import sys

# 修复导入路径
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from infrastructure.resource_manager import ResourceManager, ResourceMonitor, ResourceMonitorConfig
from common_test_base import DeepSeekQuantTestBase


class TestResourceManager(DeepSeekQuantTestBase):
    """资源管理器测试"""

    def setUp(self):
        self.monitor = ResourceMonitor(ResourceMonitorConfig(), "TestProcessor")
        # 使用模拟的资源使用，避免依赖psutil
        self.monitor.has_psutil = False
        self.monitor.process = None
        self.manager = ResourceManager("TestProcessor", self.monitor)

    def test_01_allocate_release_success(self):
        # 设置当前使用，使得有足够剩余内存
        self.monitor.usage.memory_mb = 100.0
        self.monitor.config.max_memory_mb = 512
        ok = self.manager.allocate_resource("memory", "res1", 100)
        self.assertTrue(ok)
        self.assertIn("res1", self.manager.allocated_resources)
        # 释放
        self.manager.release_resource("res1")
        self.assertNotIn("res1", self.manager.allocated_resources)

    def test_02_allocate_fail_due_to_limit(self):
        # 剩余不足
        self.monitor.usage.memory_mb = 510.0
        self.monitor.config.max_memory_mb = 512
        ok = self.manager.allocate_resource("memory", "res2", 20)
        self.assertFalse(ok)
        self.assertNotIn("res2", self.manager.allocated_resources)

    def test_03_cpu_limit_check(self):
        self.monitor.usage.cpu_percent = 75.0
        self.monitor.config.max_cpu_percent = 80
        # 可用约5，申请5应通过
        ok_pass = self.manager.allocate_resource("cpu", "cpu1", 5)
        self.assertTrue(ok_pass)
        # 申请更多应失败
        ok_fail = self.manager.allocate_resource("cpu", "cpu2", 10)
        self.assertFalse(ok_fail)


# 追加 ResourceMonitor 的测试用例

class TestResourceMonitor(DeepSeekQuantTestBase):
    def setUp(self):
        self.config = ResourceMonitorConfig()
        self.monitor = ResourceMonitor(self.config, "TestProcessor")

    def test_01_start_stop(self):
        self.monitor.start()
        self.assertTrue(self.monitor.running)
        self.monitor.stop()
        self.assertFalse(self.monitor.running)

    def test_02_update_usage_without_psutil(self):
        self.monitor.has_psutil = False
        self.monitor.process = None
        self.monitor._update_usage()
        usage = self.monitor.get_usage()
        self.assertEqual(usage['memory_mb'], 100.0)
        self.assertEqual(usage['cpu_percent'], 25.0)
        self.assertGreaterEqual(usage['thread_count'], 1)

if __name__ == '__main__':
    unittest.main(verbosity=2)
