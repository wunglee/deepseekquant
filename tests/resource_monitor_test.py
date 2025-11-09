#!/usr/bin/env python3
"""
DeepSeekQuant ResourceMonitor 测试模块 - 独立文件
"""

import unittest
import os
import sys
from unittest.mock import Mock

# 修复导入路径
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.components.resource_monitor import ResourceMonitor, ResourceMonitorConfig
from test_common import DeepSeekQuantTestBase


class TestResourceMonitor(DeepSeekQuantTestBase):
    """资源监控组件测试"""

    def setUp(self):
        self.config = ResourceMonitorConfig()
        self.monitor = ResourceMonitor(self.config, "TestProcessor")

    def test_01_start_stop(self):
        """测试启动和停止"""
        self.monitor.start()
        self.assertTrue(self.monitor.running)

        self.monitor.stop()
        self.assertFalse(self.monitor.running)

    # 在测试文件中修复资源监控测试
    def test_02_update_usage_with_psutil(self):
        """测试使用psutil更新资源使用情况"""
        # 创建Mock对象
        mock_process = Mock()

        # 设置内存信息Mock - 使用嵌套Mock结构
        mock_memory_info = Mock()
        memory_bytes = 33.46875 * 1024 * 1024  # 33.46875MB
        mock_memory_info.rss = memory_bytes
        mock_process.memory_info.return_value = mock_memory_info

        # 设置CPU和线程Mock
        mock_process.cpu_percent.return_value = 25.0
        mock_process.num_threads.return_value = 5

        # 将Mock对象设置到monitor中
        self.monitor.process = mock_process
        self.monitor.has_psutil = True

        self.monitor._update_usage()
        usage = self.monitor.get_usage()

        # 使用近似比较，允许微小差异
        self.assertAlmostEqual(usage['memory_mb'], 33.47, places=2)
        self.assertEqual(usage['cpu_percent'], 25.0)
        self.assertEqual(usage['thread_count'], 5)

    def test_03_update_usage_without_psutil(self):
        """测试没有psutil时的资源监控"""
        # 模拟没有psutil的情况
        self.monitor.has_psutil = False
        self.monitor.process = None

        self.monitor._update_usage()
        usage = self.monitor.get_usage()

        # 应该使用模拟数据
        self.assertEqual(usage['memory_mb'], 100.0)
        self.assertEqual(usage['cpu_percent'], 25.0)
        self.assertGreaterEqual(usage['thread_count'], 1)  # 至少有一个线程

    def test_04_assess_health(self):
        """测试健康评估"""
        # 设置准确的资源使用情况
        self.monitor.config.max_memory_mb = 512  # 使用配置的默认值
        self.monitor.usage.memory_mb = 450.0  # 450/512 ≈ 0.8789

        self.monitor.config.max_cpu_percent = 80
        self.monitor.usage.cpu_percent = 72.0  # 72/80 = 0.9

        health = self.monitor.assess_health()

        # 修正期望值
        self.assertEqual(health['memory_ratio'], 450.0 / 512)  # ≈0.8789
        self.assertEqual(health['cpu_ratio'], 0.9)

    def test_05_config_update(self):
        """测试配置更新"""
        new_config = ResourceMonitorConfig(
            monitor_interval=5,
            max_memory_mb=1024,
            max_cpu_percent=90
        )

        self.monitor.update_config(new_config)

        self.assertEqual(self.monitor.config.monitor_interval, 5)
        self.assertEqual(self.monitor.config.max_memory_mb, 1024)
        self.assertEqual(self.monitor.config.max_cpu_percent, 90)


if __name__ == '__main__':
    unittest.main(verbosity=2)
