#!/usr/bin/env python3
"""
DeepSeekQuant BaseProcessor 测试模块 - 重构版本
测试新的组件化架构
"""

import unittest
import time
import threading
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from concurrent.futures import Future
from typing import Any

# 统一测试基类

# 修复导入路径
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入被测试的模块
from core.processors.base_processor import (
    BaseProcessor, ProcessorConfig, ProcessorState, HealthStatus
)

# 导入组件
from infrastructure.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState
from infrastructure.resource_monitor import ResourceMonitor, ResourceMonitorConfig, ResourceUsage
from infrastructure.performance_tracker import PerformanceTracker, PerformanceConfig
from error_handler import ErrorHandler, ErrorHandlerConfig, ErrorRecord
from core.managers.task_manager import TaskManager, TaskManagerConfig, TaskInfo

class TestProcessor(BaseProcessor):
    """测试用处理器实现"""

    def __init__(self, *args, **kwargs):
        # 测试参数
        self.process_delay = kwargs.pop('process_delay', 0)
        self.should_fail = kwargs.pop('should_fail', False)
        self.raise_exception = kwargs.pop('raise_exception', False)

        # 状态标志
        self.initialize_called = False
        self.cleanup_called = False
        self.process_called = False

        super().__init__(*args, **kwargs)

    def _initialize_core(self) -> bool:
        """测试初始化逻辑"""
        self.initialize_called = True
        time.sleep(0.01)  # 模拟初始化延迟
        return True

    def _process_core(self, *args, **kwargs) -> Any:
        """测试处理逻辑"""
        self.process_called = True

        if self.raise_exception:
            raise ValueError("测试异常")

        if self.should_fail:
            return {"status": "error", "message": "处理失败"}

        # 模拟处理延迟
        if self.process_delay > 0:
            time.sleep(self.process_delay)

        # 处理数据
        data = None
        if args and len(args) > 0:
            if isinstance(args[0], dict):
                data = args[0].get('data', 'default')
            else:
                data = str(args[0])
        elif kwargs:
            data = kwargs.get('data', 'default')

        return {
            "status": "success",
            "data": data if data is not None else 'default',
            "timestamp": datetime.now().isoformat()
        }

    def _cleanup_core(self):
        """测试清理逻辑"""
        self.cleanup_called = True
        time.sleep(0.01)  # 模拟清理延迟












class TestBaseProcessor(unittest.TestCase):
    """BaseProcessor 基础功能测试"""

    def setUp(self):
        """测试前准备"""
        self.test_config = {
            "processor_config": {
                "enabled": True,
                "max_threads": 4,  # 这里应该是4，测试期望是4
                "processing_timeout": 5,
                "retry_attempts": 2,
                "performance_monitoring": True,
                "resource_monitor": {  # 使用正确的字段名
                    "max_memory_mb": 100,
                    "max_cpu_percent": 50
                }
            }
        }

    def test_01_initialization(self):
        """测试处理器初始化"""
        processor = TestProcessor(config=self.test_config)

        # 验证初始状态
        self.assertEqual(processor.state, ProcessorState.UNINITIALIZED)
        self.assertEqual(processor.health_status, HealthStatus.UNKNOWN)
        self.assertIsNotNone(processor.logger)

        # 验证组件已初始化
        self.assertIsInstance(processor.circuit_breaker, CircuitBreaker)
        self.assertIsInstance(processor.resource_monitor, ResourceMonitor)
        self.assertIsInstance(processor.performance_tracker, PerformanceTracker)
        self.assertIsInstance(processor.error_handler, ErrorHandler)
        self.assertIsInstance(processor.task_manager, TaskManager)

    def test_02_successful_initialization(self):
        """测试成功初始化"""
        processor = TestProcessor(config=self.test_config)
        result = processor.initialize()

        self.assertTrue(result)
        self.assertEqual(processor.state, ProcessorState.READY)
        self.assertTrue(processor.initialize_called)
        self.assertEqual(processor.health_status, HealthStatus.HEALTHY)

    def test_03_configuration_management(self):
        """测试配置管理"""
        processor = TestProcessor(config=self.test_config)
        processor.initialize()

        self.assertIsNotNone(processor.processor_config)
        expected_threads = 4  # 应该与test_config中的配置一致
        self.assertEqual(processor.processor_config.max_threads, expected_threads)
        self.assertTrue(processor.processor_config.enabled)

    def test_04_basic_processing(self):
        """测试基本处理功能"""
        processor = TestProcessor(config=self.test_config)
        processor.initialize()

        # 测试正常处理
        test_data = {"data": "test_data", "value": 42}
        result = processor.process(**test_data)

        self.assertTrue(processor.process_called)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"], "test_data")
        self.assertIn("timestamp", result)

    def test_05_error_handling(self):
        """测试错误处理"""
        processor = TestProcessor(
            config=self.test_config,
            should_fail=True
        )
        processor.initialize()

        # 确保错误处理器已正确初始化
        self.assertIsNotNone(processor.error_handler)

        # 处理一个应该失败的任务
        result = processor.process(data="error_test")

        # 验证错误状态
        self.assertEqual(result["status"], "error")

        # 验证错误计数增加
        self.assertEqual(processor.error_handler.error_count, 1)

    def test_06_exception_handling(self):
        """测试异常处理"""
        processor = TestProcessor(
            config=self.test_config,
            raise_exception=True
        )
        processor.initialize()

        result = processor.process(data="exception_test")

        self.assertIn("error", result)
        self.assertEqual(processor.error_handler.error_count, 1)
        self.assertIsNotNone(processor.error_handler.last_error)
        self.assertIsNotNone(processor.error_handler.last_error)
        self.assertEqual(getattr(processor.error_handler.last_error, "error_type", None), "ValueError")

    def test_07_performance_metrics(self):
        """测试性能指标收集"""
        processor = TestProcessor(
            config=self.test_config,
            process_delay=0.1
        )
        processor.initialize()

        # 执行多次处理
        for i in range(5):
            processor.process(data=f"test_{i}")

        metrics = processor.get_performance_report()

        self.assertEqual(metrics["total_operations"], 5)
        self.assertEqual(metrics["successful_operations"], 5)
        self.assertGreater(metrics["avg_processing_time"], 0)
        self.assertLessEqual(metrics["error_rate"], 0.0)

    def test_08_health_status(self):
        """测试健康状态检查"""
        processor = TestProcessor(config=self.test_config)
        processor.initialize()

        health_status = processor.get_health_status()

        self.assertTrue(health_status["is_healthy"])
        self.assertEqual(processor.state, ProcessorState.READY)
        self.assertIn("resource_usage", health_status)
        self.assertIn("performance", health_status)
        self.assertIn("circuit_breaker", health_status)

    def test_09_circuit_breaker(self):
        """测试熔断器机制"""
        processor = TestProcessor(
            config=self.test_config,
            raise_exception=True
        )
        processor.initialize()

        # 触发多次失败，应该打开熔断器
        for i in range(5):
            processor.process(data=f"cb_test_{i}")

        self.assertEqual(processor.circuit_breaker.state.state, "OPEN")
        self.assertEqual(processor.circuit_breaker.state.failure_count, 5)

        # 验证熔断器拒绝请求
        result = processor.process(data="should_be_rejected")
        self.assertIn("circuit_breaker", result)
        self.assertEqual(result["circuit_breaker"], "open")

    def test_10_async_processing(self):
        """测试异步处理"""
        processor = TestProcessor(config=self.test_config)
        processor.initialize()

        # 提交异步任务
        def test_task(x):
            return x * 2

        future = processor.submit_task(test_task, 21)
        result = future.result(timeout=5)

        self.assertEqual(result, 42)

    def test_11_batch_processing(self):
        """测试批量处理"""
        processor = TestProcessor(config=self.test_config)
        processor.initialize()

        # 创建测试项目
        test_items = [{"data": f"item_{i}"} for i in range(10)]
        results = processor.batch_process(test_items, batch_size=3)

        self.assertEqual(len(results), 10)
        for i, result in enumerate(results):
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["data"], f"item_{i}")

    def test_12_cleanup(self):
        """测试资源清理"""
        processor = TestProcessor(config=self.test_config)
        processor.initialize()
        processor.process(data="test")

        processor.cleanup()

        self.assertTrue(processor.cleanup_called)
        self.assertEqual(processor.state, ProcessorState.TERMINATED)
        self.assertIsNone(processor.task_manager.thread_pool)

    def test_13_context_manager(self):
        """测试上下文管理器"""
        with TestProcessor(config=self.test_config) as processor:
            self.assertEqual(processor.state, ProcessorState.READY)
            result = processor.process(data="context_test")
            self.assertEqual(result["status"], "success")

        # 退出上下文后应该已清理
        self.assertEqual(processor.state, ProcessorState.TERMINATED)

    def test_14_restart_functionality(self):
        """测试重启功能"""
        processor = TestProcessor(config=self.test_config)

        # 先初始化
        processor.initialize()

        # 先制造一些状态
        processor.process(data="before_restart")
        original_metrics = processor.get_performance_report()

        # 执行重启
        success = processor.restart()

        self.assertTrue(success)
        self.assertEqual(processor.state, ProcessorState.READY)

        # 验证可以继续处理
        result = processor.process(data="after_restart")
        self.assertEqual(result["status"], "success")

        # 清理
        processor.cleanup()

    def test_15_emergency_stop(self):
        """测试紧急停止"""
        processor = TestProcessor(config=self.test_config)
        processor.initialize()

        processor.emergency_stop()

        self.assertEqual(processor.state, ProcessorState.ERROR)
        self.assertFalse(processor.resource_monitor.running)
        self.assertIsNone(processor.task_manager.thread_pool)





if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)