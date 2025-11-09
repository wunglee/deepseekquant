#!/usr/bin/env python3
"""
DeepSeekQuant ProcessorManager 测试模块 - 独立文件
"""

import unittest
import os
import sys

# 修复导入路径
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infrastructure.processor_manager import ProcessorManager
from common import ProcessorState
from base_processor_test import TestProcessor


def add_unregister_method():

    def unregister_processor(self, processor_name: str) -> bool:
        """取消注册处理器"""
        with self.manager_lock:
            if processor_name not in self.processors:
                return False
            processor = self.processors[processor_name]
            try:
                processor.cleanup()
            except:
                pass
            del self.processors[processor_name]
            return True

    def cleanup_all(self):
        """清理所有处理器"""
        with self.manager_lock:
            for name, processor in list(self.processors.items()):
                try:
                    processor.cleanup()
                except Exception as e:
                    self.logger.error(f"处理器清理异常 {name}: {e}")
                del self.processors[name]

    # 动态添加方法
    setattr(ProcessorManager, "unregister_processor", unregister_processor)  # type: ignore[attr-defined]
    setattr(ProcessorManager, "cleanup_all", cleanup_all)  # type: ignore[attr-defined]


# 在运行测试前调用
add_unregister_method()


class TestProcessorManager(unittest.TestCase):
    """ProcessorManager 测试"""

    def setUp(self):
        self.manager = ProcessorManager()

    def tearDown(self):
        # 清理所有处理器
        for name, processor in list(self.manager.processors.items()):
            processor.cleanup()
            getattr(self.manager, "unregister_processor")(name)

    def test_01_processor_registration(self):
        """测试处理器注册"""
        processor = TestProcessor()

        success = self.manager.register_processor(processor)

        self.assertTrue(success)
        self.assertIn("TestProcessor", self.manager.processors)
        self.assertEqual(
            self.manager.processors["TestProcessor"],
            processor
        )

    def test_02_duplicate_registration(self):
        """测试重复注册"""
        processor1 = TestProcessor()
        processor2 = TestProcessor()

        success1 = self.manager.register_processor(processor1)
        success2 = self.manager.register_processor(processor2)

        self.assertTrue(success1)
        self.assertFalse(success2)  # 同名处理器应该注册失败

    def test_03_bulk_operations(self):
        """测试批量操作"""
        processors = []
        for i in range(3):
            processor = TestProcessor(processor_name=f"Processor_{i}")
            processors.append(processor)
            self.manager.register_processor(processor)

        # 测试批量初始化
        results = self.manager.initialize_all()

        self.assertEqual(len(results), 3)
        for name, success in results.items():
            self.assertTrue(success)

        # 测试健康报告
        health_report = self.manager.get_health_report()

        self.assertEqual(health_report["total_processors"], 3)
        self.assertEqual(health_report["healthy_processors"], 3)
        self.assertIn("processor_details", health_report)

        # 测试批量清理
        getattr(self.manager, "cleanup_all")()

        for processor in processors:
            self.assertEqual(processor.state, ProcessorState.TERMINATED)


if __name__ == '__main__':
    unittest.main(verbosity=2)
