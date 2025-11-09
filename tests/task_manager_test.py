#!/usr/bin/env python3
"""
DeepSeekQuant TaskManager 测试模块 - 独立文件
"""

import unittest
import os
import sys

# 修复导入路径
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.managers.task_manager import TaskManager, TaskManagerConfig


class TestTaskManager(unittest.TestCase):
    """任务管理组件测试"""

    def setUp(self):
        self.config = TaskManagerConfig()
        self.manager = TaskManager(self.config, "TestProcessor")

    def test_01_initialization(self):
        """测试初始化"""
        self.assertIsNone(self.manager.thread_pool)
        self.manager.initialize()
        self.assertIsNotNone(self.manager.thread_pool)

    def test_02_task_lifecycle(self):
        """测试任务生命周期"""
        self.manager.initialize()

        # 生成任务ID
        task_id = self.manager.generate_task_id()
        self.assertTrue(task_id.startswith("task_"))

        # 记录任务开始
        args = ("test_arg",)
        kwargs = {"test_key": "test_value"}
        self.manager.record_task_start(task_id, args, kwargs)

        self.assertIn(task_id, self.manager.active_tasks)
        self.assertIn(task_id, self.manager.task_queue)

        # 记录任务成功
        result = {"status": "success"}
        self.manager.record_task_success(task_id, result, 0.1)

        task_info = self.manager.active_tasks[task_id]
        self.assertEqual(task_info.result, result)
        self.assertEqual(task_info.processing_time, 0.1)

        # 记录任务结束
        self.manager.record_task_end(task_id)
        self.assertNotIn(task_id, self.manager.active_tasks)
        self.assertNotIn(task_id, self.manager.task_queue)

    def test_03_submit_task(self):
        """测试提交任务"""
        self.manager.initialize()

        def test_task(x):
            return x * 2

        future = self.manager.submit_task(test_task, 21)
        result = future.result(timeout=5)

        self.assertEqual(result, 42)

    def test_04_queue_status(self):
        """测试队列状态"""
        self.manager.initialize()

        status = self.manager.get_queue_status()
        self.assertEqual(status['active_tasks'], 0)
        self.assertEqual(status['queue_size'], 0)
        self.assertTrue(status['thread_pool_active'])

    def test_05_emergency_stop(self):
        """测试紧急停止"""
        self.manager.initialize()

        # 提交一些任务
        for i in range(3):
            task_id = self.manager.generate_task_id()
            self.manager.record_task_start(task_id, (i,), {})

        self.assertEqual(len(self.manager.active_tasks), 3)
        self.assertEqual(len(self.manager.task_queue), 3)

        # 紧急停止
        self.manager.emergency_stop()

        self.assertIsNone(self.manager.thread_pool)
        self.assertEqual(len(self.manager.active_tasks), 0)
        self.assertEqual(len(self.manager.task_queue), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
