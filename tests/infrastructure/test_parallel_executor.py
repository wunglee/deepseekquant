"""
并行执行器测试
"""

import unittest
import time
from core_bak_refactored.infrastructure.parallel_executor import (
    ParallelExecutor,
    ParallelConfig,
    get_parallel_executor
)


def cpu_intensive_task(n: int) -> int:
    """CPU密集型任务：计算斐波那契数"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def io_intensive_task(delay: float) -> str:
    """I/O密集型任务：模拟网络延迟"""
    time.sleep(delay)
    return f"completed after {delay}s"


def simple_square(x: int) -> int:
    """简单计算任务"""
    return x * x


class TestParallelConfig(unittest.TestCase):
    """测试并行配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = ParallelConfig()
        
        self.assertGreater(config.max_workers_cpu, 0)
        self.assertGreater(config.max_workers_io, 0)
        self.assertEqual(config.min_items_for_parallel_cpu, 10)
        self.assertEqual(config.min_items_for_parallel_io, 5)
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = ParallelConfig(
            max_workers_cpu=4,
            max_workers_io=8,
            min_items_for_parallel_cpu=20
        )
        
        self.assertEqual(config.max_workers_cpu, 4)
        self.assertEqual(config.max_workers_io, 8)
        self.assertEqual(config.min_items_for_parallel_cpu, 20)


class TestParallelExecutor(unittest.TestCase):
    """测试并行执行器"""
    
    def setUp(self):
        """测试前准备"""
        self.config = ParallelConfig(
            max_workers_cpu=2,
            max_workers_io=4,
            min_items_for_parallel_cpu=5,
            min_items_for_parallel_io=3
        )
        self.executor = ParallelExecutor(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.executor.config)
        self.assertIsNotNone(self.executor.metrics)
        self.assertEqual(self.executor.metrics.total_tasks, 0)
    
    def test_should_parallelize_positive(self):
        """测试并行判断 - 正面案例"""
        should, reason = self.executor.should_parallelize(10, 'cpu')
        self.assertTrue(should)
        self.assertIn("满足", reason)
    
    def test_should_parallelize_negative(self):
        """测试并行判断 - 负面案例"""
        should, reason = self.executor.should_parallelize(3, 'cpu')
        self.assertFalse(should)
        self.assertIn("小于", reason)
    
    def test_map_cpu_intensive_small_dataset(self):
        """测试CPU密集型映射 - 小数据集（串行）"""
        items = [10, 15, 20]  # 小于阈值，应该串行
        results = self.executor.map_cpu_intensive(cpu_intensive_task, items)
        
        self.assertEqual(len(results), len(items))
        self.assertEqual(results[0], cpu_intensive_task(10))
        self.assertEqual(results[1], cpu_intensive_task(15))
    
    def test_map_cpu_intensive_large_dataset(self):
        """测试CPU密集型映射 - 大数据集（并行）"""
        items = list(range(10, 20))  # 10个任务，应该并行
        results = self.executor.map_cpu_intensive(simple_square, items)
        
        self.assertEqual(len(results), len(items))
        expected = [x * x for x in items]
        self.assertEqual(results, expected)
        
        # 验证指标
        metrics = self.executor.get_metrics()
        self.assertGreater(metrics['completed_tasks'], 0)
    
    def test_map_io_intensive(self):
        """测试I/O密集型映射"""
        items = [0.01, 0.01, 0.01, 0.01]  # 4个任务
        
        start_time = time.time()
        results = self.executor.map_io_intensive(io_intensive_task, items)
        elapsed = time.time() - start_time
        
        self.assertEqual(len(results), len(items))
        # 并行执行应该比串行快
        self.assertLess(elapsed, 0.04 * 1.5)  # 允许50%误差
    
    def test_map_with_progress_callback(self):
        """测试带进度回调的映射"""
        items = list(range(10))
        progress_updates = []
        
        def progress_callback(completed, total):
            progress_updates.append((completed, total))
        
        results = self.executor.map_with_progress(
            simple_square,
            items,
            task_type='cpu',
            progress_callback=progress_callback
        )
        
        self.assertEqual(len(results), len(items))
        self.assertGreater(len(progress_updates), 0)
        
        # 验证最后的进度是100%
        if progress_updates:
            last_completed, last_total = progress_updates[-1]
            self.assertEqual(last_completed, last_total)
    
    def test_metrics_tracking(self):
        """测试性能指标追踪"""
        items = list(range(5, 15))
        
        # 重置指标
        self.executor.reset_metrics()
        
        # 执行任务
        self.executor.map_cpu_intensive(simple_square, items)
        
        # 获取指标
        metrics = self.executor.get_metrics()
        
        self.assertEqual(metrics['total_tasks'], len(items))
        self.assertEqual(metrics['completed_tasks'], len(items))
        self.assertEqual(metrics['failed_tasks'], 0)
        self.assertGreater(metrics['total_time_seconds'], 0)
    
    def test_reset_metrics(self):
        """测试指标重置"""
        items = list(range(10))
        self.executor.map_cpu_intensive(simple_square, items)
        
        # 重置前应该有数据
        metrics_before = self.executor.get_metrics()
        self.assertGreater(metrics_before['total_tasks'], 0)
        
        # 重置
        self.executor.reset_metrics()
        
        # 重置后应该清零
        metrics_after = self.executor.get_metrics()
        self.assertEqual(metrics_after['total_tasks'], 0)
        self.assertEqual(metrics_after['completed_tasks'], 0)


class TestGlobalExecutor(unittest.TestCase):
    """测试全局执行器"""
    
    def test_get_parallel_executor_singleton(self):
        """测试单例模式"""
        executor1 = get_parallel_executor()
        executor2 = get_parallel_executor()
        
        self.assertIs(executor1, executor2)


if __name__ == '__main__':
    unittest.main()
