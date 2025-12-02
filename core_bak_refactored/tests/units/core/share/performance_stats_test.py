"""
性能统计管理器测试
"""

import unittest
from core_bak_refactored.core.share.performance_stats import (
    PerformanceStatsManager, PerformanceMetrics
)


class TestPerformanceStatsManager(unittest.TestCase):
    """测试性能统计管理器"""
    
    def test_init_default(self):
        """测试默认初始化（不启用请求追踪）"""
        manager = PerformanceStatsManager()
        self.assertIsNotNone(manager._stats)
        self.assertFalse(manager._enable_request_tracking)
        self.assertIsNone(manager._request_tracker)
    
    def test_init_with_request_tracking(self):
        """测试启用请求追踪初始化"""
        manager = PerformanceStatsManager(enable_request_tracking=True)
        self.assertTrue(manager._enable_request_tracking)
        self.assertIsNotNone(manager._request_tracker)
    
    def test_increment_counter(self):
        """测试递增计数器"""
        manager = PerformanceStatsManager()
        manager.increment_counter('data_points_processed', 100)
        summary = manager.get_summary()
        self.assertIn('data_points_processed', summary)
    
    def test_record_request_without_tracking(self):
        """测试未启用追踪时记录请求"""
        manager = PerformanceStatsManager(enable_request_tracking=False)
        # 不应抛出异常
        manager.record_request('AAPL', True, 0.5, 'yahoo')
    
    def test_record_request_with_tracking(self):
        """测试启用追踪时记录请求"""
        manager = PerformanceStatsManager(enable_request_tracking=True)
        manager.record_request('AAPL', True, 0.5, 'yahoo')
        summary = manager.get_summary()
        # 应包含请求追踪统计
        self.assertIn('requests_total', summary)
    
    def test_record_cache_hit(self):
        """测试记录缓存命中"""
        manager = PerformanceStatsManager(enable_request_tracking=True)
        manager.record_cache_hit()
        summary = manager.get_summary()
        self.assertGreaterEqual(summary.get('cache_hits', 0), 1)
    
    def test_record_error(self):
        """测试记录错误"""
        manager = PerformanceStatsManager(enable_request_tracking=True)
        manager.record_error('TestError')
        summary = manager.get_summary()
        self.assertIn('error_counts', summary)


class TestPerformanceMetrics(unittest.TestCase):
    """测试性能指标数据类"""
    
    def test_metrics_creation(self):
        """测试创建性能指标"""
        metrics = PerformanceMetrics()
        self.assertEqual(metrics.throughput, 0.0)
        self.assertEqual(metrics.success_rate, 1.0)
    
    def test_metrics_with_request_tracking_fields(self):
        """测试带请求追踪字段的指标"""
        metrics = PerformanceMetrics(
            requests_total=100,
            cache_hits=50,
            cache_misses=50
        )
        self.assertEqual(metrics.requests_total, 100)
        self.assertEqual(metrics.cache_hits, 50)


if __name__ == '__main__':
    unittest.main()
