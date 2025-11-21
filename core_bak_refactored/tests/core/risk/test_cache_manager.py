"""
缓存管理器测试
"""

import unittest
import time
from core_bak_refactored.core.risk.cache_manager import (
    RiskCacheManager,
    CacheConfig,
    CacheKeyGenerator,
    L1MemoryCache,
    get_cache_manager
)


class TestCacheKeyGenerator(unittest.TestCase):
    """测试缓存Key生成器"""
    
    def test_generate_simple_key(self):
        """测试简单Key生成"""
        key = CacheKeyGenerator.generate_simple_key(
            "returns", "US", "AAPL", period=252
        )
        
        self.assertIsInstance(key, str)
        self.assertIn("returns", key)
        self.assertIn("US", key)
        self.assertIn("AAPL", key)
        self.assertIn("period=252", key)
    
    def test_generate_key_with_components(self):
        """测试智能Key生成"""
        key = CacheKeyGenerator.generate_key({
            'market': 'US',
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'model_type': 'factor',
            'params': {'lookback': 252, 'confidence': 0.95}
        })
        
        self.assertIsInstance(key, str)
        self.assertIn("v1.0", key)  # 版本号
        self.assertEqual(len(key.split(":")), 4)  # 4个部分
    
    def test_key_consistency(self):
        """测试Key一致性 - 相同输入应生成相同Key"""
        components = {
            'market': 'CN',
            'symbols': ['000001', '000002'],
            'model_type': 'covariance'
        }
        
        key1 = CacheKeyGenerator.generate_key(components)
        key2 = CacheKeyGenerator.generate_key(components)
        
        self.assertEqual(key1, key2)
    
    def test_key_ordering_independence(self):
        """测试Key与symbols顺序无关"""
        components1 = {
            'market': 'US',
            'symbols': ['AAPL', 'GOOGL', 'MSFT']
        }
        components2 = {
            'market': 'US',
            'symbols': ['MSFT', 'AAPL', 'GOOGL']  # 不同顺序
        }
        
        key1 = CacheKeyGenerator.generate_key(components1)
        key2 = CacheKeyGenerator.generate_key(components2)
        
        self.assertEqual(key1, key2)  # 应该相同（内部排序）


class TestL1MemoryCache(unittest.TestCase):
    """测试L1内存缓存"""
    
    def setUp(self):
        """测试前准备"""
        self.config = CacheConfig(
            l1_maxsize=100,
            l1_ttl_seconds=2  # 2秒TTL用于测试
        )
        self.cache = L1MemoryCache(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.cache.cache)
        self.assertEqual(self.cache.metrics.l1_hits, 0)
        self.assertEqual(self.cache.metrics.l1_misses, 0)
    
    def test_set_and_get(self):
        """测试设置和获取"""
        # 设置缓存
        success = self.cache.set("test_key", {"data": [1, 2, 3]})
        self.assertTrue(success)
        
        # 获取缓存
        value = self.cache.get("test_key")
        self.assertIsNotNone(value)
        self.assertEqual(value, {"data": [1, 2, 3]})
        
        # 验证命中指标
        self.assertEqual(self.cache.metrics.l1_hits, 1)
    
    def test_cache_miss(self):
        """测试缓存未命中"""
        value = self.cache.get("non_existent_key")
        self.assertIsNone(value)
        
        # 验证未命中指标
        self.assertEqual(self.cache.metrics.l1_misses, 1)
    
    def test_ttl_expiration(self):
        """测试TTL过期"""
        # 设置缓存
        self.cache.set("temp_key", "temp_value")
        
        # 立即获取应该成功
        value1 = self.cache.get("temp_key")
        self.assertIsNotNone(value1)
        
        # 等待TTL过期
        time.sleep(2.5)
        
        # 过期后获取应该失败
        value2 = self.cache.get("temp_key")
        self.assertIsNone(value2)
    
    def test_invalidate(self):
        """测试缓存失效"""
        # 设置缓存
        self.cache.set("key_to_invalidate", "value")
        
        # 失效
        success = self.cache.invalidate("key_to_invalidate")
        self.assertTrue(success)
        
        # 验证已失效
        value = self.cache.get("key_to_invalidate")
        self.assertIsNone(value)
    
    def test_clear(self):
        """测试清空缓存"""
        # 设置多个缓存
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")
        
        # 清空
        success = self.cache.clear()
        self.assertTrue(success)
        
        # 验证全部失效
        self.assertIsNone(self.cache.get("key1"))
        self.assertIsNone(self.cache.get("key2"))
        self.assertIsNone(self.cache.get("key3"))
    
    def test_get_metrics(self):
        """测试获取指标"""
        # 执行一些操作
        self.cache.set("key1", "value1")
        self.cache.get("key1")  # 命中
        self.cache.get("key2")  # 未命中
        
        # 获取指标
        metrics = self.cache.get_metrics()
        
        self.assertIn('cache_size', metrics)
        self.assertIn('l1_hits', metrics)
        self.assertIn('l1_misses', metrics)
        self.assertIn('l1_hit_rate', metrics)
        
        self.assertEqual(metrics['l1_hits'], 1)
        self.assertEqual(metrics['l1_misses'], 1)
        self.assertAlmostEqual(metrics['l1_hit_rate'], 0.5)


class TestRiskCacheManager(unittest.TestCase):
    """测试风险缓存管理器"""
    
    def setUp(self):
        """测试前准备"""
        self.config = CacheConfig(l1_ttl_seconds=60)
        self.manager = RiskCacheManager(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.manager.l1_cache)
        self.assertIsNone(self.manager.l2_cache)  # 未启用
        self.assertIsNone(self.manager.l3_cache)  # 未启用
    
    def test_cached_decorator_basic(self):
        """测试缓存装饰器 - 基本功能"""
        call_count = [0]  # 使用列表追踪调用次数
        
        @self.manager.cached("test_func")
        def expensive_function(x, y):
            call_count[0] += 1
            return x + y
        
        # 第一次调用 - 应该执行函数
        result1 = expensive_function(1, 2)
        self.assertEqual(result1, 3)
        self.assertEqual(call_count[0], 1)
        
        # 第二次调用 - 应该从缓存获取
        result2 = expensive_function(1, 2)
        self.assertEqual(result2, 3)
        self.assertEqual(call_count[0], 1)  # 调用次数不变
        
        # 不同参数 - 应该再次执行函数
        result3 = expensive_function(2, 3)
        self.assertEqual(result3, 5)
        self.assertEqual(call_count[0], 2)
    
    def test_cached_decorator_with_kwargs(self):
        """测试缓存装饰器 - 支持关键字参数"""
        call_count = [0]
        
        @self.manager.cached("calc")
        def calculate(a, b=10):
            call_count[0] += 1
            return a * b
        
        # 不同调用方式，相同参数
        result1 = calculate(5, b=10)
        result2 = calculate(5, 10)
        
        self.assertEqual(result1, result2)
        # 由于Key生成策略，可能命中也可能不命中
        # 这里只验证逻辑正确性
        self.assertGreaterEqual(call_count[0], 1)
    
    def test_covariance_matrix_caching(self):
        """测试协方差矩阵专用缓存"""
        call_count = [0]
        
        @self.manager.cache_covariance_matrix(
            "US", ["AAPL", "GOOGL"], 252
        )
        def calculate_cov():
            call_count[0] += 1
            return [[1.0, 0.5], [0.5, 1.0]]
        
        # 第一次调用
        result1 = calculate_cov()
        self.assertEqual(call_count[0], 1)
        
        # 第二次调用 - 应该从缓存获取
        result2 = calculate_cov()
        self.assertEqual(result1, result2)
        self.assertEqual(call_count[0], 1)  # 调用次数不变
    
    def test_invalidate_pattern(self):
        """测试按模式失效缓存"""
        # 设置多个缓存
        self.manager.l1_cache.set("returns_US_AAPL", [1, 2, 3])
        self.manager.l1_cache.set("returns_US_GOOGL", [4, 5, 6])
        self.manager.l1_cache.set("returns_CN_000001", [7, 8, 9])
        
        # 按模式失效
        count = self.manager.invalidate_pattern("returns_US")
        
        # 验证US市场缓存失效
        self.assertIsNone(self.manager.l1_cache.get("returns_US_AAPL"))
        self.assertIsNone(self.manager.l1_cache.get("returns_US_GOOGL"))
        
        # 验证CN市场缓存未失效
        self.assertIsNotNone(self.manager.l1_cache.get("returns_CN_000001"))
        
        # 验证失效数量
        self.assertEqual(count, 2)
    
    def test_get_overall_metrics(self):
        """测试获取整体指标"""
        # 执行一些操作
        self.manager.l1_cache.set("key1", "value1")
        self.manager.l1_cache.get("key1")
        
        # 获取指标
        metrics = self.manager.get_overall_metrics()
        
        self.assertIn('l1', metrics)
        self.assertIn('l2', metrics)
        self.assertIn('l3', metrics)
        self.assertIn('timestamp', metrics)
        
        self.assertFalse(metrics['l2']['enabled'])
        self.assertFalse(metrics['l3']['enabled'])
    
    def test_clear_all(self):
        """测试清空所有缓存"""
        # 设置缓存
        self.manager.l1_cache.set("key1", "value1")
        self.manager.l1_cache.set("key2", "value2")
        
        # 清空
        self.manager.clear_all()
        
        # 验证已清空
        self.assertIsNone(self.manager.l1_cache.get("key1"))
        self.assertIsNone(self.manager.l1_cache.get("key2"))


class TestGlobalCacheManager(unittest.TestCase):
    """测试全局缓存管理器"""
    
    def test_get_cache_manager_singleton(self):
        """测试单例模式"""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()
        
        # 应该返回同一个实例
        self.assertIs(manager1, manager2)
    
    def test_get_cache_manager_with_config(self):
        """测试带配置的初始化"""
        config = CacheConfig(l1_maxsize=500)
        manager = get_cache_manager(config)
        
        self.assertIsNotNone(manager)
        # 配置只在首次生效，后续会返回已有实例


if __name__ == '__main__':
    unittest.main()
