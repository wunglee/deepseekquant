"""
增强版缓存服务测试
"""

import unittest
import time
from infrastructure.cache_service import (
    CacheService,
    CacheConfig,
    CacheKeyGenerator,
    get_cache_service
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
        self.assertIn("v1.0", key)
        self.assertEqual(len(key.split(":")), 4)
    
    def test_key_consistency(self):
        """测试Key一致性"""
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
            'symbols': ['MSFT', 'AAPL', 'GOOGL']
        }
        
        key1 = CacheKeyGenerator.generate_key(components1)
        key2 = CacheKeyGenerator.generate_key(components2)
        
        self.assertEqual(key1, key2)


class TestCacheService(unittest.TestCase):
    """测试缓存服务"""
    
    def setUp(self):
        """测试前准备"""
        self.config = CacheConfig(
            l1_maxsize=100,
            l1_ttl_seconds=2
        )
        self.cache = CacheService(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.cache._l1_cache)
        self.assertEqual(self.cache.metrics.l1_hits, 0)
        self.assertEqual(self.cache.metrics.l1_misses, 0)
    
    def test_set_and_get(self):
        """测试设置和获取"""
        self.cache.set("test_key", {"data": [1, 2, 3]})
        value = self.cache.get("test_key")
        
        self.assertIsNotNone(value)
        self.assertEqual(value, {"data": [1, 2, 3]})
        self.assertEqual(self.cache.metrics.l1_hits, 1)
    
    def test_cache_miss(self):
        """测试缓存未命中"""
        value = self.cache.get("non_existent_key")
        self.assertIsNone(value)
        self.assertEqual(self.cache.metrics.l1_misses, 1)
    
    def test_ttl_expiration(self):
        """测试TTL过期"""
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
        self.cache.set("key_to_invalidate", "value")
        self.cache.invalidate("key_to_invalidate")
        
        value = self.cache.get("key_to_invalidate")
        self.assertIsNone(value)
    
    def test_invalidate_pattern(self):
        """测试按模式失效缓存"""
        self.cache.set("returns_US_AAPL", [1, 2, 3])
        self.cache.set("returns_US_GOOGL", [4, 5, 6])
        self.cache.set("returns_CN_000001", [7, 8, 9])
        
        count = self.cache.invalidate_pattern("returns_US")
        
        self.assertIsNone(self.cache.get("returns_US_AAPL"))
        self.assertIsNone(self.cache.get("returns_US_GOOGL"))
        self.assertIsNotNone(self.cache.get("returns_CN_000001"))
        self.assertEqual(count, 2)
    
    def test_clear(self):
        """测试清空缓存"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        
        self.cache.clear()
        
        self.assertIsNone(self.cache.get("key1"))
        self.assertIsNone(self.cache.get("key2"))
    
    def test_get_metrics(self):
        """测试获取指标"""
        self.cache.set("key1", "value1")
        self.cache.get("key1")  # 命中
        self.cache.get("key2")  # 未命中
        
        metrics = self.cache.get_metrics()
        
        self.assertIn('cache_size', metrics)
        self.assertIn('l1_hits', metrics)
        self.assertIn('l1_misses', metrics)
        self.assertIn('l1_hit_rate', metrics)
        
        self.assertEqual(metrics['l1_hits'], 1)
        self.assertEqual(metrics['l1_misses'], 1)
        self.assertAlmostEqual(metrics['l1_hit_rate'], 0.5)
    
    def test_legacy_process_interface(self):
        """测试兼容旧接口"""
        # 测试set操作
        result = self.cache._process_core(op='set', key='test', value='data')
        self.assertEqual(result['status'], 'success')
        
        # 测试get操作
        result = self.cache._process_core(op='get', key='test')
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['value'], 'data')
        
        # 测试invalidate操作
        result = self.cache._process_core(op='invalidate', key='test')
        self.assertEqual(result['status'], 'success')


class TestGlobalCacheService(unittest.TestCase):
    """测试全局缓存服务"""
    
    def test_get_cache_service_singleton(self):
        """测试单例模式"""
        service1 = get_cache_service()
        service2 = get_cache_service()
        
        self.assertIs(service1, service2)


if __name__ == '__main__':
    unittest.main()
