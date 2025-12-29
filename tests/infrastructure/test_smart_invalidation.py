"""
智能失效策略测试（旧版，已由 smart_invalidation_test.py 替代）
"""

# Deleted: legacy duplicate tests, see smart_invalidation_test.py for maintained suite.

import unittest
import sys
import os


# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from infrastructure.cache_service import (
    CacheService,
    CacheConfig,
    SmartInvalidationManager,
    InvalidationRule,
    get_smart_invalidation_manager
)


class TestSmartInvalidation(unittest.TestCase):
    """测试智能失效策略"""
    
    def setUp(self):
        """测试前准备"""
        self.config = CacheConfig(l1_maxsize=100, l1_ttl_seconds=300)
        self.cache_service = CacheService(self.config)
        self.invalidation_manager = SmartInvalidationManager(self.cache_service)
    
    def tearDown(self):
        """测试后清理"""
        self.cache_service.clear()
    
    def test_default_rules_initialization(self):
        """测试默认规则初始化"""
        self.assertGreaterEqual(len(self.invalidation_manager.rules), 3)
    
    def test_param_version_change_invalidation(self):
        """测试参数版本变化失效"""
        key1 = "v1.0:hash123:time1:oldparam"
        key2 = "v1.0:hash456:time1:oldparam"
        self.cache_service.set(key1, "value1")
        self.cache_service.set(key2, "value2")
        
        context = {'param_version': 'newparam'}
        invalidated = self.invalidation_manager.check_and_invalidate(context)
        
        self.assertEqual(invalidated, 2)
    
    def test_market_data_update_invalidation(self):
        """测试市场数据更新失效"""
        self.cache_service.set("market:US:cov", "cov_matrix")
        self.cache_service.set("market:CN:cov", "cov_matrix")
        self.cache_service.set("other:data", "other_value")
        
        context = {'market_data_updated': True}
        invalidated = self.invalidation_manager.check_and_invalidate(context)
        
        self.assertEqual(invalidated, 2)
        self.assertIsNotNone(self.cache_service.get("other:data"))
    
    def test_invalidate_by_condition(self):
        """测试条件失效"""
        self.cache_service.set("user:1:profile", "profile1")
        self.cache_service.set("user:2:profile", "profile2")
        self.cache_service.set("system:config", "config")
        
        invalidated = self.invalidation_manager.invalidate_by_condition(
            lambda k: k.startswith("user:")
        )
        
        self.assertEqual(invalidated, 2)
        self.assertIsNone(self.cache_service.get("user:1:profile"))
        self.assertIsNotNone(self.cache_service.get("system:config"))
    
    def test_schedule_preload(self):
        """测试预加载"""
        def loader(key):
            return f"loaded_value_for_{key}"
        
        keys = ["key1", "key2", "key3"]
        success_count = self.invalidation_manager.schedule_preload(keys, loader)
        
        self.assertEqual(success_count, 3)
        self.assertEqual(self.cache_service.get("key1"), "loaded_value_for_key1")


if __name__ == '__main__':
    unittest.main()
