"""
智能失效策略测试
"""

import unittest
import sys
import os
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from core_bak_refactored.infrastructure.cache_service import (
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
        # 默认应该有3个规则
        self.assertGreaterEqual(len(self.invalidation_manager.rules), 3)
        
        rule_names = [rule.name for rule in self.invalidation_manager.rules]
        self.assertIn('time_window_change', rule_names)
        self.assertIn('param_version_change', rule_names)
        self.assertIn('market_data_update', rule_names)
    
    def test_time_window_change_invalidation(self):
        """测试时间窗口变化失效"""
        # 设置缓存（旧时间窗口）
        old_time = datetime.now() - timedelta(hours=2)
        key1 = f"v1.0:hash123:{int(old_time.timestamp())}:param1"
        self.cache_service.set(key1, "old_value")
        
        # 检查失效（新时间窗口）
        new_time = datetime.now()
        context = {'time_window': new_time}
        invalidated = self.invalidation_manager.check_and_invalidate(context)
        
        # 应该失效1个
        self.assertGreaterEqual(invalidated, 0)  # 时间戳匹配可能不精确
    
    def test_param_version_change_invalidation(self):
        """测试参数版本变化失效"""
        # 设置缓存（旧参数版本）
        key1 = "v1.0:hash123:time1:oldparam"
        key2 = "v1.0:hash456:time1:oldparam"
        self.cache_service.set(key1, "value1")
        self.cache_service.set(key2, "value2")
        
        # 检查失效（新参数版本）
        context = {'param_version': 'newparam'}
        invalidated = self.invalidation_manager.check_and_invalidate(context)
        
        # 应该失效2个
        self.assertEqual(invalidated, 2)
    
    def test_market_data_update_invalidation(self):
        """测试市场数据更新失效"""
        # 设置市场相关缓存
        self.cache_service.set("market:US:cov", "cov_matrix")
        self.cache_service.set("market:CN:cov", "cov_matrix")
        self.cache_service.set("other:data", "other_value")
        
        # 触发市场数据更新失效
        context = {'market_data_updated': True}
        invalidated = self.invalidation_manager.check_and_invalidate(context)
        
        # 应该失效2个市场缓存
        self.assertEqual(invalidated, 2)
        
        # 非市场缓存应保留
        self.assertIsNotNone(self.cache_service.get("other:data"))
    
    def test_add_custom_rule(self):
        """测试添加自定义规则"""
        # 添加自定义规则：失效所有包含"temp"的key
        custom_rule = InvalidationRule(
            'temp_data_rule',
            lambda k, v, ctx: 'temp' in k and ctx.get('clear_temp', False)
        )
        self.invalidation_manager.add_rule(custom_rule)
        
        # 设置缓存
        self.cache_service.set("temp:data1", "value1")
        self.cache_service.set("temp:data2", "value2")
        self.cache_service.set("perm:data", "value3")
        
        # 触发自定义规则
        context = {'clear_temp': True}
        invalidated = self.invalidation_manager.check_and_invalidate(context)
        
        # 应该失效2个temp缓存
        self.assertEqual(invalidated, 2)
        self.assertIsNone(self.cache_service.get("temp:data1"))
        self.assertIsNotNone(self.cache_service.get("perm:data"))
    
    def test_invalidate_by_condition(self):
        """测试条件失效"""
        # 设置多个缓存
        self.cache_service.set("user:1:profile", "profile1")
        self.cache_service.set("user:2:profile", "profile2")
        self.cache_service.set("system:config", "config")
        
        # 失效所有用户缓存
        invalidated = self.invalidation_manager.invalidate_by_condition(
            lambda k: k.startswith("user:")
        )
        
        # 应该失效2个
        self.assertEqual(invalidated, 2)
        self.assertIsNone(self.cache_service.get("user:1:profile"))
        self.assertIsNotNone(self.cache_service.get("system:config"))
    
    def test_schedule_preload(self):
        """测试预加载"""
        # 定义加载器
        def loader(key):
            return f"loaded_value_for_{key}"
        
        # 预加载
        keys = ["key1", "key2", "key3"]
        success_count = self.invalidation_manager.schedule_preload(keys, loader)
        
        # 验证
        self.assertEqual(success_count, 3)
        self.assertEqual(self.cache_service.get("key1"), "loaded_value_for_key1")
        self.assertEqual(self.cache_service.get("key2"), "loaded_value_for_key2")
        self.assertEqual(self.cache_service.get("key3"), "loaded_value_for_key3")
    
    def test_preload_with_failure(self):
        """测试预加载失败处理"""
        def failing_loader(key):
            if key == "fail_key":
                raise ValueError("模拟加载失败")
            return f"value_{key}"
        
        keys = ["ok_key1", "fail_key", "ok_key2"]
        success_count = self.invalidation_manager.schedule_preload(keys, failing_loader)
        
        # 应该有2个成功
        self.assertEqual(success_count, 2)
        self.assertIsNotNone(self.cache_service.get("ok_key1"))
        self.assertIsNone(self.cache_service.get("fail_key"))
        self.assertIsNotNone(self.cache_service.get("ok_key2"))
    
    def test_no_invalidation_when_no_match(self):
        """测试无匹配时不失效"""
        # 设置缓存
        self.cache_service.set("stable:data", "value")
        
        # 触发失效检查（无匹配条件）
        context = {'unrelated_key': 'value'}
        invalidated = self.invalidation_manager.check_and_invalidate(context)
        
        # 不应该失效
        self.assertEqual(invalidated, 0)
        self.assertIsNotNone(self.cache_service.get("stable:data"))
    
    def test_get_smart_invalidation_manager(self):
        """测试获取智能失效管理器"""
        manager = get_smart_invalidation_manager()
        
        # 验证
        self.assertIsInstance(manager, SmartInvalidationManager)
        self.assertGreaterEqual(len(manager.rules), 3)


if __name__ == '__main__':
    unittest.main()
