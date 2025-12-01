"""
缓存键生成器单元测试

测试路径同构：
- 被测试代码：core_bak_refactored/infrastructure/cache/key_generator.py
- 测试代码：core_bak_refactored/tests/units/infrastructure/cache/key_generator_test.py

测试范围：
1. generate_key() - 智能缓存键生成
2. generate_simple_key() - 简单缓存键生成
"""
import unittest
from datetime import datetime
from core_bak_refactored.infrastructure.cache.key_generator import CacheKeyGenerator


class TestGenerateKey(unittest.TestCase):
    """测试智能缓存键生成"""

    def test_generate_key_basic(self):
        """测试基本键生成"""
        components = {
            'market': 'US',
            'symbols': ['AAPL', 'GOOGL'],
            'model_type': 'risk',
            'params': {'alpha': 0.05}
        }
        key = CacheKeyGenerator.generate_key(components)
        
        # 验证格式
        self.assertTrue(key.startswith('v1.0:'))
        parts = key.split(':')
        self.assertEqual(len(parts), 4)  # version:stable_hash:time:param_hash
    
    def test_generate_key_with_custom_version(self):
        """测试自定义版本号"""
        components = {
            'market': 'CN',
            'symbols': ['000001.SZ'],
            'model_type': 'factor'
        }
        key = CacheKeyGenerator.generate_key(components, data_version='v2.5')
        
        self.assertTrue(key.startswith('v2.5:'))
    
    def test_generate_key_deterministic(self):
        """测试键生成的确定性（相同输入生成相同键）"""
        components = {
            'market': 'US',
            'symbols': ['AAPL', 'MSFT'],
            'model_type': 'covariance',
            'params': {'method': 'shrinkage'}
        }
        
        key1 = CacheKeyGenerator.generate_key(components)
        key2 = CacheKeyGenerator.generate_key(components)
        
        self.assertEqual(key1, key2)
    
    def test_generate_key_symbols_order_agnostic(self):
        """测试符号顺序无关性（不同顺序应生成相同稳定部分）"""
        components1 = {
            'market': 'US',
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'model_type': 'risk'
        }
        components2 = {
            'market': 'US',
            'symbols': ['MSFT', 'AAPL', 'GOOGL'],  # 不同顺序
            'model_type': 'risk'
        }
        
        key1 = CacheKeyGenerator.generate_key(components1)
        key2 = CacheKeyGenerator.generate_key(components2)
        
        # 稳定哈希部分（第2段）应该相同
        stable_hash1 = key1.split(':')[1]
        stable_hash2 = key2.split(':')[1]
        self.assertEqual(stable_hash1, stable_hash2)
    
    def test_generate_key_with_time_window_datetime(self):
        """测试带时间窗口（datetime对象）"""
        test_time = datetime(2025, 1, 15, 10, 30, 45)
        components = {
            'market': 'US',
            'symbols': ['AAPL'],
            'time_window': test_time
        }
        
        key = CacheKeyGenerator.generate_key(components)
        
        # 时间应该被对齐到整点（10:00:00）
        parts = key.split(':')
        aligned_time = datetime(2025, 1, 15, 10, 0, 0)
        expected_timestamp = str(int(aligned_time.timestamp()))
        self.assertEqual(parts[2], expected_timestamp)
    
    def test_generate_key_with_time_window_string(self):
        """测试带时间窗口（字符串）"""
        components = {
            'market': 'CN',
            'symbols': ['000001.SZ'],
            'time_window': '2025-01-15'
        }
        
        key = CacheKeyGenerator.generate_key(components)
        
        # 字符串时间窗口应直接使用
        parts = key.split(':')
        self.assertEqual(parts[2], '2025-01-15')
    
    def test_generate_key_without_time_window(self):
        """测试无时间窗口（应使用static）"""
        components = {
            'market': 'US',
            'symbols': ['AAPL']
        }
        
        key = CacheKeyGenerator.generate_key(components)
        
        parts = key.split(':')
        self.assertEqual(parts[2], 'static')
    
    def test_generate_key_empty_components(self):
        """测试空组件（使用默认值）"""
        components = {}
        
        key = CacheKeyGenerator.generate_key(components)
        
        # 应该能成功生成（使用默认值）
        self.assertTrue(key.startswith('v1.0:'))
        parts = key.split(':')
        self.assertEqual(len(parts), 4)
    
    def test_generate_key_empty_symbols(self):
        """测试空符号列表"""
        components = {
            'market': 'US',
            'symbols': [],
            'model_type': 'risk'
        }
        
        key = CacheKeyGenerator.generate_key(components)
        
        # 应该能成功生成
        self.assertTrue(key.startswith('v1.0:'))
    
    def test_generate_key_params_affect_result(self):
        """测试参数变化影响键"""
        components1 = {
            'market': 'US',
            'symbols': ['AAPL'],
            'params': {'alpha': 0.05}
        }
        components2 = {
            'market': 'US',
            'symbols': ['AAPL'],
            'params': {'alpha': 0.10}  # 不同参数
        }
        
        key1 = CacheKeyGenerator.generate_key(components1)
        key2 = CacheKeyGenerator.generate_key(components2)
        
        # 参数哈希（第4段）应该不同
        param_hash1 = key1.split(':')[3]
        param_hash2 = key2.split(':')[3]
        self.assertNotEqual(param_hash1, param_hash2)
    
    def test_generate_key_params_order_invariant(self):
        """测试参数顺序无关性"""
        components1 = {
            'market': 'US',
            'symbols': ['AAPL'],
            'params': {'alpha': 0.05, 'beta': 0.10}
        }
        components2 = {
            'market': 'US',
            'symbols': ['AAPL'],
            'params': {'beta': 0.10, 'alpha': 0.05}  # 不同顺序
        }
        
        key1 = CacheKeyGenerator.generate_key(components1)
        key2 = CacheKeyGenerator.generate_key(components2)
        
        # 参数哈希应该相同（因为使用 sort_keys=True）
        param_hash1 = key1.split(':')[3]
        param_hash2 = key2.split(':')[3]
        self.assertEqual(param_hash1, param_hash2)
    
    def test_generate_key_complex_params(self):
        """测试复杂参数"""
        components = {
            'market': 'US',
            'symbols': ['AAPL'],
            'params': {
                'window': 252,
                'method': 'exponential',
                'weights': [0.1, 0.2, 0.7],
                'config': {'nested': True}
            }
        }
        
        key = CacheKeyGenerator.generate_key(components)
        
        # 应该能处理复杂参数
        self.assertTrue(key.startswith('v1.0:'))
        parts = key.split(':')
        self.assertEqual(len(parts), 4)
    
    def test_generate_key_special_characters_in_symbols(self):
        """测试符号中的特殊字符"""
        components = {
            'market': 'CN',
            'symbols': ['000001.SZ', '600036.SS', '000300.SH'],
            'model_type': 'risk'
        }
        
        key = CacheKeyGenerator.generate_key(components)
        
        # 应该能处理特殊字符
        self.assertTrue(key.startswith('v1.0:'))
    
    def test_generate_key_reproducibility(self):
        """测试可重现性（多次调用应生成相同键）"""
        components = {
            'market': 'US',
            'symbols': ['AAPL', 'GOOGL'],
            'model_type': 'factor',
            'params': {'n_factors': 5}
        }
        
        keys = [CacheKeyGenerator.generate_key(components) for _ in range(5)]
        
        # 所有键应该相同
        self.assertEqual(len(set(keys)), 1)


class TestGenerateSimpleKey(unittest.TestCase):
    """测试简单缓存键生成"""

    def test_generate_simple_key_prefix_only(self):
        """测试仅前缀"""
        key = CacheKeyGenerator.generate_simple_key('test')
        
        self.assertEqual(key, 'test')
    
    def test_generate_simple_key_with_args(self):
        """测试带位置参数"""
        key = CacheKeyGenerator.generate_simple_key('cache', 'user', 123, 'profile')
        
        self.assertEqual(key, 'cache:user:123:profile')
    
    def test_generate_simple_key_with_kwargs(self):
        """测试带关键字参数"""
        key = CacheKeyGenerator.generate_simple_key('cache', region='US', type='risk')
        
        # 关键字参数应该被排序
        self.assertIn('region=US', key)
        self.assertIn('type=risk', key)
        self.assertTrue(key.startswith('cache:'))
    
    def test_generate_simple_key_with_args_and_kwargs(self):
        """测试混合参数"""
        key = CacheKeyGenerator.generate_simple_key(
            'cache', 'user', 123,
            action='fetch', limit=100
        )
        
        self.assertTrue(key.startswith('cache:user:123:'))
        self.assertIn('action=fetch', key)
        self.assertIn('limit=100', key)
    
    def test_generate_simple_key_kwargs_ordering(self):
        """测试关键字参数排序一致性"""
        key1 = CacheKeyGenerator.generate_simple_key('cache', z=3, a=1, m=2)
        key2 = CacheKeyGenerator.generate_simple_key('cache', a=1, m=2, z=3)
        
        # 不同顺序输入应生成相同键
        self.assertEqual(key1, key2)
    
    def test_generate_simple_key_empty_args(self):
        """测试空参数"""
        key = CacheKeyGenerator.generate_simple_key('prefix')
        
        self.assertEqual(key, 'prefix')
    
    def test_generate_simple_key_numeric_types(self):
        """测试数值类型参数"""
        key = CacheKeyGenerator.generate_simple_key('data', 123, 45.67, True)
        
        self.assertEqual(key, 'data:123:45.67:True')
    
    def test_generate_simple_key_special_characters(self):
        """测试特殊字符"""
        key = CacheKeyGenerator.generate_simple_key('cache', 'user@test.com', 'role:admin')
        
        self.assertIn('user@test.com', key)
        # 注意：冒号会影响解析，但生成器不做转义
    
    def test_generate_simple_key_none_values(self):
        """测试None值"""
        key = CacheKeyGenerator.generate_simple_key('cache', None, value=None)
        
        self.assertIn('None', key)
    
    def test_generate_simple_key_reproducibility(self):
        """测试可重现性"""
        keys = [
            CacheKeyGenerator.generate_simple_key('test', 'a', 'b', x=1, y=2)
            for _ in range(5)
        ]
        
        # 所有键应该相同
        self.assertEqual(len(set(keys)), 1)


class TestKeyGeneratorEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_very_long_symbol_list(self):
        """测试超长符号列表"""
        components = {
            'market': 'US',
            'symbols': [f'SYM{i:04d}' for i in range(1000)],
            'model_type': 'risk'
        }
        
        key = CacheKeyGenerator.generate_key(components)
        
        # 应该能处理（哈希会截断）
        self.assertTrue(key.startswith('v1.0:'))
        stable_hash = key.split(':')[1]
        self.assertEqual(len(stable_hash), 12)  # MD5截断到12位
    
    def test_very_long_params(self):
        """测试超长参数"""
        components = {
            'market': 'US',
            'symbols': ['AAPL'],
            'params': {f'param_{i}': i for i in range(100)}
        }
        
        key = CacheKeyGenerator.generate_key(components)
        
        # 参数哈希应该固定长度
        param_hash = key.split(':')[3]
        self.assertEqual(len(param_hash), 8)  # MD5截断到8位
    
    def test_unicode_characters(self):
        """测试Unicode字符"""
        components = {
            'market': '中国',
            'symbols': ['贵州茅台', '平安银行'],
            'model_type': '风险模型'
        }
        
        key = CacheKeyGenerator.generate_key(components)
        
        # 应该能处理Unicode
        self.assertTrue(key.startswith('v1.0:'))
    
    def test_time_window_at_midnight(self):
        """测试午夜时间"""
        midnight = datetime(2025, 1, 15, 0, 0, 0)
        components = {
            'market': 'US',
            'symbols': ['AAPL'],
            'time_window': midnight
        }
        
        key = CacheKeyGenerator.generate_key(components)
        
        # 午夜应该对齐到自己
        parts = key.split(':')
        expected_timestamp = str(int(midnight.timestamp()))
        self.assertEqual(parts[2], expected_timestamp)
    
    def test_time_window_one_second_before_midnight(self):
        """测试午夜前1秒"""
        before_midnight = datetime(2025, 1, 15, 23, 59, 59)
        components = {
            'market': 'US',
            'symbols': ['AAPL'],
            'time_window': before_midnight
        }
        
        key = CacheKeyGenerator.generate_key(components)
        
        # 应该对齐到23:00:00
        parts = key.split(':')
        aligned = datetime(2025, 1, 15, 23, 0, 0)
        expected_timestamp = str(int(aligned.timestamp()))
        self.assertEqual(parts[2], expected_timestamp)


if __name__ == '__main__':
    unittest.main()
