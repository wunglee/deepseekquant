"""
MemoryCache 单元测试

测试内存缓存层功能：
1. 基本读写操作
2. LRU 淘汰机制
3. TTL 过期机制
4. 边界场景（满缓存、空数据、过期数据等）
"""

import unittest
import time
import pandas as pd

from core_bak_refactored.infrastructure.cache.memory import MemoryCache


class MemoryCacheTest(unittest.TestCase):
    """MemoryCache 功能测试"""
    
    def setUp(self):
        """测试初始化"""
        # 使用较小的缓存和短TTL便于测试
        self.cache = MemoryCache(max_windows=3, ttl=2)
    
    def _create_test_df(self, rows=5):
        """创建测试数据"""
        return pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=rows, freq='D'),
            'close': [100 + i for i in range(rows)],
            'volume': [1000 + i*100 for i in range(rows)]
        })
    
    # ========== 基本读写测试 ==========
    
    def test_set_and_get_success(self):
        """测试基本写入和读取"""
        df = self._create_test_df()
        
        self.cache.set('399006.SZ', 'monthly', '2025-01', df)
        result = self.cache.get('399006.SZ', 'monthly', '2025-01')
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)
        pd.testing.assert_frame_equal(result, df)
    
    def test_get_non_existent(self):
        """测试读取不存在的缓存"""
        result = self.cache.get('399006.SZ', 'monthly', '2025-01')
        self.assertIsNone(result)
    
    def test_set_empty_dataframe(self):
        """测试写入空DataFrame"""
        empty_df = pd.DataFrame()
        self.cache.set('399006.SZ', 'monthly', '2025-01', empty_df)
        
        # 空DataFrame不应被缓存
        result = self.cache.get('399006.SZ', 'monthly', '2025-01')
        self.assertIsNone(result)
    
    def test_set_none_data(self):
        """测试写入None"""
        self.cache.set('399006.SZ', 'monthly', '2025-01', None)
        
        # None不应被缓存
        result = self.cache.get('399006.SZ', 'monthly', '2025-01')
        self.assertIsNone(result)
    
    def test_multiple_symbols(self):
        """测试多个不同symbol的缓存"""
        df1 = self._create_test_df(3)
        df2 = self._create_test_df(4)
        
        self.cache.set('399006.SZ', 'monthly', '2025-01', df1)
        self.cache.set('000001.SZ', 'monthly', '2025-01', df2)
        
        result1 = self.cache.get('399006.SZ', 'monthly', '2025-01')
        result2 = self.cache.get('000001.SZ', 'monthly', '2025-01')
        
        self.assertEqual(len(result1), 3)
        self.assertEqual(len(result2), 4)
    
    def test_multiple_periods(self):
        """测试多个不同period的缓存"""
        df_monthly = self._create_test_df(3)
        df_daily = self._create_test_df(4)
        
        self.cache.set('399006.SZ', 'monthly', '2025-01', df_monthly)
        self.cache.set('399006.SZ', 'daily', '2025-01-01', df_daily)
        
        result_monthly = self.cache.get('399006.SZ', 'monthly', '2025-01')
        result_daily = self.cache.get('399006.SZ', 'daily', '2025-01-01')
        
        self.assertEqual(len(result_monthly), 3)
        self.assertEqual(len(result_daily), 4)
    
    # ========== LRU 淘汰机制测试 ==========
    
    def test_lru_eviction_when_full(self):
        """测试缓存满时的LRU淘汰"""
        df1 = self._create_test_df(1)
        df2 = self._create_test_df(2)
        df3 = self._create_test_df(3)
        df4 = self._create_test_df(4)
        
        # 填满缓存（max_windows=3）
        self.cache.set('399006.SZ', 'monthly', '2025-01', df1)
        self.cache.set('399006.SZ', 'monthly', '2025-02', df2)
        self.cache.set('399006.SZ', 'monthly', '2025-03', df3)
        
        # 再写入一个，应淘汰最老的（2025-01）
        self.cache.set('399006.SZ', 'monthly', '2025-04', df4)
        
        # 2025-01应被淘汰
        self.assertIsNone(self.cache.get('399006.SZ', 'monthly', '2025-01'))
        
        # 其他应存在
        self.assertIsNotNone(self.cache.get('399006.SZ', 'monthly', '2025-02'))
        self.assertIsNotNone(self.cache.get('399006.SZ', 'monthly', '2025-03'))
        self.assertIsNotNone(self.cache.get('399006.SZ', 'monthly', '2025-04'))
    
    def test_lru_update_on_access(self):
        """测试访问后LRU更新"""
        df1 = self._create_test_df(1)
        df2 = self._create_test_df(2)
        df3 = self._create_test_df(3)
        df4 = self._create_test_df(4)
        
        # 填满缓存
        self.cache.set('399006.SZ', 'monthly', '2025-01', df1)
        self.cache.set('399006.SZ', 'monthly', '2025-02', df2)
        self.cache.set('399006.SZ', 'monthly', '2025-03', df3)
        
        # 访问2025-01，使其成为最近使用
        self.cache.get('399006.SZ', 'monthly', '2025-01')
        
        # 再写入一个，应淘汰2025-02（现在是最老的）
        self.cache.set('399006.SZ', 'monthly', '2025-04', df4)
        
        # 2025-02应被淘汰
        self.assertIsNone(self.cache.get('399006.SZ', 'monthly', '2025-02'))
        
        # 2025-01因为被访问过，应该还在
        self.assertIsNotNone(self.cache.get('399006.SZ', 'monthly', '2025-01'))
    
    # ========== TTL 过期机制测试 ==========
    
    def test_ttl_expiration(self):
        """测试TTL过期"""
        df = self._create_test_df()
        
        self.cache.set('399006.SZ', 'monthly', '2025-01', df)
        
        # 立即读取应成功
        result = self.cache.get('399006.SZ', 'monthly', '2025-01')
        self.assertIsNotNone(result)
        
        # 等待超过TTL（2秒）
        time.sleep(2.5)
        
        # 再次读取应返回None（已过期）
        result = self.cache.get('399006.SZ', 'monthly', '2025-01')
        self.assertIsNone(result)
    
    def test_ttl_not_expired(self):
        """测试TTL未过期时仍可读取"""
        df = self._create_test_df()
        
        self.cache.set('399006.SZ', 'monthly', '2025-01', df)
        
        # 等待一半TTL时间
        time.sleep(1)
        
        # 应仍可读取
        result = self.cache.get('399006.SZ', 'monthly', '2025-01')
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)
    
    def test_expired_entry_removed_on_access(self):
        """测试过期条目在访问时被移除"""
        df = self._create_test_df()
        
        self.cache.set('399006.SZ', 'monthly', '2025-01', df)
        
        # 等待过期
        time.sleep(2.5)
        
        # 访问过期条目
        result = self.cache.get('399006.SZ', 'monthly', '2025-01')
        self.assertIsNone(result)
        
        # 缓存统计应显示0个窗口（过期条目已被移除）
        stats = self.cache.get_stats()
        self.assertEqual(stats['total_windows'], 0)
    
    # ========== 缓存统计测试 ==========
    
    def test_get_stats_empty(self):
        """测试空缓存统计"""
        stats = self.cache.get_stats()
        
        self.assertEqual(stats['total_windows'], 0)
        self.assertEqual(stats['max_windows'], 3)
        self.assertEqual(stats['usage_percent'], 0.0)
    
    def test_get_stats_partial(self):
        """测试部分填充缓存统计"""
        df = self._create_test_df()
        
        self.cache.set('399006.SZ', 'monthly', '2025-01', df)
        self.cache.set('399006.SZ', 'monthly', '2025-02', df)
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats['total_windows'], 2)
        self.assertEqual(stats['max_windows'], 3)
        self.assertAlmostEqual(stats['usage_percent'], 66.67, places=1)
    
    def test_get_stats_full(self):
        """测试满缓存统计"""
        df = self._create_test_df()
        
        self.cache.set('399006.SZ', 'monthly', '2025-01', df)
        self.cache.set('399006.SZ', 'monthly', '2025-02', df)
        self.cache.set('399006.SZ', 'monthly', '2025-03', df)
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats['total_windows'], 3)
        self.assertEqual(stats['max_windows'], 3)
        self.assertEqual(stats['usage_percent'], 100.0)
    
    # ========== 清空缓存测试 ==========
    
    def test_clear_cache(self):
        """测试清空缓存"""
        df = self._create_test_df()
        
        self.cache.set('399006.SZ', 'monthly', '2025-01', df)
        self.cache.set('399006.SZ', 'monthly', '2025-02', df)
        
        # 清空前应有数据
        self.assertIsNotNone(self.cache.get('399006.SZ', 'monthly', '2025-01'))
        
        # 清空
        self.cache.clear()
        
        # 清空后应无数据
        self.assertIsNone(self.cache.get('399006.SZ', 'monthly', '2025-01'))
        self.assertIsNone(self.cache.get('399006.SZ', 'monthly', '2025-02'))
        
        # 统计应为0
        stats = self.cache.get_stats()
        self.assertEqual(stats['total_windows'], 0)
    
    # ========== 数据隔离测试 ==========
    
    def test_data_independence(self):
        """测试缓存数据独立性（修改原数据不影响缓存）"""
        df = self._create_test_df()
        original_value = df.iloc[0]['close']
        
        self.cache.set('399006.SZ', 'monthly', '2025-01', df)
        
        # 修改原DataFrame
        df.iloc[0, df.columns.get_loc('close')] = 999
        
        # 从缓存读取
        cached_df = self.cache.get('399006.SZ', 'monthly', '2025-01')
        
        # 缓存中的数据应未被修改
        self.assertEqual(cached_df.iloc[0]['close'], original_value)
        self.assertNotEqual(cached_df.iloc[0]['close'], 999)
    
    # ========== 边界场景测试 ==========
    
    def test_large_dataframe(self):
        """测试大DataFrame缓存"""
        large_df = self._create_test_df(rows=10000)
        
        self.cache.set('399006.SZ', 'daily', '2025-01-01', large_df)
        result = self.cache.get('399006.SZ', 'daily', '2025-01-01')
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 10000)
    
    def test_special_characters_in_symbol(self):
        """测试symbol中的特殊字符"""
        df = self._create_test_df()
        
        symbols = ['399006.SZ', '000001.SH', 'AAPL', 'BRK.B', 'SPX:INDEX']
        for symbol in symbols:
            self.cache.set(symbol, 'monthly', '2025-01', df)
            result = self.cache.get(symbol, 'monthly', '2025-01')
            self.assertIsNotNone(result, f"Failed for symbol: {symbol}")
    
    def test_window_key_edge_cases(self):
        """测试窗口键边界场景"""
        df = self._create_test_df()
        
        # 测试各种窗口键格式
        window_keys = [
            ('monthly', '2025-01'),
            ('monthly', '2024-12'),
            ('weekly', '2025-W01'),
            ('daily', '2025-01-01'),
            ('daily', '2025-12-31'),
            ('daily', '2024-02-29'),  # 闰年
        ]
        
        for period, window_key in window_keys:
            self.cache.set('399006.SZ', period, window_key, df)
            result = self.cache.get('399006.SZ', period, window_key)
            self.assertIsNotNone(result, 
                f"Failed for period={period}, window_key={window_key}")


if __name__ == '__main__':
    unittest.main()
