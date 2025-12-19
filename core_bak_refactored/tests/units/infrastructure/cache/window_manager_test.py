"""
WindowManager 单元测试

测试窗口键管理工具类的功能：
1. 窗口键生成（月/周/日）
2. 窗口键与日期的转换
3. 窗口连续性判断
4. 边界场景测试（跨年、月末、闰年等）
"""

import unittest
from datetime import datetime
import pandas as pd

from core_bak_refactored.infrastructure.cache.window_manager import WindowManager


class WindowManagerTest(unittest.TestCase):
    """WindowManager 功能测试"""
    
    def setUp(self):
        """测试初始化"""
        self.mgr = WindowManager()
    
    # ========== 窗口键生成测试 ==========
    
    def test_make_window_key_monthly(self):
        """测试月度窗口键生成"""
        date = pd.Timestamp('2025-03-15')
        key = self.mgr.make_window_key(date, 'monthly')
        self.assertEqual(key, '2025-03')
    
    def test_make_window_key_weekly(self):
        """测试周度窗口键生成"""
        date = pd.Timestamp('2025-03-15')  # 2025年第10周
        key = self.mgr.make_window_key(date, 'weekly')
        self.assertTrue(key.startswith('2025-W'))
    
    def test_make_window_key_daily(self):
        """测试日度窗口键生成"""
        date = pd.Timestamp('2025-03-15')
        key = self.mgr.make_window_key(date, 'daily')
        self.assertEqual(key, '2025-03-15')
    
    def test_generate_window_keys_single_month(self):
        """测试生成单月窗口键"""
        keys = self.mgr.generate_window_keys('2025-01-01', '2025-01-31', 'monthly')
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0], '2025-01')
    
    def test_generate_window_keys_multiple_months(self):
        """测试生成多月窗口键"""
        keys = self.mgr.generate_window_keys('2025-01-01', '2025-03-31', 'monthly')
        self.assertEqual(len(keys), 3)
        self.assertEqual(keys, ['2025-01', '2025-02', '2025-03'])
    
    def test_generate_window_keys_cross_year(self):
        """测试生成跨年窗口键"""
        keys = self.mgr.generate_window_keys('2024-11-01', '2025-02-28', 'monthly')
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys, ['2024-11', '2024-12', '2025-01', '2025-02'])
    
    def test_generate_window_keys_weekly(self):
        """测试生成周度窗口键"""
        keys = self.mgr.generate_window_keys('2025-01-01', '2025-01-31', 'weekly')
        # 1月通常包含4-5周
        self.assertGreater(len(keys), 0)
        self.assertTrue(all(k.startswith('2025-W') for k in keys))
    
    def test_generate_window_keys_daily(self):
        """测试生成日度窗口键"""
        keys = self.mgr.generate_window_keys('2025-01-01', '2025-01-05', 'daily')
        self.assertEqual(len(keys), 5)
        self.assertEqual(keys, [
            '2025-01-01', '2025-01-02', '2025-01-03', 
            '2025-01-04', '2025-01-05'
        ])
    
    # ========== 边界场景测试 ==========
    
    def test_generate_window_keys_leap_year_february(self):
        """测试闰年2月窗口键"""
        # 2024是闰年
        keys = self.mgr.generate_window_keys('2024-02-01', '2024-02-29', 'monthly')
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0], '2024-02')
        
        # 验证2月29日包含在内
        keys_daily = self.mgr.generate_window_keys('2024-02-28', '2024-02-29', 'daily')
        self.assertEqual(len(keys_daily), 2)
        self.assertIn('2024-02-29', keys_daily)
    
    def test_generate_window_keys_non_leap_year_february(self):
        """测试非闰年2月窗口键"""
        # 2025不是闰年
        keys = self.mgr.generate_window_keys('2025-02-01', '2025-02-28', 'monthly')
        self.assertEqual(len(keys), 1)
        
        # 2月29日不应存在
        keys_daily = self.mgr.generate_window_keys('2025-02-28', '2025-03-01', 'daily')
        self.assertEqual(len(keys_daily), 2)
        self.assertNotIn('2025-02-29', keys_daily)
    
    def test_generate_window_keys_month_end_boundaries(self):
        """测试月末边界场景"""
        # 31天的月
        keys = self.mgr.generate_window_keys('2025-01-31', '2025-02-01', 'monthly')
        self.assertEqual(len(keys), 2)
        
        # 30天的月
        keys = self.mgr.generate_window_keys('2025-04-30', '2025-05-01', 'monthly')
        self.assertEqual(len(keys), 2)
    
    def test_generate_window_keys_year_boundary(self):
        """测试年末边界场景"""
        keys = self.mgr.generate_window_keys('2024-12-31', '2025-01-01', 'monthly')
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys, ['2024-12', '2025-01'])
        
        keys_daily = self.mgr.generate_window_keys('2024-12-31', '2025-01-01', 'daily')
        self.assertEqual(keys_daily, ['2024-12-31', '2025-01-01'])
    
    # ========== 窗口键转换测试 ==========
    
    def test_window_key_to_date_monthly_start(self):
        """测试月度窗口键转为开始日期"""
        date = self.mgr.window_key_to_date('2025-03', 'monthly', is_start=True)
        self.assertEqual(date, '2025-03-01')
    
    def test_window_key_to_date_monthly_end(self):
        """测试月度窗口键转为结束日期"""
        date = self.mgr.window_key_to_date('2025-03', 'monthly', is_start=False)
        self.assertEqual(date, '2025-03-31')
    
    def test_window_key_to_date_february_end(self):
        """测试2月月末转换"""
        # 闰年2月
        date = self.mgr.window_key_to_date('2024-02', 'monthly', is_start=False)
        self.assertEqual(date, '2024-02-29')
        
        # 非闰年2月
        date = self.mgr.window_key_to_date('2025-02', 'monthly', is_start=False)
        self.assertEqual(date, '2025-02-28')
    
    def test_window_key_to_date_december_end(self):
        """测试12月月末转换"""
        date = self.mgr.window_key_to_date('2025-12', 'monthly', is_start=False)
        self.assertEqual(date, '2025-12-31')
    
    def test_window_key_to_date_daily(self):
        """测试日度窗口键转换"""
        date = self.mgr.window_key_to_date('2025-03-15', 'daily', is_start=True)
        self.assertEqual(date, '2025-03-15')
        
        date = self.mgr.window_key_to_date('2025-03-15', 'daily', is_start=False)
        self.assertEqual(date, '2025-03-15')
    
    # ========== 窗口连续性测试 ==========
    
    def test_is_continuous_monthly_consecutive(self):
        """测试连续月份判断"""
        self.assertTrue(self.mgr.is_continuous('2025-01', '2025-02', 'monthly'))
        self.assertTrue(self.mgr.is_continuous('2024-12', '2025-01', 'monthly'))
    
    def test_is_continuous_monthly_non_consecutive(self):
        """测试非连续月份判断"""
        self.assertFalse(self.mgr.is_continuous('2025-01', '2025-03', 'monthly'))
        self.assertFalse(self.mgr.is_continuous('2024-11', '2025-01', 'monthly'))
    
    def test_is_continuous_daily_consecutive(self):
        """测试连续日期判断"""
        self.assertTrue(self.mgr.is_continuous('2025-01-15', '2025-01-16', 'daily'))
        self.assertTrue(self.mgr.is_continuous('2025-01-31', '2025-02-01', 'daily'))
        self.assertTrue(self.mgr.is_continuous('2024-12-31', '2025-01-01', 'daily'))
    
    def test_is_continuous_daily_non_consecutive(self):
        """测试非连续日期判断"""
        self.assertFalse(self.mgr.is_continuous('2025-01-15', '2025-01-17', 'daily'))
        self.assertFalse(self.mgr.is_continuous('2025-01-01', '2025-02-01', 'daily'))
    
    # ========== 异常输入测试 ==========
    
    def test_generate_window_keys_invalid_date_range(self):
        """测试无效日期范围（开始日期晚于结束日期）"""
        keys = self.mgr.generate_window_keys('2025-03-01', '2025-01-01', 'monthly')
        # 应返回空列表或抛出异常（取决于实现）
        self.assertEqual(len(keys), 0)
    
    def test_generate_window_keys_same_date(self):
        """测试相同日期"""
        keys = self.mgr.generate_window_keys('2025-01-15', '2025-01-15', 'daily')
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0], '2025-01-15')
    
    def test_window_key_to_date_invalid_format(self):
        """测试无效窗口键格式"""
        # 这应该抛出异常或返回None（取决于实现）
        try:
            date = self.mgr.window_key_to_date('invalid-key', 'monthly', is_start=True)
            # 如果没有抛出异常，至少应返回None或空字符串
            self.assertIn(date, [None, ''])
        except (ValueError, AttributeError):
            # 抛出异常是可接受的
            pass


if __name__ == '__main__':
    unittest.main()
