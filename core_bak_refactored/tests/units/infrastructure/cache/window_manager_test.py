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

from core_bak_refactored.infrastructure.cache.window_cache import WindowsCache


class WindowManagerTest(unittest.TestCase):
    """WindowManager 功能测试"""
    
    def setUp(self):
        """测试初始化"""
        self.mgr = WindowsCache()
    
    # ========== 窗口键生成测试 ==========
    
    def test_make_window_key_monthly(self):
        """测试月度窗口键生成"""
        date = pd.Timestamp('2025-03-15')
        key = self.mgr._make_window_key(date, 'monthly', window_size=1)
        self.assertEqual(key, '2025-03_03')
    
    def test_make_window_key_weekly(self):
        """测试周度窗口键生成"""
        date = pd.Timestamp('2025-03-15')  # 2025年第10周
        key = self.mgr._make_window_key(date, 'weekly', window_size=1)
        self.assertTrue(key.startswith('2025-W'))
    
    def test_make_window_key_daily(self):
        """测试日度窗口键生成"""
        # 使用一个确定是交易日的日期
        date = pd.Timestamp('2025-03-10')  # 周一，交易日
        key = self.mgr._make_window_key(date, 'daily', window_size=1)
        # window_size=1时，是从年初开始的第69天（0-based index）
        self.assertIsNotNone(key)
        self.assertTrue('2025' in key)
    
    def test_generate_window_keys_single_month(self):
        """测试生成单月窗口键"""
        keys = self.mgr._generate_window_keys('2025-01-01', '2025-01-31', 'monthly', window_size=1)
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0], '2025-01_01')
    
    def test_generate_window_keys_multiple_months(self):
        """测试生成多月窗口键"""
        keys = self.mgr._generate_window_keys('2025-01-01', '2025-03-31', 'monthly', window_size=1)
        self.assertEqual(len(keys), 3)
        self.assertEqual(keys, ['2025-01_01', '2025-02_02', '2025-03_03'])
    
    def test_generate_window_keys_cross_year(self):
        """测试生成跨年窗口键"""
        keys = self.mgr._generate_window_keys('2024-11-01', '2025-02-28', 'monthly', window_size=1)
        self.assertEqual(len(keys), 4)
        self.assertEqual(keys, ['2024-11_11', '2024-12_12', '2025-01_01', '2025-02_02'])
    
    def test_generate_window_keys_weekly(self):
        """测试生成周度窗口键"""
        keys = self.mgr._generate_window_keys('2025-01-01', '2025-01-31', 'weekly', window_size=1)
        # 1月通常包含4-5周
        self.assertGreater(len(keys), 0)
        self.assertTrue(all(k.startswith('2025-W') for k in keys))
    
    def test_generate_window_keys_daily(self):
        """测试生成日度窗口键（window_size=1，每天一个窗口）"""
        # 使用一个不包含节假日的区间（避免调整）
        keys = self.mgr._generate_window_keys('2025-03-10', '2025-03-14', 'daily', window_size=1)
        # window_size=1时，每天是一个独立的窗口，格式为YYYYMMDD_YYYYMMDD
        # 3月10-14日是周一至周五，5个交易日
        self.assertEqual(len(keys), 5)
        # 验证每个键都是单日窗口格式
        for key in keys:
            start, end = key.split('_')
            self.assertEqual(start, end)  # 单日窗口，起止日期相同
    
    # ========== 边界场景测试 ==========
    
    def test_generate_window_keys_leap_year_february(self):
        """测试闰年2月窗口键"""
        # 2024是闰年
        keys = self.mgr._generate_window_keys('2024-02-01', '2024-02-29', 'monthly', window_size=1)
        self.assertEqual(len(keys), 1)
        self.assertEqual(keys[0], '2024-02_02')
        
        # 验证2月29日包含在内
        keys_daily = self.mgr._generate_window_keys('2024-02-28', '2024-02-29', 'daily', window_size=1)
        self.assertEqual(len(keys_daily), 2)
    
    def test_generate_window_keys_non_leap_year_february(self):
        """测试非闰年2月窗口键"""
        # 2025不是闰年
        keys = self.mgr._generate_window_keys('2025-02-01', '2025-02-28', 'monthly', window_size=1)
        self.assertEqual(len(keys), 1)
        
        # 2月29日不应存在，使用交易日测试
        # 2025-02-27（周四）和2025-02-28（周五）是交易日
        keys_daily = self.mgr._generate_window_keys('2025-02-27', '2025-02-28', 'daily', window_size=1)
        self.assertEqual(len(keys_daily), 2)
    
    def test_generate_window_keys_month_end_boundaries(self):
        """测试月末边界场景"""
        # 31天的月
        keys = self.mgr._generate_window_keys('2025-01-31', '2025-02-01', 'monthly', window_size=1)
        self.assertEqual(len(keys), 2)
        
        # 30天的月
        keys = self.mgr._generate_window_keys('2025-04-30', '2025-05-01', 'monthly', window_size=1)
        self.assertEqual(len(keys), 2)
    
    def test_generate_window_keys_year_boundary(self):
        """测试年末边界场景"""
        keys = self.mgr._generate_window_keys('2024-12-31', '2025-01-01', 'monthly', window_size=1)
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys, ['2024-12_12', '2025-01_01'])
        
        # 🔧 注意：2025-01-01是元旦，窗口起始日会调整到交易日
        # 使用不包含节假日的区间测试日度窗口
        keys_daily = self.mgr._generate_window_keys('2024-12-30', '2024-12-31', 'daily', window_size=1)
        self.assertEqual(len(keys_daily), 2)
    
    # ========== 窗口键转换测试 ==========
    
    def test_window_key_to_date_monthly_range(self):
        """测试月度窗口键转为日期范围"""
        start, end = self.mgr._window_key_to_date_range('2025-03_03', 'monthly')
        self.assertEqual(start, '2025-03-01')
        self.assertEqual(end, '2025-03-31')
    
    def test_window_key_to_date_february_range(self):
        """测试2月窗口键转换"""
        # 闰年2月
        start, end = self.mgr._window_key_to_date_range('2024-02_02', 'monthly')
        self.assertEqual(start, '2024-02-01')
        self.assertEqual(end, '2024-02-29')
        
        # 非闰年2月
        start, end = self.mgr._window_key_to_date_range('2025-02_02', 'monthly')
        self.assertEqual(start, '2025-02-01')
        self.assertEqual(end, '2025-02-28')
    
    def test_window_key_to_date_december_range(self):
        """测试12月窗口键转换"""
        start, end = self.mgr._window_key_to_date_range('2025-12_12', 'monthly')
        self.assertEqual(start, '2025-12-01')
        self.assertEqual(end, '2025-12-31')
    
    def test_window_key_to_date_daily_range(self):
        """测试日度窗口键转换"""
        start, end = self.mgr._window_key_to_date_range('20250315_20250315', 'daily')
        self.assertEqual(start, '2025-03-15')
        self.assertEqual(end, '2025-03-15')
    
    # ========== 窗口连续性测试 ==========
    # 注意：WindowManager没有is_continuous方法，连续性判断在CacheManager中
    # 这里我们测试通过window_key_to_date_range验证窗口边界
    
    def test_window_boundary_monthly_consecutive(self):
        """测试连续月份的窗口边界"""
        _, end1 = self.mgr._window_key_to_date_range('2025-01_01', 'monthly')
        start2, _ = self.mgr._window_key_to_date_range('2025-02_02', 'monthly')
        
        # 验证连续性：end1 + 1天 = start2
        from datetime import datetime, timedelta
        end1_dt = datetime.strptime(end1, '%Y-%m-%d')
        start2_dt = datetime.strptime(start2, '%Y-%m-%d')
        self.assertEqual((start2_dt - end1_dt).days, 1)
    
    def test_window_boundary_daily_consecutive(self):
        """测试连续日期的窗口边界"""
        _, end1 = self.mgr._window_key_to_date_range('20250115_20250115', 'daily')
        start2, _ = self.mgr._window_key_to_date_range('20250116_20250116', 'daily')
        
        # 验证连续性
        from datetime import datetime, timedelta
        end1_dt = datetime.strptime(end1, '%Y-%m-%d')
        start2_dt = datetime.strptime(start2, '%Y-%m-%d')
        self.assertEqual((start2_dt - end1_dt).days, 1)
    
    # ========== 异常输入测试 ==========
    
    def test_generate_window_keys_invalid_date_range(self):
        """测试无效日期范围（开始日期晚于结束日期）"""
        keys = self.mgr._generate_window_keys('2025-03-01', '2025-01-01', 'monthly', window_size=1)
        # 应返回空列表或抛出异常（取决于实现）
        self.assertEqual(len(keys), 0)
    
    def test_generate_window_keys_same_date(self):
        """测试相同日期"""
        keys = self.mgr._generate_window_keys('2025-01-15', '2025-01-15', 'daily', window_size=1)
        self.assertEqual(len(keys), 1)
    
    def test_window_key_to_date_invalid_format(self):
        """测试无效窗口键格式"""
        # 这应该抛出异常或返回None（取决于实现）
        try:
            result = self.mgr._window_key_to_date_range('invalid-key', 'monthly')
            # 如果没有抛出异常，至少应返回None或空字符串
            self.assertIn(result, [None, ('', ''), (None, None)])
        except (ValueError, AttributeError, IndexError):
            # 抛出异常是可接受的
            pass


if __name__ == '__main__':
    unittest.main()
