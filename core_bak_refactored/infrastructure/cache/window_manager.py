"""
时间窗口管理器（重构版） - 基于 period 和 window_size 的窗口管理

核心概念：
- period（数据粒度）: daily/weekly/monthly，K线类型
- window_size（窗口大小）: 整数，表示包含多少个period单位
  例如：period=daily, window_size=7 → 7天一个窗口
       period=weekly, window_size=4 → 4周一个窗口
       period=monthly, window_size=3 → 3个月一个窗口

窗口键格式：
- daily窗口: YYYYMMDD_YYYYMMDD (起始日期_结束日期，例如：20250113_20250119)
- weekly窗口: YYYY-Www_Www (起始周_结束周，例如：2025-W02_W02 表示单周窗口)
- monthly窗口: YYYY-MM_MM (起始月_结束月，例如：2025-01_03 表示1-3月)

注意：
- 窗口边界对齐到period的自然边界
- daily: 自然日
- weekly: ISO周（周一到周日）
- monthly: 月初到月末
"""

import logging
from typing import List, Tuple
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger('DeepSeekQuant.WindowManager')


class WindowManager:
    """时间窗口管理工具类（重构版）"""
    
    @staticmethod
    def make_window_key(date: pd.Timestamp, period: str, window_size: int) -> str:
        """
        生成时间窗口键
        
        Args:
            date: 日期
            period: 数据粒度 (daily/weekly/monthly)
            window_size: 窗口大小（period的整数倍）
        
        Returns:
            窗口键字符串
        
        Examples:
            >>> make_window_key(pd.Timestamp('2025-01-15'), 'daily', 7)
            '20250113_20250119'  # 2025-01-13 到 2025-01-19 (7天窗口，对齐到周一)
            
            >>> make_window_key(pd.Timestamp('2025-01-15'), 'weekly', 4)
            '2025-W02_W05'  # 第2周到第5周 (4周窗口)
            
            >>> make_window_key(pd.Timestamp('2025-02-15'), 'monthly', 3)
            '2025-01_03'  # 1月到3月 (3个月窗口)
        """
        if period == 'daily':
            # Daily窗口：按window_size天一个窗口
            # 🔧 修复BUG：直接使用date计算，而不是对齐到周一
            year_start = pd.Timestamp(f'{date.year}-01-01')
            days_from_year_start = (date - year_start).days
            window_index = days_from_year_start // window_size
            
            # 计算窗口边界
            window_start = year_start + pd.Timedelta(days=window_index * window_size)
            window_end = window_start + pd.Timedelta(days=window_size - 1)
            
            return f"{window_start.strftime('%Y%m%d')}_{window_end.strftime('%Y%m%d')}"
        
        elif period == 'weekly':
            # Weekly窗口：基于ISO周，window_size周一个窗口
            iso_year, iso_week, _ = date.isocalendar()
            
            # 计算窗口索引（从第1周开始，每window_size周一个窗口）
            window_index = (iso_week - 1) // window_size
            start_week = window_index * window_size + 1
            end_week = start_week + window_size - 1
            
            return f"{iso_year}-W{start_week:02d}_W{end_week:02d}"
        
        elif period == 'monthly':
            # Monthly窗口：基于月份，window_size月一个窗口
            # 计算窗口索引（从1月开始，每window_size月一个窗口）
            window_index = (date.month - 1) // window_size
            start_month = window_index * window_size + 1
            end_month = start_month + window_size - 1
            
            return f"{date.year}-{start_month:02d}_{end_month:02d}"
        
        else:
            raise ValueError(f"不支持的 period: {period}，必须是 'daily', 'weekly' 或 'monthly'")
    
    @staticmethod
    def generate_window_keys(start_date: str, end_date: str, period: str, window_size: int) -> List[str]:
        """
        生成指定范围内的所有窗口键
        
        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            period: 数据粒度 (daily/weekly/monthly)
            window_size: 窗口大小（period的整数倍）
        
        Returns:
            窗口键列表（去重且排序）
        
        Examples:
            >>> generate_window_keys('2025-01-01', '2025-01-31', 'weekly', 1)
            ['2025-W01_W01', '2025-W02_W02', '2025-W03_W03', '2025-W04_W04', '2025-W05_W05']
            
            >>> generate_window_keys('2025-01-01', '2025-03-31', 'monthly', 3)
            ['2025-01_03']
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if start > end:
            return []
        
        # 生成日期范围内的所有代表性日期
        if period == 'daily':
            # 每天生成一个日期
            dates = pd.date_range(start=start, end=end, freq='D')
        elif period == 'weekly':
            # 每周生成一个日期（周一）
            dates = pd.date_range(start=start, end=end, freq='W-MON')
            # 确保包含起始日期所在的周
            if dates.empty or dates[0] > start:
                dates = pd.DatetimeIndex([start]).union(dates)
        elif period == 'monthly':
            # 每月生成一个日期（月初）
            dates = pd.date_range(start=start, end=end, freq='MS')
            # 确保包含起始月份
            if dates.empty or dates[0] > start:
                dates = pd.DatetimeIndex([start]).union(dates)
        else:
            raise ValueError(f"不支持的 period: {period}")
        
        # 为每个日期生成窗口键，去重
        window_keys = set()
        for date in dates:
            window_key = WindowManager.make_window_key(date, period, window_size)
            window_keys.add(window_key)
        
        # 排序返回
        return sorted(list(window_keys))
    
    @staticmethod
    def window_key_to_date_range(window_key: str, period: str) -> Tuple[str, str]:
        """
        将窗口键转换为日期范围
        
        Args:
            window_key: 窗口键
            period: 数据粒度 (daily/weekly/monthly)
        
        Returns:
            (start_date, end_date) 元组，格式为 YYYY-MM-DD
        
        Examples:
            >>> window_key_to_date_range('20250113_20250119', 'daily')
            ('2025-01-13', '2025-01-19')
            
            >>> window_key_to_date_range('2025-W02_W05', 'weekly')
            ('2025-01-06', '2025-02-02')  # 第2周周一到第5周周日
            
            >>> window_key_to_date_range('2025-01_03', 'monthly')
            ('2025-01-01', '2025-03-31')
        """
        if period == 'daily':
            # Daily窗口格式: YYYYMMDD_YYYYMMDD
            start_str, end_str = window_key.split('_')
            start_date = pd.to_datetime(start_str, format='%Y%m%d')
            end_date = pd.to_datetime(end_str, format='%Y%m%d')
            return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        
        elif period == 'weekly':
            # Weekly窗口格式: YYYY-Www_Www
            parts = window_key.split('_')
            year_week_start = parts[0]  # YYYY-Www
            week_end = parts[1]  # Www
            
            # 解析起始周
            year, week_start = year_week_start.split('-W')
            year = int(year)
            week_start = int(week_start)
            week_end = int(week_end.replace('W', ''))
            
            # 计算起始周的周一
            start_date = pd.to_datetime(f'{year}-W{week_start:02d}-1', format='%G-W%V-%u')
            
            # 计算结束周的周日
            end_date = pd.to_datetime(f'{year}-W{week_end:02d}-7', format='%G-W%V-%u')
            
            return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        
        elif period == 'monthly':
            # Monthly窗口格式: YYYY-MM_MM
            year_month, end_month = window_key.split('_')
            year, start_month = year_month.split('-')
            year = int(year)
            start_month = int(start_month)
            end_month = int(end_month)
            
            # 起始月第一天
            start_date = pd.Timestamp(year=year, month=start_month, day=1)
            
            # 结束月最后一天
            if end_month == 12:
                end_date = pd.Timestamp(year=year + 1, month=1, day=1) - pd.Timedelta(days=1)
            else:
                end_date = pd.Timestamp(year=year, month=end_month + 1, day=1) - pd.Timedelta(days=1)
            
            return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        
        else:
            raise ValueError(f"不支持的 period: {period}")
    
    @staticmethod
    def is_current_window(window_key: str, period: str, current_date: pd.Timestamp = None) -> bool:
        """
        判断窗口是否为当前未完成窗口
        
        Args:
            window_key: 窗口键
            period: 数据粒度
            current_date: 当前日期（默认为今天）
        
        Returns:
            True 如果是当前未完成窗口
        """
        if current_date is None:
            current_date = pd.Timestamp.now().normalize()
        
        start_str, end_str = WindowManager.window_key_to_date_range(window_key, period)
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)
        
        # 当前日期在窗口范围内，且窗口尚未结束
        return start_date <= current_date <= end_date
