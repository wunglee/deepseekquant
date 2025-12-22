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
from typing import List, Tuple, Optional, Union, Dict
import pandas as pd

from core_bak_refactored.core.share.market.market_enums import MarketCode
from core_bak_refactored.core.share.market.trading_calendar_service import get_trading_calendar_service
from core_bak_refactored.infrastructure.cache.memory import MemoryCache
from core_bak_refactored.infrastructure.cache.redis import RedisCache

logger = logging.getLogger('DeepSeekQuant.WindowManager')


class WindowsCache:
    """时间窗口管理工具类（重构版）"""

    def __init__(self, redis_client=None):
        """初始化窗口管理器"""
        # 从配置文件加载缓存配置
        from core_bak_refactored.core.share.config_manager import ConfigManager
        config_manager = ConfigManager()
        cache_config = config_manager.get_cache_config()

        # 参数优先级：显式传入 > 配置文件
        self._calendar_service = get_trading_calendar_service()
        self._cache_mode = cache_config.cache_mode
        if self._cache_mode == 'memory':
            self._fast_cache = MemoryCache(max_windows=cache_config.memory_max_windows, ttl=cache_config.memory_ttl)
            logger.info(
                f"✅ 使用内存缓存: max_windows={cache_config.memory_max_windows}, ttl={cache_config.memory_ttl}s [从配置加载]")
        elif self._cache_mode == 'redis':
            self._fast_cache = RedisCache(redis_client=redis_client, ttl=cache_config.redis_ttl)
            logger.info(f"✅ 使用Redis缓存: ttl={cache_config.redis_ttl}s [从配置加载]")
        else:
            raise ValueError(f"无效的 cache_mode: {cache_config.cache_mode}，必须是 'memory' 或 'redis'")
        # 缓存窗口大小（静态配置，period的整数倍）
        self._window_size: dict[str, int] = cache_config.window_size
        logger.info(
            f"✅ WindowsCache 初始化完成: cache_mode={cache_config.cache_mode}, window_size={self._window_size}[从配置加载]")

    def _make_window_key(self, date: pd.Timestamp, period: str, market_code: Optional[MarketCode] = MarketCode.CN) -> \
            Optional[str]:
        """
        生成时间窗口键
        
        Args:
            date: 日期
            period: 数据粒度 (daily/weekly/monthly)
            market_code: 市场代码
        
        Returns:
            窗口键字符串，如果窗口无效（调整后window_start > window_end）则返回None
        
        Examples:
            >>> self._make_window_key(pd.Timestamp('2025-01-15'), 'daily')
            '20250113_20250119'  # 2025-01-13 到 2025-01-19 (7天窗口，对齐到周一)
            
            >>> self._make_window_key(pd.Timestamp('2025-01-15'), 'weekly')
            '2025-W02_W05'  # 第2周到第5周 (4周窗口)
            
            >>> self._make_window_key(pd.Timestamp('2025-02-15'), 'monthly')
            '2025-01_03'  # 1月到3月 (3个月窗口)
        """
        if not isinstance(market_code, MarketCode):
            raise TypeError("market_code must be MarketCode")
        if not isinstance(date, pd.Timestamp):
            raise TypeError("date must be pd.Timestamp")

        # 默认为中国市场
        if market_code is None:
            market_code = MarketCode.CN

        if period == 'daily':
            window_size = self._window_size.get('daily')
            # Daily窗口：按window_size天一个窗口
            # 🔧 修复BUG：直接使用date计算，而不是对齐到周一
            year_start = pd.Timestamp(f'{date.year}-01-01')
            days_from_year_start = (date - year_start).days
            window_index = days_from_year_start // window_size

            # 计算窗口边界
            window_start = year_start + pd.Timedelta(days=window_index * window_size)
            window_end = window_start + pd.Timedelta(days=window_size - 1)

            # 🔧 关键：调整窗口边界到交易日
            # window_start: 向后推到下一个交易日
            # window_end: 向前推到上一个交易日
            # 这个调整是确定性的，只依赖于窗口边界本身，不依赖查询时间

            # 调整window_start到下一个交易日
            if not self._calendar_service.is_trading_day(market_code, window_start):
                next_trading = self._calendar_service.get_next_trading_day(market_code, window_start)
                if next_trading:
                    window_start = pd.Timestamp(next_trading)
                    logger.debug(
                        f"📅 窗口起始日调整: {window_start.date()} (非交易日) → {next_trading.date()} (交易日)")
                else:
                    logger.warning(f"⚠️ 无法找到 {window_start.date()} 之后的交易日，返回None")
                    return None

            # 调整window_end到上一个交易日
            if not self._calendar_service.is_trading_day(market_code, window_end):
                prev_trading = self._calendar_service.get_previous_trading_day(market_code, window_end)
                if prev_trading:
                    window_end = pd.Timestamp(prev_trading)
                    logger.debug(
                        f"📅 窗口结束日调整: {window_end.date()} (非交易日) → {prev_trading.date()} (交易日)")
                else:
                    logger.warning(f"⚠️ 无法找到 {window_end.date()} 之前的交易日，返回None")
                    return None

            # 检查调整后的窗口是否有效（起始日期必须早于或等于结束日期）
            if window_start > window_end:
                return None

            return f"{window_start.strftime('%Y%m%d')}_{window_end.strftime('%Y%m%d')}"

        elif period == 'weekly':
            window_size = self._window_size.get('weekly')
            # Weekly窗口：基于ISO周，window_size周一个窗口
            iso_year, iso_week, _ = date.isocalendar()

            # 计算窗口索引（从第1周开始，每window_size周一个窗口）
            window_index = (iso_week - 1) // window_size
            start_week = window_index * window_size + 1
            end_week = start_week + window_size - 1

            # 🔧 计算结束周的实际日期，处理跨年情况
            end_date = pd.to_datetime(f'{iso_year}-W{end_week:02d}-7', format='%G-W%V-%u')
            # ⚠️ 使用 ISO 年份，而不是日历年份
            end_year = end_date.isocalendar()[0]  # ISO year

            # 始终包含结束年份，保持格式一致
            return f"{iso_year}-W{start_week:02d}_{end_year}-W{end_week:02d}"

        elif period == 'monthly':
            window_size = self._window_size.get('monthly')
            # Monthly窗口：基于月份，window_size月一个窗口
            # 计算窗口索引（从1月开始，每window_size月一个窗口）
            window_index = (date.month - 1) // window_size
            start_month = window_index * window_size + 1
            end_month = start_month + window_size - 1

            # 🔧 处理跨年情况
            start_year = date.year
            if end_month > 12:
                # 跨年情况：结束月份在下一年
                end_year = start_year + (end_month - 1) // 12
                end_month = ((end_month - 1) % 12) + 1
            else:
                # 同年情况
                end_year = start_year

            # 始终包含结束年份，保持格式一致
            return f"{start_year}-{start_month:02d}_{end_year}-{end_month:02d}"

        else:
            raise ValueError(f"不支持的 period: {period}，必须是 'daily', 'weekly' 或 'monthly'")

    def _generate_window_keys(self, start: pd.Timestamp, end: pd.Timestamp, period: str,
                              market_code: Optional[MarketCode] = None) -> List[str]:
        """
        生成指定范围内的所有窗口键
        
        Args:
            start: 开始日期（YYYY-MM-DD）
            end: 结束日期（YYYY-MM-DD）
            period: 数据粒度 (daily/weekly/monthly)
            market_code:市场代码
        
        Returns:
            窗口键列表（去重且排序）
        
        Examples:
            >>> self._generate_window_keys(pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-31'), 'weekly')
            ['2025-W01_W01', '2025-W02_W02', '2025-W03_W03', '2025-W04_W04', '2025-W05_W05']
            
            >>> self._generate_window_keys(pd.Timestamp('2025-01-01'), pd.Timestamp('2025-03-31'), 'monthly')
            ['2025-01_03']
        """
        if not isinstance(market_code, MarketCode):
            raise TypeError("market_code must be MarketCode")
        if not isinstance(start, pd.Timestamp):
            raise TypeError("start must be pd.Timestamp")
        if not isinstance(end, pd.Timestamp):
            raise TypeError("end must be pd.Timestamp")

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

        # 为每个日期生成窗口键，去重，过滤None
        window_keys = set()
        for date in dates:
            window_key = self._make_window_key(date, period, market_code)
            if window_key is not None:  # 过滤无效窗口
                window_keys.add(window_key)
        logger.info(f"📦 需要 {len(window_keys)} 个窗口 (period={period}, window_size={self._window_size.get(period)})")
        return sorted(list(window_keys))

    @staticmethod
    def _window_key_to_date_range(window_key: str, period: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        将窗口键转换为日期范围
        
        Args:
            window_key: 窗口键
            period: 数据粒度 (daily/weekly/monthly)
        
        Returns:
            (start_date, end_date) 元组，格式为 YYYY-MM-DD
        
        Examples:
            >>> WindowsCache._window_key_to_date_range('20250113_20250119', 'daily')
            ('2025-01-13', '2025-01-19')
            
            >>> WindowsCache._window_key_to_date_range('2025-W02_W05', 'weekly')
            ('2025-01-06', '2025-02-02')  # 第2周周一到第5周周日
            
            >>> WindowsCache._window_key_to_date_range('2025-01_03', 'monthly')
            ('2025-01-01', '2025-03-31')
        """
        if period == 'daily':
            # Daily窗口格式: YYYYMMDD_YYYYMMDD
            start_str, end_str = window_key.split('_')
            start_date = pd.to_datetime(start_str, format='%Y%m%d')
            end_date = pd.to_datetime(end_str, format='%Y%m%d')
            return start_date, end_date

        elif period == 'weekly':
            # Weekly窗口格式: YYYY-Www_YYYY-Www (始终包含结束年份)
            parts = window_key.split('_')
            year_week_start = parts[0]  # YYYY-Www
            year_week_end = parts[1]  # YYYY-Www

            # 解析起始周
            year_start, week_start = year_week_start.split('-W')
            year_start = int(year_start)
            week_start = int(week_start)

            # 解析结束周
            year_end, week_end = year_week_end.split('-W')
            year_end = int(year_end)
            week_end = int(week_end)

            # 计算起始周的周一
            start_date = pd.to_datetime(f'{year_start}-W{week_start:02d}-1', format='%G-W%V-%u')

            # 计算结束周的周日
            end_date = pd.to_datetime(f'{year_end}-W{week_end:02d}-7', format='%G-W%V-%u')

            return start_date, end_date

        elif period == 'monthly':
            # Monthly窗口格式: YYYY-MM_YYYY-MM (始终包含结束年份)
            parts = window_key.split('_')
            year_month_start = parts[0]  # YYYY-MM
            year_month_end = parts[1]  # YYYY-MM

            # 解析起始月
            year_start, start_month = year_month_start.split('-')
            year_start = int(year_start)
            start_month = int(start_month)

            # 解析结束月
            year_end, end_month = year_month_end.split('-')
            year_end = int(year_end)
            end_month = int(end_month)

            # 起始月第一天
            start_date = pd.Timestamp(year=year_start, month=start_month, day=1)

            # 结束月最后一天
            if end_month == 12:
                end_date = pd.Timestamp(year=year_end + 1, month=1, day=1) - pd.Timedelta(days=1)
            else:
                end_date = pd.Timestamp(year=year_end, month=end_month + 1, day=1) - pd.Timedelta(days=1)

            return start_date, end_date

        else:
            raise ValueError(f"不支持的 period: {period}")

    @staticmethod
    def is_date_in_window(window_key: str, period: str, date: pd.Timestamp) -> bool:
        """
        判断窗口是否为当前未完成窗口

        Args:
            window_key: 窗口键
            period: 数据粒度
            date: 当前日期

        Returns:
            True 如果是当前未完成窗口
        """
        start_str, end_str = WindowsCache._window_key_to_date_range(window_key, period)
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)

        # 当前日期在窗口范围内，且窗口尚未结束
        return start_date <= date <= end_date

    def merge_continuous_windows(self, window_keys: list, period: str, market_code: MarketCode) -> list:
        """
        合并连续的窗口键，减少网络请求次数

        Args:
            window_keys: 缺失窗口键列表 (已排序)
            period: 数据粒度
            market_code: 市场代码，用于交易日历判断

        Returns:
            合并后的连续范围列表，每个元素包含:
            - start: 范围起始日期
            - end: 范围结束日期
            - windows: 该范围包含的窗口键列表

        Example:
            Input: ['2024-01_01', '2024-02_02', '2024-03_03', '2024-05_05', '2024-06_06']
            Output: [
                {'start': '2024-01-01', 'end': '2024-03-31', 'windows': ['2024-01_01', '2024-02_02', '2024-03_03']},
                {'start': '2024-05-01', 'end': '2024-06-30', 'windows': ['2024-05_05', '2024-06_06']}
            ]
        """
        if not window_keys:
            return []

        # 按窗口键排序
        sorted_keys = sorted(window_keys)

        merged_ranges = []
        current_range_windows = [sorted_keys[0]]

        for i in range(1, len(sorted_keys)):
            prev_key = sorted_keys[i - 1]
            curr_key = sorted_keys[i]

            # 检查是否连续：下一个窗口紧跟上一个窗口
            if self.is_consecutive_windows(prev_key, curr_key, period, market_code):
                # 连续，加入当前范围
                current_range_windows.append(curr_key)
            else:
                # 不连续，保存当前范围，开始新范围
                range_start, _ = self._window_key_to_date_range(current_range_windows[0], period)
                _, range_end = self._window_key_to_date_range(current_range_windows[-1], period)
                merged_ranges.append({
                    'start': pd.DataFrame(range_start),
                    'end': pd.DataFrame(range_end),
                    'windows': current_range_windows.copy()
                })
                current_range_windows = [curr_key]

        # 添加最后一个范围
        range_start, _ = self._window_key_to_date_range(current_range_windows[0], period)
        _, range_end = self._window_key_to_date_range(current_range_windows[-1], period)
        merged_ranges.append({
            'start': pd.Timestamp(range_start),
            'end': pd.Timestamp(range_end),
            'windows': current_range_windows.copy()
        })

        return merged_ranges

    def is_consecutive_windows(self, key1: str, key2: str, period: str, market_code: MarketCode) -> bool:
        """
        判断两个窗口是否连续（基于不同周期的判断逻辑）

        Args:
            key1: 第一个窗口键
            key2: 第二个窗口键
            period: 数据粒度 (daily/weekly/monthly)
            market_code: 市场代码枚举（MarketCode.CN/US/HK/JP/EU/SG）

        Returns:
            True 表示连续，False 表示不连续

        逻辑：
            - daily: 使用交易日历判断连续性（考虑节假日）
            - weekly: 判断ISO周号是否连续
            - monthly: 判断月份是否连续
        """

        if period == 'daily':
            # Daily周期：使用交易日历判断连续性
            # 获取两个窗口的日期范围
            _, end1 = self._window_key_to_date_range(key1, period)
            start2, _ = self._window_key_to_date_range(key2, period)

            # 转换为pd.Timestamp对象
            end1_dt = pd.to_datetime(end1)
            start2_dt = pd.to_datetime(start2)

            # 使用交易日历服务判断连续性
            return self._calendar_service.is_consecutive_trading_days(market_code, end1_dt, start2_dt)

        elif period == 'weekly':
            # Weekly周期：判断ISO周号是否连续
            # 窗口键格式：YYYY-Www_YYYY-Www（例如：2020-W52_2020-W52, 2020-W53_2021-W53）
            # 注意：2020-W53_2021-W53 表示 2020年第53周，结束日落在 2021年
            try:
                # 解析起始周和结束周的日期，然后计算 ISO 周号
                parts1 = key1.split('_')
                year1_week_start = parts1[0]  # YYYY-Www (起始周)
                year1_week_end = parts1[1]  # YYYY-Www (结束周)

                parts2 = key2.split('_')
                year2_week_start = parts2[0]  # YYYY-Www (起始周)

                # 计算 key1 的结束周的实际 ISO 周号
                year1_end, week1_end = year1_week_end.split('-W')
                end1_date = pd.to_datetime(f'{year1_end}-W{week1_end}-7', format='%G-W%V-%u')
                end1_iso_year, end1_iso_week, _ = end1_date.isocalendar()

                # 计算 key2 的起始周的实际 ISO 周号
                year2_start, week2_start = year2_week_start.split('-W')
                start2_date = pd.to_datetime(f'{year2_start}-W{week2_start}-1', format='%G-W%V-%u')
                start2_iso_year, start2_iso_week, _ = start2_date.isocalendar()

                # 判断连续性：比较 ISO 年份和 ISO 周号
                if end1_iso_year == start2_iso_year:
                    # 同一年：周号连续
                    return (end1_iso_week + 1) == start2_iso_week
                elif end1_iso_year + 1 == start2_iso_year:
                    # 跨年：第53周 → 第1周
                    return end1_iso_week >= 52 and start2_iso_week == 1
                else:
                    return False
            except (ValueError, IndexError) as e:
                logger.warning(f"⚠️ 解析周窗口键失败: {key1}, {key2}, error: {e}")
                return False

        elif period == 'monthly':
            # Monthly周期：判断月份是否连续
            # 窗口键格式：YYYY-MM_YYYY-MM（例如：2025-01_2025-03）
            try:
                # 解析key1的结束月
                parts1 = key1.split('_')
                year_month1_end = parts1[1]  # YYYY-MM (结束月)
                year1, month1_end = year_month1_end.split('-')
                year1 = int(year1)
                month1_end = int(month1_end)

                # 解析key2的起始月
                parts2 = key2.split('_')
                year_month2_start = parts2[0]  # YYYY-MM (起始月)
                year2, month2_start = year_month2_start.split('-')
                year2 = int(year2)
                month2_start = int(month2_start)

                # 判断连续性
                if year1 == year2:
                    # 同一年：月份连续（month1_end + 1 == month2_start）
                    return (month1_end + 1) == month2_start
                elif year1 + 1 == year2:
                    # 跨年：year1的12月 → year2的1月
                    return month1_end == 12 and month2_start == 1
                else:
                    return False
            except (ValueError, IndexError) as e:
                logger.warning(f"⚠️ 解析月窗口键失败: {key1}, {key2}, error: {e}")
                return False

        else:
            logger.warning(f"⚠️ 不支持的周期: {period}")
            return False

    def _is_first_window(self, window_start: pd.Timestamp, search_from_data: pd.Timestamp, period: str, market_code: MarketCode) -> bool:
        """
        判断窗口是否为起始窗口（封装所有判断逻辑）

        不同周期的判断逻辑：
        1. 日周期(daily)：
           - 需要精确到交易日判断
           - 确保查询开始时间所在交易日没有数据

        2. 周周期(weekly)：
           - 只需精确到周的比较
           - 确保查询开始时间所在的整个"交易周"没有数据
           - 如果整周都是假期（春节、五一、十一等），需向后推到交易周

        3. 月周期(monthly)：
           - 只需精确到月的比较
           - 确保查询开始时间所在整个月没有数据

        Args:
            window_start:窗口起始时间
            period: 数据粒度 (daily/weekly/monthly)
            search_from_data: 查询起始时间
            market_code: 市场代码

        Returns:
            bool: True表示是起始窗口
        """
        # 🔧 根据周期类型进行不同的判断
        if period == 'daily':
            # 日周期：精确到交易日判断
            is_first = self._is_first_window_daily(window_start, search_from_data, search_from_data, market_code)
        elif period == 'weekly':
            # 周周期：精确到周判断
            is_first = self._is_first_window_weekly(window_start,
                                                    search_from_data, search_from_data, market_code)
        elif period == 'monthly':
            # 月周期：精确到月判断
            is_first = self._is_first_window_monthly(window_start, search_from_data, search_from_data)
        else:
            # 未知周期，使用保守判断
            is_first = True
            logger.warning(f"未知周期类型: {period}，使用保守判断")
        return is_first

    def _is_first_window_daily(self, window_start: pd.Timestamp, query_start: pd.Timestamp,
                               search_from_data: pd.Timestamp, market_code: MarketCode) -> bool:
        """
        日周期的起始窗口判断

        逻辑：确保查询开始时间所在交易日没有数据
        """
        query_start_dt = query_start  # 已经是 pd.Timestamp 类型，无需转换
        if not self._calendar_service.is_trading_day(market_code, query_start_dt):
            next_trading_day = self._calendar_service.get_next_trading_day(market_code, query_start_dt)
            if next_trading_day:
                query_start = next_trading_day
                logger.debug(
                    f"📅 调整查询起始日期: {query_start_dt.strftime('%Y-%m-%d')} (非交易日) → {next_trading_day.strftime('%Y-%m-%d')} (交易日)")

        return query_start < window_start <= search_from_data

    def _is_first_window_weekly(self, window_start: pd.Timestamp, query_start: pd.Timestamp,
                                search_from_data: pd.Timestamp, market_code: MarketCode) -> bool:
        """
        周周期的起始窗口判断

        逻辑：确保查询开始时间所在的整个"交易周"没有数据
        - 获取 query_start 所在周的周一和周日
        - 如果整周都不是交易周（春节、五一、十一等长假），向后推到下一个交易周
        - 比较交易周的起始时间和 actual_earliest_start
        """

        # 获取 query_start 所在周的周一
        query_week_start = query_start - pd.Timedelta(days=query_start.weekday())
        query_week_end = query_week_start + pd.Timedelta(days=6)

        # 检查这一周是否有交易日
        has_trading_day = False
        current_date = query_week_start
        while current_date <= query_week_end:
            # 已经是 pd.Timestamp 类型，无需转换
            if self._calendar_service.is_trading_day(market_code, current_date):
                has_trading_day = True
                break
            current_date += pd.Timedelta(days=1)

        # 如果整周都不是交易周，向后推到下一个交易周
        if not has_trading_day:
            # 已经是 pd.Timestamp 类型，无需转换
            next_trading_day = self._calendar_service.get_next_trading_day(
                market_code, query_week_end
            )
            if next_trading_day:
                # 获取下一个交易日所在周的周一
                next_trading_ts = next_trading_day
                query_week_start = next_trading_ts - pd.Timedelta(days=next_trading_ts.weekday())
                logger.debug(f"📅 整周非交易周，调整到下一个交易周: {query_week_start.strftime('%Y-%m-%d')}")

        # 只比较周，不比较具体日期
        query_week = query_week_start.isocalendar()[1]  # ISO 周数
        window_week = window_start.isocalendar()[1]
        actual_earliest_week = search_from_data.isocalendar()[1]
        query_year = query_week_start.year
        window_year = window_start.year
        actual_earliest_year = search_from_data.year

        # 比较年份和周数
        is_first_window_weekly = (query_year < window_year <= actual_earliest_year) or (
                query_year == window_year == actual_earliest_year and query_week < window_week <= actual_earliest_week)
        if is_first_window_weekly:
            logger.debug(
                f"📅遇到起始窗口：query: {query_year}-{query_week},window: {window_year}-{window_week},actual_earliest: {actual_earliest_year}-{actual_earliest_week}")
        return is_first_window_weekly

    def _is_first_window_monthly(self, window_start: pd.Timestamp, query_start: pd.Timestamp,
                                 search_from_data: pd.Timestamp) -> bool:
        """
        月周期的起始窗口判断

        逻辑：确保查询开始时间所在整个月没有数据
        """
        # 只比较年月，不比较具体日期
        query_year_month = (query_start.year, query_start.month)
        window_start_month = (window_start.year, window_start.month)
        actual_earliest_year_month = (search_from_data.year, search_from_data.month)

        return query_year_month < window_start_month <= actual_earliest_year_month

    def distribute_data_to_windows(self, symbol: str, period: str, data: pd.DataFrame,
                                   window_keys: list, cached_windows: dict, search_from_data: pd.Timestamp,
                                   market_code: MarketCode) -> None:
        """
        将大范围数据分配到各个窗口，并写入缓存

        Args:
            symbol: 股票/指数代码
            period: 数据粒度
            data: 大范围查询返回的数据
            window_keys: 需要分配的窗口键列表
            cached_windows: 已缓存窗口字典 (输出参数)
            query_start_date: 查询起始日期（用于判断起始窗口）
            market_code: 市场代码枚举，用于交易日历判断
        """
        if data.empty:
            return

        # 确保数据有 date 列
        if 'date' not in data.columns:
            logger.warning("⚠️ 数据缺少 date 列，无法分配到窗口")
            return

        data['date'] = pd.to_datetime(data['date'])

        # 分配数据到各个窗口
        window_keys = sorted(window_keys, reverse=True)
        found_first_window = False

        for idx, window_key in enumerate(window_keys):
            if found_first_window:
                break

            window_start, window_end = self._window_key_to_date_range(window_key, period)

            # 筛选该窗口的数据
            window_data = data[(data['date'] >= window_start) & (data['date'] <= window_end)].copy()

            if not window_data.empty:
                is_first_window = self._is_first_window(
                    window_start=window_start,
                    search_from_data=search_from_data,
                    period=period,
                    market_code=market_code
                )

                self._fast_cache.set(symbol, period, window_key, window_data, is_first_window=is_first_window)
                cached_windows[window_key] = window_data

                logger.debug(f"  ✅ 窗口 {window_key}-{window_end}: {len(window_data)} 条")

                if is_first_window:
                    found_first_window = True
                    logger.info(
                        f"🅰️ 检测到起始窗口: {window_key} (查询条件从 {search_from_data.strftime('%Y-%m-%d')}，但数据源最早从 {search_from_data.strftime('%Y-%m-%d')} 开始)")

    def _backfill_first_window_flag(self, symbol: str, period: str, actual_earliest_start: pd.Timestamp,
                                    current_window_keys: list, market_code: MarketCode = MarketCode.CN) -> None:
        """
        回溯更新起始窗口标记

        场景：
        - 首次查询正好从上市日开始（如 2025-01-08 ~ 2025-01-12）
        - query_start == actual_earliest_start，未被标记为起始窗口
        - 第二次查询包含更早日期（如 2025-01-06 ~ 2025-01-12）
        - 检测到 query_start < actual_earliest_start，确认为起始窗口
        - 需要回溯更新之前缓存中包含 actual_earliest_start 的窗口

        Args:
            symbol: 股票/指数代码
            period: 数据粒度/K线类型
            actual_earliest_start: 数据源返回的实际最早日期（上市日）
            current_window_keys: 当前查询涉及的窗口键列表（避免重复更新）
        """
        # 生成包含 actual_earliest_start 的窗口键
        first_data_window_key = self._make_window_key(actual_earliest_start, period, market_code)

        # 如果当前查询已经处理过这个窗口，无需回溯
        if first_data_window_key in current_window_keys:
            return

        # 尝试更新该窗口的标记
        if hasattr(self._fast_cache, 'update_first_window_flag'):
            updated = self._fast_cache.update_first_window_flag(
                symbol, period, first_data_window_key, is_first_window=True
            )

            if updated:
                logger.info(f"✅ 回溯成功: 更新窗口 {first_data_window_key} 为起始窗口")

    def get_cached_and_missing_windows(self,
                                       symbol: str,
                                       start_date:pd.Timestamp,
                                       end_date:pd.Timestamp,
                                       market_code: MarketCode = MarketCode.CN,
                                       period: str = "daily"):
        # ========== 第1步：生成所需的所有窗口键 ==========
        window_keys = self._generate_window_keys(start_date, end_date, period, market_code)
        # ========== 第2步：从快速缓存获取已有窗口 ==========
        cached_windows = {}
        missing_windows = []
        first_window_key = None  # 记录起始窗口（最早数据）
        current_window_key = self._make_window_key(pd.Timestamp.now(), period, market_code)
        logger.debug(f"📅 当前窗口: {current_window_key}")
        for window_key in window_keys:
            cached_value = self._fast_cache.get(symbol, period, window_key)
            if cached_value is not None:
                # 两种缓存现在都返回dict：{'data': df, 'is_first_window': bool, 'timestamp': float}
                cached_df = cached_value.get('data')
                is_first = cached_value.get('is_first_window', False)

                # 记录起始窗口
                if is_first:
                    first_window_key = window_key
                    logger.info(f"🅰️ 检测到起始窗口: {window_key} (最早数据)")

                # 检查是否为当前窗口（未完成窗口）
                is_current_window = (window_key == current_window_key)

                if not is_current_window:
                    # 已完成的窗口，直接使用缓存
                    cached_windows[window_key] = cached_df
                else:
                    # 当前窗口，需要重新查询
                    logger.info(f"🔄 当前窗口 {window_key} 需要更新（窗口尚未结束）")
                    missing_windows.append(window_key)
            else:
                # 缺失窗口
                missing_windows.append(window_key)
        # 检查是否有比起始窗口更早的查询
        if first_window_key is not None:
            # 移除所有早于起始窗口的请求
            original_count = len(missing_windows)
            missing_windows = [w for w in missing_windows if w >= first_window_key]
            removed_count = original_count - len(missing_windows)

            if removed_count > 0:
                logger.info(f"🚫 已忽略 {removed_count} 个早于起始窗口 {first_window_key} 的查询")
        logger.info(f"✅ {self._cache_mode}命中: {len(cached_windows)}/{len(window_keys)} 个窗口")
        return cached_windows, missing_windows

    def clear_all_cache(self) -> None:
        """清空所有层级的缓存"""
        self._fast_cache.clear()
        logger.info(f"✅ 所有缓存已清空 (cache_mode={self._cache_mode})")

    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            'cache_mode': self._cache_mode,
            self._cache_mode: self._fast_cache.get_stats()
        }
