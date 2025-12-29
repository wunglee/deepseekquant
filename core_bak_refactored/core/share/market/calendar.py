"""
市场日历模块（从Da taFetcher._is_market_open 和 _is_market_holiday 迁移而来）

职责：
1. 判断市场是否开盘
2. 检查市场假日
3. 时区转换（转换为东部时间）
4. 开盘时间验证

注意：所有时间处理统一使用 pd.Timestamp
"""

import logging
from datetime import time

import pandas as pd
import pytz

logger = logging.getLogger(__name__)


def is_market_open(dt: pd.Timestamp) -> bool:
    """
    判断市场是否开盘（从 DataFetcher._is_market_open 迁移而来）。
    
    美国股市开盘时间: 工作日 9:30-16:00 ET（东部时间）
    
    Args:
        dt: 待检查的时间（可以是任意时区）
    
    Returns:
        True如果市场开盘，False否则
    
    Example:
        >>>
        >>> import pytz
        >>> ny_time = pytz.timezone('US/Eastern').localize(datetime(2024, 1, 15, 10, 30))
        >>> is_market_open(ny_time)
        True  # 工作日上午10:30在开盘时间内
    """
    try:
        # 检查是否为周末
        if dt.weekday() >= 5:  # 周六(5)或周日(6)
            logger.debug(f"市场未开盘: {dt.strftime('%Y-%m-%d')} 是周末")
            return False

        # 检查节假日
        if is_market_holiday(dt):
            logger.debug(f"市场未开盘: {dt.strftime('%Y-%m-%d')} 是假日")
            return False

        # 转换为东部时间
        eastern = pytz.timezone('US/Eastern')
        if dt.tzinfo is None:
            # 如果没有时区信息，假设为UTC
            dt = pytz.UTC.localize(dt)
        dt_eastern = dt.astimezone(eastern)

        # 检查是否在开盘时间内
        market_open = time(9, 30)  # 9:30 AM
        market_close = time(16, 0)  # 4:00 PM

        is_open = market_open <= dt_eastern.time() <= market_close
        
        if is_open:
            logger.debug(f"市场开盘: {dt_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        else:
            logger.debug(f"市场未开盘: {dt_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')} 不在开盘时间内")
        
        return is_open

    except Exception as e:
        logger.error(f"判断市场开盘状态失败: {e}")
        return False


def is_market_holiday(dt: pd.Timestamp) -> bool:
    """
    检查是否为市场假日（从 DataFetcher._is_market_holiday 迁移而来）。
    
    主要美国市场假日包括：
    - 新年（1月1日）
    - 马丁·路德·金纪念日（1月第三个周一）
    - 总统日（2月第三个周一）
    - 耶稣受难日（复活节前的星期五，日期可变）
    - 阵亡将士纪念日（5月最后一个周一）
    - 六月节（6月19日）
    - 独立日（7月4日）
    - 劳动节（9月第一个周一）
    - 感恩节（11月第四个周四）
    - 圣诞节（12月25日）
    
    Args:
        dt: 待检查的日期
    
    Returns:
        True如果是假日，False否则
    
    Example:
        >>>
        >>> is_market_holiday(datetime(2024, 12, 25))
        True  # 圣诞节
        >>> is_market_holiday(datetime(2024, 1, 15))
        False  # 普通工作日
    """
    try:
        # 固定日期的假日（简化版，实际生产中应使用更精确的规则）
        # 注意：某些假日如果在周末，会顺延到周一
        fixed_holidays = {
            (1, 1): "New Year's Day",
            (6, 19): "Juneteenth",
            (7, 4): "Independence Day",
            (12, 25): "Christmas Day"
        }

        # 检查固定日期假日
        date_tuple = (dt.month, dt.day)
        if date_tuple in fixed_holidays:
            logger.debug(f"检测到假日: {fixed_holidays[date_tuple]} ({dt.strftime('%Y-%m-%d')})")
            return True

        # 浮动假日（基于周几的假日）
        # 马丁·路德·金纪念日：1月第三个周一
        if dt.month == 1:
            third_monday = get_nth_weekday(dt.year, 1, 0, 3)  # 0表示周一
            if dt.day == third_monday.day:
                logger.debug(f"检测到假日: Martin Luther King Jr. Day ({dt.strftime('%Y-%m-%d')})")
                return True

        # 总统日：2月第三个周一
        if dt.month == 2:
            third_monday = get_nth_weekday(dt.year, 2, 0, 3)
            if dt.day == third_monday.day:
                logger.debug(f"检测到假日: Presidents' Day ({dt.strftime('%Y-%m-%d')})")
                return True

        # 阵亡将士纪念日：5月最后一个周一
        if dt.month == 5:
            last_monday = get_last_weekday(dt.year, 5, 0)
            if dt.day == last_monday.day:
                logger.debug(f"检测到假日: Memorial Day ({dt.strftime('%Y-%m-%d')})")
                return True

        # 劳动节：9月第一个周一
        if dt.month == 9:
            first_monday = get_nth_weekday(dt.year, 9, 0, 1)
            if dt.day == first_monday.day:
                logger.debug(f"检测到假日: Labor Day ({dt.strftime('%Y-%m-%d')})")
                return True

        # 感恩节：11月第四个周四
        if dt.month == 11:
            fourth_thursday = get_nth_weekday(dt.year, 11, 3, 4)  # 3表示周四
            if dt.day == fourth_thursday.day:
                logger.debug(f"检测到假日: Thanksgiving Day ({dt.strftime('%Y-%m-%d')})")
                return True

        return False

    except Exception as e:
        logger.error(f"检查市场假日失败: {e}")
        return False


def get_nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """
    获取某月的第N个星期几。
    
    Args:
        year: 年份
        month: 月份
        weekday: 星期几（0=周一, 1=周二, ..., 6=周日）
        n: 第几个（1-5）
    
    Returns:
        对应的日期 (pd.Timestamp)
    """
    # 从该月第一天开始查找
    first_day = pd.Timestamp(year=year, month=month, day=1)
    first_weekday = first_day.weekday()
    
    # 计算第一个目标星期几的日期
    if weekday >= first_weekday:
        days_until_target = weekday - first_weekday
    else:
        days_until_target = 7 - (first_weekday - weekday)
    
    # 第N个目标星期几
    target_day = 1 + days_until_target + (n - 1) * 7
    
    return pd.Timestamp(year=year, month=month, day=target_day)


def get_last_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    """
    获取某月的最后一个星期几。
    
    Args:
        year: 年份
        month: 月份
        weekday: 星期几（0=周一, 1=周二, ..., 6=周日）
    
    Returns:
        对应的日期 (pd.Timestamp)
    """
    # 从该月最后一天往前查找
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    last_date = pd.Timestamp(year=year, month=month, day=last_day)
    last_weekday = last_date.weekday()
    
    # 计算最后一个目标星期几的日期
    if last_weekday >= weekday:
        days_back = last_weekday - weekday
    else:
        days_back = 7 - (weekday - last_weekday)
    
    target_day = last_day - days_back
    
    return pd.Timestamp(year=year, month=month, day=target_day)
