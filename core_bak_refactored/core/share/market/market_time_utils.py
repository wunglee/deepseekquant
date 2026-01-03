"""
市场时间工具类

设计原则：
1. 所有方法都接收市场本地时间（不带时区信息）
2. API层负责将UTC时间转换为市场本地时间（移除时区）
3. 本工具类内部不进行时区转换，只处理本地时间逻辑
4. 所有判断（交易时段、节假日、K线日期等）都基于传入的本地时间

使用示例：
    # ✅ 正确用法：使用 MarketTimeUtils.get_market_time_now 获取不带时区的市场本地时间
    >>> market_local_time = MarketTimeUtils.get_market_time_now('000001.SH')
    >>> phase = MarketTimeUtils.determine_trading_phase(MarketCode.CN, market_local_time)
"""

import logging
from datetime import time as dt_time
import pandas as pd
from numpy.f2py.auxfuncs import throw_error

from core_bak_refactored.core.share.market.market_enums import MarketCode, TradingPhase

logger = logging.getLogger(__name__)


class MarketTimeUtils:
    """市场时间工具类
    
    核心设计：
    - 接收市场本地时间（不带时区）
    - 不进行时区转换
    - 所有判断基于本地时间
    """
    
    @staticmethod
    def _get_market_timezone(market: MarketCode) -> pd.Timestamp.tzinfo:
        """
        获取市场时区
        
        Args:
            market: 市场代码
        
        Returns:
            市场时区对象
        """
        from core_bak_refactored.core.share.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        trading_hours = config_manager.get_trading_hours(market.value)
        
        if not trading_hours:
            logger.warning(f"未找到市场 {market.value} 的配置，使用UTC时区")
            return pd.Timestamp.now(tz='UTC').tz
        
        timezone_str = trading_hours.get('timezone', 'UTC')
        return pd.Timestamp.now(tz=timezone_str).tz
    
    @staticmethod
    def convert_client_time_to_utc(date_str: str, client_timezone: str) -> pd.Timestamp:
        """
        将浏览器本地时间转换为UTC时间
        
        Args:
            date_str: 浏览器本地时间字符串（如 '2024-01-15' 或 '2024-01-15 10:30:00'）
            client_timezone: 浏览器时区（如 'Asia/Shanghai', 'America/New_York'）
        
        Returns:
            pd.Timestamp: UTC时间（带时区信息）
        
        Raises:
            ValueError: 如果时区无效或日期格式错误
        
        Examples:
            >>> MarketTimeUtils.convert_client_time_to_utc('2024-01-15', 'Asia/Shanghai')
            Timestamp('2024-01-14 16:00:00+0000', tz='UTC')
            
            >>> MarketTimeUtils.convert_client_time_to_utc('2024-01-15 10:30', 'America/New_York')
            Timestamp('2024-01-15 15:30:00+0000', tz='UTC')
        """
        try:
            # 解析浏览器时区
            local_dt = pd.to_datetime(date_str).tz_localize(client_timezone)
            
            # 转换为UTC时间，确保返回 pd.Timestamp 类型
            utc_time = pd.Timestamp(local_dt.tz_convert('UTC'))
            
            return utc_time
            
        except Exception as e:
            if "Unknown timezone" in str(e):
                raise ValueError(f"无效的时区: {client_timezone}") from e
            else:
                raise ValueError(f"日期转换失败: {str(e)}") from e
    
    @staticmethod
    def convert_client_time_to_market_time(
        date_str: str, 
        client_timezone: str, 
        market: MarketCode
    ) -> pd.Timestamp:
        """
        将浏览器本地时间转换为目标市场的本地时间（不带时区）
        
        Args:
            date_str: 浏览器本地时间字符串（如 '2024-01-15' 或 '2024-01-15 10:30:00'）
            client_timezone: 浏览器时区（如 'Asia/Shanghai', 'America/New_York'）
            market: 目标市场代码
        
        Returns:
            pd.Timestamp: 市场本地时间（不带时区信息）
        
        Raises:
            ValueError: 如果时区无效或日期格式错误
        
        Examples:
            >>> # 上海浏览器的2024-01-15 -> A股市场时间
            >>> MarketTimeUtils.convert_client_time_to_market_time(
            ...     '2024-01-15', 'Asia/Shanghai', MarketCode.CN
            ... )
            Timestamp('2024-01-15 00:00:00')  # 不带时区
            
            >>> # 纽约浏览器的2024-01-15 -> A股市场时间  
            >>> MarketTimeUtils.convert_client_time_to_market_time(
            ...     '2024-01-15', 'America/New_York', MarketCode.CN
            ... )
            Timestamp('2024-01-15 13:00:00')  # 不带时区
        """
        # 1. 浏览器本地时间 -> UTC时间
        utc_time = MarketTimeUtils.convert_client_time_to_utc(date_str, client_timezone)
        
        # 2. 获取目标市场时区
        market_tz = MarketTimeUtils._get_market_timezone(market)
        
        # 3. UTC时间 -> 市场本地时间（移除时区）
        market_time_with_tz = utc_time.tz_convert(market_tz)
        market_time = market_time_with_tz.replace(tzinfo=None)
        
        return market_time
    
    @staticmethod
    def get_market_time_now(symbol: str) -> pd.Timestamp:
        """
        获取指定symbol所属市场的当前本地时间（不带时区信息）
        
        工作流程：
        1. 获取UTC时间
        2. 推断symbol所属市场
        3. 获取市场时区
        4. 转换为市场本地时间（移除时区信息）
        
        Args:
            symbol: 证券代码（如 '000001.SH', 'AAPL', '600000.SS'）
        
        Returns:
            pd.Timestamp: 市场本地时间（不带时区信息）
        
        Examples:
            >>> # 获取A股市场当前时间
            >>> MarketTimeUtils.get_market_time_now('000001.SH')
            Timestamp('2024-01-15 14:30:00')
            
            >>> # 获取美股市场当前时间
            >>> MarketTimeUtils.get_market_time_now('AAPL')
            Timestamp('2024-01-15 10:30:00')
        """
        from core_bak_refactored.core.share.market.market_utils import MarketUtils
        
        # 1. 获取UTC时间
        utc_now = pd.Timestamp.now(tz='UTC')
        
        # 2. 推断市场代码
        market_code = MarketUtils.infer_market_from_symbol(symbol)
        
        # 3. 获取市场时区
        market_tz = MarketTimeUtils._get_market_timezone(market_code)
        
        # 4. 转换为市场本地时间（移除时区信息）
        market_time = utc_now.tz_convert(market_tz).replace(tzinfo=None)
        
        return market_time

    @staticmethod
    def determine_trading_phase(market: MarketCode, market_local_time: pd.Timestamp) -> TradingPhase:
        """
        判断市场交易时段
        
        Args:
            market: 市场代码
            market_local_time: 市场本地时间（不带时区信息）
        
        Returns:
            TradingPhase: 交易时段枚举
        
        Examples:
            >>> local_time = pd.Timestamp('2024-01-15 10:00')  # 市场本地时间10:00
            >>> phase = MarketTimeUtils.determine_trading_phase(MarketCode.CN, local_time)
            >>> print(phase)
            TradingPhase.TRADING
        """
        from core_bak_refactored.core.share.config_manager import ConfigManager
        from core_bak_refactored.core.share.market.trading_calendar_service import get_trading_calendar_service
        
        # 确保market是MarketCode枚举
        if not isinstance(market, MarketCode):
            market = MarketCode.parse(market)
        
        # 提取本地日期和时间
        market_local_date = market_local_time.date()
        current_time = market_local_time.time()
        
        logger.debug(f"🌍 [{market.value}] 本地时间: {market_local_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # 获取市场配置
        config_manager = ConfigManager()
        trading_hours = config_manager.get_trading_hours(market.value)
        
        if not trading_hours:
            logger.warning(f"未找到市场 {market.value} 的交易时段配置")
            return TradingPhase.AFTER_CLOSE
        
        # 判断是否为交易日（基于市场本地日期）
        calendar_service = get_trading_calendar_service()
        market_date_ts = pd.Timestamp(market_local_date)
        is_trading_day = calendar_service.is_trading_day(market, market_date_ts)
        
        if not is_trading_day:
            logger.debug(f"📅 [{market.value}] {market_local_date} 不是交易日")
            return TradingPhase.AFTER_CLOSE
        
        # 解析配置中的时间（都是市场本地时间）
        def parse_local_time(time_str: str) -> dt_time:
            """解析时间字符串为time对象"""
            parts = time_str.split(':')
            return dt_time(int(parts[0]), int(parts[1]))
        
        open_time = parse_local_time(trading_hours['open'])
        close_time = parse_local_time(trading_hours['close'])
        
        # 集合竞价开始时间（开盘前30分钟）
        call_auction_start = (pd.Timestamp.combine(market_local_date, open_time) - pd.Timedelta(minutes=30)).time()

        # 判断交易时段（基于市场本地时间）
        if call_auction_start <= current_time < open_time:
            logger.debug(f"⏰ [{market.value}] 集合竞价时段")
            return TradingPhase.BEFORE_OPEN
        
        # 处理午休
        has_lunch_break = trading_hours.get('has_lunch_break', False)
        if has_lunch_break:
            lunch_start_str = trading_hours.get('lunch_start')
            lunch_end_str = trading_hours.get('lunch_end')
            
            if lunch_start_str and lunch_end_str:
                lunch_start = parse_local_time(lunch_start_str)
                lunch_end = parse_local_time(lunch_end_str)
                
                if open_time <= current_time < lunch_start:
                    logger.debug(f"⏰ [{market.value}] 上午交易时段")
                    return TradingPhase.TRADING
                elif lunch_start <= current_time < lunch_end:
                    logger.debug(f"⏰ [{market.value}] 午休时段")
                    return TradingPhase.NOON_BREAK
                elif lunch_end <= current_time <= close_time:
                    logger.debug(f"⏰ [{market.value}] 下午交易时段")
                    return TradingPhase.TRADING
                else:
                    logger.debug(f"⏰ [{market.value}] 收盘后")
                    return TradingPhase.AFTER_CLOSE
            else:
                logger.warning(f"市场 {market.value} 配置了午休但未提供午休时间")
        
        # 无午休市场或午休配置无效
        if open_time <= current_time <= close_time:
            logger.debug(f"⏰ [{market.value}] 交易时段")
            return TradingPhase.TRADING
        else:
            logger.debug(f"⏰ [{market.value}] 收盘后")
            return TradingPhase.AFTER_CLOSE
    
    @staticmethod
    def get_last_trade_date(market: MarketCode, market_local_time: pd.Timestamp) -> pd.Timestamp:
        """
        获取最后一个交易日（用于分时数据显示）
        
        规则（基于市场本地时间判断）：
        1. 盘前时段：显示前一个交易日的分时数据
        2. 盘中/盘后：显示当天的分时数据（如果是交易日）
        3. 非交易日：显示前一个交易日的分时数据
        
        Args:
            market: 市场代码
            market_local_time: 市场本地时间（不带时区信息）
        
        Returns:
            最后一个交易日（市场本地日期，naive类型）
        
        Examples:
            >>> local_time = pd.Timestamp('2024-01-15 08:00')  # 市场本地时间08:00（盘前）
            >>> date = MarketTimeUtils.get_last_trade_date(MarketCode.CN, local_time)
            >>> # 返回前一交易日
        """
        from core_bak_refactored.core.share.market.trading_calendar_service import get_trading_calendar_service
        from core_bak_refactored.core.share.config_manager import ConfigManager

        # 如果传入的是带时区的时间，移除时区信息
        if market_local_time.tzinfo is not None:
            throw_error("传入的时间必须为naive类型")

        # 提取本地日期
        market_local_date =pd.Timestamp(market_local_time.date())

        # 判断交易时段
        trading_phase = MarketTimeUtils.determine_trading_phase(market, market_local_time)
        
        calendar_service = get_trading_calendar_service()
        
        if trading_phase == TradingPhase.BEFORE_OPEN:
            # 集合竞价时段：返回前一个交易日
            prev_trading_day = calendar_service.get_previous_trading_day(market, market_local_date)
            if prev_trading_day:
                logger.debug(f"📊 [{market.value}] 盘前，返回前一交易日: {prev_trading_day.date()}")
                return prev_trading_day
            else:
                # 降级处理
                if market_local_date.weekday() == 0:  # 周一
                    return market_local_date - pd.Timedelta(days=3)
                else:
                    return market_local_date - pd.Timedelta(days=1)
        
        elif trading_phase == TradingPhase.AFTER_CLOSE:
            # 收盘后：需要区分是否在今天集合竞价之前
            config_manager = ConfigManager()
            trading_hours = config_manager.get_trading_hours(market.value)
            
            if trading_hours:
                # 解析开盘时间
                parts = trading_hours['open'].split(':')
                open_time = dt_time(int(parts[0]), int(parts[1]))
                call_auction_start = (pd.Timestamp.combine(market_local_date.date(), open_time) - pd.Timedelta(minutes=30)).time()
                
                current_time = market_local_time.time()
                
                if current_time < call_auction_start:
                    # 在今天集合竞价之前（例如早上08:00）-> 返回前一交易日
                    prev_trading_day = calendar_service.get_previous_trading_day(market, market_local_date)
                    if prev_trading_day:
                        logger.debug(f"📊 [{market.value}] 集合竞价前，返回前一交易日: {prev_trading_day.date()}")
                        return prev_trading_day
                    else:
                        if market_local_date.weekday() == 0:
                            return market_local_date - pd.Timedelta(days=3)
                        else:
                            return market_local_date - pd.Timedelta(days=1)
            
            # 收盘后：判断当天是否为交易日
            is_trading_day = calendar_service.is_trading_day(market, market_local_date)
            
            if is_trading_day:
                logger.debug(f"📊 [{market.value}] 收盘后，返回当天: {market_local_date.date()}")
                return market_local_date
            else:
                # 非交易日：返回前一个交易日
                prev_trading_day = calendar_service.get_previous_trading_day(market, market_local_date)
                if prev_trading_day:
                    logger.debug(f"📊 [{market.value}] 非交易日，返回前一交易日: {prev_trading_day.date()}")
                    return prev_trading_day
                else:
                    weekday = market_local_date.weekday()
                    if weekday >= 5:  # 周末
                        days_to_subtract = weekday - 4
                        return market_local_date - pd.Timedelta(days=days_to_subtract)
                    else:
                        for i in range(1, 8):
                            prev_date = market_local_date - pd.Timedelta(days=i)
                            if prev_date.weekday() < 5:
                                return prev_date
                        return market_local_date - pd.Timedelta(days=1)
        
        else:
            # 盘中/午休：判断当天是否为交易日
            is_trading_day = calendar_service.is_trading_day(market, market_local_date)
            
            if is_trading_day:
                logger.debug(f"📊 [{market.value}] 盘中/午休，返回当天: {market_local_date.date()}")
                return market_local_date
            else:
                # 非交易日：返回前一个交易日
                prev_trading_day = calendar_service.get_previous_trading_day(market, market_local_date)
                if prev_trading_day:
                    logger.debug(f"📊 [{market.value}] 非交易日，返回前一交易日: {prev_trading_day.date()}")
                    return prev_trading_day
                else:
                    weekday = market_local_date.weekday()
                    if weekday >= 5:
                        days_to_subtract = weekday - 4
                        return market_local_date - pd.Timedelta(days=days_to_subtract)
                    else:
                        for i in range(1, 8):
                            prev_date = market_local_date - pd.Timedelta(days=i)
                            if prev_date.weekday() < 5:
                                return prev_date
                        return market_local_date - pd.Timedelta(days=1)
