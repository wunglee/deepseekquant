"""
交易日历服务

职责：
1. 基于 pandas_market_calendars 提供多市场交易日历
2. 判断指定日期是否为交易日
3. 计算两个日期之间的交易日数
4. 判断两个日期是否为连续交易日

支持市场：
- CN (中国A股)
- US (美国股市)
- HK (香港股市)
- JP (日本股市)
- EU (欧洲股市)
- SG (新加坡股市)
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Union
import pandas as pd

from core_bak_refactored.core.share.market.market_enums import MarketCode

logger = logging.getLogger(__name__)

# 延迟导入，避免启动时错误
_mcal = None


def _get_mcal():
    """延迟导入 pandas_market_calendars"""
    global _mcal
    if _mcal is None:
        try:
            import pandas_market_calendars as mcal
            _mcal = mcal
        except ImportError:
            logger.warning("pandas_market_calendars 未安装，交易日历功能将降级为简单周末判断")
            _mcal = False  # 标记为不可用
    return _mcal


class TradingCalendarService:
    """交易日历服务"""
    
    # 市场代码映射到 pandas_market_calendars 的交易所代码
    MARKET_CALENDAR_MAP = {
        'CN': 'SSE',      # 上海证券交易所 (Shanghai Stock Exchange)
        'US': 'NYSE',     # 纽约证券交易所 (New York Stock Exchange)
        'HK': 'HKEX',     # 香港交易所 (Hong Kong Stock Exchange)
        'JP': 'JPX',      # 日本交易所集团 (Japan Exchange Group)
        'EU': 'LSE',      # 伦敦证券交易所 (London Stock Exchange) 代表欧洲
        'SG': 'SGX',      # 新加坡交易所 (Singapore Exchange)
    }
    
    def __init__(self):
        """初始化交易日历服务"""
        self._calendars = {}
        self._cache = {}  # 缓存交易日判断结果
        mcal = _get_mcal()
        
        if mcal and mcal is not False:
            logger.info("✅ 交易日历服务初始化: 使用 pandas_market_calendars")
            self._available = True
        else:
            logger.warning("⚠️ 交易日历服务降级: pandas_market_calendars 不可用，仅支持周末判断")
            self._available = False
    
    def _get_calendar(self, market_code: Union[str, MarketCode]):
        """
                获取指定市场的交易日历
        
        Args:
            market_code: 市场代码枚举或字符串 (MarketCode.CN 或 'CN')
        
        Returns:
            交易日历对象，如果不可用则返回None
        """
        if not self._available:
            return None
        
        # 支持 MarketCode 枚举和字符串
        market_code_str = str(market_code) if isinstance(market_code, MarketCode) else market_code
        
        if market_code_str not in self._calendars:
            mcal = _get_mcal()
            exchange_code = self.MARKET_CALENDAR_MAP.get(market_code_str)
            
            if not exchange_code:
                logger.warning(f"未知市场代码: {market_code_str}，回退到NYSE")
                exchange_code = 'NYSE'
            
            try:
                calendar = mcal.get_calendar(exchange_code)
                self._calendars[market_code_str] = calendar
                logger.debug(f"加载交易日历: {market_code_str} -> {exchange_code}")
            except Exception as e:
                logger.error(f"加载交易日历失败 ({market_code_str}): {e}")
                return None
        
        return self._calendars.get(market_code_str)
    
    def is_trading_day(self, market_code: Union[str, MarketCode], date: datetime) -> bool:
        """
        判断指定日期是否为交易日
        
        Args:
            market_code: 市场代码枚举或字符串 (MarketCode.CN 或 'CN')
            date: 日期
        
        Returns:
            bool: True表示是交易日，False表示非交易日
        
        Examples:
            >>> service = TradingCalendarService()
            >>> service.is_trading_day(MarketCode.CN, datetime(2024, 10, 1))  # 国庆节
            False
            >>> service.is_trading_day('CN', datetime(2024, 10, 8))  # 工作日
            True
        """
        # 支持 MarketCode 枚举和字符串
        market_code_str = str(market_code) if isinstance(market_code, MarketCode) else market_code
        
        # 缓存键
        cache_key = f"{market_code_str}_{date.strftime('%Y-%m-%d')}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 降级模式：仅判断周末
        if not self._available:
            result = date.weekday() < 5  # 周一到周五
            self._cache[cache_key] = result
            return result
        
        # 使用交易日历
        calendar = self._get_calendar(market_code_str)
        if calendar is None:
            # 回退到周末判断
            result = date.weekday() < 5
            self._cache[cache_key] = result
            return result
        
        try:
            # 获取该日期所在月份的交易日
            year = date.year
            month = date.month
            schedule = calendar.schedule(
                start_date=f'{year}-{month:02d}-01',
                end_date=f'{year}-{month:02d}-{pd.Timestamp(year, month, 1).days_in_month}'
            )
            
            # 检查日期是否在交易日列表中
            date_str = pd.Timestamp(date).normalize()
            result = date_str in schedule.index
            
            self._cache[cache_key] = result
            return result
        except Exception as e:
            logger.warning(f"交易日判断失败 ({market_code_str}, {date}): {e}，回退到周末判断")
            result = date.weekday() < 5
            self._cache[cache_key] = result
            return result
    
    def get_trading_days_between(self, market_code: Union[str, MarketCode], start_date: datetime, 
                                 end_date: datetime) -> List[datetime]:
        """
        获取两个日期之间的所有交易日
        
        Args:
            market_code: 市场代码枚举或字符串
            start_date: 起始日期（包含）
            end_date: 结束日期（包含）
        
        Returns:
            交易日列表
        
        Examples:
            >>> service = TradingCalendarService()
            >>> days = service.get_trading_days_between(MarketCode.CN, 
            ...     datetime(2024, 9, 30), datetime(2024, 10, 8))
            >>> len(days)  # 跳过国庆假期
            2  # 9月30日 和 10月8日
        """
        # 支持 MarketCode 枚举和字符串
        market_code_str = str(market_code) if isinstance(market_code, MarketCode) else market_code
        # 降级模式：仅排除周末
        if not self._available:
            trading_days = []
            current = start_date
            while current <= end_date:
                if current.weekday() < 5:
                    trading_days.append(current)
                current += timedelta(days=1)
            return trading_days
        
        # 使用交易日历
        calendar = self._get_calendar(market_code_str)
        if calendar is None:
            # 回退到周末判断
            trading_days = []
            current = start_date
            while current <= end_date:
                if current.weekday() < 5:
                    trading_days.append(current)
                current += timedelta(days=1)
            return trading_days
        
        try:
            schedule = calendar.schedule(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # 转换为 datetime 列表
            trading_days = [dt.to_pydatetime() for dt in schedule.index]
            return trading_days
        except Exception as e:
            logger.warning(f"获取交易日列表失败 ({market_code_str}): {e}，回退到周末判断")
            trading_days = []
            current = start_date
            while current <= end_date:
                if current.weekday() < 5:
                    trading_days.append(current)
                current += timedelta(days=1)
            return trading_days
    
    def is_consecutive_trading_days(self, market_code: Union[str, MarketCode], date1: datetime, 
                                    date2: datetime) -> bool:
        """
        判断两个日期是否为连续交易日（中间没有其他交易日）
        
        Args:
            market_code: 市场代码枚举或字符串
            date1: 第一个日期（应早于date2）
            date2: 第二个日期
        
        Returns:
            bool: True表示连续，False表示不连续
        
        Examples:
            >>> service = TradingCalendarService()
            >>> # 周五 -> 下周一（连续）
            >>> service.is_consecutive_trading_days(MarketCode.CN,
            ...     datetime(2024, 1, 5), datetime(2024, 1, 8))
            True
            >>> # 国庆前 -> 国庆后（不连续，中间有假期）
            >>> service.is_consecutive_trading_days(MarketCode.CN,
            ...     datetime(2024, 9, 30), datetime(2024, 10, 8))
            True  # 实际上是连续的（因为中间都是假期）
        """
        if date1 >= date2:
            return False
        
        # 获取两个日期之间的所有交易日
        trading_days = self.get_trading_days_between(market_code, date1, date2)
        
        # 连续交易日：只有date1和date2两个交易日
        if len(trading_days) == 2 and trading_days[0].date() == date1.date() and trading_days[1].date() == date2.date():
            return True
        
        return False
    
    def get_next_trading_day(self, market_code: Union[str, MarketCode], date: datetime) -> Optional[datetime]:
        """
        获取下一个交易日
        
        Args:
            market_code: 市场代码枚举或字符串
            date: 当前日期
        
        Returns:
            下一个交易日，如果未来30天内没有则返回None
        """
        # 搜索未来30天
        end_date = date + timedelta(days=30)
        trading_days = self.get_trading_days_between(market_code, date + timedelta(days=1), end_date)
        
        return trading_days[0] if trading_days else None
    
    def get_previous_trading_day(self, market_code: Union[str, MarketCode], date: datetime) -> Optional[datetime]:
        """
        获取上一个交易日
        
        Args:
            market_code: 市场代码枚举或字符串
            date: 当前日期
        
        Returns:
            上一个交易日，如果过去30天内没有则返回None
        """
        # 搜索过去30天
        start_date = date - timedelta(days=30)
        trading_days = self.get_trading_days_between(market_code, start_date, date - timedelta(days=1))
        
        return trading_days[-1] if trading_days else None
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("交易日历缓存已清空")


# 全局单例
_service_instance = None


def get_trading_calendar_service() -> TradingCalendarService:
    """获取交易日历服务单例"""
    global _service_instance
    if _service_instance is None:
        _service_instance = TradingCalendarService()
    return _service_instance
