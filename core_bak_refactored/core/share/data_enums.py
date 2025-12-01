"""
数据相关枚举定义（共享模块）

职责：定义标准化的数据相关枚举类型
用途：替换项目中所有字符串常量，提供类型安全和自动补全
"""

from enum import Enum


class DataType(str, Enum):
    """
    数据类型枚举
    """
    OHLCV = 'ohlcv'
    DIVIDENDS = 'dividends'
    SPLITS = 'splits'
    ALL = 'all'
    
    def __str__(self) -> str:
        """支持直接字符串转换"""
        return self.value


class DataInterval(str, Enum):
    """
    数据间隔枚举
    """
    MINUTE_1 = '1m'
    MINUTE_5 = '5m'
    MINUTE_15 = '15m'
    MINUTE_30 = '30m'
    HOUR_1 = '1h'
    HOUR_4 = '4h'
    DAY_1 = '1d'
    WEEK_1 = '1wk'
    MONTH_1 = '1mo'
    
    def __str__(self) -> str:
        """支持直接字符串转换"""
        return self.value


class DataPeriod(str, Enum):
    """
    数据期间枚举
    """
    DAY_1 = '1d'
    DAY_5 = '5d'
    MONTH_1 = '1mo'
    MONTH_3 = '3mo'
    MONTH_6 = '6mo'
    YEAR_1 = '1y'
    YEAR_2 = '2y'
    YEAR_5 = '5y'
    YEAR_10 = '10y'
    YTD = 'ytd'
    MAX = 'max'
    
    def __str__(self) -> str:
        """支持直接字符串转换"""
        return self.value


class DataSource(str, Enum):
    """
    数据源枚举
    """
    YAHOO_FINANCE = 'yahoo_finance'
    ALPHA_VANTAGE = 'alpha_vantage'
    MOCK = 'mock'
    DATABASE = 'database'
    
    def __str__(self) -> str:
        """支持直接字符串转换"""
        return self.value


class DataFormat(str, Enum):
    """
    数据格式枚举
    """
    JSON = 'json'
    CSV = 'csv'
    PARQUET = 'parquet'
    PICKLE = 'pickle'
    
    def __str__(self) -> str:
        """支持直接字符串转换"""
        return self.value