"""
共享枚举定义（业务层）

职责：
- 定义标准化的业务枚举类型
- 提供跨模块共享的枚举定义
- 替换项目中所有字符串常量，提供类型安全和自动补全

从 core/data/data_fetcher.py 提取并整合
"""

from enum import Enum


class DataSourceType(Enum):
    """数据源类型枚举"""
    YAHOO_FINANCE = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    IEX_CLOUD = "iex_cloud"
    POLYGON = "polygon"
    TWELVE_DATA = "twelve_data"
    FINNHUB = "finnhub"
    TIINGO = "tiingo"
    QUANDL = "quandl"
    INTRINIO = "intrinio"
    EOD_HISTORICAL = "eod_historical"
    CUSTOM_API = "custom_api"
    DATABASE = "database"
    BROKER_API = "broker_api"


class DataFrequency(Enum):
    """数据频率枚举"""
    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


# 导出所有枚举类型
__all__ = [
    'DataSourceType',
    'DataFrequency',
]
