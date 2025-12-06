"""
数据模块枚举定义

职责：
- 定义数据模块专属的枚举类型
- 数据源、频率、类型、间隔、周期、格式等
"""

from enum import Enum


class DataSourceType(Enum):
    """
    数据源类型枚举（生产环境）
    
    支持的真实数据源：
    - YAHOO_FINANCE: Yahoo Finance（全球市场）
    - ALPHA_VANTAGE: Alpha Vantage API
    - IEX_CLOUD: IEX Cloud API
    - POLYGON: Polygon.io API
    - TWELVE_DATA: Twelve Data API
    - FINNHUB: Finnhub API
    - TIINGO: Tiingo API
    - QUANDL: Quandl/NASDAQ Data Link
    - INTRINIO: Intrinio API
    - EOD_HISTORICAL: EOD Historical Data API
    - CUSTOM_API: 自定义API
    - DATABASE: 数据库
    - BROKER_API: 券商API
    
    注：Mock数据源已移至测试模块，仅用于单元测试
    """
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


class DataType(str, Enum):
    """数据类型枚举"""
    OHLCV = 'ohlcv'
    DIVIDENDS = 'dividends'
    SPLITS = 'splits'
    ALL = 'all'
    
    def __str__(self) -> str:
        return self.value


class DataInterval(str, Enum):
    """数据间隔枚举"""
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
        return self.value


class DataPeriod(str, Enum):
    """数据期间枚举"""
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
        return self.value


class DataFormat(str, Enum):
    """数据格式枚举"""
    JSON = 'json'
    CSV = 'csv'
    PARQUET = 'parquet'
    PICKLE = 'pickle'
    
    def __str__(self) -> str:
        return self.value


class DataQualityIssueType(str, Enum):
    """数据质量问题类型枚举"""
    MISSING_DATA = 'missing_data'
    DUPLICATE_DATA = 'duplicate_data'
    OUTLIER = 'outlier'
    INCONSISTENCY = 'inconsistency'
    TIMELINESS = 'timeliness'
    COMPLETENESS = 'completeness'
    ACCURACY = 'accuracy'
    
    def __str__(self) -> str:
        return self.value


__all__ = [
    'DataSourceType',
    'DataFrequency',
    'DataType',
    'DataInterval',
    'DataPeriod',
    'DataFormat',
    'DataQualityIssueType',
]
