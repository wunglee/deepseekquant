"""
数据获取系统 - 枚举和数据模型
拆分自: core_bak/data_fetcher.py (line 35-133)
职责: 定义数据获取相关的枚举和数据类
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum
from datetime import datetime


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


@dataclass
class MarketData:
    """市场数据容器类"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: Optional[float] = None
    dividends: Optional[float] = None
    splits: Optional[float] = None
    vwap: Optional[float] = None
    trades: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    implied_volatility: Optional[float] = None
    open_interest: Optional[float] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'adj_close': self.adj_close,
            'dividends': self.dividends,
            'splits': self.splits,
            'vwap': self.vwap,
            'trades': self.trades,
            'bid': self.bid,
            'ask': self.ask,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'implied_volatility': self.implied_volatility,
            'open_interest': self.open_interest,
            'metadata': self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MarketData':
        """从字典创建"""
        return cls(
            symbol=data['symbol'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            volume=data['volume'],
            adj_close=data.get('adj_close'),
            dividends=data.get('dividends'),
            splits=data.get('splits'),
            vwap=data.get('vwap'),
            trades=data.get('trades'),
            bid=data.get('bid'),
            ask=data.get('ask'),
            bid_size=data.get('bid_size'),
            ask_size=data.get('ask_size'),
            implied_volatility=data.get('implied_volatility'),
            open_interest=data.get('open_interest'),
            metadata=data.get('metadata', {})
        )
