"""
共享数据模型（业务层）

职责：
- 定义标准化的数据模型类
- 提供跨模块共享的数据结构
"""

from typing import Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass


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


# 导出所有数据模型
__all__ = [
    'MarketData',
]