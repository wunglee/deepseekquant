"""
历史数据提供者协议接口

职责：
- 定义历史数据提供者的标准接口契约
- 支持多种实现（Mock/Real/自定义）的无缝切换
- 为数据模块提供统一的接口规范

设计原则：
- Protocol接口，支持鸭子类型
- 接口稳定，向后兼容
"""

import pandas as pd
from typing import Protocol, Dict, Any, List, Union, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class OHLCVRecord:
    """
    单条OHLCV数据记录
    
    数据标准：
    - date: pd.Timestamp 类型，交易日期时间
    - open: float，开盘价
    - high: float，最高价
    - low: float，最低价
    - close: float，收盘价
    - volume: float，成交量
    """
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class IntradayTickRecord:
    """
    分时Tick数据记录（1分钟级别）
    
    属性：
        time: str - 交易时间（HH:MM格式）
        price: float - 当前价格
        volume: int - 成交量（手）
        avg_price: float - 均价
    """
    time: str
    price: float
    volume: int
    avg_price: float


@dataclass
class OrderBookLevel:
    """
    盘口档位数据
    
    属性：
        price: float - 价格
        volume: int - 挂单量（手）
    """
    price: float
    volume: int


@dataclass
class TickerRecord:
    """
    成交明细记录
    
    属性：
        time: str - 成交时间（HH:MM:SS格式）
        price: float - 成交价格
        volume: int - 成交量（手）
        direction: str - 买卖方向（'buy'/'sell'）
    """
    time: str
    price: float
    volume: int
    direction: str


@dataclass
class IntradayData:
    """
    分时图完整数据结构
    
    属性：
        symbol: str - 证券代码
        name: str - 证券名称
        current_price: float - 当前价格
        yesterday_close: float - 昨收价
        change: float - 涨跌额
        change_percent: float - 涨跌幅（%）
        ticks: List[IntradayTickRecord] - 分时tick数据列表
        order_book_bids: List[OrderBookLevel] - 买盘档位（从高到低）
        order_book_asks: List[OrderBookLevel] - 卖盘档位（从低到高）
        tickers: List[TickerRecord] - 成交明细列表
        trade_date: str - 交易日期（YYYY-MM-DD）
        order_book_message: str - 盘口数据提示信息（如果为空）
        tickers_message: str - 成交明细提示信息（如果为空）
        is_tradable: bool - 是否可交易（指数为False，股票为True）
    """
    symbol: str
    name: str
    current_price: float
    yesterday_close: float
    change: float
    change_percent: float
    ticks: List[IntradayTickRecord]
    order_book_bids: List[OrderBookLevel]
    order_book_asks: List[OrderBookLevel]
    tickers: List[TickerRecord]
    trade_date: str
    order_book_message: str = ''  # 默认为空
    tickers_message: str = ''  # 默认为空
    is_tradable: bool = True  # 默认可交易


@dataclass
class PriceData:
    """
    标准价格数据结构 - 明确的属性字段
    
    属性：
        records: List[OHLCVRecord] - OHLCV数据记录列表
        symbol: str - 证券代码
        start_date: pd.Timestamp - 开始日期
        end_date: pd.Timestamp - 结束日期
        count: int - 记录数量
    """
    records: List[OHLCVRecord]
    symbol: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    count: int
    
    def __post_init__(self):
        """验证数据结构"""
        if self.records is not None:
            if not isinstance(self.records, list):
                raise ValueError("records must be a list of OHLCVRecord")
            if len(self.records) != self.count:
                raise ValueError("count must match the number of records")
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        转换为DataFrame格式（为了向后兼容）
        
        Returns:
            pd.DataFrame: 包含date, open, high, low, close, volume列的DataFrame
        """
        if not self.records:
            return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        
        data = []
        for record in self.records:
            data.append({
                'date': record.date,
                'open': record.open,
                'high': record.high,
                'low': record.low,
                'close': record.close,
                'volume': record.volume
            })
        
        return pd.DataFrame(data)
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, symbol: str = "") -> 'PriceData':
        """
        从DataFrame创建PriceData对象
        
        Args:
            df: 包含OHLCV数据的DataFrame
            symbol: 证券代码
            
        Returns:
            PriceData: 标准化的价格数据对象
        """
        required_columns = {'date', 'open', 'high', 'low', 'close', 'volume'}
        if not required_columns.issubset(set(df.columns)):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        records = []
        for _, row in df.iterrows():
            record = OHLCVRecord(
                date=pd.to_datetime(row['date']),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume'])
            )
            records.append(record)
        
        start_date = records[0].date if records else pd.Timestamp.now()
        end_date = records[-1].date if records else pd.Timestamp.now()
        
        return cls(
            records=records,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            count=len(records)
        )


class HistoricalDataProvider(Protocol):
    """
    历史数据提供者接口（数据模块标准接口）
    
    设计目的：
    - 解耦业务逻辑与数据来源
    - 支持模拟数据（当前）和真实数据（未来）无缝切换
    - 为core_bak_refactored/core/data模块集成预留标准接口
    
    数据标准：
    所有实现必须返回标准的OHLCV数据格式：
    - date: pd.Timestamp 类型，交易日期时间
    - open: float，开盘价
    - high: float，最高价
    - low: float，最低价
    - close: float，收盘价
    - volume: float，成交量
    
    注意事项：
    - 所有价格字段必须为float类型
    - 日期字段必须为pd.Timestamp类型
    - 成交量字段必须为float类型
    - 数据必须按日期升序排列
    - 不得包含缺失值（NaN）
    """

    def get_index_prices(self, index_id: str, start_date: str, end_date: str) -> PriceData:
        """
        获取指数价格数据
        
        Args:
            index_id: 指数代码（如'000300.SH'沪深300）
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
        
        Returns:
            PriceData: 包含标准OHLCV数据的结构化对象，具有明确的属性字段：
                - records: List[OHLCVRecord] - OHLCV数据记录列表
                - symbol: str - 指数代码
                - start_date: pd.Timestamp - 开始日期
                - end_date: pd.Timestamp - 结束日期
                - count: int - 记录数量
            
        数据标准：
        - date: pd.Timestamp 类型，交易日期
        - open: float，开盘价
        - high: float，最高价
        - low: float，最低价
        - close: float，收盘价
        - volume: float，成交量
            
        注意：所有实现必须返回完整的OHLCV数据，用于技术指标计算
        """
        ...

    def get_index_returns(self, index_id: str, start_date: str, end_date: str) -> pd.Series:
        """
        获取指数收益率序列
        
        Args:
            index_id: 指数代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            Series with date index and return values
        """
        ...

    def get_stock_prices(self, symbol: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> PriceData:
        """
        获取个股历史价格数据

        Args:
            symbol: 股票代码（支持市场后缀，如 '000001.SZ'）
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            PriceData: 包含标准OHLCV数据的结构化对象，具有明确的属性字段：
                - records: List[OHLCVRecord] - OHLCV数据记录列表
                - symbol: str - 股票代码
                - start_date: pd.Timestamp - 开始日期
                - end_date: pd.Timestamp - 结束日期
                - count: int - 记录数量
            
        数据标准：与 get_index_prices 相同
        """
        ...

    def get_volatility_index(self, index_id: str, start_date: str, end_date: str) -> pd.Series:
        """
        获取波动率指数（如VIX）
        
        Args:
            index_id: 波动率指数代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            Series with date index and volatility values
        """
        ...
        
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        数据质量验证报告
        
        Args:
            data: 待验证的数据
            
        Returns:
            质量报告字典，包含completeness_score、consistency_score等
        """
        ...
    
    def get_intraday_data(self, symbol: str, trade_date: Optional[str] = None) -> IntradayData:
        """
        获取分时图数据（1分钟级别）
        
        Args:
            symbol: 证券代码（如'000001.SH'上证指数）
            trade_date: 交易日期 'YYYY-MM-DD'（默认为当前日期）
        
        Returns:
            IntradayData: 包含完整分时数据的结构化对象：
                - symbol: 证券代码
                - name: 证券名称
                - current_price: 当前价格
                - yesterday_close: 昨收价
                - change: 涨跌额
                - change_percent: 涨跌幅（%）
                - ticks: List[IntradayTickRecord] - 分时tick数据
                - order_book_bids: List[OrderBookLevel] - 买盘10档
                - order_book_asks: List[OrderBookLevel] - 卖盘10档
                - tickers: List[TickerRecord] - 成交明细（最近20笔）
                - trade_date: 交易日期
        
        数据标准：
        - ticks: 按时间升序排列，覆盖交易时段（09:30-11:30, 13:00-15:00）
        - order_book: 买盘从高到低，卖盘从低到高
        - tickers: 按时间降序排列（最新的在前）
        
        注意：实现类可以返回实时数据或历史分时数据
        """
        ...