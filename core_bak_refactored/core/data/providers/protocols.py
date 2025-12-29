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
from dataclasses import dataclass
from typing import Protocol, Dict, Any, List, Union, Optional

import pandas as pd

# 导入市场数据类型
from core_bak_refactored.core.share.market.data_types import PriceData
# 导入 TradingPhase 枚举
from core_bak_refactored.core.share.market.market_enums import TradingPhase


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
class TradeDetailRecord:
    """
    成交明细记录（逐笔成交）
    
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
        trade_records: List[TradeDetailRecord] - 成交明细列表（逐笔成交）
        trade_date: pd.Timestamp - 交易日期（YYYY-MM-DD）
        order_book_message: str - 盘口数据提示信息（如果为空）
        trade_records_message: str - 成交明细提示信息（如果为空）
        is_index: bool - 是否为指数（True=指数不可交易，False=个股可交易）
        should_poll: bool - 是否应该轮询（盘前或盘中为True）
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
    trade_records: List[TradeDetailRecord]
    trade_date: pd.Timestamp
    order_book_message: str = ''  # 默认为空
    trade_records_message: str = ''  # 默认为空
    is_index: bool = False  # 默认为个股（可交易）
    should_poll: bool = False  # 默认不轮询

    @classmethod
    def from_any(cls, data: Union['IntradayData', dict, Any]) -> Optional['IntradayData']:
        """
        从任意数据类型转换为 IntradayData 对象
        
        Args:
            data: 输入数据，可以是：
                - IntradayData 对象：直接返回
                - dict 字典：使用 **kwargs 构造
                - 其他类型：返回 None 并记录警告
        
        Returns:
            IntradayData 对象或 None（如果转换失败）
        
        示例：
            >>> data_dict = {'symbol': '000001.SH', 'name': '上证指数', ...}
            >>> intraday = IntradayData.from_any(data_dict)
            >>> 
            >>> intraday_obj = IntradayData(...)
            >>> same_obj = IntradayData.from_any(intraday_obj)  # 直接返回
        """
        if isinstance(data, cls):
            # 已经是 IntradayData 对象，直接返回
            return data
        elif isinstance(data, dict):
            # 字典类型，尝试构造对象
            try:
                return cls(**data)
            except (TypeError, ValueError) as e:
                # 记录错误但不抛出异常
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"从字典构造 IntradayData 失败: {e}")
                return None
        else:
            # 不支持的类型
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"不支持的数据类型转换为 IntradayData: {type(data)}")
            return None

    @classmethod
    def from_akshare_df(cls, df: pd.DataFrame, symbol: str, trade_date: pd.Timestamp,
                        interpolate_func=None) -> 'IntradayData':
        """
        从 AKShare 返回的 DataFrame 构建 IntradayData 对象
        
        AKShare DataFrame 格式：时间,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
        
        Args:
            df: AKShare 返回的 DataFrame
            symbol: 证券代码
            trade_date: 交易日期
            interpolate_func: 插值函数（可选，用于将1分钟数据插值为5秒数据）
        
        Returns:
            IntradayData 对象（不包含盘口和成交明细）
        
        注意：
        - 本方法仅负责 DataFrame 到 IntradayData 的数据转换
        - 盘口和成交明细为空，由调用方设置
        """
        from core_bak_refactored.core.share.market.market_utils import MarketUtils
        import logging
        logger = logging.getLogger(__name__)

        # 获取股票名称
        name_map = {
            '000001.SH': '上证指数',
            '000300.SH': '沪深300',
            '399001.SZ': '深证成指',
            '399006.SZ': '创业板指'
        }
        name = name_map.get(symbol, symbol)

        # 处理空DataFrame（集合竞价时段等）
        if df.empty:
            yesterday_close = 0.0
            ticks = []
            current_price = 0.0
            change = 0.0
            change_percent = 0.0
        else:
            # 获取昨收价（从第一条数据推算）
            if '涨跌额' in df.columns and '收盘' in df.columns:
                first_close = float(df.iloc[0]['收盘'])
                first_change = float(df.iloc[0].get('涨跌额', 0))
                yesterday_close = first_close - first_change
            else:
                yesterday_close = float(df.iloc[0].get('收盘', 0)) * 0.99  # 估算

            # 构建 ticks
            ticks = []
            total_volume = 0
            total_amount = 0

            for _, row in df.iterrows():
                time_str = str(row['时间']).split(' ')[-1]  # 提取时间部分 HH:MM:SS
                # 确保时间格式为 HH:MM:SS
                if len(time_str) == 5:  # HH:MM
                    time_str += ':00'
                elif len(time_str) > 8:  # 可能包含毫秒，截断到秒
                    time_str = time_str[:8]

                price = float(row.get('收盘', row.get('最新价', 0)))
                volume = int(row.get('成交量', 0))

                total_volume += volume
                total_amount += price * volume
                avg_price = total_amount / total_volume if total_volume > 0 else price

                ticks.append(IntradayTickRecord(
                    time=time_str,
                    price=round(price, 2),
                    volume=volume,
                    avg_price=round(avg_price, 2)
                ))

            # 插值：将1分钟数据插值为5秒数据（如果提供了插值函数）
            if interpolate_func and ticks:
                ticks = interpolate_func(ticks)
                logger.info(f"📊 插值完成: {len(ticks)}个5秒粒度数据点")

            # 当前价格
            current_price = ticks[-1].price if ticks else yesterday_close
            change = current_price - yesterday_close
            change_percent = (change / yesterday_close * 100) if yesterday_close > 0 else 0

        # 判断是否为指数
        is_index = MarketUtils.is_index(symbol)

        return cls(
            symbol=symbol,
            name=name,
            current_price=round(current_price, 2),
            yesterday_close=round(yesterday_close, 2),
            change=round(change, 2),
            change_percent=round(change_percent, 2),
            ticks=ticks,
            order_book_bids=[],  # 由调用方设置
            order_book_asks=[],  # 由调用方设置
            trade_records=[],  # 由调用方设置
            trade_date=trade_date,
            order_book_message='',  # 由调用方设置
            trade_records_message='',  # 由调用方设置
            is_index=is_index,
            should_poll=False
        )


@dataclass
class TickRange:
    """
    Tick 数据时间范围
    
    属性：
        start_time: pd.Timestamp - 开始时间（包含）
        end_time: pd.Timestamp - 结束时间（包含）
        period_seconds: int - 时间粒度（秒），默认5秒
    
    示例：
        >>> # 获取 09:30 到 10:00 的分时数据，5秒粒度
        >>> tick_range = TickRange(
        ...     start_time=pd.Timestamp('2025-12-14 09:30:00'),
        ...     end_time=pd.Timestamp('2025-12-14 10:00:00'),
        ...     period_seconds=5
        ... )
    """
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    period_seconds: int = 5  # 默认5秒粒度

    def get_tick_count(self) -> int:
        """
        计算时间范围内的tick数量
        
        Returns:
            tick数量
        """
        total_seconds = int((self.end_time - self.start_time).total_seconds())
        return (total_seconds // self.period_seconds) + 1

    @classmethod
    def from_trading_phase(cls, trading_phase: 'TradingPhase', trade_date: pd.Timestamp,
                          current_time: Optional[pd.Timestamp] = None) -> 'TickRange':
        """
        根据交易时段创建 TickRange
        
        Args:
            trading_phase: 交易时段枚举
            trade_date: 交易日期
            current_time: 当前时间，如果为None则使用系统时间
        
        Returns:
            TickRange 对象
        """
        if current_time is None:
            current_time = pd.Timestamp.now()

        # 确保 trade_date 是 pd.Timestamp 类型
        if isinstance(trade_date, str):
            trade_date = pd.to_datetime(trade_date)

        # 提取日期部分并格式化为字符串用于构造时间
        trade_date_str = trade_date.strftime('%Y-%m-%d')

        if trading_phase.value == 'after_close':
            # 盘后：返回全天数据 09:30-15:00
            start_time = pd.Timestamp(f"{trade_date_str} 09:30:00")
            end_time = pd.Timestamp(f"{trade_date_str} 15:00:00")
        elif trading_phase.value == 'before_open':
            # 盘前：返回空范围
            start_time = pd.Timestamp(f"{trade_date_str} 09:30:00")
            end_time = start_time
        else:  # trading
            # 盘中：返回开盘至当前时间
            start_time = pd.Timestamp(f"{trade_date_str} 09:30:00")
            # 确保不超过当前时间和收盘时间
            end_time = min(
                current_time,
                pd.Timestamp(f"{trade_date_str} 15:00:00")
            )

        return cls(start_time=start_time, end_time=end_time, period_seconds=5)


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

    def get_index_prices(self, index_id: str,
                         start_date: pd.Timestamp,
                         end_date: pd.Timestamp,
                         current_time: pd.Timestamp,
                         period: str = 'daily') -> PriceData:
        """
        获取指数价格数据
        
        Args:
            index_id: 指数代码（如'000300.SH'沪深300）
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            current_time:当前时间
            period:周期
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

    def get_index_returns(self, index_id: str,
                          start_date: pd.Timestamp,
                          end_date: pd.Timestamp) -> pd.Series:
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

    def get_stock_prices(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp,
                         current_time: pd.Timestamp,period: str = 'daily') -> PriceData:
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

    def get_volatility_index(self, index_id: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
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

    def get_intraday_data(self, symbol: str, tick_range: TickRange = None,
                          current_time: pd.Timestamp = None) -> IntradayData:
        """
        获取分时图数据（1分钟级别）
        
        Args:
            symbol: 证券代码（如'000001.SH'上证指数）
            tick_range: 时间范围
            current_time: 当前时间（用于测试，默认使用系统时间）
        
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
                - trade_records: List[TradeDetailRecord] - 成交明细（最近20笔）
                - trade_date: 交易日期
        
        数据标准：
        - ticks: 按时间升序排列，覆盖交易时段（09:30-11:30, 13:00-15:00）
        - order_book: 买盘从高到低，卖盘从低到高
        - trade_records: 按时间降序排列（最新的在前）
        
        注意：实现类可以返回实时数据或历史分时数据
        """
        ...
