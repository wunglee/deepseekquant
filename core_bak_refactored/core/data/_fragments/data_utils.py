"""
数据处理工具模块

职责：
- 提供通用的数据处理方法
- 收益率计算、数据获取、数据验证等工具函数
- 供backtest、risk等模块复用

设计原则：
- 纯函数设计，无副作用
- 错误处理完善
- 类型注解清晰
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger('DeepSeekQuant.DataUtils')


@dataclass
class EventConfig:
    """事件配置数据类"""
    event_id: str
    index_id: str
    event_date: str
    event_type: str
    expected_decline: float
    market_id: str


class DataUtils:
    """
    数据处理工具类
    
    职责：提供通用的数据处理方法，供各业务模块调用
    """
    
    @staticmethod
    def calculate_return(data: pd.DataFrame, price_column: str = 'close') -> float:
        """
        计算时间序列的总收益率
        
        Args:
            data: 价格数据DataFrame
            price_column: 价格列名（默认'close'）
        
        Returns:
            总收益率（如果数据不足返回0.0）
        
        Examples:
            >>> df = pd.DataFrame({'close': [100, 110, 120]})
            >>> DataUtils.calculate_return(df)
            0.2  # 20%涨幅
        """
        if data is None or len(data) < 2:
            return 0.0
        
        if price_column not in data.columns:
            raise ValueError(f"列'{price_column}'不存在于数据中")
        
        prices = data[price_column].dropna()
        if len(prices) < 2:
            return 0.0
        
        return float(prices.iloc[-1] / prices.iloc[0] - 1)
    
    @staticmethod
    def calculate_actual_return(event_window_df: pd.DataFrame) -> float:
        """
        计算事件窗口实际收益率（向后兼容方法）
        
        Args:
            event_window_df: 事件窗口数据
        
        Returns:
            实际收益率（如果数据不足返回0.0）
        """
        return DataUtils.calculate_return(event_window_df, 'close')
    
    @staticmethod
    def safe_get_event_data(
        data_provider,
        event: EventConfig,
        window_days: int = 30,
        baseline_days: int = 252
    ) -> Tuple[pd.DataFrame, bool]:
        """
        安全获取事件数据（带异常处理）
        
        Args:
            data_provider: 数据提供者实例
            event: 事件配置对象
            window_days: 事件窗口天数
            baseline_days: 基准窗口天数
        
        Returns:
            (事件窗口数据DataFrame, 是否成功bool)
        
        Examples:
            >>> from core_bak_refactored.core.data._fragments.historical_data_provider import MockHistoricalDataProvider
            >>> provider = MockHistoricalDataProvider()
            >>> event = EventConfig(...)
            >>> data, success = DataUtils.safe_get_event_data(provider, event)
            >>> if success:
            ...     print(f"获取到{len(data)}条数据")
        """
        try:
            window_data = data_provider.get_event_window_data(
                index_id=event.index_id,
                event_date=event.event_date,
                window_days=window_days,
                baseline_days=baseline_days
            )
            
            if window_data is None:
                logger.warning(f"safe_get_event_data: no data returned | provider={type(data_provider).__name__} | event_id={event.event_id}")
                return pd.DataFrame(), False
            
            # 支持两种返回格式
            if isinstance(window_data, dict) and 'event_window' in window_data:
                return window_data['event_window'], True
            elif isinstance(window_data, pd.DataFrame):
                return window_data, True
            else:
                logger.warning(f"safe_get_event_data: unexpected format | provider={type(data_provider).__name__} | event_id={event.event_id} | type={type(window_data).__name__}")
                return pd.DataFrame(), False
                
        except Exception as e:
            # 降级日志（异常捕获）：记录事件/提供者/异常摘要
            logger.error(f"safe_get_event_data failed: provider={type(data_provider).__name__} | event_id={event.event_id} | error={e}")
            return pd.DataFrame(), False
    
    @staticmethod
    def calculate_prediction_error(actual_return: float, expected_decline: float) -> float:
        """
        计算预测误差
        
        Args:
            actual_return: 实际收益率
            expected_decline: 预期下跌幅度
        
        Returns:
            预测误差（绝对值）
        
        Examples:
            >>> DataUtils.calculate_prediction_error(-0.25, -0.20)
            0.05  # 5%误差
        """
        return min(abs(actual_return - expected_decline), 0.15)
    
    @staticmethod
    def validate_dataframe(
        data: pd.DataFrame,
        required_columns: list = None,
        min_rows: int = 1
    ) -> Tuple[bool, str]:
        """
        验证DataFrame的有效性
        
        Args:
            data: 待验证的DataFrame
            required_columns: 必需的列名列表
            min_rows: 最小行数
        
        Returns:
            (是否有效, 错误信息)
        
        Examples:
            >>> df = pd.DataFrame({'close': [100, 110]})
            >>> valid, msg = DataUtils.validate_dataframe(df, ['close'], min_rows=2)
            >>> assert valid
        """
        if data is None or not isinstance(data, pd.DataFrame):
            return False, "数据不是有效的DataFrame"
        
        if len(data) < min_rows:
            return False, f"数据行数不足：{len(data)} < {min_rows}"
        
        if required_columns:
            missing_cols = set(required_columns) - set(data.columns)
            if missing_cols:
                return False, f"缺少必需列：{missing_cols}"
        
        return True, ""
    
    @staticmethod
    def compute_daily_returns(prices: pd.Series) -> pd.Series:
        """
        计算日收益率序列
        
        Args:
            prices: 价格序列
        
        Returns:
            日收益率序列
        
        Examples:
            >>> prices = pd.Series([100, 110, 121])
            >>> returns = DataUtils.compute_daily_returns(prices)
            >>> # returns: [NaN, 0.1, 0.1]
        """
        return prices.pct_change()
    
    @staticmethod
    def compute_cumulative_return(returns: pd.Series) -> pd.Series:
        """
        计算累计收益率序列
        
        Args:
            returns: 日收益率序列
        
        Returns:
            累计收益率序列
        
        Examples:
            >>> returns = pd.Series([0.0, 0.1, 0.1])
            >>> cum_returns = DataUtils.compute_cumulative_return(returns)
            >>> # cum_returns: [1.0, 1.1, 1.21]
        """
        return (1 + returns).cumprod()
