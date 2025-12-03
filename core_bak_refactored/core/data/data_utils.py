"""
数据处理工具模块

职责：
- 提供数据源的基本处理方法
- 收益率计算、时间序列转换等基础工具

设计原则：
- 纯函数设计，无副作用
- 错误处理完善
- 类型注解清晰
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('DeepSeekQuant.DataUtils')


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
        # 调用基础设施层通用简单收益计算（消除重复）
        from core_bak_refactored.infrastructure.statistical_calculators import StatisticalCalculator
        
        simple_returns = StatisticalCalculator.calculate_simple_returns(prices.values)
        # 返回 pd.Series 并保持索引对齐（第一个值为NaN）
        return pd.Series([np.nan] + list(simple_returns), index=prices.index)
    
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
    
    @staticmethod
    def compute_log_returns(prices: pd.Series) -> pd.Series:
        """
        计算对数收益率序列
        
        Args:
            prices: 价格序列
        
        Returns:
            对数收益率序列（第一个值为NaN）
        
        Examples:
            >>> prices = pd.Series([100, 110, 121])
            >>> log_returns = DataUtils.compute_log_returns(prices)
            >>> # log_returns: [NaN, 0.0953, 0.0953]
        """
        # 调用基础设施层通用对数收益计算（消除重复）
        from core_bak_refactored.infrastructure.statistical_calculators import StatisticalCalculator
        
        log_returns = StatisticalCalculator.calculate_log_returns(prices.values)
        # 返回 pd.Series 并保持索引对齐（第一个值为NaN）
        return pd.Series([np.nan] + list(log_returns), index=prices.index)
    

