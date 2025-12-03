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
from typing import Protocol, Dict, Any


class HistoricalDataProvider(Protocol):
    """
    历史数据提供者接口（数据模块标准接口）
    
    设计目的：
    - 解耦业务逻辑与数据来源
    - 支持模拟数据（当前）和真实数据（未来）无缝切换
    - 为core_bak_refactored/core/data模块集成预留标准接口
    """
    
    def get_index_prices(self, index_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取指数价格数据
        
        Args:
            index_id: 指数代码（如'000300.SH'沪深300）
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
        
        Returns:
            DataFrame with columns: ['date', 'close', 'volume']
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
    
    def get_stock_prices(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取个股价格数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            DataFrame with columns: ['date', 'close', 'volume']
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
