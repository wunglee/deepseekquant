"""
历史回测框架 - 压力测试场景验证
从第5轮专家指导实施
职责: 历史事件回测、合成组合构造、预测误差验证

设计原则：
- 数据抽象层：预留真实数据集成点
- 模拟优先：使用模拟数据快速验证框架
- 接口稳定：真实数据集成时无需修改业务逻辑

使用示例：
    # Phase 3A: 使用模拟数据
    from core_bak_refactored.core.risk.backtest_framework import create_data_provider
    provider = create_data_provider('mock')
    
    # Phase 3B: 使用Yahoo Finance真实数据
    provider = create_data_provider('yahoo', fallback_to_mock=True)
    
    # 推荐：自动选择（优先真实数据，失败回退）
    provider = create_data_provider('auto')
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Protocol, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

logger = logging.getLogger('DeepSeekQuant.BacktestFramework')


# =============================================================================
# 数据接口抽象层（预留真实数据集成点）
# =============================================================================

class HistoricalDataProvider(Protocol):
    """
    历史数据提供者接口（抽象层）
    
    设计目的：
    - 解耦业务逻辑与数据来源
    - 支持模拟数据（当前）和真实数据（未来）无缝切换
    - 为core_bak_refactored/core/data模块集成预留标准接口
    """
    
    def get_index_prices(self, index_id: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        获取指数价格数据
        
        Args:
            index_id: 指数代码（如'000300.SH'沪深300）
            start_date: 开始日期 'YYYY-MM-DD' 或 datetime 对象
            end_date: 结束日期 'YYYY-MM-DD' 或 datetime 对象
        
        Returns:
            DataFrame with columns: ['date', 'close', 'volume']
        """
        ...
    
    def get_index_returns(self, index_id: str, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.Series:
        """
        获取指数收益率序列
        
        Args:
            index_id: 指数代码
            start_date: 开始日期 'YYYY-MM-DD' 或 datetime 对象
            end_date: 结束日期 'YYYY-MM-DD' 或 datetime 对象
        
        Returns:
            Series with date index and return values
        """
        ...


# =============================================================================
# 模拟数据提供者（Phase 3A实现）
# =============================================================================

from core_bak_refactored.core.data._fragments.historical_data_provider import MockHistoricalDataProvider


# =============================================================================
# 真实数据提供者（Phase 3B实现）
# =============================================================================

from core_bak_refactored.core.data._fragments.yahoo_finance_provider import YahooFinanceDataProvider


# =============================================================================
# 数据提供者工厂函数
# =============================================================================

def create_data_provider(provider_type: str = 'mock', **kwargs) -> HistoricalDataProvider:
    """
    创建历史数据提供者
    
    Args:
        provider_type: 数据提供者类型
            - 'mock': 模拟数据提供者（Phase 3A）
            - 'yahoo': Yahoo Finance真实数据（Phase 3B）
            - 'auto': 自动选择（优先yahoo，失败回退mock）
        **kwargs: 传递给数据提供者的额外参数
    
    Returns:
        HistoricalDataProvider实例
    
    Examples:
        # Phase 3A: 使用模拟数据
        provider = create_data_provider('mock')
        
        # Phase 3B: 使用Yahoo Finance真实数据
        provider = create_data_provider('yahoo', fallback_to_mock=True)
        
        # 自动选择（推荐）
        provider = create_data_provider('auto')
    """
    if provider_type == 'mock':
        logger.info("Creating MockHistoricalDataProvider (Phase 3A)")
        return MockHistoricalDataProvider()
    
    elif provider_type == 'yahoo':
        logger.info("Creating YahooFinanceDataProvider (Phase 3B)")
        fallback = kwargs.get('fallback_to_mock', True)  # 默认启用回退
        return YahooFinanceDataProvider(fallback_to_mock=fallback)
    
    elif provider_type == 'auto':
        logger.info("Auto-selecting data provider (Yahoo with fallback)")
        try:
            # 尝试创建Yahoo提供者，启用回退
            return YahooFinanceDataProvider(fallback_to_mock=True)
        except Exception as e:
            logger.warning(f"Yahoo provider unavailable, falling back to Mock: {e}")
            return MockHistoricalDataProvider()
    
    else:
        raise ValueError(f"Unknown provider_type: {provider_type}. Use 'mock', 'yahoo', or 'auto'")


# =============================================================================
# 合成组合构造器（基于专家answer.md 1.3节）
# =============================================================================

from core_bak_refactored.core.portfolio._fragments.synthetic_portfolio import SyntheticPortfolio


from core_bak_refactored.core.portfolio._fragments.synthetic_portfolio import SyntheticPortfolioBuilder


# =============================================================================
# 事件窗口回测引擎（基于专家answer.md 1.3节）
# =============================================================================

from core_bak_refactored.core.backtest._fragments.event_window_backtester import BacktestEvent


from core_bak_refactored.core.backtest._fragments.event_window_backtester import BacktestResult


from core_bak_refactored.core.backtest._fragments.event_window_backtester import EventWindowBacktester


# =============================================================================
# 回测报告生成器
# =============================================================================

from core_bak_refactored.core.backtest._fragments.event_window_backtester import BacktestReporter
