"""
历史回测框架 - 压力测试场景验证
从第5轮专家指导实施
职责: 历史事件回测、合成组合构造、预测误差验证

设计原则：
- 数据抽象层：预留真实数据集成点
- 模拟优先：使用模拟数据快速验证框架
- 接口稳定：真实数据集成时无需修改业务逻辑
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Protocol
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


# =============================================================================
# 模拟数据提供者（Phase 3A实现）
# =============================================================================

from core_bak_refactored.core.data._fragments.historical_data_provider import MockHistoricalDataProvider


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
