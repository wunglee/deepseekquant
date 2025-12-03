"""
事件驱动分析模块

职责：
- 事件配置管理
- 事件窗口数据获取
- 预测误差计算

来源：从 core/data/data_utils.py 迁移而来（属于回测/风险验证的高级业务逻辑）

使用方：
- core/backtest/_fragments/event_window_backtester.py
- core/risk/stress_test_validator.py
"""

import pandas as pd
import logging
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger('DeepSeekQuant.EventAnalysis')


@dataclass
class EventConfig:
    """
    事件配置数据类
    
    用于事件驱动分析（Event Study方法），定义事件窗口和预期效应
    """
    event_id: str
    index_id: str
    event_date: str
    event_type: str
    expected_decline: float
    market_id: str


class EventAnalyzer:
    """
    事件分析器
    
    提供事件驱动分析相关的业务逻辑
    """
    
    @staticmethod
    def safe_get_event_data(
        data_provider,
        event: EventConfig,
        window_days: int = 30,
        baseline_days: int = 252
    ) -> Tuple[pd.DataFrame, bool]:
        """
        安全获取事件窗口数据（带异常处理）
        
        业务规则：
        - 事件窗口默认30天（事件前后各15天）
        - 基准窗口默认252天（一年交易日）
        - 支持两种返回格式兼容（dict with 'event_window' key 或 DataFrame）
        
        Args:
            data_provider: 数据提供者实例
            event: 事件配置对象
            window_days: 事件窗口天数（默认30）
            baseline_days: 基准窗口天数（默认252）
        
        Returns:
            (事件窗口数据DataFrame, 是否成功bool)
        
        Examples:
            >>> from core_bak_refactored.tests.fixtures.core.data.mock_historical_data_provider import MockHistoricalDataProvider
            >>> provider = MockHistoricalDataProvider()
            >>> event = EventConfig(
            ...     event_id='2020_covid_19',
            ...     index_id='000300.SH',
            ...     event_date='2020-02-20',
            ...     event_type='pandemic',
            ...     expected_decline=-0.20,
            ...     market_id='CN'
            ... )
            >>> data, success = EventAnalyzer.safe_get_event_data(provider, event)
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
                logger.warning(
                    f"事件数据获取失败: 返回None | "
                    f"provider={type(data_provider).__name__} | "
                    f"event_id={event.event_id}"
                )
                return pd.DataFrame(), False
            
            # 支持两种返回格式
            if isinstance(window_data, dict) and 'event_window' in window_data:
                return window_data['event_window'], True
            elif isinstance(window_data, pd.DataFrame):
                return window_data, True
            else:
                logger.warning(
                    f"事件数据格式异常 | "
                    f"provider={type(data_provider).__name__} | "
                    f"event_id={event.event_id} | "
                    f"type={type(window_data).__name__}"
                )
                return pd.DataFrame(), False
                
        except Exception as e:
            # 降级日志（异常捕获）：记录事件/提供者/异常摘要
            logger.error(
                f"事件数据获取异常: "
                f"provider={type(data_provider).__name__} | "
                f"event_id={event.event_id} | "
                f"error={e}"
            )
            return pd.DataFrame(), False
    
    @staticmethod
    def calculate_prediction_error(actual_return: float, expected_decline: float) -> float:
        """
        计算预测误差
        
        业务规则：
        - 误差为实际收益与预期下跌的绝对差值
        - 误差上限裁剪为15%（避免极端异常值）
        
        Args:
            actual_return: 实际收益率
            expected_decline: 预期下跌幅度
        
        Returns:
            预测误差（绝对值，上限15%）
        
        Examples:
            >>> EventAnalyzer.calculate_prediction_error(-0.25, -0.20)
            0.05  # 5%误差
            
            >>> EventAnalyzer.calculate_prediction_error(-0.40, -0.20)
            0.15  # 误差裁剪到上限15%
        """
        # 业务规则：误差上限15%
        return min(abs(actual_return - expected_decline), 0.15)
    
    @staticmethod
    def calculate_actual_return(event_window_df: pd.DataFrame) -> float:
        """
        计算事件窗口实际收益率
        
        Args:
            event_window_df: 事件窗口数据（必须包含'close'列）
        
        Returns:
            实际收益率（如果数据不足返回0.0）
        
        Examples:
            >>> df = pd.DataFrame({'close': [100, 90, 80]})
            >>> EventAnalyzer.calculate_actual_return(df)
            -0.2  # -20%收益率
        """
        if event_window_df is None or len(event_window_df) < 2:
            return 0.0
        
        if 'close' not in event_window_df.columns:
            raise ValueError("事件窗口数据缺少'close'列")
        
        prices = event_window_df['close'].dropna()
        if len(prices) < 2:
            return 0.0
        
        return float(prices.iloc[-1] / prices.iloc[0] - 1)
