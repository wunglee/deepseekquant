"""
风险数据预处理器 - 基础设施层
职责：统一数据提取与转换逻辑
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger('DeepSeekQuant.Infrastructure.RiskDataPreprocessor')


class RiskDataPreprocessor:
    """
    风险数据预处理器
    
    设计原则：
    - 统一数据提取逻辑，避免重复代码
    - 提供清晰的数据转换接口
    - 无状态工具类，所有方法为静态方法
    """
    
    @staticmethod
    def extract_returns_from_dict(data: Dict[str, Any]) -> pd.Series:
        """
        从数据字典提取收益序列
        
        支持的输入格式：
        1. 直接提供 'returns' 键
        2. 提供 'prices' 键，自动计算对数收益
        
        Args:
            data: 数据字典
            
        Returns:
            收益率序列（pandas Series）
        """
        try:
            # 方式1：直接提供收益率
            if 'returns' in data:
                returns = data['returns']
                if isinstance(returns, pd.Series):
                    return returns
                elif isinstance(returns, (list, np.ndarray)):
                    return pd.Series(returns)
            
            # 方式2：从价格计算收益率
            if 'prices' in data:
                prices = data['prices']
                if isinstance(prices, (list, np.ndarray)) and len(prices) > 1:
                    # 委托给 StatisticalCalculator 计算对数收益
                    from infrastructure.risk_metrics import StatisticalCalculator
                    log_returns = StatisticalCalculator.calculate_log_returns(np.array(prices))
                    return pd.Series(log_returns)
            
            logger.warning("无法从数据字典提取收益序列")
            return pd.Series()
            
        except Exception as e:
            logger.error(f"收益序列提取失败: {e}")
            return pd.Series()
    
    @staticmethod
    def extract_market_returns_from_dict(data: Dict[str, Any]) -> Optional[pd.Series]:
        """
        从数据字典提取市场收益序列
        
        支持的输入格式：
        1. 直接提供 'market_returns' 键
        2. 提供 'benchmark_prices' 键，自动计算对数收益
        
        Args:
            data: 数据字典
            
        Returns:
            市场收益率序列（pandas Series）或 None
        """
        try:
            # 方式1：直接提供市场收益率
            if 'market_returns' in data:
                market_returns = data['market_returns']
                if isinstance(market_returns, pd.Series):
                    return market_returns
                elif isinstance(market_returns, (list, np.ndarray)):
                    return pd.Series(market_returns)
            
            # 方式2：从基准价格计算收益率
            if 'benchmark_prices' in data:
                prices = data['benchmark_prices']
                if isinstance(prices, (list, np.ndarray)) and len(prices) > 1:
                    from infrastructure.risk_metrics import StatisticalCalculator
                    log_returns = StatisticalCalculator.calculate_log_returns(np.array(prices))
                    return pd.Series(log_returns)
            
            # 市场收益可选，返回 None 不记录警告
            return None
            
        except Exception as e:
            logger.error(f"市场收益序列提取失败: {e}")
            return None
    
    @staticmethod
    def extract_returns_from_prices(prices: np.ndarray) -> np.ndarray:
        """
        从价格序列计算对数收益率（便捷方法）
        
        Args:
            prices: 价格序列
            
        Returns:
            对数收益率序列
        """
        from infrastructure.risk_metrics import StatisticalCalculator
        return StatisticalCalculator.calculate_log_returns(prices)
    
    @staticmethod
    def align_time_series(series1: pd.Series, series2: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        对齐两个时间序列（按索引）
        
        Args:
            series1: 序列1
            series2: 序列2
            
        Returns:
            (对齐后的序列1, 对齐后的序列2)
        """
        try:
            # 取交集索引
            common_index = series1.index.intersection(series2.index)
            
            if len(common_index) == 0:
                logger.warning("两个序列无交集，按长度对齐")
                min_len = min(len(series1), len(series2))
                return series1.iloc[-min_len:].reset_index(drop=True), \
                       series2.iloc[-min_len:].reset_index(drop=True)
            
            return series1.loc[common_index], series2.loc[common_index]
            
        except Exception as e:
            logger.error(f"时间序列对齐失败: {e}")
            return series1, series2
    
    @staticmethod
    def validate_returns_data(returns: pd.Series, min_length: int = 20) -> bool:
        """
        验证收益数据是否满足计算要求
        
        Args:
            returns: 收益率序列
            min_length: 最小数据长度要求
            
        Returns:
            是否有效
        """
        if returns is None or len(returns) < min_length:
            logger.warning(f"收益数据不足: 需要至少{min_length}个数据点，实际{len(returns) if returns is not None else 0}个")
            return False
        
        # 检查是否全为 NaN
        if returns.isna().all():
            logger.warning("收益数据全为NaN")
            return False
        
        # 检查是否全为零
        if (returns == 0).all():
            logger.warning("收益数据全为零")
            return False
        
        return True
