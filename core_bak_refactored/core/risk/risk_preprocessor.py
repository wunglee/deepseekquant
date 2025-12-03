"""
风险数据预处理器

职责：
- 从风险模块的数据字典中提取收益率序列
- 对齐风险计算所需的时间序列
- 验证风险数据的有效性

依赖层次：
risk (业务层) → risk_preprocessor (业务预处理) → data_utils (基础预处理) → pandas (第三方库)

从 infrastructure/data_preprocessor.py 迁移而来
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import logging

# 依赖基础预处理层（数据源基础处理）与共享业务逻辑
from core_bak_refactored.core.data.data_utils import DataUtils  # 仅使用数据源基本处理方法
from core_bak_refactored.core.share.data_analysis_utils import DataAnalysisUtils  # 共享的高级数据分析能力

logger = logging.getLogger('DeepSeekQuant.RiskDataPreprocessor')


class RiskDataPreprocessor:
    """
    风险数据预处理器（业务预处理层）
    
    设计原则：
    - 理解风险模块的数据结构和业务规则
    - 委托通用计算给 DataUtils（基础预处理层）
    - 无状态工具类，所有方法为静态方法
    """
    
    @staticmethod
    def extract_returns_from_dict(data: Dict[str, Any]) -> pd.Series:
        """
        从风险数据字典提取收益序列
        
        支持的输入格式：
        1. 直接提供 'returns' 键
        2. 提供 'prices' 键，自动计算对数收益
        
        Args:
            data: 风险模块的数据字典
            
        Returns:
            收益率序列（pandas Series）
            
        Examples:
            >>> data = {'prices': [100, 110, 121]}
            >>> returns = RiskDataPreprocessor.extract_returns_from_dict(data)
        """
        try:
            # 方式1：直接提供收益率
            if 'returns' in data:
                returns = data['returns']
                # 委托给共享业务逻辑层（类型保障）
                return DataAnalysisUtils.ensure_series(returns, "returns")
            
            # 方式2：从价格计算收益率
            if 'prices' in data:
                prices = data['prices']
                if isinstance(prices, (list, np.ndarray)) and len(prices) > 1:
                    # 委托给基础层计算对数收益率
                    prices_series = DataAnalysisUtils.ensure_series(prices, "prices")
                    return DataUtils.compute_log_returns(prices_series).dropna()
            
            logger.warning("无法从数据字典提取收益序列")
            return pd.Series()
            
        except Exception as e:
            logger.error(f"收益序列提取失败: {e}")
            return pd.Series()
    
    @staticmethod
    def extract_market_returns_from_dict(data: Dict[str, Any]) -> Optional[pd.Series]:
        """
        从风险数据字典提取市场收益序列
        
        支持的输入格式：
        1. 直接提供 'market_returns' 键
        2. 提供 'benchmark_prices' 键，自动计算对数收益
        
        Args:
            data: 风险模块的数据字典
            
        Returns:
            市场收益率序列（pandas Series）或 None
            
        Examples:
            >>> data = {'benchmark_prices': [3000, 3100, 3200]}
            >>> market_returns = RiskDataPreprocessor.extract_market_returns_from_dict(data)
        """
        try:
            # 方式1：直接提供市场收益率
            if 'market_returns' in data:
                market_returns = data['market_returns']
                # 委托给共享业务逻辑层（类型保障）
                return DataAnalysisUtils.ensure_series(market_returns, "market_returns")
            
            # 方式2：从基准价格计算收益率
            if 'benchmark_prices' in data:
                prices = data['benchmark_prices']
                if isinstance(prices, (list, np.ndarray)) and len(prices) > 1:
                    # 委托给基础层计算对数收益率
                    prices_series = DataAnalysisUtils.ensure_series(prices, "benchmark_prices")
                    return DataUtils.compute_log_returns(prices_series).dropna()
            
            # 市场收益可选，返回 None 不记录警告
            return None
            
        except Exception as e:
            logger.error(f"市场收益序列提取失败: {e}")
            return None
    
    @staticmethod
    def extract_returns_from_prices(prices: np.ndarray) -> np.ndarray:
        """
        从价格序列计算对数收益率（便捷方法，返回numpy数组）
        
        Args:
            prices: 价格序列（numpy数组）
            
        Returns:
            对数收益率序列（numpy数组）
            
        Examples:
            >>> prices = np.array([100, 110, 121])
            >>> returns = RiskDataPreprocessor.extract_returns_from_prices(prices)
        """
        # 委托给共享业务逻辑层
        prices_series = DataAnalysisUtils.ensure_series(prices, "prices")
        log_returns = DataUtils.compute_log_returns(prices_series).dropna()
        return log_returns.values
    
    @staticmethod
    def align_time_series(series1: pd.Series, series2: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        对齐两个时间序列（风险计算专用包装）
        
        Args:
            series1: 序列1
            series2: 序列2
            
        Returns:
            (对齐后的序列1, 对齐后的序列2)
            
        Examples:
            >>> returns1 = pd.Series([0.01, 0.02], index=['2020-01-01', '2020-01-02'])
            >>> returns2 = pd.Series([0.015, 0.025], index=['2020-01-01', '2020-01-02'])
            >>> aligned1, aligned2 = RiskDataPreprocessor.align_time_series(returns1, returns2)
        """
        # 委托给共享业务逻辑层
        return DataAnalysisUtils.align_time_series(series1, series2)
    
    @staticmethod
    def validate_returns_data(returns: pd.Series, min_length: int = 20) -> bool:
        """
        验证收益数据是否满足风险计算要求（风险模块的业务规则）
        
        Args:
            returns: 收益率序列
            min_length: 最小数据长度要求（风险计算默认20）
            
        Returns:
            是否有效
            
        Examples:
            >>> returns = pd.Series([0.01, 0.02, 0.015] * 10)
            >>> is_valid = RiskDataPreprocessor.validate_returns_data(returns, min_length=20)
        """
        # 长度检查
        if returns is None or len(returns) < min_length:
            logger.warning(
                f"收益数据不足: 需要至少{min_length}个数据点，"
                f"实际{len(returns) if returns is not None else 0}个"
            )
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
