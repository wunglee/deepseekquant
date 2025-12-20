# Package marker for core_bak_refactored.core.risk

from typing import List
import numpy as np
import pandas as pd
from core_bak_refactored.infrastructure.statistical_calculators import StatisticalCalculator

# 导出配置类
from core_bak_refactored.core.risk.config import RiskConfig

def calculate_hhi(weights: List[float]) -> float:
    """
    计算Herfindahl–Hirschman Index (HHI) 集中度指数
    
    Args:
        weights: 组合权重列表（0-1之间，和通常为1）
    
    Returns:
        float: HHI（0-1），越高表示越集中
    
    说明：
        - 技术性统一：在risk层统一集中度计算口径，避免分散实现（如 sum(w**2)）
        - 不改变业务口径：如需系数或阈值，请在调用处参数化
    """
    try:
        return float(sum((float(w) ** 2) for w in weights))
    except Exception:
        return 0.0

def calculate_historical_var(
    returns: pd.Series, 
    confidence_level: float = 0.95,
    absolute: bool = True
) -> float:
    """
    计算历史模拟VaR（业务层封装）
    
    Args:
        returns: 收益率序列（pandas Series）
        confidence_level: 置信水平（默认0.95）
        absolute: 是否取绝对值（默认True）
    
    Returns:
        float: VaR值（正数表示损失）
    
    说明：
        - 业务层封装：接受pd.Series，调用Infrastructure层纯数学计算
        - 统一的历史分位数VaR计算逻辑
        - 适用于 position_risk.py 中8处重复的分位数计算
    
    示例：
        >>> var_95 = calculate_historical_var(returns, 0.95)
        >>> var_99 = calculate_historical_var(returns, 0.99)
    """
    try:
        if returns is None or len(returns) == 0:
            return 0.0
        
        # 确保是pandas Series
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        # 调用Infrastructure层纯数学计算
        quantile_level = (1 - confidence_level) * 100
        var = StatisticalCalculator.calculate_percentile(returns.values, quantile_level)
        
        # 取绝对值并转换为float
        return float(abs(var)) if absolute else float(var)
    except Exception:
        return 0.0


def calculate_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    计算条件在险价值（CVaR / Expected Shortfall）【业务层】
    
    业务含义：VaR以下损失的平均值，比VaR更保守的风险度量
    
    Args:
        returns: 收益率序列（pandas Series）
        confidence_level: 置信水平（默认0.95）
    
    Returns:
        float: CVaR值（正数表示损失）
    
    示例：
        >>> cvar_95 = calculate_cvar(returns, 0.95)
    """
    try:
        if returns is None or len(returns) == 0:
            return 0.0
        
        # 调用Infrastructure层纯数学计算
        cvar = StatisticalCalculator.calculate_cvar(
            returns.values, 
            confidence_level=confidence_level
        )
        
        return float(abs(cvar))
    except Exception:
        return 0.0


def calculate_tail_risk(
    returns: pd.Series,
    threshold: float = -0.05
) -> float:
    """
    计算尾部风险概率【业务层】
    
    业务含义：损失超过阈值的概率（如5%损失的发生频率）
    
    Args:
        returns: 收益率序列（pandas Series）
        threshold: 损失阈值（默认-0.05，即5%损失）
    
    Returns:
        float: 尾部事件发生概率（0-1）
    
    示例：
        >>> tail_prob = calculate_tail_risk(returns, threshold=-0.05)
    """
    try:
        if returns is None or len(returns) == 0:
            return 0.0
        
        # 统计超过阈值的事件数
        tail_events = returns[returns < threshold]
        return float(len(tail_events) / len(returns))
    except Exception:
        return 0.0
