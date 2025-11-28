"""
增量计算优化模块

职责：
1. 协方差矩阵增量更新 (Sherman-Morrison公式)
2. VaR增量计算策略
3. 增量计算边界条件判断
4. 误差累积监控和自动回退

设计原则：
- 80%场景收益：权重微调、单点数据更新
- 精度保障：误差<0.5%警告，>1%回退全量
- 性能目标：50资产权重调整<50ms (当前400ms)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import logging

from core_bak_refactored.infrastructure.statistical_calculators import StatisticalCalculator

logger = logging.getLogger(__name__)


@dataclass
class IncrementalBoundary:
    """增量计算边界条件"""
    max_changed_asset_ratio: float = 0.20  # 变化资产数≤20%
    max_new_asset_ratio: float = 0.10  # 新增资产≤10%
    max_market_value_ratio: float = 0.05  # 新增市值占比≤5%
    max_condition_number_change: float = 0.10  # 条件数变化<10%
    max_consecutive_updates: int = 50  # 连续增量更新≤50次
    error_warning_threshold: float = 0.005  # 0.5%警告
    error_fallback_threshold: float = 0.01  # 1%回退


class IncrementalCovarianceCalculator:
    """
    协方差矩阵增量计算器
    
    基于Sherman-Morrison-Woodbury公式实现高效增量更新
    """
    
    def __init__(self, boundary: Optional[IncrementalBoundary] = None):
        """
        初始化增量计算器
        
        Parameters:
        boundary: 边界条件配置
        """
        self.boundary = boundary or IncrementalBoundary()
        self.consecutive_updates = 0
        self.cumulative_error = 0.0
        
        # 缓存当前状态
        self.current_cov = None
        self.current_returns = None
        self.current_mean = None
        self.last_full_calculation_time = None
        
    def can_use_incremental(
        self,
        old_assets: List[str],
        new_assets: List[str],
        weights_changed_ratio: float,
        market_condition_change: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        判断是否可以使用增量计算
        
        Parameters:
        old_assets: 原资产列表
        new_assets: 新资产列表
        weights_changed_ratio: 权重变化资产占比
        market_condition_change: 市场条件数变化（可选）
        
        Returns:
        (can_use, reason): 是否可用和原因
        """
        # 1. 检查连续更新次数
        if self.consecutive_updates >= self.boundary.max_consecutive_updates:
            return False, f"连续增量更新超过{self.boundary.max_consecutive_updates}次"
        
        # 2. 检查误差累积
        if self.cumulative_error > self.boundary.error_warning_threshold:
            return False, f"累积误差{self.cumulative_error:.4f}超过阈值"
        
        # 3. 检查资产结构变化
        old_set = set(old_assets)
        new_set = set(new_assets)
        
        added_assets = new_set - old_set
        removed_assets = old_set - new_set
        
        # 新增资产占比检查
        if len(added_assets) > len(old_assets) * self.boundary.max_new_asset_ratio:
            return False, f"新增资产{len(added_assets)}超过{self.boundary.max_new_asset_ratio*100}%"
        
        # 移除资产检查（移除资产需要全量计算）
        if len(removed_assets) > 0:
            return False, f"存在{len(removed_assets)}个移除资产，需要全量计算"
        
        # 4. 检查权重变化资产占比
        if weights_changed_ratio > self.boundary.max_changed_asset_ratio:
            return False, f"权重变化资产占比{weights_changed_ratio:.2%}超过阈值"
        
        # 5. 检查市场条件数变化（如果提供）
        if market_condition_change is not None:
            if abs(market_condition_change) > self.boundary.max_condition_number_change:
                return False, f"协方差矩阵条件数变化{market_condition_change:.2%}超过阈值"
        
        return True, "满足增量计算条件"
    
    def incremental_update(
        self,
        current_cov: np.ndarray,
        current_returns: np.ndarray,
        new_return: np.ndarray,
        old_return: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        增量更新协方差矩阵
        
        Parameters:
        current_cov: (n, n) 当前协方差矩阵
        current_returns: (T, n) 历史收益率矩阵
        new_return: (n,) 新收益率数据
        old_return: (n,) 要移除的旧数据（滑动窗口，可选）
        
        Returns:
        (updated_cov, metadata): 更新后的协方差矩阵和元数据
        """
        start_time = datetime.now()
        n = current_cov.shape[0]
        T = current_returns.shape[0]
        
        # 验证输入维度
        if new_return.shape[0] != n:
            raise ValueError(f"新数据维度{new_return.shape[0]}与协方差矩阵维度{n}不匹配")
        
        # 计算当前均值
        current_mean = np.mean(current_returns, axis=0)
        
        if old_return is None:
            # 场景1: 只添加新数据（扩展窗口）
            updated_cov, error = self._add_new_data(
                current_cov, current_mean, new_return, T
            )
        else:
            # 场景2: 滑动窗口更新（移除旧数据，添加新数据）
            updated_cov, error = self._sliding_window_update(
                current_cov, current_returns, new_return, old_return
            )
        
        # 更新状态
        self.consecutive_updates += 1
        self.cumulative_error += error
        self.current_cov = updated_cov
        
        # 计算元数据
        computation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        metadata = {
            'computation_time_ms': computation_time,
            'incremental_error': error,
            'cumulative_error': self.cumulative_error,
            'consecutive_updates': self.consecutive_updates,
            'method': 'sliding_window' if old_return is not None else 'add_new',
            'timestamp': datetime.now().isoformat()
        }
        
        # 错误检查和警告
        if self.cumulative_error > self.boundary.error_warning_threshold:
            logger.warning(
                f"增量计算累积误差{self.cumulative_error:.4f}超过警告阈值"
                f"{self.boundary.error_warning_threshold}"
            )
        
        if self.cumulative_error > self.boundary.error_fallback_threshold:
            logger.error(
                f"增量计算累积误差{self.cumulative_error:.4f}超过回退阈值"
                f"{self.boundary.error_fallback_threshold}，建议执行全量计算"
            )
            metadata['fallback_recommended'] = True
        
        return updated_cov, metadata
    
    def _add_new_data(
        self,
        current_cov: np.ndarray,
        current_mean: np.ndarray,
        new_return: np.ndarray,
        T: int
    ) -> Tuple[np.ndarray, float]:
        """
        仅添加新数据的增量更新（Sherman-Morrison公式）
        
        公式: Cov_new = (T-1)/T * Cov_old + (1/T) * delta * delta'
        其中 delta = new_return - new_mean
        """
        # 更新均值
        new_mean = (T * current_mean + new_return) / (T + 1)
        
        # 计算偏差向量
        delta_old = new_return - current_mean  # 相对于旧均值
        delta_new = new_return - new_mean  # 相对于新均值
        
        # Sherman-Morrison秩一更新
        # 这是一个高效的更新公式，避免了完整的协方差重新计算
        updated_cov = (T - 1) / T * current_cov + \
                      np.outer(delta_old, delta_new) / T
        
        # 估计误差（相对Frobenius范数）
        update_magnitude = np.linalg.norm(np.outer(delta_old, delta_new))
        cov_magnitude = np.linalg.norm(current_cov)
        error = update_magnitude / (cov_magnitude * T) if cov_magnitude > 0 else 0
        
        return updated_cov, error
    
    def _sliding_window_update(
        self,
        current_cov: np.ndarray,
        current_returns: np.ndarray,
        new_return: np.ndarray,
        old_return: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        滑动窗口更新（Woodbury公式）
        
        步骤：
        1. 移除最旧的数据点
        2. 添加最新的数据点
        """
        T, n = current_returns.shape
        current_mean = np.mean(current_returns, axis=0)
        
        # 步骤1: 移除旧数据的影响
        delta_old = old_return - current_mean
        cov_after_remove = (T / (T - 1)) * (
            current_cov - np.outer(delta_old, delta_old) / T
        )
        
        # 更新均值（移除旧数据后）
        mean_after_remove = (T * current_mean - old_return) / (T - 1)
        
        # 步骤2: 添加新数据
        delta_new = new_return - mean_after_remove
        updated_cov = ((T - 2) / (T - 1)) * cov_after_remove + \
                      np.outer(delta_new, delta_new) / (T - 1)
        
        # 估计误差
        remove_magnitude = np.linalg.norm(np.outer(delta_old, delta_old))
        add_magnitude = np.linalg.norm(np.outer(delta_new, delta_new))
        cov_magnitude = np.linalg.norm(current_cov)
        
        error = (remove_magnitude + add_magnitude) / (cov_magnitude * T) \
                if cov_magnitude > 0 else 0
        
        return updated_cov, error
    
    def reset(self):
        """重置增量计算状态（执行全量计算后调用）"""
        self.consecutive_updates = 0
        self.cumulative_error = 0.0
        self.last_full_calculation_time = datetime.now()
        logger.info("增量计算状态已重置")


class IncrementalVaRCalculator:
    """
    VaR增量计算器
    
    适用场景：
    - 权重微调后快速重新计算VaR
    - 新增单个数据点后更新VaR
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        初始化VaR增量计算器
        
        Parameters:
        confidence_level: 置信水平
        """
        self.confidence_level = confidence_level
        self.base_returns = None
        self.base_var = None
        
    def update_var_on_weight_change(
        self,
        base_portfolio_returns: np.ndarray,
        base_weights: np.ndarray,
        new_weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        权重变化后快速更新VaR（参数法）
        
        Parameters:
        base_portfolio_returns: 基准组合收益率
        base_weights: 原权重
        new_weights: 新权重
        cov_matrix: 协方差矩阵
        
        Returns:
        (new_var, metadata): 新VaR和计算元数据
        """
        start_time = datetime.now()
        
        # 计算新组合的波动率
        portfolio_variance = new_weights.T @ cov_matrix @ new_weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # 计算新组合的期望收益
        mean_returns = np.mean(base_portfolio_returns)
        
        # 参数法VaR (正态分布假设)
        from scipy import stats
        z_score = stats.norm.ppf(1 - self.confidence_level)
        new_var = -(mean_returns + z_score * portfolio_volatility)
        
        computation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        metadata = {
            'method': 'parametric_incremental',
            'computation_time_ms': computation_time,
            'portfolio_volatility': portfolio_volatility,
            'timestamp': datetime.now().isoformat()
        }
        
        return new_var, metadata
    
    def update_var_on_data_change(
        self,
        base_returns: np.ndarray,
        new_return: float,
        window_size: int = 252
    ) -> Tuple[float, Dict]:
        """
        新增数据点后更新VaR（历史模拟法）
        
        Parameters:
        base_returns: 历史收益率
        new_return: 新收益率数据点
        window_size: 滚动窗口大小
        
        Returns:
        (new_var, metadata): 新VaR和计算元数据
        """
        start_time = datetime.now()
        
        # 滑动窗口：保留最近window_size个数据点
        if len(base_returns) >= window_size:
            updated_returns = np.concatenate([
                base_returns[-(window_size-1):],
                [new_return]
            ])
        else:
            updated_returns = np.concatenate([base_returns, [new_return]])
        
        # 历史模拟法计算VaR（使用基础设施层统一方法）
        new_var = StatisticalCalculator.calculate_percentile(
            -updated_returns,  # 负号转换为损失
            self.confidence_level * 100
        )
        
        computation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        metadata = {
            'method': 'historical_simulation_incremental',
            'computation_time_ms': computation_time,
            'sample_size': len(updated_returns),
            'timestamp': datetime.now().isoformat()
        }
        
        return new_var, metadata


def compare_incremental_vs_full(
    incremental_result: np.ndarray,
    full_result: np.ndarray
) -> Dict:
    """
    对比增量计算与全量计算的结果差异
    
    Parameters:
    incremental_result: 增量计算结果
    full_result: 全量计算结果
    
    Returns:
    comparison_metrics: 对比指标字典
    """
    # Frobenius范数相对误差
    frobenius_error = np.linalg.norm(incremental_result - full_result) / \
                      np.linalg.norm(full_result)
    
    # 最大元素绝对误差
    max_abs_error = np.max(np.abs(incremental_result - full_result))
    
    # 最大元素相对误差
    max_rel_error = np.max(
        np.abs(incremental_result - full_result) / 
        (np.abs(full_result) + 1e-10)
    )
    
    # 对角线元素误差（方差）
    diag_error = np.mean(
        np.abs(np.diag(incremental_result) - np.diag(full_result))
    )
    
    return {
        'frobenius_relative_error': frobenius_error,
        'max_absolute_error': max_abs_error,
        'max_relative_error': max_rel_error,
        'diagonal_mean_error': diag_error,
        'is_acceptable': frobenius_error < 0.01  # 1%阈值
    }
