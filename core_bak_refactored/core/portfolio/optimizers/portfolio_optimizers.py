"""
组合优化算法库 - 基础设施层
从 core_bak/portfolio_manager.py 拆分
职责: 提供通用的组合优化算法
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy.optimize import minimize
import logging

logger = logging.getLogger('DeepSeekQuant.Infrastructure.PortfolioOptimizers')


class PortfolioOptimizers:
    """组合优化算法库"""
    
    @staticmethod
    def mean_variance_optimization(expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   risk_aversion: float = 1.0) -> np.ndarray:
        """
        均值-方差优化
        
        Args:
            expected_returns: 预期收益向量
            covariance_matrix: 协方差矩阵
            risk_aversion: 风险厌恶系数
            
        Returns:
            最优权重向量
        """
        n_assets = len(expected_returns)
        
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            return -portfolio_return + risk_aversion * portfolio_variance
        
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(objective, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else initial_weights
    
    @staticmethod
    def risk_parity(covariance_matrix: np.ndarray) -> np.ndarray:
        """
        风险平价优化
        
        Args:
            covariance_matrix: 协方差矩阵
            
        Returns:
            最优权重向量
        """
        n_assets = covariance_matrix.shape[0]
        
        def risk_budget_objective(weights):
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            marginal_contrib = np.dot(covariance_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_variance
            return np.sum((risk_contrib - 1/n_assets)**2)
        
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(risk_budget_objective, initial_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else initial_weights
    
    @staticmethod
    def minimum_variance(covariance_matrix: np.ndarray) -> np.ndarray:
        """
        最小方差优化
        
        Args:
            covariance_matrix: 协方差矩阵
            
        Returns:
            最优权重向量
        """
        n_assets = covariance_matrix.shape[0]
        
        def objective(weights):
            return np.dot(weights, np.dot(covariance_matrix, weights))
        
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(objective, initial_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else initial_weights
    
    @staticmethod
    def max_sharpe_ratio(expected_returns: np.ndarray,
                         covariance_matrix: np.ndarray,
                         risk_free_rate: float = 0.0) -> np.ndarray:
        """
        最大夏普比率优化
        
        Args:
            expected_returns: 预期收益向量
            covariance_matrix: 协方差矩阵
            risk_free_rate: 无风险利率
            
        Returns:
            最优权重向量
        """
        n_assets = len(expected_returns)
        
        def neg_sharpe(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_std = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            return -(portfolio_return - risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
        
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(neg_sharpe, initial_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else initial_weights
