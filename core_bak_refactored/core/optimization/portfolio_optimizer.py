"""
组合权重优化 - 业务层
从 core_bak/bayesian_optimizer.py 拆分
职责: 投资组合权重优化、多样化分析
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging

from .optimization_models import OptimizationResult
from .bayesian_core import BayesianOptimizer

logger = logging.getLogger('DeepSeekQuant.PortfolioOptimizer')


class PortfolioOptimizer:
    """组合权重优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化组合优化器
        
        Args:
            config: 优化配置
        """
        self.optimizer = BayesianOptimizer(config)
        self.logger = logger
    
    def optimize_portfolio_weights(self,
                                   expected_returns: pd.Series,
                                   covariance_matrix: pd.DataFrame,
                                   constraints: List[Dict[str, Any]] = None) -> OptimizationResult:
        """
        优化投资组合权重
        从 core_bak/bayesian_optimizer.py:optimize_portfolio_weights 提取
        
        Args:
            expected_returns: 预期收益序列
            covariance_matrix: 协方差矩阵
            constraints: 优化约束条件
            
        Returns:
            优化结果
        """
        try:
            # 设置参数边界（权重在0-1之间）
            symbols = expected_returns.index.tolist()
            parameter_bounds = {symbol: (0.0, 1.0) for symbol in symbols}
            self.optimizer.set_parameter_bounds(parameter_bounds)
            
            # 添加权重和为1的约束
            def weight_sum_constraint(weights):
                return sum(weights.values()) - 1.0
            
            self.optimizer.add_constraint(weight_sum_constraint, 'eq')
            
            # 添加用户定义的约束
            if constraints:
                for constraint in constraints:
                    self.optimizer.add_constraint(
                        constraint['func'],
                        constraint.get('type', 'ineq')
                    )
            
            # 定义目标函数（最大化夏普比率）
            def portfolio_objective(weights):
                try:
                    # 转换为权重向量
                    weight_vector = np.array([weights[symbol] for symbol in symbols])
                    
                    # 计算组合收益和风险
                    portfolio_return = np.dot(weight_vector, expected_returns)
                    portfolio_risk = np.sqrt(
                        weight_vector.T @ covariance_matrix @ weight_vector
                    )
                    
                    # 计算夏普比率（假设无风险利率为0）
                    if portfolio_risk > 0:
                        sharpe_ratio = portfolio_return / portfolio_risk
                    else:
                        sharpe_ratio = 0
                    
                    # 返回负值用于最小化
                    return -sharpe_ratio
                    
                except Exception as e:
                    self.logger.error(f"组合目标函数计算失败: {e}")
                    return float('inf')
            
            # 执行优化
            result = self.optimizer.optimize(portfolio_objective)
            
            # 添加组合特定信息
            if result.success:
                optimal_weights = result.optimal_parameters
                weight_vector = np.array([optimal_weights[symbol] for symbol in symbols])
                
                # 计算组合指标
                portfolio_return = np.dot(weight_vector, expected_returns)
                portfolio_risk = np.sqrt(
                    weight_vector.T @ covariance_matrix @ weight_vector
                )
                sharpe_ratio = (portfolio_return / portfolio_risk 
                               if portfolio_risk > 0 else 0)
                
                result.metadata.update({
                    'portfolio_return': portfolio_return,
                    'portfolio_risk': portfolio_risk,
                    'sharpe_ratio': sharpe_ratio,
                    'diversification_ratio': self._calculate_diversification_ratio(
                        weight_vector, covariance_matrix
                    ),
                    'concentration_index': self._calculate_concentration_index(
                        weight_vector
                    ),
                    'effective_number': self._calculate_effective_number(weight_vector)
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"组合权重优化失败: {e}")
            return self._create_error_result(str(e))
    
    def _calculate_diversification_ratio(self,
                                         weights: np.ndarray,
                                         covariance_matrix: pd.DataFrame) -> float:
        """
        计算分散化比率
        从 core_bak/bayesian_optimizer.py:_calculate_diversification_ratio 提取
        """
        try:
            # 计算加权平均波动率
            volatilities = np.sqrt(np.diag(covariance_matrix))
            weighted_vol = np.sum(weights * volatilities)
            
            # 计算组合波动率
            portfolio_vol = np.sqrt(weights.T @ covariance_matrix @ weights)
            
            if portfolio_vol > 0:
                return weighted_vol / portfolio_vol
            else:
                return 1.0
        except:
            return 1.0
    
    def _calculate_concentration_index(self, weights: np.ndarray) -> float:
        """
        计算集中度指数（赫芬达尔指数）
        从 core_bak/bayesian_optimizer.py:_calculate_concentration_index 提取
        """
        try:
            return float(np.sum(weights ** 2))
        except:
            return 0.0
    
    def _calculate_effective_number(self, weights: np.ndarray) -> float:
        """
        计算有效资产数量
        从 core_bak/bayesian_optimizer.py:_calculate_effective_number 提取
        """
        try:
            hhi = self._calculate_concentration_index(weights)
            if hhi > 0:
                return 1.0 / hhi
            else:
                return len(weights)
        except:
            return len(weights)
    
    def _analyze_weight_sensitivity(self,
                                    weights: np.ndarray,
                                    expected_returns: pd.Series,
                                    covariance_matrix: pd.DataFrame) -> Dict[str, float]:
        """分析权重敏感性"""
        sensitivities = {}
        symbols = expected_returns.index.tolist()
        
        for i, symbol in enumerate(symbols):
            # 小幅调整权重
            delta = 0.01
            weights_up = weights.copy()
            weights_up[i] += delta
            weights_up = weights_up / np.sum(weights_up)  # 重新标准化
            
            # 计算夏普比率变化
            original_sharpe = self._calculate_sharpe_ratio(
                weights, expected_returns, covariance_matrix
            )
            adjusted_sharpe = self._calculate_sharpe_ratio(
                weights_up, expected_returns, covariance_matrix
            )
            
            sensitivities[symbol] = (adjusted_sharpe - original_sharpe) / delta
        
        return sensitivities
    
    def _calculate_sharpe_ratio(self,
                                weights: np.ndarray,
                                expected_returns: pd.Series,
                                covariance_matrix: pd.DataFrame) -> float:
        """计算夏普比率"""
        try:
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(weights.T @ covariance_matrix @ weights)
            
            if portfolio_risk > 0:
                return portfolio_return / portfolio_risk
            else:
                return 0.0
        except:
            return 0.0
    
    def _create_error_result(self, error_message: str) -> OptimizationResult:
        """创建错误结果"""
        return OptimizationResult(
            success=False,
            optimal_parameters={},
            optimal_value=float('inf'),
            iterations=0,
            convergence=False,
            convergence_history=[],
            execution_time=0.0,
            evaluations=0,
            acquisition_function_values=[],
            uncertainty_estimates=[],
            confidence_intervals={},
            model_hyperparameters={},
            metadata={'error': error_message}
        )
