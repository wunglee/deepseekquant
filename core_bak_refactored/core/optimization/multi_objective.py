"""
多目标优化 - 业务层
从 core_bak/bayesian_optimizer.py 拆分
职责: 多目标贝叶斯优化、Pareto前沿估计
"""

import numpy as np
from typing import Dict, List, Any, Callable, Tuple
import logging

from .optimization_models import OptimizationResult
from .bayesian_core import BayesianOptimizer

logger = logging.getLogger('DeepSeekQuant.MultiObjective')


class MultiObjectiveOptimizer:
    """多目标优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化多目标优化器"""
        self.optimizer = BayesianOptimizer(config)
        self.logger = logger
    
    def multi_objective_optimization(self,
                                    objective_functions: List[Callable],
                                    weights: List[float] = None) -> OptimizationResult:
        """
        多目标优化
        从 core_bak/bayesian_optimizer.py:multi_objective_optimization 提取
        
        Args:
            objective_functions: 目标函数列表
            weights: 权重列表（加权法）
            
        Returns:
            优化结果
        """
        try:
            n_objectives = len(objective_functions)
            
            # 默认权重（平均）
            if weights is None:
                weights = [1.0 / n_objectives] * n_objectives
            
            # 定义加权目标函数
            def weighted_objective(parameters):
                try:
                    values = [func(parameters) for func in objective_functions]
                    return sum(w * v for w, v in zip(weights, values))
                except Exception as e:
                    self.logger.error(f"多目标函数评估失败: {e}")
                    return float('inf')
            
            # 执行优化
            result = self.optimizer.optimize(weighted_objective)
            
            # 添加多目标特定信息
            if result.success:
                # 计算每个目标的值
                objective_values = [
                    func(result.optimal_parameters)
                    for func in objective_functions
                ]
                
                # 估计Pareto前沿
                pareto_front = self._estimate_pareto_front(
                    objective_functions,
                    weights
                )
                
                result.metadata.update({
                    'objective_values': objective_values,
                    'weights': weights,
                    'n_objectives': n_objectives,
                    'pareto_front': pareto_front,
                    'tradeoff_analysis': self._analyze_tradeoffs(
                        objective_values,
                        weights
                    )
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"多目标优化失败: {e}")
            return self._create_error_result(str(e))
    
    def _estimate_pareto_front(self,
                               objective_functions: List[Callable],
                               weights: List[float],
                               n_points: int = 10) -> List[Dict[str, Any]]:
        """
        估计Pareto前沿
        从 core_bak/bayesian_optimizer.py:_estimate_pareto_front 提取
        """
        pareto_points = []
        
        # 生成不同权重组合
        for i in range(n_points):
            # 随机权重
            random_weights = np.random.dirichlet([1.0] * len(objective_functions))
            
            try:
                # 使用当前权重优化
                result = self.multi_objective_optimization(
                    objective_functions,
                    random_weights.tolist()
                )
                
                if result.success:
                    pareto_points.append({
                        'parameters': result.optimal_parameters,
                        'objectives': [
                            func(result.optimal_parameters)
                            for func in objective_functions
                        ],
                        'weights': random_weights.tolist()
                    })
            except:
                continue
        
        return pareto_points
    
    def _analyze_tradeoffs(self,
                           objective_values: List[float],
                           weights: List[float]) -> Dict[str, Any]:
        """
        分析目标之间的权衡
        从 core_bak/bayesian_optimizer.py:_analyze_tradeoffs 提取
        """
        analysis = {
            'weighted_sum': sum(w * v for w, v in zip(weights, objective_values)),
            'max_value': max(objective_values),
            'min_value': min(objective_values),
            'value_range': max(objective_values) - min(objective_values),
            'normalized_values': [
                v / sum(objective_values) if sum(objective_values) > 0 else 0
                for v in objective_values
            ]
        }
        
        # 计算主导目标
        weighted_values = [w * v for w, v in zip(weights, objective_values)]
        dominant_idx = weighted_values.index(max(weighted_values))
        analysis['dominant_objective'] = dominant_idx
        
        return analysis
    
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
