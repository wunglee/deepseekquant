"""
超参数调优 - 业务层
从 core_bak/bayesian_optimizer.py 拆分
职责: 模型超参数优化
"""

import numpy as np
from typing import Dict, Any, Tuple
import logging

from .optimization_models import OptimizationResult
from .bayesian_core import BayesianOptimizer

logger = logging.getLogger('DeepSeekQuant.HyperparameterTuner')


class HyperparameterTuner:
    """超参数调优器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化超参数调优器"""
        self.optimizer = BayesianOptimizer(config)
        self.logger = logger
    
    def hyperparameter_tuning(self,
                              model_class: Any,
                              parameter_space: Dict[str, Any],
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_val: np.ndarray,
                              y_val: np.ndarray,
                              scoring_metric: str = 'accuracy') -> OptimizationResult:
        """
        超参数调优
        从 core_bak/bayesian_optimizer.py:hyperparameter_tuning 提取
        
        Args:
            model_class: 模型类
            parameter_space: 参数空间
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            scoring_metric: 评分指标
            
        Returns:
            优化结果
        """
        try:
            # 设置参数边界
            self.optimizer.set_parameter_bounds(parameter_space)
            
            # 定义目标函数
            def model_objective(parameters):
                try:
                    # 创建模型实例
                    model = model_class(**parameters)
                    
                    # 训练模型
                    model.fit(X_train, y_train)
                    
                    # 在验证集上评估
                    if scoring_metric == 'accuracy':
                        score = model.score(X_val, y_val)
                        return -score  # 返回负值用于最小化
                    elif scoring_metric == 'loss':
                        predictions = model.predict(X_val)
                        loss = np.mean((predictions - y_val) ** 2)
                        return loss
                    else:
                        # 自定义评分
                        score = model.score(X_val, y_val)
                        return -score
                        
                except Exception as e:
                    self.logger.error(f"模型评估失败: {e}")
                    return float('inf')
            
            # 执行优化
            result = self.optimizer.optimize(model_objective)
            
            # 添加超参数调优特定信息
            if result.success:
                # 使用最优参数重新训练模型
                final_model = model_class(**result.optimal_parameters)
                final_model.fit(X_train, y_train)
                
                # 评估最终性能
                train_score = final_model.score(X_train, y_train)
                val_score = final_model.score(X_val, y_val)
                
                result.metadata.update({
                    'train_score': train_score,
                    'val_score': val_score,
                    'scoring_metric': scoring_metric,
                    'parameter_space': parameter_space,
                    'model_type': model_class.__name__
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"超参数调优失败: {e}")
            return self._create_error_result(str(e))
    
    def _check_hessian_positive_definite(self, hessian: np.ndarray) -> bool:
        """
        检查Hessian矩阵是否正定
        从 core_bak/bayesian_optimizer.py:_check_hessian_positive_definite 提取
        """
        try:
            eigenvalues = np.linalg.eigvals(hessian)
            return np.all(eigenvalues > 0)
        except:
            return False
    
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
