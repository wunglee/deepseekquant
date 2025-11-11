"""
优化分析诊断 - 业务层
从 core_bak/bayesian_optimizer.py 拆分
职责: 收敛分析、参数敏感性、优化报告生成
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger('DeepSeekQuant.OptimizationAnalyzer')


class OptimizationAnalyzer:
    """优化分析器"""
    
    def __init__(self):
        """初始化优化分析器"""
        self.logger = logger
    
    def get_parameter_importance(self,
                                 parameters: List[Dict[str, float]],
                                 values: List[float]) -> Dict[str, float]:
        """
        计算参数重要性
        从 core_bak/bayesian_optimizer.py:get_parameter_importance 提取
        
        Args:
            parameters: 参数历史
            values: 目标值历史
            
        Returns:
            参数重要性字典
        """
        try:
            if len(parameters) < 2:
                return {}
            
            # 获取所有参数名
            param_names = list(parameters[0].keys())
            importance = {}
            
            for param_name in param_names:
                # 计算参数与目标值的相关性
                param_values = [p[param_name] for p in parameters]
                correlation = np.corrcoef(param_values, values)[0, 1]
                importance[param_name] = abs(correlation)
            
            # 归一化
            total = sum(importance.values())
            if total > 0:
                importance = {k: v / total for k, v in importance.items()}
            
            return importance
            
        except Exception as e:
            self.logger.error(f"参数重要性计算失败: {e}")
            return {}
    
    def calculate_convergence_score(self,
                                    convergence_history: List[float],
                                    window_size: int = 5) -> float:
        """
        计算收敛评分
        从 core_bak/bayesian_optimizer.py:_calculate_convergence_score 提取
        
        Args:
            convergence_history: 收敛历史
            window_size: 窗口大小
            
        Returns:
            收敛评分
        """
        try:
            if len(convergence_history) < window_size:
                return 0.0
            
            # 计算最近窗口的改进率
            recent_values = convergence_history[-window_size:]
            improvements = [
                abs(recent_values[i] - recent_values[i-1])
                for i in range(1, len(recent_values))
            ]
            
            # 平均改进率
            avg_improvement = np.mean(improvements)
            
            # 收敛评分（改进越小，收敛越好）
            if avg_improvement > 0:
                score = 1.0 / (1.0 + avg_improvement)
            else:
                score = 1.0
            
            return score
            
        except:
            return 0.0
    
    def estimate_parameter_sensitivity(self,
                                       parameters: Dict[str, float],
                                       objective_function: callable,
                                       parameter_bounds: Dict[str, Tuple[float, float]],
                                       delta: float = 0.01) -> Dict[str, float]:
        """
        估计参数敏感性
        从 core_bak/bayesian_optimizer.py:_estimate_parameter_sensitivity 提取
        
        Args:
            parameters: 当前参数
            objective_function: 目标函数
            parameter_bounds: 参数边界
            delta: 扰动量
            
        Returns:
            参数敏感性字典
        """
        sensitivity = {}
        
        try:
            base_value = objective_function(parameters)
            
            for param_name, param_value in parameters.items():
                # 向上扰动
                params_up = parameters.copy()
                lower, upper = parameter_bounds[param_name]
                params_up[param_name] = min(param_value + delta, upper)
                value_up = objective_function(params_up)
                
                # 向下扰动
                params_down = parameters.copy()
                params_down[param_name] = max(param_value - delta, lower)
                value_down = objective_function(params_down)
                
                # 计算敏感性（数值梯度）
                sensitivity[param_name] = abs(value_up - value_down) / (2 * delta)
                
        except Exception as e:
            self.logger.error(f"参数敏感性估计失败: {e}")
        
        return sensitivity
    
    def analyze_parameter_landscape(self,
                                    parameters: List[Dict[str, float]],
                                    values: List[float]) -> Dict[str, Any]:
        """
        分析参数空间景观
        从 core_bak/bayesian_optimizer.py:_analyze_parameter_landscape 提取
        """
        analysis = {
            'n_evaluations': len(values),
            'best_value': min(values),
            'worst_value': max(values),
            'mean_value': np.mean(values),
            'std_value': np.std(values),
            'value_range': max(values) - min(values)
        }
        
        # 分析改进趋势
        if len(values) > 1:
            improvements = [values[i] - values[i-1] for i in range(1, len(values))]
            analysis['improvements'] = {
                'mean': np.mean(improvements),
                'std': np.std(improvements),
                'positive_rate': sum(1 for x in improvements if x < 0) / len(improvements)
            }
        
        return analysis
    
    def analyze_parameter_correlations(self,
                                       parameters: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        分析参数相关性
        从 core_bak/bayesian_optimizer.py:_analyze_parameter_correlations 提取
        """
        try:
            if len(parameters) < 2:
                return {}
            
            param_names = list(parameters[0].keys())
            n_params = len(param_names)
            
            # 构建参数矩阵
            param_matrix = np.array([
                [p[name] for name in param_names]
                for p in parameters
            ])
            
            # 计算相关系数矩阵
            corr_matrix = np.corrcoef(param_matrix.T)
            
            # 提取强相关参数对
            correlations = {}
            for i in range(n_params):
                for j in range(i+1, n_params):
                    corr = corr_matrix[i, j]
                    if abs(corr) > 0.5:  # 强相关阈值
                        pair = f"{param_names[i]}-{param_names[j]}"
                        correlations[pair] = corr
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"参数相关性分析失败: {e}")
            return {}
    
    def analyze_convergence_patterns(self,
                                    convergence_history: List[float]) -> Dict[str, Any]:
        """
        分析收敛模式
        从 core_bak/bayesian_optimizer.py:_analyze_convergence_patterns 提取
        """
        patterns = {
            'n_iterations': len(convergence_history),
            'initial_value': convergence_history[0] if convergence_history else 0,
            'final_value': convergence_history[-1] if convergence_history else 0,
            'total_improvement': 0,
            'convergence_rate': 0
        }
        
        if len(convergence_history) > 1:
            patterns['total_improvement'] = (
                convergence_history[0] - convergence_history[-1]
            )
            
            # 估计收敛速率（指数拟合）
            try:
                x = np.arange(len(convergence_history))
                y = np.array(convergence_history)
                # 简化：使用线性回归估计
                slope = np.polyfit(x, y, 1)[0]
                patterns['convergence_rate'] = -slope  # 负斜率表示改进
            except:
                patterns['convergence_rate'] = 0
        
        return patterns
    
    def assess_robustness(self,
                         parameters: Dict[str, float],
                         objective_function: callable,
                         n_samples: int = 100,
                         noise_level: float = 0.1) -> Dict[str, Any]:
        """
        评估解的鲁棒性
        从 core_bak/bayesian_optimizer.py:_assess_robustness 提取
        """
        try:
            base_value = objective_function(parameters)
            perturbed_values = []
            
            for _ in range(n_samples):
                # 添加随机噪声
                perturbed_params = {
                    k: v * (1 + np.random.normal(0, noise_level))
                    for k, v in parameters.items()
                }
                
                try:
                    value = objective_function(perturbed_params)
                    perturbed_values.append(value)
                except:
                    continue
            
            if perturbed_values:
                return {
                    'base_value': base_value,
                    'mean_perturbed': np.mean(perturbed_values),
                    'std_perturbed': np.std(perturbed_values),
                    'max_deviation': max(abs(v - base_value) for v in perturbed_values),
                    'robustness_score': 1.0 / (1.0 + np.std(perturbed_values))
                }
            else:
                return {'robustness_score': 0.0}
                
        except Exception as e:
            self.logger.error(f"鲁棒性评估失败: {e}")
            return {'robustness_score': 0.0}
    
    def generate_optimization_report(self,
                                    state: Dict[str, Any],
                                    parameter_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """
        生成优化报告
        从 core_bak/bayesian_optimizer.py:get_optimization_report 提取
        """
        report = {
            'success': True,
            'optimal_value': state['best_value'],
            'optimal_parameters': state['best_parameters'],
            'iterations': state['iteration'],
            'evaluations': len(state['values']),
            'model_quality': state['model_quality'],
            'convergence_score': state['convergence_score']
        }
        
        # 添加参数重要性
        if state['parameters'] and state['values']:
            report['parameter_importance'] = self.get_parameter_importance(
                state['parameters'],
                state['values']
            )
        
        # 添加景观分析
        if state['values']:
            report['landscape_analysis'] = self.analyze_parameter_landscape(
                state['parameters'],
                state['values']
            )
        
        # 添加收敛分析
        if len(state['values']) > 1:
            report['convergence_patterns'] = self.analyze_convergence_patterns(
                state['values']
            )
        
        return report
