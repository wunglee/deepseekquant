"""
贝叶斯优化工具 - 业务层
从 core_bak/bayesian_optimizer.py 拆分
职责: 参数转换、重要性分析、可视化
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("DeepSeekQuant.BayesianUtils")


class BayesianUtils:
    """贝叶斯优化工具集"""

    @staticmethod
    def calculate_convergence_score(convergence_history: List[float],
                                    window_size: int = 5) -> float:
        """计算收敛评分"""
        if len(convergence_history) < window_size:
            return 0.0
        recent = convergence_history[-window_size:]
        improvements = [abs(recent[i] - recent[i-1]) for i in range(1, len(recent))]
        avg_improvement = np.mean(improvements)
        return 1.0 / (1.0 + avg_improvement) if avg_improvement > 0 else 1.0

    def _parameters_to_array(self, parameters: List[Dict[str, float]]) -> np.ndarray:
        """参数字典转换为数组"""
        param_names = sorted(self.parameter_bounds.keys())
        arrays = []

        for params in parameters:
            array = [params[name] for name in param_names]
            arrays.append(array)

        return np.array(arrays)

    def _array_to_parameters(self, array: np.ndarray) -> Dict[str, float]:
        """数组转换为参数字典"""
        param_names = sorted(self.parameter_bounds.keys())
        return {name: array[i] for i, name in enumerate(param_names)}

    def _random_initial_guess(self) -> np.ndarray:
        """生成随机初始点"""
        return np.array([self.rng.uniform(low, high)
                         for low, high in self.parameter_bounds.values()])

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """绘制优化历史（需要matplotlib）"""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # 绘制目标值历史
            iterations = range(len(self.state['values']))
            ax1.plot(iterations, self.state['values'], 'b-', label='Objective Value')
            ax1.plot(iterations, [self.state['best_value']] * len(iterations), 'r--', label='Best Value')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Objective Value')
            ax1.set_title('Optimization History')
            ax1.legend()
            ax1.grid(True)

            # 绘制采集函数值
            if self.state['acquisition_values']:
                ax2.plot(range(len(self.state['acquisition_values'])),
                         self.state['acquisition_values'], 'g-', label='Acquisition Value')
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Acquisition Value')
                ax2.set_title('Acquisition Function History')
                ax2.legend()
                ax2.grid(True)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Plotting failed: {e}")

    def get_parameter_importance(self) -> Dict[str, float]:
        """获取参数重要性"""
        try:
            # 使用模型的特征重要性
            if hasattr(self.gp_model, 'kernel_') and hasattr(self.gp_model.kernel_, 'k1'):
                if hasattr(self.gp_model.kernel_.k1, 'length_scale'):
                    length_scales = self.gp_model.kernel_.k1.length_scale
                    if isinstance(length_scales, np.ndarray):
                        param_names = sorted(self.parameter_bounds.keys())
                        importance = {}
                        for i, name in enumerate(param_names):
                            # 长度尺度越小，参数越重要
                            importance[name] = 1.0 / length_scales[i] if length_scales[i] > 0 else 0.0

                        # 归一化
                        total = sum(importance.values())
                        if total > 0:
                            importance = {k: v / total for k, v in importance.items()}

                        return importance

            # 回退方法：基于参数敏感性
            return self._estimate_parameter_sensitivity()

        except Exception as e:
            logger.error(f"参数重要性计算失败: {e}")
            return {}

    def _estimate_parameter_sensitivity(self) -> Dict[str, float]:
        """估计参数敏感性"""
        sensitivity = {}
        param_names = sorted(self.parameter_bounds.keys())

        if len(self.state['values']) < 10:
            return {name: 1.0 / len(param_names) for name in param_names}

        # 准备数据
        X = self._parameters_to_array(self.state['parameters'])
        y = np.array(self.state['values'])

        # 计算每个参数的方差贡献
        for i, name in enumerate(param_names):
            param_values = X[:, i]
            if len(np.unique(param_values)) > 1:
                # 计算参数值与目标值的相关性
                correlation = np.corrcoef(param_values, y)[0, 1]
                sensitivity[name] = abs(correlation)
            else:
                sensitivity[name] = 0.0

        # 归一化
        total = sum(sensitivity.values())
        if total > 0:
            sensitivity = {k: v / total for k, v in sensitivity.items()}

        return sensitivity

    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        return {
            'success': True,
            'optimal_value': self.state['best_value'],
            'optimal_parameters': self.state['best_parameters'],
            'iterations': self.state['iteration'],
            'evaluations': len(self.state['values']),
            'model_quality': self.state['model_quality'],
            'parameter_importance': self.get_parameter_importance(),
            'convergence_status': self._check_convergence(float('inf')),
            'execution_time': time.time() - float(self.state['timestamp']),
            'constraints_violated': self._check_constraints_violations(),
            'recommendations': self._generate_optimization_recommendations()
        }

    def _check_constraints_violations(self) -> List[Dict[str, Any]]:
        """检查约束违反情况"""
        violations = []

        for constraint in self.constraints:
            try:
                value = constraint['func'](self.state['best_parameters'])
                if constraint['type'] == 'ineq' and value < 0:
                    violations.append({
                        'constraint_type': 'inequality',
                        'constraint_value': value,
                        'violation_amount': abs(value),
                        'severity': 'minor' if abs(value) < 0.1 else 'major'
                    })
                elif constraint['type'] == 'eq' and abs(value) > 1e-6:
                    violations.append({
                        'constraint_type': 'equality',
                        'constraint_value': value,
                        'violation_amount': abs(value),
                        'severity': 'minor' if abs(value) < 0.01 else 'major'
                    })
            except Exception as e:
                violations.append({
                    'constraint_type': 'unknown',
                    'error': str(e),
                    'severity': 'error'
                })

        return violations

    def _generate_optimization_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # 基于收敛情况的建议
        if not self._check_convergence(float('inf')):
            recommendations.append("优化未完全收敛，建议增加迭代次数或调整收敛容忍度")

        # 基于模型质量的建议
        if self.state['model_quality'] < -0.5:
            recommendations.append("高斯过程模型质量较差，建议增加初始点数量或调整核函数参数")

        # 基于参数重要性的建议
        importance = self.get_parameter_importance()
        if importance:
            most_important = max(importance.items(), key=lambda x: x[1])
            least_important = min(importance.items(), key=lambda x: x[1])

            if most_important[1] > 0.5:
                recommendations.append(f"参数 '{most_important[0]}' 对目标函数影响显著，建议重点优化")

            if least_important[1] < 0.05:
                recommendations.append(f"参数 '{least_important[0]}' 影响较小，可考虑固定或移除")

        # 基于约束违反的建议
        violations = self._check_constraints_violations()
        if violations:
            recommendations.append("存在约束违反，建议调整约束条件或使用罚函数方法")

        # 基于采集函数的建议
        if self.state['acquisition_values']:
            last_acquisition = self.state['acquisition_values'][-1]
            if last_acquisition < 1e-6:
                recommendations.append("采集函数值较低，可能已接近最优解，可终止优化")

        return recommendations
