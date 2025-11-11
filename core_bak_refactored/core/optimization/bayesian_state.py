"""
贝叶斯优化状态管理 - 业务层
从 core_bak/bayesian_optimizer.py 拆分
职责: 状态持久化、序列化/反序列化
"""

import numpy as np
import json
from typing import Dict, List, Any
from datetime import datetime
from dataclasses import asdict
import logging

from .optimization_models import OptimizationConfig, BayesianOptimizationState

logger = logging.getLogger("DeepSeekQuant.BayesianState")


class BayesianStateManager:
    """贝叶斯优化状态管理器"""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def save_state(self, filepath: str):
        """保存优化状态"""
        try:
            state_data = {
                'config': asdict(self.optimization_config),
                'state': self.state,
                'parameter_bounds': self.parameter_bounds,
                'constraints': [{'func': str(c['func']), 'type': c['type']} for c in self.constraints],
                'gp_model': self._serialize_gp_model(),
                'scaler': self._serialize_scaler(),
                'timestamp': datetime.now().isoformat()
            }

            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)

            logger.info(f"优化状态保存成功: {filepath}")

        except Exception as e:
            logger.error(f"优化状态保存失败: {e}")

    def load_state(self, filepath: str):
        """加载优化状态"""
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)

            self.optimization_config = OptimizationConfig(**state_data['config'])
            self.state = state_data['state']
            self.parameter_bounds = state_data['parameter_bounds']
            self.constraints = state_data['constraints']
            self._deserialize_gp_model(state_data['gp_model'])
            self._deserialize_scaler(state_data['scaler'])

            logger.info(f"优化状态加载成功: {filepath}")

        except Exception as e:
            logger.error(f"优化状态加载失败: {e}")

    def _serialize_gp_model(self) -> Dict[str, Any]:
        """序列化高斯过程模型"""
        try:
            return {
                'kernel_params': self.gp_model.kernel_.get_params(),
                'alpha': self.gp_model.alpha,
                'normalize_y': self.gp_model.normalize_y
            }
        except:
            return {}

    def _deserialize_gp_model(self, model_data: Dict[str, Any]):
        """反序列化高斯过程模型"""
        try:
            kernel = Matern(**model_data['kernel_params'])
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=model_data['alpha'],
                normalize_y=model_data['normalize_y']
            )
        except:
            self.gp_model = self._create_gaussian_process()

    def _serialize_scaler(self) -> Dict[str, Any]:
        """序列化标准化器"""
        try:
            return {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist()
            }
        except:
            return {}

    def _deserialize_scaler(self, scaler_data: Dict[str, Any]):
        """反序列化标准化器"""
        try:
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(scaler_data['mean'])
            self.scaler.scale_ = np.array(scaler_data['scale'])
        except:
            self.scaler = StandardScaler()

    def get_optimization_history(self) -> List[BayesianOptimizationState]:
        """获取优化历史"""
        history = []

        for i in range(len(self.state['values'])):
            history.append(BayesianOptimizationState(
                iteration=i,
                parameters=self.state['parameters'][:i + 1],
                values=self.state['values'][:i + 1],
                best_value=np.min(self.state['values'][
                                  :i + 1]) if self.optimization_config.objective == OptimizationObjective.MINIMIZE else np.max(
                    self.state['values'][:i + 1]),
                best_parameters=self.state['parameters'][np.argmin(self.state['values'][
                                                                   :i + 1])] if self.optimization_config.objective == OptimizationObjective.MINIMIZE else
                self.state['parameters'][np.argmax(self.state['values'][:i + 1])],
                acquisition_values=self.state['acquisition_values'][:i + 1] if i < len(
                    self.state['acquisition_values']) else [],
                model_quality=self.state['model_quality'],
                convergence_score=self._calculate_convergence_score(i),
                timestamp=datetime.now().isoformat()
            ))

        return history

