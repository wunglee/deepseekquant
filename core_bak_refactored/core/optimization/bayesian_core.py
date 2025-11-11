"""
贝叶斯优化器核心 - 业务层
从 core_bak/bayesian_optimizer.py 拆分
职责: 核心优化循环、模型训练、点选择
"""

"""
贝叶斯优化器核心 - 业务层
从 core_bak/bayesian_optimizer.py 拆分
职责: 核心优化循环、参数管理、状态更新、结果生成
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
from dataclasses import asdict
import logging
import time
import warnings
import json
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from .optimization_models import (
    OptimizationConfig, OptimizationResult, BayesianOptimizationState,
    AcquisitionFunctionType, OptimizationObjective
)

logger = logging.getLogger("DeepSeekQuant.BayesianCore")


from .optimization_models import (
    OptimizationConfig, OptimizationResult, BayesianOptimizationState,
    AcquisitionFunctionType, OptimizationObjective
)
from .parameter_space import ParameterSpace
from .bayesian_state import BayesianStateManager
from .bayesian_utils import BayesianUtils


class BayesianOptimizer:
    """贝叶斯优化器 - 核心优化引擎"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化贝叶斯优化器

        Args:
            config: 配置字典
        """
        self.config = config
        self.optimization_config = self._create_optimization_config()

        # 优化状态
        self.state = {
            'iteration': 0,
            'parameters': [],
            'values': [],
            'best_value': float('inf'),
            'best_parameters': {},
            'acquisition_values': [],
            'model_quality': 0.0,
            'convergence_score': 0.0,
            'timestamp': datetime.now().isoformat()
        }

        # 高斯过程模型
        self.gp_model = self._create_gaussian_process()
        self.scaler = StandardScaler()

        # 随机数生成器
        self.rng = np.random.RandomState(self.optimization_config.random_seed)

        # 边界和约束
        self.parameter_bounds = {}
        self.constraints = []

        logger.info("贝叶斯优化器初始化完成")

    def _create_optimization_config(self) -> OptimizationConfig:
        """创建优化配置"""
        return OptimizationConfig(
            objective=OptimizationObjective(self.config.get('objective', 'minimize')),
            acquisition_function=AcquisitionFunctionType(
                self.config.get('acquisition_function', 'expected_improvement')
            ),
            max_iterations=self.config.get('max_iterations', 100),
            initial_points=self.config.get('initial_points', 10),
            kappa=self.config.get('kappa', 2.576),
            xi=self.config.get('xi', 0.01),
            noise_level=self.config.get('noise_level', 0.1),
            convergence_tolerance=self.config.get('convergence_tolerance', 1e-6),
            patience=self.config.get('patience', 10),
            random_seed=self.config.get('random_seed'),
            parallel_evaluations=self.config.get('parallel_evaluations', True),
            max_parallel=self.config.get('max_parallel', 4),
            bounds_scaling=self.config.get('bounds_scaling', True),
            normalization=self.config.get('normalization', True),
            early_stopping=self.config.get('early_stopping', True),
            verbose=self.config.get('verbose', True)
        )

    def _create_gaussian_process(self) -> GaussianProcessRegressor:
        """创建高斯过程回归器"""
        # 核函数配置
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=1.0,
            nu=2.5
        ) + WhiteKernel(noise_level=self.optimization_config.noise_level)

        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.optimization_config.random_seed
        )

    def set_parameter_bounds(self, parameter_bounds: Dict[str, Tuple[float, float]]):
        """设置参数边界"""
        self.parameter_bounds = parameter_bounds
        logger.info(f"参数边界设置完成: {len(parameter_bounds)} 个参数")

    def add_constraint(self, constraint_func: Callable, constraint_type: str = "ineq"):
        """添加优化约束"""
        self.constraints.append({
            'func': constraint_func,
            'type': constraint_type
        })
        logger.info("优化约束已添加")

    def optimize(self, objective_function: Callable,
                 initial_parameters: Optional[List[Dict[str, float]]] = None) -> OptimizationResult:
        """
        执行贝叶斯优化

        Args:
            objective_function: 目标函数
            initial_parameters: 初始参数点

        Returns:
            优化结果
        """
        start_time = time.time()

        try:
            # 初始化参数空间
            if initial_parameters is None:
                initial_parameters = self._generate_initial_points()

            # 评估初始点
            initial_values = self._evaluate_points(objective_function, initial_parameters)

            # 更新状态
            self.state['parameters'] = initial_parameters
            self.state['values'] = initial_values
            self._update_best_point()

            # 主优化循环
            convergence_count = 0
            acquisition_values = []
            uncertainty_estimates = []

            for iteration in range(self.optimization_config.max_iterations):
                # 训练高斯过程模型
                self._train_gaussian_process()

                # 选择下一个评估点
                next_point, acquisition_value = self._select_next_point()

                # 评估新点
                new_value = objective_function(next_point)

                # 更新状态
                self.state['parameters'].append(next_point)
                self.state['values'].append(new_value)
                self.state['acquisition_values'].append(acquisition_value)
                self.state['iteration'] = iteration + 1

                # 更新最佳点
                old_best = self.state['best_value']
                self._update_best_point()

                # 记录采集函数值和不确定性
                acquisition_values.append(acquisition_value)
                uncertainty_estimates.append(self._estimate_uncertainty(next_point))

                # 检查收敛
                if self._check_convergence(old_best):
                    convergence_count += 1
                else:
                    convergence_count = 0

                if (self.optimization_config.early_stopping and
                        convergence_count >= self.optimization_config.patience):
                    logger.info(f"提前停止: 迭代 {iteration}, 收敛次数 {convergence_count}")
                    break

                # 输出进度
                if self.optimization_config.verbose and iteration % 10 == 0:
                    logger.info(f"迭代 {iteration}: 最佳值 = {self.state['best_value']:.6f}")

            # 生成最终结果
            execution_time = time.time() - start_time
            result = self._create_optimization_result(execution_time, acquisition_values, uncertainty_estimates)

            logger.info(f"贝叶斯优化完成: 最佳值 = {result.optimal_value:.6f}, "
                        f"迭代 = {result.iterations}, 耗时 = {execution_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"贝叶斯优化失败: {e}")
            return OptimizationResult(
                success=False,
                optimal_parameters={},
                optimal_value=float('inf'),
                iterations=self.state['iteration'],
                convergence=False,
                convergence_history=[],
                execution_time=time.time() - start_time,
                evaluations=len(self.state['values']),
                acquisition_function_values=[],
                uncertainty_estimates=[],
                confidence_intervals={},
                model_hyperparameters={},
                metadata={'error': str(e)}
            )

    def _generate_initial_points(self) -> List[Dict[str, float]]:
        """生成初始参数点"""
        initial_points = []

        for i in range(self.optimization_config.initial_points):
            point = {}
            for param_name, (lower, upper) in self.parameter_bounds.items():
                point[param_name] = self.rng.uniform(lower, upper)
            initial_points.append(point)

        return initial_points

    def _evaluate_points(self, objective_function: Callable,
                         parameters: List[Dict[str, float]]) -> List[float]:
        """评估参数点"""
        values = []

        for params in parameters:
            try:
                value = objective_function(params)
                values.append(float(value))
            except Exception as e:
                logger.error(f"参数评估失败 {params}: {e}")
                values.append(float('inf'))

        return values

    def _train_gaussian_process(self):
        """训练高斯过程模型"""
        try:
            # 准备训练数据
            X = self._parameters_to_array(self.state['parameters'])
            y = np.array(self.state['values'])

            if self.optimization_config.normalization:
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = X

            # 训练模型
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp_model.fit(X_scaled, y)

            # 评估模型质量
            self.state['model_quality'] = self._evaluate_model_quality(X_scaled, y)

        except Exception as e:
            logger.error(f"高斯过程训练失败: {e}")
            raise

    def _evaluate_model_quality(self, X: np.ndarray, y: np.ndarray) -> float:
        """评估模型质量"""
        try:
            # 使用交叉验证评估模型
            if len(y) >= 5:
                scores = cross_val_score(
                    self.gp_model, X, y,
                    cv=min(5, len(y)),
                    scoring='neg_mean_squared_error'
                )
                return float(np.mean(scores))
            else:
                return 0.0
        except:
            return 0.0

    def _select_next_point(self) -> Tuple[Dict[str, float], float]:
        """选择下一个评估点"""

        # 定义采集函数优化问题
        def acquisition_optimization(x):
            x_array = np.array(x).reshape(1, -1)
            return -self._acquisition_function(x_array)

        # 参数边界
        bounds = list(self.parameter_bounds.values())
        initial_guess = self._random_initial_guess()

        # 优化采集函数
        result = minimize(
            acquisition_optimization,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )

        if result.success:
            best_x = result.x
            best_acquisition = -result.fun[0]
        else:
            # 失败时使用随机搜索
            best_x, best_acquisition = self._random_search_acquisition()

        # 转换回参数字典
        next_point = self._array_to_parameters(best_x)

        return next_point, best_acquisition

    def _acquisition_function(self, x: np.ndarray) -> float:
        """采集函数计算"""
        if self.optimization_config.normalization:
            x_scaled = self.scaler.transform(x.reshape(1, -1))
        else:
            x_scaled = x.reshape(1, -1)

        # 预测均值和标准差
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred, sigma = self.gp_model.predict(x_scaled, return_std=True)

        y_pred = y_pred[0]
        sigma = sigma[0]

        # 根据采集函数类型计算
        if self.optimization_config.acquisition_function == AcquisitionFunctionType.EXPECTED_IMPROVEMENT:
            return self._expected_improvement(y_pred, sigma)
        elif self.optimization_config.acquisition_function == AcquisitionFunctionType.UPPER_CONFIDENCE_BOUND:
            return self._upper_confidence_bound(y_pred, sigma)
        elif self.optimization_config.acquisition_function == AcquisitionFunctionType.PROBABILITY_OF_IMPROVEMENT:
            return self._probability_of_improvement(y_pred, sigma)
        else:
            return self._expected_improvement(y_pred, sigma)  # 默认

    def _expected_improvement(self, mu: float, sigma: float) -> float:
        """期望改进采集函数"""
        if sigma <= 0:
            return 0

        best = self.state['best_value']
        if self.optimization_config.objective == OptimizationObjective.MAXIMIZE:
            best = -best
            mu = -mu

        z = (best - mu - self.optimization_config.xi) / sigma
        return sigma * (z * norm.cdf(z) + norm.pdf(z))

    def _upper_confidence_bound(self, mu: float, sigma: float) -> float:
        """上置信界采集函数"""
        if self.optimization_config.objective == OptimizationObjective.MAXIMIZE:
            return mu + self.optimization_config.kappa * sigma
        else:
            return -mu + self.optimization_config.kappa * sigma

    def _probability_of_improvement(self, mu: float, sigma: float) -> float:
        """改进概率采集函数"""
        if sigma <= 0:
            return 0

        best = self.state['best_value']
        if self.optimization_config.objective == OptimizationObjective.MAXIMIZE:
            best = -best
            mu = -mu

        z = (best - mu - self.optimization_config.xi) / sigma
        return norm.cdf(z)

    def _random_search_acquisition(self) -> Tuple[np.ndarray, float]:
        """随机搜索采集函数最大值"""
        best_acquisition = -float('inf')
        best_x = None

        for _ in range(1000):  # 随机采样1000个点
            x = self._random_initial_guess()
            acquisition = self._acquisition_function(x)

            if acquisition > best_acquisition:
                best_acquisition = acquisition
                best_x = x

        return best_x, best_acquisition

    def _update_best_point(self):
        """更新最佳点"""
        if self.optimization_config.objective == OptimizationObjective.MINIMIZE:
            best_idx = np.argmin(self.state['values'])
        else:
            best_idx = np.argmax(self.state['values'])

        self.state['best_value'] = self.state['values'][best_idx]
        self.state['best_parameters'] = self.state['parameters'][best_idx]

    def _check_convergence(self, old_best: float) -> bool:
        """检查收敛条件"""
        improvement = abs(self.state['best_value'] - old_best)
        return improvement < self.optimization_config.convergence_tolerance

    def _estimate_uncertainty(self, parameters: Dict[str, float]) -> float:
        """估计参数点的不确定性"""
        try:
            x = self._parameters_to_array([parameters])
            if self.optimization_config.normalization:
                x_scaled = self.scaler.transform(x)
            else:
                x_scaled = x

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, sigma = self.gp_model.predict(x_scaled, return_std=True)

            return float(sigma[0])
        except:
            return 0.0

    def _create_optimization_result(self, execution_time: float,
                                    acquisition_values: List[float],
                                    uncertainty_estimates: List[float]) -> OptimizationResult:
        """创建优化结果"""
        # 计算置信区间
        confidence_intervals = {}
        for param_name in self.parameter_bounds.keys():
            values = [p[param_name] for p in self.state['parameters']]
            if len(values) > 1:
                mean = np.mean(values)
                std = np.std(values)
                confidence_intervals[param_name] = (
                    mean - 1.96 * std,
                    mean + 1.96 * std
                )

        return OptimizationResult(
            success=True,
            optimal_parameters=self.state['best_parameters'],
            optimal_value=self.state['best_value'],
            iterations=self.state['iteration'],
            convergence=self._check_convergence(float('inf')),
            convergence_history=self.state['values'],
            execution_time=execution_time,
            evaluations=len(self.state['values']),
            acquisition_function_values=acquisition_values,
            uncertainty_estimates=uncertainty_estimates,
            confidence_intervals=confidence_intervals,
            model_hyperparameters=self.gp_model.kernel_.get_params(),
            metadata={
                'model_quality': self.state['model_quality'],
                'final_acquisition': acquisition_values[-1] if acquisition_values else 0,
                'parameter_bounds': self.parameter_bounds,
                'constraints_count': len(self.constraints)
            }
        )

