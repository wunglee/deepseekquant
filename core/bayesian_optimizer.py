"""
DeepSeekQuant 贝叶斯优化器模块
基于高斯过程的贝叶斯优化，用于组合权重优化和超参数调优
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import time
from datetime import datetime

logger = logging.getLogger('DeepSeekQuant.BayesianOptimizer')


class AcquisitionFunctionType(Enum):
    """采集函数类型枚举"""
    EXPECTED_IMPROVEMENT = "expected_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    PROBABILITY_OF_IMPROVEMENT = "probability_of_improvement"
    THOMPSON_SAMPLING = "thompson_sampling"
    ENTROPY_SEARCH = "entropy_search"


class OptimizationObjective(Enum):
    """优化目标枚举"""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class OptimizationConfig:
    """优化配置"""
    objective: OptimizationObjective = OptimizationObjective.MINIMIZE
    acquisition_function: AcquisitionFunctionType = AcquisitionFunctionType.EXPECTED_IMPROVEMENT
    max_iterations: int = 100
    initial_points: int = 10
    kappa: float = 2.576  # UCB参数
    xi: float = 0.01     # EI和POI参数
    noise_level: float = 0.1
    convergence_tolerance: float = 1e-6
    patience: int = 10
    random_seed: Optional[int] = None
    parallel_evaluations: bool = True
    max_parallel: int = 4
    bounds_scaling: bool = True
    normalization: bool = True
    early_stopping: bool = True
    verbose: bool = True


@dataclass
class OptimizationResult:
    """优化结果"""
    success: bool
    optimal_parameters: Dict[str, float]
    optimal_value: float
    iterations: int
    convergence: bool
    convergence_history: List[float]
    execution_time: float
    evaluations: int
    acquisition_function_values: List[float]
    uncertainty_estimates: List[float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    model_hyperparameters: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BayesianOptimizationState:
    """贝叶斯优化状态"""
    iteration: int
    parameters: List[Dict[str, float]]
    values: List[float]
    best_value: float
    best_parameters: Dict[str, float]
    acquisition_values: List[float]
    model_quality: float
    convergence_score: float
    timestamp: str


class BayesianOptimizer:
    """贝叶斯优化器 - 基于高斯过程的参数优化"""

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

    def _random_initial_guess(self) -> np.ndarray:
        """生成随机初始点"""
        return np.array([self.rng.uniform(low, high)
                         for low, high in self.parameter_bounds.values()])

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

    def _calculate_convergence_score(self, iteration: int) -> float:
        """计算收敛分数"""
        if iteration < 2:
            return 0.0

        recent_values = self.state['values'][max(0, iteration - 10):iteration + 1]
        if len(recent_values) < 2:
            return 0.0

        improvements = np.diff(recent_values)
        if self.optimization_config.objective == OptimizationObjective.MAXIMIZE:
            improvements = -improvements  # 对于最大化，我们希望负的改进

        # 计算平均改进率
        avg_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)

        if std_improvement > 0:
            return max(0, min(1, 1 - abs(avg_improvement / std_improvement)))
        else:
            return 1.0 if avg_improvement == 0 else 0.0

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

    def optimize_portfolio_weights(self,
                                   expected_returns: pd.Series,
                                   covariance_matrix: pd.DataFrame,
                                   constraints: List[Dict[str, Any]] = None) -> OptimizationResult:
        """
        优化投资组合权重

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
            self.set_parameter_bounds(parameter_bounds)

            # 添加权重和为1的约束
            def weight_sum_constraint(weights):
                return sum(weights.values()) - 1.0

            self.add_constraint(weight_sum_constraint, 'eq')

            # 添加用户定义的约束
            if constraints:
                for constraint in constraints:
                    self.add_constraint(constraint['func'], constraint.get('type', 'ineq'))

            # 定义目标函数（最大化夏普比率）
            def portfolio_objective(weights):
                try:
                    # 转换为权重向量
                    weight_vector = np.array([weights[symbol] for symbol in symbols])

                    # 计算组合收益和风险
                    portfolio_return = np.dot(weight_vector, expected_returns)
                    portfolio_risk = np.sqrt(weight_vector.T @ covariance_matrix @ weight_vector)

                    # 计算夏普比率（假设无风险利率为0）
                    if portfolio_risk > 0:
                        sharpe_ratio = portfolio_return / portfolio_risk
                    else:
                        sharpe_ratio = 0

                    # 返回负值用于最小化
                    return -sharpe_ratio

                except Exception as e:
                    logger.error(f"组合目标函数计算失败: {e}")
                    return float('inf')

            # 执行优化
            result = self.optimize(portfolio_objective)

            # 添加组合特定信息
            if result.success:
                optimal_weights = result.optimal_parameters
                weight_vector = np.array([optimal_weights[symbol] for symbol in symbols])

                # 计算组合指标
                portfolio_return = np.dot(weight_vector, expected_returns)
                portfolio_risk = np.sqrt(weight_vector.T @ covariance_matrix @ weight_vector)
                sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0

                result.metadata.update({
                    'portfolio_return': portfolio_return,
                    'portfolio_risk': portfolio_risk,
                    'sharpe_ratio': sharpe_ratio,
                    'diversification_ratio': self._calculate_diversification_ratio(weight_vector, covariance_matrix),
                    'concentration_index': self._calculate_concentration_index(weight_vector),
                    'effective_number': self._calculate_effective_number(weight_vector)
                })

            return result

        except Exception as e:
            logger.error(f"组合权重优化失败: {e}")
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
                metadata={'error': str(e)}
            )

    def _calculate_diversification_ratio(self, weights: np.ndarray,
                                         covariance_matrix: pd.DataFrame) -> float:
        """计算分散化比率"""
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
        """计算集中度指数（赫芬达尔指数）"""
        try:
            return float(np.sum(weights ** 2))
        except:
            return 0.0

    def _calculate_effective_number(self, weights: np.ndarray) -> float:
        """计算有效资产数量"""
        try:
            hhi = self._calculate_concentration_index(weights)
            if hhi > 0:
                return 1.0 / hhi
            else:
                return len(weights)
        except:
            return len(weights)

    def optimize_trading_parameters(self,
                                    strategy_function: Callable,
                                    parameter_space: Dict[str, Tuple[float, float]],
                                    historical_data: Dict[str, Any],
                                    performance_metric: str = 'sharpe_ratio') -> OptimizationResult:
        """
        优化交易策略参数

        Args:
            strategy_function: 策略函数
            parameter_space: 参数空间 {参数名: (最小值, 最大值)}
            historical_data: 历史数据
            performance_metric: 性能指标

        Returns:
            优化结果
        """
        try:
            # 设置参数边界
            self.set_parameter_bounds(parameter_space)

            # 定义目标函数
            def strategy_objective(parameters):
                try:
                    # 运行策略
                    results = strategy_function(parameters, historical_data)

                    # 提取性能指标
                    if performance_metric == 'sharpe_ratio':
                        metric_value = results.get('sharpe_ratio', 0)
                    elif performance_metric == 'total_return':
                        metric_value = results.get('total_return', 0)
                    elif performance_metric == 'calmar_ratio':
                        metric_value = results.get('calmar_ratio', 0)
                    else:
                        metric_value = results.get(performance_metric, 0)

                    # 返回负值用于最小化（如果指标需要最大化）
                    if performance_metric in ['sharpe_ratio', 'total_return', 'calmar_ratio']:
                        return -metric_value
                    else:
                        return metric_value

                except Exception as e:
                    logger.error(f"策略评估失败: {e}")
                    return float('inf')

            # 执行优化
            result = self.optimize(strategy_objective)

            # 添加策略特定信息
            if result.success:
                # 重新运行最优策略获取完整结果
                final_results = strategy_function(result.optimal_parameters, historical_data)
                result.metadata.update({
                    'strategy_performance': final_results,
                    'performance_metric': performance_metric,
                    'parameter_space': parameter_space,
                    'data_period': f"{len(historical_data)} periods"
                })

            return result

        except Exception as e:
            logger.error(f"交易参数优化失败: {e}")
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
                metadata={'error': str(e)}
            )

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
            self.set_parameter_bounds(parameter_space)

            # 定义目标函数
            def model_objective(parameters):
                try:
                    # 创建模型实例
                    model = model_class(**parameters)

                    # 训练模型
                    model.fit(X_train, y_train)

                    # 在验证集上评估
                    if hasattr(model, 'predict_proba'):
                        y_pred = model.predict_proba(X_val)
                        if scoring_metric == 'accuracy':
                            score = np.mean(np.argmax(y_pred, axis=1) == y_val)
                        elif scoring_metric == 'log_loss':
                            from sklearn.metrics import log_loss
                            score = -log_loss(y_val, y_pred)
                        else:
                            score = model.score(X_val, y_val)
                    else:
                        y_pred = model.predict(X_val)
                        if scoring_metric == 'accuracy':
                            score = np.mean(y_pred == y_val)
                        elif scoring_metric == 'r2':
                            from sklearn.metrics import r2_score
                            score = r2_score(y_val, y_pred)
                        else:
                            score = model.score(X_val, y_val)

                    # 返回负值用于最小化
                    return -score

                except Exception as e:
                    logger.error(f"模型训练失败: {e}")
                    return float('inf')

            # 执行优化
            result = self.optimize(model_objective)

            # 添加模型特定信息
            if result.success:
                # 使用最优参数重新训练模型
                best_model = model_class(**result.optimal_parameters)
                best_model.fit(X_train, y_train)

                result.metadata.update({
                    'best_model': best_model,
                    'scoring_metric': scoring_metric,
                    'training_samples': len(X_train),
                    'validation_samples': len(X_val),
                    'feature_dimension': X_train.shape[1]
                })

            return result

        except Exception as e:
            logger.error(f"超参数调优失败: {e}")
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
                metadata={'error': str(e)}
            )

    def multi_objective_optimization(self,
                                     objective_functions: List[Callable],
                                     weights: Optional[List[float]] = None) -> OptimizationResult:
        """
        多目标优化

        Args:
            objective_functions: 目标函数列表
            weights: 目标权重

        Returns:
            优化结果
        """
        try:
            if weights is None:
                weights = [1.0] * len(objective_functions)

            if len(weights) != len(objective_functions):
                raise ValueError("权重数量必须与目标函数数量一致")

            # 定义综合目标函数
            def combined_objective(parameters):
                try:
                    objectives = []
                    for obj_func in objective_functions:
                        value = obj_func(parameters)
                        objectives.append(value)

                    # 加权和
                    weighted_sum = sum(w * obj for w, obj in zip(weights, objectives))
                    return weighted_sum

                except Exception as e:
                    logger.error(f"多目标函数评估失败: {e}")
                    return float('inf')

            # 执行优化
            result = self.optimize(combined_objective)

            # 计算各目标函数值
            if result.success:
                individual_objectives = []
                for obj_func in objective_functions:
                    value = obj_func(result.optimal_parameters)
                    individual_objectives.append(value)

                result.metadata.update({
                    'individual_objectives': individual_objectives,
                    'weights': weights,
                    'pareto_front': self._estimate_pareto_front(objective_functions),
                    'tradeoff_analysis': self._analyze_tradeoffs(individual_objectives, weights)
                })

            return result

        except Exception as e:
            logger.error(f"多目标优化失败: {e}")
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
                metadata={'error': str(e)}
            )

    def _estimate_pareto_front(self, objective_functions: List[Callable]) -> List[Dict[str, Any]]:
        """估计帕累托前沿"""
        try:
            # 简化实现：使用历史点估计帕累托前沿
            pareto_points = []

            for i, params in enumerate(self.state['parameters']):
                objectives = []
                for obj_func in objective_functions:
                    try:
                        value = obj_func(params)
                        objectives.append(value)
                    except:
                        objectives.append(float('inf'))

                # 检查是否为帕累托最优
                is_pareto = True
                for j, other_params in enumerate(self.state['parameters']):
                    if i != j:
                        other_objectives = []
                        for obj_func in objective_functions:
                            try:
                                value = obj_func(other_params)
                                other_objectives.append(value)
                            except:
                                other_objectives.append(float('inf'))

                        # 检查是否被支配
                        dominated = all(o1 <= o2 for o1, o2 in zip(other_objectives, objectives))
                        if dominated and any(o1 < o2 for o1, o2 in zip(other_objectives, objectives)):
                            is_pareto = False
                            break

                if is_pareto:
                    pareto_points.append({
                        'parameters': params,
                        'objectives': objectives,
                        'iteration': i
                    })

            return pareto_points

        except Exception as e:
            logger.error(f"帕累托前沿估计失败: {e}")
            return []

    def _analyze_tradeoffs(self, objectives: List[float], weights: List[float]) -> Dict[str, Any]:
        """分析目标间权衡"""
        try:
            if len(objectives) < 2:
                return {}

            # 计算目标间的相关性（基于历史数据）
            objective_matrix = []
            for params in self.state['parameters']:
                point_objectives = []
                for obj_func in self._get_objective_functions():  # 需要存储目标函数
                    try:
                        value = obj_func(params)
                        point_objectives.append(value)
                    except:
                        point_objectives.append(float('nan'))
                objective_matrix.append(point_objectives)

            # 计算相关系数
            correlation_matrix = np.corrcoef(np.array(objective_matrix).T)

            # 分析权衡关系
            tradeoffs = {}
            for i in range(len(objectives)):
                for j in range(i+1, len(objectives)):
                    correlation = correlation_matrix[i, j]
                    if correlation < -0.5:
                        tradeoffs[f'objective_{i}_vs_{j}'] = 'strong_tradeoff'
                    elif correlation > 0.5:
                        tradeoffs[f'objective_{i}_vs_{j}'] = 'aligned'
                    else:
                        tradeoffs[f'objective_{i}_vs_{j}'] = 'independent'

            return {
                'correlation_matrix': correlation_matrix.tolist(),
                'tradeoff_relationships': tradeoffs,
                'weight_sensitivity': self._analyze_weight_sensitivity(weights)
            }

        except Exception as e:
            logger.error(f"权衡分析失败: {e}")
            return {}

    def _analyze_weight_sensitivity(self, weights: List[float]) -> Dict[str, float]:
        """分析权重敏感性"""
        try:
            sensitivities = {}

            for i, weight in enumerate(weights):
                # 小扰动权重
                perturbed_weights = weights.copy()
                perturbed_weights[i] = weight * 1.1  # 增加10%

                # 重新计算目标值（简化实现）
                # 实际中需要重新优化
                sensitivity = 0.1  # 占位符
                sensitivities[f'weight_{i}'] = sensitivity

            return sensitivities

        except Exception as e:
            logger.error(f"权重敏感性分析失败: {e}")
            return {}

    def get_optimization_insights(self) -> Dict[str, Any]:
        """获取优化洞察"""
        return {
            'parameter_landscape': self._analyze_parameter_landscape(),
            'convergence_patterns': self._analyze_convergence_patterns(),
            'sensitivity_analysis': self._perform_sensitivity_analysis(),
            'robustness_assessment': self._assess_robustness(),
            'optimality_certificate': self._certify_optimality()
        }

    def _analyze_parameter_landscape(self) -> Dict[str, Any]:
        """分析参数空间地形"""
        try:
            # 分析参数分布
            param_stats = {}
            for param_name in self.parameter_bounds.keys():
                values = [p[param_name] for p in self.state['parameters']]
                param_stats[param_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'optimal': self.state['best_parameters'].get(param_name, 0)
                }

            # 分析目标函数地形
            objective_values = self.state['values']
            landscape_analysis = {
                'global_minima_candidates': len([v for v in objective_values if v <= self.state['best_value'] * 1.1]),
                'local_minima_count': self._estimate_local_minima_count(),
                'smoothness_score': self._estimate_landscape_smoothness(),
                'multimodality_index': self._assess_multimodality()
            }

            return {
                'parameter_statistics': param_stats,
                'landscape_characteristics': landscape_analysis,
                'correlation_structure': self._analyze_parameter_correlations()
            }

        except Exception as e:
            logger.error(f"参数空间分析失败: {e}")
            return {}

    def _estimate_local_minima_count(self) -> int:
        """估计局部最小值数量"""
        try:
            if len(self.state['values']) < 10:
                return 1

            # 简单启发式：基于值序列的局部最小值
            values = self.state['values']
            local_minima = 0

            for i in range(1, len(values)-1):
                if values[i] < values[i-1] and values[i] < values[i+1]:
                    local_minima += 1

            return max(1, local_minima)

        except:
            return 1

    def _estimate_landscape_smoothness(self) -> float:
        """估计地形平滑度"""
        try:
            if len(self.state['values']) < 3:
                return 1.0

            values = np.array(self.state['values'])
            gradients = np.diff(values)
            second_derivatives = np.diff(gradients)

            smoothness = 1.0 / (1.0 + np.std(second_derivatives))
            return float(max(0.0, min(1.0, smoothness)))

        except:
            return 0.5

    def _assess_multimodality(self) -> float:
        """评估多峰性"""
        try:
            values = np.array(self.state['values'])
            if len(values) < 10:
                return 0.0

            # 使用峰度作为多峰性指标
            from scipy.stats import kurtosis
            kurt = kurtosis(values)

            # 高峰度表示重尾，可能有多峰
            multimodality = max(0.0, (kurt - 1.0) / 10.0)  # 标准化到0-1
            return float(min(1.0, multimodality))

        except:
            return 0.0

    def _analyze_parameter_correlations(self) -> Dict[str, float]:
        """分析参数间相关性"""
        try:
            correlations = {}
            param_names = list(self.parameter_bounds.keys())

            if len(param_names) < 2:
                return {}

            # 构建参数矩阵
            param_matrix = []
            for params in self.state['parameters']:
                row = [params[name] for name in param_names]
                param_matrix.append(row)

            param_matrix = np.array(param_matrix)
            corr_matrix = np.corrcoef(param_matrix.T)

            for i in range(len(param_names)):
                for j in range(i+1, len(param_names)):
                    key = f"{param_names[i]}_{param_names[j]}"
                    correlations[key] = float(corr_matrix[i, j])

            return correlations

        except Exception as e:
            logger.error(f"参数相关性分析失败: {e}")
            return {}

    def _analyze_convergence_patterns(self) -> Dict[str, Any]:
        """分析收敛模式"""
        try:
            values = self.state['values']
            if len(values) < 5:
                return {}

            # 计算收敛指标
            improvements = np.diff(values)
            if self.optimization_config.objective == OptimizationObjective.MAXIMIZE:
                improvements = -improvements  # 对于最大化，改进应为正

            convergence_metrics = {
                'total_improvement': float(values[0] - values[-1]),
                'average_improvement_per_iteration': float(np.mean(improvements)),
                'improvement_consistency': float(1.0 - np.std(improvements) / (np.mean(np.abs(improvements)) + 1e-10)),
                'final_convergence_rate': float(improvements[-1] / improvements[0] if improvements[0] != 0 else 0),
                'plateau_detected': len([imp for imp in improvements if abs(imp) < 1e-6]) > len(improvements) * 0.2
            }

            return convergence_metrics

        except Exception as e:
            logger.error(f"收敛模式分析失败: {e}")
            return {}

    def _perform_sensitivity_analysis(self) -> Dict[str, float]:
        """执行敏感性分析"""
        try:
            sensitivities = {}

            for param_name in self.parameter_bounds.keys():
                # 分析参数变化对目标函数的影响
                param_values = [p[param_name] for p in self.state['parameters']]
                objective_values = self.state['values']

                if len(param_values) > 10:
                    # 计算参数与目标值的相关性
                    correlation = np.corrcoef(param_values, objective_values)[0, 1]
                    sensitivities[param_name] = float(abs(correlation))
                else:
                    sensitivities[param_name] = 0.5  # 默认中等敏感性

            return sensitivities

        except Exception as e:
            logger.error(f"敏感性分析失败: {e}")
            return {}

    def _assess_robustness(self) -> Dict[str, Any]:
        """评估解的鲁棒性"""
        try:
            # 分析最优解周围的稳定性
            optimal_point = self.state['best_parameters']
            optimal_value = self.state['best_value']

            # 检查邻近点的性能
            neighbor_performance = []
            for params in self.state['parameters'][-10:]:  # 最近的点
                if params != optimal_point:
                    distance = self._parameter_distance(params, optimal_point)
                    performance_ratio = self.state['values'][self.state['parameters'].index(params)] / optimal_value
                    neighbor_performance.append({
                        'distance': distance,
                        'performance_ratio': performance_ratio
                    })

            # 计算鲁棒性指标
            if neighbor_performance:
                avg_performance_ratio = np.mean([n['performance_ratio'] for n in neighbor_performance])
                robustness_score = min(1.0, avg_performance_ratio)
            else:
                robustness_score = 0.5

            return {
                'robustness_score': robustness_score,
                'stability_radius': self._estimate_stability_radius(),
                'performance_variation': self._calculate_performance_variation(),
                'sensitivity_to_perturbations': self._assess_perturbation_sensitivity()
            }

        except Exception as e:
            logger.error(f"鲁棒性评估失败: {e}")
            return {}

    def _parameter_distance(self, params1: Dict[str, float], params2: Dict[str, float]) -> float:
        """计算参数空间距离"""
        try:
            squared_distance = 0
            for key in params1.keys():
                if key in params2:
                    # 归一化距离，考虑参数边界
                    low, high = self.parameter_bounds[key]
                    range_val = high - low
                    if range_val > 0:
                        normalized_diff = (params1[key] - params2[key]) / range_val
                        squared_distance += normalized_diff ** 2

            return float(np.sqrt(squared_distance))

        except Exception as e:
            logger.error(f"参数距离计算失败: {e}")
            return float('inf')

    def _estimate_stability_radius(self) -> float:
        """估计稳定性半径"""
        try:
            optimal_value = self.state['best_value']
            optimal_params = self.state['best_parameters']

            # 查找性能下降不超过10%的最近邻点
            min_distance = float('inf')

            for i, params in enumerate(self.state['parameters']):
                if params != optimal_params:
                    performance_ratio = self.state['values'][i] / optimal_value
                    if performance_ratio >= 0.9:  # 性能下降不超过10%
                        distance = self._parameter_distance(params, optimal_params)
                        if distance < min_distance:
                            min_distance = distance

            return min_distance if min_distance < float('inf') else 0.0

        except Exception as e:
            logger.error(f"稳定性半径估计失败: {e}")
            return 0.0

    def _calculate_performance_variation(self) -> float:
        """计算性能变异系数"""
        try:
            values = np.array(self.state['values'])
            if len(values) < 2:
                return 0.0

            # 计算变异系数
            std_dev = np.std(values)
            mean_val = np.mean(values)

            if mean_val != 0:
                return float(std_dev / mean_val)
            else:
                return 0.0

        except:
            return 0.0

    def _assess_perturbation_sensitivity(self) -> Dict[str, float]:
        """评估扰动敏感性"""
        try:
            sensitivities = {}
            optimal_params = self.state['best_parameters']
            optimal_value = self.state['best_value']

            for param_name, (low, high) in self.parameter_bounds.items():
                # 对每个参数进行小扰动
                perturbed_params = optimal_params.copy()
                range_val = high - low
                perturbation = 0.05 * range_val  # 5%的扰动

                # 正向扰动
                perturbed_params[param_name] = min(high, optimal_params[param_name] + perturbation)
                # 这里需要重新评估，简化实现使用历史数据近似
                # 实际中应该调用目标函数

                # 使用历史数据中最近的点来估计
                nearest_performance = self._find_nearest_performance(perturbed_params)
                if nearest_performance is not None:
                    sensitivity = abs(nearest_performance - optimal_value) / optimal_value
                    sensitivities[param_name] = float(sensitivity)
                else:
                    sensitivities[param_name] = 0.5  # 默认值

            return sensitivities

        except Exception as e:
            logger.error(f"扰动敏感性评估失败: {e}")
            return {}

    def _find_nearest_performance(self, params: Dict[str, float]) -> Optional[float]:
        """查找最近参数的性能"""
        try:
            min_distance = float('inf')
            nearest_performance = None

            for i, stored_params in enumerate(self.state['parameters']):
                distance = self._parameter_distance(params, stored_params)
                if distance < min_distance:
                    min_distance = distance
                    nearest_performance = self.state['values'][i]

            return nearest_performance

        except:
            return None

    def _certify_optimality(self) -> Dict[str, Any]:
        """验证最优性"""
        try:
            # 多种最优性验证方法
            convergence_certificate = self._check_convergence_certificate()
            local_optimality = self._check_local_optimality()
            global_coverage = self._assess_global_coverage()

            return {
                'convergence_certificate': convergence_certificate,
                'local_optimality': local_optimality,
                'global_coverage': global_coverage,
                'optimality_confidence': self._calculate_optimality_confidence(
                    convergence_certificate, local_optimality, global_coverage
                ),
                'recommendations': self._generate_optimality_recommendations(
                    convergence_certificate, local_optimality, global_coverage
                )
            }

        except Exception as e:
            logger.error(f"最优性验证失败: {e}")
            return {}

    def _check_convergence_certificate(self) -> Dict[str, Any]:
        """检查收敛证书"""
        try:
            values = self.state['values']
            if len(values) < 10:
                return {'certified': False, 'reason': 'insufficient_data'}

            # 检查最后几次迭代的改进
            last_n = min(10, len(values))
            recent_improvements = np.diff(values[-last_n:])

            if self.optimization_config.objective == OptimizationObjective.MAXIMIZE:
                recent_improvements = -recent_improvements

            # 计算收敛指标
            max_recent_improvement = np.max(np.abs(recent_improvements))
            avg_recent_improvement = np.mean(np.abs(recent_improvements))

            certified = (max_recent_improvement < self.optimization_config.convergence_tolerance and
                         avg_recent_improvement < self.optimization_config.convergence_tolerance * 0.1)

            return {
                'certified': bool(certified),
                'max_recent_improvement': float(max_recent_improvement),
                'avg_recent_improvement': float(avg_recent_improvement),
                'tolerance_threshold': float(self.optimization_config.convergence_tolerance)
            }

        except Exception as e:
            logger.error(f"收敛证书检查失败: {e}")
            return {'certified': False, 'error': str(e)}

    def _check_local_optimality(self) -> Dict[str, Any]:
        """检查局部最优性"""
        try:
            # 使用模型预测检查局部最优性
            optimal_params = self.state['best_parameters']
            x_optimal = self._parameters_to_array([optimal_params])[0]

            if self.optimization_config.normalization:
                x_optimal_scaled = self.scaler.transform([x_optimal])[0]
            else:
                x_optimal_scaled = x_optimal

            # 计算梯度（数值近似）
            gradient = self._estimate_gradient(x_optimal_scaled)
            gradient_norm = np.linalg.norm(gradient)

            # 检查海森矩阵（数值近似）
            hessian_positive_definite = self._check_hessian_positive_definite(x_optimal_scaled)

            return {
                'gradient_norm': float(gradient_norm),
                'stationary_point': gradient_norm < 1e-4,
                'hessian_positive_definite': hessian_positive_definite,
                'local_minimum': gradient_norm < 1e-4 and hessian_positive_definite
            }

        except Exception as e:
            logger.error(f"局部最优性检查失败: {e}")
            return {'local_minimum': False, 'error': str(e)}

    def _estimate_gradient(self, x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """估计梯度（数值差分）"""
        try:
            n = len(x)
            gradient = np.zeros(n)

            for i in range(n):
                x_plus = x.copy()
                x_minus = x.copy()

                x_plus[i] += epsilon
                x_minus[i] -= epsilon

                # 使用模型预测
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_plus = self.gp_model.predict([x_plus])[0]
                    y_minus = self.gp_model.predict([x_minus])[0]

                gradient[i] = (y_plus - y_minus) / (2 * epsilon)

            return gradient

        except Exception as e:
            logger.error(f"梯度估计失败: {e}")
            return np.zeros(len(x))

    def _check_hessian_positive_definite(self, x: np.ndarray, epsilon: float = 1e-6) -> bool:
        """检查海森矩阵是否正定"""
        try:
            n = len(x)
            hessian = np.zeros((n, n))

            # 计算海森矩阵（数值近似）
            for i in range(n):
                for j in range(n):
                    # 四点点差分
                    x_pp = x.copy()
                    x_pm = x.copy()
                    x_mp = x.copy()
                    x_mm = x.copy()

                    x_pp[i] += epsilon
                    x_pp[j] += epsilon

                    x_pm[i] += epsilon
                    x_pm[j] -= epsilon

                    x_mp[i] -= epsilon
                    x_mp[j] += epsilon

                    x_mm[i] -= epsilon
                    x_mm[j] -= epsilon

                    # 使用模型预测
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        y_pp = self.gp_model.predict([x_pp])[0]
                        y_pm = self.gp_model.predict([x_pm])[0]
                        y_mp = self.gp_model.predict([x_mp])[0]
                        y_mm = self.gp_model.predict([x_mm])[0]

                    hessian[i, j] = (y_pp - y_pm - y_mp + y_mm) / (4 * epsilon * epsilon)

            # 检查正定性：所有特征值大于0
            eigenvalues = np.linalg.eigvals(hessian)
            return np.all(eigenvalues > -1e-8)  # 考虑数值误差

        except Exception as e:
            logger.error(f"海森矩阵检查失败: {e}")
            return False

    def _assess_global_coverage(self) -> Dict[str, Any]:
        """评估全局覆盖度"""
        try:
            # 分析参数空间的探索程度
            coverage_metrics = {}

            # 计算每个维度的探索范围
            dimension_coverage = {}
            for param_name, (low, high) in self.parameter_bounds.items():
                values = [p[param_name] for p in self.state['parameters']]
                explored_range = max(values) - min(values)
                total_range = high - low

                coverage_ratio = explored_range / total_range if total_range > 0 else 0
                dimension_coverage[param_name] = {
                    'coverage_ratio': float(coverage_ratio),
                    'explored_range': float(explored_range),
                    'total_range': float(total_range)
                }

            # 计算总体覆盖度
            coverage_ratios = [dim['coverage_ratio'] for dim in dimension_coverage.values()]
            overall_coverage = np.mean(coverage_ratios)

            # 评估探索模式
            exploration_pattern = self._analyze_exploration_pattern()

            return {
                'overall_coverage': float(overall_coverage),
                'dimension_coverage': dimension_coverage,
                'exploration_sufficiency': overall_coverage > 0.8,
                'exploration_pattern': exploration_pattern,
                'unexplored_regions': self._identify_unexplored_regions()
            }

        except Exception as e:
            logger.error(f"全局覆盖度评估失败: {e}")
            return {}

    def _analyze_exploration_pattern(self) -> Dict[str, Any]:
        """分析探索模式"""
        try:
            # 分析探索是随机的还是有策略的
            acquisition_values = self.state['acquisition_values']

            if len(acquisition_values) < 10:
                return {'pattern': 'insufficient_data', 'confidence': 0.0}

            # 分析采集函数值的变化模式
            acquisition_changes = np.diff(acquisition_values)

            # 计算探索-利用平衡
            exploration_ratio = len([v for v in acquisition_values if v > np.median(acquisition_values)]) / len(
                acquisition_values)

            # 分析探索策略的一致性
            if len(acquisition_changes) > 1:
                change_consistency = 1.0 - (
                            np.std(acquisition_changes) / (np.mean(np.abs(acquisition_changes)) + 1e-10))
            else:
                change_consistency = 0.5

            # 判断探索模式
            if exploration_ratio > 0.7:
                pattern = 'exploration_dominant'
            elif exploration_ratio < 0.3:
                pattern = 'exploitation_dominant'
            else:
                pattern = 'balanced'

            return {
                'pattern': pattern,
                'exploration_ratio': float(exploration_ratio),
                'change_consistency': float(change_consistency),
                'confidence': min(exploration_ratio, 1 - exploration_ratio) * 2  # 平衡时置信度高
            }

        except Exception as e:
            logger.error(f"探索模式分析失败: {e}")
            return {'pattern': 'unknown', 'confidence': 0.0}

    def _identify_unexplored_regions(self) -> List[Dict[str, Any]]:
        """识别未探索区域"""
        try:
            unexplored_regions = []

            for param_name, (low, high) in self.parameter_bounds.items():
                values = [p[param_name] for p in self.state['parameters']]

                if len(values) < 5:
                    continue

                # 识别参数空间中的间隙
                sorted_values = sorted(values)
                gaps = []

                for i in range(1, len(sorted_values)):
                    gap = sorted_values[i] - sorted_values[i - 1]
                    if gap > (high - low) * 0.1:  # 间隙大于范围10%
                        gaps.append({
                            'start': sorted_values[i - 1],
                            'end': sorted_values[i],
                            'gap_size': gap,
                            'gap_ratio': gap / (high - low)
                        })

                if gaps:
                    unexplored_regions.append({
                        'parameter': param_name,
                        'gaps': gaps,
                        'largest_gap': max(gaps, key=lambda x: x['gap_size']) if gaps else None
                    })

            return unexplored_regions

        except Exception as e:
            logger.error(f"未探索区域识别失败: {e}")
            return []

    def _calculate_optimality_confidence(self, convergence_cert: Dict,
                                         local_opt: Dict, global_cov: Dict) -> float:
        """计算最优性置信度"""
        try:
            confidence_factors = []

            # 收敛置信度
            if convergence_cert.get('certified', False):
                convergence_confidence = 0.9
            else:
                improvement = convergence_cert.get('max_recent_improvement', 1.0)
                tolerance = convergence_cert.get('tolerance_threshold', 1e-6)
                convergence_confidence = max(0.1, 1.0 - (improvement / (tolerance * 10)))

            confidence_factors.append(convergence_confidence)

            # 局部最优性置信度
            if local_opt.get('local_minimum', False):
                local_confidence = 0.8
            else:
                gradient_norm = local_opt.get('gradient_norm', 1.0)
                local_confidence = max(0.1, 1.0 - min(1.0, gradient_norm * 10))

            confidence_factors.append(local_confidence)

            # 全局覆盖置信度
            coverage = global_cov.get('overall_coverage', 0.0)
            global_confidence = coverage * 0.5 + 0.5  # 覆盖度影响但非决定性

            confidence_factors.append(global_confidence)

            # 综合置信度（几何平均）
            optimality_confidence = np.prod(confidence_factors) ** (1 / len(confidence_factors))

            return float(optimality_confidence)

        except Exception as e:
            logger.error(f"最优性置信度计算失败: {e}")
            return 0.5

    def _generate_optimality_recommendations(self, convergence_cert: Dict,
                                             local_opt: Dict, global_cov: Dict) -> List[str]:
        """生成最优性建议"""
        recommendations = []

        # 基于收敛证书的建议
        if not convergence_cert.get('certified', False):
            recommendations.append("优化未完全收敛，建议增加迭代次数或调整收敛容忍度")

        # 基于局部最优性的建议
        if not local_opt.get('local_minimum', False):
            gradient_norm = local_opt.get('gradient_norm', 1.0)
            if gradient_norm > 0.1:
                recommendations.append("当前点可能不是局部最优，建议从不同初始点重新优化")

        # 基于全局覆盖度的建议
        coverage = global_cov.get('overall_coverage', 0.0)
        if coverage < 0.7:
            recommendations.append("参数空间探索不足，建议增加初始点数量或使用更积极的探索策略")

        # 基于探索模式的建议
        exploration_pattern = global_cov.get('exploration_pattern', {})
        if exploration_pattern.get('pattern') == 'exploitation_dominant':
            recommendations.append("当前策略过于偏向利用，可能错过全局最优，建议增加探索")
        elif exploration_pattern.get('pattern') == 'exploration_dominant':
            recommendations.append("当前策略过于偏向探索，收敛较慢，建议平衡探索与利用")

        return recommendations

    def cleanup(self):
        """清理资源"""
        try:
            # 清理模型和缓存
            self.gp_model = None
            self.scaler = None
            self.state.clear()
            self.parameter_bounds.clear()
            self.constraints.clear()

            logger.info("贝叶斯优化器资源清理完成")

        except Exception as e:
            logger.error(f"贝叶斯优化器资源清理失败: {e}")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.cleanup()

    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except:
            pass  # 避免析构函数中的异常

# 辅助函数和工具类
def create_bayesian_optimizer(config: Dict[str, Any]) -> BayesianOptimizer:
    """创建贝叶斯优化器辅助函数"""
    return BayesianOptimizer(config)

def validate_optimization_result(result: OptimizationResult) -> bool:
    """验证优化结果有效性"""
    try:
        if not result.success:
            return False

        if not result.optimal_parameters:
            return False

        if np.isinf(result.optimal_value) or np.isnan(result.optimal_value):
            return False

        return True

    except:
        return False

def compare_optimization_results(results: List[OptimizationResult]) -> Dict[str, Any]:
    """比较多个优化结果"""
    try:
        if not results:
            return {}

        valid_results = [r for r in results if validate_optimization_result(r)]

        if not valid_results:
            return {}

        # 找到最佳结果
        if valid_results[0].metadata.get('objective', 'minimize') == 'minimize':
            best_result = min(valid_results, key=lambda x: x.optimal_value)
        else:
            best_result = max(valid_results, key=lambda x: x.optimal_value)

        # 计算统计信息
        optimal_values = [r.optimal_value for r in valid_results]
        execution_times = [r.execution_time for r in valid_results]

        comparison = {
            'best_result_index': valid_results.index(best_result),
            'best_optimal_value': best_result.optimal_value,
            'value_range': (min(optimal_values), max(optimal_values)),
            'value_std': float(np.std(optimal_values)),
            'time_range': (min(execution_times), max(execution_times)),
            'time_std': float(np.std(execution_times)),
            'result_consistency': len(valid_results) / len(results),
            'performance_comparison': {
                'mean_value': float(np.mean(optimal_values)),
                'median_value': float(np.median(optimal_values)),
                'best_improvement': best_result.optimal_value - np.mean(optimal_values)
            }
        }

        return comparison

    except Exception as e:
        logger.error(f"优化结果比较失败: {e}")
        return {}

if __name__ == "__main__":
    # 测试代码
    config = {
        'objective': 'minimize',
        'acquisition_function': 'expected_improvement',
        'max_iterations': 50,
        'initial_points': 10,
        'convergence_tolerance': 1e-6,
        'verbose': True
    }

    # 创建优化器
    optimizer = BayesianOptimizer(config)

    # 测试函数：Rastrigin函数（多峰测试函数）
    def rastrigin_function(x):
        A = 10
        n = len(x)
        return A * n + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x.values()])

    # 设置参数边界
    parameter_bounds = {f'x{i}': (-5.12, 5.12) for i in range(2)}
    optimizer.set_parameter_bounds(parameter_bounds)

    # 执行优化
    result = optimizer.optimize(rastrigin_function)

    print("优化结果:")
    print(f"成功: {result.success}")
    print(f"最优值: {result.optimal_value}")
    print(f"最优参数: {result.optimal_parameters}")
    print(f"迭代次数: {result.iterations}")
    print(f"执行时间: {result.execution_time:.2f}s")

    # 获取优化洞察
    insights = optimizer.get_optimization_insights()
    print(f"最优性置信度: {insights.get('optimality_certificate', {}).get('optimality_confidence', 0):.3f}")

    # 清理
    optimizer.cleanup()