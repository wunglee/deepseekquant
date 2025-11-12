"""
采集函数模块
从 core_bak/bayesian_optimizer.py 拆分
职责: 实现各种贝叶斯优化采集函数
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from enum import Enum
import logging
from typing import Tuple, Dict, Any, Optional
import warnings

logger = logging.getLogger('DeepSeekQuant.AcquisitionFunctions')


class AcquisitionFunctionType(Enum):
    """采集函数类型枚举"""
    EXPECTED_IMPROVEMENT = "expected_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    PROBABILITY_OF_IMPROVEMENT = "probability_of_improvement"
    THOMPSON_SAMPLING = "thompson_sampling"
    ENTROPY_SEARCH = "entropy_search"


class AcquisitionFunction:
    """采集函数计算器"""
    
    def __init__(self, 
                 acquisition_type: AcquisitionFunctionType,
                 gp_model: Any,
                 scaler: Any,
                 best_value: float,
                 objective: str = "minimize",
                 kappa: float = 2.576,
                 xi: float = 0.01,
                 normalization: bool = True):
        """
        初始化采集函数
        
        Args:
            acquisition_type: 采集函数类型
            gp_model: 高斯过程模型
            scaler: 数据缩放器
            best_value: 当前最优值
            objective: 优化目标 (minimize/maximize)
            kappa: UCB参数
            xi: EI和POI参数
            normalization: 是否标准化
        """
        self.acquisition_type = acquisition_type
        self.gp_model = gp_model
        self.scaler = scaler
        self.best_value = best_value
        self.objective = objective
        self.kappa = kappa
        self.xi = xi
        self.normalization = normalization
    
    def compute(self, x: np.ndarray) -> float:
        """
        计算采集函数值
        
        Args:
            x: 输入点
            
        Returns:
            采集函数值
        """
        # 标准化输入
        if self.normalization:
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
        if self.acquisition_type == AcquisitionFunctionType.EXPECTED_IMPROVEMENT:
            return self._expected_improvement(y_pred, sigma)
        elif self.acquisition_type == AcquisitionFunctionType.UPPER_CONFIDENCE_BOUND:
            return self._upper_confidence_bound(y_pred, sigma)
        elif self.acquisition_type == AcquisitionFunctionType.PROBABILITY_OF_IMPROVEMENT:
            return self._probability_of_improvement(y_pred, sigma)
        else:
            return self._expected_improvement(y_pred, sigma)  # 默认
    
    def _expected_improvement(self, mu: float, sigma: float) -> float:
        """期望改进采集函数"""
        if sigma <= 0:
            return 0
        
        best = self.best_value
        if self.objective == "maximize":
            best = -best
            mu = -mu
        
        z = (best - mu - self.xi) / sigma
        return sigma * (z * norm.cdf(z) + norm.pdf(z))
    
    def _upper_confidence_bound(self, mu: float, sigma: float) -> float:
        """上置信界采集函数"""
        if self.objective == "maximize":
            return mu + self.kappa * sigma
        else:
            return -mu + self.kappa * sigma
    
    def _probability_of_improvement(self, mu: float, sigma: float) -> float:
        """改进概率采集函数"""
        if sigma <= 0:
            return 0
        
        best = self.best_value
        if self.objective == "maximize":
            best = -best
            mu = -mu
        
        z = (best - mu - self.xi) / sigma
        return norm.cdf(z)
    
    def optimize(self, parameter_bounds: Dict[str, Tuple[float, float]], 
                 rng: np.random.RandomState) -> Tuple[Dict[str, float], float]:
        """
        优化采集函数寻找下一个采样点
        
        Args:
            parameter_bounds: 参数边界
            rng: 随机数生成器
            
        Returns:
            最优点和采集函数值
        """
        # 定义采集函数优化问题
        def acquisition_optimization(x):
            return -self.compute(np.array(x))
        
        # 参数边界
        bounds = list(parameter_bounds.values())
        initial_guess = self._random_initial_guess(parameter_bounds, rng)
        
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
            best_acquisition = -result.fun
        else:
            # 失败时使用随机搜索
            best_x, best_acquisition = self._random_search(parameter_bounds, rng)
        
        # 转换回参数字典
        param_names = sorted(parameter_bounds.keys())
        next_point = {name: best_x[i] for i, name in enumerate(param_names)}
        
        return next_point, best_acquisition
    
    def _random_search(self, parameter_bounds: Dict[str, Tuple[float, float]],
                      rng: np.random.RandomState) -> Tuple[np.ndarray, float]:
        """随机搜索采集函数最大值"""
        best_acquisition = -float('inf')
        best_x = None
        
        for _ in range(1000):  # 随机采样1000个点
            x = self._random_initial_guess(parameter_bounds, rng)
            acquisition = self.compute(x)
            
            if acquisition > best_acquisition:
                best_acquisition = acquisition
                best_x = x
        
        return best_x, best_acquisition
    
    def _random_initial_guess(self, parameter_bounds: Dict[str, Tuple[float, float]],
                             rng: np.random.RandomState) -> np.ndarray:
        """生成随机初始点"""
        return np.array([rng.uniform(low, high)
                        for low, high in parameter_bounds.values()])
