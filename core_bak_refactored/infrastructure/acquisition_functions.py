"""
采集函数库 - 基础设施层
从 core_bak/bayesian_optimizer.py 拆分
职责: 提供通用的贝叶斯优化采集函数
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Tuple, Dict, Callable
import warnings
import logging

logger = logging.getLogger('DeepSeekQuant.Infrastructure.AcquisitionFunctions')


class AcquisitionFunction:
    """采集函数计算器"""
    
    def __init__(self, 
                 function_type: str,
                 gp_model,
                 scaler,
                 best_value: float,
                 objective: str = "minimize",
                 kappa: float = 2.576,
                 xi: float = 0.01,
                 normalization: bool = True):
        """
        初始化采集函数
        
        Args:
            function_type: 采集函数类型 (ei/ucb/poi)
            gp_model: 高斯过程模型
            scaler: 数据缩放器
            best_value: 当前最优值
            objective: 优化目标
            kappa: UCB参数
            xi: EI和POI参数
            normalization: 是否标准化
        """
        self.function_type = function_type
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
        从 core_bak/bayesian_optimizer.py:_acquisition_function 提取
        
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
        
        # 根据类型计算
        if self.function_type == "expected_improvement":
            return self._expected_improvement(y_pred, sigma)
        elif self.function_type == "upper_confidence_bound":
            return self._upper_confidence_bound(y_pred, sigma)
        elif self.function_type == "probability_of_improvement":
            return self._probability_of_improvement(y_pred, sigma)
        else:
            return self._expected_improvement(y_pred, sigma)
    
    def _expected_improvement(self, mu: float, sigma: float) -> float:
        """
        期望改进采集函数
        从 core_bak/bayesian_optimizer.py:_expected_improvement 提取
        """
        if sigma <= 0:
            return 0
        
        best = self.best_value
        if self.objective == "maximize":
            best = -best
            mu = -mu
        
        z = (best - mu - self.xi) / sigma
        return sigma * (z * norm.cdf(z) + norm.pdf(z))
    
    def _upper_confidence_bound(self, mu: float, sigma: float) -> float:
        """
        上置信界采集函数
        从 core_bak/bayesian_optimizer.py:_upper_confidence_bound 提取
        """
        if self.objective == "maximize":
            return mu + self.kappa * sigma
        else:
            return -mu + self.kappa * sigma
    
    def _probability_of_improvement(self, mu: float, sigma: float) -> float:
        """
        改进概率采集函数
        从 core_bak/bayesian_optimizer.py:_probability_of_improvement 提取
        """
        if sigma <= 0:
            return 0
        
        best = self.best_value
        if self.objective == "maximize":
            best = -best
            mu = -mu
        
        z = (best - mu - self.xi) / sigma
        return norm.cdf(z)
    
    def optimize(self, parameter_bounds: Dict[str, Tuple[float, float]], 
                 rng) -> Tuple[Dict[str, float], float]:
        """
        优化采集函数寻找下一个采样点
        从 core_bak/bayesian_optimizer.py:_select_next_point 部分提取
        
        Args:
            parameter_bounds: 参数边界
            rng: 随机数生成器
            
        Returns:
            最优点和采集函数值
        """
        def acquisition_optimization(x):
            return -self.compute(np.array(x))
        
        bounds = list(parameter_bounds.values())
        initial_guess = self._random_guess(parameter_bounds, rng)
        
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
            best_x, best_acquisition = self._random_search(parameter_bounds, rng)
        
        param_names = sorted(parameter_bounds.keys())
        next_point = {name: best_x[i] for i, name in enumerate(param_names)}
        
        return next_point, best_acquisition
    
    def _random_search(self, parameter_bounds: Dict[str, Tuple[float, float]], 
                      rng) -> Tuple[np.ndarray, float]:
        """
        随机搜索采集函数最大值
        从 core_bak/bayesian_optimizer.py:_random_search_acquisition 提取
        """
        best_acquisition = -float('inf')
        best_x = None
        
        for _ in range(1000):
            x = self._random_guess(parameter_bounds, rng)
            acquisition = self.compute(x)
            
            if acquisition > best_acquisition:
                best_acquisition = acquisition
                best_x = x
        
        return best_x, best_acquisition
    
    def _random_guess(self, parameter_bounds: Dict[str, Tuple[float, float]], 
                     rng) -> np.ndarray:
        """生成随机初始点"""
        return np.array([rng.uniform(low, high)
                        for low, high in parameter_bounds.values()])
