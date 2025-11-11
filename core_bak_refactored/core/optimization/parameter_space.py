"""
参数空间管理
从 core_bak/bayesian_optimizer.py 拆分
职责: 管理优化参数的边界、转换和采样
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger('DeepSeekQuant.ParameterSpace')


class ParameterSpace:
    """参数空间管理器"""
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        初始化参数空间管理器
        
        Args:
            random_seed: 随机种子
        """
        self.parameter_bounds: Dict[str, Tuple[float, float]] = {}
        self.rng = np.random.RandomState(random_seed)
    
    def set_bounds(self, parameter_bounds: Dict[str, Tuple[float, float]]):
        """
        设置参数边界
        从 core_bak/bayesian_optimizer.py:set_parameter_bounds 提取
        
        Args:
            parameter_bounds: 参数边界字典
        """
        self.parameter_bounds = parameter_bounds
        logger.info(f"参数边界设置完成: {len(parameter_bounds)} 个参数")
    
    def parameters_to_array(self, parameters: List[Dict[str, float]]) -> np.ndarray:
        """
        参数字典转换为数组
        从 core_bak/bayesian_optimizer.py:_parameters_to_array 提取
        
        Args:
            parameters: 参数字典列表
            
        Returns:
            参数数组
        """
        param_names = sorted(self.parameter_bounds.keys())
        arrays = []
        
        for params in parameters:
            array = [params[name] for name in param_names]
            arrays.append(array)
        
        return np.array(arrays)
    
    def array_to_parameters(self, array: np.ndarray) -> Dict[str, float]:
        """
        数组转换为参数字典
        从 core_bak/bayesian_optimizer.py:_array_to_parameters 提取
        
        Args:
            array: 参数数组
            
        Returns:
            参数字典
        """
        param_names = sorted(self.parameter_bounds.keys())
        return {name: array[i] for i, name in enumerate(param_names)}
    
    def random_sample(self) -> np.ndarray:
        """
        随机采样一个参数点
        从 core_bak/bayesian_optimizer.py:_random_initial_guess 提取
        
        Returns:
            随机参数点数组
        """
        return np.array([self.rng.uniform(low, high)
                        for low, high in self.parameter_bounds.values()])
    
    def generate_initial_points(self, n_points: int) -> List[Dict[str, float]]:
        """
        生成初始参数点
        从 core_bak/bayesian_optimizer.py:_generate_initial_points 提取
        
        Args:
            n_points: 初始点数量
            
        Returns:
            初始参数点列表
        """
        initial_points = []
        
        for i in range(n_points):
            point = {}
            for param_name, (lower, upper) in self.parameter_bounds.items():
                point[param_name] = self.rng.uniform(lower, upper)
            initial_points.append(point)
        
        return initial_points
    
    def validate_parameters(self, parameters: Dict[str, float]) -> bool:
        """
        验证参数是否在边界内
        
        Args:
            parameters: 参数字典
            
        Returns:
            是否有效
        """
        for param_name, value in parameters.items():
            if param_name not in self.parameter_bounds:
                return False
            
            lower, upper = self.parameter_bounds[param_name]
            if not (lower <= value <= upper):
                return False
        
        return True
