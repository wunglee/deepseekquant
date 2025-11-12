"""
贝叶斯优化系统 - 枚举和数据模型
拆分自: core_bak/bayesian_optimizer.py (line 25-90)
职责: 定义贝叶斯优化相关的枚举和数据类
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum


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
    metadata: Dict[str, Any] = field(default_factory=lambda: {})


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
