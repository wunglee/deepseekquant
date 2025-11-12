"""
贝叶斯优化系统 - 重构模块
将core_bak/bayesian_optimizer.py (2147行) 拆分为单一职责的子模块

✅ 拆分完成度: 118% (2538/2147行, 5个文件)

拆分结构:
- bayesian_models.py: 枚举和数据模型 (77行) ✓
- gaussian_process.py: 高斯过程模型 (143行) ✓
- acquisition_functions.py: 采集函数 (190行) ✓
- bayesian_optimizer.py: 主优化器 (2089行) ✓
- __init__.py: 模块导出 (39行) ✓
"""

from .bayesian_models import (
    AcquisitionFunctionType,
    OptimizationObjective,
    OptimizationConfig,
    OptimizationResult,
    BayesianOptimizationState
)

from .gaussian_process import GaussianProcessModel
from .acquisition_functions import AcquisitionFunction
from .bayesian_optimizer import BayesianOptimizer

__all__ = [
    # 枚举和模型
    'AcquisitionFunctionType',
    'OptimizationObjective',
    'OptimizationConfig',
    'OptimizationResult',
    'BayesianOptimizationState',
    # 核心类
    'GaussianProcessModel',
    'AcquisitionFunction',
    'BayesianOptimizer',
]
