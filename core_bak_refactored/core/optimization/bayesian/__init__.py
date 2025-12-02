"""
贝叶斯优化模块

职责：
- 高斯过程回归模型（Gaussian Process Regression）
- 采集函数（Acquisition Functions）
  - EI (Expected Improvement)
  - UCB (Upper Confidence Bound)
  - POI (Probability of Improvement)

注意：
- 本模块从 infrastructure 迁移而来（2025-12-02）
- 原因：贝叶斯优化是参数优化业务逻辑，非通用技术基础设施
"""

from .gaussian_process import GaussianProcessModel
from .acquisition_functions import AcquisitionFunction

__all__ = [
    'GaussianProcessModel',
    'AcquisitionFunction',
]
