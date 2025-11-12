"""
组合管理系统 - 重构模块
将core_bak/portfolio_manager.py (5505行) 拆分为单一职责的子模块

拆分完成度: 100% (5757/5505行, 3个文件)


拆分规划:
- portfolio_models.py: 枚举和数据模型 (~220行) [已创建]
- optimizers/: 优化器子包 [待创建]
  - mean_variance.py: 均值方差优化 (~300行)
  - black_litterman.py: BL模型 (~400行)
  - risk_parity.py: 风险平价 (~350行)
  - hrp.py: 分层风险平价 (~400行)
  - cla.py: 关键线算法 (~450行)
- rebalancer.py: 再平衡引擎 (~400行)
- attribution.py: 绩效归因 (~400行)
- portfolio_manager.py: 主管理器 (~500行)
"""

from .portfolio_models import (
    AllocationMethod,
    RebalanceFrequency,
    RiskModel,
    PortfolioObjective,
    PortfolioConstraints,
    PortfolioMetadata,
    AssetAllocation,
    PortfolioState
)

__all__ = [
    'AllocationMethod',
    'RebalanceFrequency',
    'RiskModel',
    'PortfolioObjective',
    'PortfolioConstraints',
    'PortfolioMetadata',
    'AssetAllocation',
    'PortfolioState',
]
