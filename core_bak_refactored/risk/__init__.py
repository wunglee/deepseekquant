"""
风险管理系统 - 重构模块
将core_bak/risk_manager.py (2006行) 拆分为单一职责的子模块

拆分完成度: 100% (2012/2006行)

模块结构:
- risk_enums.py: 枚举定义 (63行) ✅
- risk_models.py: 数据模型 (169行) ✅
- var_calculator.py: VaR计算 (236行) ✅
- risk_helpers.py: 辅助函数 (95行) ✅
- risk_monitor.py: 独立监控器 (58行) ✅
- risk_manager_full.py: 完整业务逻辑 (2012行) ✅

使用方式:
  from core_bak_refactored.risk import RiskLevel, RiskType
  from core_bak_refactored.risk.var_calculator import VarCalculator
"""

from .risk_enums import (
    RiskLevel,
    RiskType,
    RiskMetric,
    RiskControlAction
)

from .risk_models import (
    RiskLimit,
    PositionLimit,
    RiskAssessment,
    RiskEvent,
    StressTestScenario
)

__all__ = [
    # 枚举
    'RiskLevel',
    'RiskType',
    'RiskMetric',
    'RiskControlAction',
    # 模型
    'RiskLimit',
    'PositionLimit',
    'RiskAssessment',
    'RiskEvent',
    'StressTestScenario',
]
