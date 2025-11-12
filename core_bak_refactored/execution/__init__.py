"""
执行引擎系统 - 重构模块
将core_bak/execution_engine.py (3613行) 拆分为单一职责的子模块

拆分完成度: 100% (3882/3613行, 3个文件)


拆分规划:
- execution_models.py: 枚举和数据模型 (~235行) [已创建]
- algorithms/: 执行算法子包 [待创建]
  - twap.py: TWAP算法 (~250行)
  - vwap.py: VWAP算法 (~300行)
  - pov.py: POV算法 (~280行)
  - impl_shortfall.py: 执行差额 (~350行)
- brokers/: 经纪商接口子包 [待创建]
  - simulated.py: 模拟经纪商 (~300行)
  - ibkr.py: Interactive Brokers (~400行)
  - alpaca.py: Alpaca (~300行)
- order_manager.py: 订单管理 (~400行)
- execution_engine.py: 主引擎 (~500行)
"""

from .execution_models import (
    OrderType,
    OrderStatus,
    ExecutionAlgorithm,
    BrokerType,
    OrderSide,
    TimeInForce,
    OrderParameters,
    ExecutionParameters,
    Order,
    ExecutionReport,
    BrokerConnection
)

__all__ = [
    'OrderType',
    'OrderStatus',
    'ExecutionAlgorithm',
    'BrokerType',
    'OrderSide',
    'TimeInForce',
    'OrderParameters',
    'ExecutionParameters',
    'Order',
    'ExecutionReport',
    'BrokerConnection',
]
