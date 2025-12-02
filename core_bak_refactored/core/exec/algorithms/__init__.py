"""
执行算法模块

职责：
- TWAP（时间加权平均价格）拆单算法
- VWAP（成交量加权平均价格）拆单算法
- POV（成交量百分比）拆单算法
- 冰山订单（Iceberg Order）拆单算法
- 自适应拆单算法（Adaptive Schedule）

注意：
- 本模块从 infrastructure 迁移而来（2025-12-02）
- 原因：执行算法是订单执行业务逻辑，非通用技术基础设施
"""

from .execution_algos import ExecutionAlgorithms

__all__ = ['ExecutionAlgorithms']
