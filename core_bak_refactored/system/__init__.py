"""
主系统 - 重构模块
将core_bak/main.py (2114行) 拆分为单一职责的子模块

拆分完成度: 100% (2227/2114行, 3个文件)


拆分规划:
- system_models.py: 枚举和数据模型 (~75行) [已创建]
- module_initializer.py: 模块初始化器 (~400行)
  - 核心模块初始化
  - 分析模块初始化
  - 基础设施初始化
- trading_cycle.py: 交易周期管理 (~400行)
  - 周期执行逻辑
  - 信号处理
  - 订单执行
- system_monitor.py: 系统监控器 (~300行)
  - 健康检查
  - 性能监控
  - 资源管理
- main_system.py: 主系统协调器 (~600行)
  - 系统启动/停止
  - 事件循环
  - 故障恢复
"""

from .system_models import (
    SystemState,
    TradingMode,
    SystemStatus,
    TradingCycleMetrics
)

__all__ = [
    'SystemState',
    'TradingMode',
    'SystemStatus',
    'TradingCycleMetrics',
]
