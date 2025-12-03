"""
监控基础设施模块

职责：
- 性能监控
- 请求追踪
- 指标统计

设计原则：
- 通用化，适用于所有模块
- 轻量级，无侵入
- 易扩展
"""

from core_bak_refactored.infrastructure.monitoring.performance_monitor import (
    PerformanceMonitor,
    create_performance_report
)

__all__ = [
    'PerformanceMonitor',
    'create_performance_report'
]
