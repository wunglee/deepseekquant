from typing import Any, Dict


def aggregate(monitor: Any) -> Dict[str, Any]:
    """聚合质量/性能/警报数据（职责单一：读取聚合）"""
    return {
        'quality': monitor.get_quality_history(24),
        'alerts': monitor.get_alert_history(24),
        'performance': monitor.get_performance_statistics(),
    }
