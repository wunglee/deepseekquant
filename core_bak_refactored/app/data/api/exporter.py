from typing import Any


def export_quality(monitor: Any, filename: str, fmt: str = 'json') -> bool:
    """导出质量数据（职责单一：委派导出）"""
    return monitor.export_monitoring_data(filename, fmt)
