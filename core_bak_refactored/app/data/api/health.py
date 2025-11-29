from typing import Any, Dict


def check_health(monitor: Any) -> Dict[str, Any]:
    """健康检查（职责单一：返回最小健康状态）"""
    return {'status': 'healthy'}
