from typing import Any, Dict


class DataQualityControllers:
    """API控制器（职责单一：参数到监控的委派）"""

    def __init__(self, quality_monitor: Any) -> None:
        self._qm = quality_monitor

    def get_quality_current(self, hours: int = 24) -> Dict:
        return {
            'data': self._qm.get_quality_history(hours),
            'alerts': self._qm.get_alert_history(hours),
        }
