from typing import Any

from core_bak_refactored.app.data_quality.api_service import DataQualityAPIService


class DataQualityAPIRouting:
    """API路由装配（职责单一：路由与入口）"""

    def __init__(self, quality_monitor: Any) -> None:
        self._service = DataQualityAPIService(quality_monitor)
        self._app = self._service.app

    def get_app(self):
        return self._app
